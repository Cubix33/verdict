"""FastAPI backend for the VERDICT dashboard.

Serves the same four-stage pipeline the CLI runs, staged as separate endpoints
so the frontend can animate Triage -> Prober -> Adjudicator -> Verifier one
step at a time. Every endpoint operates on one "current claim", tracked
server-side and reset by `/api/reset` - which matches how the original single-
page dashboard behaves, but now goes through `verdict.ledger.Ledger`'s
claim-scoped storage instead of a process-global cache shared by every image
ever analysed.

This file was rewritten in the same change that reworked the agents. The
previous version called `process_probes(path)`, `adjudicate_from_ledger(LEDGER)`
and `verify_claims(LEDGER, claims)` - signatures that no longer exist, since
every agent now requires an explicit `claim_id` and a `Calibration` (see
verdict/calibrate.py). Those calls would raise `TypeError` on the very first
request.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import sys
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verdict import calibrate as cal_mod  # noqa: E402
from verdict.agents import adjudicator, prober, triage  # noqa: E402
from verdict.agents.verifier import verify_claims  # noqa: E402
from verdict.ledger import Ledger  # noqa: E402

import cv2  # noqa: E402

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/data", StaticFiles(directory=str(ROOT / "data")), name="data")

LEDGER = Ledger(root=str(ROOT / "data"), db=str(ROOT / "data" / "ledger.db"))
DEFAULT_CUSTOMER = os.environ.get("VERDICT_CUSTOMER", "sroie")

# One active claim per dashboard session, matching the frontend's linear
# Triage -> Prober -> Adjudicator -> Verifier flow. A real multi-user
# deployment would key this on a session id instead of a module global.
_state: dict = {"claim_id": None, "image_path": None, "triage": None}


class ImageRequest(BaseModel):
    image_path: str
    customer_id: str | None = None


def _calibration_or_400(customer_id: str):
    try:
        return cal_mod.load(customer_id)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e) + " The demo ships a 'sroie' profile - pass "
                            "customer_id='sroie', or run "
                            "`python -m verdict.calibrate <dir> --customer <name>` "
                            "on documents you certify as authentic.")


@app.post("/api/triage")
def api_triage(req: ImageRequest):
    path = pathlib.Path(req.image_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail="image not found on server")

    claim_id = f"CLM-{uuid.uuid4().hex[:8]}"
    res = triage.process_claim(str(path), claim_id=claim_id)
    _state.update(claim_id=claim_id, image_path=str(path), triage=res,
                 customer_id=req.customer_id or DEFAULT_CUSTOMER)
    res["claim_id"] = claim_id
    return res


@app.post("/api/probe")
def api_probe(req: ImageRequest):
    if not _state["claim_id"]:
        raise HTTPException(status_code=400, detail="run /api/triage first")
    cal = _calibration_or_400(_state.get("customer_id", DEFAULT_CUSTOMER))
    res = prober.process_probes(_state["image_path"], _state["claim_id"],
                                LEDGER, cal, triage=_state["triage"])
    return res


@app.get("/api/manifest")
def api_manifest():
    claim_id = _state["claim_id"]
    if not claim_id:
        return {"manifest": "", "cards": []}

    cards_list = []
    for card in LEDGER.cards(claim_id):
        web_path = None
        if card.crop_path:
            rel = os.path.relpath(card.crop_path, str(ROOT)).replace("\\", "/")
            web_path = f"http://localhost:8000/{rel}"
        cards_list.append({
            "id": card.id, "probe": card.probe, "params": card.params,
            "bbox": card.bbox, "px_on_target": card.px_on_target,
            "crop_sha256": card.crop_sha256, "crop_path": web_path,
            "observation": card.observation, "numeric": card.numeric,
            "cost_units": card.cost_units, "ts": card.ts,
        })
    return {"manifest": LEDGER.manifest(claim_id), "cards": cards_list}


@app.post("/api/adjudicate")
def api_adjudicate():
    if not _state["claim_id"]:
        raise HTTPException(status_code=400, detail="run /api/triage and /api/probe first")
    cal = _calibration_or_400(_state.get("customer_id", DEFAULT_CUSTOMER))
    res = adjudicator.adjudicate_from_ledger(LEDGER, _state["claim_id"], cal)
    res["claims"] = [c.__dict__ for c in res["claims"]]
    return res


@app.post("/api/verify")
def api_verify():
    if not _state["claim_id"]:
        raise HTTPException(status_code=400, detail="run /api/triage and /api/probe first")
    claim_id = _state["claim_id"]
    cal = _calibration_or_400(_state.get("customer_id", DEFAULT_CUSTOMER))
    cards = LEDGER.cards(claim_id)

    adj = adjudicator.adjudicate(cards, cal)
    img = cv2.imread(_state["image_path"])
    ver = verify_claims(adj["claims"], cards, LEDGER, claim_id, cal, img)

    adj_out = {**adj, "claims": [c.__dict__ for c in adj["claims"]]}
    return {"adjudication": adj_out, "verification": ver}


@app.post("/api/upload")
def api_upload(file: UploadFile = File(...)):
    try:
        data_dir = ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"image_path": f"data/{file.filename}", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset")
def api_reset():
    try:
        if _state["claim_id"]:
            LEDGER.purge(_state["claim_id"])
        _state.update(claim_id=None, image_path=None, triage=None)
        return {"status": "success", "message": "Session cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
