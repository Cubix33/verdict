"""FastAPI backend for the VERDICT dashboard.
Provides simple endpoints to run triage, prober, adjudication and verification on server-side images.

Notes:
- For MVP this accepts an image_path on the server (deferred: file uploads)
- CORS is enabled for a React dev server on localhost:3000
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pathlib
import sys
import shutil

# Ensure the project root (two levels up) is on sys.path so 'verdict' package is importable
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verdict.agents.triage import process_claim
from verdict.agents.prober import process_probes, LEDGER
from verdict.agents.adjudicator import adjudicate_from_ledger
from verdict.agents.verifier import verify_claims

from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files to serve the crops and data images
app.mount("/data", StaticFiles(directory=str(ROOT / "data")), name="data")

class ImageRequest(BaseModel):
    image_path: str


@app.post('/api/triage')
def api_triage(req: ImageRequest):
    path = pathlib.Path(req.image_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail='image not found on server')
    res = process_claim(str(path))
    return res


@app.post('/api/probe')
def api_probe(req: ImageRequest):
    path = pathlib.Path(req.image_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail='image not found on server')
    res = process_probes(str(path))
    return res


@app.get('/api/manifest')
def api_manifest():
    cards_list = []
    for k, v in LEDGER._cache.items():
        # Make crop path web-accessible
        web_path = None
        if v.crop_path:
            rel_path = os.path.relpath(v.crop_path, str(ROOT)).replace("\\", "/")
            web_path = f"http://localhost:8000/{rel_path}"
        
        cards_list.append({
            "id": v.id,
            "probe": v.probe,
            "params": v.params,
            "bbox": v.bbox,
            "px_on_target": v.px_on_target,
            "crop_sha256": v.crop_sha256,
            "crop_path": web_path,
            "observation": v.observation,
            "numeric": v.numeric,
            "cost_units": v.cost_units,
            "ts": v.ts
        })
    return {"manifest": LEDGER.manifest(), "cards": cards_list}


@app.post('/api/adjudicate')
def api_adjudicate():
    res = adjudicate_from_ledger(LEDGER)
    return res


@app.post('/api/verify')
def api_verify():
    adjud = adjudicate_from_ledger(LEDGER)
    claims = adjud.get('claims', [])
    res = verify_claims(LEDGER, claims)
    return {'adjudication': adjud, 'verification': res}


@app.post('/api/upload')
def api_upload(file: UploadFile = File(...)):
    try:
        # ensure data directory exists
        data_dir = ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Return path relative to project root
        return {"image_path": f"data/{file.filename}", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/reset')
def api_reset():
    try:
        # Clear SQLite tables
        LEDGER.db.execute("DELETE FROM cards")
        LEDGER.db.execute("DELETE FROM claims")
        LEDGER.db.commit()
        
        # Clear in-memory cache
        LEDGER._cache.clear()
        
        # Delete crop files
        crops_dir = ROOT / "data" / "crops"
        if crops_dir.exists():
            for f in crops_dir.glob("*"):
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass
        return {"status": "success", "message": "Ledger cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
