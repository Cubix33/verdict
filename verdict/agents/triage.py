"""TRIAGE - cheap deterministic probes only. Zero model calls.

Triage produces a *risk assessment*, never an accusation. That separation is
the fix for the reported failure. The previous version escalated any file
without EXIF and called the reason `missing_exif`, and since PNGs, scanner
output, screenshots and anything routed through a messaging app all lack EXIF,
every document escalated - after which the Adjudicator turned the escalation
into a tamper claim. A signal that fires on everything carries no information,
and turning one into an accusation is how a detector reaches a 100% false
positive rate.

What triage now does with EXIF absence is record it and move on. What it
escalates on is structure: arithmetic that does not reconcile, a duplicate of
a document already on file, or an explicit editor tag.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

import cv2

from ..probes import cheap

# Weights are risk, not guilt. Escalation buys a closer look; only the
# Adjudicator, working from probe evidence, may accuse.
RISK_WEIGHTS = {
    "duplicate_submission": 0.90,
    "arithmetic_mismatch": 0.55,
    "editor_signature": 0.45,
    "datetime_conflict": 0.30,
    "no_exif": 0.05,          # near-zero on purpose: true of most honest uploads
}
ESCALATE_AT = 0.25


def process_claim(image_path: str, claim_text: Optional[str] = None,
                  phash_index: Optional[dict] = None,
                  claim_id: Optional[str] = None) -> dict:
    """Run the free probes and decide whether the case deserves the budget."""
    path = str(pathlib.Path(image_path))
    out = {"agent": "triage", "vlm_calls": 0, "probes": {},
           "flags": [], "risk": 0.0, "decision": None, "reason": None,
           "target_bbox": None}

    img = cv2.imread(path)
    if img is None:
        out.update(decision="error", reason="imread_failed")
        return out

    ex = cheap.exif_audit(path)
    out["probes"]["exif_audit"] = ex

    ph_hex = None
    try:
        from PIL import Image
        ph = cheap.perceptual_hash(Image.open(path))
        out["probes"]["perceptual_hash"] = ph
        ph_hex = ph["dhash_hex"]
    except Exception as e:
        out["probes"]["perceptual_hash"] = {"error": str(e)}

    cf = cheap.cross_field(path, img_bgr=img)
    out["probes"]["cross_field"] = cf

    risk = 0.0
    flags: list[str] = []

    # --- duplicate submission -------------------------------------------
    duplicates = []
    if ph_hex and phash_index:
        mine = int(ph_hex, 16)
        for other_id, other_hex in phash_index.items():
            if other_id == claim_id:
                continue
            d = cheap.hamming(mine, int(other_hex, 16))
            if d <= 6:
                duplicates.append({"claim_id": other_id, "distance": d})
    if duplicates:
        flags.append(f"duplicate_submission:{duplicates[0]['claim_id']}")
        risk += RISK_WEIGHTS["duplicate_submission"]
    out["duplicates"] = duplicates

    # --- arithmetic -------------------------------------------------------
    # Only a genuine reconciliation failure counts. `insufficient_structure`
    # means the document was not legible as a receipt, which is a statement
    # about our OCR, not about the document's honesty.
    if cf.get("status") == "mismatch":
        flags.append("arithmetic_mismatch")
        risk += RISK_WEIGHTS["arithmetic_mismatch"]
        # Point the Prober at the field the arithmetic implicates.
        out["target_bbox"] = cf.get("total_bbox")

    # --- provenance -------------------------------------------------------
    if ex.get("editor_signature"):
        flags.append(f"editor_signature:{ex['editor_signature']}")
        risk += RISK_WEIGHTS["editor_signature"]
    if ex.get("datetime_conflict"):
        flags.append("datetime_conflict")
        risk += RISK_WEIGHTS["datetime_conflict"]
    if not ex.get("has_exif"):
        # Recorded for the audit trail. Deliberately almost weightless: on its
        # own this must never be enough to escalate, let alone to accuse.
        flags.append("no_exif")
        risk += RISK_WEIGHTS["no_exif"]

    out["flags"] = flags
    out["risk"] = round(min(risk, 1.0), 3)
    out["phash"] = ph_hex

    if risk >= ESCALATE_AT:
        out["decision"] = "escalate"
        out["reason"] = next((f for f in flags if not f.startswith("no_exif")), "risk")
    else:
        out["decision"] = "resolved"
        out["reason"] = "low_risk"
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m verdict.agents.triage <image_path>")
        sys.exit(2)
    print(json.dumps(process_claim(sys.argv[1]), indent=2, default=str))
