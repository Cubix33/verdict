"""Triage agent: runs cheap, deterministic probes and resolves easy cases with zero VLM calls.

Public API:
- process_claim(image_path: str, claim_text: str|None) -> dict

Decision logic (simple MVP):
- Run exif_audit, perceptual_hash, cross_field
- If cross_field reports delta != 0 and count>1 -> escalate (needs prober) with reason
- If exif missing and perceptual_hash matches existing ledger (not implemented here) -> escalate
- Otherwise: resolve as low_risk and return probe outputs

This is intentionally small and readable for the hackathon MVP.
"""
from __future__ import annotations
from typing import Optional
from ..probes import cheap
import pathlib
import json


def process_claim(image_path: str, claim_text: Optional[str] = None) -> dict:
    path = str(pathlib.Path(image_path))
    out = {"agent": "triage", "probes": {}, "decision": None, "reason": None}
    # exif
    ex = cheap.exif_audit(path)
    out["probes"]["exif_audit"] = ex
    # open image for pHash
    try:
        from PIL import Image
        pil = Image.open(path)
        ph = cheap.perceptual_hash(pil)
        out["probes"]["perceptual_hash"] = ph
    except Exception as e:
        out["probes"]["perceptual_hash"] = {"error": str(e)}
    # cross-field arithmetic
    cf = cheap.cross_field(path)
    out["probes"]["cross_field"] = cf

    # decision rules (MVP heuristics)
    if cf.get("ok") and cf.get("status") == "computed":
        delta = cf.get("delta", 0.0)
        count = cf.get("count", 0)
        # if there are at least two numeric fields and delta is substantial -> escalate
        if count >= 2 and abs(delta) > 0.5:
            out["decision"] = "escalate"
            out["reason"] = "arithmetic_mismatch"
            return out
    # if no exif and no phash match (we don't have a ledger yet) -> escalate if suspicious text mentions total
    if not ex.get("has_exif"):
        # heuristic: missing exif often suspicious for receipts; escalate for manual review
        out["decision"] = "escalate"
        out["reason"] = "missing_exif"
        return out
    # otherwise low risk
    out["decision"] = "resolved"
    out["reason"] = "low_risk"
    return out


# simple CLI for quick manual testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python verdict/agents/triage.py <image_path>")
        sys.exit(2)
    r = process_claim(sys.argv[1])
    print(json.dumps(r, indent=2))
