"""DCT quantization table extractor (MVP).
This reads quantization tables from JPEGs using Pillow where available and returns a stable hash.
"""
from __future__ import annotations
from PIL import Image
import hashlib, json
import pathlib


def extract_qtable_hash(path: str) -> dict:
    p = pathlib.Path(path)
    try:
        img = Image.open(p)
    except Exception as e:
        return {"ok": False, "reason": f"open_failed:{e}"}
    out = {"ok": True, "is_jpeg": img.format == 'JPEG', "qtable_hash": None}
    if img.format == 'JPEG':
        q = getattr(img, 'quantization', None) or img.info.get('quantization')
        if q:
            try:
                s = json.dumps(q, sort_keys=True)
                h = hashlib.sha256(s.encode('utf-8')).hexdigest()
                out['qtable_hash'] = h
            except Exception:
                out['qtable_hash'] = None
        else:
            out['qtable_hash'] = None
    return out
