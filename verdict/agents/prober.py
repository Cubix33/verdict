"""Prober agent (MVP): runs cheap probes, creates EvidenceCards in the Ledger.

Public API:
- process_probes(image_path: str, claim_id: str|None = None) -> dict

CLI usage:
  python verdict\agents\prober.py <image_path> [--demo-corrupt]

The demo mode creates one or more cards and optionally corrupts the first crop to show provenance verification failing.
"""
from __future__ import annotations
import uuid, pathlib, json
import cv2
import numpy as np
from ..probes import cheap
from ..ledger import Ledger


LEDGER = Ledger(root="data", db="data/ledger.db")


def _make_card_id() -> str:
    return f"EV-{uuid.uuid4().hex[:8]}"


def process_probes(image_path: str, claim_id: str | None = None) -> dict:
    p = str(pathlib.Path(image_path))
    img_bgr = cv2.imread(p)
    if img_bgr is None:
        return {"ok": False, "reason": "imread_failed"}
    out = {"agent": "prober", "created_cards": []}
    # run cheap probes
    ex = cheap.exif_audit(p)
    ph = None
    try:
        from PIL import Image
        pil = Image.open(p)
        ph = cheap.perceptual_hash(pil)
    except Exception as e:
        ph = {"error": str(e)}
    cf = cheap.cross_field(p)
    boxes = cheap.find_numeric_boxes(img_bgr, min_h=12)

    # Decide a crop bbox: if numeric boxes exist, crop their union; else center crop (512x512)
    if boxes:
        xs = [b["bbox"][0] for b in boxes] + [b["bbox"][2] for b in boxes]
        ys = [b["bbox"][1] for b in boxes] + [b["bbox"][3] for b in boxes]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        # ensure within image bounds
        h, w = img_bgr.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        crop = img_bgr[y1:y2, x1:x2]
        bbox = [x1, y1, x2, y2]
    else:
        h, w = img_bgr.shape[:2]
        ch, cw = min(512, h), min(512, w)
        x1 = max(0, (w - cw) // 2); y1 = max(0, (h - ch) // 2)
        x2 = x1 + cw; y2 = y1 + ch
        crop = img_bgr[y1:y2, x1:x2]
        bbox = [x1, y1, x2, y2]

    # create one EvidenceCard for cross_field observation
    card_id = _make_card_id()
    obs = f"cross_field: {cf.get('status')}"
    numeric = {"stated_total": cf.get('stated_total'), "sum_line_items": cf.get('sum_line_items'), "delta": cf.get('delta')}
    card = LEDGER.add(claim_id or "unassigned", card_id, probe="cross_field", params={}, bbox=bbox, crop_bgr=crop, observation=obs, numeric=numeric, cost=1.0)
    out["created_cards"].append({"card_id": card.id, "sha": card.crop_sha256, "path": card.crop_path})

    # create another card for perceptual hash (full image crop)
    card_id2 = _make_card_id()
    obs2 = f"perceptual_hash: {ph.get('dhash_hex') if isinstance(ph, dict) else ph}"
    numeric2 = {"dhash_int": ph.get('dhash_int') if isinstance(ph, dict) else None}
    # store a small resized version as crop
    small = cv2.resize(img_bgr, (256, 256)) if img_bgr is not None else crop
    card2 = LEDGER.add(claim_id or "unassigned", card_id2, probe="perceptual_hash", params={}, bbox=None, crop_bgr=small, observation=obs2, numeric=numeric2, cost=0.0)
    out["created_cards"].append({"card_id": card2.id, "sha": card2.crop_sha256, "path": card2.crop_path})

    # record exif as a 0-cost card (no crop needed) -> store a 32x32 blank PNG with exif summary as observation
    card_id3 = _make_card_id()
    obs3 = f"exif: has_exif={ex.get('has_exif')}"
    # create a 32x32 white image as placeholder crop
    placeholder = 255 * (np.ones((32, 32, 3), dtype=np.uint8))
    card3 = LEDGER.add(claim_id or "unassigned", card_id3, probe="exif_audit", params=ex, bbox=None, crop_bgr=placeholder, observation=obs3, numeric={'has_exif': bool(ex.get('has_exif'))}, cost=0.0)
    out["created_cards"].append({"card_id": card3.id, "sha": card3.crop_sha256, "path": card3.crop_path})

    return out


# CLI demo with optional corruption to test verify_provenance
if __name__ == "__main__":
    import sys
    import argparse
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--corrupt-first', action='store_true', help='Corrupt the first crop file to demonstrate provenance failure')
    args = ap.parse_args()
    res = process_probes(args.image)
    print(json.dumps(res, indent=2))
    # verify provenance for created cards
    for c in res.get('created_cards', []):
        # find card_id and map to ledger cache lookup
        card_obj = LEDGER._cache.get(c['card_id'])
        ok = LEDGER.verify_provenance(c['card_id']) if card_obj else False
        print(f"Card {c['card_id']} provenance ok: {ok}  path: {c['path']}")
    if args.corrupt_first and res.get('created_cards'):
        first = res['created_cards'][0]
        p = pathlib.Path(first['path'])
        b = p.read_bytes()
        # flip a byte in the middle
        arr = bytearray(b)
        if len(arr) > 50:
            arr[50] = (arr[50] + 1) % 256
            p.write_bytes(bytes(arr))
            print('Corrupted first crop on disk to demonstrate provenance mismatch.')
            card_obj = LEDGER._cache.get(first['card_id'])
            print('verify_provenance after corruption:', LEDGER.verify_provenance(first['card_id']))
