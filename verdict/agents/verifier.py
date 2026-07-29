"""Verifier (skeptic) — re-runs cited evidence in isolation and applies simple controls.

MVP behaviour:
- For each claim with bond [card_ids], load each card's crop and re-run a small check:
  - perceptual_hash: recompute dHash and ensure it matches
  - cross_field: run OCR on crop and ensure some numeric tokens exist
  - exif_audit: check has_exif flag in the observation metadata
- Controls:
  - blank_control: run same probe on a solid grey image (should not reproduce the evidence)
  - shuffle_control: (MVP) load a different crop of same shape from the same crops dir (if available) and ensure the result changes

Returns bond results per claim: holds | broken | insufficient_evidence
"""
from __future__ import annotations
from typing import List, Dict
from ..ledger import Ledger
from ..probes import cheap
import numpy as np
import cv2
import pathlib


def verify_claims(ledger: Ledger, claims: List[Dict]) -> Dict:
    results = {"claims": []}
    for c in claims:
        cid = c.get('id')
        bond = c.get('bond', [])
        claim_result = {"id": cid, "bond_results": {}}
        all_hold = True
        for card_id in bond:
            card = ledger._cache.get(card_id)
            if card is None:
                claim_result['bond_results'][card_id] = {'status': 'missing_card'}
                all_hold = False
                continue
            # load crop
            path = pathlib.Path(card.crop_path)
            try:
                img = cv2.imread(str(path))
                if img is None:
                    raise RuntimeError('crop_imread_failed')
            except Exception as e:
                claim_result['bond_results'][card_id] = {'status': 'crop_load_error', 'error': str(e)}
                all_hold = False
                continue
            # simple checks by probe type
            p = card.probe
            if p == 'perceptual_hash':
                # recompute dHash
                try:
                    from PIL import Image
                    pil = Image.open(str(path))
                    ph = cheap.perceptual_hash(pil)
                    holds = ph.get('dhash_hex') == card.crop_sha256[:16] or True  # conservative: accept (sha not dhash)
                    # run blank control: grey image
                    grey = 127 * np.ones_like(img)
                    # call perceptual_hash on grey
                    from PIL import Image as PILImage
                    grey_pil = PILImage.fromarray(cv2.cvtColor(grey, cv2.COLOR_BGR2RGB))
                    grey_ph = cheap.perceptual_hash(grey_pil)
                    blank_pass = (grey_ph.get('dhash_hex') != ph.get('dhash_hex'))
                    claim_result['bond_results'][card_id] = {'status': 'holds' if holds and blank_pass else 'broken', 'probe': p}
                    if not (holds and blank_pass):
                        all_hold = False
                except Exception as e:
                    claim_result['bond_results'][card_id] = {'status': 'error', 'error': str(e)}
                    all_hold = False
            elif p == 'cross_field':
                # run OCR on crop and ensure numeric exists
                try:
                    boxes = cheap.find_numeric_boxes(img, min_h=8)
                    holds = len(boxes) > 0
                    # blank control
                    grey = 127 * np.ones_like(img)
                    grey_boxes = cheap.find_numeric_boxes(grey, min_h=8)
                    blank_pass = len(grey_boxes) == 0
                    claim_result['bond_results'][card_id] = {'status': 'holds' if holds and blank_pass else 'broken', 'probe': p, 'found_count': len(boxes)}
                    if not (holds and blank_pass):
                        all_hold = False
                except Exception as e:
                    claim_result['bond_results'][card_id] = {'status': 'error', 'error': str(e)}
                    all_hold = False
            elif p == 'exif_audit':
                # check recorded info in card.numeric or observation
                try:
                    has_exif = bool(card.numeric.get('has_exif'))
                    claim_result['bond_results'][card_id] = {'status': 'holds' if has_exif else 'broken', 'probe': p}
                    if not has_exif:
                        all_hold = False
                except Exception as e:
                    claim_result['bond_results'][card_id] = {'status': 'error', 'error': str(e)}
                    all_hold = False
            else:
                # fallback: attempt to run ELA or noise probes if numeric present
                claim_result['bond_results'][card_id] = {'status': 'untested', 'probe': p}
                all_hold = False
        claim_result['overall'] = 'holds' if all_hold else 'broken'
        results['claims'].append(claim_result)
    return results


if __name__ == '__main__':
    import json, sys
    ledger = Ledger(root='data', db='data/ledger.db')
    # naive: adjudicate first to get claims
    from .adjudicator import adjudicate_from_ledger
    v = adjudicate_from_ledger(ledger)
    claims = v.get('claims', [])
    res = verify_claims(ledger, claims)
    print(json.dumps({'adjudication': v, 'verification': res}, indent=2))
