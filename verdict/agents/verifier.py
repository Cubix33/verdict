"""VERIFIER - the skeptic. Re-derives each claim from its own evidence alone.

A bond holds only if all four of these pass.

  provenance   every cited card exists and its PNG still hashes to the value
               recorded when it was written
  reproduction the cited statistic is recomputed from the crop on disk, and
               must land in the same place and still exceed the calibrated
               bound
  blank        the same probe on a flat grey patch of identical size must NOT
               reproduce the finding
  shuffle      the same probe on several random same-size regions of this same
               document must NOT reproduce the finding

The shuffle control is the sharpest of the four. If a random patch of the page
trips the same threshold, the finding was never about the cited region - it
was about the probe being trigger-happy on this particular document. That is
lazy perception caught at inference time, and it is the reason a claim can be
struck even when it happens to be true.

The original verifier did none of this. Its cross-field bond only asked
whether OCR could find any digit inside the crop, which is true of every
receipt crop and tests nothing about the arithmetic. Its EXIF bond was
inverted - it held only when `has_exif` was True, so a claim founded on
missing metadata always broke. Its perceptual-hash bond compared a dHash
against the first sixteen characters of a SHA-256 and then wrote
`or True`, so the comparison could not fail. Bonds broke and held for reasons
unrelated to the claims they were meant to test.
"""
from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np

from ..probes import compression, copymove, geometric, noise, ocr, typography

# How many random same-size regions the shuffle control draws, and how many of
# them may reproduce the finding before the bond is judged unearned.
SHUFFLE_DRAWS = 6
SHUFFLE_TOLERANCE = 1

# Which recomputed metric backs which claimed signal.
SIGNAL_METRIC = {
    "ela_exceeds": "ela_z",
    "grid_exceeds": "grid_ratio",
    "hf_exceeds": "hf_ratio",
    "flat_exceeds": "flat_pixel_frac",
    "dup_exceeds": "dup_score",
    "stroke_exceeds": "stroke_ratio",
    "baseline_exceeds": "baseline_z",
}


def _measure(img, bbox, refs, ctx, metric: str):
    """Recompute one metric for one region, from scratch."""
    if metric == "ela_z":
        if ctx.get("ela") is None:
            return None
        return compression.robust_region_z(
            ctx["ela"], bbox, refs, ctx.get("energy")).get("z")
    if metric == "hf_ratio":
        r = noise.region_hf_ratio(ctx["var"], bbox, refs)
        return r.get("hf_ratio") if r.get("ok") else None
    if metric == "flat_pixel_frac":
        r = noise.flatness(img, bbox)
        return r.get("flat_pixel_frac") if r.get("ok") else None
    if metric == "grid_ratio":
        r = compression.block_grid_energy(img, bbox)
        return r.get("grid_ratio") if r.get("ok") else None
    if metric == "dup_score":
        r = copymove.region_duplicate_score(img, bbox)
        return r.get("best_score") if r.get("ok") else None
    if metric in ("stroke_ratio", "baseline_z"):
        r = typography.region_metrics(img, bbox)
        return r.get(metric) if r.get("ok") else None
    return None


def verify_claim(claim, cards_by_id, ledger, claim_id, cal, img, ctx, rng) -> dict:
    """Test one bonded claim. Returns the detail written back onto the claim."""
    detail: dict = {"checks": {}, "cited": list(claim.bond)}

    # --- 1. provenance -----------------------------------------------------
    for cid in claim.bond:
        card = cards_by_id.get(cid)
        if card is None:
            detail["reason"] = f"cited card {cid} is not in this claim's evidence"
            detail["checks"]["provenance"] = False
            return detail
        if not ledger.verify_provenance(claim_id, cid):
            detail["reason"] = f"crop bytes for {cid} no longer match its recorded hash"
            detail["checks"]["provenance"] = False
            return detail
    detail["checks"]["provenance"] = True

    signals = claim.bond_detail.get("signals", [])
    metrics = [SIGNAL_METRIC[s] for s in signals if s in SIGNAL_METRIC]

    # --- an arithmetic-only claim is re-derived from the numbers ----------
    if not metrics:
        return _verify_non_pixel(claim, cards_by_id, detail)

    region_card = next((cards_by_id[c] for c in claim.bond
                        if cards_by_id[c].probe == "region_forensics"), None)
    if region_card is None or not region_card.bbox:
        detail["reason"] = "claim cites no region card, so it cannot be re-measured"
        detail["checks"]["evidence_type"] = False
        return detail
    detail["checks"]["evidence_type"] = True

    bbox = region_card.bbox
    refs = ctx["refs"]

    # --- 2. reproduction ---------------------------------------------------
    reproduced, observed = [], {}
    for m in metrics:
        v = _measure(img, bbox, refs, ctx, m)
        observed[m] = None if v is None else round(float(v), 4)
        if v is not None and cal.exceeds(m, v):
            reproduced.append(m)
    detail["recomputed"] = observed
    detail["checks"]["reproduction"] = bool(reproduced)
    if not reproduced:
        detail["reason"] = ("re-measuring the cited region did not reproduce any "
                            "flagged statistic")
        return detail

    # --- 3. blank control --------------------------------------------------
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    grey = np.full((max(h, 8), max(w, 8), 3), 128, np.uint8)
    grey_hits = []
    for m in reproduced:
        if m in ("ela_z", "hf_ratio"):
            continue           # relative metrics need peers; grey has none
        v = _measure(grey, [0, 0, grey.shape[1], grey.shape[0]], refs, ctx, m)
        if v is not None and cal.exceeds(m, v):
            grey_hits.append(m)
    detail["checks"]["blank_control"] = not grey_hits
    if grey_hits:
        detail["reason"] = (f"a flat grey patch of the same size also trips "
                            f"{grey_hits} - the finding is not about this region")
        detail["blank_hits"] = grey_hits
        return detail

    # --- 4. shuffle control ------------------------------------------------
    # Draw the decoys from other *text* regions of this document rather than
    # from anywhere on the page. A patch of blank margin fails every probe
    # trivially, so a control built on random coordinates is one the claim
    # always passes and therefore tests nothing. Peer text regions are the
    # population the finding implicitly claims this region does not belong to,
    # so they are what it has to be distinguishable from.
    survivors = 0
    draws = []
    peers = [b for b in refs if list(b) != list(bbox)]
    if len(peers) >= SHUFFLE_DRAWS:
        idx = rng.choice(len(peers), size=SHUFFLE_DRAWS, replace=False)
        decoys = [peers[int(i)] for i in idx]
    else:
        decoys = [b for _, b in
                  ((geometric.random_same_size_region(img, bbox, rng))
                   for _ in range(SHUFFLE_DRAWS)) if b is not None]
    for rbox in decoys:
        hits = [m for m in reproduced
                if (v := _measure(img, rbox, refs, ctx, m)) is not None
                and cal.exceeds(m, v)]
        draws.append({"bbox": list(rbox), "hits": hits})
        if hits:
            survivors += 1
    detail["shuffle_draws"] = len(draws)
    detail["shuffle_survivors"] = survivors
    detail["checks"]["shuffle_control"] = survivors <= SHUFFLE_TOLERANCE
    if survivors > SHUFFLE_TOLERANCE:
        detail["reason"] = (f"{survivors} of {len(draws)} random regions of this "
                            f"same document reproduce the finding - the claim does "
                            f"not depend on its cited evidence")
        return detail

    detail["reason"] = (f"independently re-derived {reproduced} from the cited crop; "
                        f"survived blank and shuffle controls")
    return detail


def _verify_non_pixel(claim, cards_by_id, detail: dict) -> dict:
    """Bond test for claims that rest on arithmetic or metadata, not pixels."""
    kind = claim.bond_detail.get("rule")
    if kind == "editor_tag":
        card = cards_by_id[claim.bond[0]]
        sig = (card.numeric or {}).get("editor_signature")
        detail["checks"]["reproduction"] = bool(sig)
        detail["reason"] = (f"editor tag '{sig}' is present on the cited card"
                            if sig else "cited card carries no editor tag")
        return detail

    card = next((cards_by_id[c] for c in claim.bond
                 if cards_by_id[c].probe == "cross_field"), None)
    if card is None:
        detail["checks"]["evidence_type"] = False
        detail["reason"] = "claim cites no evidence of a type that could support it"
        return detail

    n = card.numeric or {}
    holds = bool(n.get("arithmetic_mismatch")) and abs(n.get("delta", 0)) > n.get("tolerance", 0)
    detail["checks"]["reproduction"] = holds
    detail["reason"] = (f"stated total minus line items is {n.get('delta')}, beyond "
                        f"the {n.get('tolerance')} tolerance"
                        if holds else "the arithmetic reconciles on re-check")
    return detail


def verify_claims(claims, cards, ledger, claim_id, cal, img, seed: int = 0) -> dict:
    """Test every claim. Each is judged only against the evidence it cited."""
    rng = np.random.default_rng(seed)
    cards_by_id = {c.id: c for c in cards}

    page = ocr.read(img)
    ctx = {"refs": ocr.text_regions(page, img),
           "ela": compression.ela_map(img)[0],
           "energy": compression.edge_energy(img),
           "var": noise.local_variance_map(noise.residual_map(img))}

    results = []
    for claim in claims:
        detail = verify_claim(claim, cards_by_id, ledger, claim_id, cal, img, ctx, rng)
        holds = all(v for v in detail["checks"].values())
        claim.bond_status = "holds" if holds else "broken"
        claim.bond_detail = {**claim.bond_detail, **detail}
        results.append({"id": claim.id, "overall": claim.bond_status,
                        "text": claim.text, "detail": detail})

    return {"agent": "verifier", "claims": results,
            "holds": sum(1 for r in results if r["overall"] == "holds"),
            "broken": sum(1 for r in results if r["overall"] == "broken")}


if __name__ == "__main__":
    import sys
    from .. import calibrate as cal_mod
    from ..ledger import Ledger
    from .adjudicator import adjudicate
    if len(sys.argv) < 3:
        print("Usage: python -m verdict.agents.verifier <claim_id> <image> [customer]")
        sys.exit(2)
    led = Ledger()
    cid, img_path = sys.argv[1], sys.argv[2]
    cal = cal_mod.load(sys.argv[3] if len(sys.argv) > 3 else "default")
    cards = led.cards(cid)
    adj = adjudicate(cards, cal)
    res = verify_claims(adj["claims"], cards, led, cid, cal, cv2.imread(img_path))
    print(json.dumps(res, indent=2, default=str))
