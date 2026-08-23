"""PROBER - spends a budget acquiring evidence. Plans, but cannot decide.

The original prober called only the cheap probes and then wrote three fixed
cards: a crop of every number on the page, a 256x256 thumbnail, and a blank
32x32 white square standing in for the EXIF result. None of the forensic
modules were ever invoked, so `ela.py`, `noise.py`, `dct_quant.py` and
`typography.py` sat unused and the Adjudicator's ELA branch was unreachable
code. The system issued verdicts with no forensic evidence behind them.

This prober selects candidate regions, runs the real probes on each, and
registers one card per region carrying every calibrated statistic that region
was large enough to support. Probes that cannot resolve a region are skipped
rather than guessed - idea.md Finding 6: gate on measured pixels, never on
confidence.
"""
from __future__ import annotations

import json
import pathlib
import uuid

import cv2
import numpy as np

from .. import calibrate as cal_mod
from ..probes import cheap, compression, copymove, geometric, noise, ocr, typography
from ..probes.ocr import parse_amount

COST = {"zoom": 1.0, "ela": 1.0, "dct_quant": 1.0, "noise_residual": 2.0,
        "font_metrics": 1.0, "copy_move": 2.0, "block_grid": 1.0, "cross_field": 0.0,
        "exif_audit": 0.0, "perceptual_hash": 0.0}

MAX_REGIONS = 14


def _card_id(n: int) -> str:
    return f"EV-{n}"


def propose_anomaly_regions(emap, vmap, refs, page, energy=None) -> list[dict]:
    """Regions the pixels themselves nominate, independent of what OCR read.

    OCR-driven selection has a blind spot that matters: a retyped or erased
    field often no longer reads as an amount - or as anything - so the very
    region that was tampered with never gets proposed. Screening the residual
    maps directly closes that hole, and it is also how the Prober earns its
    keep, by looking where the evidence is rather than where the text is.
    """
    if emap is None or not refs:
        return []
    stats = []
    for b in refs:
        x1, y1, x2, y2 = b
        e = compression._region_stat(emap, energy, (x1, y1, x2, y2))
        if e is None:
            continue
        vsub = vmap[y1:y2, x1:x2]
        stats.append((e, float(np.median(vsub)), b))
    if len(stats) < 6:
        return []

    e_vals = np.asarray([s[0] for s in stats])
    v_vals = np.asarray([s[1] for s in stats])
    e_med, v_med = float(np.median(e_vals)), float(np.median(v_vals))
    e_mad = float(np.median(np.abs(e_vals - e_med))) * 1.4826 or 1e-6
    v_mad = float(np.median(np.abs(v_vals - v_med))) * 1.4826 or 1e-6

    scored = []
    for e, v, b in stats:
        # Either tail is interesting: unusual compression residual, or texture
        # that is unusually smooth (a fill) or unusually rough (a composite).
        score = max(abs(e - e_med) / e_mad, abs(v - v_med) / v_mad)
        scored.append((score, b))
    scored.sort(key=lambda s: -s[0])
    return [{"bbox": list(b), "why": f"residual_outlier(z={s:.1f})"}
            for s, b in scored[:MAX_REGIONS] if s >= 2.0]


def select_regions(img_bgr, page, triage: dict, target_bbox=None,
                   emap=None, vmap=None, refs=None, energy=None) -> list[dict]:
    """Choose where to spend the budget, most informative first.

    The arithmetic target comes first because it is the only region the system
    can already name a motive for. Pixel outliers come next, because they do
    not depend on the tampered text remaining legible. Amounts follow, since
    they are what forgers edit.
    """
    regions: list[dict] = []
    seen: set[tuple] = set()

    def push(bbox, why):
        key = tuple(int(v) for v in bbox)
        if key in seen or key[2] <= key[0] or key[3] <= key[1]:
            return
        seen.add(key)
        regions.append({"bbox": list(key), "why": why})

    if target_bbox:
        push(target_bbox, "arithmetic_target")

    for r in propose_anomaly_regions(emap, vmap, refs or [], page, energy):
        push(r["bbox"], r["why"])

    amounts = sorted(((w, v) for w, v in page.amount_words() if w.height >= 10),
                     key=lambda wv: -(wv[0].height * wv[0].width))
    for w, _ in amounts:
        push(w.bbox, "amount")

    if len(regions) < 4:
        for w in sorted(page.words, key=lambda w: -(w.height * w.width)):
            push(w.bbox, "large_text")

    return regions[:MAX_REGIONS]


def process_probes(image_path: str, claim_id: str, ledger, cal,
                   triage: dict | None = None, budget: float = 90.0) -> dict:
    """Run the forensic suite and register one evidence card per region."""
    p = str(pathlib.Path(image_path))
    img = cv2.imread(p)
    if img is None:
        return {"ok": False, "reason": "imread_failed", "created_cards": []}

    triage = triage or {}
    out = {"agent": "prober", "ok": True, "created_cards": [],
           "budget": budget, "budget_spent": 0.0, "regions": []}

    page = ocr.read(img)
    refs = ocr.text_regions(page, img)

    # Whole-page maps, computed once and shared by every region. They are also
    # what lets the Prober nominate regions the OCR never named.
    emap, ela_stats = compression.ela_map(img)
    energy = compression.edge_energy(img)
    res = noise.residual_map(img)
    vmap = noise.local_variance_map(res)
    mins = cal.px_on_target_minimums

    regions = select_regions(img, page, triage, triage.get("target_bbox"),
                             emap=emap, vmap=vmap, refs=refs, energy=energy)
    if not regions:
        out["reason"] = "no_regions_of_interest"
        return out

    spent = 0.0
    n = 0
    dropped = 0
    cards = []

    def emit(probe, params, bbox, crop, obs, numeric, cost, px):
        nonlocal spent, n
        n += 1
        spent += cost
        card = ledger.add(claim_id, _card_id(n), probe, params, bbox, crop,
                          obs, numeric, cost, px_on_target=px)
        cards.append(card)
        out["created_cards"].append({"card_id": card.id, "probe": probe,
                                     "sha": card.crop_sha256, "path": card.crop_path,
                                     "bbox": bbox, "px_on_target": card.px_on_target})
        return card

    # --- one card per region, carrying every statistic it can support -----
    for reg in regions:
        if spent >= budget:
            # Never truncate silently: a run that ran out of budget covered
            # less of the page than the report implies, and the reader has to
            # be able to see that.
            dropped += 1
            continue
        bbox = reg["bbox"]
        crop, zi = geometric.zoom(img, bbox)
        if crop is None:
            continue
        px = zi["px_on_target"]
        area = zi["area_px"]

        numeric: dict = {"px_on_target": px, "area_px": area}
        notes: list[str] = []
        cost = COST["zoom"]

        if emap is not None and area >= mins.get("ela", 120):
            z = compression.robust_region_z(emap, bbox, refs, energy)
            if z.get("ok"):
                numeric["ela_z"] = z["z"]
                numeric["ela_exceeds"] = bool(cal.exceeds("ela_z", z["z"]))
                notes.append(f"ELA residual z={z['z']}")
                cost += COST["ela"]

        if area >= mins.get("noise_residual", 120):
            hf = noise.region_hf_ratio(vmap, bbox, refs)
            if hf.get("ok"):
                numeric["hf_ratio"] = hf["hf_ratio"]
                numeric["hf_exceeds"] = bool(cal.exceeds("hf_ratio", hf["hf_ratio"]))
                notes.append(f"high-frequency energy {hf['hf_ratio']}x peer median")
                cost += COST["noise_residual"]
            flat = noise.flatness(img, bbox)
            if flat.get("ok"):
                numeric["flat_pixel_frac"] = flat["flat_pixel_frac"]
                numeric["flat_exceeds"] = bool(
                    cal.exceeds("flat_pixel_frac", flat["flat_pixel_frac"]))
                notes.append(f"{flat['flat_pixel_frac']:.0%} of pixels have no gradient")

        if area >= mins.get("block_grid", 576):
            grid = compression.block_grid_energy(img, bbox)
            if grid.get("ok"):
                numeric["grid_ratio"] = grid["grid_ratio"]
                numeric["grid_exceeds"] = bool(cal.exceeds("grid_ratio", grid["grid_ratio"]))
                cost += COST["block_grid"]

        if area >= mins.get("copy_move", 200):
            dup = copymove.region_duplicate_score(img, bbox)
            if dup.get("ok"):
                numeric["dup_score"] = dup["best_score"]
                numeric["dup_distance_px"] = dup["distance_px"]
                numeric["dup_exceeds"] = bool(cal.exceeds("dup_score", dup["best_score"]))
                if numeric["dup_exceeds"]:
                    numeric["dup_match_bbox"] = dup["match_bbox"]
                    notes.append(f"near-identical copy at {dup['match_bbox']}")
                cost += COST["copy_move"]

        if area >= mins.get("font_metrics", 400):
            typo = typography.region_metrics(img, bbox)
            if typo.get("ok") and typo.get("scope") == "row":
                numeric["stroke_ratio"] = typo["stroke_ratio"]
                numeric["baseline_z"] = typo["baseline_z"]
                numeric["stroke_exceeds"] = bool(
                    cal.exceeds("stroke_ratio", typo["stroke_ratio"]))
                numeric["baseline_exceeds"] = bool(
                    cal.exceeds("baseline_z", typo["baseline_z"]))
                notes.append(f"stroke width {typo['stroke_ratio']}x its row")
                cost += COST["font_metrics"]
        else:
            numeric["font_metrics_skipped"] = "below_px_minimum"

        text = _text_at(page, bbox)
        obs = f"region {bbox} ({text or 'no text read'}): " + "; ".join(notes) \
            if notes else f"region {bbox} ({text or 'no text read'}): no anomaly measured"
        emit("region_forensics", {"why": reg["why"], "text": text}, bbox, crop,
             obs, numeric, cost, px)

    # --- document-level cards --------------------------------------------
    cf = triage.get("probes", {}).get("cross_field") or cheap.cross_field(p, img_bgr=img)
    if cf.get("status") == "mismatch":
        tb = cf.get("total_bbox")
        crop, zi = geometric.zoom(img, tb) if tb else (None, {"px_on_target": 0})
        if crop is None:
            crop, zi = img, {"px_on_target": img.shape[0]}
        emit("cross_field", {"reconciliation": cf.get("reconciliation")}, tb, crop,
             f"stated total {cf['stated_total']} vs line items "
             f"{cf['sum_line_items']} (delta {cf['delta']}, tolerance {cf['tolerance']})",
             {"stated_total": cf["stated_total"], "sum_line_items": cf["sum_line_items"],
              "delta": cf["delta"], "rel_error": cf["rel_error"],
              "tolerance": cf["tolerance"], "item_count": cf["item_count"],
              "arithmetic_mismatch": True},
             COST["cross_field"], zi["px_on_target"])

    ex = triage.get("probes", {}).get("exif_audit") or cheap.exif_audit(p)
    q = compression.dct_quant(p)
    in_bank = bool(q.get("qtable_hash")) and q["qtable_hash"] in (cal.qtable_bank or [])
    thumb = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    emit("provenance", {"exif": ex, "qtable": q}, None, thumb,
         f"exif={'present' if ex.get('has_exif') else 'absent'}"
         + (f", editor tag {ex['editor_signature']}" if ex.get("editor_signature") else "")
         + (f", qtable {'in' if in_bank else 'not in'} customer bank"
            if q.get("qtable_hash") else ", no qtable (not a JPEG)"),
         {"has_exif": bool(ex.get("has_exif")),
          "editor_signature": ex.get("editor_signature"),
          "datetime_conflict": bool(ex.get("datetime_conflict")),
          "qtable_hash": q.get("qtable_hash"),
          "qtable_in_bank": in_bank,
          "qtable_bank_known": bool(cal.qtable_bank)},
         COST["dct_quant"], 0)

    out["budget_spent"] = round(spent, 2)
    out["regions"] = regions
    out["regions_probed"] = len(regions) - dropped
    out["regions_dropped_over_budget"] = dropped
    out["ela_page"] = ela_stats
    return out


def _text_at(page, bbox) -> str:
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    for w in page.words:
        x1, y1, x2, y2 = w.bbox
        if x1 - 4 <= cx <= x2 + 4 and y1 - 4 <= cy <= y2 + 4:
            return w.text
    return ""


if __name__ == "__main__":
    import sys
    from ..ledger import Ledger
    if len(sys.argv) < 2:
        print("Usage: python -m verdict.agents.prober <image> [customer]")
        sys.exit(2)
    customer = sys.argv[2] if len(sys.argv) > 2 else "default"
    led = Ledger()
    cid = f"CLM-{uuid.uuid4().hex[:8]}"
    r = process_probes(sys.argv[1], cid, led, cal_mod.load(customer))
    print(json.dumps(r, indent=2, default=str))
    for c in r["created_cards"]:
        print(f"  {c['card_id']} provenance ok:",
              led.verify_provenance(cid, c["card_id"]))
