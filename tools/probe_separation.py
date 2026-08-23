"""PLAN.md section 3 acceptance gate.

    "On 10 forged + 10 authentic images, print each probe's numbers.
     At least three probes must show visible separation between the two sets.
     If none do, stop and fix the probes before writing any agent code."

For every image this measures the probes at one region: for a forgery, the
region that was actually altered; for an authentic image, the largest amount
on the page - the region a forger would have gone for. That keeps the two
populations comparable, which a random-region baseline would not.

Separation is reported as AUC, because it needs no threshold: 0.5 is a coin
flip and 1.0 is perfect. Thresholds are the calibrator's job, not this
script's.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verdict.probes import compression, copymove, noise, ocr, typography  # noqa: E402
from verdict.probes.ocr import parse_amount  # noqa: E402

CORPUS = ROOT / "testdata" / "corpus"


def target_region(img, meta):
    """The altered region for a forgery; the biggest amount for an authentic page."""
    if meta.get("bbox"):
        return list(meta["bbox"])
    page = ocr.read(img)
    cands = [w for w in page.words if parse_amount(w.text) is not None and w.height >= 12]
    if not cands:
        return None
    w = max(cands, key=lambda w: w.height * w.width)
    return list(w.bbox)


def measure(path: pathlib.Path, meta: dict) -> dict | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    bbox = target_region(img, meta)
    if bbox is None:
        return None

    page = ocr.read(img)
    refs = ocr.text_regions(page, img)
    if len(refs) < 6:
        return None

    out = {"file": str(path.relative_to(CORPUS)), "label": meta["label"], "bbox": bbox}

    emap, _ = compression.ela_map(img)
    if emap is not None:
        out["ela_z"] = compression.robust_region_z(
            emap, bbox, refs, compression.edge_energy(img)).get("z", 0.0)

    res = noise.residual_map(img)
    vmap = noise.local_variance_map(res)
    hf = noise.region_hf_ratio(vmap, bbox, refs)
    out["hf_ratio"] = hf.get("hf_ratio", 1.0)

    flat = noise.flatness(img, bbox)
    out["modal_fraction"] = flat.get("modal_fraction", 0.0)
    out["flat_pixel_frac"] = flat.get("flat_pixel_frac", 0.0)

    grid = compression.block_grid_energy(img, bbox)
    out["grid_ratio"] = grid.get("grid_ratio", 1.0)

    typo = typography.region_metrics(img, bbox)
    out["stroke_ratio"] = typo.get("stroke_ratio", 1.0)
    out["baseline_z"] = typo.get("baseline_z", 0.0)
    out["height_ratio"] = typo.get("height_ratio", 1.0)

    dup = copymove.region_duplicate_score(img, bbox)
    out["dup_score"] = dup.get("best_score", 0.0) if dup.get("ok") else 0.0

    return out


def auc(pos: list[float], neg: list[float]) -> float:
    """Probability a random positive outranks a random negative (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> int:
    labels = json.loads((CORPUS / "labels.json").read_text())
    rows = []
    for rel, meta in sorted(labels.items()):
        r = measure(CORPUS / rel, meta)
        if r:
            rows.append(r)
        print(".", end="", flush=True)
    print()

    if not rows:
        print("no measurable images")
        return 1

    metrics = [k for k in rows[0] if k not in ("file", "label", "bbox")]
    forged = [r for r in rows if r["label"] == "forged"]
    auth = [r for r in rows if r["label"] == "authentic"]
    print(f"\nmeasured {len(forged)} forged / {len(auth)} authentic\n")

    print(f"{'probe metric':20}{'forged med':>12}{'auth med':>12}{'AUC':>8}"
          f"{'|AUC-.5|':>10}  separation")
    print("-" * 76)
    scored = []
    for m in metrics:
        f = [r[m] for r in forged if r.get(m) is not None]
        a = [r[m] for r in auth if r.get(m) is not None]
        if not f or not a:
            continue
        # Direction-agnostic: a probe that reliably runs *low* on forgeries is
        # just as useful as one that runs high.
        val = auc(f, a)
        strength = abs(val - 0.5)
        scored.append((strength, m, np.median(f), np.median(a), val))

    for strength, m, mf, ma, val in sorted(scored, reverse=True):
        bar = "#" * int(strength * 60)
        flag = "  <== STRONG" if strength >= 0.25 else ("  <- usable" if strength >= 0.15 else "")
        print(f"{m:20}{mf:12.4f}{ma:12.4f}{val:8.3f}{strength:10.3f}  {bar}{flag}")

    strong = [s for s in scored if s[0] >= 0.15]
    print(f"\n{len(strong)} probes show usable separation "
          f"(PLAN.md section 3 requires at least 3)")

    # Per-mode breakdown. A probe suite that only separates the easy modes is
    # a probe suite that has learned this generator, not forgery in general -
    # splice_digit and recompress_patch reuse the document's own pixels, so
    # they are the honest test.
    ops = sorted({labels[r["file"].replace("\\", "/")].get("op")
                  for r in forged if labels.get(r["file"].replace("\\", "/"))} - {None})
    if ops:
        print(f"\n{'metric':20}" + "".join(f"{o[:13]:>15}" for o in ops))
        print("-" * (20 + 15 * len(ops)))
        for _, m, _, _, _ in sorted(scored, reverse=True)[:6]:
            cells = ""
            for o in ops:
                f = [r[m] for r in forged
                     if labels.get(r["file"].replace("\\", "/"), {}).get("op") == o
                     and r.get(m) is not None]
                a = [r[m] for r in auth if r.get(m) is not None]
                v = auc(f, a)
                cells += f"{v:15.3f}" if v == v else f"{'-':>15}"
            print(f"{m:20}{cells}")
        print("\n(AUC per forgery mode; 0.5 = no separation, far from 0.5 = good)")

    (CORPUS / "separation.json").write_text(json.dumps(rows, indent=2))
    return 0 if len(strong) >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
