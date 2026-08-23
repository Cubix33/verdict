"""Mechanism 3 - N=10 self-calibration. VERDICT ships with no thresholds.

This module was specified in PLAN.md section 4 and was absent from the
repository. Its absence is the direct cause of the reported failure: the
Adjudicator compared against literals (`abs(delta) > 0.5`, `z > 3.0`) that
were never measured against anything, so every document tripped them. The
project's own ablation table predicts the result - `no_calibration` drives
false accusations from 1.7% to 21.4%. Measured on real receipts it was 100%.

What gets learned
-----------------
Every probe here already reports a *within-document* statistic: a region
compared against the other text regions on its own page. Calibration then asks
a second question, which is the one that actually controls false positives:

    on a document that is known authentic, how extreme does the most extreme
    region get?

That distinction matters because of multiple comparisons. A receipt has fifty
or more text regions. Even with a perfectly behaved statistic, the maximum
z-score over fifty draws is routinely 2.5-3.0 - so a fixed `z > 3.0` threshold
fires on a large share of genuine documents. Calibrating on the per-document
*maximum* prices that in.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import cv2
import numpy as np

from .probes import compression, copymove, noise, ocr, typography

DEFAULT_DIR = pathlib.Path("config/customers")

# Statistics collected per region. `side` says which tail is suspicious:
#   "high"  large values are anomalous
#   "low"   small values are anomalous
#   "both"  either tail is
PROBE_SPECS = {
    "ela_z":           {"side": "high", "probe": "ela"},
    "hf_ratio":        {"side": "both", "probe": "noise_residual"},
    "flat_pixel_frac": {"side": "high", "probe": "noise_residual"},
    "dup_score":       {"side": "high", "probe": "copy_move"},
    "grid_ratio":      {"side": "low",  "probe": "block_grid"},
    "stroke_ratio":    {"side": "both", "probe": "font_metrics"},
    "baseline_z":      {"side": "high", "probe": "font_metrics"},
}

# A probe may not opine on a region it cannot resolve (idea.md Finding 6:
# gate on measured pixels, never on model confidence).
PX_ON_TARGET_MINIMUMS = {"font_metrics": 400, "zoom": 300, "ela": 120,
                         "noise_residual": 120, "copy_move": 200, "block_grid": 576}


@dataclass
class Calibration:
    customer_id: str
    n_samples: int = 0
    target_fpr: float = 0.02
    thresholds: dict = field(default_factory=dict)
    qtable_bank: list = field(default_factory=list)
    typography: dict = field(default_factory=dict)
    px_on_target_minimums: dict = field(default_factory=lambda: dict(PX_ON_TARGET_MINIMUMS))
    notes: str = ""

    def to_dict(self) -> dict:
        return {"customer_id": self.customer_id, "n_samples": self.n_samples,
                "target_fpr": self.target_fpr, "thresholds": self.thresholds,
                "qtable_bank": self.qtable_bank, "typography": self.typography,
                "px_on_target_minimums": self.px_on_target_minimums,
                "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(customer_id=d.get("customer_id", "default"),
                   n_samples=d.get("n_samples", 0),
                   target_fpr=d.get("target_fpr", 0.02),
                   thresholds=d.get("thresholds", {}),
                   qtable_bank=d.get("qtable_bank", []),
                   typography=d.get("typography", {}),
                   px_on_target_minimums=d.get("px_on_target_minimums",
                                               dict(PX_ON_TARGET_MINIMUMS)),
                   notes=d.get("notes", ""))

    def exceeds(self, metric: str, value: float) -> bool:
        """Is this value beyond the calibrated bound for this metric?"""
        t = self.thresholds.get(metric)
        if t is None or value is None:
            return False
        side = PROBE_SPECS.get(metric, {}).get("side", "high")
        if side == "high":
            return value > t["hi"]
        if side == "low":
            return value < t["lo"]
        return value > t["hi"] or value < t["lo"]

    def severity(self, metric: str, value: float) -> float:
        """How far past the bound, in calibrated units. 0.0 means within normal.

        Reported alongside every claim so a reader can see not just that a
        threshold was crossed but by how much.
        """
        t = self.thresholds.get(metric)
        if t is None or value is None:
            return 0.0
        scale = max(t.get("scale", 1.0), 1e-9)
        side = PROBE_SPECS.get(metric, {}).get("side", "high")
        out = 0.0
        if side in ("high", "both") and value > t["hi"]:
            out = (value - t["hi"]) / scale
        if side in ("low", "both") and value < t["lo"]:
            out = max(out, (t["lo"] - value) / scale)
        return round(float(out), 3)


def region_features(img_bgr: np.ndarray, bbox, ctx: dict) -> dict:
    """Every calibrated statistic for one region, sharing one set of maps."""
    refs = ctx["refs"]
    out: dict[str, float] = {}

    if ctx.get("ela") is not None:
        out["ela_z"] = compression.robust_region_z(
            ctx["ela"], bbox, refs, ctx.get("energy")).get("z", 0.0)

    hf = noise.region_hf_ratio(ctx["var"], bbox, refs)
    if hf.get("ok"):
        out["hf_ratio"] = hf["hf_ratio"]

    flat = noise.flatness(img_bgr, bbox)
    if flat.get("ok"):
        out["flat_pixel_frac"] = flat["flat_pixel_frac"]
        out["modal_fraction"] = flat["modal_fraction"]

    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area >= PX_ON_TARGET_MINIMUMS["block_grid"]:
        grid = compression.block_grid_energy(img_bgr, bbox)
        if grid.get("ok"):
            out["grid_ratio"] = grid["grid_ratio"]

    if area >= PX_ON_TARGET_MINIMUMS["copy_move"]:
        dup = copymove.region_duplicate_score(img_bgr, bbox)
        if dup.get("ok"):
            out["dup_score"] = dup["best_score"]

    if (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) >= PX_ON_TARGET_MINIMUMS["font_metrics"]:
        typo = typography.region_metrics(img_bgr, bbox)
        if typo.get("ok") and typo.get("scope") == "row":
            out["stroke_ratio"] = typo["stroke_ratio"]
            out["baseline_z"] = typo["baseline_z"]
    return out


def build_context(img_bgr: np.ndarray) -> dict:
    """Compute the whole-page maps once; every region reuses them."""
    page = ocr.read(img_bgr)
    refs = ocr.text_regions(page, img_bgr)
    emap, _ = compression.ela_map(img_bgr)
    res = noise.residual_map(img_bgr)
    return {"page": page, "refs": refs, "ela": emap,
            "energy": compression.edge_energy(img_bgr),
            "var": noise.local_variance_map(res)}


def scan_document(img_bgr: np.ndarray) -> list[dict]:
    """Features for every text region on a page."""
    ctx = build_context(img_bgr)
    if len(ctx["refs"]) < 6:
        return []
    return [region_features(img_bgr, b, ctx) for b in ctx["refs"]]


def calibrate(authentic_paths, customer_id: str = "default",
              out_dir=DEFAULT_DIR, target_fpr: float = 0.02,
              verbose: bool = False) -> Calibration:
    """Learn this pipeline's normal from documents certified authentic.

    Thresholds are set on the distribution of each document's *worst* region,
    at the requested false-positive rate, with a margin that widens when the
    corpus is small - ten documents cannot pin a 2% tail precisely, and
    pretending otherwise is how a calibrator becomes a false-accusation engine.
    """
    per_doc_extremes: dict[str, list[float]] = {}
    qhashes: list[str] = []
    jitters: list[float] = []
    strokes: list[float] = []
    n = 0

    for p in authentic_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        feats = scan_document(img)
        if not feats:
            if verbose:
                print(f"  skip (too little text): {p}")
            continue
        n += 1

        for metric, spec in PROBE_SPECS.items():
            vals = [f[metric] for f in feats if metric in f]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            side = spec["side"]
            # The per-document extreme in the suspicious direction.
            if side == "high":
                per_doc_extremes.setdefault(metric + "|hi", []).append(float(arr.max()))
            elif side == "low":
                per_doc_extremes.setdefault(metric + "|lo", []).append(float(arr.min()))
            else:
                per_doc_extremes.setdefault(metric + "|hi", []).append(float(arr.max()))
                per_doc_extremes.setdefault(metric + "|lo", []).append(float(arr.min()))

        q = compression.dct_quant(str(p))
        if q.get("qtable_hash"):
            qhashes.append(q["qtable_hash"])
        pm = typography.page_metrics(img)
        if pm.get("status") == "ok":
            jitters.append(pm["doc_baseline_jitter_px"])
            strokes.append(pm["doc_stroke_width_px"])
        if verbose:
            print(f"  calibrated on {pathlib.Path(p).name} ({len(feats)} regions)")

    # Small-sample margin. With N documents the tightest tail that can be
    # estimated honestly is about 1/N, so widen the bound when N is small.
    margin = 1.0 + 2.0 / max(n, 1)

    thresholds: dict[str, dict] = {}
    for metric, spec in PROBE_SPECS.items():
        hi_vals = per_doc_extremes.get(metric + "|hi")
        lo_vals = per_doc_extremes.get(metric + "|lo")
        if not hi_vals and not lo_vals:
            continue
        entry: dict[str, float] = {}
        pool = np.asarray((hi_vals or []) + (lo_vals or []), dtype=np.float64)
        scale = float(np.std(pool)) or max(float(np.mean(np.abs(pool))) * 0.1, 1e-3)
        entry["scale"] = round(scale, 5)

        if hi_vals:
            a = np.asarray(hi_vals, dtype=np.float64)
            q = float(np.quantile(a, 1.0 - target_fpr))
            entry["hi"] = round(q + (margin - 1.0) * (float(a.std()) or scale), 5)
            entry["observed_max"] = round(float(a.max()), 5)
        else:
            entry["hi"] = float("inf")
        if lo_vals:
            a = np.asarray(lo_vals, dtype=np.float64)
            q = float(np.quantile(a, target_fpr))
            entry["lo"] = round(q - (margin - 1.0) * (float(a.std()) or scale), 5)
            entry["observed_min"] = round(float(a.min()), 5)
        else:
            entry["lo"] = float("-inf")
        entry["n"] = len(hi_vals or lo_vals)
        thresholds[metric] = entry

    cal = Calibration(
        customer_id=customer_id, n_samples=n, target_fpr=target_fpr,
        thresholds=thresholds, qtable_bank=sorted(set(qhashes)),
        typography={"baseline_jitter_px": round(float(np.median(jitters)), 3) if jitters else 0.4,
                    "stroke_width_px": round(float(np.median(strokes)), 3) if strokes else 2.0},
        notes=f"per-document extreme quantiles at fpr={target_fpr}, "
              f"small-sample margin x{margin:.2f} over n={n} documents",
    )
    save(cal, out_dir)
    return cal


def save(cal: Calibration, out_dir=DEFAULT_DIR) -> pathlib.Path:
    d = pathlib.Path(out_dir) / cal.customer_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "calibration.json"
    p.write_text(json.dumps(cal.to_dict(), indent=2))
    return p


def load(customer_id: str = "default", out_dir=DEFAULT_DIR) -> Calibration:
    """Load a calibration, or fail loudly.

    There is deliberately no default set of thresholds to fall back on. A
    pipeline running on numbers nobody measured is the bug this module exists
    to prevent, so an uncalibrated customer must be an error the caller has to
    handle, not a silent guess.
    """
    p = pathlib.Path(out_dir) / customer_id / "calibration.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no calibration for '{customer_id}' at {p}. "
            f"VERDICT ships no thresholds - run "
            f"`python -m verdict.calibrate <authentic-dir> --customer {customer_id}` "
            f"on at least 5 documents you certify as authentic.")
    return Calibration.from_dict(json.loads(p.read_text()))


def exists(customer_id: str = "default", out_dir=DEFAULT_DIR) -> bool:
    return (pathlib.Path(out_dir) / customer_id / "calibration.json").exists()


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Calibrate VERDICT on authentic documents.")
    ap.add_argument("directory", help="folder of documents certified authentic")
    ap.add_argument("--customer", default="default")
    ap.add_argument("--out-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--target-fpr", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(p for p in pathlib.Path(args.directory).iterdir()
                   if p.suffix.lower() in exts)
    if args.limit:
        paths = paths[:args.limit]
    if len(paths) < 3:
        print(f"need at least 3 authentic documents, found {len(paths)}")
        return 2

    print(f"calibrating '{args.customer}' on {len(paths)} documents...")
    cal = calibrate(paths, customer_id=args.customer, out_dir=args.out_dir,
                    target_fpr=args.target_fpr, verbose=True)
    print(f"\ncalibrated on {cal.n_samples} documents at fpr={cal.target_fpr}")
    for m, t in sorted(cal.thresholds.items()):
        lo = "-inf" if t["lo"] == float("-inf") else f"{t['lo']:.4f}"
        hi = "inf" if t["hi"] == float("inf") else f"{t['hi']:.4f}"
        print(f"  {m:18} lo={lo:>10}  hi={hi:>10}  (n={t['n']})")
    print(f"\nwritten to {pathlib.Path(args.out_dir) / args.customer / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
