"""Noise-residual probes - the anti-synthesis and anti-fill detectors.

Two things matter here, and the previous implementation had both wrong.

First, direction. FLAME (ICML 2026, arXiv 2606.02178) shows that diffusion
*suppresses* local high-frequency variance, so generated and machine-filled
regions look anomalously **smooth**. The same is true of the crudest edit
there is: flood-filling a number with the paper colour before retyping it.
The old probe reported a raw variance ratio with no notion of which direction
was suspicious, so a perfectly flat forged patch and a noisy authentic one
scored the same magnitude.

Second, the reference. A ratio against the whole-image variance compares a
text region against mostly blank paper. Regions must be compared against other
text regions from the same document.
"""
from __future__ import annotations

import cv2
import numpy as np


def residual_map(img_bgr: np.ndarray, fast: bool = True) -> np.ndarray:
    """High-pass sensor/render residual.

    The fast path subtracts a small median blur, which preserves the grain that
    matters while costing a fraction of non-local means denoising. The slow
    path is a better sensor-noise estimate and is used during calibration,
    where per-image cost is amortised over the whole corpus.
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    if fast:
        base = cv2.medianBlur(g, 3)
    else:
        base = cv2.fastNlMeansDenoising(g, None, h=7, templateWindowSize=7,
                                        searchWindowSize=21)
    return g.astype(np.float32) - base.astype(np.float32)


def block_variance(res: np.ndarray, block: int = 32) -> tuple[np.ndarray, dict]:
    """Per-block variance grid of the residual, plus page-level statistics."""
    H, W = res.shape
    gh, gw = max(1, H // block), max(1, W // block)
    var = np.zeros((gh, gw), np.float32)
    for i in range(gh):
        for j in range(gw):
            var[i, j] = res[i * block:(i + 1) * block, j * block:(j + 1) * block].var()
    med = float(np.median(var))
    return var, {"median_block_var": round(med, 4),
                 "low_var_frac": round(float((var < 0.35 * max(med, 1e-9)).mean()), 5),
                 "min_block_var": round(float(var.min()), 4)}


def local_variance_map(res: np.ndarray, win: int = 9) -> np.ndarray:
    """Per-pixel local variance, so small regions still get a usable statistic.

    A retyped amount can be under 32 px tall, which is smaller than one block
    of the coarse grid. Sliding-window variance keeps the measurement valid at
    the scale forgeries actually occur.
    """
    r = res.astype(np.float32)
    mean = cv2.boxFilter(r, -1, (win, win), normalize=True)
    sq = cv2.boxFilter(r * r, -1, (win, win), normalize=True)
    return np.maximum(sq - mean * mean, 0.0)


def region_hf_ratio(var_map: np.ndarray, bbox, reference_bboxes) -> dict:
    """Region high-frequency energy relative to other text regions.

    A ratio well below 1 means the region is smoother than the text around it -
    the signature of a fill, a paste from a cleaner source, or a generated
    patch. A ratio well above 1 means it is noisier, which happens when text is
    composited over a different background.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    sub = var_map[y1:y2, x1:x2]
    if sub.size == 0:
        return {"ok": False, "hf_ratio": 1.0}
    r = float(np.median(sub))

    peers = []
    for b in reference_bboxes:
        bx1, by1, bx2, by2 = (int(v) for v in b)
        if [bx1, by1, bx2, by2] == [x1, y1, x2, y2]:
            continue
        s = var_map[by1:by2, bx1:bx2]
        if s.size:
            peers.append(float(np.median(s)))
    if len(peers) < 4:
        return {"ok": False, "hf_ratio": 1.0, "reason": "too_few_reference_regions"}

    arr = np.asarray(peers, dtype=np.float64)
    med = float(np.median(arr))
    ratio = r / max(med, 1e-9)
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad or (float(arr.std()) or 1e-9)
    return {"ok": True, "hf_ratio": round(ratio, 4), "region_var": round(r, 5),
            "peer_median_var": round(med, 5),
            "z": round(float((r - med) / sigma), 3), "n_peers": len(peers)}


def flatness(img_bgr: np.ndarray, bbox) -> dict:
    """Fraction of a region occupied by perfectly uniform pixels.

    Scanned paper is never flat: even a white margin carries grain. A run of
    identical values across a text region means something wrote a constant
    colour there. This is the single most reliable indicator of an erase-and-
    retype edit, and it needs no reference population at all.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    sub = img_bgr[y1:y2, x1:x2]
    if sub.size == 0:
        return {"ok": False}
    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if sub.ndim == 3 else sub
    counts = np.bincount(g.ravel(), minlength=256)
    top = int(counts.max())
    total = int(g.size)
    # Local gradient magnitude: a genuine scan has texture everywhere.
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return {"ok": True,
            "modal_fraction": round(top / max(total, 1), 4),
            "unique_levels": int((counts > 0).sum()),
            "flat_pixel_frac": round(float((grad < 2.0).mean()), 4)}


def dead_grain_regions(img_bgr: np.ndarray, exclude_bboxes=None,
                       grain_fraction: float = 0.4, min_px: int = 300,
                       max_regions: int = 3) -> list[dict]:
    """Regions of paper with no grain - the footprint of an erase fill.

    Scanned or photographed paper always carries sensor grain: measured on
    real receipts the local 5x5 standard deviation of blank paper sits around
    4-6 grey levels. A field that was flood-filled with a sampled background
    colour has almost none (~0.8 on the same page). This scans for
    paper-coloured pixels whose grain is far below the page's own paper
    median - a per-document relative test, so it ships no absolute threshold.

    This scan exists because erasures are invisible to every other proposal
    path: an erased field has no OCR text, so it appears in no text box and
    no amount list, and block-level statistics dilute a word-sized patch
    against its neighbours. The most anomalous region on the page was the one
    place nothing ever looked.

    MEASURED LIMITATION - read before wiring this into claim generation. On
    already-JPEG-compressed scans, compression itself produces natural
    grain-dead patches on content rows whose grain ratio (measured 0.07-0.41
    of the paper median across authentic SROIE receipts) brackets a genuine
    erase fill (0.22) - and identical regions fire on forged images and
    their authentic originals alike. The Prober therefore does NOT use these
    proposals; they are only trustworthy on high-quality inputs (phone
    photos, PNGs) where paper grain is strong and uncompressed.
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    f = g.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (5, 5), normalize=True)
    sq = cv2.boxFilter(f * f, -1, (5, 5), normalize=True)
    local_std = np.sqrt(np.maximum(sq - mean * mean, 0.0))

    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = g > thr
    if not paper.any():
        return []
    grain = float(np.median(local_std[paper]))
    if grain < 1e-3:
        return []              # synthetic image: everything is flat, nothing stands out

    dead = ((local_std < grain_fraction * grain) & paper).astype(np.uint8)
    dead = cv2.morphologyEx(dead, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if not dead.any():
        return []

    n_comp, comp, cstats, _ = cv2.connectedComponentsWithStats(dead)
    H, W = g.shape[:2]

    # Erasures happen where the content is. Scanner output legitimately has
    # grainless strips at the page borders (the scanner lid, software white
    # fill), so a candidate must sit inside the area the text actually spans.
    if exclude_bboxes:
        span = [min(b[0] for b in exclude_bboxes), min(b[1] for b in exclude_bboxes),
                max(b[2] for b in exclude_bboxes), max(b[3] for b in exclude_bboxes)]
    else:
        span = [0, 0, W, H]

    out = []
    for c in range(1, n_comp):
        x, y, w, h, area = cstats[c]
        if area < min_px or w < 20 or h < 10:
            # JPEG legitimately flattens thin blank strips between printed
            # sections; a filled FIELD is a solid block with the footprint of
            # a word. Requiring field-like dimensions is what separates them.
            continue
        if w >= 0.9 * W or h >= 0.9 * H:
            continue           # page-scale artefact, not a field-sized fill
        if x <= 4 or y <= 4 or x + w >= W - 4 or y + h >= H - 4:
            continue           # touches the page border: scanner background
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        if not _inside(bbox, span):
            continue
        if exclude_bboxes and any(_inside(bbox, b) for b in exclude_bboxes):
            continue
        # An erased FIELD sits on a content row: other words share its
        # vertical band (the item name is still printed; only the amount is
        # gone). A flat strip BETWEEN rows shares its band with nothing.
        # This is the difference between "a hole in a line of text" and
        # "the gap the layout put there".
        if exclude_bboxes:
            band = max((min(bbox[3], b[3]) - max(bbox[1], b[1]))
                       for b in exclude_bboxes)
            if band < 0.5 * h:
                continue
        out.append((int(area), bbox, round(float(
            np.median(local_std[y:y + h, x:x + w])), 3)))
    out.sort(key=lambda r: -r[0])
    return [{"bbox": b, "area_px": a, "grain": s, "paper_grain": round(grain, 3)}
            for a, b, s in out[:max_regions]]


def _inside(bbox, other) -> bool:
    """Is bbox's centre inside `other`? (loose overlap test for exclusion)"""
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    return other[0] <= cx <= other[2] and other[1] <= cy <= other[3]


# --- backwards-compatible entry point used by the original code ------------
def noise_residual(img_bgr: np.ndarray, bbox=None) -> dict:
    res = residual_map(img_bgr)
    var_full = float(np.var(res))
    if bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        var_crop = float(np.var(res[y1:y2, x1:x2]))
    else:
        var_crop = var_full
    return {"ok": True, "var_full": var_full, "var_crop": var_crop,
            "ratio": var_crop / (var_full + 1e-12)}
