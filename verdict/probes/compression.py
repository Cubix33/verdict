"""Compression-history probes: multi-scale ELA, JPEG ghost, quantization tables.

Every edit to a JPEG leaves a compression fingerprint. A region that was
pasted in, re-rendered, or inpainted has been through a different number of
encode cycles than its neighbourhood, and that difference is measurable
without any model.

The old `ela.py` re-encoded the crop *standalone* and compared it against the
whole image re-encoded separately. Those are two different contexts - a small
crop and a full page do not quantise alike - so the resulting z-score measured
crop size far more than crop authenticity. Here the residual map is computed
once over the whole image, and regions are compared against each other inside
that single map.
"""
from __future__ import annotations

import hashlib
import pathlib

import cv2
import numpy as np
from PIL import Image

# Multi-quality ELA: arXiv 2607.06615 reports +0.180 AUC over single-quality.
DEFAULT_QUALITIES = (70, 80, 90)


def edge_energy(img_bgr: np.ndarray, win: int = 9) -> np.ndarray:
    """Local gradient magnitude - a proxy for how much detail a region holds."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.boxFilter(mag, -1, (win, win), normalize=True)


def ela_map(img_bgr: np.ndarray, qualities=DEFAULT_QUALITIES):
    """Multi-scale Error Level Analysis residual over the whole image.

    Returns the raw residual. Normalisation for local detail happens at
    *region* level in `robust_region_z`, not pixel by pixel: dividing two maps
    pointwise blows up wherever both go to zero, and blank paper between
    glyphs does exactly that, which drags a handful of pixels to enormous
    ratios and makes the region median meaningless.
    """
    maps = []
    for q in qualities:
        ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        if not ok:
            continue
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is None or dec.shape != img_bgr.shape:
            continue
        maps.append(cv2.absdiff(img_bgr, dec).astype(np.float32).max(axis=2))
    if not maps:
        return None, {"ok": False, "reason": "encode_failed"}
    m = np.stack(maps).mean(axis=0)
    return m, {"ok": True, "mean": float(m.mean()), "std": float(m.std()),
               "p99": float(np.percentile(m, 99)),
               "cross_q_ratio": float(maps[-1].mean() / max(maps[0].mean(), 1e-6))}


def _region_stat(value_map, energy_map, box, log: bool = True) -> float | None:
    """Mean residual per unit of local detail, over one region.

    Reported in log space by default. Residual-per-detail is a positive,
    multiplicative quantity and its distribution across the regions of a page
    is strongly right-skewed: measured on real receipts the per-region values
    span an order of magnitude, so a z-score on the linear scale puts the
    upper tail of a perfectly authentic document at z = 10 to 50 and no
    threshold can separate that from a forgery. Taking logs makes the spread
    roughly symmetric, which is the condition a median/MAD z-score assumes.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    sub = value_map[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    v = float(sub.mean())
    if energy_map is not None:
        e = energy_map[y1:y2, x1:x2]
        v = v / (float(e.mean()) + 4.0)
    return float(np.log(v + 1e-6)) if log else v


def robust_region_z(value_map: np.ndarray, bbox, reference_bboxes,
                    energy_map: np.ndarray | None = None) -> dict:
    """Score one region against the same statistic measured on its peers.

    The reference population is other *text* regions from the same document,
    not the whole page. Comparing a dense glyph region against a page that is
    mostly blank paper guarantees a large z-score for every word on every
    document, forged or not - which is precisely the false-positive engine
    this replaces.

    When an energy map is supplied the statistic becomes residual per unit of
    local detail. Without it, ELA ranks regions by how bold their text is: a
    re-encoded headline always produces a larger residual than body text, so
    the biggest glyphs on a perfectly honest receipt score highest and get
    accused. Normalising by detail asks the question that actually matters -
    is this region's compression history unusual *for how much ink it holds*.

    Uses median and MAD rather than mean and standard deviation: a genuinely
    tampered region would otherwise inflate the very spread it is measured
    against, masking itself.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    r_mean = _region_stat(value_map, energy_map, (x1, y1, x2, y2))
    if r_mean is None:
        return {"ok": False, "z": 0.0}

    peers = []
    for b in reference_bboxes:
        bx1, by1, bx2, by2 = (int(v) for v in b)
        if [bx1, by1, bx2, by2] == [x1, y1, x2, y2]:
            continue                                  # never compare against itself
        s = _region_stat(value_map, energy_map, (bx1, by1, bx2, by2))
        if s is not None:
            peers.append(s)
    if len(peers) < 4:
        return {"ok": False, "z": 0.0, "reason": "too_few_reference_regions",
                "region_mean": round(r_mean, 4), "n_peers": len(peers)}

    arr = np.asarray(peers, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad                              # MAD -> normal-equivalent sigma
    if sigma < 1e-6:
        sigma = float(arr.std()) or 1e-6
    z = (r_mean - med) / sigma
    return {"ok": True, "z": round(float(z), 3), "region_mean": round(r_mean, 4),
            "peer_median": round(med, 4), "peer_sigma": round(sigma, 4),
            "n_peers": len(peers)}


def jpeg_ghost(img_bgr: np.ndarray, qualities=range(55, 100, 5), block: int = 16):
    """Locate each region's original JPEG quality.

    A region originally compressed at q* shows a local minimum in reconstruction
    error when re-encoded at q*. A spliced or re-rendered region carries a
    different q* from the rest of the page, which shows up as a ghost.
    """
    qs = list(qualities)
    diffs = []
    for q in qs:
        ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        if not ok:
            continue
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is None or dec.shape != img_bgr.shape:
            continue
        d = ((img_bgr.astype(np.float32) - dec.astype(np.float32)) ** 2).mean(axis=2)
        diffs.append(cv2.blur(d, (block, block)))
    if not diffs:
        return None, {"ok": False}
    D = np.stack(diffs)
    argmin_q = np.asarray(qs[:len(diffs)])[D.argmin(axis=0)].astype(np.float32)
    med = float(np.median(argmin_q))
    return argmin_q, {"ok": True, "median_q": med,
                      "outlier_frac": round(float((np.abs(argmin_q - med) > 12).mean()), 5)}


def region_ghost_delta(ghost_map: np.ndarray, bbox) -> dict:
    """How far a region's estimated original quality sits from the page median."""
    if ghost_map is None:
        return {"ok": False}
    x1, y1, x2, y2 = (int(v) for v in bbox)
    sub = ghost_map[y1:y2, x1:x2]
    if sub.size == 0:
        return {"ok": False}
    med = float(np.median(ghost_map))
    r = float(np.median(sub))
    return {"ok": True, "region_q": r, "page_q": med,
            "delta_q": round(r - med, 2)}


def block_grid_energy(img_bgr: np.ndarray, bbox=None) -> dict:
    """Measure alignment to the JPEG 8x8 block grid.

    Untouched JPEG content has discontinuities on the 8-pixel lattice. A region
    that was pasted at an arbitrary offset, or re-rendered from vector text,
    loses that periodicity. The ratio below is near 1.0 when the grid is absent
    and rises when it is present.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        gray = gray[y1:y2, x1:x2]
        offset_x, offset_y = x1 % 8, y1 % 8
    else:
        offset_x = offset_y = 0
    if gray.shape[0] < 24 or gray.shape[1] < 24:
        return {"ok": False, "reason": "region_too_small"}

    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    col = dx.mean(axis=0)
    row = dy.mean(axis=1)

    def _ratio(profile, offset):
        idx = np.arange(profile.size)
        on = profile[(idx + offset + 1) % 8 == 0]
        off = profile[(idx + offset + 1) % 8 != 0]
        if on.size == 0 or off.size == 0:
            return 1.0
        return float(on.mean() / max(off.mean(), 1e-6))

    rx, ry = _ratio(col, offset_x), _ratio(row, offset_y)
    return {"ok": True, "grid_ratio_x": round(rx, 4), "grid_ratio_y": round(ry, 4),
            "grid_ratio": round((rx + ry) / 2.0, 4)}


def dct_quant(path: str) -> dict:
    """JPEG quantization tables.

    Per DocQT (arXiv 2605.19688) a real capture pipeline uses a small, stable
    bank of tables, so an unfamiliar table is a meaningful provenance signal -
    but only once the calibrator has seen what this customer's bank looks like.
    """
    try:
        im = Image.open(pathlib.Path(path))
        fmt = im.format
        qt = getattr(im, "quantization", None) or {}
        tabs = {str(k): list(v) for k, v in qt.items()}
        h = hashlib.sha256(repr(sorted(tabs.items())).encode()).hexdigest()[:16] if tabs else None
        luma = round(float(np.mean(list(qt.values())[0])), 2) if qt else None
        return {"ok": True, "is_jpeg": fmt == "JPEG", "has_qtable": bool(tabs),
                "qtable_hash": h, "n_tables": len(tabs), "luma_mean": luma}
    except Exception as e:
        return {"ok": False, "is_jpeg": False, "has_qtable": False,
                "qtable_hash": None, "error": str(e)}


def extract_qtable_hash(path: str) -> dict:
    """Backwards-compatible shim for the original dct_quant module."""
    d = dct_quant(path)
    return {"ok": d.get("ok", False), "is_jpeg": d.get("is_jpeg", False),
            "qtable_hash": d.get("qtable_hash")}
