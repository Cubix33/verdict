"""Typography probes - text edits rarely reproduce the original rasteriser.

A forger who retypes a number is compositing glyphs from a different renderer
into a page whose other glyphs went through a scanner or a camera. Three
things give that away: stroke width, baseline position relative to the rest of
its own row, and glyph height relative to its own row.

The previous version measured `baseline_std` across the *entire page* and
reported it as a single number. Every receipt has many rows at many heights,
so that statistic is dominated by page layout and says nothing about whether
any particular word is anomalous. Everything here is measured against a word's
own row.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import ocr


def stroke_width(gray_roi: np.ndarray) -> float:
    """Mean stroke thickness in pixels, via the distance transform of the ink."""
    if gray_roi is None or gray_roi.size == 0:
        return 0.0
    if gray_roi.shape[0] < 4 or gray_roi.shape[1] < 4:
        return 0.0
    _, bw = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if bw.mean() < 1.0 or bw.mean() > 254.0:
        return 0.0                              # no ink, or all ink: nothing to measure
    dt = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    v = dt[dt > 0]
    return float(2.0 * np.mean(v)) if v.size else 0.0


def page_metrics(img_bgr: np.ndarray) -> dict:
    """Document-level typography baseline: stroke width and per-row jitter."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    page = ocr.read(img_bgr)
    if len(page.words) < 5:
        return {"ok": True, "status": "insufficient_text", "n_words": len(page.words)}

    jitter = []
    for line in page.lines:
        if len(line.words) < 2:
            continue
        b = np.array([w.bbox[3] for w in line.words], np.float32)
        jitter.append(float(b.std()))

    sws = []
    for w in page.words:
        s = stroke_width(g[w.bbox[1]:w.bbox[3], w.bbox[0]:w.bbox[2]])
        if s > 0:
            sws.append(s)

    return {"ok": True, "status": "ok", "n_words": len(page.words),
            "doc_baseline_jitter_px": round(float(np.median(jitter)), 3) if jitter else 0.0,
            "doc_stroke_width_px": round(float(np.median(sws)), 3) if sws else 0.0,
            "doc_stroke_width_mad": round(float(np.median(np.abs(np.asarray(sws) -
                                          np.median(sws)))), 4) if sws else 0.0}


def region_metrics(img_bgr: np.ndarray, bbox) -> dict:
    """Typography of one region, scored against the row it sits in.

    Row-local comparison is the point. A total printed in a larger bold face is
    perfectly normal on a receipt, so comparing it to the page mean would flag
    every well-designed document. Comparing it to the other words *on its own
    row* asks the right question: does this word belong with its neighbours?
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    page = ocr.read(img_bgr)
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    target, row = None, None
    for line in page.lines:
        for w in line.words:
            wx1, wy1, wx2, wy2 = w.bbox
            if wx1 - 6 <= cx <= wx2 + 6 and wy1 - 6 <= cy <= wy2 + 6:
                target, row = w, line
                break
        if target:
            break
    if target is None:
        return {"ok": False, "reason": "no_word_at_bbox"}

    r_sw = stroke_width(g[target.bbox[1]:target.bbox[3], target.bbox[0]:target.bbox[2]])

    peers = [w for w in row.words if w is not target]
    if len(peers) < 1:
        doc = page_metrics(img_bgr)
        doc_sw = doc.get("doc_stroke_width_px", 0.0) or 1e-6
        return {"ok": True, "scope": "document", "region_stroke_width_px": round(r_sw, 3),
                "reference_stroke_width_px": round(doc_sw, 3),
                "stroke_ratio": round(r_sw / doc_sw, 3),
                "baseline_offset_px": 0.0, "baseline_z": 0.0,
                "height_ratio": 1.0, "n_peers": 0}

    peer_sw = [s for s in (stroke_width(g[w.bbox[1]:w.bbox[3], w.bbox[0]:w.bbox[2]])
                           for w in peers) if s > 0]
    ref_sw = float(np.median(peer_sw)) if peer_sw else 0.0

    peer_baselines = np.array([w.bbox[3] for w in peers], np.float32)
    ref_base = float(np.median(peer_baselines))
    offset = abs(float(target.bbox[3]) - ref_base)
    # Jitter floor: sub-pixel agreement is not physically meaningful, and
    # without a floor a perfectly aligned row divides by ~0 and reports a
    # spectacular z-score for a one-pixel OCR rounding difference.
    jitter = max(float(peer_baselines.std()), 0.5, 0.04 * float(np.median(
        [w.height for w in peers])))

    peer_h = float(np.median([w.height for w in peers])) or 1e-6

    return {"ok": True, "scope": "row",
            "region_stroke_width_px": round(r_sw, 3),
            "reference_stroke_width_px": round(ref_sw, 3),
            "stroke_ratio": round(r_sw / max(ref_sw, 1e-6), 3) if ref_sw else 1.0,
            "baseline_offset_px": round(offset, 3),
            "baseline_z": round(offset / jitter, 3),
            "height_ratio": round(target.height / peer_h, 3),
            "row_text": row.text[:60], "n_peers": len(peers)}


def font_metrics(img_bgr: np.ndarray, bbox=None) -> dict:
    """Combined page and region view, matching the PLAN.md probe signature."""
    out = page_metrics(img_bgr)
    if bbox is not None and out.get("status") == "ok":
        out.update({k: v for k, v in region_metrics(img_bgr, bbox).items()
                    if k not in ("ok",)})
    return out
