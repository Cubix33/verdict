"""Zoom and enhance - the evidence-acquisition probes.

`zoom` is the workhorse described in idea.md section 4: a VLM ingests 448x448,
so a forged digit in a 4000x3000 photo arrives as roughly 45x34 px and the
3 px stroke detail that betrays it is sub-pixel. Cropping at native resolution
is the only way that information ever reaches a decision. This module was
specified in PLAN.md section 3 and was missing from the repository entirely,
which is why nothing downstream had any pixels to reason about.
"""
from __future__ import annotations

import cv2
import numpy as np


def zoom(img_bgr: np.ndarray, bbox, pad: float = 0.10, min_side: int = 448):
    """Native-resolution crop of a region, with a small context margin.

    Returns `(crop, info)`. `px_on_target` reports the *true* pixel height of
    the source region, before any upscaling for model ingestion - a crop that
    was enlarged has not gained information and must not be allowed to claim
    it has. Downstream gating reads this number to decide whether a probe is
    entitled to an opinion at all.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None, {"px_on_target": 0}
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
    if x2 <= x1 or y2 <= y1:
        return None, {"px_on_target": 0}

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, {"px_on_target": 0}

    true_px = int(crop.shape[0])
    upscaled = False
    if max(crop.shape[:2]) < min_side:
        s = min_side / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
        upscaled = True
    return crop, {"px_on_target": true_px, "bbox": [x1, y1, x2, y2],
                  "area_px": int((x2 - x1) * (y2 - y1)), "upscaled": upscaled}


def enhance(img_bgr: np.ndarray, mode: str = "clahe") -> np.ndarray:
    """Reveal what was captured but is not perceptible.

    Enhancement does not increase pixels on target. It is a rendering change,
    not evidence acquisition, and callers must never let it raise a card's
    `px_on_target`.
    """
    if mode == "clahe":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if mode == "gamma_lift":
        t = np.array([((i / 255.0) ** 0.45) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img_bgr, t)
    if mode == "unsharp":
        return cv2.addWeighted(img_bgr, 1.6, cv2.GaussianBlur(img_bgr, (0, 0), 3), -0.6, 0)
    if mode == "edge":
        e = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 60, 160)
        return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
    return img_bgr


def isolate(img_bgr: np.ndarray, bboxes, fill: int = 0) -> np.ndarray:
    """Black out everything outside the cited regions.

    This is the physical form of an evidence bond: the Verifier sees the cited
    crops in their true page positions and nothing else, so it cannot borrow
    context the Adjudicator never cited.
    """
    out = np.full_like(img_bgr, fill)
    H, W = img_bgr.shape[:2]
    for b in bboxes or []:
        x1, y1, x2, y2 = (int(v) for v in b)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = img_bgr[y1:y2, x1:x2]
    return out


def random_same_size_region(img_bgr: np.ndarray, bbox, rng) -> tuple:
    """A random region of identical size, for the shuffle control.

    idea.md calls the shuffle control the sharpest tool in the box: if a claim
    survives having its evidence swapped for an equal-sized patch of the same
    document, the claim was never using that evidence.
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = min(y2 - y1, H), min(x2 - x1, W)
    if h <= 0 or w <= 0:
        return None, None
    y = int(rng.integers(0, max(1, H - h)))
    x = int(rng.integers(0, max(1, W - w)))
    return img_bgr[y:y + h, x:x + w], [x, y, x + w, y + h]
