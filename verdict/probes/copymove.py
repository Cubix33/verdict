"""Copy-move detection: find regions duplicated from elsewhere in the same page.

Two complementary methods, because they fail differently.

ORB self-matching finds duplicated *structure* (a stamp, a signature, a logo)
anywhere on the page, at any offset, and tolerates mild rescaling. It needs
texture, so it is weak on small runs of plain digits.

Normalised cross-correlation of a specific region against the rest of the page
is the direct test for "was this amount pasted from another amount on this same
receipt" - which is exactly the copy-move forgery a receipt actually suffers.
It is targeted rather than global, so the Prober runs it only on regions it
already cares about.
"""
from __future__ import annotations

import cv2
import numpy as np


def orb_self_match(img_bgr: np.ndarray, min_dist: float = 40.0,
                   ratio: float = 0.75, nfeatures: int = 6000) -> tuple[list, dict]:
    """Keypoints in this image that match other keypoints in the same image."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(g, None)
    if des is None or len(kp) < 20:
        return [], {"ok": True, "n_pairs": 0, "frac_of_kp": 0.0, "n_keypoints": len(kp or [])}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des, des, k=3)
    pairs = []
    for ms in knn:
        if len(ms) < 3:
            continue
        for m in ms[1:]:                       # ms[0] is the point matching itself
            if ms[1].distance < ratio * ms[-1].distance:
                p, q = kp[m.queryIdx].pt, kp[m.trainIdx].pt
                if np.hypot(p[0] - q[0], p[1] - q[1]) > min_dist:
                    pairs.append((p, q))
            break
    return pairs, {"ok": True, "n_pairs": len(pairs),
                   "frac_of_kp": round(len(pairs) / max(len(kp), 1), 4),
                   "n_keypoints": len(kp)}


def region_duplicate_score(img_bgr: np.ndarray, bbox, exclude_pad: int = 8) -> dict:
    """Best normalised correlation of this region against the rest of the page.

    A score near 1.0 means an almost pixel-identical copy of this region exists
    elsewhere on the same document. Authentic receipts do repeat short strings
    (`0.00`, `1.00`) so the score alone is not proof, but combined with the
    match being *near-exact* rather than merely similar it is strong evidence.
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    tpl = img_bgr[y1:y2, x1:x2]
    if tpl.size == 0 or tpl.shape[0] < 8 or tpl.shape[1] < 8:
        return {"ok": False, "reason": "region_too_small"}
    if tpl.shape[0] >= H or tpl.shape[1] >= W:
        return {"ok": False, "reason": "region_covers_page"}

    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    t = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if tpl.ndim == 3 else tpl
    if float(t.std()) < 3.0:
        return {"ok": False, "reason": "region_has_no_structure"}

    res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
    # Suppress the trivial self-match at the region's own location.
    sx1 = max(0, x1 - exclude_pad)
    sy1 = max(0, y1 - exclude_pad)
    sx2 = min(res.shape[1], x1 + exclude_pad + 1)
    sy2 = min(res.shape[0], y1 + exclude_pad + 1)
    res[sy1:sy2, sx1:sx2] = -1.0

    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return {"ok": True, "best_score": round(float(max_val), 4),
            "match_bbox": [int(max_loc[0]), int(max_loc[1]),
                           int(max_loc[0]) + t.shape[1], int(max_loc[1]) + t.shape[0]],
            "distance_px": round(float(np.hypot(max_loc[0] - x1, max_loc[1] - y1)), 1)}


def copy_move(img_bgr: np.ndarray, min_dist: float = 40.0, ratio: float = 0.75):
    """Backwards-compatible alias for the global ORB detector."""
    return orb_self_match(img_bgr, min_dist=min_dist, ratio=ratio)
