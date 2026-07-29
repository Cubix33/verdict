"""Error Level Analysis (ELA) probe.
Produces residual scores by re-encoding crops at lower JPEG quality and measuring per-pixel difference.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional


def ela_residual(img_bgr: np.ndarray, bbox: Optional[Tuple[int,int,int,int]] = None, qualities=(90,75,50)) -> dict:
    """Compute ELA residual statistics for the (optionally cropped) image.
    Returns dict with doc_mean, doc_std, crop_mean, crop_std, z_score (crop vs doc)
    """
    if bbox is not None:
        x1,y1,x2,y2 = bbox
        crop = img_bgr[y1:y2, x1:x2]
    else:
        crop = img_bgr
    # reference grayscale
    ref_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ref_mean = float(ref_gray.mean())
    ref_std = float(ref_gray.std())

    # operate on crop
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scores = []
    for q in qualities:
        ok, enc = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            continue
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        dec_gray = cv2.cvtColor(dec, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(crop_gray, dec_gray).astype(np.float32)
        scores.append(float(diff.mean()))
    if not scores:
        return {"ok": False, "reason": "encode_failed"}
    crop_mean = float(np.mean(scores))
    crop_std = float(np.std(scores))
    # approximate doc-level residual by sampling the whole image reconstructed similarly
    ok, enc_whole = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qualities[0]])
    dec_whole = cv2.imdecode(enc_whole, cv2.IMREAD_COLOR)
    whole_gray = cv2.cvtColor(dec_whole, cv2.COLOR_BGR2GRAY)
    whole_diff = cv2.absdiff(ref_gray, whole_gray).astype(np.float32)
    doc_mean = float(whole_diff.mean())
    doc_std = float(whole_diff.std())
    # compute z-score of crop_mean against doc distribution (use doc_mean/doc_std)
    z = (crop_mean - doc_mean) / (doc_std + 1e-9)
    return {"ok": True, "doc_mean": doc_mean, "doc_std": doc_std, "crop_mean": crop_mean, "crop_std": crop_std, "z_score": float(z)}
