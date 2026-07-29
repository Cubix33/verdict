"""Noise residual probe (MVP).
Computes a high-pass residual (simple sensor-noise proxy) and returns variance ratios.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple


def noise_residual(img_bgr: np.ndarray, bbox: Optional[Tuple[int,int,int,int]] = None) -> dict:
    # gray
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # approximate sensor noise by subtracting a heavy Gaussian blur
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=3)
    residual = gray.astype('float32') - blur.astype('float32')
    var_full = float(np.var(residual))
    if bbox is not None:
        x1,y1,x2,y2 = bbox
        crop_res = residual[y1:y2, x1:x2]
        var_crop = float(np.var(crop_res))
    else:
        var_crop = var_full
    # ratio of crop variance to full-image variance
    ratio = var_crop / (var_full + 1e-12)
    return {"ok": True, "var_full": var_full, "var_crop": var_crop, "ratio": ratio}
