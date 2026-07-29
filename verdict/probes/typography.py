"""Typography / font metrics probe (MVP).
Uses OCR bounding boxes to estimate baseline offsets and stroke-width-like heuristics.
"""
from __future__ import annotations
from typing import Tuple, Optional
import os, shutil
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Point to standard install folder if PATH environment variable is not updated yet in the current terminal
if not shutil.which("tesseract"):
    win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(win_path):
        pytesseract.pytesseract.tesseract_cmd = win_path


def font_metrics(img_bgr: np.ndarray, bbox: Optional[Tuple[int,int,int,int]] = None) -> dict:
    if bbox is not None:
        x1,y1,x2,y2 = bbox
        crop = img_bgr[y1:y2, x1:x2]
    else:
        crop = img_bgr
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    d = pytesseract.image_to_data(gray, output_type=Output.DICT)
    baselines = []
    heights = []
    for i, txt in enumerate(d.get('text', [])):
        t = str(txt).strip()
        if not t:
            continue
        try:
            top = int(d['top'][i]); h = int(d['height'][i])
            baselines.append(top + h)  # bottom of box as proxy for baseline
            heights.append(h)
        except Exception:
            continue
    if not baselines:
        return {"ok": True, "count": 0}
    import statistics
    baseline_mean = statistics.mean(baselines)
    baseline_std = statistics.pstdev(baselines) if len(baselines) > 1 else 0.0
    height_mean = statistics.mean(heights)
    return {"ok": True, "count": len(baselines), "baseline_mean": baseline_mean, "baseline_std": baseline_std, "stroke_height_mean": height_mean}
