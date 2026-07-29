"""Deterministic, zero-VLM probes used by the Triage agent.

Probes implemented:
- exif_audit(path) -> dict
- perceptual_hash(pil_image) -> dict
- cross_field(path) -> dict (uses OCR to find numeric fields and compare sums)
- find_numeric_boxes(img_bgr, min_h=14) -> list of boxes (helper)

These are intentionally simple, fast, and auditable.
"""
from __future__ import annotations
import io, pathlib, typing, os, shutil
from typing import List
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
import piexif
import hashlib

# Point to standard install folder if PATH environment variable is not updated yet in the current terminal
if not shutil.which("tesseract"):
    win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(win_path):
        pytesseract.pytesseract.tesseract_cmd = win_path


def exif_audit(path: str) -> dict:
    """Return a small EXIF summary for the image at path."""
    out = {"has_exif": False, "make": None, "model": None, "software": None,
           "datetime": None, "flags": []}
    try:
        p = pathlib.Path(path)
        img = Image.open(p)
        info = img.info
        if "exif" in info:
            out["has_exif"] = True
            try:
                ex = piexif.load(info["exif"])
                zeroth = ex.get("0th", {})
                out["make"] = zeroth.get(piexif.ImageIFD.Make)
                out["model"] = zeroth.get(piexif.ImageIFD.Model)
                out["software"] = zeroth.get(piexif.ImageIFD.Software)
                out["datetime"] = zeroth.get(piexif.ImageIFD.DateTime)
            except Exception:
                # fallback: try parsing via PIL tag lookup
                pass
        else:
            # Try to read common keys from info dict
            for k in ("Software", "DateTime", "make", "model"):
                if k in info:
                    out["has_exif"] = True
                    out[k.lower()] = info[k]
    except Exception as e:
        out["flags"].append(f"exif_error:{e}")
    return out


def perceptual_hash(pil_image: Image.Image) -> dict:
    """Compute a simple perceptual hash (dHash) for dedup/phash-like check.
    Returns hex string and int hamming distance helper.
    """
    try:
        img = pil_image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    except Exception:
        img = pil_image.convert("L").resize((9, 8))
    pixels = np.array(img, dtype=np.uint8)
    diff = pixels[:, 1:] > pixels[:, :-1]
    # pack into 64-bit int
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    hexh = f"{h:016x}"
    return {"dhash_hex": hexh, "dhash_int": h}


def find_numeric_boxes(img_bgr: np.ndarray, min_h: int = 14) -> List[dict]:
    """Return OCR-detected numeric boxes with some filtering.
    Each box: {bbox: [x1,y1,x2,y2], text: str, conf: int}
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    d = pytesseract.image_to_data(gray, output_type=Output.DICT)
    boxes: List[dict] = []
    for i, txt in enumerate(d.get("text", [])):
        t = str(txt).strip().replace(",", "").replace(".", "")
        try:
            conf = int(d.get("conf", [])[i])
        except Exception:
            conf = -1
        h = int(d.get("height", [])[i]) if d.get("height") else 0
        if t.isdigit() and len(t) >= 1 and conf > 30 and h >= min_h:
            x = int(d.get("left", [])[i]); y = int(d.get("top", [])[i])
            w = int(d.get("width", [])[i]); hh = h
            boxes.append({"bbox": [x, y, x + w, y + hh], "text": txt.strip(), "conf": conf})
    return boxes


def cross_field(path: str) -> dict:
    """Very small heuristic to detect simple arithmetic mismatches in receipts.
    Extracts numbers via OCR and tries to find a declared total that doesn't match the sum of line items.
    Returns a dict with status and values found.
    """
    try:
        img = cv2.imread(str(path))
        if img is None:
            return {"ok": False, "reason": "imread_failed"}
        boxes = find_numeric_boxes(img, min_h=12)
        nums = []
        for b in boxes:
            txt = b["text"].replace("₹", "").replace("Rs", "")
            try:
                val = float(txt)
                nums.append((b, val))
            except Exception:
                continue
        if not nums:
            return {"ok": True, "status": "no_numeric_found", "count": 0}
        # crude heuristic: largest number is total, others are line-items
        vals = [v for _, v in nums]
        largest = max(vals)
        others = [v for v in vals if v != largest]
        sothers = sum(others) if others else 0.0
        delta = largest - sothers
        return {"ok": True, "status": "computed", "stated_total": largest, "sum_line_items": sothers, "delta": delta, "count": len(vals)}
    except Exception as e:
        return {"ok": False, "reason": f"exception:{e}"}
