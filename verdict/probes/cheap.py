"""Cheap deterministic probes: EXIF provenance, perceptual hash, arithmetic.

Zero model calls. These run first and, per PLAN.md section 6, resolve a large
fraction of cases outright. Their job is to *describe* the document, not to
accuse it - the previous version escalated on `missing_exif` and reported an
`arithmetic_mismatch` on every receipt it saw, which is why the pipeline
called authentic documents forged.
"""
from __future__ import annotations

import pathlib
from typing import Any

import cv2
import numpy as np
import piexif
from PIL import Image

from . import ocr
from .ocr import (DISCOUNT_WORDS, NON_TOTAL_WORDS, PAYMENT_WORDS, TAX_WORDS,
                  TOTAL_WORDS, parse_amount)

# Software strings that mean a raster editor touched the file. A camera or a
# scanner never writes these.
EDITOR_SIGNATURES = ("photoshop", "gimp", "snapseed", "lightroom", "paint.net",
                     "pixlr", "picsart", "affinity", "krita", "canva",
                     "inkscape", "imagemagick", "faststone", "photopea")


def _jsonable(v: Any) -> Any:
    """EXIF values arrive as bytes; JSON and SQLite need text.

    piexif returns raw bytes for Make/Model/Software and the prober handed
    that dict straight to json.dumps, so the pipeline crashed outright on
    every real camera photo.
    """
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace").strip("\x00").strip()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def exif_audit(path: str) -> dict:
    """Summarise provenance metadata.

    Note what this does *not* do: it does not treat absent EXIF as evidence of
    tampering. PNGs, screenshots, scanner output, and anything that has been
    through a messaging app or a web CDN all legitimately have no EXIF, so
    absence carries almost no information. PLAN.md scenario F06 (screenshot of
    a genuine receipt) exists precisely to keep this honest.
    """
    out: dict[str, Any] = {
        "has_exif": False, "make": None, "model": None, "software": None,
        "datetime": None, "datetime_original": None, "format": None,
        "editor_signature": None, "datetime_conflict": False, "flags": [],
    }
    try:
        img = Image.open(pathlib.Path(path))
        out["format"] = img.format
        raw = img.info.get("exif")
        if raw:
            out["has_exif"] = True
            try:
                ex = piexif.load(raw)
                zeroth, exif_ifd = ex.get("0th", {}), ex.get("Exif", {})
                out["make"] = _jsonable(zeroth.get(piexif.ImageIFD.Make))
                out["model"] = _jsonable(zeroth.get(piexif.ImageIFD.Model))
                out["software"] = _jsonable(zeroth.get(piexif.ImageIFD.Software))
                out["datetime"] = _jsonable(zeroth.get(piexif.ImageIFD.DateTime))
                out["datetime_original"] = _jsonable(
                    exif_ifd.get(piexif.ExifIFD.DateTimeOriginal))
            except Exception as e:
                out["flags"].append("exif_parse_failed:" + str(e))
        else:
            for k in ("Software", "DateTime", "Make", "Model"):
                if k in img.info:
                    out["has_exif"] = True
                    out[k.lower()] = _jsonable(img.info[k])

        sw = (out.get("software") or "").lower()
        for sig in EDITOR_SIGNATURES:
            if sig in sw:
                out["editor_signature"] = sig
                out["flags"].append("editor_tag:" + sig)
                break

        dt, dto = out.get("datetime"), out.get("datetime_original")
        if dt and dto and dt != dto:
            out["datetime_conflict"] = True
            out["flags"].append("datetime_conflict")
    except Exception as e:
        out["flags"].append("exif_error:" + str(e))
    return _jsonable(out)


def perceptual_hash(pil_image: Image.Image) -> dict:
    """dHash - a fingerprint for spotting the same image resubmitted twice."""
    try:
        img = pil_image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    except Exception:
        img = pil_image.convert("L").resize((9, 8))
    px = np.array(img, dtype=np.int16)
    diff = px[:, 1:] > px[:, :-1]
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return {"dhash_hex": "%016x" % h, "dhash_int": h}


def phash(img_bgr: np.ndarray, hash_size: int = 8) -> str:
    """DCT perceptual hash - more robust to rescaling than dHash."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:hash_size, :hash_size].flatten()
    med = float(np.median(d[1:]))          # drop the DC term
    return "".join("1" if v > med else "0" for v in d)


def hamming(a, b) -> int:
    if isinstance(a, int) and isinstance(b, int):
        return bin(a ^ b).count("1")
    return sum(x != y for x, y in zip(a, b))


def _label_of(line_text: str) -> str | None:
    """Classify a receipt line by its label, if it has a recognisable one.

    Order matters: subtotal and "total qty" must be tested before "total",
    otherwise a substring match promotes them to the grand total.
    """
    t = " ".join(line_text.lower().split())
    for w in NON_TOTAL_WORDS:
        if w in t:
            return "subtotal" if "sub" in w else "count"
    for w in TAX_WORDS:
        if w in t:
            return "tax"
    for w in DISCOUNT_WORDS:
        if w in t:
            return "discount"
    for w in PAYMENT_WORDS:
        if w in t:
            return "payment"
    for w in TOTAL_WORDS:
        if w in t:
            return "total"
    return None


def _rightmost_amount(line):
    """The amount column sits on the right; the label sits on the left."""
    amts = line.amounts()
    return max(amts, key=lambda wv: wv[0].bbox[2]) if amts else None


def cross_field(path: str, img_bgr: np.ndarray | None = None) -> dict:
    """Reconcile a receipt's stated total against its own line items.

    The rule this replaces was: take the largest number anywhere on the page,
    call it the total, subtract every other number, and report a mismatch if
    the remainder was non-zero. On a real receipt those other numbers include
    the date, the phone number, the till id and the quantities, so it never
    balanced and every document was accused.

    This version refuses to answer unless the document is legible as a
    receipt: it needs a line whose *label* says total, and either two item
    lines above it or an explicit subtotal. Otherwise it returns
    insufficient_structure, which raises no claim. Silence beats a guess -
    this is the one probe that can accuse a *specific field*, so it must never
    fire on a document it did not understand.
    """
    if img_bgr is None:
        img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return {"ok": False, "reason": "imread_failed"}
    if not ocr.tesseract_available():
        return {"ok": False, "reason": "tesseract_unavailable"}

    page = ocr.read(img_bgr)
    if not page.lines:
        return {"ok": True, "status": "no_text", "count": 0}

    # --- locate the grand total -------------------------------------------
    total_hit = None
    for line in page.lines:                    # the last labelled total wins
        if _label_of(line.text) != "total":
            continue
        hit = _rightmost_amount(line)
        if hit:
            total_hit = (line, hit[0], hit[1])
    if total_hit is None:
        return {"ok": True, "status": "insufficient_structure",
                "reason": "no_labelled_total",
                "count": len(page.amount_words())}

    total_line, total_word, stated_total = total_hit
    if stated_total <= 0:
        return {"ok": True, "status": "insufficient_structure",
                "reason": "non_positive_total", "stated_total": stated_total}

    # --- collect the item lines above it ----------------------------------
    items: list[dict] = []
    tax_total, discount_total, subtotal = 0.0, 0.0, None
    for line in page.lines:
        if line.bbox[1] >= total_line.bbox[1]:
            continue                           # below the total = change / payment
        hit = _rightmost_amount(line)
        if not hit:
            continue
        word, val = hit
        kind = _label_of(line.text)
        if kind == "tax":
            tax_total += val
        elif kind == "discount":
            discount_total += val
        elif kind == "subtotal":
            subtotal = val
        elif kind in ("payment", "count", "total"):
            pass                               # not an item, not part of the sum
        else:
            items.append({"text": line.text[:60], "value": val,
                          "bbox": list(word.bbox), "kind": kind})

    if len(items) < 2 and subtotal is None:
        return {"ok": True, "status": "insufficient_structure",
                "reason": "too_few_line_items", "stated_total": stated_total,
                "item_count": len(items)}

    # A receipt may present items+tax = total or subtotal+tax = total; accept
    # whichever reconciles, since both layouts are legitimate.
    cand = {"items_plus_tax": sum(i["value"] for i in items) + tax_total - discount_total,
            "items_only": sum(i["value"] for i in items)}
    if subtotal is not None:
        cand["subtotal_plus_tax"] = subtotal + tax_total - discount_total
        cand["subtotal"] = subtotal

    # Tolerance absorbs per-line rounding and the occasional OCR digit slip.
    tol = max(0.05 * max(1.0, len(items)), 0.02 * stated_total, 1.0)
    best_name, best_delta = None, None
    for name, val in cand.items():
        delta = stated_total - val
        if best_delta is None or abs(delta) < abs(best_delta):
            best_name, best_delta = name, delta

    balanced = abs(best_delta) <= tol
    return {
        "ok": True,
        "status": "balanced" if balanced else "mismatch",
        "stated_total": round(stated_total, 2),
        "total_bbox": list(total_word.bbox),
        "total_line": total_line.text[:80],
        "reconciliation": best_name,
        "sum_line_items": round(cand[best_name], 2),
        "delta": round(best_delta, 2),
        "rel_error": round(abs(best_delta) / max(stated_total, 1.0), 4),
        "tolerance": round(tol, 2),
        "tax": round(tax_total, 2),
        "discount": round(discount_total, 2),
        "subtotal": subtotal,
        "item_count": len(items),
        "items": items[:40],
        "count": len(page.amount_words()),
    }


def find_numeric_boxes(img_bgr: np.ndarray, min_h: int = 12) -> list[dict]:
    """Amount-shaped OCR boxes. Kept for the verifier's presence check."""
    page = ocr.read(img_bgr)
    out = []
    for w in page.words:
        if w.height >= min_h and parse_amount(w.text) is not None:
            out.append({"bbox": list(w.bbox), "text": w.text, "conf": w.conf})
    return out
