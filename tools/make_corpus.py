"""Build a labelled test corpus for the VERDICT pipeline.

Sources
-------
`testdata/raw/`  real images downloaded from the web (SROIE scanned receipts,
                 real camera JPEGs with EXIF, a landscape photo) plus the four
                 images that ship in `data/`.

Output
------
`testdata/corpus/authentic/`  untouched copies of the real images
`testdata/corpus/forged/`     ground-truth forgeries derived from them
`testdata/corpus/labels.json` {relative_path: {"label": "authentic"|"forged", ...}}

The forgeries mimic what a human actually does to a receipt: they retype a
number.  Each one records the pixel region that was altered so the evaluation
harness can also score *localisation*, not just the binary verdict.
"""
from __future__ import annotations

import json
import pathlib
import random
import shutil
import sys

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "testdata" / "raw"
CORPUS = ROOT / "testdata" / "corpus"
SEED = 20260823


def _tesseract_ready() -> None:
    import os
    import shutil as sh
    if sh.which("tesseract"):
        return
    win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(win):
        pytesseract.pytesseract.tesseract_cmd = win


def amount_boxes(img_bgr: np.ndarray) -> list[dict]:
    """OCR boxes that look like a money amount and are big enough to edit."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    d = pytesseract.image_to_data(gray, output_type=Output.DICT)
    out = []
    for i, raw in enumerate(d["text"]):
        t = str(raw).strip()
        if not t:
            continue
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 55:
            continue
        digits = t.replace(",", "").replace(".", "")
        if not digits.isdigit() or len(digits) < 3:
            continue
        x, y = int(d["left"][i]), int(d["top"][i])
        w, h = int(d["width"][i]), int(d["height"][i])
        if h < 12 or w < 18:
            continue
        out.append({"bbox": [x, y, x + w, y + h], "text": t, "conf": conf})
    return out


def forge_retype(img_bgr: np.ndarray, box: dict, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Erase an amount with the local background and re-render a bigger one.

    This is the classic expense-fraud edit: the total is inflated and the file
    is saved again.  The re-rendered glyphs come from a different rasteriser
    than the rest of the page, and the patch has been through one fewer
    compression generation than its neighbourhood.
    """
    x1, y1, x2, y2 = box["bbox"]
    out = img_bgr.copy()
    patch = out[y1:y2, x1:x2]

    # background colour = the brightest decile of the patch (paper, not ink)
    bg = np.percentile(patch.reshape(-1, 3), 90, axis=0).astype(np.uint8)
    ink = np.percentile(patch.reshape(-1, 3), 5, axis=0).astype(np.uint8)
    out[y1:y2, x1:x2] = bg

    digits = box["text"].replace(",", "")
    try:
        value = float(digits)
    except ValueError:
        value = 0.0
    inflated = value * rng.uniform(2.0, 4.0) + rng.uniform(10, 90)
    new_text = f"{inflated:.2f}" if "." in digits else f"{int(inflated)}"

    h = y2 - y1
    scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, max(8, int(h * 0.72)), 1)
    (tw, th), _ = cv2.getTextSize(new_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    while tw > (x2 - x1) and scale > 0.15:
        scale *= 0.9
        (tw, th), _ = cv2.getTextSize(new_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)

    org = (x2 - tw, y1 + (h + th) // 2)
    cv2.putText(out, new_text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                tuple(int(c) for c in ink), 1, cv2.LINE_AA)
    return out, {"op": "retype_amount", "bbox": [x1, y1, x2, y2],
                 "from": box["text"], "to": new_text}


def forge_copymove(img_bgr: np.ndarray, boxes: list[dict], rng: random.Random):
    """Paste one amount over another - a splice with identical local statistics."""
    if len(boxes) < 2:
        return None, None
    src, dst = rng.sample(boxes, 2)
    sx1, sy1, sx2, sy2 = src["bbox"]
    dx1, dy1, dx2, dy2 = dst["bbox"]
    patch = img_bgr[sy1:sy2, sx1:sx2]
    if patch.size == 0:
        return None, None
    resized = cv2.resize(patch, (dx2 - dx1, dy2 - dy1), interpolation=cv2.INTER_LANCZOS4)
    out = img_bgr.copy()
    out[dy1:dy2, dx1:dx2] = resized
    return out, {"op": "copy_move_amount", "bbox": [dx1, dy1, dx2, dy2],
                 "src_bbox": [sx1, sy1, sx2, sy2],
                 "from": dst["text"], "to": src["text"]}


def forge_splice_digit(img_bgr: np.ndarray, boxes: list[dict], rng: random.Random):
    """Overwrite the leading digit of one amount with a digit cut from another.

    The hardest case in the corpus, and the most realistic. Every pixel comes
    from this same document, so it survived the same scanner, the same JPEG
    quantiser and the same lighting. Nothing about the rasteriser gives it
    away - only the compression and grid discontinuity at the paste boundary
    can, which is exactly what the compression probes are for.
    """
    if len(boxes) < 2:
        return None, None
    src, dst = rng.sample(boxes, 2)
    sx1, sy1, sx2, sy2 = src["bbox"]
    dx1, dy1, dx2, dy2 = dst["bbox"]

    # A single glyph is roughly width/len(text) wide; take the first of each.
    s_w = max(4, (sx2 - sx1) // max(len(src["text"]), 1))
    d_w = max(4, (dx2 - dx1) // max(len(dst["text"]), 1))
    glyph = img_bgr[sy1:sy2, sx1:sx1 + s_w]
    if glyph.size == 0:
        return None, None
    glyph = cv2.resize(glyph, (d_w, dy2 - dy1), interpolation=cv2.INTER_LANCZOS4)
    out = img_bgr.copy()
    out[dy1:dy2, dx1:dx1 + d_w] = glyph
    return out, {"op": "splice_digit", "bbox": [dx1, dy1, dx1 + d_w, dy2],
                 "src_bbox": [sx1, sy1, sx1 + s_w, sy2],
                 "from": dst["text"], "to": "leading digit of " + src["text"]}


def forge_recompress_patch(img_bgr: np.ndarray, box: dict, rng: random.Random):
    """Re-encode one field at a lower quality and paste it back.

    A compression-history attack with no visible change at all: the pixels
    still say the same thing, but that field has been through one more encode
    cycle than the page around it. This is what a real edit leaves behind once
    the forger has been careful about everything else.
    """
    x1, y1, x2, y2 = box["bbox"]
    patch = img_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return None, None
    ok, enc = cv2.imencode(".jpg", patch, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
    if not ok:
        return None, None
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None or dec.shape != patch.shape:
        return None, None
    out = img_bgr.copy()
    out[y1:y2, x1:x2] = dec
    return out, {"op": "recompress_patch", "bbox": [x1, y1, x2, y2],
                 "from": box["text"], "to": box["text"] + " (q45 re-encode)"}


def forge_erase_field(img_bgr: np.ndarray, box: dict, rng: random.Random):
    """Remove a line item entirely, so the stated total no longer reconciles."""
    x1, y1, x2, y2 = box["bbox"]
    out = img_bgr.copy()
    patch = out[y1:y2, x1:x2]
    if patch.size == 0:
        return None, None
    bg = np.percentile(patch.reshape(-1, 3), 90, axis=0).astype(np.uint8)
    out[y1:y2, x1:x2] = bg
    return out, {"op": "erase_field", "bbox": [x1, y1, x2, y2],
                 "from": box["text"], "to": "(erased)"}


def main() -> int:
    _tesseract_ready()
    rng = random.Random(SEED)

    auth_dir = CORPUS / "authentic"
    forged_dir = CORPUS / "forged"
    for d in (auth_dir, forged_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    sources = sorted(RAW.glob("*.jpg")) + sorted(RAW.glob("*.png"))
    sources += [p for p in sorted((ROOT / "data").glob("*")) if p.suffix.lower() in {".jpg", ".png"}]

    labels: dict[str, dict] = {}

    for src in sources:
        img = cv2.imread(str(src))
        if img is None:
            print(f"  skip (unreadable): {src.name}")
            continue

        dest = auth_dir / src.name
        shutil.copy2(src, dest)
        labels[f"authentic/{src.name}"] = {"label": "authentic", "source": src.name}

        boxes = amount_boxes(img)
        if len(boxes) < 2:
            print(f"  {src.name}: only {len(boxes)} amount boxes - no forgery made")
            continue

        # Forge the largest amount on the page; that is what a forger targets.
        target = max(boxes, key=lambda b: (b["bbox"][3] - b["bbox"][1]) * (b["bbox"][2] - b["bbox"][0]))

        made = 0
        for fn, args in ((forge_retype, (img, target, rng)),
                         (forge_copymove, (img, boxes, rng)),
                         (forge_splice_digit, (img, boxes, rng)),
                         (forge_recompress_patch, (img, target, rng)),
                         (forge_erase_field, (img, rng.choice(boxes), rng))):
            forged, meta = fn(*args)
            if forged is None:
                continue
            name = f"{src.stem}__{meta['op']}{src.suffix}"
            _write(forged_dir / name, forged, src)
            labels[f"forged/{name}"] = {"label": "forged", "source": src.name, **meta}
            made += 1

        print(f"  {src.name}: {len(boxes)} amount boxes -> {made} forgeries")

    (CORPUS / "labels.json").write_text(json.dumps(labels, indent=2))
    n_auth = sum(1 for v in labels.values() if v["label"] == "authentic")
    n_forged = len(labels) - n_auth
    print(f"\ncorpus: {n_auth} authentic, {n_forged} forged -> {CORPUS}")
    return 0


def _write(path: pathlib.Path, img: np.ndarray, like: pathlib.Path) -> None:
    """Save a forgery the way a forger would: same container, one more save."""
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    else:
        cv2.imwrite(str(path), img)


if __name__ == "__main__":
    sys.exit(main())
