"""One OCR pass per image, shared by every probe.

Before this module each probe called `image_to_data` on its own with slightly
different filters, so `cheap.find_numeric_boxes` and `typography.font_metrics`
could disagree about what text was even on the page - and a single analysis
paid for four full OCR passes.  Everything now goes through `read()`, which
memoises on the image's content hash.

The unit the rest of the system reasons about is the `Line`: a run of words
sharing an OCR block/paragraph/line index.  Receipts are line-structured
documents, and both the arithmetic check and the typography check need to
compare a word against *its own line* rather than against the whole page.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

if not shutil.which("tesseract"):
    _WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_WIN):
        pytesseract.pytesseract.tesseract_cmd = _WIN


def tesseract_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# A money amount: optional currency sign, digit groups, and - crucially - a
# decimal part.  Requiring the cents is what keeps dates, phone numbers, item
# codes and quantities out of the arithmetic check.
_AMOUNT_RE = re.compile(
    r"""^[^\d\-+%]{0,3}?         # leading currency symbol / stray glyph
        (?P<sign>-)?
        (?P<int>\d{1,3}(?:[,\s]\d{3})+|\d+)
        (?P<dec>[.,]\d{2})
        (?![\d.,])               # no further digits: 1.00.00 is an OCR merge
        [^\d%]{0,3}$             # trailing '-', '*', 'CR'; '%' is NOT money
    """,
    re.VERBOSE,
)

# Words that mark the line carrying the document's grand total.
TOTAL_WORDS = ("total", "amount due", "amountdue", "grand total", "balance due",
               "nett total", "net total", "amount payable", "sum", "to pay")
# Total-like labels that are NOT the grand total and must not be mistaken for it.
NON_TOTAL_WORDS = ("subtotal", "sub total", "sub-total", "total qty", "total item",
                   "total quantity", "item total", "total items", "total b/f")
TAX_WORDS = ("tax", "gst", "vat", "sst", "service charge", "svc", "cess", "duty")
DISCOUNT_WORDS = ("discount", "disc", "rounding", "round off", "adjustment",
                  "voucher", "coupon", "redeem")
PAYMENT_WORDS = ("cash", "change", "tendered", "card", "visa", "master", "paid",
                 "payment", "debit", "credit", "balance")


@dataclass
class Word:
    text: str
    bbox: list[int]          # [x1, y1, x2, y2]
    conf: float
    line_key: tuple

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]


@dataclass
class Line:
    key: tuple
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def bbox(self) -> list[int]:
        xs = [w.bbox[0] for w in self.words] + [w.bbox[2] for w in self.words]
        ys = [w.bbox[1] for w in self.words] + [w.bbox[3] for w in self.words]
        return [min(xs), min(ys), max(xs), max(ys)]

    def amounts(self) -> list[tuple[Word, float]]:
        out = []
        for w in self.words:
            v = parse_amount(w.text)
            if v is not None:
                out.append((w, v))
        return out


@dataclass
class Page:
    words: list[Word]
    lines: list[Line]
    shape: tuple[int, int]

    def amount_words(self) -> list[tuple[Word, float]]:
        out = []
        for w in self.words:
            v = parse_amount(w.text)
            if v is not None:
                out.append((w, v))
        return out


def parse_amount(token: str) -> float | None:
    """Parse a money amount, or return None if the token is not one.

    Deliberately strict: `12.50` and `1,234.00` parse, but `2024`, `0123456789`
    and `12` do not.  The old code accepted any run of digits, which is why a
    phone number could end up being treated as a receipt total.
    """
    t = str(token).strip()
    if not t:
        return None
    m = _AMOUNT_RE.match(t)
    if not m:
        return None
    whole = m.group("int").replace(",", "").replace(" ", "")
    dec = m.group("dec")[1:]
    try:
        val = float(f"{whole}.{dec}")
    except ValueError:
        return None
    if m.group("sign"):
        val = -val
    # A receipt amount with more than 9 integer digits is an OCR merge artefact.
    if abs(val) >= 1e9:
        return None
    return val


def _key(d: dict, i: int) -> tuple:
    return (d["page_num"][i], d["block_num"][i], d["par_num"][i], d["line_num"][i])


_CACHE: dict[str, Page] = {}


# Page-segmentation modes worth trying, cheapest-looking first.  Receipts are
# a single narrow column, which psm 4 and 6 model far better than the default
# psm 3 - on tall scans psm 3 routinely drops the flush-right amount column.
_PSM_MODES = ("--oem 3 --psm 6", "--oem 3 --psm 4", "--oem 3 --psm 3")

# Below this height Tesseract's classifier degrades badly; receipts scanned at
# 150 dpi land here often, and upscaling first is the single biggest accuracy
# win available without a different engine.
_MIN_OCR_HEIGHT = 1400


def _preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Grayscale, upscale small pages, and flatten uneven scan lighting.

    Returns the prepared image and the scale factor applied, so callers can
    map boxes back to original pixel coordinates - every downstream probe
    measures *original* pixels, so boxes must never stay in upscaled space.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    scale = 1.0
    h = gray.shape[0]
    if h < _MIN_OCR_HEIGHT:
        scale = min(3.0, _MIN_OCR_HEIGHT / max(h, 1))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Divide out the illumination field: thermal receipts and phone photos both
    # have strong gradients that push whole regions below the binarisation point.
    bg = cv2.medianBlur(gray, 31)
    norm = cv2.divide(gray, bg, scale=255)
    return norm, scale


def _run_psm(prepared: np.ndarray, config: str) -> dict | None:
    try:
        return pytesseract.image_to_data(prepared, config=config, output_type=Output.DICT)
    except Exception:
        return None


def read(img_bgr: np.ndarray, min_conf: float = 40.0) -> Page:
    """OCR an image once and cache the result on its pixel content.

    Tries several page-segmentation modes and keeps whichever recovers the most
    money amounts.  That choice is deliberate: the arithmetic probe is the only
    one that can accuse a named field, and it is useless if the amount column
    was never read.
    """
    key = hashlib.sha256(np.ascontiguousarray(img_bgr).tobytes()).hexdigest()
    ck = f"{key}:{min_conf}"
    hit = _CACHE.get(ck)
    if hit is not None:
        return hit

    prepared, scale = _preprocess(img_bgr)
    shape = img_bgr.shape[:2]

    best_words: list[Word] = []
    best_amounts = -1
    for config in _PSM_MODES:
        d = _run_psm(prepared, config)
        if d is None:
            continue
        words = _words_from(d, min_conf, scale)
        n_amounts = sum(1 for w in words if parse_amount(w.text) is not None)
        if n_amounts > best_amounts:
            best_words, best_amounts = words, n_amounts
        if n_amounts >= 6:
            break              # plenty of structure; no need to pay for more passes

    page = Page(words=best_words, lines=_group_rows(best_words), shape=shape)
    _CACHE[ck] = page
    return page


def _words_from(d: dict, min_conf: float, scale: float) -> list[Word]:
    words: list[Word] = []
    inv = 1.0 / scale if scale else 1.0
    for i, raw in enumerate(d["text"]):
        t = str(raw).strip()
        if not t:
            continue
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < min_conf:
            continue
        x, y = int(d["left"][i]), int(d["top"][i])
        w, hh = int(d["width"][i]), int(d["height"][i])
        if w <= 0 or hh <= 0:
            continue
        bbox = [int(round(x * inv)), int(round(y * inv)),
                int(round((x + w) * inv)), int(round((y + hh) * inv))]
        words.append(Word(text=t, bbox=bbox, conf=conf, line_key=_key(d, i)))
    return words


def _group_rows(words: list[Word]) -> list[Line]:
    """Group words into visual rows by vertical overlap.

    Tesseract's own block/paragraph/line indices split a receipt row whenever
    the label and the amount are separated by a wide gap - which is every row
    on a receipt, because the amount column is flush right.  Grouping on
    geometry instead recovers the label and its amount as one row, which is
    what the arithmetic check needs.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.bbox[1] + w.bbox[3]) / 2.0)
    median_h = float(np.median([w.height for w in ordered])) or 10.0
    tol = max(4.0, median_h * 0.6)

    rows: list[list[Word]] = []
    current: list[Word] = [ordered[0]]
    centre = (ordered[0].bbox[1] + ordered[0].bbox[3]) / 2.0
    for w in ordered[1:]:
        c = (w.bbox[1] + w.bbox[3]) / 2.0
        if abs(c - centre) <= tol:
            current.append(w)
            centre = sum((x.bbox[1] + x.bbox[3]) / 2.0 for x in current) / len(current)
        else:
            rows.append(current)
            current, centre = [w], c
    rows.append(current)

    lines = []
    for i, ws in enumerate(rows):
        ws.sort(key=lambda w: w.bbox[0])
        lines.append(Line(key=(i,), words=ws))
    return lines


def text_regions(page: Page, img_bgr: np.ndarray | None = None,
                 min_h: int = 8, min_w: int = 10,
                 min_contrast: float = 12.0) -> list[list[int]]:
    """Word boxes usable as a reference population for per-region statistics.

    Anomaly scores are only meaningful relative to how the *rest of this
    document* behaves, so probes need a consistent set of comparable regions.

    When the image is supplied, boxes with no real ink are dropped. Tesseract
    reliably hallucinates a few low-confidence words in blank margins, and
    those regions are perfectly uniform - leaving them in makes "this region is
    suspiciously flat" true of every document ever calibrated, which silently
    disables the probe that catches erasures.
    """
    out = []
    for w in page.words:
        if w.height < min_h or w.width < min_w:
            continue
        if img_bgr is not None:
            x1, y1, x2, y2 = w.bbox
            sub = img_bgr[y1:y2, x1:x2]
            if sub.size == 0:
                continue
            g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if sub.ndim == 3 else sub
            if float(g.std()) < min_contrast:
                continue
        out.append(list(w.bbox))
    return out


def pad_bbox(bbox, shape, pad: int = 3) -> list[int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    return [max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)]


def clear_cache() -> None:
    _CACHE.clear()
