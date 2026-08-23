# VERDICT — Architecture

This is the low/high-level design for the rebuilt pipeline. For the product
concept and motivation, see [`idea.md`](idea.md); for the original build plan
this follows, see [`PLAN.md`](PLAN.md).

## 1. Data flow

```
image path
   │
   ▼
┌─────────────┐   risk ∈ [0,1], flags,        ┌──────────────┐
│   TRIAGE    │──►optional target_bbox ───────►│    PROBER    │
└─────────────┘   (never an accusation)        └──────┬───────┘
                                                        │ registers EvidenceCards
                                                        ▼
                                                ┌──────────────┐
                                                │    LEDGER    │  SQLite, content-addressed
                                                │ (per claim)  │  PRIMARY KEY (claim_id, card_id)
                                                └──────┬───────┘
                                       manifest() text │ (no image, ever)
                                                        ▼
                                                ┌──────────────┐
                                                │ ADJUDICATOR  │  🔒 starved
                                                └──────┬───────┘
                                              bonded Claim list
                                                        ▼
                                                ┌──────────────┐
                                                │   VERIFIER   │  🔒 independent re-derivation
                                                └──────┬───────┘
                                             claim.bond_status
                                                        ▼
                                                ┌──────────────┐
                                                │ ORCHESTRATOR │  only bonded claims -> verdict
                                                └──────────────┘
```

Everything is keyed by a `claim_id` generated at the start of a run. The
Ledger's SQLite schema uses `(claim_id, card_id)` as its primary key
specifically so that two unrelated analyses can never share evidence — see
[§6](#6-what-changed-and-why-it-was-necessary) for why the original schema
allowed exactly that.

## 2. Core types (`verdict/types.py`)

```python
@dataclass(frozen=True)
class EvidenceCard:
    id: str
    probe: str                  # "region_forensics" | "cross_field" | "provenance"
    params: dict
    bbox: list | None
    px_on_target: int           # TRUE pixel size before any upscaling
    crop_sha256: str            # the bond anchor
    crop_path: str | None
    observation: str            # short, factual, machine-generated
    numeric: dict                # every calibrated statistic this region supports
    cost_units: float

@dataclass
class Claim:
    id: str
    text: str
    bond: list[str]              # EvidenceCard ids this claim is bonded to
    bond_status: str = "untested"  # untested | holds | broken
    bond_detail: dict = field(default_factory=dict)

@dataclass
class Verdict:
    claim_id: str
    verdict: str      # tamper_detected | authentic | undetermined | insufficient_evidence
    confidence: float | None
    claims: list[Claim]
    controls: dict
    cost: dict
    latency_s: float
    evidence_chain: list[str]
```

`px_on_target` is load-bearing: a probe that ran on a region below its
calibrated minimum resolution is not entitled to an opinion, and this is the
field that gate is checked against (`idea.md` Finding 6 — gate on measured
pixels, never on model confidence).

## 3. The Ledger (`verdict/ledger.py`)

- Backed by SQLite (`data/ledger.db`) plus a `data/crops/` directory of
  content-addressed PNGs.
- `add()` computes the SHA-256 of the PNG-encoded crop and only writes the
  file if that hash isn't already on disk — identical crops across claims
  share storage but never share metadata rows, because the row key includes
  `claim_id`.
- `verify_provenance(claim_id, card_id)` re-hashes the crop on disk and
  compares it to the value recorded at write time. This is the mechanism a
  Verifier check calls before doing anything else: if a crop was tampered
  with *after* being registered, the bond fails before any probe re-runs.
- `manifest(claim_id)` — the *only* representation the Adjudicator ever
  receives. It is always scoped to one claim.
- `purge(claim_id)` removes a claim's rows and deletes crop files not
  referenced by any other claim.

## 4. Probes (`verdict/probes/`)

All probes are deterministic, offline, and side-effect-free on their inputs.
Every probe that scores a region does so **relative to other text regions on
the same document** (`ocr.text_regions`), never against the whole page or a
fixed constant — a receipt is mostly blank paper, and comparing dense glyph
regions to that background guarantees every word looks anomalous.

| Module | Statistic | Anomalous direction | Independent of |
|---|---|---|---|
| `compression.py` | `ela_z` — multi-scale ELA residual per unit of local detail, log-scaled, MAD z-scored against peer regions | high | `noise.py`, `typography.py` |
| `compression.py` | `grid_ratio` — alignment to the JPEG 8×8 block lattice | low (grid absent) | — |
| `noise.py` | `hf_ratio` — local high-frequency variance vs peer median | either (composited-in noisier; filled-in smoother) | `compression.py` |
| `noise.py` | `flat_pixel_frac` — fraction of gradient-free pixels | high | `compression.py` |
| `copymove.py` | `dup_score` — best normalised template-match of a region against the rest of the page | high | everything else |
| `typography.py` | `stroke_ratio`, `baseline_z` — stroke width and baseline offset vs the *other words on the same row* | either / high | everything else |
| `cheap.py` | `cross_field` — reconciles a labelled total against its own line items | mismatch beyond tolerance | pixel probes |

`ocr.py` underlies all of them: one OCR pass per image (memoised on pixel
content), multi-PSM voting for low-DPI scans, geometric row grouping (not
Tesseract's block/paragraph/line indices, which split a receipt row wherever
the amount column is far from the label), and a strict money-amount regex
that requires a decimal part — this is what keeps dates, phone numbers and
item codes out of the arithmetic check.

## 5. Agents

### Triage (`agents/triage.py`)
Zero model cost. Computes `exif_audit`, `perceptual_hash`, `cross_field`, and
turns them into a **risk score**, not a verdict — the weight on "no EXIF" is
deliberately near zero, since PNGs, screenshots and scans legitimately lack
it. Escalation is a request for the Prober to look closer, never an
accusation on its own.

### Prober (`agents/prober.py`)
Selects up to `MAX_REGIONS` candidate regions from three sources, in order:
the field the arithmetic check implicates, **pixel-level outliers found by
scanning the ELA and noise maps directly** (this is what lets the system
notice a region that was erased or retyped badly enough that it no longer
OCRs as text at all), and OCR-detected money amounts. Every region gets one
`EvidenceCard` carrying every statistic its pixel area was large enough to
support, each already compared against this deployment's calibration
(`numeric["ela_exceeds"]`, etc.) so the Adjudicator never has to see a raw
image to know a bound was crossed.

### Adjudicator (`agents/adjudicator.py`) — 🔒 starved
Receives `cards: list[EvidenceCard]` and a `Calibration`. **Never receives an
image.** Requires corroboration from **two independent probe families**
(`compression`, `texture`, `duplication`, `typography` — grouped so that
`hf_ratio` and `flat_pixel_frac`, which measure the same underlying texture
statistic, cannot satisfy the rule by themselves) before raising a
`tamper_detected`-track claim. A single-family signal may still raise a claim
if it is `SOLO_SEVERITY` calibrated units past the bound — set high on
purpose, since one probe alone should almost never be enough. Verdicts:

- `tamper_detected` — at least one claim, later bonded by the Verifier
- `authentic` — regions were probed and none crossed calibration
- `undetermined` — only provenance-level findings (e.g. an editor tag), which describe the file without accusing a specific pixel region
- `insufficient_evidence` — no region carried enough pixels to probe at all

### Verifier (`agents/verifier.py`) — 🔒 independent
A bond holds only if all four pass:

1. **Provenance** — the cited card's crop still hashes to its recorded value.
2. **Reproduction** — the flagged statistic is recomputed from scratch on the crop on disk and still exceeds the calibrated bound.
3. **Blank control** — the same probe on a flat grey patch of identical size must *not* reproduce the finding.
4. **Shuffle control** — the same probe on several other text regions of *this same document* must not reproduce it in more than `SHUFFLE_TOLERANCE` of them. Decoys are drawn from other OCR-detected text regions, not random coordinates — a patch of blank margin fails every probe trivially, so a control built on it passes by construction and tests nothing.

### Orchestrator (`orchestrator.py`)
Wires the four agents and enforces that **only claims whose bond holds
contribute to the verdict**. An Adjudicator that raised a confident
`tamper_detected` with zero surviving bonds ships as `undetermined` — the
accusation is structurally void, not merely downweighted.

## 6. What changed, and why it was necessary

See the [root-cause analysis](../aidlc-docs/efforts/001-fix-false-positive-pipeline/root-cause-analysis.md)
for the full list with line references. The two structural changes worth
calling out specifically:

- **Claim-scoped storage everywhere.** The original `Ledger._cache` was a
  single `dict[card_id, EvidenceCard]` shared by the whole process; every
  agent function took the ledger and iterated its full cache. This branch
  changes every read path to take an explicit `claim_id` and scopes both the
  in-memory cache and the SQL schema's primary key to `(claim_id, card_id)`.
  Any code that calls `ledger.cards()` or `adjudicate_from_ledger()` without
  a claim id is a signal the scoping regressed.
- **Calibration is now a hard dependency, not an optional nicety.**
  `verdict.calibrate.load()` raises `FileNotFoundError` with an actionable
  message rather than falling back to a guessed threshold. A pipeline running
  on numbers nobody measured is the exact defect this rebuild exists to
  remove, so an uncalibrated deployment is made to fail loudly instead of
  quietly accusing everyone.

## 7. Known gaps

Tracked in detail in the repository issue tracker; summarised in the main
[README §7](../README.md#7-current-status-and-honest-limitations).
