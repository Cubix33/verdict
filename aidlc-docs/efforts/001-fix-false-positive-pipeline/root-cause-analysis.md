# Root-cause analysis — 100% false-positive rate

## Symptom

> "run it, test on various real images from web search — it's failing, like
> it's saying false [tampered] to even correct [authentic] ones"

## Reproduction

13 real, unmodified images were tested against the pipeline as originally
committed (commit `1c5f276`): 10 scanned receipts from the public SROIE
dataset, `data/example.png`, `data/bill1.png`, and a Wikimedia landscape
photo with no receipt content at all.

| image | triage decision | reason | claims raised (cumulative) | bonds held |
|---|---|---|---|---|
| sroie_000.jpg | escalate | arithmetic_mismatch | 1 | 1 |
| sroie_001.jpg | escalate | arithmetic_mismatch | 2 | 2 |
| sroie_003.jpg | escalate | arithmetic_mismatch | 3 | 3 |
| … | … | … | … | … |
| sroie_010.jpg | escalate | arithmetic_mismatch | 10 | 10 |
| example.png | escalate | missing_exif | 11 | 10 |
| bill1.png | escalate | missing_exif | 12 | 11 |
| wm_landscape.jpg (no receipt at all) | escalate | missing_exif | 12 (+0 of its own) | 11 |

**13/13 authentic images → escalated, and every one accumulated tamper
claims.** The "claims raised" column climbing monotonically (1, 2, 3, … 12)
rather than resetting per image is itself a second, independent bug — see
root cause 3. `wm_landscape.jpg` contributed zero evidence of its own (it has
no receipt content to OCR) yet still inherited all 12 prior claims and 11
prior bonds from unrelated images analysed earlier in the same process. Real
camera JPEGs (as opposed to PNGs) crashed the pipeline outright before even
reaching this table — see root cause 5.

## Root causes

### 1. No calibration layer — thresholds were unmeasured literals

`verdict/agents/adjudicator.py` (original):
```python
if abs(delta) > 0.5:
    claims.append(Claim(..., text="total field digitally altered", ...))
if z > 3.0:
    claims.append(Claim(..., text="localized ELA anomaly", ...))
```
Neither `0.5` nor `3.0` was ever measured against a real document. Building
`verdict/calibrate.py` (new) and running it on 12 real, disjoint authentic
receipts shows the *normal* worst-scoring region on an honest document
reaches an ELA z-score of 15–20 — a receipt has 50–100 OCR-detected text
regions, and the maximum of that many draws from any real-world statistic is
large even when nothing is wrong. `z > 3.0` was not a strict threshold; it
was inside the noise floor.

`docs/PLAN.md` §4 specifies exactly this calibrator ("VERDICT ships with no
thresholds") and its own ablation table (§13.2) predicts the consequence of
skipping it: `no_calibration` → false-accusation rate `1.7% → 21.4%`. The
measured reality (100%) is worse than the plan's own worst-case estimate,
consistent with the arithmetic bug (root cause 2) firing independently on
every document regardless of calibration.

**Fix:** `verdict/calibrate.py`. Thresholds are the target-FPR quantile of
each metric's *per-document maximum* over the authentic corpus, with a
small-sample margin. `calibrate.load()` raises `FileNotFoundError` with an
actionable message if no calibration exists — an uncalibrated deployment must
fail loudly, not silently guess.

### 2. Cross-field arithmetic accused every receipt by construction

`verdict/probes/cheap.py` (original):
```python
largest = max(vals)              # any number anywhere on the page
others = [v for v in vals if v != largest]
delta = largest - sothers        # subtract EVERY other number: dates, phones, qty...
```
On a real receipt, "every other number" includes the date, the phone number,
the till ID, and every item quantity. This provably cannot balance to zero
except by chance, so `arithmetic_mismatch` fired on essentially all input —
confirmed by 10/10 SROIE receipts in the reproduction table above.

**Fix:** `verdict/probes/cheap.py::cross_field`. Requires OCR to find a line
whose *label* says "total" (with `subtotal`/`total qty` explicitly excluded
from matching as a substring), sums only line items above that line, and
returns `insufficient_structure` (raising nothing) when the document is not
legible as a structured receipt. Money amounts are now parsed with a strict
regex requiring a decimal part (`verdict/probes/ocr.py::parse_amount`), which
alone removes dates, phone numbers, and bare quantities from consideration.

### 3. Evidence leaked across unrelated images

`verdict/ledger.py` (original):
```python
class Ledger:
    def __init__(self, ...):
        self._cache: dict[str, EvidenceCard] = {}   # keyed by card_id ONLY

    def manifest(self, ids=None) -> str:
        cards = [c for k, c in self._cache.items() if ids is None or k in ids]
```
`self._cache` is one dictionary for the entire process lifetime, and
`verdict/agents/prober.py` held a **module-level `LEDGER` singleton**
imported by the dashboard. Every agent function (`adjudicate_from_ledger`,
`verify_claims`) iterated this cache directly with no concept of "this
claim's evidence" versus "everyone's evidence." In the reproduction run this
is why image #10 in the batch carried 10 claims and the unrelated landscape
photo — which has no receipt content and generated zero evidence of its own
— still inherited 12 tamper accusations belonging to other people's receipts.

**Fix:** `verdict/ledger.py` schema change: `cards` table primary key is now
`(claim_id, card_id)`, not `sha`. Every read path (`cards()`, `get()`,
`manifest()`, `verify_provenance()`) requires an explicit `claim_id`
parameter. `verdict/agents/adjudicator.py::adjudicate()` takes
`cards: list[EvidenceCard]` directly rather than a ledger handle, so it is
structurally unable to see any claim's evidence but the one it was given. A
migration (`Ledger._migrate`) handles pre-existing `claims` table rows; the
`cards` table primary-key change requires a fresh database, documented in
`verdict/ledger.py`.

### 4. The forensic probes existed but were never invoked

`verdict/probes/ela.py`, `noise.py`, `dct_quant.py`, `typography.py` were
present in the original repository and completely unused: `verdict/agents/
prober.py` called only `cheap.exif_audit`, `cheap.perceptual_hash`, and
`cheap.cross_field`, then wrote three fixed cards (a numeric-region crop, a
256×256 thumbnail, and a blank 32×32 placeholder standing in for the EXIF
result). The Adjudicator's `z > 3.0` ELA branch (root cause 1) was checking a
field (`numeric["z_score"]`) that no code path ever populated — dead code
that happened to never fire, masking the fact that no forensic evidence was
ever being gathered at all. Every verdict the system issued was backed by
arithmetic and EXIF-absence alone.

**Fix:** `verdict/agents/prober.py` now runs the full probe suite —
`compression.py` (rewritten from `ela.py`/`dct_quant.py`), `noise.py`
(rewritten), `copymove.py` (new), `typography.py` (rewritten) — against every
selected region, writing every statistic the region's pixel area can support
onto its `EvidenceCard.numeric`.

### 5. `piexif` bytes crashed `json.dumps` on every real camera photo

`verdict/agents/prober.py` (original):
```python
card3 = LEDGER.add(..., params=ex, ...)   # ex["make"]/["model"]/["software"] are raw bytes
# -> verdict/ledger.py: json.dumps(params)
# TypeError: Object of type bytes is not JSON serializable
```
`piexif.load()` returns `Make`/`Model`/`Software`/`DateTime` as `bytes`. Any
PNG or already-stripped image skipped this path; any authentic JPEG straight
off a phone or camera hit it and crashed the pipeline before a verdict was
ever produced — a second failure mode entirely separate from the false
positives, and arguably worse, since it affects the single most common real
input (an unedited camera photo).

**Fix:** `verdict/probes/cheap.py::_jsonable` and `verdict/ledger.py::_safe`
recursively decode bytes and coerce numpy scalars before anything reaches
`json.dumps`.

### 6. The Verifier's bond tests could not fail

The original Adjudicator only ever bonded a claim to the `cross_field`
probe's card (the `z_score` branch was dead per root cause 4, so the
`perceptual_hash`/`exif_audit` branches of the Verifier below were never
actually reached by the live code path — latent bugs, not contributors to
the measured 100%). The branch that *was* exercised, in
`verdict/agents/verifier.py` (original):
```python
boxes = cheap.find_numeric_boxes(img, min_h=8)
holds = len(boxes) > 0                       # true of nearly every receipt crop
```
This checks whether OCR found *any* digit inside the crop at all — unrelated
to whether the arithmetic the claim accused actually failed to reconcile. A
claim saying "the total was altered" held as long as the cited crop
contained a number, which every cross-field evidence card does by
construction. The bond test could not meaningfully fail, which is why the
reproduced accusations in the table above came back bonded at a 10/10 to
11/12 rate rather than being caught and struck.

Two further branches were real bugs but not on the path that produced the
reproduced numbers: the perceptual-hash check compared a 64-bit dHash
against the first 16 hex characters of an unrelated SHA-256 and then
discarded the comparison with `or True`; the EXIF check's pass condition was
inverted (`holds = has_exif`, so a claim founded on the *absence* of EXIF
metadata could never hold). Both are fixed as part of the same rewrite,
since the new Adjudicator can bond claims to region-forensics cards that
exercise every Verifier branch.

**Fix:** `verdict/agents/verifier.py` rewritten around four sequential
checks — provenance, independent reproduction of the flagged statistic,
blank control, shuffle control against peer text regions on the same
document — documented in full in `docs/ARCHITECTURE.md` §5.

## Verification performed

- Bounded manual runs (not a full corpus sweep, per instruction to keep
  commands short) confirm authentic documents now return `authentic`.
- `tools/probe_separation.py` — the acceptance gate `docs/PLAN.md` §3
  specifies before any agent code should be trusted — passes on the code in
  this PR, measured on 105 forged / 22 authentic regions across 5 forgery
  modes:

  ```
  probe metric          forged med    auth med     AUC  |AUC-.5|
  ela_z                     5.5390      0.3090   0.897     0.397   <== STRONG
  dup_score                 0.4497      0.7983   0.200     0.300   <== STRONG
  grid_ratio                1.0000      1.0000   0.324     0.176   <- usable
  height_ratio              1.0000      1.0335   0.367     0.133
  flat_pixel_frac           0.0259      0.0109   0.606     0.106
  hf_ratio                  0.9025      0.6248   0.604     0.104
  baseline_z                0.8020      1.5835   0.396     0.104
  stroke_ratio              1.0000      1.0020   0.430     0.070
  modal_fraction            0.1429      0.0810   0.564     0.064

  per-mode AUC (ela_z):  copy_move .882  erase_field .950  recompress .816
                         retype .933     splice_digit .903
  ```

  `ela_z` separates every one of the 5 forgery modes strongly on its own
  (AUC 0.82-0.95), including the two modes built specifically to defeat a
  naive detector by reusing the document's own pixels (`splice_digit`: only
  one glyph is foreign; `recompress_patch`: every pixel is untouched, only
  its JPEG generation count differs). It is not the only signal that
  matters, though: `dup_score` and `hf_ratio` both go completely blind on
  `erase_field` (AUC 0.000 — erasure removes content rather than adding or
  moving it, so there is nothing to duplicate-match and no new high-frequency
  texture to detect), while `flat_pixel_frac` catches exactly that mode
  best (AUC 0.998, its strongest result of the five). No single probe wins
  on every mode, which is the empirical case for the Adjudicator's
  corroboration rule treating these as independent families rather than
  picking one "best" probe.
- `FastAPI.TestClient` smoke test confirms all five dashboard endpoints
  respond correctly against the rewritten agent signatures.
- **Not yet verified**: full-corpus precision/recall at the corrected
  (log-space) calibration. See `effort-state.md` and the linked GitHub issue.
