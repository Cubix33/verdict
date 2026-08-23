# Test images

A small, curated sample for trying the pipeline without downloading anything.
The full evaluation corpus (130+ images) is generated, not stored in git —
see [Regenerating the full corpus](#regenerating-the-full-corpus) below.

## Layout

```
authentic/
  002.jpg        real scanned receipt (SROIE dataset)
  016.jpg        real scanned receipt (SROIE dataset)
  bill1.png      sample bill shipped with the original repo
  example.png    sample receipt shipped with the original repo

forged/
  002__retype_amount.jpg      the total was erased and re-rendered larger
  002__copy_move_amount.jpg   one amount pasted over another, same document
  002__splice_digit.jpg       leading digit spliced in from elsewhere on the page
  002__recompress_patch.jpg   total field re-encoded at a lower JPEG quality
  002__erase_field.jpg        a line item was erased, breaking the arithmetic

labels.json      ground truth + the bounding box and generator parameters
                 used for each forgery
```

All five forged images share one source (`002.jpg`), so you can compare each
forgery directly against the same authentic original.

## Why these five modes

Two are visible edits (`retype_amount`, `copy_move_amount`); three are the
hard cases a naive detector misses, listed easiest-to-catch to hardest:

| Mode | What survives from the original file | Which probe should catch it |
|---|---|---|
| `retype_amount` | nothing at the edit site - new pixels entirely | ELA, flat-pixel-fraction |
| `copy_move_amount` | nothing local, but the pasted pixels exist elsewhere on the page | copy-move duplicate score |
| `splice_digit` | most of the page - only one glyph is foreign | ELA, block-grid alignment |
| `recompress_patch` | every pixel, including its own JPEG history | ELA (compression-generation mismatch) |
| `erase_field` | nothing - the amount is gone, the total no longer reconciles | cross-field arithmetic |

`recompress_patch` is the one worth studying first if you are trying to
understand why this project treats forensics as more than "diff the pixels":
the message the number encodes never changes, only how many times that patch
has been through a JPEG encoder.

## Try it

```powershell
# one-off analysis (uses the "sroie" calibration profile checked in for the demo -
# see docs/PLAN.md section 4 if you want to calibrate your own)
python -m verdict.orchestrator docs/test-images/forged/002__retype_amount.jpg --customer sroie
```

## Regenerating the full corpus

```powershell
# 1. download real receipts (SROIE) into testdata/raw and testdata/calib -
#    see docs/PLAN.md section 1 for the exact URLs used
# 2. generate labelled forgeries from every image in testdata/raw
python tools/make_corpus.py

# 3. calibrate on the disjoint testdata/calib set (never on eval images)
python -m verdict.calibrate testdata/calib --customer sroie --limit 12

# 4. run the acceptance gate and the full evaluation
python tools/probe_separation.py
python tools/evaluate.py --customer sroie --exclude-sources sroie_021.jpg,sroie_022.jpg,...
```
