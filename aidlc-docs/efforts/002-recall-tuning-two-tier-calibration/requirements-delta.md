# Requirements delta — Effort 002

Baseline: `docs/idea.md`, `docs/PLAN.md`, plus effort 001's rebuilt pipeline.
Driven by issue #2 (detection recall unmeasured / too conservative at the
corrected calibration).

## CHANGED requirements

| ID | Was (effort 001) | Now | Why |
|---|---|---|---|
| C1 | One calibration tier: quantiles of each authentic document's *worst* region, applied to every region individually | Two tiers. `screen`: pooled per-region quantiles (~2% of honest regions cross; a lead, never an accusation — needs a second independent family plus the Verifier's controls). `strong`: the per-document-extreme quantiles from effort 001 (individually damning, may accuse alone). | Applying a max-of-N bound to individual regions is a population mismatch: a forged region only needs to be unusual among its peers, not more extreme than the most extreme honest region ever calibrated. Measured cost of the one-tier design: **0/105 forgeries detected** at 0% false accusations — a detector that never fires either way. |
| C2 | Adjudicator trusts the Prober's `*_exceeds` flags | Adjudicator re-derives exceedance from the raw statistics on each card, at both tiers | The agent with answer authority should not inherit another agent's thresholding; the Prober's flags are display hints only. |
| C3 | `SOLO_SEVERITY = 3.0` calibrated units past the (single) bound lets one family accuse alone | One family may accuse alone only past the **strong** tier (beyond any authentic document's worst region) | Same intent, but expressed against a bound with a meaning, instead of an arbitrary multiplier. |
| C4 | Verifier re-tests claims at the single calibration bound | Verifier re-tests each metric at the tier the Adjudicator crossed it at (recorded per claim in `bond_detail.tiers`) | Re-testing a screen-tier claim at the strong bound would break every bond the two-tier rulebook exists to allow; the reverse would let a strong claim pass a weaker test than it made. |

## NEW requirements

| ID | Requirement | Status |
|---|---|---|
| N1 | Calibration files must be strict JSON — no bare `Infinity` tokens (issue #2, secondary item) | Done — one-sided bounds serialise as `null`, round-trip restores ±inf; regression-tested |
| N2 | Pre-two-tier calibration files must still load and behave as before (conservative), not crash | Done — missing screen bounds fall back to strong bounds; regression-tested |
| N3 | The blank control must not test metrics it trips by construction | Done — `flat_pixel_frac` exempted: a synthetic grey patch is 100% flat by definition, so the old check made every erasure claim structurally unable to hold. Flatness is controlled by the shuffle instead, where decoys are real text regions that must NOT be flat. |
| N4 | A unit-test suite must pin the tier logic and serialisation behaviour | Done — `tests/test_calibrate.py` |
| N5 | Screen-tier corroboration may only accuse **motivated regions** — money amounts, the arithmetic-implicated field. Decorative text (logos, headers) must meet the strong tier. | Done — `adjudicator._is_motivated`. Removed all 7 measured intermediate false positives, every one a screen-tier claim on decorative text. |
| N6 | Duplication may accuse alone only at the strong tier, and may not corroborate the arithmetic rule | Done — receipts legitimately repeat short amounts, and printing the total twice is the most common honest layout |
| N7 | Region proposals must not generate claims from a signal measured to be non-discriminative on the corpus class | Done — the dead-grain (erase-fill) scan was built, measured, found to fire identically on forged and authentic scans (natural JPEG grain-dead patches bracket the true fill), and removed from claim generation. Negative result recorded in `eval-results.md` and in the probe's docstring. |

## OUT of scope (unchanged from effort 001)

- VLM claim-generation layer (`docs/PLAN.md` §5, §8–9)
- AI-inpainting forgery mode (mode 6)
- Metadata-ablation control
