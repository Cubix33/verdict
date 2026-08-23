# Effort 002 — Recall tuning: two-tier calibration

**State:** `complete` (awaiting PR review)
**Depth dial:** standard
**Origin:** Issue #2 — detection recall unmeasured at the corrected
calibration; full-corpus evaluation needed.

## Stages

| Stage | Status | Artifact |
|---|---|---|
| Baseline measurement (full corpus, pre-change) | ✅ done | `eval-results.md` |
| Root-cause of zero recall | ✅ done | below |
| Requirements delta | ✅ done | [`requirements-delta.md`](requirements-delta.md) |
| Construction — two-tier calibration + tier-aware agents | ✅ done | `verdict/calibrate.py`, `verdict/agents/adjudicator.py`, `verdict/agents/verifier.py` |
| Strict-JSON calibration files (issue #2 secondary item) | ✅ done | `verdict/calibrate.py`, `tests/test_calibrate.py` |
| Blank-control flatness exemption | ✅ done | `verdict/agents/verifier.py` |
| Regression tests | ✅ done | `tests/test_calibrate.py` (7 tests) |
| Tuned full-corpus evaluation | ✅ done | `eval-results.md` |

## Root cause of the zero recall

The effort-001 calibrator set each metric's threshold from the distribution
of **per-document maxima** — the right statistic for controlling
document-level false positives — and then applied that bound to **individual
regions**. That is a population mismatch: over 50–100 regions, an honest
document's *worst* region routinely reaches an ELA z-score of 15–20, so the
strong bound lands around 20, while real forged regions measure 7–11. The
detector could not fire on a forgery without also firing on the worst honest
region of some calibration document — so it fired on nothing. Measured
baseline: **0/105 forgeries detected, 0/29 authentic accused.**

## The fix

Two tiers per metric (see `docs/ARCHITECTURE.md`, "Calibration tiers"):

- **screen** — pooled per-region quantile at `target_fpr` (q98 by default,
  ~2% of honest regions cross). Permissive on purpose: a screen hit alone
  never accuses. Two *independent* probe families must cross on the same
  region, and the Verifier's shuffle control must then confirm the region is
  anomalous among its own page's peers.
- **strong** — the effort-001 per-document-extreme bound, kept as the
  individually-damning tier: one family past it may accuse alone (this
  replaces the arbitrary `SOLO_SEVERITY = 3.0` multiplier).

The false-positive budget moves from "each bound alone must be rare" to "the
*conjunction* must be rare" — which is where a corroboration-plus-adversarial-
controls architecture should have been spending it all along.

Also fixed while here, both from issue #2's checklist:

- **Strict JSON calibration files** — one-sided bounds serialised as `null`
  instead of the bare `Infinity` token Python's `json` emits; ±inf restored
  on load; pre-two-tier files fall back to strong-only behaviour instead of
  crashing.
- **Blank control vs flatness** — a synthetic grey patch is 100% flat by
  construction, so testing `flat_pixel_frac` against it made every erasure
  claim structurally unable to hold. Exempted; the shuffle control (real
  text-region decoys that must NOT be flat) is the meaningful control there.

## Results

Full before/after numbers, per forgery mode, in [`eval-results.md`](eval-results.md).

## Approval gate

Requesting review on the PR. Same discipline as effort 001: every number in
`eval-results.md` is from a completed run of `tools/evaluate.py` on the
134-image corpus with calibration and evaluation sources disjoint, and
false-accusation rate is reported next to every recall figure.
