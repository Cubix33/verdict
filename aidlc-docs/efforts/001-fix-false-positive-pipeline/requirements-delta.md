# Requirements delta — Effort 001

Compared against the baseline design in `docs/idea.md` and `docs/PLAN.md`.
Nothing here changes the product concept; this delta is entirely "make the
shipped code match the design it already committed to."

## NEW requirements (design existed, code did not)

| ID | Requirement | Baseline reference | Status |
|---|---|---|---|
| R1 | A calibration mechanism must set every anomaly threshold from documents a deployment certifies as authentic. No threshold may be a shipped literal. | `idea.md` Mechanism 3; `PLAN.md` §4 | Done — `verdict/calibrate.py` |
| R2 | The Prober must run the forensic probes (ELA, noise, typography, copy-move, JPEG quantization) and attach their output to evidence cards. | `PLAN.md` §7 | Done — `verdict/agents/prober.py` |
| R3 | The Verifier must apply a blank control and a shuffle control to every claim before its bond can hold. | `idea.md` Mechanism 2 | Done — `verdict/agents/verifier.py` |
| R4 | Evidence and claims must be scoped per analysis (`claim_id`); one analysis must never see another's evidence. | Implicit in `idea.md`'s bond model — a bond that can cite another claim's evidence is not a bond | Done — `verdict/ledger.py` schema change |
| R5 | The system must be able to return `authentic` as a first-class, positively-evidenced outcome, not just "no accusation raised." | `idea.md` §5 refusal/authenticity framing | Done — `verdict/agents/adjudicator.py` |

## CHANGED requirements (design existed, code contradicted it)

| ID | Was | Now | Why |
|---|---|---|---|
| C1 | Triage escalates on `missing_exif` alone, treated as near-accusation-strength | Triage records EXIF absence at near-zero weight; escalation requires reconciliation failure, an editor tag, or a phash duplicate | `idea.md` explicitly reserves accusation for the Adjudicator; screenshots/scans/PNGs legitimately lack EXIF |
| C2 | Cross-field arithmetic takes the largest number on the page as "the total" | Cross-field arithmetic requires a line whose *label* says total, and returns `insufficient_structure` (raises nothing) if the document isn't legible as a receipt | The naive version accused 100% of real receipts; matching `PLAN.md`'s own framing that this probe should be "the cheapest, sharpest" one, not the noisiest |
| C3 | The Adjudicator accuses on any single flagged statistic | Requires corroboration from 2 independent probe families, or one family at high calibrated severity | Single-signal accusation is a false-positive engine on a document with 50+ regions; not explicitly specified in the baseline docs but required to satisfy R1's intent |

## Explicitly OUT of scope for this effort

- The VLM/LLM claim-generation layer described in `PLAN.md` §5 and §8–9
  (grammar-locked natural-language claims). The Adjudicator here reasons
  directly over calibrated statistics — a stronger and fully deterministic
  MVP of the starvation mechanism, but not what the plan describes long-term.
- AI-inpainting forgery mode (`PLAN.md` §1.3, "Mode 6", marked optional there).
- Metadata-ablation control (`PLAN.md` §10, third of three controls) — blank
  and shuffle are implemented; metadata ablation is not.
- Full precision/recall evaluation at the corrected (log-space) calibration —
  see `effort-state.md` and the linked GitHub issue.
