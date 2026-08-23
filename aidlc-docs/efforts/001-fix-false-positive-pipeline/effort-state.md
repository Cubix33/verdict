# Effort 001 — Fix the false-positive detection pipeline

**State:** `in-progress` (fix-fast-path, bug track)
**Depth dial:** standard
**Owner:** AI agent session, reviewed by @Cubix33 (repo owner)

## Origin

User report: *"run it, test on various real images from web search — it's
failing, like it's saying false [tampered] to even correct [authentic]
ones."* No stack trace, no repro steps — this started as a bug
characterization, not a known fix.

## Stages

| Stage | Status | Artifact |
|---|---|---|
| Characterize root cause | ✅ done | [`root-cause-analysis.md`](root-cause-analysis.md) |
| Requirements delta | ✅ done | [`requirements-delta.md`](requirements-delta.md) |
| Build test-first (reproduction harness) | ✅ done | `tools/make_corpus.py`, `tools/probe_separation.py` |
| Construction — probes, agents, calibration, ledger | ✅ done | `verdict/probes/*`, `verdict/agents/*`, `verdict/calibrate.py`, `verdict/ledger.py`, `verdict/orchestrator.py` |
| Dashboard API compatibility | ✅ done | `dashboard/app.py` rewritten against new agent signatures |
| Full held-out evaluation run | ❌ not done | see below |
| Detection-recall tuning | ❌ not done | see below |

## What is verified vs. what is claimed

**Verified in this session** (small, bounded runs — not a full held-out eval,
by explicit instruction to keep runs short):

- The false-positive bug is fixed: authentic documents, including ones never
  seen by the calibrator, return `verdict=authentic`, not `tamper_detected`,
  across every image checked (`docs/test-images/authentic/*`,
  `testdata/corpus/authentic/*`).
- `tools/probe_separation.py` (the PLAN.md §3 acceptance gate — "at least 3
  probes must show visible separation, else stop before writing agent code")
  passes on the current code, measured on 105 forged / 22 authentic regions
  across 5 forgery modes: `ela_z` separates strongly (AUC 0.90 overall,
  0.82-0.95 per individual mode, including `splice_digit` and
  `recompress_patch`, which reuse the document's own pixels rather than the
  synthetic generator's own artifacts); `dup_score` and `grid_ratio` add 2
  more usable signals. Full table in `root-cause-analysis.md`.
- The full pipeline runs end to end without crashing on real camera JPEGs
  (previously: guaranteed crash via `json.dumps` on `piexif` bytes) and on a
  stale pre-migration `ledger.db` (previously: guaranteed `sqlite3.OperationalError`).
- All five dashboard endpoints respond correctly against the rewritten agents
  (`FastAPI.TestClient`, one request per endpoint).

**Not verified — this is the open work**, tracked in the GitHub issue this PR
links:

- A full precision/recall run of `tools/evaluate.py` over the 130-image
  labelled corpus (`tools/make_corpus.py`'s output) was started and killed
  before completion, at the user's request to stop long-running commands.
  The last bounded sample (2 authentic, 2 forged) showed 0/2 forged detected
  post-recalibration — expected, since calibrating the ELA statistic in log
  space (necessary to fix a threshold-explosion bug, see root-cause doc)
  makes the detector materially more conservative, and that new operating
  point has not been re-tuned or measured at scale.
- Per-mode **pipeline recall** (does the full run end in `tamper_detected`,
  after thresholding, corroboration, and bond verification) is unknown at the
  current calibration. What is measured is per-mode **probe separation** —
  a weaker, upstream signal that a statistic *can* distinguish forged from
  authentic regions before any threshold is applied — reported in
  `root-cause-analysis.md`. The two numbers can diverge: a conservative
  calibration can leave a probe with excellent separation (`ela_z`, AUC
  0.82-0.95 on every mode) still failing to cross its threshold often enough
  to drive the pipeline's actual recall up. Closing that gap is the point of
  the tuning work below.

## Approval gate

Requesting review before merge. The honest framing: **the reported bug is
fixed and the fix is verified**; **the detector's sensitivity at the new,
correct operating point is not yet re-measured**, which is exactly the kind
of regression a calibration-driven system can hide if nobody checks. Filed as
a follow-up issue rather than silently shipped.
