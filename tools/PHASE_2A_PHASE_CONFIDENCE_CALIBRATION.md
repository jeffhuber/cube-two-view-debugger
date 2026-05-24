# Phase 2a: phase-confidence trust-signal calibration

> **Legacy-data caution (2026-05-23):** this report depends on
> `post_218_baseline.json`, which depends on the legacy `near_*` axis
> fixture. Treat its row-level phase/chirality conclusions as provisional
> until regenerated from `Va/Vb + 0..5` full-corner truth.

## Question

Can `|phase_darkness_separation|` from the global cube model serve
as a trust signal that routes low-confidence geometry to retake,
hitting the Phase 2 success criterion?

> Phase 2 target: **≥80% catastrophic recall at ≤10% false-retake on GOOD cases.**

## Method

On the 116-run post-#218 baseline:

- `n_good` (GOOD outcome, err < 10°): **76**
- `n_catastrophic` (CHIRALITY_MISS + CHIRALITY_FALSE_FLIP + TRUE_GEOMETRY_FAIL): **24**
- `n_marginal` (10°–25° err — not part of success criterion): **16**

For each threshold T, the retake decision is `|phase_sep| < T`. We compute:

- `catastrophic_recall` = (catastrophic runs flagged for retake) / (all catastrophic)
- `good_false_retake_rate` = (GOOD runs flagged for retake) / (all GOOD)
- `marginal_routed_rate` = (MARGINAL runs flagged for retake) / (all MARGINAL)  — informational

## Chosen operating point

**Status: `below_recall_target`**

Target false-retake holds but recall is below 80%.

| metric | value |
|---|---|
| Threshold T (route to retake when \|sep\| < T) | **11.70** |
| Catastrophic recall | **45.8%** (11/24) |
| GOOD false-retake rate | **9.2%** (7/76) |
| MARGINAL routed rate (informational) | 37.5% (6/16) |

## Threshold curve (selected rows)

Showing thresholds where one of the metrics changes meaningfully.

| T  | catastrophic_recall | good_false_retake | marginal_routed |
|---:|--------------------:|------------------:|----------------:|
|  0.00 |   0.0% ( 0/24) |   0.0% ( 0/76) |   0.0% |
|  2.80 |   4.2% ( 1/24) |   0.0% ( 0/76) |  12.5% |
|  4.70 |  12.5% ( 3/24) |   0.0% ( 0/76) |  18.8% |
|  5.00 |  12.5% ( 3/24) |   1.3% ( 1/76) |  18.8% |
|  6.90 |  16.7% ( 4/24) |   1.3% ( 1/76) |  18.8% |
|  7.10 |  20.8% ( 5/24) |   1.3% ( 1/76) |  25.0% |
|  7.20 |  20.8% ( 5/24) |   2.6% ( 2/76) |  25.0% |
|  7.60 |  20.8% ( 5/24) |   3.9% ( 3/76) |  25.0% |
|  7.70 |  25.0% ( 6/24) |   3.9% ( 3/76) |  25.0% |
|  7.90 |  29.2% ( 7/24) |   3.9% ( 3/76) |  25.0% |
|  8.20 |  29.2% ( 7/24) |   5.3% ( 4/76) |  25.0% |
|  8.50 |  33.3% ( 8/24) |   5.3% ( 4/76) |  25.0% |
|  8.60 |  33.3% ( 8/24) |   6.6% ( 5/76) |  25.0% |
|  8.90 |  33.3% ( 8/24) |   7.9% ( 6/76) |  25.0% |
| 10.10 |  33.3% ( 8/24) |   9.2% ( 7/76) |  37.5% |
| 10.70 |  37.5% ( 9/24) |   9.2% ( 7/76) |  37.5% |
| 10.90 |  41.7% (10/24) |   9.2% ( 7/76) |  37.5% |
| 11.70 |  45.8% (11/24) |   9.2% ( 7/76) |  37.5% |
| 11.80 |  45.8% (11/24) |  11.8% ( 9/76) |  37.5% |
| 11.90 |  45.8% (11/24) |  13.2% (10/76) |  37.5% |
| 12.20 |  45.8% (11/24) |  14.5% (11/76) |  37.5% |
| 13.20 |  45.8% (11/24) |  17.1% (13/76) |  43.8% |
| 13.40 |  45.8% (11/24) |  18.4% (14/76) |  43.8% |
| 13.90 |  45.8% (11/24) |  19.7% (15/76) |  43.8% |
| 15.30 |  50.0% (12/24) |  19.7% (15/76) |  50.0% |
| 15.70 |  50.0% (12/24) |  22.4% (17/76) |  50.0% |
| 16.90 |  50.0% (12/24) |  23.7% (18/76) |  56.2% |
| 17.50 |  54.2% (13/24) |  25.0% (19/76) |  56.2% |
| 18.00 |  54.2% (13/24) |  26.3% (20/76) |  56.2% |
| 18.40 |  54.2% (13/24) |  27.6% (21/76) |  56.2% |
| ... | 64 more rows truncated | | |

## Interpretation

⚠️ The signal can meet the false-retake budget but not the catastrophic-
recall target alone. **Phase 2 needs additional signals** — either
compose with cv-local face-quad consistency (#225 finding), two-view
consistency (#200), or vertex-ensemble disagreement.

## Phase 2b candidates if multi-signal is needed

Per the Phase 2 plan in `STATE_OF_THE_WORLD.md`:

1. **cv-local face-quad structural consistency** — Phase 1 (#225) showed
   90% structural fit-fail on cv-local. Strongly predicts unreliable
   geometry independently of the phase detector.
2. **Two-view consistency** — Codex's #200 diagnostic; if A and B
   disagree, low confidence.
3. **Vertex-ensemble disagreement** — the mean-of-3 ensemble already
   computes 3 candidate vertices; their pairwise disagreement is a
   vertex-uncertainty proxy that should correlate with phase miscalls.

## Reproducing

```bash
.venv/bin/python tools/phase2a_phase_confidence_calibration.py
```
