# Phase 4 trust ranker v1

**Status: diagnostics-only.** First learned-classifier pass at the
Phase 2 bar after Phase 2B's hand-tuning hit its ceiling (see
`PHASE_2B_TRUST_SIGNAL_MATRIX_RECOMPUTED.md`). 4 model classes
evaluated on the same 58-case eval via leave-one-case-out CV.
No production behavior change.

## Headline

**No learned model clears the Phase 2 bar on out-of-fold CV (≥80% catastrophic recall AND ≤10% GOOD FPR).** Best-in-class is `random_forest` at 80% recall / 18% FPR — meaningfully better than Phase 2B's hand-tuned ceiling of 80% / 31% at the same recall, but still above the 10% FPR target. The learned model captures multi-feature structure (non-axis-aligned cuts) that hand-tuning missed; what's still missing is data and/or a stronger feature.

## Models evaluated

| model | clears bar? | out-of-fold FPR at 80% recall (vs Phase 2B ceiling: 31%) | out-of-fold recall at 10% FPR (vs Phase 2B: ~50%) | in-sample FPR at 80% recall |
|---|---|---|---|---|
| `logistic_regression` | ❌ | 39% (80% recall) | 50% (9% FPR) | 26% |
| `gradient_boosting` | ❌ | 30% (80% recall) | 35% (9% FPR) | 0% |
| `random_forest` | ❌ | 18% (80% recall) | 40% (9% FPR) | 0% |
| `mlp_16_8` | ❌ | 27% (80% recall) | 50% (9% FPR) | 0% |

**Reading the table:**
- "Clears bar" = some threshold achieves both ≥80% catastrophic recall AND ≤10% GOOD FPR on out-of-fold predictions.
- Lower FPR is better at fixed recall.
- Higher recall is better at fixed FPR.
- The in-sample column is what you'd get if you trained AND evaluated on the same data (no held-out). Compare to out-of-fold to see the generalization gap.

## Dataset

- 116 per-case-per-run rows across 58 cases
- Outcome breakdown:

| category | n |
|---|---|
| CHIRALITY_FALSE_FLIP | 7 |
| CHIRALITY_MISS | 12 |
| GOOD | 74 |
| MARGINAL | 22 |
| TRUE_GEOMETRY_FAIL | 1 |

- Catastrophic = `CHIRALITY_MISS` ∪ `CHIRALITY_FALSE_FLIP` ∪ `TRUE_GEOMETRY_FAIL`
- GOOD = `GOOD`; MARGINAL is intermediate (not penalized in bar)

## Features

6 continuous signals from the Phase 2B matrix
(no new features in v1; v2 candidates documented in
`STATE_OF_THE_WORLD.md`). Order:

- `fit_residual_rms_px`
- `pnp_rms_px`
- `hexagon_centroid_vs_bezel_vertex_offset_px`
- `junction_score_at_ensemble`
- `ensemble_shift_px`
- `phase_darkness_separation`

### Headline model feature importance (`random_forest`)

| feature | importance |
|---|---|
| phase_darkness_separation | 0.272 |
| junction_score_at_ensemble | 0.180 |
| pnp_rms_px | 0.151 |
| ensemble_shift_px | 0.150 |
| fit_residual_rms_px | 0.137 |
| hexagon_centroid_vs_bezel_vertex_offset_px | 0.110 |


## Cross-validation methodology

- **Leave-one-case-out** (group-based on `case_key`): both runs of a
  case_key go into the same held-out fold. Both runs share the same
  input image — leaving only one run out would leak the underlying
  difficulty of that case.
- 58 folds total. Per fold: fit on the remaining
  116 − 2 ≈ 114 rows, predict on the 2 held-out.

## What the result means

Even the best learned model (`random_forest` at out-of-fold)
doesn't clear the Phase 2 bar on this 58-case corpus. But the gap is
real and shrinking:

- Phase 2A (`phase_sep` alone, calibrated): 46% recall / 9% FPR
- Phase 2B hand-tuned ceiling: 80% recall / 31% FPR (or 50% recall / 10% FPR)
- Phase 4 v1 best learned: see table above

The remaining gap is consistent with two simultaneous limits:
1. **Sample size.** 20 catastrophic samples means leave-one-case-out
   gives noisy folds. The Phase 4 corpus expansion (in-flight
   worktree `claude/phase4-corpus-expansion`, blocked on the labeling
   gallery viewport bug) adds 6 sets ≈ 4-8 more catastrophic samples,
   shrinking CV variance.
2. **Feature set.** 6 features per row from the global model fit.
   The proposed two-view orientation consistency signal (Codex's
   pickup, per `COORDINATION.md`) would add a 7th feature derived
   from comparing A and B view orientations — a fundamentally
   different geometric signal than the others.

Either lever (data, features) could close the gap. Both together
should.

## Files

- Tool: `tools/phase4_trust_ranker.py`
- Trained snapshot: `tests/fixtures/phase4_trust_ranker_v1.json`
  (per-model full-data fit + out-of-fold predictions + threshold sweep
  + named operating points)
- Source data: `tests/fixtures/phase2b_recomputed_signals.json`

## See also

- [`PHASE_2B_TRUST_SIGNAL_MATRIX_RECOMPUTED.md`](PHASE_2B_TRUST_SIGNAL_MATRIX_RECOMPUTED.md) —
  hand-tuned ceiling this learned model is graded against.
- [`STATE_OF_THE_WORLD.md`](STATE_OF_THE_WORLD.md) — Phase 4
  positioning in the phased roadmap.
