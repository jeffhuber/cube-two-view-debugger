# Phase 4 trust ranker v1

**Status: diagnostics-only.** First learned-classifier pass at the
Phase 2 bar after Phase 2B's hand-tuning hit its ceiling (see
`PHASE_2B_TRUST_SIGNAL_MATRIX_RECOMPUTED.md`). 4 model classes
evaluated on the 70-case eval via leave-one-case-out CV.
No production behavior change.

> **Legacy-data caution (2026-05-23):** this ranker was trained/evaluated
> on outcomes derived from legacy `near_*` axis labels. Treat it as a
> historical experiment until the labels and baselines are regenerated from
> `Va/Vb + 0..5` full-corner truth.

## Headline

**No learned model clears the Phase 2 bar on out-of-fold CV (≥80% catastrophic recall AND ≤10% GOOD FPR).** Best-in-class is `mlp_16_8` at 82% recall / 27% FPR — meaningfully better than Phase 2B's hand-tuned ceiling of 80% / 31% at the same recall, but still above the 10% FPR target. The learned model captures multi-feature structure (non-axis-aligned cuts) that hand-tuning missed; what's still missing is data and/or a stronger feature.

## Models evaluated

| model | clears bar? | out-of-fold FPR at 80% recall (vs Phase 2B ceiling: 31%) | out-of-fold recall at 10% FPR (vs Phase 2B: ~50%) | in-sample FPR at 80% recall |
|---|---|---|---|---|
| `logistic_regression` | ❌ | 40% (82% recall) | 36% (10% FPR) | 35% |
| `gradient_boosting` | ❌ | 34% (82% recall) | 52% (10% FPR) | 0% |
| `random_forest` | ❌ | 36% (82% recall) | 27% (10% FPR) | 0% |
| `mlp_16_8` | ❌ | 27% (82% recall) | 30% (10% FPR) | 0% |

**Reading the table:**
- "Clears bar" = some threshold achieves both ≥80% catastrophic recall AND ≤10% GOOD FPR on out-of-fold predictions.
- Lower FPR is better at fixed recall.
- Higher recall is better at fixed FPR.
- The in-sample column is what you'd get if you trained AND evaluated on the same data (no held-out). Compare to out-of-fold to see the generalization gap.

## Dataset

- 140 per-case-per-run rows across 70 cases
- Outcome breakdown:

| category | n |
|---|---|
| CHIRALITY_FALSE_FLIP | 12 |
| CHIRALITY_MISS | 20 |
| GOOD | 91 |
| MARGINAL | 16 |
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

### Headline model feature importance (`mlp_16_8`)

(Not extracted for this model class — MLP coefficients aren't trivially interpretable.)


## Cross-validation methodology

- **Leave-one-case-out** (group-based on `case_key`): both runs of a
  case_key go into the same held-out fold. Both runs share the same
  input image — leaving only one run out would leak the underlying
  difficulty of that case.
- 70 folds total. Per fold: fit on the remaining
  140 − 2 = 138 rows, predict on the 2 held-out.

## What the result means

Even the best learned model (`mlp_16_8` at out-of-fold)
doesn't clear the Phase 2 bar on this 70-case corpus.
But the gap is real and shrinking:

- Phase 2A (`phase_sep` alone, calibrated): 46% recall / 9% FPR
- Phase 2B hand-tuned ceiling: 80% recall / 31% FPR (or 50% recall / 10% FPR)
- Phase 4 best learned: see table above

The remaining gap is consistent with two simultaneous limits:
1. **Sample size.** 33 catastrophic samples means
   leave-one-case-out gives noisy folds. Phase 4 v1.1 expanded the
   corpus from 58 to 70 cases (adding ~13 catastrophic samples) but
   the bar still wasn't cleared — strong evidence the data lever
   alone is insufficient.
2. **Feature set.** 6 features per row from the global model fit.
   The two-view orientation consistency signal (shipped in
   `tools/two_view_consistency.py`, PR #243) would add a 7th feature
   derived from comparing A and B view orientations — a fundamentally
   different geometric signal than the others. See
   `tools/TWO_VIEW_CONSISTENCY.md` for the integration plan.

Phase 4 v1.1 demonstrated the data lever alone doesn't suffice. Phase
4 v2 (feature integration) is the highest-leverage remaining lever.

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
