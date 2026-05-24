# Canonical full-corner geometry baseline

Status: canonical full-corner evidence. This report uses
`tests/fixtures/full_corner_ground_truth.json` and the `Va/Vb + 0..5`
human convention from `tools/FULL_CORNER_LABELING.md`. It does not use
legacy `near_*` semantics.

## Summary

- Rows scored: 12 / 12
- Categories: `GOOD: 3, MARGINAL: 1, PHASE_SWAPPED: 8`
- Median vertex error: `153.1` px
- Median one-edge angle error: `58.76` deg
- Median far angle error: `58.66` deg
- Median swapped-phase angle error: `7.42` deg

## Row Details

| row | category | vertex px | one-edge deg | far deg | swapped deg | phase check | sep |
|---|---:|---:|---:|---:|---:|---|---:|
| `20_A` | `PHASE_SWAPPED` | 104.1 | 58.93 | 58.85 | 2.68 | `correct` | -24.0 |
| `20_B` | `PHASE_SWAPPED` | 155.6 | 59.10 | 59.71 | 4.12 | `corrected_60deg_flip` | 67.7 |
| `38_A` | `PHASE_SWAPPED` | 200.2 | 58.89 | 58.80 | 6.98 | `ambiguous_no_correction` | -2.3 |
| `38_B` | `GOOD` | 209.7 | 2.99 | 6.99 | 58.22 | `corrected_60deg_flip` | 24.2 |
| `40_A` | `GOOD` | 136.3 | 4.20 | 11.61 | 58.89 | `corrected_60deg_flip` | 10.4 |
| `40_B` | `PHASE_SWAPPED` | 464.7 | 59.19 | 58.70 | 10.36 | `correct` | -12.6 |
| `41_A` | `GOOD` | 111.2 | 3.84 | 6.70 | 58.42 | `correct` | -11.3 |
| `41_B` | `PHASE_SWAPPED` | 54.7 | 59.25 | 58.29 | 7.85 | `corrected_60deg_flip` | 18.2 |
| `43_A` | `PHASE_SWAPPED` | 115.1 | 58.63 | 59.11 | 4.96 | `corrected_60deg_flip` | 53.6 |
| `43_B` | `PHASE_SWAPPED` | 150.6 | 58.51 | 58.61 | 6.15 | `corrected_60deg_flip` | 21.0 |
| `45_A` | `PHASE_SWAPPED` | 168.4 | 59.68 | 59.49 | 4.67 | `corrected_60deg_flip` | 11.9 |
| `45_B` | `MARGINAL` | 290.7 | 7.25 | 18.17 | 59.03 | `corrected_60deg_flip` | 37.4 |

## Interpretation

This is a 12-row seed baseline, not the final 58-case migration.
The strong signal is phase parity: most failures are near-exact
one-edge/far swaps under the human convention, not arbitrary geometry
drift. That keeps the phase/chirality problem real, but the old
`near_*`-derived category names should remain historical until this
canonical path is expanded.

- `PHASE_SWAPPED` rows by current `phase_check`: `ambiguous_no_correction: 1, correct: 2, corrected_60deg_flip: 5`

## Source

- Model: `tools.global_cube_model.fit_global_cube_model`
- Truth: `tests/fixtures/full_corner_ground_truth.json`
- Max processing image dimension: `1600` px
- Run selection: `min(aligned_one_edge_far_mean_deg, swapped_phase_mean_deg)`
- Image root: resolved from corpus manifests (local corpus path not recorded)

Rows marked `PHASE_SWAPPED` mean the model's one-edge triplet matches
the human far/double-axis triplet, and vice versa. This is the canonical
full-corner version of the old near/far phase ambiguity.
