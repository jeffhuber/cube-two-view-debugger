# Vertex Hypothesis Ensemble V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This report drives the current three-step path to a conclusion: canonicalize vertex/axis feedback, expand the hypothesis pool, then test whether agreement can become a safe confidence signal.

## Summary

### `gcm_fullres`

- Rows: 23
- Strict threshold: 30 px
- Plausible threshold: 50 px
- Candidate count: mean 5.0, max 5
- Oracle-best candidate: strict 6 / 23, plausible 12 / 23

| Policy | Selected | Abstained | Strict-ready | Plausible | False-confident | Mean selected error |
|---|---:|---:|---:|---:|---:|---:|
| `source_priority_top1_v0` | 23 | 0 | 6 | 9 | 14 | 72.2 px |
| `agreement_cluster_v0` | 23 | 0 | 4 | 8 | 15 | 69.7 px |
| `strict_agreement_cluster_v0` | 23 | 0 | 4 | 7 | 16 | 70.5 px |
| `oracle_best_candidate` | 23 | 0 | 6 | 12 | 11 | 58.2 px |

### `easy_processing`

- Rows: 16
- Strict threshold: 10 px
- Plausible threshold: 20 px
- Candidate count: mean 21.0, max 21
- Oracle-best candidate: strict 7 / 16, plausible 10 / 16

| Policy | Selected | Abstained | Strict-ready | Plausible | False-confident | Mean selected error |
|---|---:|---:|---:|---:|---:|---:|
| `source_priority_top1_v0` | 16 | 0 | 1 | 3 | 13 | 42.5 px |
| `agreement_cluster_v0` | 9 | 7 | 0 | 4 | 5 | 35.1 px |
| `strict_agreement_cluster_v0` | 3 | 13 | 0 | 1 | 2 | 25.7 px |
| `oracle_best_candidate` | 16 | 0 | 7 | 10 | 6 | 22.4 px |

## Agreement Policy Readout

| Lane | Row | Agreement | Strict agreement | Oracle-best |
|---|---|---|---|---|
| `gcm_fullres` | `12_B` | `agreement_cluster` / 23.83 px / `strict_ready` | `agreement_cluster` / 22.89 px / `strict_ready` | `inverse_residual_weighted` / 22.98 px / `strict_ready` |
| `gcm_fullres` | `14_B` | `agreement_cluster` / 19.85 px / `strict_ready` | `agreement_cluster` / 19.85 px / `strict_ready` | `sam3` / 16.19 px / `strict_ready` |
| `gcm_fullres` | `15_A` | `agreement_cluster` / 53.76 px / `false_confident` | `agreement_cluster` / 53.76 px / `false_confident` | `sam3` / 47.29 px / `plausible` |
| `gcm_fullres` | `15_B` | `agreement_cluster` / 46.81 px / `plausible` | `agreement_cluster` / 46.63 px / `plausible` | `inverse_residual_weighted` / 46.07 px / `plausible` |
| `gcm_fullres` | `17_B` | `agreement_cluster` / 122.49 px / `false_confident` | `agreement_cluster` / 122.49 px / `false_confident` | `rembg` / 109.96 px / `false_confident` |
| `gcm_fullres` | `21_A` | `agreement_cluster` / 44.08 px / `plausible` | `agreement_cluster` / 46.79 px / `plausible` | `rembg` / 38.72 px / `plausible` |
| `gcm_fullres` | `21_B` | `agreement_cluster` / 94.27 px / `false_confident` | `agreement_cluster` / 87.81 px / `false_confident` | `rembg` / 75.69 px / `false_confident` |
| `gcm_fullres` | `24_A` | `agreement_cluster` / 86.67 px / `false_confident` | `agreement_cluster` / 86.67 px / `false_confident` | `sam3` / 86.37 px / `false_confident` |
| `gcm_fullres` | `26_A` | `agreement_cluster` / 86.21 px / `false_confident` | `agreement_cluster` / 85.01 px / `false_confident` | `rembg` / 72.54 px / `false_confident` |
| `gcm_fullres` | `26_B` | `agreement_cluster` / 19.41 px / `strict_ready` | `agreement_cluster` / 29.92 px / `strict_ready` | `sam3` / 24.1 px / `strict_ready` |
| `gcm_fullres` | `28_A` | `agreement_cluster` / 95.44 px / `false_confident` | `agreement_cluster` / 95.01 px / `false_confident` | `sam3` / 78.49 px / `false_confident` |
| `gcm_fullres` | `28_B` | `agreement_cluster` / 104.08 px / `false_confident` | `agreement_cluster` / 104.08 px / `false_confident` | `rembg` / 102.57 px / `false_confident` |
| `gcm_fullres` | `29_A` | `agreement_cluster` / 50.22 px / `false_confident` | `agreement_cluster` / 50.22 px / `false_confident` | `sam3` / 45.3 px / `plausible` |
| `gcm_fullres` | `30_A` | `agreement_cluster` / 27.65 px / `strict_ready` | `agreement_cluster` / 27.65 px / `strict_ready` | `sam3` / 20.03 px / `strict_ready` |
| `gcm_fullres` | `31_A` | `agreement_cluster` / 66.98 px / `false_confident` | `agreement_cluster` / 66.57 px / `false_confident` | `rembg` / 47.18 px / `plausible` |
| `gcm_fullres` | `32_A` | `agreement_cluster` / 64.25 px / `false_confident` | `agreement_cluster` / 64.25 px / `false_confident` | `sam3` / 56.51 px / `false_confident` |
| `gcm_fullres` | `32_B` | `agreement_cluster` / 55.19 px / `false_confident` | `agreement_cluster` / 55.22 px / `false_confident` | `sam3` / 47.65 px / `plausible` |
| `gcm_fullres` | `36_B` | `agreement_cluster` / 85.29 px / `false_confident` | `agreement_cluster` / 85.29 px / `false_confident` | `sam3` / 69.62 px / `false_confident` |
| `gcm_fullres` | `42_B` | `agreement_cluster` / 136.14 px / `false_confident` | `agreement_cluster` / 136.14 px / `false_confident` | `rembg` / 89.25 px / `false_confident` |
| `gcm_fullres` | `44_A` | `agreement_cluster` / 152.83 px / `false_confident` | `agreement_cluster` / 156.4 px / `false_confident` | `sam3` / 148.34 px / `false_confident` |
| `gcm_fullres` | `44_B` | `agreement_cluster` / 48.04 px / `plausible` | `agreement_cluster` / 59.87 px / `false_confident` | `sam3` / 14.96 px / `strict_ready` |
| `gcm_fullres` | `61_A` | `agreement_cluster` / 44.28 px / `plausible` | `agreement_cluster` / 44.12 px / `plausible` | `sam3` / 18.88 px / `strict_ready` |
| `gcm_fullres` | `61_B` | `agreement_cluster` / 74.24 px / `false_confident` | `agreement_cluster` / 74.24 px / `false_confident` | `rembg` / 60.04 px / `false_confident` |
| `easy_processing` | `15_A` | `abstain` | `abstain` | `model_local_grid` / 14.71 px / `plausible` |
| `easy_processing` | `15_B` | `agreement_cluster` / 25.5 px / `false_confident` | `abstain` | `model_local_grid` / 12.35 px / `plausible` |
| `easy_processing` | `23_A` | `agreement_cluster` / 15.34 px / `plausible` | `abstain` | `bezel_axis_ray` / 7.55 px / `strict_ready` |
| `easy_processing` | `23_B` | `agreement_cluster` / 45.78 px / `false_confident` | `abstain` | `bezel_axis_ray` / 2.47 px / `strict_ready` |
| `easy_processing` | `26_A` | `abstain` | `abstain` | `model_local_grid` / 33.71 px / `false_confident` |
| `easy_processing` | `26_B` | `abstain` | `abstain` | `model_local_grid` / 6.58 px / `strict_ready` |
| `easy_processing` | `29_A` | `agreement_cluster` / 15.11 px / `plausible` | `agreement_cluster` / 25.49 px / `false_confident` | `model_ranked` / 6.05 px / `strict_ready` |
| `easy_processing` | `29_B` | `agreement_cluster` / 18.09 px / `plausible` | `agreement_cluster` / 18.28 px / `plausible` | `model_ranked` / 10.01 px / `plausible` |
| `easy_processing` | `32_A` | `agreement_cluster` / 14.81 px / `plausible` | `abstain` | `model_ranked` / 6.88 px / `strict_ready` |
| `easy_processing` | `32_B` | `agreement_cluster` / 24.32 px / `false_confident` | `agreement_cluster` / 33.38 px / `false_confident` | `model_ranked` / 8.59 px / `strict_ready` |
| `easy_processing` | `36_A` | `abstain` | `abstain` | `model_local_grid` / 47.77 px / `false_confident` |
| `easy_processing` | `36_B` | `agreement_cluster` / 50.64 px / `false_confident` | `abstain` | `model_ranked` / 27.59 px / `false_confident` |
| `easy_processing` | `37_A` | `abstain` | `abstain` | `bezel_axis_ray` / 26.07 px / `false_confident` |
| `easy_processing` | `37_B` | `agreement_cluster` / 106.65 px / `false_confident` | `abstain` | `model_ranked` / 93.46 px / `false_confident` |
| `easy_processing` | `42_A` | `abstain` | `abstain` | `model_ranked` / 47.13 px / `false_confident` |
| `easy_processing` | `42_B` | `abstain` | `abstain` | `model_ranked` / 6.84 px / `strict_ready` |

## Conclusion

- Production wiring recommendation: `wait`.
- Reason: Agreement-based deployable policies still make false-confident vertex selections: gcm_fullres/agreement_cluster_v0, gcm_fullres/strict_agreement_cluster_v0, easy_processing/agreement_cluster_v0, easy_processing/strict_agreement_cluster_v0.
- The canonical fixture is now reusable for later probes, but the current deployable agreement policies still emit false-confident selections.
- The expanded hypothesis pool contains more signal than the current rankers can safely select, especially in the easy-corpus lane. That makes ranking/confidence the blocker, not face splitting.
- The next useful input is richer labels/features around the visible vertex and axes, or a model objective that scores face-boundary consistency directly. More single-score source selection is unlikely to close the gap by itself.
