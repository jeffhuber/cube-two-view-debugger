# Vertex Candidate Source Probe

Diagnostics/data-only artifact. This does not alter recognition behavior.

This report evaluates whether new candidate-source families contain the human-visible trihedral vertex. The current model-ranked candidates remain as the baseline.

## Summary

- Rows: 16
- Labeled rows: 16
- Error rows: 0
- Thresholds: 10px, 20px

## Source Recall

| Source | Rows with candidates | Mean candidates | Top-1 @10 | Top-3 @10 | Top-5 @10 | Oracle @10 | Top-3 @20 | Top-5 @20 | Oracle @20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `model_ranked` | 16 | 5.0 | 1 | 2 | 4 | 4 | 4 | 7 | 7 |
| `bezel_line_intersection` | 16 | 1.0 | 0 | 0 | 0 | 0 | 2 | 2 | 2 |
| `bezel_axis_ray` | 16 | 35.8 | 3 | 3 | 3 | 4 | 6 | 6 | 11 |
| `model_local_grid` | 16 | 40.0 | 1 | 2 | 3 | 10 | 5 | 8 | 13 |
| `dark_junction_grid` | 16 | 40.0 | 0 | 0 | 0 | 1 | 0 | 0 | 4 |

## Combined Oracle

- Best source within 10px: 11 / 16
- Best source within 20px: 14 / 16

## Per-Row Best Source

| Set | Side | Status | Best source | Best dist | Best rank | Top source @10 | Source notes |
|---:|---|---|---|---:|---:|---|---|
| 15 | A | `labeled` | `model_local_grid` | 1.74 | 11 | `model_local_grid` |  |
| 15 | B | `labeled` | `model_local_grid` | 4.44 | 11 | `model_local_grid` |  |
| 23 | A | `labeled` | `bezel_axis_ray` | 7.55 | 1 | `bezel_axis_ray`, `model_local_grid` |  |
| 23 | B | `labeled` | `model_local_grid` | 1.16 | 23 | `bezel_axis_ray`, `model_local_grid` |  |
| 26 | A | `labeled` | `model_local_grid` | 13.64 | 20 |  |  |
| 26 | B | `labeled` | `model_local_grid` | 6.58 | 5 | `model_local_grid` |  |
| 29 | A | `labeled` | `model_ranked` | 6.05 | 3 | `model_ranked`, `model_local_grid` |  |
| 29 | B | `labeled` | `model_local_grid` | 9.23 | 6 | `model_local_grid` |  |
| 32 | A | `labeled` | `model_ranked` | 6.88 | 5 | `model_ranked`, `bezel_axis_ray`, `model_local_grid` |  |
| 32 | B | `labeled` | `model_local_grid` | 8.02 | 12 | `model_ranked`, `model_local_grid` |  |
| 36 | A | `labeled` | `model_local_grid` | 16.41 | 30 |  |  |
| 36 | B | `labeled` | `dark_junction_grid` | 7.43 | 34 | `bezel_axis_ray`, `dark_junction_grid` |  |
| 37 | A | `labeled` | `bezel_axis_ray` | 10.86 | 6 |  |  |
| 37 | B | `labeled` | `dark_junction_grid` | 27.25 | 16 |  |  |
| 42 | A | `labeled` | `model_local_grid` | 24.76 | 30 |  |  |
| 42 | B | `labeled` | `model_ranked` | 6.84 | 1 | `model_ranked`, `model_local_grid` |  |

## Interpretation

- `Top-N` columns use each source's own deterministic ranking; low top-N with high oracle means the source sees the neighborhood but still needs a better ranking policy.
- `Oracle` columns ignore ranking and ask whether the source contains any candidate within the threshold.
- The next geometry step should only use a source for model fitting once source top-3 recall is strong on easy rows, or once a downstream fitter can safely consume larger candidate sets.
