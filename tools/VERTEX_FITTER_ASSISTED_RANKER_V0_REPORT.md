# Vertex Fitter-Assisted Ranker V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This report tests whether coherent projected-cube fit quality can select the human-visible trihedral vertex from the expanded #189 candidate pool.

## Headline

At 10px, `fitter_model_score_v0` top-3 recall is 2 / 16 and `fitter_assisted_v0` top-3 recall is 1 / 16, versus 2 / 16 for the baseline, 3 / 16 for the simple source heuristic, and 11 / 16 source-pool oracle. The current fit objective therefore does not reliably select the true vertex even when the pool contains it.

## Summary

- Rows: 16
- Labeled rows: 16
- Error rows: 0
- Mean candidate pool: 121.8
- Mean fitted candidates: 121.8
- Thresholds: 10px, 20px
- Edge steps per candidate: 10
- Scoring max dimension: 360px

## Policy Metrics

| Policy | Rows | Mean pool | Top-1 @10 | Top-3 @10 | Top-5 @10 | Oracle @10 | Top-3 @20 | Top-5 @20 | Oracle @20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_model_ranked` | 16 | 121.8 | 1 | 2 | 4 | 4 | 4 | 7 | 7 |
| `source_heuristic_v0` | 16 | 121.8 | 3 | 3 | 4 | 11 | 6 | 7 | 14 |
| `fitter_model_score_v0` | 16 | 121.8 | 1 | 2 | 4 | 11 | 5 | 5 | 14 |
| `fitter_assisted_v0` | 16 | 121.8 | 0 | 1 | 1 | 11 | 4 | 4 | 14 |
| `combined_oracle` | 16 | 121.8 | 11 | 11 | 11 | 11 | 14 | 14 | 14 |

## Per-Row Readout

| Set | Side | Pool | Fitted | Best oracle | Baseline top3 | Fitter top3 | Assisted top3 | Assisted top source | Overlay | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 15 | A | 122 | 122 | 1.74 | 30.69 | 34.09 | 30.69 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_15_A_fitter_assisted_vertex_v0.png` |  |
| 15 | B | 122 | 122 | 4.44 | 15.65 | 17.67 | 17.67 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_15_B_fitter_assisted_vertex_v0.png` |  |
| 23 | A | 122 | 122 | 7.55 | 20.7 | 10.27 | 20.7 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_23_A_fitter_assisted_vertex_v0.png` |  |
| 23 | B | 122 | 122 | 1.16 | 22.78 | 28.45 | 31.06 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_23_B_fitter_assisted_vertex_v0.png` |  |
| 26 | A | 122 | 122 | 13.64 | 48.38 | 48.34 | 45.64 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_26_A_fitter_assisted_vertex_v0.png` |  |
| 26 | B | 122 | 122 | 6.58 | 21.7 | 36.22 | 20.53 | `dark_junction_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_26_B_fitter_assisted_vertex_v0.png` |  |
| 29 | A | 122 | 122 | 6.05 | 6.05 | 37.13 | 21.65 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_29_A_fitter_assisted_vertex_v0.png` |  |
| 29 | B | 122 | 122 | 9.23 | 10.01 | 10.01 | 10.01 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_29_B_fitter_assisted_vertex_v0.png` |  |
| 32 | A | 122 | 122 | 6.88 | 22.23 | 7.48 | 10.0 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_32_A_fitter_assisted_vertex_v0.png` |  |
| 32 | B | 122 | 122 | 8.02 | 29.51 | 26.45 | 29.51 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_32_B_fitter_assisted_vertex_v0.png` |  |
| 36 | A | 119 | 119 | 16.41 | 63.68 | 47.77 | 47.77 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_36_A_fitter_assisted_vertex_v0.png` |  |
| 36 | B | 122 | 122 | 7.43 | 49.94 | 34.79 | 49.94 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_36_B_fitter_assisted_vertex_v0.png` |  |
| 37 | A | 122 | 122 | 10.86 | 58.0 | 71.12 | 109.0 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_37_A_fitter_assisted_vertex_v0.png` |  |
| 37 | B | 122 | 122 | 27.25 | 93.46 | 141.92 | 114.58 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_37_B_fitter_assisted_vertex_v0.png` |  |
| 42 | A | 122 | 122 | 24.76 | 47.13 | 24.95 | 24.95 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_42_A_fitter_assisted_vertex_v0.png` |  |
| 42 | B | 122 | 122 | 6.84 | 6.84 | 6.84 | 11.6 | `model_local_grid` | `/tmp/vertex_fitter_assisted_ranker_v0_overlays/set_42_B_fitter_assisted_vertex_v0.png` |  |

## Interpretation

- `baseline_model_ranked` preserves the original #188/#189 ranking.
- `source_heuristic_v0` preserves the best simple #190 static ranker.
- `fitter_model_score_v0` ranks each expanded-pool candidate by its best coherent projected cube model.
- `fitter_assisted_v0` adds weak anchor-mesh and source-prior terms to the model fit score.
- `combined_oracle` is the source-pool ceiling and uses the human label only for evaluation.
- If fitter policies stay far below oracle, the model objective is still missing the signal needed to identify the visible trihedral vertex.
