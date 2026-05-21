# Vertex Candidate Ranker V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This report tests whether simple label-driven ranking policies can move the human-visible trihedral vertex from the expanded #189 source pool into top-1/top-3/top-5.

## Summary

- Rows: 16
- Rows with notes/errors: 0
- Thresholds: 10px, 20px

## Policy Metrics

| Policy | Rows | Mean pool | Top-1 @10 | Top-3 @10 | Top-5 @10 | Oracle @10 | Top-3 @20 | Top-5 @20 | Oracle @20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_model_ranked` | 16 | 121.8 | 1 | 2 | 4 | 4 | 4 | 7 | 7 |
| `source_heuristic_v0` | 16 | 121.8 | 3 | 3 | 4 | 11 | 6 | 7 | 14 |
| `leave_one_out_feature_prior_v0` | 16 | 121.8 | 1 | 3 | 3 | 11 | 5 | 6 | 14 |
| `combined_oracle` | 16 | 121.8 | 11 | 11 | 11 | 11 | 14 | 14 | 14 |

## Per-Row Readout

| Set | Side | Pool | Best oracle | Baseline top3 | Heuristic top3 | Feature-prior top3 | Notes |
|---:|---|---:|---:|---:|---:|---:|---|
| 15 | A | 122 | 1.74 | 30.69 | 30.69 | 30.69 |  |
| 15 | B | 122 | 4.44 | 15.65 | 17.67 | 15.65 |  |
| 23 | A | 122 | 7.55 | 20.7 | 20.7 | 20.7 |  |
| 23 | B | 122 | 1.16 | 22.78 | 2.47 | 2.47 |  |
| 26 | A | 122 | 13.64 | 48.38 | 157.67 | 48.38 |  |
| 26 | B | 122 | 6.58 | 21.7 | 21.7 | 21.7 |  |
| 29 | A | 122 | 6.05 | 6.05 | 11.35 | 37.13 |  |
| 29 | B | 122 | 9.23 | 10.01 | 10.01 | 10.01 |  |
| 32 | A | 122 | 6.88 | 22.23 | 10.0 | 10.0 |  |
| 32 | B | 122 | 8.02 | 29.51 | 25.0 | 29.51 |  |
| 36 | A | 119 | 16.41 | 63.68 | 63.68 | 63.68 |  |
| 36 | B | 122 | 7.43 | 49.94 | 49.94 | 49.24 |  |
| 37 | A | 122 | 10.86 | 58.0 | 58.0 | 80.44 |  |
| 37 | B | 122 | 27.25 | 93.46 | 93.46 | 103.57 |  |
| 42 | A | 122 | 24.76 | 47.13 | 86.75 | 86.75 |  |
| 42 | B | 122 | 6.84 | 6.84 | 6.84 | 6.84 |  |

## Interpretation

- `baseline_model_ranked` is the #188/#189 starting point.
- `source_heuristic_v0` is the best current deployable-shaped static score, but it remains far below the oracle ceiling.
- `leave_one_out_feature_prior_v0` uses labels from other rows only; it is a sanity check against tiny-sample overfitting, not a production model.
- The gap between ranker top-3 and `combined_oracle` means the next useful work is richer ranking signal, not production wiring.
