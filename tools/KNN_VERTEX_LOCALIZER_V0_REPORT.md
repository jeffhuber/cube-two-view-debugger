# KNN Vertex Localizer V0

Diagnostics-only artifact. This does not alter recognition behavior.

This probe evaluates a leave-one-row-out k-nearest positive/negative prototype scorer over the same local candidate grid and features as the linear learned V0 localizer.

## Summary

- Rows: 28
- Evaluated rows: 28
- Axis-good rows: 17
- Axis-blocked rows: 11
- Baseline strict/plausible: 6 / 9
- Candidate-grid oracle strict/plausible: 28 / 28
- KNN top-1 strict/plausible: 11 / 23
- KNN gated strict/plausible: 10 / 14
- Axis-good strict baseline/KNN-gated: 5 / 6
- KNN gated accepted rows: 13
- KNN top-1 improved/worsened rows by >5px: 20 / 5
- KNN gated improved/worsened rows by >5px: 11 / 1
- Mean vertex error baseline/oracle/top-1/gated: 67.0 px / 5.6 px / 39.7 px / 52.5 px
- Median vertex error baseline/oracle/top-1/gated: 70.7 px / 6.2 px / 32.7 px / 46.6 px
- Best non-empty low-worsen KNN gate: accepted 2, improved 2, worsened 0, strict/plausible 8 / 10, mean 64.7 px (gain>=2.0, score>=-6.0)
- Best non-empty zero-worsen KNN gate: accepted 2, improved 2, worsened 0, strict/plausible 8 / 10, mean 64.7 px (gain>=2.0, score>=-6.0)
- Best non-empty KNN gate with <=2 worsens: accepted 14, improved 12, worsened 2, strict/plausible 10 / 17, mean 49.3 px (gain>=0.75, score>=-1.0)

## Rows

| Row | Axis | Base | Oracle | KNN top-1 | Gated | Accepted | Delta | Score gain |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `12_B` | `axis_blocked` | 26.9 px | 7.5 px | 28.9 px | 26.9 px | no | 0.0 px | 0.585 |
| `14_B` | `axis_blocked` | 53.9 px | 9.4 px | 26.3 px | 26.3 px | yes | 27.6 px | 1.470 |
| `15_A` | `axis_good` | 63.5 px | 8.3 px | 14.7 px | 14.7 px | yes | 48.9 px | 1.054 |
| `15_B` | `axis_good` | 72.0 px | 6.3 px | 26.4 px | 72.0 px | no | 0.0 px | 0.831 |
| `17_B` | `axis_good` | 107.2 px | 1.8 px | 49.8 px | 107.2 px | no | 0.0 px | 0.870 |
| `21_A` | `axis_good` | 38.7 px | 5.4 px | 34.0 px | 38.7 px | no | 0.0 px | 0.201 |
| `21_B` | `axis_good` | 75.7 px | 2.4 px | 16.4 px | 75.7 px | no | 0.0 px | 0.722 |
| `24_A` | `axis_blocked` | 87.6 px | 7.2 px | 39.4 px | 87.6 px | no | 0.0 px | 0.474 |
| `26_A` | `axis_good` | 72.5 px | 4.6 px | 38.4 px | 38.4 px | yes | 34.1 px | 1.863 |
| `26_B` | `axis_blocked` | 80.9 px | 9.0 px | 22.6 px | 80.9 px | no | 0.0 px | 0.681 |
| `27_B` | `axis_good` | 7.0 px | 7.0 px | 55.3 px | 55.3 px | yes | -48.3 px | 1.678 |
| `28_A` | `axis_good` | 124.3 px | 1.9 px | 64.8 px | 64.8 px | yes | 59.5 px | 1.852 |
| `28_B` | `axis_good` | 102.6 px | 9.2 px | 42.4 px | 42.4 px | yes | 60.2 px | 1.278 |
| `29_A` | `axis_blocked` | 57.4 px | 6.3 px | 32.1 px | 57.4 px | no | 0.0 px | 0.425 |
| `29_B` | `axis_good` | 25.4 px | 9.0 px | 48.7 px | 25.4 px | no | 0.0 px | 0.703 |
| `30_A` | `axis_good` | 39.7 px | 1.9 px | 22.6 px | 22.6 px | yes | 17.1 px | 2.421 |
| `30_B` | `axis_good` | 10.0 px | 3.0 px | 33.0 px | 10.0 px | no | 0.0 px | 0.871 |
| `31_A` | `axis_good` | 47.2 px | 7.1 px | 30.4 px | 30.4 px | yes | 16.8 px | 1.086 |
| `31_B` | `axis_good` | 7.7 px | 2.9 px | 21.1 px | 7.7 px | no | 0.0 px | 0.762 |
| `32_A` | `axis_blocked` | 72.3 px | 6.0 px | 33.4 px | 72.3 px | no | 0.0 px | 0.843 |
| `32_B` | `axis_blocked` | 70.6 px | 3.5 px | 23.1 px | 23.1 px | yes | 47.5 px | 2.521 |
| `36_B` | `axis_good` | 103.0 px | 4.5 px | 40.9 px | 103.0 px | no | 0.0 px | 0.843 |
| `42_B` | `axis_blocked` | 92.0 px | 6.9 px | 70.9 px | 70.9 px | yes | 21.1 px | 1.170 |
| `44_A` | `axis_blocked` | 185.5 px | 2.3 px | 180.9 px | 180.9 px | yes | 4.6 px | 1.076 |
| `44_B` | `axis_blocked` | 117.6 px | 7.8 px | 50.9 px | 50.9 px | yes | 66.7 px | 1.678 |
| `57_A` | `axis_good` | 5.0 px | 5.0 px | 11.7 px | 5.0 px | no | 0.0 px | 0.203 |
| `61_A` | `axis_blocked` | 70.7 px | 4.7 px | 19.7 px | 19.7 px | yes | 51.0 px | 1.436 |
| `61_B` | `axis_good` | 60.0 px | 6.5 px | 32.4 px | 60.0 px | no | 0.0 px | 0.361 |

## Interpretation

- KNN V0 is the first learned ranker to materially beat baseline on strict/plausible counts.
- It is still not production-safe: the default confidence gate has one worsened accepted row and the broad top-1 policy has several.
- The zero-worsen gate is useful evidence that confidence is possible, but it accepts too few rows to be a recognizer policy.
- The next useful step is a real trainable localizer or more labeled rows, using this candidate-grid oracle and KNN gate as baselines.
