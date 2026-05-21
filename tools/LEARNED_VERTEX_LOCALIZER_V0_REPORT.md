# Learned Vertex Localizer V0

Diagnostics-only artifact. This does not alter recognition behavior.

This probe trains a lightweight ridge-regression scorer over local candidate features using leave-one-row-out validation on the human-labeled visible trihedral rows.

## Summary

- Rows: 28
- Evaluated rows: 28
- Axis-good rows: 17
- Axis-blocked rows: 11
- Baseline strict/plausible: 6 / 9
- Candidate-grid oracle strict/plausible: 28 / 28
- Learned top-1 strict/plausible: 4 / 13
- Learned gated strict/plausible: 4 / 12
- Axis-good strict baseline/learned-gated: 5 / 2
- Learned gated accepted rows: 16
- Learned top-1 improved/worsened rows by >5px: 16 / 5
- Learned gated improved/worsened rows by >5px: 11 / 4
- Mean vertex error baseline/oracle/top-1/gated: 67.0 px / 5.6 px / 57.2 px / 62.1 px
- Median vertex error baseline/oracle/top-1/gated: 70.7 px / 6.2 px / 50.5 px / 53.4 px
- Best non-empty low-worsen learned gate: accepted 3, improved 1, worsened 2, strict/plausible 5 / 8, mean 68.5 px (gain>=-0.05, score>=0.75)
- Best non-empty learned gate with <=2 worsens: accepted 3, improved 1, worsened 2, strict/plausible 5 / 8, mean 68.5 px (gain>=-0.05, score>=0.75)

## Rows

| Row | Axis | Base | Oracle | Learned top-1 | Gated | Accepted | Delta | Score gain |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `12_B` | `axis_blocked` | 26.9 px | 7.5 px | 21.8 px | 26.9 px | no | 0.0 px | 0.035 |
| `14_B` | `axis_blocked` | 53.9 px | 9.4 px | 27.0 px | 27.0 px | yes | 26.9 px | 0.132 |
| `15_A` | `axis_good` | 63.5 px | 8.3 px | 41.5 px | 41.5 px | yes | 22.0 px | 0.110 |
| `15_B` | `axis_good` | 72.0 px | 6.3 px | 54.1 px | 54.1 px | yes | 17.8 px | 0.058 |
| `17_B` | `axis_good` | 107.2 px | 1.8 px | 95.6 px | 95.6 px | yes | 11.6 px | 0.255 |
| `21_A` | `axis_good` | 38.7 px | 5.4 px | 50.8 px | 38.7 px | no | 0.0 px | 0.024 |
| `21_B` | `axis_good` | 75.7 px | 2.4 px | 37.9 px | 75.7 px | no | 0.0 px | 0.022 |
| `24_A` | `axis_blocked` | 87.6 px | 7.2 px | 60.4 px | 87.6 px | no | 0.0 px | 0.013 |
| `26_A` | `axis_good` | 72.5 px | 4.6 px | 36.0 px | 36.0 px | yes | 36.6 px | 0.070 |
| `26_B` | `axis_blocked` | 80.9 px | 9.0 px | 80.9 px | 80.9 px | no | 0.0 px | 0.000 |
| `27_B` | `axis_good` | 7.0 px | 7.0 px | 71.0 px | 71.0 px | yes | -64.0 px | 0.200 |
| `28_A` | `axis_good` | 124.3 px | 1.9 px | 105.5 px | 105.5 px | yes | 18.8 px | 0.117 |
| `28_B` | `axis_good` | 102.6 px | 9.2 px | 58.2 px | 58.2 px | yes | 44.4 px | 0.213 |
| `29_A` | `axis_blocked` | 57.4 px | 6.3 px | 53.2 px | 53.2 px | yes | 4.2 px | 0.066 |
| `29_B` | `axis_good` | 25.4 px | 9.0 px | 24.1 px | 25.4 px | no | 0.0 px | 0.028 |
| `30_A` | `axis_good` | 39.7 px | 1.9 px | 49.4 px | 49.4 px | yes | -9.7 px | 0.245 |
| `30_B` | `axis_good` | 10.0 px | 3.0 px | 45.0 px | 45.0 px | yes | -35.0 px | 0.275 |
| `31_A` | `axis_good` | 47.2 px | 7.1 px | 30.4 px | 30.4 px | yes | 16.8 px | 0.169 |
| `31_B` | `axis_good` | 7.7 px | 2.9 px | 50.3 px | 50.3 px | yes | -42.6 px | 0.296 |
| `32_A` | `axis_blocked` | 72.3 px | 6.0 px | 44.7 px | 44.7 px | yes | 27.7 px | 0.056 |
| `32_B` | `axis_blocked` | 70.6 px | 3.5 px | 53.6 px | 53.6 px | yes | 17.0 px | 0.167 |
| `36_B` | `axis_good` | 103.0 px | 4.5 px | 32.2 px | 103.0 px | no | 0.0 px | 0.002 |
| `42_B` | `axis_blocked` | 92.0 px | 6.9 px | 47.2 px | 47.2 px | yes | 44.8 px | 0.094 |
| `44_A` | `axis_blocked` | 185.5 px | 2.3 px | 185.5 px | 185.5 px | no | 0.0 px | 0.000 |
| `44_B` | `axis_blocked` | 117.6 px | 7.8 px | 108.7 px | 117.6 px | no | 0.0 px | 0.062 |
| `57_A` | `axis_good` | 5.0 px | 5.0 px | 5.0 px | 5.0 px | no | 0.0 px | 0.000 |
| `61_A` | `axis_blocked` | 70.7 px | 4.7 px | 70.7 px | 70.7 px | no | 0.0 px | 0.000 |
| `61_B` | `axis_good` | 60.0 px | 6.5 px | 60.0 px | 60.0 px | no | 0.0 px | 0.000 |

## Interpretation

- This is the first supervised vertex-localization diagnostic, not production wiring.
- The candidate-grid oracle tells us whether a local model-axis search could reach the human vertex if ranking were solved.
- The leave-one-row-out scorer is deliberately tiny and dependency-free; success here would justify a richer learned localizer.
- The V0 result is not production-safe: it improves mean/plausible error, but still worsens multiple already-good vertices and the best low-worsen gate underperforms baseline.
- The strongest conclusion is that local candidate generation is no longer the blocker on these rows; learned ranking/confidence is.
- Next work should train a richer vertex-localization model or collect more labels, rather than adding another hand-tuned geometric score.
