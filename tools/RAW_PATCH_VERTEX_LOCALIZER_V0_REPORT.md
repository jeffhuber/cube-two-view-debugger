# Raw Patch Vertex Localizer V0

Diagnostics-only artifact. This does not alter recognition behavior.

This probe is the first dependency-light raw image-patch vertex localizer over the 58 completed vertex+axis labels. It trains a leave-one-row-out ridge ranker on darkness/gradient patches around each local candidate, with the existing model/ray features kept only as a spatial prior.

## Summary

- Rows: 58
- Evaluated rows: 58
- Axis-good rows: 27
- Axis-blocked rows: 31
- Baseline strict/plausible: 6 / 10
- Coarse wide candidate-grid oracle strict/plausible: 39 / 53
- Raw-patch top-1 strict/plausible: 9 / 14
- Raw-patch gated strict/plausible: 6 / 10
- Axis-good strict baseline/raw-patch gated: 5 / 5
- Raw-patch gated accepted rows: 7
- Raw-patch top-1 improved/worsened rows by >5px: 9 / 15
- Raw-patch gated improved/worsened rows by >5px: 2 / 3
- Mean vertex error baseline/oracle/top-1/gated: 191.6 px / 46.3 px / 219.3 px / 195.6 px
- Median vertex error baseline/oracle/top-1/gated: 111.5 px / 24.9 px / 114.8 px / 111.5 px
- Best non-empty low-worsen raw-patch gate: accepted 3, improved 1, worsened 1, strict/plausible 6 / 10, mean 191.9 px (gain>=-0.1, score>=0.65)
- Best non-empty zero-worsen raw-patch gate: none
- Best non-empty raw-patch gate with <=2 worsens: accepted 3, improved 1, worsened 1, strict/plausible 6 / 10, mean 191.9 px (gain>=-0.1, score>=0.65)
- Production wiring recommendation: `do_not_wire`.
- Reason: Raw-patch ridge ranking is still not safe enough for recognizer wiring.

## Rows

| Row | Axis | Base | Oracle | Raw top-1 | Gated | Accepted | Delta | Score gain |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `canonical:12_B` | `axis_blocked` | 26.9 px | 26.9 px | 35.5 px | 26.9 px | no | 0.0 px | 0.013 |
| `canonical:14_B` | `axis_blocked` | 53.9 px | 24.0 px | 24.0 px | 53.9 px | no | 0.0 px | 0.062 |
| `canonical:15_A` | `axis_good` | 63.5 px | 9.7 px | 63.5 px | 63.5 px | no | 0.0 px | 0.000 |
| `canonical:15_B` | `axis_good` | 72.0 px | 17.0 px | 17.0 px | 72.0 px | no | 0.0 px | 0.054 |
| `canonical:17_B` | `axis_good` | 107.2 px | 22.2 px | 22.2 px | 107.2 px | no | 0.0 px | 0.026 |
| `canonical:21_A` | `axis_good` | 38.7 px | 25.1 px | 38.7 px | 38.7 px | no | 0.0 px | 0.000 |
| `canonical:21_B` | `axis_good` | 75.7 px | 24.5 px | 24.5 px | 75.7 px | no | 0.0 px | 0.023 |
| `canonical:24_A` | `axis_blocked` | 87.6 px | 26.3 px | 87.6 px | 87.6 px | no | 0.0 px | 0.000 |
| `canonical:26_A` | `axis_good` | 72.5 px | 16.0 px | 72.5 px | 72.5 px | no | 0.0 px | 0.000 |
| `canonical:26_B` | `axis_blocked` | 80.9 px | 36.9 px | 468.6 px | 80.9 px | no | 0.0 px | 0.088 |
| `canonical:27_B` | `axis_good` | 7.0 px | 4.4 px | 7.0 px | 7.0 px | no | 0.0 px | 0.000 |
| `canonical:28_A` | `axis_good` | 124.3 px | 32.1 px | 58.3 px | 124.3 px | no | 0.0 px | 0.042 |
| `canonical:28_B` | `axis_good` | 102.6 px | 22.4 px | 104.0 px | 102.6 px | no | 0.0 px | 0.009 |
| `canonical:29_A` | `axis_blocked` | 57.4 px | 23.8 px | 57.4 px | 57.4 px | no | 0.0 px | 0.000 |
| `canonical:29_B` | `axis_good` | 25.4 px | 25.4 px | 26.2 px | 25.4 px | no | 0.0 px | 0.006 |
| `canonical:30_A` | `axis_good` | 39.7 px | 28.4 px | 39.7 px | 39.7 px | no | 0.0 px | 0.000 |
| `canonical:30_B` | `axis_good` | 10.0 px | 10.0 px | 10.0 px | 10.0 px | no | 0.0 px | 0.000 |
| `canonical:31_A` | `axis_good` | 47.2 px | 32.9 px | 47.2 px | 47.2 px | no | 0.0 px | 0.000 |
| `canonical:31_B` | `axis_good` | 7.7 px | 7.7 px | 7.7 px | 7.7 px | no | 0.0 px | 0.000 |
| `canonical:32_A` | `axis_blocked` | 72.3 px | 19.1 px | 72.3 px | 72.3 px | no | 0.0 px | 0.000 |
| `canonical:32_B` | `axis_blocked` | 70.6 px | 20.4 px | 510.8 px | 70.6 px | no | 0.0 px | 0.103 |
| `canonical:36_B` | `axis_good` | 103.0 px | 32.6 px | 103.0 px | 103.0 px | no | 0.0 px | 0.000 |
| `canonical:42_B` | `axis_blocked` | 92.0 px | 10.3 px | 92.0 px | 92.0 px | no | 0.0 px | 0.000 |
| `canonical:44_A` | `axis_blocked` | 185.5 px | 40.1 px | 185.5 px | 185.5 px | no | 0.0 px | 0.000 |
| `canonical:44_B` | `axis_blocked` | 117.6 px | 34.8 px | 111.5 px | 117.6 px | no | 0.0 px | 0.061 |
| `canonical:57_A` | `axis_good` | 5.0 px | 5.0 px | 5.0 px | 5.0 px | no | 0.0 px | 0.000 |
| `canonical:61_A` | `axis_blocked` | 70.7 px | 24.8 px | 70.7 px | 70.7 px | no | 0.0 px | 0.000 |
| `canonical:61_B` | `axis_good` | 60.0 px | 12.5 px | 60.0 px | 60.0 px | no | 0.0 px | 0.000 |
| `active:42_A` | `axis_blocked` | 301.9 px | 27.6 px | 301.9 px | 301.9 px | no | 0.0 px | 0.000 |
| `active:36_A` | `axis_blocked` | 1018.0 px | 513.9 px | 1054.3 px | 1054.3 px | yes | -36.3 px | 0.471 |
| `active:23_B` | `axis_blocked` | 120.4 px | 8.2 px | 120.4 px | 120.4 px | no | 0.0 px | 0.000 |
| `active:37_A` | `axis_blocked` | 266.5 px | 22.2 px | 266.5 px | 266.5 px | no | 0.0 px | 0.000 |
| `active:23_A` | `axis_blocked` | 99.6 px | 26.3 px | 99.6 px | 99.6 px | no | 0.0 px | 0.000 |
| `active:37_B` | `axis_blocked` | 372.2 px | 28.3 px | 383.0 px | 372.2 px | no | 0.0 px | 0.021 |
| `active:12_A` | `axis_good` | 115.9 px | 28.9 px | 127.1 px | 115.9 px | no | 0.0 px | 0.016 |
| `active:27_A` | `axis_good` | 190.8 px | 33.1 px | 197.7 px | 190.8 px | no | 0.0 px | 0.004 |
| `active:24_B` | `axis_blocked` | 142.7 px | 23.0 px | 138.4 px | 138.4 px | yes | 4.3 px | 0.158 |
| `active:14_A` | `axis_good` | 435.3 px | 34.5 px | 617.9 px | 435.3 px | no | 0.0 px | 0.041 |
| `active:17_A` | `axis_blocked` | 118.2 px | 32.2 px | 118.2 px | 118.2 px | no | 0.0 px | 0.000 |
| `active:57_B` | `axis_blocked` | 829.8 px | 329.6 px | 808.5 px | 808.5 px | yes | 21.3 px | 0.214 |
| `active:62_A` | `axis_good` | 120.0 px | 40.4 px | 120.0 px | 120.0 px | no | 0.0 px | 0.000 |
| `active:62_B` | `axis_blocked` | 181.6 px | 17.9 px | 181.6 px | 181.6 px | no | 0.0 px | 0.000 |
| `active:22_B` | `axis_good` | 73.3 px | 32.9 px | 62.3 px | 73.3 px | no | 0.0 px | 0.019 |
| `active:58_A` | `axis_blocked` | 86.1 px | 14.6 px | 86.1 px | 86.1 px | no | 0.0 px | 0.000 |
| `active:58_B` | `axis_blocked` | 33.8 px | 24.5 px | 33.8 px | 33.8 px | no | 0.0 px | 0.000 |
| `active:39_A` | `axis_blocked` | 184.5 px | 10.1 px | 184.5 px | 184.5 px | no | 0.0 px | 0.000 |
| `active:47_A` | `axis_good` | 263.4 px | 23.8 px | 263.4 px | 263.4 px | no | 0.0 px | 0.000 |
| `active:47_B` | `axis_good` | 323.2 px | 15.0 px | 411.1 px | 323.2 px | no | 0.0 px | 0.055 |
| `active:49_A` | `axis_blocked` | 390.1 px | 24.7 px | 780.0 px | 390.1 px | no | 0.0 px | 0.088 |
| `active:22_A` | `axis_good` | 167.4 px | 30.3 px | 167.4 px | 167.4 px | no | 0.0 px | 0.000 |
| `active:49_B` | `axis_blocked` | 183.6 px | 6.0 px | 183.6 px | 183.6 px | no | 0.0 px | 0.000 |
| `active:48_B` | `axis_blocked` | 156.2 px | 26.5 px | 169.2 px | 156.2 px | no | 0.0 px | 0.017 |
| `active:39_B` | `axis_good` | 238.3 px | 30.9 px | 233.2 px | 233.2 px | yes | 5.1 px | 0.120 |
| `active:46_B` | `axis_blocked` | 573.8 px | 101.2 px | 629.0 px | 629.0 px | yes | -55.2 px | 0.156 |
| `active:25_B` | `axis_good` | 362.4 px | 38.0 px | 426.5 px | 362.4 px | no | 0.0 px | 0.037 |
| `active:48_A` | `axis_blocked` | 369.4 px | 13.5 px | 439.1 px | 369.4 px | no | 0.0 px | 0.044 |
| `active:46_A` | `axis_blocked` | 818.1 px | 308.9 px | 820.1 px | 820.1 px | yes | -2.0 px | 0.075 |
| `active:25_A` | `axis_blocked` | 702.4 px | 203.6 px | 871.6 px | 871.6 px | yes | -169.2 px | 0.397 |

## Interpretation

- This is a supervised raw-patch diagnostic, not production wiring.
- It is deliberately small: no CNN dependency and no persistent model artifact. The only question is whether local image patches contain ranking signal beyond hand-authored features.
- If it produces a meaningful zero-worsen gate, the next step is a proper trainable patch model with more labels and held-out validation.
- If it does not improve over the patch/junction feature probe, the limiting factor is likely data volume or candidate/model geometry, not another small ranker.
