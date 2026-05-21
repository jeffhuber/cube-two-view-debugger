# Patch/Junction Vertex Localizer V0

Diagnostics-only artifact. This does not alter recognition behavior.

This probe stops adding scalar gates to the prior localizer path and instead asks whether richer local image evidence can rank the visible trihedral vertex candidates better. It uses the 58 completed human vertex+axis labels for leave-one-row-out evaluation.

## Summary

- Rows: 58
- Evaluated rows: 58
- Axis-good rows: 27
- Axis-blocked rows: 31
- Baseline strict/plausible: 6 / 10
- Coarse wide candidate-grid oracle strict/plausible: 39 / 53
- Patch/junction top-1 strict/plausible: 15 / 22
- Patch/junction gated strict/plausible: 7 / 13
- Axis-good strict baseline/patch-gated: 5 / 5
- Patch/junction gated accepted rows: 6
- Patch/junction top-1 improved/worsened rows by >5px: 30 / 27
- Patch/junction gated improved/worsened rows by >5px: 6 / 0
- Mean vertex error baseline/oracle/top-1/gated: 191.6 px / 46.3 px / 230.9 px / 181.7 px
- Median vertex error baseline/oracle/top-1/gated: 111.5 px / 24.9 px / 85.7 px / 101.1 px
- Best non-empty low-worsen patch/junction gate: accepted 6, improved 6, worsened 0, strict/plausible 7 / 13, mean 181.7 px (gain>=1.5, score>=-3.0)
- Best non-empty zero-worsen patch/junction gate: accepted 6, improved 6, worsened 0, strict/plausible 7 / 13, mean 181.7 px (gain>=1.5, score>=-3.0)
- Best non-empty patch/junction gate with <=2 worsens: accepted 6, improved 6, worsened 0, strict/plausible 7 / 13, mean 181.7 px (gain>=1.5, score>=-3.0)
- Production wiring recommendation: `diagnostics_only_needs_more_validation`.
- Reason: A zero-worsen gate improves strict coverage on this label set, but this remains too small and too local for production wiring.

## Feature Families

- Patch texture: local darkness, dark-pixel fraction, gradient strength, and structure-tensor cornerness at 10/22/44 px radii.
- Explicit junction structure: radial dark-line support sampled around the candidate, including top-3 arm strength and peak count.
- Face-boundary consistency: near/far forward ray support, side contrast, and backward-continuation penalties along the three projected cube axes.

## Rows

| Row | Axis | Base | Oracle | Patch top-1 | Gated | Accepted | Delta | Score gain |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `canonical:12_B` | `axis_blocked` | 26.9 px | 26.9 px | 199.7 px | 26.9 px | no | 0.0 px | 2.156 |
| `canonical:14_B` | `axis_blocked` | 53.9 px | 24.0 px | 43.9 px | 53.9 px | no | 0.0 px | 0.341 |
| `canonical:15_A` | `axis_good` | 63.5 px | 9.7 px | 9.7 px | 63.5 px | no | 0.0 px | 1.699 |
| `canonical:15_B` | `axis_good` | 72.0 px | 17.0 px | 51.3 px | 72.0 px | no | 0.0 px | 1.666 |
| `canonical:17_B` | `axis_good` | 107.2 px | 22.2 px | 22.2 px | 107.2 px | no | 0.0 px | 1.380 |
| `canonical:21_A` | `axis_good` | 38.7 px | 25.1 px | 25.1 px | 38.7 px | no | 0.0 px | 1.861 |
| `canonical:21_B` | `axis_good` | 75.7 px | 24.5 px | 90.0 px | 75.7 px | no | 0.0 px | 1.196 |
| `canonical:24_A` | `axis_blocked` | 87.6 px | 26.3 px | 161.8 px | 87.6 px | no | 0.0 px | 1.591 |
| `canonical:26_A` | `axis_good` | 72.5 px | 16.0 px | 67.5 px | 72.5 px | no | 0.0 px | 4.977 |
| `canonical:26_B` | `axis_blocked` | 80.9 px | 36.9 px | 36.9 px | 36.9 px | yes | 44.0 px | 1.889 |
| `canonical:27_B` | `axis_good` | 7.0 px | 4.4 px | 386.1 px | 7.0 px | no | 0.0 px | 1.906 |
| `canonical:28_A` | `axis_good` | 124.3 px | 32.1 px | 32.1 px | 124.3 px | no | 0.0 px | 2.328 |
| `canonical:28_B` | `axis_good` | 102.6 px | 22.4 px | 522.8 px | 102.6 px | no | 0.0 px | 0.889 |
| `canonical:29_A` | `axis_blocked` | 57.4 px | 23.8 px | 23.8 px | 57.4 px | no | 0.0 px | 0.959 |
| `canonical:29_B` | `axis_good` | 25.4 px | 25.4 px | 25.4 px | 25.4 px | no | 0.0 px | 0.000 |
| `canonical:30_A` | `axis_good` | 39.7 px | 28.4 px | 390.4 px | 39.7 px | no | 0.0 px | 3.093 |
| `canonical:30_B` | `axis_good` | 10.0 px | 10.0 px | 371.6 px | 10.0 px | no | 0.0 px | 10.688 |
| `canonical:31_A` | `axis_good` | 47.2 px | 32.9 px | 340.8 px | 47.2 px | no | 0.0 px | 1.319 |
| `canonical:31_B` | `axis_good` | 7.7 px | 7.7 px | 161.6 px | 7.7 px | no | 0.0 px | 8.604 |
| `canonical:32_A` | `axis_blocked` | 72.3 px | 19.1 px | 405.4 px | 72.3 px | no | 0.0 px | 1.035 |
| `canonical:32_B` | `axis_blocked` | 70.6 px | 20.4 px | 481.6 px | 70.6 px | no | 0.0 px | 7.042 |
| `canonical:36_B` | `axis_good` | 103.0 px | 32.6 px | 238.9 px | 103.0 px | no | 0.0 px | 0.177 |
| `canonical:42_B` | `axis_blocked` | 92.0 px | 10.3 px | 439.4 px | 92.0 px | no | 0.0 px | 4.132 |
| `canonical:44_A` | `axis_blocked` | 185.5 px | 40.1 px | 40.1 px | 185.5 px | no | 0.0 px | 3.454 |
| `canonical:44_B` | `axis_blocked` | 117.6 px | 34.8 px | 56.5 px | 56.5 px | yes | 61.1 px | 1.555 |
| `canonical:57_A` | `axis_good` | 5.0 px | 5.0 px | 187.4 px | 5.0 px | no | 0.0 px | 1.065 |
| `canonical:61_A` | `axis_blocked` | 70.7 px | 24.8 px | 24.8 px | 70.7 px | no | 0.0 px | 1.026 |
| `canonical:61_B` | `axis_good` | 60.0 px | 12.5 px | 115.8 px | 60.0 px | no | 0.0 px | 2.127 |
| `active:42_A` | `axis_blocked` | 301.9 px | 27.6 px | 423.1 px | 301.9 px | no | 0.0 px | 12.587 |
| `active:36_A` | `axis_blocked` | 1018.0 px | 513.9 px | 905.2 px | 1018.0 px | no | 0.0 px | 8.030 |
| `active:23_B` | `axis_blocked` | 120.4 px | 8.2 px | 8.2 px | 8.2 px | yes | 112.2 px | 2.033 |
| `active:37_A` | `axis_blocked` | 266.5 px | 22.2 px | 81.1 px | 266.5 px | no | 0.0 px | 2.527 |
| `active:23_A` | `axis_blocked` | 99.6 px | 26.3 px | 26.3 px | 99.6 px | no | 0.0 px | 1.706 |
| `active:37_B` | `axis_blocked` | 372.2 px | 28.3 px | 319.3 px | 319.3 px | yes | 52.9 px | 6.426 |
| `active:12_A` | `axis_good` | 115.9 px | 28.9 px | 429.4 px | 115.9 px | no | 0.0 px | 7.994 |
| `active:27_A` | `axis_good` | 190.8 px | 33.1 px | 51.4 px | 190.8 px | no | 0.0 px | 1.176 |
| `active:24_B` | `axis_blocked` | 142.7 px | 23.0 px | 23.0 px | 142.7 px | no | 0.0 px | 8.778 |
| `active:14_A` | `axis_good` | 435.3 px | 34.5 px | 551.9 px | 435.3 px | no | 0.0 px | 8.994 |
| `active:17_A` | `axis_blocked` | 118.2 px | 32.2 px | 32.2 px | 32.2 px | yes | 85.9 px | 9.438 |
| `active:57_B` | `axis_blocked` | 829.8 px | 329.6 px | 976.6 px | 829.8 px | no | 0.0 px | 7.790 |
| `active:62_A` | `axis_good` | 120.0 px | 40.4 px | 40.4 px | 120.0 px | no | 0.0 px | 2.280 |
| `active:62_B` | `axis_blocked` | 181.6 px | 17.9 px | 17.9 px | 181.6 px | no | 0.0 px | 1.669 |
| `active:22_B` | `axis_good` | 73.3 px | 32.9 px | 81.3 px | 73.3 px | no | 0.0 px | 0.945 |
| `active:58_A` | `axis_blocked` | 86.1 px | 14.6 px | 14.6 px | 86.1 px | no | 0.0 px | 3.167 |
| `active:58_B` | `axis_blocked` | 33.8 px | 24.5 px | 24.5 px | 33.8 px | no | 0.0 px | 0.309 |
| `active:39_A` | `axis_blocked` | 184.5 px | 10.1 px | 53.9 px | 184.5 px | no | 0.0 px | 7.409 |
| `active:47_A` | `axis_good` | 263.4 px | 23.8 px | 404.2 px | 263.4 px | no | 0.0 px | 5.166 |
| `active:47_B` | `axis_good` | 323.2 px | 15.0 px | 15.0 px | 323.2 px | no | 0.0 px | 11.360 |
| `active:49_A` | `axis_blocked` | 390.1 px | 24.7 px | 24.7 px | 390.1 px | no | 0.0 px | 12.511 |
| `active:22_A` | `axis_good` | 167.4 px | 30.3 px | 30.3 px | 167.4 px | no | 0.0 px | 1.251 |
| `active:49_B` | `axis_blocked` | 183.6 px | 6.0 px | 6.0 px | 183.6 px | no | 0.0 px | 5.641 |
| `active:48_B` | `axis_blocked` | 156.2 px | 26.5 px | 436.3 px | 156.2 px | no | 0.0 px | 2.509 |
| `active:39_B` | `axis_good` | 238.3 px | 30.9 px | 542.5 px | 238.3 px | no | 0.0 px | 13.337 |
| `active:46_B` | `axis_blocked` | 573.8 px | 101.2 px | 579.3 px | 573.8 px | no | 0.0 px | 0.444 |
| `active:25_B` | `axis_good` | 362.4 px | 38.0 px | 553.3 px | 362.4 px | no | 0.0 px | 12.451 |
| `active:48_A` | `axis_blocked` | 369.4 px | 13.5 px | 439.1 px | 369.4 px | no | 0.0 px | 3.236 |
| `active:46_A` | `axis_blocked` | 818.1 px | 308.9 px | 868.2 px | 818.1 px | no | 0.0 px | 7.559 |
| `active:25_A` | `axis_blocked` | 702.4 px | 203.6 px | 479.0 px | 479.0 px | yes | 223.4 px | 25.750 |

## Interpretation

- This is a supervised diagnostics pass, not a recognizer policy.
- The key success test is not just top-1 improvement. A production-shaped signal would need a non-empty confidence gate that improves coverage without accepting worsened rows.
- Compared with the expanded-label KNN baseline from the prior report, patch/junction features modestly improve top-1 strict count but still have poor ungated reliability.
- The conservative gate is intentionally small. It is useful evidence for a possible confidence feature, not a production policy.
- If patch/junction features do not materially improve top-1 and gated safety over the expanded KNN baseline, the next step should be a real trained image-patch model or direct line/junction detector, not more hand-authored thresholds.
- If a zero-worsen gate appears, it should remain diagnostics-only until validated on more labeled rows and hard-background cases.
