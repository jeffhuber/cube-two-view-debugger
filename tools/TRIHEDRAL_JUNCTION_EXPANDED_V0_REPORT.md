# Trihedral Junction Expanded V0

Diagnostics-only artifact. This does not alter recognition behavior.

This report reruns the explicit dark-line trihedral junction extractor on the current 58-row vertex+axis label set. It answers whether the old structural detector becomes useful once evaluated against the expanded labels and fixed active-label coordinate space.

## Label Set

- Labeled rows: 58
- Canonical rows: 28
- Active-learning rows: 30

## Summary

- Evaluated rows: 58
- Axis-good rows: 27
- Axis-blocked rows: 31
- Baseline strict/plausible: 6 / 10
- Model junction strict/plausible: 4 / 5
- Model junction gated strict/plausible: 1 / 2
- Human-axis oracle strict/plausible: 5 / 9
- Human-axis oracle gated strict/plausible: 1 / 3
- Axis-good strict baseline/model-gated: 5 / 1
- Model junction gated accepted rows: 19
- Model junction gated improved/worsened rows by >5px: 5 / 11
- Human-axis oracle gated improved/worsened rows by >5px: 7 / 9
- Mean vertex error baseline/model-gated/oracle-gated: 191.6 px / 195.8 px / 194.8 px
- Median vertex error baseline/model-gated/oracle-gated: 111.5 px / 104.9 px / 88.1 px
- Best non-empty low-worsen model gate: accepted 3, improved 1, worsened 1, strict/plausible 6 / 9, mean 190.7 px (spread<=20.0, score>=0.45, contrast>=0.05, move<=160.0)
- Best non-empty model gate with <=2 worsens: accepted 7, improved 4, worsened 2, strict/plausible 6 / 9, mean 187.2 px (spread<=20.0, score>=0.45, contrast>=-0.02, move<=220.0)
- Production wiring recommendation: `do_not_wire`.
- Reason: Expanded-label explicit junction extraction underperforms baseline and accepts 11 worsened rows versus 5 improved rows.

## Rows

| Row | Axis | Base | Model best | Model gated | Accepted | Delta | Spread | Min score | Oracle gated | Oracle delta |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `canonical:12_B` | `axis_blocked` | 26.9 px | 66.3 px | 66.3 px | yes | -39.5 px | 51.2 px | 0.735 | 26.9 px | 0.0 px |
| `canonical:14_B` | `axis_blocked` | 53.9 px | 78.7 px | 78.7 px | yes | -24.8 px | 12.1 px | 0.795 | 71.8 px | -17.9 px |
| `canonical:15_A` | `axis_good` | 63.5 px | 24.2 px | 63.5 px | no | 0.0 px | 86.1 px | 0.781 | 63.5 px | 0.0 px |
| `canonical:15_B` | `axis_good` | 72.0 px | 78.9 px | 78.9 px | yes | -6.9 px | 36.4 px | 0.808 | 61.2 px | 10.7 px |
| `canonical:17_B` | `axis_good` | 107.2 px | 75.5 px | 107.2 px | no | 0.0 px | 60.6 px | 0.875 | 80.8 px | 26.4 px |
| `canonical:21_A` | `axis_good` | 38.7 px | 63.1 px | 38.7 px | no | 0.0 px | 48.3 px | 0.740 | 38.7 px | 0.0 px |
| `canonical:21_B` | `axis_good` | 75.7 px | 68.8 px | 75.7 px | no | 0.0 px | 92.6 px | 0.746 | 75.7 px | 0.0 px |
| `canonical:24_A` | `axis_blocked` | 87.6 px | 76.7 px | 87.6 px | no | 0.0 px | 55.7 px | 0.848 | 87.6 px | 0.0 px |
| `canonical:26_A` | `axis_good` | 72.5 px | 67.6 px | 67.6 px | yes | 5.0 px | 50.9 px | 0.786 | 72.5 px | 0.0 px |
| `canonical:26_B` | `axis_blocked` | 80.9 px | 17.9 px | 80.9 px | no | 0.0 px | 126.4 px | 0.800 | 80.9 px | 0.0 px |
| `canonical:27_B` | `axis_good` | 7.0 px | 88.9 px | 88.9 px | yes | -81.9 px | 24.7 px | 0.917 | 88.7 px | -81.7 px |
| `canonical:28_A` | `axis_good` | 124.3 px | 11.6 px | 124.3 px | no | 0.0 px | 116.4 px | 0.839 | 86.9 px | 37.4 px |
| `canonical:28_B` | `axis_good` | 102.6 px | 85.0 px | 102.6 px | no | 0.0 px | 77.2 px | 0.755 | 102.6 px | 0.0 px |
| `canonical:29_A` | `axis_blocked` | 57.4 px | 86.2 px | 57.4 px | no | 0.0 px | 59.1 px | 0.840 | 57.4 px | 0.0 px |
| `canonical:29_B` | `axis_good` | 25.4 px | 77.3 px | 77.3 px | yes | -51.9 px | 52.4 px | 0.869 | 78.3 px | -52.9 px |
| `canonical:30_A` | `axis_good` | 39.7 px | 72.6 px | 72.6 px | yes | -32.9 px | 18.0 px | 0.994 | 79.4 px | -39.6 px |
| `canonical:30_B` | `axis_good` | 10.0 px | 65.5 px | 65.5 px | yes | -55.5 px | 47.3 px | 0.919 | 61.1 px | -51.1 px |
| `canonical:31_A` | `axis_good` | 47.2 px | 77.5 px | 77.5 px | yes | -30.3 px | 35.2 px | 0.945 | 69.7 px | -22.5 px |
| `canonical:31_B` | `axis_good` | 7.7 px | 56.2 px | 56.2 px | yes | -48.6 px | 32.7 px | 0.942 | 69.7 px | -62.0 px |
| `canonical:32_A` | `axis_blocked` | 72.3 px | 68.5 px | 68.5 px | yes | 3.8 px | 10.1 px | 0.873 | 68.7 px | 3.6 px |
| `canonical:32_B` | `axis_blocked` | 70.6 px | 67.8 px | 67.8 px | yes | 2.9 px | 40.1 px | 0.717 | 72.5 px | -1.9 px |
| `canonical:36_B` | `axis_good` | 103.0 px | 77.7 px | 77.7 px | yes | 25.2 px | 25.0 px | 0.781 | 77.9 px | 25.1 px |
| `canonical:42_B` | `axis_blocked` | 92.0 px | 51.6 px | 51.6 px | yes | 40.4 px | 20.5 px | 0.766 | 50.9 px | 41.1 px |
| `canonical:44_A` | `axis_blocked` | 185.5 px | 9.2 px | 185.5 px | no | 0.0 px | 23.5 px | 0.659 | 185.5 px | 0.0 px |
| `canonical:44_B` | `axis_blocked` | 117.6 px | 49.4 px | 117.6 px | no | 0.0 px | 33.2 px | 0.725 | 117.6 px | 0.0 px |
| `canonical:57_A` | `axis_good` | 5.0 px | 98.4 px | 5.0 px | no | 0.0 px | 63.7 px | 0.779 | 78.8 px | -73.9 px |
| `canonical:61_A` | `axis_blocked` | 70.7 px | 63.3 px | 70.7 px | no | 0.0 px | 61.3 px | 0.807 | 70.7 px | 0.0 px |
| `canonical:61_B` | `axis_good` | 60.0 px | 53.5 px | 53.5 px | yes | 6.5 px | 37.0 px | 0.878 | 60.0 px | 0.0 px |
| `active:42_A` | `axis_blocked` | 301.9 px | 385.5 px | 301.9 px | no | 0.0 px | 156.3 px | 0.797 | 417.7 px | -115.9 px |
| `active:36_A` | `axis_blocked` | 1018.0 px | 1164.0 px | 1018.0 px | no | 0.0 px | 162.6 px | 0.660 | 1018.0 px | 0.0 px |
| `active:23_B` | `axis_blocked` | 120.4 px | 74.2 px | 120.4 px | no | 0.0 px | 20.0 px | 0.693 | 120.4 px | 0.0 px |
| `active:37_A` | `axis_blocked` | 266.5 px | 155.0 px | 266.5 px | no | 0.0 px | 64.5 px | 0.653 | 266.5 px | 0.0 px |
| `active:23_A` | `axis_blocked` | 99.6 px | 65.9 px | 99.6 px | no | 0.0 px | 74.4 px | 0.761 | 99.6 px | 0.0 px |
| `active:37_B` | `axis_blocked` | 372.2 px | 514.2 px | 372.2 px | no | 0.0 px | 73.6 px | 0.794 | 372.2 px | 0.0 px |
| `active:12_A` | `axis_good` | 115.9 px | 77.2 px | 115.9 px | no | 0.0 px | 103.3 px | 0.721 | 115.9 px | 0.0 px |
| `active:27_A` | `axis_good` | 190.8 px | 58.9 px | 190.8 px | no | 0.0 px | 42.1 px | 0.692 | 190.8 px | 0.0 px |
| `active:24_B` | `axis_blocked` | 142.7 px | 60.6 px | 60.6 px | yes | 82.0 px | 11.0 px | 0.770 | 142.7 px | 0.0 px |
| `active:14_A` | `axis_good` | 435.3 px | 613.0 px | 435.3 px | no | 0.0 px | 113.3 px | 0.693 | 435.3 px | 0.0 px |
| `active:17_A` | `axis_blocked` | 118.2 px | 57.5 px | 118.2 px | no | 0.0 px | 50.2 px | 0.675 | 118.2 px | 0.0 px |
| `active:57_B` | `axis_blocked` | 829.8 px | 908.4 px | 829.8 px | no | 0.0 px | 205.8 px | 0.706 | 829.8 px | 0.0 px |
| `active:62_A` | `axis_good` | 120.0 px | 85.2 px | 120.0 px | no | 0.0 px | 78.7 px | 0.685 | 120.0 px | 0.0 px |
| `active:62_B` | `axis_blocked` | 181.6 px | 71.3 px | 181.6 px | no | 0.0 px | 25.3 px | 0.752 | 77.0 px | 104.6 px |
| `active:22_B` | `axis_good` | 73.3 px | 96.5 px | 96.5 px | yes | -23.2 px | 34.5 px | 0.745 | 73.3 px | 0.0 px |
| `active:58_A` | `axis_blocked` | 86.1 px | 68.0 px | 68.0 px | yes | 18.1 px | 40.3 px | 0.718 | 86.1 px | 0.0 px |
| `active:58_B` | `axis_blocked` | 33.8 px | 65.1 px | 65.1 px | yes | -31.3 px | 24.2 px | 0.797 | 33.8 px | 0.0 px |
| `active:39_A` | `axis_blocked` | 184.5 px | 75.5 px | 184.5 px | no | 0.0 px | 0.5 px | 0.681 | 184.5 px | 0.0 px |
| `active:47_A` | `axis_good` | 263.4 px | 156.2 px | 263.4 px | no | 0.0 px | 151.9 px | 0.666 | 263.4 px | 0.0 px |
| `active:47_B` | `axis_good` | 323.2 px | 478.9 px | 323.2 px | no | 0.0 px | 47.7 px | 0.674 | 323.2 px | 0.0 px |
| `active:49_A` | `axis_blocked` | 390.1 px | 535.5 px | 390.1 px | no | 0.0 px | 110.7 px | 0.654 | 390.1 px | 0.0 px |
| `active:22_A` | `axis_good` | 167.4 px | 91.9 px | 167.4 px | no | 0.0 px | 4.0 px | 0.679 | 167.4 px | 0.0 px |
| `active:49_B` | `axis_blocked` | 183.6 px | 71.6 px | 183.6 px | no | 0.0 px | 183.1 px | 0.693 | 183.6 px | 0.0 px |
| `active:48_B` | `axis_blocked` | 156.2 px | 266.8 px | 156.2 px | no | 0.0 px | 265.9 px | 0.706 | 69.8 px | 86.4 px |
| `active:39_B` | `axis_good` | 238.3 px | 327.2 px | 238.3 px | no | 0.0 px | 196.5 px | 0.685 | 238.3 px | 0.0 px |
| `active:46_B` | `axis_blocked` | 573.8 px | 604.2 px | 573.8 px | no | 0.0 px | 284.0 px | 0.694 | 573.8 px | 0.0 px |
| `active:25_B` | `axis_good` | 362.4 px | 562.3 px | 362.4 px | no | 0.0 px | 47.0 px | 0.992 | 362.4 px | 0.0 px |
| `active:48_A` | `axis_blocked` | 369.4 px | 540.7 px | 369.4 px | no | 0.0 px | 92.6 px | 0.686 | 369.4 px | 0.0 px |
| `active:46_A` | `axis_blocked` | 818.1 px | 952.5 px | 818.1 px | no | 0.0 px | 163.6 px | 0.724 | 818.1 px | 0.0 px |
| `active:25_A` | `axis_blocked` | 702.4 px | 615.5 px | 702.4 px | no | 0.0 px | 41.4 px | 0.823 | 702.4 px | 0.0 px |

## Interpretation

- This is an expanded-label rerun of explicit line/junction extraction, not production wiring.
- The result is a strong negative for this V0 structural detector: model-gated output underperforms baseline strict/plausible counts and accepts more worsened rows than improved rows.
- The human-axis oracle also underperforms baseline, which means the failure is not only model-axis choice; the extracted line intersections themselves are not stable enough.
- The useful conclusion is to stop revisiting this particular dark-line intersection objective. A future line path would need a materially different detector, such as segment grouping with actual face-boundary topology or a trained patch model.
