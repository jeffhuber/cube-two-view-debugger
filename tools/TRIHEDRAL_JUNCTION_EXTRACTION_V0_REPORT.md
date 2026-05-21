# Trihedral Junction Extraction V0

Diagnostics-only artifact. This does not alter recognition behavior.

This probe extracts three dark line hypotheses near the visible trihedral region, intersects them, and evaluates the resulting junction against human labels.

## Summary

- Rows: 28
- Evaluated rows: 28
- Axis-good rows: 17
- Axis-blocked rows: 11
- Baseline strict/plausible: 6 / 9
- Model junction strict/plausible: 4 / 5
- Model junction gated strict/plausible: 1 / 2
- Human-axis oracle gated strict/plausible: 1 / 2
- Axis-good strict baseline/model-gated: 5 / 1
- Model junction gated accepted rows: 15
- Human-axis oracle gated accepted rows: 15
- Model junction gated improved/worsened rows by >5px: 3 / 9
- Human-axis oracle gated improved/worsened rows by >5px: 5 / 8
- Mean vertex error baseline/model-gated/oracle-gated: 67.0 px / 77.3 px / 76.3 px
- Median vertex error baseline/model-gated/oracle-gated: 70.7 px / 74.2 px / 72.5 px
- Best non-empty low-worsen model gate: accepted 2, improved 0, worsened 1, strict/plausible 6 / 8, mean 68.1 px (spread<=20.0, score>=0.45, contrast>=0.05, move<=160.0)
- Best non-empty model gate with <=2 worsens: accepted 2, improved 0, worsened 1, strict/plausible 6 / 8, mean 68.1 px (spread<=20.0, score>=0.45, contrast>=0.05, move<=160.0)

## Rows

| Row | Axis | Base | Model gated | Accepted | Delta | Spread | Min line score | Oracle gated | Oracle accepted | Oracle delta |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|
| `12_B` | `axis_blocked` | 26.9 px | 66.3 px | yes | -39.5 px | 51.2 px | 0.735 | 26.9 px | no | 0.0 px |
| `14_B` | `axis_blocked` | 53.9 px | 78.7 px | yes | -24.8 px | 12.1 px | 0.795 | 71.8 px | yes | -17.9 px |
| `15_A` | `axis_good` | 63.5 px | 63.5 px | no | 0.0 px | 86.1 px | 0.781 | 63.5 px | no | 0.0 px |
| `15_B` | `axis_good` | 72.0 px | 78.9 px | yes | -6.9 px | 36.4 px | 0.808 | 61.2 px | yes | 10.7 px |
| `17_B` | `axis_good` | 107.2 px | 107.2 px | no | 0.0 px | 60.6 px | 0.875 | 80.8 px | yes | 26.4 px |
| `21_A` | `axis_good` | 38.7 px | 38.7 px | no | 0.0 px | 48.3 px | 0.740 | 38.7 px | no | 0.0 px |
| `21_B` | `axis_good` | 75.7 px | 75.7 px | no | 0.0 px | 92.6 px | 0.746 | 75.7 px | no | 0.0 px |
| `24_A` | `axis_blocked` | 87.6 px | 87.6 px | no | 0.0 px | 55.7 px | 0.848 | 87.6 px | no | 0.0 px |
| `26_A` | `axis_good` | 72.5 px | 67.6 px | yes | 5.0 px | 50.9 px | 0.786 | 72.5 px | no | 0.0 px |
| `26_B` | `axis_blocked` | 80.9 px | 80.9 px | no | 0.0 px | 126.4 px | 0.800 | 80.9 px | no | 0.0 px |
| `27_B` | `axis_good` | 7.0 px | 88.9 px | yes | -81.9 px | 24.7 px | 0.917 | 88.7 px | yes | -81.7 px |
| `28_A` | `axis_good` | 124.3 px | 124.3 px | no | 0.0 px | 116.4 px | 0.839 | 86.9 px | yes | 37.4 px |
| `28_B` | `axis_good` | 102.6 px | 102.6 px | no | 0.0 px | 77.2 px | 0.755 | 102.6 px | no | 0.0 px |
| `29_A` | `axis_blocked` | 57.4 px | 57.4 px | no | 0.0 px | 59.1 px | 0.840 | 57.4 px | no | 0.0 px |
| `29_B` | `axis_good` | 25.4 px | 77.3 px | yes | -51.9 px | 52.4 px | 0.869 | 78.3 px | yes | -52.9 px |
| `30_A` | `axis_good` | 39.7 px | 72.6 px | yes | -32.9 px | 18.0 px | 0.994 | 79.4 px | yes | -39.6 px |
| `30_B` | `axis_good` | 10.0 px | 65.5 px | yes | -55.5 px | 47.3 px | 0.919 | 61.1 px | yes | -51.1 px |
| `31_A` | `axis_good` | 47.2 px | 77.5 px | yes | -30.3 px | 35.2 px | 0.945 | 69.7 px | yes | -22.5 px |
| `31_B` | `axis_good` | 7.7 px | 56.2 px | yes | -48.6 px | 32.7 px | 0.942 | 69.7 px | yes | -62.0 px |
| `32_A` | `axis_blocked` | 72.3 px | 68.5 px | yes | 3.8 px | 10.1 px | 0.873 | 68.7 px | yes | 3.6 px |
| `32_B` | `axis_blocked` | 70.6 px | 67.8 px | yes | 2.9 px | 40.1 px | 0.717 | 72.5 px | yes | -1.9 px |
| `36_B` | `axis_good` | 103.0 px | 77.7 px | yes | 25.2 px | 25.0 px | 0.781 | 77.9 px | yes | 25.1 px |
| `42_B` | `axis_blocked` | 92.0 px | 51.6 px | yes | 40.4 px | 20.5 px | 0.766 | 50.9 px | yes | 41.1 px |
| `44_A` | `axis_blocked` | 185.5 px | 185.5 px | no | 0.0 px | 23.5 px | 0.659 | 185.5 px | no | 0.0 px |
| `44_B` | `axis_blocked` | 117.6 px | 117.6 px | no | 0.0 px | 33.2 px | 0.725 | 117.6 px | no | 0.0 px |
| `57_A` | `axis_good` | 5.0 px | 5.0 px | no | 0.0 px | 63.7 px | 0.779 | 78.8 px | yes | -73.9 px |
| `61_A` | `axis_blocked` | 70.7 px | 70.7 px | no | 0.0 px | 61.3 px | 0.807 | 70.7 px | no | 0.0 px |
| `61_B` | `axis_good` | 60.0 px | 53.5 px | yes | 6.5 px | 37.0 px | 0.878 | 60.0 px | no | 0.0 px |

## Interpretation

- This is explicit line/junction extraction, not production wiring.
- The current implementation is a negative result: default gating reduces strict/plausible rows and accepts too many worsened vertices.
- A useful promotion-shaped signal would increase strict/plausible rows while keeping worsened accepted rows near zero.
- The best low-worsen gate still underperforms baseline, so this line-extraction path should remain a diagnostic and the next step should be learned vertex localization.
