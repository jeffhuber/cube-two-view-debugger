# Ray-Start Vertex Refinement V0

Diagnostics-only artifact. This does not alter recognition behavior.

This probe searches for a visible-trihedral vertex by rewarding dark outgoing axis rays and penalizing dark continuation behind the candidate point.

## Summary

- Rows: 28
- Evaluated rows: 28
- Axis-good rows: 17
- Axis-blocked rows: 11
- Baseline strict/plausible: 6 / 9
- Model-axis best strict/plausible: 4 / 7
- Model-axis gated strict/plausible: 5 / 8
- Human-axis oracle best strict/plausible: 6 / 14
- Human-axis oracle gated strict/plausible: 5 / 13
- Axis-good strict baseline/model-gated/oracle-gated: 5 / 4 / 3
- Model-axis gated accepted rows: 17
- Human-axis oracle gated accepted rows: 27
- Model-axis gated improved/worsened rows by >5px: 9 / 8
- Human-axis oracle gated improved/worsened rows by >5px: 16 / 9
- Mean vertex error baseline/model-gated/oracle-gated: 67.0 px / 60.9 px / 49.2 px
- Median vertex error baseline/model-gated/oracle-gated: 70.7 px / 58.3 px / 52.6 px
- Best non-empty low-worsen model gate: accepted 5, improved 2, worsened 3, strict/plausible 5 / 7, mean 68.8 px (gain>=0.0, meanStart>=-0.05, minStart>=0.1)
- Best non-empty model gate with <=2 worsens: none

## Rows

| Row | Axis | Base | Model gated | Accepted | Delta | Oracle gated | Oracle accepted | Oracle delta |
|---|---|---:|---:|---|---:|---:|---|---:|
| `12_B` | `axis_blocked` | 26.9 px | 26.9 px | no | 0.0 px | 47.0 px | yes | -20.1 px |
| `14_B` | `axis_blocked` | 53.9 px | 53.9 px | no | 0.0 px | 56.6 px | yes | -2.7 px |
| `15_A` | `axis_good` | 63.5 px | 40.4 px | yes | 23.1 px | 40.4 px | yes | 23.1 px |
| `15_B` | `axis_good` | 72.0 px | 54.2 px | yes | 17.8 px | 56.0 px | yes | 16.0 px |
| `17_B` | `axis_good` | 107.2 px | 62.3 px | yes | 44.9 px | 62.3 px | yes | 44.9 px |
| `21_A` | `axis_good` | 38.7 px | 63.4 px | yes | -24.7 px | 56.5 px | yes | -17.7 px |
| `21_B` | `axis_good` | 75.7 px | 15.9 px | yes | 59.8 px | 15.9 px | yes | 59.8 px |
| `24_A` | `axis_blocked` | 87.6 px | 87.6 px | no | 0.0 px | 55.6 px | yes | 32.0 px |
| `26_A` | `axis_good` | 72.5 px | 48.9 px | yes | 23.7 px | 48.0 px | yes | 24.6 px |
| `26_B` | `axis_blocked` | 80.9 px | 80.9 px | no | 0.0 px | 40.8 px | yes | 40.1 px |
| `27_B` | `axis_good` | 7.0 px | 67.0 px | yes | -60.1 px | 67.0 px | yes | -60.1 px |
| `28_A` | `axis_good` | 124.3 px | 52.9 px | yes | 71.4 px | 57.2 px | yes | 67.1 px |
| `28_B` | `axis_good` | 102.6 px | 54.0 px | yes | 48.5 px | 54.0 px | yes | 48.5 px |
| `29_A` | `axis_blocked` | 57.4 px | 57.4 px | no | 0.0 px | 58.6 px | yes | -1.2 px |
| `29_B` | `axis_good` | 25.4 px | 51.8 px | yes | -26.4 px | 51.8 px | yes | -26.4 px |
| `30_A` | `axis_good` | 39.7 px | 67.0 px | yes | -27.3 px | 53.4 px | yes | -13.7 px |
| `30_B` | `axis_good` | 10.0 px | 60.3 px | yes | -50.3 px | 57.5 px | yes | -47.5 px |
| `31_A` | `axis_good` | 47.2 px | 59.2 px | yes | -12.1 px | 59.2 px | yes | -12.1 px |
| `31_B` | `axis_good` | 7.7 px | 47.7 px | yes | -40.0 px | 56.8 px | yes | -49.1 px |
| `32_A` | `axis_blocked` | 72.3 px | 72.3 px | no | 0.0 px | 33.3 px | yes | 39.0 px |
| `32_B` | `axis_blocked` | 70.6 px | 70.6 px | no | 0.0 px | 46.0 px | yes | 24.6 px |
| `36_B` | `axis_good` | 103.0 px | 27.1 px | yes | 75.9 px | 27.1 px | yes | 75.9 px |
| `42_B` | `axis_blocked` | 92.0 px | 92.0 px | no | 0.0 px | 34.0 px | yes | 58.0 px |
| `44_A` | `axis_blocked` | 185.5 px | 185.5 px | no | 0.0 px | 185.5 px | no | 0.0 px |
| `44_B` | `axis_blocked` | 117.6 px | 117.6 px | no | 0.0 px | 8.6 px | yes | 108.9 px |
| `57_A` | `axis_good` | 5.0 px | 12.0 px | yes | -7.1 px | 33.0 px | yes | -28.0 px |
| `61_A` | `axis_blocked` | 70.7 px | 70.7 px | no | 0.0 px | 9.8 px | yes | 61.0 px |
| `61_B` | `axis_good` | 60.0 px | 5.5 px | yes | 54.5 px | 5.5 px | yes | 54.5 px |

## Interpretation

- This is still a diagnostics-only image objective, not production wiring.
- A useful production-shaped signal would increase strict/plausible rows while keeping worsened accepted rows near zero.
- The threshold sweep could not find a non-empty model-axis gate with two-or-fewer worsened rows, so the current hand-tuned ray-start score is not a safe promotion signal.
- If gated model-axis results remain no better than baseline, the next step should be explicit line/junction extraction or learned vertex localization rather than hand-tuned darkness scoring.
