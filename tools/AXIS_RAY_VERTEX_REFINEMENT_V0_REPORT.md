# Axis-Ray Vertex Refinement V0

Diagnostics-only artifact. This does not alter recognition behavior.

This probe searches for a better visible-trihedral vertex by holding the three outgoing axis directions fixed and scoring dark-line support along those rays.

## Summary

- Rows: 28
- Evaluated rows: 28
- Axis-good rows: 17
- Axis-blocked rows: 11
- Baseline strict/plausible: 6 / 9
- Model-axis refined strict/plausible: 4 / 9
- Human-axis oracle strict/plausible: 4 / 11
- Axis-good strict baseline/model-axis/oracle: 5 / 2 / 2
- Mean vertex error baseline/model-axis/oracle: 67.0 px / 65.8 px / 48.7 px
- Median vertex error baseline/model-axis/oracle: 70.7 px / 61.2 px / 52.4 px
- Model-axis improved/worsened rows by >5px: 13 / 13
- Human-axis oracle improved/worsened rows by >5px: 15 / 9

## Rows

| Row | Axis category | Baseline | Model-axis refined | Delta | Human-axis oracle | Oracle delta |
|---|---|---:|---:|---:|---:|---:|
| `12_B` | `axis_blocked` | 26.9 px | 90.1 px | -63.2 px | 47.0 px | -20.1 px |
| `14_B` | `axis_blocked` | 53.9 px | 103.7 px | -49.8 px | 56.6 px | -2.7 px |
| `15_A` | `axis_good` | 63.5 px | 39.7 px | 23.8 px | 52.2 px | 11.3 px |
| `15_B` | `axis_good` | 72.0 px | 54.2 px | 17.8 px | 56.0 px | 16.0 px |
| `17_B` | `axis_good` | 107.2 px | 62.3 px | 44.9 px | 62.3 px | 44.9 px |
| `21_A` | `axis_good` | 38.7 px | 63.4 px | -24.7 px | 56.5 px | -17.7 px |
| `21_B` | `axis_good` | 75.7 px | 15.9 px | 59.8 px | 15.9 px | 59.8 px |
| `24_A` | `axis_blocked` | 87.6 px | 69.6 px | 17.9 px | 52.5 px | 35.0 px |
| `26_A` | `axis_good` | 72.5 px | 48.9 px | 23.7 px | 48.0 px | 24.6 px |
| `26_B` | `axis_blocked` | 80.9 px | 182.7 px | -101.8 px | 40.8 px | 40.1 px |
| `27_B` | `axis_good` | 7.0 px | 67.0 px | -60.1 px | 67.0 px | -60.1 px |
| `28_A` | `axis_good` | 124.3 px | 40.0 px | 84.3 px | 66.0 px | 58.3 px |
| `28_B` | `axis_good` | 102.6 px | 54.0 px | 48.5 px | 54.0 px | 48.5 px |
| `29_A` | `axis_blocked` | 57.4 px | 25.7 px | 31.7 px | 58.6 px | -1.2 px |
| `29_B` | `axis_good` | 25.4 px | 51.8 px | -26.4 px | 51.8 px | -26.4 px |
| `30_A` | `axis_good` | 39.7 px | 67.0 px | -27.3 px | 53.4 px | -13.7 px |
| `30_B` | `axis_good` | 10.0 px | 65.2 px | -55.2 px | 57.5 px | -47.5 px |
| `31_A` | `axis_good` | 47.2 px | 59.2 px | -12.1 px | 59.2 px | -12.1 px |
| `31_B` | `axis_good` | 7.7 px | 47.7 px | -40.0 px | 47.7 px | -40.0 px |
| `32_A` | `axis_blocked` | 72.3 px | 143.0 px | -70.7 px | 50.7 px | 21.6 px |
| `32_B` | `axis_blocked` | 70.6 px | 103.1 px | -32.5 px | 49.9 px | 20.7 px |
| `36_B` | `axis_good` | 103.0 px | 50.0 px | 53.0 px | 38.0 px | 64.9 px |
| `42_B` | `axis_blocked` | 92.0 px | 77.9 px | 14.0 px | 30.1 px | 61.8 px |
| `44_A` | `axis_blocked` | 185.5 px | 6.0 px | 179.5 px | 26.0 px | 159.5 px |
| `44_B` | `axis_blocked` | 117.6 px | 111.8 px | 5.8 px | 8.6 px | 108.9 px |
| `57_A` | `axis_good` | 5.0 px | 5.0 px | 0.0 px | 25.3 px | -20.4 px |
| `61_A` | `axis_blocked` | 70.7 px | 76.8 px | -6.1 px | 70.7 px | 0.0 px |
| `61_B` | `axis_good` | 60.0 px | 60.0 px | 0.0 px | 60.0 px | 0.0 px |

## Interpretation

- This is an image-objective probe, not production wiring.
- Current readout: model-axis refinement is mixed and not safe. It improves many rows but also worsens many rows, and strict rows do not improve over baseline.
- The human-axis oracle improves mean/median error, but still does not produce a stable strict hit-rate. That points to a weak image objective, not only axis-family selection.
- If model-axis refinement improves the axis-good subset without creating many worsened rows, the next step is to refine the scoring objective and add overlays for manual inspection.
- If the human-axis oracle improves but model-axis refinement does not, axis-family selection remains the blocker.
- If neither improves, the dark-line objective is not enough and we should pivot to stronger junction/line extraction.
