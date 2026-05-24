# Axis-correctness diagnostic

For each (row, hypothesis), measures how close the production fit's 3 axis vectors are to the oracle's ground-truth single-axis hex-corner directions (per `ONE_EDGE_CORNERS_BY_SIDE`). Cross-references with vertex error and partial visual quality labels from the rectification-quality follow-up dig.

## Source

- Tool: `tools/measure_axis_correctness.py`
- Truth: `tests/fixtures/full_corner_ground_truth.json`
- Manifest: `tests/fixtures/corpus_manifest.json`
- Max image dim: `1600`
- Run selection: single deterministic run per row/hypothesis

## Per-row metrics

| Row | Hypothesis | flip? | vertex err px | total axis misfit ° | per-axis errors ° | predicted axis lengths px | gt axis lengths px | visual |
|---|---|:-:|---:|---:|---|---|---|---|
| `20_A` | corr_true | N | 44.2 | 177.4 | [63.6, 56.3, 57.5] | [408.9, 411.4, 410.8] | [502.2, 590.4, 547.0] | broken |
| `20_A` | corr_false | N | 44.2 | 177.4 | [63.6, 56.3, 57.5] | [408.9, 411.4, 410.8] | [502.2, 590.4, 547.0] | marginal |
| `20_B` | corr_true | N | 77.6 | 179.6 | [55.3, 65.1, 59.2] | [418.1, 417.2, 419.0] | [504.3, 588.3, 591.8] | broken |
| `20_B` | corr_false | N | 77.6 | 179.6 | [55.3, 65.1, 59.2] | [418.1, 417.2, 419.0] | [504.3, 588.3, 591.8] | unknown |
| `38_A` | corr_true | N | 79.8 | 179.4 | [53.1, 63.4, 63.0] | [399.2, 394.7, 399.9] | [579.2, 475.0, 491.4] | unknown |
| `38_A` | corr_false | N | 79.8 | 179.4 | [53.1, 63.4, 63.0] | [399.2, 394.7, 399.9] | [579.2, 475.0, 491.4] | unknown |
| `38_B` | corr_true | Y | 83.2 | 8.9 | [2.1, 1.6, 5.2] | [371.3, 338.0, 368.0] | [423.4, 493.7, 454.9] | unknown |
| `38_B` | corr_false | N | 86.0 | 175.0 | [58.1, 61.9, 54.9] | [365.5, 360.7, 351.1] | [423.4, 493.7, 454.9] | unknown |
| `40_A` | corr_true | N | 78.6 | 178.8 | [65.7, 55.4, 57.7] | [473.5, 482.8, 471.8] | [515.7, 577.6, 437.1] | unknown |
| `40_A` | corr_false | N | 78.6 | 178.8 | [65.7, 55.4, 57.7] | [473.5, 482.8, 471.8] | [515.7, 577.6, 437.1] | unknown |
| `40_B` | corr_true | N | 184.9 | 178.2 | [67.1, 60.5, 50.6] | [431.3, 402.6, 411.2] | [605.9, 511.4, 465.2] | unknown |
| `40_B` | corr_false | N | 184.9 | 178.2 | [67.1, 60.5, 50.6] | [431.3, 402.6, 411.2] | [605.9, 511.4, 465.2] | unknown |
| `41_A` | corr_true | N | 101.8 | 12.9 | [6.2, 2.4, 4.4] | [475.3, 470.3, 464.7] | [510.8, 568.9, 434.3] | marginal |
| `41_A` | corr_false | N | 101.8 | 12.9 | [6.2, 2.4, 4.4] | [475.3, 470.3, 464.7] | [510.8, 568.9, 434.3] | clean |
| `41_B` | corr_true | N | 37.3 | 179.8 | [56.3, 63.7, 59.8] | [386.8, 401.5, 395.2] | [535.8, 453.4, 479.2] | unknown |
| `41_B` | corr_false | N | 37.3 | 179.8 | [56.3, 63.7, 59.8] | [386.8, 401.5, 395.2] | [535.8, 453.4, 479.2] | unknown |
| `43_A` | corr_true | Y | 27.0 | 179.5 | [64.1, 66.0, 49.3] | [384.6, 530.0, 526.5] | [490.1, 523.9, 455.1] | unknown |
| `43_A` | corr_false | N | 27.0 | 5.7 | [2.6, 1.2, 1.9] | [456.1, 453.7, 522.5] | [490.1, 523.9, 455.1] | unknown |
| `43_B` | corr_true | N | 78.1 | 177.9 | [65.2, 53.4, 59.2] | [395.4, 383.2, 393.2] | [458.0, 499.1, 560.9] | unknown |
| `43_B` | corr_false | N | 78.1 | 177.9 | [65.2, 53.4, 59.2] | [395.4, 383.2, 393.2] | [458.0, 499.1, 560.9] | unknown |
| `45_A` | corr_true | Y | 41.5 | 178.1 | [50.5, 54.3, 73.4] | [477.6, 597.9, 429.2] | [555.0, 431.3, 536.9] | unknown |
| `45_A` | corr_false | N | 41.6 | 5.9 | [1.6, 1.3, 3.0] | [539.1, 424.0, 517.5] | [555.0, 431.3, 536.9] | unknown |
| `45_B` | corr_true | Y | 116.0 | 24.8 | [6.3, 10.7, 7.8] | [443.9, 407.8, 370.4] | [431.3, 580.1, 612.1] | broken |
| `45_B` | corr_false | N | 114.7 | 176.6 | [69.3, 51.9, 55.5] | [414.3, 406.0, 402.0] | [431.3, 580.1, 612.1] | broken |

## Cross-reference: vertex error vs total axis misfit, by visual quality bucket

Only rows with a visual quality label are shown. Bucket counts: clean=1, marginal=2, broken=4, unknown=17

| Visual | Count | Median vertex err px | Median axis misfit ° |
|---|---:|---:|---:|
| clean | 1 | 101.8 | 12.9 |
| marginal | 2 | 73.0 | 95.2 |
| broken | 4 | 96.2 | 177.0 |

## Interpretation guide

- If `broken` rows have notably higher median axis misfit than `clean` rows (separation > say 20°), axis correctness is a useful predictor.
- If `broken` rows have similar axis misfit to `clean` rows, rectification breakage has another cause (e.g. axis-length error / non-Procrustes scale issues) not captured by angle.
- Low angle misfit is necessary but not sufficient for clean rectification; compare predicted vs GT axis lengths too.
- Compare to vertex error: which is the cleaner predictor?
