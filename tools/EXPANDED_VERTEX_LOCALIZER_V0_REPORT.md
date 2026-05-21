# Expanded Vertex Localizer V0

Diagnostics-only artifact. This does not alter recognition behavior.

This report uses the completed active-learning vertex+axis labels to test whether the larger label set makes the local visible-trihedral vertex localizer reliable enough to consider wiring. It also fixes the active-label current-model coordinate space before evaluation.

## Label Set

- Labeled rows: 58
- Canonical rows: 28
- Active-learning rows: 30

## Benchmark Summary

| Benchmark | Rows | Base strict/plausible | Oracle strict/plausible | Top-1 strict/plausible | Gated strict/plausible | Accepted | Top-1 +/- | Gated +/- | Mean base/oracle/top-1/gated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `knn_default_radius220` | 58 | 6 / 10 | 44 / 45 | 13 / 27 | 11 / 17 | 38 | 38 / 14 | 25 / 11 | 191.6 px / 73.8 px / 159.9 px / 169.2 px |
| `knn_wide_radius520` | 58 | 6 / 10 | 53 / 53 | 13 / 27 | 10 / 24 | 38 | 37 / 18 | 26 / 10 | 191.6 px / 36.7 px / 168.8 px / 171.9 px |
| `ridge_default_radius220` | 58 | 6 / 10 | 44 / 45 | 4 / 15 | 4 / 12 | 22 | 29 / 11 | 14 / 5 | 191.6 px / 73.8 px / 184.4 px / 185.4 px |

## Coordinate-Space Finding

- The active queue's model hypotheses were produced in processing-image coordinates, while human clicks are in EXIF-correct full-resolution image coordinates.
- This PR scales the active queue `currentModel` vertex and axis vectors into the label coordinate space before using the labels for localizer training/evaluation.
- Without that bridge, active-row baselines were off by roughly 1800 px and the localizer result was meaningless.

## Interpretation

- Best candidate reach: `knn_wide_radius520` reaches 53 / 58 strict rows.
- Best top-1 ranker: `knn_default_radius220` reaches 13 / 58 strict rows.
- Best gated ranker: `knn_default_radius220` reaches 11 / 58 strict rows.
- Production wiring recommendation: `do_not_wire`.
- Reason: Expanded labels and wider candidate reach still produce false-confident worsened rows; ranking/confidence remains unsafe.
- The wide candidate grid shows that candidate reach is mostly recoverable: oracle strict improves from 44/58 to 53/58.
- The ranker still cannot safely select those candidates: top-1 remains 13/58 strict and every non-empty confidence gate still has worsened rows.
- The next useful move is not another scalar gate. It should be a stronger vertex localizer trained on image patches/line junctions or a model objective that scores face-boundary consistency directly.
