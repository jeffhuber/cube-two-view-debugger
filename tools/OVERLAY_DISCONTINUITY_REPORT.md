# Overlay Cell Discontinuity Probe

Diagnostics-only scoring over the same hybrid overlay slots that received human visual feedback.
High scores indicate cell-internal color variance or half-cell contrast that may be consistent with a multi-face span.

## Summary

- Rows scored: 30 / 30
- Human-bad rows: 28
- Human-good rows: 2
- Mean score, human-bad rows: 10.812
- Mean score, human-good rows: 3.977

## Highest Discontinuity Rows

| Rank | Set | Slot | Human label | Score | Max std | Max half-delta | Source | Failure modes |
|---:|---:|---|---|---:|---:|---:|---|---|
| 1 | 61 | `B:L` | bad | 16.445 | 152.513 | 276.204 | U | `bad_quad`, `bad_rectified`, `wrong_source_face` |
| 2 | 47 | `A:F` | bad | 16.215 | 154.994 | 278.419 | L | `bad_quad`, `bad_rectified`, `wrong_source_face` |
| 3 | 17 | `B:B` | bad | 16.12 | 168.128 | 257.243 | U | `bad_quad`, `bad_rectified` |
| 4 | 47 | `B:B` | bad | 16.0 | 160.96 | 261.036 | U | `bad_rectified`, `wrong_source_face`, `rectification_bad_despite_good_quad` |
| 5 | 17 | `B:L` | bad | 15.651 | 176.228 | 272.981 | B | `bad_quad`, `bad_rectified`, `wrong_source_face` |
| 6 | 49 | `B:B` | bad | 15.599 | 177.987 | 268.36 | B | `bad_rectified`, `rectification_bad_despite_good_quad` |
| 7 | 61 | `B:B` | bad | 14.821 | 155.165 | 199.699 | L | `bad_quad`, `bad_rectified`, `wrong_source_face` |
| 8 | 61 | `A:U` | bad | 13.781 | 136.271 | 192.925 | U | `bad_rectified`, `rectification_bad_despite_good_quad` |
| 9 | 17 | `B:D` | bad | 13.575 | 151.465 | 258.635 | D | `bad_quad`, `bad_rectified` |
| 10 | 47 | `B:L` | bad | 13.455 | 176.473 | 142.313 | B | `bad_quad`, `bad_rectified`, `wrong_source_face` |
| 11 | 49 | `B:L` | bad | 13.396 | 146.537 | 178.157 | F | `bad_quad`, `bad_rectified`, `wrong_source_face` |
| 12 | 47 | `A:U` | bad | 12.167 | 136.085 | 152.065 | U | `bad_rectified`, `rectification_bad_despite_good_quad` |

## Interpretation

- This probe is not a production guard and does not change recognition behavior.
- Treat high-score human-bad rows as candidates for a future multi-face-span ranker.
- Treat high-score human-good rows as potential false positives that a future guard must avoid.
- The current metric is intentionally simple: rectified-cell RGB variance plus half-cell contrast.
