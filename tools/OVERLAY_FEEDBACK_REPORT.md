# Overlay Feedback Visual Labels

Diagnostics/data-only artifact generated from the human overlay feedback workbook.
It records whether each hybrid overlay slot's quad and rectified face looked good or bad.

## Summary

- Sets reviewed: 5
- Slots reviewed: 30
- Slots with at least one bad signal: 28

## Failure Modes

| Failure mode | Count |
|---|---:|
| `bad_quad` | 22 |
| `bad_rectified` | 22 |
| `rectification_bad_despite_good_quad` | 6 |
| `rectification_survives_bad_quad` | 6 |
| `wrong_source_face` | 16 |

## Ranked Slots

| Slot | Bad signals | Modes |
|---|---:|---|
| `B:L` | 15 | `bad_quad`=5, `bad_rectified`=5, `wrong_source_face`=5 |
| `A:F` | 14 | `bad_quad`=5, `bad_rectified`=5, `wrong_source_face`=4 |
| `B:B` | 13 | `bad_quad`=2, `bad_rectified`=5, `rectification_bad_despite_good_quad`=3, `wrong_source_face`=3 |
| `A:R` | 12 | `bad_quad`=4, `bad_rectified`=1, `rectification_survives_bad_quad`=3, `wrong_source_face`=4 |
| `B:D` | 10 | `bad_quad`=5, `bad_rectified`=2, `rectification_survives_bad_quad`=3 |
| `A:U` | 8 | `bad_quad`=1, `bad_rectified`=4, `rectification_bad_despite_good_quad`=3 |

## Per-Set Labels

| Set | Bad slots | Slot labels |
|---|---:|---|
| 17 | 4 | `A:F` quad=bad rect=bad src=B (bad_quad, bad_rectified, wrong_source_face)<br>`B:D` quad=bad rect=bad src=D (bad_quad, bad_rectified)<br>`B:L` quad=bad rect=bad src=B (bad_quad, bad_rectified, wrong_source_face)<br>`B:B` quad=bad rect=bad src=B (bad_quad, bad_rectified) |
| 21 | 6 | `A:U` quad=good rect=bad src=U (bad_rectified, rectification_bad_despite_good_quad)<br>`A:R` quad=bad rect=good src=B (bad_quad, wrong_source_face, rectification_survives_bad_quad)<br>`A:F` quad=bad rect=bad src=D (bad_quad, bad_rectified, wrong_source_face)<br>`B:D` quad=bad rect=good src=D (bad_quad, rectification_survives_bad_quad)<br>`B:L` quad=bad rect=bad src=F (bad_quad, bad_rectified, wrong_source_face)<br>`B:B` quad=good rect=bad src=R (bad_rectified, wrong_source_face, rectification_bad_despite_good_quad) |
| 47 | 6 | `A:U` quad=good rect=bad src=U (bad_rectified, rectification_bad_despite_good_quad)<br>`A:R` quad=bad rect=good src=F (bad_quad, wrong_source_face, rectification_survives_bad_quad)<br>`A:F` quad=bad rect=bad src=L (bad_quad, bad_rectified, wrong_source_face)<br>`B:D` quad=bad rect=good src=D (bad_quad, rectification_survives_bad_quad)<br>`B:L` quad=bad rect=bad src=B (bad_quad, bad_rectified, wrong_source_face)<br>`B:B` quad=good rect=bad src=U (bad_rectified, wrong_source_face, rectification_bad_despite_good_quad) |
| 49 | 6 | `A:U` quad=bad rect=bad src=U (bad_quad, bad_rectified)<br>`A:R` quad=bad rect=bad src=B (bad_quad, bad_rectified, wrong_source_face)<br>`A:F` quad=bad rect=bad src=F (bad_quad, bad_rectified)<br>`B:D` quad=bad rect=good src=D (bad_quad, rectification_survives_bad_quad)<br>`B:L` quad=bad rect=bad src=F (bad_quad, bad_rectified, wrong_source_face)<br>`B:B` quad=good rect=bad src=B (bad_rectified, rectification_bad_despite_good_quad) |
| 61 | 6 | `A:U` quad=good rect=bad src=U (bad_rectified, rectification_bad_despite_good_quad)<br>`A:R` quad=bad rect=good src=F (bad_quad, wrong_source_face, rectification_survives_bad_quad)<br>`A:F` quad=bad rect=bad src=B (bad_quad, bad_rectified, wrong_source_face)<br>`B:D` quad=bad rect=bad src=D (bad_quad, bad_rectified)<br>`B:L` quad=bad rect=bad src=U (bad_quad, bad_rectified, wrong_source_face)<br>`B:B` quad=bad rect=bad src=L (bad_quad, bad_rectified, wrong_source_face) |

## Notes

- `wrong_source_face` means the rectified face was sourced from a different detected face than the slot label.
- `rectification_bad_despite_good_quad` is especially useful for sampling/rectification debugging.
- This report is not a recognizer behavior change; it is supervision for future diagnostics.
