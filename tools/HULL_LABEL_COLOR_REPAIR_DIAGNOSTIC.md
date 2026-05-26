# Hull-Label Color Repair Diagnostic

## Purpose

This diagnostic asks whether deterministic color bookkeeping can clean
up hull-label rectified panels before involving an LLM. It uses the
same slot/yaw WCA assignment as the hull-label path, then compares
plain Lab nearest-color classification, center forcing, and exact
9-per-face count repair.

Git head: `8f8cf23d3a063e6cfc2a1d5072a5314716f8ac0d`
Generated: `2026-05-26T04:36:43.058103+00:00`

## Summary

| Yaw source | Method | Assembled | Exact | Legal | Mean stickers | Median hamming |
|---|---|---:|---:|---:|---:|---:|
| `ground_truth_captureYaw` | `canonical` | 1 | 0 | 0 | 48 | 6 |
| `ground_truth_captureYaw` | `canonical_center_forced` | 1 | 0 | 0 | 48 | 6 |
| `ground_truth_captureYaw` | `canonical_count_repaired` | 1 | 0 | 0 | 51 | 3 |
| `ground_truth_captureYaw` | `adaptive` | 1 | 0 | 0 | 42 | 12 |
| `ground_truth_captureYaw` | `adaptive_center_forced` | 1 | 0 | 0 | 42 | 12 |
| `ground_truth_captureYaw` | `adaptive_count_repaired` | 1 | 0 | 0 | 40 | 14 |
| `hull_label_center_colors` | `canonical` | 3 | 0 | 0 | 45 | 10 |
| `hull_label_center_colors` | `canonical_center_forced` | 3 | 0 | 0 | 45 | 10 |
| `hull_label_center_colors` | `canonical_count_repaired` | 3 | 0 | 0 | 51 | 3 |
| `hull_label_center_colors` | `adaptive` | 3 | 0 | 0 | 48 | 3 |
| `hull_label_center_colors` | `adaptive_center_forced` | 3 | 0 | 0 | 48 | 3 |
| `hull_label_center_colors` | `adaptive_count_repaired` | 3 | 1 | 1 | 48.67 | 2 |
| `white_up_default` | `canonical` | 3 | 0 | 0 | 45 | 10 |
| `white_up_default` | `canonical_center_forced` | 3 | 0 | 0 | 45 | 10 |
| `white_up_default` | `canonical_count_repaired` | 3 | 1 | 1 | 52 | 2 |
| `white_up_default` | `adaptive` | 3 | 0 | 0 | 50.67 | 3 |
| `white_up_default` | `adaptive_center_forced` | 3 | 0 | 0 | 50.67 | 3 |
| `white_up_default` | `adaptive_count_repaired` | 3 | 2 | 2 | 53.33 | 0 |

## Sets 69-73

| Set | Preferred yaw source | Best method | Best hamming | Canonical hamming | Adaptive+count hamming | Status |
|---:|---|---|---:|---:|---:|---|
| 69 | `ground_truth_captureYaw` | `canonical_count_repaired` | 3 | 6 | 14 | `assembled` |
| 70 | `ground_truth_captureYaw` | `n/a` | n/a | None | None | `sample_failed` |
| 71 | `white_up_default` | `adaptive_count_repaired` | 0 | 6 | 0 | `assembled` |
| 72 | `white_up_default` | `adaptive_count_repaired` | 2 | 11 | 2 | `assembled` |
| 73 | `white_up_default` | `adaptive_count_repaired` | 0 | 10 | 0 | `assembled` |

## Current Run Notes

- Sets 71 and 73 become exact legal cubes after the best deterministic
  count-repair variant in this run; Set 72 is left with two sticker
  errors. These are the cases where the rectified panels are visually
  good and the remaining problem is mostly duplicated/missing color
  reads.
- Set 69 improves under canonical count repair but adaptive-center repair
  is worse, which is useful evidence that adaptive palettes should be
  gated rather than blindly preferred.
- Set 70 does not reach color repair because its A-side hull-label model
  is rejected by the existing acceptance gates. That is the right failure
  class: do not count-repair bad geometry.
- The muddy side-face panels in these rows are photometric failures,
  not rembg failures. rembg supplies the silhouette mask; the rectified
  panels sample the original RGB image. Grazing side faces stretch shadow,
  black bevels, reflections, and sticker texture into a square, so humans
  can still read the colors while static Lab distance struggles.
- Set 70 should be inspected with yaw-aware panel labels. Its current
  yaw=2 Image B slots map to D/F/R; older no-yaw D/L/B contact sheets
  are useful visually but misleading for face identity.

## Interpretation

- `center_forced` is a cheap sanity step: the center sticker of each WCA
  face is known once slot/yaw assignment succeeds, so a center-color
  miss should not be allowed to poison the state.
- `count_repaired` is deliberately not a legality solver. It only enforces
  the physical requirement that each WCA face color appears exactly nine
  times. This catches duplicated/missing color reads while preserving the
  sampled geometry.
- The adaptive palette uses the six known center samples as anchors. It is
  still deterministic and local to the two input photos; no GT colors or
  LLM output are used.
- Rows that remain high-hamming after adaptive count repair are likely
  geometry/panel-quality failures rather than cube-count failures. Those
  should be handled by hull-label acceptance gates or a visual repair UI.
