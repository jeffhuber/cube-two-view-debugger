# Hull-Label Color Repair Diagnostic

## Purpose

This diagnostic asks whether deterministic color bookkeeping can clean
up hull-label rectified panels before involving an LLM. It uses the
same slot/yaw WCA assignment as the hull-label path, then compares
plain Lab nearest-color classification, center forcing, and exact
9-per-color count repair.

Git head: `83d941e94474f432a1205ae8eab9cab5ed2f02d0`
Generated: `2026-05-26T16:27:31.591774+00:00`

## Headline

The production-like yaw source is `hull_label_center_colors`, because it
does not use ground-truth yaw metadata and is available for every pair.
On the 46-pair manifest corpus, count repair changes the story from
mostly-correct panels to mostly-exact cubes:

| Stage | Assembled | Exact | Legal | Mean stickers | Median hamming | Hamming distribution |
|---|---:|---:|---:|---:|---:|---|
| `canonical` | 46 | 17 | 17 | 51.52 | 2.0 | `{0: 17, 1: 5, 2: 8, 3: 6, 5: 1, 6: 4, 7: 2, 10: 2, 12: 1}` |
| `canonical_center_forced` | 46 | 17 | 17 | 51.57 | 2.0 | `{0: 17, 1: 5, 2: 9, 3: 5, 5: 1, 6: 4, 7: 2, 10: 2, 11: 1}` |
| `canonical_count_repaired` | 46 | 42 | 42 | 53.72 | 0.0 | `{0: 42, 2: 2, 3: 1, 6: 1}` |
| `adaptive` | 46 | 16 | 16 | 51.63 | 1.0 | `{0: 16, 1: 8, 2: 8, 3: 4, 4: 3, 5: 2, 7: 2, 10: 1, 12: 1, 15: 1}` |
| `adaptive_count_repaired` | 46 | 36 | 36 | 52.7 | 0.0 | `{0: 36, 2: 4, 3: 1, 4: 1, 6: 1, 8: 1, 14: 1, 17: 1}` |

`canonical_count_repaired` is the current best deterministic candidate:
42/46 exact/legal, 45/46 within 3 stickers, and only 1 row above 3 stickers.

## Full Summary By Yaw Source

| Yaw source | Method | Assembled | Exact | Legal | Mean stickers | Median hamming |
|---|---|---:|---:|---:|---:|---:|
| `ground_truth_captureYaw` | `adaptive` | 13 | 4 | 4 | 50.23 | 2 |
| `ground_truth_captureYaw` | `adaptive_center_forced` | 13 | 4 | 4 | 50.31 | 2 |
| `ground_truth_captureYaw` | `adaptive_count_repaired` | 13 | 8 | 8 | 50.69 | 0 |
| `ground_truth_captureYaw` | `canonical` | 13 | 2 | 2 | 51 | 2 |
| `ground_truth_captureYaw` | `canonical_center_forced` | 13 | 2 | 2 | 51.15 | 2 |
| `ground_truth_captureYaw` | `canonical_count_repaired` | 13 | 11 | 11 | 53.31 | 0 |
| `hull_label_center_colors` | `adaptive` | 46 | 16 | 16 | 51.63 | 1.0 |
| `hull_label_center_colors` | `adaptive_center_forced` | 46 | 16 | 16 | 51.65 | 1.0 |
| `hull_label_center_colors` | `adaptive_count_repaired` | 46 | 36 | 36 | 52.7 | 0.0 |
| `hull_label_center_colors` | `canonical` | 46 | 17 | 17 | 51.52 | 2.0 |
| `hull_label_center_colors` | `canonical_center_forced` | 46 | 17 | 17 | 51.57 | 2.0 |
| `hull_label_center_colors` | `canonical_count_repaired` | 46 | 42 | 42 | 53.72 | 0.0 |
| `manifest_expectedYaw` | `adaptive` | 7 | 2 | 2 | 51 | 3 |
| `manifest_expectedYaw` | `adaptive_center_forced` | 7 | 2 | 2 | 51 | 3 |
| `manifest_expectedYaw` | `adaptive_count_repaired` | 7 | 5 | 5 | 53.29 | 0 |
| `manifest_expectedYaw` | `canonical` | 7 | 2 | 2 | 51.14 | 2 |
| `manifest_expectedYaw` | `canonical_center_forced` | 7 | 2 | 2 | 51.14 | 2 |
| `manifest_expectedYaw` | `canonical_count_repaired` | 7 | 6 | 6 | 53.71 | 0 |
| `manifest_notes` | `adaptive` | 2 | 0 | 0 | 53 | 1.0 |
| `manifest_notes` | `adaptive_center_forced` | 2 | 0 | 0 | 53 | 1.0 |
| `manifest_notes` | `adaptive_count_repaired` | 2 | 2 | 2 | 54 | 0.0 |
| `manifest_notes` | `canonical` | 2 | 0 | 0 | 51.5 | 2.5 |
| `manifest_notes` | `canonical_center_forced` | 2 | 0 | 0 | 51.5 | 2.5 |
| `manifest_notes` | `canonical_count_repaired` | 2 | 2 | 2 | 54 | 0.0 |
| `white_up_default` | `adaptive` | 24 | 7 | 7 | 39.04 | 2.5 |
| `white_up_default` | `adaptive_center_forced` | 24 | 7 | 7 | 39.38 | 2.5 |
| `white_up_default` | `adaptive_count_repaired` | 24 | 14 | 16 | 40 | 0.0 |
| `white_up_default` | `canonical` | 24 | 8 | 8 | 37.25 | 5.5 |
| `white_up_default` | `canonical_center_forced` | 24 | 8 | 8 | 38.58 | 5.5 |
| `white_up_default` | `canonical_count_repaired` | 24 | 15 | 16 | 40.33 | 0.0 |

## Per-Set Snapshot

| Set | Source | Best method | Best hamming | Canonical | Canonical+count | Adaptive+count | Status |
|---:|---|---|---:|---:|---:|---:|---|
| 12 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 7 | 0 | 0 | `assembled` |
| 14 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 7 | 0 | 0 | `assembled` |
| 15 | `hull_label_center_colors` | `canonical_count_repaired` | 0 | 5 | 0 | 2 | `assembled` |
| 23 | `hull_label_center_colors` | `adaptive` | 0 | 2 | 0 | 0 | `assembled` |
| 24 | `hull_label_center_colors` | `adaptive` | 0 | 0 | 0 | 0 | `assembled` |
| 26 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 0 | 0 | 0 | `assembled` |
| 27 | `hull_label_center_colors` | `adaptive` | 0 | 0 | 0 | 0 | `assembled` |
| 28 | `hull_label_center_colors` | `adaptive` | 0 | 0 | 0 | 0 | `assembled` |
| 29 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 0 | 0 | 0 | `assembled` |
| 31 | `hull_label_center_colors` | `canonical` | 0 | 0 | 0 | 8 | `assembled` |
| 32 | `hull_label_center_colors` | `canonical_count_repaired` | 0 | 3 | 0 | 2 | `assembled` |
| 36 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 0 | 0 | 0 | `assembled` |
| 37 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 2 | 0 | 0 | `assembled` |
| 42 | `hull_label_center_colors` | `adaptive` | 0 | 0 | 0 | 0 | `assembled` |
| 44 | `hull_label_center_colors` | `adaptive` | 0 | 2 | 0 | 0 | `assembled` |
| 20 | `hull_label_center_colors` | `adaptive` | 0 | 1 | 0 | 0 | `assembled` |
| 38 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 0 | 0 | 0 | `assembled` |
| 40 | `hull_label_center_colors` | `adaptive` | 0 | 0 | 0 | 0 | `assembled` |
| 41 | `hull_label_center_colors` | `adaptive` | 0 | 0 | 0 | 0 | `assembled` |
| 43 | `hull_label_center_colors` | `adaptive` | 0 | 0 | 0 | 0 | `assembled` |
| 45 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 2 | 0 | 0 | `assembled` |
| 17 | `hull_label_center_colors` | `canonical` | 0 | 0 | 0 | 6 | `assembled` |
| 21 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 3 | 0 | 0 | `assembled` |
| 22 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 3 | 0 | 0 | `assembled` |
| 25 | `hull_label_center_colors` | `adaptive` | 0 | 0 | 0 | 0 | `assembled` |
| 30 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 0 | 0 | 0 | `assembled` |
| 39 | `hull_label_center_colors` | `adaptive` | 0 | 1 | 0 | 0 | `assembled` |
| 46 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 2 | 0 | 0 | `assembled` |
| 47 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 1 | 0 | 0 | `assembled` |
| 48 | `hull_label_center_colors` | `adaptive` | 0 | 0 | 0 | 0 | `assembled` |
| 49 | `hull_label_center_colors` | `canonical_count_repaired` | 0 | 3 | 0 | 2 | `assembled` |
| 57 | `hull_label_center_colors` | `adaptive` | 0 | 1 | 0 | 0 | `assembled` |
| 58 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 1 | 0 | 0 | `assembled` |
| 61 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 0 | 0 | 0 | `assembled` |
| 62 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 2 | 0 | 0 | `assembled` |
| 63 | `hull_label_center_colors` | `adaptive` | 0 | 2 | 0 | 0 | `assembled` |
| 64 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 3 | 0 | 0 | `assembled` |
| 65 | `hull_label_center_colors` | `canonical_count_repaired` | 0 | 6 | 0 | 4 | `assembled` |
| 66 | `hull_label_center_colors` | `adaptive` | 0 | 3 | 0 | 0 | `assembled` |
| 67 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 2 | 0 | 0 | `assembled` |
| 68 | `hull_label_center_colors` | `canonical_count_repaired` | 2 | 6 | 2 | 3 | `assembled` |
| 69 | `hull_label_center_colors` | `canonical_count_repaired` | 3 | 6 | 3 | 14 | `assembled` |
| 70 | `hull_label_center_colors` | `canonical_count_repaired` | 6 | 12 | 6 | 17 | `assembled` |
| 71 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 6 | 0 | 0 | `assembled` |
| 72 | `hull_label_center_colors` | `adaptive` | 2 | 10 | 2 | 2 | `assembled` |
| 73 | `hull_label_center_colors` | `adaptive_count_repaired` | 0 | 10 | 0 | 0 | `assembled` |

## Current Run Notes

- The raw `canonical` classifier is already close: 17/46 exact, 36/46
  within 3 stickers. The dominant issue is duplicated/missing color
  counts, not WCA face assignment.
- Greedy count repair is a large deterministic jump: 42/46 exact/legal
  with the production-like yaw source. This supersedes the older
  20/46 exact headline for raw hull-label `prefer` panels.
- Canonical Lab count repair beats the adaptive-palette count repair in
  this run (42/46 exact versus 36/46). Adaptive palettes should stay
  diagnostic or gated; do not blindly prefer them.
- Sets 69-73 remain useful stress cases: Set 69 is 3 stickers off after
  canonical count repair, Set 70 is 6 off, Set 72 is 2 off, and Sets
  71/73 are exact.
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
