# Hull-Label Color Repair Diagnostic

## Purpose

This diagnostic asks whether deterministic color bookkeeping can clean
up hull-label rectified panels before involving an LLM. It uses the
same slot/yaw WCA assignment as the hull-label path, then compares
plain Lab nearest-color classification, center forcing, exact
9-per-color count repair, and guarded cubie-legality repair.

Git head: `f699ebf11f54c36bac702db2f52a3089a7fedcc5`
Generated: `2026-05-26T18:52:49.694503+00:00`

## Headline

The production-like yaw source is `hull_label_center_colors`, because it
does not use ground-truth yaw metadata. It is available for most pairs;
rows without an accepted center-yaw inference are still shown with their
metadata/default yaw fallback in the per-set table.
On the 46-pair manifest corpus, count repair changes the story from
mostly-correct panels to mostly-exact cubes:

| Stage | Assembled | Exact | Legal | Mean stickers | Median hamming | Hamming distribution |
|---|---:|---:|---:|---:|---:|---|
| `canonical` | 45 | 18 | 18 | 51.78 | 2 | `{0: 18, 1: 4, 2: 10, 3: 3, 4: 2, 5: 2, 6: 1, 7: 2, 9: 1, 10: 2}` |
| `canonical_center_forced` | 45 | 18 | 18 | 51.82 | 1 | `{0: 18, 1: 5, 2: 9, 3: 3, 4: 2, 5: 2, 6: 1, 7: 2, 9: 2, 10: 1}` |
| `canonical_count_repaired` | 45 | 42 | 42 | 53.8 | 0 | `{0: 42, 2: 1, 3: 1, 4: 1}` |
| `conservative_legal_repaired` | 45 | 43 | 43 | 54 | 0 | `{0: 43}` |
| `guarded_broad_legal_repaired` | 45 | 44 | 44 | 54 | 0.0 | `{0: 44}` |
| `adaptive` | 45 | 19 | 19 | 51.82 | 1 | `{0: 19, 1: 5, 2: 6, 3: 7, 4: 2, 5: 2, 7: 1, 11: 1, 12: 2}` |
| `adaptive_count_repaired` | 45 | 36 | 36 | 52.87 | 0 | `{0: 36, 2: 4, 4: 1, 6: 1, 8: 1, 11: 1, 14: 1}` |

`canonical_count_repaired` is the stable deterministic baseline:
42/45 exact/legal, 44/45 within 3 stickers, and only 1 row above 3 stickers.
The payload's recommended-method selector is now 44/45 exact with hamming distribution `{0: 44, 4: 1}`.

## Full Summary By Yaw Source

| Yaw source | Method | Assembled | Exact | Legal | Mean stickers | Median hamming |
|---|---|---:|---:|---:|---:|---:|
| `ground_truth_captureYaw` | `canonical` | 13 | 3 | 3 | 51.54 | 2 |
| `ground_truth_captureYaw` | `canonical_center_forced` | 13 | 3 | 3 | 51.62 | 2 |
| `ground_truth_captureYaw` | `canonical_count_repaired` | 13 | 11 | 11 | 53.62 | 0 |
| `ground_truth_captureYaw` | `conservative_legal_repaired` | 13 | 12 | 12 | 54 | 0.0 |
| `ground_truth_captureYaw` | `guarded_broad_legal_repaired` | 13 | 13 | 13 | 54 | 0 |
| `ground_truth_captureYaw` | `adaptive` | 13 | 3 | 3 | 50.54 | 3 |
| `ground_truth_captureYaw` | `adaptive_center_forced` | 13 | 3 | 3 | 50.54 | 3 |
| `ground_truth_captureYaw` | `adaptive_count_repaired` | 13 | 7 | 7 | 51.69 | 0 |
| `hull_label_center_colors` | `canonical` | 45 | 18 | 18 | 51.78 | 2 |
| `hull_label_center_colors` | `canonical_center_forced` | 45 | 18 | 18 | 51.82 | 1 |
| `hull_label_center_colors` | `canonical_count_repaired` | 45 | 42 | 42 | 53.8 | 0 |
| `hull_label_center_colors` | `conservative_legal_repaired` | 45 | 43 | 43 | 54 | 0 |
| `hull_label_center_colors` | `guarded_broad_legal_repaired` | 45 | 44 | 44 | 54 | 0.0 |
| `hull_label_center_colors` | `adaptive` | 45 | 19 | 19 | 51.82 | 1 |
| `hull_label_center_colors` | `adaptive_center_forced` | 45 | 19 | 19 | 51.84 | 1 |
| `hull_label_center_colors` | `adaptive_count_repaired` | 45 | 36 | 36 | 52.87 | 0 |
| `manifest_expectedYaw` | `canonical` | 7 | 2 | 2 | 51.29 | 2 |
| `manifest_expectedYaw` | `canonical_center_forced` | 7 | 2 | 2 | 51.29 | 2 |
| `manifest_expectedYaw` | `canonical_count_repaired` | 7 | 7 | 7 | 54 | 0 |
| `manifest_expectedYaw` | `conservative_legal_repaired` | 7 | 7 | 7 | 54 | 0 |
| `manifest_expectedYaw` | `guarded_broad_legal_repaired` | 7 | 7 | 7 | 54 | 0 |
| `manifest_expectedYaw` | `adaptive` | 7 | 2 | 2 | 51.29 | 3 |
| `manifest_expectedYaw` | `adaptive_center_forced` | 7 | 2 | 2 | 51.29 | 3 |
| `manifest_expectedYaw` | `adaptive_count_repaired` | 7 | 6 | 6 | 53.71 | 0 |
| `manifest_notes` | `canonical` | 2 | 0 | 0 | 51.5 | 2.5 |
| `manifest_notes` | `canonical_center_forced` | 2 | 0 | 0 | 51.5 | 2.5 |
| `manifest_notes` | `canonical_count_repaired` | 2 | 2 | 2 | 54 | 0.0 |
| `manifest_notes` | `conservative_legal_repaired` | 2 | 2 | 2 | 54 | 0.0 |
| `manifest_notes` | `guarded_broad_legal_repaired` | 2 | 2 | 2 | 54 | 0.0 |
| `manifest_notes` | `adaptive` | 2 | 1 | 1 | 53.5 | 0.5 |
| `manifest_notes` | `adaptive_center_forced` | 2 | 1 | 1 | 53.5 | 0.5 |
| `manifest_notes` | `adaptive_count_repaired` | 2 | 2 | 2 | 54 | 0.0 |
| `white_up_default` | `canonical` | 24 | 8 | 8 | 37.25 | 5.0 |
| `white_up_default` | `canonical_center_forced` | 24 | 8 | 8 | 38.62 | 5.0 |
| `white_up_default` | `canonical_count_repaired` | 24 | 15 | 16 | 40.25 | 0.0 |
| `white_up_default` | `conservative_legal_repaired` | 24 | 15 | 23 | 39.78 | 0 |
| `white_up_default` | `guarded_broad_legal_repaired` | 24 | 15 | 23 | 39.74 | 0 |
| `white_up_default` | `adaptive` | 24 | 9 | 9 | 38.92 | 2.0 |
| `white_up_default` | `adaptive_center_forced` | 24 | 9 | 9 | 39.29 | 2.0 |
| `white_up_default` | `adaptive_count_repaired` | 24 | 15 | 17 | 39.58 | 0.0 |

## Per-Set Snapshot

| Set | Source | Recommended | Best safe method | Best hamming | Canonical | Canonical+count | Guarded legal | Adaptive+count | Status |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 12 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 7 | 0 | 0 | 0 | `assembled` |
| 14 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 4 | 10 | 4 | None | 11 | `assembled` |
| 15 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 4 | 0 | 0 | 0 | `assembled` |
| 23 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 2 | 0 | 0 | 0 | `assembled` |
| 24 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 26 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 27 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 28 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 29 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 31 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical` | 0 | 0 | 0 | 0 | 8 | `assembled` |
| 32 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 3 | 0 | 0 | 2 | `assembled` |
| 36 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 37 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 2 | 0 | 0 | 0 | `assembled` |
| 42 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 44 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 2 | 0 | 0 | 0 | `assembled` |
| 20 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 1 | 0 | 0 | 0 | `assembled` |
| 38 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 40 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 41 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 43 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 45 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 2 | 0 | 0 | 0 | `assembled` |
| 17 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical` | 0 | 0 | 0 | 0 | 6 | `assembled` |
| 21 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 2 | 0 | 0 | 0 | `assembled` |
| 22 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 3 | 0 | 0 | 0 | `assembled` |
| 25 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 30 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 39 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 1 | 0 | 0 | 0 | `assembled` |
| 46 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 2 | 0 | 0 | 0 | `assembled` |
| 47 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 1 | 0 | 0 | 0 | `assembled` |
| 48 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 49 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 2 | 0 | 0 | 2 | `assembled` |
| 57 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 1 | 0 | 0 | 2 | `assembled` |
| 58 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 61 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | 0 | 0 | `assembled` |
| 62 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 2 | 0 | 0 | 2 | `assembled` |
| 63 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 2 | 0 | 0 | 0 | `assembled` |
| 64 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 3 | 0 | 0 | 0 | `assembled` |
| 65 | `hull_label_center_colors` | `guarded_broad_legal_repaired` | `guarded_broad_legal_repaired` | 0 | 7 | 2 | 0 | 4 | `assembled` |
| 66 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 4 | 0 | 0 | 0 | `assembled` |
| 67 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 2 | 0 | 0 | 0 | `assembled` |
| 68 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 5 | 0 | 0 | 0 | `assembled` |
| 69 | `hull_label_center_colors` | `conservative_legal_repaired` | `conservative_legal_repaired` | 0 | 6 | 3 | 0 | 14 | `assembled` |
| 70 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 5 | 0 | 0 | 0 | `assembled` |
| 71 | `white_up_default` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 6 | 0 | 0 | 0 | `assembled` |
| 72 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 9 | 0 | 0 | 0 | `assembled` |
| 73 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 10 | 0 | 0 | 0 | `assembled` |

## Current Run Notes

- The raw `canonical` classifier is already close: 18/45 exact, 35/45
  within 3 stickers. The dominant issue is duplicated/missing color
  counts, not WCA face assignment.
- Greedy count repair is a large deterministic jump: 42/45 exact/legal
  with the production-like yaw source. This supersedes the older
  20/46 exact headline for raw hull-label `prefer` panels.
- Guarded cubie-legality repair is now part of the color-repair payload:
  it is 44/45 exact here and exposes
  conservative and guarded-broad legal candidates, while the
  ungated broad legal candidate remains diagnostic-only.
- Canonical Lab count repair beats the adaptive-palette count repair in
  this run (42/45 exact versus 36/45). Adaptive palettes should stay
  diagnostic or gated; do not blindly prefer them.
- Sets 69-73 remain useful stress cases. With the Fixer-equivalent 1600px
  geometry path, Sets 70-73 are exact after count repair; Set 69 still
  needs the conservative legal layer to resolve a 3-sticker count-repair
  ambiguity. This replaces the older lower-res diagnostic read where
  Sets 70 and 72 looked like remaining tails.
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
- `conservative_legal_repaired` and `guarded_broad_legal_repaired` add the
  next constraint layer: cubie legality. The guarded broad method uses
  the same no-ground-truth cost/change gate as the legal-repair diagnostic;
  the raw `broad_legal_repaired` method is emitted only for traceability.
- The adaptive palette uses the six known center samples as anchors. It is
  still deterministic and local to the two input photos; no GT colors or
  LLM output are used.
- Rows that remain high-hamming after adaptive count repair are likely
  geometry/panel-quality failures rather than cube-count failures. Those
  should be handled by hull-label acceptance gates or a visual repair UI.
