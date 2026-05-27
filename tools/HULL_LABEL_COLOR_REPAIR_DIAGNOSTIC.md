# Hull-Label Color Repair Diagnostic

## Purpose

This diagnostic asks whether deterministic color bookkeeping can clean
up hull-label rectified panels before involving an LLM. It uses the
same slot/yaw WCA assignment as the hull-label path, then compares
plain Lab nearest-color classification, center forcing, exact
9-per-color count repair, guarded two-view cubie consistency, and
guarded cubie-legality repair.

Git head: `0078609caa58f3455805650d576991f655ad2586`
Generated: `2026-05-27T14:40:31.905677+00:00`

## Headline

The production-like yaw source is `hull_label_center_colors`, because it
does not use ground-truth yaw metadata. It is available for most pairs;
rows without an accepted center-yaw inference are still shown with their
metadata/default yaw fallback in the per-set table.
On the current 71-pair GT corpus, count repair changes the story from
mostly-correct panels to mostly-exact cubes:

| Stage | Assembled | Exact | Legal | Mean stickers | Median hamming | Hamming distribution |
|---|---:|---:|---:|---:|---:|---|
| `canonical` | 71 | 29 | 29 | 51.65 | 1 | `{0: 29, 1: 7, 2: 11, 3: 5, 4: 3, 5: 6, 6: 3, 7: 2, 8: 1, 9: 1, 10: 2, 12: 1}` |
| `canonical_center_forced` | 71 | 30 | 30 | 51.72 | 1 | `{0: 30, 1: 7, 2: 10, 3: 5, 4: 4, 5: 5, 6: 3, 7: 3, 9: 2, 10: 1, 12: 1}` |
| `canonical_count_repaired` | 71 | 66 | 66 | 53.82 | 0 | `{0: 66, 2: 3, 3: 1, 4: 1}` |
| `conservative_legal_repaired` | 71 | 68 | 68 | 54 | 0.0 | `{0: 68}` |
| `two_view_consistency_repaired` | 71 | 2 | 2 | 54 | 0.0 | `{0: 2}` |
| `guarded_broad_legal_repaired` | 71 | 70 | 70 | 54 | 0.0 | `{0: 70}` |
| `adaptive` | 71 | 25 | 25 | 51.32 | 2 | `{0: 25, 1: 9, 2: 7, 3: 8, 4: 6, 5: 6, 6: 3, 7: 1, 9: 2, 11: 2, 12: 2}` |
| `adaptive_count_repaired` | 71 | 49 | 49 | 52.69 | 0 | `{0: 49, 2: 13, 4: 1, 6: 5, 8: 1, 11: 1, 14: 1}` |

`canonical_count_repaired` is the stable deterministic baseline:
66/71 exact/legal, 70/71 within 3 stickers, and only 1 row above 3 stickers.
The payload's recommended-method selector is now 70/71 exact with hamming distribution `{0: 70, 4: 1}`.
`two_view_consistency_repaired` is intentionally narrower than
`guarded_broad_legal_repaired`: it promoted
2/71 rows in this run,
only when the current count-repaired state had split-cubie
inconsistency across the A/B views and the candidate cleared that
inconsistency.

## Full Summary By Yaw Source

| Yaw source | Method | Assembled | Exact | Legal | Mean stickers | Median hamming |
|---|---|---:|---:|---:|---:|---:|
| `ground_truth_captureYaw` | `canonical` | 26 | 8 | 8 | 51.35 | 2.0 |
| `ground_truth_captureYaw` | `canonical_center_forced` | 26 | 8 | 8 | 51.42 | 1.5 |
| `ground_truth_captureYaw` | `canonical_count_repaired` | 26 | 23 | 23 | 53.73 | 0.0 |
| `ground_truth_captureYaw` | `conservative_legal_repaired` | 26 | 24 | 24 | 54 | 0.0 |
| `ground_truth_captureYaw` | `two_view_consistency_repaired` | 26 | 2 | 2 | 54 | 0.0 |
| `ground_truth_captureYaw` | `guarded_broad_legal_repaired` | 26 | 26 | 26 | 54 | 0.0 |
| `ground_truth_captureYaw` | `adaptive` | 26 | 5 | 5 | 50.38 | 3.0 |
| `ground_truth_captureYaw` | `adaptive_center_forced` | 26 | 5 | 5 | 50.38 | 3.0 |
| `ground_truth_captureYaw` | `adaptive_count_repaired` | 26 | 12 | 12 | 52.08 | 2.0 |
| `hull_label_center_colors` | `canonical` | 71 | 29 | 29 | 51.65 | 1 |
| `hull_label_center_colors` | `canonical_center_forced` | 71 | 30 | 30 | 51.72 | 1 |
| `hull_label_center_colors` | `canonical_count_repaired` | 71 | 66 | 66 | 53.82 | 0 |
| `hull_label_center_colors` | `conservative_legal_repaired` | 71 | 68 | 68 | 54 | 0.0 |
| `hull_label_center_colors` | `two_view_consistency_repaired` | 71 | 2 | 2 | 54 | 0.0 |
| `hull_label_center_colors` | `guarded_broad_legal_repaired` | 71 | 70 | 70 | 54 | 0.0 |
| `hull_label_center_colors` | `adaptive` | 71 | 25 | 25 | 51.32 | 2 |
| `hull_label_center_colors` | `adaptive_center_forced` | 71 | 25 | 25 | 51.37 | 2 |
| `hull_label_center_colors` | `adaptive_count_repaired` | 71 | 49 | 49 | 52.69 | 0 |
| `manifest_expectedYaw` | `canonical` | 7 | 2 | 2 | 51.29 | 2 |
| `manifest_expectedYaw` | `canonical_center_forced` | 7 | 2 | 2 | 51.29 | 2 |
| `manifest_expectedYaw` | `canonical_count_repaired` | 7 | 7 | 7 | 54 | 0 |
| `manifest_expectedYaw` | `conservative_legal_repaired` | 7 | 7 | 7 | 54 | 0 |
| `manifest_expectedYaw` | `two_view_consistency_repaired` | 7 | 0 | 0 | None | None |
| `manifest_expectedYaw` | `guarded_broad_legal_repaired` | 7 | 7 | 7 | 54 | 0 |
| `manifest_expectedYaw` | `adaptive` | 7 | 2 | 2 | 51.29 | 3 |
| `manifest_expectedYaw` | `adaptive_center_forced` | 7 | 2 | 2 | 51.29 | 3 |
| `manifest_expectedYaw` | `adaptive_count_repaired` | 7 | 6 | 6 | 53.71 | 0 |
| `manifest_notes` | `canonical` | 3 | 1 | 1 | 52.33 | 2 |
| `manifest_notes` | `canonical_center_forced` | 3 | 1 | 1 | 52.33 | 2 |
| `manifest_notes` | `canonical_count_repaired` | 3 | 3 | 3 | 54 | 0 |
| `manifest_notes` | `conservative_legal_repaired` | 3 | 3 | 3 | 54 | 0 |
| `manifest_notes` | `two_view_consistency_repaired` | 3 | 0 | 0 | None | None |
| `manifest_notes` | `guarded_broad_legal_repaired` | 3 | 3 | 3 | 54 | 0 |
| `manifest_notes` | `adaptive` | 3 | 1 | 1 | 53.33 | 1 |
| `manifest_notes` | `adaptive_center_forced` | 3 | 1 | 1 | 53.33 | 1 |
| `manifest_notes` | `adaptive_count_repaired` | 3 | 2 | 2 | 53.33 | 0 |
| `white_up_default` | `canonical` | 35 | 13 | 13 | 41.83 | 2 |
| `white_up_default` | `canonical_center_forced` | 35 | 14 | 14 | 42.83 | 2 |
| `white_up_default` | `canonical_count_repaired` | 35 | 25 | 26 | 44.51 | 0 |
| `white_up_default` | `conservative_legal_repaired` | 35 | 26 | 34 | 44.38 | 0.0 |
| `white_up_default` | `two_view_consistency_repaired` | 35 | 0 | 0 | None | None |
| `white_up_default` | `guarded_broad_legal_repaired` | 35 | 26 | 34 | 44.35 | 0.0 |
| `white_up_default` | `adaptive` | 35 | 13 | 13 | 42.57 | 3 |
| `white_up_default` | `adaptive_center_forced` | 35 | 13 | 13 | 42.89 | 2 |
| `white_up_default` | `adaptive_count_repaired` | 35 | 22 | 24 | 43.54 | 0 |

## Per-Set Snapshot

| Set | Source | Recommended | Best safe method | Best hamming | Canonical | Canonical+count | Two-view | Guarded legal | Adaptive+count | Status |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 8 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 5 | 0 | None | 0 | 2 | `assembled` |
| 9 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 3 | 0 | None | 0 | 2 | `assembled` |
| 10 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 5 | 0 | None | 0 | 2 | `assembled` |
| 11 | `hull_label_center_colors` | `guarded_broad_legal_repaired` | `guarded_broad_legal_repaired` | 0 | 12 | 2 | None | 0 | 2 | `assembled` |
| 12 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 7 | 0 | None | 0 | 0 | `assembled` |
| 13 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 6 | 0 | None | 0 | 0 | `assembled` |
| 14 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 4 | 10 | 4 | None | None | 11 | `assembled` |
| 15 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 4 | 0 | None | 0 | 0 | `assembled` |
| 16 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 5 | 0 | None | 0 | 2 | `assembled` |
| 17 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical` | 0 | 0 | 0 | None | 0 | 6 | `assembled` |
| 18 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 19 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical` | 0 | 0 | 0 | None | 0 | 6 | `assembled` |
| 20 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 1 | 0 | None | 0 | 0 | `assembled` |
| 21 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 2 | 0 | None | 0 | 0 | `assembled` |
| 22 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 3 | 0 | None | 0 | 0 | `assembled` |
| 23 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 2 | 0 | None | 0 | 0 | `assembled` |
| 24 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 25 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 26 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 27 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 28 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 29 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 30 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 31 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical` | 0 | 0 | 0 | None | 0 | 8 | `assembled` |
| 32 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 3 | 0 | None | 0 | 2 | `assembled` |
| 33 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 2 | 0 | None | 0 | 0 | `assembled` |
| 34 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 4 | 0 | None | 0 | 6 | `assembled` |
| 35 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 3 | 0 | None | 0 | 6 | `assembled` |
| 36 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 37 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 2 | 0 | None | 0 | 0 | `assembled` |
| 38 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 39 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 1 | 0 | None | 0 | 0 | `assembled` |
| 40 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 41 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 42 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 43 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 44 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 2 | 0 | None | 0 | 0 | `assembled` |
| 45 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 2 | 0 | None | 0 | 0 | `assembled` |
| 46 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 2 | 0 | None | 0 | 0 | `assembled` |
| 47 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 1 | 0 | None | 0 | 0 | `assembled` |
| 48 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 49 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 2 | 0 | None | 0 | 2 | `assembled` |
| 50 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 51 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 1 | 0 | None | 0 | 0 | `assembled` |
| 52 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 53 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 54 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 55 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 1 | 0 | None | 0 | 0 | `assembled` |
| 56 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical` | 0 | 0 | 0 | None | 0 | 2 | `assembled` |
| 57 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 1 | 0 | None | 0 | 2 | `assembled` |
| 58 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 59 | `hull_label_center_colors` | `conservative_legal_repaired` | `conservative_legal_repaired` | 0 | 8 | 2 | None | 0 | 6 | `assembled` |
| 60 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 5 | 0 | None | 0 | 2 | `assembled` |
| 61 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 62 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 2 | 0 | None | 0 | 2 | `assembled` |
| 63 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 2 | 0 | None | 0 | 0 | `assembled` |
| 64 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 3 | 0 | None | 0 | 0 | `assembled` |
| 65 | `hull_label_center_colors` | `two_view_consistency_repaired` | `guarded_broad_legal_repaired` | 0 | 7 | 2 | 0 | 0 | 4 | `assembled` |
| 66 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 4 | 0 | None | 0 | 0 | `assembled` |
| 67 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 2 | 0 | None | 0 | 0 | `assembled` |
| 68 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 5 | 0 | None | 0 | 0 | `assembled` |
| 69 | `hull_label_center_colors` | `conservative_legal_repaired` | `conservative_legal_repaired` | 0 | 6 | 3 | 0 | 0 | 14 | `assembled` |
| 70 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 5 | 0 | None | 0 | 0 | `assembled` |
| 71 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 6 | 0 | None | 0 | 0 | `assembled` |
| 72 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 9 | 0 | None | 0 | 0 | `assembled` |
| 73 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 10 | 0 | None | 0 | 0 | `assembled` |
| 74 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 75 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical` | 0 | 0 | 0 | None | 0 | 2 | `assembled` |
| 76 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |
| 77 | `hull_label_center_colors` | `canonical_count_repaired` | `canonical_count_repaired` | 0 | 1 | 0 | None | 0 | 2 | `assembled` |
| 78 | `hull_label_center_colors` | `canonical_count_repaired` | `adaptive_count_repaired` | 0 | 0 | 0 | None | 0 | 0 | `assembled` |

## Current Run Notes

- The raw `canonical` classifier is already close: 29/71 exact, 52/71
  within 3 stickers. The dominant issue is duplicated/missing color
  counts, not WCA face assignment.
- Greedy count repair is a large deterministic jump: 66/71 exact/legal
  with the production-like yaw source. This supersedes the older
  20/46 exact headline for raw hull-label `prefer` panels.
- Guarded two-view and cubie-legality repair are now part of the color-repair payload:
  two-view promotes 2/71 rows,
  while guarded broad is 70/71 exact here.
  The payload exposes conservative, split-cubie-gated, and guarded-broad
  legal candidates; the ungated broad legal candidate remains diagnostic-only.
- Canonical Lab count repair beats the adaptive-palette count repair in
  this run (66/71 exact versus 49/71). Adaptive palettes should stay
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
- GAN Sets 74-78 are the first GAN-brand tricky-lighting expansion in this
  scoreboard. The default recognizer still rejects them, but hull-label
  center-yaw inference gets yaw 0/1/2/3 correct for Sets 75-78 and
  `canonical_count_repaired` is exact on all five rows.

## Interpretation

- `center_forced` is a cheap sanity step: the center sticker of each WCA
  face is known once slot/yaw assignment succeeds, so a center-color
  miss should not be allowed to poison the state.
- `count_repaired` is deliberately not a legality solver. It only enforces
  the physical requirement that each WCA face color appears exactly nine
  times. This catches duplicated/missing color reads while preserving the
  sampled geometry.
- `two_view_consistency_repaired` is the first explicit A/B consistency
  gate in the repair payload. It requires split-cubie inconsistency in
  `canonical_count_repaired`, then promotes an already legal candidate
  only when that candidate clears cubie consistency within the same
  cost/state-delta limits as guarded broad repair.
- `conservative_legal_repaired` and `guarded_broad_legal_repaired` add the
  broader cubie-legality layer. The raw `broad_legal_repaired` method is
  emitted only for traceability.
- The adaptive palette uses the six known center samples as anchors. It is
  still deterministic and local to the two input photos; no GT colors or
  LLM output are used.
- Rows that remain high-hamming after adaptive count repair are likely
  geometry/panel-quality failures rather than cube-count failures. Those
  should be handled by hull-label acceptance gates or a visual repair UI.

## Decision Path

1. Treat `two_view_consistency_repaired` as a transparent diagnostic/selector,
   not as the main accuracy lever. It explains the split-cubie rescue rows
   and gives the Fixer trace better evidence, but guarded broad remains the
   wider legal-repair candidate.
2. Keep the current guarded-broad state-delta gate as the production-shaped
   repair candidate until out-of-corpus rows show a new failure mode.
3. Pull the next accuracy lever in front of repair: Lab + LLM evidence
   ensemble per sticker, then confidence-gated auto-merge of repair variants.
4. Re-run this scoreboard whenever the manifest/GT corpus changes, especially
   as additional GAN/tricky-lighting rows are added.
