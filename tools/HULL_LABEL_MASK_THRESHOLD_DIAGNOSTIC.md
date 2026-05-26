# Hull-Label Mask Threshold Diagnostic

## Purpose

This diagnostic tests whether a fixed rembg alpha threshold (`alpha > 128`)
is too fragile for hull-label geometry. It sweeps candidate thresholds
after one rembg pass per image and records hull-label acceptance and
sticker-score quality.

Git head: `942bdfa89bb958426845f1e5d5489f0a28566553`
Generated: `2026-05-26T04:57:01.860908+00:00`
Thresholds: `[64, 128, 160, 192, 224]`

## Summary

- Pairs: 5
- Sides: 10
- Best-by-score threshold counts: `{'128': 1, '160': 3, '224': 3, '64': 3}`
- Best accepted threshold counts: `{'128': 1, '160': 3, '192': 1, '224': 2, '64': 3}`
- Accepted side count by threshold: `{'128': 9, '160': 9, '192': 10, '224': 9, '64': 9}`
- Median score improvement vs `128`: `6.900000000000034`

## Per Side

| Set | Side | Best any | Any accepted? | Best accepted | alpha>128 | alpha>224 | Notes |
|---:|---|---:|---|---:|---|---|---|
| 69 | A | 64 (793.14) | True | 64 (793.14) | accepted/794.45 | accepted/794.77 | fixed-128 not best |
| 69 | B | 160 (755.91) | True | 160 (755.91) | accepted/757.51 | accepted/757.5 | fixed-128 not best |
| 70 | A | 224 (739.99) | False | 192 (837.56) | rejected/922.5 | rejected/739.99 | fixed-128 not best; 224 best but rejected |
| 70 | B | 224 (556.04) | True | 224 (556.04) | accepted/645.11 | accepted/556.04 | fixed-128 not best |
| 71 | A | 224 (798.85) | True | 224 (798.85) | accepted/817.17 | accepted/798.85 | fixed-128 not best |
| 71 | B | 64 (760.61) | True | 64 (760.61) | accepted/761.71 | accepted/765.81 | fixed-128 not best |
| 72 | A | 160 (665.28) | True | 160 (665.28) | accepted/711.6 | accepted/702.37 | fixed-128 not best |
| 72 | B | 128 (794.7) | True | 128 (794.7) | accepted/794.7 | accepted/813.25 |  |
| 73 | A | 64 (710.29) | True | 64 (710.29) | accepted/711.32 | accepted/714.76 | fixed-128 not best |
| 73 | B | 160 (741.3) | True | 160 (741.3) | accepted/753.5 | accepted/749.6 | fixed-128 not best |

## Interpretation

- Set 70 demonstrates why a fixed `alpha > 128` mask is brittle: shadow
  pixels can be included in the silhouette, distorting the convex hull
  and downstream vertex/corner geometry.
- A candidate selector should not simply hard-code `224`. It should try
  a small threshold set and choose a candidate using geometry/color
  quality, then apply acceptance gates.
- Rows where the lowest-score candidate is rejected are especially useful
  for gate tuning: they show where visual quality and current hard gates
  disagree.
