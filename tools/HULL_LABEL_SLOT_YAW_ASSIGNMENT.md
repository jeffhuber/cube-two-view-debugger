# Hull-Label Slot/Yaw Assignment Diagnostic

## Purpose

This diagnostic tests whether hull-label `upper` / `right` / `front`
slots can be mapped directly to canonical WCA faces using capture yaw,
instead of asking the existing joint face-ID layer to infer each face
from center sticker color alone.

Two scores are reported:

- `raw`: rectified face row-major order as sampled.
- `convention`: non-oracle in-plane orientation derived from the shared
  corner/facelet convention.
- `gt_aligned`: oracle per-face orientation alignment against ground truth.
  This isolates slot/yaw face identity from in-plane face rotation and
  should be read as an upper bound, not production behavior.

Git head: `e021b8907fbd8c054c4eeb25f54f58da22fb1ed8`
Generated: `2026-05-26T19:20:39.129861+00:00`

## Summary By Yaw Source

| Yaw source | Rows | Assembled | Yaw counts | Raw exact | Raw mean stickers | Convention exact | Convention legal | Convention mean stickers | GT-aligned exact | GT-aligned mean stickers |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `assumed_zero` | 46 | 46 | `{'0': 46}` | 0 | 11.93 | 6 | 6 | 26.46 | 6 | 36.83 |
| `detected` | 46 | 28 | `{'0': 11, '1': 10, '2': 5, '3': 2, 'None': 18}` | 0 | 16.11 | 15 | 15 | 53 | 15 | 53 |
| `hull_label_center_colors` | 46 | 46 | `{'0': 19, '1': 15, '2': 9, '3': 3}` | 0 | 15.43 | 20 | 20 | 52.04 | 20 | 52.04 |
| `manifest_expectedYaw` | 9 | 9 | `{'0': 1, '1': 3, '2': 3, '3': 2}` | 0 | 17.44 | 6 | 6 | 53.33 | 6 | 53.33 |
| `manifest_notes` | 7 | 7 | `{'0': 2, '1': 1, '2': 3, '3': 1}` | 0 | 14.71 | 1 | 1 | 50.71 | 1 | 50.71 |

## Key Findings

- Convention-derived in-plane orientation tracks the oracle orientation
  closely when yaw is right. In this run, the preferred yaw source produced
  20 convention-exact rows versus
  20 oracle-exact rows under `gt_aligned`
  for `hull_label_center_colors`.
- Capture yaw is load-bearing. `assumed_zero` exact rows are much lower
  than detected/manifest yaw on non-zero-yaw captures, and wrong yaw
  commonly produces about 40+ sticker errors.
- Hull-label center-color yaw is the production-shaped fallback source:
  it comes from the same rectified slot centers the hull-label path already
  computes, so it can survive rows where the legacy recognizer cannot emit
  `captureYaw`.
- Remaining small hamming counts under correct yaw are not face-identity
  failures; they point at sticker color sampling/classification quality.

## Interpretation Guide

- If `convention` is strong, the shared geometry convention can assign
  both WCA face identity and in-plane face orientation.
- If `gt_aligned` is strong while `convention` is weak, slot/yaw face
  identity is promising but the convention-derived orientation is wrong.
- If both `convention` and `gt_aligned` are weak, the problem is still
  color sampling, rectification, or the yaw source.
- If `manifest` beats `detected`, yaw detection is the bottleneck.
- If `assumed_zero` is close to `manifest`, this corpus is mostly yaw=0.

## Per-Pair Snapshot

| Set | Source | Yaw | Raw hamming | Convention hamming | Convention legal | GT-aligned hamming | GT-aligned legal |
|---|---|---:|---:|---:|---|---:|---|
| 12 | `assumed_zero` | 0 | 47 | 44 | False | 26 | False |
| 12 | `manifest_expectedYaw` | 3 | 41 | 0 | True | 0 | True |
| 12 | `hull_label_center_colors` | 3 | 41 | 0 | True | 0 | True |
| 12 | `detected` | 3 | 41 | 0 | True | 0 | True |
| 14 | `assumed_zero` | 0 | 38 | 0 | True | 0 | True |
| 14 | `hull_label_center_colors` | 0 | 38 | 0 | True | 0 | True |
| 14 | `detected` | 0 | 38 | 0 | True | 0 | True |
| 15 | `assumed_zero` | 0 | 44 | 1 | False | 1 | False |
| 15 | `hull_label_center_colors` | 0 | 44 | 1 | False | 1 | False |
| 15 | `detected` | 0 | 44 | 1 | False | 1 | False |
| 23 | `assumed_zero` | 0 | 50 | 44 | False | 30 | False |
| 23 | `manifest_expectedYaw` | 2 | 37 | 0 | True | 0 | True |
| 23 | `hull_label_center_colors` | 2 | 37 | 0 | True | 0 | True |
| 23 | `detected` | 2 | 37 | 0 | True | 0 | True |
| 24 | `assumed_zero` | 0 | 43 | 3 | False | 3 | False |
| 24 | `hull_label_center_colors` | 0 | 43 | 3 | False | 3 | False |
| 24 | `detected` | 0 | 43 | 3 | False | 3 | False |
| 26 | `assumed_zero` | 0 | 44 | 2 | False | 2 | False |
| 26 | `hull_label_center_colors` | 0 | 44 | 2 | False | 2 | False |
| 26 | `detected` | 0 | 44 | 2 | False | 2 | False |
| 27 | `assumed_zero` | 0 | 37 | 1 | False | 1 | False |
| 27 | `hull_label_center_colors` | 0 | 37 | 1 | False | 1 | False |
| 27 | `detected` | 0 | 37 | 1 | False | 1 | False |
| 28 | `assumed_zero` | 0 | 46 | 3 | False | 3 | False |
| 28 | `hull_label_center_colors` | 0 | 46 | 3 | False | 3 | False |
| 28 | `detected` | None | None | None | None | None | None |
| 29 | `assumed_zero` | 0 | 35 | 1 | False | 1 | False |
| 29 | `hull_label_center_colors` | 0 | 35 | 1 | False | 1 | False |
| 29 | `detected` | 0 | 35 | 1 | False | 1 | False |
| 31 | `assumed_zero` | 0 | 46 | 43 | False | 25 | False |
| 31 | `hull_label_center_colors` | 1 | 44 | 2 | False | 2 | False |
| 31 | `detected` | None | None | None | None | None | None |
| 32 | `assumed_zero` | 0 | 39 | 43 | False | 26 | False |
| 32 | `manifest_expectedYaw` | 1 | 38 | 0 | True | 0 | True |
| 32 | `hull_label_center_colors` | 1 | 38 | 0 | True | 0 | True |
| 32 | `detected` | 1 | 38 | 0 | True | 0 | True |
| 36 | `assumed_zero` | 0 | 40 | 42 | False | 24 | False |
| 36 | `manifest_expectedYaw` | 2 | 36 | 0 | True | 0 | True |
| 36 | `hull_label_center_colors` | 2 | 36 | 0 | True | 0 | True |
| 36 | `detected` | 2 | 36 | 0 | True | 0 | True |
| 37 | `assumed_zero` | 0 | 44 | 43 | False | 27 | False |
| 37 | `manifest_expectedYaw` | 1 | 34 | 0 | True | 0 | True |
| 37 | `hull_label_center_colors` | 1 | 34 | 0 | True | 0 | True |
| 37 | `detected` | 1 | 34 | 0 | True | 0 | True |
| 42 | `assumed_zero` | 0 | 37 | 45 | False | 26 | False |
| 42 | `manifest_expectedYaw` | 1 | 33 | 0 | True | 0 | True |
| 42 | `hull_label_center_colors` | 1 | 33 | 0 | True | 0 | True |
| 42 | `detected` | 1 | 33 | 0 | True | 0 | True |
| 44 | `assumed_zero` | 0 | 42 | 0 | True | 0 | True |
| 44 | `hull_label_center_colors` | 0 | 42 | 0 | True | 0 | True |
| 44 | `detected` | None | None | None | None | None | None |
| 20 | `assumed_zero` | 0 | 30 | 0 | True | 0 | True |
| 20 | `hull_label_center_colors` | 0 | 30 | 0 | True | 0 | True |
| 20 | `detected` | 0 | 30 | 0 | True | 0 | True |
| 38 | `assumed_zero` | 0 | 45 | 45 | False | 26 | False |
| 38 | `hull_label_center_colors` | 1 | 42 | 0 | True | 0 | True |
| 38 | `detected` | 1 | 42 | 0 | True | 0 | True |
| 40 | `assumed_zero` | 0 | 33 | 0 | True | 0 | True |
| 40 | `hull_label_center_colors` | 0 | 33 | 0 | True | 0 | True |
| 40 | `detected` | 0 | 33 | 0 | True | 0 | True |
| 41 | `assumed_zero` | 0 | 47 | 45 | False | 28 | False |
| 41 | `hull_label_center_colors` | 1 | 41 | 0 | True | 0 | True |
| 41 | `detected` | 1 | 41 | 0 | True | 0 | True |
| 43 | `assumed_zero` | 0 | 47 | 43 | False | 25 | False |
| 43 | `hull_label_center_colors` | 1 | 41 | 0 | True | 0 | True |
| 43 | `detected` | 1 | 41 | 0 | True | 0 | True |
| 45 | `assumed_zero` | 0 | 36 | 0 | True | 0 | True |
| 45 | `hull_label_center_colors` | 0 | 36 | 0 | True | 0 | True |
| 45 | `detected` | 0 | 36 | 0 | True | 0 | True |
| 17 | `assumed_zero` | 0 | 41 | 46 | False | 27 | False |
| 17 | `hull_label_center_colors` | 1 | 37 | 0 | True | 0 | True |
| 17 | `detected` | None | None | None | None | None | None |
| 21 | `assumed_zero` | 0 | 45 | 47 | False | 27 | False |
| 21 | `hull_label_center_colors` | 1 | 36 | 0 | True | 0 | True |
| 21 | `detected` | 1 | 36 | 0 | True | 0 | True |
| 22 | `assumed_zero` | 0 | 51 | 44 | False | 30 | False |
| 22 | `hull_label_center_colors` | 2 | 38 | 0 | True | 0 | True |
| 22 | `detected` | 2 | 38 | 0 | True | 0 | True |
| 25 | `assumed_zero` | 0 | 38 | 2 | False | 2 | False |
| 25 | `hull_label_center_colors` | 0 | 38 | 2 | False | 2 | False |
| 25 | `detected` | None | None | None | None | None | None |
| 30 | `assumed_zero` | 0 | 48 | 46 | False | 31 | False |
| 30 | `hull_label_center_colors` | 1 | 42 | 2 | False | 2 | False |
| 30 | `detected` | 1 | 42 | 2 | False | 2 | False |
| 39 | `assumed_zero` | 0 | 46 | 45 | False | 27 | False |
| 39 | `hull_label_center_colors` | 1 | 40 | 0 | True | 0 | True |
| 39 | `detected` | None | None | None | None | None | None |
| 46 | `assumed_zero` | 0 | 39 | 5 | False | 5 | False |
| 46 | `manifest_notes` | 0 | 39 | 5 | False | 5 | False |
| 46 | `hull_label_center_colors` | 0 | 39 | 5 | False | 5 | False |
| 46 | `detected` | None | None | None | None | None | None |
| 47 | `assumed_zero` | 0 | 48 | 47 | False | 30 | False |
| 47 | `manifest_notes` | 1 | 39 | 4 | False | 4 | False |
| 47 | `hull_label_center_colors` | 1 | 39 | 4 | False | 4 | False |
| 47 | `detected` | None | None | None | None | None | None |
| 48 | `assumed_zero` | 0 | 47 | 48 | False | 27 | False |
| 48 | `manifest_notes` | 2 | 39 | 1 | False | 1 | False |
| 48 | `hull_label_center_colors` | 2 | 39 | 1 | False | 1 | False |
| 48 | `detected` | None | None | None | None | None | None |
| 49 | `assumed_zero` | 0 | 48 | 47 | False | 30 | False |
| 49 | `manifest_notes` | 3 | 40 | 2 | False | 2 | False |
| 49 | `hull_label_center_colors` | 3 | 40 | 2 | False | 2 | False |
| 49 | `detected` | None | None | None | None | None | None |
| 57 | `assumed_zero` | 0 | 41 | 49 | False | 28 | False |
| 57 | `hull_label_center_colors` | 1 | 36 | 2 | False | 2 | False |
| 57 | `detected` | 1 | 36 | 2 | False | 2 | False |
| 58 | `assumed_zero` | 0 | 50 | 39 | False | 27 | False |
| 58 | `hull_label_center_colors` | 2 | 43 | 1 | False | 1 | False |
| 58 | `detected` | 2 | 43 | 1 | False | 1 | False |
| 61 | `assumed_zero` | 0 | 35 | 5 | False | 5 | False |
| 61 | `hull_label_center_colors` | 0 | 35 | 5 | False | 5 | False |
| 61 | `detected` | 0 | 35 | 5 | False | 5 | False |
| 62 | `assumed_zero` | 0 | 43 | 45 | False | 27 | False |
| 62 | `hull_label_center_colors` | 1 | 38 | 4 | False | 4 | False |
| 62 | `detected` | 1 | 38 | 4 | False | 4 | False |
| 63 | `assumed_zero` | 0 | 42 | 41 | False | 29 | False |
| 63 | `manifest_expectedYaw` | 2 | 31 | 1 | False | 1 | False |
| 63 | `hull_label_center_colors` | 2 | 31 | 1 | False | 1 | False |
| 63 | `detected` | 2 | 31 | 1 | False | 1 | False |
| 64 | `assumed_zero` | 0 | 34 | 0 | True | 0 | True |
| 64 | `manifest_notes` | 0 | 34 | 0 | True | 0 | True |
| 64 | `hull_label_center_colors` | 0 | 34 | 0 | True | 0 | True |
| 64 | `detected` | None | None | None | None | None | None |
| 65 | `assumed_zero` | 0 | 37 | 45 | False | 24 | False |
| 65 | `hull_label_center_colors` | 1 | 32 | 0 | True | 0 | True |
| 65 | `detected` | None | None | None | None | None | None |
| 66 | `assumed_zero` | 0 | 50 | 42 | False | 29 | False |
| 66 | `hull_label_center_colors` | 2 | 41 | 1 | False | 1 | False |
| 66 | `detected` | None | None | None | None | None | None |
| 67 | `assumed_zero` | 0 | 44 | 44 | False | 24 | False |
| 67 | `manifest_expectedYaw` | 3 | 41 | 2 | False | 2 | False |
| 67 | `hull_label_center_colors` | 3 | 41 | 2 | False | 2 | False |
| 67 | `detected` | 3 | 41 | 2 | False | 2 | False |
| 68 | `assumed_zero` | 0 | 38 | 3 | False | 3 | False |
| 68 | `manifest_expectedYaw` | 0 | 38 | 3 | False | 3 | False |
| 68 | `hull_label_center_colors` | 0 | 38 | 3 | False | 3 | False |
| 68 | `detected` | 0 | 38 | 3 | False | 3 | False |
| 69 | `assumed_zero` | 0 | 36 | 48 | False | 25 | False |
| 69 | `manifest_notes` | 2 | 37 | 1 | False | 1 | False |
| 69 | `hull_label_center_colors` | 2 | 37 | 1 | False | 1 | False |
| 69 | `detected` | None | None | None | None | None | None |
| 70 | `assumed_zero` | 0 | 46 | 50 | False | 28 | False |
| 70 | `manifest_notes` | 2 | 47 | 10 | False | 10 | False |
| 70 | `hull_label_center_colors` | 2 | 47 | 10 | False | 10 | False |
| 70 | `detected` | None | None | None | None | None | None |
| 71 | `assumed_zero` | 0 | 41 | 11 | False | 11 | False |
| 71 | `hull_label_center_colors` | 0 | 41 | 11 | False | 11 | False |
| 71 | `detected` | None | None | None | None | None | None |
| 72 | `assumed_zero` | 0 | 40 | 9 | False | 9 | False |
| 72 | `hull_label_center_colors` | 0 | 40 | 9 | False | 9 | False |
| 72 | `detected` | None | None | None | None | None | None |
| 73 | `assumed_zero` | 0 | 37 | 11 | False | 11 | False |
| 73 | `hull_label_center_colors` | 0 | 37 | 11 | False | 11 | False |
| 73 | `detected` | None | None | None | None | None | None |
