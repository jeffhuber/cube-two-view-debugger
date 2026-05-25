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

Git head: `18ec7d5370478f6719d8ab620a447f004a94f0cb`
Generated: `2026-05-25T18:19:05.376787+00:00`

## Summary By Yaw Source

| Yaw source | Rows | Assembled | Yaw counts | Raw exact | Raw mean stickers | Convention exact | Convention legal | Convention mean stickers | GT-aligned exact | GT-aligned mean stickers |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `assumed_zero` | 41 | 40 | `{'0': 41}` | 0 | 11.6 | 5 | 5 | 26.32 | 5 | 36.8 |
| `detected` | 41 | 27 | `{'0': 11, '1': 10, '2': 5, '3': 2, 'None': 13}` | 0 | 15.52 | 14 | 14 | 52.11 | 14 | 52.15 |
| `hull_label_center_colors` | 41 | 40 | `{'0': 16, '1': 15, '2': 7, '3': 3}` | 0 | 15.45 | 19 | 19 | 52.15 | 19 | 52.17 |
| `manifest_expectedYaw` | 9 | 9 | `{'0': 1, '1': 3, '2': 3, '3': 2}` | 0 | 15.44 | 5 | 5 | 51.44 | 5 | 51.44 |
| `manifest_notes` | 5 | 5 | `{'0': 2, '1': 1, '2': 1, '3': 1}` | 0 | 15.4 | 0 | 0 | 50.8 | 0 | 50.8 |

## Key Findings

- Convention-derived in-plane orientation tracks the oracle orientation
  closely when yaw is right. In this run, the preferred yaw source produced
  19 convention-exact rows versus
  19 oracle-exact rows under `gt_aligned`
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
| 14 | `assumed_zero` | 0 | 39 | 6 | False | 5 | False |
| 14 | `hull_label_center_colors` | 0 | 39 | 6 | False | 5 | False |
| 14 | `detected` | 0 | 39 | 6 | False | 5 | False |
| 15 | `assumed_zero` | 0 | 37 | 0 | True | 0 | True |
| 15 | `hull_label_center_colors` | 0 | 37 | 0 | True | 0 | True |
| 15 | `detected` | 0 | 37 | 0 | True | 0 | True |
| 23 | `assumed_zero` | 0 | 42 | 44 | False | 30 | False |
| 23 | `manifest_expectedYaw` | 2 | 29 | 0 | True | 0 | True |
| 23 | `hull_label_center_colors` | 2 | 29 | 0 | True | 0 | True |
| 23 | `detected` | 2 | 29 | 0 | True | 0 | True |
| 24 | `assumed_zero` | 0 | 35 | 3 | False | 3 | False |
| 24 | `hull_label_center_colors` | 0 | 35 | 3 | False | 3 | False |
| 24 | `detected` | 0 | 35 | 3 | False | 3 | False |
| 26 | `assumed_zero` | 0 | 46 | 3 | False | 3 | False |
| 26 | `hull_label_center_colors` | 0 | 46 | 3 | False | 3 | False |
| 26 | `detected` | 0 | 46 | 3 | False | 3 | False |
| 27 | `assumed_zero` | 0 | 45 | 1 | False | 1 | False |
| 27 | `hull_label_center_colors` | 0 | 45 | 1 | False | 1 | False |
| 27 | `detected` | 0 | 45 | 1 | False | 1 | False |
| 28 | `assumed_zero` | 0 | 38 | 3 | False | 3 | False |
| 28 | `hull_label_center_colors` | 0 | 38 | 3 | False | 3 | False |
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
| 36 | `assumed_zero` | 0 | 47 | 43 | False | 25 | False |
| 36 | `manifest_expectedYaw` | 2 | 41 | 2 | False | 2 | False |
| 36 | `hull_label_center_colors` | 2 | 41 | 2 | False | 2 | False |
| 36 | `detected` | 2 | 41 | 2 | False | 2 | False |
| 37 | `assumed_zero` | 0 | 52 | 43 | False | 27 | False |
| 37 | `manifest_expectedYaw` | 1 | 42 | 0 | True | 0 | True |
| 37 | `hull_label_center_colors` | 1 | 42 | 0 | True | 0 | True |
| 37 | `detected` | 1 | 42 | 0 | True | 0 | True |
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
| 43 | `assumed_zero` | 0 | 40 | 43 | False | 25 | False |
| 43 | `hull_label_center_colors` | 1 | 34 | 0 | True | 0 | True |
| 43 | `detected` | 1 | 34 | 0 | True | 0 | True |
| 45 | `assumed_zero` | 0 | 36 | 0 | True | 0 | True |
| 45 | `hull_label_center_colors` | 0 | 36 | 0 | True | 0 | True |
| 45 | `detected` | 0 | 36 | 0 | True | 0 | True |
| 17 | `assumed_zero` | 0 | 40 | 46 | False | 27 | False |
| 17 | `hull_label_center_colors` | 1 | 36 | 0 | True | 0 | True |
| 17 | `detected` | None | None | None | None | None | None |
| 21 | `assumed_zero` | 0 | 45 | 47 | False | 27 | False |
| 21 | `hull_label_center_colors` | 1 | 36 | 0 | True | 0 | True |
| 21 | `detected` | 1 | 36 | 0 | True | 0 | True |
| 22 | `assumed_zero` | 0 | 50 | 44 | False | 30 | False |
| 22 | `hull_label_center_colors` | 2 | 37 | 0 | True | 0 | True |
| 22 | `detected` | 2 | 37 | 0 | True | 0 | True |
| 25 | `assumed_zero` | 0 | 38 | 2 | False | 2 | False |
| 25 | `hull_label_center_colors` | 0 | 38 | 2 | False | 2 | False |
| 25 | `detected` | None | None | None | None | None | None |
| 30 | `assumed_zero` | 0 | None | None | None | None | None |
| 30 | `hull_label_center_colors` | 1 | None | None | None | None | None |
| 30 | `detected` | 1 | None | None | None | None | None |
| 39 | `assumed_zero` | 0 | 46 | 45 | False | 27 | False |
| 39 | `hull_label_center_colors` | 1 | 40 | 0 | True | 0 | True |
| 39 | `detected` | None | None | None | None | None | None |
| 46 | `assumed_zero` | 0 | 33 | 5 | False | 5 | False |
| 46 | `manifest_notes` | 0 | 33 | 5 | False | 5 | False |
| 46 | `hull_label_center_colors` | 0 | 33 | 5 | False | 5 | False |
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
| 57 | `assumed_zero` | 0 | 50 | 50 | False | 29 | False |
| 57 | `hull_label_center_colors` | 1 | 45 | 3 | False | 3 | False |
| 57 | `detected` | 1 | 45 | 3 | False | 3 | False |
| 58 | `assumed_zero` | 0 | 50 | 38 | False | 27 | False |
| 58 | `hull_label_center_colors` | 2 | 43 | 2 | False | 2 | False |
| 58 | `detected` | 2 | 43 | 2 | False | 2 | False |
| 61 | `assumed_zero` | 0 | 33 | 5 | False | 5 | False |
| 61 | `hull_label_center_colors` | 0 | 33 | 5 | False | 5 | False |
| 61 | `detected` | 0 | 33 | 5 | False | 5 | False |
| 62 | `assumed_zero` | 0 | 50 | 45 | False | 27 | False |
| 62 | `hull_label_center_colors` | 1 | 45 | 4 | False | 4 | False |
| 62 | `detected` | 1 | 45 | 4 | False | 4 | False |
| 63 | `assumed_zero` | 0 | 49 | 44 | False | 29 | False |
| 63 | `manifest_expectedYaw` | 2 | 43 | 8 | False | 8 | False |
| 63 | `hull_label_center_colors` | 2 | 43 | 8 | False | 8 | False |
| 63 | `detected` | 2 | 43 | 8 | False | 8 | False |
| 64 | `assumed_zero` | 0 | 42 | 4 | False | 4 | False |
| 64 | `manifest_notes` | 0 | 42 | 4 | False | 4 | False |
| 64 | `hull_label_center_colors` | 0 | 42 | 4 | False | 4 | False |
| 64 | `detected` | None | None | None | None | None | None |
| 65 | `assumed_zero` | 0 | 37 | 45 | False | 24 | False |
| 65 | `hull_label_center_colors` | 1 | 32 | 0 | True | 0 | True |
| 65 | `detected` | None | None | None | None | None | None |
| 66 | `assumed_zero` | 0 | 50 | 42 | False | 30 | False |
| 66 | `hull_label_center_colors` | 2 | 40 | 0 | True | 0 | True |
| 66 | `detected` | None | None | None | None | None | None |
| 67 | `assumed_zero` | 0 | 44 | 44 | False | 24 | False |
| 67 | `manifest_expectedYaw` | 3 | 42 | 9 | False | 9 | False |
| 67 | `hull_label_center_colors` | 3 | 42 | 9 | False | 9 | False |
| 67 | `detected` | 3 | 42 | 9 | False | 9 | False |
| 68 | `assumed_zero` | 0 | 38 | 4 | False | 4 | False |
| 68 | `manifest_expectedYaw` | 0 | 38 | 4 | False | 4 | False |
| 68 | `hull_label_center_colors` | 0 | 38 | 4 | False | 4 | False |
| 68 | `detected` | 0 | 38 | 4 | False | 4 | False |
