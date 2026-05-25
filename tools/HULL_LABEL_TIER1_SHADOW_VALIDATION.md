# Hull-Label Tier 1 Shadow Validation

## Purpose

This report validates the feature-flagged Tier 1 hull-label path added
to `tools.global_cube_model.fit_global_cube_model()`. It compares the
legacy path (`off`), trace-only shadow mode (`shadow`), and accepted
candidate mode with fallback (`prefer`) on A+B corpus pairs.

The pair-level score is diagnostic: it rectifies faces from the global
model quads, classifies stickers, performs joint face-ID, assembles a
54-sticker URFDLB state, and validates that state. It does not change
the production `WhiteUpRecognizer` path.

Git head: `8de0941ce1f4b7fa7c2f524b4e021edc19dc7aad`
Generated: `2026-05-25T07:26:01.642841+00:00`

## Summary By Mode

| Mode | Pairs | Assembled | Legal | Exact | Rectified sticker acc | Assembled sticker acc |
|---|---:|---:|---:|---:|---:|---:|
| `off` | 41 | 41 | 0 | 0 | 33.1% | 33.1% |
| `shadow` | 41 | 41 | 0 | 0 | 33.1% | 33.1% |
| `prefer` | 41 | 41 | 19 | 19 | 95.9% | 95.9% |

## Shadow Trace

- Side traces: `82`
- Accepted sides: `81`
- Selected sides: `0`
- Status counts: `{'accepted': 81, 'rejected': 1}`
- Hard failures: `{'projective_residual_norm=0.0273; max 0.0250': 1}`
- Warnings: `{'projective_residual_norm=0.0189; warning 0.0180': 1, 'projective_residual_norm=0.0229; warning 0.0180': 1, 'sticker_score_total=706.5; warning 700.0': 1, 'sticker_score_total=712.6; warning 700.0': 1, 'sticker_score_total=728.7; warning 700.0': 1, 'sticker_score_worst_face=351.6; warning 350.0': 1, 'sticker_score_worst_face=357.4; warning 350.0': 1, 'sticker_score_worst_face=360.3; warning 350.0': 1, 'sticker_score_worst_face=372.5; warning 350.0': 1}`

## Prefer Trace

- Side traces: `82`
- Accepted sides: `81`
- Selected sides: `81`
- Status counts: `{'accepted': 81, 'rejected': 1}`
- Hard failures: `{'projective_residual_norm=0.0273; max 0.0250': 1}`
- Warnings: `{'projective_residual_norm=0.0189; warning 0.0180': 1, 'projective_residual_norm=0.0229; warning 0.0180': 1, 'sticker_score_total=706.5; warning 700.0': 1, 'sticker_score_total=712.6; warning 700.0': 1, 'sticker_score_total=728.7; warning 700.0': 1, 'sticker_score_worst_face=351.6; warning 350.0': 1, 'sticker_score_worst_face=357.4; warning 350.0': 1, 'sticker_score_worst_face=360.3; warning 350.0': 1, 'sticker_score_worst_face=372.5; warning 350.0': 1}`

## Prefer Versus Legacy

- Improved hamming: `41`
- Regressed hamming: `0`
- Same hamming/incomplete status: `0`
- Incomplete-status changes: `0`

Improved rows (first 20 of 41; see per-pair snapshot below):
- Set 12: `32` -> `0`
- Set 14: `28` -> `5`
- Set 15: `38` -> `0`
- Set 23: `40` -> `0`
- Set 24: `36` -> `3`
- Set 26: `35` -> `3`
- Set 27: `38` -> `1`
- Set 28: `37` -> `3`
- Set 29: `34` -> `1`
- Set 31: `36` -> `2`
- Set 32: `37` -> `0`
- Set 36: `33` -> `2`
- Set 37: `36` -> `0`
- Set 42: `37` -> `0`
- Set 44: `33` -> `0`
- Set 20: `35` -> `0`
- Set 38: `37` -> `0`
- Set 40: `35` -> `0`
- Set 41: `37` -> `0`
- Set 43: `36` -> `0`

## Per-Pair Snapshot

| Set | Off hamming | Prefer hamming | Prefer selected sides | Prefer trace statuses |
|---|---:|---:|---:|---|
| 12 | 32 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 14 | 28 | 5 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 15 | 38 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 23 | 40 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 24 | 36 | 3 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 26 | 35 | 3 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 27 | 38 | 1 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 28 | 37 | 3 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 29 | 34 | 1 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 31 | 36 | 2 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 32 | 37 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 36 | 33 | 2 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 37 | 36 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 42 | 37 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 44 | 33 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 20 | 35 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 38 | 37 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 40 | 35 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 41 | 37 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 43 | 36 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 45 | 33 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 17 | 39 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 21 | 38 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 22 | 37 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 25 | 28 | 2 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 30 | 35 | 18 | 1 | `{'A': 'rejected', 'B': 'accepted'}` |
| 39 | 42 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 46 | 40 | 5 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 47 | 36 | 4 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 48 | 34 | 1 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 49 | 41 | 2 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 57 | 36 | 3 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 58 | 39 | 2 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 61 | 33 | 5 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 62 | 36 | 4 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 63 | 39 | 8 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 64 | 38 | 4 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 65 | 39 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 66 | 38 | 0 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 67 | 36 | 9 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
| 68 | 35 | 4 | 2 | `{'A': 'accepted', 'B': 'accepted'}` |
