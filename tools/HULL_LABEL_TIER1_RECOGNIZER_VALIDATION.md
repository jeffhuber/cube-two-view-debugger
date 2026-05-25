# Hull-Label Tier 1 Recognizer Validation

## Purpose

This report runs the recognizer itself on corpus A+B pairs after the
Hull-Label Tier 1 path and direct yaw assembly wiring. Unlike the
geometry-only shadow validation, this includes the production-shaped
face identity, color repair, legal-state, fallback, and diagnostic
signals that cube-snap would consume.

Git head: `a03e154bf51c750712f62101d41723c62de4add4`
Generated: `2026-05-25T18:06:29.795176+00:00`

## Summary By Mode

| Mode | Pairs | Success | Legal | Exact | Mean stickers | Sticker acc |
|---|---:|---:|---:|---:|---:|---:|
| `off` | 41 | 29 | 29 | 19 | 34.463 | 63.8% |
| `shadow` | 41 | 29 | 29 | 19 | 34.463 | 63.8% |
| `prefer_candidate` | 41 | 4 | 4 | 4 | 5.268 | 9.8% |
| `prefer_effective` | 41 | 30 | 30 | 20 | 35.78 | 66.3% |

`prefer_candidate` is the raw hull-label recognizer candidate. Rejected
candidate rows contribute 0 recognized stickers to its aggregate score;
`prefer_effective` is the production-shaped result after falling back to
legacy unless both hull-label sides are selected and the candidate succeeds.

## Interpretation

- Shadow mode is output-identical to legacy in aggregate: `19` exact / `29` success versus legacy `19` exact / `29` success.
- Hull-label geometry/yaw is usually available: `81/82` sides accepted and yaw accepted on `41` pairs.
- Effective prefer selected `4/41` rows, improved `1`, regressed `0`, and moved exact solves from `19` to `20`.
- Default-on prefer is still premature. The remaining bottleneck is not
  hull geometry; it is color/slot-to-WCA face identity after rectification.

## Prefer Effective Versus Legacy

- Improved hamming: `1`
- Regressed hamming: `0`
- Same hamming/incomplete status: `40`
- Hull-label selected rows: `4`
- Fallback rows: `37`
- Selected setIds: `32, 42, 45, 39`

Improved rows:
- Set 39: `None` -> `0`

## Hull-Label Trace

### Shadow

- Side traces: `82`
- Accepted sides: `81`
- Selected sides: `0`
- Side statuses: `{'accepted': 81, 'rejected': 1}`
- Yaw statuses: `{'accepted': 41}`
- Best-yaw counts: `{'0': 16, '1': 15, '2': 7, '3': 3}`
- Vertex sources: `{'affine': 75, 'projective': 7}`
- Hard failures: `projective_residual_norm=0.0273; max 0.0250`: `1`
- Warnings: `projective_residual_norm=0.0189; warning 0.0180`: `1`, `projective_residual_norm=0.0229; warning 0.0180`: `1`, `sticker_score_total=706.5; warning 700.0`: `1`, `sticker_score_total=712.6; warning 700.0`: `1`, `sticker_score_total=728.7; warning 700.0`: `1`, `sticker_score_worst_face=351.6; warning 350.0`: `1`, `sticker_score_worst_face=357.4; warning 350.0`: `1`, `sticker_score_worst_face=360.3; warning 350.0`: `1`

### Prefer Candidate

- Side traces: `82`
- Accepted sides: `81`
- Selected sides: `81`
- Side statuses: `{'accepted': 81, 'rejected': 1}`
- Yaw statuses: `{'accepted': 41}`
- Best-yaw counts: `{'0': 16, '1': 15, '2': 7, '3': 3}`
- Vertex sources: `{'affine': 75, 'projective': 7}`
- Hard failures: `projective_residual_norm=0.0273; max 0.0250`: `1`
- Warnings: `projective_residual_norm=0.0189; warning 0.0180`: `1`, `projective_residual_norm=0.0229; warning 0.0180`: `1`, `sticker_score_total=706.5; warning 700.0`: `1`, `sticker_score_total=712.6; warning 700.0`: `1`, `sticker_score_total=728.7; warning 700.0`: `1`, `sticker_score_worst_face=351.6; warning 350.0`: `1`, `sticker_score_worst_face=357.4; warning 350.0`: `1`, `sticker_score_worst_face=360.3; warning 350.0`: `1`

## Candidate Fallback Diagnostics

- Candidate categories: `{'needs_manual_review': 4, 'reject_retake': 37}`
- Candidate failed checks: `B_count_not_9`: `26`, `D_count_not_9`: `26`, `R_count_not_9`: `26`, `L_center_invalid`: `25`, `U_count_not_9`: `23`, `L_count_not_9`: `22`, `F_center_invalid`: `21`, `F_count_not_9`: `20`, `R_center_invalid`: `19`, `U_center_invalid`: `10`, `B_center_invalid`: `8`, `D_center_invalid`: `4`

Capture / rollout guidance buckets:
- `37`: face counts/centers invalid; inspect color classifier and slot assignment
- `5`: rectified sticker score high; improve lighting/focus/glare
- `2`: hull/projective residual high; avoid background edges and steep tilt

## Per-Pair Snapshot

| Set | Off | Prefer candidate | Prefer effective | Selected | Candidate category | Top failed checks |
|---|---:|---:|---:|---:|---|---|
| 12 | 2 | None | 2 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 14 | 4 | None | 4 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_count_not_9` |
| 15 | 0 | None | 0 | False | `reject_retake` | `B_center_invalid, piece_legality_invalid` |
| 23 | 0 | None | 0 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 24 | 0 | None | 0 | False | `reject_retake` | `B_center_invalid, F_center_invalid` |
| 26 | 0 | None | 0 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 27 | 12 | None | 12 | False | `reject_retake` | `B_center_invalid, F_center_invalid, L_center_invalid` |
| 28 | 24 | None | 24 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 29 | 0 | None | 0 | False | `reject_retake` | `B_count_not_9, D_count_not_9, L_center_invalid` |
| 31 | None | None | None | False | `reject_retake` | `B_center_invalid, L_center_invalid` |
| 32 | 0 | 0 | 0 | True | `needs_manual_review` | `` |
| 36 | 0 | None | 0 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 37 | 0 | None | 0 | False | `reject_retake` | `B_count_not_9, D_count_not_9, L_count_not_9` |
| 42 | 0 | 0 | 0 | True | `needs_manual_review` | `` |
| 44 | None | None | None | False | `reject_retake` | `B_count_not_9, D_center_invalid, D_count_not_9` |
| 20 | 0 | None | 0 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_count_not_9` |
| 38 | 6 | None | 6 | False | `reject_retake` | `B_count_not_9, D_count_not_9, L_count_not_9` |
| 40 | 0 | None | 0 | False | `reject_retake` | `B_center_invalid, U_center_invalid` |
| 41 | 0 | None | 0 | False | `reject_retake` | `F_center_invalid, U_center_invalid, piece_legality_invalid` |
| 43 | 0 | None | 0 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_count_not_9` |
| 45 | 0 | 0 | 0 | True | `needs_manual_review` | `` |
| 17 | None | None | None | False | `reject_retake` | `B_center_invalid, D_center_invalid` |
| 21 | 0 | None | 0 | False | `reject_retake` | `D_center_invalid, R_center_invalid, U_center_invalid` |
| 22 | 0 | None | 0 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 25 | None | None | None | False | `reject_retake` | `B_count_not_9, D_center_invalid, D_count_not_9` |
| 30 | 36 | None | 36 | False | `reject_retake` | `F_center_invalid, L_center_invalid` |
| 39 | None | 0 | 0 | True | `needs_manual_review` | `` |
| 46 | None | None | None | False | `reject_retake` | `B_center_invalid, F_center_invalid, L_center_invalid` |
| 47 | None | None | None | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 48 | None | None | None | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_count_not_9` |
| 49 | None | None | None | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_count_not_9` |
| 57 | 25 | None | 25 | False | `reject_retake` | `B_count_not_9, D_count_not_9, L_center_invalid` |
| 58 | 12 | None | 12 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 61 | 20 | None | 20 | False | `reject_retake` | `B_center_invalid, U_center_invalid` |
| 62 | 0 | None | 0 | False | `reject_retake` | `B_count_not_9, D_count_not_9, L_center_invalid` |
| 63 | 0 | None | 0 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 64 | None | None | None | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 65 | None | None | None | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 66 | None | None | None | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 67 | 12 | None | 12 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
| 68 | 0 | None | 0 | False | `reject_retake` | `B_count_not_9, D_count_not_9, F_center_invalid` |
