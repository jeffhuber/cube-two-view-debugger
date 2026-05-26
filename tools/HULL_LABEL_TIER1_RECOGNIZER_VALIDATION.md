# Hull-Label Tier 1 Recognizer Validation

## Purpose

This report runs the recognizer itself on corpus A+B pairs after the
Hull-Label Tier 1 path and direct yaw assembly wiring. Unlike the
geometry-only shadow validation, this includes the production-shaped
face identity, color repair, legal-state, fallback, and diagnostic
signals that cube-snap would consume.

Git head: `87f9282d38b3aad7ff4de733fe54adcb4f458dd7`
Generated: `2026-05-26T00:05:00.889834+00:00`

## Summary By Mode

| Mode | Pairs | Success | Legal | Exact | Mean stickers | Sticker acc |
|---|---:|---:|---:|---:|---:|---:|
| `off` | 46 | 29 | 29 | 19 | 30.717 | 56.9% |
| `shadow` | 46 | 29 | 29 | 19 | 30.717 | 56.9% |
| `prefer_candidate` | 46 | 4 | 4 | 4 | 4.696 | 8.7% |
| `prefer_effective` | 46 | 30 | 30 | 20 | 31.891 | 59.1% |

`prefer_candidate` is the raw hull-label recognizer candidate. Rejected
candidate rows contribute 0 recognized stickers to its aggregate score;
`prefer_effective` is the production-shaped result after falling back to
legacy unless both hull-label sides are selected and the candidate succeeds.

## Interpretation

- Shadow mode is output-identical to legacy in aggregate: `19` exact / `29` success versus legacy `19` exact / `29` success.
- Hull-label geometry/yaw is usually available: `90/92` sides accepted and yaw accepted on `45` pairs.
- Effective prefer selected `4/46` rows, improved `1`, regressed `0`, and moved exact solves from `19` to `20`.
- Default-on prefer is still premature. The remaining bottleneck is not
  hull geometry; it is color/slot-to-WCA face identity after rectification.

## Prefer Effective Versus Legacy

- Improved hamming: `1`
- Regressed hamming: `0`
- Same hamming/incomplete status: `45`
- Hull-label selected rows: `4`
- Fallback rows: `42`
- Selected setIds: `32, 42, 45, 39`

Improved rows:
- Set 39: `None` -> `0`

## Hull-Label Trace

### Shadow

- Side traces: `92`
- Accepted sides: `90`
- Selected sides: `0`
- Side statuses: `{'accepted': 90, 'rejected': 2}`
- Yaw statuses: `{'accepted': 45, 'ambiguous': 1}`
- Best-yaw counts: `{'0': 19, '1': 15, '2': 9, '3': 3}`
- Vertex sources: `{'affine': 84, 'projective': 8}`
- Hard failures: `projective_residual_norm=0.0273; max 0.0250`: `1`, `sticker_score_total=922.5; max 900.0`: `1`
- Warnings: `projective_residual_norm=0.0189; warning 0.0180`: `1`, `projective_residual_norm=0.0207; warning 0.0180`: `1`, `projective_residual_norm=0.0229; warning 0.0180`: `1`, `sticker_score_total=706.5; warning 700.0`: `1`, `sticker_score_total=711.3; warning 700.0`: `1`, `sticker_score_total=711.6; warning 700.0`: `1`, `sticker_score_total=712.6; warning 700.0`: `1`, `sticker_score_total=728.7; warning 700.0`: `1`

### Prefer Candidate

- Side traces: `92`
- Accepted sides: `90`
- Selected sides: `90`
- Side statuses: `{'accepted': 90, 'rejected': 2}`
- Yaw statuses: `{'accepted': 45, 'ambiguous': 1}`
- Best-yaw counts: `{'0': 19, '1': 15, '2': 9, '3': 3}`
- Vertex sources: `{'affine': 84, 'projective': 8}`
- Hard failures: `projective_residual_norm=0.0273; max 0.0250`: `1`, `sticker_score_total=922.5; max 900.0`: `1`
- Warnings: `projective_residual_norm=0.0189; warning 0.0180`: `1`, `projective_residual_norm=0.0207; warning 0.0180`: `1`, `projective_residual_norm=0.0229; warning 0.0180`: `1`, `sticker_score_total=706.5; warning 700.0`: `1`, `sticker_score_total=711.3; warning 700.0`: `1`, `sticker_score_total=711.6; warning 700.0`: `1`, `sticker_score_total=712.6; warning 700.0`: `1`, `sticker_score_total=728.7; warning 700.0`: `1`

## Candidate Fallback Diagnostics

- Candidate categories: `{'reject_retake': 42, 'needs_manual_review': 4}`
- Candidate failed checks: `B_count_not_9`: `28`, `D_count_not_9`: `28`, `R_count_not_9`: `28`, `L_center_invalid`: `27`, `U_count_not_9`: `25`, `L_count_not_9`: `24`, `F_center_invalid`: `22`, `F_count_not_9`: `22`, `R_center_invalid`: `20`, `U_center_invalid`: `10`, `B_center_invalid`: `9`, `D_center_invalid`: `6`

Capture / rollout guidance buckets:
- `40`: face counts/centers invalid; inspect color classifier and slot assignment
- `10`: rectified sticker score high; improve lighting/focus/glare
- `3`: hull/projective residual high; avoid background edges and steep tilt
- `1`: U/D anchor not reliable; ensure A is white-up and B is yellow-up
- `1`: visible 3-face grid not reliable; improve framing/focus

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
| 69 | None | None | None | False | `reject_retake` | `B_center_invalid, L_center_invalid` |
| 70 | None | None | None | False | `reject_retake` | `image_b_no_reliable_face_triple` |
| 71 | None | None | None | False | `reject_retake` | `image_b_D_anchor_missing` |
| 72 | None | None | None | False | `reject_retake` | `B_count_not_9, D_center_invalid, D_count_not_9` |
| 73 | None | None | None | False | `reject_retake` | `B_count_not_9, D_center_invalid, D_count_not_9` |
