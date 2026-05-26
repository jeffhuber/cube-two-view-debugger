# Hull-Label Legal Repair Diagnostic

## Purpose

This diagnostic probes the next constraint layer after deterministic
9-per-color count repair: cubie legality. It reuses the existing
recognizer cubie repair helper on hull-label rectified Lab evidence.

Git head: `e021b8907fbd8c054c4eeb25f54f58da22fb1ed8`
Generated: `2026-05-26T19:12:59.047344+00:00`

## Summary

| Method | Assembled | Legal | Exact | <=3 stickers | Median hamming | Hamming distribution | Median repair cost | Median changes | Max changes |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `canonical_count_repaired` | 46 | 43 | 43 | 45 | 0.0 | `{0: 43, 2: 1, 3: 1, 4: 1}` | None | None | None |
| `conservative_legal_repaired` | 46 | 44 | 44 | 44 | 0.0 | `{0: 44}` | 0.0 | 0.0 | 1 |
| `guarded_broad_legal_repaired` | 46 | 45 | 45 | 45 | 0 | `{0: 45}` | 0.0 | 0 | 4 |
| `broad_legal_repaired` | 46 | 46 | 45 | 45 | 0.0 | `{0: 45, 4: 1}` | 0.0 | 0.0 | 5 |

## Interpretation

- Conservative cubie repair is the safer signal: it only considers the
  normal low-cost color alternatives allowed by the recognizer.
- Broad cubie repair is diagnostic-only: it marks samples as grid samples
  so the helper can consider all color fallbacks. A perfect broad result
  proves the true legal state is rankable by existing constraints, but
  it still needs cost/change/margin gates before production use.
- Guarded broad repair applies a provisional no-ground-truth gate to that
  same broad result: repair cost <= 20
  and repair changes <= 4. This is still
  diagnostic, but it estimates the slice that looks safe enough to consider
  for production confidence gating.
- This diagnostic does not use ground truth for selection. Ground truth is
  only used after each candidate state is selected to compute hamming and
  exact-match metrics.

## Non-Exact / No-Repair Rows

| Set | Count hamming | Conservative hamming | Conservative status | Guarded hamming | Guarded status | Broad hamming | Broad cost | Broad changes |
|---:|---:|---:|---|---:|---|---:|---:|---:|
| 14 | 4 | None | `no_legal_repair` | None | `rejected_guarded_broad_legal_repair` | 4 | 26.6913 | 5 |
| 65 | 2 | None | `no_legal_repair` | 0 | `accepted_guarded_broad_legal_repair` | 0 | 19.5798 | 4 |
| 69 | 3 | 0 | `legal_repair_found` | 0 | `accepted_guarded_broad_legal_repair` | 0 | 3.1484 | 1 |
