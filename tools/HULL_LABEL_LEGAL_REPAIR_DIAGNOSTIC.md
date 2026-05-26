# Hull-Label Legal Repair Diagnostic

## Purpose

This diagnostic probes the next constraint layer after deterministic
9-per-color count repair: cubie legality. It reuses the existing
recognizer cubie repair helper on hull-label rectified Lab evidence.

Git head: `21a7dd9d4efd5d3c1049fe0e4e07032f06b40803`
Generated: `2026-05-26T17:07:09.588650+00:00`

## Summary

| Method | Assembled | Legal | Exact | <=3 stickers | Median hamming | Hamming distribution | Median repair cost | Median changes | Max changes |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `canonical_count_repaired` | 46 | 42 | 42 | 45 | 0.0 | `{0: 42, 2: 2, 3: 1, 6: 1}` | None | None | None |
| `conservative_legal_repaired` | 46 | 43 | 43 | 43 | 0 | `{0: 43}` | 0.0 | 0 | 1 |
| `guarded_broad_legal_repaired` | 46 | 44 | 44 | 44 | 0.0 | `{0: 44}` | 0.0 | 0.0 | 3 |
| `broad_legal_repaired` | 46 | 46 | 46 | 46 | 0.0 | `{0: 46}` | 0.0 | 0.0 | 11 |

## Interpretation

- Conservative cubie repair is the safer signal: it only considers the
  normal low-cost color alternatives allowed by the recognizer.
- Broad cubie repair is diagnostic-only: it marks samples as grid samples
  so the helper can consider all color fallbacks. A perfect broad result
  proves the true legal state is rankable by existing constraints, but
  it still needs cost/change/margin gates before production use.
- Guarded broad repair applies a provisional no-ground-truth gate to that
  same broad result: repair cost <= 16
  and repair changes <= 4. This is still
  diagnostic, but it estimates the slice that looks safe enough to consider
  for production confidence gating.
- This diagnostic does not use ground truth for selection. Ground truth is
  only used after each candidate state is selected to compute hamming and
  exact-match metrics.

## Non-Exact / No-Repair Rows

| Set | Count hamming | Conservative hamming | Conservative status | Guarded hamming | Guarded status | Broad hamming | Broad cost | Broad changes |
|---:|---:|---:|---|---:|---|---:|---:|---:|
| 68 | 2 | None | `no_legal_repair` | 0 | `accepted_guarded_broad_legal_repair` | 0 | 10.0137 | 3 |
| 69 | 3 | 0 | `legal_repair_found` | 0 | `accepted_guarded_broad_legal_repair` | 0 | 3.1484 | 1 |
| 70 | 6 | None | `no_legal_repair` | None | `rejected_guarded_broad_legal_repair` | 0 | 33.7985 | 10 |
| 72 | 2 | None | `no_legal_repair` | None | `rejected_guarded_broad_legal_repair` | 0 | 12.6398 | 11 |
