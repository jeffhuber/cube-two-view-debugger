# Hull-Label Legal Repair Diagnostic

## Purpose

This diagnostic probes the next constraint layer after deterministic
9-per-color count repair: cubie legality. It reuses the existing
recognizer cubie repair helper on hull-label rectified Lab evidence.

Git head: `fb2cecb939d1c0c9c5a51944d750b5dfb1182ca0`
Generated: `2026-05-26T16:43:15.799173+00:00`

## Summary

| Method | Assembled | Legal | Exact | <=3 stickers | Median hamming | Hamming distribution | Median repair cost | Median changes | Max changes |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `canonical_count_repaired` | 46 | 42 | 42 | 45 | 0.0 | `{0: 42, 2: 2, 3: 1, 6: 1}` | None | None | None |
| `conservative_legal_repaired` | 46 | 43 | 43 | 43 | 0 | `{0: 43}` | 0.0 | 0 | 1 |
| `broad_legal_repaired` | 46 | 46 | 46 | 46 | 0.0 | `{0: 46}` | 0.0 | 0.0 | 11 |

## Interpretation

- Conservative cubie repair is the safer signal: it only considers the
  normal low-cost color alternatives allowed by the recognizer.
- Broad cubie repair is diagnostic-only: it marks samples as grid samples
  so the helper can consider all color fallbacks. A perfect broad result
  proves the true legal state is rankable by existing constraints, but
  it still needs cost/change/margin gates before production use.
- This diagnostic does not use ground truth for selection. Ground truth is
  only used after each candidate state is selected to compute hamming and
  exact-match metrics.

## Non-Exact / No-Repair Rows

| Set | Count hamming | Conservative hamming | Conservative status | Broad hamming | Broad cost | Broad changes |
|---:|---:|---:|---|---:|---:|---:|
| 68 | 2 | None | `no_legal_repair` | 0 | 10.0137 | 3 |
| 69 | 3 | 0 | `legal_repair_found` | 0 | 3.1484 | 1 |
| 70 | 6 | None | `no_legal_repair` | 0 | 33.7985 | 10 |
| 72 | 2 | None | `no_legal_repair` | 0 | 12.6398 | 11 |
