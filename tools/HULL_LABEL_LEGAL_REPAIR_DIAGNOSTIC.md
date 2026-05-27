# Hull-Label Legal Repair Diagnostic

## Purpose

This diagnostic probes the next constraint layer after deterministic
9-per-color count repair: cubie legality. It reuses the existing
recognizer cubie repair helper on hull-label rectified Lab evidence.

Git head: `38629f56fd7c0d3740399263869d127282a1c6b3`
Generated: `2026-05-27T02:43:54.092966+00:00`

## Summary

| Method | Assembled | Legal | Exact | <=3 stickers | Median hamming | Hamming distribution | Median repair cost | Median changes | Max changes | Median state delta | Max state delta |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `canonical_count_repaired` | 66 | 61 | 61 | 65 | 0.0 | `{0: 61, 2: 3, 3: 1, 4: 1}` | None | None | None | None | None |
| `conservative_legal_repaired` | 66 | 63 | 63 | 63 | 0 | `{0: 63}` | 0.0 | 0 | 5 | 0 | 3 |
| `guarded_broad_legal_repaired` | 66 | 65 | 65 | 65 | 0 | `{0: 65}` | 0.0 | 0 | 6 | 0 | 3 |
| `broad_legal_repaired` | 66 | 66 | 65 | 65 | 0.0 | `{0: 65, 4: 1}` | 0.0 | 0.0 | 6 | 0.0 | 6 |

## Interpretation

- Conservative cubie repair is the safer signal: it only considers the
  normal low-cost color alternatives allowed by the recognizer.
- Broad cubie repair is diagnostic-only: it marks samples as grid samples
  so the helper can consider all color fallbacks. A perfect broad result
  proves the true legal state is rankable by existing constraints, but
  it still needs cost/state-delta/margin gates before production use.
- Guarded broad repair applies a no-ground-truth gate to that
  same broad result: repair cost <= 20
  and state delta from `canonical_count_repaired` <= 4.
  `repairChanges` remains diagnostic metadata only because it is measured
  against raw observations, not the count-repaired baseline. This is still
  diagnostic, but it estimates the slice that looks safe enough to consider
  for production confidence gating.
- This diagnostic does not use ground truth for selection. Ground truth is
  only used after each candidate state is selected to compute hamming and
  exact-match metrics.

## Non-Exact / No-Repair Rows

| Set | Count hamming | Conservative hamming | Conservative status | Guarded hamming | Guarded status | Broad hamming | Broad cost | Broad changes | Broad state delta |
|---:|---:|---:|---|---:|---|---:|---:|---:|---:|
| 11 | 2 | None | `no_legal_repair` | 0 | `accepted_guarded_broad_legal_repair` | 0 | 10.1513 | 6 | 2 |
| 14 | 4 | None | `no_legal_repair` | None | `rejected_guarded_broad_legal_repair` | 4 | 26.6913 | 5 | 6 |
| 59 | 2 | 0 | `legal_repair_found` | 0 | `accepted_guarded_broad_legal_repair` | 0 | 6.4646 | 5 | 2 |
| 65 | 2 | None | `no_legal_repair` | 0 | `accepted_guarded_broad_legal_repair` | 0 | 19.5798 | 4 | 2 |
| 69 | 3 | 0 | `legal_repair_found` | 0 | `accepted_guarded_broad_legal_repair` | 0 | 3.1484 | 1 | 3 |
