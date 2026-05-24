# Procrustes correspondence diagnostic

This diagnostic reruns the 720-way detected-hexagon-to-template correspondence search and scores every permutation against canonical full-corner truth. It asks whether canonical-good assignments are available but ranked below the minimum-residual assignment.

Canonical categories are triplet-angle categories, not exact vertex/corner point-error categories. This keeps the diagnostic focused on correspondence assignment before PnP, phase correction, and vertex refinement.

## Source

- Tool: `tools/diagnose_procrustes_correspondence.py`
- Truth: `tests/fixtures/full_corner_ground_truth.json`
- Max image dim: `1600`
- Rows glob: `*`
- Search: all 720 detected-hexagon-to-template permutations
- Selection metric: minimum affine residual before PnP/phase correction

## Aggregate

- Rows traced: 12 / 12
- Diagnosis counts: `{'residual_selects_canonical': 12}`
- Selected category counts: `{'GOOD': 12}`
- Median canonical residual RMS gap px: `0.0`

## Per-row summary

| Row | Selected category | Selected RMS px | Canonical rank | Canonical RMS gap px | Canonical aligned mean deg | Diagnosis |
|---|---|---:|---:|---:|---:|---|
| `20_A` | GOOD | 60.843 | 1 | 0.0 | 3.8 | residual_selects_canonical |
| `20_B` | GOOD | 64.433 | 1 | 0.0 | 5.21 | residual_selects_canonical |
| `38_A` | GOOD | 58.366 | 1 | 0.0 | 4.41 | residual_selects_canonical |
| `38_B` | GOOD | 45.359 | 1 | 0.0 | 3.29 | residual_selects_canonical |
| `40_A` | GOOD | 35.72 | 1 | 0.0 | 7.9 | residual_selects_canonical |
| `40_B` | GOOD | 40.353 | 1 | 0.0 | 6.52 | residual_selects_canonical |
| `41_A` | GOOD | 31.973 | 1 | 0.0 | 7.3 | residual_selects_canonical |
| `41_B` | GOOD | 31.468 | 1 | 0.0 | 3.98 | residual_selects_canonical |
| `43_A` | GOOD | 56.815 | 1 | 0.0 | 3.86 | residual_selects_canonical |
| `43_B` | GOOD | 56.591 | 1 | 0.0 | 5.52 | residual_selects_canonical |
| `45_A` | GOOD | 57.925 | 1 | 0.0 | 6.97 | residual_selects_canonical |
| `45_B` | GOOD | 65.967 | 1 | 0.0 | 8.77 | residual_selects_canonical |

## Interpretation

- `residual_selects_canonical`: the current residual objective already selects a GOOD/MARGINAL full-corner assignment.
- `canonical_available_but_outranked`: a canonical assignment exists in the 720 candidates, but residual ranks another permutation first. This points at a correspondence-ranking or bias problem.
- `canonical_absent_or_not_within_threshold`: no permutation scores GOOD/MARGINAL against full-corner truth. This points at hexagon extraction, model shape, or the affine correspondence family itself.
- This instruments the initial affine correspondence layer; downstream PnP, phase correction, and vertex refinement can still help or hurt later.
