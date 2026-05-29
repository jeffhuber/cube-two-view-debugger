# Deployed constrained recognizer scoreboard

Generated: `2026-05-29T08:31:58.094763+00:00`
Endpoint: `https://api.cubesnap.app/api/recognize?slim=1&hullLabelTier1=constrained`
Manifest: `/Users/jhuber/cube-two-view-debugger/tests/fixtures/corpus_manifest.json`

## Summary

| Rows | Scored | Exact | Within 3 | Rejected | Missing local files |
|---:|---:|---:|---:|---:|---:|
| 71 | 71 | 71 | 71 | 0 | 0 |

Recommended methods:

- `canonical_count_repaired`: `67`
- `conservative_legal_repaired`: `2`
- `guarded_broad_legal_repaired`: `1`
- `two_view_consistency_repaired`: `1`

Two-view repair statuses:

- `accepted_two_view_consistency_repair`: `2`
- `rejected_two_view_consistency_repair`: `69`

Performance schemas:

- `constrained_recognize_performance_v1`: `71`

Timing summary:

| Metric | Count | Min | P50 | P90 | Max | Avg |
|---|---:|---:|---:|---:|---:|---:|
| `latencyMs` | 71 | 1983.0 | 2798.0 | 3789.0 | 8021.0 | 3041.08 |
| `recognizeTotalMs` | 71 | 1680.48 | 2360.65 | 3278.5 | 7446.44 | 2628.7 |
| `prepareTotalMs` | 71 | 1680.4 | 2360.57 | 3278.42 | 7446.31 | 2628.6 |
| `prepareConstrainedInputMs` | 71 | 1680.46 | 2360.63 | 3278.48 | 7446.42 | 2628.68 |
| `importsMs` | 71 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 |
| `rembgSessionMs` | 71 | 0.0 | 0.01 | 0.01 | 0.01 | 0.01 |
| `loadImagesMs` | 71 | 260.93 | 272.29 | 282.04 | 290.04 | 273.17 |
| `rembgAMs` | 71 | 376.69 | 866.61 | 1315.98 | 2027.16 | 900.91 |
| `rembgBMs` | 71 | 399.07 | 699.28 | 1103.59 | 1786.29 | 760.02 |
| `hullFitAMs` | 71 | 118.69 | 203.72 | 276.76 | 299.32 | 195.41 |
| `hullFitBMs` | 71 | 113.56 | 208.78 | 287.71 | 297.95 | 215.41 |
| `selectGuardedPairMs` | 71 | 31.12 | 32.63 | 34.3 | 4360.4 | 283.53 |

## Non-exact Rows

_None._
