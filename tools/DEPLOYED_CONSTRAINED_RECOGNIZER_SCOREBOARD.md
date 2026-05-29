# Deployed constrained recognizer scoreboard

Generated: `2026-05-29T11:01:06.777394+00:00`
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
| `latencyMs` | 71 | 2165.0 | 2532.0 | 2800.0 | 7304.0 | 2794.9 |
| `recognizeTotalMs` | 71 | 1873.5 | 2058.9 | 2220.47 | 6834.43 | 2323.22 |
| `prepareTotalMs` | 71 | 1873.37 | 2058.68 | 2220.32 | 6833.14 | 2322.99 |
| `prepareConstrainedInputMs` | 71 | 1873.48 | 2058.88 | 2220.45 | 6834.4 | 2323.2 |
| `importsMs` | 71 | 0.02 | 0.02 | 0.03 | 0.09 | 0.02 |
| `rembgSessionMs` | 71 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 |
| `loadImagesMs` | 71 | 358.83 | 375.82 | 389.42 | 395.33 | 376.65 |
| `rembgAMs` | 71 | 232.01 | 325.12 | 343.95 | 392.73 | 324.8 |
| `rembgBMs` | 71 | 382.81 | 428.76 | 496.76 | 548.54 | 440.5 |
| `hullFitAMs` | 71 | 518.12 | 709.18 | 802.59 | 899.15 | 717.28 |
| `hullFitBMs` | 71 | 615.25 | 707.1 | 798.77 | 873.64 | 717.69 |
| `selectGuardedPairMs` | 71 | 115.1 | 192.24 | 209.49 | 4938.7 | 437.19 |

## Non-exact Rows

_None._
