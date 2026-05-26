# Fresh GT constrained-inference generalization

Diagnostic-only. This report compares the old `/api/recognize` baseline pinned in
`tests/fixtures/corpus_manifest.json` against the current hidden Fixer-style
constrained-inference path on the 20 freshly added GT rows.

Git head: `fb81baf65a12ef864ae4c07ea6691621dc0f7ddf`
Generated: `2026-05-26T23:18:55.766308+00:00`

Command:

```bash
.venv/bin/python tools/diagnose_pair_threshold_repair.py --only-sets 8 9 10 11 13 16 18 19 33 34 35 50 51 52 53 54 55 56 59 60 --out-json /tmp/fresh_gt_pair_threshold.json --report /tmp/fresh_gt_pair_threshold.md
```

## Headline

| Path | Clean/exact | Legal | <=3 stickers | Notes |
|---|---:|---:|---:|---|
| Old recognizer baseline | 2/20 (10%) success_clean | n/a | n/a | Manifest baseline: 6 manual-review, 12 retake |
| Constrained recommended path | 19/20 (95%) exact | 19/20 (95%) | 20/20 (100%) | Guarded pair-threshold + deterministic repair recommendation |
| Broad legal candidate | 20/20 (100%) exact | 20/20 (100%) | 20/20 (100%) | Diagnostic ceiling only; not the accepted production recommendation |

## Per Set

| Set | Baseline category | Baseline score | Selected thresholds | Recommended hamming | Valid | Broad-legal hamming | Note |
|---:|---|---:|---|---:|---|---:|---|
| 8 | `needs_manual_review` | 54 | `{'A': 192, 'B': 128}` | 0 | true | 0 |  |
| 9 | `needs_manual_review` | 54 | `{'A': 224, 'B': 160}` | 0 | true | 0 |  |
| 10 | `success_clean` | 54 | `{'A': 64, 'B': 224}` | 0 | true | 0 |  |
| 11 | `needs_manual_review` | 39 | `{'A': 160, 'B': 128}` | 2 | false | 0 | remaining recommended miss |
| 13 | `success_clean` | 54 | `{'A': 224, 'B': 128}` | 0 | true | 0 |  |
| 16 | `needs_manual_review` | 34 | `{'A': 128, 'B': 128}` | 0 | true | 0 |  |
| 18 | `reject_retake` | 0 | `{'A': 64, 'B': 128}` | 0 | true | 0 |  |
| 19 | `needs_manual_review` | 48 | `{'A': 224, 'B': 64}` | 0 | true | 0 |  |
| 33 | `needs_manual_review` | 54 | `{'A': 224, 'B': 192}` | 0 | true | 0 |  |
| 34 | `reject_retake` | 0 | `{'A': 192, 'B': 160}` | 0 | true | 0 |  |
| 35 | `reject_retake` | 0 | `{'A': 160, 'B': 128}` | 0 | true | 0 |  |
| 50 | `reject_retake` | 0 | `{'A': 224, 'B': 160}` | 0 | true | 0 | GAN brand, checkerboard |
| 51 | `reject_retake` | 0 | `{'A': 192, 'B': 64}` | 0 | true | 0 | GAN brand |
| 52 | `reject_retake` | 0 | `{'A': 160, 'B': 128}` | 0 | true | 0 |  |
| 53 | `reject_retake` | 0 | `{'A': 192, 'B': 224}` | 0 | true | 0 |  |
| 54 | `reject_retake` | 0 | `{'A': 128, 'B': 224}` | 0 | true | 0 |  |
| 55 | `reject_retake` | 0 | `{'A': 192, 'B': 160}` | 0 | true | 0 |  |
| 56 | `reject_retake` | 0 | `{'A': 128, 'B': 64}` | 0 | true | 0 |  |
| 59 | `reject_retake` | 0 | `{'A': 128, 'B': 224}` | 0 | true | 0 |  |
| 60 | `reject_retake` | 0 | `{'A': 160, 'B': 192}` | 0 | true | 0 |  |

## Interpretation

- The fresh out-of-corpus rows are not overfit-only evidence: the constrained path is exact on 19/20 rows and within two stickers on the remaining row.
- The old recognizer baseline is intentionally much weaker on the same rows: only Sets 10 and 13 are `success_clean`.
- GAN Sets 50/51, including the checkerboard GAN 50 row, are exact under the constrained path even though the old baseline rejects them.
- Set 11 is the useful remaining tail: the recommended canonical count repair is 52/54 and invalid, while the broad legal candidate contains the GT-exact state but is rejected by the current guard. That makes Set 11 a good next visual/debug target before broadening legal repair acceptance.

## Caveats

- This exercises the hidden Fixer-style constrained path, not the default production `/api/recognize` route.
- `broadLegalCandidate` uses a wider legal-repair search and is reported as a diagnostic ceiling, not as a production-safe recommendation.
