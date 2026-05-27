# Constrained Recognize Mode Validation

Diagnostic-only. This report runs the hidden
`/api/recognize?hullLabelTier1=constrained` path on the corpus and
compares it with the unchanged legacy/default recognizer path.

Git head: `d6577882fbac4684ba4df582a3bbb51aacb14d72`
Generated: `2026-05-27T18:32:43.078967+00:00`
Manifest: `tests/fixtures/corpus_manifest.json`

## Summary By Mode

| Mode | Pairs | Success | Legal | Exact | <=3 | Mean stickers | Sticker acc | Hamming distribution |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `legacy` | 71 | 37 | 37 | 24 | 25 | 25.408 | 47.1% | `{'0': 24, '12': 3, '15': 1, '2': 1, '20': 2, '24': 1, '25': 1, '36': 1, '4': 1, '6': 2}` |
| `constrained` | 71 | 71 | 71 | 71 | 71 | 54.0 | 100.0% | `{'0': 71}` |

## Gate Behavior

- Selected constrained candidate: `71/71`
- Fell back to legacy: `0/71`
- Gate accepted: `71/71`
- Threshold switches: `1`
- Recommended methods: `{'canonical_count_repaired': 67, 'conservative_legal_repaired': 2, 'guarded_broad_legal_repaired': 1, 'two_view_consistency_repaired': 1}`
- Selection reasons: `{'kept_current_valid_repair': 70, 'current_invalid_selected_best_pair': 1}`

## Delta Versus Legacy

- Legacy exact: `24/71`
- Constrained exact: `71/71`
- Improved hamming: `47`
- Regressed hamming: `0`
- Same hamming/incomplete status: `24`

Improved rows:
- Set 11: `15` -> `0` (`guarded_broad_legal_repaired`)
- Set 12: `2` -> `0` (`canonical_count_repaired`)
- Set 14: `4` -> `0` (`canonical_count_repaired`)
- Set 16: `20` -> `0` (`canonical_count_repaired`)
- Set 17: `None` -> `0` (`canonical_count_repaired`)
- Set 18: `None` -> `0` (`canonical_count_repaired`)
- Set 19: `6` -> `0` (`canonical_count_repaired`)
- Set 25: `None` -> `0` (`canonical_count_repaired`)
- Set 27: `12` -> `0` (`canonical_count_repaired`)
- Set 28: `24` -> `0` (`canonical_count_repaired`)
- Set 30: `36` -> `0` (`canonical_count_repaired`)
- Set 31: `None` -> `0` (`canonical_count_repaired`)
- Set 34: `None` -> `0` (`canonical_count_repaired`)
- Set 35: `None` -> `0` (`canonical_count_repaired`)
- Set 38: `6` -> `0` (`canonical_count_repaired`)
- Set 39: `None` -> `0` (`canonical_count_repaired`)
- Set 44: `None` -> `0` (`canonical_count_repaired`)
- Set 46: `None` -> `0` (`canonical_count_repaired`)
- Set 47: `None` -> `0` (`canonical_count_repaired`)
- Set 48: `None` -> `0` (`canonical_count_repaired`)
- Set 49: `None` -> `0` (`canonical_count_repaired`)
- Set 50: `None` -> `0` (`canonical_count_repaired`)
- Set 51: `None` -> `0` (`canonical_count_repaired`)
- Set 52: `None` -> `0` (`canonical_count_repaired`)
- Set 53: `None` -> `0` (`canonical_count_repaired`)
- Set 54: `None` -> `0` (`canonical_count_repaired`)
- Set 55: `None` -> `0` (`canonical_count_repaired`)
- Set 56: `None` -> `0` (`canonical_count_repaired`)
- Set 57: `25` -> `0` (`canonical_count_repaired`)
- Set 58: `12` -> `0` (`canonical_count_repaired`)
- Set 59: `None` -> `0` (`conservative_legal_repaired`)
- Set 60: `None` -> `0` (`canonical_count_repaired`)
- Set 61: `20` -> `0` (`canonical_count_repaired`)
- Set 64: `None` -> `0` (`canonical_count_repaired`)
- Set 65: `None` -> `0` (`two_view_consistency_repaired`)
- Set 66: `None` -> `0` (`canonical_count_repaired`)
- Set 67: `12` -> `0` (`canonical_count_repaired`)
- Set 69: `None` -> `0` (`conservative_legal_repaired`)
- Set 70: `None` -> `0` (`canonical_count_repaired`)
- Set 71: `None` -> `0` (`canonical_count_repaired`)
- Set 72: `None` -> `0` (`canonical_count_repaired`)
- Set 73: `None` -> `0` (`canonical_count_repaired`)
- Set 74: `None` -> `0` (`canonical_count_repaired`)
- Set 75: `None` -> `0` (`canonical_count_repaired`)
- Set 76: `None` -> `0` (`canonical_count_repaired`)
- Set 77: `None` -> `0` (`canonical_count_repaired`)
- Set 78: `None` -> `0` (`canonical_count_repaired`)

## Per-Pair Snapshot

| Set | Legacy | Constrained | Selected | Gate | Method | Thresholds |
|---|---:|---:|---:|---|---|---|
| 8 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 192, 'B': 128}` |
| 9 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 160}` |
| 10 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 224}` |
| 11 | 15 | 0 | True | `True` | `guarded_broad_legal_repaired` | `{'A': 160, 'B': 128}` |
| 12 | 2 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 160, 'B': 224}` |
| 13 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 128}` |
| 14 | 4 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 192}` |
| 15 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 192, 'B': 192}` |
| 16 | 20 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 128, 'B': 128}` |
| 17 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 128, 'B': 64}` |
| 18 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 128}` |
| 19 | 6 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 64}` |
| 20 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 64}` |
| 21 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 192, 'B': 64}` |
| 22 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 160}` |
| 23 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 128, 'B': 160}` |
| 24 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 192, 'B': 64}` |
| 25 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 160, 'B': 224}` |
| 26 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 64}` |
| 27 | 12 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 192}` |
| 28 | 24 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 160, 'B': 224}` |
| 29 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 160}` |
| 30 | 36 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 192}` |
| 31 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 192, 'B': 192}` |
| 32 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 160, 'B': 160}` |
| 33 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 192}` |
| 34 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 192, 'B': 160}` |
| 35 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 160, 'B': 128}` |
| 36 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 160, 'B': 128}` |
| 37 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 192}` |
| 38 | 6 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 224}` |
| 39 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 192}` |
| 40 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 128}` |
| 41 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 192}` |
| 42 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 64}` |
| 43 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 128, 'B': 224}` |
| 44 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 160}` |
| 45 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 128, 'B': 192}` |
| 46 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 64}` |
| 47 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 192, 'B': 128}` |
| 48 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 160, 'B': 224}` |
| 49 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 192, 'B': 224}` |
| 50 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 160}` |
| 51 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 192, 'B': 64}` |
| 52 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 160, 'B': 128}` |
| 53 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 192, 'B': 224}` |
| 54 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 128, 'B': 224}` |
| 55 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 192, 'B': 160}` |
| 56 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 128, 'B': 64}` |
| 57 | 25 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 192, 'B': 64}` |
| 58 | 12 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 128, 'B': 64}` |
| 59 | None | 0 | True | `True` | `conservative_legal_repaired` | `{'A': 128, 'B': 224}` |
| 60 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 160, 'B': 192}` |
| 61 | 20 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 128, 'B': 224}` |
| 62 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 192}` |
| 63 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 192}` |
| 64 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 160, 'B': 192}` |
| 65 | None | 0 | True | `True` | `two_view_consistency_repaired` | `{'A': 64, 'B': 224}` |
| 66 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 160}` |
| 67 | 12 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 64}` |
| 68 | 0 | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 160}` |
| 69 | None | 0 | True | `True` | `conservative_legal_repaired` | `{'A': 64, 'B': 160}` |
| 70 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 224}` |
| 71 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 64}` |
| 72 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 224}` |
| 73 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 224, 'B': 192}` |
| 74 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 64}` |
| 75 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 128}` |
| 76 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 160, 'B': 192}` |
| 77 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 64}` |
| 78 | None | 0 | True | `True` | `canonical_count_repaired` | `{'A': 64, 'B': 160}` |

## Interpretation

- This report validates the hidden runtime mode added for staged
  recognizer rollout. It does not flip the default `/api/recognize` path.
- A clean result here means the shared constrained-inference gate is
  behaving at the recognizer boundary, not just in offline repair traces.
- Any default promotion should still run in shadow first on real traffic
  and preserve fallback/manual-review behavior when the gate rejects.
