# Pair-level threshold repair diagnostic

Diagnostic-only. This report asks whether A/B mask thresholds should be
chosen after yaw + deterministic repair, rather than independently per side.

Git head: `0078609caa58f3455805650d576991f655ad2586`
Generated: `2026-05-27T14:48:33.560308+00:00`

## Summary

| Selector | Assembled | Exact | Legal | <=3 stickers | Hamming distribution |
|---|---:|---:|---:|---:|---|
| Current per-side selector | 71 | 70 | 70 | 70 | `{'0': 70, '4': 1}` |
| Aggressive pair-selected by production signals | 71 | 70 | 71 | 71 | `{'0': 70, '2': 1}` |
| Guarded pair-selected by production signals | 71 | 71 | 71 | 71 | `{'0': 71}` |
| Oracle best threshold pair | 71 | 71 | 71 | 71 | `{'0': 71}` |

## Changed Rows

| Set | Current thresholds | Selected thresholds | Current hamming | Selected hamming |
|---|---|---|---:|---:|
| 14 | `{'A': 160, 'B': 160}` | `{'A': 64, 'B': 192}` | 4 | 0 |

## Guarded Rows

| Set | Guarded thresholds | Aggressive thresholds | Guarded hamming | Aggressive hamming | Reason |
|---|---|---|---:|---:|---|
| 73 | `{'A': 224, 'B': 192}` | `{'A': 224, 'B': 128}` | 0 | 2 | `kept_current_valid_repair` |

## Notes

- The pair selector rank uses only production-available signals:
  valid canonical count repair first, then legal repair candidates, then
  balanced count repair, with state-delta/repair moves, cost, and
  sticker score as tie-breakers.
- The guarded selector keeps the current per-side threshold pair when it
  already yields a valid repaired cube. It only switches threshold pairs
  when current repair is invalid or unavailable.
- `oracleBest` uses ground-truth hamming only to show the ceiling; it is
  not a production selector.
- The hidden rectified Fixer path now uses the same guarded pair-level
  threshold gate. If this result holds, the next production-shaped
  decision is whether to validate it beyond Fixer toward broader
  recognizer use.
