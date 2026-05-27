# Pair-level threshold repair diagnostic

Diagnostic-only. This report asks whether A/B mask thresholds should be
chosen after yaw + deterministic repair, rather than independently per side.

Git head: `38629f56fd7c0d3740399263869d127282a1c6b3`
Generated: `2026-05-27T03:59:58.737898+00:00`

## Summary

| Selector | Assembled | Exact | Legal | <=3 stickers | Hamming distribution |
|---|---:|---:|---:|---:|---|
| Current per-side selector | 66 | 65 | 65 | 65 | `{'0': 65, '4': 1}` |
| Aggressive pair-selected by production signals | 66 | 65 | 66 | 66 | `{'0': 65, '2': 1}` |
| Guarded pair-selected by production signals | 66 | 66 | 66 | 66 | `{'0': 66}` |
| Oracle best threshold pair | 66 | 66 | 66 | 66 | `{'0': 66}` |

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
