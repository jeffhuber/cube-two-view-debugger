# Pair-level threshold repair diagnostic

Diagnostic-only. This report asks whether A/B mask thresholds should be
chosen after yaw + deterministic repair, rather than independently per side.

Git head: `e021b8907fbd8c054c4eeb25f54f58da22fb1ed8`
Generated: `2026-05-26T19:58:54.527841+00:00`

## Summary

| Selector | Assembled | Exact | Legal | <=3 stickers | Hamming distribution |
|---|---:|---:|---:|---:|---|
| Current per-side selector | 45 | 44 | 44 | 44 | `{'0': 44, '4': 1}` |
| Aggressive pair-selected by production signals | 45 | 44 | 45 | 45 | `{'0': 44, '2': 1}` |
| Guarded pair-selected by production signals | 45 | 45 | 45 | 45 | `{'0': 45}` |
| Oracle best threshold pair | 45 | 45 | 45 | 45 | `{'0': 45}` |

## Changed Rows

| Set | Current thresholds | Selected thresholds | Current hamming | Selected hamming |
|---|---|---|---:|---:|
| 14 | `{'A': 160, 'B': 160}` | `{'A': 64, 'B': 192}` | 4 | 0 |

## Notes

- The pair selector rank uses only production-available signals:
  valid canonical count repair first, then legal repair candidates, then
  balanced count repair, with repair moves/cost and sticker score as
  tie-breakers.
- The guarded selector keeps the current per-side threshold pair when it
  already yields a valid repaired cube. It only switches threshold pairs
  when current repair is invalid or unavailable.
- `oracleBest` uses ground-truth hamming only to show the ceiling; it is
  not a production selector.
- If pair selection improves exact/legal without regressions, the next
  production-shaped step is to fold the same pair-level search into the
  hidden hull-label Fixer path behind a gate.
