# SAM3 Whole-Cube Silhouette Bakeoff V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This report compares a refined global cube model when the silhouette source is `rembg` versus when it is a cached SAM3 `whole_cube` mask. The score is the fitted visible trihedral vertex distance to the human label.

## Summary

- Compared rows: 23
- Skipped rows: 5
- rembg mean / median error: 78.8 / 72.3 px
- SAM3 mean / median error: 72.2 / 56.5 px
- Mean delta, SAM3 minus rembg: -6.6 px
- SAM3 better by >5 px: 14
- rembg better by >5 px: 7
- Within 5 px tie: 2
- SAM3 large regressions (>30 px worse): 4

## Thresholds

| Threshold | rembg rows below | SAM3 rows below |
|---:|---:|---:|
| <30 px | 0 | 6 |
| <50 px | 5 | 9 |
| <75 px | 13 | 13 |
| <100 px | 17 | 17 |

## Rows

| Row | rembg err | SAM3 err | Delta | Winner | rembg refine | SAM3 refine |
|---|---:|---:|---:|---|---|---|
| `44_B` | 118 | 15 | -103 | `sam3` | `applied` | `applied` |
| `26_B` | 81 | 24 | -57 | `sam3` | `skipped_high_base_score` | `applied` |
| `61_A` | 71 | 19 | -52 | `sam3` | `skipped_high_base_score` | `skipped_high_base_score` |
| `28_A` | 124 | 78 | -46 | `sam3` | `skipped_high_base_score` | `skipped_high_base_score` |
| `44_A` | 186 | 148 | -37 | `sam3` | `applied` | `applied` |
| `36_B` | 103 | 70 | -33 | `sam3` | `skipped_high_base_score` | `skipped_high_base_score` |
| `32_B` | 71 | 48 | -23 | `sam3` | `applied` | `skipped_high_base_score` |
| `30_A` | 40 | 20 | -20 | `sam3` | `skipped_high_base_score` | `skipped_high_base_score` |
| `15_B` | 72 | 54 | -18 | `sam3` | `skipped_high_base_score` | `skipped_high_base_score` |
| `14_B` | 33 | 16 | -17 | `sam3` | `skipped_high_base_score` | `skipped_high_base_score` |
| `15_A` | 64 | 47 | -16 | `sam3` | `skipped_high_base_score` | `skipped_high_base_score` |
| `32_A` | 72 | 56 | -16 | `sam3` | `skipped_high_base_score` | `skipped_high_base_score` |
| `29_A` | 57 | 45 | -12 | `sam3` | `skipped_high_base_score` | `skipped_high_base_score` |
| `12_B` | 37 | 26 | -11 | `sam3` | `skipped_high_base_score` | `skipped_high_base_score` |
| `24_A` | 88 | 86 | -1 | `tie` | `skipped_high_base_score` | `skipped_high_base_score` |
| `28_B` | 103 | 106 | +3 | `tie` | `skipped_high_base_score` | `skipped_high_base_score` |
| `21_A` | 39 | 56 | +18 | `rembg` | `skipped_high_base_score` | `skipped_high_base_score` |
| `17_B` | 110 | 137 | +27 | `rembg` | `skipped_high_base_score` | `skipped_high_base_score` |
| `61_B` | 60 | 89 | +29 | `rembg` | `skipped_high_base_score` | `skipped_high_base_score` |
| `21_B` | 76 | 118 | +42 | `rembg` | `applied` | `skipped_high_base_score` |
| `31_A` | 47 | 92 | +44 | `rembg` | `skipped_high_base_score` | `applied` |
| `26_A` | 73 | 121 | +48 | `rembg` | `applied` | `applied` |
| `42_B` | 89 | 186 | +97 | `rembg` | `applied` | `applied` |

## Interpretation

- SAM3 whole-cube masks are materially more promising than SAM3 face text/box prompts: they improve mean and median vertex error in this paired refined-model bakeoff.
- They are still not safe as a sole silhouette source. The regression tail is real, including several rows where SAM3 is more than 30 px worse than rembg.
- Treat SAM3 whole-cube masks as an alternate geometry hypothesis or cross-check, not production wiring.
- The next path should be geometry-first face splitting from a trusted whole-cube silhouette and vertex/axis model, because SAM3 is useful for object isolation but not for semantic face separation.
