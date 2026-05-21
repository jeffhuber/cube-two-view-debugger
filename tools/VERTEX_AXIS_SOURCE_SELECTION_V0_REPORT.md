# Vertex/Axis Source Selection V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This report tests whether existing global-model fit signals can choose between rembg- and SAM3-driven vertex/axis hypotheses, and abstain when confidence is low.

## Summary

- Rows: 23
- Strict threshold: 30 px
- Plausible threshold: 50 px

| Policy | Selected | Abstained | Best-source correct | Strict-ready | Plausible | False-confident >50px | Mean selected error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `always_rembg` | 23 | 0 | 8 | 0 | 5 | 18 | 78.8 px |
| `always_sam3` | 23 | 0 | 15 | 6 | 9 | 14 | 72.2 px |
| `global_model_score_v0` | 23 | 0 | 17 | 6 | 8 | 15 | 64.4 px |
| `strict_residual_margin_confidence_v0` | 1 | 22 | 1 | 1 | 1 | 0 | 20.0 px |
| `high_fit_quality_margin_confidence_v0` | 1 | 22 | 1 | 1 | 1 | 0 | 20.0 px |
| `oracle_best_source` | 23 | 0 | 23 | 6 | 11 | 12 | 58.7 px |

## Per-Row Readout

| Row | Best source | rembg err | SAM3 err | Score policy | Strict confidence |
|---|---|---:|---:|---|---|
| `12_B` | `sam3` | 37 | 26 | `sam3` / 26.43 | `abstain` /  |
| `14_B` | `sam3` | 33 | 16 | `sam3` / 16.19 | `abstain` /  |
| `15_A` | `sam3` | 64 | 47 | `rembg` / 63.5 | `abstain` /  |
| `15_B` | `sam3` | 72 | 54 | `sam3` / 54.43 | `abstain` /  |
| `17_B` | `rembg` | 110 | 137 | `rembg` / 109.95 | `abstain` /  |
| `21_A` | `rembg` | 39 | 56 | `sam3` / 56.44 | `abstain` /  |
| `21_B` | `rembg` | 76 | 118 | `sam3` / 118.16 | `abstain` /  |
| `24_A` | `sam3` | 88 | 86 | `sam3` / 86.36 | `abstain` /  |
| `26_A` | `rembg` | 73 | 121 | `rembg` / 72.54 | `abstain` /  |
| `26_B` | `sam3` | 81 | 24 | `sam3` / 24.1 | `abstain` /  |
| `28_A` | `sam3` | 124 | 78 | `sam3` / 78.48 | `abstain` /  |
| `28_B` | `rembg` | 103 | 106 | `sam3` / 105.89 | `abstain` /  |
| `29_A` | `sam3` | 57 | 45 | `sam3` / 45.3 | `abstain` /  |
| `30_A` | `sam3` | 40 | 20 | `sam3` / 20.03 | `sam3` / 20.03 |
| `31_A` | `rembg` | 47 | 92 | `rembg` / 47.18 | `abstain` /  |
| `32_A` | `sam3` | 72 | 56 | `sam3` / 56.5 | `abstain` /  |
| `32_B` | `sam3` | 71 | 48 | `rembg` / 70.61 | `abstain` /  |
| `36_B` | `sam3` | 103 | 70 | `sam3` / 69.62 | `abstain` /  |
| `42_B` | `rembg` | 89 | 186 | `rembg` / 89.25 | `abstain` /  |
| `44_A` | `sam3` | 186 | 148 | `sam3` / 148.34 | `abstain` /  |
| `44_B` | `sam3` | 118 | 15 | `sam3` / 14.96 | `abstain` /  |
| `61_A` | `sam3` | 71 | 19 | `sam3` / 18.88 | `abstain` /  |
| `61_B` | `rembg` | 60 | 89 | `sam3` / 88.87 | `abstain` /  |

## Interpretation

- The broad global-model score policy improves source choice versus a fixed source, but it still selects too many bad vertices to trust.
- The conservative residual-margin confidence policy can abstain its way to zero false-confident rows here, but it selects only one row. That is evidence of signal, not a usable recognizer path.
- The oracle across rembg/SAM3 remains far better than the deployable policies, so the missing piece is confidence/source selection, not deterministic face splitting.
- Production wiring should continue to wait. The next useful move is adding richer confidence features or more labeled rows, then re-running this exact source-selection report.
