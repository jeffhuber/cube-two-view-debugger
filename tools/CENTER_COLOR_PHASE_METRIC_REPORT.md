# Center-color phase metric — validation report

Diagnostic-only. Tests whether sum-of-CIELAB-distance from canonical Rubik's colors, computed over the 3 visible center stickers per side, reliably prefers the **correct** face labeling (per oracle / human-validated yaw + corners) over the two alternative 120°-rotated hypotheses that the chirality detector might otherwise pick.

> ⚠️ **Reproducibility note (Codex P3 on PR #262).** This report's numeric scores depend on the rectified-face oracle's output, which is pinned to the canonical environment per `tests/fixtures/corpus_manifest.json` (native ARM64 macOS, Python 3.12.13, numpy 2.3.5, Pillow 12.2.0). Other architectures or library versions can produce small (~0.1-1 dE) per-row score differences from BLAS path / image decode differences. The qualitative VERDICT is robust under these perturbations (margins are 100+ dE), but downstream parsing of exact numeric values should regenerate this report on the canonical environment via `python tools/build_oracle_rectified_faces.py && python tools/probe_center_color_phase_metric.py`.

Each row has 3 hypothesis scores:

- `identity` — slots labeled per oracle (the correct chirality assignment)
- `cyclic_120` — slot->face assignment rotated 120° around the body diagonal
- `cyclic_240` — slot->face assignment rotated 240° (equivalent to -120°)

Lower score = better match to canonical colors. The metric is sound iff `identity` wins on all 12 rows by a clear margin.

## Headline

- `identity` wins on **12/12** rows
- Of those, **12** wins are strict (margin > 0; the rest are ties)
- Margin when identity wins: median 157.4, min 132.7, max 243.1

## Verdict

**Metric is sound.** Identity wins strictly on all rows. Safe to wire into the production pipeline as a per-row phase-correction gate.

## Per-row scores

| Row | Yaw | Oracle (upper,right,front) | id | cyc120 | cyc240 | Winner | Margin |
|---|---:|---|---:|---:|---:|---|---:|
| `20_A` | 0 | U,R,F | 65.7 | 198.3 | 217.0 | identity | 132.7 |
| `20_B` | 0 | D,B,L | 67.9 | 240.1 | 234.7 | identity | 166.8 |
| `38_A` | 1 | U,B,R | 60.3 | 196.3 | 223.3 | identity | 135.9 |
| `38_B` | 1 | D,L,F | 55.4 | 192.2 | 200.6 | identity | 136.8 |
| `40_A` | 0 | U,R,F | 58.7 | 215.0 | 235.5 | identity | 156.3 |
| `40_B` | 0 | D,B,L | 30.3 | 273.4 | 283.1 | identity | 243.1 |
| `41_A` | 1 | U,B,R | 48.9 | 219.7 | 235.8 | identity | 170.8 |
| `41_B` | 1 | D,L,F | 38.4 | 195.8 | 206.5 | identity | 157.3 |
| `43_A` | 1 | U,B,R | 46.3 | 216.1 | 214.8 | identity | 168.5 |
| `43_B` | 1 | D,L,F | 40.7 | 206.6 | 198.2 | identity | 157.5 |
| `45_A` | 0 | U,R,F | 59.0 | 217.2 | 214.4 | identity | 155.4 |
| `45_B` | 0 | D,B,L | 67.2 | 259.4 | 250.5 | identity | 183.3 |

## Where the margin comes from

For each row, the identity hypothesis assigns each center sticker to its canonical color (e.g. U center → white). The alternative hypotheses assign that same center pixel to a different canonical color (e.g. U center → red), producing a much larger Lab distance because the actual center color is nowhere near red. The metric works because the 6 canonical Rubik's colors are well separated in Lab space (median pairwise distance ≈ 83), much larger than the per-row within-face color variation due to lighting / wear.
