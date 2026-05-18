# `rembg` foundation-model proposers

Follow-up to PR #127 (auto-geometry evaluation framework). Adds 5 new
proposers backed by background-removal foundation models (U²-Net,
BiRefNet via the `rembg` package), plus a diagnostic tool that answers
Codex's PR #128 follow-up question: "which automatic hull/proposer would
have rejected the recognizer's bad grid?"

## What's added

- **5 new proposers** in `tools/propose_geometry_labels.py` via a
  parameterized `RembgProposer` class:
  - `rembg_u2net_hull` — precise convex hull from U²-Net mask, no face quads
  - `rembg_u2net_hexagon` — 6-vertex hexagon fit + template-formula face quads
  - `rembg_u2net_hybrid` — precise hull + hexagon-derived face quads
  - `rembg_birefnet_general_hull` — same but BiRefNet (~400MB model, slightly more precise)
  - `rembg_birefnet_general_hybrid` — same
- **`tools/diagnose_grid_rejection.py`** — reads Codex's hard-case /
  corpus / label-baseline artifacts (`/tmp/ctvd-*-current.json`),
  cross-references against any proposer in `runs/auto_geometry_report.json`,
  and prints a "would this proposer have rejected the recognizer's
  chosen grids?" table.

## Headline results (60-target sweep)

| proposer | hullIoU | hull≥0.85 pass | face IoU | face≥0.75 pass |
|---|---|---|---|---|
| recognizer_grids | 0.851 | 60% (36/60) | 0.364 | 0% |
| rembg_u2net_hull | **0.962** | **100% (60/60)** | — | — |
| rembg_u2net_hybrid | **0.962** | **100% (60/60)** | 0.565 | 5% (3/60) |
| rembg_birefnet_general_hull | **0.967** | **100% (60/60)** | — | — |
| rembg_birefnet_general_hybrid | **0.967** | **100% (60/60)** | 0.555 | 3% (2/60) |

**Cube hull localization is solved** — rembg_u2net_hull gets 100% pass
at hullIoU≥0.85, mean 0.962. Worst case 0.923 (still above the gate).
BiRefNet gives +0.005 marginal precision at 5-10× compute cost — U²-Net
is the right default.

**Face quad accuracy is NOT solved by either** — bottleneck is in the
hexagon → face quad post-processing, not mask quality. The marginal
improvement in mask precision (BiRefNet vs U²-Net) actually *hurts*
face IoU slightly (0.555 vs 0.565) because the more detailed mask
catches small surface artifacts that confuse the angular-sector
hexagon extraction.

## Codex's follow-up answered

For each of Codex's named failure cases (Sets 46/47/48/49 A & B,
"background_sticker_noise" failure class), **`rembg_u2net_hull` rejects
all 6 of the recognizer's chosen grids on every case** — used as a
pre-acceptance filter, it would have caught every wood-grain failure
where the recognizer was wrongly picking grids that extend into the
background.

Reproduce:

```bash
.venv/bin/python tools/diagnose_grid_rejection.py \
  --only-sets 46 47 48 49 30 39 27 28 25 17
```

The classical baselines (saturation_*, roi_bbox) also reject these but
they reject *everything indiscriminately* (would false-positive on
success cases). `rembg_u2net_hull` is the clean filter: high hull IoU
(0.96+) AND high rejection rate on actual failures.

## Cost

- U²-Net model: ~176MB, downloaded automatically on first call to
  `~/.u2net/u2net.onnx`
- BiRefNet-general: ~400MB
- New runtime dep: `pip install rembg onnxruntime`
- Inference: U²-Net ~500ms-1s per image (CPU); BiRefNet 3-5× slower
- Per-sweep cost: cached `analyze_image` + cached rembg sessions; ~7
  minutes for U²-Net-only across 60 targets, ~25 minutes adding BiRefNet

## Not in this PR

- **Integration into production recognizer** — none of these proposers
  touch `rubik_recognizer/`. The diagnostic shows that wiring
  `rembg_u2net_hull` as a pre-acceptance filter on grid candidates
  would catch the wood-grain failure cases, but actually doing that is
  a separate PR with proper corpus-contract gates.
- **InSPyReNet evaluation** — BiRefNet's marginal gain suggests
  InSPyReNet's would also be marginal for hull. Defer unless we get a
  reason to push hullIoU past 0.97.
- **Face-quad accuracy improvement** — the open problem. Needs better
  hexagon fit (RANSAC?) or a learned face-corner regressor. Out of
  scope for this PR.

## Tests

`tests/test_auto_geometry_metrics.py` from PR #127 covers the metrics
pipeline. The new rembg proposers are integration-tested by the full
sweep; explicit unit tests would require mocking the ONNX session,
which has limited ROI. Add if a regression appears.
