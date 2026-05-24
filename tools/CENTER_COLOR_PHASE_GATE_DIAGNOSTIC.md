# Center-color phase gate diagnostic

Status: diagnostic-only. Do not wire this as a production gate yet.

## What changed

`tools/diagnose_center_color_phase_gate.py` evaluates the production-shaped
center-color idea from PR #262 against actual recognizer geometry. Under
production geometry, the score is **not a pure phase/chirality signal**:
it is center color plus rectification quality. If a fitted face quad is bad,
the sampled "center" patch can contain bezel or pixels from the wrong physical
face, and the lower Lab score may simply identify the less-broken fit.

1. Run the global model with `apply_phase_correction=False` to get the raw
   unflipped phase hypothesis.
2. Build the explicit forced-flip hypothesis from that raw model by using the
   current far corners as the new one-edge axes.
3. Run today's production path separately with `apply_phase_correction=True`.
4. Rectify each model's three visible face quads, sample only the center
   stickers, and score the model's current face assignment by sum of CIELAB
   distance to canonical WCA center colors.
5. Compare the lower-scoring center-color choice with today's production
   category under `tests/fixtures/full_corner_ground_truth.json`.

The support API change is intentionally narrow:

```python
fit_global_cube_model(..., apply_phase_correction=True)
```

The default remains `True`, so existing production behavior is unchanged.

## Latest local diagnostic run

Command:

```bash
PYTHONPATH=. .venv/bin/python tools/diagnose_center_color_phase_gate.py \
  --n-runs 3 \
  --out-json /tmp/center_color_phase_gate_trace_n3.json \
  --out-md /tmp/center_color_phase_gate_report_n3.md
```

Summary from that run:

- Rows: 12 traced / 12 total
- Fully stable rows: 3 / 12
- Center-choice modal counts: `forced_flip:5`, `unflipped:7`
- Effect vs production modal counts:
  `center_choice_would_help:3`, `center_choice_would_hurt:4`,
  `same_as_production:4`, `center_choice_tie:1`
- Production geometry modal counts: `GOOD:4`, `MARGINAL:2`,
  `PHASE_SWAPPED:6`
- Selected geometry modal counts: `GOOD:6`, `MARGINAL:1`,
  `PHASE_SWAPPED:5`

## Interpretation

Center color is useful evidence, but not yet a standalone phase gate.

The good news: it can rescue real phase-swapped rows. In the 3-run diagnostic,
the modal center-color choice improved rows such as `20_B`, `38_A`, and `43_B`
from production `PHASE_SWAPPED` to selected `GOOD`.

The bad news: it also hurts some rows where production was already good
or marginal, and the per-row choice is unstable on most rows. That means
absolute center-color identity score is not enough by itself under recognizer
geometry. Model quads can sample the wrong part of a sticker or face when the
geometry is noisy, so the center-color score is contaminated by geometry error.

Claude's visual rectification audit sharpened this interpretation:

- `20_A`: one hypothesis produced visibly broken rectified faces spanning
  multiple cube faces, while the other had two usable faces. The color metric
  preferred the less-broken geometry. That is useful, but it is a fit-quality
  signal more than a clean phase signal.
- `41_A`: both hypotheses produced recognizable faces. This is the kind of
  row where center color may be measuring genuine phase evidence.
- `45_B`: both hypotheses produced visibly bad rectifications, so the color
  "winner" is not trustworthy.

So the diagnostic is valuable, but the current result should be read as:

> Center color can expose bad phase/fit combinations, but production
> rectification quality is now the bottleneck. A production override needs a
> rectification-quality gate before center color can safely influence phase.

## Recommendation

Keep this as a diagnostic feature and use it to design a safer compound rule.
Likely next useful probes:

- Require a large center-color margin before overriding production.
- Add a rectification-quality/contact-sheet diagnostic for both hypotheses,
  and only evaluate phase color evidence when both hypotheses produce usable
  face grids.
- Combine center-color evidence with full-corner geometry proxies rather than
  letting color act alone.
- Inspect hurt rows (`40_A`, `41_A`, `41_B`, `45_A`, `45_B`) visually to see
  whether the sampled center patches are off-center, on bezel, or correctly
  sampled but color-ambiguous.
- Reduce upstream nondeterminism in `detect_hexagon_anchors` / model fitting
  before treating repeated diagnostics as stable measurements.
