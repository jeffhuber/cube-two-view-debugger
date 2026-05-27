# Tier 1 Hull-Label Wiring

`tools.global_cube_model.fit_global_cube_model()` now has a default-off
hull-label candidate path. The legacy Procrustes/PnP/chirality pipeline remains
the default unless a caller opts in.

## Modes

Use the explicit `hull_label_mode` argument when possible. If it is omitted,
the function reads `CUBE_RECOGNIZER_HULL_LABEL_TIER1`.

- `off` / unset: legacy behavior only.
- `shadow`: run the hull-label candidate, attach `model.debug["hull_label_tier1"]`,
  but always return the legacy model.
- `prefer`: return the hull-label model when acceptance gates pass; otherwise
  fall back to the legacy model and attach the rejection trace.

The hull-label path also requires `hull_label_side="A"` or `"B"` because the
same silhouette positions map to different numbered corners on side A vs side B.

## Acceptance Gates

The production-shaped gates live in `tools/hull_label_acceptance.py`. They are
intentionally based on signals available without human ground truth:

- exactly six hull corners
- side-specific convention sanity
- three vertex estimates
- expected face slots
- vertex-cloud spread
- sticker-score total and worst-face score
- projective residual normalized by hexagon diameter

Any hard failure in `prefer` mode falls back to the legacy model. Warnings are
recorded in the trace but do not force fallback.

## Trace

When enabled, `model.debug["hull_label_tier1"]` records:

- mode, side, status, selected/fallback decision
- hard failures and warnings from the acceptance gate
- score totals and per-face score
- selected vertex plus affine/projective candidates
- raw vertex estimates
- hexagon diameter, spread, normalized spread, and projective residual
- `slot_center_faces` with the classified center color/face for each
  `upper` / `right` / `front` slot
- `corner_0..corner_5` and face quads by slot

This is the handoff surface for shadow-mode audits and cube-snap capture UX:
bad-hull rows should show projective residual or sticker-score failures, while
perspective-heavy rows should show high vertex spread and a projective vertex
selection.

When both image traces include slot-center observations, recognition signals
also include `hullLabelTier1Yaw`. This is the center-color yaw inference from
`tools/hull_label_yaw.py`: score yaw candidates 0..3 by the six visible center
faces, accept only with at least 5/6 matches and margin >= 2, and expose the
winning `yawQuarterTurns` plus candidate scores. It is diagnostic/hidden-path
metadata today.

In hidden `prefer` mode, accepted center-color yaw now also enables a direct
slot/yaw candidate assembled through `tools/hull_label_assembly.py` and
`corner_conventions`, bypassing the legacy visible-face identity search for
that candidate. The normal legality checks, pair fallback, and default-off
feature flag still apply: if the direct candidate does not produce a legal
state, `prefer` falls back to the legacy recognizer result.

## Constrained-Inference API Modes

`/api/recognize?hullLabelTier1=constrained-shadow` runs the legacy recognizer
result, then evaluates the rectified hull-label constrained-inference payload
and attaches `recognitionSignals.constrainedInference`; it always returns the
legacy state.

`/api/recognize?hullLabelTier1=constrained` returns the constrained candidate
only when `tools/constrained_inference_gate.py` accepts it. If the gate rejects
or rectification fails, the endpoint falls back to the legacy recognizer and
records the rejection/error in `recognitionSignals.constrainedInference`.

These modes are hidden and default-off. They are the shadow/candidate bridge
between the Fixer-side rectified path and any future default recognizer flip.

To run the constrained path in shadow for every `/api/recognize` request without
changing callers, start the server with:

```bash
CUBE_CONSTRAINED_INFERENCE_MODE=shadow .venv/bin/python app.py
```

The env var is intentionally opt-in and default-off. Explicit request query
params still win: `?hullLabelTier1=off` disables the constrained path for that
request, and `?hullLabelTier1=constrained` runs the candidate-return mode.
Accepted values are `shadow`/`constrained-shadow` for log-only shadowing and
`prefer`/`constrained` for gate-controlled candidate return.

When either constrained mode runs, the server appends one compact JSONL event
to `runs/constrained_inference_shadow.jsonl` by default. Set
`CUBE_CONSTRAINED_SHADOW_LOG=/path/to/events.jsonl` to write elsewhere, or set
it to `off` to disable. The event intentionally stores only run metadata,
gate decision, selected repair method, thresholds, yaw, and input hashes; it
does not duplicate images or overlays. This is the real-traffic shadow audit
trail to compare against the 71-pair corpus before any default flip.

Use `tools/summarize_constrained_shadow_log.py` to inspect the JSONL:

```bash
.venv/bin/python tools/summarize_constrained_shadow_log.py \
  --report runs/constrained_inference_shadow_summary.md
```
