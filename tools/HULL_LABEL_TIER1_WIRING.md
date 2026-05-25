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
- `corner_0..corner_5` and face quads by slot

This is the handoff surface for shadow-mode audits and cube-snap capture UX:
bad-hull rows should show projective residual or sticker-score failures, while
perspective-heavy rows should show high vertex spread and a projective vertex
selection.
