# Chirality detection in global_cube_model — v0 report

## Bug

The Procrustes brute-force fit in `fit_cube_template_to_anchors` searches
all 720 (6!) permutations of detected hexagon corners → template
positions, picks lowest residual. The cube has a 60° symmetry around its
body diagonal that yields the SAME silhouette with the NEAR/FAR labels
swapped — i.e., 6 symmetry-equivalent permutations all give nearly the
same residual. Which one wins is then determined by tiny noise in the fit,
making the chirality choice effectively non-deterministic.

Discovered via the axis-labeling tool's ground truth (4 user-labeled
cases on 2026-05-22): set 12_A's model output had its near-corner axes
~58° off from the user-labeled directions — a clear 60° flip rather than
a small drift.

## What this PR adds

A diagnostic probe — `_apply_chirality_correction` — that inspects each
of the 6 hexagon corners after the fit:

1. For each of the 6 corners, compute mean pixel **darkness** along the
   line from the model vertex to that corner.
2. Mean of the 3 model-labeled NEAR corners' darkness vs mean of the
   3 model-labeled FAR corners' darkness.
3. **Signed separation** = mean_near − mean_far.
4. Emit `chirality_check` debug status + per-corner darkness +
   separation in `model.debug`.

### Status values

- `correct`               — model.near corners are darker than model.far (signal looks right)
- `flip_suggested_diagnostic_only` — model.near appear lighter than model.far (signal suggests flip)
- `ambiguous_no_correction`        — |separation| < 5 (no clear signal)
- `skipped_no_image`               — no RGB image passed in
- `skipped_missing_corners`        — model didn't populate visible_corners

### Geometric rationale

In a clean iso-projection render:
- vertex→NEAR corner runs along a visible cube edge = DARK BEZEL along
  its full length → high darkness
- vertex→FAR corner cuts across a colored sticker face (face diagonal)
  → lighter

So darker NEAR than FAR means chirality correct; the inverse suggests
the model labeled the FAR positions as near.

## Why diagnostic-only (no auto-correction by default)

Validated against the 4 user-labeled cases (12_A, 12_B, 17_A, 17_B):
the synthetic-image signal is clean and works perfectly (all 8 unit
tests pass) but the **real-photo signal is noisy and frequently
inverted**. Empirically auto-correction makes the geometry WORSE in
3 of 4 cases.

Suspected causes:

1. **Vertex offset.** The model vertex from `fit_cube_template_to_anchors`
   has typical 10–50 px error (the post-PnP mean-of-3 ensemble brings
   it down, but the ensemble is applied AFTER the chirality check
   currently). When the model vertex is shifted from the true cube
   vertex, the line vertex→near doesn't actually lie on the bezel; it
   skims into the adjacent sticker interior.
2. **Dark sticker colors.** Red, blue, dark green stickers along far
   diagonals compete with bezel darkness — the face-diagonal line can
   sample darker pixels than expected.
3. **Lighting glare on bezels.** Reflective bezel plastic + glossy
   stickers cause bright spots on bezels that drag near-line darkness
   down.

## What's next

To enable `apply_correction=True` safely we need either:

- **Robust geometric discriminator.** Distance-from-vertex is degenerate
  in true iso projection (all 6 corners equidistant under orthographic);
  with perspective there's a signal but it's noisy in cases with yaw
  (verified by the probe — 17_A had 1 of 3 user-near corners among the
  3 farthest hex corners). Bezel-line angle matching modulo 180°
  ALSO degenerate: near and far corners share the same 3 lines through
  the vertex.
- **Better image-based discriminator.** Possibly: sample lateral
  perpendicular gradient transitions (a bezel produces sharp
  perpendicular edges; a face diagonal does not). Or: orient detected
  bezel half-lines and match to the model's near-corner directions —
  but this requires reliable bezel detection (current `signal_quality`
  is often < 0.1 on these photos).
- **Larger ground-truth set.** Full axis-labeled gallery (in progress)
  will let us pick a separation threshold that's empirically reliable
  rather than the 5-degree minimum we have now.

## How to consume the diagnostic

```python
model = fit_global_cube_model(detection, image_rgb, mask)
print(model.debug["chirality_check"])
# 'correct' | 'flip_suggested_diagnostic_only' | 'ambiguous_no_correction'
# | 'skipped_no_image' | 'skipped_missing_corners'

print(model.debug["chirality_darkness_separation"])  # signed, in [-255, 255]
print(model.debug["chirality_near_line_darkness"])   # 3 floats
print(model.debug["chirality_far_line_darkness"])    # 3 floats
```

To opt into the experimental auto-correction (use at your own risk):

```python
from tools.global_cube_model import _apply_chirality_correction
corrected, debug = _apply_chirality_correction(
    model, detection, image_rgb, apply_correction=True
)
```
