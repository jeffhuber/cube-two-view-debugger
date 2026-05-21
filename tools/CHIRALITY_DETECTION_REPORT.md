# Chirality detection in global_cube_model

## Bug

The Procrustes brute-force fit in `fit_cube_template_to_anchors` searches
all 720 (6!) permutations of detected hexagon corners → template
positions and picks the lowest residual. The cube has a 60° symmetry
around its body diagonal that yields the SAME silhouette with the NEAR/FAR
labels swapped — 6 symmetry-equivalent permutations all give nearly the
same residual. Which one wins is determined by tiny noise in the fit,
making the chirality choice effectively non-deterministic.

Discovered via the axis-labeling tool's ground truth (4 user-labeled
cases on 2026-05-22): set 12_A's model output had its near-corner axes
~58° off from the user-labeled directions — a clear 60° flip rather than
a small drift.

## Detector

`_apply_chirality_correction` inspects each of the 6 hexagon corners
after the fit:

1. For each of the 6 corners, compute mean pixel **darkness** along the
   line from the model vertex to that corner (15–85% of the line, to
   skip endpoints).
2. Mean of the 3 model-labeled NEAR corners' darkness (`mean_near`) vs
   the 3 model-labeled FAR corners' darkness (`mean_far`).
3. **Signed separation** = `mean_near − mean_far`.
4. Emit `chirality_check` status + per-corner darkness + separation in
   `model.debug`.

### Status values

| status                              | meaning                                                                |
|-------------------------------------|------------------------------------------------------------------------|
| `correct`                           | sep < −10 → empirically the chirality is correct                       |
| `corrected_60deg_flip`              | sep > +10 → flipped; rebuilt model with far corners as new axes        |
| `flip_suggested_diagnostic_only`    | sep > +10 but `apply_correction=False` → model left unchanged           |
| `ambiguous_no_correction`           | \|sep\| < 10 → no clear signal                                         |
| `skipped_no_image`                  | no RGB image passed in                                                 |
| `skipped_missing_corners`           | model didn't populate `visible_corners`                                |

## Empirical polarity (validated 2026-05-21 on 58-case axis-labeled gallery)

**The polarity is COUNTER-INTUITIVE.** Naive bezel-darkness reasoning
says vertex→near (along a cube edge) should sample more dark bezel
pixels than vertex→far (across a face diagonal), so sep > 0 should mean
"correct". The data says the opposite:

| pos. truth | sep sign | examples |
|------------|----------|----------|
| CORRECT    | negative | sep ≈ −34, −25, −18, −30, −29 (stable-CORRECT cases) |
| FLIPPED    | positive | sep ≈ +94, +73, +60, +51, +42 (stable-FLIPPED cases) |

So the code uses the EMPIRICAL polarity: `sep < 0` ≡ correct,
`sep > 0` ≡ flipped.

### Validation cross-tab (58 cases × 2 runs = 116 model runs)

Rows = detector says, cols = position-based ground truth (bearings
comparison against user-labeled near corners, scale-invariant):

|             | CORRECT | FLIPPED | AMBIGUOUS | total |
|-------------|--------:|--------:|----------:|------:|
| CORRECT     | 45      | 7       | 2         | 54    |
| FLIPPED     | 5       | 48      | 1         | 54    |
| AMBIG       | 3       | 5       | 0         | 8     |
| **total**   | 53      | 60      | 3         | 116   |

Detector agreement on non-ambiguous truth (113 runs):
- **agree: 93/113 (82.3%)**
- disagree: 12/113 (10.6%)
- detector ambiguous: 8/113 (7.1%)

Base rate of chirality flips in the model: **60/116 = 51.7%** —
essentially a coin flip per Procrustes run.

Case-level stability across 2 runs: only 26/58 cases (45%) are stable;
32/58 cases (55%) flip outcome between runs.

### Suspected cause of the polarity inversion

The model vertex from PnP is typically 10–50 px off from the true cube
vertex (the post-PnP mean-of-3 ensemble brings it down, but the
ensemble is applied AFTER the chirality check). When the model vertex
is shifted from the true cube vertex, the line from model_vertex to
model.h_x (the geometrically NEAR corner) doesn't actually lie on the
cube-edge bezel — it skims off into adjacent sticker interior. Meanwhile
the line to model.h_xy (geometrically FAR via face diagonal) crosses
multiple internal between-sticker bezels at near-perpendicular angles,
racking up MORE total darkness than the off-bezel line to model.h_x.

The empirical polarity reflects this systematic vertex-offset regime
rather than the idealized "bezel-along-edge" reasoning.

## What's next

- **Pre-condition vertex precision.** With a better vertex localizer
  (PRs #207/#208/#209/#211 chained), the off-bezel skim could shrink
  and the polarity might revert toward the geometric ideal. Worth
  re-validating once the vertex KNN pipeline lands at higher accuracy.
- **Deterministic Procrustes tie-breaker.** Even with a 82%-accurate
  detector, fixing the root non-determinism in the 6! brute-force
  would eliminate the symptom entirely. Candidate rule: among the
  permutations within ε of best residual, prefer the one whose
  detected near corners match detected bezel-line angles (mod 180°).
- **Lateral gradient discriminator.** Sample perpendicular to the
  vertex→corner line: a bezel produces sharp transverse edges, a face
  diagonal does not. More robust to vertex offset than along-line
  darkness.

## How to consume

```python
model = fit_global_cube_model(detection, image_rgb, mask)
print(model.debug["chirality_check"])
# 'correct' | 'corrected_60deg_flip' | 'ambiguous_no_correction' | …

print(model.debug["chirality_darkness_separation"])   # signed
print(model.debug["chirality_near_line_darkness"])    # 3 floats
print(model.debug["chirality_far_line_darkness"])     # 3 floats
```

`fit_global_cube_model` now calls `_apply_chirality_correction` with
`apply_correction=True` by default, so the returned model reflects the
chirality-corrected geometry whenever the detector fires. To skip
correction (diagnostic-only), call `_apply_chirality_correction`
directly with `apply_correction=False`.
