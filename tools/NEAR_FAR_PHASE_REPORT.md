# Near/far phase detection in global_cube_model

## Naming note (2026-05-22 rename)

Earlier code and PRs (#210, #213, #218) called this "chirality
detection." That's a slight misnomer:

- **Strict-geometry chirality** means handedness under reflection —
  a left-handed vs right-handed mirror invariant.
- **What we actually face** is a **60° near/far phase ambiguity**: a
  rotational degeneracy from the cube's 3-fold symmetry around its
  body diagonal. The 6 hexagon-silhouette corners alternate near/far,
  and the labels can be swapped by a 60° rotation that leaves the
  silhouette unchanged.

The mechanical rename (`_apply_chirality_correction` →
`_resolve_near_far_phase`, `chirality_check` → `phase_check`, all
`chirality_*` debug fields → `phase_*`) was done by a single PR
following the strategic synthesis from Codex+Devin. Per
[`tools/POST_218_BASELINE_AND_TAXONOMY.md`](POST_218_BASELINE_AND_TAXONOMY.md),
the naming distinction matters because the current darkness-based
detector is an empirical stopgap, not a first-principles signal —
treating it as "chirality" pulled in wrong intuitions about what
kind of evidence would settle the ambiguity.

There is ONE genuinely-chirality concern still in
`fit_cube_template_to_anchors` (the CCW vs CW hexagon ordering),
and the comment there is explicit about distinguishing the two.

## Corner convention update (2026-05-23)

The canonical human convention for geometry labels is now
[`FULL_CORNER_LABELING.md`](FULL_CORNER_LABELING.md), backed by
`tools/corner_conventions.py`:

```text
Image A slots: upper=Va+1,0,5; right=Va+3,2,1; front=Va+5,4,3
Image B slots: upper=Vb+2,3,4; right=Vb+0,1,2; front=Vb+4,5,0
```

Canonical WCA face names for the side slots depend on capture yaw; the
corner numbering itself does not.

The one-edge triplet is side-specific:

```text
A one-edge = 1,3,5; A far = 0,2,4
B one-edge = 0,2,4; B far = 1,3,5
```

Initial audit against `tests/fixtures/full_corner_ground_truth.json` shows
the legacy `near_x/near_y/near_z` labels match the far/double-axis triplet on
the 12 seed photos (`A -> 0,2,4`, `B -> 1,3,5`). Treat row-level
`CHIRALITY_MISS` / `CHIRALITY_FALSE_FLIP` evidence as provisional until the
legacy baseline is regenerated from full-corner labels.

## Bug

The Procrustes brute-force fit in `fit_cube_template_to_anchors` searches
all 720 (6!) permutations of detected hexagon corners → template
positions and picks the lowest residual. The cube has a 60° symmetry
around its body diagonal that yields the SAME silhouette with the NEAR/FAR
labels swapped — 6 symmetry-equivalent permutations all give nearly the
same residual. Which one wins is determined by tiny noise in the fit,
making the near/far phase choice effectively non-deterministic.

Discovered via the axis-labeling tool's ground truth (4 user-labeled
cases on 2026-05-22): set 12_A's model output had its near-corner axes
~58° off from the user-labeled directions — a clear 60° flip rather than
a small drift.

## Detector

`_resolve_near_far_phase` inspects each of the 6 hexagon corners
after the fit:

1. For each of the 6 corners, compute mean pixel **darkness** along the
   line from the model vertex to that corner (15–85% of the line, to
   skip endpoints).
2. Mean of the 3 model-labeled NEAR corners' darkness (`mean_near`) vs
   the 3 model-labeled FAR corners' darkness (`mean_far`).
3. **Signed separation** = `mean_near − mean_far`.
4. Emit `phase_check` status + per-corner darkness + separation in
   `model.debug`.

### Status values

| status                              | meaning                                                                |
|-------------------------------------|------------------------------------------------------------------------|
| `correct`                           | sep < −10 → empirically the phase is correct                       |
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

### Validation cross-tabs (58 cases × 2 runs = 116 model runs)

Two regimes measured. Bearings comparison against user-labeled near
corners is scale+translation invariant, so we work in gallery coords
directly.

**Regime A — phase check runs BEFORE mean-of-3 vertex ensemble**
(initial PR #213 wiring). Detector verdict vs position truth:

|                                 | truth=CORRECT | truth=FLIPPED | total |
|---------------------------------|--------------:|--------------:|------:|
| correct (sep < −10, no flip)    |            45 |             7 |    52 |
| corrected_60deg_flip (sep > +10)|             5 |            48 |    53 |
| ambiguous (\|sep\| < 10)        |             3 |             5 |     8 |
| **total**                       |            53 |            60 |   116 |

End-to-end accuracy: **53/116 = 45.7% correct** on the final returned
model. Base rate of flips that pass through: **60/116 = 51.7%**.

**Regime B — phase check runs AFTER mean-of-3 vertex ensemble**
(the order in current main). Detector verdict vs position truth:

|                                 | truth=CORRECT | truth=FLIPPED | total |
|---------------------------------|--------------:|--------------:|------:|
| correct (sep < −10, no flip)    |            32 |             4 |    36 |
| corrected_60deg_flip (sep > +10)|            42 |             8 |    50 |
| ambiguous (\|sep\| < 10)        |            18 |            12 |    30 |
| **total**                       |            92 |            24 |   116 |

End-to-end accuracy: **92/116 = 79.3% correct** — a 33.6-percentage-
point absolute improvement over Regime A (or ~71% relative).
Base rate of flips that pass through: **24/116 = 20.7%** (down from
51.7%).

Detector accuracy when it commits a verdict (correct ∪
corrected_60deg_flip): 32+42 = 74 right / 86 commits = **86%**.
The 30 "ambiguous" runs land 60% correct vs 40% flipped — that's the
underlying Procrustes coin-flip when neither the darkness signal nor a
useful flip can salvage it.

Case-level stability across 2 runs:
- Regime A: 11/58 always CORRECT, 15/58 always FLIPPED, 32/58 mixed.
- Regime B: **40/58 always CORRECT**, 6/58 always FLIPPED, 12/58 mixed.

### Why the order matters

The model vertex from PnP-only is typically 10–50 px off from the true
cube vertex. When the phase detector runs with that pre-ensemble
vertex (Regime A), the line from model_vertex to model.h_x doesn't
actually lie on the cube-edge bezel — it skims off into adjacent
sticker interior. The detector then fires off a signal that's
counter-intuitively inverted from the naive bezel-darkness reasoning
(`sep < 0` ≡ correct), accidentally calibrated to that off-bezel
regime.

The mean-of-3 ensemble (PnP + bezel + hexagon-centroid) brings the
vertex closer to the true cube vertex. Running the phase detector
on the ensemble-corrected vertex (Regime B) doesn't revert the
polarity to the geometric ideal (`sep > 0` ≡ correct) — `sep < 0`
still means correct — but the discriminator becomes much more
confident: the |sep| < 10 ambiguous band shrinks the disagree rate
from 10.6% to 3.4%, the commit rate goes up, and end-to-end accuracy
nearly doubles.

So the polarity inversion isn't a pure vertex-offset artifact. It
holds in both regimes; vertex precision just makes the signal cleaner.

## What's next

- **Tune the ambiguous threshold.** With |sep| < 10 still putting 30
  runs into ambiguous (and 12 of those landing FLIPPED), a slightly
  wider commit band (e.g., flip when sep > +5) might claw back a few
  more flipped cases at the cost of accepting more low-confidence
  flips. Empirical sweep over the 58-case gallery would calibrate it.
- **Deterministic Procrustes tie-breaker.** Even with an 86%-accurate
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
print(model.debug["phase_check"])
# 'correct' | 'corrected_60deg_flip' | 'ambiguous_no_correction' | …

print(model.debug["phase_darkness_separation"])   # signed
print(model.debug["phase_near_line_darkness"])    # 3 floats
print(model.debug["phase_far_line_darkness"])     # 3 floats
```

`fit_global_cube_model` now calls `_resolve_near_far_phase` with
`apply_correction=True` by default, so the returned model reflects the
phase-corrected geometry whenever the detector fires. To skip
correction (diagnostic-only), call `_resolve_near_far_phase`
directly with `apply_correction=False`.
