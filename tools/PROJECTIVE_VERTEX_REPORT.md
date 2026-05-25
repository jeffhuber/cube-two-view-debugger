# Projective vertex via vanishing-point construction — corpus diagnostic

## Question this PR answers

Can projective vertex correction (via vanishing-point construction)
recover perspective-tilted rows like **37_B** without regressing
clean rows or bad-hull-input rows like **30_A**?

## Headline result

**The vanishing-point math works (geometry IS measurably more accurate
on perspective-heavy rows), but the production-trust metric is insensitive
to the improvement.** Specifically:

- **37_B** (the named case study): vertex_err **80 → 38 px** (−52%),
  axis_misfit **32.2° → 14.3°** (drops below the 30° fallback gate),
  sticker_score **594.78 → 593.15** (essentially unchanged — tie).
- **10/70 rows** see vertex_err improvements > 15 px from projective.
- **66/70 rows** are sticker-score ties (within ±30) between affine and
  projective. Of the 4 non-ties, projective wins 1, affine wins 3.
- **`projective_residual_norm` is a new bad-input detector** — strongly
  correlates with 30_A's known-bad-hull case (which has the highest
  residual in the corpus).

So this is a **decision-quality null result**: the projective math is
correct and elegant; the question of whether to wire it into production
hinges on whether the *downstream* metric (sticker-color classification
under the canonical 3×3 sampling) is sensitive to small face_quad
shifts, and on this corpus it is not. Recommendation below.

## Decision-quality output schema

Per Codex review item 5 (don't let sticker score alone decide
"better"), each row carries the full multi-signal record:

| column | source | use |
|---|---|---|
| `affine_vertex_err_px` | vs GT | diagnostic (production cannot compute) |
| `projective_vertex_err_px` | vs GT | diagnostic |
| `affine_axis_misfit_deg` | vs GT | diagnostic |
| `projective_axis_misfit_deg` | vs GT | diagnostic |
| `affine_sticker_score_total` | rectified-face classifier distance | **production** |
| `projective_sticker_score_total` | same with projective vertex | **production** |
| `projective_residual_px` | 3-line LSQ residual (perpendicular px) | **production** |
| `projective_residual_norm` | residual_px / hexagon_diameter | **production**, resolution-independent |
| `projective_degeneracy` | `finite_projective` / `near_affine` / `degenerate` | **production** classification |
| `winner_by_sticker` | which vertex gives lower sticker | per-row decision |
| `winner_by_gt_vertex` | which is closer to GT (px) | diagnostic-only |

(Production-usable signals can be computed at recognition time from
the rembg mask alone — they don't need GT.)

## 37_B case study

The named row this PR exists to explain. See
`projective_vertex_case_37B.png` for the visual panel showing
affine-quad-overlay (red) vs projective-quad-overlay (green) on the
source image, plus the two rectified-face triples below it.

| Metric | Affine | Projective | Δ |
|---|---:|---:|---:|
| vertex_err vs GT | 79.7 px | **37.6 px** | **−52%** |
| axis_misfit | 32.2° | **14.3°** | **−56%** |
| sticker_score | 594.78 | 593.15 | −0.3% (tie) |
| projective_residual_norm | — | 0.0117 | — (sane, well below 0.05 gate) |
| projective degeneracy | — | `finite_projective` | well-conditioned |

So projective IS geometrically right on 37_B (the math validates the
hypothesis), but the downstream rectified faces score almost identical
because the 9-sticker centroids interior to each face quad are
insensitive to ~30 px shifts in the vertex corner.

## Top rows where projective recovers most vertex error

| Row | aff_v_err | prj_v_err | Δv | aff_sticker | prj_sticker | Δsticker |
|---|---:|---:|---:|---:|---:|---:|
| 45_B | 58.9 | **8.9**  | +50.0 | 596.1 | 593.7 | +2.4 |
| 45_A | 44.4 | **1.8**  | +42.6 | 529.1 | 528.3 | +0.8 |
| 37_B | 79.7 | **37.6** | +42.1 | 594.8 | 593.1 | +1.6 |
| 49_B | 69.6 | **41.9** | +27.7 | 686.6 | 661.2 | +25.4 |
| 37_A | 71.1 | **45.4** | +25.7 | 659.3 | 654.6 | +4.8 |
| 44_A | 36.7 | **14.2** | +22.5 | 495.9 | 502.4 | −6.5 |
| 62_B | 41.0 | **18.9** | +22.1 | 580.9 | 581.4 | −0.5 |
| 46_A | 25.2 | **5.1**  | +20.1 | 432.0 | 433.4 | −1.5 |
| 40_B | 28.2 | **8.3**  | +19.9 | 312.0 | 311.6 | +0.4 |
| 23_A | 27.8 | **8.2**  | +19.6 | 605.9 | 605.2 | +0.7 |

Pattern: projective consistently improves the vertex geometry by 20–50 px
in the top recovered rows, but Δsticker is almost always |Δ| < 7 — the
classifier doesn't care about these vertex shifts because the sticker
samples land inside the same physical stickers regardless.

The single exception in the table is **49_B** (Δsticker = +25.4 in
projective's favor, just under the 30-margin gate). This is the only
row where projective vertex meaningfully cleans up sticker sampling
enough to register in the production metric.

## 30_A (known bad-hull-input case)

Per Codex review item 6, **30_A is not evidence against the projective
approach** — it's a bad-input case (rembg picked up a wall-edge
artifact as the silhouette TOP). The projective math correctly:

| Metric | Affine | Projective |
|---|---:|---:|
| vertex_err vs GT | 51.1 px | 60.8 px (worse) |
| sticker_score | 575.11 | 624.62 (worse) |
| projective_residual_norm | — | **0.0315 (highest in corpus)** |
| projective degeneracy | — | `near_affine` |

The high `projective_residual_norm` is the diagnostic signal — projective
correctly *detects* that the 3 vanishing-point lines don't meet cleanly,
because the 6 input corners are not all on the cube. **`projective_residual_norm`
is a new gate signal** that PR #285's acceptance gates could adopt to
flag bad-input rows that don't have visible degenerate geometry but DO
have inconsistent multi-face evidence.

## Degeneracy classification

Per Codex review item 4:

| Class | Count | Meaning |
|---|---:|---|
| `finite_projective` | 69 / 70 | All 3 vanishing points are finite AND lines meet cleanly → projective math is well-conditioned |
| `near_affine` | 1 / 70 (30_A) | ≥1 vanishing point is at infinity or far enough that lines are effectively parallel → projective ≈ affine, no improvement expected |
| `degenerate` | 0 / 70 | Residual exceeds 5% of hexagon diameter → neither geometry is trustworthy |

Worst-case residuals (`projective_residual_norm`):

| Row | resN | degeneracy | prj_err | aff_err | known_bad |
|---|---:|---|---:|---:|---|
| 30_A | 0.0315 | `near_affine` | 60.8 | 51.1 | **YES** ✓ |
| 30_B | 0.0199 | `finite_projective` | 21.5 | 37.3 | — |
| 25_A | 0.0163 | `finite_projective` | 62.6 | 56.2 | — |
| 29_A | 0.0128 | `finite_projective` | 43.9 | 33.0 | — |
| 28_A | 0.0118 | `finite_projective` | 9.3 | 19.9 | — |

The residual signal correctly flags 30_A as the worst-conditioned row
in the corpus — exactly the row known to have bad hull input.
30_B, 25_A, 29_A are candidates worth a visual spot-check to see if
they also have subtle hull-detection artifacts that we haven't
catalogued yet.

## Why sticker score doesn't differentiate

The sticker scorer samples the 9 centroids of a 3×3 grid inside each
rectified face. Each centroid is the center of a ~30×30 px sticker
region. A 30-50 px vertex shift moves the face_quad by a similar
amount — but the 9 sticker centers are interior to the face quad
(spaced 30-50% of face dimensions from each edge), so they typically
stay inside the same physical sticker even when the quad shifts.

So the sticker scorer is **robust** to ~50 px vertex perturbations,
which is exactly what makes it a good production-trust metric for the
existing affine pipeline — but also what makes it insensitive to the
geometry improvement from projective.

A more sensitive metric would be something like:
- **bezel-overlap penalty**: how much do face_quads overlap with detected
  bezel pixels (which mark face boundaries)?
- **edge-distance variance**: how much does each face_quad's edge length
  vary from the median? Should be small for a true cube.
- **inter-face sticker consistency**: do shared cube corners give
  consistent sticker boundaries when sampled from both adjacent faces?

Any of these would likely show the projective improvement that
sticker-score misses.

## Recommendation

**Diagnostic-only PR; no production behavior change. Specific guidance:**

1. **DO NOT replace affine with projective wholesale.** Sticker
   score (the current production trust metric) shows tie on 66/70
   rows. Wholesale replacement has no expected production benefit
   and adds a code path. **Affine stays the default.**

2. **DO add `projective_residual_norm` as a NEW gate signal in
   `tools/hull_label_acceptance.py`** (PR #285). It strongly
   correlates with bad-hull-input rows (30_A's 0.0315 is the
   corpus max). Suggested initial threshold: `> 0.025` →
   warn / fallback, supplementing the existing `vertex_cloud_spread`
   signal.

3. **DEFER projective-vertex production wiring** until either:
   - A more sensitive end-to-end metric (full recognition accuracy,
     bezel-overlap, etc.) shows that the geometry improvement
     matters at the user-visible level, OR
   - A row class emerges where affine spends > 0 sticker-score
     budget that projective recovers (the 49_B-class is the only
     such case at this corpus size).

4. **Production candidate hybrid switch (later, after #2 lands):**
   ```
   trust_projective = (
     projective_degeneracy == "finite_projective"
     AND projective_residual_norm < 0.02
     AND projective_sticker_score_total + MARGIN < affine_sticker_score_total
   )
   ```
   On the current corpus this would fire on ~1 row (49_B). Not zero,
   but not the headline justification either.

The single concrete win for this PR is **#2 — the new gate signal**.
The projective math itself is interesting, the case study on 37_B is
visually compelling, and the corpus null-result is honest evidence
that simple vertex-geometry improvements don't translate to
production-trust improvements with the current scoring.

## Reproducing

```bash
.venv/bin/python tools/diagnose_projective_vertex.py
```

Writes `tests/fixtures/projective_vertex_trace.json` and the
`projective_vertex_case_37B.png` visual panel.

## What this PR contains

- `tools/projective_vertex.py` — core vanishing-point solver with
  `derive_axis_edge_pairs(side)` (computed from `FACE_DEFS_BY_SIDE`
  per Codex review item 1, no hand-coded A/B), degeneracy
  classification (per item 4), and normalized residuals (per item 3).
- `tools/diagnose_projective_vertex.py` — corpus runner emitting
  the multi-signal record (per item 5).
- `tools/PROJECTIVE_VERTEX_REPORT.md` — this report.
- `tools/projective_vertex_case_37B.png` — visual panel (per item 7).
- `tests/fixtures/projective_vertex_trace.json` — committed canonical trace.
- `tests/test_projective_vertex.py` — 10 unit tests pinning the
  math + side-A/B conventions + degeneracy + exact-on-synthetic-perspective.

This PR does NOT modify `tools/rectify_via_hull_labels.py`,
`tools/hull_label_acceptance.py`, or any production code path.

## Suggested next steps

1. **Codex review of this report's recommendation** — decide whether
   `projective_residual_norm` belongs in PR #285's acceptance gates
   (probably yes) and whether the projective-vertex code itself
   should live behind the future production feature flag (likely
   only if a more sensitive metric makes the case).

2. **Visual spot-check of 30_B / 25_A / 29_A** — they have elevated
   `projective_residual_norm` and might be undiagnosed bad-hull cases
   similar to 30_A. Cheap check; could uncover a small new failure
   mode.

3. **Bezel-overlap or edge-length-variance metric** — if we want to
   make a stronger case for projective production wiring, build a
   metric that's sensitive to the geometry improvement that sticker
   score misses. Out of scope for this PR.
