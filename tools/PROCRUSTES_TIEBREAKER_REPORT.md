# Procrustes chirality tiebreaker — 12-row before/after

## Background

PR #268's axis-correctness diagnostic + PR #270's Procrustes-search
diagnostic together showed:

- On the 12 approved oracle rows, all 12 have at least one permutation
  in the 720-perm search with axis misfit <30° from ground truth.
- Production's `fit_cube_template_to_anchors` uses bare `<` comparison
  over `itertools.permutations(range(6))` iteration order. Among the
  ~12 perms that tie at the residual floor (the cube's 3-fold body-
  diagonal symmetry × 2 chirality × 2 near/far phase), production picks
  the lex-first perm.
- On 8/12 rows, that lex-first pick is geometrically wrong (~178° axis
  misfit), even though a GOOD perm (<30°) exists at the same residual.

This PR adds a phase-separation tiebreaker that scores each tied perm
via `mean_near - mean_far` darkness (the same signal
`_resolve_near_far_phase` uses post-PnP) and picks the most-negative
("GOOD by darkness").

## Measurement on 12-row corpus

| Row | Before T | After T | ΔT | Before F | After F | ΔF | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| 20_A | 177.4 | 177.4 | +0.0 | 177.4 | 177.4 | +0.0 | no-op |
| 20_B | 179.6 | 179.6 | +0.0 | 179.6 | 179.6 | +0.0 | no-op |
| 38_A | 179.4 | 179.4 | +0.0 | 179.4 | 179.4 | +0.0 | no-op |
| 38_B | 8.9 | 8.9 | +0.0 | 175.0 | 175.0 | +0.0 | no-op |
| **40_A** | **178.8** | **14.4** | **-164.4** | **178.8** | **14.4** | **-164.4** | **fixed** |
| 40_B | 178.2 | 178.2 | +0.0 | 178.2 | 178.2 | +0.0 | no-op (Mode B) |
| 41_A | 12.9 | 12.9 | +0.0 | 12.9 | 12.9 | +0.0 | already GOOD |
| 41_B | 179.8 | 179.8 | +0.0 | 179.8 | 179.8 | +0.0 | no-op |
| **43_A** | 179.5 | 179.4 | -0.1 | **5.7** | **179.4** | **+173.7** | **regressed corr_false** |
| 43_B | 177.9 | 177.9 | +0.0 | 177.9 | 177.9 | +0.0 | no-op (Mode B) |
| 45_A | 178.1 | 178.1 | +0.0 | 5.9 | 5.9 | +0.0 | no-op |
| **45_B** | **24.8** | **3.0** | **-21.8** | **176.6** | **3.0** | **-173.6** | **fixed** |

### Per-row tiebreaker probe

Every row produced exactly 12 tied perms at the residual floor (6 with
negative phase-separation, 6 with positive — a clean bimodal). The
tiebreaker picked the most-negative-sep perm on all 12 rows, and the
existing `_resolve_near_far_phase` agreed ("correct" or
"ambiguous_no_correction" on all 12, never "flip_suggested").

So the tiebreaker is firing as designed, the empirical polarity holds,
and the post-PnP chirality detector is consistent with the pre-PnP
pick.

## Headline

| | Count |
|---|---:|
| Cleanly fixed (both hypotheses 178° → <30°) | 1/12 (40_A) |
| Partial improvement | 1/12 (45_B: T -21.8°, F -173.6°) |
| Regression on at least one hypothesis | 1/12 (43_A corr_false: 5.7° → 179°) |
| No-op (already GOOD or unchanged broken) | 9/12 |

## Why this is mixed

The phase-separation signal is **one bit** — it distinguishes
"model's labeled-near corners are LIGHTER" (= correct near/far phase)
from "model's labeled-near corners are DARKER" (= flipped near/far
phase). On a perfect hexagonal silhouette, the cube's symmetry group
has more than one bit's worth of perms in the residual-tied set:

- 3-fold body-diagonal rotations (h_x → h_y → h_z → h_x): 3
- Near/far phase swap (60° body-diagonal flip): × 2
- Chirality (left-handed vs right-handed assignment): × 2
- = 12 perms per row, matching the observed `procrustes_n_tied`

The negative-separation set contains 6 perms:
- 3 are body-diagonal rotations of the GOOD chirality (all equivalent
  under relabeling — axis-correctness diagnostic gives the same misfit
  for any of these 3 because it computes best-permutation match)
- 3 are body-diagonal rotations of the MIRROR chirality (silhouette-
  identical to GOOD but the inner/outer labeling is mirror-swapped —
  axes point at wrong GT corners)

The tiebreaker picks lex-first within the negative-sep group. On 40_A
and 45_B that happens to land on a GOOD chirality perm. On 43_A
`corr_false` it happens to land on a MIRROR chirality perm. On most
rows it lands on the same perm production already chose.

## V2: tried bezel-alignment as the second signal — did not help

Added `_score_bezel_alignment` (mean angular distance from each inner
template direction to the nearest detected bezel angle) and made the
tiebreaker two-stage: stage 1 filters to negative-phase-separation
(phase-correct) set, stage 2 picks min bezel alignment within them.

**Empirical result: V2 numbers are bit-identical to V1 on all 12
rows.** Per-row probe reveals the cause: on every row, all 6 negative-
sep perms have IDENTICAL alignment scores (~37° to ~47° depending on
the row, but flat across the 6 perms). The GOOD and MIRROR chirality
sets don't separate at all under this signal on real images.

Why: on canonical iso geometry the 6 hexagon vertices sit at exact
60° spacing and the 3 detected bezels point at the GOOD chirality's
inner-vertex angles. GOOD would score ~0°, MIRROR ~60°. But on the
oracle corpus the hexagons are NOT regular — perspective + yaw +
non-zero camera roll distort vertex spacing enough that the affine
fit's inner directions sit between bezels in BOTH chirality sets.
Both score ~40° from the nearest bezel.

Per-row evidence (each row's `procrustes_tiebreaker_all_alignments_deg`
field is a list of 6 identical values):

| Row | Alignment scores (6 perms, all GOOD-sign) |
|---|---|
| 20_A | [39.92, 39.92, 39.92, 39.92, 39.92, 39.92] |
| 40_A | [46.68, 46.68, 46.68, 46.68, 46.68, 46.68] |
| 45_B | [38.59, 38.59, 38.59, 38.59, 38.59, 38.59] |

(12/12 rows showed this flat-across-6 pattern.)

So bezel alignment is the wrong second signal for THIS corpus. Two
alternative directions for the chirality-disambiguator slot, neither
of which this PR tries:

1. **Per-corner darkness (richer than phase-separation mean)**:
   `_resolve_near_far_phase` already computes per-corner darkness for
   all 6 hexagon vertices. The "true near" indices are the 3 LIGHTEST.
   Match those against each perm's claimed-near indices; pick the
   perm whose top-3-darkness pattern best aligns with its h_x/y/z
   labeling.
2. **Push disambiguation downstream**: keep top-K Procrustes
   candidates, run PnP on each, score the final models (sticker color
   coverage, axis-length sanity, junction strength). 6 PnP calls is
   cheap. Likely the most robust path.

## Shipping recommendation

**DRAFT, not for merge.** V2's bezel-alignment stage runs and is
correctly wired (debug fields confirm) but produces zero behavior
change vs V1 on the 12-row corpus, so the combined V1+V2 net is the
same wash as V1 alone: 1 fully fixed (40_A), 1 partial (45_B), 1
regressed (43_A corr_false), 9 no-op.

The implementation is honest and instrumented. Keeping the PR open
preserves the building blocks; the next iteration should swap stage 2
to per-corner darkness or move to top-K downstream re-ranking.

## Test plan

- [x] 10 new unit tests in `tests/test_procrustes_chirality_tiebreaker.py`:
  - `_score_phase_separation`: None without `derive_geometry`,
    polarity (negative on near-lighter, positive on near-darker)
  - `_score_bezel_alignment`: None for <3 bezels, zero on exact-match,
    GOOD<MIRROR on a synthetic GOOD-vs-MIRROR comparison
  - `fit_cube_template_to_anchors`: no-image path preserves
    pre-tiebreaker behavior + records iteration_order; image path
    records the two-stage tiebreaker name + n_tied + chosen sep +
    all seps; invalid hexagon (<6 vertices) returns None
- [x] `pytest tests/` — 932/932 pass (was 922; +10 new)
- [x] 12-row corpus measurement reproduced via `measure_axis_correctness.py`
  from PR #268 (V1 and V2 both run; numbers bit-identical, per-row
  probe shows why)
