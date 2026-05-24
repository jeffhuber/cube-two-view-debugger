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

## What's needed next

A SECOND signal to distinguish chirality among the 6 negative-sep
perms. The function already receives `bezel_angles_rad` — the 3 angles
where interior-bezel detection found a dark cube-edge line from
`cube_center`. The GOOD chirality's 3 inner template vertices
(h_x, h_y, h_z) project to directions that should align with these 3
bezel angles. The MIRROR chirality projects them to angles between
the bezels (along face diagonals, not edges).

Concrete next-step proposal:

1. Among the residual-tied perms, filter to negative-phase-separation
   set (preserves Mode A win).
2. Score each remaining perm by **mean angular distance from each of
   its 3 inner directions to the nearest detected bezel angle**.
3. Pick the perm with the smallest mean angular distance.

This combines the phase-separation tiebreaker (this PR) with bezel-
alignment as a chirality disambiguator. Bezel angles are already a
required input to `fit_cube_template_to_anchors`, so no plumbing
change is needed.

## Shipping recommendation

**Land as DRAFT for review and as a stepping stone, NOT for merge.**

Net effect on the 12-row corpus is roughly a wash (1 fix + 1 partial -
1 regression). The implementation works as designed; the design itself
is one bit short.

If merged as-is, `corr_false`-style downstream consumers that happened
to land on GOOD chirality by lex order would regress. The pre-fix
behavior is at least stable (deterministic on the same input); this
intermediate version is also deterministic but with a different and
not-uniformly-better distribution.

Best path forward: keep this PR open, add the bezel-alignment second
signal in a follow-up commit, re-measure, then either land both
together or revisit.

## Test plan

- [x] 7 new unit tests in `tests/test_procrustes_chirality_tiebreaker.py`:
  - `_score_phase_separation` returns None without `derive_geometry`
  - `_score_phase_separation` polarity (negative on near-lighter image,
    positive on near-darker)
  - No-image path preserves pre-tiebreaker behavior + records
    `procrustes_tiebreaker = "iteration_order"`
  - Image path records `procrustes_tiebreaker = "phase_separation"`
    + tied perm count + chosen sep + all seps
  - Invalid hexagon (<6 vertices) still returns None
- [x] `pytest tests/` — 929/929 pass (was 922; +7 new)
- [x] 12-row corpus measurement reproduced via `measure_axis_correctness.py`
  from PR #268
