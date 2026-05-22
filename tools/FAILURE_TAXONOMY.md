# Failure taxonomy

> Single source of truth for categorizing recognizer failures.
> Cite this from PR bodies, reports, and ad-hoc analyses so we're
> all using the same labels.

## Top-level categories

| Category | Where it shows up | What it means | Typical fix path |
|---|---|---|---|
| **GEOMETRY_FAIL** | Global model fit has high residual; or no fit at all. | The 6-DOF cube model can't be fit to the silhouette+hexagon anchors. Underlying causes: hull bad, anchor extraction bad, perspective extreme. | Improve detection/anchors upstream; route to retake if hull is the issue. |
| **NEAR_FAR_PHASE_FAIL** | Model fits but with 60° rotation around body diagonal — model.near corners are actually at face-diagonal positions. | The cube's 3-fold body-diagonal symmetry makes Procrustes pick one of two valid options based on sub-noise residual. The "chirality" problem. | Phase detector + auto-correction (in production, 79% end-to-end accuracy post-#218). Long-term: learned multi-evidence Procrustes. |
| **VERTEX_OFFSET** | Geometry fits but vertex is 30-100+ px from the true trihedral junction. | PnP under perspective is noisy. The mean-of-3 ensemble (PnP + bezel + hex centroid) reduces this but doesn't eliminate it. | Better vertex localizer. Mostly captured by Phase 4 learned ranker. |
| **HULL_BACKGROUND_FAIL** | rembg mask is wrong (cuts off corners, includes background, misses cube entirely). | Background interferes (wood grain, shadows, low contrast). Caused real failures on Set 46. | Foundation-mask alternatives (SAM3 explored — bakeoffs without new signal are off the table). Route to retake when hull quality is low. |
| **GRID_SOURCE_CONTAMINATION** | `analyze_image` returns a 3×3 grid whose 9 stickers span multiple physical cube faces. | The face-quad detector latched onto a quad that crosses a cube edge. Produces plausible-but-wrong sticker colors. | Hull guard (PR #141, landed). Per-face source filtering. Cv-local-side fix. |
| **COLOR_AMBIGUITY** | Geometry is correct but 2-3 sticker color labels are ambiguous (e.g., red vs orange under glare, blue vs green under low light). | Color is noisy on real-world cubes under varied lighting. | Robust classifier (knn5_lab_full helps). Per-image color anchors. Manual-fixer flow for residuals. |
| **LEGAL_AMBIGUITY** | Color labels are individually reasonable but no legal cube state fits, or multiple legal states fit equally well. | The 54-sticker arrangement must obey cube invariants (9 of each color, valid centers/cubies/parity). | Legal-state assignment with top-N color probabilities. Repair must be guarded by confidence margins — repairing low-confidence geometry hides errors. |
| **CAPTURE_QUALITY_FAIL** | Photo is blurry, glare-heavy, missing a face, oblique, or off-convention (e.g., wrong face-up). | User-side issue. | Capture-flow improvements: orientation guide, blur detector, retake prompts. Phase 5 work. |

## Sub-categories under NEAR_FAR_PHASE_FAIL

Used in `tools/baseline_post_218.py` and the post-#218 baseline:

| Sub-category | Meaning |
|---|---|
| `CHIRALITY_MISS` | Model.far matches user.near; detector said `correct` or `ambiguous` — flip needed but missed. |
| `CHIRALITY_FALSE_FLIP` | Model.far matches user.near; detector said `corrected_60deg_flip` — wrongly flipped a previously-correct model. |

(Names retained for backward compatibility with the baseline JSON
field even after the chirality → near_far_phase rename. Future
baselines may use `PHASE_MISS` / `PHASE_FALSE_FLIP`.)

## Severity bands (bearing error on 58-case eval)

These come from `tools/baseline_post_218.py`'s categorization:

| Band | Bearing error | Recognizer outcome | Typical category |
|---|---|---|---|
| **GOOD** | < 10° mean | Model is essentially right | — |
| **MARGINAL** | 10° – 25° mean | Small jitter, color sampling probably still OK | VERTEX_OFFSET (mild) |
| **catastrophic** | > 25° mean | Wrong geometry, wrong colors downstream | NEAR_FAR_PHASE_FAIL / GEOMETRY_FAIL / HULL_BACKGROUND_FAIL |

Per [`POST_218_BASELINE_AND_TAXONOMY.md`](POST_218_BASELINE_AND_TAXONOMY.md),
the post-#218 distribution is:

| Band | % of runs |
|---|---|
| GOOD + MARGINAL | 77.6% |
| CHIRALITY_MISS / FALSE_FLIP | 20.7% of total (95% of catastrophic) |
| TRUE_GEOMETRY_FAIL | 1.7% |

**Implication**: 95%+ of remaining catastrophic failures are phase-
decision miscalls. Vertex precision is the dominant single lever.

## How to use this doc

- **In PR bodies**: cite the relevant category when describing a
  fix or a regression. Example: "Fixes CHIRALITY_MISS on Set 12_A —
  see baseline diff."
- **In reports**: keep terminology consistent. If a failure doesn't
  fit a listed category, propose a new one in a PR that updates this
  file.
- **In code comments**: use the category name verbatim when explaining
  what a guard catches. Example: `# HULL_BACKGROUND_FAIL guard`.
- **In active-label queue selection**: target cases that fall into
  under-sampled failure modes so labels span the failure surface, not
  the visually-interesting tail.

## Categories deliberately NOT in this taxonomy

- Per-pixel classifier confusions (orange-vs-red on stickers) —
  rolled into COLOR_AMBIGUITY.
- Synthetic-corpus-only failures — those are out of scope; the
  production failure modes are what matter.
- "Glitch" or "edge case" without a hypothesis — if you can't say
  which category a failure belongs to, that's a signal the analysis
  isn't done yet, not a sign we need a new bucket.
