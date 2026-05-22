# First-Principles Cube Recognizer Design

> **Status (2026-05-22 rewrite)**: The original version of this doc
> framed first-principles geometry as a near-term *replacement* for the
> production `cv-local` recognizer. Per the Codex+Devin strategic
> synthesis ([`POST_218_BASELINE_AND_TAXONOMY.md`](POST_218_BASELINE_AND_TAXONOMY.md)),
> that framing has been updated. First-principles geometry is now
> positioned as **scaffolding around `cv-local`** — a trust layer, a
> labeled-training-data source, and a benchmark harness — not as a
> replacement.
>
> The pipeline architecture below remains the **north-star vision**.
> The current-state pieces and the phased roadmap to get there are
> in [`STATE_OF_THE_WORLD.md`](STATE_OF_THE_WORLD.md).

## Goal

Provide a consistently accurate, solveable cube state from two guided
isometric photos, with a target delivered-solve accuracy near 99%. The
system should be allowed to abstain, request a retake, or ask for
small human confirmation when the image evidence is not strong enough.

The fallback should distinguish two different failure classes:

- If geometry fails, prefer abstention or guided retake.
- If geometry is strong but 2–3 colors are ambiguous, prefer a
  low-friction manual-review / Fixer flow over forcing a retake.

## Core principle

Build the recognizer around a **single coherent projected cube model
per photo**, not around independent local 3x3 grid detections.

A valid input image should be explained as three visible 3x3 cube
faces sharing one cube corner, with three families of bezel/grid
lines, fixed cube topology, and model-derived sticker cells. Every
sampled sticker should come from this same projected cube model.

## The Devin/Codex policy bar (2026-05-22)

> A plausible cube model is cheap. A trustworthy cube model is hard.

First-principles work — anything touching `tools/global_cube_model.py`
or downstream — must satisfy at least one of:

1. **Improve guardrails around `cv-local`** (e.g., supply a trust
   signal that catches confident-wrong solves at acceptable
   abstention).
2. **Produce labeled training data** for a future learned ranker
   (extending or refining `tests/fixtures/gcm_axis_ground_truth.json`).
3. **Demonstrate safe held-out improvement** on the 58-case axis-
   labeled gallery — measured by `tools/baseline_post_218.py --diff`
   against the committed snapshot, with row-level deltas pasted into
   the PR body. Aggregate-metric-only A/B is insufficient.

If a proposed change doesn't fit one of those three, it's almost
certainly the wrong next move and should be reconsidered.

## What's explicitly off the table

Per the same synthesis:

- **More handcrafted vertex/phase heuristics** (dark-line variants,
  junction extractors, scalar scorers). The sprint repeatedly
  falsified them at the safe-coverage bar; diminishing returns.
- **More SAM3 prompt bakeoffs** without a materially new signal.
- **Replacing `cv-local`** in production with the global cube model.
  The global model is scaffolding.
- **Aggregate-metric-only A/B comparisons.** Per Devin: *"some changes
  improve averages while worsening critical rows."*

## North-star pipeline (unchanged target architecture)

1. **Guided capture**

   Two convention-driven photos:
   - Image A: white-up view.
   - Image B: yellow-up / flipped complementary view.
   - Each shows exactly three cube faces.
   - Across A+B, all six centers are observed.

   If a photo is blurry, cropped, too oblique, glare-heavy, missing a
   center, or inconsistent with the capture convention, prefer retake
   over guessing.

2. **Cube object detection**

   Segment and crop the cube region using silhouette, edges, and black
   plastic bezel structure. This stage finds the cube region and rough
   pose only; it does not commit to sticker identities.

3. **Global cube geometry fit**

   Fit the whole projected cube as three connected face planes:
   - one shared visible cube corner / junction,
   - three visible faces,
   - three families of parallel bezel/grid lines,
   - fixed 3x3 spacing per visible face under perspective,
   - physically valid cube topology.

   Score candidate fits against original-image evidence:
   - predicted grid lines align with dark bezel edges,
   - sticker interiors are color-homogeneous,
   - cells do not cross face-boundary bezels,
   - all three faces agree on one coherent cube pose.

   **Current gap (2026-05-22)**: silhouette + 6 hexagon corners gives
   the geometry an unavoidable 60° near/far phase ambiguity — see
   [`NEAR_FAR_PHASE_REPORT.md`](NEAR_FAR_PHASE_REPORT.md). The
   architectural answer is to fold additional evidence (bezel-line
   angles, two-view consistency, learned priors) into the fit itself
   rather than detect-and-correct downstream.

4. **Model-derived facelets**

   Once the global cube model is trusted, derive all visible sticker
   quadrilaterals from that model. Do not independently discover
   sticker cells outside the model.

5. **Per-face rectification**

   Rectify each trusted visible face into a canonical flat square
   grid. Use rectification for color sampling and debugging, not as
   the source of geometry truth.

   ```text
   photo → global cube pose → trusted face quads → rectified faces → color samples
   ```

   **Key constraint**: rectification helps color classification *only
   when geometry is trusted*. Rectifying an untrusted quad can make
   invalid geometry look clean — that's worse than not rectifying.

6. **Robust color classification**

   Sample shrunken interior patches from each rectified cell. Use
   center stickers as per-image color anchors, combine Lab/HSV/RGB
   features, and keep top-N color probabilities instead of committing
   too early.

   Color classification is not currently believed to be the binding
   constraint when geometry is correct.

7. **Two-view fusion**

   Map the three visible faces from A and B into the six cube faces
   using center colors and the capture convention. Reject, request
   retake, or request human confirmation if faces are missing,
   duplicated inconsistently, or if A/B evidence conflicts.

8. **Legal cube assignment**

   Choose the lowest-cost legal cube state from the color
   probabilities while enforcing:
   - exactly nine of each color,
   - valid centers,
   - valid cubies,
   - legal orientation and parity,
   - a unique winner with a real margin.

   Legal solveability should be the final consistency constraint, not
   a repair mechanism for bad geometry. *Legal repair must not hide
   low-confidence geometry.*

9. **Quality gates and UX**

   Three possible outcomes:
   - confident solved cube,
   - retake one or both photos when geometry/capture quality fails,
   - confirm a small number of ambiguous stickers when geometry is
     good but residual color evidence is weak.

   To reach a 99% accurate delivered-solve rate, abstention, guided
   retake, and manual confirmation are part of the design, not
   failures. The split matters: retake is appropriate for geometry
   failure; manual review is appropriate for residual color ambiguity
   after a trusted geometry fit.

## Current lessons (post-sprint)

What we learned during the first-principles geometry sprint:

- **Silhouette-only geometry is underdetermined.** The 7 anchor points
  (vertex + 6 hex corners) leave a 60° near/far phase ambiguity from
  the cube's body-diagonal rotation symmetry. Procrustes brute-force
  picks one of the two valid options based on sub-noise residual
  differences — effectively random.
- **The "chirality" problem is real but the current darkness detector
  is empirical, not first-principles.** See
  [`NEAR_FAR_PHASE_REPORT.md`](NEAR_FAR_PHASE_REPORT.md) for the full
  framing. PR #218's 33pp accuracy lift from a single block reorder
  (vertex ensemble BEFORE phase check) is evidence that the detector
  is sensitive to non-geometric noise. The right structural answer is
  multi-evidence fusion in the Procrustes fit, which is what a learned
  ranker delivers.
- **Vertex precision is the dominant remaining lever.** 95%+ of
  catastrophic failures in the post-#218 baseline are phase-decision
  miscalls; only 1.7% are pure-geometry failures regardless of phase.
  Better vertex precision tightens the phase confidence signal and
  removes the dominant failure mode.
- **Hand-crafted vertex localizers don't graduate.** Multiple variants
  (dark-line, junction extraction, raw patch, patch/junction) failed
  the safe-coverage bar. The candidate-oracle is often strong;
  ranking/confidence calibration is the unsolved problem.
- **Labels are the most valuable artifact produced by the sprint.**
  58 user-labeled cases (vertex + 3 near corners per photo) make
  failures measurable instead of anecdotal. Treat them as both:
  - the *training set* for a future learned vertex/axis/phase ranker
    (extend via active-label queues, not opportunistically), and
  - the *regression gate* for every geometry-sensitive PR
    (`tools/baseline_post_218.py --diff`).
- **Rectification helps color only when geometry is trusted.** The
  hybrid-pipeline experiment (PR #152) confirmed that rectifying
  arbitrary local quads produces bimodal accuracy distributions —
  perfect on faces where geometry was right, near-random on faces
  where it was wrong.
- **Legal-state repair can make a bad geometry look solveable; it
  cannot make geometry true.** Confidence/abstention is the right
  guard, not more aggressive repair.

## Recommended long-term direction

The highest-leverage architectural shift remains a global cube model
fitter — but reframed as the trust layer for `cv-local`, not as a
replacement:

```text
detect cube object  (cv-local + global model agree?)
  → fit coherent projected 3-face cube model  (global model produces candidate + confidence)
  → derive visible sticker quads from that model  (only if confidence threshold met)
  → rectify trusted faces  (skip if low confidence — route to retake)
  → classify colors from clean rectified samples  (cv-local primary)
  → fuse A+B into six faces
  → choose a unique legal cube state
  → abstain/retake/confirm when margins are weak  (driven by global-model confidence)
```

The roadmap from here is [`STATE_OF_THE_WORLD.md`](STATE_OF_THE_WORLD.md)
Phase 0 → Phase 5. The short version: consolidate now (Phase 0), then
re-baseline both sides (cv-local + global model) on the same 58 cases
(Phase 1), add diagnostics-only trust signals (Phase 2), wire them to
production via guardrails (Phase 3), train the learned ranker
(Phase 4), and improve capture/UX with the resulting signals
(Phase 5).

## See also

- [`STATE_OF_THE_WORLD.md`](STATE_OF_THE_WORLD.md) — current architecture map + phased roadmap.
- [`POST_218_BASELINE_AND_TAXONOMY.md`](POST_218_BASELINE_AND_TAXONOMY.md) — the decision spine.
- [`FAILURE_TAXONOMY.md`](FAILURE_TAXONOMY.md) — single source of truth for failure-mode categories.
- [`BENCHMARK_INDEX.md`](BENCHMARK_INDEX.md) — which fixture/report answers which question.
- [`NEAR_FAR_PHASE_REPORT.md`](NEAR_FAR_PHASE_REPORT.md) — geometric framing of the phase ambiguity.
- [`GLOBAL_CUBE_MODEL.md`](GLOBAL_CUBE_MODEL.md) — current global-model implementation spec.
- `../COORDINATION.md` — role split between Claude and Codex + regression-gate policy.
