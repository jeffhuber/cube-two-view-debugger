# First-Principles Cube Recognizer Design

This memo captures the proposed longer-term recognizer architecture based on
what we have learned from the production recognizer, hard-case probes, overlay
feedback, rectification experiments, and interior-bezel diagnostics. It is a
memory aid, not a claim that the current implementation has moved to this
architecture.

## Goal

Provide a consistently accurate, solveable cube state from two guided
isometric photos, with a target delivered-solve accuracy near 99%. The system
should be allowed to abstain, request a retake, or ask for small human
confirmation when the image evidence is not strong enough.

The fallback should distinguish two different failure classes:

- If geometry fails, prefer abstention or guided retake.
- If geometry is strong but 2-3 colors are ambiguous, prefer a low-friction
  manual-review/Fixer flow over forcing a retake.

## Core Principle

Build the recognizer around a single coherent projected cube model per photo,
not around independent local 3x3 grid detections.

A valid input image should be explained as three visible 3x3 cube faces sharing
one cube corner, with three families of bezel/grid lines, fixed cube topology,
and model-derived sticker cells. Every sampled sticker should come from this
same projected cube model.

## Proposed Pipeline

1. Guided capture

   Require two convention-driven photos:

   - Image A: white-up view.
   - Image B: yellow-up / flipped complementary view.
   - Each image should show exactly three cube faces.
   - Across A+B, all six centers should be observed.

   If a photo is blurry, cropped, too oblique, glare-heavy, missing a center, or
   inconsistent with the capture convention, prefer retake over guessing.

2. Cube object detection

   Segment and crop the cube region using silhouette, edges, and black plastic
   bezel structure. This stage should find the cube region and rough pose only;
   it should not commit to sticker identities.

3. Global cube geometry fit

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

4. Model-derived facelets

   Once the global cube model is trusted, derive all visible sticker
   quadrilaterals from that model. Do not independently discover sticker cells
   outside the model.

5. Per-face rectification

   Rectify each trusted visible face into a canonical flat square grid. Use
   rectification for color sampling and debugging, not as the source of geometry
   truth.

   The intended flow is:

   ```text
   photo -> global cube pose -> trusted face quads -> rectified faces -> color samples
   ```

6. Robust color classification

   Sample shrunken interior patches from each rectified cell. Use center
   stickers as per-image color anchors, combine Lab/HSV/RGB features, and keep
   top-N color probabilities instead of committing too early.

   Color classification is not currently believed to be the binding constraint
   when geometry is correct.

7. Two-view fusion

   Map the three visible faces from A and B into the six cube faces using center
   colors and the capture convention. Reject, request retake, or request human
   confirmation if faces are missing, duplicated inconsistently, or if A/B
   evidence conflicts.

8. Legal cube assignment

   Choose the lowest-cost legal cube state from the color probabilities while
   enforcing:

   - exactly nine of each color,
   - valid centers,
   - valid cubies,
   - legal orientation and parity,
   - a unique winner with a real margin.

   Legal solveability should be the final consistency constraint, not a repair
   mechanism for bad geometry.

9. Quality gates and UX

   The recognizer should have three possible outcomes:

   - confident solved cube,
   - retake one or both photos when geometry/capture quality fails,
   - confirm a small number of ambiguous stickers when geometry is good but
     residual color evidence is weak.

   To reach a 99% accurate delivered-solve rate, abstention, guided retake, and
   manual confirmation are part of the design, not failures. The split matters:
   retake is appropriate for geometry failure; manual review is appropriate for
   residual color ambiguity after a trusted geometry fit.

## Role of Rectification

Rectification remains valuable, but it should be downstream of trusted geometry.
It helps color detection by turning each visible isometric face into a stable
2D square grid, normalizing perspective, and allowing fixed interior sampling.

The failure mode to avoid is rectifying an arbitrary local quad that actually
spans multiple physical cube faces. Rectification can make invalid geometry look
clean; therefore, it should not be used as the geometry source of truth.

## Current Lessons That Motivate This

- Clean-label color classification appears near ceiling when samples come from
  correct face geometry.
- The hardest current failures are geometry failures: local grids or cells can
  cross physical face boundaries while still producing plausible color labels.
- Direct legal-state repair can make a bad geometry result look solveable; that
  should remain guarded by quality margins and hard-case evidence.
- Interior-bezel and cell-discontinuity diagnostics are useful because they
  measure whether candidate cells are consistent with physical face boundaries.
- Overlay feedback is especially valuable for identifying slot-level geometry
  failures and should guide future model-fit diagnostics.

## Role of Existing Diagnostics

The recent interior-bezel and cell-discontinuity work should be treated as
initializers and cross-checks for the global model, not as the primary geometry
path.

- Interior-bezel lines can initialize or validate the global model's cube
  center and face-boundary directions.
- Hull simplification and h-vertex candidates can provide outer-boundary
  anchors.
- Cell-discontinuity and bezel-crossing diagnostics can cross-check whether
  model-derived sticker cells remain inside a single physical face.

The important invariant remains stronger than any individual diagnostic:
the final sampled facelets should be generated by one coherent projected cube
model.

## Recommended Long-Term Direction

The highest-leverage architectural shift is a global cube model fitter:

```text
detect cube object
-> fit coherent projected 3-face cube model
-> derive visible sticker quads from that model
-> rectify trusted faces
-> classify colors from clean rectified samples
-> fuse A+B into six faces
-> choose a unique legal cube state
-> abstain/retake/confirm when margins are weak
```

This does not replace the current incremental trajectory immediately. It gives
us a north-star architecture for interpreting what the ongoing diagnostics are
teaching us.
