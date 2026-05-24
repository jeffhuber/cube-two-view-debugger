# cv-local improvement design — diagnostics-first

Four targeted improvements to the cv-local recognizer's recognition pipeline,
prioritized by leverage-per-effort. Each is independent and verifiable against
the labeled corpus (28 pairs; cv-local currently 8/15 full match,
41.5/54 avg sticker accuracy on pairs with cv-local data).

Designs derived from the global-cube-model + per-photo-classifier prototyping
work (PR #182 and follow-up experiments, 2026-05-21):

- Vertex precision below ~150 px doesn't affect color sampling reliability —
  face quad SHAPE matters, not exact vertex position.
- Per-photo K-means + center anchors works for color separation when lighting
  is good; fails under shadow / low contrast.
- ~30% of cell-center samples catch bezel pixels with fixed-radius windows;
  saturation-aware sampling cuts this nearly in half.
- 7/15 cv-local non-full-matches fall into two patterns: 3 severe (>70%
  mismatches, likely face-grid disagreement) + 4 subtle (<25% mismatches,
  often orange↔red, white↔green confusions).

## Common terminology

- **Photo pair**: two photos of one cube (A = white-up; B = yellow-up
  after the single 180° camera-X flip). At yaw=0 the visible WCA faces
  are URF/DLB; non-zero capture yaw rotates the side-face identities.
- **Sticker**: one of 54 colored squares on the cube. Each photo shows 27.
- **Face grid**: the 3×3 arrangement of 9 stickers per visible face.
- **Cell**: one of 9 positions within a face grid; we sample color from it.
- **Bezel**: the black border between stickers (usually 5-15% of cell width).
- **Sample patch**: small pixel region we read color from at each cell center.

---

## Improvement #2 — Saturation-aware cell sampling (TESTED, NEGATIVE RESULT — abandoned)

**Status update 2026-05-22:** Implemented and tested against the labeled
corpus. **Result: -37 stickers aggregate, -1 full match** (17 pairs
evaluable, baseline 83.2% sticker accuracy, new 79.2%). Even after
surgical scoping (apply filter only at final-read call sites, not at
grid-candidate-annotation paths) the regressions persisted: set 29 went
54→20, set 61 34→21, set 27 42→31, set 24 54→44. The 70th-percentile
filter is too aggressive — it discards legitimate sticker pixels along
with bezel pixels, biasing reads on uniformly-colored stickers.

Lesson learned: cv-local's existing donut sampling (avoiding inner 38% of
the patch) already handles most bezel exposure. The remaining contamination
isn't dominated by patch-edge pixels — it's from genuinely-misaligned
patches (face grid is off, patch isn't on the sticker at all). Filtering
pixel-by-pixel doesn't help when the whole patch is wrong.

What would help: (a) better grid alignment in the first place (covered
by #1 — global cube model), or (b) sample-window quality scoring that
detects "this patch is contaminated" and emits low confidence, rather
than trying to median-out the bezel.

**Hypothesis (original):** cv-local's `_sample_patch` uses a fixed donut-shaped
median window. Pixels at the outer ring catch bezel (~30% of cells observed
in my prototyping). Filtering those out via per-pixel saturation/value
scoring before taking median lifts color quality without changing geometry.

**Why the hypothesis was wrong:** ~30% bezel contamination doesn't directly
translate to mis-classification because the median is already robust to
30% outliers (50th percentile passes through). The actual failure cases
aren't "bezel pixels in the patch dragging the median" — they're "the
patch is entirely on the wrong sticker."

**Specific change:**
In `rubik_recognizer/image_pipeline.py` `_sample_patch`:
- Current: median of all donut pixels, equal-weighted
- New: for each pixel in the donut, compute `score = max(saturation, value/255)`.
  Take median only of pixels with `score >= percentile(scores, 70)`.
- Falls back to median of brightest 30% if no pixel has high saturation
  (uniformly-white sticker case).

**Why this works (theoretically):**
- Bezel pixels are dark and unsaturated → score < 0.3
- White stickers are bright but unsaturated → score = brightness ≈ 0.8+
- Other cube colors (R/O/G/B/Y) are highly saturated → score = saturation ≈ 0.6-0.9
- Robust to per-pixel noise (still takes median, just over a curated subset)

**Risks:**
- Could hurt on very dark cube colors under low light (blue under shadow
  might have score < 0.3 → falls back to brightest-30%, biases toward
  any glare/specular highlight)
- Cell at the bezel-edge case (whole patch is bezel) — falls back to
  brightest-30%, picks up edge of neighboring sticker. We're not making
  this worse than current behavior.

**Expected impact:**
- 5-15 percentage point per-sticker accuracy lift on shadowed/low-light cases
- Smaller impact on already-clean cases (most pixels score high anyway)

**Test plan:**
1. Apply to `_sample_patch`. Verify no behavior change on synthetic well-lit
   stickers (unit test).
2. Re-run cv-local on the full labeled corpus.
3. Compare per-pair sticker mismatches: new vs current. Net positive expected.
4. Check the 7 failure cases specifically: should help most on the "subtle
   color confusion" cases (sets 57, 58, 27, 61), less impact on severe-failure
   cases (24, 28, 31, which have geometric issues).

**Files changed:** `rubik_recognizer/image_pipeline.py`. ~30 lines.

**Diagnostic-only:** no, this changes recognition behavior. Gated by feature
flag `RUBIK_SATURATION_AWARE_SAMPLING=1` initially; default off until
validated.

---

## Improvement #3 — Two-view geometric consistency (TESTED, SHIPPED AS DIAGNOSTIC)

**Status update 2026-05-22:** Implemented as a new `twoViewGeometryConsistency`
field in `recognition_signals`. Validated against the labeled corpus:

| accuracy tier | n  | median ratio | max ratio |
|---------------|----|--------------|-----------|
| full match (54/54) | 10 | 1.03 | 1.19 |
| partial (30-53) | 6 | 1.08 | 1.14 |
| failure (<30) | 4 | 1.05 | 1.18 |
| no recognition (N/A) | 7 | 1.07 | **1.54** |

Signal is **weakly predictive**: high spacing ratios (>1.20) correlate
strongly with "no recognition produced" (3/3 such cases have
ratio ≥ 1.24), but the partial-vs-success distinction is weak (ratios
mostly overlap in 1.0-1.15 range). Ships as a diagnostic-only signal
that's available to downstream consumers (e.g., the
needs_manual_review category could use it as one of several inputs)
but doesn't drive recognition behavior on its own.

Tolerance for `inconsistent=True` flag defaults to 1.4 (configurable
via `RUBIK_TWO_VIEW_RATIO_TOLERANCE`), which fires on the most
extreme cases (1/26 in the corpus eval). A tighter threshold (1.20)
catches more but at risk of false positives on cases like set 62
(ratio 1.19, recognized perfectly).



**Hypothesis:** image A and image B are photos of the same physical cube.
Their fitted face-grid geometries should have compatible parameters:
- Same sticker spacing (up to image-scale ratio)
- Compatible cube rotations (the cube's 3D orientation is constrained
  between photos by the user's flip convention)

When the two photos produce geometrically incompatible fits, it's a signal
that one of them has a bad fit. Currently cv-local fits each photo
independently with no cross-check.

**Specific change:**
In `rubik_recognizer/recognizer.py` add a `_two_view_consistency_check`
function called after `_recognize_from_analyses` builds initial state:
- Extract sticker spacing per photo (median across 3 faces × multiple cells)
- Check ratio: image_A_spacing / image_B_spacing should be near 1.0 (the
  cubes are the same size; only image scale differs from camera distance,
  which is bounded by capture convention)
- If ratio < 0.7 or > 1.4: flag as `two_view_inconsistent`
- Add new recognition signal field

**Why this works (theoretically):**
- Same physical cube → ~same physical size in pixels under similar capture
  conditions
- A real outlier (wrong face grid fitted) will have very different spacing
- Catches the geometric-failure class (sets 24, 28, 31) where one or both
  photos have face-grid mis-fits

**Risks:**
- False positives on cases where one photo is genuinely closer to camera
  (user moved between A and B captures). Tolerance ratio loose enough to
  absorb 30% scale difference.
- Doesn't directly fix anything; just adds a confidence signal

**Expected impact:**
- Detect 2-4 of the 7 cv-local failures pre-emptively
- Can be wired into existing `needs_manual_review` category trigger
- Doesn't lift accuracy on its own; it's a "refuse early" signal

**Test plan:**
1. Compute spacing per face per photo for all corpus pairs
2. Compare A/B spacing ratio on full-match vs non-full-match cv-local cases
3. Verify: failures should cluster at extreme ratios; successes at ratio ≈ 1.0
4. If signal is clean, wire into category logic as new "weak signal"

**Files changed:** `rubik_recognizer/recognizer.py` + tests. ~50 lines.

**Diagnostic-only:** yes initially. New signal field; existing categories
unchanged unless signal is statistically validated.

---

## Improvement #1 — Global cube model as additional grid candidate (BIGGER PR)

**Hypothesis:** cv-local fits 3 face grids per photo independently and
selects the best per face. This can choose 3 face grids that are
individually plausible but mutually inconsistent. The 3 severe failures
(sets 24, 28, 31) show this pattern: 70%+ mismatches consistent with
swapped face IDs.

A single 6-DOF global cube model (PR #182's `fit_global_cube_model`)
produces 3 coherent face quads by construction. Feeding these as
ADDITIONAL grid candidates lets cv-local's existing grid selector pick
them OR pick its own — but it now has a global-consistent option available.

**Specific change:**
In `rubik_recognizer/image_pipeline.py` `_fit_face_grids`:
- After running existing grid candidate generation
- Call `tools.global_cube_model.fit_global_cube_model` (already on main
  via PR #182)
- Convert the resulting 3 face quads into the same FaceGrid format
- Add them to the candidate list with a special source tag
  (`source='global_model'`)
- Existing selection logic chooses among them as usual

**Why this works (theoretically):**
- When cv-local's per-face fit is bad, the global model offers an
  alternative that's geometrically self-consistent
- The selector picks whichever scores higher
- If global model fits poorly (rare), cv-local's own candidate is selected
  → no regression

**Risks:**
- Global model adds ~1-3 sec per photo (rembg + PnP + ensemble)
- Bad global model fits could "win" selection if scored generously
- Increased complexity of `_fit_face_grids`

**Expected impact:**
- Lift on the 3 severe-failure cases (24, 28, 31) — these are exactly the
  cases the global model is built to solve
- Small or no change on cases where cv-local's per-face fit was already good

**Test plan:**
1. Run global model on the 7 cv-local failure cases, capture face quads
2. Visually inspect: are global-model face quads better than cv-local's?
3. Implement candidate injection
4. Re-run full corpus, measure full-match rate change
5. Specifically check the 3 severe cases — do they recover?

**Files changed:** `rubik_recognizer/image_pipeline.py` (~80 lines new code
+ existing global_cube_model.py import). ~100 LOC.

**Diagnostic-only:** yes initially. Behavior change gated by feature flag
`RUBIK_GLOBAL_MODEL_CANDIDATES=1`.

---

## Improvement #4 — Probabilistic state output (BIGGEST CHANGE)

**Hypothesis:** cv-local commits to one color label per sticker, then runs
repair if the resulting state is invalid. A more principled approach:
output a probability distribution per sticker (P(color)), then search the
space of valid cube states for the one with highest joint likelihood.

**Specific change:**
- `classifySample` (already returns nearest + confidence) → return all 6
  distances → softmax to P(color)
- New `_solve_state_likelihood` that enumerates legal corner/edge cubie
  configurations and scores by joint per-sticker probability
- Replace `_legal_repair_candidates` with this likelihood-based search

**Why this works (theoretically):**
- Subsumes cv-local's repair path as a special case (deterministic
  classification = degenerate Dirac probability distribution)
- Uses cube physics + per-sticker confidence holistically
- Handles ambiguous stickers (e.g., orange vs red on a borderline sample)
  by deferring decision to whichever produces a more-likely valid state

**Risks:**
- Likely the most invasive change to cv-local
- Need to be careful: likelihood-search must be bounded (8 corners × 24
  orientations + 12 edges × 2 orientations is small; the full state space
  is huge but constrained)
- Validation against current repair-path results requires reproducing the
  existing test suite

**Expected impact:**
- Lift on subtle-confusion cases (sets 57, 58 — currently "high confidence
  but wrong"); the joint likelihood would prefer states with more
  cube-physics-consistent corner/edge arrangements
- Modest cost: needs to be implementable without slowing down by >2×

**Test plan:**
1. Add softmax-distance output to classifySample
2. Stub likelihood-search returning a single argmax result (= current
   behavior); make sure tests still pass
3. Replace argmax with proper likelihood search
4. Measure: full-match rate vs current

**Files changed:** `rubik_recognizer/colorCalibration.py` (or equivalent),
`rubik_recognizer/recognizer.py`. ~200 LOC.

**Diagnostic-only:** behavior change.

---

## Implementation order + checkpoints

1. **#2 first** — smallest surgical change, clear hypothesis, low risk.
   Ship as feature-flagged improvement. ~1 day.
2. **#3 second** — purely additive (new signal); zero regression risk.
   Wire later when signal proves predictive. ~1 day.
3. **#1 third** — bigger change but builds on shipped global_cube_model.py.
   Behind feature flag; doesn't replace existing path until validated.
   ~1-2 days.
4. **#4 last** — architectural; do only after #1-#3 validate the
   approach. ~2-3 days.

Each PR independently mergeable. Each goes through Devin audit via
the `needs-devin-audit` / `devin-audit-done` label protocol.
