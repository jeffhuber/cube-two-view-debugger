# Post-#218 baseline + failure taxonomy

## Role of this document

This is the **legacy decision spine** for first-principles geometry
work as of 2026-05-22, per the Codex+Devin strategic synthesis:

> A plausible cube model is cheap. A trustworthy cube model is
> hard.
>
> First-principles work must either improve guardrails around
> `cv-local`, produce labeled training data, or demonstrate
> safe held-out improvement.

The global cube model is **NOT** on a near-term path to replace
`cv-local` as the primary recognizer. Its role is:

- a **trust layer** around `cv-local`: low-confidence geometry
  should route to manual/retake, not to legal repair;
- a **source of labeled training data** for a future learned
  vertex/axis ranker;
- a **benchmark harness**: every geometry-sensitive PR should
  emit row-level before/after deltas against this baseline, not
  just aggregate pass rates (see `--diff` mode below).

**2026-05-23 convention caution:** this report was computed against
`tests/fixtures/gcm_axis_ground_truth.json`, whose `near_x/near_y/near_z`
semantics are legacy. Initial audit against the 12-row full-corner seed
fixture shows those fields match the far/double-axis triplet (`A -> 0,2,4`,
`B -> 1,3,5`), not canonical one-edge labels. The canonical corner convention is
[`FULL_CORNER_LABELING.md`](FULL_CORNER_LABELING.md):
`A slots: upper=Va+1,0,5; right=Va+3,2,1; front=Va+5,4,3` and
`B slots: upper=Vb+2,3,4; right=Vb+0,1,2; front=Vb+4,5,0`. Canonical WCA
face names for side slots depend on capture yaw. Use this report for
historical context, but do not treat row-level `CHIRALITY_*` evidence as
canonical until the baseline is regenerated from full-corner truth.

## What the "chirality" concept actually is

The detector and code use the word "chirality" but it's a
slight misnomer — strict-geometry chirality means handedness
under reflection. What we actually face is a **60° near/far
phase ambiguity** stemming from the cube's 3-fold rotational
symmetry around its body diagonal.

In iso projection, the 6 outer hexagon-silhouette vertices
alternate between 3 NEAR corners (1 cube-edge from the visible
trihedral vertex) and 3 FAR corners (2 cube-edges via face
diagonal). Swap the labels and you get a different valid cube
pose that projects to the SAME silhouette under orthographic
projection — the 7-anchor Procrustes fit has identical residual
for either assignment.

**This is real first-principles geometry, not invented
complexity.** The 7 anchor points (vertex + 6 hex corners) give
14 measurements for a 6-DOF cube model — over-determined
linearly, but the solution set has two equally-valid options
under the body-diagonal symmetry. With perspective there's a
small distinguishing signal from foreshortening, but it's
noise-shaped at our anchor-extraction precision.

### What's principled vs empirical in the current detector

- **Principled (keep)**: that we need to resolve the phase at
  all, that the resolution requires evidence beyond the 7
  anchors, and that the model should expose a confidence
  signal for whether the resolution is trustworthy.
- **Empirical (treat as stopgap)**: the line-darkness signal,
  its inverted polarity (`sep<0` ≡ correct, opposite of naive
  bezel-darkness reasoning), and the |sep| < 10 ambiguous band.
  These are calibrated against the 58 cases, not derived from
  first principles. PR #218's 33pp accuracy lift from a single
  block reorder (vertex ensemble BEFORE phase check) is
  itself evidence that the detector is sensitive to non-
  geometric noise.

The architectural answer is to fold additional evidence INTO
the Procrustes fit (bezel-line angles, two-view consistency,
learned priors) rather than detect-and-correct downstream.
That's what the recommended sequence's step 6 (learned
vertex/axis ranker with calibrated abstention) actually is.
The current darkness detector becomes vestigial once that
lands.

## Baseline snapshot

Re-baseline of the global cube model after PR #213 (phase
auto-correct) and PR #218 (vertex ensemble before phase
check) landed.

**Eval set**: 58 cases × 2 runs each = 116 total runs.

Ground truth: legacy user-labeled vertex + 3 `near_*` corners per photo
(`tests/fixtures/gcm_axis_ground_truth.json`). The eval compares
model bearings (in gallery coords) to user bearings (in original
coords); bearings are exactly invariant under the gallery's
uniform-scale-and-translation crop, so no crop reconstruction is
needed. The `near_*` target semantics are provisional pending the
full-corner migration.

## Headline accuracy

| accuracy band                    | runs |  %  |
|----------------------------------|-----:|----:|
| <5°                              |   27 | 23.3% |
| 5-10°                            |   49 | 42.2% |
| 10-25°                           |   16 | 13.8% |
| 25-45°                           |    2 | 1.7% |
| >45°                             |   22 | 19.0% |

- **79.3% of runs land at <25° bearing error** (GOOD + MARGINAL).
- **20.7% of runs are catastrophic** (>25° err in one of the failure
  modes below).

## Failure taxonomy

| category                  | runs |  %  | meaning |
|---------------------------|-----:|----:|---------|
| GOOD                      |   76 | 65.5% | All axes within 10° of user labels — model is essentially right. |
| MARGINAL                  |   16 | 13.8% | 10–25° err — small jitter, color sampling probably still OK. |
| CHIRALITY_MISS            |   12 | 10.3% | Model.far matches user.near; detector said `correct` or `ambiguous` — flip needed but missed. |
| CHIRALITY_FALSE_FLIP      |   10 | 8.6% | Model.far matches user.near; detector said `corrected_60deg_flip` — wrongly flipped a previously-correct model. |
| TRUE_GEOMETRY_FAIL        |    2 | 1.7% | Neither model.near nor model.far matches user.near — fit is bad regardless of phase. |

## Case-level stability across runs

| outcome                       | cases |  %  |
|-------------------------------|------:|----:|
| always GOOD across runs       |    32 | 55.2% |
| always BAD (catastrophic+)    |    11 | 19.0% |
| mixed / varies between runs   |    15 | 25.9% |

Mixed cases are where the Procrustes 6! brute-force still picks
different symmetry-equivalent permutations across runs and the
phase detector doesn't reliably rescue all of them. The
deterministic-tie-breaker path was scoped (see NEAR_FAR_PHASE_REPORT.md
"What's next") but deprioritized in favor of higher-leverage work
on vertex localization.

## 10 worst cases (worst run per case)

| set | err_near | category | chir_check | sep |
|-----|---------:|----------|------------|----:|
| 62_B |   59.9° |   CHIRALITY_FALSE_FLIP |           corrected_60deg_flip | +13.9 |
| 46_B |   59.7° |         CHIRALITY_MISS |        ambiguous_no_correction | -7.6 |
| 47_B |   59.7° |   CHIRALITY_FALSE_FLIP |           corrected_60deg_flip | +34.4 |
| 62_A |   59.2° |   CHIRALITY_FALSE_FLIP |           corrected_60deg_flip | +28.5 |
| 25_B |   58.4° |   CHIRALITY_FALSE_FLIP |           corrected_60deg_flip | +16.9 |
| 44_A |   57.1° |         CHIRALITY_MISS |                        correct | -35.5 |
| 57_A |   55.7° |         CHIRALITY_MISS |                        correct | -10.9 |
| 46_A |   55.5° |   CHIRALITY_FALSE_FLIP |           corrected_60deg_flip | +10.1 |
| 58_A |   55.1° |         CHIRALITY_MISS |        ambiguous_no_correction | -6.9 |
| 30_A |   52.6° |         CHIRALITY_MISS |                        correct | -22.0 |

## What this baseline says

1. The phase auto-correction (PRs #210/#213/#218) handles the
   dominant failure mode at the symptom level. The remaining
   catastrophic band is largely phase-decision miscalls
   (detector said correct/ambig but should have flipped, or
   flipped when shouldn't have).
2. Detector confidence (|sep|) tracks success. Strong-|sep| commits
   are almost always right; the failures cluster in the weak-|sep|
   band where neither commit-correct nor commit-flip is reliable.
3. The case-level non-determinism is real — the Procrustes 6!
   brute-force is the root cause. A deterministic tie-breaker
   would address it directly, but is **explicitly deprioritized**
   per the Codex+Devin strategic shift (no more handcrafted
   vertex/phase heuristics — the bar is now "safe held-out
   improvement" or "feeds the trust layer").

## Recommended next sequence (per Codex+Devin)

In order, gated on the previous step demonstrating value:

1. **This baseline + taxonomy artifact** — done by this PR. Sets
   the regression gate.
2. **Geometry trust-policy diagnostics** — add `model.debug`
   fields for phase confidence, axis agreement against the
   detected bezels, face-quad-consistency, grid/source-
   contamination. Diagnostics-only; no behavior change.
3. **Guardrail experiment** — route low-trust cases to
   manual/retake. Success metric is *fewer confident wrong
   solves with tolerable abstention*, NOT "more solves."
   The product framing here is Devin's: "use first-principles
   diagnostics to decide when to ask the user for a second
   capture / manual fixer path."
4. **Stable labeled benchmark harness** — every geometry-
   sensitive PR runs `tools/baseline_post_218.py --diff`
   against the merge-base baseline and reports row-level
   deltas. PR body must include the diff summary. Aggregate
   mean/p95 metrics are NOT sufficient (Devin: "some changes
   improve averages while worsening critical rows").
5. **Active-label queue for the eval set** — extend the 58
   labels systematically rather than opportunistically. Pick
   the next photos to label based on which model-disagreement
   regions are currently under-sampled, not on which photos
   look visually interesting.
6. **Learned vertex/axis ranker** — only after (2)–(5)
   demonstrate the trust layer works. Must train/eval with
   held-out splits and calibrated abstention. Note Devin's
   caveat: "a candidate oracle means a localizer is near
   production" is FALSE — the search space containing truth
   doesn't mean ranking/confidence is solved. Calibrated
   abstention is the actual deliverable, not top-1 accuracy.

## What's explicitly off the table

Per the Codex+Devin synthesis, these are NOT next bets:

- More handcrafted vertex/phase heuristics (dark-line variants,
  junction extractors, scalar scorers). Diminishing returns; the
  sprint repeatedly falsified them at the safe-coverage bar.
- More SAM3 prompt bakeoffs without a materially new signal.
- Replacing `cv-local` with the global cube model. The global
  model is scaffolding, not a recognizer.
- Aggregate-metric-only A/B comparisons (without row-level diff).

## How to use this as a regression gate

Before a geometry-sensitive PR:
```bash
.venv/bin/python tools/baseline_post_218.py \
  --out /tmp/baseline_my_branch.json --report /dev/null
```

Then diff:
```bash
.venv/bin/python tools/baseline_post_218.py \
  --diff tests/fixtures/post_218_baseline.json /tmp/baseline_my_branch.json
```

Paste the diff summary into the PR body. A PR that regresses
any case from GOOD → catastrophic without offsetting wins is a
blocker for merge regardless of aggregate metrics.

## Reproducing

```bash
.venv/bin/python tools/baseline_post_218.py \
  --truth tests/fixtures/gcm_axis_ground_truth.json \
  --gallery ~/axis_labeling \
  --runs 2 \
  --out tests/fixtures/post_218_baseline.json \
  --report tools/POST_218_BASELINE_AND_TAXONOMY.md
```
