# Phase 1: cv-local baseline on 58-case axis-labeled gallery

## Role of this document

Companion to `POST_218_BASELINE_AND_TAXONOMY.md`. Same eval
set, same metric (per-axis bearing error vs user labels),
same categorization. The difference: this measures the
**production `cv-local` recognizer** instead of the global
cube model.

cv-local's `analyze_image` outputs face-quads, not vertex+
near-corners directly. This baseline derives the comparable
(vertex, [3 near corners]) signal by clustering the 12 corner
instances (4 per face-quad × 3 face-quads) across faces:

- 1 cluster of 3 points (one per face) → trihedral vertex
- 3 clusters of 2 points (each shared by 2 adjacent faces) → 3 near corners
- 3 clusters of 1 point (one per face) → 3 far corners

Algorithm: union-find on cross-face edges in ascending
distance order, threshold 120 px, with the
constraint that the first 3-face cluster found is the vertex
(no other cluster can grow to 3 face members afterward).
Cases where the expected 1×3 + 3×2 + 3×1 cluster pattern
doesn't emerge count as `CV_LOCAL_FIT_FAIL`.

## Headline accuracy

**58 cases** (6 scorable, 52 cv-local fit-fail).

| accuracy band | cases |  %  |
|---------------|------:|----:|
| <5°           |     0 | 0.0% |
| 5-10°         |     0 | 0.0% |
| 10-25°        |     0 | 0.0% |
| 25-45°        |     1 | 1.7% |
| >45°          |     5 | 8.6% |

## Category breakdown

| category               | cases |  %  |
|------------------------|------:|----:|
| GOOD                   |     0 | 0.0% |
| MARGINAL               |     0 | 0.0% |
| CHIRALITY_MISS         |     3 | 5.2% |
| TRUE_GEOMETRY_FAIL     |     3 | 5.2% |
| CV_LOCAL_FIT_FAIL      |    52 | 89.7% |

## 10 worst (scorable) cases

| case | err_near | category |
|------|---------:|----------|
| 17_A |   80.3° | TRUE_GEOMETRY_FAIL |
| 12_B |   59.9° | CHIRALITY_MISS |
| 23_A |   59.9° | CHIRALITY_MISS |
| 31_A |   51.6° | TRUE_GEOMETRY_FAIL |
| 37_B |   46.4° | CHIRALITY_MISS |
| 29_B |   39.9° | TRUE_GEOMETRY_FAIL |

## cv-local fit-failures

| case | reason |
|------|--------|
| 12_A | cluster_pattern_mismatch (n_face_quads=3) |
| 14_A | cluster_pattern_mismatch (n_face_quads=3) |
| 14_B | fewer_than_3_face_quads (n_face_quads=2) |
| 15_A | cluster_pattern_mismatch (n_face_quads=3) |
| 15_B | cluster_pattern_mismatch (n_face_quads=3) |
| 17_B | cluster_pattern_mismatch (n_face_quads=3) |
| 21_A | cluster_pattern_mismatch (n_face_quads=3) |
| 21_B | cluster_pattern_mismatch (n_face_quads=3) |
| 22_A | cluster_pattern_mismatch (n_face_quads=3) |
| 22_B | cluster_pattern_mismatch (n_face_quads=3) |
| 23_B | cluster_pattern_mismatch (n_face_quads=3) |
| 24_A | cluster_pattern_mismatch (n_face_quads=3) |
| 24_B | cluster_pattern_mismatch (n_face_quads=3) |
| 25_A | cluster_pattern_mismatch (n_face_quads=3) |
| 25_B | cluster_pattern_mismatch (n_face_quads=3) |
| 26_A | cluster_pattern_mismatch (n_face_quads=3) |
| 26_B | fewer_than_3_face_quads (n_face_quads=2) |
| 27_A | cluster_pattern_mismatch (n_face_quads=3) |
| 27_B | fewer_than_3_face_quads (n_face_quads=2) |
| 28_A | cluster_pattern_mismatch (n_face_quads=3) |
| 28_B | cluster_pattern_mismatch (n_face_quads=3) |
| 29_A | fewer_than_3_face_quads (n_face_quads=0) |
| 30_A | cluster_pattern_mismatch (n_face_quads=3) |
| 30_B | cluster_pattern_mismatch (n_face_quads=3) |
| 31_B | cluster_pattern_mismatch (n_face_quads=3) |
| 32_A | cluster_pattern_mismatch (n_face_quads=3) |
| 32_B | cluster_pattern_mismatch (n_face_quads=3) |
| 36_A | cluster_pattern_mismatch (n_face_quads=3) |
| 36_B | cluster_pattern_mismatch (n_face_quads=3) |
| 37_A | cluster_pattern_mismatch (n_face_quads=3) |
| 39_A | cluster_pattern_mismatch (n_face_quads=3) |
| 39_B | fewer_than_3_face_quads (n_face_quads=1) |
| 42_A | cluster_pattern_mismatch (n_face_quads=3) |
| 42_B | cluster_pattern_mismatch (n_face_quads=3) |
| 44_A | cluster_pattern_mismatch (n_face_quads=3) |
| 44_B | fewer_than_3_face_quads (n_face_quads=1) |
| 46_A | fewer_than_3_face_quads (n_face_quads=1) |
| 46_B | cluster_pattern_mismatch (n_face_quads=3) |
| 47_A | fewer_than_3_face_quads (n_face_quads=0) |
| 47_B | cluster_pattern_mismatch (n_face_quads=3) |
| 48_A | fewer_than_3_face_quads (n_face_quads=1) |
| 48_B | cluster_pattern_mismatch (n_face_quads=3) |
| 49_A | fewer_than_3_face_quads (n_face_quads=2) |
| 49_B | cluster_pattern_mismatch (n_face_quads=3) |
| 57_A | fewer_than_3_face_quads (n_face_quads=0) |
| 57_B | cluster_pattern_mismatch (n_face_quads=3) |
| 58_A | cluster_pattern_mismatch (n_face_quads=3) |
| 58_B | cluster_pattern_mismatch (n_face_quads=3) |
| 61_A | cluster_pattern_mismatch (n_face_quads=3) |
| 61_B | cluster_pattern_mismatch (n_face_quads=3) |
| 62_A | cluster_pattern_mismatch (n_face_quads=3) |
| 62_B | fewer_than_3_face_quads (n_face_quads=2) |

## Headline finding: cv-local face-quads are not geometrically consistent

**90% of cases fail the structural consistency check.** This
isn't a bug in the derivation — it's a real property of cv-
local's face-quad output that this script measures honestly.

`cv-local` produces face-quads by **independently extrapolating**
a 4-corner quad from each detected 3×3 sticker grid. There's
no constraint enforcing that the 3 face-quads share a
trihedral vertex or pairwise-share near corners. Each face
sees only its own stickers; it doesn't know about the others.

So on most cases, the 3 cv-local face-quads taken together
don't represent a single coherent projected cube — they're
3 disconnected quadrilaterals. The structural-clustering
derivation correctly reports this as a fit failure.

Of the 6 cases where cv-local DID produce a structurally
consistent set (lucky alignment of grid extrapolations), all 6
are catastrophic-error categorizations (>25° err) — meaning
even when the structure is consistent, the geometry is wrong.

### What this means for Phase 1's stated goal

The Phase 1 plan said: "Two committed JSON snapshots, both
runnable via `--diff` for row-level deltas between the two
sides." The mechanical infrastructure for that is now in
place. But the actual finding is more useful than the diff:

- **The two sides are not comparable via this derivation.**
  The global model produces well-formed (vertex, 3 near, 3
  far) tuples on 116/116 runs (post-#218 baseline). cv-local
  produces them on 6/58 cases.
- **cv-local doesn't "see" the cube as a single object** at
  the geometry layer. It detects faces independently. This is
  a structural difference, not a calibration issue.
- **For Phase 2 trust diagnostics, we want a different cross-
  system metric.** Options:
  1. **cv-local-side metric**: per-sticker accuracy on the
     same 58 photos (production-style eval, not derived
     geometry). Uses `tools/evaluate_hybrid_pipeline.py`.
  2. **Consistency-as-trust-signal**: the fact that cv-local's
     face-quads ARE/ARE NOT structurally consistent is itself
     a candidate trust signal. "If the 3 face-quads don't
     share a vertex within X px, route to retake."
  3. **Hybrid pipeline that uses global model for geometry,
     cv-local for color** — Codex's #152 hybrid experiment
     went this direction but found that arbitrary cv-local
     quads produce bimodal accuracy.

Option 2 is the most natural input for Phase 2's trust-policy
diagnostics.

## How to compare against the global model

```bash
.venv/bin/python tools/baseline_post_218.py \
  --diff tests/fixtures/post_218_baseline.json \
        tests/fixtures/cv_local_baseline.json
```

This produces row-level deltas: which cases the global model
gets right but cv-local misses, which the reverse, which both
agree on.

## Reproducing

```bash
.venv/bin/python tools/baseline_cv_local.py
```
