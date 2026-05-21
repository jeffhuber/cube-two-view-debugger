# Global cube projection model — diagnostics

Diagnostics-only probe that fits a single global cube model per photo
so that all 27 sticker quads come from one coherent geometry. This is
the post-pivot Tier 2 work (per Decision Log entry **2026-05-20** in
`COORDINATION.md`).

**Status:** prototype validated against 27-case user ground truth
(`tests/fixtures/gcm_vertex_ground_truth.json`). **Median vertex error
77 px, p95 120 px, 11/27 cases within 50 px.** Not the "17/18 strong"
claim from earlier iterations of this doc — that was overconfident
visual spot-checking before ground truth landed.

## Motivation

Three independent first-principles design proposals (Claude, Codex,
Devin) converged on the same architecture:

> Replace per-sticker / per-grid local detection + post-hoc
> reconciliation with a SINGLE GLOBAL CUBE PROJECTION MODEL per photo
> whose 27 sticker quads are coherent **by construction**.

The structural failure mode of "local detection then reconcile"
disappears when the parameterization itself guarantees coherence.

## Architecture (current)

```
silhouette (rembg)
  → 6-corner Visvalingam-Whyatt simplification of convex hull
    → cv2.solvePnP (fx=0.82*image_max_dim, calibrated iPhone-like K)
      → mean-of-3 vertex ensemble (PnP + bezel + hex_centroid)
        → score-gated image-based vertex refinement
          → 6-DOF cube pose → 27 sticker cells
```

Each stage is independently testable; vertex error is dominated by
upstream silhouette quality.

### Camera intrinsics

`fx = fy = 0.82 × max(image_h, image_w)`. iPhone main cameras have
~26mm-equivalent focal length on a 1/1.7" sensor, giving fx ≈ 3300 px
on a 4032-wide capture. Defaulting to `image_max_dim` (4032) overshoots
by ~22% and biased PnP toward systematic vertex offset of 50-100 px
under perspective+yaw. Validated 2026-05-21: switching to 0.82× reduces
RMS on 6 of 7 user-flagged-wrong cases.

### Mean-of-3 vertex ensemble

After PnP fits the hexagon corners, we override the trihedral vertex
with the mean of three independent estimates:

- **PnP-derived vertex** — from the 6-corner fit
- **Bezel-detected vertex** — from `interior_bezel_detection.py` Hough line intersection
- **Hexagon centroid** — mean of the 6 detected hexagon corners

On the 27-case ground truth, this ensemble has mean error 77 px vs
99 px for any single method alone (-22%). PnP's axes are kept; only
the vertex (and derived geometry, shifted by the same delta) is
overridden.

Rationale: each method has semi-independent error sources (PnP from
silhouette noise propagation, bezel from Hough line uncertainty, hex
centroid from perspective bias). Averaging regresses toward truth.
Tried weighted variants — uniform mean is within 6 px of an oracle that
picks the best per case.

### Score-gated image refinement

After the ensemble, we search a ±40 px window around the candidate
vertex for the strongest "3-way dark-line junction" using the
PnP-derived axis directions. Score is `min(darkness_along_3_dirs)` —
penalizes false junctions where only 1-2 directions have dark lines
(sticker corners look "dark" in one direction, not three).

Gated: refinement applied only when score at the ensemble vertex is
below threshold (default 200). High-score cases mean the ensemble is
already at a real junction; refinement on them tends to drift to
nearby sticker corners with marginally higher local score.

On the 27-case ground truth this lowers mean error 77 → 71 px (8%),
median 77 → 69 px (-10%), and helps the worst cases substantially
(42_B 161 → 78 px, 32_B 158 → 36 px). One small regression on a
borderline case (31_B 26 → 34 px). Doesn't fix the truly bad cases
(44_A stayed at 186 px) — those are silhouette-input failures, not
refinement-window failures.

## Ground-truth-validated stats

| metric | rembg + mean3 + refinement |
|---|---|
| n cases evaluated | 23 (with user-marked true_vertex) |
| mean vertex error | 77 px |
| median | 76 px |
| p95 | 120 px |
| max | 161 px |
| within 30 px | 5 |
| within 50 px | 9 |
| within 75 px | 13 |
| within 100 px | 17 |

User's "correct" threshold: ~75 px in this round. By that bar, 13/23
cases pass (57%), 17/23 (74%) are within 100 px (close enough for
sticker sampling given 200-300 px cell width), and 2 cases (44_A,
44_B) are >100 px.

Per-case details and the user's true_vertex marks are in
`tests/fixtures/gcm_vertex_ground_truth.json`.

## Why 44_A doesn't fix

Tested image-based refinement diagnostic on 44_A:

- True vertex has junction score 231 (HIGHER than any candidate in
  the ±40 search window — so the score metric is correct)
- BUT the true vertex is 94+ px from the ensemble starting point,
  outside the refinement window
- Widening the window introduces drift on already-good cases
- Better silhouette (SAM 3, tested 2026-05-21) recovers 44_B (118 → 15)
  but only partially helps 44_A (186 → 148)

44_A is therefore a "fundamentally hard capture" case for silhouette-
based methods. In production this is what guided capture should
prompt-to-retake, not algorithmically rescue.

## SAM 3 silhouette source (optional)

`--silhouette-source sam3` swaps rembg for SAM 3 masks (pre-computed
via `tools/extract_sam3_masks.py`; SAM 3 requires Python 3.13 +
CUDA-or-MLX environment, so it's an offline step).

| metric | rembg | SAM 3 |
|---|---|---|
| mean | 79 | 72 (-9%) |
| median | 72 | 57 (-21%) |
| <30 px | 0 | 6 |
| <50 px | 5 | 9 |
| p95 | 124 | 147 (worse) |
| max | 186 | 186 (44_A) |

SAM 3 is moderately better on average and dramatically better on some
cases (44_B 118 → 15 px), but introduces 5 new regressions (worst:
42_B 89 → 186 px) because the slightly different mask shape can push
Visvalingam into a different PnP basin. **Default remains rembg**
since it's <1s/image vs ~7s for SAM 3, and the mean improvement is
modest. SAM 3 is documented as a diagnostics / server-side option.

Per-face concept prompts ("top face", "yellow sticker") don't cleanly
work — would require fine-tuning SAM 3 on a cube-specific dataset.
Worth revisiting if we ever want to skip the PnP pipeline entirely.

## Where this fits in the system

**Diagnostics-only.** This pipeline is NOT wired into the recognizer.
It produces per-photo vertex + axes + face quads + 27 sticker cells,
which downstream consumers can use to:

- Validate capture quality (vertex confidence ≥ threshold → "OK"; lower → "retake")
- Drive guided-capture UI (live overlay showing where the model thinks the cube is)
- Provide geometric backbone to a color classifier (sample each cell, classify)

Wiring into actual recognition (replacing the cloud LLM recognizer or
the cv-local Fixer path) is a separate decision predicated on:
- Per-cell color sampling robustness against the floor's vertex error
- Two-view stitching (A + B sides → 54-char state)
- Production reliability vs the current cloud recognizer

## Dependencies

| Dependency | Status |
|---|---|
| numpy | required |
| Pillow | required |
| scipy | optional (`spatial.ConvexHull` for hexagon extraction, `optimize` legacy path) |
| opencv-python | required (`cv2.solvePnP`) |
| rembg | required for default silhouette path |
| sam3 / mlx-sam3 | optional, for `--silhouette-source sam3` |

## Files

| File | Purpose |
|---|---|
| `tools/global_cube_model.py` | Core module: PnP fit, mean3 ensemble, score-gated refinement |
| `tools/test_global_cube_model.py` | Test driver + 3-panel visualization |
| `tools/extract_sam3_masks.py` | Offline SAM 3 mask extraction (separate Python 3.13 env) |
| `tools/GLOBAL_CUBE_MODEL.md` | This doc |
| `tests/test_global_cube_model.py` | Unit tests (5 passing) |
| `tests/fixtures/gcm_vertex_ground_truth.json` | User-marked true vertex positions for 27 cases (durable regression fixture) |

## Reproducing the headline numbers

```bash
# Default: rembg silhouette
.venv/bin/python tools/test_global_cube_model.py \
    --sets 12 14 15 17 21 23 24 26 27 28 29 30 31 32 36 37 42 44 47 57 58 61 \
    --out /tmp/gcm_run

# With SAM 3 (pre-compute masks first, in SAM 3 venv):
# (in mlx_sam3 env, Python 3.13)
.venv/bin/python /path/to/tools/extract_sam3_masks.py \
    --sets 12 14 15 17 21 23 24 26 27 28 29 30 31 32 36 37 42 44 47 57 58 61 \
    --out /tmp/sam3_masks
# (back in cube-two-view-debugger env)
.venv/bin/python tools/test_global_cube_model.py \
    --sets 12 14 15 17 21 23 24 26 27 28 29 30 31 32 36 37 42 44 47 57 58 61 \
    --silhouette-source sam3 --sam3-mask-dir /tmp/sam3_masks \
    --out /tmp/gcm_sam3_run
```

Then compare against `tests/fixtures/gcm_vertex_ground_truth.json`
to compute per-case vertex error.

PNGs from `--out` are not committed (large binaries); the generator,
this doc, and the JSON ground-truth fixture are the durable artifacts.

## What to validate next

The natural follow-ups, in priority order:

1. **Guided capture wiring** in `cube-snap` — the architectural answer
   to the 2/28 unrecoverable cases is "refuse to commit on bad
   captures." Real-time vertex computation in mobile Safari, live
   overlay shows confidence, prompts user to adjust until locked.
   Phase 0 (feasibility): can the pipeline (or a stripped version)
   run at 10 fps or even 1 fps in browser?
2. **Two-view consistency**: photos A and B of the same cube should
   produce geometrically consistent outputs (same cube edge length,
   compatible yaw). Useful cross-check + could tighten the floor by
   joint fit.
3. **Face-identity resolution + sticker color sampling**: combine
   model output with center-sticker color reads to assign U/F/R/D/L/B
   labels, then test downstream solveable-cube delivery rate.
4. **SAM 3 fine-tune for per-face segmentation** (deferred — see
   user nice-to-have backlog). Would eliminate the PnP/vertex
   pipeline entirely if successful.
