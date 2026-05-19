# AMG face-quad refiner — research findings

`tools/amg_face_refiner.py` is a **standalone research tool**, NOT wired
into the production recognizer or `tools/evaluate_hybrid_pipeline.py`.
This document records what we learned, why we didn't ship to production,
and how to reproduce / extend the experiment.

## TL;DR

We tried using SAM2's Automatic Mask Generator (AMG) to detect cube
stickers, then refining the geometry proposer's face quads by snapping
their corners to detected sticker positions.

**Result**: small aggregate gain (+0.06pp), real per-pair wins (Set 44
+7.4pp, Set 17 +3.7pp, Set 31 +1.9pp), modest losses (Set 12 −1.9pp,
Set 61 −1.9pp). 27/33 pairs unchanged.

**Decision**: not shipped to the production proposer. The aggregate gain
doesn't justify the 150 MB SAM2 dependency, the scipy monkey-patch
needed for the missing C++ extension, the 25–30 min sweep cost on
gated CPU, and the new failure modes (model load, weight download).
Kept as a research tool for future revisiting.

## Aggregate results on the 33-pair corpus (knn5_lab_full classifier)

| Variant | rect | assembled | exact-54 |
|---|---|---|---|
| Baseline (hull-guard only) | 0.6532 | 0.6532 | 0 |
| PR #160 (clip-to-hull) | 0.7970 | 0.7970 | 2 |
| PR #163 (hull-shared + cache) | 0.8381 | 0.8374 | 6 |
| PR #166 (post-derivation guard) | 0.8438 | 0.8438 | 6 |
| PR #168 (min-edge calibration) | 0.8455 | 0.8455 | 6 |
| **+ AMG refine (no gate)** | 0.8443 | 0.8443 | 7 |
| **+ AMG refine (per-slot gated)** | **0.8460** | **0.8461** | 6 |
| Existing recognizer (reference) | — | 0.827 | — |
| Human-quad rectification (ceiling) | — | 0.9929 | — |

The per-slot gated path is the **non-regressive** one: it triggers AMG
ONLY on slots where `topologyFallbackRejectedByHullGuard=True` (i.e.,
the topology fallback was attempted but produced a degenerate quad
and was reverted to the original analyze_image grid). Slots whose
existing geometry is "good" (no fallback OR fallback's derived quad
passed the post-derivation hull guard) are left alone.

## Per-pair deltas (per-slot gated AMG vs PR #168 baseline)

| Set | Before | After | Delta |
|---|---|---|---|
| **44** | 0.6481 | **0.7222** | **+7.4pp** ✓ |
| **17** | 0.6852 | **0.7222** | **+3.7pp** ✓ |
| 31 | 0.6111 | 0.6296 | +1.9pp ✓ |
| 12 | 0.7963 | 0.7778 | −1.9pp ✗ (−1 sticker) |
| 61 (OOD) | 0.7963 | 0.7778 | −1.9pp ✗ (−1 sticker) |
| 27 other pairs | — | — | 0 (preserved) |

The 5 perfect-baseline pairs (15, 28, 29, 41, 42) all stayed at 1.000 —
gating correctly skipped AMG on those.

## What works

1. **SAM2 AMG detects 27–32 sticker-sized masks per cube photo.** On the
   Set 47 A canonical case, AMG returns 30 clean masks: 27 actual cube
   stickers + 2-3 background regions (desk, monitor).

2. **Constrained k-means (max 9 per cluster) partitions cleanly.** Init
   from `rembg`-hull bounding box (top/right-lower/left-lower thirds)
   yields a balanced 9/9/9 partition without any hexagon dependency.

3. **The `topologyFallbackRejectedByHullGuard` signal is the right
   gate.** When the gate fires, the existing pipeline has explicitly
   admitted defeat on this slot — that's where AMG has signal to add.
   When the gate doesn't fire, the existing pipeline is confident and
   AMG can only introduce sample-position noise.

4. **The bilinear extrapolation formula recovers face corners cleanly
   from the 4 outer sticker centroids:**

   ```
   face_corner_i = 1.5 × outer_sticker_i − 0.5 × face_center
   face_center   = mean of the 4 outer sticker centroids
   ```

   Exact for parallelogram faces. Off by ~9 px for perspective-quad
   faces, but within the noise floor (each sticker is ~70 px wide).

## What doesn't work

1. **`samv2` 0.0.4 pip wheel ships without its C++ extension.** The
   missing `sam2._C.get_connected_componnets` is needed for AMG mask
   post-processing. We work around it with a scipy.ndimage.label shim
   in `_install_scipy_connected_components_shim()`. ~5–10× slower than
   the C extension but functionally equivalent.

2. **AMG on MPS triggers a `float64 not supported` error.** Forced to
   CPU (~45 s per 1150-max-dim image). With per-slot gating, only ~half
   of the corpus pairs invoke AMG, bringing sweep cost to ~25–30 min.

3. **Aggregate gain is small.** +0.06pp over a 33-pair corpus = ~3
   stickers net. Concentrated wins (Set 44, 17) cancel small losses
   (Set 12, 61) at the corpus level.

4. **Naïve AMG-as-replacement regresses everywhere.** The PCA-bucketing
   and convex-hull-simplification variants we tried both produced
   wrong 3×3 sticker orderings on perspective-projected faces, leading
   to face quads that were the wrong shape (kite, mirror, etc.). The
   bilinear-snap-from-existing-quad approach (this module) is what
   actually works.

## Decision logic (per-slot gate)

The gate that produced the net-positive result:

```python
slots_to_refine = [
    slot for slot, meta in orig_debug["selectedPerFace"].items()
    if meta.get("topologyFallbackRejectedByHullGuard")
]
if slots_to_refine:
    refined, _ = amg_refine_quads_per_slot(
        image, orig_quads, slots_to_refine, snap_tolerance_px=25.0,
    )
```

Set-by-set: ~12 of 33 pairs trigger the gate; ~21 don't and skip AMG
entirely.

## Reproducing

```bash
# Install dependencies
.venv/bin/pip install samv2

# Download SAM2 tiny weights (~150 MB)
mkdir -p /tmp/sam2_checkpoints
curl -sSL -o /tmp/sam2_checkpoints/sam2_hiera_tiny.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

# Run on the canonical wins
.venv/bin/python tools/run_amg_face_refinement.py --sets 44 17 47 12

# Full corpus sweep (~25-30 min)
.venv/bin/python tools/run_amg_face_refinement.py --all \
    --report /tmp/amg_corpus_report.json \
    --summary /tmp/amg_corpus_summary.txt
```

## When to revisit

1. **SAM2 ships a prebuilt wheel with the C++ extension.** Removes the
   scipy monkey-patch and the perf hit from it. AMG cost drops 5–10×.

2. **A user reports a real-world Set-44-style failure** that the
   current production recognizer can't recover from. AMG-refinement is
   then a proven recovery path (we know it works on Set 44).

3. **A different per-sticker segmenter becomes available** (e.g., SAM3,
   a smaller open-weights alternative, or a cube-specific fine-tune).
   The pipeline math (`predict_sticker_positions_from_quad`,
   `extrapolate_face_corners_from_4_outer`, `constrained_kmeans`,
   `amg_refine_quads_per_slot`) is portable to any segmenter that gives
   us 27+ sticker-mask candidates per image.

## Files

- `tools/amg_face_refiner.py` — refiner module with pure-math helpers,
  SAM2 setup, scipy monkey-patch, per-slot refinement function
- `tools/run_amg_face_refinement.py` — CLI for running on specific pairs
  or the full corpus
- `tools/AMG_FACE_REFINER.md` — this document
- `tests/test_amg_face_refiner.py` — unit tests for the pure-math
  helpers (no SAM2 required)
