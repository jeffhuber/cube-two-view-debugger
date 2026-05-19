"""SAM2 Automatic Mask Generator (AMG) face-quad refiner — research tool.

This module is a STANDALONE research tool. It is NOT wired into the
production recognizer or `evaluate_hybrid_pipeline.py`. It explores
whether per-sticker segmentation from SAM2 can refine the geometry
proposer's face quads on hard pairs where the topology fallback
fails (`topologyFallbackRejectedByHullGuard=True`).

Empirical result on the 33-pair corpus (knn5_lab_full classifier):

    Variant                              rect    assembled  exact-54
    PR #168 baseline                     0.8455  0.8455     6
    + AMG refine (no gate)               0.8443  0.8443     7
    + AMG refine (per-slot gated by      0.8460  0.8461     6
      topologyFallbackRejectedByHullGuard)

Aggregate gain is ~+0.06pp (within noise). Per-pair signal IS real:

    Set 44:  0.6481 → 0.7222  (+7.4pp, ~+4 stickers)
    Set 17:  0.6852 → 0.7222  (+3.7pp, ~+2 stickers)
    Set 31:  0.6111 → 0.6296  (+1.9pp, ~+1 sticker)
    Set 12:  0.7963 → 0.7778  (-1.9pp, -1 sticker)
    Set 61:  0.7963 → 0.7778  (-1.9pp, -1 sticker)

The Set 44 / Set 17 / Set 31 wins are concentrated on pairs where the
existing topology fallback produced degenerate quads (collapsed
corners → rectified output samples desk/background instead of
stickers). AMG re-detects the actual sticker positions and snaps
existing-quad-corners to those positions, recovering geometry.

The wins don't propagate to the aggregate because:
- ~half the corpus has clean baseline geometry (5/5 perfect-pairs stay
  preserved under gating)
- The 2 minor regressions (-1 sticker each) cancel the smaller wins
- The Set 44 +4 stickers and Set 17 +2 stickers are real net positives

NOT shipped to production because:
- SAM2 dependency (~150MB weights, must be downloaded separately)
- scipy monkey-patch required (pip wheel `samv2` 0.0.4 is missing the
  C++ extension that backs `sam2.utils.misc.get_connected_components`)
- Aggregate gain is too small to justify the dependency surface

Future use cases:
1. Re-evaluate when SAM2 ships pre-compiled wheels with the C++ extension
2. Use as a recovery path for "low-confidence" production photos when
   user reports a real-world Set-44-style failure
3. Reference implementation for "per-sticker → face-quad" geometry math
   (the bilinear extrapolation formula is reusable)


Setup
-----

Install SAM2 + weights:

    .venv/bin/pip install samv2
    mkdir -p /tmp/sam2_checkpoints
    curl -sSL -o /tmp/sam2_checkpoints/sam2_hiera_tiny.pt \\
        https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

(The user must approve these external downloads in their
`.claude/settings.json` — see the curl rule in this repo.)


Usage
-----

    from tools.amg_face_refiner import amg_refine_quads_per_slot
    from tools.evaluate_hybrid_pipeline import _proposer_face_quads
    image, _ = _load_processing_image(image_path)
    quads, debug = _proposer_face_quads(image_path, "A", hull_guard=True,
                                         fit_error_fallback=True,
                                         fit_error_threshold=4.0,
                                         processing_image=image)
    # Identify slots where topology fallback was rejected
    slots_to_refine = [
        slot for slot, meta in debug["selectedPerFace"].items()
        if meta.get("topologyFallbackRejectedByHullGuard")
    ]
    if slots_to_refine:
        refined, refine_debug = amg_refine_quads_per_slot(
            image, quads, slots_to_refine, snap_tolerance_px=25.0,
        )
        # Use `refined` quads instead of `quads`
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Lazy imports of heavy deps (torch, sam2, scipy) — only required when
# you actually call into AMG. Keeps `import tools.amg_face_refiner` cheap
# for unit tests that only exercise the pure-math helpers.


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SAM2_CHECKPOINT = Path("/tmp/sam2_checkpoints/sam2_hiera_tiny.pt")
DEFAULT_SAM2_CONFIG = "sam2_hiera_t.yaml"
DEFAULT_SAM2_DEVICE = "cpu"  # MPS triggers a float64 error inside AMG; CPU works


# ---------------------------------------------------------------------------
# Pure-math helpers (no SAM2 / torch dependency)
# ---------------------------------------------------------------------------


def predict_sticker_positions_from_quad(
    quad: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """For a face quad in any 4-point order, return 9 sticker centroids
    at face fractions (1/6, 3/6, 5/6) via bilinear interpolation.

    The result is in row-major order based on the canonical (CW-from-N)
    re-ordering of the input quad — matching the `rectify_face` and
    `extract_stickers_from_rectified` sampling convention.
    """
    # Local import to avoid circular dep when this module is imported
    # by the proposer
    from tools.sample_stickers_from_hull import canonical_corner_order
    canon = canonical_corner_order([tuple(p) for p in quad])
    A = np.array(canon[0], dtype=float)
    B = np.array(canon[1], dtype=float)
    C = np.array(canon[2], dtype=float)
    D = np.array(canon[3], dtype=float)
    stickers: List[Tuple[float, float]] = []
    for v_idx in range(3):
        v = (2 * v_idx + 1) / 6.0
        for u_idx in range(3):
            u = (2 * u_idx + 1) / 6.0
            p = ((1 - u) * (1 - v) * A + u * (1 - v) * B
                 + u * v * C + (1 - u) * v * D)
            stickers.append((float(p[0]), float(p[1])))
    return stickers


def extrapolate_face_corners_from_4_outer(
    outer_stickers: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Given the 4 outer sticker centroids of a 3×3 face grid
    (at face fractions (1/6, 1/6), (5/6, 1/6), (5/6, 5/6), (1/6, 5/6)),
    extrapolate the face quad's 4 corners.

    Math: face_center = mean of 4 outer-sticker centroids (= face center
    by 3×3 symmetry for parallelogram faces). Each face corner is at
    distance 1.5× the sticker's distance-from-center, in the same
    direction.

      face_corner_i = 1.5 × outer_sticker_i − 0.5 × face_center

    Returns 4 corners in the SAME ORDER as the input outer_stickers.
    The caller's downstream `canonical_corner_order` resorts CW from
    image-north for consistent rectification.

    Caveat: this formula is exact for parallelogram faces and slightly
    biased (~9 px on typical 1150-max images) for non-parallelogram
    quads with perspective. Within the noise floor of sticker sampling
    (each sticker is ~70 px wide).
    """
    if len(outer_stickers) != 4:
        raise ValueError(f"expected 4 outer stickers, got {len(outer_stickers)}")
    outer = np.array(outer_stickers, dtype=float)
    face_center = outer.mean(axis=0)
    return [
        tuple((1.5 * outer[i] - 0.5 * face_center).tolist())
        for i in range(4)
    ]


def constrained_kmeans(
    points: np.ndarray,
    init_centers: np.ndarray,
    max_per_cluster: int = 9,
    n_iter: int = 15,
) -> Tuple[List[int], np.ndarray]:
    """K-means with a hard cap on cluster size.

    Used to partition the 27-30 AMG-detected sticker centroids into 3
    clusters of ≤9 (one per cube face). Capacity constraint enforces
    that a face has exactly 9 stickers — the constraint isn't elegant
    but it eliminates the failure mode where 1 cluster captures
    12+ stickers because k-means is greedy.

    Assignment: sort all (point, cluster) pairs by distance, greedily
    assign each point to its nearest non-full cluster. Re-compute
    centers, iterate until convergence.

    Returns (assignments_per_point, final_centers).
    """
    centers = init_centers.copy()
    n = len(points)
    k = len(centers)
    assigned = [-1] * n
    for _ in range(n_iter):
        dists = np.linalg.norm(
            points[:, None, :] - centers[None, :, :], axis=2,
        )
        pairs = sorted(
            ((float(dists[i, j]), i, j) for i in range(n) for j in range(k))
        )
        assigned = [-1] * n
        counts = [0] * k
        for _d, i, j in pairs:
            if assigned[i] != -1:
                continue
            if counts[j] >= max_per_cluster:
                continue
            assigned[i] = j
            counts[j] += 1
        new_centers = np.array([
            points[[i for i in range(n) if assigned[i] == j]].mean(axis=0)
            if counts[j] > 0 else centers[j]
            for j in range(k)
        ])
        if np.allclose(new_centers, centers, atol=1.0):
            break
        centers = new_centers
    return assigned, centers


# ---------------------------------------------------------------------------
# SAM2 setup (lazy)
# ---------------------------------------------------------------------------


_AMG_CACHE: Dict = {}


def _install_scipy_connected_components_shim() -> None:
    """Replace sam2.utils.misc.get_connected_components with a pure-Python
    implementation backed by scipy.ndimage.label.

    The pip-installed samv2 wheel (0.0.4) ships without its C++ extension
    (`sam2._C`), so calling `get_connected_components` raises ImportError.
    The shim is functionally equivalent (8-connected components per
    batched slice) using scipy's label routine. ~5-10× slower than the
    C extension but acceptable for the per-image AMG cost we're paying.
    """
    import sam2.utils.misc as _sam2_misc  # type: ignore
    if getattr(_sam2_misc, "_scipy_shim_installed", False):
        return
    import scipy.ndimage as _ndi  # type: ignore
    import torch  # type: ignore

    def _gcc(mask):  # type: ignore[no-redef]
        arr = mask.to(torch.uint8).cpu().numpy()
        n_batch, _, h, w = arr.shape
        labels_out = np.zeros((n_batch, 1, h, w), dtype=np.int32)
        counts_out = np.zeros((n_batch, 1, h, w), dtype=np.int32)
        structure = np.ones((3, 3), dtype=np.uint8)
        for n in range(n_batch):
            labeled, num_features = _ndi.label(arr[n, 0], structure=structure)
            labels_out[n, 0] = labeled
            if num_features > 0:
                areas = np.bincount(labeled.ravel())
                counts_out[n, 0] = areas[labeled]
        return (
            torch.from_numpy(labels_out).to(mask.device),
            torch.from_numpy(counts_out).to(mask.device),
        )

    _sam2_misc.get_connected_components = _gcc
    _sam2_misc._scipy_shim_installed = True  # type: ignore[attr-defined]


def get_amg_predictor(
    checkpoint: Path = DEFAULT_SAM2_CHECKPOINT,
    config: str = DEFAULT_SAM2_CONFIG,
    device: str = DEFAULT_SAM2_DEVICE,
):
    """Load (and cache) the SAM2 AutomaticMaskGenerator.

    Raises FileNotFoundError if the checkpoint is missing — must be
    downloaded separately (see module docstring for the curl command).
    """
    cache_key = (str(checkpoint), config, device)
    if cache_key in _AMG_CACHE:
        return _AMG_CACHE[cache_key]
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint missing at {checkpoint}. "
            f"Download with:\n"
            f"  mkdir -p {Path(checkpoint).parent}\n"
            f"  curl -sSL -o {checkpoint} "
            f"https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
            f"sam2_hiera_tiny.pt"
        )
    _install_scipy_connected_components_shim()
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
    from sam2.build_sam import build_sam2  # type: ignore

    sam2 = build_sam2(config, str(checkpoint), device=device)
    amg = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=32,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.90,
        min_mask_region_area=200,
    )
    _AMG_CACHE[cache_key] = amg
    return amg


def amg_sticker_centroids(
    image,
    cube_hull: Optional[List[Tuple[float, float]]] = None,
    min_area: int = 1500,
    max_area: int = 12000,
    hull_inside_min_frac: float = 0.85,
) -> List[Tuple[float, float]]:
    """Run SAM2 AMG on `image` and return the centroids of all
    sticker-sized masks that fall inside the rembg cube hull.

    Sticker-sized = `min_area <= mask.area <= max_area`. Default range
    (1500-12000 px²) covers typical Rubik's stickers in the
    1150-max-dimension processing image.

    Returns list of (x, y) centroids.
    """
    from tools.inspect_cube_isolation import point_in_polygon

    amg = get_amg_predictor()
    img_arr = np.array(image)
    masks = amg.generate(img_arr)

    centroids: List[Tuple[float, float]] = []
    for m in masks:
        if not (min_area <= m["area"] <= max_area):
            continue
        seg = m["segmentation"]
        ys, xs = np.nonzero(seg)
        if cube_hull and len(cube_hull) >= 3:
            n_inside = sum(
                1 for x, y in zip(xs, ys)
                if point_in_polygon((float(x), float(y)), cube_hull)
            )
            if n_inside / max(len(xs), 1) < hull_inside_min_frac:
                continue
        centroids.append((float(xs.mean()), float(ys.mean())))
    return centroids


# ---------------------------------------------------------------------------
# Per-slot face-quad refinement
# ---------------------------------------------------------------------------


def amg_refine_quads_per_slot(
    image,
    existing_quads: Dict[str, List[Tuple[float, float]]],
    slots_to_refine: Iterable[str],
    *,
    snap_tolerance_px: float = 25.0,
    cube_hull: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict]:
    """Refine ONLY the requested slots using AMG sticker centroids.

    For each slot:
      1. Predict 9 sticker positions from the existing face quad
      2. Find the nearest AMG centroid for each predicted position
      3. If d < snap_tolerance_px, use the AMG centroid (refined);
         else keep the predicted position
      4. Extract the 4 outer-corner refined positions (indices 0, 2,
         8, 6 in row-major order)
      5. Extrapolate face corners via the (1/6, 5/6) face-fraction
         identity

    Slots not in `slots_to_refine` are returned unchanged.

    Args:
      image: PIL.Image in processing-image coords (max 1150 dimension)
      existing_quads: {slot_label: [(x, y), ...]} from the geometry proposer
      slots_to_refine: which slots to refine (caller decides via gating)
      snap_tolerance_px: max distance for sticker-snap (default 25 px,
        ~30% of a sticker width; tighter avoids snapping to wrong-face
        neighbors)
      cube_hull: optional rembg cube hull polygon for inside-hull
        filtering (auto-computed via `tools.evaluate_hybrid_pipeline.
        _rembg_cube_hull` if None)

    Returns:
      (refined_quads, debug_info). `refined_quads` has all slots from
      `existing_quads`; slots in `slots_to_refine` are refined, others
      are passed through unchanged. `debug_info` records per-slot snap
      counts and AMG sticker totals.
    """
    slots_to_refine_list = list(slots_to_refine)
    if not slots_to_refine_list:
        return dict(existing_quads), {
            "method": "amg_refine_per_slot",
            "amgRan": False,
            "slotsRefined": [],
        }

    if cube_hull is None:
        # Lazy import — only needed if caller didn't supply hull
        from tools.evaluate_hybrid_pipeline import _rembg_cube_hull
        cube_hull = _rembg_cube_hull(image)

    amg_centroids = amg_sticker_centroids(image, cube_hull=cube_hull)
    if len(amg_centroids) < 9:
        return dict(existing_quads), {
            "method": "amg_refine_per_slot",
            "amgRan": True,
            "stickersDetected": len(amg_centroids),
            "error": "fewer than 9 AMG stickers found",
        }
    amg_array = np.array(amg_centroids)

    refined_quads = dict(existing_quads)
    debug_per_slot: Dict[str, Dict] = {}
    for slot in slots_to_refine_list:
        if slot not in existing_quads:
            continue
        existing_q = existing_quads[slot]
        predicted = predict_sticker_positions_from_quad(existing_q)
        refined: List[Tuple[float, float]] = []
        n_snapped = 0
        for pred in predicted:
            pred_arr = np.array(pred)
            dists = np.linalg.norm(amg_array - pred_arr, axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_d = float(dists[nearest_idx])
            if nearest_d < snap_tolerance_px:
                refined.append(tuple(amg_array[nearest_idx].tolist()))
                n_snapped += 1
            else:
                refined.append(pred)
        # Outer corners at row-major indices 0, 2, 8, 6
        outer = [refined[0], refined[2], refined[8], refined[6]]
        face_corners = extrapolate_face_corners_from_4_outer(outer)
        refined_quads[slot] = face_corners
        debug_per_slot[slot] = {
            "snapped": n_snapped,
            "totalStickers": len(refined),
        }

    return refined_quads, {
        "method": "amg_refine_per_slot",
        "amgRan": True,
        "slotsRefined": slots_to_refine_list,
        "amgStickers": len(amg_centroids),
        "perSlot": debug_per_slot,
    }
