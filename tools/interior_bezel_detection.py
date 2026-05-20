"""Interior bezel-line detection — find the cube-center vertex (where 3
visible cube faces meet in image space) by detecting the dark
face-boundary lines INSIDE the rembg silhouette.

Motivation (from PR #176's hex-fitter failure taxonomy):
  On yawed cubes (67% of the worst-pair corpus), only 3 of the 6
  iso-projection hexagon vertices project onto the convex hull. The
  other 3 (h1 / h3 / h5 — the cube corners adjacent to the cube-center
  vertex) are INTERIOR to the silhouette. No hull-based fitter (angular
  sectors, Visvalingam-Whyatt, RANSAC) can find them — the information
  isn't on the hull.

What IS in the image: the dark bezel lines that mark the boundaries
BETWEEN the 3 visible cube faces. These run from the cube-center
vertex out toward h1, h3, h5 (or extensions thereof). Detect them
via gradient + a constrained angular sweep through the silhouette
centroid.

Algorithm (iterative refinement, post-human-review iteration on the
18 worst pairs):

  1. Compute the silhouette centroid as an initial cube-center seed.
  2. Compute Sobel gradient magnitude on grayscale, masked to an
     eroded silhouette (erode by `erosion_radius` px to avoid the
     outer hull boundary).
  3. ITERATIVE LOOP (up to `max_iter` rounds, typically 3-5):
       a. For each angle θ in [0, π) at 1° steps, sum the gradient
          magnitude along a line through (cx, cy) at angle θ.
       b. Pick the top-3 angles with non-max suppression.
       c. Locally search for the (cx, cy) that maximizes the SUM
          of line-mass for the 3 chosen angles.
       d. If (cx, cy) moved < `convergence_px` from the previous
          iteration, stop.
  4. Build boundary line segments from the converged cube-center
     outward along each of the 3 angles, terminating at the
     silhouette edge.
  5. Score each line individually: per-line quality = how much
     stronger that line's mass is relative to the local baseline
     (90th-percentile of mass over the full sweep). Exposes which
     of the 3 lines are robust vs which are sticker-grid contaminants.
  6. Return InteriorBezelDetection with cube-center, 3 boundary
     lines, per-line qualities, and an aggregate signal_quality.

The first-pass single-iteration version (committed in PR #177) scored
the 18 worst pairs at 5/18 human-overall_pass. The iterative refinement
targets the 13/18 cases where the centroid seed was correct on the
cube but the angle-picker had landed on sticker-grid contaminants —
re-picking angles after refining the center should let those cases
escape the local optimum.

This module is DIAGNOSTICS-ONLY (per the project decision-log entries).
The wiring into `_derive_face_quad_topology_aware` (using the detected
center + h1/h3/h5 as authoritative when available) is a future step,
gated on zero-FP mining across the corpus first.

Dependencies:
  * numpy (in `requirements.txt`) — required at module import
  * scipy — **optional research dependency**, not in `requirements.txt`.
    Used for binary erosion + Sobel-filter convolution. When absent,
    `detect_interior_bezel_lines()` returns an InteriorBezelDetection
    with `signal_quality=0.0` and `debug["error"]` explaining the
    missing dependency (matches the pattern used by
    `propose_geometry_labels.py:_fit_hexagon_optimized` and
    `amg_face_refiner.py`). Install with `.venv/bin/pip install scipy`.
  * No `rembg` dependency at module top — the caller supplies the
    silhouette mask.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


Point = Tuple[float, float]
Line = Tuple[Point, Point]  # (cube-center end, hull-edge end)


@dataclass
class InteriorBezelDetection:
    """Result of running interior bezel-line detection on one image."""

    cube_center: Optional[Point] = None
    # 0-3 boundary line segments, each from the cube-center outward
    # to where the line exits the silhouette
    boundary_lines: List[Line] = field(default_factory=list)
    # 0-3 angles in radians, one per boundary line
    boundary_angles: List[float] = field(default_factory=list)
    # 0-3 line equations as (a, b, c) tuples such that ax + by + c = 0
    # (same order as boundary_angles). Pre-computed for downstream
    # geometric joins — point-to-line distance is
    # `abs(a*x + b*y + c) / hypot(a, b)`.
    line_equations: List[Tuple[float, float, float]] = field(default_factory=list)
    # 0-3 per-line quality scores in [0, 1], same order as boundary_angles
    line_qualities: List[float] = field(default_factory=list)
    # Heuristic overall confidence 0.0-1.0
    signal_quality: float = 0.0
    debug: dict = field(default_factory=dict)


# ----------------- low-level helpers (unchanged from v1) -----------------


def _try_import_scipy_ndimage():
    """Return `scipy.ndimage` if installed, else None. Cached on first
    call. Lets `detect_interior_bezel_lines` return a graceful
    debug["error"] instead of raising ImportError when scipy is missing.

    scipy is an optional research dependency in this repo — it's not in
    `requirements.txt`. The same pattern is used by
    `propose_geometry_labels.py:_fit_hexagon_optimized` and
    `amg_face_refiner.py`.
    """
    cached = getattr(_try_import_scipy_ndimage, "_cached", "MISSING")
    if cached != "MISSING":
        return cached
    try:
        from scipy import ndimage  # type: ignore
    except ImportError:
        ndimage = None
    _try_import_scipy_ndimage._cached = ndimage  # type: ignore[attr-defined]
    return ndimage


def _sobel_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude on a grayscale image.

    Caller is responsible for checking that `_try_import_scipy_ndimage()`
    is not None before calling this. Raises ImportError otherwise.
    """
    ndimage = _try_import_scipy_ndimage()
    if ndimage is None:
        raise ImportError("scipy.ndimage is required")
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = ndimage.convolve(gray.astype(np.float32), kx, mode="reflect")
    gy = ndimage.convolve(gray.astype(np.float32), ky, mode="reflect")
    return np.hypot(gx, gy)


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Luma (ITU-R BT.601)."""
    return (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    ).astype(np.uint8)


def _erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary erosion via `scipy.ndimage`.

    Caller is responsible for checking that `_try_import_scipy_ndimage()`
    is not None before calling this. Raises ImportError otherwise.
    """
    ndimage = _try_import_scipy_ndimage()
    if ndimage is None:
        raise ImportError("scipy.ndimage is required")
    return ndimage.binary_erosion(mask, iterations=radius)


# ----------------- core: Hough-style angular sweep -----------------


def _line_mass(
    cx: float,
    cy: float,
    theta: float,
    silhouette_mask: np.ndarray,
    grad: np.ndarray,
    max_radius: int,
    min_samples: int = 50,
) -> float:
    """Sum gradient magnitude along a line through (cx, cy) at angle θ,
    sampling only pixels inside the silhouette mask. Returns 0.0 if the
    line doesn't have at least `min_samples` valid pixels.

    Vectorized via numpy fancy-indexing — Python-loop version (~5000
    iterations per call × 180 angles × 5 seeds × multiple iterations
    per pair) was the dominant cost; this is ~50x faster.
    """
    h, w = grad.shape
    dx = math.cos(theta)
    dy = math.sin(theta)
    r_arr = np.arange(-max_radius, max_radius + 1)
    xs = np.round(cx + r_arr * dx).astype(np.int32)
    ys = np.round(cy + r_arr * dy).astype(np.int32)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs = xs[inside]
    ys = ys[inside]
    in_mask = silhouette_mask[ys, xs]
    xs = xs[in_mask]
    ys = ys[in_mask]
    if len(xs) < min_samples:
        return 0.0
    return float(grad[ys, xs].sum())


def _top_k_with_nms(arr: np.ndarray, k: int, min_separation: int) -> List[int]:
    """Top-k indices of `arr` (a 1-D periodic array of length n) with
    non-max suppression: no two picked indices may be closer than
    `min_separation` (circular distance)."""
    n = len(arr)
    order = np.argsort(arr)[::-1]
    picked: List[int] = []
    for idx in order:
        if all(min(abs(int(idx) - p), n - abs(int(idx) - p)) >= min_separation
               for p in picked):
            picked.append(int(idx))
            if len(picked) == k:
                break
    return picked


def _angular_sweep(
    cx: float,
    cy: float,
    silhouette_mask: np.ndarray,
    grad: np.ndarray,
    max_radius: int,
    n_theta: int = 180,
) -> np.ndarray:
    """1-D angular Hough: line-mass as a function of θ in [0, π).

    Returns a length-`n_theta` array of total gradient magnitude along
    each candidate line direction. Lines through (cx, cy) at angle θ
    and θ+π are the same line, so we only sweep [0, π).
    """
    masses = np.zeros(n_theta, dtype=np.float32)
    for i in range(n_theta):
        theta = math.pi * i / n_theta
        masses[i] = _line_mass(
            cx, cy, theta, silhouette_mask, grad, max_radius
        )
    return masses


def _refine_cube_center(
    cx0: float,
    cy0: float,
    angles: List[float],
    silhouette_mask: np.ndarray,
    grad: np.ndarray,
    max_radius: int,
    *,
    coarse_window: int = 200,
    coarse_step: int = 20,
    fine_window: int = 30,
    fine_step: int = 4,
) -> Tuple[float, float, float]:
    """Coarse-to-fine search for (cx, cy) that maximizes the SUM of
    line-masses for the given fixed `angles` (radians). Returns
    (cx_refined, cy_refined, total_mass).

    HARD CONSTRAINT: (cx, cy) must be inside the silhouette mask.
    Without this, multi-seed restart can walk the center outside the
    silhouette where line-masses are spuriously high (most line passes
    through empty image regions that the mask filter doesn't reject
    completely because of the min_samples threshold).
    """
    h, w = silhouette_mask.shape

    def inside(cx: float, cy: float) -> bool:
        ix = int(round(cx))
        iy = int(round(cy))
        return 0 <= ix < w and 0 <= iy < h and bool(silhouette_mask[iy, ix])

    def score(cx: float, cy: float) -> float:
        if not inside(cx, cy):
            return -1.0
        return sum(
            _line_mass(cx, cy, t, silhouette_mask, grad, max_radius)
            for t in angles
        )

    # If seed is outside silhouette, fall back to (0, 0) score = -1
    # so subsequent search still tries to find a valid point.
    best_cx, best_cy = cx0, cy0
    best_score = score(cx0, cy0)
    for dx in range(-coarse_window, coarse_window + 1, coarse_step):
        for dy in range(-coarse_window, coarse_window + 1, coarse_step):
            cx = cx0 + dx
            cy = cy0 + dy
            s = score(cx, cy)
            if s > best_score:
                best_score = s
                best_cx, best_cy = cx, cy

    # Fine
    coarse_cx, coarse_cy = best_cx, best_cy
    for dx in range(-fine_window, fine_window + 1, fine_step):
        for dy in range(-fine_window, fine_window + 1, fine_step):
            cx = coarse_cx + dx
            cy = coarse_cy + dy
            s = score(cx, cy)
            if s > best_score:
                best_score = s
                best_cx, best_cy = cx, cy

    return best_cx, best_cy, best_score


def _trace_boundary_segment(
    cx: float,
    cy: float,
    theta: float,
    silhouette_mask: np.ndarray,
    *,
    step: int = 4,
) -> Optional[Line]:
    """From (cx, cy), step outward along angle θ until exiting the
    silhouette in BOTH directions. Return the longer-traversal endpoint
    as the boundary-line "outer" end. Returns None if neither
    direction can reach 30 px."""
    h, w = silhouette_mask.shape
    dx_unit = math.cos(theta)
    dy_unit = math.sin(theta)
    best_end = None
    best_length = 0
    for sign in (1.0, -1.0):
        last_inside = None
        for r in range(step, max(h, w), step):
            x = int(round(cx + sign * r * dx_unit))
            y = int(round(cy + sign * r * dy_unit))
            if not (0 <= x < w and 0 <= y < h):
                break
            if not silhouette_mask[y, x]:
                break
            last_inside = (cx + sign * r * dx_unit, cy + sign * r * dy_unit)
            if r > best_length:
                best_length = r
                best_end = last_inside
    if best_end is None or best_length < 30:
        return None
    return ((cx, cy), best_end)


# ----------------- main entry point -----------------


def detect_interior_bezel_lines(
    image_rgb: np.ndarray,
    silhouette_mask: np.ndarray,
    *,
    erosion_radius: int = 30,
    n_theta: int = 90,
    nms_separation_deg: int = 15,
    n_lines: int = 3,
    max_iter: int = 4,
    convergence_px: float = 6.0,
    max_drift_px: float = 180.0,
) -> InteriorBezelDetection:
    """Detect the cube-center vertex + up to `n_lines` face-boundary lines
    via iterative (angle-pick, center-refine) until convergence.

    Args:
      image_rgb: HxWx3 uint8 array of the input photo
      silhouette_mask: HxW bool array of the rembg cube silhouette
      erosion_radius: shrink the mask inward by N pixels before
        searching for edges; prevents picking the OUTER hull boundary
        as a "bezel line"
      n_theta: number of angular bins in [0, π); default 180 = 1° steps
      nms_separation_deg: min angular separation between picked lines
      n_lines: number of boundary lines to pick (default 3 — one per
        face boundary of a 3-face iso projection)
      max_iter: maximum (angle-pick, center-refine) iterations
      convergence_px: stop iterating when center moves < N px

    Returns:
      InteriorBezelDetection with cube_center, up to 3 boundary lines,
      per-line qualities, and an aggregate signal_quality score.
    """
    debug: dict = {}
    # Hard-gate scipy availability before any other work — if scipy is
    # missing we want a clear diagnostic, not a bare ModuleNotFoundError
    # inside the first call to a helper. scipy is an optional research
    # dependency (not in requirements.txt); see module docstring.
    if _try_import_scipy_ndimage() is None:
        return InteriorBezelDetection(
            debug={"error": (
                "scipy is not installed; this diagnostic probe requires "
                "scipy.ndimage for Sobel + binary_erosion. Install with: "
                ".venv/bin/pip install scipy"
            )},
        )
    if image_rgb.shape[:2] != silhouette_mask.shape:
        return InteriorBezelDetection(
            debug={"error": "image and mask shape mismatch"},
        )

    h, w = silhouette_mask.shape
    if int(silhouette_mask.sum()) < 1000:
        return InteriorBezelDetection(
            debug={"error": "silhouette mask too small"},
        )

    # Silhouette centroid as initial cube_center seed
    ys, xs = np.where(silhouette_mask)
    cx0 = float(xs.mean())
    cy0 = float(ys.mean())
    debug["centroid_seed"] = [round(cx0, 1), round(cy0, 1)]

    # Erode mask + compute gradient magnitude
    eroded = _erode_mask(silhouette_mask, erosion_radius)
    debug["eroded_pixel_count"] = int(eroded.sum())
    if debug["eroded_pixel_count"] < 1000:
        return InteriorBezelDetection(
            debug={**debug, "error": "eroded mask too small"},
        )

    gray = _rgb_to_gray(image_rgb)
    grad = _sobel_gradient_magnitude(gray).astype(np.float32)
    grad_inside = np.where(eroded, grad, 0.0).astype(np.float32)
    max_r = int(math.hypot(w, h) / 2)

    # ---- Iterative (angle-pick, center-refine) from centroid seed ----
    # Multi-seed restart (5- and 9-seed variants with offsets ±150-200 px)
    # was tried and EMPIRICALLY MADE THINGS WORSE on Sets 30 B and 44 B:
    # the optimization landscape has many local maxima from sticker-grid
    # edges, and "highest total line-mass" doesn't reliably distinguish
    # the true cube-center from a sticker-grid-aligned center. The
    # discarded multi-seed code is preserved in git history at commits
    # on this branch prior to the "drop multi-seed" commit. Single-seed
    # iteration + the new per-line quality score gives more honest signal.
    cx, cy = cx0, cy0
    angles: List[float] = []
    masses: np.ndarray = np.zeros(n_theta, dtype=np.float32)
    iteration_history = []
    converged = False
    for iteration in range(max_iter):
        masses = _angular_sweep(cx, cy, eroded, grad_inside, max_r, n_theta)
        if float(masses.max()) <= 0.0:
            return InteriorBezelDetection(
                debug={**debug, "error": "no line-mass signal at center"},
            )
        top_idx = _top_k_with_nms(masses, n_lines, nms_separation_deg)
        angles = [math.pi * i / n_theta for i in top_idx]
        if len(angles) < 2:
            return InteriorBezelDetection(
                debug={**debug, "error": "fewer than 2 lines detected"},
            )
        cx_new, cy_new, _ = _refine_cube_center(
            cx, cy, angles, eroded, grad_inside, max_r
        )
        shift = math.hypot(cx_new - cx, cy_new - cy)
        iteration_history.append({
            "iter": iteration,
            "center_before": [round(cx, 1), round(cy, 1)],
            "angles_picked_deg": [round(math.degrees(a), 1) for a in angles],
            "center_after": [round(cx_new, 1), round(cy_new, 1)],
            "shift_px": round(shift, 1),
        })
        cx, cy = cx_new, cy_new
        # Drift cap: if iteration has carried us too far from the
        # centroid seed, the iteration is probably chasing a
        # sticker-grid local optimum. Roll back to centroid + the
        # angles picked from centroid in iteration 0.
        cumulative_shift = math.hypot(cx - cx0, cy - cy0)
        if cumulative_shift > max_drift_px:
            iteration_history.append({
                "iter": iteration + 1,
                "action": "drift_cap_rollback",
                "cumulative_shift_px": round(cumulative_shift, 1),
                "max_drift_px": max_drift_px,
            })
            cx, cy = cx0, cy0
            # Re-sweep + re-pick at centroid (gives us angles for downstream)
            masses = _angular_sweep(cx, cy, eroded, grad_inside, max_r, n_theta)
            top_idx = _top_k_with_nms(masses, n_lines, nms_separation_deg)
            angles = [math.pi * i / n_theta for i in top_idx]
            converged = False
            break
        if shift < convergence_px:
            converged = True
            break

    debug["iterations"] = iteration_history
    debug["iter_count"] = len(iteration_history)
    debug["converged"] = converged
    debug["centroid_to_final_shift_px"] = round(
        math.hypot(cx - cx0, cy - cy0), 1
    )
    debug["mass_max"] = round(float(masses.max()), 1)
    debug["mass_median"] = round(float(np.median(masses)), 1)
    debug["mass_p90"] = round(float(np.percentile(masses, 90)), 1)

    # Build boundary segments at the final center
    boundary_lines: List[Line] = []
    for theta in angles:
        seg = _trace_boundary_segment(cx, cy, theta, silhouette_mask)
        if seg is not None:
            boundary_lines.append(seg)
    debug["boundary_line_count"] = len(boundary_lines)

    # ---- Per-line quality ----
    # For each picked angle θ, score how much its line-mass exceeds the
    # local baseline (90th percentile of mass over the full sweep).
    # Lines that are 1.5x+ above the p90 = a real bezel; lines barely
    # above are likely sticker-grid contaminants.
    p90 = max(1.0, float(np.percentile(masses, 90)))
    line_qualities: List[float] = []
    # angle θ = math.pi * i / n_theta → idx i = round(θ / (math.pi / n_theta))
    angle_step_rad = math.pi / n_theta
    for theta in angles:
        idx = int(round(theta / angle_step_rad)) % n_theta
        mass_here = float(masses[idx])
        # ratio: 1.0 = at p90, 2.0 = double p90; clamp to [0,1] via
        # min(1.0, (ratio - 1.0) / 1.0); below p90 → 0
        line_q = min(1.0, max(0.0, (mass_here / p90) - 1.0))
        line_qualities.append(line_q)
    debug["per_line_mass"] = [
        round(float(masses[int(round(t / angle_step_rad)) % n_theta]), 1)
        for t in angles
    ]
    debug["per_line_quality"] = [round(q, 3) for q in line_qualities]

    # ---- Aggregate signal_quality (recalibrated post-human-review) ----
    # The previous heuristic over-rewarded angle_regularity. New scheme:
    # the aggregate is the WEIGHTED MIN of the per-line qualities — a
    # single bad line caps the aggregate, which matches the human
    # review's "1 bad line means overall fail" pattern.
    if not line_qualities:
        signal_quality = 0.0
    else:
        # Mean of the bottom-2 quality values (worst-case-biased).
        sorted_q = sorted(line_qualities)
        if len(sorted_q) == 1:
            signal_quality = sorted_q[0]
        elif len(sorted_q) == 2:
            signal_quality = sum(sorted_q) / 2
        else:
            signal_quality = (sorted_q[0] + sorted_q[1]) / 2
    # Multiply by completeness so partial detections don't claim full score
    completeness = len(boundary_lines) / max(1, n_lines)
    signal_quality *= completeness

    debug["quality_components"] = {
        "per_line_quality": [round(q, 3) for q in line_qualities],
        "completeness": round(completeness, 3),
        "aggregator": "mean_of_bottom_2 * completeness",
    }

    # Line equations: for a line through (cx, cy) at angle θ, the unit
    # NORMAL is (-sin θ, cos θ) and the line is `-sin(θ)*x + cos(θ)*y +
    # (sin(θ)*cx - cos(θ)*cy) = 0`.
    line_equations: List[Tuple[float, float, float]] = []
    for theta in angles:
        a = -math.sin(theta)
        b = math.cos(theta)
        c = -(a * cx + b * cy)
        line_equations.append((float(a), float(b), float(c)))

    return InteriorBezelDetection(
        cube_center=(float(cx), float(cy)),
        boundary_lines=boundary_lines,
        boundary_angles=[float(a) for a in angles],
        line_equations=line_equations,
        line_qualities=line_qualities,
        signal_quality=float(signal_quality),
        debug=debug,
    )


# ----------------- helpers for downstream slot/cell-level joins -----------------


def point_to_line_distance(
    point: Point, line_eq: Tuple[float, float, float]
) -> float:
    """Distance from a point to a line in ax + by + c = 0 form."""
    a, b, c = line_eq
    denom = math.hypot(a, b)
    if denom < 1e-9:
        return float("inf")
    return abs(a * point[0] + b * point[1] + c) / denom


def line_crosses_quad(
    quad_pts: Sequence[Point], line_eq: Tuple[float, float, float]
) -> bool:
    """True iff the infinite line `ax + by + c = 0` passes through the
    interior of the convex polygon `quad_pts` (4 vertices in CW or CCW
    order). Detected by checking that quad vertices straddle the line —
    i.e., not all on one side."""
    a, b, c = line_eq
    signs = []
    for x, y in quad_pts:
        v = a * x + b * y + c
        if v > 1e-6:
            signs.append(1)
        elif v < -1e-6:
            signs.append(-1)
        else:
            signs.append(0)
    return any(s == 1 for s in signs) and any(s == -1 for s in signs)


# Default thresholds for the `crosses_high_quality_bezel` derived flag.
# Picked from this branch's 18-pair human-review walkthroughs (#177 fixture):
#   * line_q >= 0.40 selects the magenta-vertical bezel that's correct
#     on 13/18 pairs (and excludes the yellow-dimmest line that's
#     essentially noise on 10/18 pairs)
#   * cell-centroid distance <= 30 px is a conservative "actually
#     crosses the cell's center, not just clipping a corner"
# These are NOT validated as zero-FP thresholds — they exist to give
# the broader-corpus mining a concrete starting point. The raw
# per-line `quality`, `distance_from_centroid_px`, and `crosses_cell`
# fields are preserved alongside so downstream consumers can re-tune
# without re-running the detector.
DEFAULT_LINE_QUALITY_THRESHOLD = 0.40
DEFAULT_DISTANCE_THRESHOLD_PX = 30.0


def cell_line_diagnostics(
    detection: InteriorBezelDetection,
    cell_quad: Sequence[Point],
    *,
    min_line_quality: float = 0.0,
    high_quality_threshold: float = DEFAULT_LINE_QUALITY_THRESHOLD,
    max_distance_px: float = DEFAULT_DISTANCE_THRESHOLD_PX,
    detector_version: str = "iterative-v1",
) -> dict:
    """Compute per-line diagnostics for a single cell quad against the
    detected bezel lines. Intended for downstream slot/cell-level joins
    against #175's overlay_feedback per-slot rows.

    Args:
      detection: an InteriorBezelDetection (image-level result)
      cell_quad: 4-vertex sequence of (x, y) image-space points
        describing the cell's quadrilateral in image coords
      min_line_quality: when computing the aggregate `any_crossing_high_quality`
        flag, ignore lines whose per-line quality is below this
        threshold (default 0.0 = consider all detected lines)
      high_quality_threshold: line-quality threshold for the derived
        `crosses_high_quality_bezel` flag (default 0.40, see
        DEFAULT_LINE_QUALITY_THRESHOLD).
      max_distance_px: maximum distance from cell centroid to a line
        for that line to count as "crossing through the cell" (used by
        the derived `crosses_high_quality_bezel` flag — independent of
        the geometric `crosses_cell` boolean which only checks straddle).
      detector_version: tag stamped on the output so future detector
        iterations can be distinguished in cross-tab mining without
        re-running upstream.

    Returns dict with keys:
      detector_version: str (passed through; for cross-tab join keys)
      cell_centroid: (cx, cy)
      cell_center_to_cube_center_px: float (proxy for "is this cell
        near the cube-center vertex?")
      per_line: list of {angle_deg, quality, distance_from_centroid_px,
        crosses_cell} — RAW per-line fields preserved
      any_crossing: bool (any detected line straddles the cell?)
      any_crossing_high_quality: bool (any line with quality >=
        `min_line_quality` straddles the cell?)
      crosses_high_quality_bezel: bool (derived flag — any line with
        quality >= `high_quality_threshold` AND distance from cell
        centroid <= `max_distance_px` crosses through the cell)
      thresholds: {"line_quality": ..., "distance_px": ...} (echoed
        back so the join row records WHICH threshold produced the flag)
      min_distance_from_centroid_px: float (closest line to cell centroid)
    """
    # Centroid of the quad
    cx = sum(p[0] for p in cell_quad) / len(cell_quad)
    cy = sum(p[1] for p in cell_quad) / len(cell_quad)
    cell_centroid: Point = (cx, cy)

    if detection.cube_center is not None:
        cc_dist = math.hypot(
            cx - detection.cube_center[0], cy - detection.cube_center[1]
        )
    else:
        cc_dist = float("inf")

    per_line = []
    for i, (eq, q, theta) in enumerate(zip(
        detection.line_equations,
        detection.line_qualities,
        detection.boundary_angles,
    )):
        per_line.append({
            "line_index": i,
            "angle_deg": round(math.degrees(theta), 2),
            "quality": round(float(q), 3),
            "distance_from_centroid_px": round(
                point_to_line_distance(cell_centroid, eq), 1
            ),
            "crosses_cell": bool(line_crosses_quad(cell_quad, eq)),
        })

    any_crossing = any(pl["crosses_cell"] for pl in per_line)
    any_crossing_high_quality = any(
        pl["crosses_cell"] and pl["quality"] >= min_line_quality
        for pl in per_line
    )
    # Derived flag: line quality >= threshold AND distance to cell
    # centroid <= threshold AND geometric crosses_cell. Strictest of
    # the three signals — meant as a starting point for broader-corpus
    # zero-FP mining, NOT a validated production threshold.
    crosses_hq = any(
        pl["crosses_cell"]
        and pl["quality"] >= high_quality_threshold
        and pl["distance_from_centroid_px"] <= max_distance_px
        for pl in per_line
    )
    min_dist = min(
        (pl["distance_from_centroid_px"] for pl in per_line),
        default=float("inf"),
    )

    return {
        "detector_version": detector_version,
        "cell_centroid": [round(cx, 1), round(cy, 1)],
        "cell_center_to_cube_center_px": round(cc_dist, 1),
        "per_line": per_line,
        "any_crossing": any_crossing,
        "any_crossing_high_quality": any_crossing_high_quality,
        "crosses_high_quality_bezel": crosses_hq,
        "thresholds": {
            "line_quality": high_quality_threshold,
            "distance_px": max_distance_px,
        },
        "min_distance_from_centroid_px": min_dist,
    }
