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

Algorithm (post-RANSAC iteration on Set 47 A — first-pass sequential
RANSAC was dominated by sticker-grid edges):

  1. Compute the silhouette centroid as an initial cube-center seed.
  2. Compute Sobel gradient magnitude on grayscale, masked to an
     eroded silhouette (erode by `erosion_radius` px to avoid the
     outer hull boundary).
  3. For each angle θ in [0, π) at 1° steps, sum the gradient
     magnitude along a line through (cx, cy) at angle θ. This is a
     1-D angular Hough transform anchored at the centroid.
  4. Pick the top-3 angles with non-max suppression (min 15°
     separation). These are 3 candidate face-boundary line angles.
  5. Locally optimize cube-center: search a small window around the
     centroid for the (cx, cy) that maximizes the SUM of line-mass
     for the 3 chosen angles. (Joint refinement; coarse-to-fine.)
  6. Build boundary line segments from the refined cube-center
     outward along each of the 3 angles, terminating at the
     silhouette edge.
  7. Return InteriorBezelDetection with the cube-center estimate,
     the 3 boundary lines, and a signal_quality score derived from
     line-mass-vs-baseline ratio + angular separation regularity.

This module is DIAGNOSTICS-ONLY (per the project decision-log entries).
The wiring into `_derive_face_quad_topology_aware` (using the detected
center + h1/h3/h5 as authoritative when available) is a future step,
gated on zero-FP mining across the corpus first.

Dependencies: numpy, scipy (already in requirements). No `rembg`
dependency at module top — the caller supplies the silhouette mask.
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
    # Heuristic confidence 0.0-1.0
    signal_quality: float = 0.0
    debug: dict = field(default_factory=dict)


# ----------------- low-level helpers (unchanged from v1) -----------------


def _sobel_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude on a grayscale image (numpy-only)."""
    from scipy import ndimage
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
    """Binary erosion via scipy."""
    from scipy import ndimage
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
    line doesn't have at least `min_samples` valid pixels."""
    h, w = grad.shape
    dx = math.cos(theta)
    dy = math.sin(theta)
    total = 0.0
    n_samples = 0
    # 1-pixel-spaced sampling along the line
    for r in range(-max_radius, max_radius + 1):
        x = int(round(cx + r * dx))
        y = int(round(cy + r * dy))
        if 0 <= x < w and 0 <= y < h and silhouette_mask[y, x]:
            total += grad[y, x]
            n_samples += 1
    if n_samples < min_samples:
        return 0.0
    return total


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
    (cx_refined, cy_refined, total_mass)."""

    def score(cx: float, cy: float) -> float:
        return sum(
            _line_mass(cx, cy, t, silhouette_mask, grad, max_radius)
            for t in angles
        )

    # Coarse
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
    n_theta: int = 180,
    nms_separation_deg: int = 15,
    n_lines: int = 3,
) -> InteriorBezelDetection:
    """Detect the cube-center vertex + up to `n_lines` face-boundary lines.

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

    Returns:
      InteriorBezelDetection with cube_center, up to 3 boundary lines,
      angular signature, and signal_quality score.
    """
    debug: dict = {}
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

    # 1-D angular sweep through the centroid
    max_r = int(math.hypot(w, h) / 2)
    masses = _angular_sweep(cx0, cy0, eroded, grad_inside, max_r, n_theta)
    if float(masses.max()) <= 0.0:
        return InteriorBezelDetection(
            debug={**debug, "error": "no line-mass signal at centroid"},
        )

    debug["mass_max"] = round(float(masses.max()), 1)
    debug["mass_median"] = round(float(np.median(masses)), 1)
    debug["mass_max_to_median"] = round(
        float(masses.max()) / max(1.0, float(np.median(masses))), 3
    )

    # Top-3 angles with NMS
    top_idx = _top_k_with_nms(masses, n_lines, nms_separation_deg)
    angles = [math.pi * i / n_theta for i in top_idx]
    debug["initial_angles_deg"] = [round(math.degrees(a), 1) for a in angles]
    debug["initial_masses"] = [round(float(masses[i]), 1) for i in top_idx]

    if len(angles) < 2:
        return InteriorBezelDetection(
            debug={**debug, "error": "fewer than 2 lines detected"},
        )

    # Refine cube-center: search a window around the centroid for the
    # (cx, cy) that maximizes the sum of line-masses for these angles
    cx_ref, cy_ref, total_mass = _refine_cube_center(
        cx0, cy0, angles, eroded, grad_inside, max_r
    )
    debug["cube_center_refined"] = [round(cx_ref, 1), round(cy_ref, 1)]
    debug["refined_shift_px"] = round(
        math.hypot(cx_ref - cx0, cy_ref - cy0), 1
    )

    # Build boundary segments
    boundary_lines: List[Line] = []
    for theta in angles:
        seg = _trace_boundary_segment(cx_ref, cy_ref, theta, silhouette_mask)
        if seg is not None:
            boundary_lines.append(seg)
    debug["boundary_line_count"] = len(boundary_lines)

    # ---- Signal quality heuristic ----
    # 3 components, geometric mean → 0.0-1.0:
    # (a) line_mass_ratio: max angular peak / median ; expect ~1.5+ for
    #     a real cube, ~1.0 for noise
    # (b) angle_regularity: how close to 60° apart the 3 angles are
    #     (in iso projection cube bezels are ~60° apart)
    # (c) boundary_line_completeness: fraction of lines that traced out
    ratio = float(masses.max()) / max(1.0, float(np.median(masses)))
    ratio_score = min(1.0, max(0.0, (ratio - 1.0) / 1.0))  # 1.0→0, 2.0→1.0

    if len(angles) == 3:
        deg_angles = sorted([math.degrees(a) % 180.0 for a in angles])
        # pairwise circular distances
        gaps = [
            min(abs(deg_angles[(i + 1) % 3] - deg_angles[i]),
                180 - abs(deg_angles[(i + 1) % 3] - deg_angles[i]))
            for i in range(3)
        ]
        # ideal: 60° + 60° + 60°. Compute std around 60°.
        gap_dev = sum(abs(g - 60.0) for g in gaps) / 3.0
        angle_score = max(0.0, 1.0 - gap_dev / 30.0)  # 0° dev→1, 30°→0
    else:
        angle_score = 0.3

    completeness = len(boundary_lines) / max(1, n_lines)

    signal_quality = (ratio_score * angle_score * completeness) ** (1.0 / 3.0)
    debug["quality_components"] = {
        "line_mass_ratio_score": round(ratio_score, 3),
        "angle_regularity_score": round(angle_score, 3),
        "completeness": round(completeness, 3),
    }

    return InteriorBezelDetection(
        cube_center=(float(cx_ref), float(cy_ref)),
        boundary_lines=boundary_lines,
        boundary_angles=[float(a) for a in angles],
        signal_quality=float(signal_quality),
        debug=debug,
    )
