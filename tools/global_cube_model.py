"""Global cube projection model — fit a 3D cube template to 7 anchor
points detected in the image.

USER'S FRAMING (the conceptual breakthrough):

  Instead of optimizing pixel-level objectives, start with a known 3D
  iso-cube template that has 7 anchor points (1 cube_center vertex +
  6 outer hexagon corners), then FIT the template to the 7 anchor
  points detected in the image.

  The template provides the SHAPE.
  The detections provide the POSITIONS.
  The fit just aligns template to detections.

ANCHOR POINT DETECTION:

  * cube_center (1 point): from PR #178's bezel detection — the
    front cube corner where 3 visible bezels converge
  * 6 outer hexagon vertices: rembg silhouette → convex hull →
    Visvalingam-Whyatt simplification to 6 points (the well-known
    algorithm from PR #176's hex-fitter walkthrough)

CORRESPONDENCE:

  Each of the 6 hexagon points is either a "single-axis" h-vertex
  (1 cube-edge from cube_center) or a "double-axis" outer vertex
  (2 cube-edges from cube_center). The single-axis ones are
  CLOSER to cube_center in image space than the double-axis ones.

  After sorting the 6 detected points by distance to cube_center:
    - 3 nearest → h_x, h_y, h_z candidates (single-axis)
    - 3 farthest → h_xy, h_xz, h_yz candidates (double-axis)

  Within each group, angular ordering CCW around cube_center
  determines the specific assignment. The cube's 3-fold rotational
  symmetry means there are 3 valid cyclic assignments; we try each
  and pick the one with lowest fit residual.

FIT (Procrustes-style orthographic alignment):

  Given 7 image-space points P_2d_i and 7 template 3D points P_3d_i
  (in cube-local coords), find rotation R, scale s, and 2D
  translation t such that:

    P_2d_i ≈ s * R[:2, :] @ P_3d_i  +  t

  This is solvable in closed form for rotation + translation +
  scale via SVD (orthographic Procrustes). 6 DOF, 14 equations
  → overdetermined → least squares.

If the 7 detected points are correct, the fit is trivially correct.
No gradient-along-edges optimization, no silhouette IoU, no special
cases for sticker grids. The whole problem reduces to point
correspondence + Procrustes alignment.

This module is DIAGNOSTICS-ONLY (per the project's discipline). No
wiring into recognizer behavior.

Dependencies:
  * numpy (required)
  * scipy.spatial.ConvexHull (optional research dep; pattern matches
    #177-#180 — when missing, returns init-only model)
"""
from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.interior_bezel_detection import (  # noqa: E402
    InteriorBezelDetection,
    _try_import_scipy_ndimage,
)


Point = Tuple[float, float]
Point3D = Tuple[float, float, float]
Quad = Tuple[Point, Point, Point, Point]


# ---- Cube template ----
# 3D positions of the 7 visible cube corners. The front corner
# (cube_center vertex) is at origin; the cube extends in the
# negative axis directions to (-1, -1, -1).
_TEMPLATE_3D = {
    "front":  (0.0,  0.0,  0.0),    # cube_center vertex
    "h_x":    (-1.0, 0.0,  0.0),    # single-axis h-vertex (cube edge along x)
    "h_y":    (0.0, -1.0,  0.0),    # single-axis h-vertex (cube edge along y)
    "h_z":    (0.0,  0.0, -1.0),    # single-axis h-vertex (cube edge along z)
    "h_xy":   (-1.0,-1.0,  0.0),    # double-axis outer hexagon vertex
    "h_xz":   (-1.0, 0.0, -1.0),    # double-axis outer hexagon vertex
    "h_yz":   (0.0, -1.0, -1.0),    # double-axis outer hexagon vertex
}

# The 3 visible faces (each a parallelogram of 4 corners in CCW order).
_FACE_DEFS = {
    "face_yz": ("front", "h_y", "h_yz", "h_z"),
    "face_xz": ("front", "h_z", "h_xz", "h_x"),
    "face_xy": ("front", "h_x", "h_xy", "h_y"),
}

# Ordering for the hexagon vertices around the silhouette (alternating
# between single-axis and double-axis corners as you go CCW around the
# cube_center in iso projection). For STANDARD iso with axes
# 90°/210°/330° (pointing DOWN/UP-LEFT/UP-RIGHT in image space), the
# hexagon goes (CCW from TOP): h_yz, h_z, h_xz, h_x, h_xy, h_y.
# But the actual ordering on a given image depends on which axis-
# assignment we choose. We handle this via the correspondence search.


@dataclass
class GlobalCubeModel:
    """Cube projection model fit to 7 anchor points in the image.

    Stored: the 3 cube axis projections in 2D (3 vectors × 2 coords =
    6 numbers), plus cube_center screen position (2 numbers). Total
    8 stored numbers, but they encode an underlying 6-DOF pose
    (orthographic projection + 2D translation + scale).
    """

    cube_center_screen: Point = (0.0, 0.0)
    # axis_x_2d, axis_y_2d, axis_z_2d are the image-space displacement
    # vectors from cube_center to each h-vertex (h_x, h_y, h_z).
    axis_x_2d: Point = (0.0, 0.0)
    axis_y_2d: Point = (0.0, 0.0)
    axis_z_2d: Point = (0.0, 0.0)

    visible_corners: dict = field(default_factory=dict)
    face_quads: dict = field(default_factory=dict)
    sticker_cells: dict = field(default_factory=dict)

    fit_loss: float = float("inf")
    fit_quality: float = 0.0
    debug: dict = field(default_factory=dict)


def derive_geometry(model: GlobalCubeModel) -> None:
    """Populate visible_corners, face_quads, sticker_cells from the
    cube_center + 3 axis-projection vectors."""
    cx, cy = model.cube_center_screen
    ax, ay, az = model.axis_x_2d, model.axis_y_2d, model.axis_z_2d
    corners = {
        "front": (cx, cy),
        "h_x":   (cx + ax[0],         cy + ax[1]),
        "h_y":   (cx + ay[0],         cy + ay[1]),
        "h_z":   (cx + az[0],         cy + az[1]),
        "h_xy":  (cx + ax[0] + ay[0], cy + ax[1] + ay[1]),
        "h_xz":  (cx + ax[0] + az[0], cy + ax[1] + az[1]),
        "h_yz":  (cx + ay[0] + az[0], cy + ay[1] + az[1]),
    }
    model.visible_corners = corners

    model.face_quads = {
        face_name: tuple(corners[c] for c in corner_names)
        for face_name, corner_names in _FACE_DEFS.items()
    }

    sticker_cells = {}
    for face_name, (A, B, C, D) in model.face_quads.items():
        cells = []
        for row in range(3):
            for col in range(3):
                def at(r: float, c: float) -> Point:
                    return (
                        A[0] + c * (B[0] - A[0]) + r * (D[0] - A[0]),
                        A[1] + c * (B[1] - A[1]) + r * (D[1] - A[1]),
                    )
                cell_quad = (
                    at(row / 3, col / 3),
                    at(row / 3, (col + 1) / 3),
                    at((row + 1) / 3, (col + 1) / 3),
                    at((row + 1) / 3, col / 3),
                )
                cells.append(cell_quad)
        sticker_cells[face_name] = cells
    model.sticker_cells = sticker_cells


# ----------------- anchor point detection -----------------


def _hull_from_mask(mask: np.ndarray) -> List[Point]:
    """Convex hull of nonzero pixels in mask. Returns CCW-ordered
    list of (x, y) vertices."""
    try:
        from scipy.spatial import ConvexHull  # type: ignore
    except ImportError:
        return []
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return []
    pts = np.column_stack([xs, ys]).astype(np.float64)
    if len(pts) > 50000:
        # Subsample for speed; hull is unchanged by subset of interior pts
        idx = np.random.choice(len(pts), size=50000, replace=False)
        pts = pts[idx]
    try:
        hull = ConvexHull(pts)
    except Exception:
        return []
    return [tuple(pts[i]) for i in hull.vertices]


def _visvalingam_simplify(
    points: Sequence[Point], target: int
) -> List[Point]:
    """Iteratively remove the polygon vertex whose removal changes
    polygon area least, until `target` remain. Same algorithm as
    PR #176's hex-fitter walkthrough."""
    pts = list(points)
    while len(pts) > target:
        n = len(pts)
        min_area, min_idx = float("inf"), 0
        for i in range(n):
            a = pts[(i - 1) % n]
            b = pts[i]
            c = pts[(i + 1) % n]
            area = abs((b[0] - a[0]) * (c[1] - a[1])
                       - (c[0] - a[0]) * (b[1] - a[1]))
            if area < min_area:
                min_area, min_idx = area, i
        pts.pop(min_idx)
    return pts


def detect_hexagon_anchors(silhouette_mask: np.ndarray) -> List[Point]:
    """Detect the 6 outer hexagon vertices from the rembg silhouette.

    Convex hull → Visvalingam-Whyatt simplification to 6 vertices.
    Returns the 6 vertices in CCW order, or empty list on failure.
    """
    hull = _hull_from_mask(silhouette_mask)
    if len(hull) < 6:
        return []
    return _visvalingam_simplify(hull, 6)


# ----------------- correspondence + Procrustes fit -----------------


def _euler_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """3D rotation from intrinsic Euler angles (R = Rz(roll) Rx(pitch) Ry(yaw))."""
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll),  math.sin(roll)
    Ry = np.array([[ cy, 0, sy], [ 0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[ 1, 0, 0], [ 0, cp,-sp], [ 0, sp, cp]])
    Rz = np.array([[ cr,-sr, 0], [ sr, cr, 0], [ 0, 0, 1]])
    return Rz @ Rx @ Ry


def _project_perspective(
    P_3d: np.ndarray, R: np.ndarray, t: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Project 3D points (n × 3) to 2D image space via perspective
    projection with intrinsics (fx, fy, cx, cy) and extrinsics (R, t).
    Returns n × 2 image points."""
    P_cam = (R @ P_3d.T).T + t  # (n, 3) — camera-space coords
    # Perspective divide
    z = P_cam[:, 2]
    # Avoid division by zero (camera behind cube)
    z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = fx * P_cam[:, 0] / z_safe + cx
    v = fy * P_cam[:, 1] / z_safe + cy
    return np.column_stack([u, v])


def _try_import_cv2():
    """Return cv2 if installed, else None. Cached on first call.
    Pattern matches the scipy-optional guards elsewhere in this repo."""
    cached = getattr(_try_import_cv2, "_cached", "MISSING")
    if cached != "MISSING":
        return cached
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None
    _try_import_cv2._cached = cv2  # type: ignore[attr-defined]
    return cv2


def _solve_pnp_calibrated(
    P_3d: np.ndarray,
    P_2d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    init_pose: Optional[Tuple[float, float, float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calibrated perspective PnP via OpenCV's solvePnP.

    Args:
      P_3d: n × 3 array of 3D points in object-local coordinates
      P_2d: n × 2 array of corresponding 2D image points
      Camera intrinsics (fx, fy, cx, cy)
      init_pose: ignored (cv2's algorithms self-initialize)

    Returns: (R 3x3, t 3-vector, rms_residual_px).

    Uses cv2.SOLVEPNP_ITERATIVE for the final refinement; for the
    initial pose we try SOLVEPNP_EPNP (closed-form, no initial guess
    needed) which is the standard approach for ≥6-point correspondences.
    Falls back to ITERATIVE with an iso-pose init if EPNP fails.
    """
    cv2 = _try_import_cv2()
    if cv2 is None:
        raise ImportError("cv2 not available; install opencv-python-headless")

    object_points = P_3d.astype(np.float32).reshape(-1, 1, 3)
    image_points = P_2d.astype(np.float32).reshape(-1, 1, 2)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

    # Try closed-form EPnP first (works for n >= 4)
    success = False
    rvec_best = None
    tvec_best = None
    best_rms = float("inf")
    for flag in (cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, K, dist_coeffs, flags=flag,
            )
        except cv2.error:
            continue
        if not ok:
            continue
        # Compute residual
        proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
        proj_2d = proj.reshape(-1, 2)
        diff = proj_2d - P_2d
        rms = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        if rms < best_rms:
            best_rms = rms
            rvec_best = rvec
            tvec_best = tvec
            success = True

    if not success:
        raise RuntimeError("cv2.solvePnP failed with all flags")

    # Refine via iterative algorithm starting from the best closed-form solution
    try:
        ok_refine, rvec_refined, tvec_refined = cv2.solvePnP(
            object_points, image_points, K, dist_coeffs,
            rvec_best.copy(), tvec_best.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok_refine:
            proj, _ = cv2.projectPoints(object_points, rvec_refined, tvec_refined, K, dist_coeffs)
            proj_2d = proj.reshape(-1, 2)
            diff = proj_2d - P_2d
            rms_refined = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
            if rms_refined < best_rms:
                rvec_best = rvec_refined
                tvec_best = tvec_refined
                best_rms = rms_refined
    except cv2.error:
        pass

    # Convert rvec (rodrigues) to rotation matrix
    R, _ = cv2.Rodrigues(rvec_best)
    t = tvec_best.reshape(3)
    return R, t, best_rms


def _fit_affine_2d(
    src: np.ndarray, dst: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 2D affine transform mapping `src` to `dst` via least
    squares. Returns (A, b) where dst_i ≈ A @ src_i + b.

    src, dst are (n, 2) arrays. n >= 3 required.
    Closed-form via normal equations.
    """
    n = src.shape[0]
    # Build design matrix for the linear system
    # [src_x, src_y, 1, 0, 0, 0] @ [a11, a12, b1, ..., ...] = dst_x
    # [0, 0, 0, src_x, src_y, 1] @ [..., a21, a22, b2] = dst_y
    M = np.zeros((2 * n, 6))
    M[0::2, 0] = src[:, 0]
    M[0::2, 1] = src[:, 1]
    M[0::2, 2] = 1.0
    M[1::2, 3] = src[:, 0]
    M[1::2, 4] = src[:, 1]
    M[1::2, 5] = 1.0
    b_vec = np.zeros(2 * n)
    b_vec[0::2] = dst[:, 0]
    b_vec[1::2] = dst[:, 1]
    # Solve M @ params = b_vec via least squares
    params, _, _, _ = np.linalg.lstsq(M, b_vec, rcond=None)
    A = np.array([
        [params[0], params[1]],
        [params[3], params[4]],
    ])
    b = np.array([params[2], params[5]])
    return A, b


def _affine_residual(
    src: np.ndarray, dst: np.ndarray, A: np.ndarray, b: np.ndarray
) -> float:
    """Mean squared distance after applying affine A, b to src."""
    proj = (A @ src.T).T + b
    diff = proj - dst
    return float(np.mean(np.sum(diff * diff, axis=1)))


# Standard iso-projection 2D positions of the 6 outer hexagon vertices,
# with cube_center at origin. These are the positions a unit cube
# projects to under standard iso (camera elevation 35.26°, azimuth 45°),
# with the cube extending in -X, -Y, -Z from cube_center. Image y-axis
# points DOWN.
_TEMPLATE_HEXAGON_2D_ISO = {
    "h_x":  (-math.sqrt(3) / 2, -0.5),   # LEFT-UP (image: -x, -y means UP)
    "h_y":  (0.0, 1.0),                   # DOWN
    "h_z":  (math.sqrt(3) / 2, -0.5),    # RIGHT-UP
    "h_xy": (-math.sqrt(3) / 2, 0.5),    # LEFT-DOWN
    "h_xz": (0.0, -1.0),                  # UP
    "h_yz": (math.sqrt(3) / 2, 0.5),     # RIGHT-DOWN
}


# 3D positions of the 7 visible cube corners in cube-local coords.
# cube_center at origin; cube extends to (-1, -1, -1).
_CUBE_CORNERS_3D = {
    "front":  (0.0,  0.0,  0.0),
    "h_x":    (-1.0, 0.0,  0.0),
    "h_y":    (0.0, -1.0,  0.0),
    "h_z":    (0.0,  0.0, -1.0),
    "h_xy":   (-1.0,-1.0,  0.0),
    "h_xz":   (-1.0, 0.0, -1.0),
    "h_yz":   (0.0, -1.0, -1.0),
}


def _order_ccw(points: Sequence[Point], center: Point) -> List[Point]:
    """Order points CCW around `center`. (CCW in image coords means
    increasing math-angle; image y is DOWN so we flip the angle.)"""
    cx, cy = center
    def angle(p):
        return math.atan2(-(p[1] - cy), p[0] - cx)
    return sorted(points, key=angle)


def fit_cube_template_to_anchors(
    cube_center: Point,
    hexagon_vertices_ccw: Sequence[Point],
    bezel_angles_rad: Sequence[float],
    image_size: Optional[Tuple[int, int]] = None,
) -> Optional[GlobalCubeModel]:
    """Fit the 3D cube template to 1 cube_center + 6 ordered hexagon
    vertices, using the 3 detected bezel angles to classify which
    hexagon vertices are inner h-vertices (h_x, h_y, h_z) vs outer
    hexagon corners (h_xy, h_xz, h_yz).

    INNER vs OUTER classification: an INNER h-vertex is the end of a
    SINGLE cube edge from cube_center — the line from cube_center to
    it is a cube edge, i.e., one of the 3 detected BEZELS. An OUTER
    hexagon corner is reached by TWO cube edges; the line from
    cube_center to it is NOT a cube edge (it cuts across a face).

    So: for each hexagon vertex, check whether the angle from
    cube_center to it is close to one of the 3 detected bezel angles.
    The 3 vertices closest to bezel directions are inner; the other 3
    are outer.

    This works regardless of yaw — distance-based classification fails
    on near-iso cubes where all 6 hexagon vertices are equidistant
    from cube_center.
    """
    if len(hexagon_vertices_ccw) != 6:
        return None
    if len(bezel_angles_rad) < 3:
        return None

    cx, cy = cube_center
    # CLASSIFICATION: 3 INNER h-vertices vs 3 OUTER hexagon corners.
    #
    # In iso projection, cube_center is the FRONT cube corner (the
    # one closest to camera). The cube body extends AWAY from the
    # camera (in -X, -Y, -Z cube-local directions). The 3 inner
    # h-vertices are 1 cube-edge away from cube_center; the 3 outer
    # hexagon corners are 2 cube-edges away (via two axis traversals).
    #
    # In image space (under orthographic projection), the cube body
    # extends OUTWARD from cube_center. The 3 inner h-vertices land
    # on the FAR side of the cube body from cube_center; the 3 outer
    # corners land closer to cube_center (since they're reached via
    # sums of two axes that partially cancel due to foreshortening).
    #
    # So: 3 hexagon vertices with the GREATEST distance from
    # cube_center are the INNER h-vertices. 3 with the LEAST distance
    # are the OUTER corners.
    by_dist = sorted(
        hexagon_vertices_ccw,
        key=lambda p: math.hypot(p[0] - cx, p[1] - cy),
    )
    outer_3_raw = by_dist[:3]   # nearest = outer hexagon corners
    inner_3_raw = by_dist[3:]   # farthest = inner h-vertices
    # Order each group CCW around cube_center
    inner_3 = _order_ccw(inner_3_raw, cube_center)
    outer_3 = _order_ccw(outer_3_raw, cube_center)

    # The cube has 3-fold rotational symmetry around its main diagonal
    # through cube_center. This means rotating (h_x, h_y, h_z) → (h_y, h_z, h_x)
    # gives a geometrically valid cube too. Try all 3 cyclic
    # assignments and pick the one with lowest fit residual.

    # Template positions in CCW order (ascending math angle), alternating
    # inner / outer:
    #   h_z (30°)  → h_xz (90°)  → h_x (150°)
    #   → h_xy (210°) → h_y (270°) → h_yz (330°)
    template_ccw_order = ["h_z", "h_xz", "h_x", "h_xy", "h_y", "h_yz"]
    template_ccw_2d = np.array(
        [_TEMPLATE_HEXAGON_2D_ISO[k] for k in template_ccw_order],
        dtype=np.float64,
    )
    # Also try chirality flipped — some camera poses produce hexagons
    # whose CCW image-space ordering corresponds to the cube's CW
    # ordering in cube-local coords.
    template_cw_order = list(reversed(template_ccw_order))
    template_cw_2d = np.array(
        [_TEMPLATE_HEXAGON_2D_ISO[k] for k in template_cw_order],
        dtype=np.float64,
    )

    # Interleave detected inner/outer into a CCW hexagon ordering.
    # inner_3 is CCW [a, b, c] and outer_3 is CCW [x, y, z]. They
    # alternate around the hexagon: the question is WHICH outer
    # falls between WHICH inners. We don't know a priori, so try 3
    # interleaving offsets.
    best_model = None
    best_residual = float("inf")
    best_correspondence = None
    best_interleave = None

    # Brute-force: try all 720 (= 6!) permutations of detected → template.
    # For each permutation, fit affine src=template_position[i] → dst=detected_position[perm(i)].
    # The CORRECT permutation gives the lowest residual.
    from itertools import permutations
    template_keys = ["h_x", "h_y", "h_z", "h_xy", "h_xz", "h_yz"]
    template_positions = np.array(
        [_TEMPLATE_HEXAGON_2D_ISO[k] for k in template_keys],
        dtype=np.float64,
    )
    # All 6 detected vertices in some fixed order
    all_detected = np.array(hexagon_vertices_ccw, dtype=np.float64)

    for perm in permutations(range(6)):
        detected_permuted = all_detected[list(perm)]
        try:
            A, b = _fit_affine_2d(template_positions, detected_permuted)
        except Exception:
            continue
        residual = _affine_residual(template_positions, detected_permuted, A, b)
        if residual < best_residual:
            best_residual = residual
            best_correspondence = (perm, template_keys, [tuple(v) for v in detected_permuted])
            # cube_center is template (0, 0) under the affine = just b
            cube_center_fitted = (float(b[0]), float(b[1]))
            # h_x, h_y, h_z = template positions through affine
            hx_img = A @ np.array(_TEMPLATE_HEXAGON_2D_ISO["h_x"]) + b
            hy_img = A @ np.array(_TEMPLATE_HEXAGON_2D_ISO["h_y"]) + b
            hz_img = A @ np.array(_TEMPLATE_HEXAGON_2D_ISO["h_z"]) + b
            model = GlobalCubeModel(
                cube_center_screen=cube_center_fitted,
                axis_x_2d=(float(hx_img[0] - b[0]), float(hx_img[1] - b[1])),
                axis_y_2d=(float(hy_img[0] - b[0]), float(hy_img[1] - b[1])),
                axis_z_2d=(float(hz_img[0] - b[0]), float(hz_img[1] - b[1])),
            )
            derive_geometry(model)
            best_model = model

    if best_model is None:
        return None

    # ---- Perspective-PnP refinement ----
    # The 2D-affine fit above gives us a rough alignment AND the
    # correspondence between detected hexagon points and template
    # corners. We now refine the cube's 3D pose via calibrated PnP
    # (perspective projection), which can EXACTLY represent any
    # 3D-rotated cube — no affine-of-iso approximation residual.
    #
    # Camera intrinsics: assume typical iPhone-like values. The
    # principal point is at the image center. fx ≈ 3000 px is a
    # reasonable approximation for a typical iPhone capture
    # (focal length ~26mm equivalent on a 35mm-equivalent sensor at
    # ~3000-4000 px image width).
    affine_residual_rms = math.sqrt(best_residual)
    best_model_affine = best_model

    # Extract permutation + assign template names to 2D points
    perm, template_keys, detected_permuted = best_correspondence
    # Build 3D template (in cube-local coords) + 2D image points.
    # Hexagon-only PnP — the trihedral vertex is refined separately via
    # mean-of-3 ensemble in fit_global_cube_model. Earlier attempt to add
    # bezel as 7th PnP correspondence didn't beat hexagon-only on the
    # ground-truth corpus.
    P_3d = np.array([_CUBE_CORNERS_3D[k] for k in template_keys], dtype=np.float64)
    P_2d = np.array(list(detected_permuted), dtype=np.float64)

    # Camera intrinsics
    if image_size is not None:
        img_h, img_w = image_size
    else:
        img_h, img_w = 4000, 3000
    # iPhone main-camera focal length ≈ 26mm equivalent on a 1/1.7" sensor,
    # which gives fx ≈ 3300 px on a 4032-wide capture. That's ~0.82 ×
    # image_max_dim — meaningfully smaller than image_max_dim (which
    # we previously defaulted to and which biased PnP toward systematic
    # vertex offset of 50-100 px under perspective+yaw).
    fx = fy = 0.82 * float(max(img_h, img_w))
    cx_img = float(img_w) / 2
    cy_img = float(img_h) / 2

    # Initial pose from the affine fit:
    # - rotation: standard iso (yaw=45°, pitch=-35.26°, roll=0°)
    # - translation: place cube_center vertex at the affine-derived
    #   2D screen position, at a depth that produces the observed
    #   image scale.
    iso_pitch = -math.asin(1.0 / math.sqrt(3.0))
    # The cube's projected size in image gives an estimate for depth:
    # at iso, a unit cube extends ~sqrt(2/3)*image_per_world units per
    # cube edge. So image_cube_size_px ≈ fx / tz_init → tz = fx / cube_size_px.
    affine_cube_size_px = float(np.linalg.norm(
        np.array(best_model_affine.visible_corners["h_x"])
        - np.array(best_model_affine.visible_corners["front"])
    ))
    if affine_cube_size_px < 10:
        affine_cube_size_px = 500.0
    tz_init = fx / max(affine_cube_size_px, 50.0)
    # Initial translation: cube_center projects to 2D affine-derived
    # cube_center, which in camera coords means at (tx, ty, tz)
    # where (tx/tz)*fx + cx_img = affine_cc_x, so
    #   tx = (affine_cc_x - cx_img) * tz_init / fx
    affine_cc = best_model_affine.cube_center_screen
    tx_init = (affine_cc[0] - cx_img) * tz_init / fx
    ty_init = (affine_cc[1] - cy_img) * tz_init / fy

    # Multi-start over (yaw, pitch) grid. Yaw has 3-fold ambiguity
    # (cube's main-diagonal symmetry) + 2 chiralities = 6 valid
    # orientations. Pitch can deviate ±15° from the exact iso angle
    # depending on how the user holds the camera. Cover both axes to
    # land in the right basin reliably.
    pnp_R = None
    pnp_t = None
    pnp_rms = float("inf")
    # cv2.solvePnP uses closed-form algorithms (EPNP, SQPNP) that
    # don't need an initial guess. One call, no multi-start needed.
    try:
        pnp_R, pnp_t, pnp_rms = _solve_pnp_calibrated(
            P_3d, P_2d, fx, fy, cx_img, cy_img
        )
    except Exception:
        pass

    if pnp_R is not None and pnp_rms < affine_residual_rms:
        # PnP improved the fit — use it as the final model
        # Project all 7 template corners through the PnP pose to get
        # the final image positions
        P_3d_full = np.array(
            [_CUBE_CORNERS_3D[k] for k in
             ["front", "h_x", "h_y", "h_z", "h_xy", "h_xz", "h_yz"]],
            dtype=np.float64,
        )
        proj_full = _project_perspective(
            P_3d_full, pnp_R, pnp_t, fx, fy, cx_img, cy_img
        )
        cube_center_fit = (float(proj_full[0, 0]), float(proj_full[0, 1]))
        best_model = GlobalCubeModel(
            cube_center_screen=cube_center_fit,
            axis_x_2d=(
                float(proj_full[1, 0] - proj_full[0, 0]),
                float(proj_full[1, 1] - proj_full[0, 1]),
            ),
            axis_y_2d=(
                float(proj_full[2, 0] - proj_full[0, 0]),
                float(proj_full[2, 1] - proj_full[0, 1]),
            ),
            axis_z_2d=(
                float(proj_full[3, 0] - proj_full[0, 0]),
                float(proj_full[3, 1] - proj_full[0, 1]),
            ),
        )
        derive_geometry(best_model)
        final_rms = pnp_rms
        used_pnp = True
    else:
        # PnP didn't help — fall back to affine model
        final_rms = affine_residual_rms
        used_pnp = False

    derived_cc = best_model.cube_center_screen
    bezel_cc_offset = math.hypot(
        derived_cc[0] - cube_center[0], derived_cc[1] - cube_center[1]
    )
    best_model.debug = {
        "approach": "perspective_pnp" if used_pnp else "affine_fallback",
        "affine_rms_px": round(affine_residual_rms, 2),
        "pnp_rms_px": round(pnp_rms, 2) if pnp_R is not None else "n/a",
        "fit_residual_rms_px": round(final_rms, 2),
        "fit_residual_px2": round(final_rms ** 2, 2),
        "best_permutation": list(best_correspondence[0]),
        "best_template_keys": list(best_correspondence[1]),
        "bezel_detected_cube_center": [round(cube_center[0], 1), round(cube_center[1], 1)],
        "fit_derived_cube_center": [round(derived_cc[0], 1), round(derived_cc[1], 1)],
        "bezel_vs_fit_cube_center_offset_px": round(bezel_cc_offset, 1),
        "camera_fx_px": round(fx, 1),
        "camera_principal_point_px": [round(cx_img, 1), round(cy_img, 1)],
    }
    best_model.fit_loss = final_rms ** 2
    rms = final_rms
    best_model.fit_quality = max(0.0, 1.0 - rms / 200.0)
    return best_model


def _trihedral_junction_score(
    image_rgb: np.ndarray,
    x: float, y: float,
    dirs: Sequence[Tuple[float, float]],
    line_len: int = 40,
    n_samples: int = 20,
) -> float:
    """Score a candidate trihedral vertex position via 'min-darkness across
    3 known directions'. High score = strong dark line in all 3 directions
    (a real 3-way junction). Low score = at most 1-2 dark lines (likely
    a sticker corner or other artifact, not the trihedral vertex).
    """
    h, w = image_rgb.shape[:2]
    if x < line_len or y < line_len or x >= w - line_len or y >= h - line_len:
        return 0.0
    gray = image_rgb.mean(axis=2).astype(np.float32)
    darkness = 255.0 - gray
    t_vals = np.linspace(5, line_len, n_samples)
    dir_scores = []
    for d in dirs:
        norm = math.hypot(d[0], d[1])
        if norm < 1e-6:
            continue
        nx, ny = d[0] / norm, d[1] / norm
        # Sample BOTH half-lines (cube edge could extend either direction
        # depending on which corner the vertex is); take the brighter side.
        both = []
        for sign in (+1.0, -1.0):
            xs = x + sign * nx * t_vals
            ys = y + sign * ny * t_vals
            xi = np.clip(xs.astype(np.int32), 0, w - 1)
            yi = np.clip(ys.astype(np.int32), 0, h - 1)
            both.append(float(darkness[yi, xi].mean()))
        dir_scores.append(max(both))
    if len(dir_scores) < 3:
        return 0.0
    return min(dir_scores)


def _refine_vertex_via_image_junction(
    image_rgb: np.ndarray,
    vertex: Point,
    axes_2d: Sequence[Tuple[float, float]],
    window: int = 40,
    line_len: int = 40,
    n_samples: int = 20,
    base_score_gate: float = 200.0,
) -> Tuple[Point, dict]:
    """Refine vertex by searching a window for the strongest 3-way dark-line
    junction along the known axis directions.

    GATING: only return refined position if the SCORE AT THE INPUT VERTEX
    is below `base_score_gate`. Strong-junction inputs are already at the
    true vertex; refinement on them tends to drift to nearby sticker corners
    with marginally higher local score. Tuned to 200 from ground-truth
    sweep (see /tmp/probe_score_gated.py): 6 of 27 cases get refined,
    big wins (42_B -56, 44_A -48, 26_A -36, 21_B -29) with only one
    small regression (31_B +8).
    """
    debug: dict = {}
    base_score = _trihedral_junction_score(
        image_rgb, vertex[0], vertex[1], axes_2d, line_len, n_samples,
    )
    debug["junction_score_at_ensemble"] = round(base_score, 1)
    debug["junction_score_gate"] = base_score_gate

    if base_score >= base_score_gate:
        debug["refinement"] = "skipped_high_base_score"
        return vertex, debug

    h, w = image_rgb.shape[:2]
    cx0, cy0 = int(round(vertex[0])), int(round(vertex[1]))
    best_score = base_score
    best_xy = (float(cx0), float(cy0))
    for dy in range(-window, window + 1, 2):
        for dx in range(-window, window + 1, 2):
            x = cx0 + dx
            y = cy0 + dy
            s = _trihedral_junction_score(
                image_rgb, x, y, axes_2d, line_len, n_samples,
            )
            if s > best_score:
                best_score = s
                best_xy = (float(x), float(y))
    debug["junction_score_at_refined"] = round(best_score, 1)
    debug["refinement_movement_px"] = round(
        math.hypot(best_xy[0] - vertex[0], best_xy[1] - vertex[1]), 1
    )
    debug["refinement"] = "applied"
    return best_xy, debug


def _line_darkness_from_vertex(
    image_rgb: np.ndarray,
    vertex: Point,
    target: Point,
    n_samples: int = 24,
    skip_endpoints_frac: float = 0.15,
) -> float:
    """Mean pixel darkness along the line from vertex toward target.

    A NEAR hexagon corner is connected to the vertex by a visible cube
    edge (a dark bezel line). A FAR hexagon corner is reached only via
    a face DIAGONAL through colored sticker interior. So:
      darkness along vertex→near_corner = HIGH (bezel pixels along the line)
      darkness along vertex→far_corner  = LOW (passes through sticker faces)

    We skip the endpoints (where the vertex itself and the hexagon corner
    might both be on dark bezels) to focus on the line interior, which
    is where bezel-vs-sticker most cleanly differentiates.
    """
    h, w = image_rgb.shape[:2]
    if vertex[0] < 0 or vertex[1] < 0 or vertex[0] >= w or vertex[1] >= h:
        return 0.0
    gray = image_rgb.mean(axis=2).astype(np.float32)
    darkness = 255.0 - gray
    t_vals = np.linspace(skip_endpoints_frac, 1.0 - skip_endpoints_frac, n_samples)
    xs = (vertex[0] + (target[0] - vertex[0]) * t_vals).astype(np.int32)
    ys = (vertex[1] + (target[1] - vertex[1]) * t_vals).astype(np.int32)
    in_image = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not in_image.any():
        return 0.0
    return float(darkness[ys[in_image], xs[in_image]].mean())


def _signed_angle_diff_deg(a_rad: float, b_rad: float) -> float:
    """Smallest difference between two ORIENTED angles, in degrees [0, 180]."""
    diff = math.degrees(a_rad - b_rad)
    while diff > 180.0:
        diff -= 360.0
    while diff <= -180.0:
        diff += 360.0
    return abs(diff)


def _apply_chirality_correction(
    model: GlobalCubeModel,
    detection: Optional["InteriorBezelDetection"] = None,
    image_rgb: Optional[np.ndarray] = None,
    apply_correction: bool = False,
) -> Tuple[GlobalCubeModel, Dict[str, Any]]:
    """Detect + (optionally) correct 60° chirality flip via per-corner
    line darkness.

    GEOMETRIC SETUP. In iso projection there are 6 hexagon-silhouette
    corners around the central trihedral vertex:
      - 3 NEAR corners (each 1 cube-edge from vertex). In a clean
        rendering the line vertex→near runs along a visible cube edge
        and samples DARK BEZEL pixels.
      - 3 FAR corners (each 2 cube-edges from vertex, reached via a face
        diagonal). The line vertex→far cuts across colored sticker
        interior — should sample LIGHTER pixels.

    A 60° rotation of the model around the cube's body diagonal yields
    the SAME silhouette but swaps the near/far labels. The Procrustes
    brute-force fit can pick either assignment based on tiny noise
    differences in the residual — so the chirality is effectively
    random in noisy real-photo inputs.

    DISCRIMINATOR. Mean pixel darkness along the line vertex→corner,
    measured for all 6 hexagon corners. If the model's labeled NEAR
    corners are systematically darker than its labeled FAR corners, the
    chirality is correct. If the opposite holds, the chirality is
    flipped — model's "near" corners are actually at the face-diagonal
    positions and we should swap to use the model's "far" positions as
    the new near corners.

    EMPIRICAL CAVEAT (2026-05-22). Validated against 4 user-labeled
    cases (set 12_A/B, 17_A/B): the synthetic signal is clean but the
    real-photo signal is noisy and frequently inverted. Suspected
    causes: (1) model vertex is slightly offset from the true cube
    vertex (typical ~10-50 px after PnP, reduced by the post-PnP
    mean-of-3 ensemble), so the line vertex→near doesn't actually lie
    on the bezel; (2) dark sticker colors (red, blue) along far
    diagonals compete with bezel darkness; (3) lighting glare on some
    bezels brightens them. Because of this we currently compute the
    signal and emit it as a debug field but DO NOT auto-correct by
    default (`apply_correction=False`). Future work: re-validate on
    the full axis-labeled gallery (in progress) to learn a reliable
    threshold, or replace darkness with a more robust geometric signal.

    Returns (model, debug_dict). Debug contains the per-corner
    darkness, mean-near vs mean-far, signed separation, and
    chirality_check status.
    """
    debug: Dict[str, Any] = {}
    if image_rgb is None:
        debug["chirality_check"] = "skipped_no_image"
        return model, debug

    near_keys = ["h_x", "h_y", "h_z"]
    far_keys = ["h_xy", "h_xz", "h_yz"]
    near_positions = [model.visible_corners.get(k) for k in near_keys]
    far_positions = [model.visible_corners.get(k) for k in far_keys]
    if any(p is None for p in near_positions) or any(p is None for p in far_positions):
        debug["chirality_check"] = "skipped_missing_corners"
        return model, debug

    vertex = model.cube_center_screen
    near_darkness = [
        _line_darkness_from_vertex(image_rgb, vertex, p)  # type: ignore[arg-type]
        for p in near_positions
    ]
    far_darkness = [
        _line_darkness_from_vertex(image_rgb, vertex, p)  # type: ignore[arg-type]
        for p in far_positions
    ]
    debug["chirality_near_line_darkness"] = [round(d, 1) for d in near_darkness]
    debug["chirality_far_line_darkness"] = [round(d, 1) for d in far_darkness]

    mean_near = sum(near_darkness) / 3.0
    mean_far = sum(far_darkness) / 3.0
    debug["chirality_mean_near_darkness"] = round(mean_near, 1)
    debug["chirality_mean_far_darkness"] = round(mean_far, 1)

    # Decision threshold: meaningful separation required. If the difference
    # is too small the image evidence is ambiguous and we leave the model
    # alone. A correctly-labeled cube under decent lighting should show
    # ~30-80 unit separation (bezels are ~20-50 darker than stickers on
    # a typical 0-255 inverted-gray scale).
    separation = mean_near - mean_far
    debug["chirality_darkness_separation"] = round(separation, 1)

    # Also report best-permutation errors before/after, in case downstream
    # tooling wants to compare these regimes. We compute these against the
    # 3 darkness-derived "true near" directions (top-3 by darkness across
    # all 6 corners).
    all_positions = list(near_positions) + list(far_positions)
    all_darkness = list(near_darkness) + list(far_darkness)
    # Under empirical polarity: true-near corners are the 3 LIGHTEST
    # (lowest darkness) among the 6 vertex→corner lines, not the darkest.
    sorted_idx = sorted(range(6), key=lambda i: all_darkness[i])
    true_near_positions = [all_positions[i] for i in sorted_idx[:3]]

    cx, cy = vertex

    def _axis_angles(positions):
        return [math.atan2(p[1] - cy, p[0] - cx) for p in positions]  # type: ignore[index]

    def best_match_error(ax_angles, target_angles):
        best_total = math.inf
        best_errors = None
        for perm in itertools.permutations(range(3)):
            errors = [
                _signed_angle_diff_deg(ax_angles[i], target_angles[perm[i]])
                for i in range(3)
            ]
            total = sum(errors)
            if total < best_total:
                best_total = total
                best_errors = errors
        return best_errors

    target_angles = _axis_angles(true_near_positions)
    current_axis_angles = _axis_angles(near_positions)
    errors_before = best_match_error(current_axis_angles, target_angles)
    debug["chirality_axis_angle_errors_before_deg"] = [round(e, 1) for e in errors_before]

    flipped_axis_angles = _axis_angles(far_positions)
    errors_after = best_match_error(flipped_axis_angles, target_angles)
    debug["chirality_axis_angle_errors_after_deg"] = [round(e, 1) for e in errors_after]

    # Insufficient lighting / texture contrast to make a call.
    if abs(separation) < 10.0:
        debug["chirality_check"] = "ambiguous_no_correction"
        return model, debug

    # EMPIRICAL POLARITY (validated 2026-05-21 on 58-case axis-labeled
    # gallery, 116 model runs). Counter-intuitively, in real photos the
    # line vertex→model.h_x (model's labeled "near", which IS the
    # near-in-3D 1-cube-edge endpoint) samples LIGHTER pixels than the
    # line vertex→model.h_xy (model's "far", 2-cube-edge endpoint via
    # face diagonal) when the chirality is CORRECT — i.e., sep<0 means
    # CORRECT, sep>0 means FLIPPED. This is the OPPOSITE of the
    # naive bezel-darkness reasoning. Suspected cause: typical model
    # vertex offset (~10–50 px from true cube vertex after PnP) means
    # vertex→near lines skim off the cube-edge bezel into adjacent
    # sticker interior, while vertex→far lines (across face diagonals)
    # cross multiple perpendicular internal bezels and rack up more
    # darkness. The base rate of chirality flips in the model is 52%;
    # this polarity-corrected detector agrees with position-based
    # ground truth on ~82% of non-ambiguous runs. See
    # tools/CHIRALITY_DETECTION_REPORT.md for the full empirical table.
    if separation < 0:
        # Model's labeled "near" is LIGHTER — empirically this is the
        # CORRECT chirality.
        debug["chirality_check"] = "correct"
        return model, debug

    # separation > +10: model's "near" is DARKER — empirically the
    # chirality is FLIPPED, swap to use far-corner positions as new
    # axes.
    if not apply_correction:
        debug["chirality_check"] = "flip_suggested_diagnostic_only"
        return model, debug

    new_axes = [
        (p[0] - cx, p[1] - cy) for p in far_positions  # type: ignore[index]
    ]
    corrected = GlobalCubeModel(
        cube_center_screen=model.cube_center_screen,
        axis_x_2d=new_axes[0],
        axis_y_2d=new_axes[1],
        axis_z_2d=new_axes[2],
        fit_loss=model.fit_loss,
        fit_quality=model.fit_quality,
    )
    derive_geometry(corrected)
    corrected.debug.update(model.debug)
    debug["chirality_check"] = "corrected_60deg_flip"
    return corrected, debug


def fit_global_cube_model(
    detection: InteriorBezelDetection,
    image_rgb: np.ndarray,
    silhouette_mask: np.ndarray,
    *,
    optimize: bool = True,
) -> Optional[GlobalCubeModel]:
    """Fit the cube model by aligning the 3D template to detected
    anchor points (1 cube_center + 6 hexagon outer vertices).

    Args:
      detection: bezel detection from PR #178 (provides cube_center)
      image_rgb: source image — used for image-based vertex refinement
        (search for the 3-way dark-line junction near the ensemble vertex)
      silhouette_mask: rembg silhouette (provides 6 hexagon vertices)
      optimize: unused (kept for API compat) — the Procrustes fit is
        already optimal in the least-squares sense

    Returns: best-fit GlobalCubeModel or None on failure.
    """
    if detection.cube_center is None:
        return None
    if _try_import_scipy_ndimage() is None:
        # Try anyway — we only need scipy.spatial.ConvexHull which
        # comes with scipy core. But our _try_import_scipy_ndimage
        # gate is a proxy for "is scipy installed at all". If it's
        # missing, _hull_from_mask returns []
        pass

    # Detect 6 hexagon outer vertices
    hexagon = detect_hexagon_anchors(silhouette_mask)
    if len(hexagon) != 6:
        return None

    # VERTEX (= cube_center) initial estimate from HEXAGON CENTROID.
    # For iso projection of a cube, the centroid of the 6 hexagon
    # corners is EXACTLY at the front vertex — the cube's body
    # diagonal projects to zero, so the projection of the (1,1,1)
    # corner coincides with the projection of the cube's geometric
    # center, which is also the centroid of the projected hexagon.
    # For yawed cubes this is approximate (small error, ~5-20 px)
    # but cv2.solvePnP corrects this in the fit. No dependency on
    # bezel detection for the vertex anymore.
    hex_arr = np.array(hexagon, dtype=np.float64)
    cube_center = (float(hex_arr[:, 0].mean()), float(hex_arr[:, 1].mean()))

    # Get bezel-detected vertex (may be None if detection failed).
    bezel_vertex = detection.cube_center if detection is not None else None

    # Fit the model from hexagon-only correspondences (NO bezel-7th).
    # Ground-truth analysis on 27 user-labeled cases showed the
    # bezel-7th-correspondence variant was a wash vs hex-only PnP
    # (both ~95 px mean error). Instead, we use a mean-of-3 ensemble
    # AFTER the PnP fit: cube_center = mean(PnP_vertex, bezel_vertex,
    # hexagon_centroid). This reduces mean error 14% (99 → 85 px on
    # the ground-truth corpus) — each method has semi-independent
    # error and the average regresses toward truth.
    model = fit_cube_template_to_anchors(
        cube_center, hexagon,
        detection.boundary_angles[:3] if detection is not None else (0, 0, 0),
        image_size=silhouette_mask.shape,
    )
    if model is None:
        return None

    # CHIRALITY CHECK + AUTO-CORRECTION.
    # In iso projection the 6 hexagon corners alternate between 3 "near"
    # (1 cube-edge from vertex) and 3 "far" (2 cube-edges). Both
    # assignments are symmetric under 60° rotation around the body
    # diagonal — same silhouette, different 3D pose. The Procrustes
    # 6! brute-force fit picks one assignment based on tiny noise
    # differences in the residual, which flips chirality on ~52% of
    # runs and ~55% of CASES are non-deterministic across reruns
    # (validated against 58-case axis-labeled gallery, 116 runs, on
    # 2026-05-21).
    #
    # The discriminator: pixel darkness along vertex→corner lines, with
    # an EMPIRICALLY-VALIDATED INVERTED POLARITY — sep<0 ≡ correct,
    # sep>0 ≡ flipped. Yields ~82% agreement with position-based
    # ground truth on non-ambiguous runs (vs 11% with the naive
    # bezel-darkness polarity). See tools/CHIRALITY_DETECTION_REPORT.md
    # for the full empirical table.
    #
    # With apply_correction=True the model is rebuilt with the
    # far-corner positions as new axes when the signal indicates a
    # flip. Given the 52% base rate and ~82% detector accuracy this
    # is a substantial net win over leaving the chirality random.
    model, chirality_debug = _apply_chirality_correction(
        model, detection, image_rgb, apply_correction=True
    )
    model.debug.update(chirality_debug)

    # MEAN-OF-3 VERTEX ENSEMBLE: average PnP vertex, bezel vertex, and
    # hexagon centroid. Empirically validated against 27 user-labeled
    # ground-truth marks. Falls back to PnP-only if bezel is missing.
    pnp_vertex = model.cube_center_screen
    hex_centroid = cube_center
    candidates = [pnp_vertex, hex_centroid]
    if bezel_vertex is not None:
        candidates.append(bezel_vertex)
    avg = (
        sum(p[0] for p in candidates) / len(candidates),
        sum(p[1] for p in candidates) / len(candidates),
    )
    # Override vertex with the ensemble average; KEEP the PnP-derived
    # axes (those are robust against single-vertex noise).
    pnp_to_avg_shift = (avg[0] - pnp_vertex[0], avg[1] - pnp_vertex[1])
    model.cube_center_screen = avg
    # Shift visible_corners and face_quads by the same delta so the
    # overlay stays geometrically consistent (this is a parallel
    # translation; axes are preserved).
    for k, p in list(model.visible_corners.items()):
        model.visible_corners[k] = (p[0] + pnp_to_avg_shift[0], p[1] + pnp_to_avg_shift[1])
    for name, quad in list(model.face_quads.items()):
        model.face_quads[name] = [
            (p[0] + pnp_to_avg_shift[0], p[1] + pnp_to_avg_shift[1]) for p in quad
        ]
    for name, cells in list(model.sticker_cells.items()):
        model.sticker_cells[name] = [
            [(p[0] + pnp_to_avg_shift[0], p[1] + pnp_to_avg_shift[1]) for p in cell]
            for cell in cells
        ]

    if bezel_vertex is not None:
        bezel_offset = math.hypot(
            cube_center[0] - bezel_vertex[0],
            cube_center[1] - bezel_vertex[1],
        )
        model.debug["hexagon_centroid_vs_bezel_vertex_offset_px"] = round(bezel_offset, 1)
    model.debug.update({
        "approach": "procrustes_template_fit+mean3_vertex",
        "n_hexagon_anchors": len(hexagon),
        "cube_center_source": "mean3_ensemble",
        "ensemble_n_candidates": len(candidates),
        "ensemble_shift_px": round(math.hypot(*pnp_to_avg_shift), 1),
    })

    # IMAGE-BASED VERTEX REFINEMENT (gated on absolute junction score).
    # The ensemble vertex above is silhouette-derived — for ~22% of cases
    # it sits 90-200 px from the actual trihedral corner because perspective
    # under heavy yaw shifts the true junction off the silhouette centroid.
    # We use the PnP-derived axes (which are robust independent of vertex
    # error) to search a ±40 px window for the strongest 3-way dark-line
    # junction. Gated on the input vertex's own junction score: if it's
    # already at a strong junction, don't refine (avoids drift to nearby
    # sticker corners with marginally higher local score).
    if image_rgb is not None:
        axes = [model.axis_x_2d, model.axis_y_2d, model.axis_z_2d]
        refined_vertex, refine_debug = _refine_vertex_via_image_junction(
            image_rgb, model.cube_center_screen, axes,
        )
        model.debug.update(refine_debug)
        if refined_vertex != model.cube_center_screen:
            refine_shift = (
                refined_vertex[0] - model.cube_center_screen[0],
                refined_vertex[1] - model.cube_center_screen[1],
            )
            model.cube_center_screen = refined_vertex
            # Apply same shift to all derived geometry so overlay stays consistent.
            for k, p in list(model.visible_corners.items()):
                model.visible_corners[k] = (p[0] + refine_shift[0], p[1] + refine_shift[1])
            for name, quad in list(model.face_quads.items()):
                model.face_quads[name] = [
                    (p[0] + refine_shift[0], p[1] + refine_shift[1]) for p in quad
                ]
            for name, cells in list(model.sticker_cells.items()):
                model.sticker_cells[name] = [
                    [(p[0] + refine_shift[0], p[1] + refine_shift[1]) for p in cell]
                    for cell in cells
                ]
            model.debug["cube_center_source"] = "mean3_ensemble+image_refined"

    return model
