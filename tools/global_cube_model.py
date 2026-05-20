"""Global cube projection model — fit a single 6-DOF projected cube
per photo so that all 27 sticker quads come from one coherent geometry.

Motivation (post-pivot Decision Log entry 2026-05-20):
  Three independent first-principles design proposals converged on
  the same architecture: replace per-sticker / per-grid local detection
  + post-hoc reconciliation with a SINGLE GLOBAL CUBE PROJECTION MODEL
  per photo whose 27 sticker quads are coherent by construction.

  Per Codex: "Every sampled sticker must come from the same valid
  projected cube model."

Parameterization (6 DOF):

  * `cube_center` (cx, cy): 2D image-space position of the front cube
    corner (where 3 visible faces meet — same vertex detected by
    PR #177/#178's interior bezel detector).
  * 3 axis angles (θ_0, θ_1, θ_2): the projected directions of the 3
    cube edges from cube_center to the 3 nearest visible cube corners
    (h1, h3, h5 in PR #176's taxonomy — the "interior" hexagon
    vertices that hull-based fitters provably can't find on yawed
    cubes).
  * `edge_length` (L): the length of one projected cube edge in
    pixels. Shared across all 3 axes — the cube is rigid.

Derived geometry — from the 6 parameters compute 7 visible cube
corners + 3 face quads + 27 sticker cells:

  Cube topology — 8 corners labeled (a, b, c) ∈ {0,1}³. The "front"
  corner (1,1,1) projects to cube_center; the "opposite" corner
  (0,0,0) is hidden behind the cube. The 6 visible non-center corners
  are the 6 hexagon silhouette vertices h0-h5.

  Let A_i = L * (cos θ_i, sin θ_i) be the i-th axis displacement.

  - cube_center      = (cx, cy)         ← (1,1,1) — front corner
  - h1, h3, h5 = cube_center + A_i      ← (0,1,1), (1,0,1), (1,1,0)
  - h0, h2, h4 = cube_center + A_i+A_j  ← (0,0,1), (0,1,0), (1,0,0)
  - hidden corner    = cube_center + A_0+A_1+A_2 ← (0,0,0)

  The 3 visible faces are parallelograms — each face contains the
  front corner + 2 axes + their sum. Per face, the 9 sticker cells
  are derived by subdividing the parallelogram into a 3×3 grid via
  bilinear interpolation.

This module is DIAGNOSTICS-ONLY (per the project's diagnostics-first
discipline). The fit produces overlay PNGs + per-pair JSON sidecars
with model parameters + fit quality. NO wiring into recognizer
behavior — that comes later after broader-corpus validation.

Dependencies:
  * numpy (required)
  * scipy (optional research dependency, same pattern as PR #177-#180;
    used for the optimization step). When missing,
    `fit_global_cube_model` returns a model from the initial estimate
    only with a clear `debug["error"]` flag.
  * No `rembg` at module top — caller supplies silhouette mask
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the bezel-detection initialization plumbing
from tools.interior_bezel_detection import (  # noqa: E402
    InteriorBezelDetection,
    detect_interior_bezel_lines,
    _try_import_scipy_ndimage,
)


Point = Tuple[float, float]
Quad = Tuple[Point, Point, Point, Point]


# Cube topology — 8 corners labeled (a, b, c) ∈ {0,1}³
# These are the 6 hexagon-vertex corners (h0..h5) + the front-center +
# hidden corner. Numeric values are CUBE COORDS, not image coords.
_CORNER_111 = (1, 1, 1)  # cube_center (the front visible corner)
_CORNER_011 = (0, 1, 1)  # h-vertex along axis 0
_CORNER_101 = (1, 0, 1)  # h-vertex along axis 1
_CORNER_110 = (1, 1, 0)  # h-vertex along axis 2
_CORNER_001 = (0, 0, 1)  # hexagon vertex along (axis 0 + axis 1)
_CORNER_010 = (0, 1, 0)  # hexagon vertex along (axis 0 + axis 2)
_CORNER_100 = (1, 0, 0)  # hexagon vertex along (axis 1 + axis 2)
_CORNER_000 = (0, 0, 0)  # opposite corner (hidden behind cube)


@dataclass
class GlobalCubeModel:
    """6-DOF projected cube model + derived geometry."""

    # Parameters (the actual DOFs)
    cube_center: Point = (0.0, 0.0)
    axis_angles_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    edge_length_px: float = 0.0

    # Derived geometry (computed from parameters by `derive_geometry`)
    visible_corners: dict = field(default_factory=dict)
    face_quads: dict = field(default_factory=dict)
    sticker_cells: dict = field(default_factory=dict)

    # Fit quality (filled in by the fitter)
    fit_loss: float = float("inf")
    fit_quality: float = 0.0
    debug: dict = field(default_factory=dict)


def _corner_position(
    cube_center: Point,
    axis_displacements: Sequence[Tuple[float, float]],
    corner: Tuple[int, int, int],
) -> Point:
    """Compute the image-space position of cube corner (a, b, c).

    The cube's 8 corners are at 3D positions (a, b, c). The "front"
    corner (1,1,1) projects to cube_center. Other corners are at
    cube_center + sum of axis displacements for each "0" coordinate
    (i.e., the axis points FROM (1,1,1) TOWARD (0,1,1), (1,0,1),
    (1,1,0) — those are the corners along single axes from the front).
    """
    cx, cy = cube_center
    a, b, c = corner
    for i, present in enumerate([a, b, c]):
        if present == 0:
            dx, dy = axis_displacements[i]
            cx += dx
            cy += dy
    return (cx, cy)


def derive_geometry(model: GlobalCubeModel) -> None:
    """Populate `visible_corners`, `face_quads`, `sticker_cells` from
    the 6 DOF parameters. Mutates `model` in place."""
    L = model.edge_length_px
    angles = model.axis_angles_rad
    axis_displacements = [
        (L * math.cos(angles[i]), L * math.sin(angles[i]))
        for i in range(3)
    ]

    corners = {
        "111": _corner_position(model.cube_center, axis_displacements, _CORNER_111),
        "011": _corner_position(model.cube_center, axis_displacements, _CORNER_011),
        "101": _corner_position(model.cube_center, axis_displacements, _CORNER_101),
        "110": _corner_position(model.cube_center, axis_displacements, _CORNER_110),
        "001": _corner_position(model.cube_center, axis_displacements, _CORNER_001),
        "010": _corner_position(model.cube_center, axis_displacements, _CORNER_010),
        "100": _corner_position(model.cube_center, axis_displacements, _CORNER_100),
        # 000 is hidden behind the cube; not needed for visible geometry
    }
    model.visible_corners = corners

    # The 3 visible faces are parallelograms. Each contains the front
    # corner (111) + 2 axes + their pairwise sum. Labels are
    # placeholder — actual U/R/F assignment depends on capture
    # convention + center-sticker colors (resolved downstream).
    #
    # Each face is described by 4 corners in CCW order so that the
    # cell-grid subdivision below maps cleanly.
    #   face_AB: contains axes A and B (perpendicular to axis C)
    #
    # The face containing axes 0 and 1 is the one "above" the cube_center
    # (typically U in standard iso). Axes 0+2 = front face (F).
    # Axes 1+2 = right face (R). Etc. The labeling here is arbitrary;
    # real-cube U/F/R/D/L/B is assigned post-fit using center colors.
    face_quads = {
        # contains axes 0 and 1 (perpendicular to axis 2)
        "face_01": (corners["111"], corners["011"], corners["001"], corners["101"]),
        # contains axes 1 and 2 (perpendicular to axis 0)
        "face_12": (corners["111"], corners["101"], corners["100"], corners["110"]),
        # contains axes 0 and 2 (perpendicular to axis 1)
        "face_02": (corners["111"], corners["110"], corners["010"], corners["011"]),
    }
    model.face_quads = face_quads

    # Per face, subdivide the parallelogram into a 3×3 grid of cells.
    # `face` is (A, B, C, D) in CCW order: A is the front corner,
    # B is along axis_i, C is the far corner (along axis_i + axis_j),
    # D is along axis_j.
    sticker_cells: dict = {}
    for face_name, (A, B, C, D) in face_quads.items():
        cells = []
        for row in range(3):
            for col in range(3):
                # Bilinear interp: position = A + col/3 * (B-A) + row/3 * (D-A)
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


def _silhouette_iou(
    hexagon_vertices: Sequence[Point], silhouette_mask: np.ndarray
) -> float:
    """Compute IoU between the model's predicted hexagon silhouette
    (h0-h5 in order) and the rembg silhouette mask.

    Implemented via rasterization: rasterize the hexagon onto a binary
    mask the same size as `silhouette_mask`, then compute IoU.
    """
    h, w = silhouette_mask.shape
    poly_mask = np.zeros((h, w), dtype=bool)
    # Rasterize via PIL (we already depend on Pillow)
    from PIL import Image, ImageDraw
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    # PIL polygon expects flat list of (x, y)
    pts = [(float(p[0]), float(p[1])) for p in hexagon_vertices]
    draw.polygon(pts, fill=255)
    poly_mask = np.array(img) > 0
    inter = np.logical_and(poly_mask, silhouette_mask).sum()
    union = np.logical_or(poly_mask, silhouette_mask).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _bezel_angle_match_score(
    model: GlobalCubeModel,
    bezel_angles_rad: Sequence[float],
    bezel_qualities: Sequence[float],
) -> float:
    """Score (in [0, 1]) how well the model's 3 axis angles match the
    detected bezel angles, weighted by per-bezel detection quality.

    Each model axis can match ANY detected bezel — we pick the best
    pairing greedily.
    """
    if not bezel_angles_rad:
        return 0.5  # neutral when no bezels detected
    model_angles = list(model.axis_angles_rad)
    detected = list(zip(bezel_angles_rad, bezel_qualities))
    total_score = 0.0
    total_weight = 0.0
    for det_angle, det_quality in detected:
        # Match this detected angle to the closest model axis (in
        # [0, π) since lines are direction-agnostic mod π)
        best_diff = float("inf")
        for model_angle in model_angles:
            # Circular distance in [0, π)
            d = abs(((det_angle - model_angle + math.pi / 2) % math.pi) - math.pi / 2)
            if d < best_diff:
                best_diff = d
        # Score 1 at perfect match (0° diff), 0 at 30° diff
        score = max(0.0, 1.0 - best_diff / (math.pi / 6))
        weight = det_quality + 0.05  # small floor so low-quality bezels still register
        total_score += score * weight
        total_weight += weight
    if total_weight < 1e-9:
        return 0.5
    return total_score / total_weight


def compute_fit_loss(
    model: GlobalCubeModel,
    silhouette_mask: np.ndarray,
    bezel_angles_rad: Sequence[float],
    bezel_qualities: Sequence[float],
    *,
    weight_silhouette: float = 1.0,
    weight_bezel: float = 0.5,
) -> Tuple[float, dict]:
    """Compute the fit loss for a model. Returns (loss, debug_dict).

    Loss = weight_silhouette * (1 - IoU) + weight_bezel * (1 - bezel_match).

    Lower is better. Loss in [0, weight_silhouette + weight_bezel].
    """
    if not model.visible_corners:
        derive_geometry(model)
    # Hexagon vertices in CCW order: h0, h1, h2, h3, h4, h5
    # — h0 is at corner (0,0,1) (axes 0+1)
    # — h1 is at corner (0,1,1) (axis 0 alone)
    # — h2 is at corner (0,1,0) (axes 0+2)
    # — h3 is at corner (1,1,0) (axis 2 alone)
    # — h4 is at corner (1,0,0) (axes 1+2)
    # — h5 is at corner (1,0,1) (axis 1 alone)
    hexagon = [
        model.visible_corners["001"],  # h0
        model.visible_corners["011"],  # h1
        model.visible_corners["010"],  # h2
        model.visible_corners["110"],  # h3
        model.visible_corners["100"],  # h4
        model.visible_corners["101"],  # h5
    ]
    iou = _silhouette_iou(hexagon, silhouette_mask)
    bezel_match = _bezel_angle_match_score(model, bezel_angles_rad, bezel_qualities)

    loss = (
        weight_silhouette * (1.0 - iou)
        + weight_bezel * (1.0 - bezel_match)
    )
    debug = {
        "silhouette_iou": round(iou, 4),
        "bezel_match_score": round(bezel_match, 4),
        "weight_silhouette": weight_silhouette,
        "weight_bezel": weight_bezel,
    }
    return loss, debug


def initialize_from_bezel_detection(
    detection: InteriorBezelDetection,
    silhouette_mask: np.ndarray,
) -> Optional[GlobalCubeModel]:
    """Build an initial GlobalCubeModel from the outputs of
    `detect_interior_bezel_lines`. Returns None if the detection
    doesn't have a cube_center or sufficient bezel angles.

    The bezel detector returns LINE angles in [0, π) (lines have no
    direction). The global model needs OUTWARD AXIS angles in [0, 2π)
    (each axis points from cube_center toward an h-vertex). For 3
    detected line angles there are 8 possible sign combinations; we
    try all 8 and pick the one with highest initial silhouette IoU.

    Initial edge_length: ~0.35 × silhouette max extent. For iso
    projection of a unit cube, the silhouette spans ~2.83 × edge_length
    corner-to-corner, so edge ≈ silhouette / 2.83 ≈ silhouette * 0.354.
    Tuned to ~0.32 empirically (edge slightly underestimated → optimizer
    grows it; underestimate is safer than overestimate which can put
    the hexagon outside the silhouette and stall the optimizer).
    """
    if detection.cube_center is None:
        return None
    if len(detection.boundary_angles) < 3:
        return None

    ys, xs = np.where(silhouette_mask)
    if len(xs) < 100:
        return None
    sil_width = float(xs.max() - xs.min())
    sil_height = float(ys.max() - ys.min())
    init_edge = max(sil_width, sil_height) * 0.32

    line_angles = detection.boundary_angles[:3]

    # Try all 8 sign combinations
    best_iou = -1.0
    best_model = None
    for s0 in (1, -1):
        for s1 in (1, -1):
            for s2 in (1, -1):
                signs = (s0, s1, s2)
                outward_angles = tuple(
                    (la + (0 if s == 1 else math.pi)) % (2 * math.pi)
                    for la, s in zip(line_angles, signs)
                )
                # Geometric sanity: 3 outward axes from a cube's front
                # corner should span the 360° circle once. If two
                # outward angles are within ~30° of each other, this
                # combination has two axes "pointing the same way" — skip.
                sorted_angles = sorted(outward_angles)
                gaps = [
                    (sorted_angles[(i + 1) % 3] - sorted_angles[i]) % (2 * math.pi)
                    for i in range(3)
                ]
                min_gap_deg = min(math.degrees(g) for g in gaps)
                if min_gap_deg < 30:
                    continue
                m = GlobalCubeModel(
                    cube_center=detection.cube_center,
                    axis_angles_rad=outward_angles,
                    edge_length_px=init_edge,
                )
                derive_geometry(m)
                hexagon = [
                    m.visible_corners["001"],
                    m.visible_corners["011"],
                    m.visible_corners["010"],
                    m.visible_corners["110"],
                    m.visible_corners["100"],
                    m.visible_corners["101"],
                ]
                iou = _silhouette_iou(hexagon, silhouette_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_model = m
                    best_model.debug = {
                        "sign_combination_picked": list(signs),
                        "sign_init_iou": round(iou, 4),
                    }

    return best_model


def fit_global_cube_model(
    detection: InteriorBezelDetection,
    silhouette_mask: np.ndarray,
    *,
    optimize: bool = True,
    weight_silhouette: float = 1.0,
    weight_bezel: float = 0.5,
) -> Optional[GlobalCubeModel]:
    """Fit a 6-DOF projected cube model to the silhouette + bezel
    evidence. Returns the fitted model with `fit_loss` and
    `fit_quality` populated.

    Args:
      detection: bezel-detection output (initialization source)
      silhouette_mask: rembg silhouette
      optimize: if False, return the initialization-only model (skip
        the optimization step). Useful for debugging.
      weight_silhouette: loss weight on silhouette IoU mismatch
      weight_bezel: loss weight on bezel angle mismatch

    Returns: GlobalCubeModel or None on failure.
    """
    model = initialize_from_bezel_detection(detection, silhouette_mask)
    if model is None:
        return None

    initial_loss, initial_debug = compute_fit_loss(
        model, silhouette_mask,
        detection.boundary_angles, detection.line_qualities,
        weight_silhouette=weight_silhouette,
        weight_bezel=weight_bezel,
    )
    model.fit_loss = initial_loss
    model.debug = {
        "initial_loss": round(initial_loss, 4),
        "initial_silhouette_iou": initial_debug["silhouette_iou"],
        "initial_bezel_match": initial_debug["bezel_match_score"],
        "initial_edge_length_px": round(model.edge_length_px, 1),
        "initial_axis_angles_deg": [
            round(math.degrees(a), 2) for a in model.axis_angles_rad
        ],
        "optimized": False,
    }

    if not optimize:
        derive_geometry(model)
        # Quality heuristic: combine IoU + bezel match
        model.fit_quality = (
            initial_debug["silhouette_iou"] * 0.7
            + initial_debug["bezel_match_score"] * 0.3
        )
        return model

    # Optimize via scipy if available
    try:
        from scipy.optimize import minimize  # type: ignore
    except ImportError:
        # Fall back to initialization-only result
        model.debug["error"] = "scipy not installed; returning init-only model"
        derive_geometry(model)
        model.fit_quality = (
            initial_debug["silhouette_iou"] * 0.7
            + initial_debug["bezel_match_score"] * 0.3
        )
        return model

    bezels = detection.boundary_angles
    bezel_q = detection.line_qualities

    def objective(x):
        m = GlobalCubeModel(
            cube_center=(float(x[0]), float(x[1])),
            axis_angles_rad=(float(x[2]), float(x[3]), float(x[4])),
            edge_length_px=float(x[5]),
        )
        derive_geometry(m)
        loss, _ = compute_fit_loss(
            m, silhouette_mask, bezels, bezel_q,
            weight_silhouette=weight_silhouette,
            weight_bezel=weight_bezel,
        )
        return loss

    x0 = np.array([
        model.cube_center[0],
        model.cube_center[1],
        model.axis_angles_rad[0],
        model.axis_angles_rad[1],
        model.axis_angles_rad[2],
        model.edge_length_px,
    ])

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "xatol": 1.0,        # 1 px precision
            "fatol": 0.001,      # loss precision
            "maxiter": 300,
            "adaptive": True,
        },
    )

    fitted = GlobalCubeModel(
        cube_center=(float(result.x[0]), float(result.x[1])),
        axis_angles_rad=(float(result.x[2]), float(result.x[3]), float(result.x[4])),
        edge_length_px=float(result.x[5]),
    )
    derive_geometry(fitted)
    final_loss, final_debug = compute_fit_loss(
        fitted, silhouette_mask, bezels, bezel_q,
        weight_silhouette=weight_silhouette,
        weight_bezel=weight_bezel,
    )
    fitted.fit_loss = final_loss
    fitted.fit_quality = (
        final_debug["silhouette_iou"] * 0.7
        + final_debug["bezel_match_score"] * 0.3
    )
    fitted.debug = {
        **model.debug,
        "optimized": True,
        "optimizer_converged": bool(result.success),
        "optimizer_iterations": int(result.nit),
        "optimizer_function_evals": int(result.nfev),
        "final_loss": round(final_loss, 4),
        "final_silhouette_iou": final_debug["silhouette_iou"],
        "final_bezel_match": final_debug["bezel_match_score"],
        "final_edge_length_px": round(fitted.edge_length_px, 1),
        "final_axis_angles_deg": [
            round(math.degrees(a), 2) for a in fitted.axis_angles_rad
        ],
        "center_shift_from_init_px": round(
            math.hypot(
                fitted.cube_center[0] - model.cube_center[0],
                fitted.cube_center[1] - model.cube_center[1],
            ),
            1,
        ),
    }
    return fitted
