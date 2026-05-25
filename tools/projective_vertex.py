"""Projective vertex solver via vanishing-point construction.

Origin: Codex review of PR #282/#286 follow-up walkthrough on 2026-05-25.
The affine (parallelogram-completion) vertex in
``tools.rectify_via_hull_labels._derive_vertex_from_corners`` assumes
each visible cube face projects to a parallelogram in the image (true
under iso, false under perspective). For tilted captures (e.g.
``37_B``), the 3 per-face affine vertex estimates spread by 200+ px
and the mean lands 50-80 px off ground truth.

The projective construction exploits cube structure: for the visible
trihedral vertex V, V→NEAR_corner is along one cube axis. All cube
edges along that same axis meet at one image vanishing point (VP) under
perspective. So V lies on the line through the NEAR corner toward that
VP. Three such NEAR corners give three lines; V is at their best
intersection (3-line least-squares).

Reduces cleanly to affine: when projection is iso-ish, vanishing points
go to infinity → lines become parallels → 3-line intersection collapses
to the parallelogram-completion vertex.

No camera intrinsics needed (unlike PnP) — pure 2D projective geometry
on the 6 hexagon-corner positions.

This module is production-shaped but diagnostic-only. The companion
``tools/diagnose_projective_vertex.py`` compares affine vs projective
vertex across the 70-row corpus.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.corner_conventions import (
    FACE_DEFS_BY_SIDE,
    ONE_EDGE_CORNERS_BY_SIDE,
)


Point = Tuple[float, float]


# Degeneracy classification thresholds. Tunable.
# "near_affine": VP is further from the cube hexagon than this multiple
#                of the hexagon's longest dimension → treat as parallel.
NEAR_AFFINE_VP_DISTANCE_MULTIPLE = 50.0
# "degenerate": projective residual (best-fit line-intersection error)
#               exceeds this fraction of hexagon longest dimension →
#               lines don't meet cleanly, signal is unreliable.
DEGENERATE_RESIDUAL_FRACTION = 0.05


@dataclass(frozen=True)
class ProjectiveVertexResult:
    """Result of the projective vertex solve."""
    vertex: Point
    """The (x, y) image-space vertex estimate."""

    residual_px: float
    """Mean perpendicular distance from solved vertex to each of the 3 lines.
    A perfect projective fit has residual_px ≈ 0."""

    residual_norm: float
    """``residual_px / hexagon_diameter``. Resolution-independent — same
    threshold works on different image sizes / crops."""

    vanishing_points: Tuple[Optional[Point], Optional[Point], Optional[Point]]
    """The 3 vanishing points used (None if computed as at-infinity)."""

    degeneracy: str
    """One of:
       - ``"finite_projective"``: all 3 VPs are finite + lines meet cleanly
       - ``"near_affine"``: at least one VP is at infinity or very far
                            (projection is close to iso; projective ≈ affine)
       - ``"degenerate"``: lines don't meet cleanly (high residual)
                           — neither affine nor projective is trustworthy here"""

    hexagon_diameter_px: float
    """Longest pairwise distance among the 6 hexagon corners — used
    as the scale for ``residual_norm``."""


def derive_axis_edge_pairs(side: str) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """For each NEAR corner of V on this side, return the pairs of OTHER
    cube edges that are PARALLEL (in 3D) to V→NEAR. Those two parallel
    edges intersect (in image space) at the vanishing point for that
    cube axis.

    Output: list of ``(near_corner_num, [(c_a, c_b), (c_c, c_d)])``
    entries. Each ``(c_X, c_Y)`` is a 2-corner-number tuple naming a
    cube edge between two non-vertex visible corners.

    Derivation: each visible face has the structure
    ``(vertex, near_a, far, near_b)``. In 3D this is a parallelogram,
    so the edges ``V→near_a`` and ``far→near_b`` are parallel (opposite
    sides), as are ``V→near_b`` and ``far→near_a``. A given NEAR corner
    appears in exactly 2 of the 3 visible faces; each of those faces
    contributes ONE parallel-edge tuple (the one whose endpoint is the
    OPPOSITE NEAR corner on that face).
    """
    face_defs = FACE_DEFS_BY_SIDE[side]
    near_corner_names = ONE_EDGE_CORNERS_BY_SIDE[side]
    near_corner_nums = sorted(int(n.split("_")[1]) for n in near_corner_names)

    out: List[Tuple[int, List[Tuple[int, int]]]] = []
    for near_cn in near_corner_nums:
        parallel_pairs: List[Tuple[int, int]] = []
        for _slot, names in face_defs.items():
            if names[0] != "vertex":
                continue
            n_a = int(names[1].split("_")[1])
            far = int(names[2].split("_")[1])
            n_b = int(names[3].split("_")[1])
            if n_a == near_cn:
                # V→n_a is parallel to far→n_b in 3D
                parallel_pairs.append((far, n_b))
            elif n_b == near_cn:
                # V→n_b is parallel to far→n_a in 3D
                parallel_pairs.append((far, n_a))
        if len(parallel_pairs) != 2:
            raise ValueError(
                f"side {side!r} NEAR corner {near_cn} appears in "
                f"{len(parallel_pairs)} faces (expected 2). "
                f"FACE_DEFS_BY_SIDE may be malformed."
            )
        out.append((near_cn, parallel_pairs))
    return out


# Cache the derived axis-edge pairs per side — they're convention,
# not data, and computing once at module load is cheap.
_AXIS_EDGE_PAIRS: Dict[str, List[Tuple[int, List[Tuple[int, int]]]]] = {
    side: derive_axis_edge_pairs(side) for side in ("A", "B")
}


def line_coeffs(p: Point, q: Point) -> Tuple[float, float, float]:
    """Return (a, b, c) such that the line through p and q satisfies
    ax + by = c. Standard 2D line representation."""
    a = q[1] - p[1]
    b = p[0] - q[0]
    c = a * p[0] + b * p[1]
    return a, b, c


def intersect_lines(
    L1: Tuple[float, float, float],
    L2: Tuple[float, float, float],
) -> Optional[Point]:
    """Solve the 2x2 linear system for line intersection.

    Returns None when the lines are parallel (determinant ≈ 0)."""
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    return (x, y)


def _hexagon_diameter(corners_by_num: Dict[int, Point]) -> float:
    pts = list(corners_by_num.values())
    d = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = max(d, math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
    return d


def projective_vertex(
    corners_by_num: Dict[int, Point], side: str,
) -> ProjectiveVertexResult:
    """Compute V via 3-line vanishing-point intersection.

    Each visible NEAR corner has a corresponding cube axis. The 2 other
    visible edges along that axis meet at the image vanishing point.
    V lies on the line through the NEAR corner toward that VP.

    Returns a ``ProjectiveVertexResult`` with vertex, residual, VPs,
    and a degeneracy classification.

    See module docstring for the derivation. Reduces to parallelogram-
    completion when projection is iso (VPs at infinity).
    """
    if side not in _AXIS_EDGE_PAIRS:
        raise ValueError(f"unsupported side {side!r}; expected 'A' or 'B'")

    diameter = _hexagon_diameter(corners_by_num)
    if diameter <= 0.0:
        raise ValueError("hexagon has zero diameter (degenerate input)")

    axis_edges = _AXIS_EDGE_PAIRS[side]
    vps: List[Optional[Point]] = []
    lines_through_V: List[Tuple[float, float, float]] = []
    has_vp_at_infinity = False
    has_distant_vp = False

    for near_cn, parallel_pairs in axis_edges:
        near_pt = corners_by_num[near_cn]
        e1_p, e1_q = corners_by_num[parallel_pairs[0][0]], corners_by_num[parallel_pairs[0][1]]
        e2_p, e2_q = corners_by_num[parallel_pairs[1][0]], corners_by_num[parallel_pairs[1][1]]
        vp = intersect_lines(line_coeffs(e1_p, e1_q), line_coeffs(e2_p, e2_q))
        vps.append(vp)
        if vp is None:
            # Image-space edges are parallel → vanishing point at infinity.
            # Treat the line through NEAR corner as having the same direction.
            direction = (e1_q[0] - e1_p[0], e1_q[1] - e1_p[1])
            # Line through near_pt with direction (dx, dy): normal is (-dy, dx)
            a, b = -direction[1], direction[0]
            c = a * near_pt[0] + b * near_pt[1]
            lines_through_V.append((a, b, c))
            has_vp_at_infinity = True
        else:
            # VP is finite. Check whether it's "very far" — close to affine.
            vp_dist_from_near = math.hypot(vp[0] - near_pt[0], vp[1] - near_pt[1])
            if vp_dist_from_near > NEAR_AFFINE_VP_DISTANCE_MULTIPLE * diameter:
                has_distant_vp = True
            lines_through_V.append(line_coeffs(near_pt, vp))

    # Least-squares solve for V = intersection of 3 lines
    # Normalize each row so a^2 + b^2 = 1 → row residual is perpendicular
    # distance from the solved point to the line.
    A = np.array([[L[0], L[1]] for L in lines_through_V], dtype=np.float64)
    bvec = np.array([L[2] for L in lines_through_V], dtype=np.float64)
    norms = np.linalg.norm(A, axis=1)
    if (norms < 1e-12).any():
        raise ValueError("degenerate line in projective vertex LSQ (zero normal)")
    A = A / norms[:, None]
    bvec = bvec / norms

    V_arr, _residuals, _rank, _sv = np.linalg.lstsq(A, bvec, rcond=None)
    V: Point = (float(V_arr[0]), float(V_arr[1]))

    # Per-line perpendicular residual (since we normalized)
    per_line_residuals = np.abs(A @ V_arr - bvec)
    residual_px = float(np.mean(per_line_residuals))
    residual_norm = residual_px / diameter

    # Classify degeneracy
    if residual_norm > DEGENERATE_RESIDUAL_FRACTION:
        degeneracy = "degenerate"
    elif has_vp_at_infinity or has_distant_vp:
        degeneracy = "near_affine"
    else:
        degeneracy = "finite_projective"

    return ProjectiveVertexResult(
        vertex=V,
        residual_px=residual_px,
        residual_norm=residual_norm,
        vanishing_points=(vps[0], vps[1], vps[2]),
        degeneracy=degeneracy,
        hexagon_diameter_px=diameter,
    )
