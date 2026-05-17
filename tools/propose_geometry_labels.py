#!/usr/bin/env python3
"""Baseline auto-hull / face-quad proposers used by
`evaluate_auto_geometry.py`.

Each proposer takes a LabelTarget (image already loaded into processing
resolution) and returns a Proposal: a cube hull polygon + a dict of
face_label → 4-corner quad (matching the Geometry Labeler's format).

All proposers run in processing-resolution image space (max side 1150px,
EXIF-corrected) so their output is directly comparable to the labels
loaded by the evaluator.

Three baselines so far:

  * `recognizer_grids` — runs `analyze_image()`, takes each FaceGrid's
    4 outer sticker centers, fits a 4-point homography from unit-square
    interior to those centers, then evaluates the homography at the
    unit-square corners to get face quads. Cube hull = convex hull of
    all detected sticker centers.
  * `saturation_hull` — saturation mask → connected components → convex
    hull of the largest cube-like component. No face quads.
  * `roi_bbox` — uses `_find_cube_roi` only; reports the bbox as the
    cube hull. No face quads. Sanity-check baseline only.

Add SAM2 or a learned proposer here when/if classical baselines fall
short.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.image_pipeline import (  # noqa: E402
    _binary_dilate_square,
    _connected_components,
    _find_cube_roi,
    _rgb_to_hsv_arrays,
    analyze_image,
)
from tools.inspect_cube_isolation import convex_hull, expand_polygon  # noqa: E402
from tools.sample_stickers_from_hull import canonical_corner_order  # noqa: E402

if TYPE_CHECKING:
    from tools.evaluate_auto_geometry import LabelTarget


Point = Tuple[float, float]


@dataclass
class Proposal:
    cube_hull: List[Point] = field(default_factory=list)
    face_quads: Dict[str, List[Point]] = field(default_factory=dict)
    notes: Dict = field(default_factory=dict)


# ---------------- homography (extracted from sample_stickers_from_hull) ----------------


def homography_4_points(src_pts: Sequence[Point], dst_pts: Sequence[Point]) -> np.ndarray:
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    A = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        A.append([-sx, -sy, -1, 0, 0, 0, sx * dx, sy * dx, dx])
        A.append([0, 0, 0, -sx, -sy, -1, sx * dy, sy * dy, dy])
    A = np.array(A, dtype=np.float64)
    _, _, vh = np.linalg.svd(A)
    H = vh[-1].reshape(3, 3)
    return H / H[2, 2]


def warp(H: np.ndarray, u: float, v: float) -> Point:
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


# ---------------- recognizer-grids proposer ----------------


def _face_quad_from_grid_centers(grid_points: Sequence[Sequence[Point]]) -> Optional[List[Point]]:
    """Given a 3x3 grid of sticker centers, fit a homography from unit-square
    interior (1/6, 1/6)..(5/6, 5/6) to the 4 outer sticker centers, then
    evaluate at the unit-square corners (0,0)..(1,1) to get the face quad.

    Canonicalizes both source and destination corner order (CW from north)
    so the recognizer's arbitrary row-major grid orientation can't twist
    the homography into a self-intersecting mess."""
    if len(grid_points) != 3 or any(len(row) != 3 for row in grid_points):
        return None
    raw_dst = [
        (float(grid_points[0][0][0]), float(grid_points[0][0][1])),
        (float(grid_points[0][2][0]), float(grid_points[0][2][1])),
        (float(grid_points[2][2][0]), float(grid_points[2][2][1])),
        (float(grid_points[2][0][0]), float(grid_points[2][0][1])),
    ]
    # Sort destination CW from north → consistent corner correspondence
    dst = canonical_corner_order(raw_dst)
    # Unit-square interior CW from north: (mid-top is at u=0.5,v=1/6, but
    # we want 4 corners CW from N which is just the top-left rotation —
    # for a square the canonical sort places top-left first since "north
    # from centroid" picks the topmost corner. For our (1/6, 1/6) it picks
    # (1/6, 1/6) first because all 4 corners are equidistant from centroid
    # and the tiebreaker is angle-from-N).
    raw_src = [(1 / 6, 1 / 6), (5 / 6, 1 / 6), (5 / 6, 5 / 6), (1 / 6, 5 / 6)]
    src = canonical_corner_order(raw_src)
    try:
        H = homography_4_points(src, dst)
    except np.linalg.LinAlgError:
        return None
    # Evaluate at unit-square corners in the SAME canonical order
    raw_corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    corner_src = canonical_corner_order(raw_corners)
    return [warp(H, u, v) for (u, v) in corner_src]


class RecognizerGridsProposer:
    name = "recognizer_grids"

    def propose(self, target: "LabelTarget") -> Proposal:
        # Read the original image bytes (analyze_image does its own EXIF +
        # resize; we don't reuse target.arr because analyze_image needs raw
        # bytes for its own resize logic).
        image_bytes = target.image_path.read_bytes()
        analysis = analyze_image(image_bytes)

        # analyze_image's internal processing resolution may differ slightly
        # from target.proc_w/h if PROCESSING_MAX changes. We assume the
        # sticker centers are in the SAME processing-coordinate space as
        # target.proc_w/h (both use max-1150 EXIF-corrected resize), so no
        # rescaling needed.

        # analyze_image typically returns 20+ candidate grids per image
        # (one per stickers-subset fit). Pick the BEST grid per
        # center_face using (matched_count desc, fit_error asc). The
        # "first wins" approach took whichever appeared first which is
        # usually a spurious cross-face fit.
        grids_by_face: Dict[str, list] = {}
        for grid in analysis.grids:
            face = grid.center_face
            grids_by_face.setdefault(face, []).append(grid)

        face_quads: Dict[str, List[Point]] = {}
        for face, candidates in grids_by_face.items():
            best = min(candidates, key=lambda g: (-g.matched_count, g.fit_error))
            quad = _face_quad_from_grid_centers(best.points)
            if quad is None:
                continue
            face_quads[face] = [(float(x), float(y)) for (x, y) in quad]

        # Cube hull = convex hull of every detected sticker center, expanded
        # outward to cover the cube edge bezel (sticker centers sit ~12-18%
        # of face length inside the cube's outline; bezels add a few more %).
        all_centers = [(float(s.center[0]), float(s.center[1])) for s in analysis.stickers]
        cube_hull_raw = list(convex_hull(all_centers)) if len(all_centers) >= 3 else []
        cube_hull = list(expand_polygon(cube_hull_raw, padding_fraction=0.13))

        return Proposal(
            cube_hull=[(float(x), float(y)) for (x, y) in cube_hull],
            face_quads=face_quads,
            notes={
                "stickerCount": len(analysis.stickers),
                "gridCount": len(analysis.grids),
                "faceLabelsProposed": sorted(face_quads.keys()),
            },
        )


# ---------------- saturation-hull proposer ----------------


def _saturation_component_mask(arr: np.ndarray, sat_min: float = 0.23) -> Optional[np.ndarray]:
    """Find the largest saturation-thresholded connected component and
    return its binary mask (full image size)."""
    hsv = _rgb_to_hsv_arrays(arr)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    saturated = (sat > sat_min) & (val > 0.20)
    if saturated.sum() < 200:
        return None

    join = max(21, int(max(arr.shape[:2]) * 0.055) | 1)
    joined = _binary_dilate_square(saturated, join)
    comps = _connected_components(joined, min_area=250)
    if not comps:
        return None

    # Pick the largest component and rebuild its mask from (xs, ys)
    best = max(comps, key=lambda c: c["area"])
    mask = np.zeros(saturated.shape, dtype=bool)
    mask[best["ys"], best["xs"]] = True
    return mask


def _hull_from_mask(mask: np.ndarray) -> List[Point]:
    """Take every True pixel in `mask`, compute convex hull. For a
    well-behaved cube blob the hull is a 6-vertex hexagon-ish polygon
    that loosely traces the cube silhouette."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    # Downsample to avoid hull-on-huge-point-set; pick boundary pixels only
    # via a quick edge detection (XOR with shifted mask).
    h, w = mask.shape
    boundary = np.zeros_like(mask)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    ys, xs = np.where(boundary)
    if len(xs) < 3:
        ys, xs = np.where(mask)
    points = [(float(x), float(y)) for x, y in zip(xs, ys)]
    return list(convex_hull(points))


class SaturationHullProposer:
    name = "saturation_hull"

    def propose(self, target: "LabelTarget") -> Proposal:
        if target.arr is None:
            target.load()
        # analyze_image's internal resize may not match target.proc_w/h
        # exactly, but the sat mask runs on target.arr which IS in
        # processing-resolution, so hull lives in that space.
        mask = _saturation_component_mask(target.arr)
        if mask is None:
            return Proposal(notes={"reason": "no_saturated_component"})
        hull = _hull_from_mask(mask)
        return Proposal(
            cube_hull=hull,
            face_quads={},  # no face quads from saturation alone
            notes={"hullVertexCount": len(hull)},
        )


# ---------------- saturation hexagon + face quads proposer ----------------


def _fit_hexagon_to_hull(hull_points: Sequence[Point]) -> Optional[List[Point]]:
    """Fit a 6-vertex hexagon to a convex hull by picking the hull vertex
    farthest from centroid in each of 6 angular sectors.

    Cube viewed isometrically projects to a regular-ish hexagon; this is
    an approximation good enough for face-quad derivation via the
    Geometry Labeler's template formula."""
    if len(hull_points) < 6:
        return None
    cx = sum(p[0] for p in hull_points) / len(hull_points)
    cy = sum(p[1] for p in hull_points) / len(hull_points)
    # Find the topmost hull vertex (smallest y) — that's our "anchor" sector 0
    anchor_idx = min(range(len(hull_points)), key=lambda i: hull_points[i][1])
    anchor = hull_points[anchor_idx]
    # Angle of anchor from centroid (image y grows down, so "north" anchor
    # has dy<0; using atan2(dx, -dy) gives 0 at north, positive CW)
    from math import atan2, pi
    def angle(p):
        a = atan2(p[0] - cx, -(p[1] - cy))
        return a + 2 * pi if a < 0 else a

    anchor_angle = angle(anchor)
    sectors: List[List[Tuple[float, Point]]] = [[] for _ in range(6)]
    for p in hull_points:
        # Sector index = ((angle - anchor_angle) % 2pi) / (pi/3)
        rel = (angle(p) - anchor_angle) % (2 * pi)
        s = int(rel / (pi / 3)) % 6
        dist = ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5
        sectors[s].append((dist, p))
    hexagon: List[Point] = []
    for s in range(6):
        if not sectors[s]:
            return None  # incomplete coverage
        _, best = max(sectors[s], key=lambda dp: dp[0])
        hexagon.append(best)
    return hexagon


def _face_quads_from_hexagon(hexagon: Sequence[Point]) -> Dict[str, List[Point]]:
    """Apply the Geometry Labeler's template formula: 7 anchors (1 center
    + 6 hexagon vertices, CW from top) → 3 face quads. The labeler maps:

        top   (image A: U, image B: D): [hull[0], hull[1], center, hull[5]]
        right (image A: R, image B: ???): [hull[1], hull[2], hull[3], center]
        left  (image A: F, image B: ???): [hull[5], center, hull[3], hull[4]]

    These are anonymous "top/right/left" quads — the caller assigns
    face-name labels by side and (optionally) yaw."""
    if len(hexagon) != 6:
        return {}
    cx = sum(p[0] for p in hexagon) / 6.0
    cy = sum(p[1] for p in hexagon) / 6.0
    center = (cx, cy)
    h = list(hexagon)
    return {
        "top":   [h[0], h[1], center, h[5]],
        "right": [h[1], h[2], h[3], center],
        "left":  [h[5], center, h[3], h[4]],
    }


def _assign_face_labels(
    anonymous_quads: Dict[str, List[Point]],
    expected_faces: Sequence[str],
) -> Dict[str, List[Point]]:
    """Map anonymous positional labels (top/right/left) to face names.
    Image A: top=U, right=R, left=F. Image B: top=D, right=L, left=B
    by default — but this is the convention BEFORE yaw correction, so
    it's a best-effort; the evaluator's Hungarian-match metric is more
    informative than this strict label mapping for image B."""
    pos_to_face = {
        ("A",): {"top": "U", "right": "R", "left": "F"},
        ("B",): {"top": "D", "right": "L", "left": "B"},
    }
    # Detect which side we're on by checking which anchor face is in expected
    if "U" in expected_faces:
        mapping = pos_to_face[("A",)]
    else:
        mapping = pos_to_face[("B",)]
    out: Dict[str, List[Point]] = {}
    for pos, quad in anonymous_quads.items():
        face = mapping[pos]
        out[face] = quad
    return out


class SaturationHexagonProposer:
    name = "saturation_hexagon"

    def propose(self, target: "LabelTarget") -> Proposal:
        if target.arr is None:
            target.load()
        mask = _saturation_component_mask(target.arr)
        if mask is None:
            return Proposal(notes={"reason": "no_saturated_component"})
        hull = _hull_from_mask(mask)
        if len(hull) < 6:
            return Proposal(cube_hull=hull, notes={"reason": "hull_too_small"})
        hexagon = _fit_hexagon_to_hull(hull)
        if hexagon is None:
            return Proposal(cube_hull=hull, notes={"reason": "hexagon_fit_failed"})
        anonymous = _face_quads_from_hexagon(hexagon)
        side = "A" if "U" in target.gt_face_quads else "B"
        expected = ("U", "R", "F") if side == "A" else ("D", "L", "B")
        face_quads = _assign_face_labels(anonymous, expected)
        return Proposal(
            cube_hull=hexagon,  # the 6-vertex hexagon IS the cube outline
            face_quads=face_quads,
            notes={"hexagonVertexCount": len(hexagon)},
        )


# ---------------- ROI bbox proposer (sanity baseline) ----------------


class RoiBboxProposer:
    name = "roi_bbox"

    def propose(self, target: "LabelTarget") -> Proposal:
        if target.arr is None:
            target.load()
        roi = _find_cube_roi(target.arr)
        x0, y0, x1, y1 = roi
        bbox_hull = [
            (float(x0), float(y0)),
            (float(x1), float(y0)),
            (float(x1), float(y1)),
            (float(x0), float(y1)),
        ]
        return Proposal(
            cube_hull=bbox_hull,
            face_quads={},
            notes={"roi": [int(v) for v in roi]},
        )


# ---------------- registry ----------------


PROPOSERS = {
    cls.name: cls() for cls in (
        RecognizerGridsProposer,
        SaturationHexagonProposer,
        SaturationHullProposer,
        RoiBboxProposer,
    )
}
