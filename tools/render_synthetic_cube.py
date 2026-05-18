#!/usr/bin/env python3
"""Synthetic Rubik's-cube image renderer for the two-view recognition
corpus. v1: PIL-based, hand-rolled isometric projection with flat
shading. Renders a paired (image A, image B) for a given cube state.

Image A: standard isometric view, U/R/F visible.
Image B: same physical cube after a 180° flip that places D/L/B in the
visible positions of the labeler's image-B convention.

Outputs per render-pair:
  <prefix>_A.png  — image A (1150×862 by default)
  <prefix>_B.png  — image B
  <prefix>.json   — metadata: state, render params, cube_hull polygons
                    (per image), face_quads (per image), per_sticker
                    image-coord positions

This v1 emits geometrically-correct but visually-simple images. Day-2
work will add: per-image background variation, lighting jitter, optional
real-photo background composite, JPEG noise, mild perspective camera
warp. Day-3 work will add a configuration generator that produces
thousands of random pairs.

Conventions:
  - State string in URFDLB order (cubejs/cube-snap convention)
    Positions 0-8 = U face, 9-17 = R, 18-26 = F, 27-35 = D, 36-44 = L, 45-53 = B
    Center at position 4 within each chunk = face's own color (cube-physics invariant)
  - Each face's 9 stickers are row-major from the face's local (0,0) corner

This is investigation/eval tooling. NO production-recognizer changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


# ---------------- color palette ----------------


# Match cube-snap / ctvd CANONICAL_RGB. Centers always have these colors;
# 8 non-center stickers per face have one of these colors per state string.
CUBE_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "U": (238, 238, 232),  # white
    "R": (190, 48, 36),    # red
    "F": (58, 145, 82),    # green
    "D": (230, 210, 42),   # yellow
    "L": (218, 112, 42),   # orange
    "B": (62, 86, 150),    # blue
}
BEZEL_RGB = (15, 15, 15)  # cube body / between-sticker gaps
FACE_ORDER = "URFDLB"


# ---------------- 3D cube model ----------------


@dataclass
class CubeModel:
    """A unit cube centered at origin with axis-aligned faces.

      U face at y = +0.5 (top), normal +Y
      D face at y = -0.5 (bottom), normal -Y
      R face at x = +0.5 (right), normal +X
      L face at x = -0.5 (left), normal -X
      F face at z = +0.5 (front, toward camera), normal +Z
      B face at z = -0.5 (back), normal -Z

    Each face has 9 stickers arranged 3×3 in face-local (u, v) coordinates,
    where (u=0, v=0) is the face's row-major "first sticker" corner.
    The mapping from face-local (u, v) to world (x, y, z) is fixed per face
    to match the URFDLB sticker-position convention."""

    size: float = 1.0
    bezel_fraction: float = 0.04
    sticker_inset_fraction: float = 0.04  # gap between stickers within a face

    def face_corners_3d(self, face: str) -> List[Tuple[float, float, float]]:
        """Return the 4 corners of the face in world coords, in row-major
        order: corner at (u=0, v=0), (u=1, v=0), (u=1, v=1), (u=0, v=1)."""
        s = self.size / 2.0
        # Per-face (origin, u_direction, v_direction) in world coords. The
        # origin is the (row=0, col=0) sticker's corner; +u advances along
        # columns (row stays), +v advances along rows.
        #
        # Convention chosen so that for image A view (camera at +x, +y,
        # +z looking at origin):
        #   * U face's row 0 is at the BACK (-z side), row 2 at the FRONT
        #   * R face's row 0 is at the TOP (+y), row 2 at the BOTTOM
        #   * F face's row 0 is at the TOP (+y), row 2 at the BOTTOM
        FACE_BASIS = {
            "U": {"origin": (-s, s, -s), "u": (1, 0, 0), "v": (0, 0, 1)},   # u→+x (L→R), v→+z (back→front)
            "D": {"origin": (-s, -s, s), "u": (1, 0, 0), "v": (0, 0, -1)},  # u→+x (L→R), v→-z (front→back)
            "R": {"origin": (s, s, s), "u": (0, 0, -1), "v": (0, -1, 0)},   # u→-z (front→back), v→-y (top→bottom)
            "L": {"origin": (-s, s, -s), "u": (0, 0, 1), "v": (0, -1, 0)},  # u→+z (back→front), v→-y (top→bottom)
            "F": {"origin": (-s, s, s), "u": (1, 0, 0), "v": (0, -1, 0)},   # u→+x (L→R), v→-y (top→bottom)
            "B": {"origin": (s, s, -s), "u": (-1, 0, 0), "v": (0, -1, 0)},  # u→-x (R→L), v→-y (top→bottom)
        }
        basis = FACE_BASIS[face]
        ox, oy, oz = basis["origin"]
        ux, uy, uz = basis["u"]
        vx, vy, vz = basis["v"]
        full = self.size
        return [
            (ox, oy, oz),                                  # (u=0, v=0)
            (ox + ux * full, oy + uy * full, oz + uz * full),  # (u=1, v=0)
            (ox + ux * full + vx * full, oy + uy * full + vy * full, oz + uz * full + vz * full),  # (u=1, v=1)
            (ox + vx * full, oy + vy * full, oz + vz * full),  # (u=0, v=1)
        ]

    def sticker_corners_3d(self, face: str, row: int, col: int) -> List[Tuple[float, float, float]]:
        """Return the 4 corners of one sticker in world coords, accounting
        for bezel and inter-sticker inset. (row, col) are 0..2."""
        s = self.size / 2.0
        # Face spans -s to +s in 2 of its 3 world coords. Convert (row, col)
        # to face-local (u, v) ∈ [0, 1]² with bezel inset.
        bezel = self.bezel_fraction
        sticker_span = (1.0 - 2 * bezel) / 3.0
        inset = self.sticker_inset_fraction * sticker_span
        u0 = bezel + col * sticker_span + inset
        u1 = bezel + (col + 1) * sticker_span - inset
        v0 = bezel + row * sticker_span + inset
        v1 = bezel + (row + 1) * sticker_span - inset

        # Reuse the face basis to map (u, v) to world coords
        FACE_BASIS = {
            "U": {"origin": (-s, s, -s), "u": (1, 0, 0), "v": (0, 0, 1)},
            "D": {"origin": (-s, -s, s), "u": (1, 0, 0), "v": (0, 0, -1)},
            "R": {"origin": (s, s, s), "u": (0, 0, -1), "v": (0, -1, 0)},
            "L": {"origin": (-s, s, -s), "u": (0, 0, 1), "v": (0, -1, 0)},
            "F": {"origin": (-s, s, s), "u": (1, 0, 0), "v": (0, -1, 0)},
            "B": {"origin": (s, s, -s), "u": (-1, 0, 0), "v": (0, -1, 0)},
        }
        basis = FACE_BASIS[face]
        ox, oy, oz = basis["origin"]
        ux, uy, uz = basis["u"]
        vx, vy, vz = basis["v"]
        full = self.size
        return [
            (ox + ux * full * u + vx * full * v, oy + uy * full * u + vy * full * v, oz + uz * full * u + vz * full * v)
            for (u, v) in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
        ]


# ---------------- camera + projection ----------------


@dataclass
class Camera:
    """Simple perspective camera looking at the origin."""
    position: Tuple[float, float, float] = (3.0, 2.5, 3.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    focal_length: float = 600.0  # in pixels
    image_size: Tuple[int, int] = (862, 1150)  # portrait, matches iPhone EXIF-rotated

    def _basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos = np.array(self.position, dtype=np.float64)
        target = np.array(self.target, dtype=np.float64)
        up = np.array(self.up, dtype=np.float64)
        forward = target - pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        return pos, forward, right, true_up

    def project(self, point: Sequence[float]) -> Tuple[float, float]:
        """Project a world-space point to image pixel coords (x_img, y_img)."""
        pos, forward, right, true_up = self._basis()
        p = np.array(point, dtype=np.float64) - pos
        # Camera-space coordinates: x_cam = p·right, y_cam = p·true_up, z_cam = p·forward
        x_cam = float(np.dot(p, right))
        y_cam = float(np.dot(p, true_up))
        z_cam = float(np.dot(p, forward))
        if z_cam <= 1e-6:
            # Point behind camera or at it — clamp to a tiny positive depth
            z_cam = 1e-6
        x_img = self.focal_length * x_cam / z_cam + self.image_size[0] / 2.0
        # image y grows down; camera y grows up — flip
        y_img = self.image_size[1] / 2.0 - self.focal_length * y_cam / z_cam
        return x_img, y_img

    def project_many(self, points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
        return [self.project(p) for p in points]


# ---------------- which faces are visible from which camera ----------------


VIEW_VISIBLE_FACES = {
    "A": ("U", "R", "F"),  # standard isometric: top-right-front corner toward camera
    "B": ("D", "L", "B"),  # post-flip: bottom-left-back corner toward camera (cube rotated 180° around (1,1,0) diag)
}


def camera_for_view(view: str, image_size: Tuple[int, int]) -> Camera:
    """Pick a camera position that puts the right 3 faces toward the camera.

    For view A we look from (+x, +y, +z) octant → see +X (R), +Y (U), +Z (F).
    For view B we look from (-x, -y, -z) octant → see -X (L), -Y (D), -Z (B)."""
    if view == "A":
        return Camera(position=(3.0, 2.5, 3.0), target=(0, 0, 0), up=(0, 1, 0), image_size=image_size)
    if view == "B":
        # Camera in the opposite octant. The "up" vector still points to image-top;
        # for image B the world-up is now D (since D was on bottom and is now visible
        # from this angle, but the LABELER calls it "image B" with D at the top of the
        # image). So we set up=(0, -1, 0) so that the cube's D face appears at the top
        # of the rendered image.
        return Camera(position=(-3.0, -2.5, -3.0), target=(0, 0, 0), up=(0, -1, 0), image_size=image_size)
    raise ValueError(f"unknown view {view!r}")


# ---------------- state-string handling ----------------


def parse_state(state: str) -> Dict[str, List[List[str]]]:
    """Parse a 54-char URFDLB state string into a per-face 3×3 grid of face-color letters."""
    if len(state) != 54:
        raise ValueError(f"state must be 54 chars; got {len(state)}")
    out: Dict[str, List[List[str]]] = {}
    for i, face in enumerate(FACE_ORDER):
        chunk = state[i * 9:(i + 1) * 9]
        out[face] = [list(chunk[r * 3:(r + 1) * 3]) for r in range(3)]
    return out


# ---------------- per-face lighting (flat shading) ----------------


def shade_color(rgb: Tuple[int, int, int], intensity: float) -> Tuple[int, int, int]:
    """Multiplicative brightness scale, clamped."""
    return tuple(max(0, min(255, int(round(c * intensity)))) for c in rgb)  # type: ignore


# Approximate Lambertian shading — for the standard view, U is brightest
# (faces a top light), R and F are mid-bright, D is dimmest. Tune per view.
SHADING_BY_VIEW_FACE = {
    "A": {"U": 1.00, "R": 0.85, "F": 0.92, "D": 0.50, "L": 0.55, "B": 0.55},
    "B": {"D": 1.00, "L": 0.85, "B": 0.92, "U": 0.50, "R": 0.55, "F": 0.55},
}


# ---------------- rendering ----------------


@dataclass
class RenderResult:
    image: Image.Image
    cube_hull: List[Tuple[float, float]] = field(default_factory=list)
    face_quads: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    sticker_positions: Dict[str, List[List[Tuple[float, float]]]] = field(default_factory=dict)


def render_view(
    state: str,
    view: str,
    image_size: Tuple[int, int] = (862, 1150),
    background_rgb: Tuple[int, int, int] = (200, 200, 200),
) -> RenderResult:
    cube = CubeModel()
    cam = camera_for_view(view, image_size)
    parsed = parse_state(state)
    visible_faces = VIEW_VISIBLE_FACES[view]

    canvas = Image.new("RGB", image_size, background_rgb)
    draw = ImageDraw.Draw(canvas)

    face_quads: Dict[str, List[Tuple[float, float]]] = {}
    sticker_positions: Dict[str, List[List[Tuple[float, float]]]] = {}
    cube_hull_pts: List[Tuple[float, float]] = []

    for face in visible_faces:
        # Draw the face's bezel base first (the dark cube body behind the
        # stickers). This is the full 4-corner face polygon.
        face_3d_corners = cube.face_corners_3d(face)
        face_2d_corners = cam.project_many(face_3d_corners)
        draw.polygon(face_2d_corners, fill=BEZEL_RGB)
        face_quads[face] = [(round(x, 2), round(y, 2)) for (x, y) in face_2d_corners]
        cube_hull_pts.extend(face_2d_corners)

        # Then draw each of the 9 stickers
        face_grid = parsed[face]
        sticker_positions[face] = []
        for row in range(3):
            row_centers: List[Tuple[float, float]] = []
            for col in range(3):
                color_letter = face_grid[row][col]
                base_rgb = CUBE_COLORS_RGB[color_letter]
                shaded = shade_color(base_rgb, SHADING_BY_VIEW_FACE[view][face])
                sticker_3d = cube.sticker_corners_3d(face, row, col)
                sticker_2d = cam.project_many(sticker_3d)
                draw.polygon(sticker_2d, fill=shaded)
                # Record the sticker center (centroid of its 4 image-space corners)
                cx = sum(p[0] for p in sticker_2d) / 4.0
                cy = sum(p[1] for p in sticker_2d) / 4.0
                row_centers.append((round(cx, 2), round(cy, 2)))
            sticker_positions[face].append(row_centers)

    # Cube hull = convex hull of all visible face corners
    try:
        from scipy.spatial import ConvexHull

        pts = np.array(cube_hull_pts)
        hull_indices = ConvexHull(pts).vertices
        cube_hull = [tuple(pts[i]) for i in hull_indices]
    except Exception:
        # scipy might not be available; fall back to monotone-chain
        cube_hull = _convex_hull(cube_hull_pts)

    return RenderResult(
        image=canvas,
        cube_hull=[(round(x, 2), round(y, 2)) for (x, y) in cube_hull],
        face_quads=face_quads,
        sticker_positions=sticker_positions,
    )


def _convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Monotone-chain convex hull (fallback if scipy not present)."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts
    def cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def render_pair(state: str, image_size: Tuple[int, int] = (862, 1150),
                background_rgb: Tuple[int, int, int] = (200, 200, 200)) -> Tuple[RenderResult, RenderResult]:
    return (render_view(state, "A", image_size, background_rgb),
            render_view(state, "B", image_size, background_rgb))


# ---------------- CLI ----------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True, help="54-char URFDLB cube state")
    ap.add_argument("--output", required=True, help="output prefix (writes <prefix>_A.png, <prefix>_B.png, <prefix>.json)")
    ap.add_argument("--width", type=int, default=862)
    ap.add_argument("--height", type=int, default=1150)
    ap.add_argument("--background", default="200,200,200", help="comma-separated R,G,B")
    args = ap.parse_args()

    bg = tuple(int(c) for c in args.background.split(","))
    res_a, res_b = render_pair(args.state, (args.width, args.height), bg)

    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out_a = prefix.with_name(prefix.name + "_A.png")
    out_b = prefix.with_name(prefix.name + "_B.png")
    out_meta = prefix.with_name(prefix.name + ".json")
    res_a.image.save(out_a, "PNG", optimize=True)
    res_b.image.save(out_b, "PNG", optimize=True)

    meta = {
        "state": args.state,
        "imageSize": [args.width, args.height],
        "background": list(bg),
        "views": {
            "A": {
                "imagePath": str(out_a),
                "visibleFaces": list(VIEW_VISIBLE_FACES["A"]),
                "cubeHull": res_a.cube_hull,
                "faceQuads": res_a.face_quads,
                "stickerPositions": res_a.sticker_positions,
            },
            "B": {
                "imagePath": str(out_b),
                "visibleFaces": list(VIEW_VISIBLE_FACES["B"]),
                "cubeHull": res_b.cube_hull,
                "faceQuads": res_b.face_quads,
                "stickerPositions": res_b.sticker_positions,
            },
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    print(f"wrote {out_a}", file=sys.stderr)
    print(f"wrote {out_b}", file=sys.stderr)
    print(f"wrote {out_meta}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
