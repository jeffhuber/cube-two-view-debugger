"""Smoke + correctness tests for the synthetic cube renderer."""
from __future__ import annotations

import sys
import builtins
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from tools.render_synthetic_cube import (  # noqa: E402
    CUBE_COLORS_RGB,
    CubeModel,
    FACE_ORDER,
    VIEW_VISIBLE_FACES,
    camera_for_view,
    parse_state,
    render_pair,
    render_view,
)

SOLVED = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9


# ---- parse_state ----


def test_parse_state_solved():
    parsed = parse_state(SOLVED)
    for face in FACE_ORDER:
        for row in range(3):
            for col in range(3):
                assert parsed[face][row][col] == face


def test_parse_state_rejects_wrong_length():
    with pytest.raises(ValueError):
        parse_state("U" * 53)


def test_parse_state_set15_centers_are_canonical():
    """Set 15 corpus GT has canonical face centers (position 4 of each chunk)."""
    state = "DRULUDDUFLFRDRRFDFLBUFFLBUDULRUDBLRRRBFFLUBRLBDBBBLUFD"
    parsed = parse_state(state)
    for face in FACE_ORDER:
        assert parsed[face][1][1] == face, f"face {face} center mismatch"


# ---- 3D cube model ----


def test_cube_face_corners_are_unit_face():
    cube = CubeModel(size=1.0)
    # All 6 faces should have 4 corners spanning the full size in 2 of 3 dims
    for face in FACE_ORDER:
        corners = cube.face_corners_3d(face)
        assert len(corners) == 4
        # Each corner should be on the cube's surface (|x|=0.5 or |y|=0.5 or |z|=0.5)
        for (x, y, z) in corners:
            on_surface = (
                abs(abs(x) - 0.5) < 1e-6 or
                abs(abs(y) - 0.5) < 1e-6 or
                abs(abs(z) - 0.5) < 1e-6
            )
            assert on_surface, f"face {face} corner {(x,y,z)} not on cube surface"


def test_sticker_corners_inside_face():
    cube = CubeModel(size=1.0, bezel_fraction=0.04, sticker_inset_fraction=0.04)
    for face in FACE_ORDER:
        face_corners = cube.face_corners_3d(face)
        for row in range(3):
            for col in range(3):
                sticker = cube.sticker_corners_3d(face, row, col)
                assert len(sticker) == 4
                # Sticker corners should all be on the face's plane
                # (one of x/y/z fixed at ±0.5 matching the face)
                ref_axis_value = {"U": ("y", 0.5), "D": ("y", -0.5),
                                  "R": ("x", 0.5), "L": ("x", -0.5),
                                  "F": ("z", 0.5), "B": ("z", -0.5)}[face]
                axis_idx = {"x": 0, "y": 1, "z": 2}[ref_axis_value[0]]
                for pt in sticker:
                    assert abs(pt[axis_idx] - ref_axis_value[1]) < 1e-6


# ---- camera/projection ----


def test_camera_view_a_sees_urf():
    """For view A camera, the center of each visible face should project
    to a point within the image bounds."""
    cube = CubeModel()
    cam = camera_for_view("A", (862, 1150))
    for face in VIEW_VISIBLE_FACES["A"]:
        # Sticker (1, 1) is the face center
        sticker = cube.sticker_corners_3d(face, 1, 1)
        center_3d = tuple(sum(c[i] for c in sticker) / 4.0 for i in range(3))
        x, y = cam.project(center_3d)
        assert 0 <= x <= 862, f"face {face} center projects outside x: {x}"
        assert 0 <= y <= 1150, f"face {face} center projects outside y: {y}"


def test_camera_view_b_sees_dlb():
    cube = CubeModel()
    cam = camera_for_view("B", (862, 1150))
    for face in VIEW_VISIBLE_FACES["B"]:
        sticker = cube.sticker_corners_3d(face, 1, 1)
        center_3d = tuple(sum(c[i] for c in sticker) / 4.0 for i in range(3))
        x, y = cam.project(center_3d)
        assert 0 <= x <= 862, f"face {face} center projects outside x: {x}"
        assert 0 <= y <= 1150, f"face {face} center projects outside y: {y}"


# ---- render output ----


def test_render_solved_produces_image_and_metadata():
    res = render_view(SOLVED, "A", image_size=(400, 533))
    assert res.image.size == (400, 533)
    assert len(res.cube_hull) >= 4
    assert set(res.face_quads.keys()) == set(VIEW_VISIBLE_FACES["A"])
    for face, quad in res.face_quads.items():
        assert len(quad) == 4
    for face, stickers in res.sticker_positions.items():
        assert len(stickers) == 3
        assert all(len(row) == 3 for row in stickers)


def test_render_uses_convex_hull_fallback_without_scipy(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("blocked scipy for fallback test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    res = render_view(SOLVED, "A", image_size=(400, 533))

    assert len(res.cube_hull) >= 4


def test_render_pair_produces_distinct_a_and_b():
    res_a, res_b = render_pair(SOLVED, image_size=(400, 533))
    # Hulls should have similar bounding area (same cube) but different shapes
    # because the visible-face composition differs after the flip.
    assert set(res_a.face_quads.keys()) == set(VIEW_VISIBLE_FACES["A"])
    assert set(res_b.face_quads.keys()) == set(VIEW_VISIBLE_FACES["B"])


def test_render_solved_top_sticker_color_matches_face():
    """The center pixel of the U face on view A should be near the U-color
    (white) — within shading tolerance."""
    res = render_view(SOLVED, "A", image_size=(800, 800))
    u_center = res.sticker_positions["U"][1][1]
    # Sample the image at that pixel
    px = res.image.getpixel((round(u_center[0]), round(u_center[1])))
    # U color is (238, 238, 232); after shading (×1.00 for view A's U) still bright
    assert px[0] > 200 and px[1] > 200 and px[2] > 200, f"U center is {px}, expected near-white"


def test_render_solved_view_b_top_is_yellow_d_face():
    """View B should show D (yellow) at the top."""
    res = render_view(SOLVED, "B", image_size=(800, 800))
    d_center = res.sticker_positions["D"][1][1]
    px = res.image.getpixel((round(d_center[0]), round(d_center[1])))
    # D color (230, 210, 42); yellow = high R+G, low B
    assert px[0] > 150 and px[1] > 150 and px[2] < 100, f"D center is {px}, expected yellow"
