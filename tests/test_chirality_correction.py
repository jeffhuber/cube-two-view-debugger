"""Unit tests for chirality detection + correction in global_cube_model.

The chirality check uses per-corner line-darkness sampling: for each of
the 6 hexagon corners around the vertex, measure pixel darkness along
the line from vertex toward that corner. The 3 NEAR corners (cube edges
= dark bezels) should show HIGHER darkness than the 3 FAR corners
(face diagonals across colored stickers).
"""
import math

import numpy as np

from tools.global_cube_model import (
    GlobalCubeModel,
    _apply_chirality_correction,
    _line_darkness_from_vertex,
    _signed_angle_diff_deg,
    derive_geometry,
)
from tools.interior_bezel_detection import InteriorBezelDetection


def _solved_iso_model(edge_px: float = 100.0) -> GlobalCubeModel:
    """Synthetic chirality-correct iso projection: near corners at
    image-angle 30°, 270°, 150° (3 directions ~120° apart).
    """
    m = GlobalCubeModel(
        cube_center_screen=(500.0, 500.0),
        axis_x_2d=(edge_px * math.cos(math.radians(30)), edge_px * math.sin(math.radians(30))),
        axis_y_2d=(edge_px * math.cos(math.radians(270)), edge_px * math.sin(math.radians(270))),
        axis_z_2d=(edge_px * math.cos(math.radians(150)), edge_px * math.sin(math.radians(150))),
    )
    derive_geometry(m)
    return m


def _image_with_dark_lines(angles_rad, size: int = 1000, r_max: int = 110) -> np.ndarray:
    """Render an image with dark half-lines from center along the given
    oriented angles (going outward to the near-corner positions).

    Background is light grey (200); lines are near-black (10).
    Lines extend only up to r_max so we don't overshoot far corners.
    """
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    cx = cy = size // 2
    for ang in angles_rad:
        for r in range(8, r_max):
            x = int(cx + r * math.cos(ang))
            y = int(cy + r * math.sin(ang))
            if 0 <= x < size and 0 <= y < size:
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        xi, yi = x + dx, y + dy
                        if 0 <= xi < size and 0 <= yi < size:
                            img[yi, xi] = (10, 10, 10)
    return img


def test_signed_angle_diff_deg_basic():
    assert abs(_signed_angle_diff_deg(0, 0)) < 1e-6
    assert abs(_signed_angle_diff_deg(0, math.pi / 2) - 90.0) < 1e-6
    assert abs(_signed_angle_diff_deg(0, math.pi) - 180.0) < 1e-6


def test_line_darkness_distinguishes_bezel_vs_sticker():
    """Line through dark bezel has high darkness; line through bg has low."""
    img = _image_with_dark_lines([0.0])  # bezel going right at angle 0
    center = (500.0, 500.0)
    # Target along the bezel direction (right)
    dark_along = _line_darkness_from_vertex(img, center, (600.0, 500.0))
    # Target perpendicular (no bezel along that line)
    dark_perp = _line_darkness_from_vertex(img, center, (500.0, 600.0))
    assert dark_along > dark_perp + 50, (dark_along, dark_perp)


def test_chirality_check_no_image_skips():
    model = _solved_iso_model()
    detection = InteriorBezelDetection(
        cube_center=(500.0, 500.0),
        boundary_angles=[math.radians(30), math.radians(270), math.radians(150)],
        line_qualities=[0.9, 0.9, 0.9],
    )
    _, debug = _apply_chirality_correction(model, detection, image_rgb=None)
    assert debug["chirality_check"] == "skipped_no_image"


def test_chirality_correct_when_near_corners_have_dark_lines():
    """Bezels run along the 3 NEAR corner directions of the model.
    Per-corner darkness for near corners > far corners → correct."""
    near_angles = [math.radians(30), math.radians(270), math.radians(150)]
    img = _image_with_dark_lines(near_angles, r_max=110)
    model = _solved_iso_model(edge_px=100.0)
    detection = InteriorBezelDetection(
        cube_center=(500.0, 500.0),
        boundary_angles=[math.radians(30), math.radians(90), math.radians(150)],
        line_qualities=[0.9, 0.9, 0.9],
    )
    corrected, debug = _apply_chirality_correction(model, detection, img)
    assert debug["chirality_check"] == "correct", debug
    assert debug["chirality_mean_near_darkness"] > debug["chirality_mean_far_darkness"]
    # Axes unchanged
    assert corrected.axis_x_2d == model.axis_x_2d
    assert corrected.axis_y_2d == model.axis_y_2d
    assert corrected.axis_z_2d == model.axis_z_2d


def test_chirality_flip_suggested_when_60deg_off_diagnostic_only():
    """Dark bezels point at the TRUE near directions (30°, 270°, 150°)
    but the model's labeled near corners are at the 60°-flipped positions
    (90°, 330°, 210°). Default behavior is diagnostic-only — the debug
    flag reports flip_suggested but the model is NOT swapped."""
    true_near_angles = [math.radians(30), math.radians(270), math.radians(150)]
    img = _image_with_dark_lines(true_near_angles, r_max=110)

    edge = 100.0
    m = GlobalCubeModel(
        cube_center_screen=(500.0, 500.0),
        axis_x_2d=(edge * math.cos(math.radians(90)), edge * math.sin(math.radians(90))),
        axis_y_2d=(edge * math.cos(math.radians(330)), edge * math.sin(math.radians(330))),
        axis_z_2d=(edge * math.cos(math.radians(210)), edge * math.sin(math.radians(210))),
    )
    derive_geometry(m)

    detection = InteriorBezelDetection(
        cube_center=(500.0, 500.0),
        boundary_angles=[math.radians(30), math.radians(90), math.radians(150)],
        line_qualities=[0.9, 0.9, 0.9],
    )
    result, debug = _apply_chirality_correction(m, detection, img)
    assert debug["chirality_check"] == "flip_suggested_diagnostic_only", debug
    assert debug["chirality_mean_near_darkness"] < debug["chirality_mean_far_darkness"]
    # Model is NOT swapped by default
    assert result.axis_x_2d == m.axis_x_2d
    assert result.axis_y_2d == m.axis_y_2d
    assert result.axis_z_2d == m.axis_z_2d


def test_chirality_corrected_when_apply_correction_true():
    """With apply_correction=True the same setup actually swaps axes."""
    true_near_angles = [math.radians(30), math.radians(270), math.radians(150)]
    img = _image_with_dark_lines(true_near_angles, r_max=110)

    edge = 100.0
    m = GlobalCubeModel(
        cube_center_screen=(500.0, 500.0),
        axis_x_2d=(edge * math.cos(math.radians(90)), edge * math.sin(math.radians(90))),
        axis_y_2d=(edge * math.cos(math.radians(330)), edge * math.sin(math.radians(330))),
        axis_z_2d=(edge * math.cos(math.radians(210)), edge * math.sin(math.radians(210))),
    )
    derive_geometry(m)

    detection = InteriorBezelDetection(
        cube_center=(500.0, 500.0),
        boundary_angles=[math.radians(30), math.radians(90), math.radians(150)],
        line_qualities=[0.9, 0.9, 0.9],
    )
    corrected, debug = _apply_chirality_correction(
        m, detection, img, apply_correction=True
    )
    assert debug["chirality_check"] == "corrected_60deg_flip", debug
    assert corrected.axis_x_2d != m.axis_x_2d
    err_after = sum(debug["chirality_axis_angle_errors_after_deg"])
    err_before = sum(debug["chirality_axis_angle_errors_before_deg"])
    assert err_after < err_before


def test_chirality_corrected_model_has_consistent_geometry():
    """After correction (apply_correction=True), derived geometry is
    well-formed (7 corners, 3 face quads, 27 sticker cells)."""
    true_near_angles = [math.radians(30), math.radians(270), math.radians(150)]
    img = _image_with_dark_lines(true_near_angles, r_max=110)
    edge = 100.0
    m = GlobalCubeModel(
        cube_center_screen=(500.0, 500.0),
        axis_x_2d=(edge * math.cos(math.radians(90)), edge * math.sin(math.radians(90))),
        axis_y_2d=(edge * math.cos(math.radians(330)), edge * math.sin(math.radians(330))),
        axis_z_2d=(edge * math.cos(math.radians(210)), edge * math.sin(math.radians(210))),
    )
    derive_geometry(m)
    detection = InteriorBezelDetection(
        cube_center=(500.0, 500.0),
        boundary_angles=[math.radians(30), math.radians(90), math.radians(150)],
        line_qualities=[0.9, 0.9, 0.9],
    )
    corrected, debug = _apply_chirality_correction(
        m, detection, img, apply_correction=True
    )
    assert debug["chirality_check"] == "corrected_60deg_flip"
    expected = {"front", "h_x", "h_y", "h_z", "h_xy", "h_xz", "h_yz"}
    assert set(corrected.visible_corners.keys()) == expected
    assert len(corrected.face_quads) == 3
    assert sum(len(c) for c in corrected.sticker_cells.values()) == 27


def test_chirality_ambiguous_on_flat_image():
    """No dark lines anywhere → near/far darkness roughly equal → ambiguous."""
    flat = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    model = _solved_iso_model()
    detection = InteriorBezelDetection(
        cube_center=(500.0, 500.0),
        boundary_angles=[math.radians(30), math.radians(90), math.radians(150)],
        line_qualities=[0.5, 0.5, 0.5],
    )
    _, debug = _apply_chirality_correction(model, detection, flat)
    assert debug["chirality_check"] == "ambiguous_no_correction", debug
    assert abs(debug["chirality_darkness_separation"]) < 5.0
