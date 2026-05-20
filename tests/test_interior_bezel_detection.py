"""Smoke tests for tools/interior_bezel_detection.

scipy is an optional research dependency in this repo (not in
requirements.txt), so these tests verify:

1. The module imports without scipy (numpy only)
2. `detect_interior_bezel_lines` returns a graceful error result rather
   than raising ModuleNotFoundError when scipy is missing
3. When scipy IS available, the detector runs on a synthetic mask
   without crashing
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_module_imports_without_scipy():
    """Importing the module must succeed even in a venv without scipy."""
    mod = importlib.import_module("tools.interior_bezel_detection")
    assert mod is not None
    assert hasattr(mod, "detect_interior_bezel_lines")
    assert hasattr(mod, "InteriorBezelDetection")


def test_detect_returns_graceful_error_when_scipy_missing(monkeypatch):
    """Force the scipy-availability gate to report missing and verify
    the detector returns an InteriorBezelDetection with a useful
    `debug['error']` rather than raising."""
    from tools import interior_bezel_detection as mod

    # Reset the import cache then force it to "missing"
    if hasattr(mod._try_import_scipy_ndimage, "_cached"):
        delattr(mod._try_import_scipy_ndimage, "_cached")
    monkeypatch.setattr(mod, "_try_import_scipy_ndimage", lambda: None)

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True

    det = mod.detect_interior_bezel_lines(rgb, mask)
    assert det.signal_quality == 0.0
    assert det.cube_center is None
    assert "scipy" in det.debug.get("error", "").lower()


def test_detect_returns_graceful_error_on_shape_mismatch():
    """Shape-mismatch path must return a detection with a clear error."""
    from tools import interior_bezel_detection as mod

    # Only exercise this if scipy IS available — otherwise the scipy
    # gate fires first and obscures the shape check
    if mod._try_import_scipy_ndimage() is None:
        pytest.skip("scipy not installed in this venv")

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)  # wrong shape

    det = mod.detect_interior_bezel_lines(rgb, mask)
    assert det.signal_quality == 0.0
    assert "shape mismatch" in det.debug.get("error", "").lower()


def test_detect_runs_on_synthetic_mask_when_scipy_available():
    """When scipy IS available, the detector runs end-to-end on a
    synthetic square mask without crashing. We don't assert on
    quality — just that the dataclass shape is well-formed."""
    from tools import interior_bezel_detection as mod

    if mod._try_import_scipy_ndimage() is None:
        pytest.skip("scipy not installed in this venv")

    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=bool)
    mask[40:160, 40:160] = True

    det = mod.detect_interior_bezel_lines(rgb, mask)
    # Either a "eroded mask too small" / "no signal" graceful result,
    # or a real detection with the dataclass fields present.
    assert isinstance(det, mod.InteriorBezelDetection)
    assert isinstance(det.boundary_lines, list)
    assert isinstance(det.boundary_angles, list)
    assert 0.0 <= det.signal_quality <= 1.0


def test_cell_line_diagnostics_high_quality_threshold_flag():
    """The derived `crosses_high_quality_bezel` flag fires only when a
    line is BOTH high-quality AND close to the cell centroid AND
    geometrically crosses the cell. Verify the gates compose correctly
    so future threshold-tuning doesn't silently drop one of them."""
    from tools.interior_bezel_detection import (
        InteriorBezelDetection,
        cell_line_diagnostics,
    )

    # Three lines through (50, 50): vertical (q=0.95), horizontal
    # (q=0.50), and a diagonal (q=0.10).
    detection = InteriorBezelDetection(
        cube_center=(50.0, 50.0),
        boundary_lines=[
            ((50.0, 50.0), (50.0, 0.0)),
            ((50.0, 50.0), (100.0, 50.0)),
            ((50.0, 50.0), (0.0, 100.0)),
        ],
        boundary_angles=[1.5708, 0.0, 2.3562],
        line_equations=[
            (-1.0, 0.0, 50.0),                # x = 50
            (0.0, 1.0, -50.0),                # y = 50
            (-0.7071, -0.7071, 70.71),        # x + y = 100
        ],
        line_qualities=[0.95, 0.50, 0.10],
        signal_quality=0.5,
    )

    # Quad containing (50, 50) — vertical + horizontal (both >= 0.40)
    # cross it. HQ flag should fire.
    quad_contains = [(40, 40), (60, 40), (60, 60), (40, 60)]
    d = cell_line_diagnostics(detection, quad_contains,
                              high_quality_threshold=0.40,
                              max_distance_px=30.0)
    assert d["crosses_high_quality_bezel"] is True
    assert d["thresholds"]["line_quality"] == 0.40
    assert d["thresholds"]["distance_px"] == 30.0
    assert d["detector_version"] == "iterative-v1"

    # Quad far from any line — no crossings at all.
    quad_far = [(200, 200), (220, 200), (220, 220), (200, 220)]
    d_far = cell_line_diagnostics(detection, quad_far)
    assert d_far["any_crossing"] is False
    assert d_far["crosses_high_quality_bezel"] is False

    # Quad crossed ONLY by the low-quality diagonal (q=0.10 < 0.40).
    # `any_crossing` should fire; `crosses_high_quality_bezel` should
    # NOT (line quality below threshold).
    quad_low_only = [(0, 80), (20, 80), (20, 100), (0, 100)]
    d_low = cell_line_diagnostics(detection, quad_low_only,
                                  high_quality_threshold=0.40,
                                  max_distance_px=30.0)
    assert d_low["any_crossing"] is True
    assert d_low["crosses_high_quality_bezel"] is False, (
        "low-quality crossing must not trip the HQ-gated flag"
    )


def test_find_interior_h_vertices_filters_by_line_quality():
    """`find_interior_h_vertices` only traces bezels whose
    `line_qualities[i] >= min_line_quality`. Verify that bezels below
    the threshold are skipped (no candidate emitted for them)."""
    from tools.interior_bezel_detection import (
        find_interior_h_vertices, _try_import_scipy_ndimage,
    )

    if _try_import_scipy_ndimage() is None:
        pytest.skip("scipy not installed in this venv")

    # Build a synthetic image with no real bezels — detector will
    # produce a graceful empty/low-signal result, and h-vertex tracing
    # should return [] (nothing to trace).
    from tools.interior_bezel_detection import detect_interior_bezel_lines
    rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    mask = np.zeros((300, 300), dtype=bool)
    mask[40:260, 40:260] = True

    det = detect_interior_bezel_lines(rgb, mask)
    vertices = find_interior_h_vertices(
        det, rgb, mask, min_line_quality=0.40
    )
    # No real bezels in this synthetic image; either no detection lines
    # at all (filtered out) or detection lines with low quality (filtered
    # out by min_line_quality threshold). Either way: empty result.
    assert isinstance(vertices, list)
    for v in vertices:
        # Any vertex that DOES get emitted must have a parent line
        # quality >= the threshold we passed
        parent_q = det.line_qualities[v.parent_line_index]
        assert parent_q >= 0.40 - 1e-9, (
            f"h-vertex emitted for parent quality {parent_q} "
            f"below threshold 0.40"
        )
