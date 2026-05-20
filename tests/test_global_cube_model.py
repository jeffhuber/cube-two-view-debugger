"""Smoke tests for tools/global_cube_model.

Same scipy-optional pattern as the bezel detection module:
1. The module imports without scipy (numpy only)
2. `fit_global_cube_model` returns init-only model (graceful) when
   scipy is missing
3. When scipy IS available, the fitter runs end-to-end on a synthetic
   mask without crashing
4. Geometry derivation produces 7 visible corners + 3 face quads +
   27 sticker cells from valid 6-DOF parameters
"""
from __future__ import annotations

import importlib
import math

import numpy as np
import pytest


def test_module_imports():
    mod = importlib.import_module("tools.global_cube_model")
    assert hasattr(mod, "fit_global_cube_model")
    assert hasattr(mod, "GlobalCubeModel")
    assert hasattr(mod, "derive_geometry")


def test_derive_geometry_produces_7_corners_3_faces_27_cells():
    """Verify geometry derivation from valid iso-projection-like
    parameters produces the expected 7 visible corners, 3 face quads,
    and 27 cells (9 per face)."""
    from tools.global_cube_model import GlobalCubeModel, derive_geometry

    # Iso-projection-like angles: 3 axes 120° apart in [0, 2π)
    m = GlobalCubeModel(
        cube_center=(1500.0, 1500.0),
        axis_angles_rad=(
            math.radians(90),    # DOWN
            math.radians(210),   # UP-LEFT
            math.radians(330),   # UP-RIGHT
        ),
        edge_length_px=500.0,
    )
    derive_geometry(m)

    # 7 visible corners
    assert set(m.visible_corners.keys()) == {
        "111", "011", "101", "110", "001", "010", "100"
    }

    # 3 face quads
    assert set(m.face_quads.keys()) == {"face_01", "face_12", "face_02"}
    for quad in m.face_quads.values():
        assert len(quad) == 4

    # 27 sticker cells (9 per face)
    assert set(m.sticker_cells.keys()) == {"face_01", "face_12", "face_02"}
    for cells in m.sticker_cells.values():
        assert len(cells) == 9
        for cell in cells:
            assert len(cell) == 4

    # Cube center is corner 111
    assert m.visible_corners["111"] == (1500.0, 1500.0)


def test_fit_returns_init_only_when_no_scipy(monkeypatch):
    """When scipy is unavailable, `fit_global_cube_model` should still
    return a model from initialization (with debug["error"] flag) rather
    than crashing."""
    from tools import global_cube_model as mod
    from tools.interior_bezel_detection import InteriorBezelDetection

    # Force the bezel module's scipy gate to None — affects both the
    # bezel detection (which the fitter doesn't call directly here)
    # AND the silhouette erosion (which the fitter doesn't use directly).
    # The fitter's own scipy dependency is via scipy.optimize.minimize
    # which it imports lazily and catches ImportError on.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    import builtins
    real_import_fn = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy.optimize" or name.startswith("scipy.optimize"):
            raise ImportError("mocked: no scipy.optimize")
        return real_import_fn(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Build a synthetic detection
    detection = InteriorBezelDetection(
        cube_center=(100.0, 100.0),
        boundary_angles=[math.radians(90), math.radians(210), math.radians(330)],
        line_qualities=[0.9, 0.8, 0.7],
    )
    mask = np.zeros((300, 300), dtype=bool)
    mask[40:260, 40:260] = True

    model = mod.fit_global_cube_model(detection, mask, optimize=True)
    assert model is not None, "should return init-only model, not None"
    # The init succeeded; the optimization fallback ran
    assert "error" in model.debug
    assert "scipy" in model.debug["error"].lower()


def test_fit_runs_on_synthetic_iso_silhouette():
    """End-to-end smoke: build a synthetic hexagonal silhouette and a
    detection that roughly matches, verify the fitter runs and produces
    a model with non-trivial output."""
    from tools.global_cube_model import fit_global_cube_model
    from tools.interior_bezel_detection import (
        InteriorBezelDetection, _try_import_scipy_ndimage,
    )

    if _try_import_scipy_ndimage() is None:
        pytest.skip("scipy not installed")

    # Build a hexagonal mask
    from PIL import Image, ImageDraw
    h_size = 600
    img = Image.new("L", (h_size, h_size), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = h_size / 2, h_size / 2
    L = 150
    hexagon = [
        (cx - L * math.cos(math.radians(30)), cy - L * math.sin(math.radians(30))),
        (cx, cy - L),
        (cx + L * math.cos(math.radians(30)), cy - L * math.sin(math.radians(30))),
        (cx + L * math.cos(math.radians(30)), cy + L * math.sin(math.radians(30))),
        (cx, cy + L),
        (cx - L * math.cos(math.radians(30)), cy + L * math.sin(math.radians(30))),
    ]
    draw.polygon(hexagon, fill=255)
    mask = np.array(img) > 0

    detection = InteriorBezelDetection(
        cube_center=(cx, cy),
        boundary_angles=[
            math.radians(90),
            math.radians(210),
            math.radians(330),
        ],
        line_qualities=[1.0, 1.0, 1.0],
    )

    model = fit_global_cube_model(detection, mask, optimize=True)
    assert model is not None
    assert len(model.visible_corners) == 7
    assert len(model.face_quads) == 3
    assert sum(len(cs) for cs in model.sticker_cells.values()) == 27
    # On a clean synthetic hexagon the fit should be quite strong
    assert model.fit_quality >= 0.5, f"fit_quality {model.fit_quality} too low on clean synthetic"
