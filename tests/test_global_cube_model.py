"""Smoke tests for tools/global_cube_model.

Same scipy-optional pattern as the bezel detection module:
1. The module imports without scipy (numpy only)
2. `fit_global_cube_model` returns init-only model (graceful) when
   scipy is missing
3. When scipy IS available, the fitter runs end-to-end on a synthetic
   mask without crashing
4. Geometry derivation produces 7 visible corners + 3 face quads +
   27 sticker cells from valid 6-DOF parameters
5. Ground-truth-fixture sanity check: the fixture file shape matches
   what tools/test_global_cube_model.py expects to compare against
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_module_imports():
    mod = importlib.import_module("tools.global_cube_model")
    assert hasattr(mod, "fit_global_cube_model")
    assert hasattr(mod, "GlobalCubeModel")
    assert hasattr(mod, "derive_geometry")


def test_derive_geometry_produces_7_corners_3_faces_27_cells():
    """Verify geometry derivation from an 8-DOF parameterization
    produces 7 visible corners, 3 face quads, 27 cells, AND that
    the 3 outer corners satisfy the parallelogram closure."""
    from tools.global_cube_model import GlobalCubeModel, derive_geometry

    # cube_center + 3 axis displacement vectors
    m = GlobalCubeModel(
        cube_center_screen=(1000.0, 1000.0),
        axis_x_2d=(0.0, 500.0),     # DOWN
        axis_y_2d=(-433.0, -250.0), # UP-LEFT
        axis_z_2d=(433.0, -250.0),  # UP-RIGHT
    )
    derive_geometry(m)

    expected_corners = {"front", "h_x", "h_y", "h_z", "h_xy", "h_xz", "h_yz"}
    assert set(m.visible_corners.keys()) == expected_corners
    expected_faces = {"face_yz", "face_xz", "face_xy"}
    assert set(m.face_quads.keys()) == expected_faces
    for quad in m.face_quads.values():
        assert len(quad) == 4
    assert set(m.sticker_cells.keys()) == expected_faces
    for cells in m.sticker_cells.values():
        assert len(cells) == 9
        for cell in cells:
            assert len(cell) == 4

    # cube_center vertex ("front") is at the screen offset
    assert m.visible_corners["front"] == (1000.0, 1000.0)
    # h_x = cube_center + axis_x_2d
    assert m.visible_corners["h_x"] == (1000.0, 1500.0)
    # h_xy = cube_center + axis_x_2d + axis_y_2d (parallelogram closure)
    assert m.visible_corners["h_xy"] == (1000.0 - 433.0, 1000.0 + 500.0 - 250.0)


def test_fit_returns_none_when_scipy_missing_for_hull(monkeypatch):
    """`fit_global_cube_model` needs scipy.spatial.ConvexHull for the
    silhouette → hull → 6 vertices step. When unavailable, the fit
    returns None (no fallback possible without convex hull)."""
    from tools import global_cube_model as mod
    from tools.interior_bezel_detection import InteriorBezelDetection

    import builtins
    real_import_fn = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy.spatial" or name.startswith("scipy.spatial"):
            raise ImportError("mocked: no scipy.spatial")
        return real_import_fn(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    detection = InteriorBezelDetection(
        cube_center=(100.0, 100.0),
        boundary_angles=[math.radians(90), math.radians(210), math.radians(330)],
        line_qualities=[0.9, 0.8, 0.7],
    )
    rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    mask = np.zeros((300, 300), dtype=bool)
    mask[40:260, 40:260] = True

    model = mod.fit_global_cube_model(detection, rgb, mask, optimize=True)
    # Without scipy.spatial we can't get the convex hull → can't get
    # 6 hexagon anchors → fit returns None
    assert model is None


def test_hull_candidate_points_use_boundary_not_full_interior():
    """Hull extraction should be deterministic and avoid feeding every
    interior mask pixel to ConvexHull."""
    from tools.global_cube_model import _deterministic_hull_candidate_points

    mask = np.zeros((300, 300), dtype=bool)
    mask[50:250, 70:230] = True

    pts = _deterministic_hull_candidate_points(mask)
    assert 0 < len(pts) < int(mask.sum())
    assert (70.0, 50.0) in {tuple(p) for p in pts}
    assert (229.0, 249.0) in {tuple(p) for p in pts}


def test_hull_from_mask_does_not_use_random_subsample(monkeypatch):
    """Large masks previously used np.random.choice before ConvexHull,
    which made downstream diagnostics run-to-run unstable. Pin that
    path to fail if random sampling returns."""
    pytest.importorskip("scipy.spatial")
    from tools.global_cube_model import _hull_from_mask

    def fail_if_called(*args, **kwargs):
        raise AssertionError("hull sampling must be deterministic")

    monkeypatch.setattr(np.random, "choice", fail_if_called)

    mask = np.zeros((600, 600), dtype=bool)
    mask[::2, ::2] = True

    np.random.seed(1)
    hull_1 = _hull_from_mask(mask)
    np.random.seed(999)
    hull_2 = _hull_from_mask(mask)

    assert hull_1
    assert hull_1 == hull_2


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

    rgb_synth = np.zeros((h_size, h_size, 3), dtype=np.uint8)
    model = fit_global_cube_model(detection, rgb_synth, mask, optimize=True)
    assert model is not None
    assert len(model.visible_corners) == 7
    assert len(model.face_quads) == 3
    assert sum(len(cs) for cs in model.sticker_cells.values()) == 27
    # On a clean synthetic hexagon the template fit should be quite good
    assert model.fit_quality >= 0.5, f"fit_quality {model.fit_quality} too low on clean synthetic hexagon"


def test_vertex_ground_truth_fixture_well_formed():
    """The vertex ground-truth fixture (collected via interactive gallery,
    2026-05-21) is the durable regression resource for the global cube
    model pipeline. This test checks the file is present and
    well-structured. Use the fixture against pipeline output to compute
    per-case vertex error and gate PR merges on regression."""
    fixture_p = REPO_ROOT / "tests" / "fixtures" / "gcm_vertex_ground_truth.json"
    assert fixture_p.exists(), (
        f"Ground-truth fixture missing at {fixture_p}. "
        "This file was collected via the interactive labeling gallery on "
        "the rembg+mean3+score-gated-refinement pipeline output. "
        "It contains user-marked true_vertex positions for 23 of 28 "
        "corpus cases plus binary correct/wrong judgment for all 28."
    )
    with fixture_p.open() as f:
        gt = json.load(f)
    # Schema check
    assert len(gt) >= 25, f"Expected ~28 cases, got {len(gt)}"
    for key, v in gt.items():
        assert "_" in key, f"Bad key shape: {key}"
        assert "center_correct" in v, f"{key} missing center_correct"
        assert isinstance(v["center_correct"], bool), f"{key} center_correct not bool"
        assert "current_vertex" in v, f"{key} missing current_vertex"
        cv = v["current_vertex"]
        assert isinstance(cv, list) and len(cv) == 2, f"{key} current_vertex shape"
        if "true_vertex" in v:
            tv = v["true_vertex"]
            assert isinstance(tv, list) and len(tv) == 2, f"{key} true_vertex shape"
            assert "error_px" in v, f"{key} has true_vertex but no error_px"
    # Headline numbers (anchors so we notice if fixture is replaced silently)
    with_truth = [v for v in gt.values() if "true_vertex" in v]
    assert len(with_truth) >= 20, f"Expected >=20 cases with true_vertex, got {len(with_truth)}"
    errs = [v["error_px"] for v in with_truth]
    median_err = sorted(errs)[len(errs) // 2]
    # 2026-05-21 baseline with rembg+mean3+gated-refinement pipeline: median 72 px
    # Generous bound (catches if fixture gets corrupted to all-zeros or all-huge)
    assert 30 < median_err < 200, f"Suspicious median error: {median_err}"
