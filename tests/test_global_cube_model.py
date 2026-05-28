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
   what tools/run_global_cube_model.py expects to compare against
"""
from __future__ import annotations

import importlib
import json
import math
from types import SimpleNamespace
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


def test_hull_label_mode_normalizes_explicit_and_env(monkeypatch):
    from tools.global_cube_model import HULL_LABEL_TIER1_ENV, _hull_label_mode

    monkeypatch.delenv(HULL_LABEL_TIER1_ENV, raising=False)
    assert _hull_label_mode(None) == "off"
    assert _hull_label_mode("shadow") == "shadow"
    assert _hull_label_mode("1") == "prefer"
    assert _hull_label_mode("prefer") == "prefer"
    assert _hull_label_mode("definitely-not-a-mode") == "off"

    monkeypatch.setenv(HULL_LABEL_TIER1_ENV, "trace")
    assert _hull_label_mode(None) == "shadow"


def test_hull_label_subdivide_quad_preserves_all_four_corners():
    from tools.global_cube_model import _subdivide_quad_cells

    cells_by_face = _subdivide_quad_cells({
        "face_xy": [(0.0, 0.0), (9.0, 0.0), (12.0, 9.0), (0.0, 9.0)],
    })

    cells = cells_by_face["face_xy"]
    assert len(cells) == 9
    assert cells[0][0] == (0.0, 0.0)
    assert cells[2][1] == (9.0, 0.0)
    assert cells[8][2] == (12.0, 9.0)
    assert cells[6][3] == (0.0, 9.0)


def test_hull_label_fit_converts_to_legacy_global_model_shape():
    from tools.corner_conventions import FACE_DEFS_BY_SIDE
    from tools.global_cube_model import _global_model_from_hull_label_fit

    vertex = (100.0, 100.0)
    corners_by_num = {
        0: (100.0, 0.0),
        1: (200.0, 50.0),
        2: (200.0, 150.0),
        3: (100.0, 200.0),
        4: (0.0, 150.0),
        5: (0.0, 50.0),
    }
    face_quads = {}
    for slot, names in FACE_DEFS_BY_SIDE["A"].items():
        quad = []
        for name in names:
            quad.append(vertex if name == "vertex" else corners_by_num[int(name.split("_")[1])])
        face_quads[slot] = quad

    fit = SimpleNamespace(
        vertex=vertex,
        vertex_source="affine",
        corners_by_num=corners_by_num,
        face_quads=face_quads,
    )
    decision = SimpleNamespace(warnings=("sticker_score_total=701.0; warning 700.0",))
    trace = {"status": "accepted", "selected": True}
    model = _global_model_from_hull_label_fit(
        fit,
        "A",
        score={"total_distance": 500.0},
        decision=decision,
        trace=trace,
    )

    assert model.debug["approach"] == "hull_label_tier1"
    assert model.debug["phase_check"] == "skipped_hull_label_tier1"
    assert model.debug["hull_label_tier1"]["status"] == "accepted"
    assert set(model.face_quads) == {"face_yz", "face_xz", "face_xy"}
    assert set(model.sticker_cells) == {"face_yz", "face_xz", "face_xy"}
    assert all(len(cells) == 9 for cells in model.sticker_cells.values())
    assert model.visible_corners["h_z"] == corners_by_num[1]
    assert model.visible_corners["h_y"] == corners_by_num[3]
    assert model.visible_corners["h_x"] == corners_by_num[5]


def test_hull_label_prefer_bypasses_missing_legacy_detection(monkeypatch):
    from tools import global_cube_model as mod
    from tools.interior_bezel_detection import InteriorBezelDetection

    expected = mod.GlobalCubeModel(
        cube_center_screen=(10.0, 10.0),
        axis_x_2d=(1.0, 0.0),
        axis_y_2d=(0.0, 1.0),
        axis_z_2d=(-1.0, 0.0),
        debug={"approach": "hull_label_tier1"},
    )
    calls = []

    def fake_fit(image_rgb, silhouette_mask, *, side, mode):
        calls.append((side, mode, image_rgb.shape, silhouette_mask.shape))
        return expected, {"status": "accepted", "selected": True}

    monkeypatch.setattr(mod, "_fit_hull_label_tier1_model", fake_fit)
    detection = InteriorBezelDetection(cube_center=None)
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.ones((20, 30), dtype=bool)

    model = mod.fit_global_cube_model(
        detection,
        rgb,
        mask,
        hull_label_side="A",
        hull_label_mode="prefer",
    )

    assert model is expected
    assert calls == [("A", "prefer", (20, 30, 3), (20, 30))]


def test_hull_label_alpha_selector_records_selected_threshold(monkeypatch):
    from tools import global_cube_model as mod
    from tools import rectify_via_hull_labels as rect_mod

    selected_fit = SimpleNamespace(
        rectified_faces={},
        vertex_source="affine",
        vertex=(1.0, 2.0),
        affine_vertex=(1.0, 2.0),
        projective_vertex=None,
        hexagon_diameter_px=100.0,
        vertex_cloud_spread_px=10.0,
        vertex_cloud_spread_norm=0.1,
        projective_residual_norm=None,
        projective_degeneracy=None,
        corners_by_num={},
        face_quads={},
    )
    selected_decision = SimpleNamespace(
        accepted=True,
        hard_failures=(),
        warnings=(),
        metrics={},
    )
    selection = SimpleNamespace(
        fit=selected_fit,
        threshold=224,
        score={"total_distance": 123.0},
        decision=selected_decision,
        vertex_estimates=[(1.0, 2.0), (3.0, 4.0)],
        trace={
            "thresholds": [64, 128, 224],
            "threshold_candidates": [
                {"threshold": 128, "accepted": False, "sticker_score_total": 999.0},
                {"threshold": 224, "accepted": True, "sticker_score_total": 123.0},
            ],
            "best_any_threshold": 224,
            "best_any_accepted": True,
            "best_any_score": 123.0,
        },
    )
    calls = []

    def fake_select(image, alpha, side, *, thresholds):
        calls.append((image.size, alpha.shape, side, tuple(thresholds)))
        return selection

    def fake_model_from_fit(fit, side, *, score, decision, trace):
        assert fit is selected_fit
        assert side == "B"
        assert score == selection.score
        assert decision is selected_decision
        assert trace["selected_mask_threshold"] == 224
        return SimpleNamespace(debug={"hull_label_tier1": trace})

    monkeypatch.setattr(rect_mod, "select_hull_label_threshold_fit", fake_select)
    monkeypatch.setattr(mod, "_slot_center_faces_from_rectified", lambda _faces: {})
    monkeypatch.setattr(mod, "_global_model_from_hull_label_fit", fake_model_from_fit)

    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    alpha = np.full((20, 30), 255, dtype=np.uint8)
    model, trace = mod._fit_hull_label_tier1_model_from_alpha(
        rgb,
        alpha,
        side="B",
        mode="prefer",
        thresholds=[64, 128, 224],
    )

    assert model is not None
    assert trace["status"] == "accepted"
    assert trace["selected"] is True
    assert trace["selected_mask_threshold"] == 224
    assert trace["best_any_threshold"] == 224
    assert len(trace["threshold_candidates"]) == 2
    assert calls == [((30, 20), (20, 30), "B", (64, 128, 224))]


def test_hull_label_shadow_keeps_legacy_model_and_attaches_trace(monkeypatch):
    from tools import global_cube_model as mod
    from tools.interior_bezel_detection import InteriorBezelDetection

    hull_model = mod.GlobalCubeModel(
        cube_center_screen=(10.0, 10.0),
        axis_x_2d=(1.0, 0.0),
        axis_y_2d=(0.0, 1.0),
        axis_z_2d=(-1.0, 0.0),
    )
    legacy_model = mod.GlobalCubeModel(
        cube_center_screen=(50.0, 50.0),
        axis_x_2d=(10.0, 0.0),
        axis_y_2d=(0.0, 10.0),
        axis_z_2d=(-10.0, 0.0),
        visible_corners={},
        face_quads={},
        sticker_cells={},
        debug={"approach": "legacy"},
    )

    monkeypatch.setattr(
        mod,
        "_fit_hull_label_tier1_model",
        lambda image_rgb, silhouette_mask, *, side, mode: (
            hull_model,
            {"status": "accepted", "selected": False, "accepted": True},
        ),
    )
    monkeypatch.setattr(mod, "_try_import_scipy_ndimage", lambda: object())
    monkeypatch.setattr(
        mod,
        "detect_hexagon_anchors",
        lambda _mask: [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (2.0, 2.0), (1.0, 3.0), (0.0, 2.0)],
    )
    monkeypatch.setattr(
        mod,
        "fit_cube_template_to_anchors",
        lambda cube_center, hexagon, boundary_angles, image_size=None: legacy_model,
    )
    monkeypatch.setattr(
        mod,
        "_resolve_near_far_phase",
        lambda model, detection, image_rgb, apply_correction=True: (
            model,
            {"phase_check": "test_skipped"},
        ),
    )
    monkeypatch.setattr(
        mod,
        "_refine_vertex_via_image_junction",
        lambda image_rgb, vertex, axes: (vertex, {"vertex_refinement": "test_skipped"}),
    )

    detection = InteriorBezelDetection(
        cube_center=(50.0, 50.0),
        boundary_angles=[math.radians(90), math.radians(210), math.radians(330)],
    )
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.ones((20, 30), dtype=bool)

    model = mod.fit_global_cube_model(
        detection,
        rgb,
        mask,
        hull_label_side="A",
        hull_label_mode="shadow",
    )

    assert model is legacy_model
    assert model.debug["approach"] == "procrustes_template_fit+mean3_vertex"
    assert model.debug["hull_label_tier1"]["status"] == "accepted"
    assert model.debug["hull_label_tier1"]["selected"] is False
    assert model.debug["hull_label_tier1"]["shadow_returned_legacy"] is True


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
