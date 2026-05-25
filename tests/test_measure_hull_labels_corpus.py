"""Unit tests for ``tools/measure_hull_labels_corpus.py``.

Covers the deterministic helpers:
1. ``_ground_truth_axes_from_axis_truth`` — read the 70-row schema
   and produce vertex-relative axis vectors at the expected scale.
2. ``_classify_row`` — failure-bucket classifier; pin each bucket
   boundary so threshold tweaks are intentional.
3. The FAR-vs-NEAR axis convention (the bug that caught us during
   the first run) — guard against regressing to NEAR-corner
   predicted axes.

The full corpus run is exercised by the CLI against the committed
fixture trace; this file pins the math.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.corner_conventions import (  # noqa: E402
    FAR_CORNERS_BY_SIDE,
    ONE_EDGE_CORNERS_BY_SIDE,
)
from tools.measure_hull_labels_corpus import (  # noqa: E402
    THRESH_AXIS_MISFIT_DEG,
    THRESH_STICKER_SCORE_TOTAL,
    THRESH_VERTEX_CLOUD_SPREAD_PX,
    _classify_row,
    _ground_truth_axes_from_axis_truth,
)


# ---------------- _ground_truth_axes_from_axis_truth ----------------


def test_ground_truth_axes_returns_three_vectors_at_processing_scale():
    truth_row = {
        "vertex": [1500.0, 1800.0],
        "axis_x": [2300.0, 2500.0],  # ≈ side A corner_2 area (FAR)
        "axis_y": [500.0, 2500.0],   # ≈ corner_4 area (FAR)
        "axis_z": [1500.0, 950.0],   # ≈ corner_0 area (FAR)
    }
    scale = 0.5
    vertex_proc, axes = _ground_truth_axes_from_axis_truth(truth_row, scale)
    # Vertex scaled
    assert vertex_proc == (750.0, 900.0)
    # 3 axis vectors, each from scaled vertex to scaled axis-endpoint
    assert len(axes) == 3
    # First axis: scaled axis_x − scaled vertex
    expected_ax0 = (2300.0 * 0.5 - 750.0, 2500.0 * 0.5 - 900.0)
    assert axes[0] == expected_ax0


def test_ground_truth_axes_order_matches_axis_xyz_order():
    """The order of returned axes must be (axis_x, axis_y, axis_z) —
    callers rely on this ordering to interpret which world axis each
    vector represents (though _match_axes_to_ground_truth recovers the
    permutation anyway). Pin the order so future refactors don't shuffle.
    """
    truth_row = {
        "vertex": [0.0, 0.0],
        "axis_x": [10.0, 0.0],
        "axis_y": [0.0, 20.0],
        "axis_z": [0.0, 0.0],  # zero axis (degenerate but valid for test)
    }
    _vertex, axes = _ground_truth_axes_from_axis_truth(truth_row, 1.0)
    assert axes[0] == (10.0, 0.0), "first axis must be axis_x"
    assert axes[1] == (0.0, 20.0), "second axis must be axis_y"
    assert axes[2] == (0.0, 0.0),  "third axis must be axis_z"


def test_ground_truth_axes_backward_compat_reads_legacy_near_keys():
    """Legacy fixtures still keyed by ``near_x/y/z`` must read identically
    to the canonical ``axis_x/y/z`` form. Both spellings point at the
    same FAR-corner positions — only the key differs. The shim in
    ``_ground_truth_axes_from_axis_truth`` uses
    ``truth_row.get(new, truth_row.get(old))``.
    """
    legacy_row = {
        "vertex": [100.0, 100.0],
        "near_x": [200.0, 150.0],
        "near_y": [50.0, 250.0],
        "near_z": [100.0, 50.0],
    }
    canonical_row = {
        "vertex": [100.0, 100.0],
        "axis_x": [200.0, 150.0],
        "axis_y": [50.0, 250.0],
        "axis_z": [100.0, 50.0],
    }
    v_legacy, axes_legacy = _ground_truth_axes_from_axis_truth(legacy_row, 1.0)
    v_canon, axes_canon = _ground_truth_axes_from_axis_truth(canonical_row, 1.0)
    assert v_legacy == v_canon
    assert axes_legacy == axes_canon


# ---------------- _classify_row buckets ----------------


def test_classify_clean_row_below_all_thresholds():
    rec = {
        "vertex_cloud_spread_px": 200.0,
        "axis_total_misfit_deg": 10.0,
        "sticker_score_total": 500.0,
    }
    assert _classify_row(
        rec,
        thresh_spread=THRESH_VERTEX_CLOUD_SPREAD_PX,
        thresh_axis=THRESH_AXIS_MISFIT_DEG,
        thresh_sticker=THRESH_STICKER_SCORE_TOTAL,
    ) == "rectified_clean"


def test_classify_vertex_cloud_spread_fires_first():
    rec = {
        "vertex_cloud_spread_px": THRESH_VERTEX_CLOUD_SPREAD_PX + 1,
        "axis_total_misfit_deg": THRESH_AXIS_MISFIT_DEG + 1,  # also above
        "sticker_score_total": THRESH_STICKER_SCORE_TOTAL + 1,  # also above
    }
    # Order in _classify_row is: spread → axis → sticker → clean.
    # When all three trip, spread wins. Pin so re-ordering is intentional.
    assert _classify_row(
        rec,
        thresh_spread=THRESH_VERTEX_CLOUD_SPREAD_PX,
        thresh_axis=THRESH_AXIS_MISFIT_DEG,
        thresh_sticker=THRESH_STICKER_SCORE_TOTAL,
    ) == "vertex_cloud_high_spread"


def test_classify_axis_misfit_high():
    rec = {
        "vertex_cloud_spread_px": 100.0,
        "axis_total_misfit_deg": THRESH_AXIS_MISFIT_DEG + 0.1,
        "sticker_score_total": 500.0,
    }
    assert _classify_row(
        rec,
        thresh_spread=THRESH_VERTEX_CLOUD_SPREAD_PX,
        thresh_axis=THRESH_AXIS_MISFIT_DEG,
        thresh_sticker=THRESH_STICKER_SCORE_TOTAL,
    ) == "axis_misfit_high"


def test_classify_sticker_score_high():
    rec = {
        "vertex_cloud_spread_px": 100.0,
        "axis_total_misfit_deg": 10.0,
        "sticker_score_total": THRESH_STICKER_SCORE_TOTAL + 1,
    }
    assert _classify_row(
        rec,
        thresh_spread=THRESH_VERTEX_CLOUD_SPREAD_PX,
        thresh_axis=THRESH_AXIS_MISFIT_DEG,
        thresh_sticker=THRESH_STICKER_SCORE_TOTAL,
    ) == "sticker_score_high"


def test_classify_threshold_boundary_inclusive_below():
    """At exactly the threshold value, do NOT classify as failure —
    the comparison is strict greater-than. Pin so threshold tweaks
    are intentional."""
    rec = {
        "vertex_cloud_spread_px": THRESH_VERTEX_CLOUD_SPREAD_PX,
        "axis_total_misfit_deg": THRESH_AXIS_MISFIT_DEG,
        "sticker_score_total": THRESH_STICKER_SCORE_TOTAL,
    }
    assert _classify_row(
        rec,
        thresh_spread=THRESH_VERTEX_CLOUD_SPREAD_PX,
        thresh_axis=THRESH_AXIS_MISFIT_DEG,
        thresh_sticker=THRESH_STICKER_SCORE_TOTAL,
    ) == "rectified_clean"


# ---------------- FAR vs NEAR convention regression test ----------------


def test_far_corners_distinct_from_near_corners_per_side():
    """The bug that caught us during first-run: predicted axes were
    computed using ONE_EDGE_CORNERS (NEAR), but the 70-row GT labels
    sit at FAR positions. Sets must be disjoint per side — guarding
    against a refactor that conflates them."""
    for side in ("A", "B"):
        near = set(ONE_EDGE_CORNERS_BY_SIDE[side])
        far = set(FAR_CORNERS_BY_SIDE[side])
        assert near.isdisjoint(far), (
            f"side {side}: NEAR {near} and FAR {far} must be disjoint"
        )
        assert near | far == {f"corner_{i}" for i in range(6)}, (
            f"side {side}: NEAR ∪ FAR must cover all 6 corners"
        )


def test_far_corner_set_matches_expected_per_side():
    """Pin the exact FAR set per side — these are what the predicted
    axes for the 70-row corpus use. A regression here would silently
    rotate the axis comparison."""
    assert set(FAR_CORNERS_BY_SIDE["A"]) == {"corner_0", "corner_2", "corner_4"}
    assert set(FAR_CORNERS_BY_SIDE["B"]) == {"corner_1", "corner_3", "corner_5"}


# ---------------- empty-image-path guard (Codex P3 on head 04784014) ----------------
# ---------------- lazy rembg init (Codex P1 on head 4d1bae4) ----------------


def test_empty_image_path_in_manifest_routes_to_skipped_without_rembg(
    tmp_path, monkeypatch,
):
    """If a manifest pair exists but ``imageAPath``/``imageBPath`` is
    empty, the row must be routed to ``skipped_unresolved_image``
    rather than passed through to ``_resolve_image_path`` (which can
    silently return the corpus root as a candidate via
    ``Path("").name → root``, causing the row to later surface as an
    ``error`` trying to open a directory) — original Codex P3 finding.

    Additionally, this skip path must NOT require the ``rembg``
    package, since the package is optional and only needed when we
    actually evaluate a row. Codex P1 on head 4d1bae4: previous
    eager ``new_session("u2net")`` at the top of ``main()`` made the
    skip-path test ``ModuleNotFoundError`` on clean installs without
    rembg. Fix is lazy session init via ``_get_sess``.

    We simulate a clean-install rembg-less environment by patching
    ``builtins.__import__`` to raise ``ImportError`` on any rembg
    import. If the skip path reaches ``_get_sess()`` (and through it
    ``new_session``), this test fails — proving the lazy guard works.
    """
    import builtins
    import json
    import sys

    # Drop any cached rembg module so the patched __import__ fires.
    for mod_name in list(sys.modules):
        if mod_name == "rembg" or mod_name.startswith("rembg."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "rembg" or name.startswith("rembg."):
            raise ImportError(
                "simulated rembg-not-installed (skip path must not import rembg)"
            )
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from tools.measure_hull_labels_corpus import main

    # Minimal axis truth with one approved row pointing to a
    # set whose manifest entry has an empty image path.
    axis_truth = {
        "99_A": {
            "approved": True,
            "vertex": [100, 100],
            "near_x": [200, 100],
            "near_y": [100, 200],
            "near_z": [100, 50],
        }
    }
    full_corner_truth: dict = {}
    manifest = {"pairs": [{"setId": "99", "imageAPath": "", "imageBPath": ""}]}

    axis_path = tmp_path / "axis.json"
    full_path = tmp_path / "full.json"
    manifest_path = tmp_path / "manifest.json"
    out_json = tmp_path / "out.json"
    axis_path.write_text(json.dumps(axis_truth))
    full_path.write_text(json.dumps(full_corner_truth))
    manifest_path.write_text(json.dumps(manifest))

    rc = main([
        "--axis-truth", str(axis_path),
        "--full-corner-truth", str(full_path),
        "--manifest", str(manifest_path),
        "--out-json", str(out_json),
    ])
    assert rc == 0

    out = json.loads(out_json.read_text())
    # Row should be in skipped, NOT in per_row
    assert out["summary"]["skipped_unresolved_image"] == 1
    assert out["summary"]["total_rows_attempted"] == 0
    assert any(s["key"] == "99_A" for s in out["skipped"])
    # And the skip reason must mention the manifest path is missing,
    # not a generic "no image found" (so the human can immediately
    # tell the manifest is the problem, not the corpus).
    skip = next(s for s in out["skipped"] if s["key"] == "99_A")
    assert "no image path in manifest" in skip["reason"].lower()
