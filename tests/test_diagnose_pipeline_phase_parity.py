"""Unit tests for `tools/diagnose_pipeline_phase_parity.py`.

The load-bearing piece is `reconstruct_pre_correction_model` — it must
exactly invert the 60°-flip applied inside `_resolve_near_far_phase`.
A round-trip test (apply flip, then reconstruct) is the cleanest way to
verify the math.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.diagnose_pipeline_phase_parity import (
    _CATEGORY_RANK,
    _is_better,
    _set_id_from_key,
    _side_from_key,
    reconstruct_pre_correction_model,
    render_report,
)
from tools.global_cube_model import GlobalCubeModel, derive_geometry


REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_pre_model(
    cx: float = 100.0, cy: float = 200.0,
    ax: tuple = (50.0, 10.0),
    ay: tuple = (-20.0, 40.0),
    az: tuple = (5.0, -30.0),
) -> GlobalCubeModel:
    model = GlobalCubeModel(
        cube_center_screen=(cx, cy),
        axis_x_2d=ax,
        axis_y_2d=ay,
        axis_z_2d=az,
    )
    derive_geometry(model)
    return model


def _apply_60deg_flip(model: GlobalCubeModel) -> GlobalCubeModel:
    """Apply the same 60°-flip as `_resolve_near_far_phase` does — use
    the model's far positions (h_xy, h_xz, h_yz) as new axes."""
    cx, cy = model.cube_center_screen
    far_keys = ("h_xy", "h_xz", "h_yz")
    far_positions = [model.visible_corners[k] for k in far_keys]
    new_axes = [(p[0] - cx, p[1] - cy) for p in far_positions]
    flipped = GlobalCubeModel(
        cube_center_screen=model.cube_center_screen,
        axis_x_2d=new_axes[0],
        axis_y_2d=new_axes[1],
        axis_z_2d=new_axes[2],
        fit_loss=model.fit_loss,
        fit_quality=model.fit_quality,
    )
    derive_geometry(flipped)
    return flipped


# ----- reconstruct_pre_correction_model: the load-bearing math -----


def test_reconstruct_round_trips_a_flipped_model():
    """Apply the 60°-flip and then reconstruct — should exactly recover
    the original axes (within floating-point tolerance)."""
    original = _make_pre_model()
    flipped = _apply_60deg_flip(original)
    reconstructed = reconstruct_pre_correction_model(flipped)
    assert reconstructed.axis_x_2d == pytest.approx(original.axis_x_2d, abs=1e-9)
    assert reconstructed.axis_y_2d == pytest.approx(original.axis_y_2d, abs=1e-9)
    assert reconstructed.axis_z_2d == pytest.approx(original.axis_z_2d, abs=1e-9)


def test_reconstruct_round_trips_for_multiple_random_models():
    """Exercise round-trip on several axis combinations."""
    import random
    rng = random.Random(42)
    for _ in range(20):
        ax = (rng.uniform(-100, 100), rng.uniform(-100, 100))
        ay = (rng.uniform(-100, 100), rng.uniform(-100, 100))
        az = (rng.uniform(-100, 100), rng.uniform(-100, 100))
        original = _make_pre_model(ax=ax, ay=ay, az=az)
        flipped = _apply_60deg_flip(original)
        reconstructed = reconstruct_pre_correction_model(flipped)
        assert reconstructed.axis_x_2d == pytest.approx(ax, abs=1e-9)
        assert reconstructed.axis_y_2d == pytest.approx(ay, abs=1e-9)
        assert reconstructed.axis_z_2d == pytest.approx(az, abs=1e-9)


def test_reconstruct_preserves_cube_center():
    original = _make_pre_model(cx=512.0, cy=768.0)
    flipped = _apply_60deg_flip(original)
    reconstructed = reconstruct_pre_correction_model(flipped)
    assert reconstructed.cube_center_screen == (512.0, 768.0)


def test_reconstruct_derives_geometry():
    """After reconstruction, the visible_corners dict should be populated
    consistently with the recovered axes."""
    original = _make_pre_model()
    flipped = _apply_60deg_flip(original)
    reconstructed = reconstruct_pre_correction_model(flipped)
    assert "h_x" in reconstructed.visible_corners
    assert "h_xy" in reconstructed.visible_corners
    # h_x = vertex + axis_x_2d
    expected_hx = (
        reconstructed.cube_center_screen[0] + reconstructed.axis_x_2d[0],
        reconstructed.cube_center_screen[1] + reconstructed.axis_x_2d[1],
    )
    assert reconstructed.visible_corners["h_x"] == pytest.approx(expected_hx, abs=1e-9)


# ----- _is_better category ranking -----


def test_is_better_ranks_categories_correctly():
    """GOOD beats MARGINAL beats PHASE_SWAPPED beats GEOMETRY_FAIL."""
    assert _is_better("GOOD", "MARGINAL")
    assert _is_better("MARGINAL", "PHASE_SWAPPED")
    assert _is_better("PHASE_SWAPPED", "GEOMETRY_FAIL")
    assert _is_better("GOOD", "GEOMETRY_FAIL")
    assert not _is_better("GOOD", "GOOD")  # equal is not "better"
    assert not _is_better("PHASE_SWAPPED", "GOOD")  # worse than is not "better"


def test_is_better_unknown_categories_treated_as_worst():
    """Unknown categories rank below everything known so we don't
    accidentally call a typo 'better than GOOD'."""
    assert _is_better("GOOD", "FOO")
    assert not _is_better("FOO", "GOOD")


def test_category_rank_known_values():
    """Sanity: the rank table has the expected entries."""
    for cat in ("GOOD", "MARGINAL", "PHASE_SWAPPED", "GEOMETRY_FAIL"):
        assert cat in _CATEGORY_RANK


# ----- key parsing helpers -----


def test_side_from_key():
    assert _side_from_key("20_A") == "A"
    assert _side_from_key("38_B") == "B"


def test_set_id_from_key():
    assert _set_id_from_key("20_A") == "20"
    assert _set_id_from_key("38_B") == "38"


# ----- render_report shape -----


def test_render_report_empty_payload_is_well_formed():
    payload = {
        "schema": "pipeline_phase_parity_trace_v1",
        "source": {},
        "summary": {
            "n_total": 0,
            "n_traced": 0,
            "post_canonical_category_counts": {},
            "pre_canonical_category_counts": {},
            "phase_check_counts": {},
            "score_delta_class_counts": {},
        },
        "rows": [],
    }
    report = render_report(payload)
    assert "# Pipeline phase-parity failure modes" in report
    assert "## Aggregate" in report
    assert "## Per-row trace" in report
    assert "## Q4: Which evidence would have selected the right parity?" in report
    assert "## Findings & implications" in report


def test_render_report_includes_row_table_with_traced_status():
    payload = {
        "schema": "pipeline_phase_parity_trace_v1",
        "source": {},
        "summary": {
            "n_total": 1,
            "n_traced": 1,
            "post_canonical_category_counts": {"GOOD": 1},
            "pre_canonical_category_counts": {"GOOD": 1},
            "phase_check_counts": {"correct": 1},
            "score_delta_class_counts": {"no_flip": 1},
        },
        "rows": [{
            "key": "20_A",
            "status": "traced",
            "pre_canonical_category": "GOOD",
            "post_canonical_category": "GOOD",
            "phase_check": "correct",
            "score_delta_class": "no_flip",
            "flip_applied": False,
            "phase_debug": {"phase_darkness_separation": -15.0},
        }],
    }
    report = render_report(payload)
    assert "`20_A`" in report
    assert "ONE_EDGE ✓" in report  # Q1 column for GOOD pre


def test_render_report_handles_error_rows_gracefully():
    """If a row's status is 'error', it should still show in the table."""
    payload = {
        "schema": "pipeline_phase_parity_trace_v1",
        "source": {},
        "summary": {
            "n_total": 1, "n_traced": 0,
            "post_canonical_category_counts": {},
            "pre_canonical_category_counts": {},
            "phase_check_counts": {},
            "score_delta_class_counts": {},
        },
        "rows": [{
            "key": "99_A",
            "status": "error",
            "error": "rembg failed: ConnectionError: timed out",
        }],
    }
    report = render_report(payload)
    assert "`99_A`" in report
    assert "status=error" in report
