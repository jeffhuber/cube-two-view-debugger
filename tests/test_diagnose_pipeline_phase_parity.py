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
    """v2 multi-run schema with no per-row data still renders all
    section headers cleanly."""
    payload = {
        "schema": "pipeline_phase_parity_trace_v2_multi_run",
        "source": {"n_runs_per_row": 5},
        "summary": {
            "n_total_rows": 0,
            "n_traced_rows": 0,
            "n_stable_rows": 0,
            "n_unstable_rows": 0,
            "post_canonical_category_modal_counts": {},
            "phase_rewound_canonical_category_modal_counts": {},
            "phase_check_modal_counts": {},
            "score_delta_class_modal_counts": {},
        },
        "per_row": [],
    }
    report = render_report(payload)
    assert "# Pipeline phase-parity failure modes" in report
    assert "## Aggregate" in report
    assert "## Per-row trace" in report
    assert "## Q4: Which evidence would have selected the right parity?" in report
    assert "## Findings & implications" in report
    # P1 + P2 caveats are surfaced in the report header
    assert "Pipeline non-determinism note" in report
    assert "phase_rewound" in report.lower() or "Phase-rewound" in report


def test_render_report_includes_per_row_distribution_table():
    """Distributions across N runs per row are surfaced in the per-row
    table — including the stability marker."""
    payload = {
        "schema": "pipeline_phase_parity_trace_v2_multi_run",
        "source": {"n_runs_per_row": 5},
        "summary": {
            "n_total_rows": 1,
            "n_traced_rows": 1,
            "n_stable_rows": 1,
            "n_unstable_rows": 0,
            "post_canonical_category_modal_counts": {"GOOD": 1},
            "phase_rewound_canonical_category_modal_counts": {"GOOD": 1},
            "phase_check_modal_counts": {"correct": 1},
            "score_delta_class_modal_counts": {"no_flip": 1},
        },
        "per_row": [{
            "key": "20_A",
            "n_runs": 5,
            "runs": [
                {
                    "status": "traced",
                    "post_canonical_category": "GOOD",
                    "phase_rewound_canonical_category": "GOOD",
                    "phase_check": "correct",
                    "score_delta_class": "no_flip",
                    "phase_debug": {"phase_darkness_separation": -15.0},
                }
            ] * 5,
            "summary": {
                "n_traced": 5,
                "n_errored": 0,
                "post_canonical_category_dist": {"GOOD": 5},
                "phase_rewound_canonical_category_dist": {"GOOD": 5},
                "phase_check_dist": {"correct": 5},
                "score_delta_class_dist": {"no_flip": 5},
                "post_canonical_category_modal": "GOOD",
                "phase_rewound_canonical_category_modal": "GOOD",
                "phase_check_modal": "correct",
                "score_delta_class_modal": "no_flip",
                "post_category_stable": True,
                "delta_class_stable": True,
                "fully_stable": True,
                "is_stable": True,  # backwards-compat alias
            },
        }],
    }
    report = render_report(payload)
    assert "`20_A`" in report
    assert "GOOD:5" in report  # distribution-formatted post-category
    assert "✓✓" in report  # fully-stable marker


def test_render_report_handles_error_rows_gracefully():
    """If all of a row's N runs errored, the row still shows in the
    table with an `all_runs_errored` marker (not a hard exception)."""
    payload = {
        "schema": "pipeline_phase_parity_trace_v2_multi_run",
        "source": {"n_runs_per_row": 5},
        "summary": {
            "n_total_rows": 1, "n_traced_rows": 0,
            "n_stable_rows": 0, "n_unstable_rows": 0,
            "post_canonical_category_modal_counts": {},
            "phase_rewound_canonical_category_modal_counts": {},
            "phase_check_modal_counts": {},
            "score_delta_class_modal_counts": {},
        },
        "per_row": [{
            "key": "99_A",
            "n_runs": 5,
            "runs": [],
            "summary": {
                "status": "all_runs_errored",
                "errors": ["rembg failed: ConnectionError: timed out"] * 5,
            },
        }],
    }
    report = render_report(payload)
    assert "`99_A`" in report
    assert "all runs errored" in report


def test_aggregate_runs_partial_errors_not_marked_stable():
    """If any of the N runs errors but the surviving runs all agree,
    we must NOT mark the row fully stable — that would hide partial
    failures behind the modal value. Codex P2 round-2 on PR #255."""
    from tools.diagnose_pipeline_phase_parity import _aggregate_runs

    runs = [
        {
            "status": "traced",
            "post_canonical_category": "GOOD",
            "phase_rewound_canonical_category": "GOOD",
            "phase_check": "correct",
            "score_delta_class": "no_flip",
            "phase_debug": {},
        },
        {
            "status": "traced",
            "post_canonical_category": "GOOD",
            "phase_rewound_canonical_category": "GOOD",
            "phase_check": "correct",
            "score_delta_class": "no_flip",
            "phase_debug": {},
        },
        # One errored run — surviving runs unanimous but we should NOT
        # claim stability.
        {"status": "error", "error": "rembg failed"},
    ]
    s = _aggregate_runs(runs)
    assert s["n_traced"] == 2
    assert s["n_errored"] == 1
    assert s["fully_stable"] is False
    assert s["post_category_stable"] is False
    assert s["delta_class_stable"] is False


def test_aggregate_runs_modal_tie_breaks_deterministically():
    """When a row's runs produce tied distributions (e.g. 2 GOOD + 2
    PHASE_SWAPPED in 4 runs), the modal value should be the
    sort-smaller key regardless of run order. Codex P3 round-2 on
    PR #255: `Counter.most_common(1)` is INSERT-order-dependent on
    ties, which would let aggregate modal counts shift run-to-run."""
    from tools.diagnose_pipeline_phase_parity import _aggregate_runs

    base_run = {
        "status": "traced",
        "phase_rewound_canonical_category": "GOOD",
        "phase_check": "correct",
        "score_delta_class": "no_flip",
        "phase_debug": {},
    }
    # 4 runs: 2 GOOD then 2 PHASE_SWAPPED — Counter.most_common would
    # return GOOD (first-seen on tie), but we want sort-stable.
    runs_a = [dict(base_run, post_canonical_category=c) for c in
              ["GOOD", "GOOD", "PHASE_SWAPPED", "PHASE_SWAPPED"]]
    # Same 4 outcomes in opposite order. Sort-stable tie-breaking must
    # yield the SAME modal value regardless of insertion order.
    runs_b = [dict(base_run, post_canonical_category=c) for c in
              ["PHASE_SWAPPED", "PHASE_SWAPPED", "GOOD", "GOOD"]]
    a = _aggregate_runs(runs_a)
    b = _aggregate_runs(runs_b)
    assert a["post_canonical_category_modal"] == b["post_canonical_category_modal"]
    # And it should be "GOOD" (alphabetically first of the tied pair).
    assert a["post_canonical_category_modal"] == "GOOD"


def test_render_report_marks_partial_errors_distinctly():
    """When N-1 runs all agree but 1 errored, the per-row stability
    marker should say `partial-errors`, NOT `post-varies` — those mean
    different things and merging them hides intermittent failures
    (Codex P2 round-4 on PR #255)."""
    payload = {
        "schema": "pipeline_phase_parity_trace_v2_multi_run",
        "source": {"n_runs_per_row": 5},
        "summary": {
            "n_total_rows": 1, "n_traced_rows": 1,
            "n_post_category_stable_rows": 0,
            "n_delta_class_stable_rows": 0,
            "n_fully_stable_rows": 0,
            "n_stable_rows": 0, "n_unstable_rows": 1,
            "post_canonical_category_modal_counts": {"GOOD": 1},
            "phase_rewound_canonical_category_modal_counts": {"GOOD": 1},
            "phase_check_modal_counts": {"correct": 1},
            "score_delta_class_modal_counts": {"no_flip": 1},
        },
        "per_row": [{
            "key": "20_A", "n_runs": 5, "runs": [],
            "summary": {
                "n_traced": 4, "n_errored": 1,
                "post_canonical_category_dist": {"GOOD": 4},
                "phase_rewound_canonical_category_dist": {"GOOD": 4},
                "phase_check_dist": {"correct": 4},
                "score_delta_class_dist": {"no_flip": 4},
                "post_canonical_category_modal": "GOOD",
                "phase_rewound_canonical_category_modal": "GOOD",
                "phase_check_modal": "correct",
                "score_delta_class_modal": "no_flip",
                "post_category_stable": False,
                "delta_class_stable": False,
                "fully_stable": False,
                "is_stable": False,
            },
        }],
    }
    report = render_report(payload)
    assert "`20_A`" in report
    assert "partial-errors" in report
    assert "post-varies" not in report.split("Per-row trace")[1]


def test_render_report_marks_unstable_rows():
    """Rows where the N runs disagree on post-category should show
    `UNSTABLE` (so a reader doesn't take the modal value as gospel)."""
    payload = {
        "schema": "pipeline_phase_parity_trace_v2_multi_run",
        "source": {"n_runs_per_row": 5},
        "summary": {
            "n_total_rows": 1, "n_traced_rows": 1,
            "n_stable_rows": 0, "n_unstable_rows": 1,
            "post_canonical_category_modal_counts": {"PHASE_SWAPPED": 1},
            "phase_rewound_canonical_category_modal_counts": {"GOOD": 1},
            "phase_check_modal_counts": {"corrected_60deg_flip": 1},
            "score_delta_class_modal_counts": {"flip_hurt": 1},
        },
        "per_row": [{
            "key": "38_A",
            "n_runs": 5,
            "runs": [],
            "summary": {
                "n_traced": 5,
                "n_errored": 0,
                "post_canonical_category_dist": {"PHASE_SWAPPED": 3, "GOOD": 2},
                "phase_rewound_canonical_category_dist": {"GOOD": 5},
                "phase_check_dist": {"corrected_60deg_flip": 3, "correct": 2},
                "score_delta_class_dist": {"flip_hurt": 3, "no_flip": 2},
                "post_canonical_category_modal": "PHASE_SWAPPED",
                "phase_rewound_canonical_category_modal": "GOOD",
                "phase_check_modal": "corrected_60deg_flip",
                "score_delta_class_modal": "flip_hurt",
                "post_category_stable": False,
                "delta_class_stable": False,
                "fully_stable": False,
                "is_stable": False,  # backwards-compat alias
            },
        }],
    }
    report = render_report(payload)
    assert "`38_A`" in report
    # Post-category varies across runs → "post-varies" marker
    assert "post-varies" in report
