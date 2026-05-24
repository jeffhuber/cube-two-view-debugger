"""Unit tests for `tools/diagnose_affine_phase_tiebreakers.py`."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import diagnose_affine_phase_tiebreakers as d  # noqa: E402


def test_angle_diff_undirected_treats_opposite_directions_as_zero():
    assert d._angle_diff_undirected_deg(0.0, math.pi) == pytest.approx(0.0)
    assert d._angle_diff_undirected_deg(0.0, math.pi / 2.0) == pytest.approx(90.0)
    assert d._angle_diff_undirected_deg(math.radians(10), math.radians(170)) == (
        pytest.approx(20.0)
    )


def test_axis_bezel_alignment_finds_best_permutation():
    axes = [(1.0, 0.0), (0.0, 2.0), (-3.0, 0.0)]
    # Deliberately shuffled. The third axis is opposite the first line,
    # and line matching is undirected, so it should still be 0 error.
    bezel_angles = [math.pi / 2.0, math.pi, 0.0]

    result = d._axis_bezel_alignment(axes, bezel_angles)

    assert result is not None
    assert result["total_alignment_error_deg"] == pytest.approx(0.0)
    assert result["per_axis_alignment_error_deg"] == [0.0, 0.0, 0.0]


def test_summarize_row_reports_exact_and_near_selector_winners():
    records = [
        {
            "axis_state": "broken",
            "order_index": 0,
            "residual_px2": 1.0,
            "residual_rms_px": 1.0,
            "axis_match": {"total_misfit_deg": 178.0},
            "bezel_alignment": {"total_alignment_error_deg": 80.0},
            "center_error_to_bezel_px": 5.0,
            "center_error_to_hex_centroid_px": 5.0,
            "vertex_error_to_truth_px": 10.0,
            "perm": [0, 1, 2, 3, 4, 5],
        },
        {
            "axis_state": "usable",
            "order_index": 1,
            "residual_px2": 1.0,
            "residual_rms_px": 1.0,
            "axis_match": {"total_misfit_deg": 8.0},
            "bezel_alignment": {"total_alignment_error_deg": 4.0},
            "center_error_to_bezel_px": 8.0,
            "center_error_to_hex_centroid_px": 8.0,
            "vertex_error_to_truth_px": 12.0,
            "perm": [1, 2, 3, 4, 5, 0],
        },
        {
            "axis_state": "usable",
            "order_index": 2,
            "residual_px2": 1.21,
            "residual_rms_px": 1.1,
            "axis_match": {"total_misfit_deg": 6.0},
            "bezel_alignment": {"total_alignment_error_deg": 10.0},
            "center_error_to_bezel_px": 2.0,
            "center_error_to_hex_centroid_px": 2.0,
            "vertex_error_to_truth_px": 3.0,
            "perm": [2, 3, 4, 5, 0, 1],
        },
    ]

    summary = d.summarize_row(
        records,
        tie_rms_epsilon=0.001,
        near_rms_epsilon=0.25,
    )

    assert summary["production_selected"]["axis_state"] == "broken"
    assert summary["exact_group"]["n"] == 2
    assert summary["exact_group"]["has_usable"] is True
    assert summary["exact_group"]["bezel_alignment_range_deg"] == 76.0
    assert summary["exact_group_winners"]["bezel_alignment"]["axis_state"] == "usable"
    assert summary["exact_group_winners"]["center_to_bezel"]["axis_state"] == "broken"
    assert summary["near_group"]["n"] == 3
    assert summary["near_group_winners"]["center_to_bezel"]["order_index"] == 2


def test_summarize_payload_counts_selector_effects():
    row = {
        "status": "traced",
        "summary": {
            "production_selected": {"axis_state": "broken"},
            "exact_group": {
                "n": 2,
                "has_usable": True,
                "bezel_alignment_range_deg": 0.0,
                "center_to_bezel_range_px": 0.0,
            },
            "near_group": {"n": 3, "has_usable": True},
            "exact_group_winners": {
                "bezel_alignment": {"axis_state": "usable"},
                "center_to_bezel": {"axis_state": "broken"},
                "bezel_then_center": {"axis_state": "usable"},
            },
            "near_group_winners": {
                "bezel_alignment": {"axis_state": "usable"},
                "center_to_bezel": {"axis_state": "usable"},
                "bezel_then_center": {"axis_state": "usable"},
            },
        },
    }

    summary = d.summarize_payload([row])

    assert summary["selector_axis_state_counts"]["production"] == {"broken": 1}
    assert summary["selector_effect_counts_vs_production"]["exact_bezel"] == {
        "fixes_broken_to_usable": 1
    }
    assert summary["exact_tie_rows_with_usable_candidate"] == 1
    assert summary["near_tie_rows_with_usable_candidate"] == 1
    assert summary["exact_rows_with_nonzero_bezel_alignment_range"] == 0


def test_render_report_includes_selector_table():
    payload = {
        "source": {
            "tool": "tools/diagnose_affine_phase_tiebreakers.py",
            "truth": "tests/fixtures/full_corner_ground_truth.json",
            "manifest": "tests/fixtures/corpus_manifest.json",
            "max_image_dim": 1600,
            "tie_rms_epsilon": 0.001,
            "near_rms_epsilon": 0.25,
            "mask_path": "alpha",
            "human_truth_usage": "evaluation only",
        },
        "summary": {
            "n_rows": 1,
            "n_traced": 1,
            "selector_axis_state_counts": {"production": {"broken": 1}},
            "selector_effect_counts_vs_production": {},
            "exact_tie_rows_with_usable_candidate": 1,
            "near_tie_rows_with_usable_candidate": 1,
            "median_exact_tie_group_size": 2,
            "median_near_tie_group_size": 3,
            "exact_rows_with_nonzero_bezel_alignment_range": 0,
            "exact_rows_with_nonzero_center_to_bezel_range": 0,
        },
        "per_row": [{
            "key": "20_A",
            "status": "traced",
            "summary": {
                "exact_group": {
                    "n": 2,
                    "axis_state_counts": {"usable": 1},
                    "bezel_alignment_range_deg": 0.0,
                    "center_to_bezel_range_px": 0.0,
                },
                "near_group": {"n": 3, "axis_state_counts": {"usable": 2}},
                "production_selected": {
                    "axis_state": "broken",
                    "total_axis_misfit_deg": 178.0,
                },
                "best_axis_oracle": {
                    "axis_state": "usable",
                    "total_axis_misfit_deg": 8.0,
                },
                "exact_group_winners": {
                    "bezel_alignment": {
                        "axis_state": "usable",
                        "total_axis_misfit_deg": 8.0,
                    },
                    "center_to_bezel": {
                        "axis_state": "broken",
                        "total_axis_misfit_deg": 178.0,
                    },
                },
                "near_group_winners": {
                    "bezel_alignment": {
                        "axis_state": "usable",
                        "total_axis_misfit_deg": 8.0,
                    },
                    "center_to_bezel": {
                        "axis_state": "usable",
                        "total_axis_misfit_deg": 6.0,
                    },
                },
            },
        }],
    }

    report = d.render_report(payload)

    assert "# Affine phase tie-breaker audit" in report
    assert "`20_A`" in report
    assert "exact bezel" in report


def test_render_only_regenerates_markdown(tmp_path):
    payload = {
        "source": {},
        "summary": {},
        "per_row": [],
    }
    trace = tmp_path / "trace.json"
    report = tmp_path / "report.md"
    trace.write_text(json.dumps(payload), encoding="utf-8")

    rc = d.main(["--render-only", "--out-json", str(trace), "--out-md", str(report)])

    assert rc == 0
    assert "# Affine phase tie-breaker audit" in report.read_text(
        encoding="utf-8"
    )
