"""Unit tests for `tools/diagnose_fit_stage_transitions.py`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import diagnose_fit_stage_transitions as d  # noqa: E402
from tools.global_cube_model import _TEMPLATE_HEXAGON_2D_ISO  # noqa: E402


def test_classify_axis_misfit_buckets():
    assert d._classify_axis_misfit(None) == "unknown"
    assert d._classify_axis_misfit(30.0) == "usable"
    assert d._classify_axis_misfit(90.0) == "marginal"
    assert d._classify_axis_misfit(150.0) == "broken"


def test_selected_affine_model_recovers_affine_axes():
    A = np.array([[80.0, 20.0], [-12.0, 95.0]])
    b = np.array([320.0, 140.0])
    hexagon = [
        tuple(A @ np.array(_TEMPLATE_HEXAGON_2D_ISO[name]) + b)
        for name in d.TEMPLATE_KEYS
    ]

    model = d._selected_affine_model(hexagon)

    assert model is not None
    assert model.cube_center_screen == pytest.approx(tuple(b), abs=1e-9)
    assert model.axis_x_2d == pytest.approx(
        tuple(A @ np.array(_TEMPLATE_HEXAGON_2D_ISO["h_x"])),
        abs=1e-9,
    )
    assert model.axis_y_2d == pytest.approx(
        tuple(A @ np.array(_TEMPLATE_HEXAGON_2D_ISO["h_y"])),
        abs=1e-9,
    )
    assert model.axis_z_2d == pytest.approx(
        tuple(A @ np.array(_TEMPLATE_HEXAGON_2D_ISO["h_z"])),
        abs=1e-9,
    )
    assert model.debug["approach"] == "affine_selected"
    assert model.debug["best_permutation"] == [0, 1, 2, 3, 4, 5]


def test_summarize_payload_keeps_corr_false_and_corr_true_paths_distinct():
    rows = [{
        "status": "traced",
        "stages": [
            {
                "stage": "affine_selected",
                "status": "scored",
                "axis_state": "usable",
                "axis_match": {"total_misfit_deg": 5.0},
            },
            {
                "stage": "template_fit_pnp_or_affine",
                "status": "scored",
                "axis_state": "usable",
                "axis_match": {"total_misfit_deg": 6.0},
            },
            {
                "stage": "mean3_vertex",
                "status": "scored",
                "axis_state": "usable",
                "axis_match": {"total_misfit_deg": 6.0},
            },
            {
                "stage": "corr_false_phase_check",
                "status": "scored",
                "axis_state": "broken",
                "axis_match": {"total_misfit_deg": 178.0},
            },
            {
                "stage": "corr_true_phase_check",
                "status": "scored",
                "axis_state": "usable",
                "axis_match": {"total_misfit_deg": 8.0},
            },
        ],
    }]

    summary = d.summarize_payload(rows)

    assert summary["stage_summary"]["affine_selected"]["usable"] == 1
    assert summary["stage_summary"]["corr_false_phase_check"]["broken"] == 1
    assert summary["first_broken_stage_counts_by_path"] == {
        "corr_false": {"corr_false_phase_check": 1},
        "corr_true": {"never_broken": 1},
    }
    assert summary["phase_correction_effect_counts"] == {
        "fixes_broken_to_usable": 1
    }


def test_render_report_includes_path_level_breaks():
    payload = {
        "source": {
            "tool": "tools/diagnose_fit_stage_transitions.py",
            "truth": "tests/fixtures/full_corner_ground_truth.json",
            "manifest": "data/two_view_manifest_core.json",
            "max_image_dim": 1600,
            "run_selection": "single deterministic run per row",
        },
        "summary": {
            "n_rows": 1,
            "n_traced": 1,
            "stage_summary": {
                "affine_selected": {
                    "n": 1,
                    "usable": 1,
                    "marginal": 0,
                    "broken": 0,
                    "median_total_axis_misfit_deg": 5.0,
                },
            },
            "first_broken_stage_counts_by_path": {
                "corr_false": {"never_broken": 1},
                "corr_true": {"never_broken": 1},
            },
            "phase_correction_effect_counts": {"keeps_usable": 1},
        },
        "per_row": [{
            "key": "20_A",
            "status": "traced",
            "stages": [{
                "stage": "affine_selected",
                "axis_state": "usable",
                "axis_match": {"total_misfit_deg": 5.0},
                "delta_total_misfit_from_previous_deg": None,
                "vertex_error_processing_px": 12.0,
                "debug": {"approach": "affine_selected"},
            }],
        }],
    }

    report = d.render_report(payload)

    assert "# Fit stage transition diagnostic" in report
    assert "First broken stage counts by path" in report
    assert "`20_A`" in report
    assert "`affine_selected`" in report


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
    assert "# Fit stage transition diagnostic" in report.read_text(
        encoding="utf-8"
    )
