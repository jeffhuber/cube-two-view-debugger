from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import diagnose_procrustes_correspondence as d  # noqa: E402
from tools.global_cube_model import _TEMPLATE_HEXAGON_2D_ISO  # noqa: E402


def _synthetic_truth_a() -> tuple[str, dict]:
    key = "99_A"
    origin = np.array([100.0, 200.0])
    scale = 120.0

    def pt(name: str) -> list[float]:
        xy = origin + scale * np.array(_TEMPLATE_HEXAGON_2D_ISO[name])
        return [float(xy[0]), float(xy[1])]

    truth = {
        "approved": True,
        "vertex": [float(origin[0]), float(origin[1])],
        # Side A one-edge corners are 1, 3, 5. Triplet scoring is
        # best-permutation matched, so the exact h_x/h_y/h_z identity is
        # not load-bearing for this diagnostic.
        "corner_1": pt("h_x"),
        "corner_3": pt("h_y"),
        "corner_5": pt("h_z"),
        "corner_0": pt("h_xy"),
        "corner_2": pt("h_xz"),
        "corner_4": pt("h_yz"),
    }
    return key, truth


def test_diagnose_from_hexagon_ranks_perfect_canonical_assignment_first():
    key, truth = _synthetic_truth_a()
    hexagon = [
        tuple(_TEMPLATE_HEXAGON_2D_ISO[name])
        for name in d.TEMPLATE_KEYS
    ]

    record = d.diagnose_from_hexagon(key, truth, hexagon, processing_scale=1.0)

    assert record["status"] == "traced"
    summary = record["summary"]
    assert summary["n_scored_permutations"] == 720
    assert summary["diagnosis"] == "residual_selects_canonical"
    assert summary["selected_by_residual"]["category"] == "GOOD"
    assert summary["selected_by_residual"]["one_edge_total_axis_misfit_deg"] < 2.0
    assert summary["selected_by_residual"]["residual_rank"] == 1
    assert summary["best_canonical_by_residual"]["residual_rank"] == 1
    assert summary["best_axis_by_misfit"]["residual_rank"] == 1
    assert len(record["permutations"]) == 720


def test_summarize_row_identifies_canonical_available_but_outranked():
    records = [
        {
            "status": "scored",
            "perm": [0, 1, 2, 3, 4, 5],
            "residual_rank": 1,
            "residual_rms_px": 1.0,
            "residual_px2": 1.0,
            "category": "PHASE_SWAPPED",
            "one_edge_total_axis_misfit_deg": 174.0,
            "aligned_mean_angle_deg": 58.0,
            "swapped_mean_angle_deg": 2.0,
            "one_edge_mean_angle_deg": 59.0,
            "far_mean_angle_deg": 57.0,
            "detected_index_by_template": {},
        },
        {
            "status": "scored",
            "perm": [1, 2, 3, 4, 5, 0],
            "residual_rank": 2,
            "residual_rms_px": 1.2,
            "residual_px2": 1.44,
            "category": "GOOD",
            "one_edge_total_axis_misfit_deg": 9.0,
            "aligned_mean_angle_deg": 3.0,
            "swapped_mean_angle_deg": 55.0,
            "one_edge_mean_angle_deg": 2.0,
            "far_mean_angle_deg": 4.0,
            "detected_index_by_template": {},
        },
    ]

    summary = d.summarize_row(records)

    assert summary["diagnosis"] == "canonical_available_but_outranked"
    assert summary["selected_by_residual"]["category"] == "PHASE_SWAPPED"
    assert summary["best_canonical_by_residual"]["category"] == "GOOD"
    assert summary["best_axis_by_misfit"]["category"] == "GOOD"
    assert summary["canonical_residual_rms_gap_px"] == 0.2
    assert summary["best_axis_residual_rms_gap_px"] == 0.2


def test_render_report_includes_diagnosis_table():
    payload = {
        "source": {
            "tool": "tools/diagnose_procrustes_correspondence.py",
            "truth": "tests/fixtures/full_corner_ground_truth.json",
            "max_image_dim": 1600,
            "rows_glob": "20_*",
            "search": "all 720 detected-hexagon-to-template permutations",
            "selection_metric": "minimum affine residual",
        },
        "summary": {
            "n_rows": 1,
            "n_traced": 1,
            "diagnosis_counts": {"canonical_available_but_outranked": 1},
            "selected_category_counts": {"PHASE_SWAPPED": 1},
            "median_canonical_residual_rms_gap_px": 0.2,
        },
        "per_row": [
            {
                "key": "20_A",
                "status": "traced",
                "summary": {
                    "diagnosis": "canonical_available_but_outranked",
                    "canonical_residual_rms_gap_px": 0.2,
                    "selected_by_residual": {
                        "category": "PHASE_SWAPPED",
                        "residual_rms_px": 1.0,
                        "one_edge_total_axis_misfit_deg": 174.0,
                    },
                    "best_canonical_by_residual": {
                        "residual_rank": 2,
                        "aligned_mean_angle_deg": 3.0,
                    },
                    "best_axis_by_misfit": {
                        "residual_rank": 2,
                        "one_edge_total_axis_misfit_deg": 9.0,
                    },
                    "best_axis_residual_rms_gap_px": 0.2,
                },
            }
        ],
    }

    md = d.render_report(payload)

    assert "# Procrustes correspondence diagnostic" in md
    assert "`20_A`" in md
    assert "canonical_available_but_outranked" in md


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
    assert "# Procrustes correspondence diagnostic" in report.read_text(
        encoding="utf-8"
    )
