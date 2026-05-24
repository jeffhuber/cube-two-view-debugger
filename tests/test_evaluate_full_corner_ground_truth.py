from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.baseline_full_corner_global_model import (  # noqa: E402
    _scale_candidate,
    _select_representative_row,
)
from tools.evaluate_full_corner_ground_truth import evaluate, score_case  # noqa: E402


FIXTURE = REPO_ROOT / "tests" / "fixtures" / "full_corner_ground_truth.json"


def _row(key: str):
    return json.loads(FIXTURE.read_text(encoding="utf-8"))[key]


def test_perfect_full_corner_candidate_scores_good_with_exact_errors():
    truth = _row("20_A")
    candidate = copy.deepcopy(truth)

    score = score_case("20_A", truth, candidate)

    assert score["category"] == "GOOD"
    assert score["vertex_error_px"] == 0.0
    assert score["one_edge"]["mean_angle_error_deg"] == 0.0
    assert score["far"]["mean_angle_error_deg"] == 0.0
    assert score["swapped_mean_angle_error_deg"] > 20.0
    assert score["exact_full_corner"]["max_corner_error_px"] == 0.0


def test_triplet_candidate_does_not_require_exact_corner_identity():
    truth = _row("20_A")
    candidate = {
        "vertex": truth["vertex"],
        # A one-edge truth is corners 1,3,5. Shuffle them; best-permutation
        # matching should still score as exact.
        "one_edge": [truth["corner_5"], truth["corner_1"], truth["corner_3"]],
        "far": [truth["corner_4"], truth["corner_0"], truth["corner_2"]],
    }

    score = score_case("20_A", truth, candidate)

    assert score["category"] == "GOOD"
    assert score["one_edge"]["mean_angle_error_deg"] == 0.0
    assert score["far"]["mean_angle_error_deg"] == 0.0
    assert "exact_full_corner" not in score


def test_phase_swapped_triplets_are_categorized_canonically():
    truth = _row("20_A")
    candidate = {
        "vertex": truth["vertex"],
        "one_edge": [truth["corner_0"], truth["corner_2"], truth["corner_4"]],
        "far": [truth["corner_1"], truth["corner_3"], truth["corner_5"]],
    }

    score = score_case("20_A", truth, candidate)

    assert score["category"] == "PHASE_SWAPPED"
    assert score["one_edge"]["mean_angle_error_deg"] > 20.0
    assert score["far"]["mean_angle_error_deg"] > 20.0
    assert score["swapped_mean_angle_error_deg"] == 0.0


def test_triplet_angles_are_measured_from_candidate_vertex():
    truth = _row("20_A")
    shifted = copy.deepcopy(truth)
    shift = (100.0, -50.0)
    for name in ("vertex", "corner_0", "corner_1", "corner_2", "corner_3", "corner_4", "corner_5"):
        shifted[name] = [truth[name][0] + shift[0], truth[name][1] + shift[1]]

    score = score_case("20_A", truth, shifted)

    assert score["category"] == "GOOD"
    assert score["vertex_error_px"] > 100.0
    assert score["one_edge"]["mean_angle_error_deg"] == 0.0
    assert score["far"]["mean_angle_error_deg"] == 0.0


def test_side_b_one_edge_and_far_sets_are_opposite_side_a():
    truth = _row("20_B")
    candidate = {
        "vertex": truth["vertex"],
        # B one-edge truth is corners 0,2,4.
        "one_edge": [truth["corner_4"], truth["corner_0"], truth["corner_2"]],
        "far": [truth["corner_5"], truth["corner_1"], truth["corner_3"]],
    }

    score = score_case("20_B", truth, candidate)

    assert score["category"] == "GOOD"
    assert score["one_edge"]["mean_angle_error_deg"] == 0.0
    assert score["far"]["mean_angle_error_deg"] == 0.0


def test_evaluate_summary_marks_missing_candidates():
    truth = {"20_A": _row("20_A"), "20_B": _row("20_B")}
    candidates = {"20_A": copy.deepcopy(truth["20_A"])}

    payload = evaluate(truth, candidates)

    assert payload["schema"] == "canonical_full_corner_eval_v1"
    assert payload["summary"]["n_rows"] == 2
    assert payload["summary"]["n_scored"] == 1
    assert payload["by_case"]["20_B"]["status"] == "missing_candidate"


def test_scale_candidate_returns_predictions_to_original_coords():
    candidate = {
        "vertex": [10.0, 20.0],
        "one_edge": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "far": [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        "visible_corners": {
            "h_x": [1.0, 2.0],
            "h_y": [3.0, 4.0],
            "h_z": [5.0, 6.0],
            "h_xy": [7.0, 8.0],
            "h_xz": [9.0, 10.0],
            "h_yz": [11.0, 12.0],
        },
    }

    scaled = _scale_candidate(candidate, 2.0)

    assert scaled["vertex"] == [20.0, 40.0]
    assert scaled["one_edge"][2] == [10.0, 12.0]
    assert scaled["far"][0] == [14.0, 16.0]
    assert scaled["visible_corners"]["h_xz"] == [18.0, 20.0]
    assert scaled["processing_scale"] == 0.5


def test_baseline_representative_row_balances_aligned_and_swapped_phase_errors():
    lucky_aligned_outlier = {
        "key": "20_A",
        "status": "scored",
        "one_edge": {"mean_angle_error_deg": 4.0},
        "far": {"mean_angle_error_deg": 50.0},
        "swapped_mean_angle_error_deg": 40.0,
    }
    representative_swapped = {
        "key": "20_A",
        "status": "scored",
        "one_edge": {"mean_angle_error_deg": 58.0},
        "far": {"mean_angle_error_deg": 60.0},
        "swapped_mean_angle_error_deg": 3.0,
    }

    selected = _select_representative_row([lucky_aligned_outlier, representative_swapped])

    assert selected is representative_swapped
