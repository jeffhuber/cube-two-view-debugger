"""Unit tests for the axis-ground-truth scorer."""
import importlib.util
import math
import sys
from pathlib import Path

# Import the script as a module (it's a tool, not in a package)
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "evaluate_axis_ground_truth",
    ROOT / "tools" / "evaluate_axis_ground_truth.py",
)
mod = importlib.util.module_from_spec(spec)
sys.modules["evaluate_axis_ground_truth"] = mod
spec.loader.exec_module(mod)


def _truth_solved_cube_corner():
    """A canonical "solved cube" labeled position: vertex at origin,
    3 axes at 30, 150, 270 degrees (standard iso projection layout)."""
    L = 100.0
    return {
        "vertex": [500.0, 500.0],
        "near_x": [500.0 + L * math.cos(math.radians(30)), 500.0 + L * math.sin(math.radians(30))],
        "near_y": [500.0 + L * math.cos(math.radians(150)), 500.0 + L * math.sin(math.radians(150))],
        "near_z": [500.0 + L * math.cos(math.radians(270)), 500.0 + L * math.sin(math.radians(270))],
        "approved": True,
    }


def test_perfect_match_scores_zero():
    truth = _truth_solved_cube_corner()
    candidate = {k: v for k, v in truth.items() if k != "approved"}
    r = mod._score_pair(truth, candidate)
    assert r["vertex_error_px"] == 0.0
    assert all(a == 0.0 for a in r["axis_angle_errors_deg"])
    assert all(l == 0.0 for l in r["axis_length_errors_px"])
    assert r["composite_score"] == 0.0


def test_vertex_offset_only_scores_vertex_distance():
    truth = _truth_solved_cube_corner()
    candidate = {k: v[:] if isinstance(v, list) else v for k, v in truth.items() if k != "approved"}
    candidate["vertex"] = [510.0, 530.0]  # 30+ px offset
    candidate["near_x"] = [510.0 + 100 * math.cos(math.radians(30)), 530.0 + 100 * math.sin(math.radians(30))]
    candidate["near_y"] = [510.0 + 100 * math.cos(math.radians(150)), 530.0 + 100 * math.sin(math.radians(150))]
    candidate["near_z"] = [510.0 + 100 * math.cos(math.radians(270)), 530.0 + 100 * math.sin(math.radians(270))]
    r = mod._score_pair(truth, candidate)
    assert abs(r["vertex_error_px"] - math.hypot(10, 30)) < 0.1
    # Axes are still parallel and same length → angles 0, lengths 0
    assert all(a < 0.5 for a in r["axis_angle_errors_deg"])


def test_axis_rotation_detected():
    truth = _truth_solved_cube_corner()
    L = 100.0
    candidate = {
        "vertex": truth["vertex"][:],
        "near_x": [500.0 + L * math.cos(math.radians(40)), 500.0 + L * math.sin(math.radians(40))],   # 10 deg off
        "near_y": [500.0 + L * math.cos(math.radians(150)), 500.0 + L * math.sin(math.radians(150))],
        "near_z": [500.0 + L * math.cos(math.radians(270)), 500.0 + L * math.sin(math.radians(270))],
    }
    r = mod._score_pair(truth, candidate)
    # One axis is 10 deg off, others 0
    angles = r["axis_angle_errors_deg"]
    assert sum(1 for a in angles if abs(a - 10.0) < 0.5) == 1
    assert sum(1 for a in angles if a < 0.5) == 2


def test_axis_permutation_assigned_optimally():
    """If candidate axes are in a different order, the scorer should
    find the best assignment automatically."""
    truth = _truth_solved_cube_corner()
    candidate = {
        "vertex": truth["vertex"][:],
        "near_x": truth["near_z"][:],  # candidate "x" is actually true "z"
        "near_y": truth["near_x"][:],
        "near_z": truth["near_y"][:],
    }
    r = mod._score_pair(truth, candidate)
    # After best permutation, all axes line up → all errors ~0
    assert all(a < 0.5 for a in r["axis_angle_errors_deg"])
    assert r["permutation"] == [1, 2, 0]  # candidate idx that matches each truth idx


def test_axis_length_error_detected():
    truth = _truth_solved_cube_corner()
    L = 100.0
    candidate = {
        "vertex": truth["vertex"][:],
        "near_x": [500.0 + 150 * math.cos(math.radians(30)), 500.0 + 150 * math.sin(math.radians(30))],  # 50 px longer
        "near_y": [500.0 + L * math.cos(math.radians(150)), 500.0 + L * math.sin(math.radians(150))],
        "near_z": [500.0 + L * math.cos(math.radians(270)), 500.0 + L * math.sin(math.radians(270))],
    }
    r = mod._score_pair(truth, candidate)
    # One axis length is 50 px longer
    lengths = r["axis_length_errors_px"]
    assert any(abs(l - 50.0) < 0.5 for l in lengths)


def test_axes_displacement_schema_accepted():
    """Candidate JSON can use `axes` (displacement vectors) instead of
    explicit near_x/near_y/near_z corner positions."""
    truth = _truth_solved_cube_corner()
    L = 100.0
    candidate = {
        "vertex": truth["vertex"][:],
        "axes": [
            [L * math.cos(math.radians(30)), L * math.sin(math.radians(30))],
            [L * math.cos(math.radians(150)), L * math.sin(math.radians(150))],
            [L * math.cos(math.radians(270)), L * math.sin(math.radians(270))],
        ],
    }
    r = mod._score_pair(truth, candidate)
    assert r["vertex_error_px"] == 0.0
    assert all(a < 0.5 for a in r["axis_angle_errors_deg"])


def test_normalize_angle_diff_wraps_correctly():
    # 350 deg apparent → -10 deg actual signed difference
    assert abs(mod._normalize_angle_diff(350) - (-10)) < 1e-6
    assert abs(mod._normalize_angle_diff(-350) - 10) < 1e-6
    assert abs(mod._normalize_angle_diff(180) - 180) < 1e-6
    assert abs(mod._normalize_angle_diff(-180) - 180) < 1e-6
