"""Unit tests for `tools/probe_center_color_phase_metric.py`.

Two load-bearing pieces are pinned: the slot-permutation function
`_cycle_faces` (must be a true 3-cycle on the 3 visible faces, with
shift=0 identity / shift=1 one rotation / shift=2 the other) and the
end-to-end evaluator (must score the identity hypothesis lower than the
two cyclic permutations on synthetic perfectly-canonical centers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import CANONICAL_RGB, rgb_to_lab  # noqa: E402
from tools import probe_center_color_phase_metric as p  # noqa: E402


# --------------- _cycle_faces pinning ---------------


def test_cycle_faces_shift_zero_is_identity():
    assert p._cycle_faces(("U", "R", "F"), 0) == ("U", "R", "F")


def test_cycle_faces_shift_one_rotates_one_position():
    """shift=1 takes (a,b,c) -> (b,c,a) — slot 0 now holds the face that
    was at slot 1, etc. This corresponds to one of the two 120° rotations
    around the body diagonal."""
    assert p._cycle_faces(("U", "R", "F"), 1) == ("R", "F", "U")


def test_cycle_faces_shift_two_rotates_other_direction():
    assert p._cycle_faces(("U", "R", "F"), 2) == ("F", "U", "R")


def test_cycle_faces_shift_three_back_to_identity():
    """Three 120° rotations = 360° = identity."""
    assert p._cycle_faces(("U", "R", "F"), 3) == ("U", "R", "F")


def test_cycle_faces_works_on_b_side_faces():
    """B side has visible faces (D, L, B) — sanity-check the cycle works
    independently of the specific face labels."""
    assert p._cycle_faces(("D", "B", "L"), 1) == ("B", "L", "D")


# --------------- _score_hypothesis ---------------


def test_score_hypothesis_zero_when_center_matches_canonical_exactly():
    """A center sample whose Lab matches canonical exactly should score
    zero against the correct face label."""
    canonical_u_lab = rgb_to_lab(CANONICAL_RGB["white"])
    canonical_r_lab = rgb_to_lab(CANONICAL_RGB["red"])
    canonical_f_lab = rgb_to_lab(CANONICAL_RGB["green"])
    centers = [
        p.CenterSample("upper", "U", canonical_u_lab, (0, 0, 0)),
        p.CenterSample("right", "R", canonical_r_lab, (0, 0, 0)),
        p.CenterSample("front", "F", canonical_f_lab, (0, 0, 0)),
    ]
    score = p._score_hypothesis(centers, ("U", "R", "F"))
    assert score == pytest.approx(0.0, abs=1e-9)


def test_score_hypothesis_huge_when_centers_assigned_to_wrong_faces():
    """The same centers assigned to a cyclic-shifted face labeling
    should produce a much larger score (because the per-center Lab is
    nowhere near the wrong-face canonical)."""
    canonical_u_lab = rgb_to_lab(CANONICAL_RGB["white"])
    canonical_r_lab = rgb_to_lab(CANONICAL_RGB["red"])
    canonical_f_lab = rgb_to_lab(CANONICAL_RGB["green"])
    centers = [
        p.CenterSample("upper", "U", canonical_u_lab, (0, 0, 0)),
        p.CenterSample("right", "R", canonical_r_lab, (0, 0, 0)),
        p.CenterSample("front", "F", canonical_f_lab, (0, 0, 0)),
    ]
    score_identity = p._score_hypothesis(centers, ("U", "R", "F"))
    score_cyclic = p._score_hypothesis(centers, ("R", "F", "U"))
    assert score_identity == pytest.approx(0.0, abs=1e-9)
    assert score_cyclic > 100.0, (
        "cyclic mis-assignment should accumulate >100 dE units of "
        f"distance (got {score_cyclic}). If this fires, either the "
        "canonical palette has degenerate colors or the Lab function "
        "is broken."
    )


# --------------- end-to-end on synthetic row ---------------


def _synthetic_row(side: str, yaw: int, perturb_lab: float = 0.0):
    """Build a fake oracle row record where the 3 centers are EXACTLY
    canonical for their oracle-labeled face (optionally perturbed by
    `perturb_lab` units in each Lab channel)."""
    from tools.corner_conventions import wca_face_by_slot
    face_by_slot = wca_face_by_slot(side, yaw)

    def _face_record(slot):
        face = face_by_slot[slot]
        canonical_color_name = {
            "U": "white", "D": "yellow",
            "R": "red", "L": "orange",
            "F": "green", "B": "blue",
        }[face]
        canonical_lab = rgb_to_lab(CANONICAL_RGB[canonical_color_name])
        perturbed = tuple(v + perturb_lab for v in canonical_lab)
        # Build a 9-sticker face record where sticker_id=5 is the center.
        stickers = []
        for sid in range(1, 10):
            stickers.append({
                "row": (sid - 1) // 3,
                "col": (sid - 1) % 3,
                "sticker_id": sid,
                "facelet_id": f"{face}{sid}",
                "rgb": [0, 0, 0],
                "lab": list(perturbed if sid == 5 else (0.0, 0.0, 0.0)),
            })
        return {
            "slot": slot, "wca_face": face,
            "stickers": stickers,
        }

    return {
        "key": f"99_{side}",
        "side": side,
        "yaw_quarter_turns": yaw,
        "faces": [
            _face_record("upper"),
            _face_record("right"),
            _face_record("front"),
        ],
    }


def test_evaluate_row_identity_wins_strictly_on_clean_canonical_a_yaw_zero():
    row = _synthetic_row("A", 0)
    result = p.evaluate_row(row)
    assert result.winning_hypothesis == "identity"
    assert result.winning_score == pytest.approx(0.0, abs=1e-6)
    assert result.runner_up_score > 100.0
    assert result.margin > 100.0
    assert result.oracle_face_assignment == ("U", "R", "F")


def test_evaluate_row_identity_wins_strictly_on_clean_canonical_b_yaw_zero():
    row = _synthetic_row("B", 0)
    result = p.evaluate_row(row)
    assert result.winning_hypothesis == "identity"
    assert result.winning_score == pytest.approx(0.0, abs=1e-6)
    # B side at yaw=0: oracle assigns (D, B, L) to (upper, right, front).
    assert result.oracle_face_assignment == ("D", "B", "L")


@pytest.mark.parametrize("yaw", [0, 1, 2, 3])
def test_evaluate_row_identity_wins_on_clean_canonical_across_all_yaws(yaw):
    """The metric should be yaw-invariant: whatever 3 faces are visible
    at this yaw, the identity hypothesis should win when the centers
    match canonical colors."""
    for side in ("A", "B"):
        row = _synthetic_row(side, yaw)
        result = p.evaluate_row(row)
        assert result.winning_hypothesis == "identity", (
            f"identity should win on side {side} yaw={yaw}; "
            f"actual winner = {result.winning_hypothesis}"
        )


def test_evaluate_row_still_picks_identity_under_modest_perturbation():
    """Real centers won't be exactly canonical — there's wear, lighting,
    and rectification quantization. The metric must remain robust under
    perturbation of ±5 dE per channel (a reasonable real-world tolerance,
    well below the typical >50-dE pairwise canonical separation)."""
    for side in ("A", "B"):
        row = _synthetic_row(side, 0, perturb_lab=5.0)
        result = p.evaluate_row(row)
        assert result.winning_hypothesis == "identity"


# --------------- error cases ---------------


def test_evaluate_row_missing_slot_raises():
    """Defensive: a malformed oracle row missing one of the 3 expected
    slots should raise rather than silently emit a wrong result."""
    row = _synthetic_row("A", 0)
    row["faces"] = [f for f in row["faces"] if f["slot"] != "right"]
    with pytest.raises(ValueError, match="missing"):
        p.evaluate_row(row)


# --------------- end-to-end via main() on a temp index ---------------


def test_main_writes_outputs_and_returns_success(tmp_path: Path):
    """The CLI should consume an oracle-shaped index.json, emit the
    summary JSON + markdown report, and exit cleanly."""
    rows = [_synthetic_row("A", 0), _synthetic_row("B", 0)]
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"rows": rows}))
    out_json = tmp_path / "trace.json"
    out_md = tmp_path / "report.md"
    rc = p.main([
        "--index", str(index_path),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["schema"] == "center_color_phase_metric_v1"
    assert payload["n_rows"] == 2
    assert payload["n_identity_wins"] == 2
    md = out_md.read_text()
    assert "identity` wins on **2/2**" in md
    assert "Metric is sound" in md


def test_main_errors_when_index_missing(tmp_path: Path):
    rc = p.main([
        "--index", str(tmp_path / "nope.json"),
        "--out-json", str(tmp_path / "trace.json"),
        "--out-md", str(tmp_path / "report.md"),
    ])
    assert rc == 1
