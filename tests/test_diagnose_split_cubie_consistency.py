from __future__ import annotations

import json
from pathlib import Path

from tools.diagnose_split_cubie_consistency import (
    ALL_CUBIES,
    SPLIT_CORNERS,
    SPLIT_EDGES,
    VALID_CORNER_COLORSETS,
    VALID_EDGE_COLORSETS,
    analyze_legal_repair_json,
    check_cubie,
    check_state,
    state_diff_indices,
)


SOLVED_STATE = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9


def test_inventory_has_correct_split_counts():
    # 6 split corners, 6 split edges per the geometry
    assert len(SPLIT_CORNERS) == 6
    assert len(SPLIT_EDGES) == 6
    split_names = {c.name for c in SPLIT_CORNERS}
    assert split_names == {"UFL", "ULB", "UBR", "DFR", "DLF", "DRB"}
    split_edge_names = {e.name for e in SPLIT_EDGES}
    assert split_edge_names == {"UL", "UB", "DR", "DF", "FL", "BR"}


def test_inventory_has_20_cubies_total():
    corners = [c for c in ALL_CUBIES if c.kind == "corner"]
    edges = [c for c in ALL_CUBIES if c.kind == "edge"]
    assert len(corners) == 8
    assert len(edges) == 12


def test_valid_pools_match_expected_sizes():
    # 8 distinct corner colorsets, 12 distinct edge colorsets
    assert len(VALID_CORNER_COLORSETS) == 8
    assert len(VALID_EDGE_COLORSETS) == 12


def test_solved_state_has_all_cubies_consistent():
    result = check_state(SOLVED_STATE)
    assert result["totalCubies"] == 20
    assert result["consistentCount"] == 20
    assert result["inconsistentCount"] == 0
    assert result["inconsistentNames"] == []


def test_state_diff_indices_is_symmetric_and_sorted():
    a = "U" * 54
    b = "U" * 27 + "D" + "U" * 26
    assert state_diff_indices(a, b) == [27]
    assert state_diff_indices(b, a) == [27]


def test_check_cubie_detects_invalid_corner_triple():
    # Solved state at the URF corner has colors (U, R, F) which IS valid
    urf_cubie = next(c for c in ALL_CUBIES if c.name == "URF")
    report = check_cubie(SOLVED_STATE, urf_cubie)
    assert report["valid"] is True
    assert report["observed_colors"] == ["U", "R", "F"]

    # Corrupt one sticker (index 20, which is F[0,2] = URF's F sticker)
    # to a color that breaks the corner: (U, R, D) is not a valid corner.
    corrupted = SOLVED_STATE[:20] + "D" + SOLVED_STATE[21:]
    report = check_cubie(corrupted, urf_cubie)
    assert report["valid"] is False
    assert report["observed_colors"] == ["U", "R", "D"]


def test_analyze_legal_repair_json_handles_real_fixture():
    """Smoke-test the JSON analyzer against the checked-in summary fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "split_cubie_consistency_summary.json"
    payload = json.loads(fixture_path.read_text())

    # The fixture was generated from the 20 fresh-GT rows in #344
    assert payload["schema"] == "split_cubie_consistency_diagnostic_v1"
    assert payload["summary"]["rowCount"] == 20

    # Empirical findings from the report (these are the Phase 1 headline)
    summary = payload["summary"]
    assert summary["withSplitCubieInconsistency"] == []
    assert "11" in summary["withInImageCubieInconsistencyOnly"]
    assert "59" in summary["withNoCubieInconsistency"]
    # Exactly 2 rows needed repair (Sets 11 and 59)
    assert summary["needsRepairCount"] == 2


def test_set_11_row_has_expected_failure_shape():
    fixture_path = Path(__file__).parent / "fixtures" / "split_cubie_consistency_summary.json"
    payload = json.loads(fixture_path.read_text())
    set11 = next(r for r in payload["rows"] if r["setId"] == "11")

    # The headline finding: cc has 2 inconsistent in-image cubies
    cc_check = set11["canonicalCount"]["cubieConsistency"]
    assert cc_check["inconsistentCount"] == 2
    assert cc_check["inconsistentSplitCount"] == 0
    assert cc_check["inconsistentInImageCount"] == 2

    # The two inconsistent cubies are URF and DBL
    assert set(cc_check["inconsistentNames"]) == {"URF", "DBL"}

    # Broad legal recovers to a fully consistent state
    bl_check = set11["broadLegal"]["cubieConsistency"]
    assert bl_check["inconsistentCount"] == 0

    # The repairChanges overcount finding: reported=6, true delta=2
    assert set11["broadLegal"]["reportedRepairChanges"] == 6
    assert set11["broadLegal"]["trueStateDeltaFromCanonical"]["count"] == 2


def test_labeled_corpus_fixture_pins_combined_corpus_findings():
    """Labeled corpus sweep: 46 rows, 3 cc-failures (14, 65, 69), ALL split-cubie.

    Combined with the fresh-corpus fixture (Set 11 in-image, Set 59 parity),
    the headline becomes: 4/5 cc-failures caught by whole-cube cubie consistency,
    3/5 caught by split-cubie specifically. This pins the corrected verdict
    that the original draft report missed by only analyzing the fresh corpus.
    """
    fixture_path = Path(__file__).parent / "fixtures" / "split_cubie_consistency_labeled_corpus.json"
    payload = json.loads(fixture_path.read_text())
    assert payload["summary"]["rowCount"] == 46
    # All 3 cc-failures are in split cubies (no in-image, no parity-only)
    assert sorted(payload["summary"]["withSplitCubieInconsistency"]) == ["14", "65", "69"]
    assert payload["summary"]["withInImageCubieInconsistencyOnly"] == []
    assert payload["summary"]["withNoCubieInconsistency"] == []


def test_state_delta_gate_cleanly_separates_rescues_from_danger():
    """The corrected state_delta finding: gate of state_delta <= 4 admits
    rescue rows (65, 69, 11) and rejects the danger row (14). This is the
    structural argument for switching the gate semantic away from
    repairChanges-against-raw-observations."""
    labeled = json.loads(
        (Path(__file__).parent / "fixtures" / "split_cubie_consistency_labeled_corpus.json").read_text()
    )
    fresh = json.loads(
        (Path(__file__).parent / "fixtures" / "split_cubie_consistency_summary.json").read_text()
    )

    rows_by_set = {r["setId"]: r for r in labeled["rows"]}
    rows_by_set.update({r["setId"]: r for r in fresh["rows"]})

    # Set 14 (danger, must reject) has state_delta > 4
    assert rows_by_set["14"]["broadLegal"]["trueStateDeltaFromCanonical"]["count"] == 6

    # Rescue rows (must admit) all have state_delta <= 4
    for setid in ("65", "69", "11"):
        assert rows_by_set[setid]["broadLegal"]["trueStateDeltaFromCanonical"]["count"] <= 4, (
            f"Set {setid} state_delta {rows_by_set[setid]['broadLegal']['trueStateDeltaFromCanonical']['count']}"
            f" should be <= 4 for a state_delta gate to admit it"
        )
