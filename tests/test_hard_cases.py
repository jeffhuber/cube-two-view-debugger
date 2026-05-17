from pathlib import Path
from types import SimpleNamespace

import tools.probe_hard_cases as probe_hard_cases
from tools.probe_hard_cases import (
    _format_repair_cell,
    grid_cell_diagnostics,
    load_manifest_document,
    option_coverage_for_analysis,
    repair_probe_for_analyses,
    target_failures,
)


def test_hard_case_manifest_records_open_issue_sets():
    document = load_manifest_document(Path(__file__).parent / "fixtures" / "hard_case_manifest.json")
    rows = {str(row["setId"]): row for row in document["pairs"]}
    primary = document["supportedArchitectures"]["primary"]

    assert set(rows) == {"17", "21", "22", "25", "30", "39", "44", "46", "47", "48", "49"}
    assert primary["label"] == "native-arm64-macos-python312"
    assert primary["platform.machine"] == "arm64"
    assert rows["17"]["targetFailedChecksPresent"] == [
        "red_orange_pair_calibration_suspected",
        "image_b_visible_face_evidence_weak",
    ]
    assert rows["21"]["targetStatus"] == "success"
    assert rows["21"]["expectedScoreOnceFixed"] == 54
    assert rows["22"]["targetStatus"] == "success"
    assert rows["22"]["expectedScoreOnceFixed"] == 54
    assert {rows[set_id]["linkedIssue"] for set_id in ("25", "30")} == {51}
    assert "targetFailedChecksAbsent" not in rows["25"]
    assert rows["25"]["targetFailedChecksPresent"] == ["image_b_face_triple_overlap_low_quality"]
    assert rows["30"]["targetFailedChecksAbsent"] == ["image_a_no_reliable_face_triple"]
    assert rows["30"]["currentStatus"] == "success"
    assert {rows[set_id]["linkedIssue"] for set_id in ("46", "47", "48", "49")} == {85}
    assert rows["46"]["failureClass"] == "background_sticker_noise"
    assert rows["46"]["currentCandidates"] == 14080
    assert rows["49"]["currentFailedChecks"] == [
        "no_legal_state",
        "background_sticker_noise_suspected",
    ]
    for set_id in ("46", "47", "48", "49"):
        assert rows[set_id]["targetStatus"] == "rejected"
        assert rows[set_id]["targetCategory"] == "reject_retake"
        assert rows[set_id]["targetFailedChecksPresent"] == ["background_sticker_noise_suspected"]
    assert rows["39"]["targetFailedChecksAbsent"] == [
        "image_a_no_reliable_face_triple",
        "image_b_D_anchor_missing",
        "missing_side_face_coverage",
    ]
    assert rows["39"]["currentFailedChecks"] == ["piece_legality_invalid"]
    for set_id in ("17", "21", "22", "44", "46", "47", "48", "49"):
        assert rows[set_id]["groundTruthPath"]
        assert rows[set_id]["groundTruth_sha256_expected"]
    for row in rows.values():
        assert row["imageA_sha256_expected"]
        assert row["imageB_sha256_expected"]
        assert row["failureClass"]
        assert "currentStatus" in row
        assert "currentCandidates" in row


def test_hard_case_target_failures_check_absent_failed_checks():
    row = {"targetFailedChecksAbsent": ["image_b_no_reliable_face_triple"]}
    payload = {"failedChecks": ["image_b_no_reliable_face_triple"]}

    assert target_failures(row, payload, input_drift=False) == [
        "target_check_still_present:image_b_no_reliable_face_triple"
    ]
    assert target_failures(row, {"failedChecks": ["piece_legality_invalid"]}, input_drift=False) == []


def test_hard_case_target_failures_check_present_failed_checks():
    row = {"targetFailedChecksPresent": ["red_orange_pair_calibration_suspected"]}
    payload = {"failedChecks": ["piece_legality_invalid"]}

    assert target_failures(row, payload, input_drift=False) == [
        "target_check_missing:red_orange_pair_calibration_suspected"
    ]
    assert target_failures(
        row,
        {"failedChecks": ["piece_legality_invalid", "red_orange_pair_calibration_suspected"]},
        input_drift=False,
    ) == []


def test_hard_case_target_failures_check_expected_score_once_fixed():
    row = {"expectedScoreOnceFixed": 54}

    assert target_failures(row, {"score": 53, "failedChecks": []}, input_drift=False) == [
        "score_below_expected_once_fixed:54"
    ]
    assert target_failures(row, {"score": 54, "failedChecks": []}, input_drift=False) == []


def test_grid_cell_diagnostics_records_rgb_and_alternatives():
    sticker = type(
        "Sticker",
        (),
        {
            "id": 7,
            "source": "component",
            "shape_angle": None,
            "rgb": (230, 220, 45),
            "match": type(
                "Match",
                (),
                {
                    "color": "yellow",
                    "face": "D",
                    "confidence": 0.81,
                    "alternatives": [("yellow", 1.25), ("white", 9.5)],
                },
            )(),
        },
    )()
    grid = type(
        "Grid",
        (),
        {
            "id": 3,
            "center_face": "D",
            "matched_count": 9,
            "fit_error": 0.25,
            "stickers": [[sticker for _ in range(3)] for _ in range(3)],
        },
    )()
    analysis = type("Analysis", (), {"grids": [grid]})()

    diagnostics = grid_cell_diagnostics(analysis, "D")

    cell = diagnostics["D"]["cells"][1][1]
    assert cell["rgb"] == [230, 220, 45]
    assert cell["color"] == "yellow"
    assert cell["face"] == "D"
    assert cell["alternatives"][0] == {"color": "yellow", "distance": 1.25}


def test_repair_probe_reports_direct_and_repair_candidates(monkeypatch):
    workset = SimpleNamespace(options_a=[object()], options_b=[object(), object()], merged_candidates=[])
    recognizer = SimpleNamespace(
        _state_candidates_from_workset=lambda candidate_workset: [
            ("R" * 54, 0.6, {"sidePairA": "B/R", "sidePairB": "F/L"}),
            ("U" * 54, 0.4, {"sidePairA": "F/R", "sidePairB": "B/L"}),
        ],
        _legal_repair_candidate_details_from_workset=lambda candidate_workset, *, release_merged_candidates: [
            {
                "state": "U" * 54,
                "confidence": 0.5,
                "repairCost": 12.5,
                "repairChanges": 4,
            }
        ],
    )

    def fake_validate_state(state):
        return SimpleNamespace(valid=state == "U" * 54, errors=[] if state == "U" * 54 else ["R_count_not_9"])

    monkeypatch.setattr(probe_hard_cases, "_white_up_checks", lambda analysis_a, analysis_b: [])
    monkeypatch.setattr(probe_hard_cases, "_recognition_workset", lambda analysis_a, analysis_b: workset)
    monkeypatch.setattr(probe_hard_cases, "validate_state", fake_validate_state)
    monkeypatch.setattr(
        probe_hard_cases,
        "_validation_failed_checks",
        lambda invalid_reasons, analysis_a, analysis_b: ["R_count_not_9", "red_orange_pair_calibration_suspected"],
    )

    probe = repair_probe_for_analyses(object(), object(), recognizer, expected_state="U" * 54)

    assert probe["status"] == "probed"
    assert probe["optionsA"] == 1
    assert probe["optionsB"] == 2
    assert probe["mergedCandidateCount"] == 0
    assert probe["directCandidateCount"] == 2
    assert probe["directLegalCount"] == 1
    assert probe["directFailedChecks"] == ["R_count_not_9", "red_orange_pair_calibration_suspected"]
    assert probe["topDirectCandidatesByScore"][0]["state"] == "U" * 54
    assert probe["topDirectCandidatesByScore"][0]["score"] == 54
    assert probe["topDirectCandidatesByScore"][0]["sidePairA"] == "F/R"
    assert probe["repairCandidateCount"] == 1
    assert probe["topRepairCandidates"][0]["repairCost"] == 12.5
    assert probe["topRepairCandidates"][0]["score"] == 54


def test_repair_probe_short_circuits_white_up_rejections(monkeypatch):
    monkeypatch.setattr(probe_hard_cases, "_white_up_checks", lambda analysis_a, analysis_b: ["image_b_D_anchor_missing"])

    probe = repair_probe_for_analyses(object(), object(), SimpleNamespace())

    assert probe["status"] == "white_up_rejected"
    assert probe["whiteUpChecks"] == ["image_b_D_anchor_missing"]


def test_option_coverage_scores_generated_faces_against_truth(monkeypatch):
    options = [
        {
            "U": [["U", "U", "U"], ["U", "U", "U"], ["U", "U", "U"]],
            "R": [["R", "R", "R"], ["R", "R", "R"], ["R", "R", "R"]],
            "_score": 42.0,
            "_selection_score": 20.0,
            "_orientation_score": 2.75,
            "_orientation_rank": 3,
            "_side_pair": ("B", "R"),
            "_ordered_side_pair": ("R", "B"),
        },
        {
            "U": [["U", "R", "U"], ["U", "U", "U"], ["U", "U", "U"]],
            "_score": 50.0,
            "_selection_score": 30.0,
            "_orientation_score": 1.0,
            "_orientation_rank": 0,
            "_side_pair": ("F", "R"),
            "_ordered_side_pair": ("F", "R"),
        },
    ]
    monkeypatch.setattr(probe_hard_cases, "_oriented_face_options", lambda analysis, anchor: options)

    coverage = option_coverage_for_analysis(object(), "U", "U" * 9 + "R" * 9 + "F" * 36)

    assert coverage["optionCount"] == 2
    assert coverage["faces"]["U"][0]["score"] == 9
    assert coverage["faces"]["U"][0]["optionRank"] == 0
    assert coverage["faces"]["U"][0]["sidePair"] == "B/R"
    assert coverage["faces"]["R"][0]["score"] == 9
    assert coverage["topOptions"][0]["score"] == 18
    assert coverage["topOptions"][0]["maxScore"] == 18
    assert coverage["topOptions"][0]["optionRank"] == 0
    assert coverage["topOptions"][0]["faces"]["U"]["score"] == 9
    assert coverage["topOptions"][0]["faces"]["R"]["state"] == "RRRRRRRRR"
    assert coverage["topOptions"][1]["score"] == 8


def test_format_repair_cell_distinguishes_path_state():
    """The probe's `Repair` column must let an operator tell apart, at a
    glance: no repair attempted, standard repair fired, backfill fired
    and used, backfill fired but empty. Pins each rendering — these
    strings show up in the human-readable hard-case output and in
    follow-up audits of the PR #76 conflict-backfill path."""
    # No recognitionSignals at all (e.g., pre-PR-76 saved run).
    assert _format_repair_cell(None) == ""
    assert _format_repair_cell({}) == ""

    # Repair never attempted (most rejects fall here).
    repair_none = {
        "pathUsed": False,
        "candidateCount": 0,
        "backfillAttempted": False,
        "backfillEvaluatedMerges": None,
        "backfillUsed": None,
    }
    assert _format_repair_cell(repair_none) == "none"

    # Standard repair path: backfill gate did NOT fire.
    repair_standard = {
        "pathUsed": True,
        "candidateCount": 8,
        "backfillAttempted": False,
        "backfillEvaluatedMerges": None,
        "backfillUsed": None,
    }
    assert _format_repair_cell(repair_standard) == "std/8"

    # Backfill fired and result used (Set 22 territory).
    repair_backfill_used = {
        "pathUsed": True,
        "candidateCount": 5,
        "backfillAttempted": True,
        "backfillEvaluatedMerges": 40,
        "backfillUsed": True,
    }
    assert _format_repair_cell(repair_backfill_used) == "bf/40/used"

    # Backfill gate fired but produced zero usable candidates. This is the
    # state PR #76 made distinguishable from "gate didn't fire" via the
    # `repairBackfillAttempted` audit field — pin the rendering.
    repair_backfill_empty = {
        "pathUsed": False,
        "candidateCount": 0,
        "backfillAttempted": True,
        "backfillEvaluatedMerges": 0,
        "backfillUsed": False,
    }
    assert _format_repair_cell(repair_backfill_empty) == "bf/0/empty"
