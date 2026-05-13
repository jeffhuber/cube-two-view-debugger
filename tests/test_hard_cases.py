from pathlib import Path

from tools.probe_hard_cases import grid_cell_diagnostics, load_manifest_document, target_failures


def test_hard_case_manifest_records_open_issue_sets():
    document = load_manifest_document(Path(__file__).parent / "fixtures" / "hard_case_manifest.json")
    rows = {str(row["setId"]): row for row in document["pairs"]}
    primary = document["supportedArchitectures"]["primary"]

    assert set(rows) == {"17", "21", "22", "25", "30", "39", "44"}
    assert primary["label"] == "native-arm64-macos-python312"
    assert primary["platform.machine"] == "arm64"
    assert rows["17"]["targetFailedChecksPresent"] == ["red_orange_pair_calibration_suspected"]
    assert rows["21"]["targetFailedChecksPresent"] == ["red_orange_pair_calibration_suspected"]
    assert rows["22"]["targetFailedChecksPresent"] == ["red_orange_pair_calibration_suspected"]
    assert {rows[set_id]["linkedIssue"] for set_id in ("25", "30")} == {51}
    assert "targetFailedChecksAbsent" not in rows["25"]
    assert rows["30"]["targetFailedChecksAbsent"] == ["image_a_no_reliable_face_triple"]
    assert rows["30"]["currentStatus"] == "success"
    assert rows["39"]["targetFailedChecksAbsent"] == [
        "image_a_no_reliable_face_triple",
        "image_b_D_anchor_missing",
        "missing_side_face_coverage",
    ]
    assert rows["39"]["currentFailedChecks"] == ["piece_legality_invalid"]
    assert rows["44"]["groundTruthPath"]
    assert rows["44"]["groundTruth_sha256_expected"]
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
