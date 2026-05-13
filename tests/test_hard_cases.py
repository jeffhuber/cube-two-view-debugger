from pathlib import Path

from tools.probe_hard_cases import load_manifest_document, target_failures


def test_hard_case_manifest_records_open_issue_sets():
    document = load_manifest_document(Path(__file__).parent / "fixtures" / "hard_case_manifest.json")
    rows = {str(row["setId"]): row for row in document["pairs"]}

    assert set(rows) == {"17", "21", "22", "25", "30", "39", "44"}
    assert {rows[set_id]["issue"] for set_id in ("25", "30")} == {51}
    assert "targetFailedChecksAbsent" not in rows["25"]
    assert rows["30"]["targetFailedChecksAbsent"] == ["image_a_no_reliable_face_triple"]
    for row in rows.values():
        assert row["imageA_sha256_expected"]
        assert row["imageB_sha256_expected"]
        assert row["failureClass"]


def test_hard_case_target_failures_check_absent_failed_checks():
    row = {"targetFailedChecksAbsent": ["image_b_no_reliable_face_triple"]}
    payload = {"failedChecks": ["image_b_no_reliable_face_triple"]}

    assert target_failures(row, payload, input_drift=False) == [
        "target_check_still_present:image_b_no_reliable_face_triple"
    ]
    assert target_failures(row, {"failedChecks": ["piece_legality_invalid"]}, input_drift=False) == []
