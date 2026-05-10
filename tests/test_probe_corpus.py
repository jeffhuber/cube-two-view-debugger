from pathlib import Path

from tools.probe_corpus import classify_face_failure, load_manifest, smallest_rank_gaps


def test_probe_failure_classifier_marks_input_drift_first():
    mode = classify_face_failure(
        status="success",
        input_drift=True,
        selected_is_correct=True,
        correct_matrix_generated=True,
        correct_color_multiset_present=True,
    )

    assert mode == "image_input_drift"


def test_probe_failure_classifier_marks_rejected_as_retake():
    mode = classify_face_failure(
        status="rejected",
        input_drift=False,
        selected_is_correct=False,
        correct_matrix_generated=False,
        correct_color_multiset_present=False,
    )

    assert mode == "retake_or_low_confidence"


def test_probe_failure_classifier_distinguishes_orientation_rank():
    mode = classify_face_failure(
        status="success",
        input_drift=False,
        selected_is_correct=False,
        correct_matrix_generated=True,
        correct_color_multiset_present=True,
    )

    assert mode == "orientation_rank_failure"


def test_probe_failure_classifier_distinguishes_color_or_merge():
    mode = classify_face_failure(
        status="success",
        input_drift=False,
        selected_is_correct=False,
        correct_matrix_generated=False,
        correct_color_multiset_present=True,
    )

    assert mode == "color_or_merge_failure"


def test_probe_failure_classifier_distinguishes_candidate_generation():
    mode = classify_face_failure(
        status="success",
        input_drift=False,
        selected_is_correct=False,
        correct_matrix_generated=False,
        correct_color_multiset_present=False,
    )

    assert mode == "candidate_generation_failure"


def test_probe_manifest_records_hashes_observed_baselines_and_contracts():
    manifest = load_manifest(Path(__file__).parent / "fixtures" / "corpus_manifest.json")

    assert {row["setId"] for row in manifest} >= {"12", "14", "15", "24", "27", "29", "31", "32"}
    for row in manifest:
        assert row["imageA_sha256_expected"]
        assert row["imageB_sha256_expected"]
        assert row["groundTruth_sha256_expected"]
        assert "expectedCategory" in row
        assert "currentScoreObserved" in row


def test_probe_smallest_rank_gaps_ignores_clean_and_missing_correct():
    results = [
        {
            "setId": "x",
            "orientationDiagnostics": {
                "imageA": {
                    "faces": [
                        {
                            "face": "R",
                            "selectedVsCorrectScoreGap": 8.0,
                            "selectedIsCorrect": False,
                            "correctMatrixGenerated": True,
                            "failureMode": "orientation_rank_failure",
                        },
                        {
                            "face": "F",
                            "selectedVsCorrectScoreGap": 1.0,
                            "selectedIsCorrect": True,
                            "correctMatrixGenerated": True,
                            "failureMode": "clean",
                        },
                        {
                            "face": "B",
                            "selectedVsCorrectScoreGap": 2.0,
                            "selectedIsCorrect": False,
                            "correctMatrixGenerated": False,
                            "failureMode": "candidate_generation_failure",
                        },
                    ]
                }
            },
        }
    ]

    gaps = smallest_rank_gaps(results)

    assert gaps == [{"setId": "x", "image": "imageA", "face": "R", "gap": 8.0, "mode": "orientation_rank_failure"}]
