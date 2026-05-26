import json
import sys
from pathlib import Path

import tools.probe_corpus as probe_corpus
from tools.probe_candidate_guards import (
    candidate_grid_purity_guard,
    candidate_repair_backfill_opportunity,
    selected_grid_purity_summary,
)
from tools.probe_corpus import (
    _check_expected_yaw,
    candidate_grid_span_guard,
    classify_face_failure,
    count_deviation_summary,
    environment_policy_warnings,
    grid_weak_reasons,
    load_manifest,
    load_manifest_document,
    runtime_summary,
    score_direct_legal_candidates,
    selected_grid_span_summary,
    smallest_rank_gaps,
    timing_summary,
    write_json,
)


def test_probe_failure_classifier_marks_input_drift_first():
    mode = classify_face_failure(
        status="success",
        input_drift=True,
        selected_is_correct=True,
        correct_matrix_generated=True,
        correct_color_multiset_present=True,
    )

    assert mode == "image_input_drift"


def test_candidate_repair_backfill_opportunity_flags_set_61_shape():
    signals = {
        "repairCandidateCount": 3,
        "selectedRepairCandidate": {
            "repairChanges": 10,
            "preRepairConflicts": {"totalConflicts": 8},
        },
    }
    payload = {
        "recognitionCategory": "needs_manual_review",
        "recognitionCategoryReason": "repair_path_unstable_pre_repair_piece_evidence",
        "confidence": 0.5486,
    }

    guard = candidate_repair_backfill_opportunity(
        signals,
        payload,
        repair_backfill_gate_would_apply=True,
    )

    assert guard["wouldFire"] is True
    assert guard["firedRules"] == ["skipped_backfill_with_unstable_standard_repair"]


def test_candidate_grid_purity_guard_flags_set_30_shape():
    signals = {
        "topVisibleTripleQuality": {
            "imageA": {
                "componentOverlap": 6,
                "sidePair": "B/L",
                "grids": {
                    "B": {"gridId": 22, "cellFaceCounts": {"B": 2, "L": 5, "U": 1, "D": 1}},
                    "L": {"gridId": 5, "cellFaceCounts": {"L": 2, "F": 3, "U": 2, "D": 1, "R": 1}},
                    "U": {"gridId": 12, "cellFaceCounts": {"U": 1, "B": 4, "L": 2, "F": 1, "R": 1}},
                },
            },
            "imageB": {
                "componentOverlap": 1,
                "sidePair": "F/L",
                "grids": {
                    "D": {"gridId": 13, "cellFaceCounts": {"D": 2, "B": 4, "U": 2, "R": 1}},
                    "F": {"gridId": 5, "cellFaceCounts": {"F": 3, "R": 3, "B": 1, "D": 1, "L": 1}},
                    "L": {"gridId": 20, "cellFaceCounts": {"L": 2, "B": 2, "D": 2, "R": 1, "U": 2}},
                },
            },
        }
    }

    summary = selected_grid_purity_summary(signals)
    guard = candidate_grid_purity_guard(summary)

    assert summary["maxTopVisibleComponentOverlap"] == 6
    assert summary["topVisibleLowSelfFaceCells"] == 5
    assert summary["maxTopVisibleDominantWrongMargin"] == 3
    assert guard["wouldFire"] is True
    assert guard["firedRules"] == ["top_visible_overlap_and_low_self_purity"]


def test_candidate_grid_purity_guard_requires_high_top_visible_overlap():
    summary = {
        "maxTopVisibleComponentOverlap": 3,
        "topVisibleLowSelfFaceCells": 5,
        "maxTopVisibleDominantWrongMargin": 3,
    }

    guard = candidate_grid_purity_guard(summary)

    assert guard["wouldFire"] is False


def test_candidate_grid_purity_guard_requires_low_self_purity():
    summary = {
        "maxTopVisibleComponentOverlap": 6,
        "topVisibleLowSelfFaceCells": 4,
        "maxTopVisibleDominantWrongMargin": 3,
    }

    guard = candidate_grid_purity_guard(summary)

    assert guard["wouldFire"] is False


def test_candidate_grid_purity_guard_requires_wrong_dominant_margin():
    summary = {
        "maxTopVisibleComponentOverlap": 6,
        "topVisibleLowSelfFaceCells": 5,
        "maxTopVisibleDominantWrongMargin": 2,
    }

    guard = candidate_grid_purity_guard(summary)

    assert guard["wouldFire"] is False


def test_candidate_repair_backfill_opportunity_flags_metric_only_instability():
    signals = {
        "repairCandidateCount": 2,
        "selectedRepairCandidate": {
            "repairChanges": 8,
            "preRepairConflicts": {"totalConflicts": 7},
        },
    }
    payload = {
        "recognitionCategory": "needs_manual_review",
        "recognitionCategoryReason": "some_future_manual_reason",
        "confidence": 0.61,
    }

    guard = candidate_repair_backfill_opportunity(
        signals,
        payload,
        repair_backfill_gate_would_apply=True,
    )

    assert guard["wouldFire"] is True
    assert guard["firedRules"] == ["skipped_backfill_with_unstable_standard_repair"]


def test_candidate_repair_backfill_opportunity_flags_conflict_only_instability():
    signals = {
        "repairCandidateCount": 2,
        "selectedRepairCandidate": {
            "repairChanges": 7,
            "preRepairConflicts": {"totalConflicts": 8},
        },
    }
    payload = {
        "recognitionCategory": "needs_manual_review",
        "recognitionCategoryReason": "some_future_manual_reason",
        "confidence": 0.61,
    }

    guard = candidate_repair_backfill_opportunity(
        signals,
        payload,
        repair_backfill_gate_would_apply=True,
    )

    assert guard["wouldFire"] is True
    assert guard["firedRules"] == ["skipped_backfill_with_unstable_standard_repair"]


def test_candidate_repair_backfill_opportunity_skips_when_backfill_already_ran():
    signals = {
        "repairBackfillAttempted": True,
        "repairCandidateCount": 8,
        "selectedRepairCandidate": {
            "repairChanges": 0,
            "preRepairConflicts": {"totalConflicts": 0},
        },
    }
    payload = {
        "recognitionCategory": "success_repaired_high_confidence",
        "recognitionCategoryReason": "repair_path_high_confidence_low_penalty",
        "confidence": 0.8187,
    }

    guard = candidate_repair_backfill_opportunity(
        signals,
        payload,
        repair_backfill_gate_would_apply=True,
    )

    assert guard["wouldFire"] is False
    assert guard["repairBackfillGateWouldApply"] is True


def test_candidate_repair_backfill_opportunity_requires_backfill_gate():
    signals = {
        "repairCandidateCount": 3,
        "selectedRepairCandidate": {
            "repairChanges": 10,
            "preRepairConflicts": {"totalConflicts": 8},
        },
    }
    payload = {
        "recognitionCategory": "needs_manual_review",
        "recognitionCategoryReason": "repair_path_unstable_pre_repair_piece_evidence",
        "confidence": 0.5486,
    }

    guard = candidate_repair_backfill_opportunity(
        signals,
        payload,
        repair_backfill_gate_would_apply=False,
    )

    assert guard["wouldFire"] is False


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

    assert {row["setId"] for row in manifest} >= {"12", "14", "15", "24", "27", "29", "31", "32", "44"}
    for row in manifest:
        assert row["imageA_sha256_expected"]
        assert row["imageB_sha256_expected"]
        assert row["groundTruth_sha256_expected"]
        assert "expectedCategory" in row
        assert "currentScoreObserved" in row


def test_non_white_up_candidates_are_excluded_from_normal_corpus_manifest():
    fixture_dir = Path(__file__).parent / "fixtures"
    manifest = load_manifest(fixture_dir / "corpus_manifest.json")
    manifest_ids = {row["setId"] for row in manifest}
    candidates = json.loads((fixture_dir / "non_white_up_candidate_manifest.json").read_text())

    assert {row["setId"] for row in candidates["pairs"]} == {"1", "2", "3", "4", "5", "6", "7"}
    assert not ({row["setId"] for row in candidates["pairs"]} & manifest_ids)
    for row in candidates["pairs"]:
        assert row["declaredUpperFaceA"] != "white"
        assert row["declaredUpperFaceB"] != "yellow"
        assert "groundTruthPath" not in row


def test_probe_manifest_covers_approved_axis_truth_sets():
    """Every approved two-view axis-truth set must be runnable from the
    corpus manifest.

    The hull-label shadow analyzers use the axis-truth fixture as their
    row list and the corpus manifest as their image resolver. If these
    drift apart, diagnostics silently skip rows before reaching the
    actual acceptance gate.
    """
    manifest = load_manifest(Path(__file__).parent / "fixtures" / "corpus_manifest.json")
    manifest_ids = {row["setId"] for row in manifest}
    axis_truth_path = Path(__file__).parent / "fixtures" / "gcm_axis_ground_truth.json"
    axis_truth = json.loads(axis_truth_path.read_text(encoding="utf-8"))
    approved_ids = {
        key.rsplit("_", 1)[0]
        for key, row in axis_truth.items()
        if isinstance(row, dict) and row.get("approved")
    }

    assert approved_ids <= manifest_ids


def test_probe_manifest_records_primary_supported_architecture():
    document = load_manifest_document(Path(__file__).parent / "fixtures" / "corpus_manifest.json")
    primary = document["supportedArchitectures"]["primary"]

    assert primary["label"] == "native-arm64-macos-python312"
    assert primary["platform.machine"] == "arm64"
    assert primary["platform.system"] == "Darwin"
    assert primary["python.versionInfo"] == [3, 12, 13, "final", 0]
    assert primary["numpy.version"] == "2.3.5"
    assert primary["pillow.version"] == "12.2.0"
    assert "Issue #25" in primary["notes"]


def test_probe_environment_policy_warnings_are_empty_for_matching_runtime():
    manifest = {
        "supportedArchitectures": {
            "primary": {
                "label": "native-arm64-macos-python312",
                "platform.machine": "arm64",
                "platform.system": "Darwin",
                "python.versionInfo": [3, 12, 13, "final", 0],
                "numpy.version": "2.3.5",
                "pillow.version": "12.2.0",
                "notes": "Pinned baseline.",
            }
        }
    }
    fingerprint = {
        "platform": {"machine": "arm64", "system": "Darwin"},
        "python": {"versionInfo": [3, 12, 13, "final", 0]},
        "packages": {"numpy": {"version": "2.3.5"}, "pillow": {"version": "12.2.0"}},
    }

    assert environment_policy_warnings(manifest, fingerprint) == []


def test_probe_environment_policy_warnings_report_mismatches():
    manifest = {
        "supportedArchitectures": {
            "primary": {
                "label": "native-arm64-macos-python312",
                "platform.machine": "arm64",
                "platform.system": "Darwin",
                "python.versionInfo": [3, 12, 13, "final", 0],
                "numpy.version": "2.3.5",
                "pillow.version": "12.2.0",
                "notes": "Pinned baseline.",
            }
        }
    }
    fingerprint = {
        "platform": {"machine": "x86_64", "system": "Darwin"},
        "python": {"versionInfo": [3, 12, 11, "final", 0]},
        "packages": {"numpy": {"version": "2.3.5"}, "pillow": {"version": "12.2.0"}},
    }

    warnings = environment_policy_warnings(manifest, fingerprint)

    assert warnings == [
        {
            "architecture": "primary",
            "label": "native-arm64-macos-python312",
            "message": (
                "Corpus manifest baseline expects native-arm64-macos-python312; "
                "current runtime differs. Scores/categories may be architecture-dependent."
            ),
            "mismatches": [
                {"key": "platform.machine", "expected": "arm64", "actual": "x86_64"},
                {
                    "key": "python.versionInfo",
                    "expected": [3, 12, 13, "final", 0],
                    "actual": [3, 12, 11, "final", 0],
                },
            ],
            "notes": "Pinned baseline.",
        }
    ]


def test_probe_json_includes_environment_policy_warnings(tmp_path):
    output = tmp_path / "probe.json"
    warning = {
        "architecture": "primary",
        "label": "native-arm64-macos-python312",
        "message": "runtime differs",
        "mismatches": [],
    }

    write_json(output, [], Path("manifest.json"), {"python": {"versionInfo": [3, 12, 13]}}, [warning])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["environmentPolicyWarnings"] == [warning]


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


def test_probe_grid_weak_reasons_marks_set_44_style_down_anchor():
    reasons = grid_weak_reasons(
        {
            "matchedCount": 5,
            "fitError": 0.442,
            "quality": 76.089,
            "gridSamples": 4,
            "badSamples": 0,
            "suspectSamples": 0.0,
        }
    )

    assert reasons == ["matchedCount_below_6", "quality_below_80", "grid_sample_heavy"]


def test_probe_grid_weak_reasons_marks_grid_span_contamination():
    reasons = grid_weak_reasons(
        {
            "matchedCount": 8,
            "fitError": 0.2,
            "quality": 91.0,
            "gridSamples": 1,
            "badSamples": 0,
            "suspectSamples": 0.0,
            "gridSpanContamination": {
                "score": 8.25,
                "componentShapeSpread": 36.0,
                "extrapolatedCellCount": 3,
                "sampleCellsOutsideGridComponentHull": 3,
            },
        }
    )

    assert reasons == [
        "grid_span_contamination_score_ge_8",
        "component_shape_spread_ge_32",
        "extrapolated_cells_ge_3",
        "sample_cells_outside_grid_component_hull_ge_3",
    ]


def test_selected_grid_span_summary_flattens_selected_grid_diagnostics():
    summary = selected_grid_span_summary(
        {
            "selectedGridQuality": {
                "imageA": {
                    "U": {
                        "gridId": 1,
                        "gridSpanContamination": {
                            "score": 4.5,
                            "componentShapeSpread": 12.0,
                            "componentShapeAngleCount": 6,
                            "sampledCellCount": 2,
                            "extrapolatedCellCount": 1,
                            "unsupportedCellCount": 1,
                            "badSampleCellCount": 0,
                            "cubeHullOutsideCount": 0,
                            "maxOutsideGridComponentHullRatio": 0.8,
                            "maxNearestGridComponentRatio": 1.2,
                            "sampleCellsOutsideGridComponentHull": 1,
                            "sampleCellsFarFromGridComponents": 0,
                        },
                    },
                    "R": {
                        "gridId": 2,
                        "gridSpanContamination": {
                            "score": 9.0,
                            "componentShapeSpread": 34.0,
                            "componentShapeAngleCount": 7,
                            "sampledCellCount": 3,
                            "extrapolatedCellCount": 2,
                            "unsupportedCellCount": 0,
                            "badSampleCellCount": 1,
                            "cubeHullOutsideCount": 2,
                            "maxOutsideGridComponentHullRatio": 1.1,
                            "maxNearestGridComponentRatio": 2.0,
                            "sampleCellsOutsideGridComponentHull": 2,
                            "sampleCellsFarFromGridComponents": 1,
                        },
                    },
                }
            }
        }
    )

    assert summary["maxScore"] == 9.0
    assert summary["maxComponentShapeSpread"] == 34.0
    assert summary["maxOutsideGridComponentHullRatio"] == 1.1
    assert summary["maxNearestGridComponentRatio"] == 2.0
    assert summary["totalSampledCells"] == 5
    assert summary["totalExtrapolatedCells"] == 3
    assert summary["totalUnsupportedCells"] == 1
    assert summary["totalBadSampleCells"] == 1
    assert summary["totalCubeHullOutsideCells"] == 2
    assert summary["rows"][1]["face"] == "R"


def test_candidate_grid_span_guard_reports_diagnostics_only_rules():
    guard = candidate_grid_span_guard(
        {
            "maxScore": 8.235,
            "maxComponentShapeSpread": 30.0,
            "maxNearestGridComponentRatio": 1.284,
            "totalSampledCells": 15,
            "totalUnsupportedCells": 5,
        }
    )

    assert guard["policy"] == "diagnostics_only_no_behavior_change"
    assert guard["intendedUse"] == "candidate_manual_review_guard_not_promotion"
    assert guard["wouldFire"] is True
    assert guard["firedRules"] == [
        "shape_spread_and_sample_load",
        "sample_distance_and_unsupported_load",
        "high_span_score",
    ]
    assert guard["rules"][0]["metrics"] == {
        "maxComponentShapeSpread": 30.0,
        "totalSampledCells": 15,
    }


def test_candidate_grid_span_guard_stays_quiet_below_candidate_thresholds():
    guard = candidate_grid_span_guard(
        {
            "maxScore": 6.0,
            "maxComponentShapeSpread": 29.0,
            "maxNearestGridComponentRatio": 1.2,
            "totalSampledCells": 15,
            "totalUnsupportedCells": 4,
        }
    )

    assert guard["wouldFire"] is False
    assert guard["firedRules"] == []


def test_probe_count_deviation_summary_reports_face_count_imbalance():
    summary = count_deviation_summary(
        [
            {
                "counts": {"U": 19, "R": 9, "F": 8, "D": 6, "L": 7, "B": 5},
                "n": 12,
            }
        ]
    )

    assert summary == {
        "mostCommonCounts": {"U": 19, "R": 9, "F": 8, "D": 6, "L": 7, "B": 5},
        "mostCommonCountFrequency": 12,
        "deviationsFromNine": {"U": 10, "F": -1, "D": -3, "L": -2, "B": -4},
    }


def test_score_direct_legal_candidates_adds_score_and_hamming():
    expected = "U" * 54

    scored = score_direct_legal_candidates(
        [
            {"state": "U" * 54, "confidence": 0.8},
            {"state": "R" + "U" * 53, "confidence": 0.7},
        ],
        expected,
    )

    assert scored == [
        {"state": "U" * 54, "confidence": 0.8, "score": 54, "hamming": 0},
        {"state": "R" + "U" * 53, "confidence": 0.7, "score": 53, "hamming": 1},
    ]


def test_probe_timing_summary_reports_total_and_slowest_rows():
    summary = timing_summary(
        [
            {"setId": "12", "status": "success", "timings": {"totalSeconds": 2.5, "recognizeSeconds": 2.1}},
            {"setId": "14", "status": "success", "timings": {"totalSeconds": 4.0, "recognizeSeconds": 3.8}},
            {"setId": "missing", "status": "skipped", "timings": {"totalSeconds": 0.1}},
        ]
    )

    assert summary == {
        "totalSeconds": 6.5,
        "rowCount": 2,
        "meanSeconds": 3.25,
        "p50Seconds": 3.25,
        "p95Seconds": 4.0,
        "slowestRows": [
            {"setId": "14", "totalSeconds": 4.0, "recognizeSeconds": 3.8},
            {"setId": "12", "totalSeconds": 2.5, "recognizeSeconds": 2.1},
        ],
    }


def test_probe_rows_reports_row_progress_to_stderr(monkeypatch, capsys):
    rows = [{"setId": "12"}, {"setId": "14"}]

    def fake_probe_pair(row, manifest_path):
        return {
            "setId": row["setId"],
            "status": "success",
            "score": 54,
            "category": "success_clean",
            "contractPassed": True,
            "timings": {"totalSeconds": 1.3},
        }

    monkeypatch.setattr(probe_corpus, "probe_pair", fake_probe_pair)

    results = probe_corpus.probe_rows(rows, Path("manifest.json"), progress=True)

    captured = capsys.readouterr()
    assert [row["setId"] for row in results] == ["12", "14"]
    assert captured.out == ""
    assert "[probe] 1/2 set 12 start elapsed=" in captured.err
    assert "[probe] 1/2 set 12 done row=1.3s" in captured.err
    assert "score=54/54 category=success_clean contract=pass" in captured.err
    assert "[probe] 2/2 set 14 done row=1.3s" in captured.err
    assert "eta=" in captured.err


def test_probe_rows_can_disable_progress(monkeypatch, capsys):
    def fake_probe_pair(row, manifest_path):
        return {
            "setId": row["setId"],
            "status": "success",
            "score": 54,
            "category": "success_clean",
            "contractPassed": True,
            "timings": {"totalSeconds": 1.3},
        }

    monkeypatch.setattr(probe_corpus, "probe_pair", fake_probe_pair)

    probe_corpus.probe_rows([{"setId": "12"}], Path("manifest.json"), progress=False)

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_probe_runtime_summary_includes_key_versions():
    summary = runtime_summary(
        {
            "python": {"versionInfo": [3, 12, 13], "executable": "/tmp/python"},
            "platform": {"platform": "TestOS-arm64"},
            "packages": {"numpy": {"version": "2.3.5"}, "pillow": {"version": "12.2.0"}},
        }
    )

    assert "Python 3.12.13" in summary
    assert "Pillow 12.2.0" in summary
    assert "NumPy 2.3.5" in summary
    assert "TestOS-arm64" in summary


def test_probe_cli_analysis_only_writes_analysis_dump_without_recognition(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"pairs": [{"setId": "x"}]}', encoding="utf-8")
    analysis_output = tmp_path / "analysis.json"
    json_output = tmp_path / "probe.json"
    calls = {}

    def fake_analysis_dump_for_rows(rows, manifest_path, *, image_selection, fingerprint):
        calls["analysis"] = {
            "rows": [row["setId"] for row in rows],
            "manifest": manifest_path,
            "image_selection": image_selection,
            "fingerprint": fingerprint,
        }
        return {"ok": True}

    def fake_write_analysis_json(path, payload):
        calls["analysis_write"] = {"path": path, "payload": payload}
        path.write_text("analysis", encoding="utf-8")

    def fail_probe_pair(*args, **kwargs):
        raise AssertionError("analysis-only should not run recognition")

    def fail_write_json(*args, **kwargs):
        raise AssertionError("analysis-only should not write --json-output")

    fingerprint = {"python": {"versionInfo": [3, 12, 13], "executable": "/tmp/python"}}
    monkeypatch.setattr(probe_corpus, "runtime_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(probe_corpus, "analysis_dump_for_rows", fake_analysis_dump_for_rows)
    monkeypatch.setattr(probe_corpus, "write_analysis_json", fake_write_analysis_json)
    monkeypatch.setattr(probe_corpus, "probe_pair", fail_probe_pair)
    monkeypatch.setattr(probe_corpus, "write_json", fail_write_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_corpus.py",
            "--manifest",
            str(manifest),
            "--analysis-output",
            str(analysis_output),
            "--analysis-image",
            "imageA",
            "--analysis-only",
            "--json-output",
            str(json_output),
            "--quiet",
        ],
    )

    assert probe_corpus.main() == 0
    assert analysis_output.read_text(encoding="utf-8") == "analysis"
    assert not json_output.exists()
    assert calls["analysis"] == {
        "rows": ["x"],
        "manifest": manifest,
        "image_selection": "imageA",
        "fingerprint": fingerprint,
    }
    assert calls["analysis_write"] == {"path": analysis_output, "payload": {"ok": True}}


def test_check_expected_yaw_passes_when_manifest_omits_expectation():
    """Legacy manifest rows without an `expectedYaw` block must
    pass through — no false-negative contract failures while the
    corpus is being migrated."""
    ok, diagnostic = _check_expected_yaw(None, {"captureYaw": {"status": "standard"}})

    assert ok is True
    assert diagnostic is None


def test_check_expected_yaw_passes_when_signals_match_expectation():
    """The Set 42 / Set 32 fingerprint: nonstandard yaw=1 with
    normalization applied. Manifest pins this, probe confirms it."""
    expected = {"status": "nonstandard", "quarterTurns": 1, "normalizationApplied": True}
    signals = {
        "captureYaw": {
            "status": "nonstandard",
            "quarterTurns": 1,
            "degrees": 90,
            "normalizationApplied": True,
        }
    }
    ok, diagnostic = _check_expected_yaw(expected, signals)

    assert ok is True
    assert diagnostic == {
        "expected": expected,
        "actual": {"status": "nonstandard", "quarterTurns": 1, "normalizationApplied": True},
        "mismatchedKeys": [],
    }


def test_check_expected_yaw_fails_on_status_mismatch():
    """Future regression: a recognizer change accidentally produces
    yaw=standard for a known nonstandard pair. The probe harness
    must surface this rather than silently passing."""
    expected = {"status": "nonstandard", "quarterTurns": 1, "normalizationApplied": True}
    signals = {
        "captureYaw": {
            "status": "standard",
            "quarterTurns": 0,
            "normalizationApplied": False,
        }
    }
    ok, diagnostic = _check_expected_yaw(expected, signals)

    assert ok is False
    assert "status" in diagnostic["mismatchedKeys"]
    assert "quarterTurns" in diagnostic["mismatchedKeys"]


def test_check_expected_yaw_fails_on_quarter_turns_drift():
    """Catches the case where a future yaw-detection change picks a
    different side-pair lookup and produces e.g. quarterTurns=2 for
    Set 12 (which should remain quarterTurns=3)."""
    expected = {"status": "nonstandard", "quarterTurns": 3}
    signals = {
        "captureYaw": {
            "status": "nonstandard",
            "quarterTurns": 2,
            "normalizationApplied": True,
        }
    }
    ok, diagnostic = _check_expected_yaw(expected, signals)

    assert ok is False
    assert diagnostic["mismatchedKeys"] == ["quarterTurns"]


def test_check_expected_yaw_fails_when_signals_missing_captureyaw():
    """Pre-PR-#20 server / non-cv-local provider; pins a stronger
    signal that the yaw block didn't make it back at all."""
    expected = {"status": "nonstandard", "quarterTurns": 1}

    ok, diagnostic = _check_expected_yaw(expected, {})

    assert ok is False
    assert diagnostic["actual"] == {
        "status": None,
        "quarterTurns": None,
        "normalizationApplied": None,
    }


def test_probe_manifest_records_expected_yaw_for_post_pr20_pairs():
    """Sets 12, 32, and 42 are the post-PR-#20 / post-PR-#95 corpus
    entries — they must all carry an `expectedYaw` block pinning the
    detected yaw so the probe harness can catch future regressions."""
    manifest = load_manifest(Path(__file__).parent / "fixtures" / "corpus_manifest.json")
    rows = {row["setId"]: row for row in manifest}

    for set_id in ("12", "32", "42"):
        assert "expectedYaw" in rows[set_id], f"Set {set_id} missing expectedYaw"
        assert rows[set_id]["expectedYaw"]["status"] == "nonstandard"
        assert rows[set_id]["expectedYaw"]["normalizationApplied"] is True


def test_probe_manifest_records_executable_yaw_for_63_68_cohort():
    """Sets 63-68 are a repeated-capture/yaw cohort. Only rows where
    the current recognizer emits `captureYaw` should carry executable
    `expectedYaw`; rejected rows keep yaw documentation in notes."""
    manifest = load_manifest(Path(__file__).parent / "fixtures" / "corpus_manifest.json")
    rows = {row["setId"]: row for row in manifest}

    assert rows["63"]["expectedYaw"] == {
        "status": "nonstandard",
        "quarterTurns": 2,
        "normalizationApplied": True,
    }
    assert rows["67"]["expectedYaw"] == {
        "status": "nonstandard",
        "quarterTurns": 3,
        "normalizationApplied": True,
    }
    assert rows["68"]["expectedYaw"] == {
        "status": "standard",
        "quarterTurns": 0,
        "normalizationApplied": False,
    }
    assert "expectedYaw" not in rows["64"]
    assert "Human-confirmed capture yaw=0" in rows["64"]["notes"]
    assert "expectedYaw" not in rows["65"]
    assert "expectedYaw" not in rows["66"]
