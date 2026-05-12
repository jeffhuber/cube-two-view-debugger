import json
import sys
from pathlib import Path

import tools.probe_corpus as probe_corpus
from tools.probe_corpus import (
    _check_expected_yaw,
    classify_face_failure,
    count_deviation_summary,
    environment_policy_warnings,
    grid_weak_reasons,
    load_manifest,
    load_manifest_document,
    runtime_summary,
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
        "slowestRows": [
            {"setId": "14", "totalSeconds": 4.0, "recognizeSeconds": 3.8},
            {"setId": "12", "totalSeconds": 2.5, "recognizeSeconds": 2.1},
        ],
    }


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
