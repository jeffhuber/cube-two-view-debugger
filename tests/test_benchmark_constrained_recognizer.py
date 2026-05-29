from __future__ import annotations

from tools.benchmark_constrained_recognizer import build_summary, render_report


def test_benchmark_summary_reports_variant_stage_timings():
    rows = [
        {
            "setId": "14",
            "variant": "threaded",
            "hullFitMode": "threaded",
            "maxSide": 1200,
            "iteration": 1,
            "status": "success",
            "exactMatch": True,
            "recommendedMethod": "canonical_count_repaired",
            "wallMs": 1000.0,
            "stageTimingsMs": {
                "prepareConstrainedInput": 900.0,
                "hullFitWall": 120.0,
                "selectGuardedPair": 20.0,
            },
            "pairThresholdSelection": {"selectionReason": "current_invalid_selected_best_pair"},
        },
        {
            "setId": "14",
            "variant": "serial",
            "hullFitMode": "serial",
            "maxSide": 1200,
            "iteration": 1,
            "status": "success",
            "exactMatch": True,
            "recommendedMethod": "canonical_count_repaired",
            "wallMs": 1200.0,
            "stageTimingsMs": {
                "prepareConstrainedInput": 1100.0,
                "hullFitWall": 240.0,
                "selectGuardedPair": 25.0,
            },
            "pairThresholdSelection": {"selectionReason": "current_invalid_selected_best_pair"},
        },
    ]

    summary = build_summary(rows)

    assert summary["variants"]["threaded"]["exactCount"] == 1
    assert summary["variants"]["serial"]["stageTimingsMs"]["hullFitWall"]["p50"] == 240.0
    assert summary["slowestSelectGuardedPairRows"][0]["variant"] == "serial"


def test_benchmark_report_includes_core_latency_table():
    payload = {
        "generatedAtUtc": "2026-05-29T00:00:00+00:00",
        "gitHead": "abc123",
        "manifest": "tests/fixtures/corpus_manifest.json",
        "setIds": ["14"],
        "iterations": 1,
        "warmup": 0,
        "summary": build_summary([
            {
                "setId": "14",
                "variant": "threaded@1200",
                "hullFitMode": "threaded",
                "maxSide": 1200,
                "iteration": 1,
                "status": "success",
                "exactMatch": True,
                "recommendedMethod": "canonical_count_repaired",
                "wallMs": 1000.0,
                "stageTimingsMs": {
                    "prepareConstrainedInput": 900.0,
                    "rembgA": 300.0,
                    "rembgB": 310.0,
                    "hullFitWall": 120.0,
                    "selectGuardedPair": 20.0,
                },
                "pairThresholdSelection": {"selectionReason": "kept_current_valid_repair"},
            }
        ]),
    }

    report = render_report(payload)

    assert "# Constrained Recognizer Benchmark" in report
    assert "`threaded@1200`" in report
    assert "Slowest Guarded Pair Rows" in report
