from __future__ import annotations

import json

from tools.report_recognition_events import _endpoint_report_url, build_summary, render_report


def _event_row(**overrides):
    event = {
        "result": {"status": "success", "reason": None},
        "constrainedInference": {
            "pairThresholdSelection": {
                "selectionReason": "current_invalid_selected_best_pair",
                "searchMode": "canonical_valid_light_shortcut",
                "currentCanonicalProbeValid": False,
                "currentLegalRepairEvaluated": False,
                "currentLegalRepairSkipped": True,
                "cheapCurrentCanonicalShadow": {
                    "currentCanonicalValid": False,
                    "currentLegalValid": False,
                    "currentLegalRepairSkipped": True,
                    "couldHaveSkippedCurrentLegalForThisInput": True,
                },
            }
        },
        "performance": {
            "stageTimingsMs": {
                "recognizeTotal": 1000.0,
                "prepareConstrainedInput": 900.0,
                "selectGuardedPair": 20.0,
                "selectGuardedPair.currentCanonicalProbeEvaluatePair": 5.0,
            }
        },
    }
    row = {
        "created_at": "2026-05-29T00:00:00+00:00",
        "set_id": "41",
        "status": "success",
        "recognition_category": "success_clean",
        "failed_checks_json": "[]",
        "constrained_status": "success",
        "recommended_method": "canonical_count_repaired",
        "latency_ms": 1000.0,
        "recognize_total_ms": 1000.0,
        "prepare_constrained_input_ms": 900.0,
        "client_source": "cube-snap",
        "app_version": "0.0.1",
        "event_json": json.dumps(event),
    }
    row.update(overrides)
    return row


def test_recognition_event_summary_counts_and_stage_latency():
    rejected_event = {
        "result": {"status": "rejected", "reason": "not a cube"},
        "performance": {"stageTimingsMs": {"recognizeTotal": 100.0}},
    }
    rows = [
        _event_row(),
        _event_row(
            status="rejected",
            recognition_category="reject_retake",
            failed_checks_json=json.dumps(["non_cube_image_fast_reject"]),
            constrained_status="fast_reject",
            recommended_method=None,
            latency_ms=100.0,
            recognize_total_ms=100.0,
            prepare_constrained_input_ms=None,
            event_json=json.dumps(rejected_event),
        ),
    ]

    summary = build_summary(rows, recent_limit=1)

    assert summary["totalEvents"] == 2
    assert summary["statusCounts"] == {"rejected": 1, "success": 1}
    assert summary["failureReasonCounts"] == {"not a cube": 1}
    assert summary["stageTimingsMs"]["selectGuardedPair"]["p50"] == 20.0
    assert summary["latencyDrivers"][0]["stage"] == "recognizeTotal"
    assert summary["pairSelectionReasonCounts"] == {"current_invalid_selected_best_pair": 1}
    assert summary["pairSearchModeCounts"] == {"canonical_valid_light_shortcut": 1}
    assert summary["currentLegalRepairCounts"]["skipped"] == {"True": 1}
    assert summary["cheapCurrentCanonicalShadow"]["evaluatedCount"] == 1
    assert summary["cheapCurrentCanonicalShadow"]["currentLegalRepairSkippedCounts"] == {"True": 1}
    assert summary["cheapCurrentCanonicalShadow"]["couldHaveSkippedCurrentLegalCounts"] == {"True": 1}
    assert summary["slowestAttempts"][0]["latencyMs"] == 1000.0
    assert len(summary["recentAttempts"]) == 1


def test_recognition_event_report_renders_recent_attempts():
    payload = {
        "generatedAtUtc": "2026-05-29T00:00:00+00:00",
        "database": "/tmp/recognition_events.sqlite3",
        "sinceHours": None,
        "summary": build_summary([_event_row()], recent_limit=5),
    }

    report = render_report(payload)

    assert "# Recognition Event Report" in report
    assert "Status counts" in report
    assert "Pair selection" in report
    assert "Current legal repair" in report
    assert "Cheap current canonical shadow" in report
    assert "Latency Drivers" in report
    assert "Slowest Attempts" in report
    assert "Recent Attempts" in report


def test_endpoint_report_url_adds_camel_case_query_params():
    url = _endpoint_report_url(
        "https://api.cubesnap.app/api/recognition-events/report?recentLimit=99",
        since_hours=24,
        recent_limit=5,
    )

    assert url == (
        "https://api.cubesnap.app/api/recognition-events/report?"
        "recentLimit=5&sinceHours=24"
    )
