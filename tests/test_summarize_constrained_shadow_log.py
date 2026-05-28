from __future__ import annotations

import json

from tools.summarize_constrained_shadow_log import main, summarize_events


def _event(
    *,
    mode: str = "shadow",
    selected: bool = False,
    accepted: bool = True,
    method: str = "canonical_count_repaired",
    yaw: int = 0,
    switched: bool = False,
    candidate_available: bool = True,
    candidate_exact: bool = True,
    candidate_hamming: int = 0,
):
    selected_thresholds = {"A": 224, "B": 160} if switched else {"A": 160, "B": 160}
    return {
        "schema": "constrained_inference_shadow_event_v1",
        "mode": mode,
        "runId": f"run-{mode}-{selected}-{accepted}",
        "result": {
            "status": "success",
            "recognitionCategory": "success_clean",
        },
        "constrainedInference": {
            "selected": selected,
            "fallbackToLegacy": not selected,
            "status": "success",
            "yawQuarterTurns": yaw,
            "recommendedMethod": method,
            "candidateEvaluation": {
                "available": candidate_available,
                "exact": candidate_exact,
                "hamming": candidate_hamming,
                "expectedValid": True,
                "expectedErrors": [],
            },
            "promotionGate": {
                "accepted": accepted,
                "decision": "auto_return_candidate" if accepted else "fallback_or_manual_review",
                "rejectReasons": [] if accepted else ["recommended_confidence_low_or_missing"],
            },
            "pairThresholdSelection": {
                "selectionReason": "kept_current_valid_repair",
                "currentThresholds": {"A": 160, "B": 160},
                "selectedThresholds": selected_thresholds,
            },
        },
    }


def test_summarize_events_counts_shadow_distribution():
    summary = summarize_events([
        _event(),
        _event(mode="prefer", selected=True, method="two_view_consistency_repaired", yaw=2, switched=True),
        _event(accepted=False, method="guarded_broad_legal_repaired"),
    ])

    assert summary["eventCount"] == 3
    assert summary["selected"] == 1
    assert summary["fallbackToLegacy"] == 2
    assert summary["gateAccepted"] == 2
    assert summary["gateRejected"] == 1
    assert summary["thresholdSwitched"] == 1
    assert summary["candidateEvaluationAvailable"] == 3
    assert summary["candidateExact"] == 3
    assert summary["candidateHammingDistribution"] == {"0": 3}
    assert summary["knownBadAcceptedEventKeys"] == []
    assert summary["promotionReadiness"]["ready"] is True
    assert summary["recommendedMethodCounts"] == {
        "canonical_count_repaired": 1,
        "two_view_consistency_repaired": 1,
        "guarded_broad_legal_repaired": 1,
    }
    assert summary["gateRejectReasonCounts"] == {"recommended_confidence_low_or_missing": 1}
    assert summary["yawQuarterTurnCounts"] == {"0": 2, "2": 1}


def test_summarize_events_readiness_rejects_known_bad_auto_accept():
    summary = summarize_events([
        _event(selected=True, accepted=True, candidate_exact=False, candidate_hamming=2),
    ])

    assert summary["candidateEvaluationAvailable"] == 1
    assert summary["candidateExact"] == 0
    assert summary["knownBadAcceptedEventKeys"] == ["run-shadow-True-True"]
    assert summary["promotionReadiness"]["ready"] is False
    assert "known_bad_auto_accepts" in summary["promotionReadiness"]["reasons"]


def test_summarize_events_readiness_can_require_sample_floor_and_gt():
    summary = summarize_events([
        _event(candidate_available=False),
    ], min_events=2, require_gt_on_accepted=True)

    assert summary["acceptedWithoutCandidateEvaluation"] == 1
    assert summary["promotionReadiness"]["ready"] is False
    assert summary["promotionReadiness"]["reasons"] == [
        "insufficient_shadow_events",
        "accepted_candidates_without_gt_evaluation",
    ]


def test_cli_writes_json_and_markdown_outputs(tmp_path):
    log = tmp_path / "events.jsonl"
    log.write_text(
        "\n".join(json.dumps(event) for event in [_event(), _event(mode="prefer", selected=True)])
        + "\n",
        encoding="utf-8",
    )
    json_output = tmp_path / "summary.json"
    report = tmp_path / "summary.md"

    assert main(["--log", str(log), "--json-output", str(json_output), "--report", str(report)]) == 0

    summary = json.loads(json_output.read_text(encoding="utf-8"))
    assert summary["eventCount"] == 2
    assert summary["selected"] == 1
    assert summary["promotionReadiness"]["ready"] is True
    report_text = report.read_text(encoding="utf-8")
    assert "Constrained Shadow Log Summary" in report_text
    assert "Promotion Readiness" in report_text
    assert "`canonical_count_repaired`" in report_text


def test_cli_can_fail_when_promotion_readiness_gate_fails(tmp_path, capsys):
    log = tmp_path / "events.jsonl"
    log.write_text(
        json.dumps(_event(selected=True, candidate_exact=False, candidate_hamming=2)) + "\n",
        encoding="utf-8",
    )

    status = main(["--log", str(log), "--fail-when-not-ready"])

    captured = capsys.readouterr()
    assert status == 1
    assert "known_bad_auto_accepts" in captured.err
