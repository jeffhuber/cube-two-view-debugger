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
    assert summary["recommendedMethodCounts"] == {
        "canonical_count_repaired": 1,
        "two_view_consistency_repaired": 1,
        "guarded_broad_legal_repaired": 1,
    }
    assert summary["gateRejectReasonCounts"] == {"recommended_confidence_low_or_missing": 1}
    assert summary["yawQuarterTurnCounts"] == {"0": 2, "2": 1}


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
    report_text = report.read_text(encoding="utf-8")
    assert "Constrained Shadow Log Summary" in report_text
    assert "`canonical_count_repaired`" in report_text
