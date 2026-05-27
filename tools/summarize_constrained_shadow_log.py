#!/usr/bin/env python3
"""Summarize constrained-inference shadow JSONL events.

The server writes these events when `/api/recognize` is called with
`hullLabelTier1=constrained-shadow` or `hullLabelTier1=constrained`.
This tool is intentionally GT-free: it reports production-available
distribution signals for real-traffic shadow rollout.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO_ROOT / "runs" / "constrained_inference_shadow.jsonl"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _load_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{index}: invalid JSONL event: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{index}: event must be a JSON object")
        events.append(payload)
    return events


def _threshold_switched(event: Mapping[str, Any]) -> bool:
    selection = _mapping(_mapping(event.get("constrainedInference")).get("pairThresholdSelection"))
    current = selection.get("currentThresholds")
    selected = selection.get("selectedThresholds")
    return isinstance(current, Mapping) and isinstance(selected, Mapping) and dict(current) != dict(selected)


def _event_key(event: Mapping[str, Any]) -> str:
    return str(event.get("runId") or event.get("setId") or event.get("createdAt") or "<unknown>")


def summarize_events(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    modes: Counter[str] = Counter()
    result_status: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    gate_decisions: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    yaw_counts: Counter[str] = Counter()
    selected = 0
    fallback = 0
    gate_accepted = 0
    threshold_switched = 0
    errors: List[str] = []

    for event in events:
        constrained = _mapping(event.get("constrainedInference"))
        result = _mapping(event.get("result"))
        gate = _mapping(constrained.get("promotionGate"))

        modes[str(event.get("mode"))] += 1
        result_status[str(result.get("status"))] += 1
        categories[str(result.get("recognitionCategory"))] += 1
        methods[str(constrained.get("recommendedMethod"))] += 1
        gate_decisions[str(gate.get("decision"))] += 1
        yaw_counts[str(constrained.get("yawQuarterTurns"))] += 1

        if constrained.get("selected") is True:
            selected += 1
        if constrained.get("fallbackToLegacy") is True:
            fallback += 1
        if gate.get("accepted") is True:
            gate_accepted += 1
        if _threshold_switched(event):
            threshold_switched += 1
        if constrained.get("status") == "error":
            errors.append(_event_key(event))
        for reason in gate.get("rejectReasons") or []:
            reject_reasons[str(reason)] += 1

    return {
        "eventCount": len(events),
        "selected": selected,
        "fallbackToLegacy": fallback,
        "gateAccepted": gate_accepted,
        "gateRejected": len(events) - gate_accepted,
        "thresholdSwitched": threshold_switched,
        "errorEventKeys": errors,
        "modeCounts": dict(modes.most_common()),
        "resultStatusCounts": dict(result_status.most_common()),
        "recognitionCategoryCounts": dict(categories.most_common()),
        "recommendedMethodCounts": dict(methods.most_common()),
        "gateDecisionCounts": dict(gate_decisions.most_common()),
        "gateRejectReasonCounts": dict(reject_reasons.most_common()),
        "yawQuarterTurnCounts": dict(yaw_counts.most_common()),
    }


def _render_markdown(path: Path, summary: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        ("Events", summary.get("eventCount")),
        ("Selected constrained candidate", summary.get("selected")),
        ("Fallback to legacy", summary.get("fallbackToLegacy")),
        ("Gate accepted", summary.get("gateAccepted")),
        ("Gate rejected", summary.get("gateRejected")),
        ("Threshold switched", summary.get("thresholdSwitched")),
    ]
    lines = [
        "# Constrained Shadow Log Summary",
        "",
        f"Source: `{path}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    lines.append("")
    for title, key in (
        ("Modes", "modeCounts"),
        ("Result Status", "resultStatusCounts"),
        ("Recognition Categories", "recognitionCategoryCounts"),
        ("Recommended Methods", "recommendedMethodCounts"),
        ("Gate Decisions", "gateDecisionCounts"),
        ("Gate Reject Reasons", "gateRejectReasonCounts"),
        ("Yaw Quarter Turns", "yawQuarterTurnCounts"),
    ):
        counts = _mapping(summary.get(key))
        lines.extend([f"## {title}", "", "| Value | Count |", "|---|---:|"])
        if counts:
            lines.extend(f"| `{value}` | {count} |" for value, count in counts.items())
        else:
            lines.append("| _none_ | 0 |")
        lines.append("")

    recent = list(events)[-10:]
    lines.extend(["## Recent Events", "", "| Run | Mode | Selected | Gate | Method | Yaw |", "|---|---|---:|---|---|---:|"])
    for event in recent:
        constrained = _mapping(event.get("constrainedInference"))
        gate = _mapping(constrained.get("promotionGate"))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_event_key(event)}`",
                    f"`{event.get('mode')}`",
                    str(constrained.get("selected")),
                    f"`{gate.get('decision')}`",
                    f"`{constrained.get('recommendedMethod')}`",
                    str(constrained.get("yawQuarterTurns")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="JSONL log path.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON summary output path.")
    parser.add_argument("--report", type=Path, help="Optional Markdown report output path.")
    args = parser.parse_args(argv)

    if not args.log.exists():
        print(f"No constrained shadow log found at {args.log}", file=sys.stderr)
        return 2
    events = _load_events(args.log)
    summary = summarize_events(events)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(_render_markdown(args.log, summary, events), encoding="utf-8")
    if not args.json_output and not args.report:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
