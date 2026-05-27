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


def _candidate_evaluation(event: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(event.get("constrainedInference")).get("candidateEvaluation"))


def _promotion_readiness(
    summary: Mapping[str, Any],
    *,
    min_events: int,
    max_error_rate: float,
    max_gate_reject_rate: float,
    require_gt_on_accepted: bool,
) -> Dict[str, Any]:
    event_count = int(summary.get("eventCount") or 0)
    gate_rejected = int(summary.get("gateRejected") or 0)
    errors = list(summary.get("errorEventKeys") or [])
    known_bad = list(summary.get("knownBadAcceptedEventKeys") or [])
    accepted_without_gt = int(summary.get("acceptedWithoutCandidateEvaluation") or 0)

    error_rate = (len(errors) / event_count) if event_count else 0.0
    gate_reject_rate = (gate_rejected / event_count) if event_count else 0.0
    reasons: List[str] = []
    if event_count < min_events:
        reasons.append("insufficient_shadow_events")
    if error_rate > max_error_rate:
        reasons.append("shadow_error_rate_above_gate")
    if gate_reject_rate > max_gate_reject_rate:
        reasons.append("gate_reject_rate_above_gate")
    if known_bad:
        reasons.append("known_bad_auto_accepts")
    if require_gt_on_accepted and accepted_without_gt:
        reasons.append("accepted_candidates_without_gt_evaluation")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "thresholds": {
            "minEvents": int(min_events),
            "maxErrorRate": float(max_error_rate),
            "maxGateRejectRate": float(max_gate_reject_rate),
            "requireZeroKnownBadAccepts": True,
            "requireGtOnAccepted": bool(require_gt_on_accepted),
        },
        "observed": {
            "eventCount": event_count,
            "errorRate": round(error_rate, 6),
            "gateRejectRate": round(gate_reject_rate, 6),
            "knownBadAccepted": len(known_bad),
            "acceptedWithoutCandidateEvaluation": accepted_without_gt,
        },
    }


def summarize_events(
    events: Sequence[Mapping[str, Any]],
    *,
    min_events: int = 0,
    max_error_rate: float = 0.0,
    max_gate_reject_rate: float = 1.0,
    require_gt_on_accepted: bool = False,
) -> Dict[str, Any]:
    modes: Counter[str] = Counter()
    result_status: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    gate_decisions: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    yaw_counts: Counter[str] = Counter()
    candidate_hammings: Counter[str] = Counter()
    selected = 0
    fallback = 0
    gate_accepted = 0
    threshold_switched = 0
    candidate_eval_available = 0
    candidate_exact = 0
    accepted_without_candidate_eval = 0
    errors: List[str] = []
    known_bad_accepted: List[str] = []

    for event in events:
        constrained = _mapping(event.get("constrainedInference"))
        result = _mapping(event.get("result"))
        gate = _mapping(constrained.get("promotionGate"))
        candidate_eval = _candidate_evaluation(event)
        candidate_eval_available_here = candidate_eval.get("available") is True
        gate_accepted_here = gate.get("accepted") is True

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
        if gate_accepted_here:
            gate_accepted += 1
            if not candidate_eval_available_here:
                accepted_without_candidate_eval += 1
        if _threshold_switched(event):
            threshold_switched += 1
        if constrained.get("status") == "error":
            errors.append(_event_key(event))
        if candidate_eval_available_here:
            candidate_eval_available += 1
            if candidate_eval.get("exact") is True:
                candidate_exact += 1
            hamming = candidate_eval.get("hamming")
            if isinstance(hamming, int):
                candidate_hammings[str(hamming)] += 1
            if gate_accepted_here and candidate_eval.get("exact") is not True:
                known_bad_accepted.append(_event_key(event))
        for reason in gate.get("rejectReasons") or []:
            reject_reasons[str(reason)] += 1

    summary = {
        "eventCount": len(events),
        "selected": selected,
        "fallbackToLegacy": fallback,
        "gateAccepted": gate_accepted,
        "gateRejected": len(events) - gate_accepted,
        "thresholdSwitched": threshold_switched,
        "candidateEvaluationAvailable": candidate_eval_available,
        "candidateExact": candidate_exact,
        "candidateHammingDistribution": dict(sorted(candidate_hammings.items())),
        "acceptedWithoutCandidateEvaluation": accepted_without_candidate_eval,
        "knownBadAcceptedEventKeys": known_bad_accepted,
        "errorEventKeys": errors,
        "modeCounts": dict(modes.most_common()),
        "resultStatusCounts": dict(result_status.most_common()),
        "recognitionCategoryCounts": dict(categories.most_common()),
        "recommendedMethodCounts": dict(methods.most_common()),
        "gateDecisionCounts": dict(gate_decisions.most_common()),
        "gateRejectReasonCounts": dict(reject_reasons.most_common()),
        "yawQuarterTurnCounts": dict(yaw_counts.most_common()),
    }
    summary["promotionReadiness"] = _promotion_readiness(
        summary,
        min_events=min_events,
        max_error_rate=max_error_rate,
        max_gate_reject_rate=max_gate_reject_rate,
        require_gt_on_accepted=require_gt_on_accepted,
    )
    return summary


def _render_markdown(path: Path, summary: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        ("Events", summary.get("eventCount")),
        ("Selected constrained candidate", summary.get("selected")),
        ("Fallback to legacy", summary.get("fallbackToLegacy")),
        ("Gate accepted", summary.get("gateAccepted")),
        ("Gate rejected", summary.get("gateRejected")),
        ("Threshold switched", summary.get("thresholdSwitched")),
        ("Candidate GT eval available", summary.get("candidateEvaluationAvailable")),
        ("Candidate GT exact", summary.get("candidateExact")),
        ("Known bad auto-accepts", len(summary.get("knownBadAcceptedEventKeys") or [])),
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
        ("Candidate Hamming", "candidateHammingDistribution"),
    ):
        counts = _mapping(summary.get(key))
        lines.extend([f"## {title}", "", "| Value | Count |", "|---|---:|"])
        if counts:
            lines.extend(f"| `{value}` | {count} |" for value, count in counts.items())
        else:
            lines.append("| _none_ | 0 |")
        lines.append("")

    readiness = _mapping(summary.get("promotionReadiness"))
    lines.extend([
        "## Promotion Readiness",
        "",
        f"- Ready: `{readiness.get('ready')}`",
        f"- Reasons: `{readiness.get('reasons') or []}`",
        f"- Thresholds: `{readiness.get('thresholds') or {}}`",
        f"- Observed: `{readiness.get('observed') or {}}`",
        "",
    ])

    known_bad = list(summary.get("knownBadAcceptedEventKeys") or [])
    if known_bad:
        lines.extend(["Known bad auto-accepted events:", ""])
        lines.extend(f"- `{key}`" for key in known_bad)
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
    parser.add_argument("--min-events", type=int, default=0, help="Minimum events before the readiness gate can pass.")
    parser.add_argument("--max-error-rate", type=float, default=0.0, help="Maximum allowed constrained error-event rate.")
    parser.add_argument("--max-gate-reject-rate", type=float, default=1.0, help="Maximum allowed promotion-gate reject rate.")
    parser.add_argument(
        "--require-gt-on-accepted",
        action="store_true",
        help="Require every gate-accepted event to include candidate GT evaluation.",
    )
    args = parser.parse_args(argv)

    if not args.log.exists():
        print(f"No constrained shadow log found at {args.log}", file=sys.stderr)
        return 2
    events = _load_events(args.log)
    summary = summarize_events(
        events,
        min_events=args.min_events,
        max_error_rate=args.max_error_rate,
        max_gate_reject_rate=args.max_gate_reject_rate,
        require_gt_on_accepted=args.require_gt_on_accepted,
    )
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
