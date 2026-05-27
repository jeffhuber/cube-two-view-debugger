#!/usr/bin/env python3
"""Validate a production-shaped constrained-inference promotion gate.

This consumes ``tools/diagnose_pair_threshold_repair.py`` output and applies a
GT-free auto-return gate to the guarded pair-threshold candidate. Ground truth
is used only to score the gate in this diagnostic report.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO_ROOT / "tests" / "fixtures" / "pair_threshold_repair_diagnostic.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "constrained_inference_promotion_gate.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "CONSTRAINED_INFERENCE_PROMOTION_GATE.md"

MAX_CANONICAL_REPAIR_MOVES = 10
MAX_LEGAL_STATE_DELTA = 4
MAX_LEGAL_REPAIR_COST = 20.0
MAX_LEGAL_REPAIR_CHANGES = 6

CANONICAL_METHOD = "canonical_count_repaired"
LEGAL_METHODS = {
    "conservative_legal_repaired",
    "two_view_consistency_repaired",
    "guarded_broad_legal_repaired",
}
ALLOWED_METHODS = {CANONICAL_METHOD, *LEGAL_METHODS}
ALLOWED_SELECTION_REASONS = {
    "kept_current_valid_repair",
    "current_invalid_selected_best_pair",
}


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rank_tuple(combo: Mapping[str, Any]) -> Optional[Tuple[int, int, float, int, int]]:
    raw = combo.get("productionRank")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 5:
        return None
    try:
        return (int(raw[0]), int(raw[1]), float(raw[2]), int(raw[3]), int(raw[4]))
    except (TypeError, ValueError):
        return None


def _recommended(combo: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(combo.get("summary")).get("recommended"))


def _recommended_method(combo: Mapping[str, Any]) -> Optional[str]:
    method = _mapping(combo.get("summary")).get("recommendedMethod")
    return method if isinstance(method, str) else None


def _valid_recommended(combo: Mapping[str, Any]) -> bool:
    recommended = _recommended(combo)
    return recommended.get("validState") is True and recommended.get("countBalanced") is True


def _selected_candidate_trace(row: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    selected = _mapping(row.get("pairSelected"))
    thresholds = _mapping(selected.get("thresholds"))
    threshold = thresholds.get(side)
    diagnostics = _mapping(_mapping(row.get("thresholdDiagnostics")).get(side))
    for candidate in diagnostics.get("candidates") or []:
        if isinstance(candidate, Mapping) and candidate.get("threshold") == threshold:
            return candidate
    return {}


def evaluate_promotion_gate(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate the production-shaped auto-return gate for one diagnostic row."""
    reasons = []
    selected = _mapping(row.get("pairSelected"))
    current = _mapping(row.get("current"))
    method = _recommended_method(selected)
    recommended = _recommended(selected)
    rank = _rank_tuple(selected)
    selection_reason = selected.get("selectionReason")

    if selected.get("status") != "assembled":
        reasons.append("selected_pair_not_assembled")
    if selection_reason not in ALLOWED_SELECTION_REASONS:
        reasons.append("selection_reason_not_allowed")
    if method not in ALLOWED_METHODS:
        reasons.append("recommended_method_not_allowed")
    if not _valid_recommended(selected):
        reasons.append("recommended_not_valid_balanced")
    if recommended.get("confidence") not in {"high", "medium"}:
        reasons.append("recommended_confidence_low_or_missing")
    if rank is None:
        reasons.append("missing_production_rank")

    current_valid = bool(_mapping(_mapping(current.get("summary")).get("recommended")).get("validState"))
    selected_thresholds = _mapping(selected.get("thresholds"))
    current_thresholds = _mapping(current.get("thresholds"))
    switched_thresholds = bool(selected_thresholds and current_thresholds and selected_thresholds != current_thresholds)
    if selection_reason == "kept_current_valid_repair":
        if switched_thresholds:
            reasons.append("kept_current_reason_but_thresholds_switched")
        if not current_valid:
            reasons.append("kept_current_reason_but_current_invalid")
    if selection_reason == "current_invalid_selected_best_pair" and current_valid:
        reasons.append("switched_pair_while_current_valid")

    yaw = _mapping(selected.get("yawInference"))
    if yaw and yaw.get("accepted") is not True:
        reasons.append("yaw_not_accepted")

    for side in ("A", "B"):
        trace = _selected_candidate_trace(row, side)
        if not trace:
            reasons.append(f"{side}_selected_threshold_trace_missing")
            continue
        if trace.get("accepted") is not True:
            reasons.append(f"{side}_selected_threshold_not_accepted")
        if trace.get("hardFailures") or trace.get("hard_failures"):
            reasons.append(f"{side}_selected_threshold_has_hard_failures")

    if rank is not None and method == CANONICAL_METHOD:
        _tier, move_count, _cost, _changes, _yaw_rank = rank
        if int(move_count) > MAX_CANONICAL_REPAIR_MOVES:
            reasons.append("canonical_repair_moves_above_gate")
    elif rank is not None and method in LEGAL_METHODS:
        _tier, state_delta, repair_cost, repair_changes, _yaw_rank = rank
        if int(state_delta) > MAX_LEGAL_STATE_DELTA:
            reasons.append("legal_state_delta_above_gate")
        if float(repair_cost) > MAX_LEGAL_REPAIR_COST:
            reasons.append("legal_repair_cost_above_gate")
        if int(repair_changes) > MAX_LEGAL_REPAIR_CHANGES:
            reasons.append("legal_repair_changes_above_gate")

    accepted = not reasons
    hamming = recommended.get("hamming")
    return {
        "setId": row.get("setId"),
        "accepted": accepted,
        "decision": "auto_return_candidate" if accepted else "fallback_or_manual_review",
        "rejectReasons": reasons,
        "recommendedMethod": method,
        "recommendedConfidence": recommended.get("confidence"),
        "validState": recommended.get("validState"),
        "countBalanced": recommended.get("countBalanced"),
        "hamming": hamming if isinstance(hamming, int) else None,
        "exactMatch": hamming == 0 if isinstance(hamming, int) else None,
        "selectionReason": selection_reason,
        "currentRepairValid": current_valid,
        "thresholds": {
            "current": current_thresholds,
            "selected": selected_thresholds,
            "switched": switched_thresholds,
        },
        "productionRank": list(rank) if rank is not None else None,
    }


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    decisions = [evaluate_promotion_gate(row) for row in rows]
    accepted = [row for row in decisions if row["accepted"]]
    rejected = [row for row in decisions if not row["accepted"]]
    accepted_hammings = [row["hamming"] for row in accepted if isinstance(row.get("hamming"), int)]
    all_hammings = [row["hamming"] for row in decisions if isinstance(row.get("hamming"), int)]
    reject_reasons = Counter(reason for row in rejected for reason in row["rejectReasons"])
    return {
        "pairCount": len(decisions),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptedExact": sum(1 for value in accepted_hammings if value == 0),
        "acceptedLegal": sum(1 for row in accepted if row["validState"] is True),
        "acceptedWithin3": sum(1 for value in accepted_hammings if value <= 3),
        "allExact": sum(1 for value in all_hammings if value == 0),
        "allWithin3": sum(1 for value in all_hammings if value <= 3),
        "acceptedHammingDistribution": dict(sorted(Counter(str(value) for value in accepted_hammings).items())),
        "allHammingDistribution": dict(sorted(Counter(str(value) for value in all_hammings).items())),
        "acceptedMethodCounts": dict(Counter(str(row["recommendedMethod"]) for row in accepted).most_common()),
        "rejectedMethodCounts": dict(Counter(str(row["recommendedMethod"]) for row in rejected).most_common()),
        "selectionReasonCounts": dict(Counter(str(row["selectionReason"]) for row in decisions).most_common()),
        "thresholdSwitchCount": sum(1 for row in decisions if row["thresholds"]["switched"]),
        "acceptedThresholdSwitchCount": sum(1 for row in accepted if row["thresholds"]["switched"]),
        "rejectReasonCounts": dict(reject_reasons.most_common()),
        "rejectedSetIds": [row["setId"] for row in rejected],
    }


def render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Constrained-Inference Promotion Gate",
        "",
        "Diagnostic-only. This report evaluates a production-shaped gate for",
        "using the hull-label constrained-inference candidate outside Fixer.",
        "Ground truth is used only for scoring the gate, never for the gate",
        "decision itself.",
        "",
        f"Git head: `{payload['source']['git_sha']}`",
        f"Generated: `{payload['source']['generated_at_utc']}`",
        f"Input: `{payload['source']['input']}`",
        "",
        "## Gate",
        "",
        "A candidate may auto-return only when all of these production-available",
        "conditions hold:",
        "",
        "- the guarded pair-threshold selection assembled both sides;",
        "- the yaw inference is accepted when present;",
        "- both selected side thresholds were accepted and have no hard failures;",
        "- the recommended repair is valid, count-balanced, and not low confidence;",
        f"- the recommended method is one of `{sorted(ALLOWED_METHODS)}`;",
        f"- canonical count repair uses at most `{MAX_CANONICAL_REPAIR_MOVES}` moves;",
        f"- legal repair methods stay within state delta `{MAX_LEGAL_STATE_DELTA}`,",
        f"  repair cost `{MAX_LEGAL_REPAIR_COST}`, and repair changes `{MAX_LEGAL_REPAIR_CHANGES}`;",
        "- pair-threshold switching is allowed only when the current per-side",
        "  threshold repair was invalid.",
        "",
        "The ungated `broad_legal_repaired` method is intentionally excluded.",
        "",
        "## Summary",
        "",
        "| Pairs | Gate accepted | Gate rejected | Accepted exact | Accepted legal | Accepted <=3 | Threshold switches |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['pairCount']} | {summary['accepted']} | {summary['rejected']} | "
        f"{summary['acceptedExact']} | {summary['acceptedLegal']} | "
        f"{summary['acceptedWithin3']} | {summary['acceptedThresholdSwitchCount']} |",
        "",
        f"Accepted hamming distribution: `{summary['acceptedHammingDistribution']}`",
        "",
        "Accepted method counts:",
        "",
    ]
    for method, count in summary["acceptedMethodCounts"].items():
        lines.append(f"- `{method}`: `{count}`")
    lines.extend([
        "",
        "Selection reason counts:",
        "",
    ])
    for reason, count in summary["selectionReasonCounts"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend([
        "",
        "## Rejected Rows",
        "",
    ])
    if not summary["rejectedSetIds"]:
        lines.append("_None._")
    else:
        lines.append("| Set | Method | Reasons |")
        lines.append("|---:|---|---|")
        for row in payload["rows"]:
            if row["accepted"]:
                continue
            lines.append(
                f"| {row['setId']} | `{row['recommendedMethod']}` | "
                f"`{', '.join(row['rejectReasons'])}` |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Passing this diagnostic means the constrained-inference candidate is",
        "  coherent enough to run in a default-recognizer shadow lane or an",
        "  explicit candidate mode. It does not by itself flip `/api/recognize`.",
        "- The production flip should require the same gate to hold on shadow",
        "  traffic, plus a confidence policy for what to do when the gate rejects:",
        "  fall back to legacy, return manual-review, or ask for a retake.",
        "- The pair-threshold switch count is important: a switch is permitted",
        "  only as a rescue when current deterministic repair is invalid, which is",
        "  the Set 14 shape documented in the pair-threshold report.",
    ])
    return "\n".join(lines) + "\n"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    source = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [evaluate_promotion_gate(row) for row in source.get("rows", [])]
    payload = {
        "schema": "constrained_inference_promotion_gate_v1",
        "source": {
            "tool": "tools/validate_constrained_inference_promotion.py",
            "input": _rel(args.input),
            "git_sha": _git_head_sha(),
            "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "gate": {
            "maxCanonicalRepairMoves": MAX_CANONICAL_REPAIR_MOVES,
            "maxLegalStateDelta": MAX_LEGAL_STATE_DELTA,
            "maxLegalRepairCost": MAX_LEGAL_REPAIR_COST,
            "maxLegalRepairChanges": MAX_LEGAL_REPAIR_CHANGES,
            "allowedMethods": sorted(ALLOWED_METHODS),
            "allowedSelectionReasons": sorted(ALLOWED_SELECTION_REASONS),
        },
        "summary": build_summary(source.get("rows", [])),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
