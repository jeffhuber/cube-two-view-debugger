"""Shared constrained-inference promotion gate.

The gate decides whether a hull-label constrained-inference candidate is safe
to auto-return using only production-available signals. Ground truth scoring
belongs in diagnostics, not here.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


MAX_CANONICAL_REPAIR_MOVES = 10
MAX_LEGAL_STATE_DELTA = 4
MAX_TWO_VIEW_LEGAL_STATE_DELTA = 6
MAX_TWO_VIEW_SHADOW_RESCUE_REPAIR_COST = 10.0
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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rank_tuple(raw: Any) -> Optional[Tuple[int, int, float, int, int]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 5:
        return None
    try:
        return (int(raw[0]), int(raw[1]), float(raw[2]), int(raw[3]), int(raw[4]))
    except (TypeError, ValueError):
        return None


def _recommended_from_repair(repair: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(repair.get("recommended"))


def _recommended_from_combo(combo: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(combo.get("summary")).get("recommended"))


def _recommended_method_from_repair(repair: Mapping[str, Any]) -> Optional[str]:
    method = repair.get("recommendedMethod")
    return method if isinstance(method, str) else None


def _recommended_method_from_combo(combo: Mapping[str, Any]) -> Optional[str]:
    method = _mapping(combo.get("summary")).get("recommendedMethod")
    return method if isinstance(method, str) else None


def _valid_recommended(recommended: Mapping[str, Any]) -> bool:
    return recommended.get("validState") is True and recommended.get("countBalanced") is True


def _hard_failures(trace: Mapping[str, Any]) -> Sequence[Any]:
    raw = trace.get("hardFailures")
    if raw is None:
        raw = trace.get("hard_failures")
    return raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else []


def _runtime_rank(repair: Mapping[str, Any], yaw_quarter_turns: Optional[int]) -> Optional[Tuple[int, int, float, int, int]]:
    try:
        from tools.hull_label_pair_selector import repair_rank
    except Exception:  # noqa: BLE001
        return None
    return repair_rank({
        "repair": repair,
        "yawQuarterTurns": int(yaw_quarter_turns or repair.get("yawQuarterTurns") or 0),
    })


def evaluate_constrained_inference_gate(
    *,
    status: str,
    recommended_method: Optional[str],
    recommended: Mapping[str, Any],
    production_rank: Optional[Sequence[Any]],
    selection_reason: Optional[str],
    current_repair_valid: Optional[bool],
    current_thresholds: Mapping[str, Any],
    selected_thresholds: Mapping[str, Any],
    side_traces: Mapping[str, Mapping[str, Any]],
    yaw_inference: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate the shared GT-free auto-return gate."""
    reasons = []
    rank = _rank_tuple(production_rank)

    if status != "assembled":
        reasons.append("selected_pair_not_assembled")
    if selection_reason not in ALLOWED_SELECTION_REASONS:
        reasons.append("selection_reason_not_allowed")
    if recommended_method not in ALLOWED_METHODS:
        reasons.append("recommended_method_not_allowed")
    if not _valid_recommended(recommended):
        reasons.append("recommended_not_valid_balanced")
    if recommended.get("confidence") not in {"high", "medium"}:
        reasons.append("recommended_confidence_low_or_missing")
    if rank is None:
        reasons.append("missing_production_rank")

    switched_thresholds = bool(
        current_thresholds
        and selected_thresholds
        and dict(current_thresholds) != dict(selected_thresholds)
    )
    current_valid = bool(current_repair_valid)
    if selection_reason == "kept_current_valid_repair":
        if switched_thresholds:
            reasons.append("kept_current_reason_but_thresholds_switched")
        if not current_valid:
            reasons.append("kept_current_reason_but_current_invalid")
    if selection_reason == "current_invalid_selected_best_pair" and current_valid:
        reasons.append("switched_pair_while_current_valid")

    yaw = _mapping(yaw_inference)
    if yaw and yaw.get("accepted") is not True:
        reasons.append("yaw_not_accepted")

    for side in ("A", "B"):
        trace = _mapping(side_traces.get(side))
        if not trace:
            reasons.append(f"{side}_selected_threshold_trace_missing")
            continue
        accepted = trace.get("accepted")
        status_value = trace.get("status")
        if accepted is not True and status_value != "accepted":
            reasons.append(f"{side}_selected_threshold_not_accepted")
        if _hard_failures(trace):
            reasons.append(f"{side}_selected_threshold_has_hard_failures")

    if rank is not None and recommended_method == CANONICAL_METHOD:
        _tier, move_count, _cost, _changes, _yaw_rank = rank
        if int(move_count) > MAX_CANONICAL_REPAIR_MOVES:
            reasons.append("canonical_repair_moves_above_gate")
    elif rank is not None and recommended_method in LEGAL_METHODS:
        _tier, state_delta, repair_cost, repair_changes, _yaw_rank = rank
        max_state_delta = (
            MAX_TWO_VIEW_LEGAL_STATE_DELTA
            if recommended_method == "two_view_consistency_repaired"
            else MAX_LEGAL_STATE_DELTA
        )
        max_repair_cost = MAX_LEGAL_REPAIR_COST
        if (
            recommended_method == "two_view_consistency_repaired"
            and int(state_delta) > MAX_LEGAL_STATE_DELTA
        ):
            # Delta 5-6 two-view repairs are only considered safe after
            # _two_view_consistency_payload has accepted split-cubie evidence,
            # in-image shadow evidence, zero candidate cubie inconsistencies,
            # and the repair-change cap. The final gate mirrors the tighter
            # shadow-rescue cost cap as defense in depth.
            max_repair_cost = MAX_TWO_VIEW_SHADOW_RESCUE_REPAIR_COST
        if int(state_delta) > max_state_delta:
            reasons.append("legal_state_delta_above_gate")
        if float(repair_cost) > max_repair_cost:
            reasons.append("legal_repair_cost_above_gate")
        if int(repair_changes) > MAX_LEGAL_REPAIR_CHANGES:
            reasons.append("legal_repair_changes_above_gate")

    accepted = not reasons
    hamming = recommended.get("hamming")
    return {
        "schema": "constrained_inference_promotion_gate_decision_v1",
        "accepted": accepted,
        "decision": "auto_return_candidate" if accepted else "fallback_or_manual_review",
        "rejectReasons": reasons,
        "recommendedMethod": recommended_method,
        "recommendedConfidence": recommended.get("confidence"),
        "validState": recommended.get("validState"),
        "countBalanced": recommended.get("countBalanced"),
        "hamming": hamming if isinstance(hamming, int) else None,
        "exactMatch": hamming == 0 if isinstance(hamming, int) else None,
        "selectionReason": selection_reason,
        "currentRepairValid": current_valid,
        "thresholds": {
            "current": dict(current_thresholds),
            "selected": dict(selected_thresholds),
            "switched": switched_thresholds,
        },
        "productionRank": list(rank) if rank is not None else None,
    }


def evaluate_runtime_payload_gate(
    *,
    repair: Mapping[str, Any],
    pair_threshold_selection: Mapping[str, Any],
    side_traces: Mapping[str, Mapping[str, Any]],
    yaw_inference: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate the gate from the app/Fixer rectified payload shape."""
    rank = pair_threshold_selection.get("selectedProductionRank")
    if rank is None:
        rank = _runtime_rank(repair, repair.get("yawQuarterTurns"))
    return evaluate_constrained_inference_gate(
        status=str(repair.get("status") or ""),
        recommended_method=_recommended_method_from_repair(repair),
        recommended=_recommended_from_repair(repair),
        production_rank=rank,
        selection_reason=pair_threshold_selection.get("selectionReason"),
        current_repair_valid=pair_threshold_selection.get("currentRepairValid"),
        current_thresholds=_mapping(pair_threshold_selection.get("currentThresholds")),
        selected_thresholds=_mapping(pair_threshold_selection.get("selectedThresholds")),
        side_traces=side_traces,
        yaw_inference=yaw_inference,
    )


def _selected_candidate_trace(row: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    selected = _mapping(row.get("pairSelected"))
    thresholds = _mapping(selected.get("thresholds"))
    threshold = thresholds.get(side)
    diagnostics = _mapping(_mapping(row.get("thresholdDiagnostics")).get(side))
    for candidate in diagnostics.get("candidates") or []:
        if isinstance(candidate, Mapping) and candidate.get("threshold") == threshold:
            return candidate
    return {}


def evaluate_pair_threshold_row_gate(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate the gate from diagnose_pair_threshold_repair.py row shape."""
    selected = _mapping(row.get("pairSelected"))
    current = _mapping(row.get("current"))
    current_recommended = _recommended_from_combo(current)
    decision = evaluate_constrained_inference_gate(
        status=str(selected.get("status") or ""),
        recommended_method=_recommended_method_from_combo(selected),
        recommended=_recommended_from_combo(selected),
        production_rank=selected.get("productionRank"),
        selection_reason=selected.get("selectionReason"),
        current_repair_valid=current_recommended.get("validState") is True,
        current_thresholds=_mapping(current.get("thresholds")),
        selected_thresholds=_mapping(selected.get("thresholds")),
        side_traces={
            "A": _selected_candidate_trace(row, "A"),
            "B": _selected_candidate_trace(row, "B"),
        },
        yaw_inference=_mapping(selected.get("yawInference")),
    )
    decision["setId"] = row.get("setId")
    return decision


def build_gate_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    decisions = [evaluate_pair_threshold_row_gate(row) for row in rows]
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
