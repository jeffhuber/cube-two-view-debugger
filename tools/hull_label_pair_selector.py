"""Shared guarded pair-threshold selection for hull-label repair.

The hidden Fixer endpoint and the diagnostic report both choose among A/B
mask-threshold pairs after deterministic color repair. Keep the safety policy
in one place: retain the current per-side selector when it already produces a
valid repaired cube, and only switch threshold pairs when that current repair
is invalid.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple


RepairRank = Tuple[int, int, float, int, int]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _repair_payload(evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    repair = _mapping(evaluation.get("repair"))
    if repair:
        return repair
    payload = _mapping(evaluation.get("payload"))
    if payload:
        return payload
    if (
        isinstance(evaluation.get("methods"), Mapping)
        or isinstance(evaluation.get("recommended"), Mapping)
    ):
        return evaluation
    return {}


def _summary_payload(evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = _mapping(evaluation.get("summary"))
    return summary if summary else evaluation


def _method_from_payload(payload: Mapping[str, Any], camel_key: str, snake_key: str) -> Mapping[str, Any]:
    methods = _mapping(payload.get("methods"))
    method = _mapping(methods.get(snake_key))
    if method:
        return method
    return _mapping(payload.get(camel_key))


def recommended_repair(evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the recommended repair summary/payload for either caller schema."""
    payload = _repair_payload(evaluation)
    recommended = _mapping(payload.get("recommended"))
    if recommended:
        return recommended
    summary_recommended = _mapping(_summary_payload(evaluation).get("recommended"))
    return summary_recommended


def _state_delta_count(method: Mapping[str, Any]) -> Optional[int]:
    state_delta = _mapping(method.get("stateDeltaFromCanonical"))
    count = state_delta.get("count")
    return int(count) if isinstance(count, int) else None


def repair_valid(evaluation: Mapping[str, Any]) -> bool:
    """Whether an evaluation's recommended repair is already a valid cube."""
    return bool(recommended_repair(evaluation).get("validState"))


def repair_rank(evaluation: Mapping[str, Any]) -> RepairRank:
    """Rank a threshold-pair evaluation using production-available signals.

    Lower is better. This intentionally avoids ground-truth hamming, so the
    same rank can be used in production and diagnostics.
    """
    payload = _repair_payload(evaluation)
    if not payload:
        return (9, 99, 999.0, 99, int(evaluation.get("yawQuarterTurns") or 0))

    canonical = _method_from_payload(payload, "canonicalCount", "canonical_count_repaired")
    conservative = _method_from_payload(payload, "conservativeLegal", "conservative_legal_repaired")
    guarded = _method_from_payload(payload, "guardedBroadLegal", "guarded_broad_legal_repaired")
    recommended = recommended_repair(payload)

    if canonical.get("validState"):
        tier = 0
        primary = int(canonical.get("repairMoveCount") or 0)
        cost = 0.0
        changes = primary
    elif conservative.get("validState"):
        tier = 1
        primary = _state_delta_count(conservative)
        if primary is None:
            primary = int(conservative.get("repairChanges") or conservative.get("repairMoveCount") or 0)
        cost = float(conservative.get("repairCost") or 0.0)
        changes = int(conservative.get("repairChanges") or conservative.get("repairMoveCount") or primary)
    elif guarded.get("validState"):
        tier = 2
        primary = _state_delta_count(guarded)
        if primary is None:
            primary = int(guarded.get("repairChanges") or guarded.get("repairMoveCount") or 0)
        cost = float(guarded.get("repairCost") or 0.0)
        changes = int(guarded.get("repairChanges") or guarded.get("repairMoveCount") or primary)
    elif canonical.get("countBalanced"):
        tier = 3
        primary = int(canonical.get("repairMoveCount") or 99)
        cost = 0.0
        changes = primary
    else:
        tier = 4
        primary = _state_delta_count(recommended)
        if primary is None:
            primary = int(recommended.get("repairMoveCount") or 99)
        cost = float(recommended.get("repairCost") or 999.0)
        changes = int(recommended.get("repairChanges") or primary)
    return (
        tier,
        primary,
        cost,
        changes,
        int(evaluation.get("yawQuarterTurns") or payload.get("yawQuarterTurns") or 0),
    )


def _combo_status(combo: Mapping[str, Any]) -> Optional[str]:
    status = combo.get("status")
    if isinstance(status, str):
        return status
    evaluation = _mapping(combo.get("evaluation"))
    status = evaluation.get("status")
    return status if isinstance(status, str) else None


def _combo_rank(combo: Mapping[str, Any]) -> RepairRank:
    production_rank = combo.get("productionRank")
    if isinstance(production_rank, Sequence) and not isinstance(production_rank, (str, bytes)):
        try:
            return (
                int(production_rank[0]),
                int(production_rank[1]),
                float(production_rank[2]),
                int(production_rank[3]),
                int(production_rank[4]),
            )
        except (IndexError, TypeError, ValueError):
            pass
    evaluation = _mapping(combo.get("evaluation"))
    return repair_rank(evaluation if evaluation else combo)


def choose_pair_by_production_signals(
    combos: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Choose the best assembled threshold pair without using ground truth."""
    accepted = [combo for combo in combos if _combo_status(combo) == "assembled"]
    if not accepted:
        return None

    return min(
        accepted,
        key=lambda combo: (
            _combo_rank(combo),
            float(combo.get("stickerScoreTotal") or 999999.0),
            int(_mapping(combo.get("thresholds")).get("A") or 9999),
            int(_mapping(combo.get("thresholds")).get("B") or 9999),
        ),
    )


def choose_guarded_pair(
    *,
    current_combo: Optional[Mapping[str, Any]],
    current_eval: Optional[Mapping[str, Any]] = None,
    aggressive_pair: Optional[Mapping[str, Any]] = None,
    candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    fallback_to_current_without_alternative: bool = False,
) -> Optional[Mapping[str, Any]]:
    """Choose a threshold pair with the production safety guard.

    If the current per-side selector already yields a valid recommended state,
    keep it. Search is valuable only when current repair is invalid/unavailable;
    otherwise an alternate legal pair can still encode the wrong cube.
    """
    if current_eval is None and current_combo is not None:
        current_eval = _mapping(current_combo.get("evaluation")) or current_combo

    if current_combo is not None and current_eval is not None and repair_valid(current_eval):
        out = dict(current_combo)
        out["selectionReason"] = "kept_current_valid_repair"
        return out

    if aggressive_pair is None and candidates is not None:
        aggressive_pair = choose_pair_by_production_signals(candidates)

    if aggressive_pair is not None:
        out = dict(aggressive_pair)
        out["selectionReason"] = "current_invalid_selected_best_pair"
        return out

    if fallback_to_current_without_alternative and current_combo is not None:
        out = dict(current_combo)
        out["selectionReason"] = "kept_current_no_assembled_alternative"
        return out

    return None
