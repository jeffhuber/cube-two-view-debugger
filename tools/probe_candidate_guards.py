"""Shared diagnostics-only candidate guards for probe outputs."""

from __future__ import annotations

from typing import Any, Dict


REPAIR_BACKFILL_OPPORTUNITY_CONFIDENCE_MAX = 0.65
REPAIR_BACKFILL_OPPORTUNITY_REPAIR_CHANGES_MIN = 8
REPAIR_BACKFILL_OPPORTUNITY_CONFLICTS_MIN = 8
REPAIR_BACKFILL_OPPORTUNITY_REASON = "repair_path_unstable_pre_repair_piece_evidence"


def candidate_repair_backfill_opportunity(
    signals: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    repair_backfill_gate_would_apply: bool,
) -> Dict[str, Any]:
    """Diagnostics-only tag for Set-61-style skipped backfill opportunities."""
    selected = (
        signals.get("selectedRepairCandidate")
        if isinstance(signals.get("selectedRepairCandidate"), dict)
        else {}
    )
    conflicts = (
        selected.get("preRepairConflicts")
        if isinstance(selected.get("preRepairConflicts"), dict)
        else {}
    )
    repair_candidate_count = _int_metric(signals.get("repairCandidateCount"))
    repair_changes = _int_metric(selected.get("repairChanges"))
    total_conflicts = _int_metric(conflicts.get("totalConflicts"))
    confidence = _numeric(payload.get("confidence"))
    category = payload.get("recognitionCategory")
    category_reason = payload.get("recognitionCategoryReason")
    repair_backfill_attempted = bool(signals.get("repairBackfillAttempted"))

    unstable_manual_review_reason = (
        category == "needs_manual_review" and category_reason == REPAIR_BACKFILL_OPPORTUNITY_REASON
    )
    unstable_manual_review_metrics = (
        category == "needs_manual_review"
        and 0.0 < confidence <= REPAIR_BACKFILL_OPPORTUNITY_CONFIDENCE_MAX
        and (
            repair_changes >= REPAIR_BACKFILL_OPPORTUNITY_REPAIR_CHANGES_MIN
            or total_conflicts >= REPAIR_BACKFILL_OPPORTUNITY_CONFLICTS_MIN
        )
    )
    skipped_by_standard_repair = (
        repair_backfill_gate_would_apply
        and repair_candidate_count > 0
        and not repair_backfill_attempted
    )

    rules = [
        {
            "name": "skipped_backfill_with_unstable_standard_repair",
            "wouldFire": skipped_by_standard_repair
            and (unstable_manual_review_reason or unstable_manual_review_metrics),
            "metrics": {
                "repairBackfillGateWouldApply": repair_backfill_gate_would_apply,
                "repairBackfillAttempted": repair_backfill_attempted,
                "repairCandidateCount": repair_candidate_count,
                "recognitionCategory": category,
                "recognitionCategoryReason": category_reason,
                "confidence": confidence,
                "selectedRepairChanges": repair_changes,
                "selectedPreRepairConflicts": total_conflicts,
            },
            "thresholds": {
                "repairCandidateCountMin": 1,
                "confidenceMax": REPAIR_BACKFILL_OPPORTUNITY_CONFIDENCE_MAX,
                "selectedRepairChangesMin": REPAIR_BACKFILL_OPPORTUNITY_REPAIR_CHANGES_MIN,
                "selectedPreRepairConflictsMin": REPAIR_BACKFILL_OPPORTUNITY_CONFLICTS_MIN,
            },
        }
    ]
    fired = [rule["name"] for rule in rules if rule["wouldFire"]]
    return {
        "schemaVersion": 1,
        "policy": "diagnostics_only_no_behavior_change",
        "intendedUse": "candidate_repair_backfill_audit_not_promotion",
        "wouldFire": bool(fired),
        "firedRules": fired,
        "repairBackfillGateWouldApply": repair_backfill_gate_would_apply,
        "rules": rules,
    }


def _numeric(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _int_metric(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0
