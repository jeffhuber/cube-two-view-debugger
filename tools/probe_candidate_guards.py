"""Shared diagnostics-only candidate guards for probe outputs."""

from __future__ import annotations

from typing import Any, Dict


GRID_PURITY_TOP_COMPONENT_OVERLAP_MIN = 6
GRID_PURITY_TOP_LOW_SELF_FACE_CELLS_MAX = 2
GRID_PURITY_TOP_LOW_SELF_FACE_GRID_COUNT_MIN = 5
GRID_PURITY_TOP_WRONG_DOMINANT_MARGIN_MIN = 3
REPAIR_BACKFILL_OPPORTUNITY_CONFIDENCE_MAX = 0.65
REPAIR_BACKFILL_OPPORTUNITY_REPAIR_CHANGES_MIN = 8
REPAIR_BACKFILL_OPPORTUNITY_CONFLICTS_MIN = 8
REPAIR_BACKFILL_OPPORTUNITY_REASON = "repair_path_unstable_pre_repair_piece_evidence"


def selected_grid_purity_summary(signals: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize whether selected/top-visible grids are dominated by their expected face."""
    selected_rows = _selected_grid_purity_rows(signals.get("selectedGridQuality"))
    top_visible_rows = []
    top_component_overlaps = []
    top_visible = signals.get("topVisibleTripleQuality")
    if isinstance(top_visible, dict):
        for image, item in top_visible.items():
            if not isinstance(item, dict):
                continue
            component_overlap = _int_metric(item.get("componentOverlap"))
            top_component_overlaps.append(component_overlap)
            top_visible_rows.extend(
                _grid_purity_rows(
                    item.get("grids"),
                    image=str(image),
                    section="topVisibleTriple",
                    component_overlap=component_overlap,
                    side_pair=item.get("sidePair"),
                )
            )
    return {
        "schemaVersion": 1,
        "selectedRows": selected_rows,
        "topVisibleRows": top_visible_rows,
        "selectedLowSelfFaceCells": _count_low_self(
            selected_rows,
            threshold=GRID_PURITY_TOP_LOW_SELF_FACE_CELLS_MAX,
        ),
        "selectedVeryLowSelfFaceCells": _count_low_self(selected_rows, threshold=1),
        "topVisibleLowSelfFaceCells": _count_low_self(
            top_visible_rows,
            threshold=GRID_PURITY_TOP_LOW_SELF_FACE_CELLS_MAX,
        ),
        "topVisibleVeryLowSelfFaceCells": _count_low_self(top_visible_rows, threshold=1),
        "maxSelectedDominantWrongMargin": _max_metric(selected_rows, "dominantWrongMargin"),
        "maxTopVisibleDominantWrongMargin": _max_metric(top_visible_rows, "dominantWrongMargin"),
        "maxTopVisibleComponentOverlap": max(top_component_overlaps, default=0),
    }


def candidate_grid_purity_guard(purity_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Diagnostics-only candidate guard for Set-30-style selected-grid impurity."""
    max_overlap = _int_metric(purity_summary.get("maxTopVisibleComponentOverlap"))
    low_self_count = _int_metric(purity_summary.get("topVisibleLowSelfFaceCells"))
    max_wrong_margin = _int_metric(purity_summary.get("maxTopVisibleDominantWrongMargin"))
    rules = [
        {
            "name": "top_visible_overlap_and_low_self_purity",
            "wouldFire": max_overlap >= GRID_PURITY_TOP_COMPONENT_OVERLAP_MIN
            and low_self_count >= GRID_PURITY_TOP_LOW_SELF_FACE_GRID_COUNT_MIN
            and max_wrong_margin >= GRID_PURITY_TOP_WRONG_DOMINANT_MARGIN_MIN,
            "metrics": {
                "maxTopVisibleComponentOverlap": max_overlap,
                "topVisibleLowSelfFaceCells": low_self_count,
                "maxTopVisibleDominantWrongMargin": max_wrong_margin,
            },
            "thresholds": {
                "maxTopVisibleComponentOverlap": GRID_PURITY_TOP_COMPONENT_OVERLAP_MIN,
                "topVisibleLowSelfFaceGridCountMin": GRID_PURITY_TOP_LOW_SELF_FACE_GRID_COUNT_MIN,
                "lowSelfFaceCellsMax": GRID_PURITY_TOP_LOW_SELF_FACE_CELLS_MAX,
                "maxTopVisibleDominantWrongMargin": GRID_PURITY_TOP_WRONG_DOMINANT_MARGIN_MIN,
            },
        }
    ]
    fired = [rule["name"] for rule in rules if rule["wouldFire"]]
    return {
        "schemaVersion": 1,
        "policy": "diagnostics_only_no_behavior_change",
        "intendedUse": "candidate_manual_review_guard_not_promotion",
        "wouldFire": bool(fired),
        "firedRules": fired,
        "rules": rules,
    }


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


def _selected_grid_purity_rows(selected_grid_quality: Any) -> list[Dict[str, Any]]:
    rows = []
    if not isinstance(selected_grid_quality, dict):
        return rows
    for image, grids in selected_grid_quality.items():
        rows.extend(_grid_purity_rows(grids, image=str(image), section="selected"))
    return rows


def _grid_purity_rows(
    grids: Any,
    *,
    image: str,
    section: str,
    component_overlap: int | None = None,
    side_pair: Any = None,
) -> list[Dict[str, Any]]:
    rows = []
    if not isinstance(grids, dict):
        return rows
    for face, grid in grids.items():
        if not isinstance(grid, dict):
            continue
        face_counts = grid.get("cellFaceCounts") if isinstance(grid.get("cellFaceCounts"), dict) else {}
        source_counts = grid.get("cellSourceCounts") if isinstance(grid.get("cellSourceCounts"), dict) else {}
        self_cells = _int_metric(face_counts.get(face))
        dominant_face, dominant_cells = _dominant_face(face_counts)
        dominant_wrong_cells = dominant_cells if dominant_face not in (None, face) else 0
        row = {
            "section": section,
            "image": image,
            "face": face,
            "gridId": grid.get("gridId"),
            "sidePair": side_pair,
            "componentOverlap": component_overlap,
            "selfFaceCells": self_cells,
            "selfFaceRatio": round(self_cells / 9.0, 3),
            "dominantFace": dominant_face,
            "dominantFaceCells": dominant_cells,
            "dominantWrongFaceCells": dominant_wrong_cells,
            "dominantWrongMargin": max(0, dominant_wrong_cells - self_cells),
            "distinctFaceCount": len(face_counts),
            "componentCells": _int_metric(source_counts.get("component")),
            "gridSampleCells": _int_metric(source_counts.get("grid_sample")),
            "fitError": grid.get("fitError"),
            "componentShapeSpread": grid.get("componentShapeSpread"),
        }
        rows.append(row)
    return rows


def _dominant_face(face_counts: Dict[str, Any]) -> tuple[str | None, int]:
    if not face_counts:
        return None, 0
    face, count = max(face_counts.items(), key=lambda item: _int_metric(item[1]))
    return str(face), _int_metric(count)


def _count_low_self(rows: list[Dict[str, Any]], *, threshold: int) -> int:
    return sum(1 for row in rows if _int_metric(row.get("selfFaceCells")) <= threshold)


def _max_metric(rows: list[Dict[str, Any]], key: str) -> int:
    return max((_int_metric(row.get(key)) for row in rows), default=0)


def _numeric(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _int_metric(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0
