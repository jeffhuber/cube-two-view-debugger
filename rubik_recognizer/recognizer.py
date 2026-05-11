from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .colors import COLOR_TO_FACE, build_adaptive_palette, classify_rgb, rgb_to_hsv
from .geometry import TRANSFORMS, closest_edge, possible_transforms
from .image_pipeline import FaceGrid, ImageAnalysis, analyze_image
from .validation import CENTER_INDICES, CORNER_COLORS, CORNER_FACELETS, EDGE_COLORS, EDGE_FACELETS, FACE_ORDER, validate_state


FACE_TO_CENTER_COLOR = {
    "U": "white",
    "R": "red",
    "F": "green",
    "D": "yellow",
    "L": "orange",
    "B": "blue",
}

U_SIDE_EDGE = {"F": "bottom", "R": "right", "B": "top", "L": "left"}
D_SIDE_EDGE = {"F": "top", "R": "right", "B": "bottom", "L": "left"}
SIDE_NEIGHBORS = {
    "F": {"left": "L", "right": "R"},
    "R": {"left": "F", "right": "B"},
    "B": {"left": "R", "right": "L"},
    "L": {"left": "B", "right": "F"},
}
ADJACENT_SIDE_PAIRS = (("F", "R"), ("R", "B"), ("B", "L"), ("L", "F"))
YAW_SIDE_ORDER = ("F", "R", "B", "L")
IMAGE_A_YAW_BY_ORDERED_SIDE_PAIR = {
    ("F", "R"): 0,
    ("R", "B"): 1,
    ("B", "L"): 2,
    ("L", "F"): 3,
}
IMAGE_B_YAW_BY_ORDERED_SIDE_PAIR = {
    ("L", "B"): 0,
    ("F", "L"): 1,
    ("R", "F"): 2,
    ("B", "R"): 3,
}
MAX_ORIENTED_OPTIONS_PER_IMAGE = 220
MAX_ORIENTED_OPTIONS_PER_SIDE_PAIR = 128
MAX_VISIBLE_FACE_TRIPLES = 42
MAX_ORIENTATION_VARIANTS_PER_TRIPLE = 96
LOOSE_TRANSFORMS_PER_FACE = 8
ORIENTATION_SCORE_WEIGHT = 8.0
ANCHOR_FACE_EDGE_MATCH_WEIGHT = 2.0
SIDE_ANCHOR_EDGE_MATCH_WEIGHT = 3.0
SIDE_NEIGHBOR_EDGE_MATCH_WEIGHT = 1.0
MAX_COLOR_REPAIR_CHANGES = 7
MAX_COLOR_REPAIR_VARIANTS = 16
MAX_ANCHOR_GRID_CANDIDATES = 5
MAX_SIDE_GRID_CANDIDATES = 6
MAX_LEGAL_REPAIR_MERGES = 130
MAX_LEGAL_REPAIR_OPTIONS_PER_FACELET = 4
MAX_LEGAL_REPAIR_CHANGES = 12
MAX_LEGAL_REPAIR_PIECE_OPTIONS = 24
MAX_LEGAL_REPAIR_SOLUTIONS_PER_KEY = 5
MAX_LEGAL_REPAIR_SOLUTIONS = 90
MAX_LEGAL_REPAIR_COST = 95.0
MAX_LEGAL_REPAIR_MERGES_PER_PAIR = 60
MAX_LEGAL_REPAIR_DIVERSE_MERGES_PER_PAIR = 128
MAX_LEGAL_REPAIR_EVALUATED_MERGES = 180
MAX_LEGAL_REPAIR_RETURNED = 8
MAX_REPAIR_ALTERNATIVE_DELTA = 16.0
MAX_LOW_CONFIDENCE_REPAIR_DELTA = 24.0
LOW_CONFIDENCE_COMPONENT_REPAIR_THRESHOLD = 0.32
MAX_LOW_CONFIDENCE_COMPONENT_REPAIR_DELTA = 72.0
GRID_CONTEXT_REPAIR_THRESHOLD = 1.0
MAX_GRID_CONTEXT_COMPONENT_REPAIR_DELTA = 92.0
GRID_CONTEXT_REPAIR_DISTANCE_SCALE = 180.0
GRID_CONTEXT_REPAIR_DISTANCE_CAP = 0.55
GRID_CONTEXT_REPAIR_RANK_COST = 0.07
GRID_SAMPLE_REPAIR_FALLBACK_COST = 10.0
GRID_SAMPLE_REBALANCE_FALLBACK_COST = 38.0
MAX_DIAGNOSTIC_TRIPLES = 12
MAX_DIAGNOSTIC_MERGES = 5000
MAX_DIAGNOSTIC_EXAMPLES = 8
SUSPECT_GRID_SAMPLE_THRESHOLD = 2.5
MAX_SUSPECT_SAMPLE_ALTERNATIVE_DELTA = 58.0
MAX_TRIPLE_COMPONENT_OVERLAP = 3
# Tuned on labeled sets 12/14/15/24/26/27/28/29/31: large enough to
# down-rank conflicted repair winners, but capped so repair candidates remain
# comparable instead of being rejected by one binary threshold.
MAX_REPAIR_RANKING_PENALTY = 0.18
DIRECT_CLEAN_CONFIDENCE_THRESHOLD = 0.78
REPAIRED_HIGH_CONFIDENCE_THRESHOLD = 0.60
REPAIRED_HIGH_MAX_RANKING_PENALTY = 0.16
REPAIR_RETAKE_CONFIDENCE_THRESHOLD = 0.50
REPAIR_RETAKE_MIN_CANDIDATES = 50_000
REPAIR_ADJACENT_COLOR_PAIRS = {frozenset(("red", "orange")), frozenset(("green", "blue"))}
VALID_EDGE_COLOR_SETS = {frozenset(colors) for colors in EDGE_COLORS}
VALID_CORNER_COLOR_SETS = {frozenset(colors) for colors in CORNER_COLORS}
REPAIR_CONFLICT_PENALTY_WEIGHTS = {
    "missingCorners": 0.024,
    "duplicateColorCorners": 0.02,
    "missingUdCorners": 0.018,
    "invalidCorners": 0.024,
    "missingEdges": 0.018,
    "duplicateColorEdges": 0.014,
    "invalidEdges": 0.016,
    "duplicateCornerCubies": 0.018,
    "duplicateEdgeCubies": 0.012,
}
PIECE_CONFLICT_KEYS = (
    "missingCorners",
    "duplicateColorCorners",
    "missingUdCorners",
    "invalidCorners",
    "missingEdges",
    "duplicateColorEdges",
    "invalidEdges",
    "duplicateCornerCubies",
    "duplicateEdgeCubies",
    "validCorners",
    "validEdges",
    "totalConflicts",
)


@dataclass
class RecognitionResult:
    status: str
    state: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    failed_checks: List[str] | None = None
    image_a: Optional[ImageAnalysis] = None
    image_b: Optional[ImageAnalysis] = None
    candidates: int = 0
    recognition_signals: Optional[Dict[str, Any]] = None

    def to_api_dict(self, include_overlays: bool = True) -> Dict:
        category = _recognition_category_payload(self)
        payload = {
            "status": self.status,
            "state": self.state,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "recognitionCategory": category["category"],
            "recognitionCategoryReason": category["reason"],
            "failedChecks": self.failed_checks or [],
            "candidates": self.candidates,
        }
        if self.recognition_signals:
            payload["recognitionSignals"] = self.recognition_signals
        if self.image_a:
            payload["imageA"] = self.image_a.summary()
            payload["imageAAssignments"] = _assignment_summary(self.image_a, "U")
        if self.image_b:
            payload["imageB"] = self.image_b.summary()
            payload["imageBAssignments"] = _assignment_summary(self.image_b, "D")
        if include_overlays:
            payload["overlays"] = {
                "imageA": self.image_a.overlay_data_url if self.image_a else None,
                "imageB": self.image_b.overlay_data_url if self.image_b else None,
            }
        return payload


@dataclass(frozen=True)
class PieceOption:
    cubie: int
    orientation: int
    colors: Tuple[str, ...]
    cost: float
    changes: int


class WhiteUpRecognizer:
    def recognize(self, image_a: bytes, image_b: bytes) -> RecognitionResult:
        analysis_a = analyze_image(image_a)
        analysis_b = analyze_image(image_b)
        result = self._recognize_from_analyses(analysis_a, analysis_b)

        calibrated_a = copy.deepcopy(analysis_a)
        calibrated_b = copy.deepcopy(analysis_b)
        _apply_pair_color_calibration(calibrated_a, calibrated_b)
        calibrated_result = self._recognize_from_analyses(calibrated_a, calibrated_b)
        if calibrated_result.status == "success":
            calibrated_result.reason = f"{calibrated_result.reason} Adaptive pair color calibration was used."
            if result.status != "success" or _prefer_calibrated_result(calibrated_result, result):
                calibrated_result.image_a = calibrated_a
                calibrated_result.image_b = calibrated_b
                return calibrated_result

        if result.status == "success":
            result.image_a = analysis_a
            result.image_b = analysis_b
            return result

        if calibrated_result.status == "success":
            calibrated_result.image_a = calibrated_a
            calibrated_result.image_b = calibrated_b
            return calibrated_result

        result.image_a = analysis_a
        result.image_b = analysis_b
        return result

    def _recognize_from_analyses(self, analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> RecognitionResult:
        recognition_signals = _base_recognition_signals(analysis_a, analysis_b)
        checks = _white_up_checks(analysis_a, analysis_b)
        if checks:
            return RecognitionResult(
                status="rejected",
                reason=_reason_for_checks(checks),
                failed_checks=checks,
                recognition_signals=recognition_signals,
            )

        candidates = self._state_candidates(analysis_a, analysis_b)
        legal = []
        invalid_reasons: List[str] = []
        for state, confidence, details in candidates:
            validation = validate_state(state)
            if validation.valid:
                legal.append((state, confidence, details))
            else:
                invalid_reasons.extend(validation.errors)

        unique = {}
        unique_details = {}
        for state, confidence, details in legal:
            if confidence > unique.get(state, 0.0):
                unique[state] = confidence
                unique_details[state] = details

        if len(unique) == 1:
            state, confidence = next(iter(unique.items()))
            validation = validate_state(state)
            if validation.valid:
                return RecognitionResult(
                    status="success",
                    state=state,
                    confidence=confidence,
                    reason="Recognized a unique legal white-up cube state.",
                    candidates=len(candidates),
                    recognition_signals={**recognition_signals, **_selected_faces_signal(unique_details.get(state), state=state)},
                )
            return RecognitionResult(
                status="rejected",
                reason="The only candidate failed cube legality validation.",
                failed_checks=validation.errors,
                candidates=len(candidates),
                recognition_signals=recognition_signals,
            )

        if len(unique) > 1:
            ranked_unique = sorted(unique.items(), key=lambda item: item[1], reverse=True)
            if ranked_unique[0][1] > ranked_unique[1][1]:
                return RecognitionResult(
                    status="success",
                    state=ranked_unique[0][0],
                    confidence=ranked_unique[0][1],
                    reason="Recognized the highest-scoring legal white-up cube state.",
                    candidates=len(candidates),
                    recognition_signals={
                        **recognition_signals,
                        **_selected_faces_signal(unique_details.get(ranked_unique[0][0]), state=ranked_unique[0][0]),
                    },
                )
            return RecognitionResult(
                status="rejected",
                reason="Multiple legal cube states match the visible stickers.",
                failed_checks=["ambiguous_legal_completion"],
                candidates=len(candidates),
                recognition_signals=recognition_signals,
            )

        repair_details = self._legal_repair_candidate_details(analysis_a, analysis_b)
        repair_candidates = [(item["state"], item["confidence"]) for item in repair_details]
        recognition_signals.update(_repair_signal_summary(repair_details))
        if repair_candidates:
            candidates.extend(repair_candidates)
            repaired_unique = {}
            for state, confidence in repair_candidates:
                validation = validate_state(state)
                if validation.valid:
                    repaired_unique[state] = max(confidence, repaired_unique.get(state, 0.0))
            if len(repaired_unique) == 1:
                state, confidence = next(iter(repaired_unique.items()))
                return RecognitionResult(
                    status="success",
                    state=state,
                    confidence=confidence,
                    reason="Recognized a legal white-up cube state after cubie-level color repair.",
                    candidates=len(candidates),
                    recognition_signals={**recognition_signals, **_repair_signal_summary(repair_details, selected_state=state)},
                )
            if len(repaired_unique) > 1:
                ranked_repaired = sorted(repaired_unique.items(), key=lambda item: item[1], reverse=True)
                if ranked_repaired[0][1] > ranked_repaired[1][1]:
                    state = ranked_repaired[0][0]
                    return RecognitionResult(
                        status="success",
                        state=state,
                        confidence=ranked_repaired[0][1],
                        reason="Recognized the highest-scoring legal cube state after cubie-level color repair.",
                        candidates=len(candidates),
                        recognition_signals={**recognition_signals, **_repair_signal_summary(repair_details, selected_state=state)},
                    )

        return RecognitionResult(
            status="rejected",
            reason="No legal cube state matched the detected stickers.",
            failed_checks=_summarize_validation_errors(invalid_reasons),
            candidates=len(candidates),
            recognition_signals=recognition_signals,
        )

    def _state_candidates(self, analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> List[Tuple[str, float, Dict[str, Any]]]:
        options_a = _oriented_face_options(analysis_a, "U")
        options_b = _oriented_face_options(analysis_b, "D")
        candidates: List[Tuple[str, float, Dict[str, Any]]] = []
        for _, merged in _merged_face_candidates(options_a, options_b):
            details = _candidate_selection_detail(merged)
            for partial in _state_variants_from_faces(merged):
                confidence = _state_confidence(merged)
                candidates.append((partial, confidence, details))
        return candidates

    def _legal_repair_candidates(self, analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> List[Tuple[str, float]]:
        return [(item["state"], item["confidence"]) for item in self._legal_repair_candidate_details(analysis_a, analysis_b)]

    def _legal_repair_candidate_details(self, analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> List[Dict[str, Any]]:
        options_a = _oriented_face_options(analysis_a, "U")
        options_b = _oriented_face_options(analysis_b, "D")
        candidates: Dict[str, Dict[str, Any]] = {}
        for raw_score, merged in _repair_merged_face_candidates(options_a, options_b)[:MAX_LEGAL_REPAIR_EVALUATED_MERGES]:
            state, repair_cost, changes = _legal_repaired_state_from_faces(merged) or (None, 0.0, 0)
            if state is None:
                continue
            conflicts = _piece_conflict_summary(merged)
            base_confidence = _state_confidence(merged) - repair_cost / 650.0 - changes * 0.006
            penalty = _repair_ranking_penalty(
                conflicts,
                merged,
                repair_cost=repair_cost,
                repair_changes=changes,
            )
            confidence = base_confidence - penalty
            confidence = max(0.5, confidence)
            detail = {
                "state": state,
                "confidence": confidence,
                "baseConfidence": base_confidence,
                "repairRankingPenalty": penalty,
                "rawMergedScore": raw_score,
                "repairCost": repair_cost,
                "repairChanges": changes,
                "sidePairA": _side_pair_key(merged.get("_side_pair_a")),
                "sidePairB": _side_pair_key(merged.get("_side_pair_b")),
                "orderedSidePairA": _ordered_side_pair_key(merged.get("_ordered_side_pair_a")),
                "orderedSidePairB": _ordered_side_pair_key(merged.get("_ordered_side_pair_b")),
                "scoreA": merged.get("_score_a"),
                "scoreB": merged.get("_score_b"),
                "orientationScoreA": merged.get("_orientation_score_a"),
                "orientationScoreB": merged.get("_orientation_score_b"),
                "selectionScoreA": merged.get("_selection_score_a"),
                "selectionScoreB": merged.get("_selection_score_b"),
                "orientationRankA": merged.get("_orientation_rank_a"),
                "orientationRankB": merged.get("_orientation_rank_b"),
                "preRepairConflicts": conflicts,
                "preRepairFaceCounts": _primary_face_counts(merged),
            }
            current = candidates.get(state)
            if current is None or confidence > current["confidence"]:
                candidates[state] = detail
        return sorted(candidates.values(), key=lambda item: item["confidence"], reverse=True)[:MAX_LEGAL_REPAIR_RETURNED]


def _prefer_calibrated_result(calibrated: RecognitionResult, raw: RecognitionResult) -> bool:
    calibrated_tier = _recognition_reliability_tier(calibrated)
    raw_tier = _recognition_reliability_tier(raw)
    if calibrated_tier > raw_tier:
        return True
    if calibrated_tier == raw_tier and calibrated.confidence > raw.confidence + 0.03:
        return True
    return calibrated.confidence > raw.confidence + 0.06


def _recognition_reliability_tier(result: RecognitionResult) -> int:
    if result.status != "success":
        return -1
    if "color repair" in result.reason:
        return 1
    if "highest-scoring legal" in result.reason:
        return 2
    return 3


def _recognition_category_payload(result: RecognitionResult) -> Dict[str, str]:
    signals = result.recognition_signals or {}
    if result.status != "success" or not result.state:
        return {
            "category": "reject_retake",
            "reason": "recognizer_rejected",
        }

    repair_used = bool(signals.get("repairPathUsed")) or "color repair" in result.reason
    capture_yaw = signals.get("captureYaw") if isinstance(signals.get("captureYaw"), dict) else {}
    if capture_yaw.get("status") == "conflict":
        return {
            "category": "needs_manual_review",
            "reason": "conflicting_capture_yaw",
        }
    if capture_yaw.get("status") == "nonstandard" and not capture_yaw.get("normalizationApplied"):
        return {
            "category": "needs_manual_review",
            "reason": "nonstandard_capture_yaw_without_normalization",
        }

    if not repair_used:
        if (
            "unique legal" in result.reason
            and result.confidence >= DIRECT_CLEAN_CONFIDENCE_THRESHOLD
            and _weak_selected_grid_count(signals) == 0
        ):
            return {
                "category": "success_clean",
                "reason": "direct_unique_legal_high_confidence",
            }
        return {
            "category": "needs_manual_review",
            "reason": "direct_legal_but_low_margin_or_grid_quality",
        }

    selected = signals.get("selectedRepairCandidate") or {}
    penalty = _float_signal(selected.get("repairRankingPenalty"))
    repair_candidate_count = max(
        _int_signal(signals.get("repairCandidateCount")),
        _int_signal(result.candidates),
    )
    if result.confidence <= REPAIR_RETAKE_CONFIDENCE_THRESHOLD or repair_candidate_count < REPAIR_RETAKE_MIN_CANDIDATES:
        return {
            "category": "reject_retake",
            "reason": "repair_path_floor_confidence_or_too_few_candidates",
        }
    if (
        result.confidence >= REPAIRED_HIGH_CONFIDENCE_THRESHOLD
        and penalty < REPAIRED_HIGH_MAX_RANKING_PENALTY
    ):
        return {
            "category": "success_repaired_high_confidence",
            "reason": "repair_path_high_confidence_low_penalty",
        }
    return {
        "category": "needs_manual_review",
        "reason": "repair_path_low_confidence_or_high_conflict",
    }


def _weak_selected_grid_count(signals: Dict[str, Any]) -> int:
    count = 0
    quality_by_image = signals.get("selectedGridQuality") or {}
    selected_faces_by_image = signals.get("selectedFacesByImage") or {}
    for image_key, image_quality in quality_by_image.items():
        if not isinstance(image_quality, dict):
            continue
        selected_faces = selected_faces_by_image.get(image_key) if isinstance(selected_faces_by_image, dict) else None
        visible_faces = set(selected_faces or ())
        for face_key, grid in image_quality.items():
            # When the winning visible-face triple is known, ignore diagnostic
            # artifact grids that were not used by the recognition. If older or
            # rejected responses omit the signal, keep the original conservative
            # behavior and count every grid quality entry.
            if visible_faces and face_key not in visible_faces:
                continue
            if not isinstance(grid, dict):
                continue
            matched_count = _int_signal(grid.get("matchedCount"))
            fit_error = _float_signal(grid.get("fitError"))
            quality = _float_signal(grid.get("quality"), default=100.0)
            bad_samples = _int_signal(grid.get("badSamples"))
            suspect_samples = _float_signal(grid.get("suspectSamples"))
            if matched_count < 5 or fit_error > 12.0 or quality < 75.0 or bad_samples > 3 or suspect_samples > 4.0:
                count += 1
    return count


def _float_signal(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_signal(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def recognition_diagnostics(analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> Dict[str, Any]:
    options_a = _oriented_face_options(analysis_a, "U")
    options_b = _oriented_face_options(analysis_b, "D")
    merged = _merged_face_candidates(options_a, options_b)
    candidate_counts = _candidate_face_count_diagnostics(merged)
    return {
        "imageA": _orientation_diagnostics(analysis_a, "U", options_a),
        "imageB": _orientation_diagnostics(analysis_b, "D", options_b),
        "mergedCandidates": {
            "optionsA": len(options_a),
            "optionsB": len(options_b),
            "merged": len(merged),
            "sidePairCombos": _counter_items(
                Counter((_side_pair_key(item[1].get("_side_pair_a")), _side_pair_key(item[1].get("_side_pair_b"))) for item in merged)
            ),
            **candidate_counts,
        },
    }


def _base_recognition_signals(analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> Dict[str, Any]:
    return {
        "schemaVersion": 1,
        "repairPathUsed": False,
        "repairCandidateCount": 0,
        "selectedGridQuality": {
            "imageA": _selected_grid_quality(analysis_a, "U"),
            "imageB": _selected_grid_quality(analysis_b, "D"),
        },
    }


def _selected_grid_quality(analysis: ImageAnalysis, anchor: str) -> Dict[str, Dict[str, Any]]:
    return {
        face: _grid_signal_summary(grid)
        for face, grid in _assigned_grid_by_face(analysis, anchor).items()
    }


def _grid_signal_summary(grid: FaceGrid) -> Dict[str, Any]:
    return {
        "gridId": grid.id,
        "centerFace": grid.center_face,
        "matchedCount": grid.matched_count,
        "fitError": round(grid.fit_error, 3),
        "quality": round(_grid_quality_score(grid), 3),
        "gridSamples": _grid_sample_count(grid),
        "badSamples": _grid_bad_sample_count(grid),
        "suspectSamples": round(_grid_suspect_sample_score(grid), 3),
    }


def _repair_signal_summary(repair_details: Sequence[Dict[str, Any]], selected_state: Optional[str] = None) -> Dict[str, Any]:
    selected = next((item for item in repair_details if item.get("state") == selected_state), None) if selected_state else None
    summary: Dict[str, Any] = {
        "repairPathUsed": bool(selected_state),
        "repairCandidateCount": len(repair_details),
        "topRepairCandidates": [_public_repair_detail(item) for item in repair_details[:MAX_LEGAL_REPAIR_RETURNED]],
    }
    if selected is not None:
        summary["selectedRepairCandidate"] = _public_repair_detail(selected)
        summary.update(_selected_faces_signal(selected, state=selected_state))
    return summary


def _candidate_selection_detail(merged: Dict[str, List[List[Any]]]) -> Dict[str, Any]:
    return {
        "sidePairA": _side_pair_key(merged.get("_side_pair_a")),
        "sidePairB": _side_pair_key(merged.get("_side_pair_b")),
        "orderedSidePairA": _ordered_side_pair_key(merged.get("_ordered_side_pair_a")),
        "orderedSidePairB": _ordered_side_pair_key(merged.get("_ordered_side_pair_b")),
    }


def _selected_faces_signal(selection: Optional[Dict[str, Any]], state: Optional[str] = None) -> Dict[str, Any]:
    if not selection:
        return {}
    faces = _selected_faces_by_image(selection.get("sidePairA"), selection.get("sidePairB"))
    sides = _selected_sides_by_image(selection.get("orderedSidePairA"), selection.get("orderedSidePairB"))
    signal: Dict[str, Any] = {}
    if faces:
        signal["selectedFacesByImage"] = faces
    if sides:
        signal["selectedSidesByImage"] = sides
        signal["captureYaw"] = _capture_yaw_signal(sides, state=state)
    return signal


def _selected_faces_by_image(side_pair_a: Any, side_pair_b: Any) -> Dict[str, List[str]]:
    faces_a = _selected_anchor_faces("U", side_pair_a)
    faces_b = _selected_anchor_faces("D", side_pair_b)
    if not faces_a or not faces_b:
        return {}
    return {
        "imageA": faces_a,
        "imageB": faces_b,
    }


def _selected_sides_by_image(ordered_side_pair_a: Any, ordered_side_pair_b: Any) -> Dict[str, Dict[str, str]]:
    sides_a = _selected_side_positions(ordered_side_pair_a)
    sides_b = _selected_side_positions(ordered_side_pair_b)
    if not sides_a or not sides_b:
        return {}
    return {
        "imageA": sides_a,
        "imageB": sides_b,
    }


def _selected_side_positions(ordered_side_pair: Any) -> Dict[str, str]:
    faces = _ordered_side_pair_faces(ordered_side_pair)
    if len(faces) != 2:
        return {}
    return {"left": faces[0], "right": faces[1]}


def _capture_yaw_signal(selected_sides: Dict[str, Dict[str, str]], state: Optional[str] = None) -> Dict[str, Any]:
    observations = []
    yaw_values: List[int] = []
    for image_key, lookup in (
        ("imageA", IMAGE_A_YAW_BY_ORDERED_SIDE_PAIR),
        ("imageB", IMAGE_B_YAW_BY_ORDERED_SIDE_PAIR),
    ):
        sides = selected_sides.get(image_key) if isinstance(selected_sides, dict) else None
        if not isinstance(sides, dict):
            continue
        pair = (sides.get("left"), sides.get("right"))
        yaw = lookup.get(pair)
        observation = {
            "image": image_key,
            "orderedSidePair": "/".join(face for face in pair if face),
        }
        if yaw is not None:
            observation["yawQuarterTurns"] = yaw
            yaw_values.append(yaw)
        else:
            observation["yawQuarterTurns"] = None
        observations.append(observation)

    if not yaw_values:
        return {
            "status": "unknown",
            "quarterTurns": None,
            "degrees": None,
            "observations": observations,
        }
    if len(set(yaw_values)) > 1:
        return {
            "status": "conflict",
            "quarterTurns": None,
            "degrees": None,
            "observations": observations,
        }
    yaw = yaw_values[0]
    return {
        "status": "standard" if yaw == 0 else "nonstandard",
        "quarterTurns": yaw,
        "degrees": yaw * 90,
        "requiresNormalization": yaw != 0,
        "normalizationApplied": bool(state) and yaw != 0,
        "stateFrame": "wca",
        **({"captureFrameState": _state_to_capture_yaw(state, yaw)} if state else {}),
        "observations": observations,
    }


def _state_to_capture_yaw(state: str, yaw: int) -> str:
    """Return the photo/Fixer-frame state for a canonical WCA state.

    The recognizer's public ``state`` stays solver-ready URFDLB with WCA centers.
    For a white-up capture yawed around the U/D axis, the photo frame shifts the
    side face chunks while the U and D stickers rotate in opposite directions.
    """
    if len(state) != 54:
        return state
    yaw %= 4
    chunks = _state_chunks(state)
    capture_to_wca = _capture_to_wca_yaw_map(yaw)
    capture_chunks: Dict[str, str] = {}
    for face in FACE_ORDER:
        source_face = capture_to_wca[face]
        chunk = chunks[source_face]
        if face == "U":
            chunk = _rotate_face_clockwise_n(chunk, yaw)
        elif face == "D":
            chunk = _rotate_face_clockwise_n(chunk, -yaw)
        capture_chunks[face] = chunk
    return "".join(capture_chunks[face] for face in FACE_ORDER)


def _capture_yaw_state_to_wca(state: str, yaw: int) -> str:
    if len(state) != 54:
        return state
    yaw %= 4
    capture_chunks = _state_chunks(state)
    capture_to_wca = _capture_to_wca_yaw_map(yaw)
    wca_chunks: Dict[str, str] = {}
    for capture_face, wca_face in capture_to_wca.items():
        chunk = capture_chunks[capture_face]
        if capture_face == "U":
            chunk = _rotate_face_clockwise_n(chunk, -yaw)
        elif capture_face == "D":
            chunk = _rotate_face_clockwise_n(chunk, yaw)
        wca_chunks[wca_face] = chunk
    return "".join(wca_chunks[face] for face in FACE_ORDER)


def _capture_to_wca_yaw_map(yaw: int) -> Dict[str, str]:
    yaw %= 4
    mapping = {"U": "U", "D": "D"}
    for index, capture_face in enumerate(YAW_SIDE_ORDER):
        mapping[capture_face] = YAW_SIDE_ORDER[(index + yaw) % len(YAW_SIDE_ORDER)]
    return mapping


def _state_chunks(state: str) -> Dict[str, str]:
    return {face: state[index * 9 : (index + 1) * 9] for index, face in enumerate(FACE_ORDER)}


def _rotate_face_clockwise_n(face: str, turns: int) -> str:
    turns %= 4
    rotated = face
    for _ in range(turns):
        rotated = _rotate_face_clockwise(rotated)
    return rotated


def _rotate_face_clockwise(face: str) -> str:
    return "".join(face[index] for index in (6, 3, 0, 7, 4, 1, 8, 5, 2))


def _selected_anchor_faces(anchor: str, side_pair: Any) -> List[str]:
    sides = _side_pair_faces(side_pair)
    if len(sides) != 2:
        return []
    return sorted({anchor, *sides})


def _side_pair_faces(side_pair: Any) -> List[str]:
    key = _side_pair_key(side_pair)
    if not key:
        return []
    parts = key.split("/")
    if len(parts) != 2 or any(face not in SIDE_NEIGHBORS for face in parts):
        return []
    return parts if len(set(parts)) == len(parts) else []


def _ordered_side_pair_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return "/".join(str(part) for part in value)


def _ordered_side_pair_faces(side_pair: Any) -> List[str]:
    key = _ordered_side_pair_key(side_pair)
    if not key:
        return []
    parts = key.split("/")
    if len(parts) != 2 or any(face not in SIDE_NEIGHBORS for face in parts):
        return []
    return parts if len(set(parts)) == len(parts) else []


def _public_repair_detail(item: Dict[str, Any]) -> Dict[str, Any]:
    public = dict(item)
    for key in (
        "confidence",
        "baseConfidence",
        "repairRankingPenalty",
        "rawMergedScore",
        "repairCost",
        "scoreA",
        "scoreB",
        "orientationScoreA",
        "orientationScoreB",
        "selectionScoreA",
        "selectionScoreB",
    ):
        if public.get(key) is not None:
            public[key] = round(float(public[key]), 4)
    return public


def _orientation_diagnostics(analysis: ImageAnalysis, anchor: str, options: Sequence[Dict[str, List[List[Any]]]]) -> Dict[str, Any]:
    groups = _candidate_grids_by_face(analysis, anchor)
    triples = _ranked_visible_face_triples(groups, anchor) if anchor in groups else []
    return {
        "anchor": anchor,
        "gridGroups": {
            face: [
                {
                    "gridId": grid.id,
                    "matchedCount": grid.matched_count,
                    "fitError": round(grid.fit_error, 3),
                    "quality": round(_grid_quality_score(grid), 3),
                    "gridSamples": _grid_sample_count(grid),
                    "badSamples": _grid_bad_sample_count(grid),
                    "suspectSamples": round(_grid_suspect_sample_score(grid), 3),
                }
                for grid in grids
            ]
            for face, grids in groups.items()
        },
        "topTriples": [
            {
                "score": round(score, 3),
                "sidePair": _side_pair_key(face for face in subset if face in SIDE_NEIGHBORS),
                "grids": {face: grid.id for face, grid in subset.items()},
                "componentOverlap": _triple_overlap_count(tuple(subset.values())),
            }
            for score, subset in triples[:MAX_DIAGNOSTIC_TRIPLES]
        ],
        "tripleFilter": _triple_filter_diagnostics(groups, anchor),
        "orientedOptions": {
            "count": len(options),
            "sidePairs": _counter_items(Counter(_side_pair_key(option.get("_side_pair")) for option in options)),
        },
    }


def _candidate_face_count_diagnostics(merged: Sequence[Tuple[float, Dict[str, List[List[Any]]]]]) -> Dict[str, Any]:
    face_counts: Counter[Tuple[Tuple[str, int], ...]] = Counter()
    validation_errors: Counter[str] = Counter()
    examples = []
    legal_count = 0
    sampled = 0

    for score, faces in merged[:MAX_DIAGNOSTIC_MERGES]:
        for state in _state_variants_from_faces(faces):
            sampled += 1
            counts = tuple((face, state.count(face)) for face in FACE_ORDER)
            face_counts[counts] += 1
            validation = validate_state(state)
            if validation.valid:
                legal_count += 1
            else:
                validation_errors.update(validation.errors)
                if len(examples) < MAX_DIAGNOSTIC_EXAMPLES:
                    examples.append(
                        {
                            "score": round(score, 3),
                            "sidePairA": _side_pair_key(faces.get("_side_pair_a")),
                            "sidePairB": _side_pair_key(faces.get("_side_pair_b")),
                            "counts": {face: count for face, count in counts},
                            "validationErrors": validation.errors[:8],
                            "state": state,
                        }
                    )

    return {
        "sampled": sampled,
        "sampleLimit": min(len(merged), MAX_DIAGNOSTIC_MERGES),
        "legalInSample": legal_count,
        "validationErrors": _counter_items(validation_errors),
        "faceCounts": [
            {"counts": {face: count for face, count in counts}, "n": total}
            for counts, total in face_counts.most_common(12)
        ],
        "examples": examples,
    }


def _counter_items(counter: Counter[Any], limit: int = 12) -> List[Dict[str, Any]]:
    return [{"key": key, "n": count} for key, count in counter.most_common(limit)]


def _side_pair_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return "/".join(str(part) for part in sorted(value))


def _merged_face_candidates(
    options_a: Sequence[Dict[str, List[List[Any]]]],
    options_b: Sequence[Dict[str, List[List[Any]]]],
) -> List[Tuple[float, Dict[str, List[List[Any]]]]]:
    merged_faces: List[Tuple[float, Dict[str, List[List[Any]]]]] = []
    for faces_a in options_a:
        for faces_b in options_b:
            merged = _merge_faces(faces_a, faces_b)
            if merged is None:
                continue
            merged["_option_signature_a"] = _face_signature(faces_a)
            merged["_option_signature_b"] = _face_signature(faces_b)
            merged_faces.append((float(merged.get("_score", 0.0)), merged))
    merged_faces.sort(key=lambda item: item[0], reverse=True)
    return merged_faces


def _repair_merged_face_candidates(
    options_a: Sequence[Dict[str, List[List[Any]]]],
    options_b: Sequence[Dict[str, List[List[Any]]]],
) -> List[Tuple[float, Dict[str, List[List[Any]]]]]:
    ranked = _merged_face_candidates(options_a, options_b)
    selected: List[Tuple[float, Dict[str, List[List[Any]]]]] = ranked[:MAX_LEGAL_REPAIR_MERGES]
    by_pair: Dict[Tuple[object, object], List[Tuple[float, Dict[str, List[List[Any]]]]]] = {}
    for score, merged in ranked:
        key = (merged.get("_side_pair_a"), merged.get("_side_pair_b"))
        by_pair.setdefault(key, []).append((score, merged))
    for bucket in by_pair.values():
        selected.extend(bucket[:MAX_LEGAL_REPAIR_MERGES_PER_PAIR])
        selected.extend(_diverse_repair_items(bucket, "_option_signature_a", MAX_LEGAL_REPAIR_DIVERSE_MERGES_PER_PAIR))
        selected.extend(_diverse_repair_items(bucket, "_option_signature_b", MAX_LEGAL_REPAIR_DIVERSE_MERGES_PER_PAIR))

    deduped: List[Tuple[float, Dict[str, List[List[Any]]]]] = []
    seen = set()
    for score, merged in sorted(selected, key=lambda item: item[0], reverse=True):
        key = (
            merged.get("_side_pair_a"),
            merged.get("_side_pair_b"),
            _face_signature(merged),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append((score, merged))
    return deduped


def _diverse_repair_items(
    bucket: Sequence[Tuple[float, Dict[str, List[List[Any]]]]],
    signature_key: str,
    limit: int,
) -> List[Tuple[float, Dict[str, List[List[Any]]]]]:
    selected = []
    seen = set()
    for score, merged in bucket:
        signature = merged.get(signature_key)
        if signature is None or signature in seen:
            continue
        seen.add(signature)
        selected.append((score, merged))
        if len(selected) >= limit:
            break
    return selected


def _white_up_checks(analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> List[str]:
    checks = []
    expectations = (("image_a", analysis_a, "U"), ("image_b", analysis_b, "D"))
    for label, analysis, anchor in expectations:
        if len(analysis.grids) < 3:
            checks.append(f"{label}_missing_three_grids")
        assignments = _assigned_grid_by_face(analysis, anchor)
        if anchor not in assignments:
            checks.append(f"{label}_{anchor}_anchor_missing")
        if _can_rank_grid_triples(analysis):
            groups = _candidate_grids_by_face(analysis, anchor)
            if anchor in groups and not _ranked_visible_face_triples(groups, anchor):
                checks.append(f"{label}_no_reliable_face_triple")
    side_centers = [
        face
        for analysis, anchor in ((analysis_a, "U"), (analysis_b, "D"))
        for face in _assigned_grid_by_face(analysis, anchor)
        if face in {"R", "F", "L", "B"}
    ]
    if set(side_centers) != {"R", "F", "L", "B"}:
        checks.append("missing_side_face_coverage")
    return checks


def _reason_for_checks(checks: Sequence[str]) -> str:
    if "image_a_U_anchor_missing" in checks:
        return "Image A must contain the white/U center face; a logo is allowed if the sampled center is still white-ish."
    if "image_b_D_anchor_missing" in checks:
        return "Image B must contain the yellow/D center face after the flip."
    if "image_a_no_reliable_face_triple" in checks:
        return "Image A did not contain a reliable non-overlapping three-face grid."
    if "image_b_no_reliable_face_triple" in checks:
        return "Image B did not contain a reliable non-overlapping three-face grid."
    if "missing_side_face_coverage" in checks:
        return "The two flip photos do not expose all four side face centers."
    return "The images did not satisfy the two-view flip recognition prerequisites."


def _apply_pair_color_calibration(analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> None:
    samples = _pair_calibration_samples(analysis_a, analysis_b)
    anchors = _pair_calibration_anchors(analysis_a, analysis_b)
    palette = build_adaptive_palette(samples, anchors)
    for analysis in (analysis_a, analysis_b):
        for sticker in _analysis_stickers_for_reclassification(analysis):
            sticker.match = classify_rgb(sticker.rgb, palette)


def _pair_calibration_samples(*analyses: ImageAnalysis) -> List[Tuple[int, int, int]]:
    samples: List[Tuple[int, int, int]] = []
    seen = set()
    for analysis in analyses:
        for sticker in analysis.stickers:
            key = (sticker.id, sticker.rgb, tuple(round(value, 1) for value in sticker.center))
            if key in seen:
                continue
            seen.add(key)
            samples.append(sticker.rgb)
    return samples


def _pair_calibration_anchors(analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> Dict[str, List[Tuple[int, int, int]]]:
    anchors: Dict[str, List[Tuple[int, int, int]]] = {color: [] for color in FACE_TO_CENTER_COLOR.values()}
    for analysis, anchor in ((analysis_a, "U"), (analysis_b, "D")):
        for face, grid in _assigned_grid_by_face(analysis, anchor).items():
            color = FACE_TO_CENTER_COLOR.get(face)
            if color:
                anchors[color].append(grid.center_sticker.rgb)
    return anchors


def _analysis_stickers_for_reclassification(analysis: ImageAnalysis) -> List[Any]:
    stickers: List[Any] = []
    seen = set()
    for sticker in analysis.stickers:
        stickers.append(sticker)
        seen.add(id(sticker))
    for grid in analysis.grids:
        for row in grid.stickers:
            for sticker in row:
                if id(sticker) in seen:
                    continue
                stickers.append(sticker)
                seen.add(id(sticker))
    return stickers


def _summarize_validation_errors(errors: Sequence[str]) -> List[str]:
    if not errors:
        return ["no_legal_state"]
    unique = set(errors)
    summary = [error for error in sorted(unique) if error.endswith("_count_not_9") or error.endswith("_center_invalid")]
    if any(error.startswith("corner_") or error.startswith("edge_") for error in unique):
        summary.append("piece_legality_invalid")
    return summary or sorted(unique)


def _oriented_face_options(analysis: ImageAnalysis, anchor: str) -> List[Dict[str, List[List[Any]]]]:
    grids_by_face = _candidate_grids_by_face(analysis, anchor)
    if anchor not in grids_by_face:
        return []
    side_faces = [face for face in grids_by_face if face in SIDE_NEIGHBORS]
    if len(side_faces) < 2:
        return []

    ranked_results: List[Tuple[float, Tuple[str, ...], Tuple[str, ...], float, int, Dict[str, List[List[Any]]]]] = []
    for triple_score, subset in _ranked_visible_face_triples(grids_by_face, anchor):
        side_pair = tuple(sorted(face for face in subset if face in SIDE_NEIGHBORS))
        ordered_side_pair = _photo_ordered_side_pair(subset)
        for rank, (orientation_score, option) in enumerate(
            _oriented_options_for_grid_map(subset, anchor)[:MAX_ORIENTATION_VARIANTS_PER_TRIPLE]
        ):
            selection_score = triple_score - rank * 0.02
            ranked_results.append((selection_score, side_pair, ordered_side_pair, orientation_score, rank, option))
    ranked_results.sort(key=lambda item: item[0], reverse=True)

    scored_options = []
    seen = set()
    side_pair_counts: Counter[Tuple[str, ...]] = Counter()
    side_pair_kinds = {side_pair for _, side_pair, _, _, _, _ in ranked_results}
    per_side_pair_limit = min(
        MAX_ORIENTED_OPTIONS_PER_SIDE_PAIR,
        max(24, math.ceil(MAX_ORIENTED_OPTIONS_PER_IMAGE / max(1, len(side_pair_kinds)))),
    )
    for selection_score, side_pair, ordered_side_pair, orientation_score, orientation_rank, option in ranked_results:
        signature = (side_pair, _face_signature(option))
        if signature in seen:
            continue
        if side_pair_counts[side_pair] >= per_side_pair_limit:
            continue
        seen.add(signature)
        side_pair_counts[side_pair] += 1
        scored = dict(option)
        scored["_score"] = selection_score + orientation_score * ORIENTATION_SCORE_WEIGHT
        scored["_selection_score"] = selection_score
        scored["_orientation_score"] = orientation_score
        scored["_orientation_rank"] = orientation_rank
        scored["_side_pair"] = side_pair
        scored["_ordered_side_pair"] = ordered_side_pair
        scored_options.append(scored)
        if len(scored_options) >= MAX_ORIENTED_OPTIONS_PER_IMAGE:
            break
    return scored_options


def _ranked_visible_face_triples(grids_by_face: Dict[str, List[FaceGrid]], anchor: str) -> List[Tuple[float, Dict[str, FaceGrid]]]:
    triples: List[Tuple[float, Dict[str, FaceGrid]]] = []
    for side_pair in ADJACENT_SIDE_PAIRS:
        if side_pair[0] not in grids_by_face or side_pair[1] not in grids_by_face:
            continue
        for anchor_grid, first_grid, second_grid in product(
            grids_by_face[anchor][:MAX_ANCHOR_GRID_CANDIDATES],
            grids_by_face[side_pair[0]][:MAX_SIDE_GRID_CANDIDATES],
            grids_by_face[side_pair[1]][:MAX_SIDE_GRID_CANDIDATES],
        ):
            if len({anchor_grid.id, first_grid.id, second_grid.id}) < 3:
                continue
            if not all(_grid_usable_for_triple(grid) for grid in (anchor_grid, first_grid, second_grid)):
                continue
            if _triple_overlap_count((anchor_grid, first_grid, second_grid)) > MAX_TRIPLE_COMPONENT_OVERLAP:
                continue
            subset = {anchor: anchor_grid, side_pair[0]: first_grid, side_pair[1]: second_grid}
            triples.append((_face_plane_score(anchor_grid, first_grid, second_grid), subset))
    triples.sort(key=lambda item: item[0], reverse=True)
    return triples[:MAX_VISIBLE_FACE_TRIPLES]


def _photo_ordered_side_pair(grid_by_face: Dict[str, FaceGrid]) -> Tuple[str, ...]:
    side_items = [(face, _grid_center_x(grid)) for face, grid in grid_by_face.items() if face in SIDE_NEIGHBORS]
    if len(side_items) != 2:
        return tuple(face for face, _ in side_items)
    return tuple(face for face, _ in sorted(side_items, key=lambda item: (item[1], item[0])))


def _grid_center_x(grid: FaceGrid) -> float:
    points = getattr(grid, "points", None)
    if points:
        xs = [point[0] for row in points for point in row]
        if xs:
            return sum(xs) / len(xs)
    center = getattr(getattr(grid, "center_sticker", None), "center", None)
    if center:
        return float(center[0])
    return 0.0


def _triple_filter_diagnostics(grids_by_face: Dict[str, List[FaceGrid]], anchor: str) -> Dict[str, int]:
    counts = Counter()
    for side_pair in ADJACENT_SIDE_PAIRS:
        if side_pair[0] not in grids_by_face or side_pair[1] not in grids_by_face or anchor not in grids_by_face:
            continue
        for anchor_grid, first_grid, second_grid in product(
            grids_by_face[anchor][:MAX_ANCHOR_GRID_CANDIDATES],
            grids_by_face[side_pair[0]][:MAX_SIDE_GRID_CANDIDATES],
            grids_by_face[side_pair[1]][:MAX_SIDE_GRID_CANDIDATES],
        ):
            counts["total"] += 1
            if len({anchor_grid.id, first_grid.id, second_grid.id}) < 3:
                counts["duplicateGridId"] += 1
                continue
            if not all(_grid_usable_for_triple(grid) for grid in (anchor_grid, first_grid, second_grid)):
                counts["unusableGrid"] += 1
                continue
            if _triple_overlap_count((anchor_grid, first_grid, second_grid)) > MAX_TRIPLE_COMPONENT_OVERLAP:
                counts["componentOverlap"] += 1
                continue
            counts["accepted"] += 1
    return {key: counts[key] for key in ("total", "accepted", "duplicateGridId", "unusableGrid", "componentOverlap") if counts[key]}


def _face_plane_score(anchor_grid: FaceGrid, first_grid: FaceGrid, second_grid: FaceGrid) -> float:
    grids = (anchor_grid, first_grid, second_grid)
    score = sum(grid.matched_count * 12.0 - grid.fit_error * 0.9 for grid in grids)
    score -= sum(min(65.0, _grid_shape_spread(grid)) for grid in grids) * 0.75
    score -= sum(_grid_sample_penalty(grid) for grid in grids) * 0.9

    anchor_x, anchor_y = _grid_center(anchor_grid)
    side_centers = [_grid_center(first_grid), _grid_center(second_grid)]
    spacing = max(20.0, sum(_grid_spacing(grid) for grid in grids) / 3.0)
    triangle_area = abs(
        (side_centers[0][0] - anchor_x) * (side_centers[1][1] - anchor_y)
        - (side_centers[1][0] - anchor_x) * (side_centers[0][1] - anchor_y)
    ) / max(spacing * spacing, 1.0)
    score += max(-18.0, min(28.0, (triangle_area - 0.35) * 22.0))

    score += _axis_model_score(grids)
    score += _shape_axis_diversity_score(grids)
    score += _adjacency_score(anchor_grid, first_grid, spacing)
    score += _adjacency_score(anchor_grid, second_grid, spacing)
    score += _adjacency_score(first_grid, second_grid, spacing) * 0.45
    score -= _triple_overlap_count(grids) * 16.0
    return score


def _axis_model_score(grids: Sequence[FaceGrid]) -> float:
    pair_scores = []
    for i, grid in enumerate(grids):
        for other in grids[i + 1 :]:
            pair_scores.append(_shared_axis_score(grid, other))
    return sum(pair_scores) / max(1, len(pair_scores))


def _shape_axis_diversity_score(grids: Sequence[FaceGrid]) -> float:
    means = [_grid_shape_mean(grid) for grid in grids]
    if any(mean is None for mean in means):
        return 0.0
    score = 0.0
    for index, first in enumerate(means):
        for second in means[index + 1 :]:
            diff = _angle_diff(float(first), float(second))
            if diff < 16.0:
                score -= (16.0 - diff) * 4.2
            else:
                score += min(diff, 65.0) * 0.18
    return score


def _shared_axis_score(grid: FaceGrid, other: FaceGrid) -> float:
    diffs = [_angle_diff(a, b) for a in _grid_axis_angles(grid) for b in _grid_axis_angles(other)]
    best = min(diffs) if diffs else 90.0
    return max(-18.0, 30.0 - best * 1.7)


def _adjacency_score(grid: FaceGrid, other: FaceGrid, spacing: float) -> float:
    distance = _closest_grid_distance(grid, other)
    return max(-24.0, 28.0 - (distance / spacing) * 22.0)


def _triple_overlap_count(grids: Sequence[FaceGrid]) -> int:
    matched = [_matched_component_ids(grid) for grid in grids]
    overlap = 0
    for i, ids in enumerate(matched):
        for other in matched[i + 1 :]:
            overlap += len(ids & other)
    return overlap


def _matched_component_ids(grid: FaceGrid) -> set[int]:
    return {sticker.id for row in grid.stickers for sticker in row if sticker.source == "component"}


def _closest_grid_distance(grid: FaceGrid, other: FaceGrid) -> float:
    return min(math.hypot(a[0] - b[0], a[1] - b[1]) for row in grid.points for a in row for other_row in other.points for b in other_row)


def _grid_axis_angles(grid: FaceGrid) -> Tuple[float, float]:
    row = (grid.points[1][2][0] - grid.points[1][1][0], grid.points[1][2][1] - grid.points[1][1][1])
    col = (grid.points[2][1][0] - grid.points[1][1][0], grid.points[2][1][1] - grid.points[1][1][1])
    return _undirected_angle(row), _undirected_angle(col)


def _grid_shape_angles(grid: FaceGrid) -> List[float]:
    return [
        float(sticker.shape_angle)
        for row in getattr(grid, "stickers", [])
        for sticker in row
        if sticker.source == "component" and sticker.shape_angle is not None
    ]


def _grid_shape_mean(grid: FaceGrid) -> Optional[float]:
    angles = _grid_shape_angles(grid)
    if len(angles) < 4:
        return None
    doubled = [math.radians(angle * 2.0) for angle in angles]
    x = sum(math.cos(angle) for angle in doubled)
    y = sum(math.sin(angle) for angle in doubled)
    if abs(x) < 1e-6 and abs(y) < 1e-6:
        return None
    return (math.degrees(math.atan2(y, x)) / 2.0) % 180.0


def _grid_shape_spread(grid: FaceGrid) -> float:
    angles = _grid_shape_angles(grid)
    if len(angles) < 4:
        return 45.0
    doubled = [math.radians(angle * 2.0) for angle in angles]
    x = sum(math.cos(angle) for angle in doubled) / len(doubled)
    y = sum(math.sin(angle) for angle in doubled) / len(doubled)
    concentration = max(1e-6, min(1.0, math.hypot(x, y)))
    return math.degrees(math.sqrt(max(0.0, -2.0 * math.log(concentration)))) / 2.0


def _grid_usable_for_triple(grid: FaceGrid) -> bool:
    if grid.matched_count < 5:
        return False
    suspect = _grid_suspect_sample_score(grid)
    if grid.matched_count <= 5 and suspect >= 5.0:
        return False
    if grid.matched_count <= 6 and _grid_bad_sample_count(grid) >= 4:
        return False
    return True


def _grid_sample_penalty(grid: FaceGrid) -> float:
    return _grid_sample_count(grid) * 2.5 + _grid_suspect_sample_score(grid) * 18.0


def _grid_sample_count(grid: FaceGrid) -> int:
    return sum(1 for row in getattr(grid, "stickers", []) for sticker in row if getattr(sticker, "source", "") == "grid_sample")


def _grid_suspect_sample_score(grid: FaceGrid) -> float:
    return sum(_suspect_grid_sample_score(sticker) for row in getattr(grid, "stickers", []) for sticker in row)


def _grid_bad_sample_count(grid: FaceGrid) -> int:
    return sum(
        1
        for row in getattr(grid, "stickers", [])
        for sticker in row
        if _suspect_grid_sample_score(sticker) >= SUSPECT_GRID_SAMPLE_THRESHOLD
    )


def _suspect_grid_sample_score(sticker: Any) -> float:
    if getattr(sticker, "source", "") != "grid_sample":
        return 0.0

    hue, saturation, value = rgb_to_hsv(sticker.rgb)
    color = sticker.match.color
    confidence = sticker.match.confidence
    score = 0.0

    if value < 0.20:
        score += 3.0
    elif value < 0.34 and confidence < 0.35:
        score += 1.5

    if confidence < 0.18:
        score += 1.4
    elif confidence < 0.28:
        score += 0.7

    if color == "white":
        if saturation > 0.24:
            score += 2.2
        if value < 0.55:
            score += 1.6
    elif saturation < 0.24:
        score += 1.7

    # Warm, weakly saturated desk samples commonly sit in the orange/yellow hue
    # band while remaining much less saturated than actual cube stickers.
    if 0.04 <= hue <= 0.16 and 0.18 <= saturation <= 0.42 and color in {"white", "orange", "yellow"}:
        score += 1.6

    return score


def _undirected_angle(vector: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(vector[1], vector[0])) % 180.0


def _angle_diff(first: float, second: float) -> float:
    diff = abs(first - second) % 180.0
    return min(diff, 180.0 - diff)


def _grid_center(grid: FaceGrid) -> Point:
    return grid.points[1][1]


def _grid_spacing(grid: FaceGrid) -> float:
    distances = []
    for r in range(3):
        for c in range(2):
            distances.append(math.hypot(grid.points[r][c][0] - grid.points[r][c + 1][0], grid.points[r][c][1] - grid.points[r][c + 1][1]))
    for r in range(2):
        for c in range(3):
            distances.append(math.hypot(grid.points[r][c][0] - grid.points[r + 1][c][0], grid.points[r][c][1] - grid.points[r + 1][c][1]))
    return sorted(distances)[len(distances) // 2] if distances else 80.0


def _oriented_options_for_grid_map(grid_by_face: Dict[str, FaceGrid], anchor: str) -> List[Tuple[float, Dict[str, List[List[Any]]]]]:
    requirements: Dict[str, Dict[str, str]] = {face: {} for face in grid_by_face}
    requirement_weights: Dict[str, Dict[str, float]] = {face: {} for face in grid_by_face}
    anchor_grid = grid_by_face[anchor]
    anchor_edges = U_SIDE_EDGE if anchor == "U" else D_SIDE_EDGE
    side_anchor_edge = "top" if anchor == "U" else "bottom"
    for side, grid in grid_by_face.items():
        if side not in anchor_edges:
            continue
        requirements[anchor][anchor_edges[side]] = closest_edge(anchor_grid.points, grid.points)
        requirement_weights[anchor][anchor_edges[side]] = ANCHOR_FACE_EDGE_MATCH_WEIGHT
        requirements[side][side_anchor_edge] = closest_edge(grid.points, anchor_grid.points)
        requirement_weights[side][side_anchor_edge] = SIDE_ANCHOR_EDGE_MATCH_WEIGHT

    side_faces = [face for face in grid_by_face if face in SIDE_NEIGHBORS]
    for face in side_faces:
        for other in side_faces:
            if face == other:
                continue
            for edge, neighbor in SIDE_NEIGHBORS[face].items():
                if neighbor == other:
                    observed = closest_edge(grid_by_face[face].points, grid_by_face[other].points)
                    requirements[face][edge] = observed
                    requirement_weights[face][edge] = SIDE_NEIGHBOR_EDGE_MATCH_WEIGHT

    transform_options = {face: _ranked_transforms(reqs, requirement_weights[face]) for face, reqs in requirements.items()}

    faces = list(transform_options)
    ranked: List[Tuple[float, float, Dict[str, List[List[Any]]]]] = []
    for combo in product(*(transform_options[face] for face in faces)):
        oriented: Dict[str, List[List[Any]]] = {}
        sort_score = 0.0
        score = 0.0
        for face, transform in zip(faces, combo):
            transform_score = _transform_weighted_match_score(transform, requirements[face], requirement_weights[face])
            sort_score += transform_score
            score += transform_score
            matrix = _grid_matrix_for_orientation(grid_by_face[face])
            matrix[1][1] = face
            oriented[face] = transform.apply(matrix)  # type: ignore[assignment]
        plausibility = _visible_piece_plausibility_score(oriented)
        sort_score += plausibility
        score += plausibility
        ranked.append((sort_score, score, oriented))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(score, oriented) for _, score, oriented in ranked[:MAX_ORIENTED_OPTIONS_PER_SIDE_PAIR]]


def _grid_matrix_for_orientation(grid: FaceGrid) -> List[List[Any]]:
    flex = _grid_context_repair_score(grid)
    return [
        [_grid_contextual_facelet(sticker, grid, flex) for sticker in row]
        for row in grid.stickers
    ]


def _grid_contextual_facelet(sticker: Any, grid: FaceGrid, flex: float) -> Any:
    if flex < GRID_CONTEXT_REPAIR_THRESHOLD or getattr(sticker, "source", "component") != "component":
        return sticker
    contextual = copy.copy(sticker)
    setattr(contextual, "grid_repair_flex", flex)
    setattr(contextual, "grid_context_id", getattr(grid, "id", None))
    return contextual


def _grid_context_repair_score(grid: FaceGrid) -> float:
    score = 0.0
    if grid.matched_count <= 5:
        score += 1.6
    elif grid.matched_count <= 6:
        score += 0.9
    if grid.fit_error >= 12.0:
        score += 1.3
    elif grid.fit_error >= 7.0:
        score += 0.8
    if _grid_sample_count(grid) >= 3:
        score += 0.8
    if _grid_bad_sample_count(grid) >= 1:
        score += 0.8
    if _grid_quality_score(grid) < 35.0:
        score += 0.8
    return score


def _visible_piece_plausibility_score(faces: Dict[str, List[List[Any]]]) -> float:
    score = 0.0
    for indices in EDGE_FACELETS:
        colors = _visible_piece_colors(faces, indices)
        if colors is None:
            continue
        if len(set(colors)) != len(colors):
            score -= 8.0
        elif frozenset(colors) in VALID_EDGE_COLOR_SETS:
            score += 9.0
        else:
            score -= 14.0

    for indices in CORNER_FACELETS:
        colors = _visible_piece_colors(faces, indices)
        if colors is None:
            continue
        if len(set(colors)) != len(colors):
            score -= 10.0
        elif frozenset(colors) in VALID_CORNER_COLOR_SETS:
            score += 12.0
        else:
            score -= 18.0
    return score


def _visible_piece_colors(faces: Dict[str, List[List[Any]]], indices: Sequence[int]) -> Optional[Tuple[str, ...]]:
    colors = []
    for index in indices:
        face_index, offset = divmod(index, 9)
        face = FACE_ORDER[face_index]
        matrix = faces.get(face)
        if matrix is None:
            return None
        row, col = divmod(offset, 3)
        color = _primary_facelet_color(matrix[row][col])
        if color not in FACE_ORDER:
            return None
        colors.append(color)
    return tuple(colors)


def _piece_conflict_summary(faces: Dict[str, List[List[Any]]]) -> Dict[str, int]:
    summary = Counter()
    corner_sets: Counter[frozenset[str]] = Counter()
    edge_sets: Counter[frozenset[str]] = Counter()

    for indices in CORNER_FACELETS:
        colors = _visible_piece_colors(faces, indices)
        if colors is None:
            summary["missingCorners"] += 1
            continue
        color_set = frozenset(colors)
        if len(set(colors)) != len(colors):
            summary["duplicateColorCorners"] += 1
        if not any(color in {"U", "D"} for color in colors):
            summary["missingUdCorners"] += 1
        if color_set in VALID_CORNER_COLOR_SETS:
            corner_sets[color_set] += 1
        else:
            summary["invalidCorners"] += 1

    for indices in EDGE_FACELETS:
        colors = _visible_piece_colors(faces, indices)
        if colors is None:
            summary["missingEdges"] += 1
            continue
        color_set = frozenset(colors)
        if len(set(colors)) != len(colors):
            summary["duplicateColorEdges"] += 1
        if color_set in VALID_EDGE_COLOR_SETS:
            edge_sets[color_set] += 1
        else:
            summary["invalidEdges"] += 1

    summary["duplicateCornerCubies"] = sum(count - 1 for count in corner_sets.values() if count > 1)
    summary["duplicateEdgeCubies"] = sum(count - 1 for count in edge_sets.values() if count > 1)
    summary["validCorners"] = sum(corner_sets.values())
    summary["validEdges"] = sum(edge_sets.values())
    summary["totalConflicts"] = sum(
        summary[key]
        for key in (
            "missingCorners",
            "duplicateColorCorners",
            "missingUdCorners",
            "invalidCorners",
            "missingEdges",
            "duplicateColorEdges",
            "invalidEdges",
            "duplicateCornerCubies",
            "duplicateEdgeCubies",
        )
    )
    return {key: summary[key] for key in PIECE_CONFLICT_KEYS}


def _repair_ranking_penalty(
    conflicts: Dict[str, int],
    faces: Dict[str, List[List[Any]]],
    *,
    repair_cost: float,
    repair_changes: int,
) -> float:
    conflict_penalty = sum(conflicts.get(key, 0) * weight for key, weight in REPAIR_CONFLICT_PENALTY_WEIGHTS.items())
    orientation_penalty = min(
        0.035,
        0.001
        * (
            _nonnegative_int(faces.get("_orientation_rank_a"))
            + _nonnegative_int(faces.get("_orientation_rank_b"))
        ),
    )
    face_count_penalty = 0.003 * _face_count_deviation(_primary_face_counts(faces))
    heavy_repair_penalty = max(0, repair_changes - 4) * 0.004 + max(0.0, repair_cost - 35.0) / 900.0
    return min(
        MAX_REPAIR_RANKING_PENALTY,
        conflict_penalty + orientation_penalty + face_count_penalty + heavy_repair_penalty,
    )


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _face_count_deviation(counts: Dict[str, int]) -> int:
    return sum(abs(counts.get(face, 0) - 9) for face in FACE_ORDER) + counts.get("unknown", 0)


def _primary_face_counts(faces: Dict[str, List[List[Any]]]) -> Dict[str, int]:
    counts = Counter()
    for face in FACE_ORDER:
        matrix = faces.get(face)
        if matrix is None:
            continue
        for row in matrix:
            for facelet in row:
                color = _primary_facelet_color(facelet)
                counts[color if color in FACE_ORDER else "unknown"] += 1
    return {face: counts[face] for face in (*FACE_ORDER, "unknown") if counts[face]}


def _primary_facelet_color(facelet: Any) -> Optional[str]:
    if isinstance(facelet, str):
        return facelet if facelet in FACE_ORDER else None
    match = getattr(facelet, "match", None)
    if match is not None and getattr(match, "face", None) in FACE_ORDER:
        return match.face
    face = getattr(facelet, "face", None)
    return face if face in FACE_ORDER else None


def _ranked_transforms(requirements: Dict[str, str], weights: Optional[Dict[str, float]] = None):
    strict = possible_transforms(requirements)
    ranked = sorted(
        TRANSFORMS,
        key=lambda transform: (
            _transform_weighted_match_score(transform, requirements, weights),
            _opposite_edge_match_count(transform, requirements),
        ),
        reverse=True,
    )
    strict_names = {transform.name for transform in strict}
    ordered = strict + [transform for transform in ranked if transform.name not in strict_names]
    return ordered[:LOOSE_TRANSFORMS_PER_FACE]


def _transform_match_count(transform, requirements: Dict[str, str]) -> int:
    return sum(1 for canonical, observed in requirements.items() if transform.edge_map.get(canonical) == observed)


def _transform_weighted_match_score(transform, requirements: Dict[str, str], weights: Optional[Dict[str, float]] = None) -> float:
    return sum(
        weights.get(canonical, 1.0) if weights else 1.0
        for canonical, observed in requirements.items()
        if transform.edge_map.get(canonical) == observed
    )


def _opposite_edge_match_count(transform, requirements: Dict[str, str]) -> int:
    return sum(
        1
        for canonical, observed in requirements.items()
        if _opposite_edge(transform.edge_map.get(canonical)) == observed
    )


def _opposite_edge(edge: Optional[str]) -> Optional[str]:
    return {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}.get(edge)


def _assigned_grid_by_face(analysis: ImageAnalysis, anchor: str) -> Dict[str, FaceGrid]:
    """Assign visible grid identities for one half of the two-view flip.

    Image A is anchored by U/white; Image B is anchored by D/yellow. The U center
    can contain a multi-color logo, so a white-ish center sample is acceptable for
    A. B should have a clean yellow center after the flip.
    """
    if not analysis.grids:
        return {}

    groups = _candidate_grids_by_face(analysis, anchor)
    can_rank_triples = _can_rank_grid_triples(analysis)
    triples = _ranked_visible_face_triples(groups, anchor) if can_rank_triples and anchor in groups else []
    assigned: Dict[str, FaceGrid] = dict(triples[0][1]) if triples else {}

    if anchor not in assigned:
        anchor_grid = _assumed_anchor_grid(analysis.grids, anchor)
        if anchor_grid is not None:
            assigned[anchor] = anchor_grid
    for face, grids in groups.items():
        if face == anchor or face not in {"R", "F", "L", "B"}:
            continue
        current = assigned.get(face)
        best = grids[0]
        if current is None or _grid_quality_score(best) > _grid_quality_score(current):
            assigned[face] = best
    return assigned


def _candidate_grids_by_face(analysis: ImageAnalysis, anchor: str) -> Dict[str, List[FaceGrid]]:
    groups: Dict[str, List[FaceGrid]] = {}
    anchor_candidates = _anchor_grid_candidates(analysis.grids, anchor)
    if anchor_candidates:
        groups[anchor] = anchor_candidates
    for grid in analysis.grids:
        face = grid.center_face
        if face not in {"R", "F", "L", "B"}:
            continue
        groups.setdefault(face, []).append(grid)
    return {face: _unique_ranked_grids(grids) for face, grids in groups.items()}


def _can_rank_grid_triples(analysis: ImageAnalysis) -> bool:
    return all(hasattr(grid, "id") and hasattr(grid, "points") and hasattr(grid, "stickers") for grid in analysis.grids)


def _anchor_grid_candidates(grids: Sequence[FaceGrid], anchor: str) -> List[FaceGrid]:
    candidates = [grid for grid in grids if grid.center_face == anchor]
    if anchor == "U":
        candidates.extend(grid for grid in grids if grid.center_face != anchor and _center_sample_is_whiteish(grid))
    elif anchor == "D":
        candidates.extend(grid for grid in grids if grid.center_face != anchor and _center_sample_is_yellowish(grid))
    return _unique_ranked_grids(candidates)


def _unique_ranked_grids(grids: Sequence[FaceGrid]) -> List[FaceGrid]:
    by_id = {getattr(grid, "id", id(grid)): grid for grid in grids}
    return sorted(by_id.values(), key=_grid_quality_score, reverse=True)


def _assumed_anchor_grid(grids: Sequence[FaceGrid], anchor: str) -> Optional[FaceGrid]:
    direct_grids = [grid for grid in grids if grid.center_face == anchor]
    if direct_grids:
        return max(direct_grids, key=_grid_quality_score)
    if anchor == "U":
        whiteish_grids = [grid for grid in grids if _center_sample_is_whiteish(grid)]
        if whiteish_grids:
            return max(whiteish_grids, key=_grid_quality_score)
    if anchor == "D":
        yellowish_grids = [grid for grid in grids if _center_sample_is_yellowish(grid)]
        if yellowish_grids:
            return max(yellowish_grids, key=_grid_quality_score)
    return None


def _grid_quality_score(grid: FaceGrid) -> float:
    return grid.matched_count * 18.0 - grid.fit_error * 1.2 - min(65.0, _grid_shape_spread(grid)) * 1.2 - _grid_sample_penalty(grid)


def _center_sample_is_whiteish(grid: FaceGrid) -> bool:
    _, saturation, value = rgb_to_hsv(grid.center_sticker.rgb)
    return value >= 0.62 and saturation <= 0.34


def _center_sample_is_yellowish(grid: FaceGrid) -> bool:
    hue, saturation, value = rgb_to_hsv(grid.center_sticker.rgb)
    return 0.10 <= hue <= 0.22 and saturation >= 0.22 and value >= 0.48


def _assignment_summary(analysis: ImageAnalysis, anchor: str) -> Dict[str, Dict[str, object]]:
    assignments = _assigned_grid_by_face(analysis, anchor)
    return {
        face: {
            "gridId": grid.id,
            "rawCenterFace": grid.center_face,
            "rawCenterColor": grid.center_sticker.match.color,
            "assumed": face == anchor and grid.center_face != anchor,
        }
        for face, grid in assignments.items()
    }


def _merge_faces(a: Dict[str, List[List[Any]]], b: Dict[str, List[List[Any]]]) -> Optional[Dict[str, List[List[Any]]]]:
    merged: Dict[str, Any] = {"_score": float(a.get("_score", 0.0)) + float(b.get("_score", 0.0))}
    merged["_score_a"] = a.get("_score")
    merged["_score_b"] = b.get("_score")
    merged["_selection_score_a"] = a.get("_selection_score")
    merged["_selection_score_b"] = b.get("_selection_score")
    merged["_orientation_score_a"] = a.get("_orientation_score")
    merged["_orientation_score_b"] = b.get("_orientation_score")
    merged["_orientation_rank_a"] = a.get("_orientation_rank")
    merged["_orientation_rank_b"] = b.get("_orientation_rank")
    merged["_side_pair_a"] = a.get("_side_pair")
    merged["_side_pair_b"] = b.get("_side_pair")
    merged["_ordered_side_pair_a"] = a.get("_ordered_side_pair")
    merged["_ordered_side_pair_b"] = b.get("_ordered_side_pair")
    for face, matrix in a.items():
        if face not in FACE_ORDER:
            continue
        merged[face] = _copy_matrix(matrix)
    for face, matrix in b.items():
        if face not in FACE_ORDER:
            continue
        if face in merged and merged[face] != matrix:
            return None
        merged[face] = _copy_matrix(matrix)
    required = set(FACE_ORDER)
    if not required.issubset(merged):
        return None
    return merged


def _face_signature(faces: Dict[str, List[List[Any]]]) -> Tuple[Tuple[str, str], ...]:
    signature = []
    for face in FACE_ORDER:
        matrix = faces.get(face)
        if not matrix:
            continue
        values = []
        for row in matrix:
            for facelet in row:
                values.append(facelet if isinstance(facelet, str) else getattr(facelet, "id", id(facelet)))
        signature.append((face, ",".join(str(value) for value in values)))
    return tuple(signature)


def _state_variants_from_faces(faces: Dict[str, List[List[Any]]]) -> List[str]:
    facelets = []
    for face in FACE_ORDER:
        matrix = faces.get(face)
        if not matrix:
            return []
        facelets.extend(matrix[r][c] for r in range(3) for c in range(3))
    return _balanced_state_variants(facelets)


def _legal_repaired_state_from_faces(faces: Dict[str, List[List[Any]]]) -> Optional[Tuple[str, float, int]]:
    facelets = _facelets_from_faces(faces)
    if facelets is None:
        return None

    corner_options = [_corner_piece_options([facelets[index] for index in indices]) for indices in CORNER_FACELETS]
    edge_options = [_edge_piece_options([facelets[index] for index in indices]) for indices in EDGE_FACELETS]
    if any(not options for options in corner_options) or any(not options for options in edge_options):
        return None

    corners = _top_piece_solutions(corner_options, piece_count=8, orientation_mod=3)
    edges = _top_piece_solutions(edge_options, piece_count=12, orientation_mod=2)
    if not corners or not edges:
        return None

    best: Optional[Tuple[str, float, int]] = None
    for corner_cost, corner_changes, corner_parity, corner_solution in corners:
        for edge_cost, edge_changes, edge_parity, edge_solution in edges:
            if corner_parity != edge_parity:
                continue
            changes = corner_changes + edge_changes
            if changes > MAX_LEGAL_REPAIR_CHANGES:
                continue
            cost = corner_cost + edge_cost
            if cost > MAX_LEGAL_REPAIR_COST:
                continue
            state = _state_from_piece_solution(corner_solution, edge_solution)
            if validate_state(state).valid and (best is None or (changes, cost) < (best[2], best[1])):
                best = (state, cost, changes)
    return best


def _facelets_from_faces(faces: Dict[str, List[List[Any]]]) -> Optional[List[Any]]:
    facelets = []
    for face in FACE_ORDER:
        matrix = faces.get(face)
        if not matrix:
            return None
        facelets.extend(matrix[r][c] for r in range(3) for c in range(3))
    return facelets


def _corner_piece_options(facelets: Sequence[Any]) -> List[PieceOption]:
    options: Dict[Tuple[int, int, Tuple[str, ...]], PieceOption] = {}
    for color_options in product(*(_facelet_repair_options(facelet) for facelet in facelets)):
        colors = tuple(color for color, _, _ in color_options)
        assignment = _corner_assignment(colors)
        if assignment is None:
            continue
        cubie, orientation = assignment
        cost = sum(item[1] for item in color_options)
        changes = sum(item[2] for item in color_options)
        key = (cubie, orientation, colors)
        current = options.get(key)
        option = PieceOption(cubie, orientation, colors, cost, changes)
        if current is None or (changes, cost) < (current.changes, current.cost):
            options[key] = option
    return sorted(options.values(), key=lambda option: (option.changes, option.cost))[:MAX_LEGAL_REPAIR_PIECE_OPTIONS]


def _edge_piece_options(facelets: Sequence[Any]) -> List[PieceOption]:
    options: Dict[Tuple[int, int, Tuple[str, ...]], PieceOption] = {}
    for color_options in product(*(_facelet_repair_options(facelet) for facelet in facelets)):
        colors = tuple(color for color, _, _ in color_options)
        assignment = _edge_assignment(colors)
        if assignment is None:
            continue
        cubie, orientation = assignment
        cost = sum(item[1] for item in color_options)
        changes = sum(item[2] for item in color_options)
        key = (cubie, orientation, colors)
        current = options.get(key)
        option = PieceOption(cubie, orientation, colors, cost, changes)
        if current is None or (changes, cost) < (current.changes, current.cost):
            options[key] = option
    return sorted(options.values(), key=lambda option: (option.changes, option.cost))[:MAX_LEGAL_REPAIR_PIECE_OPTIONS]


def _facelet_repair_options(facelet: Any) -> List[Tuple[str, float, int]]:
    if isinstance(facelet, str):
        return [(facelet, 0.0, 0)]
    match = getattr(facelet, "match", None)
    if match is None:
        face = getattr(facelet, "face", None)
        return [(face, 0.0, 0)] if face in FACE_ORDER else [("U", 0.0, 1)]

    baseline = match.alternatives[0][1] if match.alternatives else match.distance
    options: List[Tuple[str, float, int]] = []
    seen = set()
    suspect = _suspect_grid_sample_score(facelet)
    flexible_repair = _broad_repair_alternatives_allowed(facelet, match, suspect)
    grid_context_flexible = _grid_context_flexible_sample(facelet)
    option_limit = len(FACE_ORDER) if flexible_repair else MAX_LEGAL_REPAIR_OPTIONS_PER_FACELET
    for rank, (color, distance) in enumerate(match.alternatives):
        face = COLOR_TO_FACE.get(color)
        if face is None or face in seen:
            continue
        if rank > 0 and not _facelet_alternative_allowed(facelet, match, color, distance, rank):
            continue
        if grid_context_flexible:
            color_cost = (
                min(GRID_CONTEXT_REPAIR_DISTANCE_CAP, max(0.0, distance - baseline) / GRID_CONTEXT_REPAIR_DISTANCE_SCALE)
                + rank * GRID_CONTEXT_REPAIR_RANK_COST
            )
        elif suspect >= SUSPECT_GRID_SAMPLE_THRESHOLD:
            color_cost = max(0.0, distance - baseline) / 12.0 + rank * 0.24 + min(2.5, suspect * 0.15)
        else:
            color_cost = max(0.0, distance - baseline) / 7.5 + rank * 0.55
        changed = 0 if face == match.face and rank == 0 else 1
        options.append((face, color_cost, changed))
        seen.add(face)
        if len(options) >= option_limit:
            break
    if not options:
        options.append((match.face, 0.0, 0))
    if getattr(facelet, "source", "") == "grid_sample":
        _append_grid_sample_repair_fallbacks(options, seen)
    return options


def _corner_assignment(colors: Tuple[str, ...]) -> Optional[Tuple[int, int]]:
    if sum(1 for color in colors if color in {"U", "D"}) != 1:
        return None
    orientation = next(idx for idx, color in enumerate(colors) if color in {"U", "D"})
    color1 = colors[(orientation + 1) % 3]
    color2 = colors[(orientation + 2) % 3]
    cubie = next((idx for idx, proto in enumerate(CORNER_COLORS) if proto[1] == color1 and proto[2] == color2), None)
    if cubie is None:
        return None
    return cubie, orientation % 3


def _edge_assignment(colors: Tuple[str, ...]) -> Optional[Tuple[int, int]]:
    for cubie, proto in enumerate(EDGE_COLORS):
        if colors == proto:
            return cubie, 0
        if colors == (proto[1], proto[0]):
            return cubie, 1
    return None


def _top_piece_solutions(
    options_by_position: Sequence[Sequence[PieceOption]],
    piece_count: int,
    orientation_mod: int,
) -> List[Tuple[float, int, int, Tuple[PieceOption, ...]]]:
    dp: Dict[Tuple[int, int, int], List[Tuple[float, int, Tuple[PieceOption, ...]]]] = {(0, 0, 0): [(0.0, 0, tuple())]}
    for options in options_by_position:
        next_dp: Dict[Tuple[int, int, int], List[Tuple[float, int, Tuple[PieceOption, ...]]]] = {}
        for (mask, orientation_sum, parity), solutions in dp.items():
            for cost, changes, selected in solutions:
                for option in options:
                    bit = 1 << option.cubie
                    if mask & bit:
                        continue
                    next_changes = changes + option.changes
                    if next_changes > MAX_LEGAL_REPAIR_CHANGES:
                        continue
                    previous_greater = (mask >> (option.cubie + 1)).bit_count()
                    next_key = (
                        mask | bit,
                        (orientation_sum + option.orientation) % orientation_mod,
                        parity ^ (previous_greater % 2),
                    )
                    bucket = next_dp.setdefault(next_key, [])
                    bucket.append((cost + option.cost, next_changes, selected + (option,)))
        dp = {
            key: sorted(bucket, key=lambda item: (item[1], item[0]))[:MAX_LEGAL_REPAIR_SOLUTIONS_PER_KEY]
            for key, bucket in next_dp.items()
        }

    full_mask = (1 << piece_count) - 1
    solutions: List[Tuple[float, int, int, Tuple[PieceOption, ...]]] = []
    for parity in (0, 1):
        for cost, changes, selected in dp.get((full_mask, 0, parity), []):
            solutions.append((cost, changes, parity, selected))
    solutions.sort(key=lambda item: (item[1], item[0]))
    return solutions[:MAX_LEGAL_REPAIR_SOLUTIONS]


def _state_from_piece_solution(corners: Sequence[PieceOption], edges: Sequence[PieceOption]) -> str:
    state: List[Optional[str]] = [None] * 54
    for face, index in CENTER_INDICES.items():
        state[index] = face
    for option, indices in zip(corners, CORNER_FACELETS):
        for index, color in zip(indices, option.colors):
            state[index] = color
    for option, indices in zip(edges, EDGE_FACELETS):
        for index, color in zip(indices, option.colors):
            state[index] = color
    return "".join(color or "U" for color in state)


def _balanced_state_variants(facelets: Sequence[Any]) -> List[str]:
    options = [_facelet_options(facelet) for facelet in facelets]
    current = [choices[0][0] for choices in options]
    counts = Counter(current)
    current_state = "".join(current)
    if all(counts[face] == 9 for face in FACE_ORDER):
        return [current_state]

    surplus = {face: counts[face] - 9 for face in FACE_ORDER if counts[face] > 9}
    deficits = {face: 9 - counts[face] for face in FACE_ORDER if counts[face] < 9}
    if sum(deficits.values()) > MAX_COLOR_REPAIR_CHANGES:
        return [current_state]

    moves = []
    for index, choices in enumerate(options):
        from_face = current[index]
        if surplus.get(from_face, 0) <= 0:
            continue
        for alt_rank, (to_face, cost) in enumerate(choices[1:], start=1):
            if to_face in deficits:
                moves.append((cost + alt_rank * 0.05, index, from_face, to_face))
    moves.sort(key=lambda item: item[0])
    moves = moves[:90]

    variants: List[Tuple[float, str]] = []

    def backtrack(state: List[str], remaining_surplus: Dict[str, int], remaining_deficits: Dict[str, int], used: set[int], cost: float) -> None:
        if len(variants) >= MAX_COLOR_REPAIR_VARIANTS:
            return
        if all(value == 0 for value in remaining_deficits.values()):
            variants.append((cost, "".join(state)))
            return
        target = max(remaining_deficits, key=lambda face: remaining_deficits[face])
        if remaining_deficits[target] <= 0:
            variants.append((cost, "".join(state)))
            return
        for move_cost, index, from_face, to_face in moves:
            if to_face != target or index in used:
                continue
            if remaining_surplus.get(from_face, 0) <= 0 or remaining_deficits.get(to_face, 0) <= 0:
                continue
            next_surplus = dict(remaining_surplus)
            next_deficits = dict(remaining_deficits)
            next_state = list(state)
            next_state[index] = to_face
            next_surplus[from_face] -= 1
            next_deficits[to_face] -= 1
            used.add(index)
            backtrack(next_state, next_surplus, next_deficits, used, cost + move_cost)
            used.remove(index)

    backtrack(current, surplus, deficits, set(), 0.0)
    if not variants:
        return [current_state]
    variants.sort(key=lambda item: item[0])
    unique = []
    seen = set()
    for _, state in variants:
        if state not in seen:
            unique.append(state)
            seen.add(state)
    return unique[:MAX_COLOR_REPAIR_VARIANTS]


def _facelet_options(facelet: Any) -> List[Tuple[str, float]]:
    if isinstance(facelet, str):
        return [(facelet, 0.0)]
    match = getattr(facelet, "match", None)
    if match is None:
        face = getattr(facelet, "face", None)
        return [(face, 0.0)] if face in FACE_ORDER else [("U", 0.0)]

    options: List[Tuple[str, float]] = []
    seen = set()
    baseline = match.alternatives[0][1] if match.alternatives else match.distance
    suspect = _suspect_grid_sample_score(facelet)
    grid_context_flexible = _grid_context_flexible_sample(facelet)
    for rank, (color, distance) in enumerate(match.alternatives):
        face = COLOR_TO_FACE.get(color)
        if face is None or face in seen:
            continue
        if rank > 0 and not _facelet_alternative_allowed(facelet, match, color, distance, rank):
            continue
        if grid_context_flexible:
            cost = (
                min(GRID_CONTEXT_REPAIR_DISTANCE_CAP, max(0.0, distance - baseline) / GRID_CONTEXT_REPAIR_DISTANCE_SCALE)
                + rank * GRID_CONTEXT_REPAIR_RANK_COST
            )
        elif suspect >= SUSPECT_GRID_SAMPLE_THRESHOLD:
            cost = max(0.0, distance - baseline) * 0.42 + rank * 0.16 + min(4.0, suspect * 0.22)
        else:
            cost = max(0.0, distance - baseline) + rank * 0.35
        options.append((face, cost))
        seen.add(face)
    if not options:
        options.append((match.face, 0.0))
    if getattr(facelet, "source", "") == "grid_sample":
        _append_grid_sample_rebalance_fallbacks(options, seen)
    return options


def _append_grid_sample_repair_fallbacks(options: List[Tuple[str, float, int]], seen: set[str]) -> None:
    for rank, face in enumerate(FACE_ORDER):
        if face in seen:
            continue
        options.append((face, GRID_SAMPLE_REPAIR_FALLBACK_COST + rank * 0.35, 1))
        seen.add(face)


def _append_grid_sample_rebalance_fallbacks(options: List[Tuple[str, float]], seen: set[str]) -> None:
    for rank, face in enumerate(FACE_ORDER):
        if face in seen:
            continue
        options.append((face, GRID_SAMPLE_REBALANCE_FALLBACK_COST + rank * 1.75))
        seen.add(face)


def _facelet_alternative_allowed(facelet: Any, match: Any, color: str, distance: float, rank: int) -> bool:
    suspect = _suspect_grid_sample_score(facelet)
    if _broad_repair_alternatives_allowed(facelet, match, suspect):
        baseline = match.alternatives[0][1] if match.alternatives else match.distance
        allowed_delta = MAX_SUSPECT_SAMPLE_ALTERNATIVE_DELTA
        if _low_confidence_component_sample(facelet, match):
            allowed_delta = max(allowed_delta, MAX_LOW_CONFIDENCE_COMPONENT_REPAIR_DELTA)
        if _grid_context_flexible_sample(facelet):
            allowed_delta = max(allowed_delta, MAX_GRID_CONTEXT_COMPONENT_REPAIR_DELTA)
        return rank <= 5 and distance - baseline <= allowed_delta
    return _repair_alternative_allowed(match, color, distance, rank)


def _broad_repair_alternatives_allowed(facelet: Any, match: Any, suspect: Optional[float] = None) -> bool:
    if getattr(facelet, "source", "") == "grid_sample":
        return True
    if suspect is None:
        suspect = _suspect_grid_sample_score(facelet)
    if suspect >= SUSPECT_GRID_SAMPLE_THRESHOLD:
        return True
    return _low_confidence_component_sample(facelet, match) or _grid_context_flexible_sample(facelet)


def _low_confidence_component_sample(facelet: Any, match: Any) -> bool:
    return (
        getattr(facelet, "source", "component") == "component"
        and getattr(match, "confidence", 1.0) < LOW_CONFIDENCE_COMPONENT_REPAIR_THRESHOLD
    )


def _grid_context_flexible_sample(facelet: Any) -> bool:
    return getattr(facelet, "grid_repair_flex", 0.0) >= GRID_CONTEXT_REPAIR_THRESHOLD


def _repair_alternative_allowed(match: Any, color: str, distance: float, rank: int) -> bool:
    baseline = match.alternatives[0][1] if match.alternatives else match.distance
    delta = distance - baseline
    if delta < 0:
        return True
    pair = frozenset((match.color, color))
    if pair in REPAIR_ADJACENT_COLOR_PAIRS and delta <= MAX_LOW_CONFIDENCE_REPAIR_DELTA:
        return True
    if rank <= 2 and delta <= MAX_REPAIR_ALTERNATIVE_DELTA:
        return True
    if match.confidence < 0.26 and rank <= 2 and delta <= MAX_LOW_CONFIDENCE_REPAIR_DELTA:
        return True
    return False


def _state_confidence(faces: Dict[str, List[List[Any]]]) -> float:
    score = float(faces.get("_score", 0.0)) if isinstance(faces, dict) else 0.0
    return max(0.5, min(0.99, 0.55 + score / 5000.0))


def _copy_matrix(matrix: Sequence[Sequence[Any]]) -> List[List[Any]]:
    return [[matrix[r][c] for c in range(3)] for r in range(3)]
