from __future__ import annotations

import copy
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .colors import COLOR_TO_FACE, build_adaptive_palette, classify_rgb, rgb_to_hsv
from .geometry import TRANSFORMS, closest_edge, possible_transforms
from .image_pipeline import FaceGrid, ImageAnalysis, Sticker, analyze_image, cube_hull_grid_penalty
from .validation import (
    CENTER_INDICES,
    CORNER_COLORS,
    CORNER_FACELETS,
    EDGE_COLORS,
    EDGE_FACELETS,
    FACE_ORDER,
    is_valid_state,
    validate_state,
)


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
MIN_IMAGE_B_D_ANCHOR_MATCHED_COUNT = 6
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
MAX_DIRECT_LEGAL_CANDIDATES_SIGNAL = 8
DIRECT_LEGAL_CONFIDENCE_TIE_EPSILON = 0.00005
# Repair-only escape hatch for Issue #31: look deeper on image A only after
# standard repair fails and the pair shows opposing red/orange skew evidence.
MAX_REPAIR_BACKFILL_OPTIONS_A = 2600
MAX_REPAIR_BACKFILL_OPTIONS_B_PER_PAIR = 80
MAX_REPAIR_BACKFILL_MERGES = 40
MAX_REPAIR_BACKFILL_CONFLICTS = 6
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
MAX_LOW_MATCH_GRID_SUSPECT_SAMPLE_SCORE = 3.0
MAX_LOW_MATCH_GRID_BAD_SAMPLES = 2
MIN_GRID_EXTRAPOLATION_GRID_SAMPLES = 3
MAX_GRID_EXTRAPOLATION_MATCHED_COUNT = 6
MIN_GRID_EXTRAPOLATION_UNSUPPORTED_SAMPLES = 3
MIN_GRID_EXTRAPOLATION_UNSUPPORTED_SCORE = 3.0
GRID_EXTRAPOLATION_PENALTY_SCALE = 7.5
MAX_GRID_EXTRAPOLATION_PENALTY = 26.0
MAX_COLLAPSED_ANCHOR_GRID_SELF_FACE_CELLS = 2
MAX_COLLAPSED_ANCHOR_SIDE_GRID_SUSPECT_SCORE = 3.0
MAX_SUSPECT_SAMPLE_ALTERNATIVE_DELTA = 58.0
MAX_TRIPLE_COMPONENT_OVERLAP = 3
# Used only when strict triple selection finds no candidates. This lets
# high-quality cluttered photos recover without making overlap the normal path.
MAX_RESCUE_TRIPLE_COMPONENT_OVERLAP = 6
MAX_RESCUE_VISIBLE_FACE_TRIPLES = 3
MIN_RESCUE_VISIBLE_FACE_TRIPLE_SCORE = 80.0
RED_ORANGE_PAIR_CALIBRATION_SUSPECTED_CHECK = "red_orange_pair_calibration_suspected"
IMAGE_B_VISIBLE_FACE_EVIDENCE_WEAK_CHECK = "image_b_visible_face_evidence_weak"
BACKGROUND_STICKER_NOISE_CHECK = "background_sticker_noise_suspected"
FACE_TRIPLE_OVERLAP_LOW_QUALITY_CHECK = "face_triple_overlap_low_quality"
VISIBLE_FACE_COLOR_COUNT_IMBALANCE_CHECK = "visible_face_color_count_imbalance"
BALANCED_COLOR_SCORING_ENV = "CUBE_RECOGNIZER_BALANCED_COLOR_SCORING"
PAIR_COLOR_EVIDENCE_COLORS = ("white", "red", "orange")
PAIR_COLOR_EVIDENCE_FACES = tuple(COLOR_TO_FACE[color] for color in PAIR_COLOR_EVIDENCE_COLORS)
# Image A may have a non-white logo on the white center. Admit that as a U
# anchor only when the whole grid is strong and the center's color is ambiguous.
MIN_U_LOGO_ANCHOR_MATCHED_COUNT = 8
MAX_U_LOGO_ANCHOR_FIT_ERROR = 3.0
MIN_U_LOGO_ANCHOR_QUALITY = 120.0
MAX_U_LOGO_ANCHOR_GRID_SAMPLES = 1
MAX_U_LOGO_ANCHOR_CENTER_CONFIDENCE = 0.42
MAX_U_LOGO_ANCHOR_WHITE_DISTANCE_DELTA = 24.0
# Issue #85 diagnostic only: tuned to tag Sets 46-49 background-driven
# sticker evidence failures without firing on Set 17's separate image-B weakness.
MAX_BACKGROUND_STICKER_NOISE_ANCHOR_SELF_FACE_CELLS = 1
MIN_BACKGROUND_STICKER_NOISE_DOMINANT_GRID_CENTER_COUNT = 14
MIN_BACKGROUND_STICKER_NOISE_DOMINANT_GRID_CENTER_SHARE = 0.60
MAX_VISIBLE_FACE_PAIR_COLOR_COUNT_IMBALANCE = 13
MAX_BALANCED_COLOR_ASSIGNMENT_EXACT_CHANGES = 7
MAX_BALANCED_COLOR_ASSIGNMENT_MOVES_PER_TARGET = 48
BALANCED_COLOR_ASSIGNMENT_HIGH_COST = 18.0
MAX_BALANCED_COLOR_ASSIGNMENT_SCORE_PENALTY = 26.0
BALANCED_COLOR_ASSIGNMENT_COST_SCALE = 0.035
BALANCED_COLOR_ASSIGNMENT_CHANGE_WEIGHT = 1.1
BALANCED_COLOR_ASSIGNMENT_HIGH_COST_WEIGHT = 1.8
BALANCED_COLOR_ASSIGNMENT_DEVIATION_WEIGHT = 1.45
# Tuned to tag Sets 17/21/22 without firing on unrelated hard-case or corpus
# rejects; revisit these gates when new red/orange captures are added.
RED_ORANGE_SKEW_MIN_GAP = 3
RED_ORANGE_SKEW_MIN_DOMINANT = 4
MIN_WEAK_IMAGE_B_SIDE_GRIDS = 2
MIN_WEAK_IMAGE_B_SIDE_GRID_SAMPLES = 3
MAX_WEAK_IMAGE_B_SIDE_GRID_MATCHED_COUNT = 6
MAX_WEAK_IMAGE_B_SIDE_GRID_QUALITY = 70.0
# Tuned on labeled sets 12/14/15/24/26/27/28/29/31: large enough to
# down-rank conflicted repair winners, but capped so repair candidates remain
# comparable instead of being rejected by one binary threshold.
MAX_REPAIR_RANKING_PENALTY = 0.18
DIRECT_CLEAN_CONFIDENCE_THRESHOLD = 0.78
DIRECT_CLEAN_MAX_SELECTED_GRID_FIT_ERROR = 16.0
DIRECT_CLEAN_MIN_SELECTED_GRID_QUALITY = 60.0
GRID_PURITY_GUARD_TOP_COMPONENT_OVERLAP_MIN = 6
GRID_PURITY_GUARD_TOP_LOW_SELF_FACE_CELLS_MAX = 2
GRID_PURITY_GUARD_TOP_LOW_SELF_FACE_GRID_COUNT_MIN = 5
GRID_PURITY_GUARD_TOP_WRONG_DOMINANT_MARGIN_MIN = 3
REPAIRED_HIGH_CONFIDENCE_THRESHOLD = 0.60
REPAIRED_HIGH_MAX_RANKING_PENALTY = 0.16
REPAIRED_HIGH_MAX_PRE_REPAIR_CONFLICTS = 5
REPAIRED_HIGH_MIN_VALID_PRE_REPAIR_CORNERS = 8
REPAIR_PRE_COUNT_SKEW_DELTA = 2
REPAIR_BACKFILL_STANDARD_UNSTABLE_CONFIDENCE_MAX = 0.65
REPAIR_BACKFILL_STANDARD_UNSTABLE_REPAIR_CHANGES_MIN = 8
REPAIR_BACKFILL_STANDARD_UNSTABLE_CONFLICTS_MIN = 8
REPAIR_RETAKE_CONFIDENCE_THRESHOLD = 0.50
REPAIR_RETAKE_MIN_CANDIDATES = 50_000
REPAIR_SKIP_DIRECT_CANDIDATE_THRESHOLD = REPAIR_RETAKE_MIN_CANDIDATES
# Tuned to improve Set 28's low-confidence, max-penalty repair result without
# firing on corpus controls: Sets 12/14/24 stay above the confidence gate, and
# Set 27 misses the orientation-rank gate. The bonus is capped at 0.03, so rank
# sums <=4 saturate instead of letting orientation rank override piece evidence.
REPAIR_ORIENTATION_RERANK_MIN_TOP_RANK_SUM = 10
REPAIR_ORIENTATION_RERANK_RANK_LIMIT = 14
REPAIR_ORIENTATION_RERANK_BONUS_PER_RANK = 0.003
REPAIR_ORIENTATION_RERANK_MAX_BONUS = 0.03
REPAIR_ADJACENT_COLOR_PAIRS = {frozenset(("red", "orange")), frozenset(("green", "blue"))}
VALID_EDGE_COLOR_SETS = {frozenset(colors) for colors in EDGE_COLORS}
VALID_CORNER_COLOR_SETS = {frozenset(colors) for colors in CORNER_COLORS}
# Corner cubie identity follows the legacy assignment rule: side-color order
# selects the cubie, while either U/D in the twist slot is allowed.
CORNER_ASSIGNMENTS = {
    colors: (cubie, orientation)
    for cubie, (_, first_side, second_side) in enumerate(CORNER_COLORS)
    for ud_color in ("U", "D")
    for colors, orientation in (
        ((ud_color, first_side, second_side), 0),
        ((second_side, ud_color, first_side), 1),
        ((first_side, second_side, ud_color), 2),
    )
}
EDGE_ASSIGNMENTS = {
    colors: (cubie, orientation)
    for cubie, (first, second) in enumerate(EDGE_COLORS)
    for colors, orientation in (((first, second), 0), ((second, first), 1))
}
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
EDGE_FACELET_COORDS = tuple(
    tuple((FACE_ORDER[index // 9], (index % 9) // 3, (index % 9) % 3) for index in indices)
    for indices in EDGE_FACELETS
)
CORNER_FACELET_COORDS = tuple(
    tuple((FACE_ORDER[index // 9], (index % 9) // 3, (index % 9) % 3) for index in indices)
    for indices in CORNER_FACELETS
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


FaceSignature = Tuple[Tuple[str, str], ...]
FaceletOptionsKey = Tuple[str, object]


@dataclass
class RecognitionWorkset:
    options_a: List[Dict[str, List[List[Any]]]]
    options_b: List[Dict[str, List[List[Any]]]]
    merged_candidates: List[Tuple[float, Dict[str, List[List[Any]]]]]
    facelet_options_by_key: Dict[FaceletOptionsKey, List[Tuple[str, float]]] = field(default_factory=dict)
    repaired_state_by_signature: Dict[FaceSignature, Optional[Tuple[str, float, int]]] = field(default_factory=dict)
    conflicts_by_signature: Dict[FaceSignature, Dict[str, int]] = field(default_factory=dict)
    face_counts_by_signature: Dict[FaceSignature, Dict[str, int]] = field(default_factory=dict)


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

        _attach_failed_pair_color_calibration_signal(
            result,
            calibrated_result,
            analysis_a,
            analysis_b,
            calibrated_a,
            calibrated_b,
        )
        result.image_a = analysis_a
        result.image_b = analysis_b
        return result

    def _recognize_from_analyses(self, analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> RecognitionResult:
        recognition_signals = _base_recognition_signals(analysis_a, analysis_b)
        checks = _failed_checks_with_context(_white_up_checks(analysis_a, analysis_b), analysis_a, analysis_b)
        if checks:
            return RecognitionResult(
                status="rejected",
                reason=_reason_for_checks(checks),
                failed_checks=checks,
                recognition_signals=_recognition_signals_with_failed_checks(
                    recognition_signals,
                    checks,
                    analysis_a,
                    analysis_b,
                ),
            )

        workset = _recognition_workset(analysis_a, analysis_b)
        candidates = self._state_candidates_from_workset(workset)
        legal = []
        invalid_reasons: List[str] = []
        collect_invalid_reasons = len(candidates) < REPAIR_SKIP_DIRECT_CANDIDATE_THRESHOLD
        for state, confidence, details in candidates:
            if collect_invalid_reasons:
                validation = validate_state(state)
                if validation.valid:
                    legal.append((state, confidence, details))
                else:
                    invalid_reasons.extend(validation.errors)
            elif is_valid_state(state):
                legal.append((state, confidence, details))

        unique = {}
        unique_details = {}
        for state, confidence, details in legal:
            if confidence > unique.get(state, 0.0):
                unique[state] = confidence
                unique_details[state] = details

        recognition_signals = {
            **recognition_signals,
            "directLegalCandidates": _direct_legal_candidate_summary(unique, unique_details),
        }

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

        # Repair recoveries below the retake candidate-count floor are still
        # categorized as retakes, so avoid the expensive cubie repair search.
        if len(candidates) < REPAIR_SKIP_DIRECT_CANDIDATE_THRESHOLD:
            recognition_signals.update(_repair_signal_summary([]))
            failed_checks = _validation_failed_checks(invalid_reasons, analysis_a, analysis_b)
            return RecognitionResult(
                status="rejected",
                reason="No legal cube state matched the detected stickers.",
                failed_checks=failed_checks,
                candidates=len(candidates),
                recognition_signals=_recognition_signals_with_failed_checks(
                    recognition_signals,
                    failed_checks,
                    analysis_a,
                    analysis_b,
                ),
            )

        repair_details = self._legal_repair_candidate_details_from_workset(workset, release_merged_candidates=True)
        repair_backfill_attempted = False
        repair_backfill_evaluated = 0
        repair_backfill_probe_reason = None
        if _repair_backfill_applies(analysis_a, analysis_b):
            if not repair_details:
                repair_backfill_probe_reason = "no_standard_repair"
            elif _repair_backfill_should_probe_standard_result(repair_details):
                repair_backfill_probe_reason = "unstable_standard_repair"
        if repair_backfill_probe_reason:
            repair_backfill_attempted = True
            repair_backfill_merges = _repair_backfill_merged_face_candidates(analysis_a, workset)
            repair_backfill_evaluated = len(repair_backfill_merges)
            repair_backfill_details = self._legal_repair_candidate_details_from_merges(
                workset,
                repair_backfill_merges,
                repair_source="conflict_backfill",
            )
            repair_details = _merge_repair_detail_sources(repair_details, repair_backfill_details)
        repair_details = _repair_details_with_orientation_selection_scores(repair_details)
        repair_candidates = [
            (
                item["state"],
                item["confidence"],
                _float_signal(item.get("repairSelectionScore"), default=_float_signal(item.get("confidence"))),
            )
            for item in repair_details
        ]
        repair_summary = _repair_signal_summary(repair_details)
        if repair_backfill_attempted:
            repair_summary.update(
                _repair_backfill_signal_fields(
                    repair_backfill_evaluated,
                    repair_details,
                    probe_reason=repair_backfill_probe_reason,
                )
            )
        recognition_signals.update(repair_summary)
        if repair_candidates:
            candidates.extend(repair_candidates)
            repaired_unique = {}
            for state, confidence, selection_score in repair_candidates:
                if is_valid_state(state):
                    current = repaired_unique.get(state)
                    if current is None or selection_score > current[1]:
                        repaired_unique[state] = (confidence, selection_score)
            if len(repaired_unique) == 1:
                state, (confidence, _) = next(iter(repaired_unique.items()))
                selected_repair_summary = _repair_signal_summary(repair_details, selected_state=state)
                if repair_backfill_attempted:
                    selected_repair_summary.update(
                        _repair_backfill_signal_fields(
                            repair_backfill_evaluated,
                            repair_details,
                            selected_state=state,
                            probe_reason=repair_backfill_probe_reason,
                        )
                    )
                return RecognitionResult(
                    status="success",
                    state=state,
                    confidence=confidence,
                    reason="Recognized a legal white-up cube state after cubie-level color repair.",
                    candidates=len(candidates),
                    recognition_signals={**recognition_signals, **selected_repair_summary},
                )
            if len(repaired_unique) > 1:
                ranked_repaired = sorted(repaired_unique.items(), key=lambda item: item[1][1], reverse=True)
                if ranked_repaired[0][1][1] > ranked_repaired[1][1][1]:
                    state = ranked_repaired[0][0]
                    confidence = ranked_repaired[0][1][0]
                    selected_repair_summary = _repair_signal_summary(repair_details, selected_state=state)
                    if repair_backfill_attempted:
                        selected_repair_summary.update(
                            _repair_backfill_signal_fields(
                                repair_backfill_evaluated,
                                repair_details,
                                selected_state=state,
                                probe_reason=repair_backfill_probe_reason,
                            )
                        )
                    return RecognitionResult(
                        status="success",
                        state=state,
                        confidence=confidence,
                        reason="Recognized the highest-scoring legal cube state after cubie-level color repair.",
                        candidates=len(candidates),
                        recognition_signals={**recognition_signals, **selected_repair_summary},
                    )

        failed_checks = _validation_failed_checks(
            invalid_reasons or _candidate_validation_errors(candidates),
            analysis_a,
            analysis_b,
        )
        return RecognitionResult(
            status="rejected",
            reason="No legal cube state matched the detected stickers.",
            failed_checks=failed_checks,
            candidates=len(candidates),
            recognition_signals=_recognition_signals_with_failed_checks(
                recognition_signals,
                failed_checks,
                analysis_a,
                analysis_b,
            ),
        )

    def _state_candidates(self, analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> List[Tuple[str, float, Dict[str, Any]]]:
        return self._state_candidates_from_workset(_recognition_workset(analysis_a, analysis_b))

    def _state_candidates_from_workset(self, workset: RecognitionWorkset) -> List[Tuple[str, float, Dict[str, Any]]]:
        candidates: List[Tuple[str, float, Dict[str, Any]]] = []
        for _, merged in workset.merged_candidates:
            details = _candidate_selection_detail(merged)
            for variant_cost, partial in _state_variants_from_faces_with_costs(
                merged,
                facelet_options_cache=workset.facelet_options_by_key,
            ):
                confidence = _state_confidence(merged)
                candidates.append((partial, confidence, {**details, "variantCost": variant_cost}))
        return candidates

    def _legal_repair_candidates(self, analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> List[Tuple[str, float]]:
        return [(item["state"], item["confidence"]) for item in self._legal_repair_candidate_details(analysis_a, analysis_b)]

    def _legal_repair_candidate_details(self, analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> List[Dict[str, Any]]:
        return self._legal_repair_candidate_details_from_workset(_recognition_workset(analysis_a, analysis_b))

    def _legal_repair_candidate_details_from_workset(
        self,
        workset: RecognitionWorkset,
        *,
        release_merged_candidates: bool = False,
    ) -> List[Dict[str, Any]]:
        repair_merges = _repair_merged_face_candidates(workset.merged_candidates)[:MAX_LEGAL_REPAIR_EVALUATED_MERGES]
        if release_merged_candidates:
            # The repair solver only needs the capped/diverse merge list below.
            # Drop the full Cartesian product before that expensive loop so
            # rejected repair-heavy cases do not carry thousands of matrices.
            workset.merged_candidates = []

        return self._legal_repair_candidate_details_from_merges(workset, repair_merges)

    def _legal_repair_candidate_details_from_merges(
        self,
        workset: RecognitionWorkset,
        repair_merges: Sequence[Tuple[float, Dict[str, List[List[Any]]]]],
        *,
        repair_source: str = "standard",
    ) -> List[Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {}
        for raw_score, merged in repair_merges:
            signature = _cached_face_signature(merged)
            if signature not in workset.repaired_state_by_signature:
                workset.repaired_state_by_signature[signature] = _legal_repaired_state_from_faces(merged)
            state, repair_cost, changes = workset.repaired_state_by_signature[signature] or (None, 0.0, 0)
            if state is None:
                continue
            if signature not in workset.conflicts_by_signature:
                workset.conflicts_by_signature[signature] = _piece_conflict_summary(merged)
            if signature not in workset.face_counts_by_signature:
                workset.face_counts_by_signature[signature] = _primary_face_counts(merged)
            conflicts = workset.conflicts_by_signature[signature]
            face_counts = workset.face_counts_by_signature[signature]
            base_confidence = _state_confidence(merged) - repair_cost / 650.0 - changes * 0.006
            penalty = _repair_ranking_penalty(
                conflicts,
                merged,
                repair_cost=repair_cost,
                repair_changes=changes,
                face_counts=face_counts,
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
                "preRepairFaceCounts": face_counts,
                "repairSource": repair_source,
            }
            balanced_assignment = merged.get("_balanced_color_assignment")
            if isinstance(balanced_assignment, dict):
                detail["balancedColorAssignment"] = balanced_assignment
            balanced_penalty = merged.get("_balanced_color_assignment_score_penalty")
            if balanced_penalty is not None:
                detail["balancedColorAssignmentScorePenalty"] = balanced_penalty
            current = candidates.get(state)
            if current is None or confidence > current["confidence"]:
                candidates[state] = detail
        return sorted(candidates.values(), key=lambda item: item["confidence"], reverse=True)[:MAX_LEGAL_REPAIR_RETURNED]


def _candidate_validation_errors(candidates: Sequence[Tuple[str, float, Dict[str, Any]]]) -> List[str]:
    errors: List[str] = []
    for state, _, _ in candidates:
        validation = validate_state(state)
        if not validation.valid:
            errors.extend(validation.errors)
    return errors


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
        if _grid_purity_guard_would_fire(signals):
            return {
                "category": "needs_manual_review",
                "reason": "grid_purity_guard",
            }
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
    if _repair_pre_count_skew_suspected(selected):
        return {
            "category": "needs_manual_review",
            "reason": "repair_path_pre_repair_color_count_skew",
        }
    if _repair_pre_piece_evidence_unstable(selected):
        return {
            "category": "needs_manual_review",
            "reason": "repair_path_unstable_pre_repair_piece_evidence",
        }
    if signals.get("repairBackfillProbeReason") == "unstable_standard_repair":
        return {
            "category": "needs_manual_review",
            "reason": "repair_backfill_from_unstable_standard_repair",
        }
    if _grid_purity_guard_would_fire(signals):
        return {
            "category": "needs_manual_review",
            "reason": "grid_purity_guard",
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


def _repair_pre_count_skew_suspected(selected: Dict[str, Any]) -> bool:
    counts = selected.get("preRepairFaceCounts")
    if not isinstance(counts, dict):
        return False
    deltas = []
    for face in FACE_ORDER:
        try:
            deltas.append(int(counts.get(face, 0)) - 9)
        except (TypeError, ValueError):
            return False
    return max(deltas) >= REPAIR_PRE_COUNT_SKEW_DELTA and min(deltas) <= -REPAIR_PRE_COUNT_SKEW_DELTA


def _repair_pre_piece_evidence_unstable(selected: Dict[str, Any]) -> bool:
    conflicts = selected.get("preRepairConflicts")
    if not isinstance(conflicts, dict):
        return False
    try:
        total_conflicts = int(conflicts.get("totalConflicts", 0))
        valid_corners = int(conflicts.get("validCorners", 8))
    except (TypeError, ValueError):
        return False
    return (
        total_conflicts > REPAIRED_HIGH_MAX_PRE_REPAIR_CONFLICTS
        or valid_corners < REPAIRED_HIGH_MIN_VALID_PRE_REPAIR_CORNERS
    )


def _grid_purity_guard_would_fire(signals: Dict[str, Any]) -> bool:
    top_visible = signals.get("topVisibleTripleQuality")
    if not isinstance(top_visible, dict):
        return False

    max_overlap = 0
    low_self_count = 0
    max_wrong_margin = 0
    for item in top_visible.values():
        if not isinstance(item, dict):
            continue
        max_overlap = max(max_overlap, _int_signal(item.get("componentOverlap")))
        grids = item.get("grids")
        if not isinstance(grids, dict):
            continue
        for face, grid in grids.items():
            if not isinstance(grid, dict):
                continue
            counts = grid.get("cellFaceCounts")
            if not isinstance(counts, dict):
                continue
            self_cells = _int_signal(counts.get(face))
            dominant_face, dominant_cells = _dominant_face_cell_count(counts)
            if self_cells <= GRID_PURITY_GUARD_TOP_LOW_SELF_FACE_CELLS_MAX:
                low_self_count += 1
            if dominant_face is not None and dominant_face != face:
                max_wrong_margin = max(max_wrong_margin, dominant_cells - self_cells)

    return (
        max_overlap >= GRID_PURITY_GUARD_TOP_COMPONENT_OVERLAP_MIN
        and low_self_count >= GRID_PURITY_GUARD_TOP_LOW_SELF_FACE_GRID_COUNT_MIN
        and max_wrong_margin >= GRID_PURITY_GUARD_TOP_WRONG_DOMINANT_MARGIN_MIN
    )


def _dominant_face_cell_count(face_counts: Dict[str, Any]) -> Tuple[Optional[str], int]:
    if not face_counts:
        return None, 0
    face, count = max(face_counts.items(), key=lambda item: _int_signal(item[1]))
    return str(face), _int_signal(count)


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
            if (
                matched_count < 5
                or fit_error > DIRECT_CLEAN_MAX_SELECTED_GRID_FIT_ERROR
                or quality < DIRECT_CLEAN_MIN_SELECTED_GRID_QUALITY
                or bad_samples > 3
                or suspect_samples > 4.0
            ):
                count += 1
    return count


def _float_signal(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float_signal(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_signal(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def recognition_diagnostics(analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> Dict[str, Any]:
    workset = _recognition_workset(analysis_a, analysis_b)
    candidate_counts = _candidate_face_count_diagnostics(
        workset.merged_candidates,
        facelet_options_cache=workset.facelet_options_by_key,
    )
    return {
        "imageA": _orientation_diagnostics(analysis_a, "U", workset.options_a),
        "imageB": _orientation_diagnostics(analysis_b, "D", workset.options_b),
        "mergedCandidates": {
            "optionsA": len(workset.options_a),
            "optionsB": len(workset.options_b),
            "merged": len(workset.merged_candidates),
            "sidePairCombos": _counter_items(
                Counter(
                    (_side_pair_key(item[1].get("_side_pair_a")), _side_pair_key(item[1].get("_side_pair_b")))
                    for item in workset.merged_candidates
                )
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
        "topVisibleTripleQuality": {
            "imageA": _top_visible_triple_quality(analysis_a, "U"),
            "imageB": _top_visible_triple_quality(analysis_b, "D"),
        },
        "topVisibleBalancedColorAssignment": _top_visible_face_pair_balanced_assignment_signal(analysis_a, analysis_b),
        "twoViewGeometryConsistency": _two_view_geometry_consistency(analysis_a, analysis_b),
        "perStickerConfidence": _per_sticker_confidence_signal(analysis_a, analysis_b),
    }


def _collect_sticker_confidences(analysis: ImageAnalysis) -> List[float]:
    """Flatten per-sticker confidences across all 3 selected face grids in
    one photo. Each grid contributes its 9 stickers' ColorMatch.confidence.

    Returns an empty list if no grids are available; otherwise typically
    27 values (3 faces × 9 stickers).
    """
    out: List[float] = []
    for grid in analysis.grids[:3]:
        for row in grid.stickers:
            for sticker in row:
                conf = float(sticker.match.confidence)
                out.append(conf)
    return out


def _confidence_stats(values: Sequence[float]) -> Dict[str, Any]:
    """Stats over a list of confidences."""
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    vals = sorted(values)
    n = len(vals)
    median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {
        "count": n,
        "min": round(vals[0], 3),
        "median": round(median, 3),
        "max": round(vals[-1], 3),
    }


def _per_sticker_confidence_signal(
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
) -> Dict[str, Any]:
    """Emit per-pair summary stats of the underlying per-sticker
    classification confidences that cv-local already computes.

    Each ColorMatch carries a confidence score in [0.0, 1.0], computed
    as `(second_distance - first_distance) / second_distance` (the
    "gap-normalized" margin between the best and second-best color
    match). Higher = more discriminated; near 0 = the sample was
    almost equidistant between two colors and the classification is
    fragile. The signal exposes:

      perPhoto.{imageA, imageB} stats (count, min, median, max)
      perPair: combined stats across all 54 stickers
      belowThresholdCount: how many stickers had confidence below
        the threshold

    Threshold default is 0.10, calibrated empirically against the
    labeled corpus (2026-05-22): full-match cases have median
    per-sticker min ~0.24; no-recognition cases have median min
    ~0.05. 0.10 sits between them, catching the genuinely-ambiguous
    reads without firing on every photo. Configurable via
    ``RUBIK_PER_STICKER_CONFIDENCE_THRESHOLD`` env var.

    DIAGNOSTIC-ONLY: no category logic depends on this field. Existing
    consumers that use min-confidence-across-pair are unaffected; this
    just exposes the richer distribution so future work (probabilistic
    state output, axis-aware scorer, retake-prompt tuning) has the
    data to use. Validation against the labeled corpus shows minConf
    correlates monotonically with accuracy tier:
        full-match (n=10):     min_conf median 0.24
        partial (n=6):         min_conf median 0.19
        failure (n=4):         min_conf median 0.15
        no-recognition (n=8):  min_conf median 0.05
    """
    threshold_str = os.environ.get("RUBIK_PER_STICKER_CONFIDENCE_THRESHOLD", "0.10")
    try:
        threshold = float(threshold_str)
    except ValueError:
        threshold = 0.10
    confs_a = _collect_sticker_confidences(analysis_a)
    confs_b = _collect_sticker_confidences(analysis_b)
    combined = confs_a + confs_b
    below = sum(1 for c in combined if c < threshold)
    return {
        "perPhoto": {
            "imageA": _confidence_stats(confs_a),
            "imageB": _confidence_stats(confs_b),
        },
        "perPair": _confidence_stats(combined),
        "belowThresholdCount": below,
        "threshold": threshold,
    }


def _grid_median_spacing(grid: FaceGrid) -> Optional[float]:
    """Median pixel spacing between adjacent cells in a 3x3 face grid.

    Returns None if the grid's points don't form a 3x3 array.
    """
    pts = grid.points
    if pts is None or len(pts) != 3 or any(len(row) != 3 for row in pts):
        return None
    distances: List[float] = []
    for r in range(3):
        for c in range(2):
            x0, y0 = pts[r][c]
            x1, y1 = pts[r][c + 1]
            distances.append(math.hypot(x1 - x0, y1 - y0))
    for r in range(2):
        for c in range(3):
            x0, y0 = pts[r][c]
            x1, y1 = pts[r + 1][c]
            distances.append(math.hypot(x1 - x0, y1 - y0))
    if not distances:
        return None
    distances.sort()
    n = len(distances)
    if n % 2 == 1:
        return float(distances[n // 2])
    return float((distances[n // 2 - 1] + distances[n // 2]) / 2)


def _photo_median_spacing(analysis: ImageAnalysis) -> Optional[float]:
    """Median sticker spacing across all selected face grids in a photo.

    Each FaceGrid contributes one spacing measurement. The median of those
    three (or fewer, if not all faces fitted) is the photo's
    characteristic scale. None if no grids are available.
    """
    grid_spacings = [_grid_median_spacing(g) for g in analysis.grids]
    valid = sorted(s for s in grid_spacings if s is not None and s > 0)
    if not valid:
        return None
    n = len(valid)
    if n % 2 == 1:
        return float(valid[n // 2])
    return float((valid[n // 2 - 1] + valid[n // 2]) / 2)


def _two_view_geometry_consistency(
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
) -> Dict[str, Any]:
    """Cross-photo geometric consistency signal: do A and B describe a
    cube at compatible scales?

    The same physical cube photographed back-to-back should produce
    similar pixel sticker spacings in A and B (within ~30%; user might
    shift slightly between captures but not dramatically). Wildly
    mismatched spacings suggest one of the photos has a face-grid fit
    that's actually following background pattern instead of cube
    geometry — the failure mode behind the severe-failure class
    (sets 24, 28, 31 on the labeled corpus: 70%+ sticker mismatches in
    cv-local's output, consistent with one or both photos having
    fundamentally wrong face geometry).

    DIAGNOSTIC-ONLY: this signal is emitted in `recognition_signals`
    but no recognition behavior currently depends on it. Validated
    against the labeled corpus, it can graduate into the
    `needs_manual_review` category trigger.

    Tolerance: configurable via env (`RUBIK_TWO_VIEW_RATIO_TOLERANCE`,
    default 1.4). Default is loose enough to absorb legitimate camera
    distance differences (the user often holds the camera at slightly
    different distances for A and B), tight enough to catch the
    grid-fit failures where one photo's "cube" is actually background.
    """
    spacing_a = _photo_median_spacing(analysis_a)
    spacing_b = _photo_median_spacing(analysis_b)
    if spacing_a is None or spacing_b is None:
        return {
            "spacingPxA": spacing_a,
            "spacingPxB": spacing_b,
            "ratio": None,
            "inconsistent": False,
            "reason": "missing_grids",
        }
    ratio = max(spacing_a, spacing_b) / max(min(spacing_a, spacing_b), 1e-6)
    try:
        tolerance = float(os.environ.get("RUBIK_TWO_VIEW_RATIO_TOLERANCE", "1.4"))
    except ValueError:
        tolerance = 1.4
    inconsistent = ratio > tolerance
    return {
        "spacingPxA": round(spacing_a, 1),
        "spacingPxB": round(spacing_b, 1),
        "ratio": round(ratio, 3),
        "toleranceRatio": round(tolerance, 3),
        "inconsistent": inconsistent,
        "reason": "two_view_inconsistent" if inconsistent else "ok",
    }


def _selected_grid_quality(analysis: ImageAnalysis, anchor: str) -> Dict[str, Dict[str, Any]]:
    return {
        face: _grid_signal_summary(grid)
        for face, grid in _assigned_grid_by_face(analysis, anchor).items()
    }


def _top_visible_triple_quality(analysis: ImageAnalysis, anchor: str) -> Optional[Dict[str, Any]]:
    groups = _candidate_grids_by_face(analysis, anchor)
    if anchor not in groups:
        return None
    triples = _ranked_visible_face_triples(groups, anchor)
    if not triples:
        return None
    score, subset = triples[0]
    return {
        "score": round(float(score), 3),
        "sidePair": _side_pair_key(face for face in subset if face in SIDE_NEIGHBORS),
        "componentOverlap": _triple_overlap_count(tuple(subset.values())),
        "grids": {face: _grid_signal_summary(grid) for face, grid in sorted(subset.items())},
        "gridSpanContamination": _grid_span_contamination_collection_summary(subset.values()),
    }


def _grid_signal_summary(grid: FaceGrid) -> Dict[str, Any]:
    summary = {
        "gridId": grid.id,
        "centerFace": grid.center_face,
        "matchedCount": grid.matched_count,
        "fitError": round(grid.fit_error, 3),
        "quality": round(_grid_quality_score(grid), 3),
        "gridSamples": _grid_sample_count(grid),
        "badSamples": _grid_bad_sample_count(grid),
        "suspectSamples": round(_grid_suspect_sample_score(grid), 3),
        "extrapolatedSamples": _grid_extrapolated_sample_count(grid),
        "extrapolatedSampleScore": round(_grid_extrapolated_sample_score(grid), 3),
        "unsupportedSamples": _grid_unsupported_sample_count(grid),
        "unsupportedSampleScore": round(_grid_unsupported_sample_score(grid), 3),
        "gridExtrapolationPenalty": round(_grid_extrapolation_penalty(grid), 3),
        "cellFaceCounts": _grid_cell_face_counts(grid),
        "cellSourceCounts": _grid_cell_source_counts(grid),
        "componentShapeAngleCount": len(_grid_shape_angles(grid)),
        "componentShapeSpread": round(_grid_shape_spread(grid), 3),
        "gridSpanContamination": _grid_span_contamination_summary(grid),
    }
    if getattr(grid, "cube_hull_inside_count", None) is not None:
        inside_count = getattr(grid, "cube_hull_inside_count", None)
        summary.update(
            {
                "cubeHullInsideCount": inside_count,
                "cubeHullOutsideCount": getattr(grid, "cube_hull_outside_count", None),
                "cubeHullSource": getattr(grid, "cube_hull_source", None),
                "cubeHullPenalty": round(_grid_cube_hull_penalty(grid), 3),
                "rawCubeHullPenalty": round(cube_hull_grid_penalty(inside_count), 3),
            }
        )
    return summary


def _grid_span_contamination_collection_summary(grids: Iterable[FaceGrid]) -> Dict[str, Any]:
    diagnostics = [_grid_span_contamination_summary(grid) for grid in grids]
    if not diagnostics:
        return {
            "maxScore": 0.0,
            "maxComponentShapeSpread": 0.0,
            "totalSampledCells": 0,
            "totalExtrapolatedCells": 0,
            "totalUnsupportedCells": 0,
        }
    return {
        "maxScore": round(max(float(item.get("score") or 0.0) for item in diagnostics), 3),
        "maxComponentShapeSpread": round(max(float(item.get("componentShapeSpread") or 0.0) for item in diagnostics), 3),
        "maxOutsideGridComponentHullRatio": round(
            max(float(item.get("maxOutsideGridComponentHullRatio") or 0.0) for item in diagnostics),
            3,
        ),
        "maxNearestGridComponentRatio": round(
            max(float(item.get("maxNearestGridComponentRatio") or 0.0) for item in diagnostics),
            3,
        ),
        "totalSampledCells": sum(int(item.get("sampledCellCount") or 0) for item in diagnostics),
        "totalExtrapolatedCells": sum(int(item.get("extrapolatedCellCount") or 0) for item in diagnostics),
        "totalUnsupportedCells": sum(int(item.get("unsupportedCellCount") or 0) for item in diagnostics),
        "totalBadSampleCells": sum(int(item.get("badSampleCellCount") or 0) for item in diagnostics),
        "totalCubeHullOutsideCells": sum(int(item.get("cubeHullOutsideCount") or 0) for item in diagnostics),
    }


def _grid_span_contamination_summary(grid: FaceGrid) -> Dict[str, Any]:
    sample_geometry = _grid_sample_geometry_summary(grid)
    angle_count = len(_grid_shape_angles(grid))
    shape_spread = _grid_shape_spread(grid)
    shape_spread_risk = max(0.0, shape_spread - 22.0) / 6.0 if angle_count >= 4 else 0.0
    hull_outside = _int_signal(getattr(grid, "cube_hull_outside_count", 0), default=0)
    hull_outside_risk = max(0.0, float(hull_outside - 1)) * 1.25
    score = (
        shape_spread_risk
        + _grid_extrapolated_sample_score(grid)
        + _grid_unsupported_sample_score(grid) * 0.5
        + _grid_bad_sample_count(grid) * 1.25
        + hull_outside_risk
    )
    return {
        "score": round(score, 3),
        "componentShapeAngleCount": angle_count,
        "componentShapeSpread": round(shape_spread, 3),
        "componentShapeSpreadRisk": round(shape_spread_risk, 3),
        "sampledCellCount": _grid_sample_count(grid),
        "badSampleCellCount": _grid_bad_sample_count(grid),
        "extrapolatedCellCount": _grid_extrapolated_sample_count(grid),
        "extrapolatedSampleScore": round(_grid_extrapolated_sample_score(grid), 3),
        "unsupportedCellCount": _grid_unsupported_sample_count(grid),
        "unsupportedSampleScore": round(_grid_unsupported_sample_score(grid), 3),
        "cubeHullOutsideCount": hull_outside,
        "cubeHullOutsideRisk": round(hull_outside_risk, 3),
        **sample_geometry,
    }


def _grid_sample_geometry_summary(grid: FaceGrid) -> Dict[str, Any]:
    outside_ratios = []
    nearest_ratios = []
    for row in getattr(grid, "stickers", []):
        for sticker in row:
            if getattr(sticker, "source", "") != "grid_sample":
                continue
            spacing = float(getattr(sticker, "grid_spacing", 0.0) or 0.0)
            if spacing <= 1e-6:
                continue
            outside_distance = float(
                getattr(
                    sticker,
                    "outside_grid_component_hull_distance",
                    getattr(sticker, "outside_component_hull_distance", 0.0),
                )
                or 0.0
            )
            nearest_distance = float(
                getattr(
                    sticker,
                    "nearest_grid_component_distance",
                    getattr(sticker, "nearest_component_distance", 0.0),
                )
                or 0.0
            )
            outside_ratios.append(outside_distance / spacing)
            nearest_ratios.append(nearest_distance / spacing)
    return {
        "maxOutsideGridComponentHullRatio": round(max(outside_ratios, default=0.0), 3),
        "maxNearestGridComponentRatio": round(max(nearest_ratios, default=0.0), 3),
        "sampleCellsOutsideGridComponentHull": sum(1 for ratio in outside_ratios if ratio > 0.75),
        "sampleCellsFarFromGridComponents": sum(1 for ratio in nearest_ratios if ratio > 1.75),
    }


def _grid_cell_face_counts(grid: FaceGrid) -> Dict[str, int]:
    counts = Counter()
    for row in getattr(grid, "stickers", []):
        for sticker in row:
            face = _primary_facelet_color(sticker)
            counts[face if isinstance(face, str) and face in FACE_ORDER else "unknown"] += 1
    return {face: counts[face] for face in (*FACE_ORDER, "unknown") if counts[face]}


def _grid_cell_source_counts(grid: FaceGrid) -> Dict[str, int]:
    counts = Counter(
        getattr(sticker, "source", "unknown") or "unknown"
        for row in getattr(grid, "stickers", [])
        for sticker in row
    )
    return {source: counts[source] for source in sorted(counts)}


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


def _repair_backfill_signal_fields(
    repair_backfill_evaluated: int,
    repair_details: Sequence[Dict[str, Any]],
    *,
    selected_state: Optional[str] = None,
    probe_reason: Optional[str] = None,
) -> Dict[str, Any]:
    selected = (
        next((item for item in repair_details if item.get("state") == selected_state), None)
        if selected_state
        else None
    )
    fields = {
        "repairBackfillAttempted": True,
        "repairBackfillEvaluatedMerges": repair_backfill_evaluated,
        "repairBackfillUsed": bool(selected and selected.get("repairSource") == "conflict_backfill"),
    }
    if probe_reason:
        fields["repairBackfillProbeReason"] = probe_reason
    return fields


def _repair_backfill_should_probe_standard_result(repair_details: Sequence[Dict[str, Any]]) -> bool:
    if not repair_details:
        return False
    selected = repair_details[0]
    if selected.get("repairSource") not in (None, "standard"):
        return False
    confidence = _float_signal(selected.get("confidence"))
    if confidence <= REPAIR_RETAKE_CONFIDENCE_THRESHOLD:
        return False
    if confidence > REPAIR_BACKFILL_STANDARD_UNSTABLE_CONFIDENCE_MAX:
        return False
    if _repair_pre_count_skew_suspected(selected):
        return False
    if not _repair_pre_piece_evidence_unstable(selected):
        return False
    conflicts = selected.get("preRepairConflicts") if isinstance(selected.get("preRepairConflicts"), dict) else {}
    repair_changes = _int_signal(selected.get("repairChanges"))
    total_conflicts = _int_signal(conflicts.get("totalConflicts"))
    return (
        repair_changes >= REPAIR_BACKFILL_STANDARD_UNSTABLE_REPAIR_CHANGES_MIN
        or total_conflicts >= REPAIR_BACKFILL_STANDARD_UNSTABLE_CONFLICTS_MIN
    )


def _merge_repair_detail_sources(
    standard_details: Sequence[Dict[str, Any]],
    backfill_details: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_state: Dict[str, Dict[str, Any]] = {}
    for detail in (*standard_details, *backfill_details):
        state = detail.get("state")
        if not isinstance(state, str):
            continue
        current = by_state.get(state)
        if current is None or _float_signal(detail.get("confidence")) > _float_signal(current.get("confidence")):
            by_state[state] = dict(detail)
    ranked = sorted(by_state.values(), key=lambda item: _float_signal(item.get("confidence")), reverse=True)
    return ranked[:MAX_LEGAL_REPAIR_RETURNED]


def _direct_legal_candidate_summary(
    unique: Dict[str, float],
    unique_details: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    ranked = sorted(unique.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return {
            "status": "none",
            "stateCount": 0,
            "topCandidates": [],
        }

    top_confidence = ranked[0][1]
    second_confidence = ranked[1][1] if len(ranked) > 1 else None
    top_tie_count = sum(
        1
        for _, confidence in ranked
        if abs(confidence - top_confidence) <= DIRECT_LEGAL_CONFIDENCE_TIE_EPSILON
    )
    if len(ranked) == 1:
        status = "unique"
    elif top_tie_count > 1:
        status = "tied"
    else:
        status = "separated"
    summary: Dict[str, Any] = {
        "status": status,
        "stateCount": len(ranked),
        "topConfidence": round(float(top_confidence), 4),
        "topTieCount": top_tie_count,
        "topCandidates": [
            _public_direct_legal_candidate_detail(rank, state, confidence, unique_details.get(state) or {})
            for rank, (state, confidence) in enumerate(ranked[:MAX_DIRECT_LEGAL_CANDIDATES_SIGNAL], start=1)
        ],
    }
    if second_confidence is not None:
        summary["secondConfidence"] = round(float(second_confidence), 4)
        summary["confidenceGap"] = round(float(top_confidence - second_confidence), 4)
    _attach_direct_legal_numeric_margin(
        summary,
        unique_details.get(ranked[0][0]) or {},
        unique_details.get(ranked[1][0]) if len(ranked) > 1 else None,
    )
    return summary


def _public_direct_legal_candidate_detail(rank: int, state: str, confidence: float, details: Dict[str, Any]) -> Dict[str, Any]:
    public = {
        "rank": rank,
        "state": state,
        "confidence": round(float(confidence), 4),
    }
    for key in (
        "rawMergedScore",
        "scoreA",
        "scoreB",
        "selectionScoreA",
        "selectionScoreB",
        "orientationScoreA",
        "orientationScoreB",
        "orientationRankA",
        "orientationRankB",
        "variantCost",
        "balancedColorAssignmentScorePenalty",
    ):
        if details.get(key) is not None:
            public[key] = round(float(details[key]), 4)
    for key in ("sidePairA", "sidePairB", "orderedSidePairA", "orderedSidePairB"):
        if details.get(key):
            public[key] = details[key]
    faces = _selected_faces_by_image(details.get("sidePairA"), details.get("sidePairB"))
    if faces:
        public["selectedFacesByImage"] = faces
    sides = _selected_sides_by_image(details.get("orderedSidePairA"), details.get("orderedSidePairB"))
    if sides:
        public["selectedSidesByImage"] = sides
    return public


def _attach_direct_legal_numeric_margin(
    summary: Dict[str, Any],
    top_details: Dict[str, Any],
    second_details: Optional[Dict[str, Any]],
) -> None:
    _attach_direct_legal_metric_margin(
        summary,
        key="rawMergedScore",
        top_key="topRawMergedScore",
        second_key="secondRawMergedScore",
        gap_key="rawMergedScoreGap",
        top_details=top_details,
        second_details=second_details,
        higher_is_better=True,
    )
    _attach_direct_legal_metric_margin(
        summary,
        key="variantCost",
        top_key="topVariantCost",
        second_key="secondVariantCost",
        gap_key="variantCostGap",
        top_details=top_details,
        second_details=second_details,
        higher_is_better=False,
    )


def _attach_direct_legal_metric_margin(
    summary: Dict[str, Any],
    *,
    key: str,
    top_key: str,
    second_key: str,
    gap_key: str,
    top_details: Dict[str, Any],
    second_details: Optional[Dict[str, Any]],
    higher_is_better: bool,
) -> None:
    top_value = _optional_float_signal(top_details.get(key))
    if top_value is None:
        return
    summary[top_key] = round(top_value, 4)
    if not second_details:
        return
    second_value = _optional_float_signal(second_details.get(key))
    if second_value is None:
        return
    summary[second_key] = round(second_value, 4)
    gap = top_value - second_value if higher_is_better else second_value - top_value
    summary[gap_key] = round(gap, 4)


def _repair_details_with_orientation_selection_scores(
    repair_details: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    should_rerank = _repair_orientation_rerank_applies(repair_details)
    annotated: List[Dict[str, Any]] = []
    for item in repair_details:
        detail = dict(item)
        confidence = _float_signal(detail.get("confidence"))
        bonus = _repair_orientation_rerank_bonus(detail) if should_rerank else 0.0
        if bonus > 0.0:
            detail["preRerankConfidence"] = confidence
            detail["confidence"] = confidence + bonus
            detail["repairOrientationRerankBonus"] = bonus
        detail["repairSelectionScore"] = _float_signal(detail.get("confidence"))
        annotated.append(detail)
    if should_rerank:
        annotated.sort(key=lambda item: _float_signal(item.get("repairSelectionScore")), reverse=True)
    return annotated


def _repair_orientation_rerank_applies(repair_details: Sequence[Dict[str, Any]]) -> bool:
    if not repair_details:
        return False
    selected = repair_details[0]
    if _float_signal(selected.get("confidence")) >= REPAIRED_HIGH_CONFIDENCE_THRESHOLD:
        return False
    if _float_signal(selected.get("repairRankingPenalty")) < MAX_REPAIR_RANKING_PENALTY:
        return False
    orientation_rank_sum = _repair_orientation_rank_sum(selected)
    return orientation_rank_sum >= REPAIR_ORIENTATION_RERANK_MIN_TOP_RANK_SUM


def _repair_orientation_rerank_bonus(detail: Dict[str, Any]) -> float:
    rank_sum = _repair_orientation_rank_sum(detail)
    remaining_rank_gap = max(0, REPAIR_ORIENTATION_RERANK_RANK_LIMIT - rank_sum)
    return min(REPAIR_ORIENTATION_RERANK_MAX_BONUS, remaining_rank_gap * REPAIR_ORIENTATION_RERANK_BONUS_PER_RANK)


def _repair_orientation_rank_sum(detail: Dict[str, Any]) -> int:
    return _nonnegative_int(detail.get("orientationRankA")) + _nonnegative_int(detail.get("orientationRankB"))


def _candidate_selection_detail(merged: Dict[str, List[List[Any]]]) -> Dict[str, Any]:
    detail = {
        "sidePairA": _side_pair_key(merged.get("_side_pair_a")),
        "sidePairB": _side_pair_key(merged.get("_side_pair_b")),
        "orderedSidePairA": _ordered_side_pair_key(merged.get("_ordered_side_pair_a")),
        "orderedSidePairB": _ordered_side_pair_key(merged.get("_ordered_side_pair_b")),
    }
    for public_key, merged_key in (
        ("rawMergedScore", "_score"),
        ("scoreA", "_score_a"),
        ("scoreB", "_score_b"),
        ("selectionScoreA", "_selection_score_a"),
        ("selectionScoreB", "_selection_score_b"),
        ("orientationScoreA", "_orientation_score_a"),
        ("orientationScoreB", "_orientation_score_b"),
        ("orientationRankA", "_orientation_rank_a"),
        ("orientationRankB", "_orientation_rank_b"),
    ):
        value = merged.get(merged_key)
        if value is not None:
            detail[public_key] = value
    balanced_assignment = merged.get("_balanced_color_assignment")
    if isinstance(balanced_assignment, dict):
        detail["balancedColorAssignment"] = balanced_assignment
    penalty = merged.get("_balanced_color_assignment_score_penalty")
    if penalty is not None:
        detail["balancedColorAssignmentScorePenalty"] = penalty
    return detail


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
    balanced_assignment = selection.get("balancedColorAssignment")
    if isinstance(balanced_assignment, dict):
        signal["selectedBalancedColorAssignment"] = balanced_assignment
    penalty = selection.get("balancedColorAssignmentScorePenalty")
    if penalty is not None:
        signal["selectedBalancedColorAssignmentScorePenalty"] = penalty
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


def _side_pair_complement(side_pair: Any) -> str:
    faces = set(_side_pair_faces(side_pair))
    if len(faces) != 2:
        return ""
    return _side_pair_key(face for face in YAW_SIDE_ORDER if face not in faces)


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
        "preRerankConfidence",
        "repairSelectionScore",
        "repairOrientationRerankBonus",
        "balancedColorAssignmentScorePenalty",
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
                    "extrapolatedSamples": _grid_extrapolated_sample_count(grid),
                    "extrapolatedSampleScore": round(_grid_extrapolated_sample_score(grid), 3),
                    "unsupportedSamples": _grid_unsupported_sample_count(grid),
                    "unsupportedSampleScore": round(_grid_unsupported_sample_score(grid), 3),
                    "gridExtrapolationPenalty": round(_grid_extrapolation_penalty(grid), 3),
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


def _candidate_face_count_diagnostics(
    merged: Sequence[Tuple[float, Dict[str, List[List[Any]]]]],
    *,
    facelet_options_cache: Optional[Dict[FaceletOptionsKey, List[Tuple[str, float]]]] = None,
) -> Dict[str, Any]:
    face_counts: Counter[Tuple[Tuple[str, int], ...]] = Counter()
    validation_errors: Counter[str] = Counter()
    examples = []
    legal_count = 0
    sampled = 0

    for score, faces in merged[:MAX_DIAGNOSTIC_MERGES]:
        for state in _state_variants_from_faces(faces, facelet_options_cache=facelet_options_cache):
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


def _recognition_workset(analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> RecognitionWorkset:
    options_a = _oriented_face_options(analysis_a, "U")
    options_b = _oriented_face_options(analysis_b, "D")
    return RecognitionWorkset(
        options_a=options_a,
        options_b=options_b,
        merged_candidates=_merged_face_candidates(options_a, options_b),
    )


def _merged_face_candidates(
    options_a: Sequence[Dict[str, List[List[Any]]]],
    options_b: Sequence[Dict[str, List[List[Any]]]],
) -> List[Tuple[float, Dict[str, List[List[Any]]]]]:
    signatures_a = [_face_signature(faces) for faces in options_a]
    signatures_b = [_face_signature(faces) for faces in options_b]
    merged_faces: List[Tuple[float, Dict[str, List[List[Any]]]]] = []
    for faces_a, signature_a in zip(options_a, signatures_a):
        for faces_b, signature_b in zip(options_b, signatures_b):
            merged = _merge_faces(faces_a, faces_b)
            if merged is None:
                continue
            merged["_option_signature_a"] = signature_a
            merged["_option_signature_b"] = signature_b
            merged_faces.append((float(merged.get("_score", 0.0)), merged))
    merged_faces.sort(key=lambda item: item[0], reverse=True)
    return merged_faces


def _repair_backfill_applies(analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> bool:
    return _red_orange_pair_calibration_suspected(("piece_legality_invalid",), analysis_a, analysis_b)


def _repair_backfill_merged_face_candidates(
    analysis_a: ImageAnalysis,
    workset: RecognitionWorkset,
) -> List[Tuple[float, Dict[str, List[List[Any]]]]]:
    existing_a_signatures = {_face_signature(option) for option in workset.options_a}
    options_b_by_pair: Dict[str, List[Dict[str, List[List[Any]]]]] = {}
    for option in workset.options_b:
        pair_key = _side_pair_key(option.get("_side_pair"))
        if not pair_key:
            continue
        bucket = options_b_by_pair.setdefault(pair_key, [])
        if len(bucket) < MAX_REPAIR_BACKFILL_OPTIONS_B_PER_PAIR:
            bucket.append(option)

    selected: List[Tuple[int, int, int, float, Dict[str, List[List[Any]]]]] = []
    seen = set()
    for option_a in _oriented_face_option_candidates(analysis_a, "U")[:MAX_REPAIR_BACKFILL_OPTIONS_A]:
        signature_a = _face_signature(option_a)
        if signature_a in existing_a_signatures:
            continue
        complement = _side_pair_complement(option_a.get("_side_pair"))
        if not complement:
            continue
        for option_b in options_b_by_pair.get(complement, ()):
            merged = _merge_faces(option_a, option_b)
            if merged is None:
                continue
            signature = _cached_face_signature(merged)
            if signature in seen:
                continue
            conflicts = _piece_conflict_summary(merged)
            total_conflicts = int(conflicts.get("totalConflicts", 0))
            valid_corners = int(conflicts.get("validCorners", 0))
            valid_edges = int(conflicts.get("validEdges", 0))
            if total_conflicts > MAX_REPAIR_BACKFILL_CONFLICTS:
                continue
            if valid_corners < 6 or valid_edges < 10:
                continue
            seen.add(signature)
            selected.append((total_conflicts, -valid_corners, -valid_edges, -float(merged.get("_score", 0.0)), merged))

    selected.sort(key=lambda item: item[:4])
    return [(-raw_score, merged) for _, _, _, raw_score, merged in selected[:MAX_REPAIR_BACKFILL_MERGES]]


def _repair_merged_face_candidates(
    ranked: Sequence[Tuple[float, Dict[str, List[List[Any]]]]],
) -> List[Tuple[float, Dict[str, List[List[Any]]]]]:
    selected: List[Tuple[float, Dict[str, List[List[Any]]]]] = list(ranked[:MAX_LEGAL_REPAIR_MERGES])
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
            _cached_face_signature(merged),
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
        elif label == "image_b" and anchor == "D" and _down_anchor_grid_too_weak(assignments[anchor]):
            checks.append("image_b_D_anchor_weak")
        if _can_rank_grid_triples(analysis):
            groups = _candidate_grids_by_face(analysis, anchor)
            if anchor in groups and not _ranked_visible_face_triples(groups, anchor):
                checks.append(_face_triple_failure_check(label, groups, anchor))
    side_centers = [
        face
        for analysis, anchor in ((analysis_a, "U"), (analysis_b, "D"))
        for face in _assigned_grid_by_face(analysis, anchor)
        if face in {"R", "F", "L", "B"}
    ]
    if set(side_centers) != {"R", "F", "L", "B"}:
        checks.append("missing_side_face_coverage")
    if _visible_face_color_count_imbalance_suspected(analysis_a, analysis_b):
        checks.append(VISIBLE_FACE_COLOR_COUNT_IMBALANCE_CHECK)
    return checks


def _face_triple_failure_check(label: str, grids_by_face: Dict[str, List[FaceGrid]], anchor: str) -> str:
    if _low_quality_overlap_face_triples_exist(grids_by_face, anchor):
        return f"{label}_{FACE_TRIPLE_OVERLAP_LOW_QUALITY_CHECK}"
    return f"{label}_no_reliable_face_triple"


def _reason_for_checks(checks: Sequence[str]) -> str:
    if BACKGROUND_STICKER_NOISE_CHECK in checks:
        return "The cube stickers appear to be mixed with a textured or high-saturation background; retake with the cube isolated on a plainer surface."
    if "image_a_U_anchor_missing" in checks:
        return "Image A must contain the white/U center face; a logo is allowed if the sampled center is still white-ish."
    if "image_b_D_anchor_missing" in checks:
        return "Image B must contain the yellow/D center face after the flip."
    if "image_b_D_anchor_weak" in checks:
        return "Image B contains a weak yellow/D center grid; retake with a clearer yellow-up face."
    if "image_a_face_triple_overlap_low_quality" in checks:
        return "Image A only produced overlapping or low-quality three-face grids; retake with clearer face separation."
    if "image_b_face_triple_overlap_low_quality" in checks:
        return "Image B only produced overlapping or low-quality three-face grids; retake with clearer face separation."
    if "image_a_no_reliable_face_triple" in checks:
        return "Image A did not contain a reliable non-overlapping three-face grid."
    if "image_b_no_reliable_face_triple" in checks:
        return "Image B did not contain a reliable non-overlapping three-face grid."
    if "missing_side_face_coverage" in checks:
        return "The two flip photos do not expose all four side face centers."
    if VISIBLE_FACE_COLOR_COUNT_IMBALANCE_CHECK in checks:
        return "The selected visible grids have implausible two-view sticker color counts."
    return "The images did not satisfy the two-view flip recognition prerequisites."


def _down_anchor_grid_too_weak(grid: FaceGrid) -> bool:
    return grid.matched_count < MIN_IMAGE_B_D_ANCHOR_MATCHED_COUNT


def _apply_pair_color_calibration(analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> None:
    samples = _pair_calibration_samples(analysis_a, analysis_b)
    anchors = _pair_calibration_anchors(analysis_a, analysis_b)
    palette = build_adaptive_palette(samples, anchors)
    for analysis in (analysis_a, analysis_b):
        for sticker in _analysis_stickers_for_reclassification(analysis):
            sticker.match = classify_rgb(sticker.rgb, palette)


def _attach_failed_pair_color_calibration_signal(
    result: RecognitionResult,
    calibrated_result: RecognitionResult,
    raw_a: ImageAnalysis,
    raw_b: ImageAnalysis,
    calibrated_a: ImageAnalysis,
    calibrated_b: ImageAnalysis,
) -> None:
    if RED_ORANGE_PAIR_CALIBRATION_SUSPECTED_CHECK not in set(result.failed_checks or ()):
        return
    signals = dict(result.recognition_signals or {})
    signals["pairColorCalibration"] = _pair_color_calibration_signal(
        raw_a,
        raw_b,
        calibrated_a,
        calibrated_b,
        result,
        calibrated_result,
    )
    result.recognition_signals = signals


def _recognition_signals_with_failed_checks(
    signals: Dict[str, Any],
    checks: Sequence[str],
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
) -> Dict[str, Any]:
    if BACKGROUND_STICKER_NOISE_CHECK not in set(checks):
        return signals
    return {
        **signals,
        "backgroundStickerNoise": _background_sticker_noise_signal(
            checks,
            analysis_a,
            analysis_b,
        ),
    }


def _background_sticker_noise_signal(
    checks: Sequence[str],
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
) -> Dict[str, Any]:
    return {
        "status": "suspected",
        "reason": _background_sticker_noise_reason(checks, analysis_a, analysis_b),
        "images": {
            "imageA": _background_sticker_noise_image_signal(analysis_a, anchor="U"),
            "imageB": _background_sticker_noise_image_signal(analysis_b, anchor="D"),
        },
        "visibleFacePairColorCounts": _visible_face_pair_color_count_signal(analysis_a, analysis_b),
    }


def _background_sticker_noise_reason(
    checks: Sequence[str],
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
) -> str:
    unique = set(checks)
    if "image_a_U_anchor_missing" in set(checks) and _dominant_grid_center_face(analysis_a)[0] == "B":
        return "image_a_u_anchor_missing_with_blue_grid_dominance"
    if "no_legal_state" in unique and _dominant_grid_center_face(analysis_a)[0] == "B":
        return "no_legal_state_with_blue_grid_dominance"
    if "no_legal_state" in unique and _anchor_evidence_collapsed(analysis_a, analysis_b):
        return "no_legal_state_with_anchor_evidence_collapse"
    if "image_a_face_triple_overlap_low_quality" in unique:
        return "image_a_face_triple_low_quality_with_anchor_evidence_collapse"
    if VISIBLE_FACE_COLOR_COUNT_IMBALANCE_CHECK in unique:
        return "visible_face_color_count_imbalance_with_anchor_evidence_collapse"
    if _many_face_counts_failed(unique):
        return "multi_face_count_failure_with_anchor_evidence_collapse"
    return "all_face_counts_failed_with_anchor_evidence_collapse"


def _background_sticker_noise_image_signal(analysis: ImageAnalysis, *, anchor: str) -> Dict[str, Any]:
    dominant_face, dominant_count, total_grids = _dominant_grid_center_face(analysis)
    selected_anchor = _assigned_grid_by_face(analysis, anchor).get(anchor)
    return {
        "roi": list(getattr(analysis, "roi", ())),
        "stickerFaceCounts": _face_count_dict(_face_counts_from_stickers(analysis)),
        "gridCenterFaceCounts": _face_count_dict(_face_counts_from_grid_centers(analysis)),
        "dominantGridCenterFace": dominant_face,
        "dominantGridCenterCount": dominant_count,
        "gridCount": total_grids,
        "selectedAnchor": _grid_signal_summary(selected_anchor) if selected_anchor is not None else None,
    }


def _pair_color_calibration_signal(
    raw_a: ImageAnalysis,
    raw_b: ImageAnalysis,
    calibrated_a: ImageAnalysis,
    calibrated_b: ImageAnalysis,
    raw_result: RecognitionResult,
    calibrated_result: RecognitionResult,
) -> Dict[str, Any]:
    samples = _pair_calibration_samples(raw_a, raw_b)
    anchors = _pair_calibration_anchors(raw_a, raw_b)
    palette = build_adaptive_palette(samples, anchors)
    return {
        "status": "attempted_no_legal_state",
        "rawFailedChecks": list(raw_result.failed_checks or []),
        "calibratedFailedChecks": list(calibrated_result.failed_checks or []),
        "sampleCount": len(samples),
        "anchorCounts": {
            color: len(anchors.get(color, ()))
            for color in FACE_TO_CENTER_COLOR.values()
        },
        "anchorRgb": {
            color: [list(rgb) for rgb in anchors.get(color, ())]
            for color in ("red", "orange")
        },
        "paletteRgb": {
            color: list(palette[color])
            for color in ("red", "orange")
            if color in palette
        },
        "images": {
            "imageA": _pair_color_calibration_image_signal(raw_a, calibrated_a, anchor="U"),
            "imageB": _pair_color_calibration_image_signal(raw_b, calibrated_b, anchor="D"),
        },
    }


def _pair_color_calibration_image_signal(raw: ImageAnalysis, calibrated: ImageAnalysis, *, anchor: str) -> Dict[str, Any]:
    raw_stickers = _face_count_dict(_face_counts_from_stickers(raw))
    calibrated_stickers = _face_count_dict(_face_counts_from_stickers(calibrated))
    raw_grids = _face_count_dict(_face_counts_from_grid_centers(raw))
    calibrated_grids = _face_count_dict(_face_counts_from_grid_centers(calibrated))
    return {
        "rawStickerFaceCounts": raw_stickers,
        "calibratedStickerFaceCounts": calibrated_stickers,
        "rawGridCenterFaceCounts": raw_grids,
        "calibratedGridCenterFaceCounts": calibrated_grids,
        "rawRedOrangeSkew": _red_orange_count_signal(raw_stickers),
        "calibratedRedOrangeSkew": _red_orange_count_signal(calibrated_stickers),
        "rawRedOrangeEvidence": _red_orange_skew_evidence(raw),
        "calibratedRedOrangeEvidence": _red_orange_skew_evidence(calibrated),
        "selectedFaceEvidence": {
            "raw": _selected_face_color_evidence(raw, anchor),
            "calibrated": _selected_face_color_evidence(calibrated, anchor),
        },
    }


def _selected_face_color_evidence(analysis: ImageAnalysis, anchor: str) -> Dict[str, Dict[str, Any]]:
    return {
        face: _face_grid_color_evidence(face, grid)
        for face, grid in _assigned_grid_by_face(analysis, anchor).items()
    }


def _face_grid_color_evidence(face: str, grid: FaceGrid) -> Dict[str, Any]:
    center = grid.center_sticker
    match = getattr(center, "match", None) or classify_rgb(center.rgb)
    return {
        "gridId": getattr(grid, "id", None),
        "expectedFace": face,
        "expectedColor": FACE_TO_CENTER_COLOR.get(face),
        "centerFace": getattr(match, "face", None),
        "centerColor": getattr(match, "color", None),
        "centerRgb": list(center.rgb),
        "centerConfidence": round(float(getattr(match, "confidence", 0.0)), 4),
        "centerDistances": _color_distance_signal(match, PAIR_COLOR_EVIDENCE_COLORS),
        "matchedCount": grid.matched_count,
        "fitError": round(grid.fit_error, 3),
        "quality": round(_grid_quality_score(grid), 3),
        "gridSamples": _grid_sample_count(grid),
        "cellFaceEvidence": {
            face: _grid_cell_face_counts(grid).get(face, 0)
            for face in PAIR_COLOR_EVIDENCE_FACES
        },
        "cellSourceCounts": _grid_cell_source_counts(grid),
    }


def _color_distance_signal(match: Any, colors: Sequence[str]) -> Dict[str, float]:
    distances = {
        color: float(distance)
        for color, distance in getattr(match, "alternatives", [])
        if color in colors
    }
    return {
        color: round(distances[color], 4)
        for color in colors
        if color in distances
    }


def _face_counts_from_stickers(analysis: ImageAnalysis) -> Counter[str]:
    return Counter(
        getattr(getattr(sticker, "match", None), "face", None)
        for sticker in getattr(analysis, "stickers", [])
    )


def _face_counts_from_grid_centers(analysis: ImageAnalysis) -> Counter[str]:
    return Counter(getattr(grid, "center_face", None) for grid in getattr(analysis, "grids", []))


def _face_count_dict(counts: Counter[str]) -> Dict[str, int]:
    return {
        face: int(counts.get(face, 0))
        for face in FACE_ORDER
        if counts.get(face, 0)
    }


def _red_orange_count_signal(counts: Dict[str, int]) -> Dict[str, Any]:
    red = int(counts.get("R", 0))
    orange = int(counts.get("L", 0))
    dominant = None
    if red != orange:
        dominant = "R" if red > orange else "L"
    return {
        "redCount": red,
        "orangeCount": orange,
        "gap": abs(red - orange),
        "dominantFace": dominant,
    }


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
            rgb = _calibration_anchor_rgb(face, anchor, grid)
            if color and rgb is not None:
                anchors[color].append(rgb)
    return anchors


def _calibration_anchor_rgb(face: str, anchor: str, grid: FaceGrid) -> Optional[Tuple[int, int, int]]:
    if face == anchor and grid.center_face != anchor:
        if anchor == "U" and not _center_sample_is_whiteish(grid):
            return None
        if anchor == "D" and not _center_sample_is_yellowish(grid):
            return None
    return grid.center_sticker.rgb


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


def _validation_failed_checks(errors: Sequence[str], analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> List[str]:
    return _failed_checks_with_context(_summarize_validation_errors(errors), analysis_a, analysis_b)


def _failed_checks_with_context(
    checks: Sequence[str],
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
) -> List[str]:
    contextual = list(checks)
    if _red_orange_pair_calibration_suspected(contextual, analysis_a, analysis_b):
        _append_failed_check(contextual, RED_ORANGE_PAIR_CALIBRATION_SUSPECTED_CHECK)
        if _image_b_visible_face_evidence_weak(analysis_b):
            _append_failed_check(contextual, IMAGE_B_VISIBLE_FACE_EVIDENCE_WEAK_CHECK)
    if _background_sticker_noise_suspected(contextual, analysis_a, analysis_b):
        _append_failed_check(contextual, BACKGROUND_STICKER_NOISE_CHECK)
    return contextual


def _append_failed_check(checks: List[str], check: str) -> None:
    if check not in checks:
        checks.append(check)


def _background_sticker_noise_suspected(
    checks: Sequence[str],
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
) -> bool:
    unique = set(checks)
    if _all_face_counts_failed(unique) or _many_face_counts_failed(unique):
        return (
            _selected_anchor_self_face_count(analysis_a, "U") <= MAX_BACKGROUND_STICKER_NOISE_ANCHOR_SELF_FACE_CELLS
            and _selected_anchor_self_face_count(analysis_b, "D") <= MAX_BACKGROUND_STICKER_NOISE_ANCHOR_SELF_FACE_CELLS
        )
    if "image_a_U_anchor_missing" in unique:
        dominant_face, dominant_count, total_grids = _dominant_grid_center_face(analysis_a)
        return (
            dominant_face == "B"
            and dominant_count >= MIN_BACKGROUND_STICKER_NOISE_DOMINANT_GRID_CENTER_COUNT
            and total_grids > 0
            and dominant_count / total_grids >= MIN_BACKGROUND_STICKER_NOISE_DOMINANT_GRID_CENTER_SHARE
        )
    if "no_legal_state" in unique:
        dominant_face, dominant_count, total_grids = _dominant_grid_center_face(analysis_a)
        blue_grid_dominance = (
            dominant_face == "B"
            and dominant_count >= MIN_BACKGROUND_STICKER_NOISE_DOMINANT_GRID_CENTER_COUNT
            and total_grids > 0
            and dominant_count / total_grids >= MIN_BACKGROUND_STICKER_NOISE_DOMINANT_GRID_CENTER_SHARE
        )
        return blue_grid_dominance or _anchor_evidence_collapsed(analysis_a, analysis_b)
    if "image_a_face_triple_overlap_low_quality" in unique:
        return _anchor_evidence_collapsed(analysis_a, analysis_b)
    if VISIBLE_FACE_COLOR_COUNT_IMBALANCE_CHECK in unique:
        return _anchor_evidence_collapsed(analysis_a, analysis_b)
    return False


def _all_face_counts_failed(checks: set[str]) -> bool:
    return all(f"{face}_count_not_9" in checks for face in FACE_ORDER)


def _many_face_counts_failed(checks: set[str]) -> bool:
    face_count_failures = sum(1 for face in FACE_ORDER if f"{face}_count_not_9" in checks)
    return face_count_failures >= 5 and "piece_legality_invalid" in checks


def _visible_face_color_count_imbalance_suspected(analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> bool:
    counts = _top_visible_face_pair_color_counts(analysis_a, analysis_b)
    if counts is None:
        return False
    imbalance = sum(abs(int(counts.get(face, 0)) - 9) for face in FACE_ORDER)
    return (
        imbalance > MAX_VISIBLE_FACE_PAIR_COLOR_COUNT_IMBALANCE
        and _anchor_evidence_collapsed(analysis_a, analysis_b)
    )


def _top_visible_face_pair_color_counts(
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
) -> Optional[Dict[str, int]]:
    counts = Counter()
    for analysis, anchor in ((analysis_a, "U"), (analysis_b, "D")):
        image_counts = _top_visible_face_color_counts(analysis, anchor)
        if image_counts is None:
            return None
        counts.update(image_counts)
    return {face: int(counts.get(face, 0)) for face in FACE_ORDER}


def _top_visible_face_color_counts(analysis: ImageAnalysis, anchor: str) -> Optional[Dict[str, int]]:
    if not _can_rank_grid_triples(analysis):
        return None
    groups = _candidate_grids_by_face(analysis, anchor)
    if anchor not in groups:
        return None
    triples = _ranked_visible_face_triples(groups, anchor)
    if not triples:
        return None
    counts = Counter()
    for grid in triples[0][1].values():
        counts.update(_grid_cell_face_counts(grid))
    return {face: int(counts.get(face, 0)) for face in FACE_ORDER}


def _visible_face_pair_color_count_signal(
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
) -> Optional[Dict[str, Any]]:
    counts = _top_visible_face_pair_color_counts(analysis_a, analysis_b)
    if counts is None:
        return None
    imbalance = sum(abs(int(counts.get(face, 0)) - 9) for face in FACE_ORDER)
    return {
        "counts": counts,
        "imbalance": imbalance,
        "maxAllowedImbalance": MAX_VISIBLE_FACE_PAIR_COLOR_COUNT_IMBALANCE,
        "suspected": imbalance > MAX_VISIBLE_FACE_PAIR_COLOR_COUNT_IMBALANCE,
    }


def _top_visible_face_pair_balanced_assignment_signal(
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
) -> Optional[Dict[str, Any]]:
    facelets = _top_visible_face_pair_facelets(analysis_a, analysis_b)
    if facelets is None:
        return None
    summary = _balanced_color_assignment_summary_from_facelets(facelets, include_changes=True)
    return {
        **summary,
        "scorePenalty": round(_balanced_color_assignment_score_penalty(summary), 4),
        "source": "top_visible_triples",
    }


def _top_visible_face_pair_facelets(
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
) -> Optional[List[Tuple[str, int, int, Any]]]:
    facelets: List[Tuple[str, int, int, Any]] = []
    for analysis, anchor in ((analysis_a, "U"), (analysis_b, "D")):
        if not _can_rank_grid_triples(analysis):
            return None
        groups = _candidate_grids_by_face(analysis, anchor)
        if anchor not in groups:
            return None
        triples = _ranked_visible_face_triples(groups, anchor)
        if not triples:
            return None
        for face, grid in triples[0][1].items():
            for row_index, row in enumerate(getattr(grid, "stickers", []) or []):
                for col_index, facelet in enumerate(row):
                    facelets.append((face, row_index, col_index, facelet))
    return facelets


def _balanced_color_assignment_summary(
    faces: Dict[str, List[List[Any]]],
    *,
    include_changes: bool = False,
) -> Dict[str, Any]:
    return _balanced_color_assignment_summary_from_facelets(
        _visible_facelets_from_faces(faces),
        include_changes=include_changes,
    )


def _balanced_color_assignment_scoring_summary(faces: Dict[str, List[List[Any]]]) -> Dict[str, Any]:
    return _balanced_color_assignment_scoring_summary_from_counts(_primary_face_counts(faces))


def _balanced_color_assignment_scoring_summary_from_counts(counts: Dict[str, int]) -> Dict[str, Any]:
    primary_counts = Counter(counts)
    imbalance = _face_count_deviation(primary_counts)
    required_changes = sum(max(0, int(primary_counts.get(face, 0)) - 9) for face in FACE_ORDER)
    status = "balanced" if imbalance == 0 else "primary_count_scored"
    summary: Dict[str, Any] = {
        "status": status,
        "cellCount": sum(int(primary_counts.get(face, 0)) for face in FACE_ORDER),
        "primaryCounts": _full_face_counts_dict(primary_counts),
        "assignedCounts": _full_face_counts_dict(primary_counts),
        "imbalance": int(imbalance),
        "requiredChanges": int(required_changes),
        "cost": None if status != "balanced" else 0.0,
        "maxChangeCost": None if status != "balanced" else 0.0,
        "highCostChanges": 0,
        "maxExactChanges": 0,
    }
    return summary


def _visible_facelets_from_faces(faces: Dict[str, List[List[Any]]]) -> List[Tuple[str, int, int, Any]]:
    facelets: List[Tuple[str, int, int, Any]] = []
    for face in FACE_ORDER:
        matrix = faces.get(face)
        if not matrix:
            continue
        for row_index, row in enumerate(matrix):
            for col_index, facelet in enumerate(row):
                facelets.append((face, row_index, col_index, facelet))
    return facelets


def _balanced_color_assignment_summary_from_facelets(
    facelets: Sequence[Tuple[str, int, int, Any]],
    *,
    include_changes: bool = False,
    max_exact_changes: int = MAX_BALANCED_COLOR_ASSIGNMENT_EXACT_CHANGES,
    facelet_options_cache: Optional[Dict[FaceletOptionsKey, List[Tuple[str, float]]]] = None,
) -> Dict[str, Any]:
    options_by_cell = [_cached_facelet_options(facelet, facelet_options_cache) for _, _, _, facelet in facelets]
    primary = [options[0][0] if options else "unknown" for options in options_by_cell]
    primary_counts = Counter(face for face in primary if face in FACE_ORDER)
    unknown_count = sum(1 for face in primary if face not in FACE_ORDER)
    if unknown_count:
        primary_counts["unknown"] = unknown_count
    imbalance = _face_count_deviation(primary_counts)
    required_changes = sum(max(0, int(primary_counts.get(face, 0)) - 9) for face in FACE_ORDER)
    base_summary: Dict[str, Any] = {
        "status": "balanced",
        "cellCount": len(facelets),
        "primaryCounts": _full_face_counts_dict(primary_counts),
        "assignedCounts": _full_face_counts_dict(primary_counts),
        "imbalance": int(imbalance),
        "requiredChanges": int(required_changes),
        "cost": 0.0,
        "maxChangeCost": 0.0,
        "highCostChanges": 0,
        "maxExactChanges": max_exact_changes,
    }
    if len(facelets) != 54:
        return {
            **base_summary,
            "status": "incomplete",
            "assignedCounts": _full_face_counts_dict(primary_counts),
            "cost": None,
        }
    if imbalance == 0:
        if include_changes:
            base_summary["changes"] = []
        return base_summary
    if required_changes > max_exact_changes:
        return {
            **base_summary,
            "status": "too_imbalanced",
            "cost": None,
        }

    surplus = {face: int(primary_counts.get(face, 0)) - 9 for face in FACE_ORDER if primary_counts.get(face, 0) > 9}
    deficits = {face: 9 - int(primary_counts.get(face, 0)) for face in FACE_ORDER if primary_counts.get(face, 0) < 9}
    moves_by_target: Dict[str, List[Tuple[float, int, str, str]]] = {face: [] for face in deficits}
    for index, (from_face, options) in enumerate(zip(primary, options_by_cell)):
        if surplus.get(from_face, 0) <= 0:
            continue
        seen_targets = set()
        for rank, (to_face, cost) in enumerate(options[1:], start=1):
            if to_face not in deficits or to_face in seen_targets:
                continue
            move_cost = float(cost) + rank * 0.05
            moves_by_target[to_face].append((move_cost, index, from_face, to_face))
            seen_targets.add(to_face)
    for moves in moves_by_target.values():
        moves.sort(key=lambda item: item[0])
        del moves[MAX_BALANCED_COLOR_ASSIGNMENT_MOVES_PER_TARGET:]

    best_cost = float("inf")
    best_moves: List[Tuple[float, int, str, str]] = []

    greedy_surplus = dict(surplus)
    greedy_deficits = dict(deficits)
    greedy_used: set[int] = set()
    greedy_moves: List[Tuple[float, int, str, str]] = []
    greedy_cost = 0.0
    for target in sorted(greedy_deficits, key=lambda face: greedy_deficits[face], reverse=True):
        while greedy_deficits.get(target, 0) > 0:
            move = next(
                (
                    item
                    for item in moves_by_target.get(target, [])
                    if item[1] not in greedy_used and greedy_surplus.get(item[2], 0) > 0
                ),
                None,
            )
            if move is None:
                break
            move_cost, index, from_face, to_face = move
            greedy_used.add(index)
            greedy_surplus[from_face] -= 1
            greedy_deficits[to_face] -= 1
            greedy_moves.append(move)
            greedy_cost += move_cost
    if all(count == 0 for count in greedy_deficits.values()):
        best_cost = greedy_cost
        best_moves = list(greedy_moves)

    def backtrack(
        remaining_surplus: Dict[str, int],
        remaining_deficits: Dict[str, int],
        used: set[int],
        chosen: List[Tuple[float, int, str, str]],
        cost: float,
    ) -> None:
        nonlocal best_cost, best_moves
        if cost >= best_cost:
            return
        targets = [face for face, count in remaining_deficits.items() if count > 0]
        if not targets:
            best_cost = cost
            best_moves = list(chosen)
            return
        target = max(targets, key=lambda face: (remaining_deficits[face], -len(moves_by_target.get(face, []))))
        for move in moves_by_target.get(target, []):
            move_cost, index, from_face, to_face = move
            if index in used:
                continue
            if remaining_surplus.get(from_face, 0) <= 0 or remaining_deficits.get(to_face, 0) <= 0:
                continue
            next_surplus = dict(remaining_surplus)
            next_deficits = dict(remaining_deficits)
            next_surplus[from_face] -= 1
            next_deficits[to_face] -= 1
            used.add(index)
            chosen.append(move)
            backtrack(next_surplus, next_deficits, used, chosen, cost + move_cost)
            chosen.pop()
            used.remove(index)

    backtrack(surplus, deficits, set(), [], 0.0)
    if not best_moves:
        return {
            **base_summary,
            "status": "unassignable",
            "cost": None,
        }

    assigned_counts = Counter(primary_counts)
    change_rows: List[Dict[str, Any]] = []
    for move_cost, index, from_face, to_face in best_moves:
        assigned_counts[from_face] -= 1
        assigned_counts[to_face] += 1
        if include_changes:
            face, row, col, _ = facelets[index]
            change_rows.append(
                {
                    "cell": f"{face}{row}{col}",
                    "from": from_face,
                    "to": to_face,
                    "cost": round(float(move_cost), 3),
                }
            )
    change_costs = [move[0] for move in best_moves]
    summary = {
        **base_summary,
        "status": "assigned",
        "assignedCounts": _full_face_counts_dict(assigned_counts),
        "cost": round(float(best_cost), 4),
        "maxChangeCost": round(max(change_costs), 4) if change_costs else 0.0,
        "highCostChanges": sum(1 for cost in change_costs if cost >= BALANCED_COLOR_ASSIGNMENT_HIGH_COST),
    }
    if include_changes:
        summary["changes"] = change_rows
    return summary


def _balanced_color_assignment_score_penalty(summary: Dict[str, Any]) -> float:
    status = summary.get("status")
    if status == "balanced":
        return 0.0
    if status == "assigned":
        penalty = (
            _int_signal(summary.get("requiredChanges")) * BALANCED_COLOR_ASSIGNMENT_CHANGE_WEIGHT
            + _float_signal(summary.get("cost")) * BALANCED_COLOR_ASSIGNMENT_COST_SCALE
            + _int_signal(summary.get("highCostChanges")) * BALANCED_COLOR_ASSIGNMENT_HIGH_COST_WEIGHT
        )
    elif status == "primary_count_scored":
        penalty = _int_signal(summary.get("imbalance")) * BALANCED_COLOR_ASSIGNMENT_DEVIATION_WEIGHT
    else:
        penalty = _int_signal(summary.get("imbalance")) * BALANCED_COLOR_ASSIGNMENT_DEVIATION_WEIGHT
    return min(MAX_BALANCED_COLOR_ASSIGNMENT_SCORE_PENALTY, max(0.0, penalty))


def _balanced_color_assignment_scoring_enabled() -> bool:
    return os.environ.get(BALANCED_COLOR_SCORING_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _full_face_counts_dict(counts: Counter[str] | Dict[str, int]) -> Dict[str, int]:
    return {face: int(counts.get(face, 0)) for face in FACE_ORDER}


def _selected_anchor_self_face_count(analysis: ImageAnalysis, anchor: str) -> int:
    if not hasattr(analysis, "grids"):
        return 9
    grid = _assigned_grid_by_face(analysis, anchor).get(anchor)
    if grid is None:
        return 9
    return int(_grid_cell_face_counts(grid).get(anchor, 0))


def _anchor_evidence_collapsed(analysis_a: ImageAnalysis, analysis_b: ImageAnalysis) -> bool:
    return (
        _selected_anchor_self_face_count(analysis_a, "U") <= MAX_BACKGROUND_STICKER_NOISE_ANCHOR_SELF_FACE_CELLS
        and _selected_anchor_self_face_count(analysis_b, "D") <= MAX_BACKGROUND_STICKER_NOISE_ANCHOR_SELF_FACE_CELLS
    )


def _dominant_grid_center_face(analysis: ImageAnalysis) -> Tuple[Optional[str], int, int]:
    counts = Counter(getattr(grid, "center_face", None) for grid in getattr(analysis, "grids", []))
    counts = Counter({face: count for face, count in counts.items() if face in FACE_ORDER})
    if not counts:
        return None, 0, 0
    face, count = counts.most_common(1)[0]
    return face, int(count), int(sum(counts.values()))


def _image_b_visible_face_evidence_weak(analysis_b: ImageAnalysis) -> bool:
    assignments = _assigned_grid_by_face(analysis_b, "D")
    weak_side_faces = [
        face
        for face in YAW_SIDE_ORDER
        if face in assignments and _selected_image_b_side_grid_weak(assignments[face])
    ]
    return len(weak_side_faces) >= MIN_WEAK_IMAGE_B_SIDE_GRIDS


def _selected_image_b_side_grid_weak(grid: FaceGrid) -> bool:
    if _grid_sample_count(grid) < MIN_WEAK_IMAGE_B_SIDE_GRID_SAMPLES:
        return False
    return (
        grid.matched_count <= MAX_WEAK_IMAGE_B_SIDE_GRID_MATCHED_COUNT
        or _grid_quality_score(grid) <= MAX_WEAK_IMAGE_B_SIDE_GRID_QUALITY
    )


def _red_orange_pair_calibration_suspected(
    checks: Sequence[str],
    analysis_a: ImageAnalysis,
    analysis_b: ImageAnalysis,
) -> bool:
    unique = set(checks)
    if "piece_legality_invalid" not in unique and not ({"R_count_not_9", "L_count_not_9"} & unique):
        return False

    image_a = _red_orange_skew_evidence(analysis_a)
    image_b = _red_orange_skew_evidence(analysis_b)
    return any(a["dominantFace"] != b["dominantFace"] for a in image_a for b in image_b)


def _red_orange_skew_evidence(analysis: ImageAnalysis) -> List[Dict[str, Any]]:
    sources = {
        "stickers": Counter(
            getattr(getattr(sticker, "match", None), "face", None)
            for sticker in getattr(analysis, "stickers", [])
        ),
        "gridCenters": Counter(getattr(grid, "center_face", None) for grid in getattr(analysis, "grids", [])),
    }
    evidence: List[Dict[str, Any]] = []
    for source, counts in sources.items():
        red = int(counts.get("R", 0))
        orange = int(counts.get("L", 0))
        gap = abs(red - orange)
        dominant = "R" if red > orange else "L"
        if red == orange or gap < RED_ORANGE_SKEW_MIN_GAP or max(red, orange) < RED_ORANGE_SKEW_MIN_DOMINANT:
            continue
        evidence.append(
            {
                "source": source,
                "dominantFace": dominant,
                "redCount": red,
                "orangeCount": orange,
                "gap": gap,
            }
        )
    return evidence


def _oriented_face_options(analysis: ImageAnalysis, anchor: str) -> List[Dict[str, List[List[Any]]]]:
    ranked_results = _oriented_face_option_candidates(analysis, anchor)
    if not ranked_results:
        return []

    scored_options = []
    seen = set()
    side_pair_counts: Counter[Tuple[str, ...]] = Counter()
    side_pair_kinds = {tuple(option.get("_side_pair", ())) for option in ranked_results}
    per_side_pair_limit = min(
        MAX_ORIENTED_OPTIONS_PER_SIDE_PAIR,
        max(24, math.ceil(MAX_ORIENTED_OPTIONS_PER_IMAGE / max(1, len(side_pair_kinds)))),
    )
    for option in ranked_results:
        side_pair = tuple(option.get("_side_pair", ()))
        signature = (side_pair, _face_signature(option))
        if signature in seen:
            continue
        if side_pair_counts[side_pair] >= per_side_pair_limit:
            continue
        seen.add(signature)
        side_pair_counts[side_pair] += 1
        scored_options.append(option)
        if len(scored_options) >= MAX_ORIENTED_OPTIONS_PER_IMAGE:
            break
    return scored_options


def _oriented_face_option_candidates(analysis: ImageAnalysis, anchor: str) -> List[Dict[str, List[List[Any]]]]:
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

    candidates = []
    balanced_scoring_enabled = _balanced_color_assignment_scoring_enabled()
    for selection_score, side_pair, ordered_side_pair, orientation_score, orientation_rank, option in ranked_results:
        scored = dict(option)
        scored["_score"] = selection_score + orientation_score * ORIENTATION_SCORE_WEIGHT
        scored["_selection_score"] = selection_score
        scored["_orientation_score"] = orientation_score
        scored["_orientation_rank"] = orientation_rank
        scored["_side_pair"] = side_pair
        scored["_ordered_side_pair"] = ordered_side_pair
        if balanced_scoring_enabled:
            scored["_visible_color_counts"] = _primary_face_counts(scored)
        candidates.append(scored)
    return candidates


def _ranked_visible_face_triples(grids_by_face: Dict[str, List[FaceGrid]], anchor: str) -> List[Tuple[float, Dict[str, FaceGrid]]]:
    if anchor not in grids_by_face:
        return []

    triples = _visible_face_triples(
        grids_by_face,
        anchor,
        max_overlap=MAX_TRIPLE_COMPONENT_OVERLAP,
        extra_overlap_penalty=0.0,
    )
    if triples:
        return triples[:MAX_VISIBLE_FACE_TRIPLES]

    triples = _visible_face_triples(
        grids_by_face,
        anchor,
        max_overlap=MAX_RESCUE_TRIPLE_COMPONENT_OVERLAP,
        extra_overlap_penalty=24.0,
    )
    triples = [item for item in triples if item[0] >= MIN_RESCUE_VISIBLE_FACE_TRIPLE_SCORE]
    return triples[:MAX_RESCUE_VISIBLE_FACE_TRIPLES]


def _low_quality_overlap_face_triples_exist(grids_by_face: Dict[str, List[FaceGrid]], anchor: str) -> bool:
    if anchor not in grids_by_face:
        return False
    triples = _visible_face_triples(
        grids_by_face,
        anchor,
        max_overlap=MAX_RESCUE_TRIPLE_COMPONENT_OVERLAP,
        extra_overlap_penalty=24.0,
    )
    return bool(triples)


def _visible_face_triples(
    grids_by_face: Dict[str, List[FaceGrid]],
    anchor: str,
    *,
    max_overlap: int,
    extra_overlap_penalty: float,
) -> List[Tuple[float, Dict[str, FaceGrid]]]:
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
            if _triple_has_collapsed_anchor_contamination(anchor_grid, first_grid, second_grid, anchor):
                continue
            overlap = _triple_overlap_count((anchor_grid, first_grid, second_grid))
            if overlap > max_overlap:
                continue
            subset = {anchor: anchor_grid, side_pair[0]: first_grid, side_pair[1]: second_grid}
            rescue_penalty = max(0, overlap - MAX_TRIPLE_COMPONENT_OVERLAP) * extra_overlap_penalty
            triples.append((_face_plane_score(anchor_grid, first_grid, second_grid) - rescue_penalty, subset))
    triples.sort(key=lambda item: item[0], reverse=True)
    return triples


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
            overlap = _triple_overlap_count((anchor_grid, first_grid, second_grid))
            if overlap > MAX_TRIPLE_COMPONENT_OVERLAP:
                counts["componentOverlap"] += 1
                if overlap <= MAX_RESCUE_TRIPLE_COMPONENT_OVERLAP:
                    counts["rescueOverlap"] += 1
                continue
            counts["accepted"] += 1
    return {
        key: counts[key]
        for key in ("total", "accepted", "duplicateGridId", "unusableGrid", "componentOverlap", "rescueOverlap")
        if counts[key]
    }


def _face_plane_score(anchor_grid: FaceGrid, first_grid: FaceGrid, second_grid: FaceGrid) -> float:
    grids = (anchor_grid, first_grid, second_grid)
    score = sum(grid.matched_count * 12.0 - grid.fit_error * 0.9 for grid in grids)
    score -= sum(min(65.0, _grid_shape_spread(grid)) for grid in grids) * 0.75
    score -= sum(_grid_sample_penalty(grid) for grid in grids) * 0.9
    score -= sum(_grid_cube_hull_penalty(grid) for grid in grids)

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
    if grid.matched_count <= 5 and suspect >= MAX_LOW_MATCH_GRID_SUSPECT_SAMPLE_SCORE:
        return False
    return True


def _triple_has_collapsed_anchor_contamination(
    anchor_grid: FaceGrid,
    first_grid: FaceGrid,
    second_grid: FaceGrid,
    anchor: str,
) -> bool:
    if _grid_cell_face_counts(anchor_grid).get(anchor, 0) > MAX_COLLAPSED_ANCHOR_GRID_SELF_FACE_CELLS:
        return False
    return any(_side_grid_contaminated_near_collapsed_anchor(grid) for grid in (first_grid, second_grid))


def _side_grid_contaminated_near_collapsed_anchor(grid: FaceGrid) -> bool:
    return (
        grid.matched_count <= 7
        and _grid_bad_sample_count(grid) >= MAX_LOW_MATCH_GRID_BAD_SAMPLES
        and _grid_suspect_sample_score(grid) >= MAX_COLLAPSED_ANCHOR_SIDE_GRID_SUSPECT_SCORE
    )


def _grid_sample_penalty(grid: FaceGrid) -> float:
    return (
        _grid_sample_count(grid) * 2.5
        + _grid_suspect_sample_score(grid) * 18.0
        + _grid_extrapolation_penalty(grid)
    )


def _grid_sample_count(grid: FaceGrid) -> int:
    return sum(1 for row in getattr(grid, "stickers", []) for sticker in row if getattr(sticker, "source", "") == "grid_sample")


def _grid_suspect_sample_score(grid: FaceGrid) -> float:
    return sum(_suspect_grid_sample_score(sticker) for row in getattr(grid, "stickers", []) for sticker in row)


def _grid_unsupported_sample_score(grid: FaceGrid) -> float:
    return sum(_unsupported_grid_sample_score(sticker) for row in getattr(grid, "stickers", []) for sticker in row)


def _grid_extrapolated_sample_score(grid: FaceGrid) -> float:
    return sum(_extrapolated_grid_sample_score(sticker) for row in getattr(grid, "stickers", []) for sticker in row)


def _grid_extrapolated_sample_count(grid: FaceGrid) -> int:
    return sum(
        1
        for row in getattr(grid, "stickers", [])
        for sticker in row
        if _extrapolated_grid_sample_score(sticker) > 0.0
    )


def _grid_unsupported_sample_count(grid: FaceGrid) -> int:
    return sum(
        1
        for row in getattr(grid, "stickers", [])
        for sticker in row
        if _unsupported_grid_sample_score(sticker) > 0.0
    )


def _grid_bad_sample_count(grid: FaceGrid) -> int:
    return sum(
        1
        for row in getattr(grid, "stickers", [])
        for sticker in row
        if _suspect_grid_sample_score(sticker) >= SUSPECT_GRID_SAMPLE_THRESHOLD
    )


def _grid_extrapolation_penalty(grid: FaceGrid) -> float:
    if (
        getattr(grid, "matched_count", 0) > MAX_GRID_EXTRAPOLATION_MATCHED_COUNT
        or _grid_sample_count(grid) < MIN_GRID_EXTRAPOLATION_GRID_SAMPLES
        or _grid_unsupported_sample_count(grid) < MIN_GRID_EXTRAPOLATION_UNSUPPORTED_SAMPLES
    ):
        return 0.0
    unsupported_score = _grid_unsupported_sample_score(grid)
    if unsupported_score <= MIN_GRID_EXTRAPOLATION_UNSUPPORTED_SCORE:
        return 0.0
    return min(
        MAX_GRID_EXTRAPOLATION_PENALTY,
        (unsupported_score - MIN_GRID_EXTRAPOLATION_UNSUPPORTED_SCORE) * GRID_EXTRAPOLATION_PENALTY_SCALE,
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


def _unsupported_grid_sample_score(sticker: Any) -> float:
    if getattr(sticker, "source", "") != "grid_sample":
        return 0.0
    color = sticker.match.color
    if color != "white":
        return 0.0

    spacing = float(getattr(sticker, "grid_spacing", 0.0) or 0.0)
    if spacing <= 1e-6:
        return 0.0

    score = 0.0
    outside_distance = float(
        getattr(
            sticker,
            "outside_grid_component_hull_distance",
            getattr(sticker, "outside_component_hull_distance", 0.0),
        )
        or 0.0
    )
    outside_ratio = outside_distance / spacing
    if outside_ratio > 0.45:
        score += min(3.0, (outside_ratio - 0.45) * 5.0)

    nearest_distance = float(
        getattr(
            sticker,
            "nearest_grid_component_distance",
            getattr(sticker, "nearest_component_distance", 0.0),
        )
        or 0.0
    )
    nearest_ratio = nearest_distance / spacing
    if nearest_ratio > 1.35:
        score += min(1.8, (nearest_ratio - 1.35) * 2.6)
    return score


def _extrapolated_grid_sample_score(sticker: Any) -> float:
    if getattr(sticker, "source", "") != "grid_sample":
        return 0.0

    spacing = float(getattr(sticker, "grid_spacing", 0.0) or 0.0)
    if spacing <= 1e-6:
        return 0.0

    outside_distance = float(
        getattr(
            sticker,
            "outside_grid_component_hull_distance",
            getattr(sticker, "outside_component_hull_distance", 0.0),
        )
        or 0.0
    )
    nearest_distance = float(
        getattr(
            sticker,
            "nearest_grid_component_distance",
            getattr(sticker, "nearest_component_distance", 0.0),
        )
        or 0.0
    )

    score = 0.0
    outside_ratio = outside_distance / spacing
    if outside_ratio > 0.75:
        score += min(3.0, (outside_ratio - 0.75) * 4.0)

    nearest_ratio = nearest_distance / spacing
    if nearest_ratio > 1.75:
        score += min(2.0, (nearest_ratio - 1.75) * 2.4)
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
    grid_context_flex = {face: _grid_context_repair_score(grid_by_face[face]) for face in faces}
    transformed_matrices: Dict[str, Dict[str, List[List[Any]]]] = {}
    for face in faces:
        matrix = _grid_matrix_for_orientation(grid_by_face[face], flex=grid_context_flex[face])
        matrix[1][1] = face
        transformed_matrices[face] = {
            transform.name: transform.apply(matrix) for transform in transform_options[face]
        }
    ranked: List[Tuple[float, float, Dict[str, List[List[Any]]]]] = []
    for combo in product(*(transform_options[face] for face in faces)):
        oriented: Dict[str, List[List[Any]]] = {}
        sort_score = 0.0
        score = 0.0
        for face, transform in zip(faces, combo):
            transform_score = _transform_weighted_match_score(transform, requirements[face], requirement_weights[face])
            sort_score += transform_score
            score += transform_score
            oriented[face] = transformed_matrices[face][transform.name]
        plausibility = _visible_piece_plausibility_score(oriented)
        sort_score += plausibility
        score += plausibility
        ranked.append((sort_score, score, oriented))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(score, oriented) for _, score, oriented in ranked[:MAX_ORIENTED_OPTIONS_PER_SIDE_PAIR]]


def _grid_matrix_for_orientation(grid: FaceGrid, *, flex: Optional[float] = None) -> List[List[Any]]:
    resolved_flex = _grid_context_repair_score(grid) if flex is None else flex
    if resolved_flex < GRID_CONTEXT_REPAIR_THRESHOLD:
        return [list(row) for row in grid.stickers]
    return [
        [_grid_contextual_facelet(sticker, grid, resolved_flex) for sticker in row]
        for row in grid.stickers
    ]


def _grid_contextual_facelet(sticker: Any, grid: FaceGrid, flex: float) -> Any:
    if flex < GRID_CONTEXT_REPAIR_THRESHOLD or getattr(sticker, "source", "component") != "component":
        return sticker
    if isinstance(sticker, Sticker):
        contextual = Sticker(
            id=sticker.id,
            center=sticker.center,
            bbox=sticker.bbox,
            rgb=sticker.rgb,
            match=sticker.match,
            area=sticker.area,
            source=sticker.source,
            shape_angle=sticker.shape_angle,
        )
    else:
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
    for coords in EDGE_FACELET_COORDS:
        colors = _visible_piece_colors_for_coords(faces, coords)
        if colors is None:
            continue
        if len(set(colors)) != len(colors):
            score -= 8.0
        elif frozenset(colors) in VALID_EDGE_COLOR_SETS:
            score += 9.0
        else:
            score -= 14.0

    for coords in CORNER_FACELET_COORDS:
        colors = _visible_piece_colors_for_coords(faces, coords)
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
    coords = tuple((FACE_ORDER[index // 9], (index % 9) // 3, (index % 9) % 3) for index in indices)
    return _visible_piece_colors_for_coords(faces, coords)


def _visible_piece_colors_for_coords(
    faces: Dict[str, List[List[Any]]],
    coords: Sequence[Tuple[str, int, int]],
) -> Optional[Tuple[str, ...]]:
    colors = []
    for face, row, col in coords:
        matrix = faces.get(face)
        if matrix is None:
            return None
        color = _primary_facelet_color(matrix[row][col])
        if color not in FACE_ORDER:
            return None
        colors.append(color)
    return tuple(colors)


def _piece_conflict_summary(faces: Dict[str, List[List[Any]]]) -> Dict[str, int]:
    summary = Counter()
    corner_sets: Counter[frozenset[str]] = Counter()
    edge_sets: Counter[frozenset[str]] = Counter()

    for coords in CORNER_FACELET_COORDS:
        colors = _visible_piece_colors_for_coords(faces, coords)
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

    for coords in EDGE_FACELET_COORDS:
        colors = _visible_piece_colors_for_coords(faces, coords)
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
    face_counts: Optional[Dict[str, int]] = None,
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
    resolved_face_counts = face_counts if face_counts is not None else _primary_face_counts(faces)
    face_count_penalty = 0.003 * _face_count_deviation(resolved_face_counts)
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
    return face if isinstance(face, str) and face in FACE_ORDER else None


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
    assigned_grid_ids = {_grid_identity(grid) for grid in assigned.values()}
    for face, grids in groups.items():
        if face == anchor or face not in {"R", "F", "L", "B"}:
            continue
        current = assigned.get(face)
        best = next((grid for grid in grids if current is grid or _grid_identity(grid) not in assigned_grid_ids), None)
        if best is None:
            continue
        if current is None or _grid_quality_score(best) > _grid_quality_score(current):
            if current is not None:
                assigned_grid_ids.discard(_grid_identity(current))
            assigned[face] = best
            assigned_grid_ids.add(_grid_identity(best))
    return assigned


def _grid_identity(grid: FaceGrid) -> Any:
    return getattr(grid, "id", id(grid))


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
        candidates.extend(grid for grid in grids if _u_logo_anchor_grid_candidate(grid))
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
    return (
        grid.matched_count * 18.0
        - grid.fit_error * 1.2
        - min(65.0, _grid_shape_spread(grid)) * 1.2
        - _grid_sample_penalty(grid)
        - _grid_cube_hull_penalty(grid)
    )


def _grid_cube_hull_penalty(grid: FaceGrid) -> float:
    penalty = cube_hull_grid_penalty(getattr(grid, "cube_hull_inside_count", None))
    if penalty <= 0.0:
        return 0.0
    if _grid_extrapolation_penalty(grid) < 5.0:
        return 0.0
    return penalty


def _center_sample_is_whiteish(grid: FaceGrid) -> bool:
    _, saturation, value = rgb_to_hsv(grid.center_sticker.rgb)
    return value >= 0.62 and saturation <= 0.34


def _center_sample_is_yellowish(grid: FaceGrid) -> bool:
    hue, saturation, value = rgb_to_hsv(grid.center_sticker.rgb)
    return 0.10 <= hue <= 0.22 and saturation >= 0.22 and value >= 0.48


def _u_logo_anchor_grid_candidate(grid: FaceGrid) -> bool:
    if grid.center_face == "U" or _center_sample_is_whiteish(grid):
        return False
    if grid.matched_count < MIN_U_LOGO_ANCHOR_MATCHED_COUNT or grid.fit_error > MAX_U_LOGO_ANCHOR_FIT_ERROR:
        return False
    if _grid_sample_count(grid) > MAX_U_LOGO_ANCHOR_GRID_SAMPLES or _grid_bad_sample_count(grid) > 0:
        return False
    if _grid_quality_score(grid) < MIN_U_LOGO_ANCHOR_QUALITY:
        return False

    center = grid.center_sticker
    match = getattr(center, "match", None) or classify_rgb(center.rgb)
    if getattr(match, "confidence", 1.0) > MAX_U_LOGO_ANCHOR_CENTER_CONFIDENCE:
        return False

    alternatives = list(getattr(match, "alternatives", ()) or ())
    baseline = alternatives[0][1] if alternatives else getattr(match, "distance", 0.0)
    white_distance = next((distance for color, distance in alternatives[:4] if color == "white"), None)
    return white_distance is not None and white_distance - baseline <= MAX_U_LOGO_ANCHOR_WHITE_DISTANCE_DELTA


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
    if _balanced_color_assignment_scoring_enabled():
        counts_a = a.get("_visible_color_counts")
        counts_b = b.get("_visible_color_counts")
        if isinstance(counts_a, dict) and isinstance(counts_b, dict):
            color_counts = {face: int(counts_a.get(face, 0)) + int(counts_b.get(face, 0)) for face in FACE_ORDER}
            balanced_assignment = _balanced_color_assignment_scoring_summary_from_counts(color_counts)
        else:
            balanced_assignment = _balanced_color_assignment_scoring_summary(merged)
        score_penalty = _balanced_color_assignment_score_penalty(balanced_assignment)
        if score_penalty > 0.0:
            merged["_score"] = float(merged.get("_score", 0.0)) - score_penalty
            merged["_balanced_color_assignment_score_penalty"] = round(score_penalty, 4)
        if score_penalty > 0.0 or balanced_assignment.get("status") != "balanced":
            merged["_balanced_color_assignment"] = balanced_assignment
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


def _cached_face_signature(faces: Dict[str, List[List[Any]]]) -> FaceSignature:
    cached = faces.get("_face_signature")
    if isinstance(cached, tuple):
        return cached
    signature = _face_signature(faces)
    faces["_face_signature"] = signature  # type: ignore[assignment]
    return signature


def _state_variants_from_faces(
    faces: Dict[str, List[List[Any]]],
    *,
    facelet_options_cache: Optional[Dict[FaceletOptionsKey, List[Tuple[str, float]]]] = None,
) -> List[str]:
    return [
        state
        for _, state in _state_variants_from_faces_with_costs(
            faces,
            facelet_options_cache=facelet_options_cache,
        )
    ]


def _state_variants_from_faces_with_costs(
    faces: Dict[str, List[List[Any]]],
    *,
    facelet_options_cache: Optional[Dict[FaceletOptionsKey, List[Tuple[str, float]]]] = None,
) -> List[Tuple[float, str]]:
    facelets = []
    for face in FACE_ORDER:
        matrix = faces.get(face)
        if not matrix:
            return []
        facelets.extend(matrix[r][c] for r in range(3) for c in range(3))
    return _balanced_state_variants_with_costs(facelets, facelet_options_cache=facelet_options_cache)


def _legal_repaired_state_from_faces(faces: Dict[str, List[List[Any]]]) -> Optional[Tuple[str, float, int]]:
    facelets = _facelets_from_faces(faces)
    if facelets is None:
        return None

    corner_options = [_corner_piece_options([facelets[index] for index in indices]) for indices in CORNER_FACELETS]
    edge_options = [_edge_piece_options([facelets[index] for index in indices]) for indices in EDGE_FACELETS]
    if any(not options for options in corner_options) or any(not options for options in edge_options):
        return None

    corners = _top_piece_solutions(corner_options, piece_count=8, orientation_mod=3)
    if not corners:
        return None

    min_corner_changes = min(changes for _, changes, _, _ in corners)
    min_corner_cost = min(cost for cost, _, _, _ in corners)
    edges = _top_piece_solutions(
        edge_options,
        piece_count=12,
        orientation_mod=2,
        max_changes=MAX_LEGAL_REPAIR_CHANGES - min_corner_changes,
        max_cost=MAX_LEGAL_REPAIR_COST - min_corner_cost,
    )
    if not edges:
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
    return CORNER_ASSIGNMENTS.get(colors)


def _edge_assignment(colors: Tuple[str, ...]) -> Optional[Tuple[int, int]]:
    return EDGE_ASSIGNMENTS.get(colors)


def _top_piece_solutions(
    options_by_position: Sequence[Sequence[PieceOption]],
    piece_count: int,
    orientation_mod: int,
    *,
    max_changes: int = MAX_LEGAL_REPAIR_CHANGES,
    max_cost: float = MAX_LEGAL_REPAIR_COST,
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
                    next_cost = cost + option.cost
                    if next_cost > max_cost:
                        continue
                    next_changes = changes + option.changes
                    if next_changes > max_changes:
                        continue
                    previous_greater = (mask >> (option.cubie + 1)).bit_count()
                    next_key = (
                        mask | bit,
                        (orientation_sum + option.orientation) % orientation_mod,
                        parity ^ (previous_greater % 2),
                    )
                    bucket = next_dp.setdefault(next_key, [])
                    bucket.append((next_cost, next_changes, selected + (option,)))
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


def _balanced_state_variants(
    facelets: Sequence[Any],
    *,
    facelet_options_cache: Optional[Dict[FaceletOptionsKey, List[Tuple[str, float]]]] = None,
) -> List[str]:
    return [
        state
        for _, state in _balanced_state_variants_with_costs(
            facelets,
            facelet_options_cache=facelet_options_cache,
        )
    ]


def _balanced_state_variants_with_costs(
    facelets: Sequence[Any],
    *,
    facelet_options_cache: Optional[Dict[FaceletOptionsKey, List[Tuple[str, float]]]] = None,
) -> List[Tuple[float, str]]:
    options = [_cached_facelet_options(facelet, facelet_options_cache) for facelet in facelets]
    current = [choices[0][0] for choices in options]
    counts = Counter(current)
    current_state = "".join(current)
    if all(counts[face] == 9 for face in FACE_ORDER):
        return [(0.0, current_state)]

    surplus = {face: counts[face] - 9 for face in FACE_ORDER if counts[face] > 9}
    deficits = {face: 9 - counts[face] for face in FACE_ORDER if counts[face] < 9}
    if sum(deficits.values()) > MAX_COLOR_REPAIR_CHANGES:
        return [(0.0, current_state)]

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
        return [(0.0, current_state)]
    variants.sort(key=lambda item: item[0])
    unique = []
    seen = set()
    for cost, state in variants:
        if state not in seen:
            unique.append((cost, state))
            seen.add(state)
    return unique[:MAX_COLOR_REPAIR_VARIANTS]


def _cached_facelet_options(
    facelet: Any,
    cache: Optional[Dict[FaceletOptionsKey, List[Tuple[str, float]]]],
) -> List[Tuple[str, float]]:
    if cache is None:
        return _facelet_options(facelet)
    key = _facelet_options_cache_key(facelet)
    cached = cache.get(key)
    if cached is None:
        cached = _facelet_options(facelet)
        cache[key] = cached
    return cached


def _facelet_options_cache_key(facelet: Any) -> FaceletOptionsKey:
    if isinstance(facelet, str):
        return ("str", facelet)
    return ("obj", id(facelet))


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
