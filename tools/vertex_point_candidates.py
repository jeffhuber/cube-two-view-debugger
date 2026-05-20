#!/usr/bin/env python3
"""Vertex-point candidate ranking for the projected cube model.

Diagnostics-only helper for the first-principles recognizer direction.

The "vertex point" is the visible trihedral corner where the three visible
cube faces meet. If this point is wrong, every downstream face quad and cell
sample is anchored to the wrong origin. This module does not alter recognizer
behavior; it only ranks candidate vertex points by fitting the same coherent
projected cube model used by ``global_cube_model_v0``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.global_cube_model_v0 import (
    CENTER_REFINEMENT_OFFSETS_PX,
    FitResult,
    ProjectedCubeModel,
    _edge_search_range,
    _model_from_axes,
    _round_point,
    _status_from_components,
    _unit_from_angle,
)
from tools.interior_bezel_detection import InteriorBezelDetection


Point = Tuple[float, float]


@dataclass(frozen=True)
class VertexPointCandidate:
    """One possible visible trihedral corner plus its best coherent model."""

    source: str
    vertex_point: Point
    offset_from_detector_center: Point
    model: ProjectedCubeModel
    model_status: str
    evaluated_models: int


@dataclass(frozen=True)
class VertexPointCandidateResult:
    """Ranked vertex-point candidates for one image."""

    candidates: Tuple[VertexPointCandidate, ...]
    status: str
    diagnostics: Dict[str, Any]

    @property
    def top_candidate(self) -> Optional[VertexPointCandidate]:
        return self.candidates[0] if self.candidates else None


def rank_vertex_point_candidates(
    detection: InteriorBezelDetection,
    silhouette_mask: np.ndarray,
    *,
    edge_steps: int = 32,
    center_offsets: Sequence[Tuple[float, float]] = CENTER_REFINEMENT_OFFSETS_PX,
    top_n: int = 5,
) -> VertexPointCandidateResult:
    """Rank plausible vertex points by coherent projected-model fit.

    Candidate sources are deliberately simple and explainable:

    * the interior-bezel detector's current center
    * bounded center-refinement offsets around that center
    * the raw silhouette centroid seed when available in detector debug data

    Ranking prefers candidates whose best model satisfies the existing V0
    thresholds, then uses the model score as a tiebreaker. This mirrors the
    V0.1 selection policy while preserving the top-N alternatives for human
    review.
    """
    if silhouette_mask.ndim != 2:
        return VertexPointCandidateResult((), "invalid_mask", {"error": "silhouette_mask must be 2-D"})
    if detection.cube_center is None:
        return VertexPointCandidateResult((), "missing_center", {"error": "interior bezel detector has no cube_center"})
    if len(detection.boundary_angles) < 3:
        return VertexPointCandidateResult((), "missing_axes", {"error": "need 3 detected boundary angles"})
    if int(silhouette_mask.sum()) <= 0:
        return VertexPointCandidateResult((), "empty_mask", {"error": "silhouette mask is empty"})
    if top_n <= 0:
        return VertexPointCandidateResult((), "invalid_top_n", {"error": "top_n must be positive"})

    detector_center = (float(detection.cube_center[0]), float(detection.cube_center[1]))
    base_units = [_unit_from_angle(angle) for angle in detection.boundary_angles[:3]]
    edge_min, edge_max = _edge_search_range(detector_center, silhouette_mask)
    if edge_max <= edge_min:
        return VertexPointCandidateResult((), "invalid_search_range", {"edgeMin": edge_min, "edgeMax": edge_max})

    candidate_points = _candidate_points(detector_center, detection, center_offsets)
    all_candidates: List[VertexPointCandidate] = []
    for source, point in candidate_points:
        model, evaluated = _best_model_for_vertex_point(
            point,
            base_units,
            edge_min,
            edge_max,
            edge_steps,
            silhouette_mask,
            detection,
        )
        if model is None:
            continue
        all_candidates.append(
            VertexPointCandidate(
                source=source,
                vertex_point=point,
                offset_from_detector_center=(
                    point[0] - detector_center[0],
                    point[1] - detector_center[1],
                ),
                model=model,
                model_status=_status_from_components(model.score_components),
                evaluated_models=evaluated,
            )
        )

    ranked = tuple(sorted(
        all_candidates,
        key=lambda candidate: (
            candidate.model_status == "ok",
            candidate.model.score,
        ),
        reverse=True,
    ))
    if not ranked:
        return VertexPointCandidateResult((), "no_candidates", {"candidatePointCount": len(candidate_points)})

    top = ranked[0]
    same_status_next = next(
        (candidate for candidate in ranked[1:] if candidate.model_status == top.model_status),
        None,
    )
    status = "ok" if top.model_status == "ok" else top.model_status
    diagnostics: Dict[str, Any] = {
        "probeVersion": "vertex-point-candidates-v0",
        "rankingPolicy": "prefer_ok_then_model_score",
        "candidatePointCount": len(candidate_points),
        "returnedCandidateCount": min(top_n, len(ranked)),
        "evaluatedModels": sum(candidate.evaluated_models for candidate in all_candidates),
        "edgeMin": round(edge_min, 2),
        "edgeMax": round(edge_max, 2),
        "detectorCenter": _round_point(detector_center),
        "detectorSignalQuality": round(float(detection.signal_quality), 4),
        "detectorLineQualities": [round(float(q), 4) for q in detection.line_qualities],
        "topModelStatus": top.model_status,
        "topSource": top.source,
        "topVertexPoint": _round_point(top.vertex_point),
        "topSameStatusScoreGap": (
            round(float(top.model.score - same_status_next.model.score), 4)
            if same_status_next is not None
            else None
        ),
    }
    return VertexPointCandidateResult(ranked[:top_n], status, diagnostics)


def serialize_vertex_candidate(candidate: VertexPointCandidate, rank: int) -> Dict[str, Any]:
    components = {
        key: round(float(value), 4)
        for key, value in sorted(candidate.model.score_components.items())
    }
    return {
        "rank": rank,
        "source": candidate.source,
        "vertexPoint": _round_point(candidate.vertex_point),
        "offsetFromDetectorCenter": _round_point(candidate.offset_from_detector_center),
        "modelStatus": candidate.model_status,
        "modelScore": round(float(candidate.model.score), 4),
        "scoreComponents": components,
        "edgeLength": round(float(candidate.model.edge_length), 3),
        "signChoice": list(candidate.model.sign_choice),
        "evaluatedModels": candidate.evaluated_models,
    }


def fit_result_from_vertex_candidate(candidate: VertexPointCandidate) -> FitResult:
    return FitResult(
        model=candidate.model,
        status=candidate.model_status,
        diagnostics={
            "fitVersion": "vertex-point-candidates-v0",
            "candidateSource": candidate.source,
            "vertexPoint": _round_point(candidate.vertex_point),
        },
    )


def _candidate_points(
    detector_center: Point,
    detection: InteriorBezelDetection,
    center_offsets: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[str, Point], ...]:
    seen = set()
    candidates: List[Tuple[str, Point]] = []

    def add(source: str, point: Point) -> None:
        key = (round(float(point[0]), 3), round(float(point[1]), 3))
        if key in seen:
            return
        seen.add(key)
        candidates.append((source, (float(point[0]), float(point[1]))))

    add("bezel_detector", detector_center)
    for dx, dy in center_offsets:
        point = (detector_center[0] + float(dx), detector_center[1] + float(dy))
        add("center_refine", point)

    centroid_seed = detection.debug.get("centroid_seed")
    if (
        isinstance(centroid_seed, list)
        and len(centroid_seed) == 2
        and all(isinstance(value, (int, float)) for value in centroid_seed)
    ):
        add("silhouette_centroid_seed", (float(centroid_seed[0]), float(centroid_seed[1])))

    return tuple(candidates)


def _best_model_for_vertex_point(
    vertex_point: Point,
    base_units: Sequence[Point],
    edge_min: float,
    edge_max: float,
    edge_steps: int,
    silhouette_mask: np.ndarray,
    detection: InteriorBezelDetection,
) -> Tuple[Optional[ProjectedCubeModel], int]:
    best: Optional[ProjectedCubeModel] = None
    evaluated = 0
    for signs in itertools.product((-1, 1), repeat=3):
        signed_units = [
            (sign * unit[0], sign * unit[1])
            for sign, unit in zip(signs, base_units)
        ]
        for edge_length in np.linspace(edge_min, edge_max, edge_steps):
            axes = tuple(
                (float(edge_length * unit[0]), float(edge_length * unit[1]))
                for unit in signed_units
            )
            model = _model_from_axes(
                vertex_point,
                axes,
                float(edge_length),
                tuple(int(sign) for sign in signs),
                silhouette_mask,
                detection,
            )
            evaluated += 1
            if best is None or model.score > best.score:
                best = model
    return best, evaluated
