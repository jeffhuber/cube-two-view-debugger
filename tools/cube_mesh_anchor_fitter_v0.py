#!/usr/bin/env python3
"""Weak-projection 7-anchor cube mesh diagnostics.

Diagnostics-only. This module does not alter recognizer behavior.

The anchor mesh is the next first-principles step after vertex-point ranking:
given a candidate visible trihedral vertex, fit one coherent projected cube
mesh with seven named anchors:

    V, X, Y, Z, XY, YZ, ZX

where V is the visible trihedral corner, X/Y/Z are one edge away along the
three visible cube axes, and XY/YZ/ZX are the three opposite corners of the
visible face parallelograms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.global_cube_model_v0 import ProjectedCubeModel, _round_point
from tools.interior_bezel_detection import InteriorBezelDetection
from tools.vertex_point_candidates import (
    VertexPointCandidate,
    VertexPointCandidateResult,
    rank_vertex_point_candidates,
    serialize_vertex_candidate,
)


Point = Tuple[float, float]

ANCHOR_NAMES = ("V", "X", "Y", "Z", "XY", "YZ", "ZX")


@dataclass(frozen=True)
class CubeMeshAnchorFit:
    """One coherent 7-anchor mesh derived from a vertex candidate."""

    source_vertex_rank: int
    source_vertex_candidate: VertexPointCandidate
    anchors: Dict[str, Point]
    score: float
    status: str
    score_components: Dict[str, float]
    warnings: Tuple[str, ...]

    @property
    def model(self) -> ProjectedCubeModel:
        return self.source_vertex_candidate.model


@dataclass(frozen=True)
class CubeMeshAnchorResult:
    """Ranked 7-anchor mesh fits for one image."""

    meshes: Tuple[CubeMeshAnchorFit, ...]
    status: str
    diagnostics: Dict[str, Any]
    vertex_result: VertexPointCandidateResult

    @property
    def top_mesh(self) -> Optional[CubeMeshAnchorFit]:
        return self.meshes[0] if self.meshes else None


def fit_cube_mesh_anchor_candidates(
    detection: InteriorBezelDetection,
    silhouette_mask: np.ndarray,
    *,
    edge_steps: int = 32,
    center_offsets: Optional[Sequence[Tuple[float, float]]] = None,
    top_vertex_candidates: int = 5,
    top_meshes: int = 5,
    anchor_near_radius_px: int = 8,
) -> CubeMeshAnchorResult:
    """Fit and rank weak-projection 7-anchor meshes from vertex candidates."""
    vertex_kwargs: Dict[str, Any] = {
        "edge_steps": edge_steps,
        "top_n": top_vertex_candidates,
    }
    if center_offsets is not None:
        vertex_kwargs["center_offsets"] = center_offsets
    vertex_result = rank_vertex_point_candidates(detection, silhouette_mask, **vertex_kwargs)
    if not vertex_result.candidates:
        return CubeMeshAnchorResult(
            (),
            vertex_result.status,
            {
                "probeVersion": "cube-mesh-anchor-v0",
                "fitMethod": "weak_projected_7_anchor_v0",
                "vertexStatus": vertex_result.status,
                "error": "no vertex candidates",
            },
            vertex_result,
        )

    fits: List[CubeMeshAnchorFit] = []
    for rank, candidate in enumerate(vertex_result.candidates, start=1):
        fits.append(_fit_from_vertex_candidate(
            candidate,
            rank,
            silhouette_mask,
            anchor_near_radius_px=anchor_near_radius_px,
        ))

    ranked = tuple(sorted(
        fits,
        key=lambda fit: (
            fit.status == "ok",
            fit.score,
        ),
        reverse=True,
    )[:top_meshes])
    top = ranked[0]
    return CubeMeshAnchorResult(
        ranked,
        top.status,
        {
            "probeVersion": "cube-mesh-anchor-v0",
            "fitMethod": "weak_projected_7_anchor_v0",
            "pnpStatus": "not_run_no_calibrated_camera_or_human_anchor_labels",
            "selectionPolicy": "prefer_ok_then_anchor_mesh_score",
            "vertexCandidateCount": len(vertex_result.candidates),
            "returnedMeshCount": len(ranked),
            "topSourceVertexRank": top.source_vertex_rank,
            "topSourceVertexSource": top.source_vertex_candidate.source,
            "topVertexPoint": _round_point(top.source_vertex_candidate.vertex_point),
            "anchorNearRadiusPx": anchor_near_radius_px,
        },
        vertex_result,
    )


def serialize_anchor_mesh(fit: CubeMeshAnchorFit, rank: int) -> Dict[str, Any]:
    """Serialize one mesh fit for committed diagnostics fixtures."""
    return {
        "rank": rank,
        "fitMethod": "weak_projected_7_anchor_v0",
        "sourceVertexRank": fit.source_vertex_rank,
        "sourceVertexCandidate": serialize_vertex_candidate(
            fit.source_vertex_candidate,
            fit.source_vertex_rank,
        ),
        "status": fit.status,
        "score": round(float(fit.score), 4),
        "scoreComponents": {
            key: round(float(value), 4)
            for key, value in sorted(fit.score_components.items())
        },
        "warnings": list(fit.warnings),
        "anchors": {
            name: _round_point(fit.anchors[name])
            for name in ANCHOR_NAMES
        },
    }


def _fit_from_vertex_candidate(
    candidate: VertexPointCandidate,
    source_vertex_rank: int,
    silhouette_mask: np.ndarray,
    *,
    anchor_near_radius_px: int,
) -> CubeMeshAnchorFit:
    anchors = _anchors_from_model(candidate.model)
    components = _score_anchor_mesh(
        candidate.model,
        anchors,
        silhouette_mask,
        anchor_near_radius_px=anchor_near_radius_px,
    )
    score = (
        candidate.model.score
        + components["anchorNearSilhouetteRatio"] * 0.35
        + components["faceAreaBalance"] * 0.20
        + components["axisAngleSeparationScore"] * 0.10
        - components["outsideImagePenalty"] * 0.25
    )
    status = _status_from_anchor_components(candidate.model_status, components)
    warnings: List[str] = []
    if components["anchorNearSilhouetteRatio"] < 0.86:
        warnings.append("low_anchor_silhouette_support")
    if components["faceAreaBalance"] < 0.15:
        warnings.append("unbalanced_face_areas")
    if components["axisAngleSeparationScore"] < 0.35:
        warnings.append("weak_axis_angle_separation")
    return CubeMeshAnchorFit(
        source_vertex_rank=source_vertex_rank,
        source_vertex_candidate=candidate,
        anchors=anchors,
        score=float(score),
        status=status,
        score_components=components,
        warnings=tuple(warnings),
    )


def _anchors_from_model(model: ProjectedCubeModel) -> Dict[str, Point]:
    v = model.cube_center
    axes = model.axes
    x = _add(v, axes[0])
    y = _add(v, axes[1])
    z = _add(v, axes[2])
    xy = _add(v, _add_vec(axes[0], axes[1]))
    yz = _add(v, _add_vec(axes[1], axes[2]))
    zx = _add(v, _add_vec(axes[2], axes[0]))
    return {
        "V": v,
        "X": x,
        "Y": y,
        "Z": z,
        "XY": xy,
        "YZ": yz,
        "ZX": zx,
    }


def _score_anchor_mesh(
    model: ProjectedCubeModel,
    anchors: Dict[str, Point],
    silhouette_mask: np.ndarray,
    *,
    anchor_near_radius_px: int,
) -> Dict[str, float]:
    mask = silhouette_mask.astype(bool)
    anchor_points = [anchors[name] for name in ANCHOR_NAMES]
    near_count = sum(
        1
        for point in anchor_points
        if _point_near_mask(point, mask, radius=anchor_near_radius_px)
    )
    face_areas = [abs(_polygon_area(face.quad)) for face in model.faces]
    min_area = min(face_areas) if face_areas else 0.0
    max_area = max(face_areas) if face_areas else 0.0
    axis_angles = [_axis_angle(axis) for axis in model.axes]
    model_components = model.score_components
    return {
        "modelScore": float(model.score),
        "silhouetteIoU": float(model_components.get("silhouetteIoU", 0.0)),
        "insideRatio": float(model_components.get("insideRatio", 0.0)),
        "maskCoverage": float(model_components.get("maskCoverage", 0.0)),
        "cellInsideRatio": float(model_components.get("cellInsideRatio", 0.0)),
        "detectorSignalQuality": float(model_components.get("detectorSignalQuality", 0.0)),
        "outsideImagePenalty": float(model_components.get("outsideImagePenalty", 0.0)),
        "anchorNearSilhouetteRatio": near_count / max(1, len(anchor_points)),
        "faceAreaBalance": min_area / max(1.0, max_area),
        "minAnchorSeparationPx": _min_pairwise_distance(anchor_points),
        "axisAngleSeparationScore": _axis_angle_separation_score(axis_angles),
    }


def _status_from_anchor_components(model_status: str, components: Dict[str, float]) -> str:
    if model_status != "ok":
        return model_status
    if components.get("faceAreaBalance", 0.0) < 0.15:
        return "unbalanced_face_area"
    if components.get("axisAngleSeparationScore", 0.0) < 0.35:
        return "weak_axis_separation"
    return "ok"


def _point_near_mask(point: Point, mask: np.ndarray, *, radius: int) -> bool:
    height, width = mask.shape
    x, y = point
    ix = int(round(x))
    iy = int(round(y))
    x0 = max(0, ix - radius)
    x1 = min(width, ix + radius + 1)
    y0 = max(0, iy - radius)
    y1 = min(height, iy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(mask[y0:y1, x0:x1].any())


def _axis_angle(axis: Point) -> float:
    return math.atan2(axis[1], axis[0])


def _axis_angle_separation_score(angles: Sequence[float]) -> float:
    if len(angles) < 3:
        return 0.0
    separations: List[float] = []
    for idx, angle in enumerate(angles):
        for other in angles[idx + 1:]:
            sep = abs((angle - other + math.pi) % (2 * math.pi) - math.pi)
            sep = min(sep, math.pi - sep)
            separations.append(sep)
    if not separations:
        return 0.0
    min_sep_deg = min(separations) * 180.0 / math.pi
    return max(0.0, min(1.0, min_sep_deg / 35.0))


def _polygon_area(points: Sequence[Point]) -> float:
    total = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def _min_pairwise_distance(points: Sequence[Point]) -> float:
    best: Optional[float] = None
    for idx, left in enumerate(points):
        for right in points[idx + 1:]:
            distance = math.hypot(left[0] - right[0], left[1] - right[1])
            if best is None or distance < best:
                best = distance
    return float(best or 0.0)


def _add(left: Point, right: Point) -> Point:
    return (float(left[0] + right[0]), float(left[1] + right[1]))


def _add_vec(left: Point, right: Point) -> Point:
    return (float(left[0] + right[0]), float(left[1] + right[1]))
