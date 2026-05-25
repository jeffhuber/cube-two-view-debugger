"""Acceptance gates for the hull-label rectification candidate.

This module is production-shaped but does not wire production behavior. It
captures the checks a feature-flagged caller should run before trusting
``tools.rectify_via_hull_labels`` over the older Procrustes/PnP path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from tools.corner_conventions import FACE_DEFS_BY_SIDE
from tools.rectify_via_hull_labels import SILHOUETTE_TO_CORNER


Point = Tuple[float, float]

SILHOUETTE_POSITION_NAMES = {
    "top",
    "upper_right",
    "lower_right",
    "bottom",
    "lower_left",
    "upper_left",
}
EXPECTED_FACE_SLOTS = {"upper", "right", "front"}


@dataclass(frozen=True)
class HullLabelGateThresholds:
    """Initial production acceptance thresholds.

    Hard thresholds mean "fallback to the old path." Warning thresholds mean
    "usable for shadow/telemetry, but do not graduate to default-on without
    inspecting the row or adding a stronger pair-level gate."

    The defaults are deliberately above the observed 70-row corpus maxima for
    hard failures (see ``tools/HULL_LABELS_CORPUS_REPORT.md``) while preserving
    warning bands near the edge of that corpus.
    """

    max_vertex_cloud_spread_px: float = 300.0
    warn_vertex_cloud_spread_px: float = 240.0
    max_sticker_score_total: float = 900.0
    warn_sticker_score_total: float = 700.0
    max_sticker_score_per_face: float = 450.0
    warn_sticker_score_per_face: float = 350.0
    # Projective residual gate — surfaces bad-hull-input rows that
    # don't trip the spread/sticker gates. Per the 70-row probe in
    # `tools/PROJECTIVE_VERTEX_REPORT.md`, residual_norm > 0.025 is a
    # strong bad-input signal (30_A's 0.0315 is the corpus max,
    # correctly flagging the wall-edge artifact case that everyone
    # else's gates missed). `warn` at 0.020 catches one more
    # borderline row (30_B at 0.0199) for shadow-mode review.
    max_projective_residual_norm: float = 0.025
    warn_projective_residual_norm: float = 0.020


DEFAULT_THRESHOLDS = HullLabelGateThresholds()


@dataclass(frozen=True)
class HullLabelGateDecision:
    accepted: bool
    hard_failures: Tuple[str, ...]
    warnings: Tuple[str, ...]
    metrics: Mapping[str, float]

    @property
    def should_fallback(self) -> bool:
        return not self.accepted


def vertex_cloud_spread(vertex_estimates: Sequence[Point]) -> float:
    """Max pairwise distance across the 3 parallelogram vertex estimates."""
    for x, y in vertex_estimates:
        if not math.isfinite(x) or not math.isfinite(y):
            return math.inf
    if len(vertex_estimates) < 2:
        return 0.0
    spread = 0.0
    for i in range(len(vertex_estimates)):
        for j in range(i + 1, len(vertex_estimates)):
            spread = max(
                spread,
                math.hypot(
                    vertex_estimates[i][0] - vertex_estimates[j][0],
                    vertex_estimates[i][1] - vertex_estimates[j][1],
                ),
            )
    return spread


def side_convention_errors(side: str) -> Tuple[str, ...]:
    """Validate that the side has the convention data needed by hull labels."""
    errors = []
    mapping = SILHOUETTE_TO_CORNER.get(side)
    if mapping is None:
        errors.append(f"unsupported side {side!r}; no SILHOUETTE_TO_CORNER entry")
    else:
        if set(mapping) != SILHOUETTE_POSITION_NAMES:
            errors.append(f"side {side!r} silhouette positions are incomplete")
        if set(mapping.values()) != set(range(6)):
            errors.append(f"side {side!r} corner mapping is not a 0..5 bijection")

    face_defs = FACE_DEFS_BY_SIDE.get(side)
    if face_defs is None:
        errors.append(f"unsupported side {side!r}; no FACE_DEFS_BY_SIDE entry")
        return tuple(errors)
    if set(face_defs) != EXPECTED_FACE_SLOTS:
        errors.append(f"side {side!r} face slots are not upper/right/front")

    for slot, names in face_defs.items():
        if len(names) != 4 or names[0] != "vertex":
            errors.append(f"side {side!r} face {slot!r} is not a vertex+3-corner quad")
            continue
        corners = []
        for name in names[1:]:
            if not name.startswith("corner_"):
                errors.append(f"side {side!r} face {slot!r} has invalid ref {name!r}")
                continue
            try:
                idx = int(name.split("_", 1)[1])
            except ValueError:
                errors.append(f"side {side!r} face {slot!r} has invalid ref {name!r}")
                continue
            corners.append(idx)
        if len(set(corners)) != 3 or any(idx not in range(6) for idx in corners):
            errors.append(f"side {side!r} face {slot!r} does not use 3 unique corners")

    return tuple(errors)


def evaluate_hull_label_acceptance(
    *,
    side: str,
    hexagon_corner_count: int,
    vertex_estimates: Sequence[Point],
    rectified_face_slots: Iterable[str],
    sticker_score_total: float,
    sticker_score_per_face: Mapping[str, float],
    projective_residual_norm: Optional[float] = None,
    thresholds: HullLabelGateThresholds = DEFAULT_THRESHOLDS,
) -> HullLabelGateDecision:
    """Evaluate production-available gates for one hull-label fit.

    This intentionally excludes ground-truth-only metrics such as axis misfit.
    Feature-flagged production wiring should fallback on any hard failure, and
    should record warnings for shadow-mode and later threshold tuning.
    """
    hard = list(side_convention_errors(side))
    warnings = []

    if hexagon_corner_count != 6:
        hard.append(f"hexagon_corner_count={hexagon_corner_count}; expected 6")

    if len(vertex_estimates) != 3:
        hard.append(f"vertex_estimate_count={len(vertex_estimates)}; expected 3")

    slots = set(rectified_face_slots)
    if slots != EXPECTED_FACE_SLOTS:
        hard.append(f"rectified_face_slots={sorted(slots)}; expected upper/right/front")

    missing_scores = EXPECTED_FACE_SLOTS - set(sticker_score_per_face)
    if missing_scores:
        hard.append(f"missing per-face sticker scores: {sorted(missing_scores)}")

    spread = vertex_cloud_spread(vertex_estimates)
    if sticker_score_per_face and all(
        math.isfinite(score) for score in sticker_score_per_face.values()
    ):
        worst_face_score = max(sticker_score_per_face.values())
    else:
        worst_face_score = math.inf
    metrics = {
        "vertex_cloud_spread_px": round(spread, 3),
        "sticker_score_total": round(float(sticker_score_total), 3),
        "sticker_score_worst_face": round(float(worst_face_score), 3),
    }

    if not math.isfinite(spread) or spread > thresholds.max_vertex_cloud_spread_px:
        hard.append(
            "vertex_cloud_spread_px="
            f"{spread:.1f}; max {thresholds.max_vertex_cloud_spread_px:.1f}"
        )
    elif spread > thresholds.warn_vertex_cloud_spread_px:
        warnings.append(
            "vertex_cloud_spread_px="
            f"{spread:.1f}; warning {thresholds.warn_vertex_cloud_spread_px:.1f}"
        )

    if (
        not math.isfinite(sticker_score_total)
        or sticker_score_total > thresholds.max_sticker_score_total
    ):
        hard.append(
            "sticker_score_total="
            f"{sticker_score_total:.1f}; max {thresholds.max_sticker_score_total:.1f}"
        )
    elif sticker_score_total > thresholds.warn_sticker_score_total:
        warnings.append(
            "sticker_score_total="
            f"{sticker_score_total:.1f}; warning {thresholds.warn_sticker_score_total:.1f}"
        )

    if (
        not math.isfinite(worst_face_score)
        or worst_face_score > thresholds.max_sticker_score_per_face
    ):
        hard.append(
            "sticker_score_worst_face="
            f"{worst_face_score:.1f}; max {thresholds.max_sticker_score_per_face:.1f}"
        )
    elif worst_face_score > thresholds.warn_sticker_score_per_face:
        warnings.append(
            "sticker_score_worst_face="
            f"{worst_face_score:.1f}; warning "
            f"{thresholds.warn_sticker_score_per_face:.1f}"
        )

    # Projective residual gate — bad-input-hull detector. When
    # provided by the caller (computed in
    # `tools/rectify_via_hull_labels.py::_choose_hybrid_vertex`),
    # surfaces rows where the 3 vanishing-point lines don't meet
    # cleanly. Strongly correlates with the 30_A-class bad-hull
    # failure mode that spread/sticker gates miss.
    # See `tools/PROJECTIVE_VERTEX_REPORT.md`.
    if projective_residual_norm is not None:
        metrics["projective_residual_norm"] = round(
            float(projective_residual_norm), 4,
        )
        if (
            not math.isfinite(projective_residual_norm)
            or projective_residual_norm > thresholds.max_projective_residual_norm
        ):
            hard.append(
                "projective_residual_norm="
                f"{projective_residual_norm:.4f}; max "
                f"{thresholds.max_projective_residual_norm:.4f}"
            )
        elif projective_residual_norm > thresholds.warn_projective_residual_norm:
            warnings.append(
                "projective_residual_norm="
                f"{projective_residual_norm:.4f}; warning "
                f"{thresholds.warn_projective_residual_norm:.4f}"
            )

    return HullLabelGateDecision(
        accepted=not hard,
        hard_failures=tuple(hard),
        warnings=tuple(warnings),
        metrics=metrics,
    )
