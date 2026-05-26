#!/usr/bin/env python3
"""End-to-end rectification via hull-position corner labels + parallelogram
vertex derivation. Diagnostic-only candidate replacement for the production
`tools/global_cube_model.py` pipeline.

Origin: a first-principles question from the user (2026-05-24): given that
rembg + convex hull produces 6 silhouette extrema with stable physical
meaning (TOP, upper-right, lower-right, BOTTOM, lower-left, upper-left),
and given the cube's per-side corner-labeling convention is fixed in
`tools/corner_conventions.py` via `FACE_DEFS_BY_SIDE`, why does production
run a 720-perm Procrustes search + PnP refinement + chirality detector +
vertex ensemble + image-based vertex refinement when the entire mapping
is deterministic?

This tool answers that question with a 7-step implementation. The original
12-row full-corner seed produced essentially-oracle-quality rectifications
(per `tools/RECTIFY_VIA_HULL_LABELS_REPORT.md`); targeted tail labels such as
Sets 63-73 extend that corpus for failure analysis.

## Pipeline

1. rembg silhouette (production rembg path: `remove(image, session=sess)`
   → alpha channel → mask threshold selector)
2. `detect_hexagon_anchors(mask)` → 6 hull-extreme corners
3. Label corners by silhouette position (TOP, upper-right, lower-right,
   BOTTOM, lower-left, upper-left) via `_label_corners_by_position`
4. Map silhouette positions → canonical corner numbers via the per-side
   convention `SILHOUETTE_TO_CORNER` (derived from `FACE_DEFS_BY_SIDE`)
5. Derive vertex by parallelogram completion: for each of the 3 face
   quads, `vertex = NEAR_A + NEAR_B - FAR_AB`. Take the mean of 3
   estimates. Exact under iso projection; approximate under perspective.
6. For each face slot (upper/right/front), construct the face_quad
   `(vertex, NEAR_A, FAR_AB, NEAR_B)` per `FACE_DEFS_BY_SIDE`.
7. Rectify each face_quad via the existing `rectify_face` helper.

## What this eliminates

- The 720-perm Procrustes search in `fit_cube_template_to_anchors`
- PnP refinement
- Mean-of-3 vertex ensemble
- `_resolve_near_far_phase` chirality detector + 60° flip correction
- Image-based vertex refinement
- The bezel-detection dependency (`detect_interior_bezel_lines`)

## When this may fail

The approach assumes the cube is held roughly upright (white-up for
side A; white-down for side B) so the silhouette-position labeling is
stable. Strong tilt (>~30° from vertical) could shuffle which hull
extremum lands at TOP vs upper-right. The per-side mapping table needs
extension for sides other than A/B. Vertex derivation is exact under
iso projection; for cubes with strong perspective the parallelogram
closure constraint is only approximate and the derived vertex may drift
20-70 px from the true trihedral junction (still better than production
on the 12-row corpus, where bezel-detected vertex error ranges 43-241
px).

## CLI

  python tools/rectify_via_hull_labels.py

Defaults to the same canonical truth + manifest fixtures the other
diagnostics use; outputs trace to
`tests/fixtures/rectify_via_hull_labels_trace.json` and a gallery to
`/tmp/rectify_via_hull_labels/`.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import datetime as _dt

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import CLASSIFIER_CANONICAL, classify_rgb_with_mode  # noqa: E402
from tools.corner_conventions import FACE_DEFS_BY_SIDE  # noqa: E402
from tools.diagnose_pipeline_phase_parity import _processing_image  # noqa: E402
from tools.global_cube_model import detect_hexagon_anchors  # noqa: E402
from tools.rectify_faces import (  # noqa: E402
    extract_stickers_from_rectified, rectify_face,
)


Point = Tuple[float, float]

DEFAULT_TRUTH = REPO_ROOT / "tests" / "fixtures" / "full_corner_ground_truth.json"
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_TRACE = REPO_ROOT / "tests" / "fixtures" / "rectify_via_hull_labels_trace.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "RECTIFY_VIA_HULL_LABELS_REPORT.md"
DEFAULT_GALLERY_DIR = Path("/tmp/rectify_via_hull_labels")
DEFAULT_MAX_IMAGE_DIM = 1600
DEFAULT_FACE_SIZE_PX = 300
DEFAULT_MASK_THRESHOLDS = (64, 128, 160, 192, 224)


# Per-side silhouette-position → corner-number mapping. Derived from
# `tools/corner_conventions.py:FACE_DEFS_BY_SIDE` plus the canonical iso
# geometry of each side's view:
#
#   Side A (white-up view):
#     NEAR set = {1, 3, 5} per ONE_EDGE_CORNERS_BY_SIDE["A"]
#     The 3 NEAR cube edges from the front vertex go to upper-right,
#     BOTTOM, upper-left silhouette positions.
#     The 3 FAR corners (outer corners of the 3 visible faces) are at
#     TOP, lower-right, lower-left.
#     Upper face = (vertex, c1=UR, c0=TOP, c5=UL) — CCW around upper.
#
#   Side B (white-down view):
#     NEAR set = {0, 2, 4} per ONE_EDGE_CORNERS_BY_SIDE["B"]
#     The 3 NEAR cube edges from side B's vertex (which is the
#     OPPOSITE corner of the cube from side A's) go to the SAME 3
#     silhouette positions (upper-right, BOTTOM, upper-left) — the
#     cube silhouette is invariant under 180° body-diagonal rotation.
#     Different LABELS though: c0=BOTTOM, c2=upper-right, c4=upper-left.
#     Upper face = (vertex, c2=UR, c3=TOP, c4=UL) — CCW around upper.
SILHOUETTE_TO_CORNER: Dict[str, Dict[str, int]] = {
    "A": {
        "top": 0,
        "upper_right": 1,
        "lower_right": 2,
        "bottom": 3,
        "lower_left": 4,
        "upper_left": 5,
    },
    "B": {
        "top": 3,
        "upper_right": 2,
        "lower_right": 1,
        "bottom": 0,
        "lower_left": 5,
        "upper_left": 4,
    },
}


@dataclass
class RectifiedFit:
    """Output of `rectify_via_hull_labels` for one image+side."""
    side: str
    corners_by_num: Dict[int, Point]
    vertex: Point
    face_quads: Dict[str, List[Point]]
    rectified_faces: Dict[str, Image.Image]

    # Hybrid-vertex telemetry (always populated since 2026-05-25). The
    # active `vertex` above is whichever of affine/projective the
    # spread-gate selected; these fields surface BOTH candidates plus
    # the decision metadata so callers can route on the same signals
    # the gate used. Per the corpus probe (see
    # `tools/HULL_LABELS_CORPUS_REPORT.md` "Hybrid vertex strategy"),
    # the gate keeps median vertex_err *better* than pure affine while
    # also unlocking projective for the perspective-heavy minority.
    affine_vertex: Optional[Point] = None
    projective_vertex: Optional[Point] = None
    vertex_cloud_spread_px: Optional[float] = None
    """Max pairwise distance across the 3 affine parallelogram-
    completion vertex estimates, in raw pixels. Backward-compat with
    PR #285's `hull_label_acceptance.py` raw-px gates."""
    vertex_cloud_spread_norm: Optional[float] = None
    """``vertex_cloud_spread_px / hexagon_diameter_px``. Resolution-
    independent — the hybrid vertex switch gates on THIS field, not
    raw px, so behavior is stable across processing scales (Codex P2
    on #289 head 48f5a66)."""
    hexagon_diameter_px: Optional[float] = None
    """Longest pairwise distance among the 6 silhouette hexagon
    corners. The scale reference for all the normalized signals."""
    projective_residual_norm: Optional[float] = None
    """Projective 3-line LSQ residual ÷ hexagon diameter. Resolution-
    independent. High → projective fit is poorly conditioned (likely
    because hull corners themselves are wrong, e.g. 30_A's wall-edge
    artifact). Useful as a standalone bad-input gate in
    `tools/hull_label_acceptance.py`."""
    projective_degeneracy: Optional[str] = None
    """One of `finite_projective` / `near_affine` / `degenerate`."""
    vertex_source: Optional[str] = None
    """`"affine"` or `"projective"` — which candidate the spread-gate
    selected for the active `vertex`."""


@dataclass
class ThresholdFitSelection:
    """Result of trying multiple rembg alpha thresholds for one side."""

    fit: Optional[RectifiedFit]
    threshold: Optional[int]
    candidates: List[Dict[str, Any]]
    decision: Optional[Any]
    score: Optional[Dict[str, Any]]
    vertex_estimates: Optional[List[Point]]
    trace: Dict[str, Any]


def _label_corners_by_position(
    hexagon: Sequence[Point], side: str,
) -> Dict[int, Point]:
    """Sort 6 hexagon corners into silhouette positions, then map to
    canonical corner numbers per `SILHOUETTE_TO_CORNER[side]`.

    Position assignment:
      - TOP    = corner with smallest y (image y points DOWN)
      - BOTTOM = corner with largest y
      - The remaining 4 corners sort into left vs right by x; within
        each side, smaller y = upper, larger y = lower.

    Robust to small tilt of the cube. Assumes the cube isn't tilted
    >~30° from vertical (which would shuffle which hull extremum
    lands at each named position).
    """
    if len(hexagon) != 6:
        raise ValueError(f"need exactly 6 hexagon corners; got {len(hexagon)}")
    by_y = sorted(hexagon, key=lambda p: p[1])
    top = by_y[0]
    bottom = by_y[-1]
    middle_4 = sorted(by_y[1:-1], key=lambda p: p[0])
    upper_left = min(middle_4[:2], key=lambda p: p[1])
    lower_left = max(middle_4[:2], key=lambda p: p[1])
    upper_right = min(middle_4[2:], key=lambda p: p[1])
    lower_right = max(middle_4[2:], key=lambda p: p[1])
    positions = {
        "top": top,
        "upper_right": upper_right,
        "lower_right": lower_right,
        "bottom": bottom,
        "lower_left": lower_left,
        "upper_left": upper_left,
    }
    mapping = SILHOUETTE_TO_CORNER[side]
    return {mapping[pos_name]: point for pos_name, point in positions.items()}


def _derive_vertex_from_corners(
    corners_by_num: Dict[int, Point], side: str,
) -> Tuple[Point, List[Point]]:
    """Vertex via parallelogram completion: each visible face is a
    parallelogram in iso projection, so for face quad
    `(vertex, NEAR_A, FAR_AB, NEAR_B)`:

        FAR_AB = NEAR_A + NEAR_B - vertex
    →   vertex = NEAR_A + NEAR_B - FAR_AB

    Each of the 3 visible faces gives one vertex estimate. Take the
    mean. Returns (mean_vertex, [estimate_upper, estimate_right,
    estimate_front]).

    Exact under iso projection. Approximate under perspective — the
    3 estimates spread by ~10-30 px on real iPhone shots; the mean is
    typically within 20-70 px of the true vertex on the 12-row corpus.
    """
    estimates: List[Point] = []
    for _slot, names in FACE_DEFS_BY_SIDE[side].items():
        # names = ("vertex", "corner_<near_a>", "corner_<far>", "corner_<near_b>")
        _, n_a_name, far_name, n_b_name = names
        n_a = corners_by_num[int(n_a_name.split("_")[1])]
        n_b = corners_by_num[int(n_b_name.split("_")[1])]
        far = corners_by_num[int(far_name.split("_")[1])]
        estimates.append((n_a[0] + n_b[0] - far[0], n_a[1] + n_b[1] - far[1]))
    vx = sum(e[0] for e in estimates) / 3.0
    vy = sum(e[1] for e in estimates) / 3.0
    return (vx, vy), estimates


# Hybrid-vertex switch threshold. When the affine 3-estimate cloud
# spreads beyond this fraction of the hexagon diameter, the iso
# parallelogram assumption is breaking — use the projective
# (vanishing-point) vertex instead.
#
# Codex P2 on PR #289 head 48f5a66: the original 240 px raw threshold
# was calibrated against `max_image_dim=1600` and would silently miss
# the rows it was added to recover on smaller processing resolutions.
# Normalizing against hexagon diameter (same pattern as
# `projective_residual_norm`) gives a resolution-independent signal.
#
# Empirical sweet spot from the 70-row corpus probe at
# `max_image_dim=1600` (see `tools/HULL_LABELS_CORPUS_REPORT.md`):
#   - spread/diameter <= 0.26 (64 rows): affine 3-estimate mean wins
#     by lower variance from corner-detection noise (iso assumption
#     is close enough that the mean denoises better than projective's
#     exact-but-noisy line intersection)
#   - spread/diameter >  0.26 (6 rows): projective wins for the rows
#     where systematic non-iso bias dominates the noise improvement.
#     Includes 37_B (vertex_err 80→38 px), 44_A (+23 px improvement),
#     45_B (+50 px improvement), 30_B (+16 px), and 2 mild regressions
#     (23_B -29 px, 30_A -10 px; the latter is bad-hull anyway and
#     caught by the projective_residual_norm gate).
# Net: hybrid > 0.26 gives strictly better median/mean/≤30-px-count
# vertex_err than pure affine on the corpus.
#
# Note: `hull_label_acceptance.HullLabelGateThresholds.warn_vertex_cloud_spread_px`
# is still in raw px (240 at max_image_dim=1600). Normalizing that gate
# too would be a parallel improvement but is out of scope for this PR.
HYBRID_PROJECTIVE_SPREAD_NORM_THRESHOLD = 0.26


def _max_pairwise_distance(points: Sequence[Point]) -> float:
    spread = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            spread = max(
                spread,
                math.hypot(points[i][0] - points[j][0],
                           points[i][1] - points[j][1]),
            )
    return spread


def _choose_hybrid_vertex(
    corners_by_num: Dict[int, Point], side: str,
    *,
    spread_norm_threshold: float = HYBRID_PROJECTIVE_SPREAD_NORM_THRESHOLD,
) -> Tuple[Point, Dict[str, Any]]:
    """Hybrid affine/projective vertex selection. Returns
    ``(vertex, telemetry)`` where telemetry carries both candidates
    and the decision metadata (for surfacing in RectifiedFit).

    The switch uses ``vertex_cloud_spread / hexagon_diameter`` — a
    resolution-independent ratio. High normalized spread → iso
    assumption is breaking → use the projective vanishing-point
    vertex. Per Codex P2 on #289 head 48f5a66, this normalization
    keeps behavior stable across processing scales (the original
    raw-px threshold would silently miss perspective-heavy rows on
    smaller processing resolutions).

    Empirical basis: `tools/HULL_LABELS_CORPUS_REPORT.md`
    "Hybrid vertex strategy" section.
    """
    # Affine candidate (current default — parallelogram completion mean)
    affine_vx, estimates = _derive_vertex_from_corners(corners_by_num, side)
    spread_px = _max_pairwise_distance(estimates)

    # Projective candidate (vanishing-point construction). Import locally
    # to keep this module loadable without the heavier projective_vertex
    # module being imported up front.
    from tools.projective_vertex import projective_vertex
    prj = projective_vertex(corners_by_num, side)
    hexagon_diameter = prj.hexagon_diameter_px

    if hexagon_diameter <= 0.0:
        # Degenerate input — fall back to affine (projective would
        # have raised earlier, so this is defensive only).
        spread_norm = 0.0
    else:
        spread_norm = spread_px / hexagon_diameter

    if spread_norm > spread_norm_threshold:
        chosen = prj.vertex
        source = "projective"
    else:
        chosen = affine_vx
        source = "affine"

    # IMPORTANT: do NOT round the numeric gate signals here. These
    # fields are surfaced on RectifiedFit specifically so callers can
    # pass them into `evaluate_hull_label_acceptance` and route on the
    # same decision signals. Rounding (e.g. 0.02504 → 0.0250) can flip
    # near-threshold gate decisions. Round only in the serialization
    # layer (e.g. trace JSON / report tables). Codex P2 on PR #289
    # head 540d891.
    telemetry = {
        "affine_vertex": affine_vx,
        "projective_vertex": prj.vertex,
        "vertex_cloud_spread_px": spread_px,
        "vertex_cloud_spread_norm": spread_norm,
        "hexagon_diameter_px": hexagon_diameter,
        "projective_residual_norm": prj.residual_norm,
        "projective_degeneracy": prj.degeneracy,
        "vertex_source": source,
    }
    return chosen, telemetry


def rectify_via_hull_labels(
    image: Image.Image,
    mask: np.ndarray,
    side: str,
    *,
    face_size_px: int = DEFAULT_FACE_SIZE_PX,
) -> Optional[RectifiedFit]:
    """End-to-end: hull → labeled corners → vertex → face_quads →
    rectified faces. Returns None on failure (e.g. <6 hexagon corners).

    `image` must be the processing-resolution PIL image; `mask` the
    rembg silhouette mask (boolean array, same H×W as image). Callers
    that have the raw rembg alpha channel should prefer
    ``select_hull_label_threshold_fit`` so the production-shaped path can
    choose among the calibrated candidate alpha thresholds.

    The vertex is selected by ``_choose_hybrid_vertex`` — affine
    (parallelogram completion mean) by default, projective (vanishing-
    point construction) when the 3-estimate spread indicates the iso
    assumption is breaking. ``RectifiedFit`` surfaces both candidates
    + decision metadata so callers can route on the same signals.
    """
    hexagon = detect_hexagon_anchors(mask)
    if len(hexagon) != 6:
        return None
    if side not in SILHOUETTE_TO_CORNER:
        raise ValueError(
            f"no per-side mapping defined for side {side!r}; "
            f"add to SILHOUETTE_TO_CORNER"
        )
    corners_by_num = _label_corners_by_position(hexagon, side)
    vertex, telemetry = _choose_hybrid_vertex(corners_by_num, side)
    face_quads: Dict[str, List[Point]] = {}
    rectified_faces: Dict[str, Image.Image] = {}
    for slot, names in FACE_DEFS_BY_SIDE[side].items():
        quad: List[Point] = []
        for name in names:
            if name == "vertex":
                quad.append(vertex)
            else:
                quad.append(corners_by_num[int(name.split("_")[1])])
        face_quads[slot] = quad
        rectified_faces[slot] = rectify_face(image, quad, output_size=face_size_px)
    return RectifiedFit(
        side=side,
        corners_by_num=corners_by_num,
        vertex=vertex,
        face_quads=face_quads,
        rectified_faces=rectified_faces,
        affine_vertex=telemetry["affine_vertex"],
        projective_vertex=telemetry["projective_vertex"],
        vertex_cloud_spread_px=telemetry["vertex_cloud_spread_px"],
        vertex_cloud_spread_norm=telemetry["vertex_cloud_spread_norm"],
        hexagon_diameter_px=telemetry["hexagon_diameter_px"],
        projective_residual_norm=telemetry["projective_residual_norm"],
        projective_degeneracy=telemetry["projective_degeneracy"],
        vertex_source=telemetry["vertex_source"],
    )


# ---------------- CLI / corpus runner ----------------


def _score_rectified_faces(faces: Dict[str, Image.Image]) -> Dict[str, Any]:
    """Sum of CIELAB distance from each sampled sticker to its nearest
    canonical color, summed over 9 stickers × 3 faces. Lower = better
    cluster (each face's stickers cleanly match canonical cube colors).
    """
    total = 0.0
    per_face: Dict[str, float] = {}
    for slot, face_img in faces.items():
        face_total = 0.0
        for row in extract_stickers_from_rectified(face_img):
            for s in row:
                face_total += classify_rgb_with_mode(s.rgb, CLASSIFIER_CANONICAL).distance
        per_face[slot] = round(face_total, 2)
        total += face_total
    return {
        "total_distance": round(total, 2),
        "per_face": per_face,
        "mean_sticker_distance": round(total / 27.0, 2),
    }


def _threshold_candidate_sort_key(candidate: Mapping[str, Any]) -> Tuple[float, int]:
    score = candidate.get("sticker_score_total")
    numeric_score = float(score) if isinstance(score, (int, float)) else float("inf")
    return numeric_score, int(candidate.get("threshold", 9999))


def choose_best_threshold_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    accepted_only: bool,
) -> Optional[Mapping[str, Any]]:
    """Choose the lowest sticker-score candidate, optionally after gates.

    This is deliberately the same policy as the mask-threshold diagnostic:
    prefer a candidate that passed the existing hull-label acceptance gates;
    use the rejected pool only for telemetry when nothing passed.
    """
    pool = [row for row in candidates if row.get("accepted")] if accepted_only else list(candidates)
    if not pool:
        return None
    return min(pool, key=_threshold_candidate_sort_key)


def _threshold_candidate_from_fit(
    *,
    threshold: int,
    side: str,
    fit: RectifiedFit,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any], List[Point]]:
    from tools.hull_label_acceptance import evaluate_hull_label_acceptance

    _affine_vertex, vertex_estimates = _derive_vertex_from_corners(
        fit.corners_by_num, side,
    )
    score = _score_rectified_faces(fit.rectified_faces)
    decision = evaluate_hull_label_acceptance(
        side=side,
        hexagon_corner_count=6,
        vertex_estimates=vertex_estimates,
        rectified_face_slots=fit.rectified_faces.keys(),
        sticker_score_total=float(score["total_distance"]),
        sticker_score_per_face=score["per_face"],
        projective_residual_norm=fit.projective_residual_norm,
    )
    candidate = {
        "threshold": int(threshold),
        "status": "accepted" if decision.accepted else "rejected",
        "accepted": bool(decision.accepted),
        "vertex_source": fit.vertex_source,
        "sticker_score_total": score.get("total_distance"),
        "sticker_score_per_face": score.get("per_face"),
        "mean_sticker_distance": score.get("mean_sticker_distance"),
        "vertex_cloud_spread_px": fit.vertex_cloud_spread_px,
        "vertex_cloud_spread_norm": fit.vertex_cloud_spread_norm,
        "hexagon_diameter_px": fit.hexagon_diameter_px,
        "projective_residual_norm": fit.projective_residual_norm,
        "projective_degeneracy": fit.projective_degeneracy,
        "hard_failures": list(decision.hard_failures),
        "warnings": list(decision.warnings),
    }
    return candidate, decision, score, vertex_estimates


def _threshold_failure_candidate(
    threshold: int,
    status: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "threshold": int(threshold),
        "status": status,
        "accepted": False,
        "sticker_score_total": None,
        "mean_sticker_distance": None,
        "hard_failures": [message],
        "warnings": [],
    }


def select_hull_label_threshold_fit(
    image: Image.Image,
    alpha: np.ndarray,
    side: str,
    *,
    thresholds: Sequence[int] = DEFAULT_MASK_THRESHOLDS,
    face_size_px: int = DEFAULT_FACE_SIZE_PX,
) -> ThresholdFitSelection:
    """Try multiple rembg alpha thresholds and choose the best accepted fit.

    The old production-shaped path used a single ``alpha > 128`` mask. Set 70
    showed that shadows can distort that mask enough to poison the hull. This
    selector runs rembg once, tries a small threshold set, applies the existing
    hull-label gates to each candidate, and returns the lowest sticker-score
    candidate among accepted fits. If no threshold passes the gates, callers
    should fall back to the legacy path.
    """
    if not thresholds:
        raise ValueError("thresholds must contain at least one alpha value")

    fit_by_threshold: Dict[int, RectifiedFit] = {}
    decision_by_threshold: Dict[int, Any] = {}
    score_by_threshold: Dict[int, Dict[str, Any]] = {}
    estimates_by_threshold: Dict[int, List[Point]] = {}
    candidates: List[Dict[str, Any]] = []

    for raw_threshold in thresholds:
        threshold = int(raw_threshold)
        mask = np.asarray(alpha, dtype=np.uint8) > threshold
        if not mask.any():
            candidates.append(
                _threshold_failure_candidate(threshold, "rejected", "empty cube mask")
            )
            continue

        fit = rectify_via_hull_labels(image, mask, side, face_size_px=face_size_px)
        if fit is None:
            candidates.append(
                _threshold_failure_candidate(
                    threshold,
                    "rejected",
                    "rectify_via_hull_labels returned None",
                )
            )
            continue

        candidate, decision, score, vertex_estimates = _threshold_candidate_from_fit(
            threshold=threshold,
            side=side,
            fit=fit,
        )
        candidates.append(candidate)
        fit_by_threshold[threshold] = fit
        decision_by_threshold[threshold] = decision
        score_by_threshold[threshold] = score
        estimates_by_threshold[threshold] = vertex_estimates

    best_any = choose_best_threshold_candidate(candidates, accepted_only=False)
    best_accepted = choose_best_threshold_candidate(candidates, accepted_only=True)
    selected_threshold = (
        int(best_accepted["threshold"]) if best_accepted is not None else None
    )
    trace: Dict[str, Any] = {
        "thresholds": [int(value) for value in thresholds],
        "threshold_candidates": candidates,
        "best_any_threshold": int(best_any["threshold"]) if best_any is not None else None,
        "best_any_accepted": bool(best_any and best_any.get("accepted")),
        "best_any_score": (
            best_any.get("sticker_score_total") if best_any is not None else None
        ),
        "selected_mask_threshold": selected_threshold,
    }

    if selected_threshold is None:
        trace.update({
            "status": "rejected",
            "accepted": False,
            "hard_failures": ["no alpha threshold candidate accepted"],
            "warnings": [],
        })
        if best_any is not None:
            trace["best_any_hard_failures"] = list(best_any.get("hard_failures") or [])
            trace["best_any_warnings"] = list(best_any.get("warnings") or [])
        return ThresholdFitSelection(
            fit=None,
            threshold=None,
            candidates=candidates,
            decision=None,
            score=None,
            vertex_estimates=None,
            trace=trace,
        )

    chosen = fit_by_threshold[selected_threshold]
    selected_candidate = next(
        row for row in candidates if int(row.get("threshold", -1)) == selected_threshold
    )
    trace.update(selected_candidate)
    trace["selected_mask_threshold"] = selected_threshold
    return ThresholdFitSelection(
        fit=chosen,
        threshold=selected_threshold,
        candidates=candidates,
        decision=decision_by_threshold[selected_threshold],
        score=score_by_threshold[selected_threshold],
        vertex_estimates=estimates_by_threshold[selected_threshold],
        trace=trace,
    )


def evaluate_row(
    sess: Any,
    key: str,
    image_path: Path,
    truth_row: Dict[str, Any],
    max_image_dim: int,
    gallery_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from rembg import remove  # noqa: E402
    side = key.rsplit("_", 1)[-1]
    rec: Dict[str, Any] = {"key": key, "side": side, "status": "pending"}
    try:
        image, scale = _processing_image(image_path, max_image_dim)
        rgba = remove(image, session=sess)
        alpha = np.array(rgba.split()[-1], dtype=np.uint8)
        selection = select_hull_label_threshold_fit(image, alpha, side)
        fit = selection.fit
        if fit is None:
            rec.update({
                "status": "no_accepted_threshold",
                "error": "no alpha threshold candidate accepted",
                "threshold_trace": selection.trace,
            })
            return rec
        # Compare against ground truth
        gt_vertex = (truth_row["vertex"][0] * scale, truth_row["vertex"][1] * scale)
        vertex_err = math.hypot(
            fit.vertex[0] - gt_vertex[0], fit.vertex[1] - gt_vertex[1],
        )
        per_corner_err = {}
        for i in range(6):
            gt = (truth_row[f"corner_{i}"][0] * scale, truth_row[f"corner_{i}"][1] * scale)
            d = math.hypot(fit.corners_by_num[i][0] - gt[0],
                           fit.corners_by_num[i][1] - gt[1])
            per_corner_err[f"corner_{i}"] = round(d, 1)
        mean_corner_err = sum(per_corner_err.values()) / 6.0
        # Score our rectification
        our_score = _score_rectified_faces(fit.rectified_faces)
        # Oracle reference rectification (using GT vertex + GT corners)
        oracle_faces: Dict[str, Image.Image] = {}
        for slot, names in FACE_DEFS_BY_SIDE[side].items():
            quad = []
            for name in names:
                if name == "vertex":
                    quad.append(gt_vertex)
                else:
                    quad.append((truth_row[name][0] * scale, truth_row[name][1] * scale))
            oracle_faces[slot] = rectify_face(
                image, quad, output_size=DEFAULT_FACE_SIZE_PX,
            )
        oracle_score = _score_rectified_faces(oracle_faces)
        if gallery_dir is not None:
            _save_gallery_panel(
                key, side, image, fit, oracle_faces,
                our_score, oracle_score, mean_corner_err, vertex_err,
                gallery_dir,
            )
        rec.update({
            "status": "rectified",
            "selected_mask_threshold": selection.threshold,
            "threshold_candidates": selection.candidates,
            "labeling_mean_corner_err_px": round(mean_corner_err, 1),
            "per_corner_err_px": per_corner_err,
            "derived_vertex_error_processing_px": round(vertex_err, 1),
            "rectified_score": our_score,
            "oracle_score": oracle_score,
            "score_delta_vs_oracle": round(
                our_score["total_distance"] - oracle_score["total_distance"], 2,
            ),
        })
    except Exception as exc:  # noqa: BLE001
        rec.update({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    return rec


def _save_gallery_panel(
    key: str, side: str, image: Image.Image,
    fit: RectifiedFit, oracle_faces: Dict[str, Image.Image],
    our_score: Dict[str, Any], oracle_score: Dict[str, Any],
    corner_err: float, vertex_err: float,
    gallery_dir: Path,
) -> None:
    """Per-row panel: source-with-overlay + our 3 faces + oracle 3
    faces. Saved under `{gallery_dir}/by_row/{key}.png`."""
    slot_order = ("upper", "right", "front")
    overlay = image.convert("RGB").copy()
    d = ImageDraw.Draw(overlay, "RGBA")
    for quad in fit.face_quads.values():
        pts = [(int(p[0]), int(p[1])) for p in quad] + [(int(quad[0][0]), int(quad[0][1]))]
        d.line(pts, fill=(220, 80, 80, 230), width=3)
    src_thumb = overlay.copy()
    src_thumb.thumbnail((360, 360))
    PANEL = 200
    margin = 16
    total_w = margin + src_thumb.width + margin + 3 * (PANEL + margin)
    total_h = 80 + max(src_thumb.height, 2 * (PANEL + 24))
    panel = Image.new("RGB", (total_w, total_h), (28, 28, 32))
    pd = ImageDraw.Draw(panel)
    pd.text(
        (margin, 8),
        f"{key} (side {side})  corner_err {corner_err:.1f}px  "
        f"vertex_err {vertex_err:.1f}px  "
        f"score {our_score['total_distance']:.0f} (oracle {oracle_score['total_distance']:.0f}, "
        f"Δ{our_score['total_distance']-oracle_score['total_distance']:+.0f})",
        fill=(255, 255, 255),
    )
    panel.paste(src_thumb, (margin, 50))
    face_x0 = margin + src_thumb.width + margin
    pd.text((face_x0, 38), "Hull-labels rectified:", fill=(220, 120, 120))
    pd.text((face_x0, 38 + PANEL + 24), "Oracle rectified (GT):", fill=(120, 220, 120))
    for i, slot in enumerate(slot_order):
        x = face_x0 + i * (PANEL + margin)
        r = fit.rectified_faces[slot].resize((PANEL, PANEL), Image.Resampling.BICUBIC)
        panel.paste(r, (x, 60))
        pd.text((x, 60 + PANEL + 4), slot, fill=(200, 200, 200))
        or_ = oracle_faces[slot].resize((PANEL, PANEL), Image.Resampling.BICUBIC)
        panel.paste(or_, (x, 60 + PANEL + 24))
        pd.text((x, 60 + 2 * PANEL + 28), slot, fill=(200, 200, 200))
    (gallery_dir / "by_row").mkdir(parents=True, exist_ok=True)
    panel.save(gallery_dir / "by_row" / f"{key}.png")


def _git_head_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--gallery-dir", type=Path, default=DEFAULT_GALLERY_DIR)
    ap.add_argument("--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM)
    args = ap.parse_args(list(argv) if argv is not None else None)

    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    set_index = {str(p["setId"]): p for p in manifest.get("pairs", [])}
    args.gallery_dir.mkdir(parents=True, exist_ok=True)
    from rembg import new_session  # noqa: E402
    # Explicit "u2net" — matches `tools/measure_axis_correctness.py` and
    # production (`rubik_recognizer/image_pipeline.py`). The default is
    # already u2net, but pinning explicitly avoids the silent drift bug
    # if rembg ever changes its default model. (Codex polish on PR #279.)
    sess = new_session("u2net")
    records: List[Dict[str, Any]] = []
    for key in sorted(truth):
        row = truth[key]
        if not row.get("approved"):
            continue
        set_id, _, side = key.rpartition("_")
        pair = set_index.get(set_id)
        if pair is None:
            records.append({"key": key, "status": "skipped",
                            "error": f"set {set_id} not in manifest"})
            continue
        img_path_str = pair.get(f"image{side}Path")
        if not img_path_str:
            records.append({"key": key, "status": "skipped",
                            "error": "no image path"})
            continue
        img_path = Path(img_path_str)
        if not img_path.exists():
            records.append({"key": key, "status": "skipped",
                            "error": f"image not found: {img_path}"})
            continue
        print(
            f"[{len([r for r in records if r.get('status')=='rectified'])+1}] {key}...",
            flush=True,
        )
        rec = evaluate_row(sess, key, img_path, row, args.max_image_dim, args.gallery_dir)
        records.append(rec)
    # Source metadata for the trace
    source = {
        "tool": "tools/rectify_via_hull_labels.py",
        "git_sha": _git_head_sha(),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "truth": str(args.truth.relative_to(REPO_ROOT)) if str(args.truth).startswith(str(REPO_ROOT)) else str(args.truth),
        "manifest": str(args.manifest.relative_to(REPO_ROOT)) if str(args.manifest).startswith(str(REPO_ROOT)) else str(args.manifest),
        "max_image_dim": args.max_image_dim,
        "face_size_px": DEFAULT_FACE_SIZE_PX,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({
            "schema": "rectify_via_hull_labels_v1",
            "source": source,
            "per_row": records,
        }, indent=2),
        encoding="utf-8",
    )
    # Quick summary
    scored = [r for r in records if r.get("status") == "rectified"]
    print(f"\n{len(scored)}/{len(records)} rows rectified")
    if scored:
        deltas = [r["score_delta_vs_oracle"] for r in scored]
        print(f"Score delta vs oracle: min {min(deltas):+.1f}, "
              f"max {max(deltas):+.1f}, median {statistics.median(deltas):+.1f}")
        verr = [r["derived_vertex_error_processing_px"] for r in scored]
        print(f"Derived vertex error: min {min(verr):.1f}, "
              f"max {max(verr):.1f}, median {statistics.median(verr):.1f}px")
        cerr = [r["labeling_mean_corner_err_px"] for r in scored]
        print(f"Labeling mean corner error: min {min(cerr):.1f}, "
              f"max {max(cerr):.1f}, median {statistics.median(cerr):.1f}px")
    print(f"Trace: {args.out_json}")
    print(f"Gallery: {args.gallery_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
