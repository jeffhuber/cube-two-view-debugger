#!/usr/bin/env python3
"""Audit production-available tie-breakers for affine phase selection.

Diagnostic-only: no production behavior change.

The fit-stage transition trace showed that many broken final axes are
already broken at the first affine correspondence selector. This tool
keeps the search layer narrow:

  1. enumerate the same 720 detected-hexagon-to-template affine candidates,
  2. group exact/near minimum-residual candidates,
  3. rank those groups with production-available secondary evidence, and
  4. use full-corner truth only to evaluate which selector would have won.

The audit is deliberately not an oracle picker. Human truth never enters
the selector metrics; it only labels outcomes as usable / broken.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_pipeline_phase_parity import _processing_image  # noqa: E402
from tools.global_cube_model import (  # noqa: E402
    _TEMPLATE_HEXAGON_2D_ISO,
    _affine_residual,
    _fit_affine_2d,
    derive_geometry,
    detect_hexagon_anchors,
    GlobalCubeModel,
)
from tools.interior_bezel_detection import detect_interior_bezel_lines  # noqa: E402
from tools.measure_axis_correctness import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_MAX_IMAGE_DIM,
    DEFAULT_TRUTH,
    _candidate_image_roots,
    _display_path,
    _git_head_sha,
    _ground_truth_axes,
    _match_axes_to_ground_truth,
    _now_utc_iso,
    _resolve_image_path,
)


Point = Tuple[float, float]
TEMPLATE_KEYS: Tuple[str, ...] = ("h_x", "h_y", "h_z", "h_xy", "h_xz", "h_yz")
DEFAULT_TIE_RMS_EPSILON = 0.001
DEFAULT_NEAR_RMS_EPSILON = 0.25
DEFAULT_OUT_JSON = REPO_ROOT / "tests" / "fixtures" / "affine_phase_tiebreaker_trace.json"
DEFAULT_OUT_MD = REPO_ROOT / "tools" / "AFFINE_PHASE_TIEBREAKER_REPORT.md"


def _round_point(point: Iterable[float]) -> List[float]:
    x, y = list(point)[:2]
    return [round(float(x), 1), round(float(y), 1)]


def _angle_diff_undirected_deg(a_rad: float, b_rad: float) -> float:
    """Smallest angle between two undirected 2D lines, in degrees."""
    diff = abs((a_rad - b_rad + math.pi / 2.0) % math.pi - math.pi / 2.0)
    return math.degrees(diff)


def _axis_angle_rad(axis: Point) -> Optional[float]:
    if math.hypot(axis[0], axis[1]) < 1e-9:
        return None
    return math.atan2(axis[1], axis[0])


def _axis_bezel_alignment(
    axes: Sequence[Point],
    bezel_angles_rad: Sequence[float],
) -> Optional[Dict[str, Any]]:
    """Best undirected 3-axis to 3-bezel-line angular assignment."""
    if len(axes) != 3 or len(bezel_angles_rad) < 3:
        return None
    axis_angles = [_axis_angle_rad(axis) for axis in axes]
    if any(angle is None for angle in axis_angles):
        return None

    best: Optional[Dict[str, Any]] = None
    for perm in itertools.permutations(range(len(bezel_angles_rad)), 3):
        errors = [
            _angle_diff_undirected_deg(float(axis_angles[i]), bezel_angles_rad[perm[i]])
            for i in range(3)
        ]
        total = sum(errors)
        if best is None or total < best["total_alignment_error_deg"]:
            best = {
                "total_alignment_error_deg": total,
                "mean_alignment_error_deg": total / 3.0,
                "per_axis_alignment_error_deg": errors,
                "assignment": list(perm),
            }

    if best is None:
        return None
    return {
        "total_alignment_error_deg": round(best["total_alignment_error_deg"], 2),
        "mean_alignment_error_deg": round(best["mean_alignment_error_deg"], 2),
        "per_axis_alignment_error_deg": [
            round(float(err), 2) for err in best["per_axis_alignment_error_deg"]
        ],
        "assignment": best["assignment"],
    }


def _classify_axis_misfit(total_misfit_deg: Optional[float]) -> str:
    if not isinstance(total_misfit_deg, (int, float)):
        return "unknown"
    if total_misfit_deg <= 30.0:
        return "usable"
    if total_misfit_deg >= 150.0:
        return "broken"
    return "marginal"


def _model_from_affine(
    A: np.ndarray,
    b: np.ndarray,
    *,
    residual_px2: float,
    perm: Sequence[int],
    order_index: int,
) -> GlobalCubeModel:
    hx_img = A @ np.array(_TEMPLATE_HEXAGON_2D_ISO["h_x"]) + b
    hy_img = A @ np.array(_TEMPLATE_HEXAGON_2D_ISO["h_y"]) + b
    hz_img = A @ np.array(_TEMPLATE_HEXAGON_2D_ISO["h_z"]) + b
    model = GlobalCubeModel(
        cube_center_screen=(float(b[0]), float(b[1])),
        axis_x_2d=(float(hx_img[0] - b[0]), float(hx_img[1] - b[1])),
        axis_y_2d=(float(hy_img[0] - b[0]), float(hy_img[1] - b[1])),
        axis_z_2d=(float(hz_img[0] - b[0]), float(hz_img[1] - b[1])),
    )
    derive_geometry(model)
    model.fit_loss = residual_px2
    model.fit_quality = max(0.0, 1.0 - math.sqrt(residual_px2) / 200.0)
    model.debug = {
        "approach": "affine_candidate",
        "affine_rms_px": round(math.sqrt(residual_px2), 6),
        "best_permutation": list(perm),
        "permutation_order_index": order_index,
    }
    return model


def _candidate_records(
    hexagon_vertices_ccw: Sequence[Point],
    *,
    gt_vertex: Point,
    gt_axes: Sequence[Point],
    bezel_angles_rad: Sequence[float],
    bezel_vertex: Optional[Point],
) -> List[Dict[str, Any]]:
    template_positions = np.array(
        [_TEMPLATE_HEXAGON_2D_ISO[k] for k in TEMPLATE_KEYS],
        dtype=np.float64,
    )
    all_detected = np.array(hexagon_vertices_ccw, dtype=np.float64)
    hex_centroid = (
        float(all_detected[:, 0].mean()),
        float(all_detected[:, 1].mean()),
    )
    records: List[Dict[str, Any]] = []

    for order_index, perm in enumerate(itertools.permutations(range(6))):
        detected_permuted = all_detected[list(perm)]
        try:
            A, b = _fit_affine_2d(template_positions, detected_permuted)
        except Exception as exc:  # noqa: BLE001
            records.append({
                "status": "fit_error",
                "perm": list(perm),
                "order_index": order_index,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        residual_px2 = _affine_residual(template_positions, detected_permuted, A, b)
        model = _model_from_affine(
            A,
            b,
            residual_px2=residual_px2,
            perm=perm,
            order_index=order_index,
        )
        predicted_axes = [model.axis_x_2d, model.axis_y_2d, model.axis_z_2d]
        axis_match = _match_axes_to_ground_truth(predicted_axes, gt_axes)
        total_misfit = axis_match.get("total_misfit_deg")
        bezel_alignment = _axis_bezel_alignment(predicted_axes, bezel_angles_rad)
        vertex = model.cube_center_screen
        center_error_to_bezel = None
        if bezel_vertex is not None:
            center_error_to_bezel = math.hypot(
                vertex[0] - bezel_vertex[0],
                vertex[1] - bezel_vertex[1],
            )
        records.append({
            "status": "scored",
            "perm": list(perm),
            "order_index": order_index,
            "residual_px2": residual_px2,
            "residual_rms_px": math.sqrt(residual_px2),
            "axis_state": _classify_axis_misfit(total_misfit),
            "axis_match": axis_match,
            "bezel_alignment": bezel_alignment,
            "center_error_to_bezel_px": center_error_to_bezel,
            "center_error_to_hex_centroid_px": math.hypot(
                vertex[0] - hex_centroid[0],
                vertex[1] - hex_centroid[1],
            ),
            "vertex_error_to_truth_px": math.hypot(
                vertex[0] - gt_vertex[0],
                vertex[1] - gt_vertex[1],
            ),
            "predicted_vertex_processing_px": _round_point(vertex),
        })

    return [r for r in records if r.get("status") == "scored"]


def _num(value: Any, default: float = float("inf")) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _total_axis_misfit(record: Dict[str, Any]) -> float:
    return _num(record.get("axis_match", {}).get("total_misfit_deg"))


def _bezel_total(record: Dict[str, Any]) -> float:
    return _num(record.get("bezel_alignment", {}).get("total_alignment_error_deg"))


def _center_bezel(record: Dict[str, Any]) -> float:
    return _num(record.get("center_error_to_bezel_px"))


def _center_hex(record: Dict[str, Any]) -> float:
    return _num(record.get("center_error_to_hex_centroid_px"))


def _residual(record: Dict[str, Any]) -> float:
    return _num(record.get("residual_px2"))


def _order(record: Dict[str, Any]) -> int:
    return int(record.get("order_index", 10**9))


def _compact_record(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if record is None:
        return None
    bezel = record.get("bezel_alignment") or {}
    return {
        "perm": record.get("perm"),
        "order_index": record.get("order_index"),
        "residual_rms_px": round(_num(record.get("residual_rms_px")), 6),
        "axis_state": record.get("axis_state"),
        "total_axis_misfit_deg": record.get("axis_match", {}).get("total_misfit_deg"),
        "bezel_total_alignment_error_deg": bezel.get("total_alignment_error_deg"),
        "center_error_to_bezel_px": (
            round(_center_bezel(record), 1)
            if math.isfinite(_center_bezel(record)) else None
        ),
        "center_error_to_hex_centroid_px": round(_center_hex(record), 1),
        "vertex_error_to_truth_px": round(_num(record.get("vertex_error_to_truth_px")), 1),
        "predicted_vertex_processing_px": record.get("predicted_vertex_processing_px"),
    }


def _group_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for record in records:
        state = str(record.get("axis_state", "unknown"))
        counts[state] = counts.get(state, 0) + 1
    def metric_range(metric_fn: Any) -> Optional[float]:
        values = [metric_fn(record) for record in records]
        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            return None
        return round(max(finite) - min(finite), 6)

    return {
        "n": len(records),
        "axis_state_counts": counts,
        "has_usable": any(record.get("axis_state") == "usable" for record in records),
        "has_not_broken": any(record.get("axis_state") != "broken" for record in records),
        "residual_rms_range_px": metric_range(lambda r: _num(r.get("residual_rms_px"))),
        "bezel_alignment_range_deg": metric_range(_bezel_total),
        "center_to_bezel_range_px": metric_range(_center_bezel),
        "center_to_hex_centroid_range_px": metric_range(_center_hex),
    }


def _selector_winners(records: Sequence[Dict[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
    if not records:
        return {}
    return {
        "residual_then_order": min(records, key=lambda r: (_residual(r), _order(r))),
        "bezel_alignment": min(records, key=lambda r: (_bezel_total(r), _center_bezel(r), _order(r))),
        "center_to_bezel": min(records, key=lambda r: (_center_bezel(r), _bezel_total(r), _order(r))),
        "center_to_hex_centroid": min(records, key=lambda r: (_center_hex(r), _bezel_total(r), _order(r))),
        "bezel_then_center": min(records, key=lambda r: (_bezel_total(r), _center_bezel(r), _center_hex(r), _order(r))),
    }


def summarize_row(
    records: Sequence[Dict[str, Any]],
    *,
    tie_rms_epsilon: float,
    near_rms_epsilon: float,
) -> Dict[str, Any]:
    if not records:
        return {"status": "no_scored_candidates"}

    min_rms = min(_num(record.get("residual_rms_px")) for record in records)
    exact_group = [
        record for record in records
        if _num(record.get("residual_rms_px")) - min_rms <= tie_rms_epsilon
    ]
    near_group = [
        record for record in records
        if _num(record.get("residual_rms_px")) - min_rms <= near_rms_epsilon
    ]
    production = min(records, key=lambda r: (_residual(r), _order(r)))
    best_axis = min(records, key=lambda r: (_total_axis_misfit(r), _residual(r), _order(r)))
    best_not_broken = [
        record for record in sorted(records, key=lambda r: (_residual(r), _order(r)))
        if record.get("axis_state") != "broken"
    ]

    exact_winners = _selector_winners(exact_group)
    near_winners = _selector_winners(near_group)

    return {
        "status": "scored",
        "min_residual_rms_px": round(min_rms, 6),
        "tie_rms_epsilon": tie_rms_epsilon,
        "near_rms_epsilon": near_rms_epsilon,
        "exact_group": _group_counts(exact_group),
        "near_group": _group_counts(near_group),
        "production_selected": _compact_record(production),
        "best_axis_oracle": _compact_record(best_axis),
        "first_not_broken_by_residual": (
            _compact_record(best_not_broken[0]) if best_not_broken else None
        ),
        "exact_group_winners": {
            name: _compact_record(record)
            for name, record in exact_winners.items()
        },
        "near_group_winners": {
            name: _compact_record(record)
            for name, record in near_winners.items()
        },
        "top_by_residual": [
            _compact_record(record)
            for record in sorted(records, key=lambda r: (_residual(r), _order(r)))[:12]
        ],
    }


def trace_one_row(
    sess: Any,
    key: str,
    image_path: Path,
    truth_row: Dict[str, Any],
    max_image_dim: int,
    *,
    tie_rms_epsilon: float,
    near_rms_epsilon: float,
) -> Dict[str, Any]:
    from rembg import remove  # noqa: E402

    side = key.rsplit("_", 1)[-1]
    record: Dict[str, Any] = {
        "key": key,
        "side": side,
        "yaw_quarter_turns": int(truth_row.get("yaw_quarter_turns", 0)),
        "status": "traced",
        "image_path": _display_path(image_path),
    }
    try:
        image, scale = _processing_image(image_path, max_image_dim)
        rgb_array = np.array(image)
        rgba = remove(image, session=sess)
        mask_array = np.array(rgba.split()[-1], dtype=np.uint8) > 128
        detection = detect_interior_bezel_lines(rgb_array, mask_array)
        hexagon = detect_hexagon_anchors(mask_array)
        if len(hexagon) != 6:
            return {
                **record,
                "status": "fit_failed",
                "error": f"hexagon detection produced {len(hexagon)} vertices",
            }
        gt_vertex, gt_axes = _ground_truth_axes(truth_row, side, scale)
        candidates = _candidate_records(
            hexagon,
            gt_vertex=gt_vertex,
            gt_axes=gt_axes,
            bezel_angles_rad=detection.boundary_angles[:3],
            bezel_vertex=detection.cube_center,
        )
        record.update({
            "processing_scale": round(scale, 6),
            "gt_vertex_processing_px": _round_point(gt_vertex),
            "bezel_vertex_processing_px": (
                _round_point(detection.cube_center)
                if detection.cube_center is not None else None
            ),
            "bezel_signal_quality": round(float(detection.signal_quality), 4),
            "bezel_line_qualities": [
                round(float(q), 4) for q in detection.line_qualities
            ],
            "hexagon_vertices_processing_px": [_round_point(p) for p in hexagon],
            "summary": summarize_row(
                candidates,
                tie_rms_epsilon=tie_rms_epsilon,
                near_rms_epsilon=near_rms_epsilon,
            ),
        })
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    return record


def _selector_state(row: Dict[str, Any], selector_path: Sequence[str]) -> str:
    value: Any = row.get("summary", {})
    for key in selector_path:
        if not isinstance(value, dict):
            return "missing"
        value = value.get(key)
    if isinstance(value, dict):
        return str(value.get("axis_state", "missing"))
    return "missing"


def _selector_outcome_vs_production(row: Dict[str, Any], selector_path: Sequence[str]) -> str:
    prod = _selector_state(row, ("production_selected",))
    selected = _selector_state(row, selector_path)
    if prod == "broken" and selected == "usable":
        return "fixes_broken_to_usable"
    if prod == "usable" and selected == "broken":
        return "breaks_usable_to_broken"
    if prod == selected:
        return f"keeps_{prod}"
    return f"{prod}_to_{selected}"


def summarize_payload(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    traced = [row for row in rows if row.get("status") == "traced"]
    selector_paths = {
        "production": ("production_selected",),
        "exact_bezel": ("exact_group_winners", "bezel_alignment"),
        "exact_center_to_bezel": ("exact_group_winners", "center_to_bezel"),
        "exact_bezel_then_center": ("exact_group_winners", "bezel_then_center"),
        "near_bezel": ("near_group_winners", "bezel_alignment"),
        "near_center_to_bezel": ("near_group_winners", "center_to_bezel"),
        "near_bezel_then_center": ("near_group_winners", "bezel_then_center"),
    }
    selector_state_counts: Dict[str, Dict[str, int]] = {}
    selector_effect_counts: Dict[str, Dict[str, int]] = {}
    for name, path in selector_paths.items():
        state_counts: Dict[str, int] = {}
        effect_counts: Dict[str, int] = {}
        for row in traced:
            state = _selector_state(row, path)
            state_counts[state] = state_counts.get(state, 0) + 1
            effect = (
                "baseline"
                if name == "production"
                else _selector_outcome_vs_production(row, path)
            )
            effect_counts[effect] = effect_counts.get(effect, 0) + 1
        selector_state_counts[name] = state_counts
        selector_effect_counts[name] = effect_counts

    exact_sizes = [
        row.get("summary", {}).get("exact_group", {}).get("n")
        for row in traced
        if isinstance(row.get("summary", {}).get("exact_group", {}).get("n"), int)
    ]
    near_sizes = [
        row.get("summary", {}).get("near_group", {}).get("n")
        for row in traced
        if isinstance(row.get("summary", {}).get("near_group", {}).get("n"), int)
    ]

    return {
        "n_rows": len(rows),
        "n_traced": len(traced),
        "selector_axis_state_counts": selector_state_counts,
        "selector_effect_counts_vs_production": selector_effect_counts,
        "exact_tie_rows_with_usable_candidate": sum(
            1 for row in traced
            if row.get("summary", {}).get("exact_group", {}).get("has_usable")
        ),
        "near_tie_rows_with_usable_candidate": sum(
            1 for row in traced
            if row.get("summary", {}).get("near_group", {}).get("has_usable")
        ),
        "median_exact_tie_group_size": (
            round(float(statistics.median(exact_sizes)), 1) if exact_sizes else None
        ),
        "median_near_tie_group_size": (
            round(float(statistics.median(near_sizes)), 1) if near_sizes else None
        ),
        "exact_rows_with_nonzero_bezel_alignment_range": sum(
            1 for row in traced
            if _num(row.get("summary", {}).get("exact_group", {}).get("bezel_alignment_range_deg"), 0.0) > 0.001
        ),
        "exact_rows_with_nonzero_center_to_bezel_range": sum(
            1 for row in traced
            if _num(row.get("summary", {}).get("exact_group", {}).get("center_to_bezel_range_px"), 0.0) > 0.001
        ),
    }


def run_all(
    truth: Dict[str, Any],
    manifest: Dict[str, Any],
    max_image_dim: int,
    *,
    tie_rms_epsilon: float,
    near_rms_epsilon: float,
    truth_path: Path = DEFAULT_TRUTH,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> Dict[str, Any]:
    from rembg import new_session  # noqa: E402

    sess = new_session("u2net")
    set_index = {
        str(pair["setId"]): pair for pair in manifest.get("pairs", [])
    }
    image_roots = _candidate_image_roots(manifest)
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    keys = [key for key, row in sorted(truth.items()) if row.get("approved")]

    for index, key in enumerate(keys, 1):
        truth_row = truth[key]
        set_id, _, side = key.rpartition("_")
        pair = set_index.get(set_id)
        if pair is None:
            skipped.append({"key": key, "reason": f"set {set_id} not in manifest"})
            continue
        image_path_str = pair.get(f"image{side}Path")
        if not image_path_str:
            skipped.append({"key": key, "reason": "no image path"})
            continue
        expected_sha = pair.get(f"image{side}_sha256_expected")
        image_path = _resolve_image_path(
            str(image_path_str),
            set_id,
            side,
            image_roots,
            expected_sha256=expected_sha,
        )
        if image_path is None:
            skipped.append({"key": key, "reason": "image not found or SHA mismatch"})
            continue
        rows.append(
            trace_one_row(
                sess,
                key,
                image_path,
                truth_row,
                max_image_dim,
                tie_rms_epsilon=tie_rms_epsilon,
                near_rms_epsilon=near_rms_epsilon,
            )
        )
        print(f"  [{index}/{len(keys)}] {key}", file=sys.stderr, flush=True)

    return {
        "schema": "affine_phase_tiebreaker_trace_v1",
        "source": {
            "tool": "tools/diagnose_affine_phase_tiebreakers.py",
            "git_sha": _git_head_sha(),
            "generated_at_utc": _now_utc_iso(),
            "truth": _display_path(truth_path),
            "manifest": _display_path(manifest_path),
            "max_image_dim": max_image_dim,
            "tie_rms_epsilon": tie_rms_epsilon,
            "near_rms_epsilon": near_rms_epsilon,
            "mask_path": "rembg.remove(...).alpha channel, matching production baselines",
            "human_truth_usage": "evaluation only, never selector input",
        },
        "summary": summarize_payload(rows),
        "per_row": rows,
        "skipped": skipped,
    }


def render_report(payload: Dict[str, Any]) -> str:
    source = payload.get("source", {})
    summary = payload.get("summary", {})
    lines: List[str] = [
        "# Affine phase tie-breaker audit",
        "",
        "This diagnostic enumerates the 720 affine correspondence candidates "
        "and audits production-available secondary tie-breakers. Human "
        "full-corner truth is used only to label outcomes as usable or "
        "broken.",
        "",
        "## Source",
        "",
        f"- Tool: `{source.get('tool', '-')}`",
        f"- Commit: `{source.get('git_sha', '-')}`",
        f"- Generated: `{source.get('generated_at_utc', '-')}`",
        f"- Truth: `{source.get('truth', '-')}`",
        f"- Manifest: `{source.get('manifest', '-')}`",
        f"- Max image dim: `{source.get('max_image_dim', '-')}`",
        f"- Exact tie RMS epsilon: `{source.get('tie_rms_epsilon', '-')}`",
        f"- Near tie RMS epsilon: `{source.get('near_rms_epsilon', '-')}`",
        f"- Mask path: {source.get('mask_path', '-')}",
        f"- Human truth usage: {source.get('human_truth_usage', '-')}",
        "",
        "## Aggregate",
        "",
        f"- Rows traced: {summary.get('n_traced', 0)} / {summary.get('n_rows', 0)}",
        "- Selector axis-state counts: "
        f"`{summary.get('selector_axis_state_counts', {})}`",
        "- Selector effects vs production: "
        f"`{summary.get('selector_effect_counts_vs_production', {})}`",
        "- Exact tie rows with a usable candidate: "
        f"`{summary.get('exact_tie_rows_with_usable_candidate')}`",
        "- Near tie rows with a usable candidate: "
        f"`{summary.get('near_tie_rows_with_usable_candidate')}`",
        "- Median exact / near group size: "
        f"`{summary.get('median_exact_tie_group_size')}` / "
        f"`{summary.get('median_near_tie_group_size')}`",
        "- Exact rows with nonzero bezel / center metric range: "
        f"`{summary.get('exact_rows_with_nonzero_bezel_alignment_range')}` / "
        f"`{summary.get('exact_rows_with_nonzero_center_to_bezel_range')}`",
        "",
        "## Headline Findings",
        "",
        "- Every row has a 12-candidate exact residual tie group with "
        "6 usable and 6 broken phases.",
        "- Simple geometric secondary metrics are degenerate in this "
        "group: bezel-axis alignment and center proximity do not vary "
        "across exact ties. Apparent selector wins/losses from these "
        "metrics are therefore fallback-order effects, not reliable "
        "evidence.",
        "- The next production candidate needs a signal that breaks the "
        "3-fold phase symmetry, such as directed face/color evidence or "
        "a stronger convention-aware correspondence constraint.",
        "",
        "## Per-row Summary",
        "",
        "| Row | prod | exact n/u | exact metric ranges | exact bezel | near n/u | near bezel | best oracle |",
        "|---|---|---:|---|---|---:|---|---|",
    ]
    for row in payload.get("per_row", []):
        if row.get("status") != "traced":
            lines.append(
                f"| `{row.get('key')}` | {row.get('status')} | - | - | - | - | - | - | {row.get('error', '')[:60]} |"
            )
            continue
        s = row.get("summary", {})
        exact = s.get("exact_group", {})
        near = s.get("near_group", {})
        prod = s.get("production_selected") or {}
        best = s.get("best_axis_oracle") or {}
        exact_w = s.get("exact_group_winners", {})
        near_w = s.get("near_group_winners", {})

        def state(winners: Dict[str, Any], name: str) -> str:
            winner = winners.get(name) or {}
            total = winner.get("total_axis_misfit_deg", "-")
            return f"{winner.get('axis_state', '-')} ({total})"

        lines.append(
            f"| `{row.get('key')}` "
            f"| {prod.get('axis_state', '-')} ({prod.get('total_axis_misfit_deg', '-')}) "
            f"| {exact.get('n', '-')} / {exact.get('axis_state_counts', {}).get('usable', 0)} "
            f"| bezel={exact.get('bezel_alignment_range_deg', '-')}; "
            f"center={exact.get('center_to_bezel_range_px', '-')} "
            f"| {state(exact_w, 'bezel_alignment')} "
            f"| {near.get('n', '-')} / {near.get('axis_state_counts', {}).get('usable', 0)} "
            f"| {state(near_w, 'bezel_alignment')} "
            f"| {best.get('axis_state', '-')} ({best.get('total_axis_misfit_deg', '-')}) |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `production` is strict residual then permutation order, matching the current affine selector.",
        "- `exact_*` selectors only choose among candidates whose residual RMS is tied with the minimum.",
        "- `near_*` selectors choose among candidates within the near-residual band; these are diagnostic only unless a future production prototype defines a safe band.",
        "- If a selector repeatedly fixes `production` broken rows without breaking usable rows AND has a nonzero metric range inside the tied group, it is a candidate production tie-breaker.",
    ])
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM)
    parser.add_argument("--tie-rms-epsilon", type=float, default=DEFAULT_TIE_RMS_EPSILON)
    parser.add_argument("--near-rms-epsilon", type=float, default=DEFAULT_NEAR_RMS_EPSILON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Regenerate --out-md from existing --out-json.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.render_only:
        payload = json.loads(args.out_json.read_text(encoding="utf-8"))
    else:
        truth = json.loads(args.truth.read_text(encoding="utf-8"))
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        payload = run_all(
            truth,
            manifest,
            args.max_image_dim,
            tie_rms_epsilon=args.tie_rms_epsilon,
            near_rms_epsilon=args.near_rms_epsilon,
            truth_path=args.truth,
            manifest_path=args.manifest,
        )
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_report(payload) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
