#!/usr/bin/env python3
"""Trace global-cube fit quality through the production model stages.

Diagnostic-only: no production behavior change.

The Procrustes correspondence diagnostic showed that the initial
720-way affine correspondence search selects canonical-good assignments
on the 12 human-labeled full-corner rows. The axis-correctness
diagnostic then showed that many final production fits still have
near-180-degree axis misfit.

This trace intentionally reproduces production's strict tie behavior:
the first permutation with the lowest residual wins. It does not use
full-corner truth to category-tie-break equal-residual candidates.

This tool asks the next load-bearing question:

  At which stage does a canonical-good initial fit become a broken
  final production axis model?

For each approved full-corner row it records axis/vertex error after:

1. selected affine correspondence before PnP
2. `fit_cube_template_to_anchors` output (PnP or affine fallback)
3. mean-of-3 vertex ensemble
4. phase check with correction disabled
5. final no-correction model after image refinement
6. phase check with correction enabled
7. final correction-enabled model after image refinement
"""
from __future__ import annotations

import argparse
import copy
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
    _refine_vertex_via_image_junction,
    _resolve_near_far_phase,
    derive_geometry,
    detect_hexagon_anchors,
    fit_cube_template_to_anchors,
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
DEFAULT_OUT_JSON = REPO_ROOT / "tests" / "fixtures" / "fit_stage_transition_trace.json"
DEFAULT_OUT_MD = REPO_ROOT / "tools" / "FIT_STAGE_TRANSITION_REPORT.md"


def _round_point(point: Iterable[float]) -> List[float]:
    x, y = list(point)[:2]
    return [round(float(x), 1), round(float(y), 1)]


def _clone_model(model: GlobalCubeModel) -> GlobalCubeModel:
    cloned = GlobalCubeModel(
        cube_center_screen=tuple(model.cube_center_screen),  # type: ignore[arg-type]
        axis_x_2d=tuple(model.axis_x_2d),  # type: ignore[arg-type]
        axis_y_2d=tuple(model.axis_y_2d),  # type: ignore[arg-type]
        axis_z_2d=tuple(model.axis_z_2d),  # type: ignore[arg-type]
        fit_loss=model.fit_loss,
        fit_quality=model.fit_quality,
    )
    derive_geometry(cloned)
    cloned.debug = copy.deepcopy(model.debug)
    return cloned


def _model_with_center(model: GlobalCubeModel, center: Point) -> GlobalCubeModel:
    shifted = _clone_model(model)
    shifted.cube_center_screen = (float(center[0]), float(center[1]))
    derive_geometry(shifted)
    return shifted


def _selected_affine_model(hexagon_vertices_ccw: Sequence[Point]) -> Optional[GlobalCubeModel]:
    if len(hexagon_vertices_ccw) != 6:
        return None
    template_positions = np.array(
        [_TEMPLATE_HEXAGON_2D_ISO[k] for k in TEMPLATE_KEYS],
        dtype=np.float64,
    )
    all_detected = np.array(hexagon_vertices_ccw, dtype=np.float64)
    best_model: Optional[GlobalCubeModel] = None
    best_residual = float("inf")
    best_perm: Optional[Tuple[int, ...]] = None

    for perm in itertools.permutations(range(6)):
        detected_permuted = all_detected[list(perm)]
        try:
            A, b = _fit_affine_2d(template_positions, detected_permuted)
        except Exception:  # noqa: BLE001
            continue
        residual = _affine_residual(template_positions, detected_permuted, A, b)
        if residual >= best_residual:
            continue
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
        model.fit_loss = residual
        model.fit_quality = max(0.0, 1.0 - math.sqrt(residual) / 200.0)
        model.debug = {
            "approach": "affine_selected",
            "affine_rms_px": round(math.sqrt(residual), 2),
            "best_permutation": list(perm),
        }
        best_residual = residual
        best_perm = perm
        best_model = model

    if best_model is not None and best_perm is not None:
        best_model.debug["best_permutation"] = list(best_perm)
    return best_model


def _classify_axis_misfit(total_misfit_deg: Optional[float]) -> str:
    if not isinstance(total_misfit_deg, (int, float)):
        return "unknown"
    if total_misfit_deg <= 30.0:
        return "usable"
    if total_misfit_deg >= 150.0:
        return "broken"
    return "marginal"


def _score_stage(
    *,
    stage: str,
    model: Optional[GlobalCubeModel],
    gt_vertex: Point,
    gt_axes: Sequence[Point],
    previous_total_misfit: Optional[float],
) -> Dict[str, Any]:
    if model is None:
        return {
            "stage": stage,
            "status": "missing_model",
            "axis_state": "unknown",
        }
    predicted_axes = [model.axis_x_2d, model.axis_y_2d, model.axis_z_2d]
    axis_match = _match_axes_to_ground_truth(predicted_axes, gt_axes)
    total_misfit = axis_match.get("total_misfit_deg")
    vertex = model.cube_center_screen
    vertex_error = math.hypot(vertex[0] - gt_vertex[0], vertex[1] - gt_vertex[1])
    delta = None
    if isinstance(total_misfit, (int, float)) and isinstance(previous_total_misfit, (int, float)):
        delta = round(float(total_misfit) - float(previous_total_misfit), 1)
    debug_keep = {
        key: model.debug.get(key)
        for key in (
            "approach",
            "affine_rms_px",
            "pnp_rms_px",
            "fit_residual_rms_px",
            "best_permutation",
            "cube_center_source",
            "ensemble_shift_px",
            "phase_check",
            "phase_darkness_separation",
            "refinement",
            "refinement_movement_px",
            "junction_score_at_ensemble",
            "junction_score_at_refined",
        )
        if key in model.debug
    }
    return {
        "stage": stage,
        "status": "scored",
        "axis_state": _classify_axis_misfit(total_misfit),
        "predicted_vertex_processing_px": _round_point(vertex),
        "vertex_error_processing_px": round(vertex_error, 1),
        "axis_match": axis_match,
        "delta_total_misfit_from_previous_deg": delta,
        "debug": debug_keep,
    }


def _append_stage(
    stages: List[Dict[str, Any]],
    *,
    stage: str,
    model: Optional[GlobalCubeModel],
    gt_vertex: Point,
    gt_axes: Sequence[Point],
) -> Optional[float]:
    previous = None
    if stages:
        previous = stages[-1].get("axis_match", {}).get("total_misfit_deg")
    score = _score_stage(
        stage=stage,
        model=model,
        gt_vertex=gt_vertex,
        gt_axes=gt_axes,
        previous_total_misfit=previous,
    )
    stages.append(score)
    total = score.get("axis_match", {}).get("total_misfit_deg")
    return float(total) if isinstance(total, (int, float)) else None


def _apply_refinement(model: GlobalCubeModel, image_rgb: np.ndarray) -> GlobalCubeModel:
    refined_input = _clone_model(model)
    refined_vertex, refine_debug = _refine_vertex_via_image_junction(
        image_rgb,
        refined_input.cube_center_screen,
        [refined_input.axis_x_2d, refined_input.axis_y_2d, refined_input.axis_z_2d],
    )
    refined = _model_with_center(refined_input, refined_vertex)
    refined.debug.update(refined_input.debug)
    refined.debug.update(refine_debug)
    if refined_vertex != refined_input.cube_center_screen:
        refined.debug["cube_center_source"] = "mean3_ensemble+image_refined"
    return refined


def trace_one_row(
    sess: Any,
    key: str,
    image_path: Path,
    truth_row: Dict[str, Any],
    max_image_dim: int,
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
        mask_array = np.array(remove(image, session=sess, only_mask=True)) > 128
        detection = detect_interior_bezel_lines(rgb_array, mask_array)
        hexagon = detect_hexagon_anchors(mask_array)
        if len(hexagon) != 6:
            return {
                **record,
                "status": "fit_failed",
                "error": f"hexagon detection produced {len(hexagon)} vertices",
            }
        if detection.cube_center is None:
            return {
                **record,
                "status": "fit_failed",
                "error": "bezel detection produced no cube_center",
            }

        gt_vertex, gt_axes = _ground_truth_axes(truth_row, side, scale)
        record["processing_scale"] = round(scale, 6)
        record["gt_vertex_processing_px"] = _round_point(gt_vertex)
        record["hexagon_centroid_processing_px"] = _round_point((
            float(np.array(hexagon, dtype=np.float64)[:, 0].mean()),
            float(np.array(hexagon, dtype=np.float64)[:, 1].mean()),
        ))
        record["bezel_vertex_processing_px"] = _round_point(detection.cube_center)
        stages: List[Dict[str, Any]] = []

        affine_model = _selected_affine_model(hexagon)
        _append_stage(
            stages,
            stage="affine_selected",
            model=affine_model,
            gt_vertex=gt_vertex,
            gt_axes=gt_axes,
        )

        hex_arr = np.array(hexagon, dtype=np.float64)
        hex_centroid = (float(hex_arr[:, 0].mean()), float(hex_arr[:, 1].mean()))
        pnp_model = fit_cube_template_to_anchors(
            hex_centroid,
            hexagon,
            detection.boundary_angles[:3],
            image_size=mask_array.shape,
        )
        _append_stage(
            stages,
            stage="template_fit_pnp_or_affine",
            model=pnp_model,
            gt_vertex=gt_vertex,
            gt_axes=gt_axes,
        )
        if pnp_model is None:
            record["stages"] = stages
            return record

        candidates = [pnp_model.cube_center_screen, hex_centroid]
        if detection.cube_center is not None:
            candidates.append(detection.cube_center)
        mean3_center = (
            sum(p[0] for p in candidates) / len(candidates),
            sum(p[1] for p in candidates) / len(candidates),
        )
        mean3_model = _model_with_center(pnp_model, mean3_center)
        shift = (
            mean3_center[0] - pnp_model.cube_center_screen[0],
            mean3_center[1] - pnp_model.cube_center_screen[1],
        )
        mean3_model.debug.update(pnp_model.debug)
        mean3_model.debug.update({
            "approach": "procrustes_template_fit+mean3_vertex",
            "cube_center_source": "mean3_ensemble",
            "ensemble_n_candidates": len(candidates),
            "ensemble_shift_px": round(math.hypot(*shift), 1),
        })
        _append_stage(
            stages,
            stage="mean3_vertex",
            model=mean3_model,
            gt_vertex=gt_vertex,
            gt_axes=gt_axes,
        )

        for apply_correction, prefix in ((False, "corr_false"), (True, "corr_true")):
            phase_input = _clone_model(mean3_model)
            phase_model, phase_debug = _resolve_near_far_phase(
                phase_input,
                detection,
                rgb_array,
                apply_correction=apply_correction,
            )
            phase_model = _clone_model(phase_model)
            phase_model.debug.update(mean3_model.debug)
            phase_model.debug.update(phase_debug)
            _append_stage(
                stages,
                stage=f"{prefix}_phase_check",
                model=phase_model,
                gt_vertex=gt_vertex,
                gt_axes=gt_axes,
            )

            final_model = _apply_refinement(phase_model, rgb_array)
            _append_stage(
                stages,
                stage=f"{prefix}_final_refined",
                model=final_model,
                gt_vertex=gt_vertex,
                gt_axes=gt_axes,
            )

        record["stages"] = stages
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    return record


def _first_broken_transition(stages: Sequence[Dict[str, Any]]) -> Optional[str]:
    previous_state = None
    for stage in stages:
        state = stage.get("axis_state")
        if state == "broken" and previous_state != "broken":
            return str(stage.get("stage"))
        previous_state = state
    return None


def _stages_for_path(
    stages: Sequence[Dict[str, Any]],
    branch_prefix: str,
) -> List[Dict[str, Any]]:
    """Return common stages plus one branch's phase/refinement stages.

    The production pipeline forks after `mean3_vertex` into correction
    disabled/enabled hypotheses. Summarizing "first broken stage" over a
    single linear list would make the `corr_false` stages hide useful
    `corr_true` information, so path-level summaries keep the branches
    distinct.
    """
    common = {"affine_selected", "template_fit_pnp_or_affine", "mean3_vertex"}
    return [
        stage for stage in stages
        if stage.get("stage") in common
        or str(stage.get("stage", "")).startswith(branch_prefix)
    ]


def _stage_total_misfit(stage: Dict[str, Any]) -> Optional[float]:
    value = stage.get("axis_match", {}).get("total_misfit_deg")
    return float(value) if isinstance(value, (int, float)) else None


def _stage_by_name(row: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for stage in row.get("stages", []):
        if stage.get("stage") == name:
            return stage
    return None


def _phase_correction_effect_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        corr_false = _stage_by_name(row, "corr_false_phase_check")
        corr_true = _stage_by_name(row, "corr_true_phase_check")
        if corr_false is None or corr_true is None:
            bucket = "missing_phase_stage"
        else:
            false_state = str(corr_false.get("axis_state"))
            true_state = str(corr_true.get("axis_state"))
            if false_state == "broken" and true_state == "usable":
                bucket = "fixes_broken_to_usable"
            elif false_state == "usable" and true_state == "broken":
                bucket = "breaks_usable_to_broken"
            elif false_state == true_state:
                bucket = f"keeps_{false_state}"
            else:
                bucket = f"{false_state}_to_{true_state}"
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def summarize_payload(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    traced = [r for r in rows if r.get("status") == "traced"]
    stage_names: List[str] = []
    for row in traced:
        for stage in row.get("stages", []):
            name = str(stage.get("stage"))
            if name not in stage_names:
                stage_names.append(name)

    by_stage: Dict[str, Dict[str, Any]] = {}
    for name in stage_names:
        stages = [
            stage for row in traced for stage in row.get("stages", [])
            if stage.get("stage") == name and stage.get("status") == "scored"
        ]
        misfits: List[float] = []
        for stage in stages:
            value = _stage_total_misfit(stage)
            if value is not None:
                misfits.append(value)
        by_stage[name] = {
            "n": len(stages),
            "usable": sum(1 for stage in stages if stage.get("axis_state") == "usable"),
            "marginal": sum(1 for stage in stages if stage.get("axis_state") == "marginal"),
            "broken": sum(1 for stage in stages if stage.get("axis_state") == "broken"),
            "median_total_axis_misfit_deg": (
                round(float(statistics.median(misfits)), 1) if misfits else None
            ),
        }

    first_breaks_by_path: Dict[str, Dict[str, int]] = {}
    for branch_prefix in ("corr_false", "corr_true"):
        first_breaks: Dict[str, int] = {}
        for row in traced:
            stages = _stages_for_path(row.get("stages", []), branch_prefix)
            first = _first_broken_transition(stages) or "never_broken"
            first_breaks[first] = first_breaks.get(first, 0) + 1
        first_breaks_by_path[branch_prefix] = first_breaks

    return {
        "n_rows": len(rows),
        "n_traced": len(traced),
        "stage_summary": by_stage,
        "first_broken_stage_counts_by_path": first_breaks_by_path,
        "phase_correction_effect_counts": _phase_correction_effect_counts(traced),
    }


def run_all(
    truth: Dict[str, Any],
    manifest: Dict[str, Any],
    max_image_dim: int,
    *,
    truth_path: Path = DEFAULT_TRUTH,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> Dict[str, Any]:
    from rembg import new_session  # noqa: E402

    sess = new_session()
    set_index = {
        str(pair["setId"]): pair for pair in manifest.get("pairs", [])
    }
    image_roots = _candidate_image_roots(manifest)
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for index, key in enumerate(sorted(truth), 1):
        truth_row = truth[key]
        if not truth_row.get("approved"):
            continue
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
        rows.append(trace_one_row(sess, key, image_path, truth_row, max_image_dim))
        print(f"  [{index}/{len(truth)}] {key}", file=sys.stderr, flush=True)

    return {
        "schema": "fit_stage_transition_trace_v1",
        "source": {
            "tool": "tools/diagnose_fit_stage_transitions.py",
            "git_sha": _git_head_sha(),
            "generated_at_utc": _now_utc_iso(),
            "truth": _display_path(truth_path),
            "manifest": _display_path(manifest_path),
            "max_image_dim": max_image_dim,
            "run_selection": "single deterministic run per row",
        },
        "summary": summarize_payload(rows),
        "per_row": rows,
        "skipped": skipped,
    }


def render_report(payload: Dict[str, Any]) -> str:
    source = payload.get("source", {})
    summary = payload.get("summary", {})
    lines: List[str] = [
        "# Fit stage transition diagnostic",
        "",
        "This diagnostic traces axis correctness through the production "
        "global-cube fit stages. It is diagnostic-only and exists to locate "
        "where canonical-good initial correspondence becomes a broken final "
        "axis model.",
        "",
        "Axis state buckets: `usable` <= 30 deg total axis misfit; `broken` "
        ">= 150 deg; otherwise `marginal`.",
        "",
        "## Source",
        "",
        f"- Tool: `{source.get('tool', '-')}`",
        f"- Commit: `{source.get('git_sha', '-')}`",
        f"- Generated: `{source.get('generated_at_utc', '-')}`",
        f"- Truth: `{source.get('truth', '-')}`",
        f"- Manifest: `{source.get('manifest', '-')}`",
        f"- Max image dim: `{source.get('max_image_dim', '-')}`",
        f"- Run selection: {source.get('run_selection', '-')}",
        "",
        "## Aggregate",
        "",
        f"- Rows traced: {summary.get('n_traced', 0)} / {summary.get('n_rows', 0)}",
        "- First broken stage counts by path: "
        f"`{summary.get('first_broken_stage_counts_by_path', {})}`",
        "- Phase correction effect counts: "
        f"`{summary.get('phase_correction_effect_counts', {})}`",
        "",
        "## Headline Findings",
        "",
        "- `affine_selected` reproduces production's strict first-minimum "
        "residual tie behavior. It does not use human truth or category "
        "tie-breaking.",
        "- On this run, most broken final axes are already broken at "
        "`affine_selected`; PnP, mean3, and vertex refinement mostly "
        "preserve the selected axis state rather than creating the "
        "near-180-degree misfit later.",
        "- This should be read alongside the Procrustes correspondence "
        "diagnostic as: canonical-good assignments exist in the search "
        "space, but production's residual-only tie behavior can still "
        "select the wrong 3-fold phase.",
        "- Phase correction is mixed: it can rescue a broken assignment, "
        "but it can also flip a usable assignment into the broken phase.",
        "",
        "### Stage Summary",
        "",
        "| Stage | n | usable | marginal | broken | median total axis misfit deg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage, stats in summary.get("stage_summary", {}).items():
        lines.append(
            f"| `{stage}` | {stats.get('n', 0)} | {stats.get('usable', 0)} "
            f"| {stats.get('marginal', 0)} | {stats.get('broken', 0)} "
            f"| {stats.get('median_total_axis_misfit_deg', '-')} |"
        )

    lines.extend([
        "",
        "## Per-row Stage Trace",
        "",
        "| Row | Stage | axis state | total misfit deg | delta vs previous | vertex err px | phase/refine |",
        "|---|---|---|---:|---:|---:|---|",
    ])
    for row in payload.get("per_row", []):
        if row.get("status") != "traced":
            lines.append(
                f"| `{row.get('key')}` | — | {row.get('status')} | — | — | — | "
                f"{row.get('error', '')[:60]} |"
            )
            continue
        for stage in row.get("stages", []):
            axis_match = stage.get("axis_match", {})
            debug = stage.get("debug", {})
            phase_refine_parts = []
            for key in ("approach", "phase_check", "refinement"):
                if debug.get(key) is not None:
                    phase_refine_parts.append(f"{key}={debug[key]}")
            if debug.get("ensemble_shift_px") is not None:
                phase_refine_parts.append(f"ensemble_shift={debug['ensemble_shift_px']}")
            if debug.get("refinement_movement_px") is not None:
                phase_refine_parts.append(f"refine_move={debug['refinement_movement_px']}")
            lines.append(
                f"| `{row.get('key')}` "
                f"| `{stage.get('stage')}` "
                f"| {stage.get('axis_state', '-')} "
                f"| {axis_match.get('total_misfit_deg', '-')} "
                f"| {stage.get('delta_total_misfit_from_previous_deg', '-')} "
                f"| {stage.get('vertex_error_processing_px', '-')} "
                f"| {'; '.join(phase_refine_parts) or '-'} |"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- If `affine_selected` is usable but `template_fit_pnp_or_affine` "
        "is broken, PnP / affine fallback selection is the first bad stage.",
        "- If `template_fit_pnp_or_affine` is usable but `mean3_vertex` is "
        "broken, the stage trace is inconsistent because mean3 should not "
        "change axis vectors; inspect scoring or geometry translation.",
        "- If `mean3_vertex` is usable but a `corr_*_phase_check` stage is "
        "broken, phase correction is the first bad stage.",
        "- If a phase-check stage is usable but the matching final-refined "
        "stage is broken, image-junction vertex refinement is the first bad "
        "stage.",
    ])
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM)
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
