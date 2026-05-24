#!/usr/bin/env python3
"""Diagnostic: can center-color/rectification evidence choose phase?

This is the production-shaped follow-up to
`tools/probe_center_color_phase_metric.py`.

The oracle probe proved the color metric on human-validated geometry:
if the visible faces are labeled correctly, the three center stickers
have a much lower CIELAB distance to their canonical WCA face colors
than either 120-degree cyclic relabeling.

This tool asks the production-shaped question without changing
production behavior, but with an important limitation: under production
geometry, center-color distance is contaminated by rectification quality.
If a model's face quads are badly fit, the "center" patch may contain
bezel or pixels from the wrong physical face, so a color win can mean
"less broken rectification" rather than "correct near/far phase."

1. Run the recognizer on each full-corner truth row with
   `apply_phase_correction=False` to get the raw unflipped hypothesis.
2. Build the detector-independent forced-flip hypothesis from that raw
   model by using the current far corners as the new one-edge axes.
3. Also run today's production behavior (`apply_phase_correction=True`)
   so the report can compare the center-color choice with the old
   darkness detector.
4. For each resulting model, rectify the three model face quads and
   sample only the center stickers.
5. Score the model's current face assignment by sum of CIELAB distance
   to canonical WCA center colors.
6. Pick unflipped vs forced-flip by lower center-color score and
   compare that choice with production and the canonical full-corner
   geometry score.

Diagnostic-only: writes JSON + Markdown reports; does not alter the
recognizer's runtime decision path.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import statistics
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import (  # noqa: E402
    CANONICAL_RGB,
    FACE_TO_COLOR,
    rgb_to_lab,
)
from tools.build_full_corner_labeling_gallery import (  # noqa: E402
    _load_manifests,
    _resolve_pair_paths,
)
from tools.corner_conventions import wca_face_by_slot  # noqa: E402
from tools.diagnose_pipeline_phase_parity import (  # noqa: E402
    _CATEGORY_RANK,
    _is_better,
    _scale_candidate_back,
    _set_id_from_key,
    _side_from_key,
    _processing_image,
    _model_candidate_dict,
)
from tools.evaluate_full_corner_ground_truth import score_case  # noqa: E402
from tools.global_cube_model import GlobalCubeModel, fit_global_cube_model  # noqa: E402
from tools.global_cube_model import derive_geometry  # noqa: E402
from tools.interior_bezel_detection import detect_interior_bezel_lines  # noqa: E402
from tools.rectify_faces import (  # noqa: E402
    DEFAULT_FACE_SIZE,
    extract_stickers_from_rectified,
    rectify_face,
)


DEFAULT_TRUTH = REPO_ROOT / "tests" / "fixtures" / "full_corner_ground_truth.json"
DEFAULT_OUT_JSON = REPO_ROOT / "runs" / "center_color_phase_gate_trace.json"
DEFAULT_OUT_MD = REPO_ROOT / "runs" / "center_color_phase_gate_report.md"
DEFAULT_MAX_IMAGE_DIM = 1600
DEFAULT_N_RUNS = 3
SLOTS: Tuple[str, str, str] = ("upper", "right", "front")

# `derive_geometry()` names the three visible model faces by axis pair.
# Under the canonical convention, these correspond to the view slots
# below; a near/far phase error cyclically relabels these slots, which is
# exactly what the center-color score is meant to detect.
MODEL_FACE_BY_SLOT: Dict[str, str] = {
    "upper": "face_yz",
    "right": "face_xz",
    "front": "face_xy",
}

_CANONICAL_LAB_BY_FACE: Dict[str, Tuple[float, float, float]] = {
    face: rgb_to_lab(CANONICAL_RGB[color])
    for face, color in FACE_TO_COLOR.items()
}


def _lab_distance(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    return (
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    ) ** 0.5


def _cycle_faces(faces: Sequence[str], shift: int) -> Tuple[str, ...]:
    n = len(faces)
    return tuple(faces[(i + shift) % n] for i in range(n))


def _score_assignment(
    center_labs: Sequence[Tuple[float, float, float]],
    face_assignment: Sequence[str],
) -> float:
    return sum(
        _lab_distance(lab, _CANONICAL_LAB_BY_FACE[face])
        for lab, face in zip(center_labs, face_assignment)
    )


def _median_rgb_from_model_center(
    image: Image.Image,
    model: GlobalCubeModel,
    model_face: str,
    face_size: int = DEFAULT_FACE_SIZE,
) -> Tuple[int, int, int]:
    if model_face not in model.face_quads:
        raise ValueError(f"model is missing face quad {model_face!r}")
    rectified = rectify_face(image, model.face_quads[model_face], face_size)
    center = extract_stickers_from_rectified(rectified)[1][1]
    return tuple(int(v) for v in center.rgb)


def score_model_center_colors(
    image: Image.Image,
    model: GlobalCubeModel,
    side: str,
    yaw_quarter_turns: int,
    *,
    face_size: int = DEFAULT_FACE_SIZE,
) -> Dict[str, Any]:
    """Score one model by center-color consistency.

    The `identity` score means "trust this model's current slot->face
    assignment." The cyclic scores show which 120-degree relabeling
    would better explain the observed center colors.
    """
    face_by_slot = wca_face_by_slot(side, yaw_quarter_turns)
    identity_assignment = tuple(face_by_slot[slot] for slot in SLOTS)
    centers: List[Dict[str, Any]] = []
    center_labs: List[Tuple[float, float, float]] = []
    for slot in SLOTS:
        model_face = MODEL_FACE_BY_SLOT[slot]
        rgb = _median_rgb_from_model_center(
            image, model, model_face, face_size=face_size
        )
        lab = rgb_to_lab(rgb)
        center_labs.append(lab)
        centers.append({
            "slot": slot,
            "model_face": model_face,
            "assigned_wca_face": face_by_slot[slot],
            "rgb": list(rgb),
            "lab": [round(float(v), 2) for v in lab],
        })

    hypotheses: Dict[str, Tuple[str, ...]] = {
        "identity": identity_assignment,
        "cyclic_120": _cycle_faces(identity_assignment, 1),
        "cyclic_240": _cycle_faces(identity_assignment, 2),
    }
    scores = {
        name: _score_assignment(center_labs, assignment)
        for name, assignment in hypotheses.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1])
    winner, winning_score = ranked[0]
    runner_up, runner_up_score = ranked[1]
    return {
        "identity_assignment": list(identity_assignment),
        "centers": centers,
        "hypothesis_assignments": {
            name: list(assignment) for name, assignment in hypotheses.items()
        },
        "hypothesis_scores": {
            name: round(score, 2) for name, score in scores.items()
        },
        "identity_score": round(scores["identity"], 2),
        "winning_hypothesis": winner,
        "winning_score": round(winning_score, 2),
        "runner_up_hypothesis": runner_up,
        "runner_up_score": round(runner_up_score, 2),
        "margin": round(runner_up_score - winning_score, 2),
    }


def force_phase_flip_model(model: GlobalCubeModel) -> GlobalCubeModel:
    """Return the detector-independent forced near/far flip hypothesis."""
    cx, cy = model.cube_center_screen
    far_keys = ("h_xy", "h_xz", "h_yz")
    if not all(key in model.visible_corners for key in far_keys):
        raise ValueError("model is missing far visible corners")
    far_positions = [model.visible_corners[key] for key in far_keys]
    new_axes = [(p[0] - cx, p[1] - cy) for p in far_positions]
    flipped = GlobalCubeModel(
        cube_center_screen=model.cube_center_screen,
        axis_x_2d=new_axes[0],
        axis_y_2d=new_axes[1],
        axis_z_2d=new_axes[2],
        fit_loss=model.fit_loss,
        fit_quality=model.fit_quality,
    )
    derive_geometry(flipped)
    flipped.debug.update(model.debug or {})
    flipped.debug["phase_check"] = "forced_60deg_flip_diagnostic"
    return flipped


def choose_by_center_identity_score(
    unflipped_score: Dict[str, Any],
    forced_flip_score: Dict[str, Any],
    *,
    min_margin: float = 1.0,
) -> Dict[str, Any]:
    """Pick between raw unflipped and forced-flip by current-label score."""
    unflipped = float(unflipped_score["identity_score"])
    forced_flip = float(forced_flip_score["identity_score"])
    delta = unflipped - forced_flip
    if abs(delta) < min_margin:
        choice = "tie"
    elif delta > 0:
        choice = "forced_flip"
    else:
        choice = "unflipped"
    return {
        "choice": choice,
        "identity_score_delta_unflipped_minus_forced_flip": round(delta, 2),
        "min_margin": min_margin,
    }


def _category_effect(
    selected_category: Optional[str],
    production_category: Optional[str],
) -> str:
    if selected_category is None or production_category is None:
        return "not_scored"
    if selected_category == production_category:
        return "same_as_production"
    if _is_better(selected_category, production_category):
        return "center_choice_would_help"
    return "center_choice_would_hurt"


def _best_category_label(categories: Dict[str, Optional[str]]) -> Optional[str]:
    valid = {
        label: cat for label, cat in categories.items()
        if cat is not None
    }
    if not valid:
        return None
    return sorted(
        valid.items(),
        key=lambda item: (_CATEGORY_RANK.get(item[1], 99), item[0]),
    )[0][0]


def trace_one_run(
    sess: Any,
    key: str,
    image_path: Path,
    truth_row: Dict[str, Any],
    *,
    max_image_dim: int = DEFAULT_MAX_IMAGE_DIM,
    face_size: int = DEFAULT_FACE_SIZE,
    min_margin: float = 1.0,
) -> Dict[str, Any]:
    from rembg import remove  # noqa: E402

    side = _side_from_key(key)
    record: Dict[str, Any] = {
        "key": key,
        "side": side,
        "status": "traced",
        "yaw_quarter_turns": int(truth_row.get("yaw_quarter_turns", 0)),
    }
    try:
        image, scale = _processing_image(image_path, max_image_dim)
        rgb_array = np.array(image)
        try:
            mask_array = (
                np.array(remove(image, session=sess, only_mask=True))
                > 128
            )
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = f"rembg failed: {type(exc).__name__}: {exc}"
            return record

        detection = detect_interior_bezel_lines(rgb_array, mask_array)
        if detection.cube_center is None:
            record["status"] = "fit_failed"
            record["error"] = "bezel detection produced no cube_center"
            return record

        unflipped_model = fit_global_cube_model(
            detection,
            rgb_array,
            mask_array,
            apply_phase_correction=False,
        )
        production_model = fit_global_cube_model(
            detection,
            rgb_array,
            mask_array,
            apply_phase_correction=True,
        )
        candidate_models: Dict[str, Optional[GlobalCubeModel]] = {
            "unflipped": unflipped_model,
            "forced_flip": (
                force_phase_flip_model(unflipped_model)
                if unflipped_model is not None else None
            ),
            "production": production_model,
        }

        candidates: Dict[str, Dict[str, Any]] = {}
        for label, model in candidate_models.items():
            if model is None:
                candidates[label] = {
                    "status": "fit_failed",
                    "error": "fit_global_cube_model returned None",
                }
                continue
            color_score = score_model_center_colors(
                image,
                model,
                side,
                record["yaw_quarter_turns"],
                face_size=face_size,
            )
            scale_factor = 1.0 / scale
            geometry_candidate = _scale_candidate_back(
                _model_candidate_dict(model), scale_factor
            )
            geometry_score = score_case(key, truth_row, geometry_candidate)
            candidates[label] = {
                "status": "scored",
                "candidate_kind": label,
                "phase_check": model.debug.get("phase_check"),
                "phase_darkness_separation": model.debug.get(
                    "phase_darkness_separation"
                ),
                "center_color": color_score,
                "geometry_category": geometry_score.get("category"),
                "geometry_score_summary": {
                    "vertex_error_px": geometry_score.get("vertex_error_px"),
                    "one_edge_mean_ang_deg": geometry_score.get(
                        "one_edge", {}
                    ).get("mean_angle_error_deg"),
                    "far_mean_ang_deg": geometry_score.get(
                        "far", {}
                    ).get("mean_angle_error_deg"),
                    "swapped_mean_ang_deg": geometry_score.get(
                        "swapped_mean_angle_error_deg"
                    ),
                },
            }

        record["candidates"] = candidates
        if any(c.get("status") != "scored" for c in candidates.values()):
            record["status"] = "partial_fit_failed"
            return record

        choice = choose_by_center_identity_score(
            candidates["unflipped"]["center_color"],
            candidates["forced_flip"]["center_color"],
            min_margin=min_margin,
        )
        record["center_choice"] = choice
        if choice["choice"] == "tie":
            selected_label = None
            selected_category = None
            effect = "center_choice_tie"
        else:
            selected_label = choice["choice"]
            selected_category = candidates[selected_label]["geometry_category"]
            effect = _category_effect(
                selected_category,
                candidates["production"]["geometry_category"],
            )
        categories = {
            label: c.get("geometry_category")
            for label, c in candidates.items()
            if label in ("unflipped", "forced_flip")
        }
        record["selected_candidate"] = selected_label
        record["selected_geometry_category"] = selected_category
        record["production_geometry_category"] = candidates[
            "production"
        ]["geometry_category"]
        record["production_phase_check"] = candidates["production"]["phase_check"]
        record["best_geometry_candidate"] = _best_category_label(categories)
        record["center_choice_effect_vs_production"] = effect
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    return record


def _aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    traced = [r for r in runs if r.get("status") == "traced"]
    if not traced:
        return {
            "status": "all_runs_untraced",
            "errors": [r.get("error") for r in runs],
        }

    center_choices = Counter(
        r.get("center_choice", {}).get("choice") for r in traced
    )
    effects = Counter(r.get("center_choice_effect_vs_production") for r in traced)
    production_categories = Counter(
        r.get("production_geometry_category") for r in traced
    )
    selected_categories = Counter(
        r.get("selected_geometry_category") for r in traced
    )

    def _modal(counter: Counter) -> Optional[str]:
        items = [(k, v) for k, v in counter.items() if k is not None]
        if not items:
            return None
        return sorted(items, key=lambda item: (-item[1], str(item[0])))[0][0]

    all_runs_traced = len(traced) == len(runs)
    fully_stable = (
        all_runs_traced
        and len(center_choices) == 1
        and len(effects) == 1
        and len(production_categories) == 1
    )
    margins = [
        abs(float(
            r.get("center_choice", {}).get(
                "identity_score_delta_unflipped_minus_forced_flip", 0.0
            )
        ))
        for r in traced
    ]
    return {
        "n_traced": len(traced),
        "n_untraced": len(runs) - len(traced),
        "center_choice_dist": dict(center_choices),
        "effect_vs_production_dist": dict(effects),
        "production_geometry_category_dist": dict(production_categories),
        "selected_geometry_category_dist": dict(selected_categories),
        "center_choice_modal": _modal(center_choices),
        "effect_vs_production_modal": _modal(effects),
        "production_geometry_category_modal": _modal(production_categories),
        "selected_geometry_category_modal": _modal(selected_categories),
        "median_abs_identity_score_delta": (
            round(statistics.median(margins), 2) if margins else None
        ),
        "fully_stable": fully_stable,
    }


def _summarize(per_row: List[Dict[str, Any]]) -> Dict[str, Any]:
    row_summaries = [r.get("summary", {}) for r in per_row]
    traced = [s for s in row_summaries if "center_choice_modal" in s]
    return {
        "n_total_rows": len(per_row),
        "n_traced_rows": len(traced),
        "n_fully_stable_rows": sum(1 for s in traced if s.get("fully_stable")),
        "center_choice_modal_counts": dict(Counter(
            s.get("center_choice_modal") for s in traced
        )),
        "effect_vs_production_modal_counts": dict(Counter(
            s.get("effect_vs_production_modal") for s in traced
        )),
        "production_geometry_category_modal_counts": dict(Counter(
            s.get("production_geometry_category_modal") for s in traced
        )),
        "selected_geometry_category_modal_counts": dict(Counter(
            s.get("selected_geometry_category_modal") for s in traced
        )),
    }


def run_diagnostic(
    truth: Dict[str, Any],
    *,
    truth_path: Optional[Path],
    rows_glob: str,
    n_runs: int,
    max_image_dim: int,
    face_size: int,
    min_margin: float,
) -> Dict[str, Any]:
    from rembg import new_session  # noqa: E402

    manifests = _load_manifests()
    sess = new_session("u2net")
    keys = [
        key for key, row in sorted(truth.items())
        if row.get("approved", True)
        and (not rows_glob or fnmatch.fnmatch(key, rows_glob))
    ]
    print(
        f"center-color phase diagnostic: {len(keys)} rows x {n_runs} runs",
        file=sys.stderr,
        flush=True,
    )
    per_row: List[Dict[str, Any]] = []
    for index, key in enumerate(keys, 1):
        paths = _resolve_pair_paths(manifests, _set_id_from_key(key))
        if paths is None:
            per_row.append({
                "key": key,
                "n_runs": 0,
                "runs": [],
                "summary": {"status": "missing_image_pair"},
            })
            continue
        image_path = paths[0] if _side_from_key(key) == "A" else paths[1]
        runs = []
        for run_index in range(n_runs):
            run = trace_one_run(
                sess,
                key,
                image_path,
                truth[key],
                max_image_dim=max_image_dim,
                face_size=face_size,
                min_margin=min_margin,
            )
            run["run_index"] = run_index
            runs.append(run)
        summary = _aggregate_runs(runs)
        per_row.append({
            "key": key,
            "n_runs": n_runs,
            "runs": runs,
            "summary": summary,
        })
        print(
            f"  [{index}/{len(keys)}] {key}: "
            f"choice={summary.get('center_choice_modal', '?')} "
            f"effect={summary.get('effect_vs_production_modal', '?')} "
            f"stable={summary.get('fully_stable', False)}",
            file=sys.stderr,
            flush=True,
        )

    source_truth = str(truth_path) if truth_path is not None else str(DEFAULT_TRUTH)
    try:
        source_truth = str(Path(source_truth).resolve().relative_to(REPO_ROOT))
    except ValueError:
        pass
    return {
        "schema": "center_color_phase_gate_trace_v1",
        "source": {
            "diagnostic": "tools/diagnose_center_color_phase_gate.py",
            "truth": source_truth,
            "rows_glob": rows_glob,
            "n_runs_per_row": n_runs,
            "max_image_dim": max_image_dim,
            "face_size": face_size,
            "min_margin": min_margin,
        },
        "summary": _summarize(per_row),
        "per_row": per_row,
    }


def _fmt_counter(counter: Dict[str, int]) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{k}:{v}" for k, v in sorted(counter.items()))


def render_report(payload: Dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines: List[str] = []
    lines.append("# Center-color phase gate diagnostic")
    lines.append("")
    lines.append(
        "Diagnostic-only. Runs the raw unflipped global model, builds a "
        "detector-independent forced-flip hypothesis, and asks whether "
        "center-color CIELAB distance would choose the better hypothesis. "
        "Today's production detector is reported separately as the "
        "baseline. Important: under production geometry this score is "
        "also a rectification/fit-quality signal, not a pure phase "
        "signal."
    )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    source = payload.get("source", {})
    lines.append(f"- Runs per row: {source.get('n_runs_per_row', '?')}")
    lines.append(f"- Rows: {summary.get('n_traced_rows', 0)} traced / {summary.get('n_total_rows', 0)} total")
    lines.append(f"- Fully stable rows: {summary.get('n_fully_stable_rows', 0)}")
    lines.append(
        f"- Center-choice modal counts: "
        f"{_fmt_counter(summary.get('center_choice_modal_counts', {}))}"
    )
    lines.append(
        f"- Effect vs production modal counts: "
        f"{_fmt_counter(summary.get('effect_vs_production_modal_counts', {}))}"
    )
    lines.append(
        f"- Production geometry modal counts: "
        f"{_fmt_counter(summary.get('production_geometry_category_modal_counts', {}))}"
    )
    lines.append(
        f"- Selected geometry modal counts: "
        f"{_fmt_counter(summary.get('selected_geometry_category_modal_counts', {}))}"
    )
    lines.append("")
    lines.append("## Per-row")
    lines.append("")
    lines.append(
        "| Row | Stable | Center choice | Effect vs production | "
        "Production cat | Selected cat | Median |delta| |"
    )
    lines.append("|---|---|---|---|---|---|---:|")
    for row in payload.get("per_row", []):
        s = row.get("summary", {})
        if "center_choice_modal" not in s:
            lines.append(
                f"| `{row.get('key')}` | - | - | {s.get('status', 'not_traced')} | - | - | - |"
            )
            continue
        stable = "yes" if s.get("fully_stable") else "no"
        lines.append(
            f"| `{row.get('key')}` | {stable} "
            f"| {s.get('center_choice_modal')} "
            f"| {s.get('effect_vs_production_modal')} "
            f"| {s.get('production_geometry_category_modal')} "
            f"| {s.get('selected_geometry_category_modal')} "
            f"| {s.get('median_abs_identity_score_delta')} |"
        )
    lines.append("")
    lines.append("## Reading The Table")
    lines.append("")
    lines.append(
        "`unflipped` is a real no-phase-correction pipeline run from the "
        "same mask and bezel detection. `forced_flip` is the explicit "
        "alternate near/far phase hypothesis built from that raw model. "
        "`production` is today's darkness-detector behavior. "
        "`center_choice_would_help` means the lower identity Lab score "
        "picked a candidate whose canonical geometry category is better "
        "than production on that run. It does not prove the win was caused "
        "by cleaner phase labeling; visual audits show some wins are "
        "broken-fit avoidance wins where one candidate's rectified faces "
        "are simply less distorted."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    ap.add_argument("--rows-glob", default="*", help="Key filter, e.g. '20_*' or '*_A'.")
    ap.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    ap.add_argument("--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM)
    ap.add_argument("--face-size", type=int, default=DEFAULT_FACE_SIZE)
    ap.add_argument("--min-margin", type=float, default=1.0)
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.n_runs <= 0:
        ap.error("--n-runs must be positive")
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    payload = run_diagnostic(
        truth,
        truth_path=args.truth,
        rows_glob=args.rows_glob,
        n_runs=args.n_runs,
        max_image_dim=args.max_image_dim,
        face_size=args.face_size,
        min_margin=args.min_margin,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_report(payload), encoding="utf-8")
    print(f"wrote {args.out_json}", file=sys.stderr)
    print(f"wrote {args.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
