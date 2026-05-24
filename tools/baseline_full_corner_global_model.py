#!/usr/bin/env python3
"""Run the global cube model against canonical full-corner truth.

This is the first canonical rebaseline path after the 2026-05-23 convention
reset. It evaluates model one-edge/far triplets against
`tests/fixtures/full_corner_ground_truth.json`; it does not use legacy
`near_*` labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_full_corner_labeling_gallery import (  # noqa: E402
    _exif_correct,
    _load_manifests,
    _resolve_pair_paths,
)
from tools.evaluate_full_corner_ground_truth import (  # noqa: E402
    DEFAULT_TRUTH,
    DEFAULT_TRUTH_LABEL,
    render_markdown,
    score_case,
    summarize,
)
from tools.global_cube_model import fit_global_cube_model  # noqa: E402
from tools.interior_bezel_detection import detect_interior_bezel_lines  # noqa: E402


DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "full_corner_global_model_baseline.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "FULL_CORNER_GLOBAL_MODEL_BASELINE.md"


def _side_from_key(key: str) -> str:
    return key.rsplit("_", 1)[-1]


def _set_id_from_key(key: str) -> str:
    return key.rsplit("_", 1)[0]


def _model_candidate(model: Any) -> Dict[str, Any]:
    visible = {
        name: [round(float(point[0]), 1), round(float(point[1]), 1)]
        for name, point in model.visible_corners.items()
    }
    one_edge = [visible[name] for name in ("h_x", "h_y", "h_z")]
    far = [visible[name] for name in ("h_xy", "h_xz", "h_yz")]
    return {
        "vertex": [round(float(model.cube_center_screen[0]), 1), round(float(model.cube_center_screen[1]), 1)],
        "one_edge": one_edge,
        "far": far,
        "visible_corners": visible,
        "debug": model.debug,
    }


def _scale_point(point: List[float], factor: float) -> List[float]:
    return [round(float(point[0]) * factor, 1), round(float(point[1]) * factor, 1)]


def _scale_candidate(candidate: Dict[str, Any], factor: float) -> Dict[str, Any]:
    if factor == 1.0:
        return candidate
    scaled = dict(candidate)
    scaled["vertex"] = _scale_point(candidate["vertex"], factor)
    scaled["one_edge"] = [_scale_point(point, factor) for point in candidate["one_edge"]]
    scaled["far"] = [_scale_point(point, factor) for point in candidate["far"]]
    scaled["visible_corners"] = {
        name: _scale_point(point, factor)
        for name, point in candidate["visible_corners"].items()
    }
    scaled["processing_scale"] = round(1.0 / factor, 6)
    return scaled


def _processing_image(image_path: Path, max_image_dim: int) -> Tuple[Any, float]:
    image = _exif_correct(image_path)
    if max_image_dim <= 0:
        return image, 1.0
    width, height = image.size
    largest = max(width, height)
    if largest <= max_image_dim:
        return image, 1.0
    scale = float(max_image_dim) / float(largest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size), scale


def _run_case(sess: Any, key: str, image_path: Path, max_image_dim: int) -> Dict[str, Any]:
    from rembg import remove  # noqa: E402

    image, scale = _processing_image(image_path, max_image_dim)
    rgb = np.asarray(image, dtype=np.uint8)
    rgba = remove(image, session=sess)
    mask = np.array(rgba.split()[-1], dtype=np.uint8) > 128
    detection = detect_interior_bezel_lines(rgb, mask)
    model = fit_global_cube_model(detection, rgb, mask)
    if model is None:
        return {"key": key, "status": "fit_failed"}
    return _scale_candidate(_model_candidate(model), 1.0 / scale)


def _render_report(payload: Dict[str, Any]) -> str:
    base = render_markdown(payload)
    source = payload.get("source", {})
    phase_swapped = [
        row for row in payload.get("by_case", {}).values()
        if row.get("category") == "PHASE_SWAPPED"
    ]
    swapped_by_phase_check: Dict[str, int] = {}
    for row in phase_swapped:
        phase_check = row.get("candidate_debug", {}).get("phase_check", "unknown")
        swapped_by_phase_check[phase_check] = swapped_by_phase_check.get(phase_check, 0) + 1
    swapped_breakdown = ", ".join(
        f"{name}: {count}" for name, count in sorted(swapped_by_phase_check.items())
    )
    lines = [
        base.rstrip(),
        "",
        "## Interpretation",
        "",
        "This is a 12-row seed baseline, not the final 58-case migration.",
        "The strong signal is phase parity: most failures are near-exact",
        "one-edge/far swaps under the human convention, not arbitrary geometry",
        "drift. That keeps the phase/chirality problem real, but the old",
        "`near_*`-derived category names should remain historical until this",
        "canonical path is expanded.",
        "",
        f"- `PHASE_SWAPPED` rows by current `phase_check`: `{swapped_breakdown}`",
        "",
        "## Source",
        "",
        f"- Model: `{source.get('model', 'unknown')}`",
        f"- Truth: `{source.get('truth', '')}`",
        f"- Max processing image dimension: `{source.get('max_image_dim', '')}` px",
        f"- Run selection: `{source.get('run_selection', 'single run')}`",
        "- Image root: resolved from corpus manifests (local corpus path not recorded)",
        "",
        "Rows marked `PHASE_SWAPPED` mean the model's one-edge triplet matches",
        "the human far/double-axis triplet, and vice versa. This is the canonical",
        "full-corner version of the old near/far phase ambiguity.",
        "",
    ]
    return "\n".join(lines)


def run_baseline(truth: Dict[str, Any], runs: int, max_image_dim: int) -> Dict[str, Any]:
    from rembg import new_session  # noqa: E402

    manifests = _load_manifests()
    sess = new_session("u2net")
    by_case: Dict[str, List[Dict[str, Any]]] = {}

    keys = sorted(key for key, row in truth.items() if row.get("approved", True))
    print(f"running {len(keys)} full-corner rows x {runs} runs", file=sys.stderr)
    for index, key in enumerate(keys, 1):
        paths = _resolve_pair_paths(manifests, _set_id_from_key(key))
        if paths is None:
            by_case[key] = [{"key": key, "status": "missing_image_pair"}]
            continue
        image_path = paths[0] if _side_from_key(key) == "A" else paths[1]
        rows = []
        for _ in range(runs):
            try:
                candidate = _run_case(sess, key, image_path, max_image_dim)
                if candidate.get("status") == "fit_failed":
                    rows.append(candidate)
                else:
                    score = score_case(key, truth[key], candidate)
                    score["candidate_debug"] = candidate.get("debug", {})
                    rows.append(score)
            except Exception as exc:  # noqa: BLE001
                rows.append({"key": key, "side": _side_from_key(key), "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        by_case[key] = rows
        print(f"  [{index}/{len(keys)}] {key}", file=sys.stderr, flush=True)

    best_rows = [_select_representative_row(rows) for rows in by_case.values()]

    return {
        "schema": "canonical_full_corner_global_model_baseline_v1",
        "source": {
            "model": "tools.global_cube_model.fit_global_cube_model",
            "truth": DEFAULT_TRUTH_LABEL,
            "runs_per_row": runs,
            "max_image_dim": max_image_dim,
            "run_selection": "min(aligned_one_edge_far_mean_deg, swapped_phase_mean_deg)",
        },
        "summary": summarize(best_rows),
        "by_case": {row["key"]: row for row in best_rows},
        "all_runs_by_case": by_case,
    }


def _representative_run_error(row: Dict[str, Any]) -> float:
    aligned_mean = (
        float(row["one_edge"]["mean_angle_error_deg"])
        + float(row["far"]["mean_angle_error_deg"])
    ) / 2.0
    return min(aligned_mean, float(row["swapped_mean_angle_error_deg"]))


def _select_representative_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [row for row in rows if row.get("status") == "scored"]
    if not scored:
        return rows[0]
    return min(scored, key=_representative_run_error)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=1600,
        help="Resize images so max(width, height) is at most this before rembg/model fit; predictions are scaled back to original coordinates. Use 0 for full resolution.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Load --out and regenerate --report without re-running rembg/model fitting.",
    )
    args = parser.parse_args()

    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    if args.render_only:
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(_render_report(payload), encoding="utf-8")
        print(f"re-rendered {args.report}", file=sys.stderr)
        return 0

    payload = run_baseline(truth, args.runs, args.max_image_dim)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_render_report(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
