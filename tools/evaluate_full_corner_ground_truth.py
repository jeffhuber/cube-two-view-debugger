#!/usr/bin/env python3
"""Score canonical full-corner geometry against human full-corner truth.

This is the canonical replacement layer for legacy `near_*` geometry
evaluation. Truth rows come from `tests/fixtures/full_corner_ground_truth.json`
and use the explicit human labels:

    vertex, corner_0, corner_1, corner_2, corner_3, corner_4, corner_5

Candidate rows may use either the same full-corner schema or a model-style
triplet schema:

    {
      "vertex": [x, y],
      "one_edge": [[x, y], [x, y], [x, y]],
      "far": [[x, y], [x, y], [x, y]]
    }

Triplet scoring is best-permutation matched. That is intentional: current
global-model/cv-local diagnostics often know "these are the one-edge corners"
without a trusted h_x/h_y/h_z to human corner_1/corner_3/corner_5 identity.
Exact per-corner metrics are emitted only when a candidate provides
`corner_0..corner_5`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.corner_conventions import (  # noqa: E402
    FAR_CORNERS_BY_SIDE,
    ONE_EDGE_CORNERS_BY_SIDE,
    POINT_NAMES,
)


Point = Tuple[float, float]
DEFAULT_TRUTH = REPO_ROOT / "tests" / "fixtures" / "full_corner_ground_truth.json"
DEFAULT_TRUTH_LABEL = "tests/fixtures/full_corner_ground_truth.json"


def _side_from_key(key: str) -> str:
    side = key.rsplit("_", 1)[-1]
    if side not in ("A", "B"):
        raise ValueError(f"cannot infer side A/B from key: {key}")
    return side


def _point(raw: Iterable[Any]) -> Point:
    x, y = list(raw)[:2]
    return float(x), float(y)


def _round_point(point: Point) -> List[float]:
    return [round(float(point[0]), 1), round(float(point[1]), 1)]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bearing(origin: Point, target: Point) -> float:
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0])) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _length(origin: Point, target: Point) -> float:
    return _distance(origin, target)


def _truth_triplets(row: Dict[str, Any], side: str) -> Tuple[List[Point], List[Point]]:
    one_edge = [_point(row[name]) for name in ONE_EDGE_CORNERS_BY_SIDE[side]]
    far = [_point(row[name]) for name in FAR_CORNERS_BY_SIDE[side]]
    return one_edge, far


def _parse_candidate_triplets(candidate: Dict[str, Any], side: str) -> Optional[Tuple[Point, List[Point], List[Point]]]:
    if "vertex" not in candidate:
        return None
    vertex = _point(candidate["vertex"])

    if "one_edge" in candidate and "far" in candidate:
        one_edge = [_point(p) for p in candidate["one_edge"]]
        far = [_point(p) for p in candidate["far"]]
        if len(one_edge) == 3 and len(far) == 3:
            return vertex, one_edge, far
        return None

    if all(name in candidate for name in POINT_NAMES):
        one_edge, far = _truth_triplets(candidate, side)
        return vertex, one_edge, far

    visible = candidate.get("visible_corners")
    if isinstance(visible, dict):
        one_edge_keys = ("h_x", "h_y", "h_z")
        far_keys = ("h_xy", "h_xz", "h_yz")
        if all(k in visible for k in one_edge_keys + far_keys):
            one_edge = [_point(visible[k]) for k in one_edge_keys]
            far = [_point(visible[k]) for k in far_keys]
            return vertex, one_edge, far

    return None


def _triplet_match(
    candidate_origin: Point,
    truth_origin: Point,
    candidate_points: List[Point],
    truth_points: List[Point],
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for perm in itertools.permutations(range(3)):
        angle_errors = []
        length_errors = []
        point_errors = []
        for truth_index, candidate_index in enumerate(perm):
            candidate = candidate_points[candidate_index]
            truth = truth_points[truth_index]
            angle_errors.append(
                _angle_diff_deg(
                    _bearing(candidate_origin, candidate),
                    _bearing(truth_origin, truth),
                )
            )
            length_errors.append(
                _length(candidate_origin, candidate) - _length(truth_origin, truth)
            )
            point_errors.append(_distance(candidate, truth))
        score = sum(angle_errors) / 3.0
        if best is None or score < best["mean_angle_error_deg"]:
            best = {
                "mean_angle_error_deg": score,
                "max_angle_error_deg": max(angle_errors),
                "mean_abs_length_error_px": sum(abs(v) for v in length_errors) / 3.0,
                "mean_point_error_px": sum(point_errors) / 3.0,
                "max_point_error_px": max(point_errors),
                "permutation": list(perm),
            }
    if best is None:
        return {"error": "empty_triplet"}
    return {
        "mean_angle_error_deg": round(best["mean_angle_error_deg"], 2),
        "max_angle_error_deg": round(best["max_angle_error_deg"], 2),
        "mean_abs_length_error_px": round(best["mean_abs_length_error_px"], 1),
        "mean_point_error_px": round(best["mean_point_error_px"], 1),
        "max_point_error_px": round(best["max_point_error_px"], 1),
        "permutation": best["permutation"],
    }


def _phase_category(one_edge_error: float, far_error: float, swapped_error: float) -> str:
    aligned = (one_edge_error + far_error) / 2.0
    if aligned < 10.0:
        return "GOOD"
    if aligned < 25.0:
        return "MARGINAL"
    if swapped_error < 25.0:
        return "PHASE_SWAPPED"
    return "GEOMETRY_FAIL"


def _exact_corner_errors(truth: Dict[str, Any], candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not all(name in candidate for name in POINT_NAMES):
        return None
    errors = {
        name: round(_distance(_point(candidate[name]), _point(truth[name])), 1)
        for name in POINT_NAMES
    }
    corner_errors = [errors[f"corner_{i}"] for i in range(6)]
    return {
        "point_errors_px": errors,
        "mean_corner_error_px": round(sum(corner_errors) / len(corner_errors), 1),
        "max_corner_error_px": round(max(corner_errors), 1),
    }


def score_case(key: str, truth: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Score one candidate row against one canonical truth row."""
    side = _side_from_key(key)
    parsed = _parse_candidate_triplets(candidate, side)
    if parsed is None:
        return {"key": key, "side": side, "status": "malformed_candidate"}

    truth_vertex = _point(truth["vertex"])
    candidate_vertex, candidate_one_edge, candidate_far = parsed
    truth_one_edge, truth_far = _truth_triplets(truth, side)

    one_edge_score = _triplet_match(candidate_vertex, truth_vertex, candidate_one_edge, truth_one_edge)
    far_score = _triplet_match(candidate_vertex, truth_vertex, candidate_far, truth_far)
    swapped_one_edge_score = _triplet_match(candidate_vertex, truth_vertex, candidate_one_edge, truth_far)
    swapped_far_score = _triplet_match(candidate_vertex, truth_vertex, candidate_far, truth_one_edge)

    if "error" in one_edge_score or "error" in far_score:
        return {"key": key, "side": side, "status": "malformed_candidate"}

    swapped_mean = (
        swapped_one_edge_score["mean_angle_error_deg"]
        + swapped_far_score["mean_angle_error_deg"]
    ) / 2.0
    category = _phase_category(
        one_edge_score["mean_angle_error_deg"],
        far_score["mean_angle_error_deg"],
        swapped_mean,
    )

    result: Dict[str, Any] = {
        "key": key,
        "side": side,
        "status": "scored",
        "category": category,
        "vertex_error_px": round(_distance(candidate_vertex, truth_vertex), 1),
        "truth_vertex": _round_point(truth_vertex),
        "candidate_vertex": _round_point(candidate_vertex),
        "one_edge": one_edge_score,
        "far": far_score,
        "swapped_mean_angle_error_deg": round(swapped_mean, 2),
        "swapped_one_edge_to_truth_far": swapped_one_edge_score,
        "swapped_far_to_truth_one_edge": swapped_far_score,
    }
    exact = _exact_corner_errors(truth, candidate)
    if exact is not None:
        result["exact_full_corner"] = exact
    return result


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [row for row in rows if row.get("status") == "scored"]
    categories = Counter(row["category"] for row in scored)

    def _median(field: str) -> Optional[float]:
        values = [float(row[field]) for row in scored if field in row]
        return round(statistics.median(values), 2) if values else None

    def _median_nested(group: str, field: str) -> Optional[float]:
        values = [float(row[group][field]) for row in scored if group in row and field in row[group]]
        return round(statistics.median(values), 2) if values else None

    return {
        "n_rows": len(rows),
        "n_scored": len(scored),
        "category_counts": dict(categories),
        "median_vertex_error_px": _median("vertex_error_px"),
        "median_one_edge_angle_error_deg": _median_nested("one_edge", "mean_angle_error_deg"),
        "median_far_angle_error_deg": _median_nested("far", "mean_angle_error_deg"),
        "median_swapped_angle_error_deg": _median("swapped_mean_angle_error_deg"),
    }


def evaluate(truth: Dict[str, Any], candidates: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for key, truth_row in sorted(truth.items()):
        if not truth_row.get("approved", True):
            continue
        candidate = candidates.get(key)
        if candidate is None:
            rows.append({"key": key, "side": _side_from_key(key), "status": "missing_candidate"})
            continue
        rows.append(score_case(key, truth_row, candidate))
    return {
        "schema": "canonical_full_corner_eval_v1",
        "truth": DEFAULT_TRUTH_LABEL,
        "summary": summarize(rows),
        "by_case": {row["key"]: row for row in rows},
    }


def render_markdown(payload: Dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = list(payload["by_case"].values())
    category_summary = ", ".join(
        f"{name}: {count}" for name, count in sorted(summary["category_counts"].items())
    )
    lines = [
        "# Canonical full-corner geometry baseline",
        "",
        "Status: canonical full-corner evidence. This report uses",
        "`tests/fixtures/full_corner_ground_truth.json` and the `Va/Vb + 0..5`",
        "human convention from `tools/FULL_CORNER_LABELING.md`. It does not use",
        "legacy `near_*` semantics.",
        "",
        "## Summary",
        "",
        f"- Rows scored: {summary['n_scored']} / {summary['n_rows']}",
        f"- Categories: `{category_summary}`",
        f"- Median vertex error: `{summary['median_vertex_error_px']}` px",
        f"- Median one-edge angle error: `{summary['median_one_edge_angle_error_deg']}` deg",
        f"- Median far angle error: `{summary['median_far_angle_error_deg']}` deg",
        f"- Median swapped-phase angle error: `{summary['median_swapped_angle_error_deg']}` deg",
        "",
        "## Row Details",
        "",
        "| row | category | vertex px | one-edge deg | far deg | swapped deg | phase check | sep |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        if row.get("status") != "scored":
            lines.append(f"| `{row['key']}` | `{row['status']}` |  |  |  |  |  |  |")
            continue
        debug = row.get("candidate_debug", {})
        phase_check = debug.get("phase_check", "")
        phase_sep = debug.get("phase_darkness_separation", "")
        phase_sep_s = f"{float(phase_sep):.1f}" if isinstance(phase_sep, (int, float)) else ""
        lines.append(
            f"| `{row['key']}` | `{row['category']}` | "
            f"{row['vertex_error_px']:.1f} | "
            f"{row['one_edge']['mean_angle_error_deg']:.2f} | "
            f"{row['far']['mean_angle_error_deg']:.2f} | "
            f"{row['swapped_mean_angle_error_deg']:.2f} | "
            f"`{phase_check}` | {phase_sep_s} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidate.read_text(encoding="utf-8"))
    payload = evaluate(truth, candidates)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(payload), encoding="utf-8")

    if args.summary_only:
        print(json.dumps(payload["summary"], indent=2))
    else:
        print(render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
