#!/usr/bin/env python3
"""Instrument the 720-way Procrustes correspondence search.

Diagnostic-only: no production behavior change.

The current global cube model picks the detected-hexagon-to-template
assignment with the lowest affine residual. This tool reruns that same
6! search and scores every permutation against
`tests/fixtures/full_corner_ground_truth.json`.

The question is deliberately narrow:

  If production picks a phase-swapped or geometry-fail assignment, was a
  canonical-good assignment present in the 720 candidates and merely
  ranked lower by residual, or was no canonical assignment available?

That distinguishes a ranker/bias problem from a deeper model/hexagon
extraction problem.

The canonical categories here come from triplet angle scoring, not exact
vertex/corner point error. That is intentional: this layer is about the
correspondence assignment before PnP, phase correction, and vertex
refinement.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baseline_full_corner_global_model import _processing_image  # noqa: E402
from tools.build_full_corner_labeling_gallery import (  # noqa: E402
    _load_manifests,
    _resolve_pair_paths,
)
from tools.evaluate_full_corner_ground_truth import (  # noqa: E402
    DEFAULT_TRUTH,
    DEFAULT_TRUTH_LABEL,
    score_case,
)
from tools.global_cube_model import (  # noqa: E402
    _TEMPLATE_HEXAGON_2D_ISO,
    _affine_residual,
    _fit_affine_2d,
    detect_hexagon_anchors,
)


Point = Tuple[float, float]
TEMPLATE_KEYS: Tuple[str, ...] = ("h_x", "h_y", "h_z", "h_xy", "h_xz", "h_yz")
DEFAULT_OUT_JSON = Path("/tmp/procrustes_correspondence_trace.json")
DEFAULT_OUT_MD = Path("/tmp/procrustes_correspondence_report.md")
DEFAULT_MAX_IMAGE_DIM = 1600


def _set_id_from_key(key: str) -> str:
    return key.rsplit("_", 1)[0]


def _side_from_key(key: str) -> str:
    return key.rsplit("_", 1)[-1]


def _round_point(point: Iterable[float]) -> List[float]:
    x, y = list(point)[:2]
    return [round(float(x), 1), round(float(y), 1)]


def _category_rank(category: str) -> int:
    return {
        "GOOD": 0,
        "MARGINAL": 1,
        "PHASE_SWAPPED": 2,
        "GEOMETRY_FAIL": 3,
    }.get(category, 4)


def _aligned_mean(score: Dict[str, Any]) -> float:
    return (
        float(score["one_edge"]["mean_angle_error_deg"])
        + float(score["far"]["mean_angle_error_deg"])
    ) / 2.0


def _is_canonical_usable(record: Dict[str, Any]) -> bool:
    return record.get("category") in ("GOOD", "MARGINAL")


def _candidate_from_affine(
    A: np.ndarray,
    b: np.ndarray,
    *,
    scale_back_to_full: float,
) -> Dict[str, Any]:
    visible: Dict[str, List[float]] = {}
    for key in TEMPLATE_KEYS:
        pt = A @ np.array(_TEMPLATE_HEXAGON_2D_ISO[key], dtype=np.float64) + b
        visible[key] = _round_point(pt * scale_back_to_full)

    vertex = _round_point(b * scale_back_to_full)
    return {
        "vertex": vertex,
        "one_edge": [visible[name] for name in ("h_x", "h_y", "h_z")],
        "far": [visible[name] for name in ("h_xy", "h_xz", "h_yz")],
        "visible_corners": visible,
    }


def _permutation_records(
    key: str,
    truth_row: Dict[str, Any],
    hexagon_vertices_ccw: Sequence[Point],
    *,
    scale_back_to_full: float,
) -> List[Dict[str, Any]]:
    template_positions = np.array(
        [_TEMPLATE_HEXAGON_2D_ISO[k] for k in TEMPLATE_KEYS],
        dtype=np.float64,
    )
    all_detected = np.array(hexagon_vertices_ccw, dtype=np.float64)
    records: List[Dict[str, Any]] = []

    for perm in itertools.permutations(range(6)):
        detected_permuted = all_detected[list(perm)]
        try:
            A, b = _fit_affine_2d(template_positions, detected_permuted)
        except Exception as exc:  # noqa: BLE001
            records.append({
                "perm": list(perm),
                "status": "fit_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        residual_px2 = _affine_residual(template_positions, detected_permuted, A, b)
        candidate = _candidate_from_affine(
            A,
            b,
            scale_back_to_full=scale_back_to_full,
        )
        score = score_case(key, truth_row, candidate)
        aligned_mean = _aligned_mean(score)
        one_edge_total = round(
            float(score["one_edge"]["mean_angle_error_deg"]) * 3.0,
            2,
        )
        records.append({
            "perm": list(perm),
            "status": "scored",
            "residual_px2": round(residual_px2, 4),
            "residual_rms_px": round(math.sqrt(residual_px2), 3),
            "category": score["category"],
            "one_edge_total_axis_misfit_deg": one_edge_total,
            "aligned_mean_angle_deg": round(aligned_mean, 2),
            "swapped_mean_angle_deg": score["swapped_mean_angle_error_deg"],
            "one_edge_mean_angle_deg": score["one_edge"]["mean_angle_error_deg"],
            "far_mean_angle_deg": score["far"]["mean_angle_error_deg"],
            "detected_index_by_template": {
                name: int(perm[i]) for i, name in enumerate(TEMPLATE_KEYS)
            },
        })

    scored = [r for r in records if r.get("status") == "scored"]
    scored.sort(key=lambda r: (
        float(r["residual_px2"]),
        _category_rank(str(r["category"])),
        float(r["aligned_mean_angle_deg"]),
        list(r["perm"]),
    ))
    for rank, record in enumerate(scored, 1):
        record["residual_rank"] = rank
    return scored + [r for r in records if r.get("status") != "scored"]


def _best_by_residual(
    records: Sequence[Dict[str, Any]],
    predicate: Callable[[Dict[str, Any]], bool],
) -> Optional[Dict[str, Any]]:
    candidates = [
        r for r in records
        if r.get("status") == "scored" and predicate(r)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: int(r["residual_rank"]))


def _best_by_canonical_error(records: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = [r for r in records if r.get("status") == "scored"]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (
        float(r["aligned_mean_angle_deg"]),
        float(r["residual_px2"]),
    ))


def _best_by_axis_misfit(records: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = [r for r in records if r.get("status") == "scored"]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (
        float(r["one_edge_total_axis_misfit_deg"]),
        float(r["residual_px2"]),
    ))


def _compact_record(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if record is None:
        return None
    keep = (
        "perm",
        "residual_rank",
        "residual_rms_px",
        "category",
        "one_edge_total_axis_misfit_deg",
        "aligned_mean_angle_deg",
        "swapped_mean_angle_deg",
        "one_edge_mean_angle_deg",
        "far_mean_angle_deg",
        "detected_index_by_template",
    )
    return {k: record[k] for k in keep if k in record}


def summarize_row(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in records if r.get("status") == "scored"]
    if not scored:
        return {
            "status": "no_scored_permutations",
            "diagnosis": "no_scored_permutations",
        }
    selected = min(scored, key=lambda r: int(r["residual_rank"]))
    best_canonical_by_residual = _best_by_residual(scored, _is_canonical_usable)
    best_swapped_by_residual = _best_by_residual(
        scored,
        lambda r: r.get("category") == "PHASE_SWAPPED",
    )
    best_canonical_by_error = _best_by_canonical_error(scored)
    best_axis_by_misfit = _best_by_axis_misfit(scored)

    if _is_canonical_usable(selected):
        diagnosis = "residual_selects_canonical"
    elif best_canonical_by_residual is not None:
        diagnosis = "canonical_available_but_outranked"
    else:
        diagnosis = "canonical_absent_or_not_within_threshold"

    selected_rms = float(selected["residual_rms_px"])
    canonical_gap = None
    if best_canonical_by_residual is not None:
        canonical_gap = round(
            float(best_canonical_by_residual["residual_rms_px"]) - selected_rms,
            3,
        )
    axis_gap = None
    if best_axis_by_misfit is not None:
        axis_gap = round(
            float(best_axis_by_misfit["residual_rms_px"]) - selected_rms,
            3,
        )

    return {
        "status": "scored",
        "diagnosis": diagnosis,
        "n_scored_permutations": len(scored),
        "selected_by_residual": _compact_record(selected),
        "best_canonical_by_residual": _compact_record(best_canonical_by_residual),
        "best_phase_swapped_by_residual": _compact_record(best_swapped_by_residual),
        "best_canonical_by_error": _compact_record(best_canonical_by_error),
        "best_axis_by_misfit": _compact_record(best_axis_by_misfit),
        "canonical_residual_rms_gap_px": canonical_gap,
        "best_axis_residual_rms_gap_px": axis_gap,
        "category_counts": dict(Counter(str(r["category"]) for r in scored)),
        "top_by_residual": [_compact_record(r) for r in scored[:10]],
    }


def diagnose_from_hexagon(
    key: str,
    truth_row: Dict[str, Any],
    hexagon_vertices_ccw: Sequence[Point],
    *,
    processing_scale: float,
) -> Dict[str, Any]:
    if len(hexagon_vertices_ccw) != 6:
        return {
            "key": key,
            "status": "hexagon_failed",
            "n_hexagon_vertices": len(hexagon_vertices_ccw),
        }
    records = _permutation_records(
        key,
        truth_row,
        hexagon_vertices_ccw,
        scale_back_to_full=1.0 / processing_scale,
    )
    return {
        "key": key,
        "side": _side_from_key(key),
        "status": "traced",
        "processing_scale": round(processing_scale, 6),
        "hexagon_vertices_processing_px": [
            _round_point(p) for p in hexagon_vertices_ccw
        ],
        "summary": summarize_row(records),
        "permutations": records,
    }


def run_one_image(
    sess: Any,
    key: str,
    image_path: Path,
    truth_row: Dict[str, Any],
    max_image_dim: int,
) -> Dict[str, Any]:
    from rembg import remove  # noqa: E402

    image, scale = _processing_image(image_path, max_image_dim)
    rgba = remove(image, session=sess)
    mask = np.array(rgba.split()[-1], dtype=np.uint8) > 128
    hexagon = detect_hexagon_anchors(mask)
    record = diagnose_from_hexagon(
        key,
        truth_row,
        hexagon,
        processing_scale=scale,
    )
    record["image_path"] = str(image_path)
    return record


def run_all(
    truth: Dict[str, Any],
    *,
    rows_glob: str,
    max_image_dim: int,
) -> Dict[str, Any]:
    from fnmatch import fnmatch
    from rembg import new_session  # noqa: E402

    manifests = _load_manifests()
    sess = new_session("u2net")
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    keys = [
        key for key, row in sorted(truth.items())
        if row.get("approved", True) and fnmatch(key, rows_glob)
    ]

    for index, key in enumerate(keys, 1):
        paths = _resolve_pair_paths(manifests, _set_id_from_key(key))
        if paths is None:
            skipped.append({"key": key, "reason": "missing image pair"})
            continue
        image_path = paths[0] if _side_from_key(key) == "A" else paths[1]
        try:
            record = run_one_image(sess, key, image_path, truth[key], max_image_dim)
        except Exception as exc:  # noqa: BLE001
            record = {
                "key": key,
                "side": _side_from_key(key),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        rows.append(record)
        print(f"  [{index}/{len(keys)}] {key}", file=sys.stderr, flush=True)

    return {
        "schema": "procrustes_correspondence_trace_v1",
        "source": {
            "tool": "tools/diagnose_procrustes_correspondence.py",
            "truth": DEFAULT_TRUTH_LABEL,
            "max_image_dim": max_image_dim,
            "rows_glob": rows_glob,
            "search": "all 720 detected-hexagon-to-template permutations",
            "selection_metric": "minimum affine residual before PnP/phase correction",
        },
        "summary": summarize_payload(rows),
        "per_row": rows,
        "skipped": skipped,
    }


def summarize_payload(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    traced = [r for r in rows if r.get("status") == "traced"]
    diagnoses = Counter(
        r.get("summary", {}).get("diagnosis", "unknown") for r in traced
    )
    selected_categories = Counter(
        r.get("summary", {})
        .get("selected_by_residual", {})
        .get("category", "unknown")
        for r in traced
    )
    gaps = [
        r.get("summary", {}).get("canonical_residual_rms_gap_px")
        for r in traced
        if isinstance(r.get("summary", {}).get("canonical_residual_rms_gap_px"), (int, float))
    ]
    return {
        "n_rows": len(rows),
        "n_traced": len(traced),
        "diagnosis_counts": dict(diagnoses),
        "selected_category_counts": dict(selected_categories),
        "median_canonical_residual_rms_gap_px": (
            round(float(statistics.median(gaps)), 3) if gaps else None
        ),
    }


def render_report(payload: Dict[str, Any]) -> str:
    source = payload.get("source", {})
    summary = payload.get("summary", {})
    lines = [
        "# Procrustes correspondence diagnostic",
        "",
        "This diagnostic reruns the 720-way detected-hexagon-to-template "
        "correspondence search and scores every permutation against canonical "
        "full-corner truth. It asks whether canonical-good assignments are "
        "available but ranked below the minimum-residual assignment.",
        "",
        "Canonical categories are triplet-angle categories, not exact "
        "vertex/corner point-error categories. This keeps the diagnostic "
        "focused on correspondence assignment before PnP, phase correction, "
        "and vertex refinement. The table also includes the one-edge total "
        "axis misfit, matching the axis-correctness diagnostic's sum of "
        "three matched axis-angle errors.",
        "",
        "## Source",
        "",
        f"- Tool: `{source.get('tool', '-')}`",
        f"- Truth: `{source.get('truth', '-')}`",
        f"- Max image dim: `{source.get('max_image_dim', '-')}`",
        f"- Rows glob: `{source.get('rows_glob', '-')}`",
        f"- Search: {source.get('search', '-')}",
        f"- Selection metric: {source.get('selection_metric', '-')}",
        "",
        "## Aggregate",
        "",
        f"- Rows traced: {summary.get('n_traced', 0)} / {summary.get('n_rows', 0)}",
        f"- Diagnosis counts: `{summary.get('diagnosis_counts', {})}`",
        f"- Selected category counts: `{summary.get('selected_category_counts', {})}`",
        "- Median canonical residual RMS gap px: "
        f"`{summary.get('median_canonical_residual_rms_gap_px')}`",
        "",
        "## Per-row summary",
        "",
        "| Row | Selected category | Selected RMS px | Selected axis misfit deg | Best-axis rank | Best-axis misfit deg | Best-axis RMS gap px | Diagnosis |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("per_row", []):
        if row.get("status") != "traced":
            lines.append(
                f"| `{row.get('key')}` | {row.get('status')} "
                f"| - | - | - | - | - | {row.get('error', '')[:40]} |"
            )
            continue
        row_summary = row.get("summary", {})
        selected = row_summary.get("selected_by_residual") or {}
        best_axis = row_summary.get("best_axis_by_misfit") or {}
        lines.append(
            f"| `{row.get('key')}` "
            f"| {selected.get('category', '-')} "
            f"| {selected.get('residual_rms_px', '-')} "
            f"| {selected.get('one_edge_total_axis_misfit_deg', '-')} "
            f"| {best_axis.get('residual_rank', '-')} "
            f"| {best_axis.get('one_edge_total_axis_misfit_deg', '-')} "
            f"| {row_summary.get('best_axis_residual_rms_gap_px', '-')} "
            f"| {row_summary.get('diagnosis', '-')} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `residual_selects_canonical`: the current residual objective "
        "already selects a GOOD/MARGINAL full-corner assignment.",
        "- `canonical_available_but_outranked`: a canonical assignment "
        "exists in the 720 candidates, but residual ranks another "
        "permutation first. This points at a correspondence-ranking or "
        "bias problem.",
        "- `canonical_absent_or_not_within_threshold`: no permutation "
        "scores GOOD/MARGINAL against full-corner truth. This points at "
        "hexagon extraction, model shape, or the affine correspondence "
        "family itself.",
        "- This instruments the initial affine correspondence layer; "
        "downstream PnP, phase correction, and vertex refinement can still "
        "help or hurt later.",
    ])
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--rows-glob", default="*")
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
        payload = run_all(
            truth,
            rows_glob=args.rows_glob,
            max_image_dim=args.max_image_dim,
        )
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_report(payload) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
