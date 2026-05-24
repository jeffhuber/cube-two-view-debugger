#!/usr/bin/env python3
"""Measure how close the production fit's axis vectors are to the
oracle ground-truth axis directions, per (row, hypothesis).

Diagnostic-only. Built to answer: is axis-direction error a better
predictor of rectification quality than vertex-position error?

Per the visual audit on the rectification-quality dig that produced this tool:
  - 41_A corr_false has vertex err 237 px (worst in dataset) but
    its rectified face_xz is a clean 3x3 sticker grid.
  - 20_A corr_true has vertex err 114 px (mid-range) and produces
    broken parallelograms slashing across multiple cube faces.

Hypothesis: rectification cleanliness depends on the AXES landing
on real hexagon corners, not on vertex position per se. If we score
the model's 3 axes (axis_x_2d/y_2d/z_2d) against the 3 ground-truth
single-axis hex corners (per ONE_EDGE_CORNERS_BY_SIDE) and find that
axis misfit correlates more cleanly with visual quality than vertex
error does, then axis improvement is the next attack surface.

## What this tool does

For each oracle row:
1. Load image at processing resolution + rembg silhouette + bezel
   detection (same scaffold as the probe).
2. Fit the global cube model TWICE (apply_phase_correction=True
   and =False).
3. For each fitted model, extract:
   - cube_center_screen (the model's vertex in processing coords)
   - axis_x_2d, axis_y_2d, axis_z_2d (3 axis vectors from vertex)
4. Compute oracle ground-truth axes (scaled to processing coords):
   - oracle vertex = oracle's "vertex" point
   - For side A: 3 single-axis hex corners are corner_1, corner_3,
     corner_5 per ONE_EDGE_CORNERS_BY_SIDE
   - For side B: corner_0, corner_2, corner_4
   - oracle axis_i = corner_i_pos - vertex_pos
5. Match each predicted axis to its best ground-truth axis (min
   angle over the 3 GT candidates). Report per-axis angle errors +
   total axis misfit.
6. Cross-reference with vertex error + visual quality classification
   on the partial sample we already eyeballed.

Default outputs:
  tests/fixtures/axis_correctness_trace.json  trace (per-row + per-hypothesis)
  tools/AXIS_CORRECTNESS_REPORT.md            markdown summary

## CLI

  python3 tools/measure_axis_correctness.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import itertools
import json
import math
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.corner_conventions import ONE_EDGE_CORNERS_BY_SIDE  # noqa: E402
from tools.diagnose_pipeline_phase_parity import _processing_image  # noqa: E402
from tools.global_cube_model import fit_global_cube_model  # noqa: E402
from tools.interior_bezel_detection import detect_interior_bezel_lines  # noqa: E402


DEFAULT_TRUTH = REPO_ROOT / "tests" / "fixtures" / "full_corner_ground_truth.json"
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_OUT_JSON = (
    REPO_ROOT / "tests" / "fixtures" / "axis_correctness_trace.json"
)
DEFAULT_OUT_MD = REPO_ROOT / "tools" / "AXIS_CORRECTNESS_REPORT.md"
DEFAULT_MAX_IMAGE_DIM = 1600


def _file_sha256(path: Path) -> Optional[str]:
    """SHA-256 hex digest of `path` contents, or None on read error."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def _git_head_sha() -> Optional[str]:
    """Best-effort current commit SHA. Returns None if not in a checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:  # noqa: BLE001
        return None


def _now_utc_iso() -> str:
    """UTC timestamp at trace generation time."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _candidate_image_roots(manifest: Dict[str, Any]) -> List[Path]:
    roots: List[Path] = []
    for pair in manifest.get("pairs", []):
        for field in ("imageAPath", "imageBPath"):
            raw = pair.get(field)
            if not raw:
                continue
            parent = Path(raw).expanduser().parent
            if parent not in roots:
                roots.append(parent)
    home_corpus = Path.home() / "cube-corpus"
    if home_corpus not in roots:
        roots.append(home_corpus)
    return roots


def _find_corpus_side(root: Path, set_id: str, side: str) -> Optional[Path]:
    for pattern in (f"Set {set_id} - {side} -*", f"Set {set_id} - {side} *"):
        candidates = sorted(p for p in root.glob(pattern) if p.is_file())
        if candidates:
            return candidates[0]
    return None


def _resolve_image_path(
    raw_path: str,
    set_id: str,
    side: str,
    image_roots: Sequence[Path],
) -> Optional[Path]:
    path = Path(raw_path).expanduser()
    if path.exists():
        return path

    for root in image_roots:
        by_name = root / path.name
        if by_name.exists():
            return by_name
        by_pattern = _find_corpus_side(root, set_id, side)
        if by_pattern is not None:
            return by_pattern
    return None


# Partial visual-quality classification from the rectification-quality
# follow-up dig. These labels are subjective, partial visual evidence;
# the question is whether axis misfit is a cleaner predictor than
# vertex error.
_VISUAL_QUALITY_SAMPLES: Dict[str, str] = {
    "41_A:corr_true:face_yz": "decent",
    "41_A:corr_true:face_xz": "decent",
    "41_A:corr_true:face_xy": "marginal",
    "41_A:corr_false:face_yz": "decent",
    "41_A:corr_false:face_xz": "clean",
    "41_A:corr_false:face_xy": "clean",
    "20_A:corr_true:face_yz": "broken",
    "20_A:corr_true:face_xz": "broken",
    "20_A:corr_true:face_xy": "broken",
    "20_A:corr_false:face_yz": "marginal",
    "20_A:corr_false:face_xz": "decent",
    "20_A:corr_false:face_xy": "clean",
    "45_B:corr_true:face_xy": "broken",
    "45_B:corr_false:face_yz": "broken",
    "20_B:corr_true:face_xy": "broken",
}


# ---------------- axis math ----------------


def _angle_between(
    a: Tuple[float, float], b: Tuple[float, float]
) -> float:
    """Angle in degrees between two 2D vectors, [0, 180]."""
    ax, ay = a
    bx, by = b
    norm_a = math.hypot(ax, ay)
    norm_b = math.hypot(bx, by)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return float("nan")
    cos_theta = (ax * bx + ay * by) / (norm_a * norm_b)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def _length(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _match_axes_to_ground_truth(
    predicted: Sequence[Tuple[float, float]],
    ground_truth: Sequence[Tuple[float, float]],
) -> Dict[str, Any]:
    """Find the best bijection from predicted axes to ground-truth axes.

    Exhaustively enumerates all 3! assignments and chooses the minimum
    total angle error. `total_misfit_deg` is therefore the sum of the
    three matched per-axis angle errors, not a single-axis maximum.
    """
    best_total = float("inf")
    best_perm = None
    best_errors = None
    for perm in itertools.permutations(range(len(ground_truth))):
        errors = [
            _angle_between(predicted[i], ground_truth[perm[i]])
            for i in range(len(predicted))
        ]
        total = sum(errors)
        if total < best_total:
            best_total = total
            best_perm = perm
            best_errors = errors
    # Reorder GT lengths to match the assignment so that the i-th
    # `gt_axis_lengths_px` entry refers to the SAME matched axis as the
    # i-th `predicted_axis_lengths_px` and `per_axis_angle_errors_deg`
    # entry. Pre-fix, GT lengths were always in raw GT order, which is
    # only correct when `best_perm` is identity. For the non-identity
    # rows in the committed trace, the report's "compare predicted vs
    # GT axis length" guidance lined up the wrong pairs (Codex P2 on
    # PR #268 head 70f90ca).
    if best_perm is not None:
        gt_lengths_matched = [
            round(_length(ground_truth[best_perm[i]]), 1)
            for i in range(len(predicted))
        ]
    else:
        gt_lengths_matched = [round(_length(v), 1) for v in ground_truth]
    return {
        "assignment": list(best_perm) if best_perm else None,
        "per_axis_angle_errors_deg": [
            round(e, 1) for e in (best_errors or [])
        ],
        "total_misfit_deg": round(best_total, 1),
        "predicted_axis_lengths_px": [
            round(_length(v), 1) for v in predicted
        ],
        # In matched (predicted-axis) order — see comment above.
        "gt_axis_lengths_px": gt_lengths_matched,
        # Preserve the raw GT order too, for callers that want it.
        "gt_axis_lengths_px_raw_order": [
            round(_length(v), 1) for v in ground_truth
        ],
    }


def _scale_point(p: Tuple[float, float], factor: float) -> Tuple[float, float]:
    return (p[0] * factor, p[1] * factor)


def _ground_truth_axes(
    truth_row: Dict[str, Any],
    side: str,
    scale: float,
) -> Tuple[Tuple[float, float], List[Tuple[float, float]]]:
    """Return (vertex_in_processing_coords, [3 single-axis-corner
    vectors from vertex in processing coords])."""
    vertex_full = (float(truth_row["vertex"][0]), float(truth_row["vertex"][1]))
    vertex_proc = _scale_point(vertex_full, scale)
    single_axis_corners = ONE_EDGE_CORNERS_BY_SIDE[side]
    axes: List[Tuple[float, float]] = []
    for corner_name in single_axis_corners:
        corner_full = (
            float(truth_row[corner_name][0]),
            float(truth_row[corner_name][1]),
        )
        corner_proc = _scale_point(corner_full, scale)
        axes.append((
            corner_proc[0] - vertex_proc[0],
            corner_proc[1] - vertex_proc[1],
        ))
    return vertex_proc, axes


# ---------------- pipeline + measurement ----------------


def evaluate_one_row(
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
    }
    try:
        image, scale = _processing_image(image_path, max_image_dim)
        rgb_array = np.array(image)
        mask_array = (
            np.array(remove(image, session=sess, only_mask=True)) > 128
        )
        detection = detect_interior_bezel_lines(rgb_array, mask_array)
        if detection.cube_center is None:
            record["status"] = "fit_failed"
            record["error"] = "bezel detection produced no cube_center"
            return record
        gt_vertex_proc, gt_axes = _ground_truth_axes(truth_row, side, scale)
        record["gt_vertex_processing_px"] = [
            round(gt_vertex_proc[0], 1), round(gt_vertex_proc[1], 1),
        ]
        for apc, tag in ((True, "corr_true"), (False, "corr_false")):
            outcome: Dict[str, Any] = {"apply_phase_correction": apc}
            try:
                model = fit_global_cube_model(
                    detection, rgb_array, mask_array,
                    apply_phase_correction=apc,
                )
            except Exception as exc:  # noqa: BLE001
                outcome["error"] = f"{type(exc).__name__}: {exc}"
                record[tag] = outcome
                continue
            if model is None:
                outcome["error"] = "fit returned None"
                record[tag] = outcome
                continue
            predicted_vertex = model.cube_center_screen
            predicted_axes = [
                model.axis_x_2d, model.axis_y_2d, model.axis_z_2d,
            ]
            vertex_error_px = math.hypot(
                predicted_vertex[0] - gt_vertex_proc[0],
                predicted_vertex[1] - gt_vertex_proc[1],
            )
            axis_match = _match_axes_to_ground_truth(
                predicted_axes, gt_axes,
            )
            outcome.update({
                "phase_check": model.debug.get("phase_check"),
                "flip_applied": model.debug.get("phase_check") == "corrected_60deg_flip",
                "predicted_vertex_processing_px": [
                    round(predicted_vertex[0], 1),
                    round(predicted_vertex[1], 1),
                ],
                "vertex_error_processing_px": round(vertex_error_px, 1),
                "axis_match": axis_match,
            })
            record[tag] = outcome
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    return record


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
        pair["setId"]: pair for pair in manifest.get("pairs", [])
    }
    image_roots = _candidate_image_roots(manifest)
    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for key in sorted(truth):
        row = truth[key]
        if not row.get("approved"):
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
        image_path = _resolve_image_path(
            str(image_path_str), set_id, side, image_roots
        )
        if image_path is None:
            skipped.append({
                "key": key,
                "reason": f"image not found: {Path(str(image_path_str)).expanduser()}",
            })
            continue
        # SHA verification: when the manifest carries an expected SHA-256
        # for this image, refuse to trace if the resolved file's actual
        # SHA differs (Codex P2 #2 on PR #268 head 7d90d30). The fuzzy
        # path resolver _resolve_image_path can land on a same-named
        # file from a different corpus root, and without this check the
        # trace would be canonical-named but contain measurements from
        # the wrong pixels.
        expected_sha = pair.get(f"image{side}_sha256_expected")
        if expected_sha:
            actual_sha = _file_sha256(image_path)
            if actual_sha is None:
                skipped.append({
                    "key": key,
                    "reason": f"could not read {image_path} for SHA check",
                })
                continue
            if actual_sha != expected_sha:
                skipped.append({
                    "key": key,
                    "reason": (
                        f"image SHA mismatch at {image_path}: "
                        f"got {actual_sha[:12]}…, expected "
                        f"{expected_sha[:12]}…"
                    ),
                })
                continue
        record = evaluate_one_row(sess, key, image_path, row, max_image_dim)
        records.append(record)
    return {
        "schema": "axis_correctness_v1",
        "source": {
            "tool": "tools/measure_axis_correctness.py",
            "git_sha": _git_head_sha(),
            "generated_at_utc": _now_utc_iso(),
            "truth": _display_path(truth_path),
            "manifest": _display_path(manifest_path),
            "max_image_dim": max_image_dim,
            "run_selection": "single deterministic run per row/hypothesis",
        },
        "per_row": records,
        "skipped": skipped,
    }


# ---------------- analysis + reporting ----------------


def _classify_face_for_row_visual(key: str, tag: str) -> str:
    """Aggregate per-face visual labels for a (key, hypothesis) into
    a single bucket: clean / marginal / broken / unknown.
    Aggregation rule: 'broken' if any labeled face is broken; 'unknown'
    if the row has no labels or only a partial non-broken sample; 'clean'
    if all three faces are clean/decent; 'marginal' otherwise."""
    face_keys = [f"{key}:{tag}:{f}" for f in ("face_yz", "face_xz", "face_xy")]
    labels = [
        _VISUAL_QUALITY_SAMPLES[k] for k in face_keys
        if k in _VISUAL_QUALITY_SAMPLES
    ]
    if not labels:
        return "unknown"
    if any(l == "broken" for l in labels):
        return "broken"
    if len(labels) < len(face_keys):
        return "unknown"
    if all(l in ("clean", "decent") for l in labels):
        return "clean"
    return "marginal"


def render_report(payload: Dict[str, Any]) -> str:
    rows = payload.get("per_row", [])
    lines: List[str] = []
    lines.append("# Axis-correctness diagnostic")
    lines.append("")
    lines.append(
        "For each (row, hypothesis), measures how close the production "
        "fit's 3 axis vectors are to the oracle's ground-truth single-axis "
        "hex-corner directions (per `ONE_EDGE_CORNERS_BY_SIDE`). "
        "Cross-references with vertex error and partial visual quality "
        "labels from the rectification-quality follow-up dig."
    )
    source = payload.get("source", {})
    if source:
        lines.append("")
        lines.append("## Source")
        lines.append("")
        lines.append(f"- Tool: `{source.get('tool', '-')}`")
        if source.get("git_sha"):
            lines.append(f"- Commit: `{source['git_sha']}`")
        if source.get("generated_at_utc"):
            lines.append(f"- Generated: `{source['generated_at_utc']}`")
        lines.append(f"- Truth: `{source.get('truth', '-')}`")
        lines.append(f"- Manifest: `{source.get('manifest', '-')}`")
        lines.append(f"- Max image dim: `{source.get('max_image_dim', '-')}`")
        lines.append(f"- Run selection: {source.get('run_selection', '-')}")
    lines.append("")
    lines.append("## Per-row metrics")
    lines.append("")
    lines.append(
        "| Row | Hypothesis | flip? | vertex err px | total axis misfit ° | "
        "per-axis errors ° | predicted axis lengths px | gt axis lengths px | visual |"
    )
    lines.append(
        "|---|---|:-:|---:|---:|---|---|---|---|"
    )
    by_visual: Dict[str, List[Dict[str, Any]]] = {
        "clean": [], "marginal": [], "broken": [], "unknown": [],
    }
    for r in rows:
        if r.get("status") != "traced":
            lines.append(
                f"| `{r.get('key')}` | — | — | — | — | — | — | — | "
                f"ERR: {r.get('error', '?')[:40]} |"
            )
            continue
        for tag in ("corr_true", "corr_false"):
            o = r.get(tag, {})
            if "error" in o:
                lines.append(
                    f"| `{r['key']}` | {tag} | — | — | — | — | — | — | "
                    f"err: {o['error'][:30]} |"
                )
                continue
            axis_match = o.get("axis_match", {})
            visual = _classify_face_for_row_visual(r["key"], tag)
            by_visual.setdefault(visual, []).append({
                "key": r["key"], "tag": tag,
                "vertex_error_px": o.get("vertex_error_processing_px"),
                "total_misfit_deg": axis_match.get("total_misfit_deg"),
            })
            lines.append(
                f"| `{r['key']}` "
                f"| {tag} "
                f"| {'Y' if o.get('flip_applied') else 'N'} "
                f"| {o.get('vertex_error_processing_px', '-')} "
                f"| {axis_match.get('total_misfit_deg', '-')} "
                f"| {axis_match.get('per_axis_angle_errors_deg', '-')} "
                f"| {axis_match.get('predicted_axis_lengths_px', '-')} "
                f"| {axis_match.get('gt_axis_lengths_px', '-')} "
                f"| {visual} |"
            )
    lines.append("")
    lines.append("## Cross-reference: vertex error vs total axis misfit, by visual quality bucket")
    lines.append("")
    lines.append(
        "Only rows with a visual quality label are shown. Bucket counts: "
        + ", ".join(f"{k}={len(v)}" for k, v in by_visual.items() if v)
    )
    lines.append("")
    lines.append("| Visual | Count | Median vertex err px | Median axis misfit ° |")
    lines.append("|---|---:|---:|---:|")
    for visual in ("clean", "marginal", "broken"):
        items = by_visual.get(visual, [])
        if not items:
            continue
        verrs = [
            i["vertex_error_px"] for i in items
            if isinstance(i["vertex_error_px"], (int, float))
        ]
        misfits = [
            i["total_misfit_deg"] for i in items
            if isinstance(i["total_misfit_deg"], (int, float))
        ]
        lines.append(
            f"| {visual} "
            f"| {len(items)} "
            f"| {round(statistics.median(verrs), 1) if verrs else '-'} "
            f"| {round(statistics.median(misfits), 1) if misfits else '-'} |"
        )
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append(
        "- If `broken` rows have notably higher median axis misfit than "
        "`clean` rows (separation > say 20°), axis correctness is a "
        "useful predictor."
    )
    lines.append(
        "- If `broken` rows have similar axis misfit to `clean` rows, "
        "rectification breakage has another cause (e.g. axis-length "
        "error / non-Procrustes scale issues) not captured by angle."
    )
    lines.append(
        "- Low angle misfit is necessary but not sufficient for clean "
        "rectification; compare predicted vs GT axis lengths too."
    )
    lines.append(
        "- Compare to vertex error: which is the cleaner predictor?"
    )
    return "\n".join(lines)


# ---------------- CLI ----------------


def _is_default_path(path: Path, default: Path) -> bool:
    """True iff `path` resolves to the committed-artifact default. Checked
    per-path so a mixed run like `--out-md /tmp/x.md` (explicit) without
    `--out-json` (still default) still protects the JSON side (Codex P2
    #2 on PR #268, head 808fc10)."""
    return path.resolve() == default.resolve()


def _hypothesis_errors(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Find rows where row-level status is 'traced' but a hypothesis-level
    fit raised or returned None. `evaluate_one_row` records those as
    `corr_true.error` / `corr_false.error` while leaving the row status
    as 'traced' (so per-hypothesis failures don't taint a whole row's
    other working hypothesis). The blocker MUST surface these — without
    this scan, a default regeneration would happily write a trace with
    missing axis metrics for the failed hypothesis (Codex P2 #1 on PR
    #268, head 808fc10).

    Returns a flat list of "row:tag" strings for diagnostics.
    """
    out: List[str] = []
    for r in rows:
        if r.get("status") != "traced":
            continue
        for tag in ("corr_true", "corr_false"):
            outcome = r.get(tag)
            if isinstance(outcome, dict) and "error" in outcome:
                out.append(f"{r.get('key', '?')}:{tag}")
    return out


def _default_output_blocker(
    payload: Dict[str, Any],
    *,
    truth_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    max_image_dim: Optional[int] = None,
) -> Optional[str]:
    """Decide whether to refuse writes to default-committed output paths.

    Returns a reason string if writes should be refused, None otherwise.
    The committed canonical trace + report are tied to a SPECIFIC
    (truth, manifest, code) tuple. Any deviation in inputs would
    silently corrupt the canonical artifact — even a "successful"
    deviation that traces every row of an exploratory dataset is
    wrong, because the resulting JSON would have a different schema
    semantics (different rows, different images) under the canonical
    schema label.

    The blocker fires if ANY of:
    1. trace is empty / partial (zero traced, skipped, errored, or
       hypothesis-level errors)
    2. caller passed a non-default `--truth` (even if every row of
       that truth traced cleanly — could be a superset, replacement,
       or differently-curated set)
    3. caller passed a non-default `--manifest` (same reasoning;
       images come from a non-canonical source)

    Codex P2 #1 + #2 on PR #268 head 988715b: a non-default truth
    superset OR a non-default manifest containing all canonical set
    IDs would each bypass the previous count-based check.
    """
    rows = payload.get("per_row", [])
    skipped = payload.get("skipped", [])
    errored = [r for r in rows if r.get("status") != "traced"]
    traced_count = sum(1 for r in rows if r.get("status") == "traced")
    if traced_count == 0:
        return "no rows were traced"
    if skipped:
        return f"{len(skipped)} row(s) were skipped"
    if errored:
        return f"{len(errored)} row(s) failed during tracing"
    hypo_errs = _hypothesis_errors(rows)
    if hypo_errs:
        sample = ", ".join(hypo_errs[:3])
        suffix = f" (sample: {sample})" if hypo_errs else ""
        return f"{len(hypo_errs)} hypothesis fit(s) failed{suffix}"
    # Canonical-input check: the committed artifacts are tied to
    # DEFAULT_TRUTH + DEFAULT_MANIFEST. Any deviation in either input
    # means the resulting trace doesn't represent the canonical 12-row
    # corpus — refuse to write to default outputs regardless of how
    # successfully the alternate inputs traced.
    if truth_path is not None and truth_path.resolve() != DEFAULT_TRUTH.resolve():
        return (
            f"non-default --truth ({_display_path(truth_path)}); committed "
            f"artifacts are tied to {_display_path(DEFAULT_TRUTH)}"
        )
    if (
        manifest_path is not None
        and manifest_path.resolve() != DEFAULT_MANIFEST.resolve()
    ):
        return (
            f"non-default --manifest ({_display_path(manifest_path)}); "
            f"committed artifacts are tied to {_display_path(DEFAULT_MANIFEST)}"
        )
    # max_image_dim check: the trace's vertex/axis coords are in the
    # processing coordinate system, which scales with `max_image_dim`.
    # Any value other than DEFAULT_MAX_IMAGE_DIM would silently produce
    # a coordinate-incompatible artifact under the canonical schema
    # label. Codex P2 #1 on PR #268 head 7d90d30.
    if max_image_dim is not None and max_image_dim != DEFAULT_MAX_IMAGE_DIM:
        return (
            f"non-default --max-image-dim ({max_image_dim}); committed "
            f"artifacts are tied to {DEFAULT_MAX_IMAGE_DIM} (axis/vertex "
            f"coords are in processing-resolution px and don't compare "
            f"across dim settings)"
        )
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    ap.add_argument(
        "--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM,
    )
    args = ap.parse_args(list(argv) if argv is not None else None)
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload = run_all(
        truth,
        manifest,
        args.max_image_dim,
        truth_path=args.truth,
        manifest_path=args.manifest,
    )
    blocker = _default_output_blocker(
        payload,
        truth_path=args.truth,
        manifest_path=args.manifest,
        max_image_dim=args.max_image_dim,
    )
    blocked_paths: List[str] = []
    if blocker is not None:
        # Per-path: protect any output that resolves to a committed
        # default. Explicit non-default paths are always written.
        # Asymmetric --out-md /tmp/x.md without --out-json would still
        # clobber the default JSON otherwise.
        if _is_default_path(args.out_json, DEFAULT_OUT_JSON):
            blocked_paths.append(str(args.out_json))
        if _is_default_path(args.out_md, DEFAULT_OUT_MD):
            blocked_paths.append(str(args.out_md))
        if blocked_paths:
            print(
                f"refusing to overwrite default committed axis-correctness "
                f"output(s) because {blocker}; pass an explicit non-default "
                f"path for each protected output to bypass.\n"
                f"  protected: {', '.join(blocked_paths)}",
                file=sys.stderr,
            )
            return 2
    # No protected defaults blocked → write everything as before.
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_report(payload), encoding="utf-8")
    n_traced = sum(
        1 for r in payload["per_row"] if r.get("status") == "traced"
    )
    print(
        f"wrote {args.out_json} and {args.out_md} "
        f"({n_traced} rows traced)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
