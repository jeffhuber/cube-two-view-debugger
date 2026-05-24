#!/usr/bin/env python3
"""Pipeline phase-parity diagnostic — answers Codex's 4 questions per row.

Per Codex's #250 review feedback after PR #251/#253: the chirality
detector is NOT the only culprit for one-edge/far phase parity swaps;
the whole correspondence + phase-correction pathway can produce them.
This diagnostic walks each of the 12 human-validated full-corner rows
through the pipeline and answers, for each row:

  Q1: Did initial correspondence choose ONE_EDGE or FAR?
  Q2: Did `_resolve_near_far_phase` flip?
  Q3: Did the flip improve or worsen canonical full-corner score?
  Q4: Which evidence would have selected the right parity?

Vocabulary note: this diagnostic uses "one-edge/far phase parity"
per Codex's vocabulary request on #250 (not "chirality").
"Chirality" is retained only when quoting the detector's
historical naming.

## Multi-run by default (Codex P1 on PR #255)

The recognizer pipeline is stochastic across runs (likely from ONNX
thread ordering in rembg + non-determinism in vertex refinement). A
single run gave headline counts that did NOT reproduce: Codex saw
`flip_hurt: 5/3/4` across reruns on the same head SHA. So this
diagnostic runs N trials per row (default N=5), records all per-run
traces, and reports DISTRIBUTIONS rather than single-shot numbers.
The report separates STABLE rows (all N runs agree on the headline
category) from UNSTABLE rows (runs disagree). Headline conclusions
quote the modal outcome + the stability fraction.

## "Phase-rewound" model — what this reconstruction means (Codex P2 on PR #255)

`fit_global_cube_model` internally calls `_resolve_near_far_phase(...,
apply_correction=True)` and then runs an image-based vertex refinement
on the (possibly-flipped) model. The pre-phase-correction model is
shadowed and not returned.

Rather than modify the production function to expose it, we
mathematically reconstruct what we call the **phase-rewound model**
from the post-pipeline output: take the final model and invert just
the 60° phase flip (when one was applied). The flip is:

  (ax, ay, az)  →  (ax+ay, ax+az, ay+az)

Inverting:

  old_ax = ( new_ax + new_ay - new_az) / 2
  old_ay = ( new_ax - new_ay + new_az) / 2
  old_az = (-new_ax + new_ay + new_az) / 2

**Important semantic caveat:** the phase-rewound model is NOT the same
as a clean `apply_phase_correction=False` pipeline run. Production
runs (1) initial Procrustes fit → (2) phase correction → (3) image-based
vertex refinement, in that order. The post-model has step 3 applied
on top of step 2's flipped axes. Our reconstruction inverts step 2's
flip but preserves step 3's vertex refinement — so the rewound axes
are positioned where vertex refinement settled them on the flipped
geometry. This is a useful inverse-axis probe (it tells us "what
canonical category does the post-pipeline output land in once you
mathematically undo the phase flip?") but it is NOT a faithful trace
of a hypothetical apply_phase_correction=False pipeline run.

A true pre-correction state would require either:
  (a) a small production-code change exposing
      `fit_global_cube_model(..., apply_phase_correction=False)`,
      then a second pipeline run with that arg, or
  (b) inlining ~100 lines of `fit_global_cube_model`'s internals
      in this diagnostic.

Both are out of scope for a diagnostic-only PR. This diagnostic
clearly labels its output as `phase_rewound_canonical_category` (not
`pre_canonical_category`) to avoid overclaiming.

When `phase_check ∈ {'correct', 'ambiguous_no_correction',
'flip_suggested_diagnostic_only'}` no flip was applied, so the
phase-rewound model IS the post model.

## Output

- Multi-run trace JSON: per-row, per-run records + aggregated
  distribution stats.
- Markdown report: aggregate distribution tables + per-row stability
  classification + findings flagged as STABLE vs PROVISIONAL based on
  how often the runs agree.

## Cost

~5-30 min × N runs wall time (rembg dominates at ~20-30s/image).
For N=5 default and 12 rows that's ~25-30 min.

Diagnostic-only — no production behavior change.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    score_case,
)
from tools.global_cube_model import (  # noqa: E402
    GlobalCubeModel,
    derive_geometry,
    fit_global_cube_model,
)
from tools.interior_bezel_detection import (  # noqa: E402
    detect_interior_bezel_lines,
)


DEFAULT_TRUTH = REPO_ROOT / "tests" / "fixtures" / "full_corner_ground_truth.json"
DEFAULT_OUT_JSON = (
    REPO_ROOT / "tests" / "fixtures" / "pipeline_phase_parity_trace.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "tools" / "PIPELINE_PHASE_PARITY_FAILURE_MODES.md"
)
DEFAULT_MAX_IMAGE_DIM = 1600


# --- Pre-correction reconstruction ---------------------------------------


def reconstruct_pre_correction_model(post_model: GlobalCubeModel) -> GlobalCubeModel:
    """Mathematically reconstruct the pre-correction model from the
    post-correction model, assuming a 60° phase flip was applied.

    The flip in `_resolve_near_far_phase`:
        new_axes = [vertex→old_h_xy, vertex→old_h_xz, vertex→old_h_yz]
    means:
        new_ax = old_ax + old_ay
        new_ay = old_ax + old_az
        new_az = old_ay + old_az
    Inverting (3 linear equations in 3 unknowns):
        old_ax = ( new_ax + new_ay - new_az) / 2
        old_ay = ( new_ax - new_ay + new_az) / 2
        old_az = (-new_ax + new_ay + new_az) / 2

    Only call this when `phase_check == 'corrected_60deg_flip'`.
    Otherwise pre_model == post_model.
    """
    nx = post_model.axis_x_2d
    ny = post_model.axis_y_2d
    nz = post_model.axis_z_2d
    old_ax = ((nx[0] + ny[0] - nz[0]) / 2.0, (nx[1] + ny[1] - nz[1]) / 2.0)
    old_ay = ((nx[0] - ny[0] + nz[0]) / 2.0, (nx[1] - ny[1] + nz[1]) / 2.0)
    old_az = ((-nx[0] + ny[0] + nz[0]) / 2.0, (-nx[1] + ny[1] + nz[1]) / 2.0)
    pre = GlobalCubeModel(
        cube_center_screen=post_model.cube_center_screen,
        axis_x_2d=old_ax,
        axis_y_2d=old_ay,
        axis_z_2d=old_az,
        fit_loss=post_model.fit_loss,
        fit_quality=post_model.fit_quality,
    )
    derive_geometry(pre)
    return pre


# --- Pipeline helpers (reused from baseline_full_corner_global_model) ----


def _processing_image(image_path: Path, max_image_dim: int):
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


def _model_candidate_dict(model: GlobalCubeModel) -> Dict[str, Any]:
    """Convert a model to the candidate dict shape that `score_case` expects."""
    visible = {
        name: [round(float(point[0]), 1), round(float(point[1]), 1)]
        for name, point in model.visible_corners.items()
    }
    one_edge = [visible[name] for name in ("h_x", "h_y", "h_z")]
    far = [visible[name] for name in ("h_xy", "h_xz", "h_yz")]
    return {
        "vertex": [
            round(float(model.cube_center_screen[0]), 1),
            round(float(model.cube_center_screen[1]), 1),
        ],
        "one_edge": one_edge,
        "far": far,
        "visible_corners": visible,
    }


def _scale_candidate_back(candidate: Dict[str, Any], factor: float) -> Dict[str, Any]:
    """Scale candidate points from processing coords back to original-image coords.
    `factor` is the multiplicative scale (e.g. 4032/1600 ≈ 2.52 if the image
    was downsized from 4032 to 1600)."""
    if factor == 1.0:
        return candidate

    def _scale_point(p):
        return [round(float(p[0]) * factor, 1), round(float(p[1]) * factor, 1)]

    scaled = dict(candidate)
    scaled["vertex"] = _scale_point(candidate["vertex"])
    scaled["one_edge"] = [_scale_point(p) for p in candidate["one_edge"]]
    scaled["far"] = [_scale_point(p) for p in candidate["far"]]
    scaled["visible_corners"] = {
        name: _scale_point(p) for name, p in candidate["visible_corners"].items()
    }
    return scaled


# --- Per-row pipeline trace ----------------------------------------------


def trace_one_row(
    sess: Any,
    key: str,
    image_path: Path,
    truth_row: Dict[str, Any],
    max_image_dim: int = DEFAULT_MAX_IMAGE_DIM,
) -> Dict[str, Any]:
    """Run the pipeline on one image and trace its phase-parity behavior.

    Returns a dict with:
      - key, side
      - status: 'traced' | 'error' | 'fit_failed'
      - post_model: candidate dict (the production-pipeline output)
      - pre_model:  candidate dict (reconstructed if a flip was applied)
      - phase_check: from the global-model debug
      - flip_applied: bool
      - canonical_post_category: e.g. 'GOOD', 'PHASE_SWAPPED'
      - canonical_pre_category:  ditto
      - score_delta_class: 'flip_helped' | 'flip_hurt' | 'no_change' | 'no_flip'
      - phase_debug: subset of relevant fields from model.debug
    """
    from rembg import remove  # noqa: E402

    record: Dict[str, Any] = {
        "key": key,
        "side": key.rsplit("_", 1)[-1],
        "status": "traced",
    }
    try:
        image, scale = _processing_image(image_path, max_image_dim)
        rgb_array = np.array(image)
        try:
            # Threshold the soft rembg mask at >128 to match the
            # production pipeline (`tools/baseline_full_corner_global_model.py`).
            # Without thresholding, low-alpha halos would change the
            # detected silhouette and the reported phase-parity outcome
            # would diverge from what the production fit actually sees.
            # (Codex P2 on PR #250-follow-up: trace must match the
            # pipeline it claims to characterize.)
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

        post_model = fit_global_cube_model(detection, rgb_array, mask_array)
        if post_model is None:
            record["status"] = "fit_failed"
            record["error"] = "fit_global_cube_model returned None"
            return record

        phase_debug = {
            k: post_model.debug.get(k)
            for k in (
                "phase_check",
                "phase_darkness_separation",
                "phase_mean_near_darkness",
                "phase_mean_far_darkness",
                "phase_near_line_darkness",
                "phase_far_line_darkness",
                "phase_axis_angle_errors_before_deg",
                "phase_axis_angle_errors_after_deg",
            )
            if k in post_model.debug
        }
        phase_check = phase_debug.get("phase_check")
        flip_applied = phase_check == "corrected_60deg_flip"

        # Phase-rewound model: if a flip was applied, mathematically
        # invert the flip (only). NOTE this is NOT the same as a clean
        # `apply_phase_correction=False` pipeline run — see module
        # docstring for the semantic caveat (vertex refinement was
        # applied to the flipped axes and is preserved here).
        phase_rewound_model = (
            reconstruct_pre_correction_model(post_model)
            if flip_applied
            else post_model
        )

        # Convert both to candidate dicts + scale back to original-image coords.
        scale_factor = 1.0 / scale
        post_candidate = _scale_candidate_back(
            _model_candidate_dict(post_model), scale_factor
        )
        phase_rewound_candidate = _scale_candidate_back(
            _model_candidate_dict(phase_rewound_model), scale_factor
        )

        post_score = score_case(key, truth_row, post_candidate)
        phase_rewound_score = score_case(key, truth_row, phase_rewound_candidate)

        post_cat = post_score.get("category", "?")
        phase_rewound_cat = phase_rewound_score.get("category", "?")

        # Score-delta classification — compares post (production output)
        # against phase_rewound (post with the flip mathematically undone).
        # Semantic note: this measures the NET effect of the phase flip
        # given the rest of the pipeline state (specifically, given the
        # vertex refinement that ran on the post-flip axes). It is NOT
        # quite "what would have happened with apply_phase_correction=False"
        # — see module docstring.
        if not flip_applied:
            score_delta_class = "no_flip"
        elif phase_rewound_cat == post_cat:
            score_delta_class = "flip_no_category_change"
        elif _is_better(post_cat, phase_rewound_cat):
            score_delta_class = "flip_helped"
        elif _is_better(phase_rewound_cat, post_cat):
            score_delta_class = "flip_hurt"
        else:
            score_delta_class = "flip_lateral"

        record.update({
            "post_canonical_category": post_cat,
            "phase_rewound_canonical_category": phase_rewound_cat,
            "phase_check": phase_check,
            "flip_applied": flip_applied,
            "score_delta_class": score_delta_class,
            "phase_debug": phase_debug,
            "post_score_summary": {
                "vertex_error_px": post_score.get("vertex_error_px"),
                "one_edge_mean_ang_deg": post_score.get("one_edge", {}).get(
                    "mean_angle_error_deg"
                ),
                "far_mean_ang_deg": post_score.get("far", {}).get(
                    "mean_angle_error_deg"
                ),
                "swapped_mean_ang_deg": post_score.get(
                    "swapped_mean_angle_error_deg"
                ),
            },
            "phase_rewound_score_summary": {
                "vertex_error_px": phase_rewound_score.get("vertex_error_px"),
                "one_edge_mean_ang_deg": phase_rewound_score.get("one_edge", {}).get(
                    "mean_angle_error_deg"
                ),
                "far_mean_ang_deg": phase_rewound_score.get("far", {}).get(
                    "mean_angle_error_deg"
                ),
                "swapped_mean_ang_deg": phase_rewound_score.get(
                    "swapped_mean_angle_error_deg"
                ),
            },
        })
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    return record


# Category ranking for "is A better than B" comparisons.
_CATEGORY_RANK = {
    "GOOD": 0,
    "MARGINAL": 1,
    "PHASE_SWAPPED": 2,
    "GEOMETRY_FAIL": 3,
}


def _is_better(cat_a: str, cat_b: str) -> bool:
    """Return True if `cat_a` represents a better outcome than `cat_b`."""
    return _CATEGORY_RANK.get(cat_a, 99) < _CATEGORY_RANK.get(cat_b, 99)


# --- Runner --------------------------------------------------------------


def _set_id_from_key(key: str) -> str:
    return key.rsplit("_", 1)[0]


def _side_from_key(key: str) -> str:
    return key.rsplit("_", 1)[-1]


def run_diagnostic(
    truth: Dict[str, Any],
    max_image_dim: int,
    n_runs: int = 5,
    truth_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the multi-run diagnostic on every approved truth row.

    For each row, runs the pipeline `n_runs` times and aggregates the
    per-run traces into a distribution. The recognizer pipeline is
    stochastic across runs (likely from ONNX thread ordering in rembg);
    a single-run diagnostic was found to be non-reproducible (Codex P1
    on PR #255), so we collect distributions and surface stable vs
    unstable rows in the report.
    """
    from rembg import new_session  # noqa: E402

    manifests = _load_manifests()
    sess = new_session("u2net")

    keys = sorted(k for k, row in truth.items() if row.get("approved", True))
    print(
        f"tracing {len(keys)} full-corner rows × {n_runs} runs each",
        file=sys.stderr, flush=True,
    )
    per_row: List[Dict[str, Any]] = []
    for index, key in enumerate(keys, 1):
        paths = _resolve_pair_paths(manifests, _set_id_from_key(key))
        if paths is None:
            per_row.append({
                "key": key, "n_runs": 0, "runs": [],
                "summary": {"status": "missing_image_pair"},
            })
            continue
        image_path = paths[0] if _side_from_key(key) == "A" else paths[1]
        runs: List[Dict[str, Any]] = []
        for run_idx in range(n_runs):
            record = trace_one_row(
                sess, key, image_path, truth[key], max_image_dim,
            )
            record["run_index"] = run_idx
            runs.append(record)
        row_summary = _aggregate_runs(runs)
        per_row.append({
            "key": key, "n_runs": n_runs, "runs": runs,
            "summary": row_summary,
        })
        stable_marker = "stable" if row_summary.get("is_stable") else "UNSTABLE"
        print(
            f"  [{index}/{len(keys)}] {key}: {stable_marker} "
            f"post_modal={row_summary.get('post_canonical_category_modal','?')} "
            f"delta_modal={row_summary.get('score_delta_class_modal','?')} "
            f"(post_dist={row_summary.get('post_canonical_category_dist', {})})",
            file=sys.stderr, flush=True,
        )

    summary = _summarize_multi_run(per_row)
    # Record the actual truth path used (Codex P2 on PR #255 round-2:
    # the source dict must match what was passed in, not a hardcoded
    # default that diverges silently when --truth points elsewhere).
    if truth_path is not None:
        try:
            truth_path_str = str(
                truth_path.resolve().relative_to(REPO_ROOT)
            )
        except ValueError:
            truth_path_str = str(truth_path)
    else:
        truth_path_str = "tests/fixtures/full_corner_ground_truth.json"
    return {
        "schema": "pipeline_phase_parity_trace_v2_multi_run",
        "source": {
            "diagnostic": "tools/diagnose_pipeline_phase_parity.py",
            "truth": truth_path_str,
            "max_image_dim": max_image_dim,
            "n_runs_per_row": n_runs,
        },
        "summary": summary,
        "per_row": per_row,
    }


def _aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the N per-run traces for one row into a distribution.

    A row is "stable" if all traced runs agree on
    `post_canonical_category` AND `score_delta_class`. Modal value is
    the most common across runs; ties broken by sort order.
    """
    traced = [r for r in runs if r.get("status") == "traced"]
    if not traced:
        return {
            "status": "all_runs_errored",
            "errors": [r.get("error") for r in runs],
        }
    post_cats = Counter(r["post_canonical_category"] for r in traced)
    phase_rewound_cats = Counter(
        r["phase_rewound_canonical_category"] for r in traced
    )
    phase_checks = Counter(r["phase_check"] for r in traced)
    delta_classes = Counter(r["score_delta_class"] for r in traced)

    def _modal(c: Counter) -> Optional[str]:
        """Most common element, with DETERMINISTIC tie-breaking by
        sort order of the key. `Counter.most_common(1)` would return
        whichever element was inserted first when tied — order-dependent
        and not what the report description promises. Codex P3 on PR #255
        round-2: sort by (-count, key) so ties resolve to the
        alphabetically/numerically smaller key consistently across runs."""
        if not c:
            return None
        return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    # Stability is tracked at two granularities:
    # - post_category_stable: do all runs agree on the FINAL outcome?
    #   This is the most important — if yes, the modal post-category is
    #   reliable as a representative value for this row.
    # - delta_class_stable: do all runs take the same PATH (no_flip vs
    #   flip_helped vs flip_hurt)? This is more sensitive because the
    #   detector can take different decisions across runs while still
    #   ending on the same final outcome.
    # - fully_stable: both AND all requested runs traced. A row with
    #   some errored runs cannot claim full stability even if the
    #   surviving runs all agree (Codex P2 on PR #255 round-2).
    all_runs_traced = len(runs) > 0 and len(traced) == len(runs)
    post_category_stable = len(post_cats) == 1 and all_runs_traced
    delta_class_stable = len(delta_classes) == 1 and all_runs_traced
    fully_stable = post_category_stable and delta_class_stable
    return {
        "n_traced": len(traced),
        "n_errored": len(runs) - len(traced),
        "post_canonical_category_dist": dict(post_cats),
        "phase_rewound_canonical_category_dist": dict(phase_rewound_cats),
        "phase_check_dist": dict(phase_checks),
        "score_delta_class_dist": dict(delta_classes),
        "post_canonical_category_modal": _modal(post_cats),
        "phase_rewound_canonical_category_modal": _modal(phase_rewound_cats),
        "phase_check_modal": _modal(phase_checks),
        "score_delta_class_modal": _modal(delta_classes),
        "post_category_stable": post_category_stable,
        "delta_class_stable": delta_class_stable,
        "fully_stable": fully_stable,
        # Backwards-compat alias for any reader expecting the old field.
        "is_stable": fully_stable,
    }


def _summarize_multi_run(per_row: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-row summaries across all rows.

    The aggregate counts use each row's MODAL outcome (most common across
    its N runs) — this collapses the per-row distribution to a single
    representative value for headline numbers, but the per-row
    distributions are preserved for any row-level claims.
    """
    rows_traced = [
        r for r in per_row
        if r["summary"].get("n_traced", 0) > 0
    ]
    post_cat_modal = Counter(
        r["summary"]["post_canonical_category_modal"] for r in rows_traced
        if r["summary"].get("post_canonical_category_modal") is not None
    )
    phase_rewound_cat_modal = Counter(
        r["summary"]["phase_rewound_canonical_category_modal"]
        for r in rows_traced
        if r["summary"].get("phase_rewound_canonical_category_modal") is not None
    )
    phase_check_modal = Counter(
        r["summary"]["phase_check_modal"] for r in rows_traced
        if r["summary"].get("phase_check_modal") is not None
    )
    delta_modal = Counter(
        r["summary"]["score_delta_class_modal"] for r in rows_traced
        if r["summary"].get("score_delta_class_modal") is not None
    )
    n_post_stable = sum(
        1 for r in rows_traced if r["summary"].get("post_category_stable")
    )
    n_delta_stable = sum(
        1 for r in rows_traced if r["summary"].get("delta_class_stable")
    )
    n_fully_stable = sum(
        1 for r in rows_traced if r["summary"].get("fully_stable")
    )
    return {
        "n_total_rows": len(per_row),
        "n_traced_rows": len(rows_traced),
        "n_post_category_stable_rows": n_post_stable,
        "n_delta_class_stable_rows": n_delta_stable,
        "n_fully_stable_rows": n_fully_stable,
        # Backwards-compat aliases for the old binary stable/unstable view.
        "n_stable_rows": n_fully_stable,
        "n_unstable_rows": len(rows_traced) - n_fully_stable,
        "post_canonical_category_modal_counts": dict(post_cat_modal),
        "phase_rewound_canonical_category_modal_counts": dict(
            phase_rewound_cat_modal
        ),
        "phase_check_modal_counts": dict(phase_check_modal),
        "score_delta_class_modal_counts": dict(delta_modal),
    }


# --- Markdown report -----------------------------------------------------


def render_report(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Pipeline phase-parity failure modes")
    lines.append("")
    lines.append(
        "Diagnostic-only. Walks each human-validated full-corner row "
        "through the recognizer pipeline and answers Codex's 4 questions "
        "(see `tools/diagnose_pipeline_phase_parity.py` docstring)."
    )
    lines.append("")
    summary = payload.get("summary", {})
    n_runs = payload.get("source", {}).get("n_runs_per_row", 1)
    n_traced = summary.get("n_traced_rows", 0)
    n_total = summary.get("n_total_rows", 0)
    n_post_stable = summary.get("n_post_category_stable_rows", 0)
    n_delta_stable = summary.get("n_delta_class_stable_rows", 0)
    n_fully_stable = summary.get("n_fully_stable_rows", 0)
    lines.append(
        f"Rows: {n_traced} traced / {n_total} total. Stability across "
        f"{n_runs} runs per row:"
    )
    lines.append("")
    lines.append(
        f"- **{n_post_stable}/{n_traced} post-category-stable** "
        f"(all runs agree on the FINAL outcome — the modal post-category "
        f"is reliable as a row-level summary)"
    )
    lines.append(
        f"- **{n_delta_stable}/{n_traced} delta-class-stable** "
        f"(all runs agree on the path the pipeline took — `flip_helped` "
        f"vs `flip_hurt` vs `no_flip`)"
    )
    lines.append(
        f"- **{n_fully_stable}/{n_traced} fully stable** (both)"
    )
    lines.append("")
    lines.append(
        "> ⚠️ **Pipeline non-determinism note (Codex P1 on PR #255).** "
        "The recognizer is stochastic across runs (likely ONNX thread "
        "ordering in rembg + vertex-refinement non-determinism). This "
        "diagnostic runs each row N times and reports DISTRIBUTIONS + "
        "modal values. Headline counts below quote MODAL per-row "
        "outcomes; row-level claims should be read against the per-row "
        "distribution table to know whether they are stable."
    )
    lines.append("")
    lines.append(
        "> ⚠️ **`phase_rewound` ≠ true pre-correction state "
        "(Codex P2 on PR #255).** The pre-flip model is reconstructed "
        "from the post-pipeline output by mathematically inverting the "
        "phase flip. But vertex refinement runs AFTER phase correction "
        "in production, so the reconstruction preserves vertex "
        "refinement that was tuned to the post-flip axes. This is a "
        "useful inverse-axis probe but is NOT the same as an "
        "`apply_phase_correction=False` pipeline run. See the module "
        "docstring for the semantic caveat."
    )
    lines.append("")

    # Aggregate counts (using modal per-row values)
    lines.append("## Aggregate (modal per-row outcomes)")
    lines.append("")
    lines.append(
        "Each row contributes its MODAL outcome across `n_runs` trials. "
        "Stable rows have one unanimous modal value; unstable rows pick "
        "the most-common run outcome (ties broken by sort order)."
    )
    lines.append("")
    lines.append("**Post-pipeline canonical category (production outcome):**")
    lines.append("")
    lines.append("| Category | Rows (modal) |")
    lines.append("|---|---:|")
    for cat, n in sorted(
        summary.get("post_canonical_category_modal_counts", {}).items(),
        key=lambda kv: -kv[1],
    ):
        lines.append(f"| `{cat}` | {n} |")
    lines.append("")
    lines.append(
        "**Phase-rewound canonical category** (caveat above):"
    )
    lines.append("")
    lines.append("| Category | Rows (modal) |")
    lines.append("|---|---:|")
    for cat, n in sorted(
        summary.get(
            "phase_rewound_canonical_category_modal_counts", {}
        ).items(),
        key=lambda kv: -kv[1],
    ):
        lines.append(f"| `{cat}` | {n} |")
    lines.append("")
    lines.append("**`phase_check` distribution (modal):**")
    lines.append("")
    lines.append("| phase_check | Rows (modal) |")
    lines.append("|---|---:|")
    for pc, n in sorted(
        summary.get("phase_check_modal_counts", {}).items(),
        key=lambda kv: -kv[1],
    ):
        lines.append(f"| `{pc}` | {n} |")
    lines.append("")
    lines.append(
        "**Score-delta classification** (modal; did the flip help, hurt, or no-op?):"
    )
    lines.append("")
    lines.append("| Class | Rows (modal) | Meaning |")
    lines.append("|---|---:|---|")
    delta_meaning = {
        "no_flip": "phase-correction made no change",
        "flip_no_category_change": "flip applied but category unchanged",
        "flip_helped": "flip moved category toward better",
        "flip_hurt": "flip moved category toward worse",
        "flip_lateral": "flip applied; same rank but different categories",
    }
    for cls, n in sorted(
        summary.get("score_delta_class_modal_counts", {}).items(),
        key=lambda kv: -kv[1],
    ):
        lines.append(f"| `{cls}` | {n} | {delta_meaning.get(cls, '?')} |")
    lines.append("")

    # Per-row table with stability
    lines.append("## Per-row trace (distributions over N runs)")
    lines.append("")
    lines.append(
        "| Key | Stable? | Post-category dist | phase_check dist | Score-delta dist |"
    )
    lines.append("|---|---|---|---|---|")
    for row in payload.get("per_row", []):
        s = row.get("summary", {})
        if s.get("status") == "missing_image_pair":
            lines.append(
                f"| `{row.get('key','?')}` | — | (missing_image_pair) | — | — |"
            )
            continue
        if s.get("status") == "all_runs_errored":
            err = (s.get("errors") or ["?"])[0]
            lines.append(
                f"| `{row.get('key','?')}` | — | (all runs errored: {str(err)[:60]}) | — | — |"
            )
            continue
        # Stability marker — distinguishes:
        #   ✓✓        fully stable (all runs traced + agree on post + delta)
        #   ✓         post-stable, delta varies (path varies; outcome stable)
        #   partial-errors  some runs errored; check n_errored / dists
        #   ✗ post-varies   surviving runs disagree on final post-category
        # The partial-errors tier (Codex P2 round-4 on PR #255) prevents
        # the marker rendering "post-varies" when actually the disagreement
        # is just one run errored and the rest unanimous.
        post_dist = s.get("post_canonical_category_dist", {})
        n_post_distinct = len(post_dist)
        if s.get("fully_stable"):
            stable = "✓✓"
        elif s.get("post_category_stable"):
            stable = "✓"
        elif s.get("n_errored", 0) > 0 and n_post_distinct <= 1:
            # Some runs errored but surviving runs (if any) agree.
            stable = f"partial-errors ({s['n_errored']} err)"
        else:
            stable = "**✗ post-varies**"
        def _fmt_dist(d: Dict[str, int]) -> str:
            return ", ".join(
                f"{k}:{v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])
            )
        lines.append(
            f"| `{row.get('key','?')}` | {stable} "
            f"| {_fmt_dist(s.get('post_canonical_category_dist', {}))} "
            f"| {_fmt_dist(s.get('phase_check_dist', {}))} "
            f"| {_fmt_dist(s.get('score_delta_class_dist', {}))} |"
        )
    lines.append("")

    # Q4 — phase_darkness_separation by canonical outcome, modal per row
    lines.append("## Q4: Which evidence would have selected the right parity?")
    lines.append("")
    lines.append(
        "Each row contributes its MEDIAN `phase_darkness_separation` "
        "across runs. Grouped by row's modal `post_canonical_category`."
    )
    lines.append("")
    seps_by_cat: Dict[str, List[float]] = defaultdict(list)
    for row in payload.get("per_row", []):
        s = row.get("summary", {})
        cat = s.get("post_canonical_category_modal")
        if cat is None:
            continue
        seps = [
            r.get("phase_debug", {}).get("phase_darkness_separation")
            for r in row.get("runs", [])
            if r.get("status") == "traced"
        ]
        seps = [v for v in seps if v is not None]
        if seps:
            seps_by_cat[cat].append(statistics.median(seps))
    lines.append("| Post canonical (modal) | n rows | sep median | sep min | sep max |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat, vals in sorted(seps_by_cat.items()):
        if not vals:
            continue
        lines.append(
            f"| `{cat}` | {len(vals)} | {statistics.median(vals):.1f} | "
            f"{min(vals):.1f} | {max(vals):.1f} |"
        )
    lines.append("")

    lines.append("## Findings & implications")
    lines.append("")
    post_phase_swapped = summary.get(
        "post_canonical_category_modal_counts", {}
    ).get("PHASE_SWAPPED", 0)
    phase_rewound_phase_swapped = summary.get(
        "phase_rewound_canonical_category_modal_counts", {}
    ).get("PHASE_SWAPPED", 0)
    flip_helped = summary.get("score_delta_class_modal_counts", {}).get(
        "flip_helped", 0
    )
    flip_hurt = summary.get("score_delta_class_modal_counts", {}).get(
        "flip_hurt", 0
    )
    no_flip = summary.get("score_delta_class_modal_counts", {}).get(
        "no_flip", 0
    )

    # Count actual disagreement vs partial-error rows separately
    # (Codex P2 round-5 on PR #255). A row with surviving-runs-unanimous
    # + some errored runs is NOT "disagreement on the final outcome";
    # it's a partial-error row. Conflating them inflates the apparent
    # outcome instability.
    n_post_disagreement = 0
    n_partial_errors = 0
    for r in payload.get("per_row", []):
        sm = r.get("summary", {})
        post_dist = sm.get("post_canonical_category_dist", {})
        if len(post_dist) > 1:
            n_post_disagreement += 1
        elif sm.get("n_errored", 0) > 0:
            n_partial_errors += 1
    caveat_parts = [
        f"{n_post_stable}/{n_traced} rows are post-category-stable across runs"
    ]
    if n_post_disagreement:
        caveat_parts.append(
            f"{n_post_disagreement} rows had run-to-run disagreement on the final outcome"
        )
    if n_partial_errors:
        caveat_parts.append(
            f"{n_partial_errors} rows had partial-errored runs (surviving runs may still agree)"
        )
    stability_caveat = "(" + "; ".join(caveat_parts) + ")"

    lines.append(
        f"1. **Phase-rewound = PHASE_SWAPPED on "
        f"{phase_rewound_phase_swapped}/{n_traced} rows (modal).** "
        f"After mathematically inverting any flip from the post-pipeline "
        f"output, these rows still land in PHASE_SWAPPED. This is an "
        f"approximate lower bound on \"how many rows the upstream "
        f"correspondence + vertex-refinement landed in the wrong "
        f"parity\" — but see the phase-rewound caveat above; the true "
        f"upstream-correspondence error rate would require a clean "
        f"apply_phase_correction=False pipeline run. {stability_caveat}."
    )
    lines.append("")
    if flip_helped or flip_hurt:
        lines.append(
            f"2. **Phase-correction's impact on canonical score (modal):** "
            f"helped {flip_helped} row(s), hurt {flip_hurt} row(s), "
            f"no flip applied on {no_flip} row(s). If `flip_hurt > "
            f"flip_helped`, the detector is on net actively creating "
            f"PHASE_SWAPPED outcomes that the initial correspondence "
            f"got right. {stability_caveat}."
        )
        lines.append("")
    lines.append(
        f"3. **End-state PHASE_SWAPPED count (modal): "
        f"{post_phase_swapped}/{n_traced}.** Compare to phase-rewound "
        f"count above — if they're similar, phase-correction isn't "
        f"making net progress."
    )
    lines.append("")

    lines.append("## Next-step recommendations")
    lines.append("")
    lines.append(
        "Based on the row-level evidence above, the fix surface is "
        "scoped per failure mode:"
    )
    lines.append("")
    lines.append(
        "- **Rows where pre=PHASE_SWAPPED, post=PHASE_SWAPPED, "
        "phase_check=`correct` or `ambiguous_no_correction`**: "
        "the correspondence picked FAR and the detector did not catch "
        "it. Fix surface: either (a) make correspondence pick ONE_EDGE "
        "more reliably, or (b) strengthen the detector to catch this "
        "subset. Look at `phase_darkness_separation` distribution for "
        "this subset to scope (b)."
    )
    lines.append(
        "- **Rows where pre=PHASE_SWAPPED, post=GOOD/MARGINAL, "
        "phase_check=`corrected_60deg_flip`**: the detector correctly "
        "caught and corrected an upstream FAR pick. This is the "
        "detector working as intended — preserve."
    )
    lines.append(
        "- **Rows where pre=GOOD/MARGINAL, post=PHASE_SWAPPED, "
        "phase_check=`corrected_60deg_flip`**: the detector flipped a "
        "correct fit into a wrong one. This is the inverted-polarity "
        "wrong-call mode from PR #250's diagnostic. Fix surface: gate "
        "the detector's polarity rule on a meta-signal that predicts "
        "when its assumption holds (PR #250 suggested "
        "`junction_score_at_ensemble`, but its categorization was "
        "provisional — re-evaluate under canonical truth here)."
    )
    lines.append("")
    lines.append(
        "Candidate fix paths from Codex's #250 review (in priority order, "
        "with the color-anchor caveat: do not use sticker colors sampled "
        "from already-wrong geometry as hard truth):"
    )
    lines.append("")
    lines.append(
        "1. **Carry both phase hypotheses forward and score**: instead "
        "of mutating one model post-hoc, run the correspondence with "
        "both phase parities, score each against orthogonal evidence "
        "(center-color consistency, white-up/yellow-up A/B convention, "
        "two-view A↔B flip constraint), pick the better-scoring one. "
        "Avoids the polarity-rule inversion risk entirely."
    )
    lines.append(
        "2. **Center-color consistency check**: even if interior "
        "stickers are sampled from wrong geometry, the visible-face "
        "CENTER stickers should still hit the same color across "
        "phase hypotheses for the correct one. Use this as a "
        "tie-breaker, not a primary signal."
    )
    lines.append(
        "3. **Two-view A↔B flip constraint**: if A and B fits both "
        "succeed under the documented 180°-camera-X flip convention, "
        "the relative parity between the two views is constrained. "
        "(Requires the canonicalization helper from #249.)"
    )
    lines.append("")

    return "\n".join(lines)


# --- CLI -----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    ap.add_argument(
        "--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM,
        help="Downscale images to this max dim before pipeline (use 0 for full res).",
    )
    ap.add_argument(
        "--out-json", type=Path, default=DEFAULT_OUT_JSON,
        help="Write the trace JSON here.",
    )
    ap.add_argument(
        "--out-md", type=Path, default=DEFAULT_OUT_MD,
        help="Write the Markdown report here.",
    )
    ap.add_argument(
        "--runs", type=int, default=5,
        help="Number of pipeline runs per row (multi-run; Codex P1 fix). Default 5.",
    )
    ap.add_argument(
        "--render-only", action="store_true",
        help="Skip pipeline runs; regenerate the report from --out-json.",
    )
    args = ap.parse_args()

    if args.render_only:
        # --render-only derives everything from the existing trace
        # payload — it must NOT require --truth to exist (Codex P3
        # round-3 on PR #255: report regeneration should work even
        # when the original truth file is unavailable).
        if not args.out_json.exists():
            print(
                f"error: --render-only requires existing --out-json at "
                f"{args.out_json}",
                file=sys.stderr,
            )
            return 1
        payload = json.loads(args.out_json.read_text(encoding="utf-8"))
        # Re-aggregate from per_row in case the on-disk summary uses
        # an older schema. The per_row records are the source of truth;
        # the summary is derived. Also recompute each row's summary so
        # any new aggregation fields (e.g. richer stability tracking)
        # are present.
        for row in payload.get("per_row", []):
            runs = row.get("runs", [])
            if runs:
                row["summary"] = _aggregate_runs(runs)
        payload["summary"] = _summarize_multi_run(
            payload.get("per_row", [])
        )
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(
            f"re-aggregated summary in {args.out_json}",
            file=sys.stderr,
        )
    else:
        if not args.truth.exists():
            print(
                f"error: truth fixture not found at {args.truth}",
                file=sys.stderr,
            )
            return 1
        truth = json.loads(args.truth.read_text(encoding="utf-8"))
        payload = run_diagnostic(
            truth, args.max_image_dim, n_runs=args.runs,
            truth_path=args.truth,
        )
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"wrote trace JSON to {args.out_json}", file=sys.stderr)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_report(payload))
    print(f"wrote Markdown report to {args.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
