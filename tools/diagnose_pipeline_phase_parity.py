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

## How pre-correction state is captured (no production code change)

`fit_global_cube_model` internally calls `_resolve_near_far_phase(...,
apply_correction=True)` and returns the post-correction model. The
pre-correction model is shadowed. To answer Q1+Q3 we need both states.

Rather than modify the production function, we reconstruct the
pre-correction model from the post-correction model when a flip
was applied. The flip is:

  (ax, ay, az)  →  (ax+ay, ax+az, ay+az)

i.e., the new axes are the displacements to the OLD far corners
(h_xy, h_xz, h_yz from the old vertex). Inverting:

  old_ax = (new_ax + new_ay - new_az) / 2
  old_ay = (new_ax - new_ay + new_az) / 2
  old_az = (-new_ax + new_ay + new_az) / 2

When phase_check ∈ {'correct', 'ambiguous_no_correction',
'flip_suggested_diagnostic_only'} no flip was applied, so
pre_model == post_model.

## Output

- Per-row Markdown table with pre/post canonical category, phase
  decision, score delta, failure-mode classification
- Aggregate failure-mode breakdown (where in the pipeline is the bug?)
- Cross-row observations on Q4 (what evidence correlates with right
  parity)

## Cost

~10-30 min wall time (rembg + global cube model fit on 12 images).
Output committed as `tests/fixtures/pipeline_phase_parity_trace.json`
+ rendered report `tools/PIPELINE_PHASE_PARITY_FAILURE_MODES.md`.

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

        # Pre-correction model: if a flip was applied, mathematically
        # reconstruct it; otherwise pre == post.
        pre_model = (
            reconstruct_pre_correction_model(post_model)
            if flip_applied
            else post_model
        )

        # Convert both to candidate dicts + scale back to original-image coords.
        scale_factor = 1.0 / scale
        post_candidate = _scale_candidate_back(
            _model_candidate_dict(post_model), scale_factor
        )
        pre_candidate = _scale_candidate_back(
            _model_candidate_dict(pre_model), scale_factor
        )

        post_score = score_case(key, truth_row, post_candidate)
        pre_score = score_case(key, truth_row, pre_candidate)

        post_cat = post_score.get("category", "?")
        pre_cat = pre_score.get("category", "?")

        # Score-delta classification
        if not flip_applied:
            score_delta_class = "no_flip"
        elif pre_cat == post_cat:
            score_delta_class = "flip_no_category_change"
        elif _is_better(post_cat, pre_cat):
            score_delta_class = "flip_helped"
        elif _is_better(pre_cat, post_cat):
            score_delta_class = "flip_hurt"
        else:
            score_delta_class = "flip_lateral"

        record.update({
            "post_canonical_category": post_cat,
            "pre_canonical_category": pre_cat,
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
            "pre_score_summary": {
                "vertex_error_px": pre_score.get("vertex_error_px"),
                "one_edge_mean_ang_deg": pre_score.get("one_edge", {}).get(
                    "mean_angle_error_deg"
                ),
                "far_mean_ang_deg": pre_score.get("far", {}).get(
                    "mean_angle_error_deg"
                ),
                "swapped_mean_ang_deg": pre_score.get(
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
) -> Dict[str, Any]:
    """Run the diagnostic on every approved truth row."""
    from rembg import new_session  # noqa: E402

    manifests = _load_manifests()
    sess = new_session("u2net")

    keys = sorted(k for k, row in truth.items() if row.get("approved", True))
    print(f"tracing {len(keys)} full-corner rows", file=sys.stderr, flush=True)
    rows: List[Dict[str, Any]] = []
    for index, key in enumerate(keys, 1):
        paths = _resolve_pair_paths(manifests, _set_id_from_key(key))
        if paths is None:
            rows.append({"key": key, "status": "missing_image_pair"})
            continue
        image_path = paths[0] if _side_from_key(key) == "A" else paths[1]
        record = trace_one_row(sess, key, image_path, truth[key], max_image_dim)
        rows.append(record)
        cat_info = (
            f"{record.get('post_canonical_category', '?'):<14}"
            f"  pre={record.get('pre_canonical_category', '?'):<14}"
            f"  delta={record.get('score_delta_class', '?')}"
        )
        print(f"  [{index}/{len(keys)}] {key}: {cat_info}", file=sys.stderr, flush=True)

    summary = _summarize(rows)
    return {
        "schema": "pipeline_phase_parity_trace_v1",
        "source": {
            "diagnostic": "tools/diagnose_pipeline_phase_parity.py",
            "truth": "tests/fixtures/full_corner_ground_truth.json",
            "max_image_dim": max_image_dim,
        },
        "summary": summary,
        "rows": rows,
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    traced = [r for r in rows if r.get("status") == "traced"]
    post_cat_counts = Counter(r.get("post_canonical_category", "?") for r in traced)
    pre_cat_counts = Counter(r.get("pre_canonical_category", "?") for r in traced)
    phase_check_counts = Counter(r.get("phase_check", "?") for r in traced)
    delta_counts = Counter(r.get("score_delta_class", "?") for r in traced)
    return {
        "n_total": len(rows),
        "n_traced": len(traced),
        "post_canonical_category_counts": dict(post_cat_counts),
        "pre_canonical_category_counts": dict(pre_cat_counts),
        "phase_check_counts": dict(phase_check_counts),
        "score_delta_class_counts": dict(delta_counts),
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
    lines.append(
        f"Rows: {summary.get('n_traced', 0)} traced / "
        f"{summary.get('n_total', 0)} total"
    )
    lines.append("")

    # Aggregate counts
    lines.append("## Aggregate")
    lines.append("")
    lines.append("**Post-pipeline canonical category (production outcome):**")
    lines.append("")
    lines.append("| Category | Rows |")
    lines.append("|---|---:|")
    for cat, n in sorted(
        summary.get("post_canonical_category_counts", {}).items(),
        key=lambda kv: -kv[1],
    ):
        lines.append(f"| `{cat}` | {n} |")
    lines.append("")
    lines.append(
        "**Pre-correction canonical category (initial correspondence output):**"
    )
    lines.append("")
    lines.append("| Category | Rows |")
    lines.append("|---|---:|")
    for cat, n in sorted(
        summary.get("pre_canonical_category_counts", {}).items(),
        key=lambda kv: -kv[1],
    ):
        lines.append(f"| `{cat}` | {n} |")
    lines.append("")
    lines.append("**`phase_check` distribution:**")
    lines.append("")
    lines.append("| phase_check | Rows |")
    lines.append("|---|---:|")
    for pc, n in sorted(
        summary.get("phase_check_counts", {}).items(),
        key=lambda kv: -kv[1],
    ):
        lines.append(f"| `{pc}` | {n} |")
    lines.append("")
    lines.append(
        "**Score-delta classification (did the flip improve the canonical score?):**"
    )
    lines.append("")
    lines.append("| Class | Rows | Meaning |")
    lines.append("|---|---:|---|")
    delta_meaning = {
        "no_flip": "phase-correction made no change (pre == post)",
        "flip_no_category_change": "flip applied but canonical category unchanged",
        "flip_helped": "flip moved category toward better (e.g. PHASE_SWAPPED → GOOD)",
        "flip_hurt": "flip moved category toward worse (e.g. GOOD → PHASE_SWAPPED)",
        "flip_lateral": "flip applied; same category rank but different categories",
    }
    for cls, n in sorted(
        summary.get("score_delta_class_counts", {}).items(),
        key=lambda kv: -kv[1],
    ):
        lines.append(f"| `{cls}` | {n} | {delta_meaning.get(cls, '?')} |")
    lines.append("")

    # Per-row table
    lines.append("## Per-row trace")
    lines.append("")
    lines.append(
        "| Key | Q1 pre→ (correspondence) | Q2 phase_check | Q3 delta | post final |"
    )
    lines.append("|---|---|---|---|---|")
    for r in payload.get("rows", []):
        if r.get("status") != "traced":
            lines.append(
                f"| `{r.get('key', '?')}` | (status={r.get('status', '?')}, "
                f"{r.get('error', '')[:60]}) | — | — | — |"
            )
            continue
        pre_cat = r.get("pre_canonical_category", "?")
        post_cat = r.get("post_canonical_category", "?")
        # Q1 interpretation: GOOD pre = correspondence picked ONE_EDGE; PHASE_SWAPPED = FAR
        q1 = {
            "GOOD": "ONE_EDGE ✓",
            "MARGINAL": "ONE_EDGE (marginal)",
            "PHASE_SWAPPED": "FAR ✗",
            "GEOMETRY_FAIL": "other",
        }.get(pre_cat, pre_cat)
        q2 = r.get("phase_check", "?")
        q3 = r.get("score_delta_class", "?")
        lines.append(
            f"| `{r.get('key', '?')}` | {q1} (`{pre_cat}`) | `{q2}` | `{q3}` | `{post_cat}` |"
        )
    lines.append("")

    # Q4 cross-row observations
    lines.append("## Q4: Which evidence would have selected the right parity?")
    lines.append("")
    lines.append(
        "Comparing `phase_darkness_separation` across rows by canonical "
        "outcome (post-pipeline):"
    )
    lines.append("")
    seps_by_cat: Dict[str, List[float]] = defaultdict(list)
    for r in payload.get("rows", []):
        if r.get("status") != "traced":
            continue
        sep = r.get("phase_debug", {}).get("phase_darkness_separation")
        cat = r.get("post_canonical_category")
        if sep is not None and cat:
            seps_by_cat[cat].append(float(sep))
    lines.append("| Post canonical | n | sep median | sep min | sep max |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat, seps in sorted(seps_by_cat.items()):
        if not seps:
            continue
        lines.append(
            f"| `{cat}` | {len(seps)} | {statistics.median(seps):.1f} | "
            f"{min(seps):.1f} | {max(seps):.1f} |"
        )
    lines.append("")

    lines.append("## Findings & implications")
    lines.append("")
    # Compute headline numbers for the findings text.
    n = summary.get("n_traced", 0)
    pre_phase_swapped = summary.get("pre_canonical_category_counts", {}).get(
        "PHASE_SWAPPED", 0
    )
    post_phase_swapped = summary.get(
        "post_canonical_category_counts", {}
    ).get("PHASE_SWAPPED", 0)
    flip_helped = summary.get("score_delta_class_counts", {}).get("flip_helped", 0)
    flip_hurt = summary.get("score_delta_class_counts", {}).get("flip_hurt", 0)
    flip_no_change = summary.get("score_delta_class_counts", {}).get(
        "flip_no_category_change", 0
    )
    no_flip = summary.get("score_delta_class_counts", {}).get("no_flip", 0)

    lines.append(
        f"1. **Initial correspondence picks FAR (PHASE_SWAPPED) on "
        f"{pre_phase_swapped}/{n} rows** — i.e., the Procrustes/template "
        f"fit assigns axis_x/y/z to far-corner positions before "
        f"`_resolve_near_far_phase` is even called. This is the upstream "
        f"bug Codex flagged: \"the whole correspondence + phase-"
        f"correction pathway,\" not just the detector."
    )
    lines.append("")
    if flip_helped or flip_hurt:
        lines.append(
            f"2. **Phase-correction's impact on canonical score:** "
            f"helped {flip_helped} row(s), hurt {flip_hurt} row(s), "
            f"no category change on {flip_no_change} row(s), "
            f"no flip applied on {no_flip} row(s). If "
            f"`flip_hurt > 0`, the detector is actively creating "
            f"PHASE_SWAPPED outcomes that the initial correspondence "
            f"got right."
        )
        lines.append("")
    lines.append(
        f"3. **End-state (post-pipeline) PHASE_SWAPPED count: "
        f"{post_phase_swapped}/{n}.** Compare to pre count above — "
        f"if they're similar, phase-correction isn't making net progress."
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
        "--render-only", action="store_true",
        help="Skip pipeline runs; regenerate the report from --out-json.",
    )
    args = ap.parse_args()

    if not args.truth.exists():
        print(f"error: truth fixture not found at {args.truth}", file=sys.stderr)
        return 1

    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    if args.render_only:
        if not args.out_json.exists():
            print(
                f"error: --render-only requires existing --out-json at "
                f"{args.out_json}",
                file=sys.stderr,
            )
            return 1
        payload = json.loads(args.out_json.read_text(encoding="utf-8"))
    else:
        payload = run_diagnostic(truth, args.max_image_dim)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"wrote trace JSON to {args.out_json}", file=sys.stderr)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_report(payload))
    print(f"wrote Markdown report to {args.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
