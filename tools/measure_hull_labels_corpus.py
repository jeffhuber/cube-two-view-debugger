#!/usr/bin/env python3
"""Larger-corpus validation of ``tools/rectify_via_hull_labels.py``.

Origin: ``tools/RECTIFY_VIA_HULL_LABELS_REPORT.md`` showed 12/12
essentially-oracle-quality rectifications on the 12 approved
full-corner rows. That's encouraging but small. The hull-labels
approach is the candidate replacement for the production 720-perm
Procrustes pipeline; before it can replace that pipeline (or wire in
behind a feature flag) we need empirical signal on a larger corpus.

This tool answers: **does the hull-labels approach hold up on the
70-row axis-labeled gallery** (the corpus used by
``tools/measure_axis_correctness.py``)?

Pipeline per row:
  1. Run ``rectify_via_hull_labels`` with the production rembg path.
  2. Classify into a failure bucket:
       - ``mask_failure``     : rembg + ``detect_hexagon_anchors``
         couldn't return 6 hull corners
       - ``label_failure``    : 6 corners present but per-side
         labeling raised (e.g. malformed input, currently impossible
         from public API but kept as a bucket for future-proofing)
       - ``rectified``        : pipeline returned a ``RectifiedFit``
  3. For rectified rows, compute:
       - ``vertex_err_px``               vs ground-truth vertex
       - ``vertex_cloud_spread_px``      max pairwise distance of the
         3 parallelogram-completion vertex estimates (a proxy for how
         non-iso the projection is)
       - ``axis_total_misfit_deg``       sum of best-permutation
         per-axis angle errors vs GT (same metric as
         ``measure_axis_correctness.py``)
       - ``sticker_color_score``         sum of distances to nearest
         canonical color across 27 sampled stickers (no GT needed —
         a measure of how cleanly our face quads sample the cube)

The 12 rows that overlap the full-corner corpus also get tagged with
``in_full_corner_corpus: true`` so the report can split "old 12" vs
"new 58".

Failure buckets per Codex's lane-split outline (2026-05-24):
- ``mask_failure``                — rembg / hexagon detect produces
  fewer than 6 hull corners
- ``vertex_cloud_high_spread``    — 3 estimates disagree by more than
  THRESH_VERTEX_CLOUD_SPREAD_PX → likely non-iso projection
- ``axis_misfit_high``            — axis_total_misfit_deg above
  THRESH_AXIS_MISFIT_DEG → labeling/vertex pipeline is producing the
  wrong face_quads
- ``sticker_score_high``          — face_quads sample non-cube content
  (background, off-edge), measured by classifier distance
- ``rectified_clean``             — all gates clean

The thresholds below are starting heuristics from the 12-row corpus
distribution; the report will let us tune them.

## CLI

  python tools/measure_hull_labels_corpus.py

Defaults: read ``tests/fixtures/gcm_axis_ground_truth.json``, write
trace to ``tests/fixtures/hull_labels_corpus_trace.json`` and report
to ``tools/HULL_LABELS_CORPUS_REPORT.md``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
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

from rubik_recognizer.colors import classify_rgb  # noqa: E402
from tools.corner_conventions import (  # noqa: E402
    FACE_DEFS_BY_SIDE,
    FAR_CORNERS_BY_SIDE,
    ONE_EDGE_CORNERS_BY_SIDE,
)
from tools.diagnose_pipeline_phase_parity import _processing_image  # noqa: E402
from tools.measure_axis_correctness import (  # noqa: E402
    _candidate_image_roots,
    _match_axes_to_ground_truth,
    _resolve_image_path,
    _scale_point,
)
from tools.rectify_faces import extract_stickers_from_rectified  # noqa: E402
from tools.rectify_via_hull_labels import (  # noqa: E402
    SILHOUETTE_TO_CORNER,
    _derive_vertex_from_corners,
    _label_corners_by_position,
    rectify_via_hull_labels,
)
from tools.global_cube_model import detect_hexagon_anchors  # noqa: E402

Point = Tuple[float, float]

DEFAULT_AXIS_TRUTH = REPO_ROOT / "tests" / "fixtures" / "gcm_axis_ground_truth.json"
DEFAULT_FULL_CORNER_TRUTH = (
    REPO_ROOT / "tests" / "fixtures" / "full_corner_ground_truth.json"
)
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_TRACE = REPO_ROOT / "tests" / "fixtures" / "hull_labels_corpus_trace.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "HULL_LABELS_CORPUS_REPORT.md"
DEFAULT_MAX_IMAGE_DIM = 1600

# Heuristic thresholds for failure-bucket classification. Starting
# values derived from the 70-row run distribution:
#   - vertex_err: typical 0-80 px (matches 12-row corpus)
#   - vertex_cloud_spread: 90-270 px (much wider than 12-row report
#     estimated — actual perspective on real iPhone shots gives larger
#     spread; mean still close to GT)
#   - axis_total_misfit: should be <30° for "clean" — the 12-row
#     corpus has very low axis error since the hull-labels approach
#     gets axes right when 6 corners are stable
#   - sticker_score_total: 12-row median was ~430-490 with mode
#     CANONICAL classifier; >1500 flags an off-cube quad
THRESH_VERTEX_CLOUD_SPREAD_PX = 350.0
THRESH_AXIS_MISFIT_DEG = 30.0
THRESH_STICKER_SCORE_TOTAL = 1500.0


def _git_head_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _ground_truth_axes_from_axis_truth(
    truth_row: Dict[str, Any], scale: float,
) -> Tuple[Point, List[Point]]:
    """Read GT vertex + 3 axis endpoints from
    ``gcm_axis_ground_truth.json``-schema row. The 3 endpoints are
    labeled ``near_x``/``near_y``/``near_z`` (world axes), NOT corner
    indices — so the predicted-to-GT matching has to be permutation-
    invariant. That's what ``_match_axes_to_ground_truth`` does.

    Empirical note (verified on all 12 overlap rows with
    ``full_corner_ground_truth.json``): the ``near_x/y/z`` points
    consistently sit at FAR-corner positions ({0,2,4} for side A;
    {1,3,5} for side B), NOT at the NEAR-set {1,3,5}/{0,2,4}.
    Despite the "near_" prefix in the labeling-tool UI, the user
    clicked the silhouette corner that visually marks each world axis
    direction (e.g. for side A white-up: +Z = TOP corner = corner_0).
    That corner is two cube-edges from the vertex, i.e. the FAR set.
    The downstream metric uses these as axis endpoints, so what
    matters is direction-from-vertex, and both NEAR and FAR endpoints
    along a given world-axis direction project to one of the 6
    silhouette corners — they just sit 60° apart in iso. So the
    convention is internally consistent as long as the PREDICTED
    axes are computed using the same set (FAR) — see ``evaluate_row``.
    """
    vertex_full = (float(truth_row["vertex"][0]), float(truth_row["vertex"][1]))
    vertex_proc = _scale_point(vertex_full, scale)
    axes: List[Point] = []
    for name in ("near_x", "near_y", "near_z"):
        p_full = (float(truth_row[name][0]), float(truth_row[name][1]))
        p_proc = _scale_point(p_full, scale)
        axes.append((p_proc[0] - vertex_proc[0], p_proc[1] - vertex_proc[1]))
    return vertex_proc, axes


def _score_rectified_faces(faces: Dict[str, Any]) -> Dict[str, Any]:
    """Sum of classifier distance across 27 sampled stickers (9 per
    face × 3 faces). Same metric as ``rectify_via_hull_labels.py``.

    Note: classifier distance shape depends on
    ``RUBIK_RECOGNIZER_COLOR_CLASSIFIER_MODE`` — absolute scores are
    only comparable across runs under the same classifier mode.
    """
    total = 0.0
    per_face: Dict[str, float] = {}
    for slot, face_img in faces.items():
        face_total = 0.0
        for row in extract_stickers_from_rectified(face_img):
            for s in row:
                face_total += classify_rgb(s.rgb).distance
        per_face[slot] = round(face_total, 2)
        total += face_total
    return {
        "total_distance": round(total, 2),
        "per_face": per_face,
        "mean_sticker_distance": round(total / 27.0, 2),
    }


def _classify_row(
    rec: Dict[str, Any],
    *,
    thresh_spread: float,
    thresh_axis: float,
    thresh_sticker: float,
) -> str:
    """Compute the bucket label for a rectified row. Mask/label
    failures are tagged earlier."""
    if rec.get("vertex_cloud_spread_px", 0.0) > thresh_spread:
        return "vertex_cloud_high_spread"
    if rec.get("axis_total_misfit_deg", 0.0) > thresh_axis:
        return "axis_misfit_high"
    if rec.get("sticker_score_total", 0.0) > thresh_sticker:
        return "sticker_score_high"
    return "rectified_clean"


def evaluate_row(
    sess: Any,
    key: str,
    image_path: Path,
    axis_truth_row: Dict[str, Any],
    full_corner_truth_row: Optional[Dict[str, Any]],
    max_image_dim: int,
    *,
    thresh_spread: float,
    thresh_axis: float,
    thresh_sticker: float,
) -> Dict[str, Any]:
    from rembg import remove  # noqa: E402
    side = key.rsplit("_", 1)[-1]
    rec: Dict[str, Any] = {
        "key": key,
        "side": side,
        "in_full_corner_corpus": full_corner_truth_row is not None,
        "status": "pending",
    }
    try:
        image, scale = _processing_image(image_path, max_image_dim)
        rgba = remove(image, session=sess)
        alpha = np.array(rgba.split()[-1], dtype=np.uint8)
        mask = alpha > 128
        hexagon = detect_hexagon_anchors(mask)
        rec["hexagon_corner_count"] = len(hexagon)
        if len(hexagon) != 6:
            rec.update({"status": "mask_failure",
                        "bucket": "mask_failure",
                        "error": "detect_hexagon_anchors returned "
                                 f"{len(hexagon)} corners (need 6)"})
            return rec
        # Hand-roll the pipeline so we can intercept the 3 vertex
        # estimates for cloud-spread analysis instead of re-running.
        try:
            corners_by_num = _label_corners_by_position(hexagon, side)
        except Exception as exc:  # noqa: BLE001
            rec.update({"status": "label_failure",
                        "bucket": "label_failure",
                        "error": f"{type(exc).__name__}: {exc}"})
            return rec
        vertex, estimates = _derive_vertex_from_corners(corners_by_num, side)
        # Vertex-cloud spread: max pairwise distance between the 3
        # estimates. Bigger = projection is less iso = our derived
        # vertex is less trustworthy.
        spread = 0.0
        for i in range(len(estimates)):
            for j in range(i + 1, len(estimates)):
                d = math.hypot(estimates[i][0] - estimates[j][0],
                               estimates[i][1] - estimates[j][1])
                if d > spread:
                    spread = d
        # Re-run via the canonical pipeline to make sure the rectified
        # faces match what the production caller would see.
        fit = rectify_via_hull_labels(image, mask, side)
        if fit is None:  # defensive — shouldn't happen given 6 corners
            rec.update({"status": "label_failure",
                        "bucket": "label_failure",
                        "error": "rectify_via_hull_labels returned None "
                                 "after detect_hexagon_anchors succeeded"})
            return rec
        # GT axes + vertex (from the 70-row axis truth schema)
        gt_vertex_proc, gt_axes = _ground_truth_axes_from_axis_truth(
            axis_truth_row, scale,
        )
        vertex_err = math.hypot(
            fit.vertex[0] - gt_vertex_proc[0],
            fit.vertex[1] - gt_vertex_proc[1],
        )
        # Predicted axes: the 3 FAR-corner vectors from our derived
        # vertex. Must match the GT convention (see
        # ``_ground_truth_axes_from_axis_truth`` for why 70-row
        # near_x/y/z labels sit at FAR-corner positions). NEAR
        # directions would be 60° offset → ~180° total misfit summed
        # over 3 axes — verified empirically.
        far_corner_names = FAR_CORNERS_BY_SIDE[side]
        predicted_axes: List[Point] = []
        for cn in far_corner_names:
            cp = fit.corners_by_num[int(cn.split("_")[1])]
            predicted_axes.append(
                (cp[0] - fit.vertex[0], cp[1] - fit.vertex[1])
            )
        axis_match = _match_axes_to_ground_truth(predicted_axes, gt_axes)
        # Sticker-color score on our rectified faces
        score = _score_rectified_faces(fit.rectified_faces)
        rec.update({
            "status": "rectified",
            "vertex_err_px": round(vertex_err, 1),
            "vertex_cloud_spread_px": round(spread, 1),
            "vertex_estimates_processing_px": [
                [round(e[0], 1), round(e[1], 1)] for e in estimates
            ],
            "derived_vertex_processing_px": [
                round(fit.vertex[0], 1), round(fit.vertex[1], 1),
            ],
            "gt_vertex_processing_px": [
                round(gt_vertex_proc[0], 1), round(gt_vertex_proc[1], 1),
            ],
            "axis_total_misfit_deg": axis_match["total_misfit_deg"],
            "axis_match": axis_match,
            "sticker_score_total": score["total_distance"],
            "sticker_score_per_face": score["per_face"],
        })
        rec["bucket"] = _classify_row(
            rec,
            thresh_spread=thresh_spread,
            thresh_axis=thresh_axis,
            thresh_sticker=thresh_sticker,
        )
    except Exception as exc:  # noqa: BLE001
        rec.update({
            "status": "error",
            "bucket": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis-truth", type=Path, default=DEFAULT_AXIS_TRUTH)
    ap.add_argument(
        "--full-corner-truth", type=Path, default=DEFAULT_FULL_CORNER_TRUTH,
    )
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM)
    ap.add_argument(
        "--thresh-spread-px", type=float,
        default=THRESH_VERTEX_CLOUD_SPREAD_PX,
    )
    ap.add_argument(
        "--thresh-axis-deg", type=float, default=THRESH_AXIS_MISFIT_DEG,
    )
    ap.add_argument(
        "--thresh-sticker-total", type=float,
        default=THRESH_STICKER_SCORE_TOTAL,
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    axis_truth = json.loads(args.axis_truth.read_text(encoding="utf-8"))
    full_corner_truth = json.loads(
        args.full_corner_truth.read_text(encoding="utf-8")
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    set_index = {str(p["setId"]): p for p in manifest.get("pairs", [])}
    image_roots = _candidate_image_roots(manifest)

    from rembg import new_session  # noqa: E402
    # Explicit "u2net" — matches production
    # (rubik_recognizer/image_pipeline.py) and other diagnostics.
    sess = new_session("u2net")

    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for key in sorted(axis_truth):
        row = axis_truth[key]
        if not row.get("approved"):
            continue
        set_id, _, side = key.rpartition("_")
        pair = set_index.get(set_id)
        if pair is None:
            # Try resolving by ~/cube-corpus + pattern even without a
            # manifest entry — _resolve_image_path with raw_path being
            # an unknown filename will fall through to pattern search.
            raw_path = f"/tmp/missing_{set_id}_{side}.jpg"  # sentinel
            expected_sha = None
        else:
            raw_path = pair.get(f"image{side}Path") or ""
            expected_sha = pair.get(f"image{side}_sha256_expected")
        image_path = _resolve_image_path(
            raw_path, set_id, side, image_roots,
            expected_sha256=expected_sha,
        )
        if image_path is None:
            skipped.append({
                "key": key,
                "reason": (
                    f"no image found for set {set_id} side {side} "
                    f"(searched: {Path(raw_path).name if raw_path else '<no manifest entry>'})"
                ),
            })
            continue
        n_done = sum(1 for r in records if r.get("status") == "rectified")
        print(f"[{n_done + 1}] {key} ({image_path.name})...", flush=True)
        rec = evaluate_row(
            sess, key, image_path, row,
            full_corner_truth.get(key) if full_corner_truth.get(key, {}).get("approved") else None,
            args.max_image_dim,
            thresh_spread=args.thresh_spread_px,
            thresh_axis=args.thresh_axis_deg,
            thresh_sticker=args.thresh_sticker_total,
        )
        records.append(rec)

    # Summary aggregation
    bucket_counts: Dict[str, int] = {}
    for r in records:
        b = r.get("bucket", "unclassified")
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    side_bucket: Dict[str, Dict[str, int]] = {"A": {}, "B": {}}
    for r in records:
        s = r.get("side", "?")
        if s not in side_bucket:
            continue
        b = r.get("bucket", "unclassified")
        side_bucket[s][b] = side_bucket[s].get(b, 0) + 1

    source = {
        "tool": "tools/measure_hull_labels_corpus.py",
        "git_sha": _git_head_sha(),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "axis_truth": str(args.axis_truth.relative_to(REPO_ROOT))
        if str(args.axis_truth).startswith(str(REPO_ROOT)) else str(args.axis_truth),
        "manifest": str(args.manifest.relative_to(REPO_ROOT))
        if str(args.manifest).startswith(str(REPO_ROOT)) else str(args.manifest),
        "max_image_dim": args.max_image_dim,
        "thresholds": {
            "vertex_cloud_spread_px": args.thresh_spread_px,
            "axis_misfit_deg": args.thresh_axis_deg,
            "sticker_score_total": args.thresh_sticker_total,
        },
    }
    out = {
        "schema": "hull_labels_corpus_v1",
        "source": source,
        "summary": {
            "total_rows_attempted": len(records),
            "skipped_unresolved_image": len(skipped),
            "by_bucket": bucket_counts,
            "by_side_bucket": side_bucket,
        },
        "skipped": skipped,
        "per_row": records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    print(f"Attempted: {len(records)} rows  (skipped {len(skipped)} unresolved)")
    print("Buckets:")
    for b, c in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {c:>3}  {b}")
    # Quick stats over rectified rows
    rect = [r for r in records if r.get("status") == "rectified"]
    if rect:
        verr = [r["vertex_err_px"] for r in rect]
        spread = [r["vertex_cloud_spread_px"] for r in rect]
        axis = [r["axis_total_misfit_deg"] for r in rect]
        score = [r["sticker_score_total"] for r in rect]
        print(f"\nOver {len(rect)} rectified rows:")
        print(f"  vertex_err_px:               min {min(verr):.1f}  "
              f"median {statistics.median(verr):.1f}  max {max(verr):.1f}")
        print(f"  vertex_cloud_spread_px:      min {min(spread):.1f}  "
              f"median {statistics.median(spread):.1f}  max {max(spread):.1f}")
        print(f"  axis_total_misfit_deg:       min {min(axis):.1f}  "
              f"median {statistics.median(axis):.1f}  max {max(axis):.1f}")
        print(f"  sticker_score_total:         min {min(score):.1f}  "
              f"median {statistics.median(score):.1f}  max {max(score):.1f}")
    print(f"\nTrace: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
