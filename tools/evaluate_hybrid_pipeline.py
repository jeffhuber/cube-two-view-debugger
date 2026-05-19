#!/usr/bin/env python3
"""End-to-end hybrid pipeline evaluator.

The hypothesis under test: take the existing recognizer's face-quad
detection (which delivers 82.7% per-sticker via WhiteUpRecognizer
with rejection) and pipe its quads through rectification +
knn5_lab_full classification (which delivers 99.29% on
rectified-from-human-quads per PR #150). Does the classification-side
improvement transfer to the end-to-end auto pipeline?

Pipeline:

  A+B images
    → analyze_image() per side  (existing recognizer's geometry)
    → grids grouped by center_face → best per face
    → 3x3 grid centers → 4-point face quad via homography
    → rectify each face to a 300x300 square (PR #136)
    → 9 sticker samples per face → classify_rgb (env-selected mode)
    → joint A+B multiset face-ID (PR #126)
    → assemble 54-state in URFDLB order
    → compare to GT

Run twice — once with `CUBE_RECOGNIZER_CLASSIFIER=canonical`, once
with `=knn5_lab_full` — to isolate the classification-side lift.

NO production-recognizer changes. Tooling-only.

Per `COORDINATION.md` sweep-logging convention: per-pair progress to
stderr with flush=True; log file should be redirected with
`> log 2>&1` (not `2>&1 > log`).

Usage:
  CUBE_RECOGNIZER_CLASSIFIER=knn5_lab_full \\
    .venv/bin/python tools/evaluate_hybrid_pipeline.py
  .venv/bin/python tools/evaluate_hybrid_pipeline.py --only-sets 46 47 61 62
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import COLOR_TO_FACE  # noqa: E402
from rubik_recognizer.image_pipeline import analyze_image  # noqa: E402
from rubik_recognizer.validation import FACE_ORDER  # noqa: E402
from tools.extract_color_samples import (  # noqa: E402
    discover_additional_tasks,
    face_colors_from_state,
    load_corpus_tasks,
    parse_ground_truth,
)
from tools.inspect_cube_isolation import point_in_polygon  # noqa: E402
from tools.propose_geometry_labels import (  # noqa: E402
    _face_quad_from_grid_centers,
    _get_rembg_session,
    _hull_from_mask,
)
from tools.rectify_faces import (  # noqa: E402
    DEFAULT_FACE_SIZE,
    extract_stickers_from_rectified,
    rectify_face,
)
from tools.sample_stickers_from_hull import (  # noqa: E402
    apply_orientation,
    discover_orientation,
    identify_faces_jointly,
)

CORPUS_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_REPORT = REPO_ROOT / "runs" / "hybrid_pipeline_report.json"
DEFAULT_SUMMARY = REPO_ROOT / "runs" / "hybrid_pipeline_summary.txt"
PROCESSING_MAX = 1150

EXPECTED_FACES_BY_SIDE = {"A": ("U", "R", "F"), "B": ("D", "L", "B")}
OOD_SETS = {"57", "58", "61", "62"}

# Hull-guard threshold (matches Codex's REMBG_GRID_INSIDE_MIN in PR #141).
# Any analyze_image grid with fewer than this many sticker-center points
# inside the rembg cube hull is rejected as spatially incoherent.
HULL_GUARD_INSIDE_MIN = 7


def _load_processing_image(image_path: Path) -> Tuple[Image.Image, np.ndarray]:
    """EXIF-correct + resize to max 1150, same as the rest of the tooling
    AND same as analyze_image's internal pipeline (so coordinates are
    directly comparable)."""
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    natural_max = max(image.size)
    if natural_max > PROCESSING_MAX:
        scale = PROCESSING_MAX / float(natural_max)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return image, np.asarray(image)


def _rembg_cube_hull(processing_image: Image.Image) -> List[Tuple[float, float]]:
    """Compute rembg u2net cube hull in processing-image coords. Returns
    [] on rembg failure (e.g. mask empty). This is the same hull that
    Codex's PR #141 uses to guard production-recognizer grid ranking.

    Cached on the underlying image object's id() because each pair runs
    this twice (proposer + verification) and rembg is the slow step."""
    cache = _rembg_cube_hull._cache  # type: ignore[attr-defined]
    key = id(processing_image)
    if key in cache:
        return cache[key]
    try:
        from rembg import remove
    except ImportError:
        cache[key] = []
        return cache[key]
    try:
        rgba = remove(processing_image, session=_get_rembg_session("u2net"))
    except Exception:
        cache[key] = []
        return cache[key]
    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    mask = alpha > 128
    if not mask.any():
        cache[key] = []
        return cache[key]
    hull = list(_hull_from_mask(mask))
    cache[key] = hull
    return hull


_rembg_cube_hull._cache = {}  # type: ignore[attr-defined]


def _grid_inside_hull_count(grid, hull: List[Tuple[float, float]]) -> int:
    """Count how many of a FaceGrid's 9 sticker centers fall inside the
    cube hull. Mirrors Codex's `cube_hull_inside_count` semantics from
    rubik_recognizer/image_pipeline.py."""
    if not hull or len(hull) < 3:
        return 9  # no hull → don't apply guard (degrade gracefully)
    inside = 0
    for row in grid.points:
        for (x, y) in row:
            if point_in_polygon((float(x), float(y)), hull):
                inside += 1
    return inside


def _proposer_face_quads(
    image_path: Path,
    side: str,
    hull_guard: bool = True,
    processing_image: Optional[Image.Image] = None,
) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict]:
    """Run analyze_image on raw bytes, pick best FaceGrid per center_face,
    convert 3x3 sticker centers → 4-point face quads, then RE-KEY the
    output to match the geometry-labeler convention that
    `identify_faces_jointly` expects (U/R/F on side A, D/L/B on side B).

    Why re-keying is necessary (Devin #152 audit caught this):
    `identify_faces_jointly` internally hardcodes
    `expected_a = ["R", "F"]` and `expected_b = ["L", "B"]` and treats
    those as literal quad-dict keys. Quads whose key isn't in that set
    are silently dropped at `_sample_multisets` (`if label not in
    quads: continue`). Without re-keying, any side A quad that
    analyze_image classified as L/B/D (Set 23-style yaw2 photos, or
    orange↔red center confusion from PR #150's diagnostic) gets
    dropped — making the eval understate the geometry that's actually
    available.

    Re-key strategy: trust analyze_image to identify the U/D anchor
    (white/yellow centers are visually distinctive and rarely confused
    by the canonical classifier). Take the best non-anchor grids by
    quality and re-key them as the geometry-labeler's side-face
    placeholders (R/F on A, L/B on B) in arbitrary order; joint
    face-ID's 16-config yaw enumeration figures out which physical
    face each really is.

    Returns (face_quads_by_label, debug_info). The label keys after
    re-key are a subset of {U, R, F} for side A or {D, L, B} for B.
    """
    assert side in ("A", "B")
    anchor_label = "U" if side == "A" else "D"
    side_face_labels = ("R", "F") if side == "A" else ("L", "B")

    image_bytes = image_path.read_bytes()
    analysis = analyze_image(image_bytes)

    # Hull guard: compute rembg cube hull, reject grids whose sticker
    # centers don't sit (mostly) within it. This addresses the
    # "catastrophic-grid" failure mode identified post-#152: analyze_image
    # sometimes returns 3x3 grids that span multiple physical faces of
    # the cube (or include non-sticker positions), producing valid-looking
    # FaceGrids whose extrapolated face quads cover the wrong region.
    # Rejecting these before rectification eliminates the bimodal
    # "perfect-or-garbage" per-face distribution seen on the pre-guard
    # eval (see PR description for the data).
    hull: List[Tuple[float, float]] = []
    grids_rejected_by_hull: List[Dict] = []
    if hull_guard and processing_image is not None:
        hull = _rembg_cube_hull(processing_image)

    def _grid_passes_guard(grid) -> Tuple[bool, int]:
        if not hull_guard or not hull:
            return True, 9
        inside = _grid_inside_hull_count(grid, hull)
        return inside >= HULL_GUARD_INSIDE_MIN, inside

    grids_by_face: Dict[str, list] = {}
    for grid in analysis.grids:
        ok, inside = _grid_passes_guard(grid)
        if not ok:
            grids_rejected_by_hull.append({
                "centerFace": grid.center_face,
                "matchedCount": grid.matched_count,
                "fitError": round(grid.fit_error, 2),
                "insideCount": inside,
            })
            continue
        grids_by_face.setdefault(grid.center_face, []).append(grid)

    # Best grid per center_face (analyze_image's color classification),
    # AFTER hull guard filtering
    best_per_face: List[Tuple[str, object]] = []
    for face, candidates in grids_by_face.items():
        best = min(candidates, key=lambda g: (-g.matched_count, g.fit_error))
        best_per_face.append((face, best))

    face_quads: Dict[str, List[Tuple[float, float]]] = {}
    selected_metrics: Dict[str, Dict] = {}

    # Anchor (U on side A, D on side B): take the grid whose center_face
    # matches the anchor letter if any exist. If none, the side degrades
    # to <3 faces and joint face-ID handles the missing anchor.
    anchor_grids = [(f, g) for f, g in best_per_face if f == anchor_label]
    if anchor_grids:
        face, grid = min(anchor_grids, key=lambda fg: (-fg[1].matched_count, fg[1].fit_error))
        quad = _face_quad_from_grid_centers(grid.points)
        if quad is not None:
            face_quads[anchor_label] = [(float(x), float(y)) for (x, y) in quad]
            selected_metrics[anchor_label] = {
                "sourceCenterFace": face,
                "matchedCount": grid.matched_count,
                "fitError": round(grid.fit_error, 2),
                "cubeHullInside": grid.cube_hull_inside_count,
            }

    # Side faces: take the best 2 non-anchor grids by quality, re-key as
    # the placeholder side-face labels. The assignment to R vs F (or L vs B)
    # is arbitrary because joint face-ID's yaw enumeration resolves both.
    non_anchor = sorted(
        [(f, g) for f, g in best_per_face if f != anchor_label],
        key=lambda fg: (-fg[1].matched_count, fg[1].fit_error),
    )
    for slot_idx, (face, grid) in enumerate(non_anchor[:2]):
        quad = _face_quad_from_grid_centers(grid.points)
        if quad is None:
            continue
        slot_label = side_face_labels[slot_idx]
        face_quads[slot_label] = [(float(x), float(y)) for (x, y) in quad]
        selected_metrics[slot_label] = {
            "sourceCenterFace": face,
            "matchedCount": grid.matched_count,
            "fitError": round(grid.fit_error, 2),
            "cubeHullInside": grid.cube_hull_inside_count,
            "rekeyedFrom": face,
        }

    return face_quads, {
        "stickerCount": len(analysis.stickers),
        "gridCount": len(analysis.grids),
        "side": side,
        "anchorFound": anchor_label in face_quads,
        "selectedPerFace": selected_metrics,
        "facesProposedAfterRekey": sorted(face_quads.keys()),
        "hullGuardEnabled": hull_guard and bool(hull),
        "gridsRejectedByHullGuard": grids_rejected_by_hull,
        "warnings": list(analysis.warnings),
    }


def _classify_face_aligned(face_img: Image.Image, gt_colors: List[str]):
    """Sample 9 stickers from rectified face, classify with the env-selected
    classifier mode, align via discover_orientation against gt_colors.
    Returns (correct_count, aligned_classified, rgbs_aligned)."""
    samples = extract_stickers_from_rectified(face_img)
    rgbs = [s.rgb for row in samples for s in row]
    classified = [s.classified_color for row in samples for s in row]
    mirror, rot, _ = discover_orientation(rgbs, gt_colors)
    aligned = apply_orientation(classified, mirror, rot)
    rgbs_aligned = apply_orientation(rgbs, mirror, rot)
    correct = sum(1 for c, g in zip(aligned, gt_colors) if c == g)
    return correct, aligned, rgbs_aligned


def evaluate_pair(
    set_id: str, image_a: Path, image_b: Path, gt_state: str,
    hull_guard: bool = True,
) -> Dict:
    """One pair: analyze_image-quads → rectify → classify → joint face-ID
    → assemble 54-state → compare to GT."""
    images: Dict[str, Image.Image] = {}
    arrs: Dict[str, np.ndarray] = {}
    proposer_quads: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    proposer_debug: Dict[str, Dict] = {}
    for side, image_path in (("A", image_a), ("B", image_b)):
        try:
            img, arr = _load_processing_image(image_path)
            images[side] = img
            arrs[side] = arr
        except Exception as e:
            return {"setId": set_id, "error": f"load {side}: {type(e).__name__}: {e}"}
        try:
            quads, debug = _proposer_face_quads(
                image_path, side,
                hull_guard=hull_guard,
                processing_image=img,
            )
        except Exception as e:
            return {"setId": set_id, "error": f"proposer {side}: {type(e).__name__}: {e}"}
        proposer_quads[side] = quads
        proposer_debug[side] = debug

    # Joint A+B face-ID using analyze_image's auto-proposed quads.
    # The function takes "expected" faces per side (URF for A, DLB for B)
    # so it can multiset-match against the GT's 6 face centers.
    prepared = {
        side: {
            "arr": arrs[side],
            "quads": proposer_quads[side],
            "expected": EXPECTED_FACES_BY_SIDE[side],
        }
        for side in ("A", "B")
    }
    label_to_true, joint_score, joint_status = identify_faces_jointly(
        prepared, gt_state, inset=0.20
    )

    per_face_aligned: Dict[str, List[str]] = {}
    per_face_metrics: List[Dict] = []
    stickers_sampled = 0
    stickers_correct = 0
    for side in ("A", "B"):
        mapping = label_to_true.get(side, {})
        for label_face, true_face in mapping.items():
            quad = proposer_quads[side].get(label_face)
            if quad is None or len(quad) != 4:
                continue
            gt_colors = face_colors_from_state(gt_state, true_face)
            try:
                rectified = rectify_face(images[side], quad,
                                         output_size=DEFAULT_FACE_SIZE)
            except Exception as e:
                per_face_metrics.append({
                    "side": side, "labelFace": label_face,
                    "trueFace": true_face,
                    "error": f"rectify: {type(e).__name__}: {e}",
                })
                continue
            try:
                correct, aligned, _ = _classify_face_aligned(rectified, gt_colors)
            except Exception as e:
                per_face_metrics.append({
                    "side": side, "labelFace": label_face,
                    "trueFace": true_face,
                    "error": f"classify: {type(e).__name__}: {e}",
                })
                continue
            stickers_sampled += 9
            stickers_correct += correct
            per_face_aligned[true_face] = aligned
            per_face_metrics.append({
                "side": side, "labelFace": label_face,
                "trueFace": true_face,
                "correct": correct, "ofTotal": 9,
            })

    # Assemble 54-state in URFDLB order
    assembled: Optional[str] = None
    if all(face in per_face_aligned for face in FACE_ORDER):
        chunks: List[str] = []
        for face in FACE_ORDER:
            colors = per_face_aligned[face]
            chunks.append("".join(COLOR_TO_FACE[c] for c in colors))
        assembled = "".join(chunks)

    exact_match = (assembled is not None and assembled == gt_state)
    sticker_matches_assembled = None
    if assembled is not None and len(gt_state) == 54:
        sticker_matches_assembled = sum(
            1 for a, g in zip(assembled, gt_state) if a == g
        )

    return {
        "setId": set_id,
        "isOOD": set_id in OOD_SETS,
        "stickersSampled": stickers_sampled,
        "stickersCorrect": stickers_correct,
        "perStickerAccuracy":
            round(stickers_correct / stickers_sampled, 4)
            if stickers_sampled else None,
        "facesRecovered": len(per_face_aligned),
        "facesExpected": 6,
        "facesProposedA": proposer_debug["A"]["facesProposedAfterRekey"],
        "facesProposedB": proposer_debug["B"]["facesProposedAfterRekey"],
        "anchorFoundA": proposer_debug["A"]["anchorFound"],
        "anchorFoundB": proposer_debug["B"]["anchorFound"],
        "hullGuardA": {
            "enabled": proposer_debug["A"].get("hullGuardEnabled"),
            "gridsRejected": len(proposer_debug["A"].get("gridsRejectedByHullGuard", [])),
            "gridsAccepted": proposer_debug["A"].get("gridCount", 0)
                - len(proposer_debug["A"].get("gridsRejectedByHullGuard", [])),
        },
        "hullGuardB": {
            "enabled": proposer_debug["B"].get("hullGuardEnabled"),
            "gridsRejected": len(proposer_debug["B"].get("gridsRejectedByHullGuard", [])),
            "gridsAccepted": proposer_debug["B"].get("gridCount", 0)
                - len(proposer_debug["B"].get("gridsRejectedByHullGuard", [])),
        },
        "jointStatus": joint_status,
        "jointScore": round(joint_score, 4) if joint_score is not None else None,
        "assembledState": assembled,
        "exactMatch": exact_match,
        "perStickerMatchesAssembled": sticker_matches_assembled,
        "perFace": per_face_metrics,
    }


def discover_pairs() -> List[Tuple[str, Path, Path, str]]:
    tasks = load_corpus_tasks(CORPUS_MANIFEST)
    tasks.extend(discover_additional_tasks({t.set_id for t in tasks}))
    out: List[Tuple[str, Path, Path, str]] = []
    for task in tasks:
        if not (task.image_a.exists() and task.image_b.exists()):
            continue
        try:
            gt = parse_ground_truth(task.ground_truth)
        except Exception:
            continue
        out.append((task.set_id, task.image_a, task.image_b, gt))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only-sets", nargs="*", default=None)
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument(
        "--no-hull-guard", action="store_true",
        help="Disable the rembg-cube-hull validation that rejects "
             "analyze_image grids whose sticker centers fall outside "
             "the cube silhouette (default: guard enabled). Useful for "
             "A/B comparison against the pre-guard baseline.",
    )
    args = ap.parse_args()

    pairs = discover_pairs()
    if args.only_sets:
        wanted = set(args.only_sets)
        pairs = [p for p in pairs if p[0] in wanted]

    hull_guard = not args.no_hull_guard
    classifier_mode = os.environ.get("CUBE_RECOGNIZER_CLASSIFIER", "canonical")
    print(f"evaluating hybrid pipeline on {len(pairs)} pairs "
          f"(classifier={classifier_mode}, hull_guard={hull_guard})",
          file=sys.stderr)
    print("", file=sys.stderr)

    rows: List[Dict] = []
    for i, (set_id, image_a, image_b, gt) in enumerate(pairs, 1):
        try:
            row = evaluate_pair(set_id, image_a, image_b, gt, hull_guard=hull_guard)
        except Exception as e:
            row = {"setId": set_id, "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        if "error" in row:
            print(f"  [{i:>2}/{len(pairs)}] set {set_id}: ERROR {row['error']}",
                  file=sys.stderr, flush=True)
        else:
            flag = " [OOD]" if row.get("isOOD") else ""
            exact = "EXACT" if row["exactMatch"] else "diff"
            psa = row["perStickerAccuracy"]
            stk = row.get("perStickerMatchesAssembled")
            stk_str = f"{stk}/54" if stk is not None else "n/a"
            print(
                f"  [{i:>2}/{len(pairs)}] set {set_id}: "
                f"faces={row['facesRecovered']}/6  "
                f"perSticker(rect)={psa}  state={exact} ({stk_str}){flag}",
                file=sys.stderr, flush=True,
            )

    valid = [r for r in rows if "error" not in r]
    total_sampled = sum(r["stickersSampled"] for r in valid)
    total_correct = sum(r["stickersCorrect"] for r in valid)
    rect_accuracy = total_correct / max(1, total_sampled)

    pairs_assembled = [r for r in valid if r["assembledState"] is not None]
    pairs_exact = [r for r in valid if r["exactMatch"]]
    pairs_failed_to_assemble = [r for r in valid if r["assembledState"] is None]

    # Per-sticker accuracy MEASURED ON ASSEMBLED 54-STATE (not just on
    # the rectified-face samples). This catches cases where joint
    # face-ID picked the wrong mapping but per-face classification
    # was internally consistent.
    assembled_stickers_total = sum(1 for r in pairs_assembled) * 54
    assembled_stickers_correct = sum(r["perStickerMatchesAssembled"]
                                     for r in pairs_assembled)
    assembled_accuracy = (
        assembled_stickers_correct / max(1, assembled_stickers_total)
    )

    ood = [r for r in valid if r.get("isOOD")]
    ood_assembled = [r for r in ood if r["assembledState"] is not None]
    ood_exact = sum(1 for r in ood if r["exactMatch"])
    ood_sticker_total = sum(1 for r in ood_assembled) * 54
    ood_sticker_correct = sum(r["perStickerMatchesAssembled"]
                              for r in ood_assembled)

    non_ood = [r for r in valid if not r.get("isOOD")]
    non_ood_assembled = [r for r in non_ood if r["assembledState"] is not None]
    non_ood_exact = sum(1 for r in non_ood if r["exactMatch"])
    non_ood_sticker_total = sum(1 for r in non_ood_assembled) * 54
    non_ood_sticker_correct = sum(r["perStickerMatchesAssembled"]
                                  for r in non_ood_assembled)

    summary_lines: List[str] = []
    summary_lines.append(
        f"Hybrid pipeline evaluation: {len(pairs)} pairs "
        f"(classifier={classifier_mode})"
    )
    summary_lines.append("")
    summary_lines.append("Stages: analyze_image → grids → face_quads → "
                         "rectify → knn5_lab_full → joint face-ID → assemble")
    summary_lines.append("")
    summary_lines.append("Pair outcomes:")
    summary_lines.append(f"  total:                {len(valid)}")
    summary_lines.append(f"  exact 54-state:       {len(pairs_exact)}  "
                         f"({len(pairs_exact)*100/max(1,len(valid)):.1f}%)")
    summary_lines.append(f"  assembled (not exact):"
                         f" {len(pairs_assembled) - len(pairs_exact)}")
    summary_lines.append(f"  failed to assemble:   {len(pairs_failed_to_assemble)}")
    summary_lines.append("")
    summary_lines.append("Per-sticker accuracy:")
    summary_lines.append(
        f"  on rectified faces (only): {rect_accuracy:.4f} "
        f"({total_correct}/{total_sampled})"
    )
    summary_lines.append(
        f"  on assembled 54-state:     {assembled_accuracy:.4f} "
        f"({assembled_stickers_correct}/{assembled_stickers_total})"
    )
    summary_lines.append("")
    summary_lines.append("OOD-set breakdown (Sets 57/58/61/62):")
    if ood:
        ood_rect_acc = sum(r["stickersCorrect"] for r in ood) / max(
            1, sum(r["stickersSampled"] for r in ood)
        )
        ood_assembled_acc = (
            ood_sticker_correct / max(1, ood_sticker_total)
            if ood_sticker_total else None
        )
        summary_lines.append(
            f"  pairs: {len(ood)}, exact: {ood_exact}, "
            f"assembled: {len(ood_assembled)}"
        )
        summary_lines.append(
            f"  rect accuracy:      {ood_rect_acc:.4f}"
        )
        if ood_assembled_acc is not None:
            summary_lines.append(
                f"  assembled accuracy: {ood_assembled_acc:.4f}"
            )
    summary_lines.append("")
    summary_lines.append("Non-OOD breakdown (28 pairs):")
    if non_ood:
        non_ood_rect_acc = sum(r["stickersCorrect"] for r in non_ood) / max(
            1, sum(r["stickersSampled"] for r in non_ood)
        )
        non_ood_assembled_acc = (
            non_ood_sticker_correct / max(1, non_ood_sticker_total)
            if non_ood_sticker_total else None
        )
        summary_lines.append(
            f"  pairs: {len(non_ood)}, exact: {non_ood_exact}, "
            f"assembled: {len(non_ood_assembled)}"
        )
        summary_lines.append(
            f"  rect accuracy:      {non_ood_rect_acc:.4f}"
        )
        if non_ood_assembled_acc is not None:
            summary_lines.append(
                f"  assembled accuracy: {non_ood_assembled_acc:.4f}"
            )
    summary_lines.append("")

    summary_lines.append("Comparison to known baselines:")
    summary_lines.append(
        "  rectified-from-human-quads + knn5_lab_full (PR #150 A/B): "
        "0.9929 per-sticker"
    )
    summary_lines.append(
        "  existing recognizer (WhiteUpRecognizer):                   "
        "~0.827 per-sticker (from #139 sweep)"
    )
    summary_lines.append(
        "  mask pipeline (rembg → optimized hexagon):                 "
        "~0.615 per-sticker (from #139 sweep)"
    )
    summary_lines.append("")

    # Worst pairs by per-sticker (rect) accuracy
    sorted_pairs = sorted(
        valid, key=lambda r: r.get("perStickerAccuracy") or 0
    )
    summary_lines.append("Worst 10 pairs by rectified per-sticker accuracy:")
    for r in sorted_pairs[:10]:
        flag = " [OOD]" if r.get("isOOD") else ""
        psa = r.get("perStickerAccuracy")
        faces = r["facesRecovered"]
        exact = "EXACT" if r["exactMatch"] else "."
        summary_lines.append(
            f"  set {r['setId']}: rect={psa} faces={faces}/6 "
            f"state={exact}{flag}"
        )

    # Failed-to-assemble pairs deserve a callout — what went wrong?
    if pairs_failed_to_assemble:
        summary_lines.append("")
        summary_lines.append(
            f"Pairs that failed to assemble (joint face-ID couldn't "
            f"recover 6 faces): {len(pairs_failed_to_assemble)}"
        )
        for r in pairs_failed_to_assemble[:10]:
            flag = " [OOD]" if r.get("isOOD") else ""
            summary_lines.append(
                f"  set {r['setId']}: faces={r['facesRecovered']}/6, "
                f"proposedA={r['facesProposedA']}, "
                f"proposedB={r['facesProposedB']}, "
                f"joint={r['jointStatus']}{flag}"
            )

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rows, indent=2))
    Path(args.summary).write_text("\n".join(summary_lines) + "\n")

    print("", file=sys.stderr)
    print("\n".join(summary_lines), file=sys.stderr)
    print(f"\nwrote {args.report}", file=sys.stderr)
    print(f"wrote {args.summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
