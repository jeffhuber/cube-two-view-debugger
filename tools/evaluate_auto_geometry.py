#!/usr/bin/env python3
"""Evaluate automatic cube-hull and face-quad proposers against the
hand-labeled ground truth in `runs/labels/`.

Multiple proposer baselines are run against every (setId, side) that has
a Geometry Labeler label. For each, the tool computes:

  * Cube hull IoU (polygon-vs-polygon, mask-based)
  * Per-face quad IoU (3 quads per labeled image)
  * Mean corner pixel error per face (Hungarian-matched to GT corners)
  * Face containment fraction (intersection_area / gt_area)
  * Per-pair pass/fail under configurable thresholds

Output:
  * runs/auto_geometry_report.json — full per-pair metrics
  * runs/auto_geometry_summary.txt — pretty per-proposer summary
  * runs/auto_geometry_overlays/<proposer>/<set>-<side>.png — visual
    overlays (proposed = solid, ground truth = dashed)

The proposers themselves live in `tools/propose_geometry_labels.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import COLOR_TO_FACE, classify_rgb  # noqa: E402
from rubik_recognizer.image_pipeline import analyze_image  # noqa: E402
from rubik_recognizer.validation import FACE_ORDER  # noqa: E402
from tools.extract_color_samples import (  # noqa: E402
    EXPECTED_FACES_BY_SIDE,
    discover_additional_tasks,
    face_colors_from_state,
    load_corpus_tasks,
    parse_ground_truth,
)
from tools.inspect_cube_isolation import (  # noqa: E402
    convex_hull,
    expand_polygon,
    point_in_polygon,
)
from tools.propose_geometry_labels import (  # noqa: E402
    Proposal,
    PROPOSERS,
)
from tools.sample_stickers_from_hull import (  # noqa: E402
    canonical_corner_order as _canonical_corner_order,
    latest_hull_label,
    load_hull_label,
    sample_rgb as _sample_rgb,
    scaled_face_quads,
    sticker_centers as _sticker_centers,
)

PROCESSING_MAX = 1150
DEFAULT_REPORT = REPO_ROOT / "runs" / "auto_geometry_report.json"
DEFAULT_SUMMARY = REPO_ROOT / "runs" / "auto_geometry_summary.txt"
DEFAULT_OVERLAY_DIR = REPO_ROOT / "runs" / "auto_geometry_overlays"


# ---------------- geometry / metric helpers ----------------


Point = Tuple[float, float]
Polygon = Sequence[Point]


def polygon_mask(points: Polygon, width: int, height: int) -> np.ndarray:
    """Rasterize a polygon onto a binary mask. Borrowed/inlined from
    evaluate_geometry_labels.polygon_mask (re-implemented here to keep
    this module self-contained)."""
    if len(points) < 3:
        return np.zeros((height, width), dtype=np.uint8)
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon([(float(x), float(y)) for x, y in points], outline=1, fill=1)
    return np.array(img, dtype=np.uint8)


def polygon_iou(a: Polygon, b: Polygon, width: int, height: int) -> float:
    """Mask-based IoU. Robust to non-convex polygons and self-intersections,
    O(width*height) per call which is fine at our resolutions."""
    if len(a) < 3 or len(b) < 3:
        return 0.0
    ma = polygon_mask(a, width, height)
    mb = polygon_mask(b, width, height)
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    if union == 0:
        return 0.0
    return inter / union


def polygon_containment(target: Polygon, container: Polygon, width: int, height: int) -> float:
    """Fraction of `target` polygon area that falls inside `container`.
    Useful for asking 'did our proposed face quad cover the true face?'
    (containment can be high even when IoU is low if proposer overshoots)."""
    if len(target) < 3 or len(container) < 3:
        return 0.0
    mt = polygon_mask(target, width, height)
    mc = polygon_mask(container, width, height)
    target_area = int(mt.sum())
    if target_area == 0:
        return 0.0
    inter = int(np.logical_and(mt, mc).sum())
    return inter / target_area


def mean_corner_error(proposed: Sequence[Point], truth: Sequence[Point]) -> float:
    """For each true corner, find the nearest proposed corner; return the
    mean of those nearest distances. Symmetric Hausdorff-ish but biased
    toward 'did we miss any GT corners' rather than 'did we add spurious
    ones'. Returns +inf if proposed is empty."""
    if not proposed:
        return float("inf")
    if not truth:
        return 0.0
    p = np.asarray(proposed, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    # For each GT corner, distance to nearest proposed corner
    dists = np.linalg.norm(t[:, None, :] - p[None, :, :], axis=2)
    nearest = dists.min(axis=1)
    return float(nearest.mean())


def canonicalize_quad_order(quad: Sequence[Point]) -> List[Point]:
    """Sort 4 corners clockwise from north (same convention used by
    sample_stickers_from_hull.canonical_corner_order). Lets us compare
    'corner 0 → corner 0' across proposers and labels."""
    if len(quad) != 4:
        return list(quad)
    cx = sum(p[0] for p in quad) / 4.0
    cy = sum(p[1] for p in quad) / 4.0

    def key(p):
        from math import atan2
        a = atan2(p[0] - cx, -(p[1] - cy))  # 0 at north, CW positive
        if a < 0:
            a += 2 * np.pi
        return a

    return sorted(quad, key=key)


# ---------------- ground-truth loading ----------------


@dataclass
class LabelTarget:
    """A single (setId, side) pair with hand-labeled ground-truth geometry."""
    set_id: str
    side: str
    image_path: Path
    label_path: Path
    gt_state: Optional[str] = None  # 54-char URFDLB ground-truth state for sticker-classification eval
    # Filled in by load():
    image: Optional[Image.Image] = None
    proc_w: int = 0
    proc_h: int = 0
    arr: Optional[np.ndarray] = None
    gt_hull: List[Point] = field(default_factory=list)
    gt_face_quads: Dict[str, List[Point]] = field(default_factory=dict)

    def load(self) -> None:
        with Image.open(self.image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        natural_max = max(image.size)
        if natural_max > PROCESSING_MAX:
            scale = PROCESSING_MAX / float(natural_max)
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        self.image = image
        self.proc_w, self.proc_h = image.size
        self.arr = np.asarray(image)
        doc = load_hull_label(self.label_path)
        self.gt_face_quads = {
            face: [(float(x), float(y)) for (x, y) in quad]
            for face, quad in scaled_face_quads(doc, self.proc_w, self.proc_h).items()
        }
        # Cube hull: scale from natural to processing coords manually
        natural_w = float((doc.get("image") or {}).get("width") or self.proc_w)
        natural_h = float((doc.get("image") or {}).get("height") or self.proc_h)
        sx = self.proc_w / max(1.0, natural_w)
        sy = self.proc_h / max(1.0, natural_h)
        raw_hull = (doc.get("labels") or {}).get("cubeHull") or []
        self.gt_hull = [(float(p["x"]) * sx, float(p["y"]) * sy) for p in raw_hull if isinstance(p, dict)]
        if len(self.gt_hull) < 3 and self.gt_face_quads:
            # Fall back to convex hull of face-quad vertices
            all_pts = [pt for q in self.gt_face_quads.values() for pt in q]
            self.gt_hull = convex_hull(all_pts)


def discover_label_targets() -> List[LabelTarget]:
    """Find every (setId, side) that has both a hull label AND a discoverable
    image path (via corpus_manifest or via Downloads discovery)."""
    tasks = load_corpus_tasks(REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json")
    tasks.extend(discover_additional_tasks({t.set_id for t in tasks}))

    targets: List[LabelTarget] = []
    for task in tasks:
        try:
            gt_state = parse_ground_truth(task.ground_truth)
        except Exception:
            gt_state = None
        for side, image_path in (("A", task.image_a), ("B", task.image_b)):
            label_path = latest_hull_label(task.set_id, side)
            if label_path is None:
                continue
            if not image_path.exists():
                continue
            targets.append(LabelTarget(
                set_id=task.set_id,
                side=side,
                image_path=image_path,
                label_path=label_path,
                gt_state=gt_state,
            ))
    return targets


# ---------------- evaluation ----------------


def evaluate_target(target: LabelTarget, proposer) -> Dict:
    """Run one proposer on one target; compute all metrics."""
    target.load()
    expected_faces = EXPECTED_FACES_BY_SIDE[target.side]
    try:
        proposal: Proposal = proposer.propose(target)
    except Exception as e:
        return {
            "setId": target.set_id,
            "side": target.side,
            "proposer": proposer.name,
            "error": f"{type(e).__name__}: {e}",
        }

    w, h = target.proc_w, target.proc_h
    hull_iou = polygon_iou(proposal.cube_hull, target.gt_hull, w, h)

    per_face: Dict[str, Dict] = {}
    for face in expected_faces:
        gt = target.gt_face_quads.get(face)
        prop = proposal.face_quads.get(face)
        if gt is None:
            continue
        if prop is None:
            per_face[face] = {"iou": 0.0, "containment_of_gt": 0.0, "mean_corner_error_px": None}
            continue
        per_face[face] = {
            "iou": round(polygon_iou(prop, gt, w, h), 4),
            "containment_of_gt": round(polygon_containment(gt, prop, w, h), 4),
            "mean_corner_error_px": round(
                mean_corner_error(canonicalize_quad_order(prop), canonicalize_quad_order(gt)), 2
            ),
        }

    face_iou_vals = [m["iou"] for m in per_face.values()]
    face_corner_vals = [m["mean_corner_error_px"] for m in per_face.values() if m.get("mean_corner_error_px") is not None]

    # Hungarian-best-match IoU: ignore proposer's face labels, find the
    # assignment of proposed-quads → GT-quads that maximises total IoU.
    # This separates "did the proposer find the geometry" from "did it
    # label faces correctly". For proposers like saturation_hexagon
    # where label assignment is a separate (and often-wrong) step, this
    # is the more honest geometry score.
    proposed_quads = list(proposal.face_quads.values())
    gt_quads = [target.gt_face_quads[f] for f in expected_faces if f in target.gt_face_quads]
    matched_face_iou = _best_match_face_iou(proposed_quads, gt_quads, w, h)

    impact = recognizer_impact_diagnostics(target, proposal)

    # Devin PR-#133 ask: sticker-center pixel error. Measures how far the
    # proposer's would-be color samples would land from where they should.
    sticker_err_px, sticker_err_per_face = mean_sticker_center_error(
        proposal.face_quads, target.gt_face_quads,
    )

    # Devin PR-#133 second ask: end-to-end sticker classification accuracy.
    # If the proposer's face quads drove a recognizer_mask path, what
    # fraction of stickers would classify correctly?
    classification = sticker_classification_accuracy(target, proposal.face_quads)

    return {
        "setId": target.set_id,
        "side": target.side,
        "proposer": proposer.name,
        "imageSize": [w, h],
        "cubeHullIoU": round(hull_iou, 4),
        "perFace": per_face,
        "meanFaceIoU_byLabel": round(float(np.mean(face_iou_vals)), 4) if face_iou_vals else 0.0,
        "meanFaceIoU_bestMatch": round(matched_face_iou, 4),
        "meanCornerErrorPx_byLabel": round(float(np.mean(face_corner_vals)), 2) if face_corner_vals else None,
        "meanStickerCenterErrorPx": sticker_err_px if sticker_err_px != float("inf") else None,
        "stickerCenterErrorPxPerFace": sticker_err_per_face,
        "stickerClassificationAccuracy": classification.get("accuracy"),
        "stickerClassificationDetail": classification,
        "recognizerImpact": impact,
        "proposerNotes": proposal.notes,
    }


_ANALYSIS_CACHE: Dict[str, object] = {}


def _cached_analyze_image(image_path: Path):
    """Single analyze_image call per image path across all proposers in
    one run. Cuts the sweep runtime ~4× since recognizer_impact_diagnostics
    runs the recognizer once per (target, proposer) and `recognizer_grids`
    also runs it once."""
    key = str(image_path)
    if key not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[key] = analyze_image(image_path.read_bytes())
    return _ANALYSIS_CACHE[key]


def sticker_classification_accuracy(
    target: "LabelTarget",
    proposed_face_quads: Dict[str, Sequence[Point]],
    inset: float = 0.20,
) -> Dict:
    """Devin's PR-#133 ask: if the proposer's face quads drove a real
    per-sticker color sampler (the recognizer_mask path we'd build next),
    what fraction of stickers would classify correctly?

    For each proposed face quad: derive 9 sticker centers via the same
    homography used by sample_stickers_from_hull.py, sample 15×15 median
    RGB at each, classify with the default classify_rgb (canonical
    palette), and compare to the ground-truth state's color at the
    matched (face, row, col) position.

    Per-face identity uses center-classification (with U/D anchor trust
    for the U-logo issue) to map the proposer's label → true face — same
    logic that handles L/B-swap and yaw rotations in PR #126.

    Returns dict with per-face accuracy + overall. None if no GT state
    or no proposed face quads."""
    if target.gt_state is None or not proposed_face_quads:
        return {"sampled": 0, "correct": 0, "accuracy": None, "perFace": {}}
    if target.arr is None:
        target.load()
    arr = target.arr
    anchor = "U" if target.side == "A" else "D"

    per_face: Dict[str, Dict] = {}
    total_sampled = 0
    total_correct = 0
    for label_face, quad in proposed_face_quads.items():
        if len(quad) != 4:
            continue
        canonical = _canonical_corner_order([tuple(p) for p in quad])
        centers = _sticker_centers(canonical, inset=inset)
        if len(centers) != 9:
            continue
        # Identify true face: trust label for the anchor (U/D — U-center has
        # the Rubik's logo so classifying it returns wrong color); otherwise
        # classify the center sticker and reverse-lookup.
        if label_face == anchor:
            true_face = anchor
        else:
            center_rgb = _sample_rgb(arr, *centers[4])
            cls = classify_rgb(center_rgb).color
            true_face = COLOR_TO_FACE.get(cls, label_face)

        gt_colors = face_colors_from_state(target.gt_state, true_face)
        face_correct = 0
        for (x, y), gt in zip(centers, gt_colors):
            rgb = _sample_rgb(arr, x, y)
            if classify_rgb(rgb).color == gt:
                face_correct += 1
        per_face[label_face] = {
            "trueFace": true_face,
            "sampled": 9,
            "correct": face_correct,
            "accuracy": round(face_correct / 9, 4),
        }
        total_sampled += 9
        total_correct += face_correct

    return {
        "sampled": total_sampled,
        "correct": total_correct,
        "accuracy": round(total_correct / total_sampled, 4) if total_sampled else None,
        "perFace": per_face,
    }


def _sticker_centers_per_face(
    face_quads: Dict[str, Sequence[Point]],
    inset: float = 0.20,
) -> Dict[str, List[Point]]:
    """For each face quad, return the 9 sticker centers a downstream
    color-sampler would land at. Uses the same homography + inset that
    sample_stickers_from_hull.py uses, so the numbers are comparable to
    the clean-label pipeline from PR #126."""
    out: Dict[str, List[Point]] = {}
    for face, quad in face_quads.items():
        if len(quad) != 4:
            continue
        canonical = _canonical_corner_order([tuple(p) for p in quad])
        out[face] = _sticker_centers(canonical, inset=inset)
    return out


def mean_sticker_center_error(
    proposed_face_quads: Dict[str, Sequence[Point]],
    truth_face_quads: Dict[str, Sequence[Point]],
    inset: float = 0.20,
) -> Tuple[float, Dict[str, float]]:
    """Per Devin's PR-#133 ask: how far (in pixels) would the proposer's
    sticker samples land from where they should? For each face present in
    BOTH proposed and truth, derive 9 sticker centers via the standard
    homography and measure mean pairwise distance (position i in proposed
    vs position i in truth). Returns (overall_mean_px, per_face_mean_px).

    A clean test of 'would my colors land in the right stickers' — if
    this error exceeds ~half a sticker spacing (typically 30-40px on
    cube-snap photos), samples bleed into adjacent stickers or bezel."""
    proposed_centers = _sticker_centers_per_face(proposed_face_quads, inset)
    truth_centers = _sticker_centers_per_face(truth_face_quads, inset)
    per_face: Dict[str, float] = {}
    all_errors: List[float] = []
    for face in proposed_centers.keys() & truth_centers.keys():
        p_pts = proposed_centers[face]
        t_pts = truth_centers[face]
        if len(p_pts) != 9 or len(t_pts) != 9:
            continue
        # Element-wise pixel distance (position i vs position i, same
        # canonicalized corner order via sample_stickers_from_hull's helper).
        errs = [
            ((p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2) ** 0.5
            for p, t in zip(p_pts, t_pts)
        ]
        per_face[face] = round(float(np.mean(errs)), 2)
        all_errors.extend(errs)
    overall = float(np.mean(all_errors)) if all_errors else float("inf")
    return round(overall, 2), per_face


def recognizer_impact_diagnostics(
    target: "LabelTarget",
    proposal: "Proposal",
) -> Dict:
    """'Would the proposed geometry have helped recognition?'

    For each proposed cube hull / face quad set, run analyze_image on the
    same photo and report:

      * outsideHullStickerCount — detected stickers whose centers fall
        outside the proposed cube hull (high count = proposer missed cube
        area OR mask is contaminated by background)
      * outsideAllFacesStickerCount — detected stickers not inside any
        proposed face quad
      * stickersPerProposedFace — per face: how many recognizer-detected
        stickers fall inside. Ideal=9 per face.
      * recognizerGridsContainedFraction — for each best-matched-count
        FaceGrid the recognizer chose per center_face, what fraction of
        its 9 sticker centers lie inside ANY proposed face quad. A
        proposer's face quad set "validates" a recognizer grid when this
        is ≥0.78 (7+/9 inside); below that the grid would effectively be
        rejected by the proposer's geometry.
      * recognizerGridsAccepted — count of recognizer-chosen grids that
        clear the 7/9-contained threshold

    These metrics turn the IoU numbers into 'so what for the recognizer's
    actual job', per Devin's PR-#127 ask."""
    # Cached so one analyze_image call serves all proposers on the same
    # image (the recognizer_grids proposer also runs analyze_image; without
    # caching this would 5× the wall time on a full sweep).
    analysis = _cached_analyze_image(target.image_path)
    stickers = analysis.stickers

    proposed_hull = proposal.cube_hull if len(proposal.cube_hull) >= 3 else []
    proposed_faces = list(proposal.face_quads.items())  # [(face_name, quad)]

    outside_hull = 0
    outside_all_faces = 0
    stickers_per_face: Dict[str, int] = {f: 0 for f, _ in proposed_faces}

    for s in stickers:
        cx, cy = float(s.center[0]), float(s.center[1])
        if proposed_hull and not point_in_polygon((cx, cy), proposed_hull):
            outside_hull += 1
        in_any_face = False
        for face_name, quad in proposed_faces:
            if len(quad) >= 3 and point_in_polygon((cx, cy), quad):
                stickers_per_face[face_name] += 1
                in_any_face = True
                break
        if proposed_faces and not in_any_face:
            outside_all_faces += 1

    # Recognizer's best grid per face (its actual choice via matched_count)
    grids_by_face: Dict[str, list] = {}
    for grid in analysis.grids:
        grids_by_face.setdefault(grid.center_face, []).append(grid)
    best_grids = {
        face: min(cands, key=lambda g: (-g.matched_count, g.fit_error))
        for face, cands in grids_by_face.items()
    }

    grid_containment: Dict[str, float] = {}
    accepted = 0
    for face, grid in best_grids.items():
        # 9 sticker centers from this grid
        centers: List[Point] = []
        for row in grid.points:
            for (gx, gy) in row:
                centers.append((float(gx), float(gy)))
        if not centers:
            continue
        inside = 0
        for cx, cy in centers:
            for _f, quad in proposed_faces:
                if len(quad) >= 3 and point_in_polygon((cx, cy), quad):
                    inside += 1
                    break
        frac = inside / len(centers)
        grid_containment[face] = round(frac, 3)
        if frac >= 7 / 9:
            accepted += 1

    return {
        "stickerCount": len(stickers),
        "outsideHullStickerCount": outside_hull,
        "outsideHullFraction": round(outside_hull / max(1, len(stickers)), 3),
        "outsideAllFacesStickerCount": outside_all_faces,
        "outsideAllFacesFraction": round(outside_all_faces / max(1, len(stickers)), 3),
        "stickersPerProposedFace": stickers_per_face,
        "recognizerBestGridContainment": grid_containment,
        "recognizerGridsAccepted": accepted,
        "recognizerGridsConsidered": len(best_grids),
    }


def _best_match_face_iou(
    proposed: Sequence[Polygon],
    truth: Sequence[Polygon],
    width: int,
    height: int,
) -> float:
    """Hungarian-style: build a |P|×|T| IoU matrix, find the assignment
    that maximises sum-IoU, return the mean IoU of matched pairs (averaged
    over the LARGER of |P|, |T| so under/over-proposal is penalized).
    Brute force over permutations — fine for ≤4 quads per side."""
    if not proposed or not truth:
        return 0.0
    from itertools import permutations
    iou_mat = np.zeros((len(proposed), len(truth)))
    for i, p in enumerate(proposed):
        for j, t in enumerate(truth):
            iou_mat[i, j] = polygon_iou(p, t, width, height)
    n, m = len(proposed), len(truth)
    if n <= m:
        # try every assignment of n proposed → some n-subset of m truth
        best = 0.0
        for choice in permutations(range(m), n):
            s = sum(iou_mat[i, choice[i]] for i in range(n))
            best = max(best, s)
        return best / max(n, m)
    else:
        best = 0.0
        for choice in permutations(range(n), m):
            s = sum(iou_mat[choice[j], j] for j in range(m))
            best = max(best, s)
        return best / max(n, m)


def render_overlay(
    target: LabelTarget,
    proposal: Proposal,
    out_path: Path,
) -> None:
    """Solid lines = proposed geometry; dashed = ground truth."""
    if target.image is None:
        target.load()
    canvas = target.image.copy()
    draw = ImageDraw.Draw(canvas)

    def draw_polygon(poly: Polygon, color, width: int = 3, dashed: bool = False):
        if len(poly) < 2:
            return
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i + 1) % len(poly)]
            if dashed:
                # Sample dashes along the segment
                dx, dy = b[0] - a[0], b[1] - a[1]
                length = (dx * dx + dy * dy) ** 0.5
                if length == 0:
                    continue
                steps = max(2, int(length / 12))
                for step in range(steps):
                    if step % 2:
                        continue
                    t0 = step / steps
                    t1 = (step + 0.6) / steps
                    p0 = (a[0] + dx * t0, a[1] + dy * t0)
                    p1 = (a[0] + dx * t1, a[1] + dy * t1)
                    draw.line([p0, p1], fill=color, width=width)
            else:
                draw.line([a, b], fill=color, width=width)

    # Ground truth — dashed
    draw_polygon(target.gt_hull, (255, 240, 80), width=3, dashed=True)
    for face, quad in target.gt_face_quads.items():
        draw_polygon(quad, (255, 240, 80), width=2, dashed=True)

    # Proposed — solid
    draw_polygon(proposal.cube_hull, (50, 200, 255), width=3, dashed=False)
    for face, quad in proposal.face_quads.items():
        draw_polygon(quad, (255, 80, 80), width=2, dashed=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposers", nargs="+", default=list(PROPOSERS.keys()),
                    choices=list(PROPOSERS.keys()),
                    help="Which proposers to evaluate")
    ap.add_argument("--set", action="append", default=None,
                    help="Limit to specific setId(s); repeatable")
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--overlays-dir", default=str(DEFAULT_OVERLAY_DIR))
    ap.add_argument("--no-overlays", action="store_true")
    args = ap.parse_args()

    targets = discover_label_targets()
    if args.set:
        wanted = set(args.set)
        targets = [t for t in targets if t.set_id in wanted]

    print(f"evaluating {len(targets)} (set, side) targets across {len(args.proposers)} proposer(s)",
          file=sys.stderr)

    rows: List[Dict] = []
    for proposer_name in args.proposers:
        proposer = PROPOSERS[proposer_name]
        print(f"\n== proposer: {proposer_name} ==", file=sys.stderr)
        for target in targets:
            row = evaluate_target(target, proposer)
            rows.append(row)
            if "error" in row:
                print(f"  set {target.set_id} {target.side}: ERROR {row['error']}", file=sys.stderr)
                continue
            imp = row.get("recognizerImpact") or {}
            print(
                f"  set {target.set_id} {target.side}: hullIoU={row['cubeHullIoU']:.3f}  "
                f"face(byLabel)={row['meanFaceIoU_byLabel']:.3f}  "
                f"face(bestMatch)={row['meanFaceIoU_bestMatch']:.3f}  "
                f"outsideHull={imp.get('outsideHullStickerCount', '?')}/{imp.get('stickerCount', '?')}  "
                f"gridsAccepted={imp.get('recognizerGridsAccepted', '?')}/{imp.get('recognizerGridsConsidered', '?')}",
                file=sys.stderr,
            )
            if not args.no_overlays:
                try:
                    proposal: Proposal = proposer.propose(target)
                    out_path = Path(args.overlays_dir) / proposer_name / f"{target.set_id}-{target.side}.png"
                    render_overlay(target, proposal, out_path)
                except Exception as e:
                    print(f"    overlay render failed: {e}", file=sys.stderr)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rows, indent=2))

    # Summary
    summary_lines: List[str] = []
    by_proposer: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        if "error" not in row:
            by_proposer[row["proposer"]].append(row)

    summary_lines.append(f"Auto-geometry evaluation: {len(targets)} targets × {len(args.proposers)} proposer(s)")
    summary_lines.append("")
    summary_lines.append(
        f"{'proposer':<30s}  {'n':>4s}  {'hullIoU':>9s}  {'face(bM)':>10s}  {'ctrErrPx':>10s}  {'stickerClsAcc':>14s}"
    )
    summary_lines.append("-" * 88)
    for proposer_name in args.proposers:
        rows_p = by_proposer.get(proposer_name, [])
        n = len(rows_p)
        if n == 0:
            summary_lines.append(f"{proposer_name:<30s}  {n:>4d}  {'—':>9s}  {'—':>10s}  {'—':>10s}  {'—':>14s}")
            continue
        m_hull = float(np.mean([r["cubeHullIoU"] for r in rows_p]))
        m_face_bm = float(np.mean([r["meanFaceIoU_bestMatch"] for r in rows_p]))
        sticker_errs = [r["meanStickerCenterErrorPx"] for r in rows_p if r.get("meanStickerCenterErrorPx") is not None]
        m_sticker_err = float(np.mean(sticker_errs)) if sticker_errs else None
        m_sticker_err_str = f"{m_sticker_err:>7.1f}px" if m_sticker_err is not None else "—"
        cls_accs = [r["stickerClassificationAccuracy"] for r in rows_p if r.get("stickerClassificationAccuracy") is not None]
        m_cls_acc = float(np.mean(cls_accs)) if cls_accs else None
        m_cls_acc_str = f"{m_cls_acc:>13.1%}" if m_cls_acc is not None else "—"
        summary_lines.append(
            f"{proposer_name:<30s}  {n:>4d}  {m_hull:>9.3f}  {m_face_bm:>10.3f}  {m_sticker_err_str:>10s}  {m_cls_acc_str:>14s}"
        )

    # Per-proposer pass thresholds (using best-match face IoU since labels
    # are sometimes wrong even when geometry is right)
    summary_lines.append("")
    summary_lines.append("pass rates (per-pair, using best-match face IoU):")
    for proposer_name in args.proposers:
        rows_p = by_proposer.get(proposer_name, [])
        if not rows_p:
            continue
        # Pass-gate raised to face >= 0.85 per Devin PR-#133 feedback
        # (was 0.75; tightened because lower thresholds don't reflect what
        # the recognizer actually needs for clean per-sticker sampling).
        hull_pass = sum(1 for r in rows_p if r["cubeHullIoU"] >= 0.85)
        face_pass = sum(1 for r in rows_p if r["meanFaceIoU_bestMatch"] >= 0.85)
        both_pass = sum(1 for r in rows_p if r["cubeHullIoU"] >= 0.85 and r["meanFaceIoU_bestMatch"] >= 0.85)
        n = len(rows_p)
        summary_lines.append(
            f"  {proposer_name:<28s}  hull≥0.85: {hull_pass}/{n} ({hull_pass/n:.0%})  "
            f"face≥0.85: {face_pass}/{n} ({face_pass/n:.0%})  "
            f"both: {both_pass}/{n} ({both_pass/n:.0%})"
        )

    # Recognizer-impact aggregates: would this proposer have helped the
    # actual recognizer's job? Average across pairs.
    summary_lines.append("")
    summary_lines.append("recognizer impact (mean across pairs):")
    summary_lines.append(
        f"  {'proposer':<28s}  {'outsideHull':>13s}  {'outsideAllFaces':>16s}  {'gridsAccepted':>13s}"
    )
    for proposer_name in args.proposers:
        rows_p = by_proposer.get(proposer_name, [])
        impacts = [r["recognizerImpact"] for r in rows_p if "recognizerImpact" in r]
        if not impacts:
            continue
        m_outside_hull = float(np.mean([i["outsideHullFraction"] for i in impacts]))
        m_outside_all = float(np.mean([i["outsideAllFacesFraction"] for i in impacts]))
        m_accepted = float(np.mean([
            i["recognizerGridsAccepted"] / max(1, i["recognizerGridsConsidered"]) for i in impacts
        ]))
        summary_lines.append(
            f"  {proposer_name:<28s}  {m_outside_hull:>12.1%}   {m_outside_all:>15.1%}   "
            f"{m_accepted:>12.1%}"
        )

    # Per-proposer worst-3 (by hullIoU)
    summary_lines.append("")
    summary_lines.append("worst 3 per proposer (by cubeHullIoU):")
    for proposer_name in args.proposers:
        rows_p = by_proposer.get(proposer_name, [])
        worst = sorted(rows_p, key=lambda r: r["cubeHullIoU"])[:3]
        if not worst:
            continue
        summary_lines.append(f"  {proposer_name}:")
        for r in worst:
            summary_lines.append(
                f"    set {r['setId']} {r['side']}: hullIoU={r['cubeHullIoU']:.3f} "
                f"face(bestMatch)={r['meanFaceIoU_bestMatch']:.3f}"
            )

    Path(args.summary).write_text("\n".join(summary_lines) + "\n")
    print("\n" + "\n".join(summary_lines), file=sys.stderr)
    print(f"\nwrote {args.report}", file=sys.stderr)
    print(f"wrote {args.summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
