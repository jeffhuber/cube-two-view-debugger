#!/usr/bin/env python3
"""Geometric per-sticker sampling from hull labels (no detection needed).

For each (set, side) with a hull label:
  1. Take the 4 corners of each face quad as the GROUND TRUTH face geometry.
  2. Sort corners clockwise around the centroid for a canonical order.
  3. Fit a projective homography from the unit square [(0,0),(1,0),(1,1),(0,1)]
     onto the 4 image corners (exact, no fitting — 4 points → 4 points).
  4. Compute 9 sticker centers at unit-square coords (1/6, 3/6, 5/6)² and
     warp them back to image space.
  5. Sample RGB as the median of a 15×15 patch at each center.
  6. Discover the correct grid orientation (4 rotations × 2 mirrors = 8)
     by matching sampled colors (via current classify_rgb) against the
     ground-truth state's 9 colors for that face. The winning orientation
     is the unique permutation that aligns sampled-position-9 with
     ground-truth-position-9.
  7. Render an overlay: each of the 27 (or 26) computed positions drawn as
     a small filled dot color-coded by GROUND TRUTH color. If geometry +
     orientation are correct, every dot lands inside its sticker AND its
     color visually matches the sticker.

Mode `--overlay`: render PNGs to /tmp/ for a single set, for visual inspection.
Mode `--dataset`: dump JSONL with (face, row, col, RGB, gtColor) for all
labeled sets.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from math import atan2
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from collections import Counter  # noqa: E402

from rubik_recognizer.colors import (  # noqa: E402
    COLOR_ORDER,
    COLOR_TO_FACE,
    FACE_TO_COLOR,
    build_adaptive_palette,
    classify_rgb,
)
from rubik_recognizer.validation import FACE_ORDER  # noqa: E402
from tools.extract_color_samples import (  # noqa: E402
    EXPECTED_FACES_BY_SIDE,
    discover_additional_tasks,
    face_colors_from_state,
    latest_hull_label,
    load_corpus_tasks,
    load_hull_label,
    parse_ground_truth,
    scaled_face_quads,
)

OUT_DIR = Path("/tmp")
CORPUS_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"

# Visual colors for overlay dots (matches Rubik's standard palette)
DOT_COLORS = {
    "white":  (250, 250, 250),
    "yellow": (255, 220, 0),
    "red":    (220, 35, 30),
    "orange": (245, 130, 35),
    "green":  (50, 165, 80),
    "blue":   (60, 90, 220),
}


# ---------- geometry ----------

def canonical_corner_order(corners: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sort 4 corners clockwise around their centroid, starting at the
    angularly-topmost (north-most) corner. Image y grows downward, so
    'north' is min y → angle = -π/2 baseline."""
    cx = sum(p[0] for p in corners) / 4.0
    cy = sum(p[1] for p in corners) / 4.0

    def key(p):
        # Angle CW from north (top). atan2 returns radians in (-π, π] with 0 east.
        # We want north = 0 and CW positive, so angle = (atan2(dx, -dy)) mod 2π.
        dx = p[0] - cx
        dy = p[1] - cy
        a = atan2(dx, -dy)  # 0 at north, +π/2 east, ±π south
        if a < 0:
            a += 2 * np.pi
        return a

    return sorted(corners, key=key)


def homography_unit_to_quad(quad: Sequence[Tuple[float, float]]) -> np.ndarray:
    """4-point planar homography: unit square [(0,0),(1,0),(1,1),(0,1)] → quad.
    Returns 3x3 H such that (x, y, 1) @ H.T projectively maps to image."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    dst = np.array(quad, dtype=np.float64)
    A = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        A.append([-sx, -sy, -1, 0, 0, 0, sx * dx, sy * dx, dx])
        A.append([0, 0, 0, -sx, -sy, -1, sx * dy, sy * dy, dy])
    A = np.array(A, dtype=np.float64)
    _, _, vh = np.linalg.svd(A)
    H = vh[-1].reshape(3, 3)
    return H / H[2, 2]


def warp(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def sticker_centers(quad: Sequence[Tuple[float, float]], inset: float = 0.20) -> List[Tuple[float, float]]:
    """Return 9 sticker centers in image coords, in row-major order
    (row 0 = closest to corner[0]). Coordinates: u along corner[0]→corner[1]
    edge, v along corner[0]→corner[3] edge.

    `inset` is the distance from a face edge to the center of the corner
    sticker, expressed as a fraction of the face's edge length. inset=1/6
    assumes the labeled quad is exactly the sticker area (no bezels);
    larger values (~0.18-0.22) account for the bezel/grout area when the
    labeler traced the cube face perimeter."""
    H = homography_unit_to_quad(quad)
    centers: List[Tuple[float, float]] = []
    coords = (inset, 0.5, 1.0 - inset)
    for v in coords:
        for u in coords:
            centers.append(warp(H, u, v))
    return centers


def sample_rgb(arr: np.ndarray, x: float, y: float, half: int = 7) -> Tuple[int, int, int]:
    """Median RGB in a (2*half+1)² patch around (x, y). Clamps to bounds."""
    h, w = arr.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)
    if x0 >= x1 or y0 >= y1:
        return (0, 0, 0)
    patch = arr[y0:y1, x0:x1].reshape(-1, 3)
    return tuple(int(np.median(patch[:, c])) for c in range(3))


# ---------- orientation discovery ----------

def apply_orientation(positions: List, mirror: bool, rot_quarter: int) -> List:
    """positions is a 9-list in row-major (r, c) order: indices 0..8 = (0,0),(0,1),(0,2),(1,0),...
    Apply optional horizontal mirror then rotate CCW by 90° * rot_quarter."""
    grid = [positions[r * 3:(r + 1) * 3] for r in range(3)]
    if mirror:
        grid = [row[::-1] for row in grid]
    for _ in range(rot_quarter % 4):
        # CCW rotation: row r col c -> row (2-c) col r
        rotated = [[None] * 3 for _ in range(3)]
        for r in range(3):
            for c in range(3):
                rotated[2 - c][r] = grid[r][c]
        grid = rotated
    return [grid[r][c] for r in range(3) for c in range(3)]


def discover_orientation(
    rgbs_row_major: List[Tuple[int, int, int]],
    gt_colors_row_major: Sequence[str],
    palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> Tuple[bool, int, float]:
    """Try all 8 orientations of the sampled-9 against ground truth.
    Returns (mirror, rot_quarter, score) for the best.

    Score is confidence-weighted: a confident-correct prediction adds its
    full confidence; a confident-wrong prediction subtracts a small
    fraction of its confidence. This lets a handful of high-confidence
    correct stickers (white, clean greens) outvote many noisy ones.

    Pass `palette` to use the adaptive (per-image) calibrated classifier
    instead of the default canonical palette. Required for shadowed faces
    (Image B's L/B faces in particular)."""
    matches = [classify_rgb(rgb, prototypes=palette) for rgb in rgbs_row_major]
    pairs = [(m.color, m.confidence) for m in matches]
    best = (False, 0, float("-inf"))
    for mirror, rot in product([False, True], range(4)):
        reordered = apply_orientation(pairs, mirror, rot)
        score = 0.0
        for (col, conf), gt in zip(reordered, gt_colors_row_major):
            if col == gt:
                score += conf + 0.05  # small floor so even low-conf correctness counts
            else:
                score -= 0.25 * conf  # penalize confident wrong
        if score > best[2]:
            best = (mirror, rot, score)
    return best


# ---------- per-face sampling ----------

def sample_face(
    arr: np.ndarray,
    quad_image_coords: Sequence[Tuple[float, float]],
    face: str,
    gt_colors: Sequence[str],
    inset: float = 0.20,
    palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> Dict:
    """Sample 9 stickers + discover orientation. Returns dict with positions,
    rgbs (in row-major sampled order), classified (sampled order), orientation,
    and the (face_position_index, image_xy, rgb, gt_color) per-sticker rows."""
    canonical_quad = canonical_corner_order([tuple(p) for p in quad_image_coords])
    centers_sampled = sticker_centers(canonical_quad, inset=inset)
    rgbs_sampled = [sample_rgb(arr, x, y) for (x, y) in centers_sampled]
    classified_sampled = [classify_rgb(rgb, prototypes=palette).color for rgb in rgbs_sampled]

    mirror, rot, matches = discover_orientation(rgbs_sampled, gt_colors, palette=palette)

    # Reorder the 9 stickers to match ground-truth position order
    centers_aligned = apply_orientation(centers_sampled, mirror, rot)
    rgbs_aligned = apply_orientation(rgbs_sampled, mirror, rot)
    classified_aligned = apply_orientation(classified_sampled, mirror, rot)

    per_sticker = []
    for i in range(9):
        per_sticker.append({
            "face": face,
            "row": i // 3,
            "col": i % 3,
            "xy": [round(centers_aligned[i][0], 1), round(centers_aligned[i][1], 1)],
            "rgb": list(rgbs_aligned[i]),
            "classifier": classified_aligned[i],
            "gtColor": gt_colors[i],
        })
    return {
        "face": face,
        "canonical_quad": canonical_quad,
        "orientation": {"mirror": mirror, "rotation_quarters_ccw": rot, "match_count": matches},
        "stickers": per_sticker,
    }


# ---------- pipeline ----------

def process_side(
    image_path: Path,
    hull_doc: Dict,
    gt_state: str,
    side: str,
    inset: float = 0.20,
) -> Dict:
    """Returns dict with image, all face samples, plus the rendered overlay path."""
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    # Match analyze_image's max-side=1150 resize so quad scaling stays consistent.
    natural_max = max(image.size)
    if natural_max > 1150:
        scale = 1150 / float(natural_max)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    arr = np.asarray(image)
    proc_w, proc_h = image.size

    quads = scaled_face_quads(hull_doc, proc_w, proc_h)
    expected = EXPECTED_FACES_BY_SIDE[side]
    return {"image": image, "arr": arr, "quads": quads, "expected": expected}


def _multiset_overlap(
    label: str,
    true_face: str,
    sampled_multisets: Dict[str, Counter],
    gt_state: str,
) -> int:
    if label not in sampled_multisets:
        return 0
    gt_mset = Counter(face_colors_from_state(gt_state, true_face))
    samp = sampled_multisets[label]
    return sum(min(samp.get(c, 0), gt_mset.get(c, 0)) for c in COLOR_ORDER)


def _sample_multisets(prepared: Dict, inset: float) -> Dict[str, Counter]:
    arr = prepared["arr"]
    quads = prepared["quads"]
    out: Dict[str, Counter] = {}
    for label in prepared["expected"]:
        if label not in quads:
            continue
        canonical_quad = canonical_corner_order([tuple(p) for p in quads[label]])
        centers = sticker_centers(canonical_quad, inset=inset)
        rgbs = [sample_rgb(arr, x, y) for (x, y) in centers]
        out[label] = Counter(classify_rgb(rgb).color for rgb in rgbs)
    return out


def identify_faces_jointly(
    prepared_sides: Dict[str, Dict],
    gt_state: str,
    inset: float,
) -> Tuple[Dict[str, Dict[str, str]], int, str]:
    """Pair-level face identification: enumerates the 16 valid (yaw ×
    side-A-label-order × side-B-label-order) combinations and picks the
    one with the highest TOTAL multiset overlap across both images.

    The pair-level invariant — across A+B, exactly 6 distinct true faces
    (U/R/F/D/L/B) — is enforced by construction. The 4 valid yaw configs
    pair side-A side-faces with side-B side-faces such that their union
    spans {R, F, L, B}:

        A_yaw0:  A side=(R, F)  →  B side=(L, B)  (canonical)
        A_yaw1:  A side=(R, B)  →  B side=(L, F)
        A_yaw2:  A side=(L, B)  →  B side=(R, F)  (Set 23 was here)
        A_yaw3:  A side=(L, F)  →  B side=(R, B)

    Returns ({side: label_to_true}, best_score, status).
    status ∈ {"ok", "ambiguous", "missing_side"}. "ambiguous" means the
    top-2 scores are within `tie_margin` of each other — caller should
    skip such pairs to avoid contaminating the clean dataset."""
    # Each yaw fixes the set of A-side faces and B-side faces. Within
    # each side, the 2 labels can be permuted onto the 2 true faces.
    yaw_configs = [
        ("yaw0", ("R", "F"), ("L", "B")),  # canonical
        ("yaw1", ("R", "B"), ("L", "F")),
        ("yaw2", ("L", "B"), ("R", "F")),  # Set 23 pair
        ("yaw3", ("L", "F"), ("R", "B")),
    ]

    sampled_by_side = {side: _sample_multisets(p, inset) for side, p in prepared_sides.items()}
    have_a = "A" in prepared_sides
    have_b = "B" in prepared_sides
    if not (have_a and have_b):
        # Fall back to per-side (rare; only when one side has no hull label)
        per_side: Dict[str, Dict[str, str]] = {}
        for side, prepared in prepared_sides.items():
            anchor = "U" if side == "A" else "D"
            mapping = {anchor: anchor}
            for label in prepared["expected"]:
                if label != anchor and label in prepared["quads"]:
                    mapping[label] = label  # best effort
            per_side[side] = mapping
        return per_side, 0, "missing_side"

    expected_a = ["R", "F"]  # labeler always writes labels in this order for A
    expected_b = ["L", "B"]  # ...and this for B

    best_score = -1
    best_second = -1
    best_mapping: Dict[str, Dict[str, str]] = {}
    for _name, a_side_faces, b_side_faces in yaw_configs:
        for a_perm in (a_side_faces, a_side_faces[::-1]):
            for b_perm in (b_side_faces, b_side_faces[::-1]):
                a_map = {"U": "U", expected_a[0]: a_perm[0], expected_a[1]: a_perm[1]}
                b_map = {"D": "D", expected_b[0]: b_perm[0], expected_b[1]: b_perm[1]}
                score = 0
                for label, true_face in a_map.items():
                    score += _multiset_overlap(label, true_face, sampled_by_side["A"], gt_state)
                for label, true_face in b_map.items():
                    score += _multiset_overlap(label, true_face, sampled_by_side["B"], gt_state)
                if score > best_score:
                    best_second = best_score
                    best_score = score
                    best_mapping = {"A": a_map, "B": b_map}
                elif score > best_second:
                    best_second = score

    status = "ok"
    if best_score - best_second < 4:  # margin: at least 4 stickers of separation
        status = "ambiguous"
    return best_mapping, best_score, status


def identify_faces_from_multisets(
    prepared: Dict,
    gt_state: str,
    inset: float,
) -> Tuple[Dict[str, str], int]:
    """Map labeler face name → true face identity using ALL 9 stickers'
    classified-color multiset, not just the center. Robust to:
      - Red/orange center misclassification (Set 27/46 image A) — 8 other
        stickers vote.
      - Yaw rotations (Set 23 image A is +180° yaw) — any rotated face
        triple {U/F/R}, {U/L/F}, {U/L/B}, {U/R/B} naturally falls out as
        the best multiset match.
      - Image B's flip-induced L/B swap — handled by the same enumeration.

    Algorithm: anchor (U for image A, D for image B) is fixed to its label.
    For the 2 non-anchor labels, enumerate all 5×4=20 ordered assignments
    of {true_face_A, true_face_B} ∈ permutations of (URFDLB \\ anchor).
    For each, score = sum over labels of |sampled_multiset ∩ gt_multiset|.
    Pick the assignment with the highest total score.

    Returns (label_to_true, best_score)."""
    arr = prepared["arr"]
    quads = prepared["quads"]
    expected = prepared["expected"]
    anchor_label = "U" if "U" in expected else "D"
    non_anchor_labels = [l for l in expected if l != anchor_label and l in quads]

    sampled_multisets: Dict[str, Counter] = {}
    for label in expected:
        if label not in quads:
            continue
        canonical_quad = canonical_corner_order([tuple(p) for p in quads[label]])
        centers = sticker_centers(canonical_quad, inset=inset)
        rgbs = [sample_rgb(arr, x, y) for (x, y) in centers]
        classified = [classify_rgb(rgb).color for rgb in rgbs]
        sampled_multisets[label] = Counter(classified)

    def score_mapping(mapping: Dict[str, str]) -> int:
        s = 0
        for label, true_face in mapping.items():
            if label not in sampled_multisets:
                continue
            gt_face_colors = face_colors_from_state(gt_state, true_face)
            gt_mset = Counter(gt_face_colors)
            samp = sampled_multisets[label]
            s += sum(min(samp.get(c, 0), gt_mset.get(c, 0)) for c in COLOR_ORDER)
        return s

    # Physical constraints on visible-face triples:
    #   - Image A (anchor U): D is hidden; the 2 side faces must come from
    #     {R, F, L, B} and must be adjacent on the cube.
    #   - Image B (anchor D): same logic — U is hidden.
    # Opposite face pairs (cannot be visible together): {L,R} and {F,B}.
    opposite = {"U": "D", "D": "U", "L": "R", "R": "L", "F": "B", "B": "F"}
    anchor_opposite = opposite[anchor_label]
    candidates = [f for f in "URFDLB" if f not in (anchor_label, anchor_opposite)]  # 4 candidates
    best_score = -1
    best_mapping: Dict[str, str] = {anchor_label: anchor_label}
    for true_a, true_b in product(candidates, repeat=2):
        if true_a == true_b:
            continue
        if opposite[true_a] == true_b:
            continue  # two visible side faces must be adjacent, not opposite
        mapping = {anchor_label: anchor_label}
        if len(non_anchor_labels) >= 1:
            mapping[non_anchor_labels[0]] = true_a
        if len(non_anchor_labels) >= 2:
            mapping[non_anchor_labels[1]] = true_b
        s = score_mapping(mapping)
        if s > best_score:
            best_score = s
            best_mapping = mapping
    return best_mapping, best_score


def collect_anchor_rgbs(
    prepared: Dict,
    inset: float,
    label_to_true: Dict[str, str],
) -> Tuple[Dict[str, List[Tuple[int, int, int]]], List[Tuple[int, int, int]]]:
    """Return (anchors-by-color, all-27-sampled-rgbs) for one side. Uses
    `label_to_true` to look up GT color by the *true* face identity.

    For the anchor face (U on image A, D on image B), we SKIP the center
    sticker when building anchor RGBs — because the U center has a logo
    that produces a non-white sample. The other 8 stickers from that face
    still feed the iterative palette refinement via the `all_rgbs` pool."""
    anchors: Dict[str, List[Tuple[int, int, int]]] = {}
    all_rgbs: List[Tuple[int, int, int]] = []
    arr = prepared["arr"]
    quads = prepared["quads"]
    side = "A" if "U" in prepared["expected"] else "B"
    anchor_face = "U" if side == "A" else "D"
    for label in prepared["expected"]:
        if label not in quads:
            continue
        canonical_quad = canonical_corner_order([tuple(p) for p in quads[label]])
        centers = sticker_centers(canonical_quad, inset=inset)
        rgbs = [sample_rgb(arr, x, y) for (x, y) in centers]
        all_rgbs.extend(rgbs)
        true_face = label_to_true.get(label, label)
        # Skip U center for anchor (logo); use the rest of the face as
        # uncalibrated training samples only.
        if true_face == "U" and label == "U":
            continue
        center_color = FACE_TO_COLOR[true_face]
        anchors.setdefault(center_color, []).append(rgbs[4])
    return anchors, all_rgbs


def finish_side(
    prepared: Dict,
    gt_state: str,
    inset: float,
    palette: Dict[str, Tuple[int, int, int]],
    label_to_true: Dict[str, str],
) -> Dict:
    """Per-face sampling using TRUE face identity (from center
    classification) for GT lookup, not the labeler's possibly-swapped
    label name."""
    face_samples = []
    for label in prepared["expected"]:
        if label not in prepared["quads"]:
            continue
        true_face = label_to_true.get(label, label)
        gt_colors_face = face_colors_from_state(gt_state, true_face)
        fs = sample_face(
            prepared["arr"], prepared["quads"][label],
            true_face, gt_colors_face, inset=inset, palette=palette,
        )
        fs["labelName"] = label  # so the report can show the swap
        face_samples.append(fs)
    return {"image": prepared["image"], "face_samples": face_samples, "palette": palette}


PER_FACE_AMBIGUOUS_THRESHOLD = 5.0  # match_count below this = mark face as uncertain


def render_overlay(
    image: Image.Image,
    face_samples: List[Dict],
    out_path: Path,
    pair_status: str = "ok",
) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    # Light dim of the original to make dots pop
    dim = Image.new("RGBA", canvas.size, (0, 0, 0, 35))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), dim).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    try:
        warn_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    except OSError:
        warn_font = ImageFont.load_default()

    # If the JOINT face identification was ambiguous, slap a red banner across
    # the top so the viewer can't miss that this overlay is suspect.
    if pair_status == "ambiguous":
        banner_h = 44
        draw.rectangle((0, 0, canvas.width, banner_h), fill=(180, 20, 20))
        text = "⚠ AMBIGUOUS PAIR-LEVEL FACE ID — dataset extractor would SKIP this pair"
        draw.text((10, 6), text, font=warn_font, fill=(255, 255, 255))

    # Draw the face quad outlines first (thin grey for ok; red+thick when
    # the per-face orientation match is below the uncertainty threshold)
    for fs in face_samples:
        q = fs["canonical_quad"]
        match_count = fs["orientation"].get("match_count", 9.0)
        uncertain = match_count < PER_FACE_AMBIGUOUS_THRESHOLD
        outline_color = (220, 30, 30) if uncertain else (200, 200, 200)
        outline_width = 5 if uncertain else 2
        for i in range(4):
            a = q[i]
            b = q[(i + 1) % 4]
            draw.line([a, b], fill=outline_color, width=outline_width)
        # If uncertain, paint a big "?" near the face centroid as a second cue
        if uncertain:
            cx = sum(p[0] for p in q) / 4.0
            cy = sum(p[1] for p in q) / 4.0
            label = f"?  (face match {match_count:.1f}/9)"
            draw.text((cx - 60, cy - 14), label, font=warn_font, fill=(220, 30, 30))

    # Draw the 9 dots per face
    for fs in face_samples:
        for s in fs["stickers"]:
            x, y = s["xy"]
            col = DOT_COLORS.get(s["gtColor"], (255, 0, 255))
            r = 11
            # Black outline ring (visibility on light stickers)
            draw.ellipse((x - r - 2, y - r - 2, x + r + 2, y + r + 2), outline=(0, 0, 0), width=2)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=col)
            # Small white outline inside (for contrast)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255), width=1)
    canvas.save(out_path, "PNG", optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("set_id", help="setId (e.g. 17)")
    ap.add_argument("--inset", type=float, default=0.20,
                    help="Sticker-center inset from face edge (0.167 = no bezel, 0.20 default)")
    ap.add_argument("--suffix", default="",
                    help="Optional suffix for output filenames (e.g. '-inset20')")
    args = ap.parse_args()

    set_id = args.set_id
    tasks = load_corpus_tasks(CORPUS_MANIFEST)
    task = next((t for t in tasks if t.set_id == set_id), None)
    if task is None:
        extra = discover_additional_tasks({t.set_id for t in tasks})
        task = next((t for t in extra if t.set_id == set_id), None)
    if task is None:
        print(f"set {set_id} not found", file=sys.stderr)
        return 2

    gt_state = parse_ground_truth(task.ground_truth)
    print(f"set {set_id}: ground truth = {gt_state}", file=sys.stderr)

    # Step 1: prepare both sides (image + quads + arr)
    prepared_sides: Dict[str, Dict] = {}
    for side, image_path in (("A", task.image_a), ("B", task.image_b)):
        hull_path = latest_hull_label(set_id, side)
        if hull_path is None:
            print(f"  side {side}: no hull label, skipping", file=sys.stderr)
            continue
        hull_doc = load_hull_label(hull_path)
        prepared_sides[side] = process_side(image_path, hull_doc, gt_state, side, inset=args.inset)

    # Step 2: JOINT face identification across A+B (enforces 6-distinct-
    # faces pair-level invariant — prevents Set 28's L-duplicate /
    # R-missing bug that single-sided matching produced).
    label_maps, joint_score, joint_status = identify_faces_jointly(
        prepared_sides, gt_state, args.inset
    )
    max_score = 9 * sum(len(m) for m in label_maps.values())
    for side, mapping in label_maps.items():
        for label, true in mapping.items():
            tag = "" if label == true else f"  ⚠ relabeled: {label} → {true}"
            print(f"  side {side} label {label} → true face {true}{tag}", file=sys.stderr)
    print(f"  joint multiset score: {joint_score}/{max_score}  ({joint_status})", file=sys.stderr)
    if joint_status == "ambiguous":
        print("", file=sys.stderr)
        print("  ⚠⚠⚠ AMBIGUOUS PAIR-LEVEL FACE ID ⚠⚠⚠", file=sys.stderr)
        print("  Top two yaw configurations are within 4 stickers of each other.", file=sys.stderr)
        print("  The dataset extractor (extract_clean_dataset.py) would SKIP this pair.", file=sys.stderr)
        print("  Rendered overlay may show ground-truth colors at WRONG sticker positions.", file=sys.stderr)
        print("  ⚠⚠⚠", file=sys.stderr)
        print("", file=sys.stderr)

    # Step 3: collect 5 face-center anchors (skip U because of logo)
    combined_anchors: Dict[str, List[Tuple[int, int, int]]] = {}
    combined_samples: List[Tuple[int, int, int]] = []
    for side, prepared in prepared_sides.items():
        anchors, all_rgbs = collect_anchor_rgbs(prepared, args.inset, label_maps[side])
        for color, rgbs in anchors.items():
            combined_anchors.setdefault(color, []).extend(rgbs)
        combined_samples.extend(all_rgbs)

    print(f"  anchors ({len(combined_anchors)} colors): "
          + ", ".join(f"{c}={tuple(rgbs[0])}" for c, rgbs in combined_anchors.items()),
          file=sys.stderr)

    palette = build_adaptive_palette(combined_samples, anchors=combined_anchors)
    print(f"  calibrated palette: " + ", ".join(f"{c}={tuple(rgb)}" for c, rgb in palette.items()),
          file=sys.stderr)

    # Step 4: per-side sampling with the calibrated palette + true face ids
    for side, prepared in prepared_sides.items():
        result = finish_side(prepared, gt_state, args.inset, palette, label_maps[side])
        out_path = OUT_DIR / f"cal-geom-set-{set_id}-{side}{args.suffix}.png"
        render_overlay(result["image"], result["face_samples"], out_path, pair_status=joint_status)
        print(f"  side {side}: wrote {out_path}", file=sys.stderr)
        for fs in result["face_samples"]:
            ori = fs["orientation"]
            mismatches = sum(1 for s in fs["stickers"] if s["classifier"] != s["gtColor"])
            warn = "  ⚠ UNCERTAIN FACE" if ori["match_count"] < PER_FACE_AMBIGUOUS_THRESHOLD else ""
            print(
                f"    face {fs['face']}: orientation (mirror={ori['mirror']}, "
                f"rot_ccw={ori['rotation_quarters_ccw']}, "
                f"orientation_match={ori['match_count']:.2f}/9), "
                f"classifier mismatches per-position vs gt = {mismatches}/9{warn}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
