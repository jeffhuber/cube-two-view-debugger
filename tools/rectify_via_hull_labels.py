#!/usr/bin/env python3
"""End-to-end rectification via hull-position corner labels + parallelogram
vertex derivation. Diagnostic-only candidate replacement for the production
`tools/global_cube_model.py` pipeline.

Origin: a first-principles question from the user (2026-05-24): given that
rembg + convex hull produces 6 silhouette extrema with stable physical
meaning (TOP, upper-right, lower-right, BOTTOM, lower-left, upper-left),
and given the cube's per-side corner-labeling convention is fixed in
`tools/corner_conventions.py` via `FACE_DEFS_BY_SIDE`, why does production
run a 720-perm Procrustes search + PnP refinement + chirality detector +
vertex ensemble + image-based vertex refinement when the entire mapping
is deterministic?

This tool answers that question with a 7-step implementation that produces
essentially-oracle-quality rectifications on all 12 approved full-corner
ground-truth rows (per the empirical comparison in
`tools/RECTIFY_VIA_HULL_LABELS_REPORT.md`).

## Pipeline

1. rembg silhouette (production rembg path: `remove(image, session=sess)`
   → alpha channel → mask > 128)
2. `detect_hexagon_anchors(mask)` → 6 hull-extreme corners
3. Label corners by silhouette position (TOP, upper-right, lower-right,
   BOTTOM, lower-left, upper-left) via `_label_corners_by_position`
4. Map silhouette positions → canonical corner numbers via the per-side
   convention `SILHOUETTE_TO_CORNER` (derived from `FACE_DEFS_BY_SIDE`)
5. Derive vertex by parallelogram completion: for each of the 3 face
   quads, `vertex = NEAR_A + NEAR_B - FAR_AB`. Take the mean of 3
   estimates. Exact under iso projection; approximate under perspective.
6. For each face slot (upper/right/front), construct the face_quad
   `(vertex, NEAR_A, FAR_AB, NEAR_B)` per `FACE_DEFS_BY_SIDE`.
7. Rectify each face_quad via the existing `rectify_face` helper.

## What this eliminates

- The 720-perm Procrustes search in `fit_cube_template_to_anchors`
- PnP refinement
- Mean-of-3 vertex ensemble
- `_resolve_near_far_phase` chirality detector + 60° flip correction
- Image-based vertex refinement
- The bezel-detection dependency (`detect_interior_bezel_lines`)

## When this may fail

The approach assumes the cube is held roughly upright (white-up for
side A; white-down for side B) so the silhouette-position labeling is
stable. Strong tilt (>~30° from vertical) could shuffle which hull
extremum lands at TOP vs upper-right. The per-side mapping table needs
extension for sides other than A/B. Vertex derivation is exact under
iso projection; for cubes with strong perspective the parallelogram
closure constraint is only approximate and the derived vertex may drift
20-70 px from the true trihedral junction (still better than production
on the 12-row corpus, where bezel-detected vertex error ranges 43-241
px).

## CLI

  python tools/rectify_via_hull_labels.py

Defaults to the same canonical truth + manifest fixtures the other
diagnostics use; outputs trace to
`tests/fixtures/rectify_via_hull_labels_trace.json` and a gallery to
`/tmp/rectify_via_hull_labels/`.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import datetime as _dt

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import CLASSIFIER_CANONICAL, classify_rgb_with_mode  # noqa: E402
from tools.corner_conventions import FACE_DEFS_BY_SIDE  # noqa: E402
from tools.diagnose_pipeline_phase_parity import _processing_image  # noqa: E402
from tools.global_cube_model import detect_hexagon_anchors  # noqa: E402
from tools.rectify_faces import (  # noqa: E402
    extract_stickers_from_rectified, rectify_face,
)


Point = Tuple[float, float]

DEFAULT_TRUTH = REPO_ROOT / "tests" / "fixtures" / "full_corner_ground_truth.json"
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_TRACE = REPO_ROOT / "tests" / "fixtures" / "rectify_via_hull_labels_trace.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "RECTIFY_VIA_HULL_LABELS_REPORT.md"
DEFAULT_GALLERY_DIR = Path("/tmp/rectify_via_hull_labels")
DEFAULT_MAX_IMAGE_DIM = 1600
DEFAULT_FACE_SIZE_PX = 300


# Per-side silhouette-position → corner-number mapping. Derived from
# `tools/corner_conventions.py:FACE_DEFS_BY_SIDE` plus the canonical iso
# geometry of each side's view:
#
#   Side A (white-up view):
#     NEAR set = {1, 3, 5} per ONE_EDGE_CORNERS_BY_SIDE["A"]
#     The 3 NEAR cube edges from the front vertex go to upper-right,
#     BOTTOM, upper-left silhouette positions.
#     The 3 FAR corners (outer corners of the 3 visible faces) are at
#     TOP, lower-right, lower-left.
#     Upper face = (vertex, c1=UR, c0=TOP, c5=UL) — CCW around upper.
#
#   Side B (white-down view):
#     NEAR set = {0, 2, 4} per ONE_EDGE_CORNERS_BY_SIDE["B"]
#     The 3 NEAR cube edges from side B's vertex (which is the
#     OPPOSITE corner of the cube from side A's) go to the SAME 3
#     silhouette positions (upper-right, BOTTOM, upper-left) — the
#     cube silhouette is invariant under 180° body-diagonal rotation.
#     Different LABELS though: c0=BOTTOM, c2=upper-right, c4=upper-left.
#     Upper face = (vertex, c2=UR, c3=TOP, c4=UL) — CCW around upper.
SILHOUETTE_TO_CORNER: Dict[str, Dict[str, int]] = {
    "A": {
        "top": 0,
        "upper_right": 1,
        "lower_right": 2,
        "bottom": 3,
        "lower_left": 4,
        "upper_left": 5,
    },
    "B": {
        "top": 3,
        "upper_right": 2,
        "lower_right": 1,
        "bottom": 0,
        "lower_left": 5,
        "upper_left": 4,
    },
}


@dataclass
class RectifiedFit:
    """Output of `rectify_via_hull_labels` for one image+side."""
    side: str
    corners_by_num: Dict[int, Point]
    vertex: Point
    face_quads: Dict[str, List[Point]]
    rectified_faces: Dict[str, Image.Image]


def _label_corners_by_position(
    hexagon: Sequence[Point], side: str,
) -> Dict[int, Point]:
    """Sort 6 hexagon corners into silhouette positions, then map to
    canonical corner numbers per `SILHOUETTE_TO_CORNER[side]`.

    Position assignment:
      - TOP    = corner with smallest y (image y points DOWN)
      - BOTTOM = corner with largest y
      - The remaining 4 corners sort into left vs right by x; within
        each side, smaller y = upper, larger y = lower.

    Robust to small tilt of the cube. Assumes the cube isn't tilted
    >~30° from vertical (which would shuffle which hull extremum
    lands at each named position).
    """
    if len(hexagon) != 6:
        raise ValueError(f"need exactly 6 hexagon corners; got {len(hexagon)}")
    by_y = sorted(hexagon, key=lambda p: p[1])
    top = by_y[0]
    bottom = by_y[-1]
    middle_4 = sorted(by_y[1:-1], key=lambda p: p[0])
    upper_left = min(middle_4[:2], key=lambda p: p[1])
    lower_left = max(middle_4[:2], key=lambda p: p[1])
    upper_right = min(middle_4[2:], key=lambda p: p[1])
    lower_right = max(middle_4[2:], key=lambda p: p[1])
    positions = {
        "top": top,
        "upper_right": upper_right,
        "lower_right": lower_right,
        "bottom": bottom,
        "lower_left": lower_left,
        "upper_left": upper_left,
    }
    mapping = SILHOUETTE_TO_CORNER[side]
    return {mapping[pos_name]: point for pos_name, point in positions.items()}


def _derive_vertex_from_corners(
    corners_by_num: Dict[int, Point], side: str,
) -> Tuple[Point, List[Point]]:
    """Vertex via parallelogram completion: each visible face is a
    parallelogram in iso projection, so for face quad
    `(vertex, NEAR_A, FAR_AB, NEAR_B)`:

        FAR_AB = NEAR_A + NEAR_B - vertex
    →   vertex = NEAR_A + NEAR_B - FAR_AB

    Each of the 3 visible faces gives one vertex estimate. Take the
    mean. Returns (mean_vertex, [estimate_upper, estimate_right,
    estimate_front]).

    Exact under iso projection. Approximate under perspective — the
    3 estimates spread by ~10-30 px on real iPhone shots; the mean is
    typically within 20-70 px of the true vertex on the 12-row corpus.
    """
    estimates: List[Point] = []
    for _slot, names in FACE_DEFS_BY_SIDE[side].items():
        # names = ("vertex", "corner_<near_a>", "corner_<far>", "corner_<near_b>")
        _, n_a_name, far_name, n_b_name = names
        n_a = corners_by_num[int(n_a_name.split("_")[1])]
        n_b = corners_by_num[int(n_b_name.split("_")[1])]
        far = corners_by_num[int(far_name.split("_")[1])]
        estimates.append((n_a[0] + n_b[0] - far[0], n_a[1] + n_b[1] - far[1]))
    vx = sum(e[0] for e in estimates) / 3.0
    vy = sum(e[1] for e in estimates) / 3.0
    return (vx, vy), estimates


def rectify_via_hull_labels(
    image: Image.Image,
    mask: np.ndarray,
    side: str,
    *,
    face_size_px: int = DEFAULT_FACE_SIZE_PX,
) -> Optional[RectifiedFit]:
    """End-to-end: hull → labeled corners → vertex → face_quads →
    rectified faces. Returns None on failure (e.g. <6 hexagon corners).

    `image` must be the processing-resolution PIL image; `mask` the
    rembg silhouette mask (boolean array, same H×W as image) — caller
    is responsible for choosing the rembg path (production uses
    `remove(image, session=sess)` → split RGBA → alpha > 128).
    """
    hexagon = detect_hexagon_anchors(mask)
    if len(hexagon) != 6:
        return None
    if side not in SILHOUETTE_TO_CORNER:
        raise ValueError(
            f"no per-side mapping defined for side {side!r}; "
            f"add to SILHOUETTE_TO_CORNER"
        )
    corners_by_num = _label_corners_by_position(hexagon, side)
    vertex, _estimates = _derive_vertex_from_corners(corners_by_num, side)
    face_quads: Dict[str, List[Point]] = {}
    rectified_faces: Dict[str, Image.Image] = {}
    for slot, names in FACE_DEFS_BY_SIDE[side].items():
        quad: List[Point] = []
        for name in names:
            if name == "vertex":
                quad.append(vertex)
            else:
                quad.append(corners_by_num[int(name.split("_")[1])])
        face_quads[slot] = quad
        rectified_faces[slot] = rectify_face(image, quad, output_size=face_size_px)
    return RectifiedFit(
        side=side,
        corners_by_num=corners_by_num,
        vertex=vertex,
        face_quads=face_quads,
        rectified_faces=rectified_faces,
    )


# ---------------- CLI / corpus runner ----------------


def _score_rectified_faces(faces: Dict[str, Image.Image]) -> Dict[str, Any]:
    """Sum of CIELAB distance from each sampled sticker to its nearest
    canonical color, summed over 9 stickers × 3 faces. Lower = better
    cluster (each face's stickers cleanly match canonical cube colors).
    """
    total = 0.0
    per_face: Dict[str, float] = {}
    for slot, face_img in faces.items():
        face_total = 0.0
        for row in extract_stickers_from_rectified(face_img):
            for s in row:
                face_total += classify_rgb_with_mode(s.rgb, CLASSIFIER_CANONICAL).distance
        per_face[slot] = round(face_total, 2)
        total += face_total
    return {
        "total_distance": round(total, 2),
        "per_face": per_face,
        "mean_sticker_distance": round(total / 27.0, 2),
    }


def evaluate_row(
    sess: Any,
    key: str,
    image_path: Path,
    truth_row: Dict[str, Any],
    max_image_dim: int,
    gallery_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from rembg import remove  # noqa: E402
    side = key.rsplit("_", 1)[-1]
    rec: Dict[str, Any] = {"key": key, "side": side, "status": "pending"}
    try:
        image, scale = _processing_image(image_path, max_image_dim)
        rgba = remove(image, session=sess)
        alpha = np.array(rgba.split()[-1], dtype=np.uint8)
        mask = alpha > 128
        fit = rectify_via_hull_labels(image, mask, side)
        if fit is None:
            rec.update({"status": "no_hexagon",
                        "error": "detect_hexagon_anchors returned <6"})
            return rec
        # Compare against ground truth
        gt_vertex = (truth_row["vertex"][0] * scale, truth_row["vertex"][1] * scale)
        vertex_err = math.hypot(
            fit.vertex[0] - gt_vertex[0], fit.vertex[1] - gt_vertex[1],
        )
        per_corner_err = {}
        for i in range(6):
            gt = (truth_row[f"corner_{i}"][0] * scale, truth_row[f"corner_{i}"][1] * scale)
            d = math.hypot(fit.corners_by_num[i][0] - gt[0],
                           fit.corners_by_num[i][1] - gt[1])
            per_corner_err[f"corner_{i}"] = round(d, 1)
        mean_corner_err = sum(per_corner_err.values()) / 6.0
        # Score our rectification
        our_score = _score_rectified_faces(fit.rectified_faces)
        # Oracle reference rectification (using GT vertex + GT corners)
        oracle_faces: Dict[str, Image.Image] = {}
        for slot, names in FACE_DEFS_BY_SIDE[side].items():
            quad = []
            for name in names:
                if name == "vertex":
                    quad.append(gt_vertex)
                else:
                    quad.append((truth_row[name][0] * scale, truth_row[name][1] * scale))
            oracle_faces[slot] = rectify_face(
                image, quad, output_size=DEFAULT_FACE_SIZE_PX,
            )
        oracle_score = _score_rectified_faces(oracle_faces)
        if gallery_dir is not None:
            _save_gallery_panel(
                key, side, image, fit, oracle_faces,
                our_score, oracle_score, mean_corner_err, vertex_err,
                gallery_dir,
            )
        rec.update({
            "status": "rectified",
            "labeling_mean_corner_err_px": round(mean_corner_err, 1),
            "per_corner_err_px": per_corner_err,
            "derived_vertex_error_processing_px": round(vertex_err, 1),
            "rectified_score": our_score,
            "oracle_score": oracle_score,
            "score_delta_vs_oracle": round(
                our_score["total_distance"] - oracle_score["total_distance"], 2,
            ),
        })
    except Exception as exc:  # noqa: BLE001
        rec.update({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    return rec


def _save_gallery_panel(
    key: str, side: str, image: Image.Image,
    fit: RectifiedFit, oracle_faces: Dict[str, Image.Image],
    our_score: Dict[str, Any], oracle_score: Dict[str, Any],
    corner_err: float, vertex_err: float,
    gallery_dir: Path,
) -> None:
    """Per-row panel: source-with-overlay + our 3 faces + oracle 3
    faces. Saved under `{gallery_dir}/by_row/{key}.png`."""
    slot_order = ("upper", "right", "front")
    overlay = image.convert("RGB").copy()
    d = ImageDraw.Draw(overlay, "RGBA")
    for quad in fit.face_quads.values():
        pts = [(int(p[0]), int(p[1])) for p in quad] + [(int(quad[0][0]), int(quad[0][1]))]
        d.line(pts, fill=(220, 80, 80, 230), width=3)
    src_thumb = overlay.copy()
    src_thumb.thumbnail((360, 360))
    PANEL = 200
    margin = 16
    total_w = margin + src_thumb.width + margin + 3 * (PANEL + margin)
    total_h = 80 + max(src_thumb.height, 2 * (PANEL + 24))
    panel = Image.new("RGB", (total_w, total_h), (28, 28, 32))
    pd = ImageDraw.Draw(panel)
    pd.text(
        (margin, 8),
        f"{key} (side {side})  corner_err {corner_err:.1f}px  "
        f"vertex_err {vertex_err:.1f}px  "
        f"score {our_score['total_distance']:.0f} (oracle {oracle_score['total_distance']:.0f}, "
        f"Δ{our_score['total_distance']-oracle_score['total_distance']:+.0f})",
        fill=(255, 255, 255),
    )
    panel.paste(src_thumb, (margin, 50))
    face_x0 = margin + src_thumb.width + margin
    pd.text((face_x0, 38), "Hull-labels rectified:", fill=(220, 120, 120))
    pd.text((face_x0, 38 + PANEL + 24), "Oracle rectified (GT):", fill=(120, 220, 120))
    for i, slot in enumerate(slot_order):
        x = face_x0 + i * (PANEL + margin)
        r = fit.rectified_faces[slot].resize((PANEL, PANEL), Image.Resampling.BICUBIC)
        panel.paste(r, (x, 60))
        pd.text((x, 60 + PANEL + 4), slot, fill=(200, 200, 200))
        or_ = oracle_faces[slot].resize((PANEL, PANEL), Image.Resampling.BICUBIC)
        panel.paste(or_, (x, 60 + PANEL + 24))
        pd.text((x, 60 + 2 * PANEL + 28), slot, fill=(200, 200, 200))
    (gallery_dir / "by_row").mkdir(parents=True, exist_ok=True)
    panel.save(gallery_dir / "by_row" / f"{key}.png")


def _git_head_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--gallery-dir", type=Path, default=DEFAULT_GALLERY_DIR)
    ap.add_argument("--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM)
    args = ap.parse_args(list(argv) if argv is not None else None)

    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    set_index = {str(p["setId"]): p for p in manifest.get("pairs", [])}
    args.gallery_dir.mkdir(parents=True, exist_ok=True)
    from rembg import new_session  # noqa: E402
    # Explicit "u2net" — matches `tools/measure_axis_correctness.py` and
    # production (`rubik_recognizer/image_pipeline.py`). The default is
    # already u2net, but pinning explicitly avoids the silent drift bug
    # if rembg ever changes its default model. (Codex polish on PR #279.)
    sess = new_session("u2net")
    records: List[Dict[str, Any]] = []
    for key in sorted(truth):
        row = truth[key]
        if not row.get("approved"):
            continue
        set_id, _, side = key.rpartition("_")
        pair = set_index.get(set_id)
        if pair is None:
            records.append({"key": key, "status": "skipped",
                            "error": f"set {set_id} not in manifest"})
            continue
        img_path_str = pair.get(f"image{side}Path")
        if not img_path_str:
            records.append({"key": key, "status": "skipped",
                            "error": "no image path"})
            continue
        img_path = Path(img_path_str)
        if not img_path.exists():
            records.append({"key": key, "status": "skipped",
                            "error": f"image not found: {img_path}"})
            continue
        print(
            f"[{len([r for r in records if r.get('status')=='rectified'])+1}] {key}...",
            flush=True,
        )
        rec = evaluate_row(sess, key, img_path, row, args.max_image_dim, args.gallery_dir)
        records.append(rec)
    # Source metadata for the trace
    source = {
        "tool": "tools/rectify_via_hull_labels.py",
        "git_sha": _git_head_sha(),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "truth": str(args.truth.relative_to(REPO_ROOT)) if str(args.truth).startswith(str(REPO_ROOT)) else str(args.truth),
        "manifest": str(args.manifest.relative_to(REPO_ROOT)) if str(args.manifest).startswith(str(REPO_ROOT)) else str(args.manifest),
        "max_image_dim": args.max_image_dim,
        "face_size_px": DEFAULT_FACE_SIZE_PX,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({
            "schema": "rectify_via_hull_labels_v1",
            "source": source,
            "per_row": records,
        }, indent=2),
        encoding="utf-8",
    )
    # Quick summary
    scored = [r for r in records if r.get("status") == "rectified"]
    print(f"\n{len(scored)}/{len(records)} rows rectified")
    if scored:
        deltas = [r["score_delta_vs_oracle"] for r in scored]
        print(f"Score delta vs oracle: min {min(deltas):+.1f}, "
              f"max {max(deltas):+.1f}, median {statistics.median(deltas):+.1f}")
        verr = [r["derived_vertex_error_processing_px"] for r in scored]
        print(f"Derived vertex error: min {min(verr):.1f}, "
              f"max {max(verr):.1f}, median {statistics.median(verr):.1f}px")
        cerr = [r["labeling_mean_corner_err_px"] for r in scored]
        print(f"Labeling mean corner error: min {min(cerr):.1f}, "
              f"max {max(cerr):.1f}, median {statistics.median(cerr):.1f}px")
    print(f"Trace: {args.out_json}")
    print(f"Gallery: {args.gallery_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
