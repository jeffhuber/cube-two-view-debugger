#!/usr/bin/env python3
"""Corpus-wide diagnostic: affine vs projective vertex on the 70-row hull-labels corpus.

Origin: Codex's design review of the projective-vertex follow-up
(2026-05-25). The core question this report answers:

  Can projective vertex correction (via vanishing-point construction)
  recover perspective-tilted rows like 37_B without regressing clean
  rows or bad-hull-input rows like 30_A?

Multi-signal output per row (per Codex feedback item 5: don't let
sticker score alone decide "better"):

  - affine_vertex_err_px           : current parallelogram vs GT
  - projective_vertex_err_px       : VP-construction vs GT
  - affine_axis_misfit_deg         : axes computed from affine vertex
  - projective_axis_misfit_deg     : axes computed from projective vertex
  - affine_sticker_score_total     : 27-sticker classifier-distance sum
  - projective_sticker_score_total : same, with projective face_quads
  - projective_residual_px         : 3-line LSQ residual (geometry self-consistency)
  - projective_residual_norm       : residual / hexagon diameter (resolution-independent)
  - projective_degeneracy          : finite_projective / near_affine / degenerate
  - winner_by_sticker              : which vertex gives lower sticker score
  - winner_by_gt_vertex            : which vertex is closer to GT (diagnostic-only)

30_A is reported but flagged as a known bad-hull-input case (per
Codex feedback item 6) — not evidence against the projective approach.

37_B is the named case study (per Codex feedback item 7) — visual
panel emitted to ``tools/projective_vertex_case_37B.png``.

Decision criterion the report supports:
  ``projective is the better choice for a row IFF``
    - degeneracy != "degenerate", AND
    - projective improves sticker score by a meaningful margin, AND
    - projective residual is sane (residual_norm < some threshold)

This is diagnostic-only — does NOT modify production behavior in
``tools/rectify_via_hull_labels.py`` or anywhere else.

## CLI

  .venv/bin/python tools/diagnose_projective_vertex.py

Writes:
  - tests/fixtures/projective_vertex_trace.json
  - tools/PROJECTIVE_VERTEX_REPORT.md  (manually written; not by this tool)
  - tools/projective_vertex_case_37B.png  (visual panel)
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
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import (  # noqa: E402
    CLASSIFIER_CANONICAL, classify_rgb_with_mode,
)
from tools.corner_conventions import (  # noqa: E402
    FACE_DEFS_BY_SIDE, FAR_CORNERS_BY_SIDE,
)
from tools.diagnose_pipeline_phase_parity import _processing_image  # noqa: E402
from tools.global_cube_model import detect_hexagon_anchors  # noqa: E402
from tools.measure_axis_correctness import (  # noqa: E402
    _candidate_image_roots, _match_axes_to_ground_truth, _resolve_image_path,
    _scale_point,
)
from tools.projective_vertex import projective_vertex  # noqa: E402
from tools.rectify_faces import extract_stickers_from_rectified, rectify_face  # noqa: E402
from tools.rectify_via_hull_labels import (  # noqa: E402
    _derive_vertex_from_corners, _label_corners_by_position,
)


Point = Tuple[float, float]

DEFAULT_AXIS_TRUTH = REPO_ROOT / "tests" / "fixtures" / "gcm_axis_ground_truth.json"
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_TRACE = REPO_ROOT / "tests" / "fixtures" / "projective_vertex_trace.json"
DEFAULT_PANEL_DIR = REPO_ROOT / "tools"
DEFAULT_MAX_IMAGE_DIM = 1600
DEFAULT_FACE_SIZE_PX = 300

# Decision-quality thresholds (initial defaults — tunable from corpus).
# "Meaningful" sticker score improvement: projective is "winner" only
# if its total score is at least this much LOWER (better) than affine.
STICKER_SCORE_MEANINGFUL_MARGIN = 30.0
# Projective residual sanity gate (matches projective_vertex.DEGENERATE_RESIDUAL_FRACTION
# default; surface here so the report can reference it).
PROJECTIVE_RESIDUAL_SANE_NORM = 0.05

# Rows flagged as bad-hull-input from the PR #282 corpus walkthrough.
# These should not be treated as evidence against the projective approach.
KNOWN_BAD_HULL_ROWS = {"30_A"}


def _git_head_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _score_face(face_img: Image.Image) -> float:
    """Sum classify_rgb distances over the 9 stickers in a rectified face."""
    total = 0.0
    for row in extract_stickers_from_rectified(face_img):
        for s in row:
            total += classify_rgb_with_mode(s.rgb, CLASSIFIER_CANONICAL).distance
    return total


def _build_face_quads(
    vertex: Point, corners_by_num: Dict[int, Point], side: str,
) -> Dict[str, List[Point]]:
    """Build the 3 visible face quads from a vertex + 6 corners.

    Mirrors what ``rectify_via_hull_labels.rectify_via_hull_labels``
    does, but factored out so we can call it with either the affine
    or projective vertex.
    """
    quads: Dict[str, List[Point]] = {}
    for slot, names in FACE_DEFS_BY_SIDE[side].items():
        quad: List[Point] = []
        for name in names:
            if name == "vertex":
                quad.append(vertex)
            else:
                quad.append(corners_by_num[int(name.split("_")[1])])
        quads[slot] = quad
    return quads


def _rectified_score(
    image: Image.Image, vertex: Point,
    corners_by_num: Dict[int, Point], side: str,
    face_size_px: int = DEFAULT_FACE_SIZE_PX,
) -> Tuple[float, Dict[str, float], Dict[str, Image.Image]]:
    """Build face_quads from vertex+corners, rectify, score.
    Returns (total, per_face, rectified_faces)."""
    quads = _build_face_quads(vertex, corners_by_num, side)
    per_face: Dict[str, float] = {}
    rectified: Dict[str, Image.Image] = {}
    total = 0.0
    for slot, quad in quads.items():
        face_img = rectify_face(image, quad, output_size=face_size_px)
        score = _score_face(face_img)
        per_face[slot] = round(score, 2)
        rectified[slot] = face_img
        total += score
    return total, per_face, rectified


def _axis_misfit(
    vertex: Point, corners_by_num: Dict[int, Point], side: str,
    gt_vertex_proc: Point, gt_axes: List[Point],
) -> float:
    """Compute axis_total_misfit_deg given a vertex + corners.
    Predicted axes use FAR_CORNERS_BY_SIDE convention (per #286)."""
    far_names = FAR_CORNERS_BY_SIDE[side]
    predicted_axes: List[Point] = []
    for cn in far_names:
        cp = corners_by_num[int(cn.split("_")[1])]
        predicted_axes.append((cp[0] - vertex[0], cp[1] - vertex[1]))
    match = _match_axes_to_ground_truth(predicted_axes, gt_axes)
    return float(match["total_misfit_deg"])


def _ground_truth_axes_from_axis_truth(
    truth_row: Dict[str, Any], scale: float,
) -> Tuple[Point, List[Point]]:
    """Same as measure_hull_labels_corpus._ground_truth_axes_from_axis_truth.
    Schema is axis_x/y/z; legacy near_x/y/z accepted via shim."""
    vertex_full = (float(truth_row["vertex"][0]), float(truth_row["vertex"][1]))
    vertex_proc = _scale_point(vertex_full, scale)
    axes: List[Point] = []
    for new_name, old_name in (
        ("axis_x", "near_x"), ("axis_y", "near_y"), ("axis_z", "near_z"),
    ):
        raw = truth_row.get(new_name, truth_row.get(old_name))
        p_full = (float(raw[0]), float(raw[1]))
        p_proc = _scale_point(p_full, scale)
        axes.append((p_proc[0] - vertex_proc[0], p_proc[1] - vertex_proc[1]))
    return vertex_proc, axes


def evaluate_row(
    sess: Any, key: str, image_path: Path, truth_row: Dict[str, Any],
    max_image_dim: int,
) -> Dict[str, Any]:
    """Run affine + projective vertex on one row; emit full multi-signal record."""
    from rembg import remove
    side = key.rsplit("_", 1)[-1]
    rec: Dict[str, Any] = {
        "key": key, "side": side,
        "known_bad_hull": key in KNOWN_BAD_HULL_ROWS,
        "status": "pending",
    }
    try:
        image, scale = _processing_image(image_path, max_image_dim)
        rgba = remove(image, session=sess)
        mask = np.array(rgba.split()[-1], dtype=np.uint8) > 128
        hexagon = detect_hexagon_anchors(mask)
        rec["hexagon_corner_count"] = len(hexagon)
        if len(hexagon) != 6:
            rec.update({"status": "mask_failure",
                        "error": f"detect_hexagon_anchors returned {len(hexagon)}"})
            return rec
        cnums = _label_corners_by_position(hexagon, side)

        # Affine (current parallelogram-completion)
        aff_v, aff_estimates = _derive_vertex_from_corners(cnums, side)
        # Projective (vanishing-point construction)
        prj = projective_vertex(cnums, side)

        # GT (diagnostic-only)
        gt_vertex_proc, gt_axes = _ground_truth_axes_from_axis_truth(truth_row, scale)

        aff_err = math.hypot(aff_v[0] - gt_vertex_proc[0], aff_v[1] - gt_vertex_proc[1])
        prj_err = math.hypot(prj.vertex[0] - gt_vertex_proc[0],
                             prj.vertex[1] - gt_vertex_proc[1])

        # Rectify + score with each vertex
        aff_total, aff_per_face, _ = _rectified_score(image, aff_v, cnums, side)
        prj_total, prj_per_face, _ = _rectified_score(image, prj.vertex, cnums, side)

        # Axis misfit with each vertex
        aff_axis_misfit = _axis_misfit(aff_v, cnums, side, gt_vertex_proc, gt_axes)
        prj_axis_misfit = _axis_misfit(prj.vertex, cnums, side, gt_vertex_proc, gt_axes)

        # Affine cloud spread (used in PR #285 gates)
        spread = 0.0
        for i in range(len(aff_estimates)):
            for j in range(i + 1, len(aff_estimates)):
                d = math.hypot(aff_estimates[i][0] - aff_estimates[j][0],
                               aff_estimates[i][1] - aff_estimates[j][1])
                spread = max(spread, d)

        # Winners by each signal
        winner_by_sticker = (
            "projective" if (aff_total - prj_total) > STICKER_SCORE_MEANINGFUL_MARGIN
            else "affine" if (prj_total - aff_total) > STICKER_SCORE_MEANINGFUL_MARGIN
            else "tie"
        )
        winner_by_gt_vertex = (
            "projective" if (aff_err - prj_err) > 1.0
            else "affine" if (prj_err - aff_err) > 1.0
            else "tie"
        )

        rec.update({
            "status": "scored",
            "scale": round(scale, 4),
            # Affine
            "affine_vertex_proc_px": [round(aff_v[0], 1), round(aff_v[1], 1)],
            "affine_vertex_cloud_spread_px": round(spread, 1),
            "affine_vertex_err_px": round(aff_err, 1),
            "affine_axis_misfit_deg": round(aff_axis_misfit, 1),
            "affine_sticker_score_total": round(aff_total, 2),
            "affine_sticker_score_per_face": aff_per_face,
            # Projective
            "projective_vertex_proc_px": [round(prj.vertex[0], 1), round(prj.vertex[1], 1)],
            "projective_vertex_err_px": round(prj_err, 1),
            "projective_axis_misfit_deg": round(prj_axis_misfit, 1),
            "projective_sticker_score_total": round(prj_total, 2),
            "projective_sticker_score_per_face": prj_per_face,
            "projective_residual_px": round(prj.residual_px, 2),
            "projective_residual_norm": round(prj.residual_norm, 4),
            "projective_degeneracy": prj.degeneracy,
            "hexagon_diameter_px": round(prj.hexagon_diameter_px, 1),
            # GT
            "gt_vertex_proc_px": [round(gt_vertex_proc[0], 1), round(gt_vertex_proc[1], 1)],
            # Winners
            "winner_by_sticker": winner_by_sticker,
            "winner_by_gt_vertex": winner_by_gt_vertex,
        })
    except Exception as exc:  # noqa: BLE001
        rec.update({"status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()})
    return rec


def _emit_case_study_37B(
    sess: Any, axis_truth: Dict[str, Any],
    set_idx: Dict[str, Any], image_roots: Sequence[Path], out_path: Path,
) -> None:
    """Per Codex feedback #7: 37_B is the named case study. Render
    a before/after panel showing both vertex placements + both
    rectified-face triples + a per-axis errors box."""
    from rembg import remove
    key = "37_B"
    side = "B"
    set_id = "37"
    pair = set_idx.get(set_id) or {}
    raw = pair.get(f"image{side}Path", f"/tmp/missing_{set_id}_{side}.jpg")
    sha = pair.get(f"image{side}_sha256_expected")
    img_path = _resolve_image_path(raw, set_id, side, image_roots, expected_sha256=sha)
    if img_path is None:
        print(f"case study 37_B: image not resolvable, skipping panel", file=sys.stderr)
        return
    image, scale = _processing_image(img_path, DEFAULT_MAX_IMAGE_DIM)
    rgba = remove(image, session=sess)
    mask = np.array(rgba.split()[-1], dtype=np.uint8) > 128
    hexagon = detect_hexagon_anchors(mask)
    if len(hexagon) != 6:
        print(f"case study 37_B: hexagon detect failed", file=sys.stderr)
        return
    cnums = _label_corners_by_position(hexagon, side)
    aff_v, _ = _derive_vertex_from_corners(cnums, side)
    prj = projective_vertex(cnums, side)
    gt = axis_truth[key]
    gt_vx = (gt["vertex"][0] * scale, gt["vertex"][1] * scale)

    # Crop to mask bbox + margin
    ys, xs = np.where(mask)
    x0 = max(0, xs.min() - 60); y0 = max(0, ys.min() - 60)
    x1 = min(image.width - 1, xs.max() + 60); y1 = min(image.height - 1, ys.max() + 60)
    crop = image.crop((x0, y0, x1, y1)).convert("RGB")

    def F(s):
        for n in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                  "/System/Library/Fonts/Helvetica.ttc"):
            try:
                return ImageFont.truetype(n, s)
            except Exception:  # noqa: BLE001
                continue
        return ImageFont.load_default()

    F_BIG = F(30); F_SM = F(20)

    # Build the overlay panel
    overlay = crop.copy()
    d = ImageDraw.Draw(overlay, "RGBA")

    def m(pt, r, col, label, off=(18, -22)):
        px, py = int(pt[0]) - x0, int(pt[1]) - y0
        d.ellipse((px-r, py-r, px+r, py+r), fill=col, outline=(0,0,0,255), width=3)
        tx, ty = px + off[0], py + off[1]
        for dx in (-2, 2):
            for dy in (-2, 2):
                d.text((tx+dx, ty+dy), label, fill=(0,0,0,255), font=F_BIG)
        d.text((tx, ty), label, fill=(255,255,255,255), font=F_BIG)

    # Affine + projective face quads (lightly drawn)
    for vertex, color in [(aff_v, (255,80,80,200)), (prj.vertex, (90,230,110,200))]:
        for slot, names in FACE_DEFS_BY_SIDE[side].items():
            pts = []
            for name in names:
                if name == "vertex":
                    pts.append((int(vertex[0])-x0, int(vertex[1])-y0))
                else:
                    cn = int(name.split("_")[1])
                    pts.append((int(cnums[cn][0])-x0, int(cnums[cn][1])-y0))
            pts.append(pts[0])
            d.line(pts, fill=color, width=3)

    m(aff_v, 14, (255,80,80,255), "affine")
    m(prj.vertex, 14, (90,230,110,255), "projective")
    m(gt_vx, 14, (250,220,80,255), "GT", off=(18, 18))

    # Compose final panel: overlay + 2x3 rectified faces below
    PANEL = 200
    pad = 12
    title_h = 100
    aff_total, _, aff_rect = _rectified_score(image, aff_v, cnums, side)
    prj_total, _, prj_rect = _rectified_score(image, prj.vertex, cnums, side)
    aff_err = math.hypot(aff_v[0]-gt_vx[0], aff_v[1]-gt_vx[1])
    prj_err = math.hypot(prj.vertex[0]-gt_vx[0], prj.vertex[1]-gt_vx[1])

    overlay.thumbnail((800, 1100))
    rect_w = 3 * PANEL + 4 * pad
    out_w = max(overlay.width, rect_w)
    out_h = title_h + overlay.height + 2 * (PANEL + 60) + 40
    out = Image.new("RGB", (out_w, out_h), (24, 24, 28))
    od = ImageDraw.Draw(out)
    od.text((pad, 8),
            "37_B case study: affine (red) vs projective (green) vertex",
            fill=(255,255,255), font=F_BIG)
    od.text((pad, 40),
            f"affine vertex_err: {aff_err:.1f}px  →  projective: {prj_err:.1f}px  "
            f"(improvement: {aff_err-prj_err:+.1f}px)",
            fill=(220,220,220), font=F_SM)
    od.text((pad, 64),
            f"affine sticker: {aff_total:.0f}  →  projective: {prj_total:.0f}  "
            f"(Δ: {aff_total-prj_total:+.0f}; lower is better)",
            fill=(220,220,220), font=F_SM)
    out.paste(overlay, ((out_w - overlay.width)//2, title_h))

    fy = title_h + overlay.height + 8
    od.text((pad, fy), "Affine rectified faces (current):", fill=(255,160,160), font=F_BIG)
    for i, slot in enumerate(("upper", "right", "front")):
        r = aff_rect[slot].resize((PANEL, PANEL), Image.Resampling.BICUBIC)
        out.paste(r, (pad + i*(PANEL+pad), fy + 36))
        od.text((pad + i*(PANEL+pad), fy + 36 + PANEL + 4), slot, fill=(200,200,200), font=F_SM)

    fy2 = fy + 36 + PANEL + 24
    od.text((pad, fy2), "Projective rectified faces (this PR):", fill=(160,230,160), font=F_BIG)
    for i, slot in enumerate(("upper", "right", "front")):
        r = prj_rect[slot].resize((PANEL, PANEL), Image.Resampling.BICUBIC)
        out.paste(r, (pad + i*(PANEL+pad), fy2 + 36))
        od.text((pad + i*(PANEL+pad), fy2 + 36 + PANEL + 4), slot, fill=(200,200,200), font=F_SM)

    out.save(out_path, optimize=True)
    print(f"case study 37_B panel saved → {out_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis-truth", type=Path, default=DEFAULT_AXIS_TRUTH)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM)
    ap.add_argument("--no-case-study", action="store_true",
                    help="Skip the 37_B visual panel.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    axis_truth = json.loads(args.axis_truth.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    set_index = {str(p["setId"]): p for p in manifest.get("pairs", [])}
    image_roots = _candidate_image_roots(manifest)

    # Lazy rembg init (matches measure_hull_labels_corpus pattern)
    sess: Any = None

    def _get_sess() -> Any:
        nonlocal sess
        if sess is None:
            from rembg import new_session
            sess = new_session("u2net")
        return sess

    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for key in sorted(axis_truth):
        row = axis_truth[key]
        if not row.get("approved"):
            continue
        set_id, _, side = key.rpartition("_")
        pair = set_index.get(set_id)
        if pair is None:
            raw_path = f"/tmp/missing_{set_id}_{side}.jpg"
            expected_sha = None
        else:
            raw_path = pair.get(f"image{side}Path") or ""
            expected_sha = pair.get(f"image{side}_sha256_expected")
        if not raw_path:
            skipped.append({"key": key,
                            "reason": f"no image path in manifest for set {set_id} side {side}"})
            continue
        image_path = _resolve_image_path(
            raw_path, set_id, side, image_roots, expected_sha256=expected_sha,
        )
        if image_path is None:
            skipped.append({"key": key,
                            "reason": f"image not found for set {set_id} side {side}"})
            continue
        n_done = sum(1 for r in records if r.get("status") == "scored")
        print(f"[{n_done + 1}] {key} ({image_path.name})...", flush=True)
        rec = evaluate_row(_get_sess(), key, image_path, row, args.max_image_dim)
        records.append(rec)

    # Aggregate
    scored = [r for r in records if r.get("status") == "scored"]
    by_winner_sticker = {"projective": 0, "affine": 0, "tie": 0}
    by_winner_gt = {"projective": 0, "affine": 0, "tie": 0}
    by_degeneracy = {"finite_projective": 0, "near_affine": 0, "degenerate": 0}
    for r in scored:
        by_winner_sticker[r["winner_by_sticker"]] += 1
        by_winner_gt[r["winner_by_gt_vertex"]] += 1
        by_degeneracy[r["projective_degeneracy"]] += 1

    source = {
        "tool": "tools/diagnose_projective_vertex.py",
        "git_sha": _git_head_sha(),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "axis_truth": str(args.axis_truth.relative_to(REPO_ROOT))
        if str(args.axis_truth).startswith(str(REPO_ROOT)) else str(args.axis_truth),
        "manifest": str(args.manifest.relative_to(REPO_ROOT))
        if str(args.manifest).startswith(str(REPO_ROOT)) else str(args.manifest),
        "max_image_dim": args.max_image_dim,
        "sticker_meaningful_margin": STICKER_SCORE_MEANINGFUL_MARGIN,
        "projective_residual_sane_norm": PROJECTIVE_RESIDUAL_SANE_NORM,
        "known_bad_hull_rows": sorted(KNOWN_BAD_HULL_ROWS),
    }
    out = {
        "schema": "projective_vertex_v1",
        "source": source,
        "summary": {
            "total_rows_attempted": len(records),
            "scored": len(scored),
            "skipped": len(skipped),
            "by_winner_sticker": by_winner_sticker,
            "by_winner_gt_vertex": by_winner_gt,
            "by_projective_degeneracy": by_degeneracy,
        },
        "skipped": skipped,
        "per_row": records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Emit 37_B case study panel
    if not args.no_case_study and "37_B" in axis_truth:
        panel_path = DEFAULT_PANEL_DIR / "projective_vertex_case_37B.png"
        try:
            _emit_case_study_37B(_get_sess(), axis_truth, set_index, image_roots, panel_path)
        except Exception as exc:  # noqa: BLE001
            print(f"case study 37_B FAILED: {exc}", file=sys.stderr)

    # Quick summary
    print()
    print(f"Scored: {len(scored)} / {len(records)} rows  (skipped: {len(skipped)})")
    print(f"\nWinners by sticker score (margin {STICKER_SCORE_MEANINGFUL_MARGIN}):")
    for k, v in by_winner_sticker.items():
        print(f"  {k:>11}: {v:>3}")
    print(f"\nWinners by GT-vertex error (diagnostic-only):")
    for k, v in by_winner_gt.items():
        print(f"  {k:>11}: {v:>3}")
    print(f"\nProjective degeneracy distribution:")
    for k, v in by_degeneracy.items():
        print(f"  {k:>20}: {v:>3}")
    if scored:
        aff = [r["affine_vertex_err_px"] for r in scored]
        prj = [r["projective_vertex_err_px"] for r in scored]
        print(f"\nVertex error (px) — min/median/max:")
        print(f"  affine:      {min(aff):.1f} / {statistics.median(aff):.1f} / {max(aff):.1f}")
        print(f"  projective:  {min(prj):.1f} / {statistics.median(prj):.1f} / {max(prj):.1f}")

    print(f"\nTrace: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
