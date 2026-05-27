#!/usr/bin/env python3
"""Render large visual panels for current scoreboard tail cases.

This diagnostic sits on top of the current hull-label repair scoreboards:

* ``tools/diagnose_hull_label_color_repair.py`` shows the per-side
  threshold selector plus deterministic color/count/legal repair.
* ``tools/diagnose_pair_threshold_repair.py`` shows the guarded pair-level
  threshold selector.

When the per-side path leaves a non-exact row, this tool renders the current
threshold pair against the guarded pair-selected threshold pair so the failure
mode is visible rather than only numeric. It intentionally uses the same
1600px/rembg/threshold geometry path as Fixer.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.validation import FACE_ORDER  # noqa: E402
from tools.audit_recognition_pair import parse_ground_truth as parse_pair_ground_truth  # noqa: E402
from tools.diagnose_pair_threshold_repair import _evaluate_side_fits, _fit_threshold_candidates  # noqa: E402
from tools.extract_color_samples import PairTask, load_corpus_tasks  # noqa: E402
from tools.hull_label_assembly import convention_orientation_for_slot  # noqa: E402
from tools.hull_label_color_repair import repair_from_hull_label_fits  # noqa: E402
from tools.rectify_faces import DEFAULT_FACE_SIZE, rectify_face  # noqa: E402


DEFAULT_COLOR_REPAIR_JSON = REPO_ROOT / "tests" / "fixtures" / "hull_label_color_repair_diagnostic.json"
DEFAULT_PAIR_JSON = REPO_ROOT / "tests" / "fixtures" / "pair_threshold_repair_diagnostic.json"
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_OUT_DIR = REPO_ROOT / "tools" / "current_scoreboard_failure_gallery"
DEFAULT_REPORT = REPO_ROOT / "tools" / "CURRENT_SCOREBOARD_FAILURE_GALLERY.md"

FACE_COLORS = {
    "U": (245, 245, 245),
    "R": (212, 50, 45),
    "F": (40, 150, 80),
    "D": (246, 220, 56),
    "L": (238, 125, 42),
    "B": (55, 95, 200),
}


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _font(size: int = 14) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _face_chunk(state: Optional[str], face: str) -> Optional[str]:
    if not state or len(state) != 54:
        return None
    offset = FACE_ORDER.index(face) * 9
    return state[offset:offset + 9]


def _hamming(actual: Optional[str], expected: str) -> Optional[int]:
    if not actual or len(actual) != len(expected):
        return None
    return sum(1 for got, want in zip(actual, expected) if got != want)


def _state_wrong_labels(actual: Optional[str], expected: str) -> List[str]:
    if not actual or len(actual) != len(expected):
        return []
    labels: List[str] = []
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got == want:
            continue
        labels.append(f"{FACE_ORDER[index // 9]}{index % 9 + 1}:{got}->{want}")
    return labels


def _transform_for_orientation(img: Image.Image, *, mirror: bool, rot_quarter: int) -> Image.Image:
    out = img
    if mirror:
        out = ImageOps.mirror(out)
    # PIL rotate is counter-clockwise for positive angles, matching
    # tools.sample_stickers_from_hull.apply_orientation.
    if rot_quarter % 4:
        out = out.rotate(90 * (rot_quarter % 4), expand=False)
    return out


def _draw_quads_on_source(image: Image.Image, fit: Any, *, title: str) -> Image.Image:
    thumb = image.convert("RGB").copy()
    thumb.thumbnail((420, 420))
    sx = thumb.width / image.width
    sy = thumb.height / image.height
    draw = ImageDraw.Draw(thumb, "RGBA")
    slot_colors = {
        "upper": (255, 230, 80, 240),
        "right": (100, 170, 255, 240),
        "front": (80, 230, 140, 240),
    }
    if fit is not None:
        for slot, quad in fit.face_quads.items():
            pts = [(int(x * sx), int(y * sy)) for x, y in quad]
            color = slot_colors.get(slot, (255, 80, 80, 240))
            draw.line(pts + [pts[0]], fill=color, width=4)
        vx, vy = fit.vertex
        draw.ellipse(
            (vx * sx - 6, vy * sy - 6, vx * sx + 6, vy * sy + 6),
            fill=(255, 70, 70, 255),
            outline=(255, 255, 255, 255),
            width=2,
        )
        for num, (cx, cy) in fit.corners_by_num.items():
            draw.ellipse(
                (cx * sx - 5, cy * sy - 5, cx * sx + 5, cy * sy + 5),
                fill=(255, 190, 0, 255),
                outline=(20, 20, 20, 255),
                width=1,
            )
            draw.text((cx * sx + 7, cy * sy - 7), str(num), fill=(255, 255, 255, 255), font=_font(13))
    label_h = 28
    canvas = Image.new("RGB", (thumb.width, thumb.height + label_h), (250, 250, 250))
    cdraw = ImageDraw.Draw(canvas)
    cdraw.text((4, 5), title, fill=(30, 30, 30), font=_font(15))
    canvas.paste(thumb, (0, label_h))
    return canvas


def _draw_face_panel(
    face_img: Image.Image,
    *,
    face: str,
    raw_chunk: Optional[str],
    final_chunk: Optional[str],
    gt_chunk: str,
    title_suffix: str = "",
    size: int = 210,
) -> Image.Image:
    img = face_img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    label_h = 58
    panel = Image.new("RGB", (size, size + label_h), (246, 246, 246))
    draw = ImageDraw.Draw(panel)
    title = f"{face}{title_suffix}"
    draw.text((6, 5), title, fill=(20, 20, 20), font=_font(17))
    if final_chunk is not None:
        draw.text((6, 29), f"final {final_chunk}", fill=(80, 80, 80), font=_font(12))
    panel.paste(img, (0, label_h))
    cell = size / 3.0
    for i in range(4):
        pos = int(round(i * cell))
        draw.line((pos, label_h, pos, label_h + size), fill=(255, 255, 255), width=2)
        draw.line((0, label_h + pos, size, label_h + pos), fill=(255, 255, 255), width=2)
    for idx in range(9):
        row, col = divmod(idx, 3)
        x0 = int(round(col * cell))
        y0 = label_h + int(round(row * cell))
        x1 = int(round((col + 1) * cell))
        y1 = label_h + int(round((row + 1) * cell))
        raw = raw_chunk[idx] if raw_chunk else "?"
        final = final_chunk[idx] if final_chunk else "?"
        gt = gt_chunk[idx]
        if final != gt:
            draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), outline=(235, 50, 50), width=5)
        elif raw != gt:
            draw.rectangle((x0 + 3, y0 + 3, x1 - 3, y1 - 3), outline=(245, 160, 35), width=3)
        face_rgb = FACE_COLORS.get(final, (230, 230, 230))
        luminance = 0.2126 * face_rgb[0] + 0.7152 * face_rgb[1] + 0.0722 * face_rgb[2]
        text_fill = (0, 0, 0) if luminance > 145 else (255, 255, 255)
        chip = (x0 + 5, y0 + 5, x0 + 30, y0 + 30)
        draw.rounded_rectangle(chip, radius=4, fill=face_rgb, outline=(30, 30, 30), width=1)
        draw.text((x0 + 11, y0 + 8), final, fill=text_fill, font=_font(14))
        draw.text((x0 + 5, y1 - 20), f"r:{raw}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), font=_font(11))
        draw.text((x1 - 28, y1 - 20), f"g:{gt}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), font=_font(11))
    return panel


def _blank_panel(text: str, width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), (245, 245, 245))
    ImageDraw.Draw(img).text((12, 12), text, fill=(40, 40, 40), font=_font(14))
    return img


def _variant_side_fits(
    task: PairTask,
    thresholds: Mapping[str, int],
    sess: Any,
) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for side, image_path in (("A", task.image_a), ("B", task.image_b)):
        by_threshold, _trace = _fit_threshold_candidates(image_path, side, sess)
        threshold = int(thresholds[side])
        if threshold not in by_threshold:
            raise ValueError(f"set {task.set_id} side {side} has no fit for threshold {threshold}")
        entry = dict(by_threshold[threshold])
        trace = dict(entry.get("trace") or {})
        trace["threshold"] = threshold
        entry["trace"] = trace
        out[side] = entry
    return out


def _oriented_faces_for_variant(
    side_fits: Mapping[str, Mapping[str, Any]],
    yaw: int,
) -> Dict[str, Image.Image]:
    faces: Dict[str, Image.Image] = {}
    for side in ("A", "B"):
        entry = side_fits[side]
        image = entry["image"]
        fit = entry["fit"]
        # Use the canonical helper because yaw rotates side-face assignments.
        from tools.hull_label_assembly import slot_face_assignments
        assignments = slot_face_assignments(yaw)
        for slot, wca_face in assignments[side].items():
            quad = fit.face_quads[slot]
            orientation = convention_orientation_for_slot(
                side=side,
                slot=slot,
                yaw_quarter_turns=yaw,
                wca_face=wca_face,
                quad=quad,
            )
            if orientation is None:
                continue
            raw_img = rectify_face(image, quad, output_size=DEFAULT_FACE_SIZE)
            faces[wca_face] = _transform_for_orientation(
                raw_img,
                mirror=bool(orientation[0]),
                rot_quarter=int(orientation[1]),
            )
    return faces


def _render_variant_column(
    *,
    title: str,
    task: PairTask,
    side_fits: Mapping[str, Mapping[str, Any]],
    gt_state: str,
    evaluation: Mapping[str, Any],
    width: int,
) -> Image.Image:
    yaw = int(evaluation["yawInference"]["yawQuarterTurns"])
    payload = repair_from_hull_label_fits(
        side_fits=side_fits,
        yaw_quarter_turns=yaw,
        gt_state=gt_state,
    )
    methods = payload["methods"]
    raw_state = methods["canonical"]["state"]
    count_state = methods["canonical_count_repaired"]["state"]
    recommended = payload["recommended"]
    recommended_state = recommended.get("state") or count_state
    hamming = _hamming(recommended_state, gt_state)
    faces = _oriented_faces_for_variant(side_fits, yaw)

    thumbs = [
        _draw_quads_on_source(side_fits["A"]["image"], side_fits["A"]["fit"], title=f"A threshold {side_fits['A']['trace'].get('threshold')}"),
        _draw_quads_on_source(side_fits["B"]["image"], side_fits["B"]["fit"], title=f"B threshold {side_fits['B']['trace'].get('threshold')}"),
    ]
    face_panels = []
    for face in FACE_ORDER:
        img = faces.get(face)
        if img is None:
            face_panels.append(_blank_panel(f"{face}: missing", 210, 268))
            continue
        face_panels.append(
            _draw_face_panel(
                img,
                face=face,
                raw_chunk=_face_chunk(raw_state, face),
                final_chunk=_face_chunk(recommended_state, face),
                gt_chunk=_face_chunk(gt_state, face) or "?????????",
                title_suffix=f"  raw/count/gt",
            )
        )

    header_h = 120
    source_h = max(t.height for t in thumbs)
    face_h = face_panels[0].height
    gap = 14
    total_h = header_h + source_h + gap + 2 * face_h + 3 * gap
    column = Image.new("RGB", (width, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(column)
    draw.rectangle((0, 0, width - 1, total_h - 1), outline=(210, 210, 210), width=2)
    draw.text((16, 12), title, fill=(20, 20, 20), font=_font(22))
    draw.text(
        (16, 44),
        f"yaw={yaw}  recommended={payload.get('recommendedMethod')}  hamming={hamming}  valid={recommended.get('validState')}",
        fill=(55, 55, 55),
        font=_font(14),
    )
    wrong = _state_wrong_labels(recommended_state, gt_state)
    wrong_text = "wrong: " + (", ".join(wrong) if wrong else "none")
    draw.text((16, 68), wrong_text[:120], fill=(150, 20, 20) if wrong else (20, 120, 50), font=_font(13))
    if len(wrong_text) > 120:
        draw.text((16, 88), wrong_text[120:240], fill=(150, 20, 20), font=_font(13))

    y = header_h
    x = 16
    for thumb in thumbs:
        column.paste(thumb, (x, y))
        x += thumb.width + gap
    y += source_h + gap
    for row in range(2):
        x = 16
        for col in range(3):
            panel = face_panels[row * 3 + col]
            column.paste(panel, (x, y))
            x += panel.width + gap
        y += face_h + gap
    return column


def _task_by_set(manifest: Path) -> Dict[str, PairTask]:
    return {task.set_id: task for task in load_corpus_tasks(manifest)}


def _candidate_rows(pair_payload: Mapping[str, Any], only_sets: Optional[Iterable[str]]) -> List[Mapping[str, Any]]:
    only = {str(item) for item in only_sets or []}
    rows = []
    for row in pair_payload.get("rows", []):
        if only and str(row.get("setId")) not in only:
            continue
        current = row.get("current", {}).get("summary", {}).get("recommended", {})
        selected = row.get("pairSelected", {}).get("summary", {}).get("recommended", {})
        if current.get("hamming") not in (None, 0) or selected.get("hamming") not in (None, 0):
            rows.append(row)
    return rows


def render_report(records: Sequence[Mapping[str, Any]], *, git_sha: str) -> str:
    lines = [
        "# Current Scoreboard Failure Gallery",
        "",
        "Diagnostic-only visual walkthrough of rows where the current per-side",
        "threshold repair path is not exact, compared against the guarded",
        "pair-threshold selector.",
        "",
        f"Git head: `{git_sha}`",
        "",
        "## Reading The Panels",
        "",
        "- Each large panel has the current per-side threshold path on the left and",
        "  the guarded pair-selected threshold path on the right.",
        "- Source thumbnails show the selected hull-label face quads. Yellow points",
        "  are silhouette corners; the red point is the derived vertex.",
        "- Rectified face cells show `r:<raw>` for raw canonical Lab classification,",
        "  the center chip/letter for the final recommended state, and `g:<gt>`",
        "  for ground truth. Red borders mark final wrong stickers; orange borders",
        "  mark raw mistakes that repair fixed.",
        "",
        "## Panels",
        "",
    ]
    for record in records:
        lines.extend([
            f"### Set {record['setId']}",
            "",
            f"- Current thresholds: `{record['currentThresholds']}` -> hamming `{record['currentHamming']}`.",
            f"- Guarded pair thresholds: `{record['selectedThresholds']}` -> hamming `{record['selectedHamming']}`.",
            "",
            f"![Set {record['setId']} walkthrough]({record['panelPath']})",
            "",
        ])
    if not records:
        lines.append("No non-exact current or selected rows were found.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pair-json", type=Path, default=DEFAULT_PAIR_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--only-sets", nargs="*", default=None)
    args = parser.parse_args(argv)

    pair_payload = json.loads(args.pair_json.read_text(encoding="utf-8"))
    tasks = _task_by_set(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from rembg import new_session  # noqa: E402
    sess = new_session("u2net")

    records: List[Dict[str, Any]] = []
    for row in _candidate_rows(pair_payload, args.only_sets):
        set_id = str(row["setId"])
        task = tasks[set_id]
        _sha, _raw_state, gt_state, _canonicalized = parse_pair_ground_truth(str(task.ground_truth))
        current_thresholds = row["current"]["thresholds"]
        selected_thresholds = row["pairSelected"]["thresholds"]
        print(f"rendering set {set_id}: current {current_thresholds} vs selected {selected_thresholds}", flush=True)
        current_fits = _variant_side_fits(task, current_thresholds, sess)
        selected_fits = _variant_side_fits(task, selected_thresholds, sess)
        current_eval = _evaluate_side_fits(current_fits, gt_state)
        selected_eval = _evaluate_side_fits(selected_fits, gt_state)
        col_w = 930
        left = _render_variant_column(
            title="Current per-side selector",
            task=task,
            side_fits=current_fits,
            gt_state=gt_state,
            evaluation=current_eval,
            width=col_w,
        )
        right = _render_variant_column(
            title="Guarded pair-threshold selector",
            task=task,
            side_fits=selected_fits,
            gt_state=gt_state,
            evaluation=selected_eval,
            width=col_w,
        )
        margin = 24
        header_h = 72
        panel = Image.new("RGB", (2 * col_w + 3 * margin, header_h + max(left.height, right.height) + margin), (238, 240, 244))
        draw = ImageDraw.Draw(panel)
        draw.text((margin, 18), f"Set {set_id}: current failure vs guarded pair-threshold repair", fill=(20, 20, 24), font=_font(24))
        draw.text((margin, 46), "Large cells expose whether the remaining error is geometry, color, or threshold selection.", fill=(70, 70, 76), font=_font(14))
        panel.paste(left, (margin, header_h))
        panel.paste(right, (2 * margin + col_w, header_h))
        out_path = args.out_dir / f"set_{set_id}_current_vs_pair_selected.png"
        panel.save(out_path)
        records.append({
            "setId": set_id,
            "currentThresholds": current_thresholds,
            "selectedThresholds": selected_thresholds,
            "currentHamming": row.get("current", {}).get("summary", {}).get("recommended", {}).get("hamming"),
            "selectedHamming": row.get("pairSelected", {}).get("summary", {}).get("recommended", {}).get("hamming"),
            "panelPath": os.path.relpath(out_path.resolve(), start=args.report.resolve().parent),
        })

    args.report.write_text(render_report(records, git_sha=_git_head_sha()), encoding="utf-8")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
