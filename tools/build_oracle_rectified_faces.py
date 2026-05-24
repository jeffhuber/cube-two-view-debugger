#!/usr/bin/env python3
"""Build oracle-quality rectified faces + per-sticker color samples.

Diagnostic-only. Per the design in `tools/ORACLE_RECTIFIED_FACES_DESIGN.md`
(PR #257), this tool consumes:

  - `tests/fixtures/full_corner_ground_truth.json` (human-validated
    vertex+corner_0..corner_5 + yaw_quarter_turns per row, post-#256)
  - `tools/corner_conventions.py` (FACE_DEFS_BY_SIDE,
    YAW0_CORNER_FACELETS, wca_facelets_for_label, wca_face_by_slot)
  - `tests/fixtures/corpus_manifest.json` (image paths)

and emits:

  /tmp/oracle_rectified_faces_v1/                  (default; ephemeral)
    by_row/{key}/{wca_face}.png                    (36 face PNGs)
    by_observation/{key}/{wca_facelet_id}.png      (324 sticker PNGs)
    patch_png/{key}_{wca_facelet_id}.png           (324 raw 41x41 patches)
    by_facelet/{wca_facelet_id}/{key}.png          (grouped comparison view)
    index.json                                     (full metadata)
    gallery.html                                   (visual inspection grid)

Isolates color classification from geometry uncertainty: every sample
uses human-validated face corners + canonical conventions, so any
color-classification mistake is purely a color problem, not geometry
contamination.

Sticker geometry after rectification is provably regular (the
perspective homography removes all projective distortion). Each cell
is `face_size / 3` per side; sticker (r, c)'s center pixel is at
`((c + 0.5) * cell, (r + 0.5) * cell)`. Standard URFDLB sticker
numbering: `sticker_id = 3*row + col + 1`. Color sample is the median
RGB over a central patch of `patch_fraction * cell_size` per side
(default 40%).

Convention mapping (load-bearing — see `_rectification_quad_for`):
`FACE_DEFS_BY_SIDE[side][slot]` lists the human face outline order
`(vertex, outer_corner_A, outer_corner_B, outer_corner_C)`. That is
NOT directly the order `_perspective_coeffs` expects. The rectification
quad order is `[TL_source, TR_source, BR_source, BL_source]`, derived
by mapping each outline label to its WCA facelet ID via
`wca_facelets_for_label(side, label, yaw)`, selecting the facelet whose
face matches `wca_face_by_slot(side, yaw)[slot]`, and placing it by
its digit: `1 -> TL, 3 -> TR, 7 -> BL, 9 -> BR`.

CLI:

  python3 tools/build_oracle_rectified_faces.py \\
      --truth tests/fixtures/full_corner_ground_truth.json \\
      --out /tmp/oracle_rectified_faces_v1/ \\
      --face-size 300 \\
      --sticker-patch 0.40 \\
      [--yaw-overrides "20:1,38:0"] \\
      [--no-patches] \\
      [--rows-glob "20_*,40_A"]

Run as a module to use the canonical sys.path:

    PYTHONPATH=. python3 tools/build_oracle_rectified_faces.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import (  # noqa: E402
    classify_rgb,
    rgb_to_hsv,
    rgb_to_lab,
)
from tools.corner_conventions import (  # noqa: E402
    FACE_DEFS_BY_SIDE,
    VERTEX_NAME_BY_SIDE,
    wca_face_by_slot,
    wca_facelets_for_label,
)
from tools.rectify_faces import _perspective_coeffs  # noqa: E402


# ---------------- defaults / constants -----------------

DEFAULT_TRUTH = REPO_ROOT / "tests" / "fixtures" / "full_corner_ground_truth.json"
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_OUT = Path("/tmp/oracle_rectified_faces_v1")
DEFAULT_FACE_SIZE = 300
DEFAULT_PATCH_FRACTION = 0.40
SLOTS: Tuple[str, str, str] = ("upper", "right", "front")

# Quad-corner-of-target convention used by `_perspective_coeffs`:
# dst = [(0,0), (size,0), (size,size), (0,size)] = [TL, TR, BR, BL].
# Sticker digit -> target-square corner.
_DIGIT_TO_QUAD_POSITION: Dict[str, int] = {
    "1": 0,  # TL
    "3": 1,  # TR
    "9": 2,  # BR
    "7": 3,  # BL
}


# ---------------- convention mapping (load-bearing) ----------------


def _rectification_quad_labels_for(
    side: str, slot: str, yaw_quarter_turns: int
) -> Tuple[str, str, str, str]:
    """Return the outline labels in `[TL, TR, BR, BL]` order.

    Derivation per the design doc:

    1. Look up the human face outline labels in
       `FACE_DEFS_BY_SIDE[side][slot]` (4 labels, starting with the
       visible trihedral vertex).
    2. For each label, compute the 3 WCA facelets it touches via
       `wca_facelets_for_label(side, label, yaw)`.
    3. Select the facelet whose face matches the slot's WCA face for
       this yaw — `wca_face_by_slot(side, yaw)[slot]`. That facelet's
       digit (1, 3, 7, or 9) names the target-square corner.
    4. Place the label at that corner.

    This is the same derivation in code form as the worked tables in
    `tools/ORACLE_RECTIFIED_FACES_DESIGN.md`. The tests pin yaw=0 +
    yaw=1 cases to ground-truth tables; this function MUST match.
    """
    outline_labels = FACE_DEFS_BY_SIDE[side][slot]
    target_face = wca_face_by_slot(side, yaw_quarter_turns)[slot]
    quad: List[Optional[str]] = [None, None, None, None]
    for label in outline_labels:
        facelets = wca_facelets_for_label(side, label, yaw_quarter_turns)
        # Exactly one of the 3 facelets at this corner is on the
        # slot's WCA face — pick it.
        digit = None
        for facelet in facelets:
            if facelet[0] == target_face:
                digit = facelet[1]
                break
        if digit is None:
            raise RuntimeError(
                f"no facelet on face {target_face} found among "
                f"{facelets} for ({side}, {slot}, yaw={yaw_quarter_turns}, "
                f"label={label})"
            )
        position = _DIGIT_TO_QUAD_POSITION[digit]
        if quad[position] is not None:
            raise RuntimeError(
                f"duplicate assignment at position {position} "
                f"({['TL','TR','BR','BL'][position]}) for "
                f"({side}, {slot}, yaw={yaw_quarter_turns}); "
                f"labels {outline_labels}"
            )
        quad[position] = label
    if any(item is None for item in quad):
        raise RuntimeError(
            f"failed to fill rectification quad for "
            f"({side}, {slot}, yaw={yaw_quarter_turns}); "
            f"labels {outline_labels} -> {quad}"
        )
    return tuple(quad)  # type: ignore[return-value]


def _facelet_ids_for_slot(
    side: str, slot: str, yaw_quarter_turns: int
) -> List[str]:
    """Return the 9 WCA facelet IDs for the rectified face in row-major
    URFDLB order (sticker 1..9 = (row=0,col=0)..(row=2,col=2))."""
    target_face = wca_face_by_slot(side, yaw_quarter_turns)[slot]
    return [f"{target_face}{i}" for i in range(1, 10)]


def sticker_id_from_row_col(row: int, col: int) -> int:
    """Standard URFDLB row-major numbering: 1 2 3 / 4 5 6 / 7 8 9."""
    return 3 * row + col + 1


# ---------------- rectification ----------------


def rectify_face_oracle(
    image: Image.Image,
    quad_image_px: Sequence[Tuple[float, float]],
    output_size: int,
) -> Image.Image:
    """Warp the source face quad (in `[TL, TR, BR, BL]` order) to a flat
    `output_size x output_size` square. Bicubic interpolation.

    Re-implements rectification without `tools.rectify_faces.rectify_face`'s
    `canonical_corner_order()` call: that ordering is "CW-from-N" and does
    not match `FACE_DEFS_BY_SIDE` order. Bypassing it keeps the convention
    mapping explicit here.
    """
    if len(quad_image_px) != 4:
        raise ValueError(
            f"quad must have 4 points; got {len(quad_image_px)}"
        )
    coeffs = _perspective_coeffs(quad_image_px, output_size)
    return image.transform(
        (output_size, output_size),
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC,
    )


# ---------------- sticker sampling ----------------


@dataclass(frozen=True)
class StickerSampleOracle:
    row: int
    col: int
    facelet_id: str
    sticker_id: int  # 1..9 within face
    rgb: Tuple[int, int, int]
    hsv: Tuple[float, float, float]
    lab: Tuple[float, float, float]
    classify_color: str
    classify_confidence: float
    patch_pixel_center_in_face: Tuple[int, int]
    cell_image: Image.Image   # full sticker cell crop (cell-sized, with
                              # surrounding bezel context); used for
                              # by_observation/{key}/{facelet_id}.png so
                              # consumers can see each sticker plus its
                              # immediate context. (Codex P2 on #259.)
    patch_image: Image.Image  # central patch_fraction crop used for
                              # color sampling; written to
                              # patch_png/{key}_{facelet_id}.png so
                              # downstream tools can re-derive any
                              # color statistic without re-rectifying.


def sample_stickers_oracle(
    face_img: Image.Image,
    side: str,
    slot: str,
    yaw_quarter_turns: int,
    patch_fraction: float = DEFAULT_PATCH_FRACTION,
) -> List[StickerSampleOracle]:
    """Sample the 9 stickers from a rectified face. Returns row-major
    list (length 9), each with the full color triplet + raw patch."""
    width, height = face_img.size
    if width != height:
        raise ValueError(
            f"rectified face must be square; got {width}x{height}"
        )
    facelet_ids = _facelet_ids_for_slot(side, slot, yaw_quarter_turns)
    cell = width / 3.0
    patch_half = int(cell * patch_fraction / 2)
    arr = np.asarray(face_img)
    samples: List[StickerSampleOracle] = []
    for row in range(3):
        cy = int((row + 0.5) * cell)
        # Full cell bounds (the whole sticker square, including
        # surrounding bezel context). Written to by_observation/
        # so consumers can see each sticker in its rectified context.
        cell_y0 = int(row * cell)
        cell_y1 = int((row + 1) * cell)
        for col in range(3):
            cx = int((col + 0.5) * cell)
            cell_x0 = int(col * cell)
            cell_x1 = int((col + 1) * cell)
            # Central patch (default 40% of cell) — for the color sample.
            y0 = max(0, cy - patch_half)
            y1 = cy + patch_half + 1
            x0 = max(0, cx - patch_half)
            x1 = cx + patch_half + 1
            patch_arr = arr[y0:y1, x0:x1]
            cell_arr = arr[cell_y0:cell_y1, cell_x0:cell_x1]
            if patch_arr.size == 0:
                rgb_tuple = (0, 0, 0)
            else:
                flat = patch_arr.reshape(-1, 3)
                rgb_tuple = tuple(
                    int(np.median(flat[:, ch])) for ch in range(3)
                )
            hsv = rgb_to_hsv(rgb_tuple)
            lab = rgb_to_lab(rgb_tuple)
            verdict = classify_rgb(rgb_tuple)
            sticker_id = sticker_id_from_row_col(row, col)
            facelet_id = facelet_ids[sticker_id - 1]
            patch_image = (
                Image.fromarray(patch_arr).copy()
                if patch_arr.size > 0
                else Image.new("RGB", (1, 1), (0, 0, 0))
            )
            cell_image = (
                Image.fromarray(cell_arr).copy()
                if cell_arr.size > 0
                else Image.new("RGB", (1, 1), (0, 0, 0))
            )
            samples.append(
                StickerSampleOracle(
                    row=row,
                    col=col,
                    facelet_id=facelet_id,
                    sticker_id=sticker_id,
                    rgb=rgb_tuple,
                    hsv=hsv,
                    lab=lab,
                    classify_color=verdict.color,
                    classify_confidence=verdict.confidence,
                    patch_pixel_center_in_face=(cx, cy),
                    cell_image=cell_image,
                    patch_image=patch_image,
                )
            )
    return samples


# ---------------- fixture loading ----------------


def _load_truth(path: Path) -> Dict[str, Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    """Return `{set_id: {"A": path, "B": path}}` from corpus_manifest."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_set: Dict[str, Dict[str, str]] = {}
    for pair in raw.get("pairs", []):
        set_id = pair.get("setId")
        if not set_id:
            continue
        entry: Dict[str, str] = {}
        if pair.get("imageAPath"):
            entry["A"] = pair["imageAPath"]
        if pair.get("imageBPath"):
            entry["B"] = pair["imageBPath"]
        if entry:
            by_set[set_id] = entry
    return by_set


def _parse_yaw_overrides(spec: str) -> Dict[str, int]:
    """Parse `--yaw-overrides "20:1,38:0"` into {"20": 1, "38": 0}."""
    if not spec:
        return {}
    out: Dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"--yaw-overrides item must be `set_id:yaw`; got {part!r}"
            )
        set_id, yaw_str = part.split(":", 1)
        out[set_id.strip()] = int(yaw_str.strip())
    return out


def _row_key_set_side(key: str) -> Tuple[str, str]:
    """`20_A` -> (`20`, `A`)."""
    set_id, _, side = key.rpartition("_")
    if not set_id or side not in ("A", "B"):
        raise ValueError(f"invalid row key: {key!r}")
    return set_id, side


def _row_yaw(
    row: Dict[str, Any],
    set_id: str,
    yaw_overrides: Dict[str, int],
) -> Tuple[int, bool]:
    """Return (yaw, was_assumed_zero)."""
    if set_id in yaw_overrides:
        return int(yaw_overrides[set_id]) % 4, False
    if "yaw_quarter_turns" in row:
        return int(row["yaw_quarter_turns"]) % 4, False
    return 0, True


def _quad_image_px(
    row: Dict[str, Any], labels: Sequence[str]
) -> List[Tuple[float, float]]:
    quad: List[Tuple[float, float]] = []
    for label in labels:
        if label not in row:
            raise KeyError(
                f"row missing point {label!r} (have: {sorted(row)})"
            )
        x, y = row[label]
        quad.append((float(x), float(y)))
    return quad


# ---------------- image loading ----------------


def load_oracle_image(image_path: Path) -> Image.Image:
    """Open + EXIF-correct + RGB-convert; preserve native resolution."""
    with Image.open(image_path) as raw:
        return ImageOps.exif_transpose(raw).convert("RGB")


# ---------------- per-row processing ----------------


@dataclass
class FaceRecord:
    slot: str
    wca_face: str
    face_png: str  # relative to out root
    outline_quad_image_px: List[Tuple[float, float]]
    rectification_quad_labels: Tuple[str, str, str, str]
    rectification_quad_image_px: List[Tuple[float, float]]
    stickers: List[Dict[str, Any]]


@dataclass
class RowRecord:
    key: str
    set_id: str
    side: str
    yaw_quarter_turns: int
    yaw_assumed_zero: bool
    image_path: str
    faces: List[FaceRecord]


def process_row(
    key: str,
    row: Dict[str, Any],
    *,
    image_path: Path,
    yaw_quarter_turns: int,
    yaw_assumed_zero: bool,
    face_size: int,
    patch_fraction: float,
    save_patches: bool,
    out_root: Path,
) -> RowRecord:
    set_id, side = _row_key_set_side(key)
    image = load_oracle_image(image_path)
    by_row_dir = out_root / "by_row" / key
    by_obs_dir = out_root / "by_observation" / key
    patch_dir = out_root / "patch_png"
    by_facelet_dir = out_root / "by_facelet"
    for d in (by_row_dir, by_obs_dir):
        d.mkdir(parents=True, exist_ok=True)
    if save_patches:
        patch_dir.mkdir(parents=True, exist_ok=True)
    by_facelet_dir.mkdir(parents=True, exist_ok=True)
    faces: List[FaceRecord] = []
    for slot in SLOTS:
        outline_labels = FACE_DEFS_BY_SIDE[side][slot]
        outline_quad = _quad_image_px(row, outline_labels)
        rect_labels = _rectification_quad_labels_for(
            side, slot, yaw_quarter_turns
        )
        rect_quad = _quad_image_px(row, rect_labels)
        face_img = rectify_face_oracle(image, rect_quad, face_size)
        wca_face = wca_face_by_slot(side, yaw_quarter_turns)[slot]
        face_filename = f"{wca_face}.png"
        face_path = by_row_dir / face_filename
        face_img.save(face_path)
        samples = sample_stickers_oracle(
            face_img, side, slot, yaw_quarter_turns, patch_fraction
        )
        stickers: List[Dict[str, Any]] = []
        for sample in samples:
            sticker_png_rel = (
                f"by_observation/{key}/{sample.facelet_id}.png"
            )
            # by_observation/{key}/{facelet_id}.png is the FULL sticker
            # cell crop (cell-sized, including surrounding bezel
            # context) — Codex P2 on #259. Consumers following the
            # documented `sticker_png` field need the per-sticker crop
            # with context, not just the central sampling patch.
            sample.cell_image.save(
                by_obs_dir / f"{sample.facelet_id}.png"
            )
            if save_patches:
                patch_filename = (
                    f"{key}_{sample.facelet_id}.png"
                )
                sample.patch_image.save(patch_dir / patch_filename)
                patch_rel: Optional[str] = f"patch_png/{patch_filename}"
            else:
                patch_rel = None
            # Grouped comparison view: same observation re-saved under
            # by_facelet/{facelet_id}/{key}.png so a downstream viewer
            # can compare the same facelet ID across observations. Uses
            # the full cell crop so the comparison shows sticker
            # context, not isolated central patches.
            grouped_dir = by_facelet_dir / sample.facelet_id
            grouped_dir.mkdir(parents=True, exist_ok=True)
            sample.cell_image.save(grouped_dir / f"{key}.png")
            stickers.append({
                "row": sample.row,
                "col": sample.col,
                "sticker_id": sample.sticker_id,
                "facelet_id": sample.facelet_id,
                "observation_id": f"{key}_{sample.facelet_id}",
                "sticker_png": sticker_png_rel,
                "patch_png": patch_rel,
                "patch_pixel_center_in_face": list(
                    sample.patch_pixel_center_in_face
                ),
                "rgb": list(sample.rgb),
                "hsv": [round(v, 4) for v in sample.hsv],
                "lab": [round(v, 2) for v in sample.lab],
                "classify_rgb": sample.classify_color,
                "classify_confidence": round(
                    float(sample.classify_confidence), 4
                ),
            })
        faces.append(FaceRecord(
            slot=slot,
            wca_face=wca_face,
            face_png=f"by_row/{key}/{face_filename}",
            outline_quad_image_px=outline_quad,
            rectification_quad_labels=rect_labels,
            rectification_quad_image_px=rect_quad,
            stickers=stickers,
        ))
    return RowRecord(
        key=key,
        set_id=set_id,
        side=side,
        yaw_quarter_turns=yaw_quarter_turns,
        yaw_assumed_zero=yaw_assumed_zero,
        image_path=str(image_path),
        faces=faces,
    )


def _serialize_row(row: RowRecord) -> Dict[str, Any]:
    return {
        "key": row.key,
        "set_id": row.set_id,
        "side": row.side,
        "yaw_quarter_turns": row.yaw_quarter_turns,
        "yaw_assumed_zero": row.yaw_assumed_zero,
        "image_path": row.image_path,
        "faces": [
            {
                "slot": face.slot,
                "wca_face": face.wca_face,
                "face_png": face.face_png,
                "outline_quad_image_px": [
                    list(p) for p in face.outline_quad_image_px
                ],
                "rectification_quad_labels": list(
                    face.rectification_quad_labels
                ),
                "rectification_quad_image_px": [
                    list(p) for p in face.rectification_quad_image_px
                ],
                "stickers": face.stickers,
            }
            for face in row.faces
        ],
    }


# ---------------- gallery HTML ----------------


def render_gallery_html(rows: List[RowRecord]) -> str:
    """Static HTML grid: each row is a band with 3 face thumbnails
    above the 9 sticker patches per face."""
    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append("<title>Oracle Rectified Faces</title>")
    parts.append("<style>")
    parts.append(
        "body{font-family:-apple-system,system-ui,sans-serif;"
        "margin:16px;color:#222;background:#fafafa}"
        "h1{font-size:18px;margin:0 0 12px}"
        "h2{font-size:14px;margin:18px 0 6px}"
        ".row{border:1px solid #ddd;background:#fff;padding:10px;"
        "margin-bottom:14px;border-radius:6px}"
        ".faces{display:flex;gap:10px;flex-wrap:wrap}"
        ".face{display:flex;flex-direction:column;align-items:flex-start}"
        ".face img{width:150px;height:150px;border:1px solid #888}"
        ".face .label{font-size:11px;margin-top:4px;color:#555}"
        ".stickers{display:grid;grid-template-columns:repeat(3,32px);"
        "gap:2px;margin-top:6px}"
        ".sticker{width:32px;height:32px;border:1px solid #ccc;"
        "display:flex;align-items:center;justify-content:center;"
        "font-size:9px;color:#333;text-shadow:0 0 2px #fff}"
        ".meta{font-size:11px;color:#666}"
    )
    parts.append("</style></head><body>")
    parts.append("<h1>Oracle rectified faces — diagnostic gallery</h1>")
    parts.append(
        "<div class='meta'>"
        f"{len(rows)} rows; "
        f"{sum(len(r.faces) for r in rows)} faces; "
        f"{sum(len(f.stickers) for r in rows for f in r.faces)} stickers."
        " Tile background = classifier verdict color (raw RGB unused for "
        "the swatch — that's by design; the swatch reflects the *decision*, "
        "not the raw pixel, so misclassifications visually pop).</div>"
    )
    for row in rows:
        parts.append("<div class='row'>")
        title = (
            f"<h2>{row.key} (side {row.side}, yaw="
            f"{row.yaw_quarter_turns}"
            f"{', ASSUMED ZERO' if row.yaw_assumed_zero else ''})</h2>"
        )
        parts.append(title)
        parts.append("<div class='faces'>")
        for face in row.faces:
            parts.append("<div class='face'>")
            parts.append(
                f"<img src='{face.face_png}' alt='{face.wca_face}'>"
            )
            parts.append(
                f"<div class='label'>"
                f"slot={face.slot}, face={face.wca_face}</div>"
            )
            parts.append("<div class='stickers'>")
            for sticker in face.stickers:
                bg = f"#{_color_swatch_hex(sticker['classify_rgb'])}"
                title_text = (
                    f"{sticker['facelet_id']} (row{sticker['row']},"
                    f"col{sticker['col']}) rgb="
                    f"{sticker['rgb']} -> {sticker['classify_rgb']}"
                    f" (conf {sticker['classify_confidence']:.2f})"
                )
                parts.append(
                    f"<div class='sticker' "
                    f"style='background:{bg}' title='{title_text}'>"
                    f"{sticker['sticker_id']}</div>"
                )
            parts.append("</div>")  # stickers
            parts.append("</div>")  # face
        parts.append("</div>")  # faces
        parts.append("</div>")  # row
    parts.append("</body></html>")
    return "\n".join(parts)


_CLASSIFIER_HEX = {
    "white": "ffffff",
    "yellow": "ffe34a",
    "red": "d33b3b",
    "orange": "f08a3a",
    "green": "3ec07a",
    "blue": "3a6dd3",
}


def _color_swatch_hex(classifier_color: str) -> str:
    return _CLASSIFIER_HEX.get(classifier_color, "888888")


# ---------------- orchestration ----------------


def select_rows(
    truth: Dict[str, Dict[str, Any]],
    rows_glob: str,
    *,
    require_approved: bool = True,
) -> List[str]:
    """Return sorted list of row keys matching `rows_glob` (comma-separated;
    `*` wildcard). Filters out unapproved rows unless `require_approved`
    is False."""
    import fnmatch
    patterns = [p.strip() for p in rows_glob.split(",") if p.strip()]
    if not patterns:
        patterns = ["*"]
    selected: List[str] = []
    for key, row in truth.items():
        if require_approved and not row.get("approved"):
            continue
        if any(fnmatch.fnmatch(key, p) for p in patterns):
            selected.append(key)
    selected.sort()
    return selected


#: Subdirectories this tool writes into. `clean_output_root` clears
#: exactly these (and the top-level index.json + gallery.html) so a
#: rerun with a narrower `--rows-glob`, different `--yaw-overrides`,
#: or `--no-patches` cannot leave stale files indistinguishable from
#: the current run's outputs. Codex P2 on #259.
_OWNED_SUBDIRS: Tuple[str, ...] = (
    "by_row",
    "by_observation",
    "by_facelet",
    "patch_png",
)
_OWNED_TOP_FILES: Tuple[str, ...] = ("index.json", "gallery.html")


def clean_output_root(out_root: Path) -> None:
    """Remove the subdirectories and top-level files this tool writes,
    leaving any unrelated content in `out_root` alone (so the
    default `/tmp/oracle_rectified_faces_v1/` is safely wiped without
    nuking sibling directories if the user pointed `--out` somewhere
    shared)."""
    import shutil
    if not out_root.exists():
        return
    for subdir in _OWNED_SUBDIRS:
        candidate = out_root / subdir
        if candidate.is_dir():
            shutil.rmtree(candidate)
    for top in _OWNED_TOP_FILES:
        candidate = out_root / top
        if candidate.is_file():
            candidate.unlink()


def build_all(
    *,
    truth_path: Path,
    manifest_path: Path,
    out_root: Path,
    face_size: int,
    patch_fraction: float,
    yaw_overrides: Dict[str, int],
    save_patches: bool,
    rows_glob: str,
) -> Dict[str, Any]:
    truth = _load_truth(truth_path)
    manifest = _load_manifest(manifest_path)
    keys = select_rows(truth, rows_glob)
    # Wipe the owned subdirs + top-level artifacts BEFORE writing. Without
    # this, a narrower rerun (different --rows-glob, different
    # --yaw-overrides, --no-patches, etc.) leaves stale files mixed in
    # with current ones, contaminating grouped comparisons and the
    # gallery view. Codex P2 on #259.
    clean_output_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    row_records: List[RowRecord] = []
    skipped: List[Dict[str, str]] = []
    for key in keys:
        row = truth[key]
        set_id, side = _row_key_set_side(key)
        image_path_str = (manifest.get(set_id) or {}).get(side)
        if not image_path_str:
            skipped.append({
                "key": key,
                "reason": "no image path in corpus_manifest for "
                          f"set {set_id} side {side}",
            })
            continue
        image_path = Path(image_path_str)
        if not image_path.exists():
            skipped.append({
                "key": key,
                "reason": f"image not found: {image_path}",
            })
            continue
        yaw, assumed = _row_yaw(row, set_id, yaw_overrides)
        try:
            record = process_row(
                key, row,
                image_path=image_path,
                yaw_quarter_turns=yaw,
                yaw_assumed_zero=assumed,
                face_size=face_size,
                patch_fraction=patch_fraction,
                save_patches=save_patches,
                out_root=out_root,
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append({"key": key, "reason": f"error: {exc!r}"})
            continue
        row_records.append(record)
    index = {
        "schema": "oracle_rectified_faces_v1",
        "source": {
            "truth_path": str(truth_path),
            "manifest_path": str(manifest_path),
            "face_size_px": face_size,
            "sticker_patch_fraction": patch_fraction,
            "save_patches": save_patches,
            "yaw_overrides": dict(yaw_overrides),
            "rows_glob": rows_glob,
        },
        "rows": [_serialize_row(r) for r in row_records],
        "skipped": skipped,
    }
    (out_root / "index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    gallery_html = render_gallery_html(row_records)
    (out_root / "gallery.html").write_text(gallery_html, encoding="utf-8")
    return index


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--face-size", type=int, default=DEFAULT_FACE_SIZE,
        help="Side length in pixels for rectified face squares (default 300).",
    )
    ap.add_argument(
        "--sticker-patch", type=float, default=DEFAULT_PATCH_FRACTION,
        help="Fraction of cell width used as central patch for color "
             "sampling (default 0.40).",
    )
    ap.add_argument(
        "--yaw-overrides", type=str, default="",
        help='Comma-separated "set_id:yaw_quarter_turns" overrides, e.g. '
             '"20:1,38:0". Defaults to each row\'s fixture yaw.',
    )
    ap.add_argument(
        "--no-patches", action="store_true",
        help="Skip per-sticker raw patch PNGs (saves ~3MB of /tmp).",
    )
    ap.add_argument(
        "--rows-glob", type=str, default="*",
        help="Comma-separated fnmatch patterns to filter row keys "
             "(e.g. '20_*,40_A'). Default '*' includes all approved rows.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)
    yaw_overrides = _parse_yaw_overrides(args.yaw_overrides)
    index = build_all(
        truth_path=args.truth,
        manifest_path=args.manifest,
        out_root=args.out,
        face_size=args.face_size,
        patch_fraction=args.sticker_patch,
        yaw_overrides=yaw_overrides,
        save_patches=not args.no_patches,
        rows_glob=args.rows_glob,
    )
    n_rows = len(index["rows"])
    n_faces = sum(len(r["faces"]) for r in index["rows"])
    n_stickers = sum(len(f["stickers"]) for r in index["rows"] for f in r["faces"])
    n_skipped = len(index["skipped"])
    print(
        f"wrote {n_rows} rows ({n_faces} faces, {n_stickers} stickers) "
        f"to {args.out}; {n_skipped} skipped",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
