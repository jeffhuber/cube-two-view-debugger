#!/usr/bin/env python3
"""Probe: determine `yaw_quarter_turns` per row by sampling the right-slot
center sticker color in each image.

Math: the right-slot face center is the centroid of the 4 quad corners
(Va/Vb + 3 outer corners per `FACE_DEFS_BY_SIDE`). Sample a small RGB
patch there, classify via the recognizer's color classifier, then map
the WCA face → yaw using `corner_conventions.wca_face_by_slot`:

  Image A (right slot under each yaw):
    yaw=0 → R (red)    yaw=1 → B (blue)
    yaw=2 → L (orange) yaw=3 → F (green)

  Image B (right slot under each yaw):
    yaw=0 → B (blue)   yaw=1 → L (orange)
    yaw=2 → F (green)  yaw=3 → R (red)

Diagnostic-only: emits a Markdown table of (set, side, sampled RGB,
classified color, proposed yaw). The user spot-checks; the actual yaw
values land in `tests/fixtures/full_corner_ground_truth.json` via a
separate small commit on this branch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import classify_rgb  # noqa: E402
from tools.corner_conventions import FACE_DEFS_BY_SIDE  # noqa: E402


# Right-slot WCA face name → cube color → expected yaw (per side).
# Derived from `wca_face_by_slot(side, yaw)` for the right slot.
RIGHT_SLOT_FACE_BY_YAW = {
    "A": {0: "R", 1: "B", 2: "L", 3: "F"},
    "B": {0: "B", 1: "L", 2: "F", 3: "R"},
}
# WCA face standard center colors (Western/Japanese scheme — used here).
FACE_COLOR = {
    "U": "white", "D": "yellow",
    "R": "red", "L": "orange",
    "F": "green", "B": "blue",
}
# Reverse: color → WCA face.
COLOR_TO_FACE = {v: k for k, v in FACE_COLOR.items()}


def _resolve_image_path(set_id: str, side: str) -> Optional[Path]:
    """Resolve the image path for (set, side) via corpus_manifest.json."""
    manifest = json.loads(
        (REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json")
        .read_text(encoding="utf-8")
    )
    for pair in manifest.get("pairs", []):
        if str(pair.get("setId")) == str(set_id):
            return Path(pair["imageAPath" if side == "A" else "imageBPath"])
    return None


def _sample_right_slot_center(
    image_path: Path, vertex_xy: Tuple[float, float],
    corner_a_xy: Tuple[float, float],
    corner_b_xy: Tuple[float, float],
    corner_c_xy: Tuple[float, float],
    patch_half_px: int = 20,
) -> Tuple[Tuple[int, int, int], Tuple[int, int]]:
    """Sample the median RGB of a small patch centered at the face quad's
    centroid. Returns (rgb, (center_x, center_y))."""
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    arr = np.asarray(image)
    cx = int(round(sum(p[0] for p in (vertex_xy, corner_a_xy, corner_b_xy, corner_c_xy)) / 4.0))
    cy = int(round(sum(p[1] for p in (vertex_xy, corner_a_xy, corner_b_xy, corner_c_xy)) / 4.0))
    h, w = arr.shape[:2]
    y0 = max(0, cy - patch_half_px)
    y1 = min(h, cy + patch_half_px + 1)
    x0 = max(0, cx - patch_half_px)
    x1 = min(w, cx + patch_half_px + 1)
    patch = arr[y0:y1, x0:x1].reshape(-1, 3)
    rgb = tuple(int(np.median(patch[:, ch])) for ch in range(3))
    return rgb, (cx, cy)


def _propose_yaw(side: str, classified_color: str) -> Optional[int]:
    """Map sampled-color → proposed yaw via RIGHT_SLOT_FACE_BY_YAW."""
    face = COLOR_TO_FACE.get(classified_color)
    if face is None:
        return None
    table = RIGHT_SLOT_FACE_BY_YAW[side]
    for yaw, expected_face in table.items():
        if expected_face == face:
            return yaw
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--truth", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "full_corner_ground_truth.json",
    )
    ap.add_argument(
        "--out-md", type=Path,
        default=REPO_ROOT / "tools" / "YAW_PROBE_PROPOSED.md",
    )
    args = ap.parse_args()

    truth = json.loads(args.truth.read_text(encoding="utf-8"))

    rows = []
    for key in sorted(truth.keys()):
        set_id, side = key.rsplit("_", 1)
        if side not in ("A", "B"):
            continue
        face_def = FACE_DEFS_BY_SIDE[side]["right"]  # vertex + 3 outer corners
        truth_row = truth[key]
        try:
            quad = [truth_row[name] for name in face_def]
        except KeyError as exc:
            rows.append({"key": key, "error": f"missing key {exc}"})
            continue
        image_path = _resolve_image_path(set_id, side)
        if image_path is None or not image_path.exists():
            rows.append({"key": key, "error": f"image not found: {image_path}"})
            continue
        try:
            rgb, (cx, cy) = _sample_right_slot_center(image_path, *quad)
        except Exception as exc:  # noqa: BLE001
            rows.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
            continue
        classified = classify_rgb(rgb)
        classified_color = getattr(classified, "color", str(classified))
        proposed = _propose_yaw(side, classified_color)
        rows.append({
            "key": key, "side": side,
            "image": image_path.name,
            "sampled_rgb": rgb,
            "sampled_center_px": (cx, cy),
            "classified_color": classified_color,
            "proposed_yaw_quarter_turns": proposed,
            "rationale": (
                f"right slot center is {classified_color} (RGB={rgb}); "
                f"per FACE_BY_YAW[{side}], that maps to yaw="
                f"{proposed}" if proposed is not None
                else f"right slot center color {classified_color} did not "
                     f"map to a WCA face — manual review needed"
            ),
        })

    # Render Markdown
    lines = [
        "# Yaw probe — proposed values for full_corner_ground_truth.json",
        "",
        "Sampled the right-slot face center sticker in each of the 12 ",
        "full-corner-truth rows and mapped its color → WCA face → yaw via ",
        "`tools/corner_conventions.wca_face_by_slot`.",
        "",
        "**For user confirmation before committing.** Spot-check rows where ",
        "the classified color seems off (lighting / sticker glare can ",
        "confuse the classifier).",
        "",
        "| Key | Image | Sampled RGB | Classified | Proposed yaw | Center px |",
        "|---|---|---|---|---:|---|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(
                f"| `{r['key']}` | (error: {r['error']}) | — | — | — | — |"
            )
            continue
        proposed = r["proposed_yaw_quarter_turns"]
        proposed_str = str(proposed) if proposed is not None else "**?**"
        lines.append(
            f"| `{r['key']}` | `{r['image']}` | "
            f"`{r['sampled_rgb']}` | `{r['classified_color']}` | "
            f"{proposed_str} | `{r['sampled_center_px']}` |"
        )
    lines.append("")
    lines.append("## Mapping reference")
    lines.append("")
    lines.append("Right-slot WCA face at each yaw_quarter_turns:")
    lines.append("")
    lines.append("| Yaw | Image A right slot | Image B right slot |")
    lines.append("|---:|---|---|")
    for y in range(4):
        a = RIGHT_SLOT_FACE_BY_YAW["A"][y]
        b = RIGHT_SLOT_FACE_BY_YAW["B"][y]
        lines.append(
            f"| {y} | `{a}` ({FACE_COLOR[a]}) | `{b}` ({FACE_COLOR[b]}) |"
        )
    lines.append("")

    args.out_md.write_text("\n".join(lines))
    print(f"wrote proposed yaw values to {args.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
