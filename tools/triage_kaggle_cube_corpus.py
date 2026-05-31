#!/usr/bin/env python3
"""Triage a local Kaggle cube-image corpus into review buckets.

The Kaggle photos are not canonical A/B recognizer fixtures, so this tool does
not try to turn them directly into solver inputs. Instead it builds a local
manifest and contact sheets that make manual curation cheap:

* three_face_candidate - likely useful for future recognizer/detector eval
* single_face - likely visible single-face photos
* hand_occluded - hands/fingers probably cover or dominate the cube
* cropped_close - cube is too close to the frame edge for geometry work
* table_isometric - plausible table-top isometric-ish photos
* retake_negative - likely negative examples for rejection/retake behavior

Images are never copied into the repository. Outputs default to
``runs/kaggle_cube_triage`` which is ignored by git.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_INPUT_ROOT = Path("/Users/jhuber/cube-corpus/kaggle-cube-data")
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "kaggle_cube_triage"
BUCKETS = (
    "three_face_candidate",
    "single_face",
    "hand_occluded",
    "cropped_close",
    "table_isometric",
    "retake_negative",
)
TIMESTAMP_RE = re.compile(r"(?P<date>\d{8})[_-]?(?P<time>\d{6})")


@dataclass(frozen=True)
class CorpusImage:
    path: Path
    rel_path: str
    label: str
    width: int
    height: int
    file_size: int
    timestamp: datetime | None


def parse_filename_timestamp(path: Path) -> datetime | None:
    """Parse camera-style YYYYMMDD_HHMMSS timestamps from filenames."""

    match = TIMESTAMP_RE.search(path.stem)
    if not match:
        return None
    raw = f"{match.group('date')}{match.group('time')}"
    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def discover_images(input_root: Path) -> list[CorpusImage]:
    if not input_root.exists():
        raise FileNotFoundError(f"input root does not exist: {input_root}")

    images: list[CorpusImage] = []
    for path in sorted(input_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative_parts = path.relative_to(input_root).parts
        if any(part.startswith((".", "_")) for part in relative_parts[:-1]):
            continue
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
        rel_path = path.relative_to(input_root).as_posix()
        label = path.parent.name
        images.append(
            CorpusImage(
                path=path,
                rel_path=rel_path,
                label=label,
                width=width,
                height=height,
                file_size=path.stat().st_size,
                timestamp=parse_filename_timestamp(path),
            )
        )
    return images


def assign_bursts(
    images: Sequence[CorpusImage],
    *,
    burst_gap_seconds: int,
) -> dict[str, dict[str, Any]]:
    """Group timestamp-near images into local burst IDs per label."""

    by_label: dict[str, list[CorpusImage]] = defaultdict(list)
    for image in images:
        if image.timestamp is not None:
            by_label[image.label].append(image)

    burst_by_path: dict[str, dict[str, Any]] = {}
    for label, label_images in by_label.items():
        current: list[CorpusImage] = []
        burst_index = 0
        previous: CorpusImage | None = None
        for image in sorted(label_images, key=lambda item: (item.timestamp, item.rel_path)):
            assert image.timestamp is not None
            if (
                previous is None
                or previous.timestamp is None
                or (image.timestamp - previous.timestamp).total_seconds() <= burst_gap_seconds
            ):
                current.append(image)
            else:
                burst_index += 1
                _record_burst(label, burst_index, current, burst_by_path)
                current = [image]
            previous = image
        if current:
            burst_index += 1
            _record_burst(label, burst_index, current, burst_by_path)
    return burst_by_path


def _record_burst(
    label: str,
    burst_index: int,
    images: Sequence[CorpusImage],
    burst_by_path: dict[str, dict[str, Any]],
) -> None:
    burst_id = f"{label}-{burst_index:04d}"
    for image in images:
        burst_by_path[image.rel_path] = {
            "burstId": burst_id,
            "burstSize": len(images),
        }


def visual_features(path: Path) -> dict[str, Any]:
    """Compute cheap, transparent pixel features for corpus triage."""

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        preview = image.copy()
        preview.thumbnail((512, 512), Image.Resampling.BILINEAR)

    arr = np.asarray(preview).astype(np.float32) / 255.0
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    max_channel = np.max(arr, axis=2)
    min_channel = np.min(arr, axis=2)
    channel_spread = max_channel - min_channel
    saturation = np.divide(
        channel_spread,
        max_channel,
        out=np.zeros_like(max_channel),
        where=max_channel > 0.001,
    )

    cube_red = (red > 0.42) & (red > green * 1.22) & (red > blue * 1.22)
    cube_blue = (blue > 0.32) & (blue > red * 1.12) & (blue > green * 1.05)
    cube_green = (green > 0.34) & (green > red * 1.04) & (green > blue * 1.10)
    cube_yellow_or_orange = (red > 0.44) & (green > 0.26) & (blue < 0.36) & (red < green * 2.35)
    colored_mask = (
        (cube_red | cube_blue | cube_green | cube_yellow_or_orange)
        & (saturation > 0.36)
        & (channel_spread > 0.16)
        & (max_channel > 0.18)
    )
    color_bbox = _mask_bbox(colored_mask)
    color_pixel_fraction = float(np.mean(colored_mask))

    skin_mask = (
        (red > 0.35)
        & (green > 0.20)
        & (blue > 0.12)
        & (red > green * 1.04)
        & (red > blue * 1.18)
        & ((red - blue) > 0.08)
        & (saturation > 0.13)
        & (saturation < 0.75)
    )
    skin_pixel_fraction = float(np.mean(skin_mask))
    skin_near_color_fraction = _skin_near_bbox_fraction(skin_mask, color_bbox)

    if color_bbox is None:
        bbox_width = 0.0
        bbox_height = 0.0
        bbox_area = 0.0
        bbox_center_y = 0.0
        edge_touch = False
        color_bbox_payload = None
    else:
        x0, y0, x1, y1 = color_bbox
        bbox_width = x1 - x0
        bbox_height = y1 - y0
        bbox_area = bbox_width * bbox_height
        bbox_center_y = (y0 + y1) / 2.0
        edge_touch = x0 <= 0.025 or y0 <= 0.025 or x1 >= 0.975 or y1 >= 0.975
        color_bbox_payload = {
            "x0": round(x0, 4),
            "y0": round(y0, 4),
            "x1": round(x1, 4),
            "y1": round(y1, 4),
        }

    return {
        "orientation": "landscape" if width >= height else "portrait",
        "colorPixelFraction": round(color_pixel_fraction, 6),
        "skinPixelFraction": round(skin_pixel_fraction, 6),
        "skinNearColorFraction": round(skin_near_color_fraction, 6),
        "colorBBox": color_bbox_payload,
        "colorBBoxAreaFraction": round(bbox_area, 6),
        "colorBBoxWidthFraction": round(bbox_width, 6),
        "colorBBoxHeightFraction": round(bbox_height, 6),
        "colorBBoxCenterY": round(bbox_center_y, 6),
        "colorBBoxTouchesFrame": edge_touch,
    }


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    height, width = mask.shape
    x0 = float(xs.min()) / float(width)
    y0 = float(ys.min()) / float(height)
    x1 = float(xs.max() + 1) / float(width)
    y1 = float(ys.max() + 1) / float(height)
    return x0, y0, x1, y1


def _skin_near_bbox_fraction(
    skin_mask: np.ndarray,
    color_bbox: tuple[float, float, float, float] | None,
) -> float:
    if color_bbox is None:
        return float(np.mean(skin_mask))
    height, width = skin_mask.shape
    x0, y0, x1, y1 = color_bbox
    pad_x = 0.08
    pad_y = 0.08
    ix0 = max(0, int((x0 - pad_x) * width))
    iy0 = max(0, int((y0 - pad_y) * height))
    ix1 = min(width, int((x1 + pad_x) * width))
    iy1 = min(height, int((y1 + pad_y) * height))
    region = skin_mask[iy0:iy1, ix0:ix1]
    if region.size == 0:
        return 0.0
    return float(np.mean(region))


def summarize_analyzer(path: Path) -> dict[str, Any]:
    from rubik_recognizer.image_pipeline import analyze_image

    analysis = analyze_image(path.read_bytes())
    summary = analysis.summary()
    grids = summary.get("grids", [])
    return {
        "stickers": summary.get("stickers", 0),
        "gridCount": len(grids),
        "gridCenterFaces": [grid.get("centerFace") for grid in grids],
        "warnings": summary.get("warnings", []),
    }


def choose_analyzer_sample(
    images: Sequence[CorpusImage],
    *,
    limit_per_label: int,
    seed: int,
) -> set[str]:
    if limit_per_label <= 0:
        return set()

    rng = random.Random(seed)
    selected: set[str] = set()
    by_label: dict[str, list[CorpusImage]] = defaultdict(list)
    for image in images:
        by_label[image.label].append(image)
    for label_images in by_label.values():
        pool = sorted(label_images, key=lambda image: image.rel_path)
        if len(pool) <= limit_per_label:
            selected.update(image.rel_path for image in pool)
        else:
            selected.update(image.rel_path for image in rng.sample(pool, limit_per_label))
    return selected


def score_buckets(
    *,
    features: dict[str, Any],
    analyzer: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, list[str]], list[str]]:
    grid_count = int(analyzer.get("gridCount", 0)) if analyzer else None
    warnings = analyzer.get("warnings", []) if analyzer else []
    color_area = float(features["colorBBoxAreaFraction"])
    color_width = float(features["colorBBoxWidthFraction"])
    color_height = float(features["colorBBoxHeightFraction"])
    color_fraction = float(features["colorPixelFraction"])
    skin_fraction = float(features["skinPixelFraction"])
    skin_near = float(features["skinNearColorFraction"])
    touches_frame = bool(features["colorBBoxTouchesFrame"])
    center_y = float(features["colorBBoxCenterY"])
    is_landscape = features["orientation"] == "landscape"

    scores: dict[str, float] = {bucket: 0.0 for bucket in BUCKETS}
    reasons: dict[str, list[str]] = {bucket: [] for bucket in BUCKETS}

    if grid_count is not None and grid_count >= 3:
        scores["three_face_candidate"] = 1.0
        reasons["three_face_candidate"].append(f"analyzer fit {grid_count} face grids")
    elif color_fraction >= 0.002 and 0.008 <= color_area <= 0.42 and not touches_frame:
        scores["three_face_candidate"] = 0.62
        reasons["three_face_candidate"].append("color component is framed and cube-sized")
        if color_width >= 0.18 and color_height >= 0.12:
            scores["three_face_candidate"] += 0.1
            reasons["three_face_candidate"].append("color component has enough span for multi-face review")
    elif color_fraction >= 0.002:
        scores["three_face_candidate"] = 0.35
        reasons["three_face_candidate"].append("cube-like color exists but framing is weak")

    if grid_count is not None and grid_count <= 1 and color_area >= 0.035 and not touches_frame:
        scores["single_face"] = 0.72
        reasons["single_face"].append("analyzer found at most one face grid")
    elif 0.08 <= color_area <= 0.55 and abs(color_width - color_height) <= 0.22 and not touches_frame:
        scores["single_face"] = 0.58
        reasons["single_face"].append("color component is large and roughly square")

    hand_frame_context = touches_frame or color_area >= 0.30 or max(color_width, color_height) >= 0.70
    color_supports_hand = color_fraction >= 0.08 or (color_fraction >= 0.04 and hand_frame_context)
    localized_skin = (
        color_supports_hand
        and skin_near >= max(0.12, skin_fraction + 0.055)
        and skin_fraction <= 0.35
    )
    compact_hand_signal = color_supports_hand and 0.035 <= skin_fraction <= 0.20 and skin_near >= 0.095
    if localized_skin or compact_hand_signal:
        hand_score = min(1.0, max(skin_near * 4.0, skin_fraction * 5.0))
        if not hand_frame_context:
            hand_score *= 0.72
        scores["hand_occluded"] = hand_score
        reasons["hand_occluded"].append("skin-tone pixels overlap or surround cube-color area")

    if touches_frame or color_area >= 0.38 or max(color_width, color_height) >= 0.78:
        scores["cropped_close"] = 0.82
        if touches_frame:
            reasons["cropped_close"].append("cube-color component touches frame edge")
        if color_area >= 0.38:
            reasons["cropped_close"].append("cube-color component occupies a large frame area")
        if max(color_width, color_height) >= 0.78:
            reasons["cropped_close"].append("cube-color component spans most of one axis")

    if (
        color_fraction >= 0.002
        and 0.010 <= color_area <= 0.36
        and center_y >= 0.38
        and not touches_frame
        and skin_near < 0.075
    ):
        scores["table_isometric"] = 0.58
        reasons["table_isometric"].append("framed cube-color area sits in lower half without hand signal")
        if is_landscape:
            scores["table_isometric"] += 0.12
            reasons["table_isometric"].append("landscape table-photo framing")

    retake_score = 0.0
    if color_fraction < 0.001:
        retake_score = max(retake_score, 0.78)
        reasons["retake_negative"].append("almost no saturated cube-color pixels")
    if grid_count is not None and grid_count < 3:
        retake_score = max(retake_score, 0.70)
        reasons["retake_negative"].append("analyzer did not fit three visible grids")
    for bucket in ("single_face", "hand_occluded", "cropped_close"):
        if scores[bucket] >= 0.58:
            retake_score = max(retake_score, scores[bucket] * 0.92)
            reasons["retake_negative"].append(f"{bucket} score is high")
    if warnings:
        retake_score = max(retake_score, 0.64)
        reasons["retake_negative"].append("analyzer emitted warnings")
    if scores["three_face_candidate"] >= 0.85 and retake_score < 0.80:
        retake_score *= 0.45
    scores["retake_negative"] = min(1.0, retake_score)

    selected = [
        bucket
        for bucket in BUCKETS
        if scores[bucket] >= (0.58 if bucket != "retake_negative" else 0.60)
    ]
    if not selected:
        selected = ["retake_negative"]
        scores["retake_negative"] = max(scores["retake_negative"], 0.60)
        reasons["retake_negative"].append("no positive triage bucket cleared threshold")

    return (
        {bucket: round(score, 4) for bucket, score in scores.items()},
        reasons,
        selected,
    )


def build_manifest(
    input_root: Path,
    *,
    output_dir: Path,
    analyze_limit_per_label: int,
    burst_gap_seconds: int,
    sheet_size: int,
    seed: int,
    verbose: bool = False,
) -> dict[str, Any]:
    images = discover_images(input_root)
    bursts = assign_bursts(images, burst_gap_seconds=burst_gap_seconds)
    analyzer_sample = choose_analyzer_sample(
        images,
        limit_per_label=analyze_limit_per_label,
        seed=seed,
    )

    entries: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    analyzed_count = 0
    analyzer_three_grid_count = 0

    for index, image in enumerate(images, start=1):
        if verbose:
            print(f"[{index}/{len(images)}] triage {image.rel_path}")
        features = visual_features(image.path)
        analyzer: dict[str, Any] | None = None
        if image.rel_path in analyzer_sample:
            try:
                analyzer = summarize_analyzer(image.path)
                analyzed_count += 1
                if int(analyzer.get("gridCount", 0)) >= 3:
                    analyzer_three_grid_count += 1
            except Exception as exc:  # pragma: no cover - defensive local-corpus path
                analyzer = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "gridCount": 0,
                    "warnings": ["analyzer failed"],
                }
                analyzed_count += 1

        scores, reasons, selected_buckets = score_buckets(features=features, analyzer=analyzer)
        for bucket in selected_buckets:
            bucket_counts[bucket] += 1
        labels[image.label] += 1
        dimensions[f"{image.width}x{image.height}"] += 1

        burst = bursts.get(image.rel_path, {"burstId": None, "burstSize": 0})
        entry = {
            "relativePath": image.rel_path,
            "label": image.label,
            "width": image.width,
            "height": image.height,
            "fileSize": image.file_size,
            "timestamp": image.timestamp.isoformat() if image.timestamp else None,
            "burstId": burst["burstId"],
            "burstSize": burst["burstSize"],
            "features": features,
            "analyzer": analyzer,
            "bucketScores": scores,
            "bucketReasons": {
                bucket: bucket_reasons
                for bucket, bucket_reasons in reasons.items()
                if bucket_reasons
            },
            "buckets": selected_buckets,
        }
        entries.append(entry)

    contact_sheet_paths = write_contact_sheets(
        input_root=input_root,
        output_dir=output_dir / "contact_sheets",
        entries=entries,
        sheet_size=sheet_size,
    )

    return {
        "schema": "ctvd.kaggleCubeCorpusTriage.v1",
        "sourceRoot": str(input_root),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "analyzeLimitPerLabel": analyze_limit_per_label,
            "burstGapSeconds": burst_gap_seconds,
            "sheetSize": sheet_size,
            "seed": seed,
        },
        "summary": {
            "totalImages": len(images),
            "labels": dict(sorted(labels.items())),
            "dimensions": dict(sorted(dimensions.items())),
            "bucketCounts": {bucket: bucket_counts.get(bucket, 0) for bucket in BUCKETS},
            "analyzedImages": analyzed_count,
            "analyzerThreeGridImages": analyzer_three_grid_count,
        },
        "contactSheets": contact_sheet_paths,
        "images": entries,
    }


def write_contact_sheets(
    *,
    input_root: Path,
    output_dir: Path,
    entries: Sequence[dict[str, Any]],
    sheet_size: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheets: dict[str, str] = {}

    for bucket in BUCKETS:
        bucket_entries = [
            entry for entry in entries if bucket in entry["buckets"]
        ]
        bucket_entries = sorted(
            bucket_entries,
            key=lambda entry: (-float(entry["bucketScores"][bucket]), entry["relativePath"]),
        )[:sheet_size]
        if not bucket_entries:
            continue

        path = output_dir / f"{bucket}.jpg"
        make_contact_sheet(
            input_root=input_root,
            entries=bucket_entries,
            bucket=bucket,
            output_path=path,
        )
        contact_sheets[bucket] = path.relative_to(output_dir.parent).as_posix()
    return contact_sheets


def make_contact_sheet(
    *,
    input_root: Path,
    entries: Sequence[dict[str, Any]],
    bucket: str,
    output_path: Path,
    thumb_size: tuple[int, int] = (220, 165),
    columns: int = 5,
) -> None:
    label_height = 46
    padding = 12
    title_height = 38
    rows = (len(entries) + columns - 1) // columns
    width = columns * (thumb_size[0] + padding) + padding
    height = title_height + rows * (thumb_size[1] + label_height + padding) + padding
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((padding, 12), f"{bucket} ({len(entries)} examples)", fill=(0, 0, 0), font=font)

    for index, entry in enumerate(entries):
        row = index // columns
        col = index % columns
        x = padding + col * (thumb_size[0] + padding)
        y = title_height + row * (thumb_size[1] + label_height + padding)
        image_path = input_root / entry["relativePath"]
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            tile = Image.new("RGB", thumb_size, (242, 242, 242))
            tile_x = (thumb_size[0] - image.width) // 2
            tile_y = (thumb_size[1] - image.height) // 2
            tile.paste(image, (tile_x, tile_y))
        sheet.paste(tile, (x, y))
        score = entry["bucketScores"][bucket]
        rel = entry["relativePath"]
        label = f"{score:.2f} {rel}"
        for line_index, line in enumerate(_wrap_label(label, 34)[:3]):
            draw.text(
                (x, y + thumb_size[1] + 4 + line_index * 13),
                line,
                fill=(0, 0, 0),
                font=font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def _wrap_label(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for part in text.split("/"):
        candidate = part if not current else f"{current}/{part}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)
    if not lines:
        return [text[:max_chars]]
    return lines


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Corpus root with solved/unsolved subfolders. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--analyze-limit-per-label",
        type=int,
        default=20,
        help="Run the CTVD image analyzer on this many deterministic samples per label. Use 0 for pixel-only triage.",
    )
    parser.add_argument(
        "--burst-gap-seconds",
        type=int,
        default=12,
        help="Timestamp gap used to group adjacent photos into local bursts.",
    )
    parser.add_argument(
        "--sheet-size",
        type=int,
        default=30,
        help="Maximum examples per bucket contact sheet.",
    )
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(
        args.input_root,
        output_dir=args.output_dir,
        analyze_limit_per_label=args.analyze_limit_per_label,
        burst_gap_seconds=args.burst_gap_seconds,
        sheet_size=args.sheet_size,
        seed=args.seed,
        verbose=args.verbose,
    )
    manifest_path = args.output_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    summary = manifest["summary"]
    print(f"wrote {manifest_path}")
    print(f"images: {summary['totalImages']} labels={summary['labels']}")
    print(f"buckets: {summary['bucketCounts']}")
    print(f"contact sheets: {manifest['contactSheets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
