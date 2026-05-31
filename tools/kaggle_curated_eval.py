#!/usr/bin/env python3
"""Build and analyze a local curated eval set from Kaggle cube triage output.

This is the second layer after ``triage_kaggle_cube_corpus.py``:

1. ``build-manifest`` creates a small, hand-editable local manifest from the
   triage manifest. It records relative paths, category labels, notes, and the
   triage evidence that caused each image to be selected. It does not copy or
   commit images.
2. ``analyze`` runs the current CTVD image analyzer over that curated manifest
   and writes JSON + Markdown reports that separate sticker detection, grid
   fitting, and warning-heavy cases.

Outputs default under the local corpus' ``_curated_eval`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.triage_kaggle_cube_corpus import (  # noqa: E402
    DEFAULT_INPUT_ROOT as TRIAGE_DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_DIR as TRIAGE_DEFAULT_OUTPUT_DIR,
    summarize_analyzer,
)


DEFAULT_CORPUS_ROOT = Path(os.environ.get("CTVD_KAGGLE_CORPUS", str(TRIAGE_DEFAULT_INPUT_ROOT)))
DEFAULT_TRIAGE_MANIFEST = Path(
    os.environ.get("CTVD_KAGGLE_TRIAGE_MANIFEST", str(TRIAGE_DEFAULT_OUTPUT_DIR / "manifest.json"))
)
DEFAULT_OUTPUT_DIR = Path(os.environ.get("CTVD_KAGGLE_CURATED_OUTPUT", str(ROOT / "runs" / "kaggle_curated_eval")))

CATEGORY_QUOTAS = {
    "usable_three_face": 10,
    "usable_table_isometric": 8,
    "single_face_negative": 8,
    "hand_occluded_negative": 8,
    "cropped_close_negative": 8,
    "noncanonical_interesting": 6,
}

CATEGORY_DESCRIPTIONS = {
    "usable_three_face": "Likely useful positive detector eval image with three visible faces.",
    "usable_table_isometric": "Table-top isometric-ish image worth testing detector generalization.",
    "single_face_negative": "Retake-negative, single-face-biased photo useful for rejection handling.",
    "hand_occluded_negative": "Hand-held or finger-occluded photo; useful retake/rejection negative.",
    "cropped_close_negative": "Cube is too close to the frame edge; useful framing negative.",
    "noncanonical_interesting": "Noncanonical but visually interesting image for exploratory review.",
}

NEGATIVE_CATEGORIES = {
    "single_face_negative",
    "hand_occluded_negative",
    "cropped_close_negative",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_curated_manifest(
    triage_manifest: Mapping[str, Any],
    *,
    triage_manifest_path: Path,
    output_dir: Path,
    quotas: Mapping[str, int] = CATEGORY_QUOTAS,
) -> dict[str, Any]:
    images = list(triage_manifest.get("images", []))
    used_paths: set[str] = set()
    selected: list[dict[str, Any]] = []

    for category, quota in quotas.items():
        candidates = _category_candidates(images, category, used_paths)
        picked = _pick_diverse(candidates, quota=quota)
        for entry in picked:
            used_paths.add(entry["relativePath"])
            selected.append(_curated_entry(entry, category, len(selected) + 1))

    summary_counts = Counter(entry["category"] for entry in selected)
    contact_sheet_paths = write_curated_contact_sheets(
        corpus_root=Path(str(triage_manifest.get("sourceRoot", DEFAULT_CORPUS_ROOT))),
        output_dir=output_dir / "contact_sheets",
        entries=selected,
    )

    return {
        "schema": "ctvd.kaggleCubeCuratedEval.v1",
        "sourceCorpusRoot": str(triage_manifest.get("sourceRoot", DEFAULT_CORPUS_ROOT)),
        "sourceTriageManifest": str(triage_manifest_path),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "totalImages": len(selected),
            "categories": {category: summary_counts.get(category, 0) for category in quotas},
            "contactSheets": contact_sheet_paths,
        },
        "curationGuidance": [
            "This is a deterministic starter manifest, not final ground truth.",
            "Edit category, tags, and notes in this JSON after manual review.",
            "Keep image files local; only relative paths are stored here.",
        ],
        "images": selected,
    }


def _category_candidates(
    entries: Sequence[Mapping[str, Any]],
    category: str,
    used_paths: set[str],
) -> list[Mapping[str, Any]]:
    candidates = [
        entry
        for entry in entries
        if entry.get("relativePath") not in used_paths and _matches_category(entry, category)
    ]
    return sorted(candidates, key=lambda entry: (-_category_score(entry, category), entry["relativePath"]))


def _matches_category(entry: Mapping[str, Any], category: str) -> bool:
    buckets = set(entry.get("buckets", []))
    label = entry.get("label")
    scores = entry.get("bucketScores", {})

    if category == "usable_three_face":
        return "three_face_candidate" in buckets and not (
            {"hand_occluded", "cropped_close", "retake_negative", "table_isometric"} & buckets
        )
    if category == "usable_table_isometric":
        return "table_isometric" in buckets and not (
            {"hand_occluded", "cropped_close", "retake_negative"} & buckets
        )
    if category == "single_face_negative":
        return "single_face" in buckets and "retake_negative" in buckets and not (
            {"hand_occluded", "cropped_close", "table_isometric"} & buckets
        )
    if category == "hand_occluded_negative":
        return "hand_occluded" in buckets
    if category == "cropped_close_negative":
        return "cropped_close" in buckets and "hand_occluded" not in buckets
    if category == "noncanonical_interesting":
        return label == "unsolved" and (
            "three_face_candidate" in buckets
            or "table_isometric" in buckets
            or float(scores.get("three_face_candidate", 0.0)) >= 0.35
        )
    raise ValueError(f"unknown category: {category}")


def _category_score(entry: Mapping[str, Any], category: str) -> float:
    scores = entry.get("bucketScores", {})
    buckets = set(entry.get("buckets", []))
    label_bonus = 0.06 if entry.get("label") == "unsolved" else 0.0

    if category == "usable_three_face":
        return float(scores.get("three_face_candidate", 0.0)) + label_bonus
    if category == "usable_table_isometric":
        return float(scores.get("table_isometric", 0.0)) + float(scores.get("three_face_candidate", 0.0)) * 0.1
    if category == "single_face_negative":
        return float(scores.get("single_face", 0.0)) + float(scores.get("retake_negative", 0.0)) * 0.1
    if category == "hand_occluded_negative":
        return float(scores.get("hand_occluded", 0.0)) + float(scores.get("retake_negative", 0.0)) * 0.1
    if category == "cropped_close_negative":
        return float(scores.get("cropped_close", 0.0)) + float(scores.get("retake_negative", 0.0)) * 0.1
    if category == "noncanonical_interesting":
        bucket_bonus = 0.08 if "table_isometric" in buckets else 0.0
        return float(scores.get("three_face_candidate", 0.0)) + bucket_bonus + label_bonus
    raise ValueError(f"unknown category: {category}")


def _pick_diverse(candidates: Sequence[Mapping[str, Any]], *, quota: int) -> list[Mapping[str, Any]]:
    if quota <= 0:
        return []

    picked: list[Mapping[str, Any]] = []
    used_bursts: set[str] = set()

    for entry in candidates:
        burst_id = str(entry.get("burstId") or entry.get("relativePath"))
        if burst_id in used_bursts:
            continue
        picked.append(entry)
        used_bursts.add(burst_id)
        if len(picked) >= quota:
            return picked

    picked_paths = {entry["relativePath"] for entry in picked}
    for entry in candidates:
        if entry["relativePath"] in picked_paths:
            continue
        picked.append(entry)
        picked_paths.add(entry["relativePath"])
        if len(picked) >= quota:
            break
    return picked


def _curated_entry(entry: Mapping[str, Any], category: str, index: int) -> dict[str, Any]:
    triage = {
        "buckets": entry.get("buckets", []),
        "bucketScores": entry.get("bucketScores", {}),
        "bucketReasons": entry.get("bucketReasons", {}),
        "burstId": entry.get("burstId"),
        "burstSize": entry.get("burstSize"),
        "features": entry.get("features", {}),
    }
    eval_tag = "retake_negative_eval" if category in NEGATIVE_CATEGORIES else "detector_positive_eval"
    tags = sorted(set([category, *entry.get("buckets", []), eval_tag]))

    return {
        "id": f"kaggle-{index:03d}",
        "relativePath": entry["relativePath"],
        "sourceLabel": entry.get("label"),
        "category": category,
        "categoryDescription": CATEGORY_DESCRIPTIONS[category],
        "tags": tags,
        "notes": "Starter selection from triage; manually review and edit before treating as ground truth.",
        "triage": triage,
    }


def write_curated_contact_sheets(
    *,
    corpus_root: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_category[str(entry["category"])].append(entry)

    for category in CATEGORY_QUOTAS:
        category_entries = by_category.get(category, [])
        if not category_entries:
            continue
        path = output_dir / f"{category}.jpg"
        _make_contact_sheet(corpus_root=corpus_root, entries=category_entries, title=category, output_path=path)
        paths[category] = path.relative_to(output_dir.parent).as_posix()

    overview = output_dir / "curated_overview.jpg"
    _make_contact_sheet(corpus_root=corpus_root, entries=entries, title="curated_overview", output_path=overview)
    paths["curated_overview"] = overview.relative_to(output_dir.parent).as_posix()
    return paths


def _make_contact_sheet(
    *,
    corpus_root: Path,
    entries: Sequence[Mapping[str, Any]],
    title: str,
    output_path: Path,
    thumb_size: tuple[int, int] = (220, 165),
    columns: int = 5,
) -> None:
    label_height = 52
    padding = 12
    title_height = 38
    rows = (len(entries) + columns - 1) // columns
    width = columns * (thumb_size[0] + padding) + padding
    height = title_height + rows * (thumb_size[1] + label_height + padding) + padding
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((padding, 12), f"{title} ({len(entries)} examples)", fill=(0, 0, 0), font=font)

    for index, entry in enumerate(entries):
        row = index // columns
        col = index % columns
        x = padding + col * (thumb_size[0] + padding)
        y = title_height + row * (thumb_size[1] + label_height + padding)
        image_path = corpus_root / entry["relativePath"]
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            tile = Image.new("RGB", thumb_size, (242, 242, 242))
            tile.paste(image, ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2))
        sheet.paste(tile, (x, y))
        label = f"{entry['id']} {entry['category']} {entry['relativePath']}"
        for line_index, line in enumerate(_wrap_label(label, 34)[:4]):
            draw.text((x, y + thumb_size[1] + 4 + line_index * 12), line, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def _wrap_label(text: str, max_chars: int) -> list[str]:
    words = text.replace("/", " / ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text[:max_chars]]


AnalyzerFn = Callable[[Path], Mapping[str, Any]]


def analyze_curated_manifest(
    curated_manifest: Mapping[str, Any],
    *,
    corpus_root: Path | None = None,
    analyzer_fn: AnalyzerFn | None = None,
) -> dict[str, Any]:
    root = corpus_root or Path(str(curated_manifest["sourceCorpusRoot"]))
    analyzer = analyzer_fn or summarize_analyzer
    rows: list[dict[str, Any]] = []

    for entry in curated_manifest.get("images", []):
        image_path = root / entry["relativePath"]
        try:
            result = dict(analyzer(image_path))
            failure_stage = _failure_stage(result)
        except Exception as exc:  # pragma: no cover - defensive local image path
            result = {
                "error": f"{type(exc).__name__}: {exc}",
                "stickers": 0,
                "gridCount": 0,
                "warnings": ["analyzer failed"],
            }
            failure_stage = "analyzer_error"
        rows.append(
            {
                "id": entry["id"],
                "relativePath": entry["relativePath"],
                "category": entry["category"],
                "sourceLabel": entry.get("sourceLabel"),
                "analyzer": result,
                "failureStage": failure_stage,
            }
        )

    return {
        "schema": "ctvd.kaggleCubeCuratedAnalysis.v1",
        "sourceManifestSchema": curated_manifest.get("schema"),
        "sourceCorpusRoot": str(root),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "summary": _analysis_summary(rows),
        "images": rows,
    }


def _failure_stage(analyzer: Mapping[str, Any]) -> str:
    if analyzer.get("error"):
        return "analyzer_error"
    stickers = int(analyzer.get("stickers", 0))
    grid_count = int(analyzer.get("gridCount", 0))
    warnings = list(analyzer.get("warnings", []))
    if grid_count >= 3:
        if stickers < 18 or warnings:
            return "warnings_with_three_grids"
        return "three_grids_clean"
    if stickers < 18:
        return "sticker_detection_low"
    return "grid_fit_low"


def _analysis_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    warning_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    sticker_counts: list[int] = []
    three_grid_count = 0

    for row in rows:
        by_category[str(row["category"])].append(row)
        analyzer = row["analyzer"]
        stickers = int(analyzer.get("stickers", 0))
        grid_count = int(analyzer.get("gridCount", 0))
        sticker_counts.append(stickers)
        if grid_count >= 3:
            three_grid_count += 1
        stage_counts[str(row["failureStage"])] += 1
        for warning in analyzer.get("warnings", []):
            warning_counts[str(warning)] += 1

    category_summary: dict[str, Any] = {}
    for category, category_rows in sorted(by_category.items()):
        stickers = [int(row["analyzer"].get("stickers", 0)) for row in category_rows]
        category_summary[category] = {
            "count": len(category_rows),
            "threeGridImages": sum(1 for row in category_rows if int(row["analyzer"].get("gridCount", 0)) >= 3),
            "warningImages": sum(1 for row in category_rows if row["analyzer"].get("warnings")),
            "medianStickers": _median(stickers),
            "failureStages": dict(Counter(str(row["failureStage"]) for row in category_rows)),
        }

    return {
        "totalImages": len(rows),
        "threeGridImages": three_grid_count,
        "warningImages": sum(1 for row in rows if row["analyzer"].get("warnings")),
        "medianStickers": _median(sticker_counts),
        "failureStages": dict(stage_counts),
        "warnings": dict(warning_counts),
        "categories": category_summary,
    }


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def markdown_report(analysis: Mapping[str, Any]) -> str:
    summary = analysis["summary"]
    lines = [
        "# Kaggle Curated Eval Analysis",
        "",
        f"Generated: `{analysis['generatedAt']}`",
        f"Source root: `{analysis['sourceCorpusRoot']}`",
        "",
        "## Summary",
        "",
        f"- Images: {summary['totalImages']}",
        f"- Three-grid images: {summary['threeGridImages']}",
        f"- Warning images: {summary['warningImages']}",
        f"- Median stickers: {summary['medianStickers']}",
        "",
        "## By Category",
        "",
        "| Category | Count | Three-grid | Warning images | Median stickers | Dominant failure |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for category, row in summary["categories"].items():
        dominant = _dominant(row["failureStages"])
        lines.append(
            f"| `{category}` | {row['count']} | {row['threeGridImages']} | "
            f"{row['warningImages']} | {row['medianStickers']} | `{dominant}` |"
        )
    lines.extend(["", "## Failure Stages", ""])
    for stage, count in sorted(summary["failureStages"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{stage}`: {count}")
    if summary["warnings"]:
        lines.extend(["", "## Analyzer Warnings", ""])
        for warning, count in sorted(summary["warnings"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {count}x {warning}")
    lines.extend(["", "## Rows", ""])
    for row in analysis["images"]:
        analyzer = row["analyzer"]
        lines.append(
            f"- `{row['id']}` `{row['category']}` `{row['failureStage']}` "
            f"stickers={analyzer.get('stickers', 0)} grids={analyzer.get('gridCount', 0)} "
            f"`{row['relativePath']}`"
        )
    return "\n".join(lines) + "\n"


def _dominant(counts: Mapping[str, int]) -> str:
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _parse_quota(values: Sequence[str]) -> dict[str, int]:
    quotas = dict(CATEGORY_QUOTAS)
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"quota must be category=count: {value}")
        category, raw_count = value.split("=", 1)
        if category not in CATEGORY_QUOTAS:
            raise argparse.ArgumentTypeError(f"unknown quota category: {category}")
        quotas[category] = int(raw_count)
    return quotas


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest", help="Build a starter curated manifest from triage output.")
    build.add_argument("--triage-manifest", type=Path, default=DEFAULT_TRIAGE_MANIFEST)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build.add_argument(
        "--quota",
        action="append",
        default=[],
        help="Override a category quota, e.g. usable_three_face=12. Repeatable.",
    )

    analyze = subparsers.add_parser("analyze", help="Run CTVD analyzer on a curated manifest.")
    analyze.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "curated_manifest.json")
    analyze.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "analysis")
    analyze.add_argument("--corpus-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build-manifest":
        triage = load_json(args.triage_manifest)
        quotas = _parse_quota(args.quota)
        manifest = build_curated_manifest(
            triage,
            triage_manifest_path=args.triage_manifest,
            output_dir=args.output_dir,
            quotas=quotas,
        )
        manifest_path = args.output_dir / "curated_manifest.json"
        write_json(manifest_path, manifest)
        print(f"wrote {manifest_path}")
        print(f"images: {manifest['summary']['totalImages']} categories={manifest['summary']['categories']}")
        print(f"contact sheets: {manifest['summary']['contactSheets']}")
        return 0

    if args.command == "analyze":
        manifest = load_json(args.manifest)
        analysis = analyze_curated_manifest(manifest, corpus_root=args.corpus_root)
        json_path = args.output_dir / "analysis.json"
        markdown_path = args.output_dir / "analysis.md"
        write_json(json_path, analysis)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown_report(analysis))
        print(f"wrote {json_path}")
        print(f"wrote {markdown_path}")
        print(f"summary: {analysis['summary']}")
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
