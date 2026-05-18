#!/usr/bin/env python3
"""Extract per-sticker (RGB, ground_truth_color) training samples for
color-classifier training.

Pipeline per image:
  1. Load image bytes, run ``analyze_image`` (which produces detected
     stickers via the existing component pipeline at max-1150px scale).
  2. If a hull label exists for this (setId, side), scale the labelled
     face quads to the analysis resolution and assign each detected
     sticker to a face via point-in-polygon. Otherwise fall back to
     grouping by the recognizer's grid output.
  3. For each face: pair the detected stickers' RGBs with the 9
     ground-truth colors for that face via greedy multiset assignment
     (cost = current classifier's Lab distance per color). The multiset
     constraint guarantees the per-face color counts match ground truth
     and resolves ambiguous (red/orange) samples by capacity.

Output: JSONL at runs/color_samples_v0.jsonl, one line per sample.

Coverage: walks both the corpus manifest (15 pairs with locked ground
truth) and the broader set of hull-labelled (setId, side) pairs that have
a matching ground-truth file in /Users/jhuber/Downloads.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import (  # noqa: E402
    FACE_TO_COLOR,
    classify_rgb,
)
from rubik_recognizer.image_pipeline import FaceGrid, ImageAnalysis, Sticker, analyze_image  # noqa: E402
from rubik_recognizer.validation import FACE_ORDER  # noqa: E402
from tools.inspect_cube_isolation import point_in_polygon  # noqa: E402


CORPUS_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
# runs/ is gitignored, so worktrees don't have hull labels. Fall back to the
# primary checkout's runs/labels if the worktree-local dir is missing.
_WORKTREE_LABEL_DIR = REPO_ROOT / "runs" / "labels"
_PRIMARY_LABEL_DIR = Path("/Users/jhuber/cube-two-view-debugger/runs/labels")
HULL_LABEL_DIR = _WORKTREE_LABEL_DIR if _WORKTREE_LABEL_DIR.is_dir() else _PRIMARY_LABEL_DIR
DOWNLOADS = Path("/Users/jhuber/Downloads")
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "color_samples_v0.jsonl"

EXPECTED_FACES_BY_SIDE = {
    "A": ("U", "R", "F"),
    "B": ("D", "L", "B"),
}


@dataclass
class PairTask:
    set_id: str
    image_a: Path
    image_b: Path
    ground_truth: Path
    source: str  # "corpus" or "discovered"


# ---------- ground-truth + manifest helpers ----------


def parse_ground_truth(path: Path) -> str:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        data = data[0]
    state = data.get("corrected")
    if not isinstance(state, str) or len(state) != 54:
        raise ValueError(f"no 54-char 'corrected' state at {path}")
    return state


def face_colors_from_state(state: str, face: str) -> List[str]:
    idx = FACE_ORDER.index(face)
    chunk = state[idx * 9:(idx + 1) * 9]
    return [FACE_TO_COLOR[ch] for ch in chunk]


def load_corpus_tasks(manifest_path: Path) -> List[PairTask]:
    manifest = json.loads(manifest_path.read_text())
    tasks = []
    for pair in manifest.get("pairs", []):
        tasks.append(PairTask(
            set_id=str(pair["setId"]),
            image_a=Path(pair["imageAPath"]),
            image_b=Path(pair["imageBPath"]),
            ground_truth=Path(pair["groundTruthPath"]),
            source="corpus",
        ))
    return tasks


def discover_additional_tasks(corpus_set_ids: Iterable[str]) -> List[PairTask]:
    """Find hull-labelled (set, A+B) pairs that have matching ground truth
    files in Downloads but aren't already in the corpus manifest.

    Returns [] when /Users/jhuber/Downloads does not exist (clean CI/VM
    environments, contributors without the local image assets). Tooling
    that depends on this function should degrade gracefully rather than
    crash, per Devin PR-#127 portability review."""
    if not DOWNLOADS.is_dir():
        return []

    seen = set(corpus_set_ids)

    # Build setId → ground truth file map (pick the most recent if multiple)
    gt_pattern = re.compile(r"Set (\d+)(?:\s*v(\d+))?.*cube-ground-truth-(\d+)\.json")
    by_set: Dict[str, Tuple[int, int, Path]] = {}
    for path in DOWNLOADS.glob("Set *cube-ground-truth-*.json"):
        m = gt_pattern.search(path.name)
        if not m:
            continue
        set_id = m.group(1)
        version = int(m.group(2) or "1")
        timestamp = int(m.group(3))
        key = (version, timestamp)
        prev = by_set.get(set_id)
        if prev is None or key > prev[:2]:
            by_set[set_id] = (version, timestamp, path)

    # Find image A/B per setId via filename pattern (require both to exist).
    # Accept both `white up` and `white-up` — newer iPhone-export naming
    # (Sets 57/58/61/62 onwards) uses the hyphen.
    img_pattern = re.compile(r"Set (\d+) - ([AB]) - white[- ]up[^.]*\.(?:JPG|jpg|jpeg|PNG|png)")
    images_by_set: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for path in DOWNLOADS.iterdir():
        m = img_pattern.match(path.name)
        if not m:
            continue
        images_by_set[m.group(1)][m.group(2)] = path

    # Find which sets have hull labels
    label_sets: set[str] = set()
    label_pattern = re.compile(r"set-(\d+)-[ab]-")
    for path in HULL_LABEL_DIR.glob("*-geometry-label.json"):
        m = label_pattern.search(path.name)
        if m:
            label_sets.add(m.group(1))

    tasks: List[PairTask] = []
    for set_id in sorted(label_sets, key=lambda s: int(s)):
        if set_id in seen:
            continue
        if set_id not in by_set:
            continue
        imgs = images_by_set.get(set_id, {})
        if "A" not in imgs or "B" not in imgs:
            continue
        tasks.append(PairTask(
            set_id=set_id,
            image_a=imgs["A"],
            image_b=imgs["B"],
            ground_truth=by_set[set_id][2],
            source="discovered",
        ))
    return tasks


# ---------- hull-label loading ----------


def latest_hull_label(set_id: str, side: str) -> Optional[Path]:
    """Return the most recently saved hull label JSON for this (set, side)."""
    pattern = f"*-set-{set_id}-{side.lower()}-geometry-label.json"
    matches = sorted(HULL_LABEL_DIR.glob(pattern))
    return matches[-1] if matches else None


def load_hull_label(path: Path) -> Dict:
    return json.loads(path.read_text())


def scaled_face_quads(
    document: Dict,
    processing_width: int,
    processing_height: int,
) -> Dict[str, List[Tuple[float, float]]]:
    image = document.get("image") or {}
    natural_w = float(image.get("width") or processing_width)
    natural_h = float(image.get("height") or processing_height)
    scale_x = processing_width / max(1.0, natural_w)
    scale_y = processing_height / max(1.0, natural_h)
    raw = (document.get("labels") or {}).get("faceQuads") or {}
    out: Dict[str, List[Tuple[float, float]]] = {}
    for face, points in raw.items():
        if not isinstance(points, list) or len(points) != 4:
            continue
        out[face] = [(float(p["x"]) * scale_x, float(p["y"]) * scale_y) for p in points]
    return out


# ---------- sticker → face assignment ----------


def stickers_by_face_via_hull(
    stickers: Sequence[Sticker],
    face_quads: Dict[str, List[Tuple[float, float]]],
    expected_faces: Iterable[str],
) -> Dict[str, List[Sticker]]:
    out: Dict[str, List[Sticker]] = {f: [] for f in expected_faces if f in face_quads}
    for s in stickers:
        center = (float(s.center[0]), float(s.center[1]))
        for face in out.keys():
            if point_in_polygon(center, face_quads[face]):
                out[face].append(s)
                break  # disjoint face quads (normally)
    return out


def stickers_by_face_via_grids(
    grids: Sequence[FaceGrid],
    expected_faces: Iterable[str],
) -> Dict[str, List[Sticker]]:
    """Fallback when no hull label: pick the best-matched grid per face."""
    by_face: Dict[str, List[FaceGrid]] = defaultdict(list)
    for grid in grids:
        if grid.center_face in expected_faces:
            by_face[grid.center_face].append(grid)
    out: Dict[str, List[Sticker]] = {}
    for face, candidates in by_face.items():
        # Highest matched_count wins; tie-break by lowest fit_error.
        best = min(candidates, key=lambda g: (-g.matched_count, g.fit_error))
        stickers: List[Sticker] = []
        for row in best.stickers:
            stickers.extend(row)
        out[face] = stickers
    return out


# ---------- multiset assignment ----------


def lab_distance_to_color(rgb: Tuple[int, int, int], color: str) -> float:
    match = classify_rgb(rgb)
    for c, d in match.alternatives:
        if c == color:
            return d
    return float("inf")


def assign_multiset(
    rgbs: Sequence[Tuple[int, int, int]],
    gt_colors: Sequence[str],
) -> List[Optional[str]]:
    """Greedy multiset assignment. Each RGB gets one color, respecting
    per-color capacity. If len(rgbs) < len(gt_colors), some colors remain
    unused (assignments are still per-RGB). If len(rgbs) > capacity for a
    color, leftover RGBs are reassigned to the next-best available color."""
    capacity = Counter(gt_colors)
    n = len(rgbs)
    assignments: List[Optional[str]] = [None] * n

    edges: List[Tuple[float, int, str]] = []
    for i, rgb in enumerate(rgbs):
        for color in capacity:
            edges.append((lab_distance_to_color(rgb, color), i, color))
    edges.sort()

    for _cost, i, color in edges:
        if assignments[i] is not None:
            continue
        if capacity[color] <= 0:
            continue
        assignments[i] = color
        capacity[color] -= 1

    return assignments


# ---------- per-pair extraction ----------


def samples_from_image(
    image_path: Path,
    set_id: str,
    side: str,
    gt_state: str,
) -> Tuple[List[Dict], str]:
    """Returns (samples, source_tag). source_tag in {'hull', 'grid', 'none'}."""
    try:
        image_bytes = image_path.read_bytes()
    except FileNotFoundError:
        print(f"  set {set_id} side {side}: image not found at {image_path}", file=sys.stderr)
        return [], "none"

    analysis: ImageAnalysis = analyze_image(image_bytes)
    expected = EXPECTED_FACES_BY_SIDE[side]

    hull_path = latest_hull_label(set_id, side)
    source_tag: str
    if hull_path is not None:
        document = load_hull_label(hull_path)
        face_quads = scaled_face_quads(document, analysis.width, analysis.height)
        # analysis.width/height are the ORIGINAL image dims (before resize). The
        # detected sticker centers are also in the processing-resolution space
        # (= original / scale). We need quads in the same space.
        # analyze_image's stickers' .center are in resized-image coords (max
        # side = 1150). Let's compute the processing resolution explicitly.
        # The Sticker centers come from the resized array; analysis.width/height
        # are the ORIGINAL image size. We need to rescale quads to processing
        # resolution: processing_max = min(1150, max(W, H)).
        proc_max = 1150
        natural_max = max(analysis.width, analysis.height)
        if natural_max <= proc_max:
            proc_w, proc_h = analysis.width, analysis.height
        else:
            scale = proc_max / float(natural_max)
            proc_w = int(analysis.width * scale)
            proc_h = int(analysis.height * scale)
        face_quads = scaled_face_quads(document, proc_w, proc_h)

        by_face = stickers_by_face_via_hull(analysis.stickers, face_quads, expected)
        source_tag = "hull"
    else:
        by_face = stickers_by_face_via_grids(analysis.grids, expected)
        source_tag = "grid"

    samples: List[Dict] = []
    for face, stickers in by_face.items():
        if not stickers:
            continue
        gt_colors = face_colors_from_state(gt_state, face)
        rgbs = [tuple(int(v) for v in s.rgb) for s in stickers]
        classified = [s.match.color for s in stickers]
        confidences = [float(s.match.confidence) for s in stickers]
        assignments = assign_multiset(rgbs, gt_colors)

        # Only keep samples where assignment succeeded.
        for sticker, rgb, cls, gt, conf in zip(stickers, rgbs, classified, assignments, confidences):
            if gt is None:
                continue
            samples.append({
                "setId": set_id,
                "side": side,
                "face": face,
                "stickerSource": sticker.source,
                "center": [round(float(sticker.center[0]), 2), round(float(sticker.center[1]), 2)],
                "rgb": list(rgb),
                "currentColor": cls,
                "currentConfidence": round(conf, 4),
                "gtColor": gt,
                "assignmentSource": source_tag,
            })
    return samples, source_tag


def samples_from_pair(task: PairTask) -> List[Dict]:
    try:
        gt_state = parse_ground_truth(task.ground_truth)
    except Exception as e:
        print(f"  set {task.set_id}: ground truth parse failed ({e})", file=sys.stderr)
        return []
    out: List[Dict] = []
    for side, image_path in (("A", task.image_a), ("B", task.image_b)):
        samples, src = samples_from_image(image_path, task.set_id, side, gt_state)
        out.extend(samples)
        print(f"  set {task.set_id} {side}: +{len(samples)} samples (source={src})", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(CORPUS_MANIFEST))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument(
        "--include-discovered",
        action="store_true",
        help="Also include hull-labelled sets outside the corpus manifest when they have a matching ground truth in Downloads.",
    )
    args = ap.parse_args()

    tasks = load_corpus_tasks(Path(args.manifest))
    if args.include_discovered:
        extra = discover_additional_tasks({t.set_id for t in tasks})
        print(f"discovered {len(extra)} additional hull-labelled sets with ground truth", file=sys.stderr)
        tasks.extend(extra)

    all_samples: List[Dict] = []
    for task in tasks:
        print(f"set {task.set_id} ({task.source}):", file=sys.stderr)
        all_samples.extend(samples_from_pair(task))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    print("", file=sys.stderr)
    print(f"wrote {len(all_samples)} samples to {output_path}", file=sys.stderr)

    # Summary
    by_gt = Counter(s["gtColor"] for s in all_samples)
    by_src = Counter(s["assignmentSource"] for s in all_samples)
    by_sticker_src = Counter(s["stickerSource"] for s in all_samples)
    print(f"per-color (ground truth): {dict(by_gt)}", file=sys.stderr)
    print(f"face-assignment source:   {dict(by_src)}", file=sys.stderr)
    print(f"sticker source:           {dict(by_sticker_src)}", file=sys.stderr)

    # Current classifier agreement with multiset-assigned ground truth.
    correct = sum(1 for s in all_samples if s["currentColor"] == s["gtColor"])
    if all_samples:
        print(
            f"\ncurrent classify_rgb agrees with multiset-assigned ground truth: "
            f"{correct}/{len(all_samples)} = {correct / len(all_samples):.1%}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
