#!/usr/bin/env python3
"""End-to-end evaluator for the proposed mask-first recognizer pipeline.

Devin's PR-#137 follow-up ask: a single tool that runs the full
proposed pipeline per (setId, side) pair and reports a single
"would this beat the current recognizer" verdict. Composes:

  rembg → optimized hexagon → rectify → sample → classify → joint face-ID
  → assemble 54-char state → compare to GT

Plus, for the same pair, runs the EXISTING recognizer
(`WhiteUpRecognizer.recognize`) and compares its state to GT.

Per (setId) outputs:
  * mask-path 54-char state + per-sticker accuracy vs GT
  * existing-recognizer 54-char state + per-sticker accuracy vs GT
  * verdict: mask-path WINS / DRAW / LOSES vs existing on per-sticker
  * confidence indicators: orientation-match per face, joint-multiset score

Aggregate output: per-pair table, win/draw/loss counts, where the
mask-path is currently strong/weak, what's blocking it from being
ready for Codex's `recognizer_mask.py` env switch.

Tooling-only. NO production-recognizer changes.

Progress: prints one line per (setId) to stderr in real time, so
long sweeps are queryable mid-run (per the lesson from PR #137).

Usage:
  .venv/bin/python tools/evaluate_mask_pipeline.py
  .venv/bin/python tools/evaluate_mask_pipeline.py --only-sets 30 46 57 58
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import COLOR_TO_FACE  # noqa: E402
from rubik_recognizer.recognizer import WhiteUpRecognizer  # noqa: E402
from rubik_recognizer.validation import FACE_ORDER  # noqa: E402
from tools.extract_color_samples import (  # noqa: E402
    discover_additional_tasks,
    face_colors_from_state,
    load_corpus_tasks,
    parse_ground_truth,
)
from tools.propose_geometry_labels import PROPOSERS  # noqa: E402
from tools.rectify_faces import (  # noqa: E402
    DEFAULT_FACE_SIZE,
    extract_stickers_from_rectified,
    rectify_face,
)
from tools.sample_stickers_from_hull import (  # noqa: E402
    apply_orientation,
    discover_orientation,
    identify_faces_jointly,
)

CORPUS_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_REPORT = REPO_ROOT / "runs" / "mask_pipeline_report.json"
DEFAULT_SUMMARY = REPO_ROOT / "runs" / "mask_pipeline_summary.txt"
PROCESSING_MAX = 1150

# Mask-path proposer to evaluate. Highest face IoU on the #137 sweep:
# 0.666 mean / 12% pass at 0.85, vs 0.565 / 5% for non-optimized.
DEFAULT_PROPOSER = "rembg_u2net_optimized_hybrid"

EXPECTED_FACES_BY_SIDE = {"A": ("U", "R", "F"), "B": ("D", "L", "B")}


# ---------------- mask-path pipeline ----------------


class _TargetShim:
    """Light shim matching the LabelTarget protocol that proposers expect.
    Populates `gt_face_quads` with empty-quad placeholders matching the
    side's expected face letters — proposers infer side from
    `target.gt_face_quads.keys()` (looking for 'U' to detect side A)."""
    def __init__(self, image_path: Path, side: str):
        self.image_path = image_path
        self.side = side
        self.image: Optional[Image.Image] = None
        self.arr: Optional[np.ndarray] = None
        self.proc_w = 0
        self.proc_h = 0
        # Placeholder keys so RembgProposer's side-inference works:
        #   side = "A" if "U" in target.gt_face_quads else "B"
        expected = EXPECTED_FACES_BY_SIDE[side]
        self.gt_face_quads: Dict = {face: [] for face in expected}

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


def _sample_face_from_proposal(
    image: Image.Image,
    face_quad,
    gt_face_colors: List[str],
) -> Optional[Dict]:
    """Rectify → sample 9 stickers → classify → discover orientation against
    gt_face_colors → return list of 9 aligned (rgb, classified, gt) tuples."""
    try:
        rectified = rectify_face(image, face_quad, output_size=DEFAULT_FACE_SIZE)
        samples_grid = extract_stickers_from_rectified(rectified)
    except Exception as e:
        return {"error": f"rectify failed: {e}"}

    # samples_grid is 3x3 row-major; flatten to 9 in row-major
    rgbs = [s.rgb for row in samples_grid for s in row]
    classified = [s.classified_color for row in samples_grid for s in row]

    # Discover orientation against gt_face_colors. discover_orientation
    # returns (mirror, rot_quarter, score).
    mirror, rot, match_score = discover_orientation(rgbs, gt_face_colors)
    aligned_rgbs = apply_orientation(rgbs, mirror, rot)
    aligned_classified = apply_orientation(classified, mirror, rot)

    correct = sum(1 for c, g in zip(aligned_classified, gt_face_colors) if c == g)
    return {
        "rgbs": aligned_rgbs,
        "classified": aligned_classified,
        "gtColors": list(gt_face_colors),
        "correct": correct,
        "orientation": {"mirror": mirror, "rot_quarter": rot, "match_score": round(match_score, 3)},
    }


def run_mask_pipeline_pair(
    image_a: Path,
    image_b: Path,
    gt_state: str,
    proposer_name: str = DEFAULT_PROPOSER,
) -> Dict:
    """Run the mask-path pipeline on a pair of images, using joint pair-level
    multiset face identification (from PR #126) to assign labels to true
    faces under the {URFDLB} distinct-faces invariant.

    Per-side per-side center-color face ID (which my v0 used) silently
    duplicates assignments when the classifier is wrong on a center
    sticker (Set 15 side A label 'R' → 'L' was a real failure). The
    joint matcher prevents this by construction."""
    target_a = _TargetShim(image_a, "A"); target_a.load()
    target_b = _TargetShim(image_b, "B"); target_b.load()
    if proposer_name not in PROPOSERS:
        return {"error": f"unknown proposer {proposer_name!r}"}
    proposer = PROPOSERS[proposer_name]

    proposals = {}
    for side, target in (("A", target_a), ("B", target_b)):
        try:
            p = proposer.propose(target)
        except Exception as e:
            return {"error": f"proposer failed on side {side}: {type(e).__name__}: {e}"}
        if not p.face_quads:
            return {"error": f"proposer produced no face quads on side {side}"}
        proposals[side] = p

    # Build prepared dicts using the PROPOSER's face_quads (not hull labels).
    # identify_faces_jointly enumerates all valid label→true mappings and
    # picks the joint best under the {URFDLB} invariant.
    prepared_sides = {
        "A": {"arr": target_a.arr, "quads": proposals["A"].face_quads, "expected": EXPECTED_FACES_BY_SIDE["A"]},
        "B": {"arr": target_b.arr, "quads": proposals["B"].face_quads, "expected": EXPECTED_FACES_BY_SIDE["B"]},
    }
    label_to_true, joint_score, joint_status = identify_faces_jointly(prepared_sides, gt_state, inset=0.20)

    # Per face: sample + classify + align via discover_orientation
    per_face: Dict[str, Dict] = {}
    chunks: Dict[str, str] = {}
    total_sampled = 0
    total_correct = 0
    for side, target in (("A", target_a), ("B", target_b)):
        side_mapping = label_to_true.get(side, {})
        for label_face, true_face in side_mapping.items():
            if label_face not in proposals[side].face_quads:
                continue
            quad = proposals[side].face_quads[label_face]
            gt_colors = face_colors_from_state(gt_state, true_face)
            result = _sample_face_from_proposal(target.image, quad, gt_colors)
            if result is None or "error" in result:
                per_face[true_face] = {"side": side, "labelFace": label_face, "error": result.get("error") if result else "unknown"}
                continue
            per_face[true_face] = {
                "side": side,
                "labelFace": label_face,
                "trueFace": true_face,
                "sampled": 9,
                "correct": result["correct"],
                "accuracy": round(result["correct"] / 9, 4),
                "orientation": result["orientation"],
                "classified": result["classified"],
                "gtColors": result["gtColors"],
            }
            chunk = "".join(COLOR_TO_FACE.get(c, "?") for c in result["classified"])
            chunks[true_face] = chunk
            total_sampled += 9
            total_correct += result["correct"]

    return {
        "proposer": proposer_name,
        "jointFaceID": {"label_to_true": label_to_true, "score": joint_score, "status": joint_status},
        "perFace": per_face,
        "chunks": chunks,
        "stickersSampled": total_sampled,
        "stickersCorrect": total_correct,
        "perStickerAccuracy": round(total_correct / total_sampled, 4) if total_sampled else None,
    }


def assemble_full_state(chunks: Dict[str, str]) -> Optional[str]:
    """Assemble 54-char URFDLB state from per-face 9-char chunks.
    Returns None if any face is missing or contains '?' (classification failure)."""
    state_parts = []
    for face in FACE_ORDER:
        chunk = chunks.get(face)
        if chunk is None or len(chunk) != 9 or "?" in chunk:
            return None
        state_parts.append(chunk)
    return "".join(state_parts)


def per_sticker_accuracy(state: Optional[str], gt_state: str) -> Optional[float]:
    if state is None or len(state) != 54 or len(gt_state) != 54:
        return None
    correct = sum(1 for s, g in zip(state, gt_state) if s == g)
    return round(correct / 54, 4)


# ---------------- existing recognizer (for comparison) ----------------


def run_existing_recognizer(image_a: Path, image_b: Path) -> Dict:
    """Run WhiteUpRecognizer.recognize on the pair; return state + reliability."""
    recognizer = WhiteUpRecognizer()
    try:
        result = recognizer.recognize(image_a.read_bytes(), image_b.read_bytes())
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "state": None}
    return {
        "state": result.state,
        "confidence": result.confidence,
    }


# ---------------- per-pair evaluation ----------------


def evaluate_pair(
    set_id: str,
    image_a: Path,
    image_b: Path,
    gt_state: str,
    proposer_name: str,
) -> Dict:
    """Run both pipelines on one corpus pair, compare to GT, return row."""
    mask = run_mask_pipeline_pair(image_a, image_b, gt_state, proposer_name)
    if "error" in mask:
        mask_state = None
        mask_acc = None
    else:
        mask_state = assemble_full_state(mask.get("chunks", {}))
        mask_acc = per_sticker_accuracy(mask_state, gt_state)

    existing = run_existing_recognizer(image_a, image_b)
    existing_acc = per_sticker_accuracy(existing.get("state"), gt_state)

    # Verdict: WIN if mask-path's per-sticker accuracy strictly beats existing;
    # DRAW if within ±2 stickers (~3.7%); LOSS otherwise. Existing missing
    # state (recognizer rejected/failed) means mask-path wins if it has any state.
    verdict = "indeterminate"
    if mask_acc is not None and existing_acc is not None:
        delta = mask_acc - existing_acc
        if abs(delta * 54) < 2:
            verdict = "draw"
        elif delta > 0:
            verdict = "mask_wins"
        else:
            verdict = "existing_wins"
    elif mask_acc is not None and existing_acc is None:
        verdict = "mask_wins_existing_rejected"
    elif mask_acc is None and existing_acc is not None:
        verdict = "existing_wins_mask_assembly_failed"

    return {
        "setId": set_id,
        "proposer": proposer_name,
        "maskPath": {
            **(mask if "error" not in mask else {"error": mask["error"]}),
            "assembledState": mask_state,
            "perStickerAccuracy": mask_acc,
        },
        "existing": {
            "state": existing.get("state"),
            "confidence": existing.get("confidence"),
            "perStickerAccuracy": existing_acc,
        },
        "verdict": verdict,
        "gtState": gt_state,
    }


# ---------------- discovery + main ----------------


def discover_pairs() -> List[Tuple[str, Path, Path, str]]:
    """Return all (setId, image_a, image_b, gt_state) tuples that have both
    images + GT + hull labels (for both sides — proposers need hull labels
    to discover the cube; but actually rembg-based proposers don't NEED hull
    labels, only the human-truth comparison does)."""
    tasks = load_corpus_tasks(CORPUS_MANIFEST)
    tasks.extend(discover_additional_tasks({t.set_id for t in tasks}))
    out: List[Tuple[str, Path, Path, str]] = []
    for task in tasks:
        if not (task.image_a.exists() and task.image_b.exists()):
            continue
        try:
            gt = parse_ground_truth(task.ground_truth)
        except Exception:
            continue
        out.append((task.set_id, task.image_a, task.image_b, gt))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposer", default=DEFAULT_PROPOSER,
                    help=f"Proposer name (default: {DEFAULT_PROPOSER})")
    ap.add_argument("--only-sets", nargs="*", default=None,
                    help="Limit to specific setIds")
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = ap.parse_args()

    pairs = discover_pairs()
    if args.only_sets:
        wanted = set(args.only_sets)
        pairs = [p for p in pairs if p[0] in wanted]

    print(f"evaluating {len(pairs)} pairs with proposer={args.proposer}", file=sys.stderr)
    print(f"(progress lines below: one per set; per-line ~10-30s with rembg+recognizer)",
          file=sys.stderr)
    print("", file=sys.stderr)

    rows: List[Dict] = []
    for i, (set_id, image_a, image_b, gt) in enumerate(pairs, 1):
        try:
            row = evaluate_pair(set_id, image_a, image_b, gt, args.proposer)
        except Exception as e:
            row = {"setId": set_id, "error": f"{type(e).__name__}: {e}"}
        rows.append(row)

        # Progress line — unbuffered, per pair
        if "error" in row:
            print(f"  [{i:>2}/{len(pairs)}] set {set_id}: ERROR {row['error']}", file=sys.stderr, flush=True)
        else:
            mask_acc = row["maskPath"]["perStickerAccuracy"]
            ex_acc = row["existing"]["perStickerAccuracy"]
            print(
                f"  [{i:>2}/{len(pairs)}] set {set_id}: "
                f"mask={mask_acc if mask_acc is not None else '—':>6}  "
                f"existing={ex_acc if ex_acc is not None else '—':>6}  "
                f"verdict={row['verdict']}",
                file=sys.stderr, flush=True,
            )

    # Aggregate
    valid = [r for r in rows if "error" not in r]
    verdict_counts = Counter(r["verdict"] for r in valid)
    mask_accs = [r["maskPath"]["perStickerAccuracy"] for r in valid if r["maskPath"]["perStickerAccuracy"] is not None]
    existing_accs = [r["existing"]["perStickerAccuracy"] for r in valid if r["existing"]["perStickerAccuracy"] is not None]

    summary_lines: List[str] = []
    summary_lines.append(f"Mask-pipeline evaluation: {len(pairs)} pairs, proposer={args.proposer}")
    summary_lines.append("")
    summary_lines.append("Per-sticker accuracy (54 stickers per pair):")
    if mask_accs:
        summary_lines.append(f"  mask-path:    mean {np.mean(mask_accs):.4f}, min {min(mask_accs):.4f}, max {max(mask_accs):.4f}, n={len(mask_accs)}")
    if existing_accs:
        summary_lines.append(f"  existing:     mean {np.mean(existing_accs):.4f}, min {min(existing_accs):.4f}, max {max(existing_accs):.4f}, n={len(existing_accs)}")
    summary_lines.append("")
    summary_lines.append("Verdict counts (per pair):")
    for verdict in ("mask_wins", "draw", "existing_wins", "mask_wins_existing_rejected",
                    "existing_wins_mask_assembly_failed", "indeterminate"):
        n = verdict_counts.get(verdict, 0)
        if n > 0:
            summary_lines.append(f"  {verdict:<40s}  {n}/{len(valid)} ({n/max(1,len(valid)):.0%})")
    summary_lines.append("")
    summary_lines.append("Worst 5 pairs by mask-path accuracy:")
    sorted_by_mask = sorted(
        (r for r in valid if r["maskPath"]["perStickerAccuracy"] is not None),
        key=lambda r: r["maskPath"]["perStickerAccuracy"],
    )
    for r in sorted_by_mask[:5]:
        summary_lines.append(
            f"  set {r['setId']:<4s} mask={r['maskPath']['perStickerAccuracy']:.4f} "
            f"existing={r['existing']['perStickerAccuracy'] if r['existing']['perStickerAccuracy'] is not None else '—':>6} "
            f"verdict={r['verdict']}"
        )
    summary_lines.append("")
    summary_lines.append("Best 5 pairs by mask-path accuracy:")
    for r in sorted_by_mask[-5:][::-1]:
        summary_lines.append(
            f"  set {r['setId']:<4s} mask={r['maskPath']['perStickerAccuracy']:.4f} "
            f"existing={r['existing']['perStickerAccuracy'] if r['existing']['perStickerAccuracy'] is not None else '—':>6} "
            f"verdict={r['verdict']}"
        )

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rows, indent=2))
    Path(args.summary).write_text("\n".join(summary_lines) + "\n")

    print("", file=sys.stderr)
    print("\n".join(summary_lines), file=sys.stderr)
    print(f"\nwrote {args.report}", file=sys.stderr)
    print(f"wrote {args.summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
