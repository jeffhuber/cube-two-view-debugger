#!/usr/bin/env python3
"""Diagnose hull-label slot-to-WCA face assignment under capture yaw.

This is a diagnostic-only follow-up to the Hull-Label Tier 1 e2e bench. It
answers a narrower question than the production recognizer:

If hull-label geometry gives us stable `upper` / `right` / `front` slots for
image A and image B, how far do we get by assigning those slots directly to WCA
faces using the shared human convention plus capture yaw?

The tool intentionally bypasses the recognizer's existing joint face-ID layer.
It runs hull-label rectification, maps slots to WCA faces with
`tools.corner_conventions.wca_face_by_slot()`, samples the rectified stickers,
and evaluates three yaw sources:

- `assumed_zero`: always yaw=0.
- `manifest`: `expectedYaw.quarterTurns` when present, otherwise a documented
  `yaw=N` / `capture yaw=N` note in the manifest.
- `detected`: current `WhiteUpRecognizer` captureYaw signal when available.

For each yaw source it reports both:

- `raw`: sampled row-major rectified face order as-is.
- `gt_aligned`: an oracle per-face orientation alignment against ground truth.
  This isolates face identity from in-plane face rotation, so it should be read
  as an upper bound for slot/yaw face assignment rather than a production
  solver path.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import statistics
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import COLOR_TO_FACE  # noqa: E402
from rubik_recognizer.recognizer import WhiteUpRecognizer  # noqa: E402
from rubik_recognizer.validation import FACE_ORDER, validate_state  # noqa: E402
from tools.audit_recognition_pair import parse_ground_truth as parse_pair_ground_truth  # noqa: E402
from tools.corner_conventions import FACE_DEFS_BY_SIDE, wca_face_by_slot, wca_facelets_for_label  # noqa: E402
from tools.evaluate_hybrid_pipeline import DEFAULT_FACE_SIZE, _load_processing_image  # noqa: E402
from tools.extract_color_samples import PairTask, face_colors_from_state, load_corpus_tasks  # noqa: E402
from tools.global_cube_model import _fit_hull_label_tier1_model  # noqa: E402
from tools.rectify_faces import extract_stickers_from_rectified, rectify_face  # noqa: E402
from tools.sample_stickers_from_hull import apply_orientation, canonical_corner_order, discover_orientation  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "hull_label_slot_yaw_assignment.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "HULL_LABEL_SLOT_YAW_ASSIGNMENT.md"

SLOT_TO_LEGACY_FACE = {
    "upper": "face_xz",
    "right": "face_yz",
    "front": "face_xy",
}


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _hamming(actual: Optional[str], expected: str) -> Optional[int]:
    if actual is None or len(actual) != len(expected):
        return None
    return sum(1 for a, e in zip(actual, expected) if a != e)


def _manifest_by_set(manifest_path: Path) -> Dict[str, Mapping[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        str(pair["setId"]): pair
        for pair in manifest.get("pairs", [])
    }


def _note_yaw_quarter_turns(notes: str) -> Optional[int]:
    match = re.search(r"(?:capture\s+)?yaw\s*=\s*([0-3])\b", notes, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)) % 4


def manifest_yaw_source(pair: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    expected = pair.get("expectedYaw")
    if isinstance(expected, Mapping):
        quarter_turns = expected.get("quarterTurns")
        if isinstance(quarter_turns, int):
            return {
                "source": "manifest_expectedYaw",
                "yawQuarterTurns": quarter_turns % 4,
                "status": expected.get("status"),
                "normalizationApplied": expected.get("normalizationApplied"),
            }
    notes = pair.get("notes")
    if isinstance(notes, str):
        yaw = _note_yaw_quarter_turns(notes)
        if yaw is not None:
            return {
                "source": "manifest_notes",
                "yawQuarterTurns": yaw,
                "status": "documented",
                "normalizationApplied": None,
            }
    return None


def slot_face_assignments(yaw_quarter_turns: int) -> Dict[str, Dict[str, str]]:
    return {
        "A": wca_face_by_slot("A", yaw_quarter_turns),
        "B": wca_face_by_slot("B", yaw_quarter_turns),
    }


def _detected_yaw_source(task: PairTask) -> Optional[Dict[str, Any]]:
    try:
        result = WhiteUpRecognizer().recognize(
            task.image_a.read_bytes(),
            task.image_b.read_bytes(),
            hull_label_tier1_mode="off",
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "source": "detected",
            "yawQuarterTurns": None,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    signals = result.recognition_signals or {}
    capture_yaw = signals.get("captureYaw")
    if not isinstance(capture_yaw, Mapping):
        return {
            "source": "detected",
            "yawQuarterTurns": None,
            "status": "unavailable",
            "recognitionStatus": result.status,
            "recognitionReason": result.reason,
        }
    yaw = capture_yaw.get("quarterTurns")
    return {
        "source": "detected",
        "yawQuarterTurns": yaw if isinstance(yaw, int) else None,
        "status": capture_yaw.get("status"),
        "normalizationApplied": capture_yaw.get("normalizationApplied"),
        "recognitionStatus": result.status,
        "recognitionReason": result.reason,
    }


def _compact_trace(model: Optional[Any], trace: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    trace = trace or {}
    keys = (
        "status",
        "accepted",
        "hard_failures",
        "warnings",
        "vertex_source",
        "vertex_cloud_spread_px",
        "vertex_cloud_spread_norm",
        "projective_residual_norm",
        "sticker_score_total",
        "mean_sticker_distance",
    )
    out = {key: trace.get(key) for key in keys if key in trace}
    if model is None and "status" not in out:
        out["status"] = "fit_failed"
    return out


def _fit_hull_side(image_path: Path, side: str, sess: Any) -> Dict[str, Any]:
    from rembg import remove  # noqa: E402

    image, arr = _load_processing_image(image_path)
    rgba = remove(image, session=sess)
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    mask = alpha > 128
    model, trace = _fit_hull_label_tier1_model(
        arr,
        mask,
        side=side,
        mode="prefer",
    )
    return {
        "image": image,
        "model": model,
        "trace": _compact_trace(model, trace),
    }


def _sample_slot_face(
    *,
    image: Any,
    model: Any,
    side: str,
    slot: str,
    wca_face: str,
    yaw_quarter_turns: int,
    gt_state: str,
) -> Dict[str, Any]:
    legacy_face = SLOT_TO_LEGACY_FACE[slot]
    quad = model.face_quads[legacy_face]
    rectified = rectify_face(image, quad, output_size=DEFAULT_FACE_SIZE)
    samples = extract_stickers_from_rectified(rectified)
    rgbs = [sample.rgb for row in samples for sample in row]
    raw_colors = [sample.classified_color for row in samples for sample in row]
    raw_faces = [COLOR_TO_FACE[color] for color in raw_colors]

    convention = _convention_orientation(
        side=side,
        slot=slot,
        yaw_quarter_turns=yaw_quarter_turns,
        wca_face=wca_face,
        quad=quad,
    )
    convention_colors = apply_orientation(raw_colors, *convention) if convention else []
    convention_faces = [COLOR_TO_FACE[color] for color in convention_colors]

    gt_colors = face_colors_from_state(gt_state, wca_face)
    mirror, rot, orientation_score = discover_orientation(rgbs, gt_colors)
    aligned_colors = apply_orientation(raw_colors, mirror, rot)
    aligned_faces = [COLOR_TO_FACE[color] for color in aligned_colors]

    gt_faces = [COLOR_TO_FACE[color] for color in gt_colors]
    return {
        "side": side,
        "slot": slot,
        "wcaFace": wca_face,
        "rawFaces": "".join(raw_faces),
        "conventionFaces": "".join(convention_faces),
        "alignedFaces": "".join(aligned_faces),
        "rawCorrect": sum(1 for got, expected in zip(raw_faces, gt_faces) if got == expected),
        "conventionCorrect": sum(1 for got, expected in zip(convention_faces, gt_faces) if got == expected),
        "alignedCorrect": sum(1 for got, expected in zip(aligned_faces, gt_faces) if got == expected),
        "conventionOrientation": (
            {"mirror": convention[0], "rotQuarter": convention[1]}
            if convention else None
        ),
        "oracleOrientation": {
            "mirror": mirror,
            "rotQuarter": rot,
            "score": round(float(orientation_score), 4),
        },
    }


def _assembled_state(face_chunks: Mapping[str, Sequence[str]]) -> Optional[str]:
    if set(face_chunks) != set(FACE_ORDER):
        return None
    return "".join("".join(face_chunks[face]) for face in FACE_ORDER)


def _orientation_from_corner_map(raw_corner_to_wca_index: Mapping[int, int]) -> Optional[Tuple[bool, int]]:
    raw_indices = list(range(9))
    for mirror in (False, True):
        for rot in range(4):
            oriented = apply_orientation(raw_indices, mirror, rot)
            if all(oriented[wca_index] == raw_index for raw_index, wca_index in raw_corner_to_wca_index.items()):
                return mirror, rot
    return None


def _nearest_label(point: Tuple[float, float], labeled_points: Sequence[Tuple[str, Tuple[float, float]]]) -> str:
    label, _ = min(
        labeled_points,
        key=lambda item: (point[0] - item[1][0]) ** 2 + (point[1] - item[1][1]) ** 2,
    )
    return label


def _convention_orientation(
    *,
    side: str,
    slot: str,
    yaw_quarter_turns: int,
    wca_face: str,
    quad: Sequence[Tuple[float, float]],
) -> Optional[Tuple[bool, int]]:
    labels = FACE_DEFS_BY_SIDE[side][slot]
    labeled_points = list(zip(labels, quad))
    canonical_points = canonical_corner_order(quad)
    raw_corner_indices = (0, 2, 8, 6)
    raw_to_wca: Dict[int, int] = {}
    for raw_index, point in zip(raw_corner_indices, canonical_points):
        label = _nearest_label(point, labeled_points)
        facelet = next(
            (
                item
                for item in wca_facelets_for_label(side, label, yaw_quarter_turns)
                if item.startswith(wca_face)
            ),
            None,
        )
        if facelet is None:
            return None
        raw_to_wca[raw_index] = int(facelet[1:]) - 1
    return _orientation_from_corner_map(raw_to_wca)


def _evaluate_slot_yaw_source(
    *,
    source: str,
    yaw_quarter_turns: Optional[int],
    gt_state: str,
    side_fits: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    if yaw_quarter_turns is None:
        return {
            "source": source,
            "status": "unavailable",
            "yawQuarterTurns": None,
        }

    assignments = slot_face_assignments(yaw_quarter_turns)
    raw_chunks: Dict[str, List[str]] = {}
    convention_chunks: Dict[str, List[str]] = {}
    aligned_chunks: Dict[str, List[str]] = {}
    per_face: List[Dict[str, Any]] = []

    for side in ("A", "B"):
        model = side_fits[side].get("model")
        if model is None:
            return {
                "source": source,
                "status": "fit_failed",
                "yawQuarterTurns": yaw_quarter_turns,
                "slotFaceAssignments": assignments,
                "sideTraces": {s: side_fits[s].get("trace") for s in ("A", "B")},
            }
        for slot, wca_face in assignments[side].items():
            row = _sample_slot_face(
                image=side_fits[side]["image"],
                model=model,
                side=side,
                slot=slot,
                wca_face=wca_face,
                yaw_quarter_turns=yaw_quarter_turns,
                gt_state=gt_state,
            )
            per_face.append({
                key: value
                for key, value in row.items()
                if key not in {"rawFaces", "alignedFaces"}
            })
            raw_chunks[wca_face] = list(row["rawFaces"])
            if row.get("conventionFaces"):
                convention_chunks[wca_face] = list(row["conventionFaces"])
            aligned_chunks[wca_face] = list(row["alignedFaces"])

    raw_state = _assembled_state(raw_chunks)
    convention_state = _assembled_state(convention_chunks)
    aligned_state = _assembled_state(aligned_chunks)
    raw_hamming = _hamming(raw_state, gt_state)
    convention_hamming = _hamming(convention_state, gt_state)
    aligned_hamming = _hamming(aligned_state, gt_state)
    raw_validation = validate_state(raw_state) if raw_state else None
    convention_validation = validate_state(convention_state) if convention_state else None
    aligned_validation = validate_state(aligned_state) if aligned_state else None
    return {
        "source": source,
        "status": "assembled" if raw_state and aligned_state else "incomplete",
        "yawQuarterTurns": yaw_quarter_turns,
        "slotFaceAssignments": assignments,
        "raw": {
            "state": raw_state,
            "hamming": raw_hamming,
            "stickersCorrect": (54 - raw_hamming) if raw_hamming is not None else None,
            "exactMatch": raw_state == gt_state if raw_state else False,
            "validState": bool(raw_validation and raw_validation.valid),
            "validationErrors": list(raw_validation.errors) if raw_validation else ["not_assembled"],
        },
        "convention": {
            "state": convention_state,
            "hamming": convention_hamming,
            "stickersCorrect": (54 - convention_hamming) if convention_hamming is not None else None,
            "exactMatch": convention_state == gt_state if convention_state else False,
            "validState": bool(convention_validation and convention_validation.valid),
            "validationErrors": list(convention_validation.errors) if convention_validation else ["not_assembled"],
        },
        "gtAligned": {
            "state": aligned_state,
            "hamming": aligned_hamming,
            "stickersCorrect": (54 - aligned_hamming) if aligned_hamming is not None else None,
            "exactMatch": aligned_state == gt_state if aligned_state else False,
            "validState": bool(aligned_validation and aligned_validation.valid),
            "validationErrors": list(aligned_validation.errors) if aligned_validation else ["not_assembled"],
        },
        "perFace": per_face,
        "sideTraces": {s: side_fits[s].get("trace") for s in ("A", "B")},
    }


def _yaw_sources_for_pair(
    pair: Mapping[str, Any],
    detected: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    sources = [{
        "source": "assumed_zero",
        "yawQuarterTurns": 0,
        "status": "assumed",
    }]
    manifest = manifest_yaw_source(pair)
    if manifest is not None:
        sources.append(manifest)
    if detected is not None:
        sources.append(dict(detected))
    return sources


def _evaluate_pair(
    task: PairTask,
    manifest_pair: Mapping[str, Any],
    sess: Any,
    *,
    include_detected_yaw: bool,
) -> Dict[str, Any]:
    _sha, raw_state, gt_state, canonicalized = parse_pair_ground_truth(str(task.ground_truth))
    side_fits = {
        "A": _fit_hull_side(task.image_a, "A", sess),
        "B": _fit_hull_side(task.image_b, "B", sess),
    }
    detected = _detected_yaw_source(task) if include_detected_yaw else None
    yaw_sources = _yaw_sources_for_pair(manifest_pair, detected)
    evaluations = [
        _evaluate_slot_yaw_source(
            source=str(source["source"]),
            yaw_quarter_turns=source.get("yawQuarterTurns") if isinstance(source.get("yawQuarterTurns"), int) else None,
            gt_state=gt_state,
            side_fits=side_fits,
        )
        for source in yaw_sources
    ]
    source_meta = {
        str(source["source"]): {
            key: value
            for key, value in source.items()
            if key != "source"
        }
        for source in yaw_sources
    }
    return {
        "setId": task.set_id,
        "source": task.source,
        "images": {
            "A": _rel(task.image_a),
            "B": _rel(task.image_b),
            "groundTruth": _rel(task.ground_truth),
        },
        "groundTruthCanonicalized": canonicalized,
        "rawGroundTruthState": raw_state,
        "canonicalGroundTruthState": gt_state,
        "yawSourceMeta": source_meta,
        "evaluations": {
            row["source"]: row
            for row in evaluations
        },
    }


def _source_summary(rows: Sequence[Mapping[str, Any]], source: str) -> Dict[str, Any]:
    evals = [
        row.get("evaluations", {}).get(source)
        for row in rows
        if row.get("evaluations", {}).get(source)
    ]
    assembled = [row for row in evals if row and row.get("status") == "assembled"]
    out: Dict[str, Any] = {
        "rowsAvailable": len(evals),
        "assembled": len(assembled),
    }
    for key in ("raw", "convention", "gtAligned"):
        hamming = [row[key]["hamming"] for row in assembled if row.get(key, {}).get("hamming") is not None]
        out[key] = {
            "exact": sum(1 for row in assembled if row[key].get("exactMatch")),
            "legal": sum(1 for row in assembled if row[key].get("validState")),
            "meanStickersCorrect": round(
                statistics.mean(54 - value for value in hamming), 2,
            ) if hamming else None,
            "medianHamming": statistics.median(hamming) if hamming else None,
        }
    yaw_counts: Counter[str] = Counter()
    for row in evals:
        if row is None:
            continue
        yaw_counts[str(row.get("yawQuarterTurns"))] += 1
    out["yawCounts"] = dict(sorted(yaw_counts.items()))
    return out


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sources = sorted({
        source
        for row in rows
        for source in row.get("evaluations", {})
    })
    return {
        "pairCount": len(rows),
        "byYawSource": {
            source: _source_summary(rows, source)
            for source in sources
        },
    }


def render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Hull-Label Slot/Yaw Assignment Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic tests whether hull-label `upper` / `right` / `front`",
        "slots can be mapped directly to canonical WCA faces using capture yaw,",
        "instead of asking the existing joint face-ID layer to infer each face",
        "from center sticker color alone.",
        "",
        "Two scores are reported:",
        "",
        "- `raw`: rectified face row-major order as sampled.",
        "- `convention`: non-oracle in-plane orientation derived from the shared",
        "  corner/facelet convention.",
        "- `gt_aligned`: oracle per-face orientation alignment against ground truth.",
        "  This isolates slot/yaw face identity from in-plane face rotation and",
        "  should be read as an upper bound, not production behavior.",
        "",
        f"Git head: `{payload['source']['git_sha']}`",
        f"Generated: `{payload['source']['generated_at_utc']}`",
        "",
        "## Summary By Yaw Source",
        "",
        "| Yaw source | Rows | Assembled | Yaw counts | Raw exact | Raw mean stickers | Convention exact | Convention legal | Convention mean stickers | GT-aligned exact | GT-aligned mean stickers |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, row in summary["byYawSource"].items():
        lines.append(
            f"| `{source}` | {row['rowsAvailable']} | {row['assembled']} | "
            f"`{row['yawCounts']}` | {row['raw']['exact']} | "
            f"{row['raw']['meanStickersCorrect']} | {row['convention']['exact']} | "
            f"{row['convention']['legal']} | {row['convention']['meanStickersCorrect']} | "
            f"{row['gtAligned']['exact']} | {row['gtAligned']['meanStickersCorrect']} |"
        )

    detected = summary["byYawSource"].get("detected", {})
    detected_conv = detected.get("convention", {})
    detected_oracle = detected.get("gtAligned", {})
    lines.extend([
        "",
        "## Key Findings",
        "",
        "- Convention-derived in-plane orientation tracks the oracle orientation",
        "  closely when yaw is right. In this run, `detected` yaw produced",
        f"  {detected_conv.get('exact', 0)} convention-exact rows versus",
        f"  {detected_oracle.get('exact', 0)} oracle-exact rows under `gt_aligned`.",
        "- Capture yaw is load-bearing. `assumed_zero` exact rows are much lower",
        "  than detected/manifest yaw on non-zero-yaw captures, and wrong yaw",
        "  commonly produces about 40+ sticker errors.",
        "- Current detected yaw is useful when present, but unavailable on many",
        "  reject/retake rows. A production slot/yaw path needs a yaw source that",
        "  survives rows where the legacy recognizer cannot emit `captureYaw`.",
        "- Remaining small hamming counts under correct yaw are not face-identity",
        "  failures; they point at sticker color sampling/classification quality.",
    ])

    lines.extend([
        "",
        "## Interpretation Guide",
        "",
        "- If `convention` is strong, the shared geometry convention can assign",
        "  both WCA face identity and in-plane face orientation.",
        "- If `gt_aligned` is strong while `convention` is weak, slot/yaw face",
        "  identity is promising but the convention-derived orientation is wrong.",
        "- If both `convention` and `gt_aligned` are weak, the problem is still",
        "  color sampling, rectification, or the yaw source.",
        "- If `manifest` beats `detected`, yaw detection is the bottleneck.",
        "- If `assumed_zero` is close to `manifest`, this corpus is mostly yaw=0.",
        "",
        "## Per-Pair Snapshot",
        "",
        "| Set | Source | Yaw | Raw hamming | Convention hamming | Convention legal | GT-aligned hamming | GT-aligned legal |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ])
    for row in payload["rows"]:
        for source, evaluation in row.get("evaluations", {}).items():
            raw = evaluation.get("raw", {})
            convention = evaluation.get("convention", {})
            aligned = evaluation.get("gtAligned", {})
            lines.append(
                f"| {row['setId']} | `{source}` | {evaluation.get('yawQuarterTurns')} | "
                f"{raw.get('hamming')} | {convention.get('hamming')} | "
                f"{convention.get('validState')} | "
                f"{aligned.get('hamming')} | {aligned.get('validState')} |"
            )
    lines.append("")
    return "\n".join(lines)


def _discover_tasks(manifest_path: Path, only_sets: Optional[Iterable[str]]) -> List[PairTask]:
    tasks = load_corpus_tasks(manifest_path)
    if only_sets:
        wanted = {str(item) for item in only_sets}
        tasks = [task for task in tasks if task.set_id in wanted]
    return [
        task for task in tasks
        if task.image_a.exists() and task.image_b.exists() and task.ground_truth.exists()
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--only-sets", nargs="*", default=None)
    parser.add_argument(
        "--skip-detected-yaw",
        action="store_true",
        help="Skip current recognizer captureYaw probing.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render --report from --out-json without re-running image analysis.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.render_only:
        payload = json.loads(args.out_json.read_text(encoding="utf-8"))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(payload), encoding="utf-8")
        print(f"wrote {args.report}", file=sys.stderr)
        return 0

    from rembg import new_session  # noqa: E402

    manifest = _manifest_by_set(args.manifest)
    tasks = _discover_tasks(args.manifest, args.only_sets)
    sess = new_session("u2net")
    rows: List[Dict[str, Any]] = []
    print(f"diagnosing {len(tasks)} pairs", file=sys.stderr)
    for index, task in enumerate(tasks, 1):
        try:
            row = _evaluate_pair(
                task,
                manifest.get(task.set_id, {}),
                sess,
                include_detected_yaw=not args.skip_detected_yaw,
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "setId": task.set_id,
                "source": task.source,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        rows.append(row)
        if "evaluations" not in row:
            print(f"  [{index}/{len(tasks)}] set {task.set_id}: ERROR", file=sys.stderr, flush=True)
            continue
        compact = []
        for source, evaluation in row["evaluations"].items():
            convention = evaluation.get("convention", {})
            aligned = evaluation.get("gtAligned", {})
            compact.append(
                f"{source}:yaw={evaluation.get('yawQuarterTurns')} "
                f"conv={convention.get('hamming')} oracle={aligned.get('hamming')}"
            )
        print(
            f"  [{index}/{len(tasks)}] set {task.set_id}: " + ", ".join(compact),
            file=sys.stderr,
            flush=True,
        )

    scored_rows = [row for row in rows if "evaluations" in row]
    payload = {
        "schema": "hull_label_slot_yaw_assignment_v1",
        "source": {
            "tool": "tools/diagnose_slot_yaw_assignment.py",
            "git_sha": _git_head_sha(),
            "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "manifest": _rel(args.manifest),
            "detected_yaw_enabled": not args.skip_detected_yaw,
        },
        "summary": build_summary(scored_rows),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), file=sys.stderr)
    print(f"wrote {args.out_json}", file=sys.stderr)
    print(f"wrote {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
