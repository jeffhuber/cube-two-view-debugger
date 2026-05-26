#!/usr/bin/env python3
"""Validate the Tier 1 hull-label path in shadow/prefer modes.

This is a diagnostic-only bridge between the newly wired
``fit_global_cube_model(..., hull_label_mode=...)`` feature flag and the
pair-level question that matters for cube-snap: if we use those generated face
quads for A+B, can we assemble a legal and accurate 54-sticker cube?

Per pair, the tool runs three geometry modes:

- ``off``: legacy global-cube model path.
- ``shadow``: legacy model returned, hull-label trace attached.
- ``prefer``: accepted hull-label model returned, with fallback to legacy on
  gate failures.

For each mode it rectifies the three visible faces per side, uses the existing
joint face-ID helper to map capture labels to true WCA faces, classifies the 54
stickers, and validates the assembled state against ground truth.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import COLOR_TO_FACE  # noqa: E402
from rubik_recognizer.validation import FACE_ORDER, validate_state  # noqa: E402
from tools.audit_recognition_pair import parse_ground_truth as parse_pair_ground_truth  # noqa: E402
from tools.evaluate_hybrid_pipeline import (  # noqa: E402
    DEFAULT_FACE_SIZE,
    EXPECTED_FACES_BY_SIDE,
    _classify_face_aligned,
    _load_processing_image,
)
from tools.extract_color_samples import (  # noqa: E402
    PairTask,
    discover_additional_tasks,
    face_colors_from_state,
    load_corpus_tasks,
)
from tools.global_cube_model import (  # noqa: E402
    _fit_hull_label_tier1_model_from_alpha,
    fit_global_cube_model,
)
from tools.interior_bezel_detection import detect_interior_bezel_lines  # noqa: E402
from tools.rectify_faces import rectify_face  # noqa: E402
from tools.sample_stickers_from_hull import identify_faces_jointly  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "hull_label_tier1_shadow_validation.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "HULL_LABEL_TIER1_SHADOW_VALIDATION.md"
MODES = ("off", "shadow", "prefer")

LEGACY_FACE_LABEL_BY_SIDE = {
    "A": {
        "face_xz": "U",
        "face_yz": "R",
        "face_xy": "F",
    },
    "B": {
        "face_xz": "D",
        "face_yz": "B",
        "face_xy": "L",
    },
}


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _hamming(actual: Optional[str], expected: str) -> Optional[int]:
    if not actual or len(actual) != len(expected):
        return None
    return sum(1 for a, e in zip(actual, expected) if a != e)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _compact_trace(model: Optional[Any]) -> Optional[Dict[str, Any]]:
    if model is None:
        return None
    trace = (getattr(model, "debug", {}) or {}).get("hull_label_tier1")
    if not isinstance(trace, dict):
        return None
    return _compact_trace_payload(trace)


def _compact_trace_payload(trace: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "mode",
        "side",
        "status",
        "selected",
        "accepted",
        "hard_failures",
        "warnings",
        "shadow_returned_legacy",
        "fallback_to_legacy",
        "vertex_source",
        "hexagon_diameter_px",
        "vertex_cloud_spread_px",
        "vertex_cloud_spread_norm",
        "projective_residual_norm",
        "sticker_score_total",
        "mean_sticker_distance",
        "selected_mask_threshold",
        "best_any_threshold",
        "best_any_score",
    )
    out = {key: trace.get(key) for key in keys if key in trace}
    metrics = trace.get("metrics")
    if isinstance(metrics, dict):
        out["metrics"] = metrics
    return out


def _face_quads_by_label(model: Any, side: str) -> Dict[str, List[Tuple[float, float]]]:
    """Map legacy model face keys to capture-frame labels expected downstream."""
    labels = LEGACY_FACE_LABEL_BY_SIDE[side]
    quads: Dict[str, List[Tuple[float, float]]] = {}
    for legacy_face, label in labels.items():
        quad = model.face_quads.get(legacy_face)
        if quad is not None and len(quad) == 4:
            quads[label] = [(float(x), float(y)) for x, y in quad]
    return quads


def _fit_side_modes(
    image_path: Path,
    side: str,
    sess: Any,
) -> Dict[str, Dict[str, Any]]:
    from rembg import remove  # noqa: E402

    image, arr = _load_processing_image(image_path)
    rgba = remove(image, session=sess)
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    mask = alpha > 128
    detection = detect_interior_bezel_lines(arr, mask)

    legacy_model = fit_global_cube_model(detection, arr, mask, hull_label_mode="off")

    def _legacy_row(trace: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        if legacy_model is None:
            return {
                "status": "fit_failed",
                "quads": {},
                "trace": _compact_trace_payload(trace) if trace else None,
                "approach": None,
            }
        return {
            "status": "fit_ok",
            "quads": _face_quads_by_label(legacy_model, side),
            "trace": _compact_trace_payload(trace) if trace else _compact_trace(legacy_model),
            "approach": (legacy_model.debug or {}).get("approach"),
            "cube_center_source": (legacy_model.debug or {}).get("cube_center_source"),
        }

    out: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        try:
            if mode == "off":
                out[mode] = _legacy_row()
                continue

            model, trace = _fit_hull_label_tier1_model_from_alpha(
                arr,
                alpha,
                side=side,
                mode=mode,
            )
            if mode == "shadow":
                out[mode] = _legacy_row(trace)
                continue
            if model is None:
                out[mode] = _legacy_row(trace)
                out[mode]["fallback_to_legacy"] = True
                continue
            out[mode] = {
                "status": "fit_ok",
                "quads": _face_quads_by_label(model, side),
                "trace": _compact_trace(model),
                "approach": (model.debug or {}).get("approach"),
                "cube_center_source": (model.debug or {}).get("cube_center_source"),
            }
        except Exception as exc:  # noqa: BLE001
            out[mode] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "quads": {},
                "trace": None,
                "approach": None,
            }
    return out


def _assemble_pair_mode(
    *,
    mode: str,
    gt_state: str,
    images: Mapping[str, Any],
    side_results: Mapping[str, Mapping[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    mode_sides = {side: side_results[side][mode] for side in ("A", "B")}
    if any(row.get("status") != "fit_ok" for row in mode_sides.values()):
        return {
            "mode": mode,
            "status": "fit_failed",
            "sideStatus": {side: row.get("status") for side, row in mode_sides.items()},
            "sideTraces": {side: row.get("trace") for side, row in mode_sides.items()},
        }

    prepared = {
        side: {
            "arr": images[side]["arr"],
            "quads": mode_sides[side]["quads"],
            "expected": EXPECTED_FACES_BY_SIDE[side],
        }
        for side in ("A", "B")
    }
    label_to_true, joint_score, joint_status = identify_faces_jointly(
        prepared, gt_state, inset=0.20,
    )

    per_face_aligned: Dict[str, List[str]] = {}
    per_face: List[Dict[str, Any]] = []
    stickers_sampled = 0
    stickers_correct = 0
    for side in ("A", "B"):
        for label_face, true_face in label_to_true.get(side, {}).items():
            quad = mode_sides[side]["quads"].get(label_face)
            if quad is None:
                per_face.append({
                    "side": side,
                    "labelFace": label_face,
                    "trueFace": true_face,
                    "error": "missing_quad",
                })
                continue
            try:
                rectified = rectify_face(images[side]["image"], quad, output_size=DEFAULT_FACE_SIZE)
                correct, aligned, _rgbs = _classify_face_aligned(
                    rectified,
                    face_colors_from_state(gt_state, true_face),
                )
            except Exception as exc:  # noqa: BLE001
                per_face.append({
                    "side": side,
                    "labelFace": label_face,
                    "trueFace": true_face,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            stickers_sampled += 9
            stickers_correct += correct
            per_face_aligned[true_face] = aligned
            per_face.append({
                "side": side,
                "labelFace": label_face,
                "trueFace": true_face,
                "correct": correct,
                "ofTotal": 9,
            })

    assembled: Optional[str] = None
    if all(face in per_face_aligned for face in FACE_ORDER):
        assembled = "".join(
            "".join(COLOR_TO_FACE[color] for color in per_face_aligned[face])
            for face in FACE_ORDER
        )
    validation = validate_state(assembled) if assembled else None
    hamming = _hamming(assembled, gt_state)
    return {
        "mode": mode,
        "status": "assembled" if assembled else "incomplete",
        "jointStatus": joint_status,
        "jointScore": joint_score,
        "facesRecovered": len(per_face_aligned),
        "stickersSampled": stickers_sampled,
        "stickersCorrect": stickers_correct,
        "perStickerAccuracy": round(stickers_correct / stickers_sampled, 4)
        if stickers_sampled else None,
        "assembledState": assembled,
        "validState": bool(validation and validation.valid),
        "validationErrors": list(validation.errors) if validation else ["not_assembled"],
        "exactMatch": assembled == gt_state if assembled else False,
        "hamming": hamming,
        "perStickerMatchesAssembled": (54 - hamming) if hamming is not None else None,
        "sideApproach": {side: row.get("approach") for side, row in mode_sides.items()},
        "sideCubeCenterSource": {side: row.get("cube_center_source") for side, row in mode_sides.items()},
        "sideTraces": {side: row.get("trace") for side, row in mode_sides.items()},
        "perFace": per_face,
    }


def _evaluate_pair(task: PairTask, sess: Any) -> Dict[str, Any]:
    _sha, raw_state, gt_state, canonicalized = parse_pair_ground_truth(str(task.ground_truth))
    images: Dict[str, Any] = {}
    side_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for side, path in (("A", task.image_a), ("B", task.image_b)):
        image, arr = _load_processing_image(path)
        images[side] = {"image": image, "arr": arr}
        side_results[side] = _fit_side_modes(path, side, sess)

    modes = {
        mode: _assemble_pair_mode(
            mode=mode,
            gt_state=gt_state,
            images=images,
            side_results=side_results,
        )
        for mode in MODES
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
        "modes": modes,
    }


def _mode_summary(rows: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    mode_rows = [row["modes"][mode] for row in rows]
    assembled = [row for row in mode_rows if row.get("assembledState")]
    exact = [row for row in mode_rows if row.get("exactMatch")]
    legal = [row for row in mode_rows if row.get("validState")]
    total_sampled = sum(int(row.get("stickersSampled") or 0) for row in mode_rows)
    total_correct = sum(int(row.get("stickersCorrect") or 0) for row in mode_rows)
    assembled_total = len(assembled) * 54
    assembled_correct = sum(int(row.get("perStickerMatchesAssembled") or 0) for row in assembled)
    return {
        "pairs": len(mode_rows),
        "assembled": len(assembled),
        "legal": len(legal),
        "exact": len(exact),
        "rectifiedStickerAccuracy": round(total_correct / total_sampled, 6)
        if total_sampled else None,
        "assembledStickerAccuracy": round(assembled_correct / assembled_total, 6)
        if assembled_total else None,
    }


def _trace_summary(rows: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    statuses: Counter[str] = Counter()
    selected = 0
    accepted = 0
    warnings: Counter[str] = Counter()
    hard_failures: Counter[str] = Counter()
    for row in rows:
        for side in ("A", "B"):
            trace = row["modes"][mode].get("sideTraces", {}).get(side)
            if not trace:
                continue
            statuses[str(trace.get("status"))] += 1
            if trace.get("selected"):
                selected += 1
            if trace.get("accepted"):
                accepted += 1
            warnings.update(str(item) for item in trace.get("warnings") or [])
            hard_failures.update(str(item) for item in trace.get("hard_failures") or [])
    return {
        "sideTraces": sum(statuses.values()),
        "acceptedSides": accepted,
        "selectedSides": selected,
        "statusCounts": dict(sorted(statuses.items())),
        "warningCounts": dict(warnings.most_common()),
        "hardFailureCounts": dict(hard_failures.most_common()),
    }


def _prefer_delta_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    improved = []
    regressed = []
    same = []
    incomplete_changed = []
    for row in rows:
        off = row["modes"]["off"]
        prefer = row["modes"]["prefer"]
        off_h = off.get("hamming")
        prefer_h = prefer.get("hamming")
        item = {
            "setId": row["setId"],
            "offHamming": off_h,
            "preferHamming": prefer_h,
            "offExact": off.get("exactMatch"),
            "preferExact": prefer.get("exactMatch"),
        }
        if off_h is None or prefer_h is None:
            if off_h != prefer_h:
                incomplete_changed.append(item)
            else:
                same.append(item)
        elif prefer_h < off_h:
            improved.append(item)
        elif prefer_h > off_h:
            regressed.append(item)
        else:
            same.append(item)
    return {
        "improved": improved,
        "regressed": regressed,
        "sameCount": len(same),
        "incompleteChanged": incomplete_changed,
    }


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "pairCount": len(rows),
        "byMode": {mode: _mode_summary(rows, mode) for mode in MODES},
        "shadowTrace": _trace_summary(rows, "shadow"),
        "preferTrace": _trace_summary(rows, "prefer"),
        "preferVsOff": _prefer_delta_summary(rows),
    }


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Hull-Label Tier 1 Shadow Validation",
        "",
        "## Purpose",
        "",
        "This report validates the feature-flagged Tier 1 hull-label path added",
        "to `tools.global_cube_model.fit_global_cube_model()`. It compares the",
        "legacy path (`off`), trace-only shadow mode (`shadow`), and accepted",
        "candidate mode with fallback (`prefer`) on A+B corpus pairs.",
        "",
        "The pair-level score is diagnostic: it rectifies faces from the global",
        "model quads, classifies stickers, performs joint face-ID, assembles a",
        "54-sticker URFDLB state, and validates that state. It does not change",
        "the production `WhiteUpRecognizer` path.",
        "",
        f"Git head: `{payload['source']['git_sha']}`",
        f"Generated: `{payload['source']['generated_at_utc']}`",
        "",
        "## Summary By Mode",
        "",
        "| Mode | Pairs | Assembled | Legal | Exact | Rectified sticker acc | Assembled sticker acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = summary["byMode"][mode]
        lines.append(
            f"| `{mode}` | {row['pairs']} | {row['assembled']} | {row['legal']} | "
            f"{row['exact']} | {_pct(row['rectifiedStickerAccuracy'])} | "
            f"{_pct(row['assembledStickerAccuracy'])} |"
        )

    for mode, title in (("shadow", "Shadow Trace"), ("prefer", "Prefer Trace")):
        trace = summary[f"{mode}Trace"]
        lines.extend([
            "",
            f"## {title}",
            "",
            f"- Side traces: `{trace['sideTraces']}`",
            f"- Accepted sides: `{trace['acceptedSides']}`",
            f"- Selected sides: `{trace['selectedSides']}`",
            f"- Status counts: `{trace['statusCounts']}`",
        ])
        if trace["hardFailureCounts"]:
            lines.append(f"- Hard failures: `{trace['hardFailureCounts']}`")
        if trace["warningCounts"]:
            lines.append(f"- Warnings: `{trace['warningCounts']}`")

    delta = summary["preferVsOff"]
    lines.extend([
        "",
        "## Prefer Versus Legacy",
        "",
        f"- Improved hamming: `{len(delta['improved'])}`",
        f"- Regressed hamming: `{len(delta['regressed'])}`",
        f"- Same hamming/incomplete status: `{delta['sameCount']}`",
        f"- Incomplete-status changes: `{len(delta['incompleteChanged'])}`",
    ])
    if delta["improved"]:
        lines.append("")
        improved_count = len(delta["improved"])
        suffix = (
            f"first 20 of {improved_count}; see per-pair snapshot below"
            if improved_count > 20
            else str(improved_count)
        )
        lines.append(f"Improved rows ({suffix}):")
        for row in delta["improved"][:20]:
            lines.append(
                f"- Set {row['setId']}: `{row['offHamming']}` -> `{row['preferHamming']}`"
            )
    if delta["regressed"]:
        lines.append("")
        regressed_count = len(delta["regressed"])
        suffix = (
            f"first 20 of {regressed_count}; see JSON for all"
            if regressed_count > 20
            else str(regressed_count)
        )
        lines.append(f"Regressed rows ({suffix}):")
        for row in delta["regressed"][:20]:
            lines.append(
                f"- Set {row['setId']}: `{row['offHamming']}` -> `{row['preferHamming']}`"
            )

    rows = payload["rows"]
    lines.extend([
        "",
        "## Per-Pair Snapshot",
        "",
        "| Set | Off hamming | Prefer hamming | Prefer selected sides | Prefer trace statuses |",
        "|---|---:|---:|---:|---|",
    ])
    for row in rows:
        prefer = row["modes"]["prefer"]
        traces = prefer.get("sideTraces", {})
        selected_count = sum(1 for trace in traces.values() if trace and trace.get("selected"))
        statuses = {
            side: trace.get("status")
            for side, trace in traces.items()
            if trace
        }
        lines.append(
            f"| {row['setId']} | {row['modes']['off'].get('hamming')} | "
            f"{prefer.get('hamming')} | {selected_count} | `{statuses}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _discover_tasks(manifest: Path, only_sets: Optional[Iterable[str]]) -> List[PairTask]:
    tasks = load_corpus_tasks(manifest)
    tasks.extend(discover_additional_tasks({task.set_id for task in tasks}))
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

    tasks = _discover_tasks(args.manifest, args.only_sets)
    sess = new_session("u2net")
    rows: List[Dict[str, Any]] = []
    print(f"validating {len(tasks)} pairs", file=sys.stderr)
    for index, task in enumerate(tasks, 1):
        try:
            row = _evaluate_pair(task, sess)
        except Exception as exc:  # noqa: BLE001
            row = {
                "setId": task.set_id,
                "source": task.source,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        rows.append(row)
        if "modes" not in row:
            print(f"  [{index}/{len(tasks)}] set {task.set_id}: ERROR", file=sys.stderr, flush=True)
        else:
            off_h = row["modes"]["off"].get("hamming")
            pref_h = row["modes"]["prefer"].get("hamming")
            print(
                f"  [{index}/{len(tasks)}] set {task.set_id}: off={off_h} prefer={pref_h}",
                file=sys.stderr,
                flush=True,
            )

    scored_rows = [row for row in rows if "modes" in row]
    payload = {
        "schema": "hull_label_tier1_shadow_validation_v1",
        "source": {
            "tool": "tools/validate_hull_label_tier1_shadow.py",
            "git_sha": _git_head_sha(),
            "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "manifest": _rel(args.manifest),
            "mode_order": list(MODES),
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
