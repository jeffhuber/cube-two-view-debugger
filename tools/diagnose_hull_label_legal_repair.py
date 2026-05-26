#!/usr/bin/env python3
"""Probe cubie-level legality repair on hull-label color evidence.

This diagnostic starts from the same hull-label rectified sticker samples as
``diagnose_hull_label_color_repair.py`` and asks the next constrained-inference
question: after count repair balances the six WCA face colors, can the existing
recognizer cubie-legality repair recover a valid cube using only Lab evidence?

Two legal-repair modes are intentionally compared:

* ``conservative_legal_repaired``: uses the normal component-sample alternative
  gates. This is the safer signal.
* ``broad_legal_repaired``: marks hull-label samples as grid samples, which lets
  the existing repair helper consider all color fallbacks. This is diagnostic
  only; it can prove that a legal solution is rankable, but it needs confidence
  gates before production use.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import (  # noqa: E402
    CANONICAL_RGB,
    CLASSIFIER_CANONICAL,
    classify_rgb_with_mode,
)
from rubik_recognizer.image_pipeline import Sticker  # noqa: E402
from rubik_recognizer.recognizer import _legal_repaired_state_from_faces  # noqa: E402
from rubik_recognizer.validation import FACE_ORDER, validate_state  # noqa: E402
from tools.audit_recognition_pair import parse_ground_truth as parse_pair_ground_truth  # noqa: E402
from tools.diagnose_hull_label_color_repair import (  # noqa: E402
    DEFAULT_MANIFEST,
    _fit_hull_side,
    _hull_label_center_yaw_source,
    _rel,
)
from tools.extract_color_samples import PairTask, load_corpus_tasks  # noqa: E402
from tools.hull_label_color_repair import (  # noqa: E402
    StickerObservation,
    evaluate_palette,
    sample_observations,
)


DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "hull_label_legal_repair_diagnostic.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "HULL_LABEL_LEGAL_REPAIR_DIAGNOSTIC.md"
GUARDED_BROAD_MAX_REPAIR_COST = 16.0
GUARDED_BROAD_MAX_REPAIR_CHANGES = 4


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _hamming(actual: Optional[str], expected: str) -> Optional[int]:
    if actual is None or len(actual) != len(expected):
        return None
    return sum(1 for got, want in zip(actual, expected) if got != want)


def _state_payload(state: Optional[str], gt_state: str) -> Dict[str, Any]:
    validation = validate_state(state) if state else None
    hamming = _hamming(state, gt_state)
    return {
        "state": state,
        "validState": bool(validation and validation.valid),
        "validationErrors": list(validation.errors) if validation else ["not_assembled"],
        "hamming": hamming,
        "stickersCorrect": (54 - hamming) if hamming is not None else None,
        "exactMatch": state == gt_state if state else False,
    }


def _observation_by_index(observations: Sequence[StickerObservation]) -> Dict[int, StickerObservation]:
    return {int(obs.index): obs for obs in observations}


def _faces_for_legal_repair(
    observations: Sequence[StickerObservation],
    *,
    source: str,
) -> Dict[str, List[List[Any]]]:
    by_index = _observation_by_index(observations)
    faces: Dict[str, List[List[Any]]] = {}
    for face in FACE_ORDER:
        matrix: List[List[Any]] = []
        for row in range(3):
            matrix_row: List[Any] = []
            for col in range(3):
                face_index = row * 3 + col
                index = FACE_ORDER.index(face) * 9 + face_index
                if face_index == 4:
                    matrix_row.append(face)
                    continue
                obs = by_index[index]
                match = classify_rgb_with_mode(obs.rgb, CLASSIFIER_CANONICAL, prototypes=CANONICAL_RGB)
                matrix_row.append(
                    Sticker(
                        id=index,
                        center=(0.0, 0.0),
                        bbox=(0.0, 0.0, 1.0, 1.0),
                        rgb=obs.rgb,
                        match=match,
                        area=1,
                        source=source,
                    )
                )
            matrix.append(matrix_row)
        faces[face] = matrix
    return faces


def _legal_repair_payload(
    observations: Sequence[StickerObservation],
    *,
    gt_state: str,
    source: str,
    baseline_state: Optional[str],
) -> Dict[str, Any]:
    if baseline_state and validate_state(baseline_state).valid:
        return {
            "status": "already_valid_count_repair",
            **_state_payload(baseline_state, gt_state),
            "repairCost": 0.0,
            "repairChanges": 0,
            "sourceMode": source,
        }
    faces = _faces_for_legal_repair(observations, source=source)
    result = _legal_repaired_state_from_faces(faces)
    if result is None:
        return {
            "status": "no_legal_repair",
            **_state_payload(None, gt_state),
            "repairCost": None,
            "repairChanges": None,
            "sourceMode": source,
        }
    state, cost, changes = result
    return {
        "status": "legal_repair_found",
        **_state_payload(state, gt_state),
        "repairCost": round(float(cost), 4),
        "repairChanges": int(changes),
        "sourceMode": source,
    }


def _guarded_broad_payload(broad_payload: Mapping[str, Any], *, gt_state: str) -> Dict[str, Any]:
    cost = broad_payload.get("repairCost")
    changes = broad_payload.get("repairChanges")
    accepted = (
        broad_payload.get("validState") is True
        and isinstance(cost, (int, float))
        and isinstance(changes, int)
        and float(cost) <= GUARDED_BROAD_MAX_REPAIR_COST
        and int(changes) <= GUARDED_BROAD_MAX_REPAIR_CHANGES
    )
    gate = {
        "maxRepairCost": GUARDED_BROAD_MAX_REPAIR_COST,
        "maxRepairChanges": GUARDED_BROAD_MAX_REPAIR_CHANGES,
        "accepted": accepted,
    }
    if accepted:
        out = dict(broad_payload)
        out["status"] = "accepted_guarded_broad_legal_repair"
        out["gate"] = gate
        return out
    return {
        "status": "rejected_guarded_broad_legal_repair",
        **_state_payload(None, gt_state),
        "repairCost": None,
        "repairChanges": None,
        "rejectedRepairCost": cost,
        "rejectedRepairChanges": changes,
        "sourceMode": broad_payload.get("sourceMode"),
        "gate": gate,
    }


def evaluate_observations(observations: Sequence[StickerObservation], *, gt_state: str) -> Dict[str, Any]:
    count_methods = evaluate_palette(
        observations=observations,
        palette=CANONICAL_RGB,
        prefix="canonical",
        gt_state=gt_state,
    )
    count_repaired = count_methods["canonical_count_repaired"]
    baseline_state = count_repaired.get("state")
    broad = _legal_repair_payload(
        observations,
        gt_state=gt_state,
        source="grid_sample",
        baseline_state=baseline_state,
    )
    return {
        "canonical_count_repaired": count_repaired,
        "conservative_legal_repaired": _legal_repair_payload(
            observations,
            gt_state=gt_state,
            source="hull_label_sample",
            baseline_state=baseline_state,
        ),
        "broad_legal_repaired": broad,
        "guarded_broad_legal_repaired": _guarded_broad_payload(broad, gt_state=gt_state),
    }


def _evaluate_pair(task: PairTask, sess: Any) -> Dict[str, Any]:
    _sha, raw_state, gt_state, canonicalized = parse_pair_ground_truth(str(task.ground_truth))
    side_fits = {
        "A": _fit_hull_side(task.image_a, "A", sess),
        "B": _fit_hull_side(task.image_b, "B", sess),
    }
    yaw_source = _hull_label_center_yaw_source(side_fits)
    yaw = yaw_source.get("yawQuarterTurns")
    if not isinstance(yaw, int):
        return {
            "setId": task.set_id,
            "source": task.source,
            "status": "missing_yaw",
            "images": {
                "A": _rel(task.image_a),
                "B": _rel(task.image_b),
                "groundTruth": _rel(task.ground_truth),
            },
            "groundTruthCanonicalized": canonicalized,
            "rawGroundTruthState": raw_state,
            "canonicalGroundTruthState": gt_state,
            "yawSource": yaw_source,
            "methods": {},
        }
    observations, panel_meta = sample_observations(
        side_fits=side_fits,
        yaw_quarter_turns=yaw,
    )
    return {
        "setId": task.set_id,
        "source": task.source,
        "status": "assembled",
        "images": {
            "A": _rel(task.image_a),
            "B": _rel(task.image_b),
            "groundTruth": _rel(task.ground_truth),
        },
        "groundTruthCanonicalized": canonicalized,
        "rawGroundTruthState": raw_state,
        "canonicalGroundTruthState": gt_state,
        "yawSource": yaw_source,
        "panelMeta": panel_meta,
        "methods": evaluate_observations(observations, gt_state=gt_state),
    }


def _method_summary(rows: Sequence[Mapping[str, Any]], method: str) -> Dict[str, Any]:
    assembled = [row for row in rows if method in row.get("methods", {})]
    values = [
        row["methods"][method]["hamming"]
        for row in assembled
        if isinstance(row["methods"][method].get("hamming"), int)
    ]
    costs = [
        row["methods"][method].get("repairCost")
        for row in assembled
        if isinstance(row["methods"][method].get("repairCost"), (int, float))
    ]
    changes = [
        row["methods"][method].get("repairChanges")
        for row in assembled
        if isinstance(row["methods"][method].get("repairChanges"), int)
    ]
    return {
        "assembled": len(assembled),
        "legal": sum(1 for row in assembled if row["methods"][method].get("validState")),
        "exact": sum(1 for row in assembled if row["methods"][method].get("exactMatch")),
        "within3": sum(1 for value in values if value <= 3),
        "medianHamming": statistics.median(values) if values else None,
        "hammingDistribution": dict(sorted(Counter(values).items())),
        "medianRepairCost": round(statistics.median(costs), 4) if costs else None,
        "medianRepairChanges": statistics.median(changes) if changes else None,
        "maxRepairChanges": max(changes) if changes else None,
    }


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    methods = (
        "canonical_count_repaired",
        "conservative_legal_repaired",
        "guarded_broad_legal_repaired",
        "broad_legal_repaired",
    )
    return {
        "pairCount": len(rows),
        "methods": {method: _method_summary(rows, method) for method in methods},
    }


def render_report(payload: Mapping[str, Any]) -> str:
    rows = payload["rows"]
    lines = [
        "# Hull-Label Legal Repair Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic probes the next constraint layer after deterministic",
        "9-per-color count repair: cubie legality. It reuses the existing",
        "recognizer cubie repair helper on hull-label rectified Lab evidence.",
        "",
        f"Git head: `{payload['source']['git_sha']}`",
        f"Generated: `{payload['source']['generated_at_utc']}`",
        "",
        "## Summary",
        "",
        "| Method | Assembled | Legal | Exact | <=3 stickers | Median hamming | Hamming distribution | Median repair cost | Median changes | Max changes |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for method, summary in payload["summary"]["methods"].items():
        lines.append(
            f"| `{method}` | {summary['assembled']} | {summary['legal']} | "
            f"{summary['exact']} | {summary['within3']} | {summary['medianHamming']} | "
            f"`{_format_distribution(summary['hammingDistribution'])}` | "
            f"{summary['medianRepairCost']} | {summary['medianRepairChanges']} | {summary['maxRepairChanges']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Conservative cubie repair is the safer signal: it only considers the",
        "  normal low-cost color alternatives allowed by the recognizer.",
        "- Broad cubie repair is diagnostic-only: it marks samples as grid samples",
        "  so the helper can consider all color fallbacks. A perfect broad result",
        "  proves the true legal state is rankable by existing constraints, but",
        "  it still needs cost/change/margin gates before production use.",
        f"- Guarded broad repair applies a provisional no-ground-truth gate to that",
        f"  same broad result: repair cost <= {GUARDED_BROAD_MAX_REPAIR_COST:g}",
        f"  and repair changes <= {GUARDED_BROAD_MAX_REPAIR_CHANGES}. This is still",
        "  diagnostic, but it estimates the slice that looks safe enough to consider",
        "  for production confidence gating.",
        "- This diagnostic does not use ground truth for selection. Ground truth is",
        "  only used after each candidate state is selected to compute hamming and",
        "  exact-match metrics.",
        "",
        "## Non-Exact / No-Repair Rows",
        "",
        "| Set | Count hamming | Conservative hamming | Conservative status | Guarded hamming | Guarded status | Broad hamming | Broad cost | Broad changes |",
        "|---:|---:|---:|---|---:|---|---:|---:|---:|",
    ])
    for row in rows:
        methods = row.get("methods", {})
        count_h = methods.get("canonical_count_repaired", {}).get("hamming")
        conservative = methods.get("conservative_legal_repaired", {})
        guarded = methods.get("guarded_broad_legal_repaired", {})
        broad = methods.get("broad_legal_repaired", {})
        if count_h == 0 and conservative.get("hamming") == 0 and guarded.get("hamming") == 0 and broad.get("hamming") == 0:
            continue
        lines.append(
            f"| {row['setId']} | {count_h} | {conservative.get('hamming')} | "
            f"`{conservative.get('status')}` | {guarded.get('hamming')} | "
            f"`{guarded.get('status')}` | {broad.get('hamming')} | "
            f"{broad.get('repairCost')} | {broad.get('repairChanges')} |"
        )
    return "\n".join(lines) + "\n"


def _format_distribution(distribution: Mapping[Any, Any]) -> str:
    if not distribution:
        return "{}"
    normalized = {int(key): int(value) for key, value in distribution.items()}
    return "{" + ", ".join(f"{key}: {normalized[key]}" for key in sorted(normalized)) + "}"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--only-sets", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    tasks = load_corpus_tasks(args.manifest)
    only = {str(item) for item in args.only_sets} if args.only_sets else None
    if only is not None:
        tasks = [task for task in tasks if task.set_id in only]
    try:
        from rembg import new_session
    except Exception as exc:  # noqa: BLE001
        print(f"failed to import rembg: {exc}", file=sys.stderr)
        return 2

    sess = new_session("u2net")
    rows: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] set {task.set_id}", flush=True)
        try:
            rows.append(_evaluate_pair(task, sess))
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "setId": task.set_id,
                "source": task.source,
                "status": "error",
                "images": {
                    "A": _rel(task.image_a),
                    "B": _rel(task.image_b),
                    "groundTruth": _rel(task.ground_truth),
                },
                "error": f"{type(exc).__name__}: {exc}",
                "methods": {},
            })
    payload = {
        "schema": "hull_label_legal_repair_diagnostic_v1",
        "source": {
            "tool": "tools/diagnose_hull_label_legal_repair.py",
            "manifest": _rel(args.manifest),
            "git_sha": _git_head_sha(),
            "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "summary": build_summary(rows),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
