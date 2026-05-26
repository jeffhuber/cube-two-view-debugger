#!/usr/bin/env python3
"""Diagnose pair-level mask-threshold selection through color repair.

The current hull-label selector chooses one alpha threshold independently per
side, using per-side geometry/sticker score. Set 14 shows the limitation:
side A threshold 160 has the lowest local sticker score, but threshold 64
produces a cube-level exact repair once A and B are assembled together.

This diagnostic enumerates accepted threshold pairs for each A/B image pair,
infers yaw, runs deterministic color/count/legal repair, and compares:

* current per-side selector threshold pair;
* best pair selected by production-available signals only;
* oracle-best pair by ground-truth hamming, for diagnostic ceiling only.

Diagnostic-only: no production behavior change.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_recognition_pair import parse_ground_truth as parse_pair_ground_truth  # noqa: E402
from tools.diagnose_hull_label_color_repair import (  # noqa: E402
    DEFAULT_MANIFEST,
    FIXER_MAX_SIDE,
    _fit_hull_side,
    _git_head_sha,
    _hull_label_center_yaw_source,
    _load_fixer_processing_image,
    _rel,
)
from tools.extract_color_samples import PairTask, load_corpus_tasks  # noqa: E402
from tools.global_cube_model import _slot_center_faces_from_rectified  # noqa: E402
from tools.hull_label_color_repair import repair_from_hull_label_fits  # noqa: E402
from tools.rectify_faces import DEFAULT_FACE_SIZE  # noqa: E402
from tools.rectify_via_hull_labels import (  # noqa: E402
    DEFAULT_MASK_THRESHOLDS,
    select_hull_label_threshold_fit,
)


DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "pair_threshold_repair_diagnostic.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "PAIR_THRESHOLD_REPAIR_DIAGNOSTIC.md"


def _method_summary(method: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not method:
        return {"status": "missing"}
    keys = (
        "method",
        "status",
        "hamming",
        "stickersCorrect",
        "validState",
        "countBalanced",
        "confidence",
        "repairMoveCount",
        "repairCost",
        "repairChanges",
    )
    return {key: method.get(key) for key in keys if key in method}


def _payload_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    methods = payload.get("methods") or {}
    recommended = payload.get("recommended") or {}
    return {
        "status": payload.get("status"),
        "recommendedMethod": payload.get("recommendedMethod"),
        "recommended": _method_summary(recommended),
        "canonicalCount": _method_summary(methods.get("canonical_count_repaired")),
        "conservativeLegal": _method_summary(methods.get("conservative_legal_repaired")),
        "guardedBroadLegal": _method_summary(methods.get("guarded_broad_legal_repaired")),
        "broadLegal": _method_summary(methods.get("broad_legal_repaired")),
    }


def _method_rank(payload: Mapping[str, Any]) -> Tuple[int, int, float, int, int]:
    """Rank a threshold-pair payload using production-available signals.

    Lower is better. The rank deliberately avoids ground-truth hamming.
    """
    methods = payload.get("methods") or {}
    canonical = methods.get("canonical_count_repaired") or {}
    guarded = methods.get("guarded_broad_legal_repaired") or {}
    conservative = methods.get("conservative_legal_repaired") or {}
    recommended = payload.get("recommended") or {}

    if canonical.get("validState"):
        tier = 0
        primary = int(canonical.get("repairMoveCount") or 0)
        cost = 0.0
        changes = primary
    elif conservative.get("validState"):
        tier = 1
        primary = int(conservative.get("repairChanges") or conservative.get("repairMoveCount") or 0)
        cost = float(conservative.get("repairCost") or 0.0)
        changes = primary
    elif guarded.get("validState"):
        tier = 2
        primary = int(guarded.get("repairChanges") or guarded.get("repairMoveCount") or 0)
        cost = float(guarded.get("repairCost") or 0.0)
        changes = primary
    elif canonical.get("countBalanced"):
        tier = 3
        primary = int(canonical.get("repairMoveCount") or 99)
        cost = 0.0
        changes = primary
    else:
        tier = 4
        primary = int(recommended.get("repairMoveCount") or 99)
        cost = float(recommended.get("repairCost") or 999.0)
        changes = int(recommended.get("repairChanges") or primary)
    return (tier, primary, cost, changes, int(payload.get("yawQuarterTurns") or 0))


def choose_pair_by_production_signals(combos: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    accepted = [combo for combo in combos if combo.get("status") == "assembled"]
    if not accepted:
        return None
    return min(
        accepted,
        key=lambda combo: (
            tuple(combo.get("productionRank") or (99, 99, 999.0, 99, 99)),
            float(combo.get("stickerScoreTotal") or 999999.0),
            int(combo["thresholds"]["A"]),
            int(combo["thresholds"]["B"]),
        ),
    )


def choose_guarded_pair(
    *,
    current_combo: Optional[Mapping[str, Any]],
    current_eval: Mapping[str, Any],
    aggressive_pair: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Choose threshold pair with a conservative production gate.

    If the current per-side selector already yields a valid recommended state,
    do not switch. Set 73 shows why: alternate threshold pairs can also be
    legal but encode the wrong cube. Pair search is valuable when the current
    selected geometry cannot assemble a valid cube, as in Set 14.
    """
    current_rec = (
        current_eval.get("summary", {})
        .get("recommended", {})
    )
    if current_rec.get("validState") and current_combo is not None:
        out = dict(current_combo)
        out["selectionReason"] = "kept_current_valid_repair"
        return out
    if aggressive_pair is not None:
        out = dict(aggressive_pair)
        out["selectionReason"] = "current_invalid_selected_best_pair"
        return out
    return None


def choose_pair_by_oracle(combos: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    candidates = [
        combo for combo in combos
        if combo.get("status") == "assembled"
        and combo.get("summary", {}).get("recommended", {}).get("hamming") is not None
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda combo: (
            int(combo["summary"]["recommended"]["hamming"]),
            tuple(combo.get("productionRank") or (99, 99, 999.0, 99, 99)),
            float(combo.get("stickerScoreTotal") or 999999.0),
        ),
    )


def _fit_threshold_candidates(
    image_path: Path,
    side: str,
    sess: Any,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    from rembg import remove  # noqa: E402

    image = _load_fixer_processing_image(image_path)
    rgba = remove(image, session=sess).convert("RGBA")
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    fits: Dict[int, Dict[str, Any]] = {}
    traces: Dict[str, Any] = {
        "image": _rel(image_path),
        "maxSide": FIXER_MAX_SIDE,
        "thresholds": [int(value) for value in DEFAULT_MASK_THRESHOLDS],
        "candidates": [],
    }
    for threshold in DEFAULT_MASK_THRESHOLDS:
        selection = select_hull_label_threshold_fit(
            image,
            alpha,
            side,
            thresholds=[int(threshold)],
            face_size_px=DEFAULT_FACE_SIZE,
        )
        trace = dict(selection.trace)
        traces["candidates"].append({
            "threshold": int(threshold),
            "accepted": bool(selection.fit is not None),
            "stickerScoreTotal": trace.get("sticker_score_total"),
            "hardFailures": trace.get("hard_failures") or [],
            "warnings": trace.get("warnings") or [],
            "vertexSource": trace.get("vertex_source"),
            "vertexCloudSpreadNorm": trace.get("vertex_cloud_spread_norm"),
            "projectiveResidualNorm": trace.get("projective_residual_norm"),
        })
        if selection.fit is None:
            continue
        trace["slot_center_faces"] = _slot_center_faces_from_rectified(selection.fit.rectified_faces)
        fits[int(threshold)] = {
            "image": image,
            "fit": selection.fit,
            "model": selection.fit,
            "trace": trace,
        }
    return fits, traces


def _current_side_selection(task: PairTask, sess: Any) -> Dict[str, Dict[str, Any]]:
    return {
        "A": _fit_hull_side(task.image_a, "A", sess),
        "B": _fit_hull_side(task.image_b, "B", sess),
    }


def _evaluate_side_fits(
    side_fits: Mapping[str, Any],
    gt_state: str,
) -> Dict[str, Any]:
    yaw = _hull_label_center_yaw_source(side_fits)
    if not yaw.get("accepted"):
        return {
            "status": "yaw_unavailable",
            "yawInference": yaw,
        }
    payload = repair_from_hull_label_fits(
        side_fits=side_fits,
        yaw_quarter_turns=int(yaw["yawQuarterTurns"]),
        gt_state=gt_state,
    )
    return {
        "status": "assembled",
        "yawInference": yaw,
        "payload": payload,
        "summary": _payload_summary(payload),
    }


def _compact_combo(
    *,
    threshold_a: int,
    threshold_b: int,
    evaluation: Mapping[str, Any],
    fit_a: Mapping[str, Any],
    fit_b: Mapping[str, Any],
) -> Dict[str, Any]:
    if evaluation.get("status") != "assembled":
        return {
            "thresholds": {"A": threshold_a, "B": threshold_b},
            "status": evaluation.get("status"),
            "yawInference": evaluation.get("yawInference"),
        }
    payload = evaluation["payload"]
    rank = _method_rank(payload)
    score_total = (
        float((fit_a.get("trace") or {}).get("sticker_score_total") or 0.0)
        + float((fit_b.get("trace") or {}).get("sticker_score_total") or 0.0)
    )
    return {
        "thresholds": {"A": threshold_a, "B": threshold_b},
        "status": "assembled",
        "yawInference": evaluation.get("yawInference"),
        "productionRank": list(rank),
        "stickerScoreTotal": round(score_total, 2),
        "summary": evaluation["summary"],
    }


def _evaluate_pair(task: PairTask, sess: Any) -> Dict[str, Any]:
    _sha, _raw_state, gt_state, _canonicalized = parse_pair_ground_truth(str(task.ground_truth))
    current_fits = _current_side_selection(task, sess)
    current_eval = _evaluate_side_fits(current_fits, gt_state)
    current_summary = current_eval.get("summary", {})
    current_thresholds = {
        side: (current_fits[side].get("trace") or {}).get("selected_mask_threshold")
        for side in ("A", "B")
    }

    fits_a, trace_a = _fit_threshold_candidates(task.image_a, "A", sess)
    fits_b, trace_b = _fit_threshold_candidates(task.image_b, "B", sess)
    combos: List[Dict[str, Any]] = []
    for threshold_a, fit_a in sorted(fits_a.items()):
        for threshold_b, fit_b in sorted(fits_b.items()):
            evaluation = _evaluate_side_fits({"A": fit_a, "B": fit_b}, gt_state)
            combos.append(
                _compact_combo(
                    threshold_a=threshold_a,
                    threshold_b=threshold_b,
                    evaluation=evaluation,
                    fit_a=fit_a,
                    fit_b=fit_b,
                )
            )

    aggressive = choose_pair_by_production_signals(combos)
    current_combo = next(
        (
            combo for combo in combos
            if combo.get("thresholds") == current_thresholds
            and combo.get("status") == "assembled"
        ),
        None,
    )
    selected = choose_guarded_pair(
        current_combo=current_combo,
        current_eval=current_eval,
        aggressive_pair=aggressive,
    )
    oracle = choose_pair_by_oracle(combos)
    top_combos = sorted(
        [combo for combo in combos if combo.get("status") == "assembled"],
        key=lambda combo: (
            tuple(combo.get("productionRank") or (99, 99, 999.0, 99, 99)),
            int((combo.get("summary") or {}).get("recommended", {}).get("hamming") or 999),
            float(combo.get("stickerScoreTotal") or 999999.0),
        ),
    )[:8]

    return {
        "setId": task.set_id,
        "source": task.source,
        "images": {
            "A": _rel(task.image_a),
            "B": _rel(task.image_b),
            "groundTruth": _rel(task.ground_truth),
        },
        "current": {
            "thresholds": current_thresholds,
            "status": current_eval.get("status"),
            "yawInference": current_eval.get("yawInference"),
            "summary": current_summary,
        },
        "aggressivePairSelected": aggressive,
        "pairSelected": selected,
        "oracleBest": oracle,
        "acceptedThresholdCounts": {"A": len(fits_a), "B": len(fits_b), "pairs": len(combos)},
        "thresholdDiagnostics": {"A": trace_a, "B": trace_b},
        "topCombos": top_combos,
    }


def _hamming(row: Mapping[str, Any], path: Sequence[str]) -> Optional[int]:
    cur: Any = row
    for key in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur if isinstance(cur, int) else None


def _valid(row: Mapping[str, Any], path: Sequence[str]) -> bool:
    cur: Any = row
    for key in path:
        if not isinstance(cur, Mapping):
            return False
        cur = cur.get(key)
    return bool(cur)


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def stats_for(path: Sequence[str]) -> Dict[str, Any]:
        hammings = [_hamming(row, path) for row in rows]
        present = [value for value in hammings if value is not None]
        return {
            "assembled": len(present),
            "exact": sum(1 for value in present if value == 0),
            "within3": sum(1 for value in present if value <= 3),
            "legal": sum(1 for row in rows if _valid(row, path[:-1] + ("validState",))),
            "hammingDistribution": {
                str(value): sum(1 for item in present if item == value)
                for value in sorted(set(present))
            },
        }

    current = stats_for(("current", "summary", "recommended", "hamming"))
    aggressive = stats_for(("aggressivePairSelected", "summary", "recommended", "hamming"))
    selected = stats_for(("pairSelected", "summary", "recommended", "hamming"))
    oracle = stats_for(("oracleBest", "summary", "recommended", "hamming"))
    changed = []
    for row in rows:
        cur_h = _hamming(row, ("current", "summary", "recommended", "hamming"))
        sel_h = _hamming(row, ("pairSelected", "summary", "recommended", "hamming"))
        if cur_h != sel_h:
            changed.append({
                "setId": row.get("setId"),
                "currentHamming": cur_h,
                "selectedHamming": sel_h,
                "currentThresholds": row.get("current", {}).get("thresholds"),
                "selectedThresholds": row.get("pairSelected", {}).get("thresholds"),
            })
    return {
        "currentPerSide": current,
        "aggressivePairSelected": aggressive,
        "pairSelected": selected,
        "oracleBest": oracle,
        "changedRows": changed,
    }


def render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Pair-level threshold repair diagnostic",
        "",
        "Diagnostic-only. This report asks whether A/B mask thresholds should be",
        "chosen after yaw + deterministic repair, rather than independently per side.",
        "",
        f"Git head: `{payload['source']['git_sha']}`",
        f"Generated: `{payload['source']['generated_at_utc']}`",
        "",
        "## Summary",
        "",
        "| Selector | Assembled | Exact | Legal | <=3 stickers | Hamming distribution |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, key in (
        ("Current per-side selector", "currentPerSide"),
        ("Aggressive pair-selected by production signals", "aggressivePairSelected"),
        ("Guarded pair-selected by production signals", "pairSelected"),
        ("Oracle best threshold pair", "oracleBest"),
    ):
        row = summary[key]
        lines.append(
            f"| {label} | {row['assembled']} | {row['exact']} | {row['legal']} | "
            f"{row['within3']} | `{row['hammingDistribution']}` |"
        )
    lines.extend([
        "",
        "## Changed Rows",
        "",
        "| Set | Current thresholds | Selected thresholds | Current hamming | Selected hamming |",
        "|---|---|---|---:|---:|",
    ])
    changed = summary.get("changedRows") or []
    if not changed:
        lines.append("| _none_ |  |  |  |  |")
    else:
        for row in changed:
            lines.append(
                f"| {row['setId']} | `{row['currentThresholds']}` | "
                f"`{row['selectedThresholds']}` | {row['currentHamming']} | "
                f"{row['selectedHamming']} |"
            )
    lines.extend([
        "",
        "## Notes",
        "",
        "- The pair selector rank uses only production-available signals:",
        "  valid canonical count repair first, then legal repair candidates, then",
        "  balanced count repair, with repair moves/cost and sticker score as",
        "  tie-breakers.",
        "- The guarded selector keeps the current per-side threshold pair when it",
        "  already yields a valid repaired cube. It only switches threshold pairs",
        "  when current repair is invalid or unavailable.",
        "- `oracleBest` uses ground-truth hamming only to show the ceiling; it is",
        "  not a production selector.",
        "- If pair selection improves exact/legal without regressions, the next",
        "  production-shaped step is to fold the same pair-level search into the",
        "  hidden hull-label Fixer path behind a gate.",
    ])
    return "\n".join(lines) + "\n"


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
                "images": {
                    "A": _rel(task.image_a),
                    "B": _rel(task.image_b),
                    "groundTruth": _rel(task.ground_truth),
                },
                "error": f"{type(exc).__name__}: {exc}",
            })

    payload = {
        "schema": "pair_threshold_repair_diagnostic_v1",
        "source": {
            "tool": "tools/diagnose_pair_threshold_repair.py",
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
