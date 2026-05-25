#!/usr/bin/env python3
"""Validate the Hull-Label Tier 1 recognizer path on the corpus.

This is the production-shaped companion to
``tools/validate_hull_label_tier1_shadow.py``. That older tool scores face
rectifications with ground-truth-aided face assignment. This one runs the
recognizer path itself so we can see the real face identity, color repair,
fallback, and API-style diagnostics that cube-snap would see.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.dataset import parse_ground_truth as parse_dataset_ground_truth  # noqa: E402
from rubik_recognizer.recognizer import WhiteUpRecognizer  # noqa: E402
from rubik_recognizer.validation import validate_state  # noqa: E402
from tools.extract_color_samples import (  # noqa: E402
    PairTask,
    discover_additional_tasks,
    load_corpus_tasks,
)


DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "hull_label_tier1_recognizer_validation.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "HULL_LABEL_TIER1_RECOGNIZER_VALIDATION.md"
LOW_LEVEL_MODES = ("off", "shadow", "prefer_candidate")
REPORT_MODES = ("off", "shadow", "prefer_candidate", "prefer_effective")


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
    if not actual or len(actual) != len(expected):
        return None
    return sum(1 for a, e in zip(actual, expected) if a != e)


def _parse_ground_truth(path: Path) -> Dict[str, Any]:
    raw_bytes = path.read_bytes()
    parsed = json.loads(raw_bytes.decode("utf-8"))
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        raw_state = parsed[0].get("corrected")
    elif isinstance(parsed, dict):
        raw_state = parsed.get("corrected") or parsed.get("expectedState")
    else:
        raw_state = None
    if not isinstance(raw_state, str) or len(raw_state) != 54:
        raise ValueError(f"no 54-character corrected state in {path}")
    raw_state = raw_state.strip().upper()

    truth_map = parse_dataset_ground_truth(raw_bytes, path.name)
    if not truth_map:
        raise ValueError(f"dataset ground-truth parser returned no states for {path}")
    canonical_state = next(iter(truth_map.values())).strip().upper()
    return {
        "rawState": raw_state,
        "canonicalState": canonical_state,
        "canonicalized": canonical_state != raw_state,
    }


def _score_payload(payload: Mapping[str, Any], expected_state: str) -> Dict[str, Any]:
    state = payload.get("state")
    state_text = state if isinstance(state, str) else None
    hamming = _hamming(state_text, expected_state)
    validation = validate_state(state_text) if state_text else None
    failed_checks = payload.get("failedChecks")
    return {
        "status": payload.get("status"),
        "category": payload.get("recognitionCategory"),
        "categoryReason": payload.get("recognitionCategoryReason"),
        "reason": payload.get("reason"),
        "failedChecks": list(failed_checks) if isinstance(failed_checks, list) else [],
        "confidence": payload.get("confidence"),
        "candidateCount": payload.get("candidates"),
        "state": state_text,
        "hamming": hamming,
        "stickersCorrect": (54 - hamming) if hamming is not None else 0,
        "exactMatch": state_text == expected_state if state_text else False,
        "validState": bool(validation and validation.valid),
        "validationErrors": list(validation.errors) if validation else ["not_assembled"],
    }


def _signals(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    signals = payload.get("recognitionSignals")
    return signals if isinstance(signals, Mapping) else {}


def _tier1(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    item = _signals(payload).get("hullLabelTier1")
    return item if isinstance(item, Mapping) else None


def _yaw(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    item = _signals(payload).get("hullLabelTier1Yaw")
    return item if isinstance(item, Mapping) else None


def _side_trace(payload: Mapping[str, Any], image_key: str) -> Optional[Mapping[str, Any]]:
    tier1 = _tier1(payload)
    if not tier1:
        return None
    images = tier1.get("images")
    if not isinstance(images, Mapping):
        return None
    trace = images.get(image_key)
    return trace if isinstance(trace, Mapping) else None


def _candidate_selected(payload: Mapping[str, Any]) -> bool:
    return all(
        (trace := _side_trace(payload, key)) is not None
        and trace.get("status") == "accepted"
        and trace.get("selected") is True
        for key in ("imageA", "imageB")
    )


def _trace_snapshot(payload: Mapping[str, Any]) -> Dict[str, Any]:
    tier1 = _tier1(payload)
    yaw = _yaw(payload)
    return {
        "hullLabelTier1": tier1,
        "hullLabelTier1Yaw": yaw,
        "selected": _candidate_selected(payload),
    }


def _payload_summary(payload: Mapping[str, Any], expected_state: str) -> Dict[str, Any]:
    out = _score_payload(payload, expected_state)
    out.update(_trace_snapshot(payload))
    return out


def _effective_prefer_payload(
    off_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    expected_state: str,
) -> Dict[str, Any]:
    candidate_score = _score_payload(candidate_payload, expected_state)
    selected = candidate_score["status"] == "success" and _candidate_selected(candidate_payload)
    base = copy.deepcopy(candidate_payload if selected else off_payload)
    signals = dict(_signals(base))
    signals["hullLabelTier1Prefer"] = {
        "selected": selected,
        "fallbackToLegacy": not selected,
        "candidateStatus": candidate_score["status"],
        "candidateCategory": candidate_score["category"],
        "candidateReason": candidate_score["reason"],
        "candidateFailedChecks": candidate_score["failedChecks"],
        "candidateHamming": candidate_score["hamming"],
        "candidateExactMatch": candidate_score["exactMatch"],
        "candidateHullLabelTier1": _tier1(candidate_payload),
        "candidateHullLabelTier1Yaw": _yaw(candidate_payload),
    }
    base["recognitionSignals"] = signals
    return base


def _compact_mode(payload: Mapping[str, Any], expected_state: str) -> Dict[str, Any]:
    summary = _payload_summary(payload, expected_state)
    return {
        "status": summary["status"],
        "category": summary["category"],
        "categoryReason": summary["categoryReason"],
        "reason": summary["reason"],
        "failedChecks": summary["failedChecks"],
        "confidence": summary["confidence"],
        "candidateCount": summary["candidateCount"],
        "hamming": summary["hamming"],
        "stickersCorrect": summary["stickersCorrect"],
        "exactMatch": summary["exactMatch"],
        "validState": summary["validState"],
        "validationErrors": summary["validationErrors"],
        "hullLabelTier1": summary["hullLabelTier1"],
        "hullLabelTier1Yaw": summary["hullLabelTier1Yaw"],
        "hullLabelSelected": summary["selected"],
        "hullLabelTier1Prefer": _signals(payload).get("hullLabelTier1Prefer"),
    }


def _recognize_low_level(
    recognizer: WhiteUpRecognizer,
    image_a: bytes,
    image_b: bytes,
    mode: str,
) -> Dict[str, Any]:
    low_level_mode = "prefer" if mode == "prefer_candidate" else mode
    result = recognizer._recognize_with_hull_label_mode(  # noqa: SLF001 - diagnostic tool.
        image_a,
        image_b,
        low_level_mode,
    )
    return result.to_api_dict(include_overlays=False)


def _evaluate_pair(task: PairTask, recognizer: WhiteUpRecognizer) -> Dict[str, Any]:
    truth = _parse_ground_truth(task.ground_truth)
    expected = truth["canonicalState"]
    image_a = task.image_a.read_bytes()
    image_b = task.image_b.read_bytes()

    payloads = {
        mode: _recognize_low_level(recognizer, image_a, image_b, mode)
        for mode in LOW_LEVEL_MODES
    }
    payloads["prefer_effective"] = _effective_prefer_payload(
        payloads["off"],
        payloads["prefer_candidate"],
        expected,
    )
    return {
        "setId": task.set_id,
        "source": task.source,
        "images": {
            "A": _rel(task.image_a),
            "B": _rel(task.image_b),
            "groundTruth": _rel(task.ground_truth),
        },
        "groundTruth": truth,
        "modes": {
            mode: _compact_mode(payloads[mode], expected)
            for mode in REPORT_MODES
        },
    }


def _mode_summary(rows: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    mode_rows = [row["modes"][mode] for row in rows]
    total = len(mode_rows)
    exact = sum(1 for row in mode_rows if row.get("exactMatch"))
    legal = sum(1 for row in mode_rows if row.get("validState"))
    success = sum(1 for row in mode_rows if row.get("status") == "success")
    total_correct = sum(int(row.get("stickersCorrect") or 0) for row in mode_rows)
    status_counts = Counter(str(row.get("status")) for row in mode_rows)
    category_counts = Counter(str(row.get("category")) for row in mode_rows)
    return {
        "pairs": total,
        "success": success,
        "legal": legal,
        "exact": exact,
        "meanStickersCorrect": round(total_correct / total, 3) if total else None,
        "stickerAccuracy": round(total_correct / (54 * total), 6) if total else None,
        "statusCounts": dict(sorted(status_counts.items())),
        "categoryCounts": dict(category_counts.most_common()),
    }


def _prefer_delta_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    improved = []
    regressed = []
    same = []
    for row in rows:
        off = row["modes"]["off"]
        effective = row["modes"]["prefer_effective"]
        item = {
            "setId": row["setId"],
            "offHamming": off.get("hamming"),
            "preferHamming": effective.get("hamming"),
            "offExact": off.get("exactMatch"),
            "preferExact": effective.get("exactMatch"),
            "selected": effective.get("hullLabelTier1Prefer", {}).get("selected"),
        }
        off_h = item["offHamming"]
        pref_h = item["preferHamming"]
        if off_h is None and pref_h is None:
            same.append(item)
        elif off_h is None:
            improved.append(item)
        elif pref_h is None:
            regressed.append(item)
        elif off_h == pref_h:
            same.append(item)
        elif pref_h < off_h:
            improved.append(item)
        else:
            regressed.append(item)
    return {
        "improved": improved,
        "regressed": regressed,
        "sameCount": len(same),
    }


def _trace_summary(rows: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    status_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    hard_failure_counts: Counter[str] = Counter()
    selected = 0
    accepted = 0
    yaw_counts: Counter[str] = Counter()
    yaw_best_counts: Counter[str] = Counter()
    vertex_source_counts: Counter[str] = Counter()
    for row in rows:
        mode_row = row["modes"][mode]
        yaw = mode_row.get("hullLabelTier1Yaw")
        if isinstance(yaw, Mapping):
            yaw_counts[str(yaw.get("status"))] += 1
            best = yaw.get("bestYawQuarterTurns")
            if best is not None:
                yaw_best_counts[str(best)] += 1
        tier1 = mode_row.get("hullLabelTier1")
        images = tier1.get("images") if isinstance(tier1, Mapping) else None
        if not isinstance(images, Mapping):
            continue
        for key in ("imageA", "imageB"):
            trace = images.get(key)
            if not isinstance(trace, Mapping):
                continue
            status_counts[str(trace.get("status"))] += 1
            if trace.get("accepted"):
                accepted += 1
            if trace.get("selected"):
                selected += 1
            source = trace.get("vertex_source")
            if isinstance(source, str):
                vertex_source_counts[source] += 1
            warning_counts.update(str(item) for item in trace.get("warnings") or [])
            hard_failure_counts.update(str(item) for item in trace.get("hard_failures") or [])
    return {
        "sideTraces": sum(status_counts.values()),
        "acceptedSides": accepted,
        "selectedSides": selected,
        "sideStatusCounts": dict(sorted(status_counts.items())),
        "yawStatusCounts": dict(sorted(yaw_counts.items())),
        "yawBestCounts": dict(sorted(yaw_best_counts.items())),
        "vertexSourceCounts": dict(vertex_source_counts.most_common()),
        "warningCounts": dict(warning_counts.most_common()),
        "hardFailureCounts": dict(hard_failure_counts.most_common()),
    }


def _capture_guidance_for_mode(row: Mapping[str, Any], mode: str) -> List[str]:
    guidance = set()
    mode_row = row["modes"][mode]
    checks = set(str(item) for item in mode_row.get("failedChecks") or [])
    if "missing_side_face_coverage" in checks:
        guidance.add("side-center coverage collapsed; inspect yaw/slot-to-WCA assignment")
    if any("U_anchor" in check or "D_anchor" in check for check in checks):
        guidance.add("U/D anchor not reliable; ensure A is white-up and B is yellow-up")
    if any("no_reliable_face_triple" in check for check in checks):
        guidance.add("visible 3-face grid not reliable; improve framing/focus")
    if "no_legal_state" in checks:
        guidance.add("sticker colors assemble to no legal state; inspect color repair/conflicts")
    if any(check.endswith("_center_invalid") or check.endswith("_count_not_9") for check in checks):
        guidance.add("face counts/centers invalid; inspect color classifier and slot assignment")

    tier1 = mode_row.get("hullLabelTier1")
    images = tier1.get("images") if isinstance(tier1, Mapping) else None
    if isinstance(images, Mapping):
        for trace in images.values():
            if not isinstance(trace, Mapping):
                continue
            messages = list(trace.get("hard_failures") or []) + list(trace.get("warnings") or [])
            for message in messages:
                text = str(message)
                if "projective_residual" in text:
                    guidance.add("hull/projective residual high; avoid background edges and steep tilt")
                if "vertex_cloud_spread" in text:
                    guidance.add("vertex estimates disagree; reduce perspective tilt")
                if "sticker_score" in text:
                    guidance.add("rectified sticker score high; improve lighting/focus/glare")
    return sorted(guidance)


def _fallback_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = []
    fallback = []
    candidate_failed_checks: Counter[str] = Counter()
    candidate_categories: Counter[str] = Counter()
    guidance_counts: Counter[str] = Counter()
    for row in rows:
        effective = row["modes"]["prefer_effective"]
        decision = effective.get("hullLabelTier1Prefer")
        if not isinstance(decision, Mapping):
            continue
        if decision.get("selected"):
            selected.append(row["setId"])
        else:
            fallback.append(row["setId"])
        candidate_categories[str(decision.get("candidateCategory"))] += 1
        candidate_failed_checks.update(str(item) for item in decision.get("candidateFailedChecks") or [])
        guidance_counts.update(_capture_guidance_for_mode(row, "prefer_candidate"))
    return {
        "selectedSetIds": selected,
        "fallbackSetIds": fallback,
        "candidateCategoryCounts": dict(candidate_categories.most_common()),
        "candidateFailedCheckCounts": dict(candidate_failed_checks.most_common()),
        "captureGuidanceCounts": dict(guidance_counts.most_common()),
    }


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "pairCount": len(rows),
        "byMode": {mode: _mode_summary(rows, mode) for mode in REPORT_MODES},
        "shadowTrace": _trace_summary(rows, "shadow"),
        "preferCandidateTrace": _trace_summary(rows, "prefer_candidate"),
        "preferVsOff": _prefer_delta_summary(rows),
        "preferFallback": _fallback_summary(rows),
    }


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _top_items(items: Mapping[str, Any], limit: int = 8) -> List[str]:
    ranked = sorted(
        items.items(),
        key=lambda item: (-int(item[1]) if isinstance(item[1], int) else 0, str(item[0])),
    )
    return [f"`{key}`: `{value}`" for key, value in ranked[:limit]]


def render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Hull-Label Tier 1 Recognizer Validation",
        "",
        "## Purpose",
        "",
        "This report runs the recognizer itself on corpus A+B pairs after the",
        "Hull-Label Tier 1 path and direct yaw assembly wiring. Unlike the",
        "geometry-only shadow validation, this includes the production-shaped",
        "face identity, color repair, legal-state, fallback, and diagnostic",
        "signals that cube-snap would consume.",
        "",
        f"Git head: `{payload['source']['git_sha']}`",
        f"Generated: `{payload['source']['generated_at_utc']}`",
        "",
        "## Summary By Mode",
        "",
        "| Mode | Pairs | Success | Legal | Exact | Mean stickers | Sticker acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in REPORT_MODES:
        row = summary["byMode"][mode]
        mean = row["meanStickersCorrect"]
        lines.append(
            f"| `{mode}` | {row['pairs']} | {row['success']} | {row['legal']} | "
            f"{row['exact']} | {mean if mean is not None else 'n/a'} | "
            f"{_pct(row['stickerAccuracy'])} |"
        )
    lines.extend([
        "",
        "`prefer_candidate` is the raw hull-label recognizer candidate. Rejected",
        "candidate rows contribute 0 recognized stickers to its aggregate score;",
        "`prefer_effective` is the production-shaped result after falling back to",
        "legacy unless both hull-label sides are selected and the candidate succeeds.",
    ])

    off_summary = summary["byMode"]["off"]
    shadow_summary = summary["byMode"]["shadow"]
    prefer_summary = summary["byMode"]["prefer_effective"]
    fallback = summary["preferFallback"]
    candidate_trace = summary["preferCandidateTrace"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Shadow mode is output-identical to legacy in aggregate: "
        f"`{shadow_summary['exact']}` exact / `{shadow_summary['success']}` success "
        f"versus legacy `{off_summary['exact']}` exact / `{off_summary['success']}` success.",
        f"- Hull-label geometry/yaw is usually available: "
        f"`{candidate_trace['acceptedSides']}/{candidate_trace['sideTraces']}` sides accepted "
        f"and yaw accepted on `{candidate_trace['yawStatusCounts'].get('accepted', 0)}` pairs.",
        f"- Effective prefer selected `{len(fallback['selectedSetIds'])}/{summary['pairCount']}` rows, "
        f"improved `{len(summary['preferVsOff']['improved'])}`, regressed "
        f"`{len(summary['preferVsOff']['regressed'])}`, and moved exact solves from "
        f"`{off_summary['exact']}` to `{prefer_summary['exact']}`.",
        "- Default-on prefer is still premature. The remaining bottleneck is not",
        "  hull geometry; it is color/slot-to-WCA face identity after rectification.",
    ])

    delta = summary["preferVsOff"]
    lines.extend([
        "",
        "## Prefer Effective Versus Legacy",
        "",
        f"- Improved hamming: `{len(delta['improved'])}`",
        f"- Regressed hamming: `{len(delta['regressed'])}`",
        f"- Same hamming/incomplete status: `{delta['sameCount']}`",
        f"- Hull-label selected rows: `{len(fallback['selectedSetIds'])}`",
        f"- Fallback rows: `{len(fallback['fallbackSetIds'])}`",
    ])
    if fallback["selectedSetIds"]:
        lines.append(f"- Selected setIds: `{', '.join(fallback['selectedSetIds'])}`")
    if delta["improved"]:
        lines.append("")
        lines.append("Improved rows:")
        for row in delta["improved"][:20]:
            lines.append(
                f"- Set {row['setId']}: `{row['offHamming']}` -> `{row['preferHamming']}`"
            )
    if delta["regressed"]:
        lines.append("")
        lines.append("Regressed rows:")
        for row in delta["regressed"][:20]:
            lines.append(
                f"- Set {row['setId']}: `{row['offHamming']}` -> `{row['preferHamming']}`"
            )

    lines.extend([
        "",
        "## Hull-Label Trace",
        "",
    ])
    for key, title in (
        ("shadowTrace", "Shadow"),
        ("preferCandidateTrace", "Prefer Candidate"),
    ):
        trace = summary[key]
        lines.extend([
            f"### {title}",
            "",
            f"- Side traces: `{trace['sideTraces']}`",
            f"- Accepted sides: `{trace['acceptedSides']}`",
            f"- Selected sides: `{trace['selectedSides']}`",
            f"- Side statuses: `{trace['sideStatusCounts']}`",
            f"- Yaw statuses: `{trace['yawStatusCounts']}`",
            f"- Best-yaw counts: `{trace['yawBestCounts']}`",
            f"- Vertex sources: `{trace['vertexSourceCounts']}`",
        ])
        if trace["hardFailureCounts"]:
            lines.append(f"- Hard failures: {', '.join(_top_items(trace['hardFailureCounts']))}")
        if trace["warningCounts"]:
            lines.append(f"- Warnings: {', '.join(_top_items(trace['warningCounts']))}")
        lines.append("")

    lines.extend([
        "## Candidate Fallback Diagnostics",
        "",
        f"- Candidate categories: `{fallback['candidateCategoryCounts']}`",
    ])
    if fallback["candidateFailedCheckCounts"]:
        lines.append(
            f"- Candidate failed checks: {', '.join(_top_items(fallback['candidateFailedCheckCounts'], 12))}"
        )
    if fallback["captureGuidanceCounts"]:
        lines.append("")
        lines.append("Capture / rollout guidance buckets:")
        guidance_items = sorted(
            fallback["captureGuidanceCounts"].items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
        for text, count in guidance_items:
            lines.append(f"- `{count}`: {text}")

    lines.extend([
        "",
        "## Per-Pair Snapshot",
        "",
        "| Set | Off | Prefer candidate | Prefer effective | Selected | Candidate category | Top failed checks |",
        "|---|---:|---:|---:|---:|---|---|",
    ])
    for row in payload["rows"]:
        off = row["modes"]["off"]
        cand = row["modes"]["prefer_candidate"]
        eff = row["modes"]["prefer_effective"]
        decision = eff.get("hullLabelTier1Prefer") or {}
        checks = decision.get("candidateFailedChecks") or []
        lines.append(
            f"| {row['setId']} | {off.get('hamming')} | {cand.get('hamming')} | "
            f"{eff.get('hamming')} | {decision.get('selected')} | "
            f"`{decision.get('candidateCategory')}` | `{', '.join(checks[:3])}` |"
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
        help="Render --report from --out-json without re-running recognition.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.render_only:
        payload = json.loads(args.out_json.read_text(encoding="utf-8"))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(payload), encoding="utf-8")
        print(f"wrote {args.report}", file=sys.stderr)
        return 0

    tasks = _discover_tasks(args.manifest, args.only_sets)
    recognizer = WhiteUpRecognizer()
    rows: List[Dict[str, Any]] = []
    print(f"validating {len(tasks)} pairs", file=sys.stderr)
    for index, task in enumerate(tasks, 1):
        try:
            row = _evaluate_pair(task, recognizer)
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
            cand_h = row["modes"]["prefer_candidate"].get("hamming")
            eff_h = row["modes"]["prefer_effective"].get("hamming")
            selected = (
                row["modes"]["prefer_effective"].get("hullLabelTier1Prefer") or {}
            ).get("selected")
            print(
                f"  [{index}/{len(tasks)}] set {task.set_id}: "
                f"off={off_h} candidate={cand_h} effective={eff_h} selected={selected}",
                file=sys.stderr,
                flush=True,
            )

    scored_rows = [row for row in rows if "modes" in row]
    payload = {
        "schema": "hull_label_tier1_recognizer_validation_v1",
        "source": {
            "tool": "tools/validate_hull_label_tier1_recognizer.py",
            "git_sha": _git_head_sha(),
            "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "manifest": _rel(args.manifest),
            "mode_order": list(REPORT_MODES),
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
