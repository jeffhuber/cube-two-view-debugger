#!/usr/bin/env python3
"""Validate hidden constrained `/api/recognize` modes on the corpus.

This is the production-boundary companion to
``tools/validate_constrained_inference_promotion.py``. It runs the recognizer
entrypoint that backs ``/api/recognize?hullLabelTier1=constrained`` and compares
that opt-in result with the unchanged legacy/default recognizer.

Diagnostic-only: this tool does not change the default recognizer path.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app as app_module  # noqa: E402
from rubik_recognizer.dataset import parse_ground_truth as parse_dataset_ground_truth  # noqa: E402
from rubik_recognizer.recognizer import WhiteUpRecognizer  # noqa: E402
from rubik_recognizer.validation import validate_state  # noqa: E402
from tools.extract_color_samples import PairTask, load_corpus_tasks  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "constrained_recognize_mode_validation.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "CONSTRAINED_RECOGNIZE_MODE_VALIDATION.md"
MODES = ("legacy", "constrained")


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
    return sum(1 for left, right in zip(actual, expected) if left != right)


def _parse_ground_truth(path: Path) -> Dict[str, Any]:
    raw_bytes = path.read_bytes()
    parsed = json.loads(raw_bytes.decode("utf-8"))
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        raw_state = parsed[0].get("corrected")
    elif isinstance(parsed, dict):
        raw_state = parsed.get("corrected") or parsed.get("expectedState")
    else:
        raw_state = None
    if not isinstance(raw_state, str) or len(raw_state.strip()) != 54:
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


def _signals(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    signals = payload.get("recognitionSignals")
    return signals if isinstance(signals, Mapping) else {}


def _constrained_signal(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    signal = _signals(payload).get("constrainedInference")
    return signal if isinstance(signal, Mapping) else {}


def _score_payload(payload: Mapping[str, Any], expected_state: str) -> Dict[str, Any]:
    state = payload.get("state")
    state_text = state if isinstance(state, str) else None
    hamming = _hamming(state_text, expected_state)
    validation = validate_state(state_text) if state_text else None
    signal = _constrained_signal(payload)
    gate = signal.get("promotionGate") if isinstance(signal.get("promotionGate"), Mapping) else {}
    pair_threshold = (
        signal.get("pairThresholdSelection")
        if isinstance(signal.get("pairThresholdSelection"), Mapping)
        else {}
    )
    recommended = signal.get("recommended") if isinstance(signal.get("recommended"), Mapping) else {}
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
        "constrainedSignal": {
            "present": bool(signal),
            "selected": signal.get("selected"),
            "fallbackToLegacy": signal.get("fallbackToLegacy"),
            "status": signal.get("status"),
            "recommendedMethod": signal.get("recommendedMethod"),
            "recommended": dict(recommended),
            "promotionGate": {
                "accepted": gate.get("accepted"),
                "decision": gate.get("decision"),
                "rejectReasons": list(gate.get("rejectReasons") or []),
                "productionRank": gate.get("productionRank"),
            },
            "pairThresholdSelection": {
                "selectionReason": pair_threshold.get("selectionReason"),
                "currentThresholds": pair_threshold.get("currentThresholds"),
                "selectedThresholds": pair_threshold.get("selectedThresholds"),
            },
            "yawQuarterTurns": signal.get("yawQuarterTurns"),
            "yawSource": signal.get("yawSource"),
        },
    }


def _run_mode(
    recognizer: WhiteUpRecognizer,
    image_a: bytes,
    image_b: bytes,
    mode: str,
) -> Dict[str, Any]:
    if mode == "legacy":
        result = recognizer.recognize(image_a, image_b, hull_label_tier1_mode="off")
    elif mode == "constrained":
        result = app_module._recognize_with_constrained_inference_mode(  # noqa: SLF001 - diagnostic boundary.
            recognizer,
            image_a,
            image_b,
            "prefer",
        )
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"unknown mode {mode!r}")
    return result.to_api_dict(include_overlays=False)


def _evaluate_pair(task: PairTask, recognizer: WhiteUpRecognizer) -> Dict[str, Any]:
    truth = _parse_ground_truth(task.ground_truth)
    expected = truth["canonicalState"]
    image_a = task.image_a.read_bytes()
    image_b = task.image_b.read_bytes()
    payloads = {
        mode: _run_mode(recognizer, image_a, image_b, mode)
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
        "groundTruth": truth,
        "modes": {
            mode: _score_payload(payloads[mode], expected)
            for mode in MODES
        },
    }


def _mode_summary(rows: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    mode_rows = [row["modes"][mode] for row in rows]
    total = len(mode_rows)
    total_correct = sum(int(row.get("stickersCorrect") or 0) for row in mode_rows)
    hammings = [row.get("hamming") for row in mode_rows if isinstance(row.get("hamming"), int)]
    return {
        "pairs": total,
        "success": sum(1 for row in mode_rows if row.get("status") == "success"),
        "legal": sum(1 for row in mode_rows if row.get("validState")),
        "exact": sum(1 for row in mode_rows if row.get("exactMatch")),
        "within3": sum(1 for value in hammings if value <= 3),
        "meanStickersCorrect": round(total_correct / total, 3) if total else None,
        "stickerAccuracy": round(total_correct / (54 * total), 6) if total else None,
        "hammingDistribution": dict(sorted(Counter(str(value) for value in hammings).items())),
        "statusCounts": dict(sorted(Counter(str(row.get("status")) for row in mode_rows).items())),
        "categoryCounts": dict(Counter(str(row.get("category")) for row in mode_rows).most_common()),
    }


def _delta_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    improved: List[Dict[str, Any]] = []
    regressed: List[Dict[str, Any]] = []
    same: List[Dict[str, Any]] = []
    for row in rows:
        legacy = row["modes"]["legacy"]
        constrained = row["modes"]["constrained"]
        item = {
            "setId": row["setId"],
            "legacyHamming": legacy.get("hamming"),
            "constrainedHamming": constrained.get("hamming"),
            "legacyExact": legacy.get("exactMatch"),
            "constrainedExact": constrained.get("exactMatch"),
            "selected": constrained.get("constrainedSignal", {}).get("selected"),
            "recommendedMethod": constrained.get("constrainedSignal", {}).get("recommendedMethod"),
        }
        legacy_h = item["legacyHamming"]
        constrained_h = item["constrainedHamming"]
        legacy_score = 999 if legacy_h is None else int(legacy_h)
        constrained_score = 999 if constrained_h is None else int(constrained_h)
        if constrained_score < legacy_score:
            improved.append(item)
        elif constrained_score > legacy_score:
            regressed.append(item)
        else:
            same.append(item)
    return {
        "improved": improved,
        "regressed": regressed,
        "sameCount": len(same),
    }


def _constrained_signal_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = []
    fallback = []
    gate_accepted = []
    gate_rejected = []
    methods: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    selection_reasons: Counter[str] = Counter()
    thresholds_switched = 0
    for row in rows:
        mode = row["modes"]["constrained"]
        signal = mode.get("constrainedSignal") or {}
        if signal.get("selected") is True:
            selected.append(row["setId"])
        else:
            fallback.append(row["setId"])
        method = signal.get("recommendedMethod")
        if method:
            methods[str(method)] += 1
        gate = signal.get("promotionGate") if isinstance(signal.get("promotionGate"), Mapping) else {}
        if gate.get("accepted") is True:
            gate_accepted.append(row["setId"])
        else:
            gate_rejected.append(row["setId"])
            reject_reasons.update(str(item) for item in gate.get("rejectReasons") or [])
        pair = signal.get("pairThresholdSelection")
        if isinstance(pair, Mapping):
            selection_reasons[str(pair.get("selectionReason"))] += 1
            if pair.get("currentThresholds") != pair.get("selectedThresholds"):
                thresholds_switched += 1
    return {
        "selectedSetIds": selected,
        "fallbackSetIds": fallback,
        "gateAcceptedSetIds": gate_accepted,
        "gateRejectedSetIds": gate_rejected,
        "recommendedMethodCounts": dict(methods.most_common()),
        "gateRejectReasonCounts": dict(reject_reasons.most_common()),
        "selectionReasonCounts": dict(selection_reasons.most_common()),
        "thresholdSwitchCount": thresholds_switched,
    }


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "pairCount": len(rows),
        "byMode": {mode: _mode_summary(rows, mode) for mode in MODES},
        "constrainedSignals": _constrained_signal_summary(rows),
        "constrainedVsLegacy": _delta_summary(rows),
    }


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Constrained Recognize Mode Validation",
        "",
        "Diagnostic-only. This report runs the hidden",
        "`/api/recognize?hullLabelTier1=constrained` path on the corpus and",
        "compares it with the unchanged legacy/default recognizer path.",
        "",
        f"Git head: `{payload['source']['git_sha']}`",
        f"Generated: `{payload['source']['generated_at_utc']}`",
        f"Manifest: `{payload['source']['manifest']}`",
        "",
        "## Summary By Mode",
        "",
        "| Mode | Pairs | Success | Legal | Exact | <=3 | Mean stickers | Sticker acc | Hamming distribution |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode in MODES:
        row = summary["byMode"][mode]
        mean = row["meanStickersCorrect"]
        lines.append(
            f"| `{mode}` | {row['pairs']} | {row['success']} | {row['legal']} | "
            f"{row['exact']} | {row['within3']} | {mean if mean is not None else 'n/a'} | "
            f"{_pct(row['stickerAccuracy'])} | `{row['hammingDistribution']}` |"
        )

    signals = summary["constrainedSignals"]
    delta = summary["constrainedVsLegacy"]
    legacy = summary["byMode"]["legacy"]
    constrained = summary["byMode"]["constrained"]
    lines.extend([
        "",
        "## Gate Behavior",
        "",
        f"- Selected constrained candidate: `{len(signals['selectedSetIds'])}/{summary['pairCount']}`",
        f"- Fell back to legacy: `{len(signals['fallbackSetIds'])}/{summary['pairCount']}`",
        f"- Gate accepted: `{len(signals['gateAcceptedSetIds'])}/{summary['pairCount']}`",
        f"- Threshold switches: `{signals['thresholdSwitchCount']}`",
        f"- Recommended methods: `{signals['recommendedMethodCounts']}`",
        f"- Selection reasons: `{signals['selectionReasonCounts']}`",
    ])
    if signals["gateRejectReasonCounts"]:
        lines.append(f"- Gate reject reasons: `{signals['gateRejectReasonCounts']}`")

    lines.extend([
        "",
        "## Delta Versus Legacy",
        "",
        f"- Legacy exact: `{legacy['exact']}/{summary['pairCount']}`",
        f"- Constrained exact: `{constrained['exact']}/{summary['pairCount']}`",
        f"- Improved hamming: `{len(delta['improved'])}`",
        f"- Regressed hamming: `{len(delta['regressed'])}`",
        f"- Same hamming/incomplete status: `{delta['sameCount']}`",
    ])
    if delta["improved"]:
        lines.append("")
        lines.append("Improved rows:")
        for row in delta["improved"]:
            lines.append(
                f"- Set {row['setId']}: `{row['legacyHamming']}` -> "
                f"`{row['constrainedHamming']}` (`{row['recommendedMethod']}`)"
            )
    if delta["regressed"]:
        lines.append("")
        lines.append("Regressed rows:")
        for row in delta["regressed"]:
            lines.append(
                f"- Set {row['setId']}: `{row['legacyHamming']}` -> "
                f"`{row['constrainedHamming']}` (`{row['recommendedMethod']}`)"
            )

    lines.extend([
        "",
        "## Per-Pair Snapshot",
        "",
        "| Set | Legacy | Constrained | Selected | Gate | Method | Thresholds |",
        "|---|---:|---:|---:|---|---|---|",
    ])
    for row in payload["rows"]:
        legacy_row = row["modes"]["legacy"]
        constrained_row = row["modes"]["constrained"]
        signal = constrained_row.get("constrainedSignal") or {}
        gate = signal.get("promotionGate") or {}
        pair = signal.get("pairThresholdSelection") or {}
        lines.append(
            f"| {row['setId']} | {legacy_row.get('hamming')} | "
            f"{constrained_row.get('hamming')} | {signal.get('selected')} | "
            f"`{gate.get('accepted')}` | `{signal.get('recommendedMethod')}` | "
            f"`{pair.get('selectedThresholds')}` |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This report validates the hidden runtime mode added for staged",
        "  recognizer rollout. It does not flip the default `/api/recognize` path.",
        "- A clean result here means the shared constrained-inference gate is",
        "  behaving at the recognizer boundary, not just in offline repair traces.",
        "- Any default promotion should still run in shadow first on real traffic",
        "  and preserve fallback/manual-review behavior when the gate rejects.",
    ])
    return "\n".join(lines) + "\n"


def _load_tasks(manifest: Path, only_sets: Optional[Iterable[str]]) -> List[PairTask]:
    tasks = load_corpus_tasks(manifest)
    if only_sets:
        wanted = {str(item) for item in only_sets}
        tasks = [task for task in tasks if task.set_id in wanted]
    return [
        task for task in tasks
        if task.image_a.exists() and task.image_b.exists() and task.ground_truth.exists()
    ]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.render_only:
        payload = json.loads(args.out_json.read_text(encoding="utf-8"))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(payload), encoding="utf-8")
        print(f"wrote {args.report}")
        return 0

    tasks = _load_tasks(args.manifest, args.only_sets)
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
                "images": {
                    "A": _rel(task.image_a),
                    "B": _rel(task.image_b),
                    "groundTruth": _rel(task.ground_truth),
                },
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        rows.append(row)
        if "modes" not in row:
            print(f"  [{index}/{len(tasks)}] set {task.set_id}: ERROR", file=sys.stderr, flush=True)
        else:
            legacy_h = row["modes"]["legacy"].get("hamming")
            constrained_h = row["modes"]["constrained"].get("hamming")
            selected = row["modes"]["constrained"].get("constrainedSignal", {}).get("selected")
            print(
                f"  [{index}/{len(tasks)}] set {task.set_id}: "
                f"legacy={legacy_h} constrained={constrained_h} selected={selected}",
                file=sys.stderr,
                flush=True,
            )

    scored_rows = [row for row in rows if "modes" in row]
    payload = {
        "schema": "constrained_recognize_mode_validation_v1",
        "source": {
            "tool": "tools/validate_constrained_recognize_mode.py",
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
