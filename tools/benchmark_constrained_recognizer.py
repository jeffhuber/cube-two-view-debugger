#!/usr/bin/env python3
"""Benchmark local constrained recognizer latency variants on the corpus.

This is a controlled in-process counterpart to the deployed scorer. It runs the
same local images through the constrained recognizer with explicit variants
such as serial versus threaded hull fitting, captures stage timings from the
recognizer signal, and writes JSON plus an optional Markdown report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app as app_module  # noqa: E402
from rubik_recognizer.dataset import parse_ground_truth as parse_dataset_ground_truth  # noqa: E402
from rubik_recognizer.recognizer import WhiteUpRecognizer  # noqa: E402
from tools.extract_color_samples import PairTask, load_corpus_tasks  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_OUT = REPO_ROOT / "runs" / "constrained_recognizer_benchmark.json"
DEFAULT_REPORT = REPO_ROOT / "runs" / "constrained_recognizer_benchmark.md"
DEFAULT_TAIL_SETS = ("11", "14", "59", "65", "69")
DEFAULT_STAGES = (
    "recognizeTotal",
    "prepareConstrainedInput",
    "prepareTotal",
    "rembgSession",
    "rembgA",
    "rembgB",
    "hullFitWall",
    "hullFitA",
    "hullFitB",
    "selectGuardedPair",
    "selectGuardedPair.thresholdPairEnumeration",
    "selectGuardedPair.currentCanonicalProbeEvaluatePair",
    "selectGuardedPair.currentPairEvaluatePair",
    "selectGuardedPair.currentGuardSelection",
    "selectGuardedPair.lightPairEvaluatePair",
    "selectGuardedPair.lightPairSearch",
    "selectGuardedPair.canonicalLightFilter",
    "selectGuardedPair.lightGuardSelection",
    "selectGuardedPair.fullPairEvaluatePair",
    "selectGuardedPair.fullPairSearch",
    "selectGuardedPair.fullGuardSelection",
    "selectGuardedPair.selectedFullPairEvaluatePair",
    "selectGuardedPair.yawInference",
    "selectGuardedPair.repairCanonicalLight",
    "selectGuardedPair.repairCanonicalOnly",
    "selectGuardedPair.repairWithLegal",
    "selectGuardedPair.rankAndScore",
    "promotionGate",
    "buildCandidate",
)


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _expected_state(path: Path) -> str:
    truth = parse_dataset_ground_truth(path.read_bytes(), path.name)
    if not truth:
        raise ValueError(f"no parseable ground truth in {path}")
    return next(iter(truth.values())).strip().upper()


def _hamming(actual: Any, expected: str) -> Optional[int]:
    if not isinstance(actual, str) or len(actual) != len(expected):
        return None
    return sum(1 for left, right in zip(actual, expected) if left != right)


def _signal(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    signals = payload.get("recognitionSignals")
    if not isinstance(signals, Mapping):
        return {}
    signal = signals.get("constrainedInference")
    return signal if isinstance(signal, Mapping) else {}


def _stage_timings(signal: Mapping[str, Any]) -> Dict[str, float]:
    performance = signal.get("performance")
    if not isinstance(performance, Mapping):
        return {}
    timings = performance.get("stageTimingsMs")
    if not isinstance(timings, Mapping):
        return {}
    return {
        str(stage): round(float(ms), 2)
        for stage, ms in timings.items()
        if isinstance(ms, (int, float)) and not isinstance(ms, bool)
    }


def _pair_selection(signal: Mapping[str, Any]) -> Mapping[str, Any]:
    pair = signal.get("pairThresholdSelection")
    return pair if isinstance(pair, Mapping) else {}


def _run_once(
    *,
    recognizer: WhiteUpRecognizer,
    task: PairTask,
    hull_fit_mode: str,
    variant: str,
    max_side: int,
    iteration: int,
) -> Dict[str, Any]:
    expected = _expected_state(task.ground_truth)
    image_a = task.image_a.read_bytes()
    image_b = task.image_b.read_bytes()
    started = time.perf_counter()
    result = app_module._recognize_with_constrained_inference_mode(  # noqa: SLF001 - benchmark boundary.
        recognizer,
        image_a,
        image_b,
        "prefer",
        expected_state=expected,
        hull_fit_mode=hull_fit_mode,
        max_side=max_side,
    )
    wall_ms = round((time.perf_counter() - started) * 1000.0, 2)
    payload = result.to_api_dict(include_overlays=False)
    signal = _signal(payload)
    timings = _stage_timings(signal)
    pair = _pair_selection(signal)
    hamming = _hamming(payload.get("state"), expected)
    return {
        "setId": task.set_id,
        "variant": variant,
        "hullFitMode": hull_fit_mode,
        "maxSide": max_side,
        "iteration": iteration,
        "status": payload.get("status"),
        "recognitionCategory": payload.get("recognitionCategory"),
        "recommendedMethod": signal.get("recommendedMethod"),
        "selected": signal.get("selected"),
        "fallbackToLegacy": signal.get("fallbackToLegacy"),
        "hamming": hamming,
        "exactMatch": hamming == 0 if hamming is not None else False,
        "wallMs": wall_ms,
        "stageTimingsMs": timings,
        "pairThresholdSelection": {
            "selectionReason": pair.get("selectionReason"),
            "currentThresholds": pair.get("currentThresholds"),
            "selectedThresholds": pair.get("selectedThresholds"),
            "searchMode": pair.get("searchMode"),
            "currentCanonicalProbeValid": pair.get("currentCanonicalProbeValid"),
            "currentLegalRepairEvaluated": pair.get("currentLegalRepairEvaluated"),
            "currentLegalRepairSkipped": pair.get("currentLegalRepairSkipped"),
            "fullEvaluatedPairCount": pair.get("fullEvaluatedPairCount"),
            "lightEvaluatedPairCount": pair.get("lightEvaluatedPairCount"),
            "possiblePairCount": pair.get("possiblePairCount"),
            "cheapCurrentCanonicalShadow": pair.get("cheapCurrentCanonicalShadow"),
        },
    }


def _metric(values: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}

    def pct(p: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return round(ordered[index], 2)

    return {
        "count": len(ordered),
        "min": round(ordered[0], 2),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": round(ordered[-1], 2),
        "avg": round(sum(ordered) / len(ordered), 2),
    }


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    counts: Counter[str] = Counter("none" if value in (None, "") else str(value) for value in values)
    return dict(sorted(counts.items()))


def _cheap_current_shadow(row: Mapping[str, Any]) -> Mapping[str, Any]:
    pair = row.get("pairThresholdSelection")
    if not isinstance(pair, Mapping):
        return {}
    shadow = pair.get("cheapCurrentCanonicalShadow")
    return shadow if isinstance(shadow, Mapping) else {}


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    variants = sorted({str(row.get("variant")) for row in rows})
    by_variant: Dict[str, Any] = {}
    for variant in variants:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        stage_metrics = {
            stage: _metric([
                float((row.get("stageTimingsMs") or {}).get(stage))
                for row in variant_rows
                if isinstance((row.get("stageTimingsMs") or {}).get(stage), (int, float))
            ])
            for stage in DEFAULT_STAGES
        }
        by_variant[variant] = {
            "rowCount": len(variant_rows),
            "exactCount": sum(1 for row in variant_rows if row.get("exactMatch") is True),
            "statusCounts": _counts(row.get("status") for row in variant_rows),
            "recommendedMethodCounts": _counts(row.get("recommendedMethod") for row in variant_rows),
            "selectionReasonCounts": _counts(
                (row.get("pairThresholdSelection") or {}).get("selectionReason")
                for row in variant_rows
            ),
            "currentLegalRepairSkippedCounts": _counts(
                (row.get("pairThresholdSelection") or {}).get("currentLegalRepairSkipped")
                for row in variant_rows
            ),
            "cheapCurrentCanonicalShadowCounts": {
                "evaluated": sum(1 for row in variant_rows if _cheap_current_shadow(row)),
                "currentLegalRepairSkipped": _counts(
                    _cheap_current_shadow(row).get("currentLegalRepairSkipped")
                    for row in variant_rows
                    if _cheap_current_shadow(row)
                ),
                "couldHaveSkippedCurrentLegal": _counts(
                    _cheap_current_shadow(row).get("couldHaveSkippedCurrentLegalForThisInput")
                    for row in variant_rows
                    if _cheap_current_shadow(row)
                ),
            },
            "wallMs": _metric([
                float(row["wallMs"])
                for row in variant_rows
                if isinstance(row.get("wallMs"), (int, float))
            ]),
            "stageTimingsMs": stage_metrics,
        }
    return {
        "rowCount": len(rows),
        "variants": by_variant,
        "slowestSelectGuardedPairRows": sorted(
            [
                {
                    "setId": row.get("setId"),
                    "variant": row.get("variant"),
                    "iteration": row.get("iteration"),
                    "selectGuardedPairMs": (row.get("stageTimingsMs") or {}).get("selectGuardedPair"),
                    "recommendedMethod": row.get("recommendedMethod"),
                    "pairThresholdSelection": row.get("pairThresholdSelection"),
                }
                for row in rows
                if isinstance((row.get("stageTimingsMs") or {}).get("selectGuardedPair"), (int, float))
            ],
            key=lambda item: float(item.get("selectGuardedPairMs") or 0.0),
            reverse=True,
        )[:10],
    }


def render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Constrained Recognizer Benchmark",
        "",
        f"Generated: `{payload['generatedAtUtc']}`",
        f"Git head: `{payload['gitHead']}`",
        f"Manifest: `{payload['manifest']}`",
        f"Sets: `{', '.join(payload['setIds'])}`",
        f"Iterations: `{payload['iterations']}` after warmup `{payload['warmup']}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Rows | Exact | Wall p50 | Wall p90 | Prepare p50 | rembg A p50 | rembg B p50 | Hull wall p50 | Guarded pair max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    shadow_lines: List[str] = []
    for variant, row in summary["variants"].items():
        stages = row["stageTimingsMs"]
        shadow = row.get("cheapCurrentCanonicalShadowCounts") or {}
        active_skips = (row.get("currentLegalRepairSkippedCounts") or {}).get("True", 0)
        shadow_wins = (shadow.get("couldHaveSkippedCurrentLegal") or {}).get("True", 0)
        lines.append(
            f"| `{variant}` | {row['rowCount']} | {row['exactCount']} | "
            f"{row['wallMs'].get('p50', 'n/a')} | {row['wallMs'].get('p90', 'n/a')} | "
            f"{stages['prepareConstrainedInput'].get('p50', 'n/a')} | "
            f"{stages['rembgA'].get('p50', 'n/a')} | {stages['rembgB'].get('p50', 'n/a')} | "
            f"{stages['hullFitWall'].get('p50', 'n/a')} | "
            f"{stages['selectGuardedPair'].get('max', 'n/a')} |"
        )
        if shadow.get("evaluated"):
            shadow_lines.append(
                f"  - `{variant}` cheap-current shadow: evaluated "
                f"`{shadow.get('evaluated')}`, active current-legal skips `{active_skips}`, "
                f"potential current-legal skips `{shadow_wins}`."
            )
    if shadow_lines:
        lines.extend(["", "Cheap-current shadow:", ""])
        lines.extend(shadow_lines)
    lines.extend(["", "## Slowest Guarded Pair Rows", ""])
    if not summary["slowestSelectGuardedPairRows"]:
        lines.append("_None._")
    else:
        lines.extend([
            "| Set | Variant | Iteration | selectGuardedPair ms | Method | Selection |",
            "|---|---|---:|---:|---|---|",
        ])
        for row in summary["slowestSelectGuardedPairRows"]:
            pair = row.get("pairThresholdSelection") or {}
            lines.append(
                f"| {row.get('setId')} | `{row.get('variant')}` | {row.get('iteration')} | "
                f"{row.get('selectGuardedPairMs')} | `{row.get('recommendedMethod')}` | "
                f"`{pair.get('selectionReason')}` |"
            )
    lines.append("")
    return "\n".join(lines)


def _load_tasks(manifest: Path, set_ids: Sequence[str]) -> List[PairTask]:
    wanted = set(str(value) for value in set_ids)
    tasks = [
        task for task in load_corpus_tasks(manifest)
        if not wanted or task.set_id in wanted
    ]
    missing = sorted(wanted - {task.set_id for task in tasks})
    if missing:
        raise ValueError(f"set ids not found in {manifest}: {', '.join(missing)}")
    return tasks


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--only-sets", nargs="*", default=list(DEFAULT_TAIL_SETS))
    parser.add_argument("--all", action="store_true", help="Run every manifest pair instead of the default tail sets.")
    parser.add_argument("--variants", nargs="+", default=["threaded", "serial"], choices=["threaded", "serial"])
    parser.add_argument(
        "--max-sides",
        nargs="+",
        type=int,
        default=[app_module._constrained_image_max_side()],  # noqa: SLF001 - benchmark boundary.
        help="Constrained pre-resize max-side values to compare.",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    set_ids = [] if args.all else list(args.only_sets or [])
    tasks = _load_tasks(args.manifest, set_ids)
    recognizer = WhiteUpRecognizer()
    rows: List[Dict[str, Any]] = []
    max_sides = [app_module._constrained_image_max_side(value) for value in args.max_sides]  # noqa: SLF001

    def variant_label(hull_fit_mode: str, max_side: int) -> str:
        if len(max_sides) == 1:
            return hull_fit_mode
        return f"{hull_fit_mode}@{max_side}"

    for variant in args.variants:
        for max_side in max_sides:
            label = variant_label(variant, max_side)
            for warmup_index in range(args.warmup):
                for task in tasks:
                    _run_once(
                        recognizer=recognizer,
                        task=task,
                        hull_fit_mode=variant,
                        variant=label,
                        max_side=max_side,
                        iteration=-(warmup_index + 1),
                    )
            for iteration in range(1, max(1, args.iterations) + 1):
                for task in tasks:
                    row = _run_once(
                        recognizer=recognizer,
                        task=task,
                        hull_fit_mode=variant,
                        variant=label,
                        max_side=max_side,
                        iteration=iteration,
                    )
                    rows.append(row)
                    timings = row.get("stageTimingsMs") or {}
                    print(
                        f"[benchmark] {label} set {task.set_id} iter {iteration}: "
                        f"{row.get('status')} h={row.get('hamming')} "
                        f"wall={row.get('wallMs')}ms rembgA={timings.get('rembgA')}ms "
                        f"rembgB={timings.get('rembgB')}ms guarded={timings.get('selectGuardedPair')}ms",
                        file=sys.stderr,
                        flush=True,
                    )

    payload = {
        "schema": "constrained_recognizer_benchmark_v1",
        "generatedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gitHead": _git_head_sha(),
        "manifest": _rel(args.manifest),
        "setIds": [task.set_id for task in tasks],
        "variants": list(args.variants),
        "maxSides": max_sides,
        "iterations": max(1, args.iterations),
        "warmup": max(0, args.warmup),
        "summary": build_summary(rows),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), file=sys.stderr)
    print(f"wrote {args.out_json}", file=sys.stderr)
    if args.report:
        print(f"wrote {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
