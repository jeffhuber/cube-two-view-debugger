#!/usr/bin/env python3
"""Aggregate current-main solvable-rate metrics from probe_corpus JSON output."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


STICKERS_PER_CUBE = 54
MANUAL_REVIEW_CATEGORIES = {"needs_manual_review"}
RETAKE_CATEGORIES = {"reject_retake", "missing_files"}


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _pct(rate: Optional[float]) -> str:
    if rate is None:
        return "n/a"
    return f"{rate * 100:.1f}%"


def _display_path(path: str, *, cwd: Path) -> str:
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(cwd.resolve()))
    except ValueError:
        return str(raw)


def _git_head(cwd: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def load_probe_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_row(group: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    score = result.get("score")
    hamming = result.get("hamming")
    recognized_state = result.get("recognizedState") or ""
    status = result.get("status")
    category = result.get("category")

    scored = isinstance(score, int) and isinstance(hamming, int)
    exact_match = bool(scored and (score == STICKERS_PER_CUBE or hamming == 0))
    legal_state = bool(status == "success" and len(recognized_state) == STICKERS_PER_CUBE)
    confident_solve = bool(
        status == "success"
        and category not in MANUAL_REVIEW_CATEGORIES
        and category not in RETAKE_CATEGORIES
    )
    confident_wrong = bool(scored and confident_solve and hamming > 0)

    return {
        "group": group,
        "setId": str(result.get("setId") or ""),
        "status": status,
        "category": category,
        "categoryReason": result.get("categoryReason"),
        "score": score if scored else None,
        "hamming": hamming if scored else None,
        "scored": scored,
        "exactMatch": exact_match,
        "legalState": legal_state,
        "confidentSolve": confident_solve,
        "confidentWrong": confident_wrong,
        "contractPassed": result.get("contractPassed"),
        "contractFailures": list(result.get("contractFailures") or []),
        "missingFiles": list(result.get("missingFiles") or []),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scored_rows = [row for row in rows if row["scored"]]
    score_sum = sum(int(row["score"] or 0) for row in scored_rows)
    scored_denominator = len(scored_rows) * STICKERS_PER_CUBE
    manifest_denominator = len(rows) * STICKERS_PER_CUBE
    exact_count = sum(1 for row in scored_rows if row["exactMatch"])
    legal_count = sum(1 for row in scored_rows if row["legalState"])
    confident_count = sum(1 for row in scored_rows if row["confidentSolve"])
    confident_wrong_count = sum(1 for row in scored_rows if row["confidentWrong"])
    skipped_count = sum(1 for row in rows if row["status"] == "skipped")

    return {
        "rowCount": len(rows),
        "scoredRowCount": len(scored_rows),
        "skippedRowCount": skipped_count,
        "scoreSum": score_sum,
        "stickerDenominator": scored_denominator,
        "manifestStickerDenominator": manifest_denominator,
        "perStickerAccuracy": _rate(score_sum, scored_denominator),
        "perStickerAccuracyAllRowsSkippedAsZero": _rate(score_sum, manifest_denominator),
        "exactMatchCount": exact_count,
        "exactMatchRate": _rate(exact_count, len(scored_rows)),
        "legalStateCount": legal_count,
        "legalStateRate": _rate(legal_count, len(scored_rows)),
        "confidentSolveCount": confident_count,
        "confidentSolveRate": _rate(confident_count, len(scored_rows)),
        "confidentWrongCount": confident_wrong_count,
        "confidentWrongRate": _rate(confident_wrong_count, len(scored_rows)),
        "confidentWrongAmongConfidentSolveRate": _rate(confident_wrong_count, confident_count),
        "categoryCounts": dict(sorted(Counter(str(row["category"]) for row in rows).items())),
    }


def build_baseline(
    inputs: Sequence[Tuple[str, Path, Mapping[str, Any]]],
    *,
    git_head: str,
    cwd: Path,
) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    all_rows: List[Dict[str, Any]] = []
    source_runs: List[Dict[str, Any]] = []

    for group, probe_path, payload in inputs:
        rows = [summarize_row(group, result) for result in payload.get("results", [])]
        groups[group] = summarize_rows(rows)
        all_rows.extend(rows)
        source_runs.append(
            {
                "group": group,
                "probeJson": _display_path(str(probe_path), cwd=cwd),
                "manifest": _display_path(str(payload.get("manifest", "")), cwd=cwd),
                "probeSchemaVersion": payload.get("schemaVersion"),
                "timingSummary": payload.get("timingSummary") or {},
                "environmentPolicyWarningCount": len(payload.get("environmentPolicyWarnings") or []),
            }
        )

    return {
        "schemaVersion": 1,
        "gitHead": git_head,
        "metricDefinitions": {
            "perStickerAccuracy": "sum(score) / (scoredRowCount * 54); skipped rows are excluded.",
            "perStickerAccuracyAllRowsSkippedAsZero": "sum(score) / (rowCount * 54); skipped rows contribute zero.",
            "exactMatchRate": "score == 54 or hamming == 0, over scored rows.",
            "legalStateRate": "recognizer status == success and emitted a 54-sticker state, over scored rows.",
            "confidentSolveRate": "status == success and category not in needs_manual_review/reject_retake, over scored rows.",
            "confidentWrongRate": "confidentSolve and hamming > 0, over scored rows.",
        },
        "confidentSuccessCategories": ["success_clean", "success_repaired_high_confidence"],
        "manualReviewCategories": sorted(MANUAL_REVIEW_CATEGORIES),
        "retakeCategories": sorted(RETAKE_CATEGORIES),
        "sourceRuns": source_runs,
        "overall": summarize_rows(all_rows),
        "groups": groups,
        "rows": all_rows,
    }


def _count_rate(summary: Mapping[str, Any], count_key: str, rate_key: str, denominator_key: str = "scoredRowCount") -> str:
    return f"{summary[count_key]}/{summary[denominator_key]} ({_pct(summary[rate_key])})"


def render_report(baseline: Mapping[str, Any]) -> str:
    corpus_summary = baseline["groups"].get("corpus") or {}
    hard_summary = baseline["groups"].get("hard") or {}
    corpus_rows = corpus_summary.get("rowCount", "current")
    hard_rows = hard_summary.get("rowCount", "current")
    skipped_rows = baseline["overall"].get("skippedRowCount", 0)
    lines: List[str] = [
        "# Current-main solvable baseline",
        "",
        "## Purpose",
        "",
        "This is the production-recognizer solvable-rate baseline generated from",
        "`tools/probe_corpus.py` on current `main`. It covers the current",
        f"`corpus_manifest.json` ({corpus_rows} rows) and `hard_case_manifest.json` ({hard_rows} rows), and",
        "keeps the per-sticker metric comparable to the older #139 solvable-rate",
        "number.",
        "",
        f"Git head: `{baseline['gitHead']}`",
        "",
        "## Headline",
        "",
    ]

    corpus = baseline["groups"].get("corpus")
    hard = baseline["groups"].get("hard")
    overall = baseline["overall"]
    if corpus:
        lines.append(
            f"- Corpus-only per-sticker accuracy: {corpus['scoreSum']} / "
            f"({corpus['scoredRowCount']} * 54) = **{_pct(corpus['perStickerAccuracy'])}**."
        )
    if hard:
        lines.append(
            f"- Hard-case per-sticker accuracy: {hard['scoreSum']} / "
            f"({hard['scoredRowCount']} * 54) = **{_pct(hard['perStickerAccuracy'])}**."
        )
    lines.append(
        f"- Combined scored-row per-sticker accuracy: {overall['scoreSum']} / "
        f"({overall['scoredRowCount']} * 54) = **{_pct(overall['perStickerAccuracy'])}**."
    )
    if overall["skippedRowCount"]:
        lines.append(
            f"- Combined all-row denominator, counting skipped rows as zero: "
            f"**{_pct(overall['perStickerAccuracyAllRowsSkippedAsZero'])}** "
            f"({overall['skippedRowCount']} skipped row)."
        )
    lines.extend(
        [
            f"- Confident-wrong count: **{overall['confidentWrongCount']}**.",
            "",
            "## Summary",
            "",
            "| Slice | Rows | Scored | Skipped | Score sum | Per-sticker | Exact match | Legal state | Confident solve | Confident wrong |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    ordered_groups: List[Tuple[str, Mapping[str, Any]]] = [
        ("corpus", baseline["groups"].get("corpus")),
        ("hard", baseline["groups"].get("hard")),
        ("overall", baseline["overall"]),
    ]
    for label, summary in ordered_groups:
        if not summary:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(summary["rowCount"]),
                    str(summary["scoredRowCount"]),
                    str(summary["skippedRowCount"]),
                    str(summary["scoreSum"]),
                    _pct(summary["perStickerAccuracy"]),
                    _count_rate(summary, "exactMatchCount", "exactMatchRate"),
                    _count_rate(summary, "legalStateCount", "legalStateRate"),
                    _count_rate(summary, "confidentSolveCount", "confidentSolveRate"),
                    _count_rate(summary, "confidentWrongCount", "confidentWrongRate"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `perStickerAccuracy` excludes skipped rows because no recognizer score exists.",
            f"- The all-row denominator is included to make skipped rows visible ({skipped_rows} in this snapshot).",
            "- `legalState` means the recognizer emitted a 54-sticker success state; manual-review successes still count as legal states.",
            "- `confidentSolve` includes `success_clean` and `success_repaired_high_confidence`; it excludes `needs_manual_review` and `reject_retake`.",
            "- `confidentWrong` is the Phase 3 guardrail target: a confident solve with hamming > 0.",
            "",
            "## Rows",
            "",
            "| Group | Set | Status | Category | Score | Hamming | Exact | Legal | Confident | Confident wrong |",
            "|---|---:|---|---|---:|---:|---|---|---|---|",
        ]
    )
    for row in baseline["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    str(row["setId"]),
                    str(row["status"]),
                    str(row["category"]),
                    "" if row["score"] is None else str(row["score"]),
                    "" if row["hamming"] is None else str(row["hamming"]),
                    "yes" if row["exactMatch"] else "no",
                    "yes" if row["legalState"] else "no",
                    "yes" if row["confidentSolve"] else "no",
                    "yes" if row["confidentWrong"] else "no",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Reproducing",
            "",
            "```bash",
            ".venv/bin/python tools/probe_corpus.py \\",
            "  --manifest tests/fixtures/corpus_manifest.json \\",
            "  --json-output /tmp/current_main_corpus_probe.json \\",
            "  --quiet",
            ".venv/bin/python tools/probe_corpus.py \\",
            "  --manifest tests/fixtures/hard_case_manifest.json \\",
            "  --json-output /tmp/current_main_hard_probe.json \\",
            "  --quiet",
            ".venv/bin/python tools/main_solvable_baseline.py \\",
            "  --corpus-json /tmp/current_main_corpus_probe.json \\",
            "  --hard-json /tmp/current_main_hard_probe.json",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_baseline(baseline: Mapping[str, Any], out_path: Path, report_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(baseline), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-json", required=True, help="probe_corpus JSON for tests/fixtures/corpus_manifest.json")
    parser.add_argument("--hard-json", required=True, help="probe_corpus JSON for tests/fixtures/hard_case_manifest.json")
    parser.add_argument("--out", default="tests/fixtures/main_solvable_baseline.json")
    parser.add_argument("--report", default="tools/MAIN_SOLVABLE_BASELINE.md")
    parser.add_argument("--git-head", help="Override git head recorded in the fixture.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cwd = Path.cwd()
    corpus_path = Path(args.corpus_json).expanduser()
    hard_path = Path(args.hard_json).expanduser()
    baseline = build_baseline(
        [
            ("corpus", corpus_path, load_probe_json(corpus_path)),
            ("hard", hard_path, load_probe_json(hard_path)),
        ],
        git_head=args.git_head or _git_head(cwd),
        cwd=cwd,
    )
    write_baseline(baseline, Path(args.out), Path(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
