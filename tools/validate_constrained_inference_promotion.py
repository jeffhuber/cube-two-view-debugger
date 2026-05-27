#!/usr/bin/env python3
"""Validate a production-shaped constrained-inference promotion gate.

This consumes ``tools/diagnose_pair_threshold_repair.py`` output and applies a
GT-free auto-return gate to the guarded pair-threshold candidate. Ground truth
is used only to score the gate in this diagnostic report.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.constrained_inference_gate import (  # noqa: E402
    ALLOWED_METHODS,
    ALLOWED_SELECTION_REASONS,
    MAX_CANONICAL_REPAIR_MOVES,
    MAX_LEGAL_REPAIR_CHANGES,
    MAX_LEGAL_REPAIR_COST,
    MAX_LEGAL_STATE_DELTA,
    build_gate_summary as build_summary,
    evaluate_pair_threshold_row_gate as evaluate_promotion_gate,
)
DEFAULT_IN = REPO_ROOT / "tests" / "fixtures" / "pair_threshold_repair_diagnostic.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "constrained_inference_promotion_gate.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "CONSTRAINED_INFERENCE_PROMOTION_GATE.md"


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


def render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Constrained-Inference Promotion Gate",
        "",
        "Diagnostic-only. This report evaluates a production-shaped gate for",
        "using the hull-label constrained-inference candidate outside Fixer.",
        "Ground truth is used only for scoring the gate, never for the gate",
        "decision itself.",
        "",
        f"Git head: `{payload['source']['git_sha']}`",
        f"Generated: `{payload['source']['generated_at_utc']}`",
        f"Input: `{payload['source']['input']}`",
        "",
        "## Gate",
        "",
        "A candidate may auto-return only when all of these production-available",
        "conditions hold:",
        "",
        "- the guarded pair-threshold selection assembled both sides;",
        "- the yaw inference is accepted when present;",
        "- both selected side thresholds were accepted and have no hard failures;",
        "- the recommended repair is valid, count-balanced, and not low confidence;",
        f"- the recommended method is one of `{sorted(ALLOWED_METHODS)}`;",
        f"- canonical count repair uses at most `{MAX_CANONICAL_REPAIR_MOVES}` moves;",
        f"- legal repair methods stay within state delta `{MAX_LEGAL_STATE_DELTA}`,",
        f"  repair cost `{MAX_LEGAL_REPAIR_COST}`, and repair changes `{MAX_LEGAL_REPAIR_CHANGES}`;",
        "- pair-threshold switching is allowed only when the current per-side",
        "  threshold repair was invalid.",
        "",
        "The ungated `broad_legal_repaired` method is intentionally excluded.",
        "",
        "## Summary",
        "",
        "| Pairs | Gate accepted | Gate rejected | Accepted exact | Accepted legal | Accepted <=3 | Threshold switches |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['pairCount']} | {summary['accepted']} | {summary['rejected']} | "
        f"{summary['acceptedExact']} | {summary['acceptedLegal']} | "
        f"{summary['acceptedWithin3']} | {summary['acceptedThresholdSwitchCount']} |",
        "",
        f"Accepted hamming distribution: `{summary['acceptedHammingDistribution']}`",
        "",
        "Accepted method counts:",
        "",
    ]
    for method, count in summary["acceptedMethodCounts"].items():
        lines.append(f"- `{method}`: `{count}`")
    lines.extend([
        "",
        "Selection reason counts:",
        "",
    ])
    for reason, count in summary["selectionReasonCounts"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend([
        "",
        "## Rejected Rows",
        "",
    ])
    if not summary["rejectedSetIds"]:
        lines.append("_None._")
    else:
        lines.append("| Set | Method | Reasons |")
        lines.append("|---:|---|---|")
        for row in payload["rows"]:
            if row["accepted"]:
                continue
            lines.append(
                f"| {row['setId']} | `{row['recommendedMethod']}` | "
                f"`{', '.join(row['rejectReasons'])}` |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Passing this diagnostic means the constrained-inference candidate is",
        "  coherent enough to run in a default-recognizer shadow lane or an",
        "  explicit candidate mode. It does not by itself flip `/api/recognize`.",
        "- The production flip should require the same gate to hold on shadow",
        "  traffic, plus a confidence policy for what to do when the gate rejects:",
        "  fall back to legacy, return manual-review, or ask for a retake.",
        "- The pair-threshold switch count is important: a switch is permitted",
        "  only as a rescue when current deterministic repair is invalid, which is",
        "  the Set 14 shape documented in the pair-threshold report.",
    ])
    return "\n".join(lines) + "\n"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    source = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [evaluate_promotion_gate(row) for row in source.get("rows", [])]
    payload = {
        "schema": "constrained_inference_promotion_gate_v1",
        "source": {
            "tool": "tools/validate_constrained_inference_promotion.py",
            "input": _rel(args.input),
            "git_sha": _git_head_sha(),
            "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "gate": {
            "maxCanonicalRepairMoves": MAX_CANONICAL_REPAIR_MOVES,
            "maxLegalStateDelta": MAX_LEGAL_STATE_DELTA,
            "maxLegalRepairCost": MAX_LEGAL_REPAIR_COST,
            "maxLegalRepairChanges": MAX_LEGAL_REPAIR_CHANGES,
            "allowedMethods": sorted(ALLOWED_METHODS),
            "allowedSelectionReasons": sorted(ALLOWED_SELECTION_REASONS),
        },
        "summary": build_summary(source.get("rows", [])),
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
