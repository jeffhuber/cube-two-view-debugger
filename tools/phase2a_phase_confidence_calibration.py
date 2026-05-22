#!/usr/bin/env python3
"""Phase 2a: calibrate the phase-darkness-separation trust signal.

The global cube model emits `phase_darkness_separation` (sep) per run.
This script analyzes the 116-run post-#218 baseline to determine the
operating point of |sep| → "route to retake" that hits the Phase 2
success criterion:

  ≥80% catastrophic recall at ≤10% false-retake rate on GOOD cases

Inputs: `tests/fixtures/post_218_baseline.json` (already in main from
#220, regenerated via #222).

Outputs:
  - `tests/fixtures/phase_confidence_calibration.json` — threshold
    curve table + chosen operating point
  - `tools/PHASE_2A_PHASE_CONFIDENCE_CALIBRATION.md` — analysis report

The signal hypothesis:

  |phase_darkness_separation| is large when the phase detector
  committed with strong evidence (either commit-correct or commit-
  flip). |sep| is small when the detector punted to `ambiguous` or
  was borderline. Phase 2 wants this as a calibrated retake signal:
  if the detector wasn't confident, route to manual/retake.

Per-run categories from post_218_baseline.json:
  - GOOD               err_near_deg < 10°
  - MARGINAL           10° ≤ err < 25°
  - CHIRALITY_MISS     detector said `correct`/`ambiguous` but model.far matches user.near
  - CHIRALITY_FALSE_FLIP detector applied a flip but flip was wrong
  - TRUE_GEOMETRY_FAIL neither matches — model fit is bad regardless

For trust-signal analysis we group:
  - "confidently right" = GOOD
  - "catastrophic" = {CHIRALITY_MISS, CHIRALITY_FALSE_FLIP, TRUE_GEOMETRY_FAIL}
  - "marginal" = MARGINAL (excluded from trust-signal scoring; would
    typically be allowed through but flagged as second-tier)

Usage:
    .venv/bin/python tools/phase2a_phase_confidence_calibration.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BASELINE = ROOT / "tests" / "fixtures" / "post_218_baseline.json"
DEFAULT_OUT_JSON = ROOT / "tests" / "fixtures" / "phase_confidence_calibration.json"
DEFAULT_OUT_MD = ROOT / "tools" / "PHASE_2A_PHASE_CONFIDENCE_CALIBRATION.md"

CATASTROPHIC = {"CHIRALITY_MISS", "CHIRALITY_FALSE_FLIP", "TRUE_GEOMETRY_FAIL"}


def _collect_runs(baseline_path: Path) -> List[Dict[str, Any]]:
    """Flatten by_case into a list of per-run dicts with the fields
    we need."""
    data = json.loads(baseline_path.read_text())
    runs: List[Dict[str, Any]] = []
    for case_id, case_runs in data["by_case"].items():
        for r in case_runs:
            if "phase_sep" not in r or "category" not in r:
                continue
            runs.append({
                "case": case_id,
                "run": r["run"],
                "phase_sep": r["phase_sep"],
                "abs_sep": abs(r["phase_sep"]),
                "category": r["category"],
                "err_near_deg": r.get("err_near_deg"),
            })
    return runs


def _calibrate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For each |sep| threshold T, compute:
      - recall on catastrophic (= fraction caught when |sep|<T → retake)
      - false-retake on GOOD (= fraction wrongly routed when |sep|<T)

    The signal: small |sep| → low confidence → route to retake. Large
    |sep| → high confidence → commit.
    """
    good = [r for r in runs if r["category"] == "GOOD"]
    catastrophic = [r for r in runs if r["category"] in CATASTROPHIC]
    marginal = [r for r in runs if r["category"] == "MARGINAL"]

    # Threshold values to sweep: every unique |sep| value plus 0.
    abs_seps = sorted(set([0.0] + [r["abs_sep"] for r in runs]))

    curve = []
    for T in abs_seps:
        # "Route to retake" condition: |sep| < T
        catastrophic_caught = sum(1 for r in catastrophic if r["abs_sep"] < T)
        good_retaken = sum(1 for r in good if r["abs_sep"] < T)
        marginal_retaken = sum(1 for r in marginal if r["abs_sep"] < T)
        catastrophic_recall = (catastrophic_caught / len(catastrophic)) if catastrophic else 0.0
        good_false_retake = (good_retaken / len(good)) if good else 0.0
        marginal_false_retake = (marginal_retaken / len(marginal)) if marginal else 0.0
        curve.append({
            "threshold": round(T, 2),
            "catastrophic_recall": round(catastrophic_recall, 3),
            "good_false_retake_rate": round(good_false_retake, 3),
            "marginal_routed_rate": round(marginal_false_retake, 3),
            "catastrophic_caught": catastrophic_caught,
            "catastrophic_total": len(catastrophic),
            "good_retaken": good_retaken,
            "good_total": len(good),
            "marginal_retaken": marginal_retaken,
            "marginal_total": len(marginal),
        })

    # Find the operating point that satisfies the Phase 2 success
    # criterion. We want the smallest T where catastrophic_recall ≥
    # 0.80 and good_false_retake_rate ≤ 0.10. If no such T exists,
    # report the closest one.
    target_recall = 0.80
    target_false_retake = 0.10
    eligible = [p for p in curve
                if p["catastrophic_recall"] >= target_recall
                and p["good_false_retake_rate"] <= target_false_retake]
    if eligible:
        # Pick the eligible point with HIGHEST recall (ties: lowest false-retake)
        chosen = max(eligible, key=lambda p: (p["catastrophic_recall"], -p["good_false_retake_rate"]))
        chosen_status = "meets_target"
    else:
        # Best Pareto point: highest recall subject to false-retake ≤ target,
        # else highest recall overall.
        below_target_false = [p for p in curve if p["good_false_retake_rate"] <= target_false_retake]
        if below_target_false:
            chosen = max(below_target_false, key=lambda p: p["catastrophic_recall"])
            chosen_status = "below_recall_target"
        else:
            chosen = max(curve, key=lambda p: p["catastrophic_recall"])
            chosen_status = "no_acceptable_point"

    return {
        "n_runs": len(runs),
        "n_good": len(good),
        "n_catastrophic": len(catastrophic),
        "n_marginal": len(marginal),
        "target": {
            "catastrophic_recall_min": target_recall,
            "good_false_retake_max": target_false_retake,
        },
        "chosen_status": chosen_status,
        "chosen_operating_point": chosen,
        "curve": curve,
    }


def _render_markdown(cal: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Phase 2a: phase-confidence trust-signal calibration")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("Can `|phase_darkness_separation|` from the global cube model serve")
    lines.append("as a trust signal that routes low-confidence geometry to retake,")
    lines.append("hitting the Phase 2 success criterion?")
    lines.append("")
    lines.append("> Phase 2 target: **≥80% catastrophic recall at ≤10% false-retake on GOOD cases.**")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("On the 116-run post-#218 baseline:")
    lines.append("")
    lines.append(f"- `n_good` (GOOD outcome, err < 10°): **{cal['n_good']}**")
    lines.append(f"- `n_catastrophic` (CHIRALITY_MISS + CHIRALITY_FALSE_FLIP + TRUE_GEOMETRY_FAIL): **{cal['n_catastrophic']}**")
    lines.append(f"- `n_marginal` (10°–25° err — not part of success criterion): **{cal['n_marginal']}**")
    lines.append("")
    lines.append("For each threshold T, the retake decision is `|phase_sep| < T`. We compute:")
    lines.append("")
    lines.append("- `catastrophic_recall` = (catastrophic runs flagged for retake) / (all catastrophic)")
    lines.append("- `good_false_retake_rate` = (GOOD runs flagged for retake) / (all GOOD)")
    lines.append("- `marginal_routed_rate` = (MARGINAL runs flagged for retake) / (all MARGINAL)  — informational")
    lines.append("")
    lines.append("## Chosen operating point")
    lines.append("")
    chosen = cal["chosen_operating_point"]
    lines.append(f"**Status: `{cal['chosen_status']}`**")
    lines.append("")
    status_msg = {
        "meets_target": "The target is achievable on this baseline.",
        "below_recall_target": "Target false-retake holds but recall is below 80%.",
        "no_acceptable_point": "Neither target dimension can be met simultaneously.",
    }.get(cal["chosen_status"], "")
    if status_msg:
        lines.append(status_msg)
        lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| Threshold T (route to retake when \\|sep\\| < T) | **{chosen['threshold']:.2f}** |")
    lines.append(f"| Catastrophic recall | **{chosen['catastrophic_recall']:.1%}** ({chosen['catastrophic_caught']}/{chosen['catastrophic_total']}) |")
    lines.append(f"| GOOD false-retake rate | **{chosen['good_false_retake_rate']:.1%}** ({chosen['good_retaken']}/{chosen['good_total']}) |")
    lines.append(f"| MARGINAL routed rate (informational) | {chosen['marginal_routed_rate']:.1%} ({chosen['marginal_retaken']}/{chosen['marginal_total']}) |")
    lines.append("")
    lines.append("## Threshold curve (selected rows)")
    lines.append("")
    lines.append("Showing thresholds where one of the metrics changes meaningfully.")
    lines.append("")
    lines.append("| T  | catastrophic_recall | good_false_retake | marginal_routed |")
    lines.append("|---:|--------------------:|------------------:|----------------:|")
    # Subsample the curve at uniform recall intervals for readability
    seen = set()
    selected = []
    for p in cal["curve"]:
        key = (round(p["catastrophic_recall"], 2), round(p["good_false_retake_rate"], 2))
        if key in seen:
            continue
        seen.add(key)
        selected.append(p)
    # Cap rows shown
    for p in selected[:30]:
        lines.append(f"| {p['threshold']:>5.2f} | {p['catastrophic_recall']:>6.1%} ({p['catastrophic_caught']:>2}/{p['catastrophic_total']:<2}) | {p['good_false_retake_rate']:>6.1%} ({p['good_retaken']:>2}/{p['good_total']:<2}) | {p['marginal_routed_rate']:>6.1%} |")
    if len(selected) > 30:
        lines.append(f"| ... | {len(selected)-30} more rows truncated | | |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if cal["chosen_status"] == "meets_target":
        lines.append("✅ The phase-confidence signal IS sufficient on its own to meet the")
        lines.append("Phase 2 success criterion on this baseline. The operating point is")
        lines.append("a usable trust signal for Phase 3 production guardrail wiring.")
    elif cal["chosen_status"] == "below_recall_target":
        lines.append("⚠️ The signal can meet the false-retake budget but not the catastrophic-")
        lines.append("recall target alone. **Phase 2 needs additional signals** — either")
        lines.append("compose with cv-local face-quad consistency (#225 finding), two-view")
        lines.append("consistency (#200), or vertex-ensemble disagreement.")
    else:
        lines.append("❌ Neither target can be hit alone with this signal. **Phase 2 will")
        lines.append("require multi-signal composition** to reach the success criterion.")
    lines.append("")
    lines.append("## Phase 2b candidates if multi-signal is needed")
    lines.append("")
    lines.append("Per the Phase 2 plan in `STATE_OF_THE_WORLD.md`:")
    lines.append("")
    lines.append("1. **cv-local face-quad structural consistency** — Phase 1 (#225) showed")
    lines.append("   90% structural fit-fail on cv-local. Strongly predicts unreliable")
    lines.append("   geometry independently of the phase detector.")
    lines.append("2. **Two-view consistency** — Codex's #200 diagnostic; if A and B")
    lines.append("   disagree, low confidence.")
    lines.append("3. **Vertex-ensemble disagreement** — the mean-of-3 ensemble already")
    lines.append("   computes 3 candidate vertices; their pairwise disagreement is a")
    lines.append("   vertex-uncertainty proxy that should correlate with phase miscalls.")
    lines.append("")
    lines.append("## Reproducing")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python tools/phase2a_phase_confidence_calibration.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    args = ap.parse_args()

    runs = _collect_runs(Path(args.baseline))
    print(f"loaded {len(runs)} runs from {args.baseline}", file=sys.stderr)
    cal = _calibrate(runs)
    print(f"chosen operating point: T={cal['chosen_operating_point']['threshold']} "
          f"(status: {cal['chosen_status']})", file=sys.stderr)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(cal, indent=2))
    print(f"wrote {args.out_json}", file=sys.stderr)

    report = _render_markdown(cal)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(report)
    print(f"wrote {args.out_md}", file=sys.stderr)

    print("\n=== Headline ===")
    chosen = cal["chosen_operating_point"]
    print(f"  threshold: T = {chosen['threshold']}")
    print(f"  catastrophic recall: {chosen['catastrophic_recall']:.1%}")
    print(f"  GOOD false-retake:   {chosen['good_false_retake_rate']:.1%}")
    print(f"  target met: {cal['chosen_status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
