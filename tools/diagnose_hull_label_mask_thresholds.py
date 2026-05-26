#!/usr/bin/env python3
"""Diagnose rembg alpha-threshold sensitivity for hull-label geometry.

Production currently builds the hull-label mask from rembg's alpha channel with
`alpha > 128`. Set 70 showed a shadow/contact patch that pulls the convex hull
into a bad shape at that threshold. This diagnostic keeps rembg fixed, sweeps
several alpha thresholds, and records how hull-label acceptance/score changes.

The goal is diagnostic evidence for a future mask-candidate selector:

1. run rembg once per image,
2. try a small set of alpha thresholds,
3. choose the best geometry/color candidate instead of failing on one fixed
   threshold.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_slot_yaw_assignment import DEFAULT_MANIFEST  # noqa: E402
from tools.evaluate_hybrid_pipeline import _load_processing_image  # noqa: E402
from tools.extract_color_samples import PairTask, load_corpus_tasks  # noqa: E402
from tools.global_cube_model import _fit_hull_label_tier1_model  # noqa: E402


DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "hull_label_mask_threshold_diagnostic.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "HULL_LABEL_MASK_THRESHOLD_DIAGNOSTIC.md"
DEFAULT_THRESHOLDS = (64, 128, 160, 192, 224)


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple:
    score = candidate.get("sticker_score_total")
    numeric_score = float(score) if isinstance(score, (int, float)) else float("inf")
    return numeric_score, int(candidate.get("threshold", 9999))


def choose_best_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    accepted_only: bool,
) -> Optional[Mapping[str, Any]]:
    pool = [row for row in candidates if row.get("accepted")] if accepted_only else list(candidates)
    if not pool:
        return None
    return min(pool, key=_candidate_sort_key)


def _trace_candidate(threshold: int, trace: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "threshold": threshold,
        "status": trace.get("status"),
        "accepted": bool(trace.get("accepted")),
        "vertex_source": trace.get("vertex_source"),
        "sticker_score_total": trace.get("sticker_score_total"),
        "mean_sticker_distance": trace.get("mean_sticker_distance"),
        "vertex_cloud_spread_px": trace.get("vertex_cloud_spread_px"),
        "vertex_cloud_spread_norm": trace.get("vertex_cloud_spread_norm"),
        "projective_residual_norm": trace.get("projective_residual_norm"),
        "hard_failures": list(trace.get("hard_failures") or []),
        "warnings": list(trace.get("warnings") or []),
    }


def _evaluate_side(
    *,
    image_path: Path,
    side: str,
    sess: Any,
    thresholds: Sequence[int],
) -> Dict[str, Any]:
    from rembg import remove  # noqa: E402

    image, arr = _load_processing_image(image_path)
    rgba = remove(image, session=sess)
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)

    candidates: List[Dict[str, Any]] = []
    for threshold in thresholds:
        mask = alpha > threshold
        _model, trace = _fit_hull_label_tier1_model(arr, mask, side=side, mode="prefer")
        candidates.append(_trace_candidate(threshold, trace))

    best_any = choose_best_candidate(candidates, accepted_only=False)
    best_accepted = choose_best_candidate(candidates, accepted_only=True)
    return {
        "side": side,
        "image": _rel(image_path),
        "candidates": candidates,
        "bestAnyThreshold": best_any.get("threshold") if best_any else None,
        "bestAnyAccepted": bool(best_any and best_any.get("accepted")),
        "bestAnyScore": best_any.get("sticker_score_total") if best_any else None,
        "bestAcceptedThreshold": best_accepted.get("threshold") if best_accepted else None,
        "bestAcceptedScore": best_accepted.get("sticker_score_total") if best_accepted else None,
    }


def _evaluate_pair(task: PairTask, sess: Any, thresholds: Sequence[int]) -> Dict[str, Any]:
    return {
        "setId": task.set_id,
        "source": task.source,
        "sides": {
            "A": _evaluate_side(image_path=task.image_a, side="A", sess=sess, thresholds=thresholds),
            "B": _evaluate_side(image_path=task.image_b, side="B", sess=sess, thresholds=thresholds),
        },
    }


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sides = [row["sides"][side] for row in rows for side in ("A", "B") if side in row.get("sides", {})]
    best_any = Counter(str(side.get("bestAnyThreshold")) for side in sides)
    best_accepted = Counter(str(side.get("bestAcceptedThreshold")) for side in sides)
    accepted_by_threshold: Dict[str, int] = Counter()
    for side in sides:
        for candidate in side.get("candidates", []):
            if candidate.get("accepted"):
                accepted_by_threshold[str(candidate.get("threshold"))] += 1
    score_improvements = []
    for side in sides:
        by_threshold = {candidate["threshold"]: candidate for candidate in side.get("candidates", [])}
        baseline = by_threshold.get(128)
        best = choose_best_candidate(side.get("candidates", []), accepted_only=False)
        if baseline and best:
            b_score = baseline.get("sticker_score_total")
            best_score = best.get("sticker_score_total")
            if isinstance(b_score, (int, float)) and isinstance(best_score, (int, float)):
                score_improvements.append(float(b_score) - float(best_score))
    return {
        "pairCount": len(rows),
        "sideCount": len(sides),
        "bestAnyThresholdCounts": dict(sorted(best_any.items())),
        "bestAcceptedThresholdCounts": dict(sorted(best_accepted.items())),
        "acceptedSideCountByThreshold": dict(sorted(accepted_by_threshold.items())),
        "medianScoreImprovementVs128": statistics.median(score_improvements) if score_improvements else None,
        "meanScoreImprovementVs128": round(statistics.mean(score_improvements), 2) if score_improvements else None,
    }


def render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Hull-Label Mask Threshold Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic tests whether a fixed rembg alpha threshold (`alpha > 128`)",
        "is too fragile for hull-label geometry. It sweeps candidate thresholds",
        "after one rembg pass per image and records hull-label acceptance and",
        "sticker-score quality.",
        "",
        f"Git head: `{payload['source']['git_sha']}`",
        f"Generated: `{payload['source']['generated_at_utc']}`",
        f"Thresholds: `{payload['source']['thresholds']}`",
        "",
        "## Summary",
        "",
        f"- Pairs: {payload['summary']['pairCount']}",
        f"- Sides: {payload['summary']['sideCount']}",
        f"- Best-by-score threshold counts: `{payload['summary']['bestAnyThresholdCounts']}`",
        f"- Best accepted threshold counts: `{payload['summary']['bestAcceptedThresholdCounts']}`",
        f"- Accepted side count by threshold: `{payload['summary']['acceptedSideCountByThreshold']}`",
        f"- Median score improvement vs `128`: `{payload['summary']['medianScoreImprovementVs128']}`",
        "",
        "## Per Side",
        "",
        "| Set | Side | Best any | Any accepted? | Best accepted | alpha>128 | alpha>224 | Notes |",
        "|---:|---|---:|---|---:|---|---|---|",
    ]
    for row in payload["rows"]:
        for side_name in ("A", "B"):
            side = row["sides"][side_name]
            by_threshold = {candidate["threshold"]: candidate for candidate in side["candidates"]}
            c128 = by_threshold.get(128, {})
            c224 = by_threshold.get(224, {})
            notes = []
            if side.get("bestAnyThreshold") != 128:
                notes.append("fixed-128 not best")
            if side.get("bestAnyThreshold") == 224 and not side.get("bestAnyAccepted"):
                notes.append("224 best but rejected")
            lines.append(
                f"| {row['setId']} | {side_name} | {side.get('bestAnyThreshold')} "
                f"({side.get('bestAnyScore')}) | {side.get('bestAnyAccepted')} | "
                f"{side.get('bestAcceptedThreshold')} ({side.get('bestAcceptedScore')}) | "
                f"{c128.get('status')}/{c128.get('sticker_score_total')} | "
                f"{c224.get('status')}/{c224.get('sticker_score_total')} | "
                f"{'; '.join(notes)} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Set 70 demonstrates why a fixed `alpha > 128` mask is brittle: shadow",
        "  pixels can be included in the silhouette, distorting the convex hull",
        "  and downstream vertex/corner geometry.",
        "- A candidate selector should not simply hard-code `224`. It should try",
        "  a small threshold set and choose a candidate using geometry/color",
        "  quality, then apply acceptance gates.",
        "- Rows where the lowest-score candidate is rejected are especially useful",
        "  for gate tuning: they show where visual quality and current hard gates",
        "  disagree.",
    ])
    return "\n".join(lines) + "\n"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--thresholds", type=int, nargs="*", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--only-sets", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    tasks = load_corpus_tasks(args.manifest)
    only = {str(item) for item in args.only_sets} if args.only_sets else None
    if only is not None:
        tasks = [task for task in tasks if task.set_id in only]

    from rembg import new_session  # noqa: E402
    sess = new_session("u2net")
    rows = [_evaluate_pair(task, sess, args.thresholds) for task in tasks]
    payload = {
        "schema": "hull_label_mask_threshold_diagnostic_v1",
        "source": {
            "tool": "tools/diagnose_hull_label_mask_thresholds.py",
            "manifest": _rel(args.manifest),
            "thresholds": list(args.thresholds),
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
