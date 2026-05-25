#!/usr/bin/env python3
"""Aggregate analyzer for Tier 1 hull-label shadow-mode traces.

PR #291 wired ``fit_global_cube_model`` with a feature-flagged
hull-label candidate path. In ``shadow`` mode the candidate runs
but the legacy model is always returned; the candidate's trace
(``model.debug["hull_label_tier1"]`` when legacy returns, or the
helper-level return value otherwise) carries the acceptance-gate
decision.

This tool answers the question PR #291 unblocked but didn't yet
answer: **if we flipped the default from ``shadow`` to ``prefer``
today, what fraction of corpus rows would the hull-label candidate
take over for, and which acceptance gates are doing the work?**

It complements but doesn't duplicate ``measure_hull_labels_corpus.py``:

- ``measure_hull_labels_corpus`` answers "does
  ``rectify_via_hull_labels`` produce clean rectifications?" — it
  measures *the rectification* against ground truth and threshold
  buckets.
- ``analyze_shadow_traces`` answers "does the production-shaped
  acceptance gate (``evaluate_hull_label_acceptance``) correctly
  identify the rectifications worth keeping?" — it measures *the
  gate* against the rectifications.

Pipeline per row:
  1. EXIF-correct + resize image (``_processing_image``).
  2. rembg silhouette (cached session across the corpus).
  3. Call ``_fit_hull_label_tier1_model(image, mask, side=side,
     mode="shadow")`` to get (model_or_none, trace).
  4. Aggregate trace by status / hard_failures / vertex_source /
     side. Surface the rejected-with-clean-rectification cases for
     manual review and the accepted-with-warnings cases for gate
     calibration.

## CLI

  python tools/analyze_shadow_traces.py

Defaults: read ``tests/fixtures/gcm_axis_ground_truth.json``, write
trace to ``tests/fixtures/shadow_trace_corpus.json`` and report to
``tools/SHADOW_TRACE_ANALYSIS.md``.

Status output is printed every row so long runs (~3–5 min on 70
rows w/ rembg first-time download) don't look hung.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_pipeline_phase_parity import _processing_image  # noqa: E402
from tools.global_cube_model import _fit_hull_label_tier1_model  # noqa: E402
from tools.measure_axis_correctness import (  # noqa: E402
    _candidate_image_roots,
    _resolve_image_path,
)

DEFAULT_AXIS_TRUTH = REPO_ROOT / "tests" / "fixtures" / "gcm_axis_ground_truth.json"
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_TRACE = REPO_ROOT / "tests" / "fixtures" / "shadow_trace_corpus.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "SHADOW_TRACE_ANALYSIS.md"
DEFAULT_MAX_IMAGE_DIM = 1600


def _git_head_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _get_rembg_session() -> Any:
    """Lazy import + session init. Matches measure_hull_labels_corpus.

    Returns a session usable as ``rembg.remove(image, session=sess)``.
    """
    from rembg import new_session

    return new_session("u2net")


def analyze_row(
    sess: Any,
    key: str,
    image_path: Path,
    *,
    max_image_dim: int,
) -> Dict[str, Any]:
    """Run shadow-mode hull-label fit on one row, extract trace.

    Returns a flattened record with status / acceptance / hard_failures
    / warnings / sticker_score_total / vertex_source / vertex_cloud_*
    fields plucked from the trace for easy aggregation.
    """
    from rembg import remove  # noqa: E402

    side = key.rsplit("_", 1)[-1]
    rec: Dict[str, Any] = {
        "key": key,
        "side": side,
        "image_path": str(image_path),
    }
    try:
        image, _scale = _processing_image(image_path, max_image_dim)
        rgba = remove(image, session=sess)
        alpha = np.array(rgba.split()[-1], dtype=np.uint8)
        mask = alpha > 128
        # Convert PIL → np.ndarray for the Tier 1 helper.
        image_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        _model, trace = _fit_hull_label_tier1_model(
            image_rgb, mask, side=side, mode="shadow",
        )
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "harness_error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        return rec

    rec["trace_status"] = trace.get("status")
    rec["accepted"] = trace.get("accepted")
    rec["selected"] = trace.get("selected")
    rec["mode"] = trace.get("mode")
    rec["hard_failures"] = list(trace.get("hard_failures", []))
    rec["warnings"] = list(trace.get("warnings", []))
    rec["sticker_score_total"] = trace.get("sticker_score_total")
    rec["mean_sticker_distance"] = trace.get("mean_sticker_distance")
    rec["vertex_source"] = trace.get("vertex_source")
    rec["vertex_cloud_spread_px"] = trace.get("vertex_cloud_spread_px")
    rec["vertex_cloud_spread_norm"] = trace.get("vertex_cloud_spread_norm")
    rec["projective_residual_norm"] = trace.get("projective_residual_norm")
    rec["hexagon_diameter_px"] = trace.get("hexagon_diameter_px")
    rec["projective_degeneracy"] = trace.get("projective_degeneracy")
    # Keep the full trace under a nested key so the JSON artifact is
    # round-trippable for forensic re-runs without re-executing rembg.
    rec["full_trace"] = trace
    return rec


def _summarize(per_row: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-row records into the report payload."""
    total = len(per_row)
    by_status = Counter(r.get("trace_status") for r in per_row)
    by_acceptance = Counter(
        ("accepted" if r.get("accepted") else "rejected")
        if r.get("accepted") is not None
        else "no_decision"
        for r in per_row
    )
    by_side = Counter(r.get("side") for r in per_row)
    by_side_accept = Counter()
    for r in per_row:
        if r.get("accepted"):
            by_side_accept[(r.get("side"), "accepted")] += 1
        elif r.get("accepted") is False:
            by_side_accept[(r.get("side"), "rejected")] += 1
        else:
            by_side_accept[(r.get("side"), "no_decision")] += 1

    # Hard-failure histogram: each row may contribute 0..N failure
    # tokens. Counter on the flattened list.
    failure_tokens = Counter()
    warning_tokens = Counter()
    for r in per_row:
        for tok in r.get("hard_failures", []):
            failure_tokens[tok] += 1
        for tok in r.get("warnings", []):
            warning_tokens[tok] += 1

    # Vertex source distribution (affine vs projective) among
    # accepted rows.
    vsource_accepted = Counter(
        r.get("vertex_source")
        for r in per_row
        if r.get("accepted") is True
    )

    # Quantitative gate-signal distributions (accepted only).
    spread_norm_accepted = [
        r["vertex_cloud_spread_norm"]
        for r in per_row
        if r.get("accepted") is True
        and isinstance(r.get("vertex_cloud_spread_norm"), (int, float))
    ]
    sticker_accepted = [
        r["sticker_score_total"]
        for r in per_row
        if r.get("accepted") is True
        and isinstance(r.get("sticker_score_total"), (int, float))
    ]
    residual_accepted = [
        r["projective_residual_norm"]
        for r in per_row
        if r.get("accepted") is True
        and isinstance(r.get("projective_residual_norm"), (int, float))
    ]

    return {
        "total_rows": total,
        "by_status": dict(by_status),
        "by_acceptance": dict(by_acceptance),
        "by_side": dict(by_side),
        "by_side_acceptance": {
            f"{side}_{verdict}": cnt
            for (side, verdict), cnt in sorted(by_side_accept.items())
        },
        "hard_failure_tokens": dict(failure_tokens),
        "warning_tokens": dict(warning_tokens),
        "vertex_source_accepted": dict(vsource_accepted),
        "spread_norm_accepted_stats": _stats(spread_norm_accepted),
        "sticker_score_accepted_stats": _stats(sticker_accepted),
        "projective_residual_accepted_stats": _stats(residual_accepted),
    }


def _stats(xs: List[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
        "median": round(statistics.median(xs), 4),
        "mean": round(statistics.mean(xs), 4),
    }


def render_report(
    summary: Dict[str, Any],
    per_row: List[Dict[str, Any]],
    *,
    head_sha: Optional[str],
    axis_truth_path: Path,
    trace_path: Path,
) -> str:
    """Markdown report. Surfaces accept/reject distribution + worst
    rows for manual review.
    """
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []
    lines.append("# Tier 1 Shadow-Mode Trace Analysis")
    lines.append("")
    lines.append(
        "Generated by `tools/analyze_shadow_traces.py` — answers "
        "*if we flipped the hull-label default from `shadow` to "
        "`prefer` today, what would change?*"
    )
    lines.append("")
    lines.append(f"- Generated: {now}")
    if head_sha:
        lines.append(f"- HEAD: `{head_sha}`")
    lines.append(f"- Source corpus: `{axis_truth_path.name}` ({summary['total_rows']} rows)")
    lines.append(f"- Trace fixture: `{trace_path.name}`")
    lines.append("")
    lines.append("## Headline: would prefer-mode flip anything?")
    lines.append("")
    accept = summary["by_acceptance"].get("accepted", 0)
    reject = summary["by_acceptance"].get("rejected", 0)
    no_dec = summary["by_acceptance"].get("no_decision", 0)
    total = summary["total_rows"]
    if total:
        accept_pct = round(100.0 * accept / total, 1)
        reject_pct = round(100.0 * reject / total, 1)
        nodec_pct = round(100.0 * no_dec / total, 1)
    else:
        accept_pct = reject_pct = nodec_pct = 0.0
    lines.append(
        f"In `prefer` mode, **{accept}/{total} ({accept_pct}%) rows** would "
        f"be served by the hull-label candidate. "
        f"**{reject}/{total} ({reject_pct}%)** would fall back to legacy "
        f"after gate rejection. "
        f"**{no_dec}/{total} ({nodec_pct}%)** never reached the gate "
        f"(harness error or early-stage skip)."
    )
    lines.append("")
    lines.append("## Per-status row distribution")
    lines.append("")
    lines.append("| status | count |")
    lines.append("|---|---|")
    for status, cnt in sorted(summary["by_status"].items(),
                              key=lambda kv: -kv[1]):
        lines.append(f"| `{status}` | {cnt} |")
    lines.append("")
    lines.append("## Per-side acceptance")
    lines.append("")
    lines.append("| side / verdict | count |")
    lines.append("|---|---|")
    for label, cnt in sorted(summary["by_side_acceptance"].items()):
        lines.append(f"| {label} | {cnt} |")
    lines.append("")
    lines.append("## Hard-failure tokens (which gates do the work)")
    lines.append("")
    if summary["hard_failure_tokens"]:
        lines.append("| token | row-count |")
        lines.append("|---|---|")
        for tok, cnt in sorted(summary["hard_failure_tokens"].items(),
                               key=lambda kv: -kv[1]):
            lines.append(f"| `{tok}` | {cnt} |")
    else:
        lines.append("_No hard failures observed — all rows passed acceptance._")
    lines.append("")
    lines.append("## Warning tokens (advisory, do not force fallback)")
    lines.append("")
    if summary["warning_tokens"]:
        lines.append("| token | row-count |")
        lines.append("|---|---|")
        for tok, cnt in sorted(summary["warning_tokens"].items(),
                               key=lambda kv: -kv[1]):
            lines.append(f"| `{tok}` | {cnt} |")
    else:
        lines.append("_No warnings observed._")
    lines.append("")
    lines.append("## Vertex source on accepted rows (affine vs projective)")
    lines.append("")
    if summary["vertex_source_accepted"]:
        lines.append("| vertex_source | accepted-row-count |")
        lines.append("|---|---|")
        for src, cnt in sorted(summary["vertex_source_accepted"].items(),
                               key=lambda kv: -kv[1]):
            lines.append(f"| `{src}` | {cnt} |")
    else:
        lines.append("_No accepted rows._")
    lines.append("")
    lines.append("## Quantitative gate-signal distributions (accepted only)")
    lines.append("")
    for label, key in (
        ("`vertex_cloud_spread_norm`", "spread_norm_accepted_stats"),
        ("`sticker_score_total`", "sticker_score_accepted_stats"),
        ("`projective_residual_norm`", "projective_residual_accepted_stats"),
    ):
        s = summary[key]
        lines.append(f"### {label}")
        if s["n"] == 0:
            lines.append("_n=0_")
            lines.append("")
            continue
        lines.append(
            f"n={s['n']}, min={s['min']}, median={s['median']}, "
            f"mean={s['mean']}, max={s['max']}"
        )
        lines.append("")
    lines.append("## Rejected-row punch list (for gate review)")
    lines.append("")
    rejected = [r for r in per_row if r.get("accepted") is False]
    if not rejected:
        lines.append("_None._")
    else:
        lines.append("| key | side | hard_failures | sticker_score | spread_norm | residual_norm |")
        lines.append("|---|---|---|---|---|---|")
        for r in rejected:
            failures = ", ".join(f"`{t}`" for t in r["hard_failures"]) or "—"
            ss = r.get("sticker_score_total")
            sp = r.get("vertex_cloud_spread_norm")
            rn = r.get("projective_residual_norm")
            lines.append(
                f"| {r['key']} | {r['side']} | {failures} | "
                f"{ss if ss is not None else '—'} | "
                f"{sp if sp is not None else '—'} | "
                f"{rn if rn is not None else '—'} |"
            )
    lines.append("")
    lines.append("## Accepted-with-warnings list (for gate calibration)")
    lines.append("")
    warn_accepted = [
        r for r in per_row
        if r.get("accepted") is True and r.get("warnings")
    ]
    if not warn_accepted:
        lines.append("_None._")
    else:
        lines.append("| key | side | warnings | sticker_score | spread_norm | residual_norm |")
        lines.append("|---|---|---|---|---|---|")
        for r in warn_accepted:
            warns = ", ".join(f"`{t}`" for t in r["warnings"]) or "—"
            ss = r.get("sticker_score_total")
            sp = r.get("vertex_cloud_spread_norm")
            rn = r.get("projective_residual_norm")
            lines.append(
                f"| {r['key']} | {r['side']} | {warns} | "
                f"{ss if ss is not None else '—'} | "
                f"{sp if sp is not None else '—'} | "
                f"{rn if rn is not None else '—'} |"
            )
    lines.append("")
    lines.append("## Harness errors (if any)")
    lines.append("")
    errs = [r for r in per_row if r.get("status") == "harness_error"]
    if not errs:
        lines.append("_None._")
    else:
        for r in errs:
            lines.append(f"- `{r['key']}` — {r.get('error')}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--axis-truth", type=Path, default=DEFAULT_AXIS_TRUTH)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--trace-out", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM)
    ap.add_argument(
        "--limit", type=int, default=0,
        help="If >0, only process the first N rows (smoke test).",
    )
    args = ap.parse_args(argv)

    axis_truth = json.loads(args.axis_truth.read_text())
    manifest_doc = (
        json.loads(args.manifest.read_text())
        if args.manifest.exists() else {}
    )
    pairs = manifest_doc.get("pairs", []) if isinstance(manifest_doc, dict) else []
    # setId → pair record (for image{A,B}Path lookup).
    set_index: Dict[str, Dict[str, Any]] = {
        str(p.get("setId")): p for p in pairs if p.get("setId") is not None
    }
    image_roots = _candidate_image_roots(manifest_doc)
    # axis_truth is a dict mapping key (e.g. "17_A") → row record.
    # Only iterate "approved" rows — matches measure_hull_labels_corpus.
    rows = []
    if isinstance(axis_truth, dict) and "rows" not in axis_truth:
        for key in sorted(axis_truth):
            row = axis_truth[key]
            if not isinstance(row, dict) or not row.get("approved"):
                continue
            rows.append({"key": key, **row})
    elif isinstance(axis_truth, list):
        rows = [r for r in axis_truth if r.get("approved")]
    else:
        rows = [r for r in axis_truth.get("rows", []) if r.get("approved")]

    sess = None  # initialized lazily on first row that actually needs it
    per_row: List[Dict[str, Any]] = []

    target_rows = rows[: args.limit] if args.limit > 0 else rows
    print(f"analyze_shadow_traces: processing {len(target_rows)} rows", flush=True)
    for i, row in enumerate(target_rows, start=1):
        key = row.get("key") or row.get("id") or f"row_{i}"
        set_id, _, side = key.rpartition("_")
        pair = set_index.get(set_id)
        if pair is None:
            raw_path = ""
            expected_sha = None
        else:
            raw_path = pair.get(f"image{side}Path") or ""
            expected_sha = pair.get(f"image{side}_sha256_expected")
        if not raw_path:
            per_row.append({
                "key": key, "side": side,
                "status": "skipped_no_image_path",
                "error": f"no manifest entry for set {set_id} side {side}",
            })
            print(f"  [{i:>3}/{len(target_rows)}] {key} SKIP (no manifest)", flush=True)
            continue
        image_path = _resolve_image_path(
            raw_path, set_id, side, image_roots,
            expected_sha256=expected_sha,
        )
        if image_path is None:
            per_row.append({
                "key": key, "side": side,
                "status": "skipped_unresolved_image",
                "error": f"no image found for {key} (searched: {Path(raw_path).name})",
            })
            print(f"  [{i:>3}/{len(target_rows)}] {key} SKIP (unresolved)", flush=True)
            continue
        if sess is None:
            print("  (initializing rembg session — may take a moment)", flush=True)
            sess = _get_rembg_session()
        rec = analyze_row(
            sess, key, image_path,
            max_image_dim=args.max_image_dim,
        )
        per_row.append(rec)
        verdict = (
            "ACCEPT" if rec.get("accepted") is True
            else "REJECT" if rec.get("accepted") is False
            else (rec.get("status") or rec.get("trace_status") or "?")
        )
        extra = ""
        if rec.get("hard_failures"):
            extra = f"  failures={rec['hard_failures']}"
        elif rec.get("warnings"):
            extra = f"  warnings={rec['warnings']}"
        print(
            f"  [{i:>3}/{len(target_rows)}] {key} {verdict}{extra}",
            flush=True,
        )

    summary = _summarize(per_row)
    head_sha = _git_head_sha()
    artifact = {
        "head_sha": head_sha,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat() + "Z",
        "axis_truth_path": str(args.axis_truth),
        "manifest_path": str(args.manifest),
        "max_image_dim": args.max_image_dim,
        "summary": summary,
        "per_row": per_row,
    }
    args.trace_out.parent.mkdir(parents=True, exist_ok=True)
    args.trace_out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nwrote trace artifact: {args.trace_out}", flush=True)

    report = render_report(
        summary, per_row,
        head_sha=head_sha,
        axis_truth_path=args.axis_truth,
        trace_path=args.trace_out,
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report)
    print(f"wrote report: {args.report_out}", flush=True)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
