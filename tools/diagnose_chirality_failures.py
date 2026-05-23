#!/usr/bin/env python3
"""Enumerate + classify chirality-detector failures in the Phase 2B matrix.

Diagnostic-only — no production behavior change. Per the v2 negative
result (PR #249), the trust-ranker lever can't move the bar because the
catastrophic mode is chirality-dominated. This script characterizes
WHY the chirality detector (`tools/global_cube_model.py:_resolve_near_far_phase`)
is failing on those rows.

The detector's decision tree:
  - sep = mean_near_line_darkness - mean_far_line_darkness  (signed)
  - |sep| < 10  → "ambiguous_no_correction"  (no decision)
  - sep < 0      → "correct"  (empirical polarity: lighter near = correct)
  - sep > +10   → "flip_suggested" or "corrected_60deg_flip"

What we know from the matrix:
  - category    in {GOOD, MARGINAL, CHIRALITY_MISS, CHIRALITY_FALSE_FLIP, TRUE_GEOMETRY_FAIL}
  - phase_check in {correct, ambiguous_no_correction, corrected_60deg_flip,
                    flip_suggested_diagnostic_only, ...}
  - phase_darkness_separation: signed sep value
  - err_near_deg / err_far_deg: post-fit bearing error

A CHIRALITY_MISS or CHIRALITY_FALSE_FLIP row means err_near ≥ 25° AND
err_far < 25°. We classify each such row's FAILURE MODE based on what
the detector decided:

  - DETECTOR_AMBIGUOUS  : detector saw |sep| < 10 → no decision; but
                          a decision was needed. Failure mode is
                          "threshold-too-tight / discriminator-too-weak".
  - DETECTOR_WRONG_CALL : detector decided "correct" but the matrix says
                          the phase IS wrong; OR decided "corrected_60deg_flip"
                          but err_near is still high. Failure mode is
                          "wrong-signal / inverted-evidence".
  - PIPELINE_BUG        : phase_check is unexpected / missing.

Output: Markdown report to stdout (or --out <path>) + JSON summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = ROOT / "tests" / "fixtures" / "phase2b_recomputed_signals.json"


def _classify_failure(row: Dict[str, Any]) -> Tuple[str, str]:
    """Return (failure_mode, rationale) for a CHIRALITY_MISS /
    CHIRALITY_FALSE_FLIP row.

    failure_mode is one of:
      - DETECTOR_AMBIGUOUS    : detector said `ambiguous_no_correction`
      - DETECTOR_WRONG_CALL   : detector said `correct` / `corrected_60deg_flip`
                                but the row is catastrophic
      - PIPELINE_BUG          : phase_check is unrecognized
    """
    pc = row.get("phase_check", "?")
    sep = row.get("phase_darkness_separation")
    if pc == "ambiguous_no_correction":
        return (
            "DETECTOR_AMBIGUOUS",
            f"|sep|<10 → no decision (sep={sep})",
        )
    if pc == "correct":
        # Detector claimed phase was already correct, but err_near is high.
        return (
            "DETECTOR_WRONG_CALL",
            f"detector said 'correct' (sep={sep}) but err_near={row.get('err_near_deg')}°",
        )
    if pc == "corrected_60deg_flip":
        return (
            "DETECTOR_WRONG_CALL",
            f"detector flipped (sep={sep}) but err_near still {row.get('err_near_deg')}°",
        )
    if pc == "flip_suggested_diagnostic_only":
        # The detector identified a flip but apply_correction=False; this
        # means the matrix recompute didn't apply the correction. Not a
        # detector bug, a pipeline configuration. Worth flagging separately.
        return (
            "FLIP_SUGGESTED_NOT_APPLIED",
            f"detector suggested flip (sep={sep}) but apply_correction=False; "
            f"err_near={row.get('err_near_deg')}°",
        )
    return ("PIPELINE_BUG", f"unexpected phase_check={pc!r}")


def _compute_meta_signal_stats(by_case: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """For each per-row matrix feature, compute IQR distributions of:
      - RIGHT-call rows: phase_check ∈ {correct, corrected_60deg_flip}
        AND category ∈ {GOOD, MARGINAL} (detector decided correctly)
      - WRONG-call rows: same phase_check values but category ∈
        {CHIRALITY_MISS, CHIRALITY_FALSE_FLIP} (detector decided wrong)

    Returns:
      {
        "right_n": int, "wrong_n": int,
        "features": {feat_name: {"right": stats, "wrong": stats,
                                  "iqr_overlap": bool}}
      }
    """
    decided_phase_checks = ("correct", "corrected_60deg_flip")
    catastrophic_cats = ("CHIRALITY_MISS", "CHIRALITY_FALSE_FLIP")
    right_rows: List[Dict[str, Any]] = []
    wrong_rows: List[Dict[str, Any]] = []
    for runs in by_case.values():
        for r in runs:
            pc = r.get("phase_check")
            cat = r.get("category")
            if pc not in decided_phase_checks:
                continue
            if cat in ("GOOD", "MARGINAL"):
                right_rows.append(r)
            elif cat in catastrophic_cats:
                wrong_rows.append(r)

    candidate_features = [
        "fit_residual_rms_px",
        "pnp_rms_px",
        "hexagon_centroid_vs_bezel_vertex_offset_px",
        "bezel_vs_fit_cube_center_offset_px",
        "junction_score_at_ensemble",
        "ensemble_shift_px",
        "ensemble_n_candidates",
        "phase_darkness_separation",
    ]

    def _quartiles(vals):
        """Return (q1, q3) using a Python 3.6+ compatible implementation
        (statistics.quantiles is 3.8+; Codex P1 on PR #250 round 5).
        Uses linear interpolation across sorted positions — equivalent
        to numpy.percentile([25, 75]) and pandas default."""
        s = sorted(vals)
        n = len(s)
        def _pct(p):
            if n == 1:
                return s[0]
            k = (n - 1) * (p / 100.0)
            lo = int(k)
            hi = min(lo + 1, n - 1)
            frac = k - lo
            return s[lo] + frac * (s[hi] - s[lo])
        return _pct(25.0), _pct(75.0)

    def _stat(rows, key):
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if len(vals) < 4:
            return None
        q1, q3 = _quartiles(vals)
        return {
            "n": len(vals),
            "median": round(statistics.median(vals), 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
        }

    def _vals(rows, key):
        return [
            float(r.get(key)) for r in rows if r.get(key) is not None
        ]

    feature_results = {}
    for feat in candidate_features:
        r = _stat(right_rows, feat)
        w = _stat(wrong_rows, feat)
        if r is None or w is None:
            feature_results[feat] = {
                "right": r, "wrong": w, "iqr_overlap": None,
                "threshold_tradeoff": None,
            }
            continue
        iqr_overlap = (max(r["q1"], w["q1"]) <= min(r["q3"], w["q3"]))
        threshold_tradeoff = None
        # For IQR-disjoint features, compute the actual count of rows
        # above/below the IQR-midpoint threshold so the report can
        # disclose the real FP/FN trade-off (Codex P2 round 3 — IQR
        # disjoint does NOT mean full ranges are disjoint).
        if not iqr_overlap:
            right_vals = _vals(right_rows, feat)
            wrong_vals = _vals(wrong_rows, feat)
            if r["q1"] > w["q3"]:
                # RIGHT above WRONG → gate "feat < threshold"
                comparison = "<"
                threshold = round((w["q3"] + r["q1"]) / 2.0, 1)
                right_in_gate = sum(1 for v in right_vals if v < threshold)
                wrong_in_gate = sum(1 for v in wrong_vals if v < threshold)
            elif w["q1"] > r["q3"]:
                comparison = ">"
                threshold = round((r["q3"] + w["q1"]) / 2.0, 1)
                right_in_gate = sum(1 for v in right_vals if v > threshold)
                wrong_in_gate = sum(1 for v in wrong_vals if v > threshold)
            else:
                # Defensive: shouldn't happen given the iqr_overlap=False
                # branch above. Skip threshold suggestion in that case.
                comparison = "?"
                threshold = None
                right_in_gate = wrong_in_gate = 0
            threshold_tradeoff = {
                "threshold": threshold,
                "comparison": comparison,
                "right_in_gate": right_in_gate,
                "wrong_in_gate": wrong_in_gate,
            }
        feature_results[feat] = {
            "right": r,
            "wrong": w,
            "iqr_overlap": iqr_overlap,
            "threshold_tradeoff": threshold_tradeoff,
        }

    return {
        "right_n": len(right_rows),
        "wrong_n": len(wrong_rows),
        "features": feature_results,
    }


def analyze(matrix_path: Path) -> Dict[str, Any]:
    """Return a dict of analysis results. Keys:
        - total_rows, total_cases
        - per_category_counts
        - chirality_rows: list of per-row dicts
        - failure_mode_counts: Counter
        - sep_stats: distribution stats per (category, phase_check) pairing
        - meta_signal_stats: live IQR-disjoint check on candidate
          meta-signals for the wrong-call subset
    """
    data = json.loads(matrix_path.read_text())
    by_case = data.get("by_case", {})

    chirality_categories = {"CHIRALITY_MISS", "CHIRALITY_FALSE_FLIP"}

    total_rows = 0
    chirality_rows: List[Dict[str, Any]] = []
    per_category: Counter = Counter()

    for case_key, runs in by_case.items():
        for r in runs:
            total_rows += 1
            cat = r.get("category", "?")
            per_category[cat] += 1
            if cat in chirality_categories:
                row = dict(r)
                row["case"] = case_key
                fmode, rationale = _classify_failure(row)
                row["_failure_mode"] = fmode
                row["_failure_rationale"] = rationale
                chirality_rows.append(row)

    failure_mode_counts: Counter = Counter(
        r["_failure_mode"] for r in chirality_rows
    )

    # Separation distribution per (category, phase_check).
    sep_by_group: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in chirality_rows:
        key = (r.get("category", "?"), r.get("phase_check", "?"))
        sep = r.get("phase_darkness_separation")
        if sep is not None:
            sep_by_group[key].append(float(sep))

    sep_stats: Dict[str, Dict[str, Any]] = {}
    for (cat, pc), seps in sep_by_group.items():
        key = f"{cat} | {pc}"
        sep_stats[key] = {
            "n": len(seps),
            "median": round(statistics.median(seps), 1) if seps else None,
            "min": round(min(seps), 1) if seps else None,
            "max": round(max(seps), 1) if seps else None,
        }

    # Render matrix_path repo-relative when possible so committed
    # reports don't leak machine-specific checkout paths or usernames.
    # Codex P3 on the diagnostic PR.
    try:
        matrix_path_for_report = str(matrix_path.resolve().relative_to(ROOT))
    except ValueError:
        matrix_path_for_report = str(matrix_path)

    meta_signal_stats = _compute_meta_signal_stats(by_case)

    return {
        "matrix_path": matrix_path_for_report,
        "total_rows": total_rows,
        "total_cases": len(by_case),
        "per_category_counts": dict(per_category),
        "chirality_rows": chirality_rows,
        "failure_mode_counts": dict(failure_mode_counts),
        "sep_stats_by_category_and_phase_check": sep_stats,
        "meta_signal_stats": meta_signal_stats,
    }


def render_report(result: Dict[str, Any]) -> str:
    """Produce a Markdown report from `analyze()` output."""
    lines: List[str] = []
    lines.append("# Chirality detector failure analysis")
    lines.append("")
    lines.append(f"Source matrix: `{result['matrix_path']}`")
    lines.append(
        f"Total rows: {result['total_rows']} "
        f"({result['total_cases']} cases × ~2 runs each)"
    )
    lines.append("")

    lines.append("## Per-category row counts")
    lines.append("")
    lines.append("| Category | Rows |")
    lines.append("|---|---|")
    for cat, n in sorted(
        result["per_category_counts"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"| `{cat}` | {n} |")
    lines.append("")

    lines.append("## Chirality-row failure modes")
    lines.append("")
    chirality_total = sum(result["failure_mode_counts"].values())
    if chirality_total == 0:
        lines.append("No chirality-failure rows in this matrix.")
        return "\n".join(lines)
    lines.append(f"Total chirality-failure rows: {chirality_total}")
    lines.append("")
    lines.append("| Failure mode | Rows | % |")
    lines.append("|---|---:|---:|")
    for fmode, n in sorted(
        result["failure_mode_counts"].items(), key=lambda kv: -kv[1]
    ):
        pct = 100.0 * n / chirality_total
        lines.append(f"| `{fmode}` | {n} | {pct:.1f}% |")
    lines.append("")

    lines.append("### What each failure mode means")
    lines.append("")
    lines.append(
        "- `DETECTOR_AMBIGUOUS`: the darkness-separation discriminator "
        "saw `|sep| < 10` and declined to make a decision. The detector "
        "needs a stronger or different signal to resolve these rows."
    )
    lines.append(
        "- `DETECTOR_WRONG_CALL`: the detector confidently chose a phase "
        "(`correct` or `corrected_60deg_flip`) but the resulting axes are "
        "still wrong (err_near ≥ 25°). The signal exists and is strong, "
        "but the decision is inverted or the threshold/polarity is "
        "mis-calibrated for these rows."
    )
    lines.append(
        "- `FLIP_SUGGESTED_NOT_APPLIED`: detector identified a flip but "
        "the recompute pipeline ran with `apply_correction=False`. This "
        "is a pipeline-configuration finding, not a detector bug."
    )
    lines.append(
        "- `PIPELINE_BUG`: `phase_check` value not in the expected set — "
        "warrants direct investigation."
    )
    lines.append("")

    lines.append("## Separation distribution by category × phase_check")
    lines.append("")
    lines.append("| Category × phase_check | n | median sep | min | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, stats in sorted(
        result["sep_stats_by_category_and_phase_check"].items()
    ):
        # Escape the pipe in the dict key (which is "category | phase_check")
        # so the Markdown renderer doesn't interpret it as a column boundary
        # (Codex P2 on PR #250 round 2).
        safe_key = key.replace("|", r"\|")
        lines.append(
            f"| {safe_key} | {stats['n']} | {stats['median']} | "
            f"{stats['min']} | {stats['max']} |"
        )
    lines.append("")

    lines.append("## Key findings")
    lines.append("")

    # Gather summary statistics ONLY for DETECTOR_WRONG_CALL rows — those
    # are the rows where the polarity-rule claim is meaningful. Codex P2:
    # the prior version unconditionally claimed polarity-inversion even
    # for matrices with only AMBIGUOUS or FLIP_SUGGESTED_NOT_APPLIED rows.
    wrong_call_rows = [
        r for r in result["chirality_rows"]
        if r.get("_failure_mode") == "DETECTOR_WRONG_CALL"
    ]
    wrong_call_pcs: Dict[str, Dict[str, Any]] = {}
    for r in wrong_call_rows:
        pc = r.get("phase_check", "?")
        sep = r.get("phase_darkness_separation")
        err_n = r.get("err_near_deg")
        d = wrong_call_pcs.setdefault(pc, {"n": 0, "seps": [], "err_nears": []})
        d["n"] += 1
        if sep is not None:
            d["seps"].append(float(sep))
        if err_n is not None:
            d["err_nears"].append(float(err_n))
    for pc, s in wrong_call_pcs.items():
        if s["seps"]:
            s["sep_median"] = round(statistics.median(s["seps"]), 1)
            s["sep_sign_all_same"] = (
                all(v >= 0 for v in s["seps"])
                or all(v <= 0 for v in s["seps"])
            )
        if s["err_nears"]:
            s["err_near_median"] = round(statistics.median(s["err_nears"]), 1)

    finding_idx = 0
    if wrong_call_rows:
        finding_idx += 1
        lines.append(
            f"{finding_idx}. **The detector's polarity rule is being correctly "
            f"applied — but to the wrong sign on a specific subset of rows "
            f"({len(wrong_call_rows)} `DETECTOR_WRONG_CALL` rows).** Looking "
            f"at the per-row patterns:"
        )
        lines.append("")
        for pc, s in sorted(wrong_call_pcs.items(), key=lambda kv: -kv[1]["n"]):
            sep_med = s.get("sep_median", "?")
            all_same = s.get("sep_sign_all_same", False)
            sign_word = "all same sign" if all_same else "mixed signs"
            err_med = s.get("err_near_median", "?")
            lines.append(
                f"   - `{pc}` ({s['n']} rows): sep median {sep_med} "
                f"({sign_word}); err_near median {err_med}° after the "
                f"detector's decision. The detector confidently chose this "
                f"branch and got it wrong on these rows."
            )
        lines.append("")
        lines.append(
            "   These rows are NOT random noise — the sep signal is unambiguous "
            "and the detector's polarity rule is firing as documented. The rule's "
            "underlying assumption (NEG sep = correct, POS sep = needs flip) "
            "**simply does not hold on this subset**. Something about these "
            "specific images (lighting? sticker color? bezel contrast? "
            "vertex offset?) inverts the polarity."
        )
        lines.append("")

    n_ambig = sum(
        1 for r in result["chirality_rows"]
        if r.get("phase_check") == "ambiguous_no_correction"
    )
    if n_ambig and chirality_total > 0:
        finding_idx += 1
        pct = 100.0 * n_ambig / chirality_total
        lines.append(
            f"{finding_idx}. **DETECTOR_AMBIGUOUS ({n_ambig} rows, "
            f"{pct:.0f}% of chirality failures) need a stronger "
            f"discriminator.** The `|sep| < 10` band leaves these rows "
            f"undecided. They have valid geometry except for the 60° "
            f"phase ambiguity — solving them would directly reduce the "
            f"`CHIRALITY_MISS` rate."
        )
        lines.append("")

    # Recommended next experiments — only emit case-specific recommendations
    # if there ARE wrong-call rows to inspect. Codex P3: avoid hard-coded
    # case lists/percentages that go stale on different matrices.
    lines.append(
        "## Recommended next experiments (out of scope for this diagnostic PR)"
    )
    lines.append("")
    if wrong_call_rows:
        wrong_case_ids = sorted({
            r.get("case", "?").rsplit("_", 1)[0]
            for r in wrong_call_rows
        }, key=lambda s: (int(s) if s.isdigit() else 99999))
        wrong_case_str = ", ".join(wrong_case_ids)
        lines.append(
            "- **Look for a meta-signal that predicts polarity inversion.** The "
            "matrix already records other per-row signals (fit_residual_rms_px, "
            "vertex_ensemble_stddev_px, junction_score_at_ensemble, "
            "ensemble_shift_px, etc). Do any correlate with the "
            "`DETECTOR_WRONG_CALL` rows above? If yes, the detector could gate "
            "its polarity rule on the meta-signal."
        )
        lines.append(
            f"- **Visual inspection of the {len(wrong_call_rows)} wrong-call "
            f"rows.** What is physically different about Sets {wrong_case_str} "
            f"that inverts the darkness polarity? Bezel reflectivity, lighting "
            f"angle, sticker color saturation are the candidate variables."
        )
    if n_ambig:
        lines.append(
            "- **Strengthen the ambiguous-band discriminator.** Candidate "
            "signals: per-line darkness variance (not just mean), "
            "color-saturation along the lines, edge-gradient orientation, "
            "or multiple lines per corner (not just vertex→corner)."
        )
    # Coverage for the other recognized failure modes — emit specific
    # recommendations rather than silently dropping them. Codex P3 on
    # the diagnostic PR.
    n_flip_suggested = sum(
        1 for r in result["chirality_rows"]
        if r.get("_failure_mode") == "FLIP_SUGGESTED_NOT_APPLIED"
    )
    n_pipeline_bug = sum(
        1 for r in result["chirality_rows"]
        if r.get("_failure_mode") == "PIPELINE_BUG"
    )
    if n_flip_suggested:
        lines.append(
            f"- **Configure `apply_correction=True` for the matrix recompute.** "
            f"{n_flip_suggested} chirality-failure rows are "
            f"`FLIP_SUGGESTED_NOT_APPLIED` — the detector identified a "
            f"flip but the pipeline ran without applying it. This is a "
            f"pipeline-configuration finding, not a detector failure."
        )
    if n_pipeline_bug:
        lines.append(
            f"- **Investigate {n_pipeline_bug} `PIPELINE_BUG` rows.** Their "
            f"`phase_check` values are outside the documented set; check "
            f"whether `_resolve_near_far_phase` has emitted a new status "
            f"that this diagnostic isn't aware of."
        )
    if chirality_total == 0:
        lines.append("- (No chirality failures in this matrix.)")
    lines.append("")

    lines.append(
        "## Meta-signal candidate for the wrong-call subset"
    )
    lines.append("")
    meta = result.get("meta_signal_stats", {})
    right_n = meta.get("right_n", 0)
    wrong_n = meta.get("wrong_n", 0)
    feat_results = meta.get("features", {})
    if right_n == 0 or wrong_n == 0:
        lines.append(
            "(No `RIGHT-call` or `WRONG-call` rows in this matrix — meta-signal "
            "analysis is not applicable.)"
        )
        lines.append("")
    else:
        lines.append(
            f"Live compare of per-row feature distributions between two "
            f"groups in `{result['matrix_path']}`:"
        )
        lines.append("")
        lines.append(
            f"- **RIGHT-call rows** (n={right_n}): detector decided "
            f"(`phase_check ∈ {{correct, corrected_60deg_flip}}`) AND the row "
            f"outcome is GOOD or MARGINAL"
        )
        lines.append(
            f"- **WRONG-call rows** (n={wrong_n}): same `phase_check` values "
            f"but the row is `CHIRALITY_MISS` or `CHIRALITY_FALSE_FLIP`"
        )
        lines.append("")
        # Order features: IQR-disjoint first (strongest separation), then
        # by median-ratio magnitude for the overlapping ones.
        ordered = []
        for feat, fr in feat_results.items():
            r = fr.get("right")
            w = fr.get("wrong")
            if r is None or w is None:
                continue
            r_med, w_med = r["median"], w["median"]
            ratio = abs((w_med - r_med) / max(abs(r_med), 1e-6))
            ordered.append((
                feat, r, w, fr.get("iqr_overlap", True), ratio
            ))
        ordered.sort(key=lambda t: (t[3], -t[4]))  # disjoint first, then by ratio desc
        lines.append(
            "| Feature | RIGHT median | RIGHT IQR | WRONG median | WRONG IQR | Verdict |"
        )
        lines.append(
            "|---|---:|---|---:|---|---|"
        )
        for feat, r, w, overlap, ratio in ordered:
            verdict = "IQR-overlap" if overlap else "**IQR-DISJOINT**"
            lines.append(
                f"| `{feat}` | {r['median']} | [{r['q1']}, {r['q3']}] | "
                f"{w['median']} | [{w['q1']}, {w['q3']}] | {verdict} |"
            )
        lines.append("")
        disjoint_features = [
            (feat, r, w) for feat, r, w, overlap, _ in ordered if not overlap
        ]
        if disjoint_features:
            # disjoint_features is a list of (feat, right_stat, wrong_stat)
            # from the IQR-disjoint pass above. Use the helper-precomputed
            # trade-off counts so the report stays matrix-agnostic.
            f, r_stat, w_stat = disjoint_features[0]
            tradeoff = feat_results[f].get("threshold_tradeoff") or {}
            comparison_op = tradeoff.get("comparison", "?")
            thr = tradeoff.get("threshold")
            right_in_gate = tradeoff.get("right_in_gate", 0)
            wrong_in_gate = tradeoff.get("wrong_in_gate", 0)
            # Direction-of-separation labels for the prose.
            if r_stat["q1"] > w_stat["q3"]:
                higher_label, lower_label = "RIGHT", "WRONG"
                higher_stat, lower_stat = r_stat, w_stat
                gate_explanation = (
                    "treat low-feature rows (the WRONG-call territory) "
                    "as ambiguous"
                )
            else:
                higher_label, lower_label = "WRONG", "RIGHT"
                higher_stat, lower_stat = w_stat, r_stat
                gate_explanation = (
                    "treat high-feature rows (the WRONG-call territory) "
                    "as ambiguous"
                )

            # Determine whether the full min/max ranges overlap so we
            # don't hard-code an "overlap" conclusion when the next
            # matrix produces a feature with clean range separation
            # (Codex P3 on PR #250 round 4).
            full_ranges_overlap = (
                max(r_stat["min"], w_stat["min"])
                <= min(r_stat["max"], w_stat["max"])
            )
            range_clause = (
                f"**The full ranges also overlap** (RIGHT min/max "
                f"{r_stat['min']}/{r_stat['max']}; WRONG min/max "
                f"{w_stat['min']}/{w_stat['max']}), so the IQR-only "
                f"separation does NOT cleanly partition the populations."
                if full_ranges_overlap
                else (
                    f"**The full min/max ranges are also disjoint** "
                    f"(RIGHT min/max {r_stat['min']}/{r_stat['max']}; "
                    f"WRONG min/max {w_stat['min']}/{w_stat['max']}), "
                    f"so a threshold within the gap would cleanly "
                    f"partition this matrix's populations."
                )
            )
            lines.append(
                f"**Interpretation (live):** `{f}` has IQR-disjoint "
                f"separation between the right-call and wrong-call "
                f"groups — {higher_label} IQR "
                f"[{higher_stat['q1']}, {higher_stat['q3']}] lies entirely "
                f"above {lower_label} IQR "
                f"[{lower_stat['q1']}, {lower_stat['q3']}]. "
                f"{range_clause}"
            )
            lines.append("")
            if thr is not None:
                lines.append(
                    f"**Candidate-threshold trade-off at `{f} "
                    f"{comparison_op} {thr}`** (midpoint of the two IQRs):"
                )
                lines.append("")
                lines.append(
                    f"- **{wrong_in_gate}/{wrong_n}** WRONG-call rows "
                    f"would be gated to `ambiguous_no_correction` "
                    f"(benefit: those rows avoid the bad polarity "
                    f"decision)."
                )
                lines.append(
                    f"- **{right_in_gate}/{right_n}** RIGHT-call rows "
                    f"would ALSO be gated (cost: those rows lose the "
                    f"correct polarity decision; some may flip back to "
                    f"`ambiguous` and remain correct anyway, others may "
                    f"regress)."
                )
                lines.append("")
                lines.append(
                    f"**Actionable hypothesis for next fix PR:** gate "
                    f"the polarity rule on `{f}`. The IQR-midpoint "
                    f"threshold `{thr}` is the starting point; the "
                    f"actual operating point should be calibrated on a "
                    f"held-out split with the FP/FN trade-off above made "
                    f"explicit. Likely the right approach is a soft "
                    f"confidence rather than a hard binary gate — i.e., "
                    f"{gate_explanation}."
                )
                lines.append("")
        else:
            lines.append(
                "**No feature has IQR-disjoint separation on this matrix.** "
                "No clean meta-signal candidate from the current per-row "
                "features; would need a new measurement or per-image "
                "inspection of the wrong-call subset to make progress."
            )
            lines.append("")

    lines.append("## Per-row detail (chirality failures only)")
    lines.append("")
    lines.append(
        "| Case | Run | Category | phase_check | sep | err_near | err_far | Failure mode |"
    )
    lines.append("|---|---:|---|---|---:|---:|---:|---|")
    for r in sorted(
        result["chirality_rows"],
        key=lambda r: (r.get("case", ""), r.get("run", -1)),
    ):
        lines.append(
            f"| `{r.get('case', '?')}` "
            f"| {r.get('run', '?')} "
            f"| `{r.get('category', '?')}` "
            f"| `{r.get('phase_check', '?')}` "
            f"| {r.get('phase_darkness_separation', '?')} "
            f"| {r.get('err_near_deg', '?')} "
            f"| {r.get('err_far_deg', '?')} "
            f"| {r['_failure_mode']} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX,
        help=f"Path to phase2b_recomputed_signals.json (default {DEFAULT_MATRIX})",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Write Markdown report to this path. If omitted, prints to stdout.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write JSON analysis to this path (machine-readable).",
    )
    args = ap.parse_args()

    if not args.matrix.exists():
        print(f"error: matrix not found at {args.matrix}", flush=True)
        return 1

    result = analyze(args.matrix)
    report = render_report(result)

    if args.out_md is None:
        print(report)
    else:
        args.out_md.write_text(report)
        print(f"wrote Markdown report to {args.out_md}")

    if args.out_json is not None:
        args.out_json.write_text(json.dumps(result, indent=2))
        print(f"wrote JSON analysis to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
