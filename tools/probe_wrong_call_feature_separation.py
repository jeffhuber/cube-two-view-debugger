#!/usr/bin/env python3
"""Probe: does any existing matrix feature correlate with the DETECTOR_WRONG_CALL
subset of chirality failures?

Per the chirality diagnostic (PR #250), 19/32 chirality-failure rows have
phase_check ∈ {correct, corrected_60deg_flip} but err_near is still ≥25°.
The detector applies its polarity rule correctly but the rule's
underlying assumption is INVERTED on this subset.

If a per-row matrix feature correlates with the wrong-call vs the
right-call rows, we could gate the polarity rule on it.

Approach: for each per-row feature, compare distributions between:
  - "Polarity-rule-rejected" rows: phase_check is 'correct' or
    'corrected_60deg_flip', and the row outcome is GOOD/MARGINAL
    (the detector got it right)
  - "Polarity-rule-wrong" rows: same phase_check values but row is
    CHIRALITY_MISS or CHIRALITY_FALSE_FLIP (the detector got it wrong)

If a feature has clearly different distributions, that's a candidate
meta-signal.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


# Derive repo root from this file's location so the probe works from
# any checkout / worktree (Codex P2 on PR #250 round 2).
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = ROOT / "tests" / "fixtures" / "phase2b_recomputed_signals.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX,
        help=f"Path to phase2b_recomputed_signals.json (default: {DEFAULT_MATRIX})",
    )
    args = ap.parse_args()
    if not args.matrix.exists():
        raise SystemExit(f"matrix not found at {args.matrix}")
    data = json.loads(args.matrix.read_text())

    # Buckets: rows where the detector made a confident phase decision
    # (not 'ambiguous_no_correction'), partitioned by whether the outcome
    # was actually correct or not.
    right_rows = []  # detector made a call AND the row is GOOD/MARGINAL
    wrong_rows = []  # detector made a call AND the row is CHIRALITY_*

    for case_key, runs in data["by_case"].items():
        for r in runs:
            pc = r.get("phase_check", "?")
            if pc not in ("correct", "corrected_60deg_flip"):
                continue  # skip ambiguous or skipped rows
            cat = r.get("category", "?")
            if cat in ("GOOD", "MARGINAL"):
                right_rows.append(dict(r, _case=case_key))
            elif cat in ("CHIRALITY_MISS", "CHIRALITY_FALSE_FLIP"):
                wrong_rows.append(dict(r, _case=case_key))

    print(f"Right-call rows (detector decided, row is GOOD/MARGINAL): {len(right_rows)}")
    print(f"Wrong-call rows (detector decided, row is CHIRALITY_*):   {len(wrong_rows)}")
    print()

    # Per-feature compare. List of feature names to probe.
    features = [
        "fit_residual_rms_px",
        "pnp_rms_px",
        "hexagon_centroid_vs_bezel_vertex_offset_px",
        "bezel_vs_fit_cube_center_offset_px",
        "junction_score_at_ensemble",
        "ensemble_shift_px",
        "ensemble_n_candidates",
        "phase_darkness_separation",
    ]

    def _pct(s, n, p):
        # Python 3.6+ compatible percentile (statistics.quantiles is 3.8+,
        # Codex P1 on PR #250 round 5).
        if n == 1:
            return s[0]
        k = (n - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, n - 1)
        frac = k - lo
        return s[lo] + frac * (s[hi] - s[lo])

    def stat(rows, key):
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        s = sorted(vals)
        n = len(s)
        return {
            "n": n,
            "median": round(statistics.median(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "q1": round(_pct(s, n, 25.0), 2) if n >= 4 else None,
            "q3": round(_pct(s, n, 75.0), 2) if n >= 4 else None,
        }

    print(f"{'Feature':<45} {'RIGHT median':>14} {'WRONG median':>14}  {'separation hint'}")
    print("-" * 100)
    for f in features:
        rs = stat(right_rows, f)
        ws = stat(wrong_rows, f)
        if rs is None or ws is None:
            print(f"{f:<45}  (insufficient data)")
            continue
        # Separation hint: median ratio and whether the IQRs overlap
        r_med = rs["median"]
        w_med = ws["median"]
        sep_ratio = (w_med - r_med) / max(abs(r_med), 1e-6)
        # IQR overlap check
        r_q1, r_q3 = rs["q1"], rs["q3"]
        w_q1, w_q3 = ws["q1"], ws["q3"]
        if r_q1 is None or w_q1 is None:
            overlap_msg = "(n too small for IQR)"
        else:
            overlap = max(r_q1, w_q1) <= min(r_q3, w_q3)
            overlap_msg = "IQR-overlap" if overlap else "IQR-DISJOINT"
        print(f"{f:<45} {r_med:>14} {w_med:>14}  {sep_ratio:+.2f}× ratio, {overlap_msg}")

    print()
    print("FULL distributions for the features with separation hints:")
    for f in features:
        rs = stat(right_rows, f)
        ws = stat(wrong_rows, f)
        if rs is None or ws is None:
            continue
        print(f"\n  {f}:")
        print(f"    RIGHT (n={rs['n']}): median {rs['median']}, IQR [{rs['q1']}, {rs['q3']}], min/max {rs['min']}/{rs['max']}")
        print(f"    WRONG (n={ws['n']}): median {ws['median']}, IQR [{ws['q1']}, {ws['q3']}], min/max {ws['min']}/{ws['max']}")


if __name__ == "__main__":
    main()
