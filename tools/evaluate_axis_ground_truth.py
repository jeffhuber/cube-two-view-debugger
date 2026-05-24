#!/usr/bin/env python
"""Score a candidate cube model output against legacy axis ground truth.

Consumes the JSON produced by tools/build_axis_labeling_gallery.py +
user labeling (one entry per approved photo: vertex + 3 legacy `near_*` corners
in original-image coordinates).

The `near_*` fixture semantics are legacy after the 2026-05-23 full-corner
convention reset. A 12-row seed audit shows those fields match the
far/double-axis triplet, not canonical one-edge labels. Treat this evaluator
as historical until targets are regenerated from explicit `Va/Vb + 0..5`
labels.

For each labeled photo, given a candidate model output (vertex + 3 axes
in screen space), reports:
  - vertex_error_px      = ||candidate_vertex - true_vertex||
  - axis_angle_error_deg = best-permutation-matched angle error per axis
  - axis_length_error_px = best-permutation-matched length error per axis
  - composite_score       = weighted combination

Decoupling the metrics matters because vertex-only ground truth (the
earlier corpus, 2026-05-21) showed that 30-200 px vertex error doesn't
break color sampling — but it couldn't measure axes, which determine
face quad SHAPE (the thing that actually drives sampling reliability).

Usage:
    .venv/bin/python tools/evaluate_axis_ground_truth.py \\
        --truth tests/fixtures/gcm_axis_ground_truth.json \\
        --candidate /path/to/predicted.json

Candidate JSON schema (per pair key like "12_A"):
  {
    "vertex": [x, y],
    "axes": [[ax_dx, ax_dy], [ay_dx, ay_dy], [az_dx, az_dy]],
    // axes are DISPLACEMENT VECTORS from vertex to each near corner
  }
  OR alternatively:
  {
    "vertex": [x, y],
    "near_x": [x, y],
    "near_y": [x, y],
    "near_z": [x, y],
  }
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_candidate(entry: Dict[str, Any]) -> Optional[Tuple[Tuple[float, float], List[Tuple[float, float]]]]:
    """Return (vertex, [axis_displacement_x3]) or None if entry is malformed."""
    if "vertex" not in entry:
        return None
    vertex = (float(entry["vertex"][0]), float(entry["vertex"][1]))
    if "axes" in entry:
        axes = [(float(a[0]), float(a[1])) for a in entry["axes"]]
        if len(axes) != 3:
            return None
        return vertex, axes
    near_keys = ["near_x", "near_y", "near_z"]
    if all(k in entry for k in near_keys):
        axes = []
        for k in near_keys:
            n = entry[k]
            axes.append((float(n[0]) - vertex[0], float(n[1]) - vertex[1]))
        return vertex, axes
    return None


def _angle_deg(v: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def _normalize_angle_diff(deg: float) -> float:
    """Wrap to (-180, 180]."""
    while deg > 180:
        deg -= 360
    while deg <= -180:
        deg += 360
    return deg


def _vec_len(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _score_pair(
    truth: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """Score one labeled pair against one candidate."""
    truth_parsed = _parse_candidate(truth)
    cand_parsed = _parse_candidate(candidate)
    if truth_parsed is None:
        return {"error": "malformed_truth"}
    if cand_parsed is None:
        return {"error": "malformed_candidate"}
    t_vertex, t_axes = truth_parsed
    c_vertex, c_axes = cand_parsed

    vertex_error = math.hypot(c_vertex[0] - t_vertex[0], c_vertex[1] - t_vertex[1])

    # Brute-force assignment: 3! = 6 permutations of candidate axes onto truth axes.
    # Choose the assignment minimizing total angle error.
    best_perm: Optional[Tuple[int, int, int]] = None
    best_total_angle_err = math.inf
    best_angles: List[float] = []
    best_lengths: List[float] = []
    for perm in itertools.permutations(range(3)):
        angles = []
        lengths = []
        total_angle = 0.0
        for ti, ci in enumerate(perm):
            t_axis = t_axes[ti]
            c_axis = c_axes[ci]
            t_angle = _angle_deg(t_axis)
            c_angle = _angle_deg(c_axis)
            ad = abs(_normalize_angle_diff(c_angle - t_angle))
            angles.append(round(ad, 2))
            lengths.append(round(_vec_len(c_axis) - _vec_len(t_axis), 1))
            total_angle += ad
        if total_angle < best_total_angle_err:
            best_total_angle_err = total_angle
            best_perm = perm
            best_angles = angles
            best_lengths = lengths

    composite = vertex_error + 5.0 * (best_total_angle_err / 3.0) + 0.2 * sum(abs(l) for l in best_lengths)
    return {
        "vertex_error_px": round(vertex_error, 1),
        "axis_angle_errors_deg": best_angles,
        "axis_length_errors_px": best_lengths,
        "mean_axis_angle_error_deg": round(best_total_angle_err / 3.0, 2),
        "permutation": list(best_perm) if best_perm else None,
        "composite_score": round(composite, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True,
                        help="Ground-truth JSON (from labeling gallery)")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="Candidate JSON (same schema, per-key model outputs)")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    truth = json.loads(args.truth.read_text())
    candidate = json.loads(args.candidate.read_text())

    rows: List[Dict[str, Any]] = []
    for key, t_entry in truth.items():
        if not t_entry.get("approved", True):
            continue
        c_entry = candidate.get(key)
        if c_entry is None:
            rows.append({"key": key, "missing_candidate": True})
            continue
        score = _score_pair(t_entry, c_entry)
        score["key"] = key
        rows.append(score)

    if not args.summary_only:
        print(f"{'key':>6s} {'vertex_px':>10s} {'mean_ax_deg':>12s} {'composite':>10s}  {'angles_deg'}")
        for r in rows:
            if "error" in r or r.get("missing_candidate"):
                print(f"{r['key']:>6s}  {r.get('error') or 'missing_candidate'}")
                continue
            print(
                f"{r['key']:>6s} "
                f"{r['vertex_error_px']:>10.1f} "
                f"{r['mean_axis_angle_error_deg']:>12.2f} "
                f"{r['composite_score']:>10.1f}  "
                f"{r['axis_angle_errors_deg']}"
            )

    scored = [r for r in rows if "vertex_error_px" in r]
    if scored:
        import statistics
        verts = [r["vertex_error_px"] for r in scored]
        angles = [r["mean_axis_angle_error_deg"] for r in scored]
        composites = [r["composite_score"] for r in scored]
        print()
        print(f"=== Aggregate ({len(scored)} pairs) ===")
        print(f"  vertex_error_px:        median={statistics.median(verts):.1f}  max={max(verts):.1f}")
        print(f"  mean axis_angle_err:    median={statistics.median(angles):.2f}  max={max(angles):.2f} deg")
        print(f"  composite_score:        median={statistics.median(composites):.1f}  max={max(composites):.1f}")
    else:
        print("\nNo scored pairs (check candidate JSON keys match truth keys).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
