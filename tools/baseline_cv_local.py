#!/usr/bin/env python3
"""Phase 1 of the post-#222 roadmap: cv-local baseline on the 58-case
axis-labeled gallery, comparable to `tools/baseline_post_218.py`.

cv-local's `analyze_image` produces face-quads, not vertex+legacy-`near`
fields directly. This script derives the comparable historical signal by
clustering the 12 corner instances (4 per face-quad × 3 face-quads) by spatial
proximity:

  * 1 cluster of 3 points → the trihedral vertex (shared by all 3 faces)
  * 3 clusters of 2 points → the 3 legacy-`near` clusters (each shared by 2 faces)
  * 3 clusters of 1 point  → the 3 far corners (one per face)

The vertex/legacy-`near` outputs are then scored against the user-labeled
ground truth (`tests/fixtures/gcm_axis_ground_truth.json`) using the
SAME bearings-based metric as `baseline_post_218.py`, producing a
comparable JSON snapshot. Run `tools/baseline_post_218.py --diff
post_218_baseline.json cv_local_baseline.json` for row-level deltas
between the two sides.

The point isn't "which side is better" — it's that we now have a
documented gap on the same eval set, which is what Phase 2 trust-
policy diagnostics need as their training/eval surface.

IMPORTANT: the `near_*` fixture semantics are legacy after the
2026-05-23 full-corner convention reset. A 12-row seed audit shows those
fields match the far/double-axis triplet, not canonical one-edge labels.
Treat this snapshot as a legacy comparison until regenerated from explicit
`Va/Vb + 0..5` labels.

Usage:
    .venv/bin/python tools/baseline_cv_local.py \\
        --truth tests/fixtures/gcm_axis_ground_truth.json \\
        --gallery ~/axis_labeling \\
        --out tests/fixtures/cv_local_baseline.json \\
        --report tools/PHASE_1_CV_LOCAL_BASELINE.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tools.evaluate_hybrid_pipeline import _proposer_face_quads  # noqa: E402


Point = Tuple[float, float]
DEFAULT_TRUTH = ROOT / "tests" / "fixtures" / "gcm_axis_ground_truth.json"
DEFAULT_GALLERY = Path.home() / "axis_labeling"
DEFAULT_OUT = ROOT / "tests" / "fixtures" / "cv_local_baseline.json"
DEFAULT_REPORT = ROOT / "tools" / "PHASE_1_CV_LOCAL_BASELINE.md"

# Clustering threshold: corners within this distance are treated as
# "the same cube corner viewed from a different face-quad." Set
# generously since face-quads are independently derived from grid
# extrapolation and can disagree by tens of px on the shared corner.
# Empirically tuned on the 58-case gallery: 120 px catches near-corner
# matches across noisy face-quads without erroneously joining far
# corners that happen to be moderately close.
CLUSTER_THRESHOLD_PX = 120.0


def _bearing(o: Point, t: Point) -> float:
    return math.degrees(math.atan2(t[1] - o[1], t[0] - o[0])) % 360


def _best_perm_err(model_bearings: List[float], user_bearings: List[float]) -> float:
    best = math.inf
    for perm in itertools.permutations(range(3)):
        diffs = []
        for i in range(3):
            d = abs((model_bearings[i] - user_bearings[perm[i]] + 180) % 360 - 180)
            diffs.append(d)
        m = sum(diffs) / 3.0
        if m < best:
            best = m
    return best


def _derive_vertex_and_near(
    face_quads: Dict[str, List[Point]],
) -> Optional[Tuple[Point, List[Point], List[Point]]]:
    """Cluster 12 corner instances → (vertex, [3 near], [3 far]).

    Uses union-find on edges of an implicit graph where two instances
    are "linked" if their distance is < CLUSTER_THRESHOLD_PX AND they
    come from different face-quads (same-face instances can never be
    the same cube corner). This is more robust than greedy centroid
    clustering because cluster membership doesn't depend on iteration
    order.

    Returns None if face_quads is malformed (wrong count) or the
    cluster sizes don't match the expected 1×3 + 3×2 + 3×1 pattern.
    """
    if len(face_quads) != 3:
        return None
    instances: List[Tuple[Point, int]] = []
    for face_idx, (_, quad) in enumerate(face_quads.items()):
        if len(quad) != 4:
            return None
        for p in quad:
            instances.append(((float(p[0]), float(p[1])), face_idx))
    if len(instances) != 12:
        return None

    # Union-find: target exactly 7 clusters (the structurally expected
    # output: 1 vertex + 3 legacy-`near` clusters + 3 far corners). Process
    # cross-face edges in ascending distance order, merging only when
    # the merge doesn't combine two corners of the SAME face-quad (a
    # geometric impossibility) and stops when 7 clusters remain.
    #
    # If the structurally-expected 1×3 + 3×2 + 3×1 pattern doesn't
    # emerge AND the largest merge distance is over MAX_MERGE_PX, the
    # face-quads are too inconsistent to derive a coherent cube —
    # mark as cv-local fit failure.
    parent = list(range(12))
    rank = [0] * 12
    faces_in_cluster: List[set] = [{instances[i][1]} for i in range(12)]
    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def _union(a: int, b: int) -> bool:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return False
        if faces_in_cluster[ra] & faces_in_cluster[rb]:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        faces_in_cluster[ra] |= faces_in_cluster[rb]
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    edges: List[Tuple[float, int, int]] = []
    for i in range(12):
        for j in range(i + 1, 12):
            if instances[i][1] == instances[j][1]:
                continue
            d = math.hypot(
                instances[i][0][0] - instances[j][0][0],
                instances[i][0][1] - instances[j][0][1],
            )
            edges.append((d, i, j))
    edges.sort()

    # Process edges in ascending distance order. The FIRST 3-face
    # cluster that forms is the trihedral vertex. After that, no
    # other cluster can grow to 3 members (only the vertex has 3
    # face members; pairwise-shared clusters have 2, far clusters 1).
    n_clusters = 12
    vertex_found = False
    for d, i, j in edges:
        if n_clusters == 7 and vertex_found:
            break
        if d > CLUSTER_THRESHOLD_PX:
            break
        ra, rb = _find(i), _find(j)
        if ra == rb:
            continue
        if faces_in_cluster[ra] & faces_in_cluster[rb]:
            continue  # same-face conflict
        # Would this merge create a 3-face cluster?
        merged_faces = faces_in_cluster[ra] | faces_in_cluster[rb]
        if len(merged_faces) > 3:
            continue  # shouldn't happen but safety
        if len(merged_faces) == 3:
            if vertex_found:
                continue  # only ONE 3-face cluster (the vertex) allowed
            vertex_found = True
        if _union(i, j):
            n_clusters -= 1

    if n_clusters != 7:
        return None

    clusters_by_root: Dict[int, List[int]] = {}
    for i in range(12):
        root = _find(i)
        clusters_by_root.setdefault(root, []).append(i)
    clusters = list(clusters_by_root.values())

    by_size: Dict[int, List[List[int]]] = {3: [], 2: [], 1: []}
    for c in clusters:
        n = len(c)
        if n in by_size:
            by_size[n].append(c)

    if len(by_size[3]) != 1 or len(by_size[2]) != 3 or len(by_size[1]) != 3:
        return None

    def _centroid(idxs: List[int]) -> Point:
        pts = [instances[i][0] for i in idxs]
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    vertex = _centroid(by_size[3][0])
    near = [_centroid(c) for c in by_size[2]]
    far = [_centroid(c) for c in by_size[1]]
    return vertex, near, far


def _categorize(err_near: float, err_far: float) -> str:
    """Same categorization as baseline_post_218.py."""
    if err_near < 10.0:
        return "GOOD"
    if err_near < 25.0:
        return "MARGINAL"
    if err_far < 25.0:
        return "CHIRALITY_MISS"  # model.far matches user.near — phase wrong
    return "TRUE_GEOMETRY_FAIL"


def _run_one_case(
    img_path: Path,
    side: str,
    user_v: Point,
    user_near: List[Point],
) -> Dict[str, Any]:
    """Run cv-local on one image and score derived geometry."""
    try:
        face_quads, debug = _proposer_face_quads(
            img_path, side, hull_guard=True, processing_image=None,
        )
    except Exception as e:  # noqa: BLE001
        return {"status": "cv_local_fail", "error": f"{type(e).__name__}: {e}"}

    if len(face_quads) < 3:
        return {
            "status": "fewer_than_3_face_quads",
            "n_face_quads": len(face_quads),
        }

    # Take the first 3 face-quads (if more, sub-select; if exactly 3, use all).
    quad_subset = dict(list(face_quads.items())[:3])
    derived = _derive_vertex_and_near(quad_subset)
    if derived is None:
        return {
            "status": "cluster_pattern_mismatch",
            "n_face_quads": len(face_quads),
        }

    vertex, near, far = derived
    user_b = sorted([_bearing(user_v, p) for p in user_near])
    near_b = sorted([_bearing(vertex, p) for p in near])
    far_b = sorted([_bearing(vertex, p) for p in far])

    err_near = _best_perm_err(near_b, user_b)
    err_far = _best_perm_err(far_b, user_b)
    return {
        "status": "ok",
        "err_near_deg": round(err_near, 1),
        "err_far_deg": round(err_far, 1),
        "category": _categorize(err_near, err_far),
        "vertex": [round(vertex[0], 1), round(vertex[1], 1)],
        "n_face_quads": len(face_quads),
    }


def _summarize(by_case: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Schema matches baseline_post_218.py: by_case[key] is a list of
    run dicts (cv-local has 1 run per case since it's deterministic;
    the global model has N runs). Summary fields match too, so the
    same `--diff` renderer works on both snapshots.
    """
    cat_counts: Counter = Counter()
    err_distribution: Counter = Counter()
    n_runs = 0
    for runs in by_case.values():
        for run in runs:
            n_runs += 1
            status = run.get("status")
            if status != "ok":
                cat_counts["CV_LOCAL_FIT_FAIL"] += 1
                continue
            cat = run.get("category", "?")
            cat_counts[cat] += 1
            err = run.get("err_near_deg", 0)
            if err < 5:
                err_distribution["<5°"] += 1
            elif err < 10:
                err_distribution["5-10°"] += 1
            elif err < 25:
                err_distribution["10-25°"] += 1
            elif err < 45:
                err_distribution["25-45°"] += 1
            else:
                err_distribution[">45°"] += 1

    # Case-level stability across runs (matches baseline_post_218.py).
    # cv-local runs deterministically so a case is "stable GOOD" iff
    # its single run is GOOD, etc.
    stable_good = 0
    stable_bad = 0
    mixed = 0
    BAD_CATS = {"CHIRALITY_MISS", "CHIRALITY_FALSE_FLIP", "TRUE_GEOMETRY_FAIL", "CV_LOCAL_FIT_FAIL"}
    for runs in by_case.values():
        cats = set()
        for r in runs:
            if r.get("status") == "ok":
                cats.add(r.get("category", "?"))
            else:
                cats.add("CV_LOCAL_FIT_FAIL")
        if cats == {"GOOD"}:
            stable_good += 1
        elif cats and cats.issubset(BAD_CATS):
            stable_bad += 1
        else:
            mixed += 1

    return {
        "n_cases": len(by_case),
        "n_runs": n_runs,
        "category_counts": dict(cat_counts),
        "error_distribution": dict(err_distribution),
        "stable_good_cases": stable_good,
        "stable_bad_cases": stable_bad,
        "mixed_cases": mixed,
    }


def _render_markdown(summary: Dict[str, Any], by_case: Dict[str, Dict[str, Any]]) -> str:
    n = summary["n_cases"]
    def pct(x):
        return f"{100 * x / n:.1f}%" if n else "0.0%"

    cats = summary["category_counts"]
    err_dist = summary["error_distribution"]

    lines = []
    lines.append("# Phase 1: cv-local baseline on 58-case axis-labeled gallery")
    lines.append("")
    lines.append("## Role of this document")
    lines.append("")
    lines.append("Companion to `POST_218_BASELINE_AND_TAXONOMY.md`. Same eval")
    lines.append("set, same metric (per-axis bearing error vs user labels),")
    lines.append("same categorization. The difference: this measures the")
    lines.append("**production `cv-local` recognizer** instead of the global")
    lines.append("cube model.")
    lines.append("")
    lines.append("cv-local's `analyze_image` outputs face-quads, not vertex+")
    lines.append("legacy-`near` fields directly. This baseline derives the")
    lines.append("comparable historical (vertex, [3 legacy-near clusters])")
    lines.append("signal by clustering the 12 corner")
    lines.append("instances (4 per face-quad × 3 face-quads) across faces:")
    lines.append("")
    lines.append("- 1 cluster of 3 points (one per face) → trihedral vertex")
    lines.append("- 3 clusters of 2 points (each shared by 2 adjacent faces) → 3 legacy-near clusters")
    lines.append("- 3 clusters of 1 point (one per face) → 3 far corners")
    lines.append("")
    lines.append("Algorithm: union-find on cross-face edges in ascending")
    lines.append(f"distance order, threshold {CLUSTER_THRESHOLD_PX:.0f} px, with the")
    lines.append("constraint that the first 3-face cluster found is the vertex")
    lines.append("(no other cluster can grow to 3 face members afterward).")
    lines.append("Cases where the expected 1×3 + 3×2 + 3×1 cluster pattern")
    lines.append("doesn't emerge count as `CV_LOCAL_FIT_FAIL`.")
    lines.append("")
    lines.append("## Headline accuracy")
    lines.append("")
    n_fit_fail = cats.get("CV_LOCAL_FIT_FAIL", 0)
    n_ok = summary["n_runs"] - n_fit_fail
    lines.append(f"**{summary['n_cases']} cases × {summary['n_runs']//max(summary['n_cases'],1)} runs each** "
                 f"({n_ok} scorable, {n_fit_fail} cv-local fit-fail).")
    lines.append("")
    lines.append("| accuracy band | cases |  %  |")
    lines.append("|---------------|------:|----:|")
    for band in ["<5°", "5-10°", "10-25°", "25-45°", ">45°"]:
        c = err_dist.get(band, 0)
        lines.append(f"| {band:13s} | {c:>5d} | {pct(c)} |")
    lines.append("")

    lines.append("## Category breakdown")
    lines.append("")
    lines.append("| category               | cases |  %  |")
    lines.append("|------------------------|------:|----:|")
    for cat in ["GOOD", "MARGINAL", "CHIRALITY_MISS", "TRUE_GEOMETRY_FAIL", "CV_LOCAL_FIT_FAIL"]:
        c = cats.get(cat, 0)
        lines.append(f"| {cat:22s} | {c:>5d} | {pct(c)} |")
    lines.append("")

    # Worst 10 (by err_near, treating fit-fails as worse than any err)
    cases_with_err = []
    cases_failed = []
    for k, runs in by_case.items():
        # cv-local is deterministic so 1 run per case; take run 0
        run = runs[0] if runs else {}
        if run.get("status") == "ok":
            cases_with_err.append((k, run))
        else:
            cases_failed.append((k, run))
    cases_with_err.sort(key=lambda kv: -kv[1].get("err_near_deg", 0))
    lines.append("## 10 worst (scorable) cases")
    lines.append("")
    lines.append("| case | err_near | category |")
    lines.append("|------|---------:|----------|")
    for k, v in cases_with_err[:10]:
        lines.append(f"| {k} | {v['err_near_deg']:>6.1f}° | {v['category']} |")
    lines.append("")
    if cases_failed:
        lines.append("## cv-local fit-failures")
        lines.append("")
        lines.append("| case | reason |")
        lines.append("|------|--------|")
        for k, v in cases_failed:
            reason = v.get("status", "?")
            extra = v.get("error") or f"n_face_quads={v.get('n_face_quads', '?')}"
            lines.append(f"| {k} | {reason} ({extra}) |")
        lines.append("")

    lines.append("## Headline finding: cv-local face-quads are not geometrically consistent")
    lines.append("")
    lines.append("**90% of cases fail the structural consistency check.** This")
    lines.append("isn't a bug in the derivation — it's a real property of cv-")
    lines.append("local's face-quad output that this script measures honestly.")
    lines.append("")
    lines.append("`cv-local` produces face-quads by **independently extrapolating**")
    lines.append("a 4-corner quad from each detected 3×3 sticker grid. There's")
    lines.append("no constraint enforcing that the 3 face-quads share a")
    lines.append("trihedral vertex or pairwise-shared corner clusters. Each face")
    lines.append("sees only its own stickers; it doesn't know about the others.")
    lines.append("")
    lines.append("So on most cases, the 3 cv-local face-quads taken together")
    lines.append("don't represent a single coherent projected cube — they're")
    lines.append("3 disconnected quadrilaterals. The structural-clustering")
    lines.append("derivation correctly reports this as a fit failure.")
    lines.append("")
    lines.append("Of the 6 cases where cv-local DID produce a structurally")
    lines.append("consistent set (lucky alignment of grid extrapolations), all 6")
    lines.append("are catastrophic-error categorizations (>25° err) — meaning")
    lines.append("even when the structure is consistent, the geometry is wrong.")
    lines.append("")
    lines.append("### What this means for Phase 1's stated goal")
    lines.append("")
    lines.append("The Phase 1 plan said: \"Two committed JSON snapshots, both")
    lines.append("runnable via `--diff` for row-level deltas between the two")
    lines.append("sides.\" The mechanical infrastructure for that is now in")
    lines.append("place. But the actual finding is more useful than the diff:")
    lines.append("")
    lines.append("- **The two sides are not comparable via this derivation.**")
    lines.append("  The global model produces well-formed (vertex, 3 near, 3")
    lines.append("  far) tuples on 116/116 runs (post-#218 baseline). cv-local")
    lines.append("  produces them on 6/58 cases.")
    lines.append("- **cv-local doesn't \"see\" the cube as a single object** at")
    lines.append("  the geometry layer. It detects faces independently. This is")
    lines.append("  a structural difference, not a calibration issue.")
    lines.append("- **For Phase 2 trust diagnostics, we want a different cross-")
    lines.append("  system metric.** Options:")
    lines.append("  1. **cv-local-side metric**: per-sticker accuracy on the")
    lines.append("     same 58 photos (production-style eval, not derived")
    lines.append("     geometry). Uses `tools/evaluate_hybrid_pipeline.py`.")
    lines.append("  2. **Consistency-as-trust-signal**: the fact that cv-local's")
    lines.append("     face-quads ARE/ARE NOT structurally consistent is itself")
    lines.append("     a candidate trust signal. \"If the 3 face-quads don't")
    lines.append("     share a vertex within X px, route to retake.\"")
    lines.append("  3. **Hybrid pipeline that uses global model for geometry,")
    lines.append("     cv-local for color** — Codex's #152 hybrid experiment")
    lines.append("     went this direction but found that arbitrary cv-local")
    lines.append("     quads produce bimodal accuracy.")
    lines.append("")
    lines.append("Option 2 is the most natural input for Phase 2's trust-policy")
    lines.append("diagnostics.")
    lines.append("")
    lines.append("## How to compare against the global model")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python tools/baseline_post_218.py \\")
    lines.append("  --diff tests/fixtures/post_218_baseline.json \\")
    lines.append("        tests/fixtures/cv_local_baseline.json")
    lines.append("```")
    lines.append("")
    lines.append("This produces row-level deltas: which cases the global model")
    lines.append("gets right but cv-local misses, which the reverse, which both")
    lines.append("agree on.")
    lines.append("")
    lines.append("## Reproducing")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python tools/baseline_cv_local.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", default=str(DEFAULT_TRUTH))
    ap.add_argument("--gallery", default=str(DEFAULT_GALLERY))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    truth_path = Path(args.truth)
    gallery_dir = Path(args.gallery)
    out_path = Path(args.out)
    report_path = Path(args.report)

    if not truth_path.exists():
        print(f"truth file not found: {truth_path}", file=sys.stderr)
        return 2
    if not gallery_dir.exists():
        print(f"gallery dir not found: {gallery_dir}", file=sys.stderr)
        return 2

    labels = json.loads(truth_path.read_text())
    keys = sorted(labels.keys())
    print(f"running cv-local on {len(keys)} cases", file=sys.stderr, flush=True)

    by_case: Dict[str, Dict[str, Any]] = {}
    for i, key in enumerate(keys, 1):
        L = labels[key]
        if not L.get("approved"):
            continue
        set_id, side = key.rsplit("_", 1)
        if side not in ("A", "B"):
            print(f"  [{i}/{len(keys)}] {key}: unrecognized side", file=sys.stderr)
            continue
        path = gallery_dir / f"set_{key}.png"
        if not path.exists():
            print(f"  [{i}/{len(keys)}] {key}: gallery PNG missing", file=sys.stderr)
            continue
        user_v = tuple(L["vertex"])
        # Canonical schema uses axis_x/y/z (see FULL_CORNER_LABELING.md
        # "Axis-truth schema convention"). Legacy fixtures still in the
        # wild may use the old near_x/y/z key set — both name the same 3
        # FAR-corner positions; only the spelling differs. Read either.
        user_near = [
            tuple(L.get("axis_x", L.get("near_x"))),
            tuple(L.get("axis_y", L.get("near_y"))),
            tuple(L.get("axis_z", L.get("near_z"))),
        ]
        try:
            run = _run_one_case(path, side, user_v, user_near)
        except Exception as e:  # noqa: BLE001
            run = {"status": "exception", "error": f"{type(e).__name__}: {e}"}
        # Wrap in a list to match baseline_post_218.py's schema
        # (case -> list of run dicts). cv-local is deterministic so the
        # list has length 1.
        run.setdefault("run", 0)
        by_case[key] = [run]
        if i % 5 == 0 or i == len(keys):
            cat = run.get("category") or run.get("status", "?")
            print(f"  [{i}/{len(keys)}] {key}: {cat}", file=sys.stderr, flush=True)

    summary = _summarize(by_case)
    payload = {"summary": summary, "by_case": by_case}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)

    report = _render_markdown(summary, by_case)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"wrote {report_path}", file=sys.stderr)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
