#!/usr/bin/env python3
"""Cross-reference auto-geometry proposers against recognizer failure
cases. Answers Codex's question (PR #128 follow-up, parallel-track ask):
"Which automatic hull/proposer would have rejected the recognizer's bad
grid?"

Inputs:
  * runs/auto_geometry_report.json (produced by evaluate_auto_geometry.py)
  * /tmp/ctvd-hard-current.json (Codex's hard-case probe artifact)
  * /tmp/ctvd-corpus-current.json (Codex's corpus contract artifact;
    optional — flags orientation_rank_failure cases like Sets 27/28)
  * /tmp/ctvd-label-baseline-current.json (Codex's label-baseline; optional
    — surfaces selected-grid-cells-outside-hull cases like 46A/47B/48A/49)

For each failing case × each proposer:
  * Total grids the recognizer's pipeline picked
  * How many of those grids the proposer's geometry would *accept*
    (≥7/9 sticker centers inside any proposed face quad)
  * Per-face containment fraction
  * Net signal: would adding this proposer as a pre-acceptance filter have
    caught the failure?

Output: a markdown table to stdout, JSON detail to runs/grid_rejection_analysis.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REPORT = REPO_ROOT / "runs" / "auto_geometry_report.json"
DEFAULT_HARD = Path("/tmp/ctvd-hard-current.json")
DEFAULT_CORPUS = Path("/tmp/ctvd-corpus-current.json")
DEFAULT_LABEL_BASELINE = Path("/tmp/ctvd-label-baseline-current.json")
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "grid_rejection_analysis.json"


def load_hard_failures(path: Path) -> List[Dict]:
    """Return one row per hard-case failure (status != success_clean)."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    for r in data.get("results", []):
        set_id = str(r.get("setId") or r.get("manifest", {}).get("setId", ""))
        status = r.get("currentStatus") or r.get("status") or ""
        cat = r.get("currentCategory") or r.get("category") or ""
        if status in ("success", "success_clean") and cat == "success_clean":
            continue
        rows.append({
            "setId": set_id,
            "status": status,
            "category": cat,
            "failureClass": r.get("failureClass", ""),
            "failedChecks": r.get("currentFailedChecks") or r.get("failedChecks", []),
        })
    return rows


def load_corpus_orientation_failures(path: Path) -> List[Dict]:
    """Return corpus rows with orientation-rank or contract failures."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    for r in data.get("results", []):
        set_id = str(r.get("setId", ""))
        cat = r.get("category", "")
        contract_failures = r.get("contractFailures") or []
        failure_modes = r.get("failureModes") or []
        if cat == "success_clean" and not contract_failures and not failure_modes:
            continue
        rows.append({
            "setId": set_id,
            "category": cat,
            "expectedCategory": r.get("expectedCategory", ""),
            "failureModes": failure_modes,
            "contractFailures": contract_failures,
            "score": r.get("currentScoreObserved"),
            "expectedScoreFloor": r.get("expectedScoreFloor"),
        })
    return rows


def load_label_baseline_outside_hull(path: Path) -> Dict[Tuple[str, str], Dict]:
    """Per (setId, side): how many recognizer-selected grid cells fell
    outside the human-labeled cube hull. High count flags the "bad grid
    that escapes the proposer's geometry" cases Codex called out (46A,
    47B, 48A, 49, 30 A/B, 39 B)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: Dict[Tuple[str, str], Dict] = {}
    for r in data.get("results", []):
        set_id = str(r.get("setId", ""))
        side = r.get("imageSide", "")
        # The schema has selectedGridCells with per-cell insideLabeledCubeHull flag
        cells = r.get("selectedGridCells") or []
        total = len(cells)
        outside = sum(1 for c in cells if not c.get("insideLabeledCubeHull", True))
        out[(set_id, side)] = {
            "selectedGridCellsTotal": total,
            "selectedGridCellsOutsideHumanHull": outside,
            "outsideFraction": (outside / total) if total else 0.0,
        }
    return out


def load_auto_geometry_report(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Auto-geometry report not found at {path}. Run "
            "`tools/evaluate_auto_geometry.py` first."
        )
    return json.loads(path.read_text())


def proposer_would_reject_grids(impact: Dict, threshold: float = 7 / 9) -> Tuple[int, int, Dict[str, float]]:
    """Returns (rejected_count, considered_count, per_face_containment).
    A grid is 'rejected' when its containment fraction falls below threshold."""
    considered = impact.get("recognizerGridsConsidered", 0)
    accepted = impact.get("recognizerGridsAccepted", 0)
    containment = impact.get("recognizerBestGridContainment", {})
    rejected = considered - accepted
    return rejected, considered, containment


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--hard", default=str(DEFAULT_HARD))
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--label-baseline", default=str(DEFAULT_LABEL_BASELINE))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--only-sets", nargs="*", default=None,
                    help="Restrict to specific set IDs (e.g. 46 47 48 49 30 39)")
    args = ap.parse_args()

    hard_failures = load_hard_failures(Path(args.hard))
    corpus_failures = load_corpus_orientation_failures(Path(args.corpus))
    label_baseline = load_label_baseline_outside_hull(Path(args.label_baseline))
    report = load_auto_geometry_report(Path(args.report))

    print(f"loaded {len(hard_failures)} hard failures, "
          f"{len(corpus_failures)} corpus issues, "
          f"{len(label_baseline)} label-baseline rows, "
          f"{len(report)} (target, proposer) report rows", file=sys.stderr)

    # Build the failing-set list: union of hard-case failures, corpus
    # failures, and label-baseline rows with >0 selected cells outside
    # the labeled hull. Codex's explicit callouts (46A, 47B, 48A, 49,
    # 30 A/B, 39 B) all come from the label-baseline source.
    failing_sets: Set[str] = set()
    for r in hard_failures:
        failing_sets.add(r["setId"])
    for r in corpus_failures:
        if r["category"] != r["expectedCategory"] or r["contractFailures"]:
            failing_sets.add(r["setId"])
    for (set_id, _side), m in label_baseline.items():
        if m["selectedGridCellsOutsideHumanHull"] > 0:
            failing_sets.add(set_id)

    if args.only_sets:
        failing_sets = {s for s in failing_sets if s in args.only_sets}
    print(f"analyzing {len(failing_sets)} failing sets", file=sys.stderr)

    # Index report by (setId, side)
    by_target: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in report:
        if "error" in row:
            continue
        by_target[(row["setId"], row["side"])].append(row)

    # Build the analysis: per (failing setId, side, proposer)
    analysis_rows: List[Dict] = []
    for set_id in sorted(failing_sets, key=lambda s: int(s) if s.isdigit() else 999):
        for side in ("A", "B"):
            rows = by_target.get((set_id, side), [])
            if not rows:
                continue
            label_row = label_baseline.get((set_id, side))
            for row in rows:
                impact = row.get("recognizerImpact") or {}
                rejected, considered, per_face = proposer_would_reject_grids(impact)
                outside_label_hull = (
                    label_row["selectedGridCellsOutsideHumanHull"] if label_row else None
                )
                analysis_rows.append({
                    "setId": set_id,
                    "side": side,
                    "proposer": row["proposer"],
                    "cubeHullIoU": row["cubeHullIoU"],
                    "meanFaceIoU_bestMatch": row["meanFaceIoU_bestMatch"],
                    "recognizerGridsRejected": rejected,
                    "recognizerGridsConsidered": considered,
                    "recognizerWouldHaveBeenFiltered": rejected > 0,
                    "perFaceContainment": per_face,
                    "outsideAllFacesFraction": impact.get("outsideAllFacesFraction"),
                    "outsideHullFraction": impact.get("outsideHullFraction"),
                    "labelBaselineGridCellsOutsideHumanHull": outside_label_hull,
                })

    # Markdown summary: per failing (set, side), one row per proposer
    print()
    print("# Auto-proposer × recognizer-failure cross-reference")
    print()
    print("For each (setId, side) where the recognizer hit a contract/hard-case failure")
    print("OR had selected-grid-cells outside the human-labeled cube hull, this table shows")
    print("**which proposers would have rejected** the recognizer's chosen grids.")
    print()
    print(f"{'set':>4s} {'side':>4s}  {'label outside hull':>20s}  {'proposer':<32s}  {'gridsRej/Cons':>14s}  {'hullIoU':>7s}  {'faceIoU':>7s}  {'wouldFilter':>11s}")
    print("-" * 130)

    by_pair: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in analysis_rows:
        by_pair[(row["setId"], row["side"])].append(row)

    for (set_id, side), rows in sorted(by_pair.items(), key=lambda x: (int(x[0][0]) if x[0][0].isdigit() else 999, x[0][1])):
        outside_label = rows[0]["labelBaselineGridCellsOutsideHumanHull"]
        outside_str = f"{outside_label}" if outside_label is not None else "—"
        for i, row in enumerate(rows):
            label_part = outside_str if i == 0 else ""
            set_part = set_id if i == 0 else ""
            side_part = side if i == 0 else ""
            print(f"{set_part:>4s} {side_part:>4s}  {label_part:>20s}  "
                  f"{row['proposer']:<32s}  "
                  f"{row['recognizerGridsRejected']:>2d}/{row['recognizerGridsConsidered']:<2d}{'':>9s}  "
                  f"{row['cubeHullIoU']:>7.3f}  {row['meanFaceIoU_bestMatch']:>7.3f}  "
                  f"{'YES' if row['recognizerWouldHaveBeenFiltered'] else 'no':>11s}")
        print()

    # Save JSON for tooling
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "failingSets": sorted(failing_sets, key=lambda s: int(s) if s.isdigit() else 999),
        "rows": analysis_rows,
    }, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
