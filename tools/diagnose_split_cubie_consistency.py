"""Split-cubie consistency diagnostic (Phase 1).

Pure-function cubie-validity checker for cube state strings, plus a
companion CLI that processes the JSON output of
``tools/diagnose_hull_label_legal_repair.py`` and emits per-row
cubie-consistency analysis.

## Motivation

A cube has 12 cubies whose stickers are split across the two photos:
photo A shows U+R+F, photo B (after 180° flip) shows D+L+B. So 6 of
the 8 corners and 6 of the 12 edges have stickers in BOTH photos.
For each split cubie, the observed colors must form a valid cubie
face-set (one of the 8 corner triples or 12 edge pairs).

In principle, this is a constraint that the existing legal-repair
layer already enforces at the whole-cube level. The interesting
diagnostic question is: are the residual recognition failures
*caused by* split-cubie inconsistency that could be caught earlier
and fixed more locally?

## What this tool answers

For each `canonical_count_repaired` state in a legal-repair JSON
output, it reports:

1. How many cubies are inconsistent (their observed colorset does
   not match any valid corner/edge face-set).
2. Whether each inconsistent cubie is a split cubie or a
   single-image cubie.
3. The true state-delta between `canonical_count_repaired` and
   `broad_legal_repaired` (this is the *actual* set of stickers
   broad-legal changed to reach a valid cube), distinct from the
   reported `repairChanges` which counts changes from raw observations.

## Scope

Phase 1 only. This tool emits structured findings; it does not
modify any production path or perform repair. If Phase 1 shows
signal (split-cubie inconsistency commonly precedes recognition
failures), Phase 2 would wire a targeted re-classification step
into ``tools/hull_label_color_repair.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.shared_cubie_consistency import (  # noqa: E402
    ALL_CUBIES,
    SPLIT_CORNERS,
    SPLIT_EDGES,
    VALID_CORNER_COLORSETS,
    VALID_EDGE_COLORSETS,
    check_cubie,
    check_state,
    index_to_face_position,
    state_diff_indices,
)


# ---------------------------------------------------------------------------
# Per-row analysis (reads legal-repair JSON)
# ---------------------------------------------------------------------------

def analyze_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Analyze one row from the legal-repair diagnostic output."""
    setid = row["setId"]
    gt = row.get("canonicalGroundTruthState")
    methods = row.get("methods", {})
    cc = methods.get("canonical_count_repaired", {})
    bl = methods.get("broad_legal_repaired", {})
    cc_state = cc.get("state")
    bl_state = bl.get("state")

    cc_check = check_state(cc_state) if cc_state else None
    bl_check = check_state(bl_state) if bl_state else None

    # True delta between canonical_count and broad_legal (the stickers
    # broad_legal actually flipped from the recommended baseline). When
    # broad_legal has no state (e.g. status: no_legal_repair), report
    # the delta as unavailable rather than collapsing to 0, which would
    # misrepresent a failed/unavailable repair as a zero-delta no-op.
    delta_available = bool(cc_state and bl_state)
    true_delta_indices = state_diff_indices(cc_state, bl_state) if delta_available else []
    true_delta_count = len(true_delta_indices) if delta_available else None

    return {
        "setId": setid,
        "groundTruthState": gt,
        "canonicalCount": {
            "state": cc_state,
            "hamming": cc.get("hamming"),
            "validState": cc.get("validState"),
            "cubieConsistency": cc_check,
        },
        "broadLegal": {
            "state": bl_state,
            "hamming": bl.get("hamming"),
            "validState": bl.get("validState"),
            "reportedRepairChanges": bl.get("repairChanges"),
            "reportedRepairCost": bl.get("repairCost"),
            "cubieConsistency": bl_check,
            "trueStateDeltaFromCanonical": {
                "available": delta_available,
                "count": true_delta_count,
                "indices": true_delta_indices,
                "facePositions": [index_to_face_position(i) for i in true_delta_indices],
            },
        },
    }


def analyze_legal_repair_json(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [analyze_row(r) for r in payload.get("rows", [])]
    return {
        "schema": "split_cubie_consistency_diagnostic_v1",
        "splitCornerNames": [c.name for c in SPLIT_CORNERS],
        "splitEdgeNames": [c.name for c in SPLIT_EDGES],
        "rows": rows,
        "summary": _summarize(rows),
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    needs_repair = [r for r in rows if r["canonicalCount"]["hamming"] and r["canonicalCount"]["hamming"] > 0]
    has_split_inconsistency = [r for r in needs_repair if r["canonicalCount"]["cubieConsistency"]["inconsistentSplitCount"] > 0]
    has_inimage_only = [
        r for r in needs_repair
        if r["canonicalCount"]["cubieConsistency"]["inconsistentSplitCount"] == 0
        and r["canonicalCount"]["cubieConsistency"]["inconsistentInImageCount"] > 0
    ]
    has_no_cubie_inconsistency = [r for r in needs_repair if r["canonicalCount"]["cubieConsistency"]["inconsistentCount"] == 0]
    return {
        "rowCount": len(rows),
        "needsRepairCount": len(needs_repair),
        "withSplitCubieInconsistency": [r["setId"] for r in has_split_inconsistency],
        "withInImageCubieInconsistencyOnly": [r["setId"] for r in has_inimage_only],
        "withNoCubieInconsistency": [r["setId"] for r in has_no_cubie_inconsistency],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="Path to a tools/diagnose_hull_label_legal_repair.py JSON output")
    p.add_argument("--out-json", type=Path, default=Path("/tmp/split_cubie_consistency.json"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = json.loads(args.input.read_text())
    result = analyze_legal_repair_json(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out_json}", file=sys.stderr)
    s = result["summary"]
    print(f"rows analyzed: {s['rowCount']}", file=sys.stderr)
    print(f"  needing repair (cc hamming > 0): {s['needsRepairCount']}", file=sys.stderr)
    print(f"  with split-cubie inconsistency: {len(s['withSplitCubieInconsistency'])}", file=sys.stderr)
    print(f"  with in-image-only cubie inconsistency: {len(s['withInImageCubieInconsistencyOnly'])}", file=sys.stderr)
    print(f"  with NO cubie inconsistency (parity/twist type): {len(s['withNoCubieInconsistency'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
