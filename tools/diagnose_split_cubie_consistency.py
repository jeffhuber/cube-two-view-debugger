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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.validation import (  # noqa: E402
    CORNER_COLORS,
    CORNER_FACELETS,
    EDGE_COLORS,
    EDGE_FACELETS,
    FACE_ORDER,
)


# ---------------------------------------------------------------------------
# Cubie inventory + image-source partition
# ---------------------------------------------------------------------------

def _image_for(index: int) -> str:
    """Image A holds the U+R+F faces (indices 0..26); image B holds D+L+B (27..53)."""
    return "A" if index < 27 else "B"


def _is_split(facelets: Sequence[int]) -> bool:
    return len({_image_for(i) for i in facelets}) > 1


@dataclass(frozen=True)
class Cubie:
    name: str  # canonical face-tuple as string, e.g. "URF" or "UB"
    kind: str  # "corner" or "edge"
    facelets: Tuple[int, ...]
    expected_colorset: FrozenSet[str]
    split: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["facelets"] = list(self.facelets)
        d["expected_colorset"] = sorted(self.expected_colorset)
        return d


def _build_cubies() -> List[Cubie]:
    out: List[Cubie] = []
    for colors, facelets in zip(CORNER_COLORS, CORNER_FACELETS):
        out.append(Cubie(
            name="".join(colors),
            kind="corner",
            facelets=tuple(facelets),
            expected_colorset=frozenset(colors),
            split=_is_split(facelets),
        ))
    for colors, facelets in zip(EDGE_COLORS, EDGE_FACELETS):
        out.append(Cubie(
            name="".join(colors),
            kind="edge",
            facelets=tuple(facelets),
            expected_colorset=frozenset(colors),
            split=_is_split(facelets),
        ))
    return out


ALL_CUBIES: List[Cubie] = _build_cubies()

VALID_CORNER_COLORSETS: FrozenSet[FrozenSet[str]] = frozenset(
    frozenset(triple) for triple in CORNER_COLORS
)
VALID_EDGE_COLORSETS: FrozenSet[FrozenSet[str]] = frozenset(
    frozenset(pair) for pair in EDGE_COLORS
)

SPLIT_CORNERS: List[Cubie] = [c for c in ALL_CUBIES if c.kind == "corner" and c.split]
SPLIT_EDGES: List[Cubie] = [c for c in ALL_CUBIES if c.kind == "edge" and c.split]


# ---------------------------------------------------------------------------
# Pure-function cubie consistency check
# ---------------------------------------------------------------------------

def check_cubie(state: str, cubie: Cubie) -> Dict[str, Any]:
    """Check whether the colors at cubie.facelets in `state` form a valid cubie."""
    if len(state) != 54:
        raise ValueError(f"expected 54-char state, got {len(state)}")
    colors = tuple(state[i] for i in cubie.facelets)
    colorset = frozenset(colors)
    valid_pool = VALID_CORNER_COLORSETS if cubie.kind == "corner" else VALID_EDGE_COLORSETS
    valid = len(colorset) == len(cubie.facelets) and colorset in valid_pool
    return {
        "name": cubie.name,
        "kind": cubie.kind,
        "split": cubie.split,
        "facelets": list(cubie.facelets),
        "observed_colors": list(colors),
        "valid": valid,
    }


def check_state(state: str) -> Dict[str, Any]:
    """Full cubie consistency check for a state string.

    Returns per-cubie reports plus aggregate counts split by
    {corner|edge} x {split|in-image}.
    """
    if len(state) != 54:
        raise ValueError(f"expected 54-char state, got {len(state)}")
    reports = [check_cubie(state, c) for c in ALL_CUBIES]
    inconsistent = [r for r in reports if not r["valid"]]
    return {
        "cubies": reports,
        "totalCubies": len(reports),
        "consistentCount": len(reports) - len(inconsistent),
        "inconsistentCount": len(inconsistent),
        "inconsistentCornerCount": sum(1 for r in inconsistent if r["kind"] == "corner"),
        "inconsistentEdgeCount": sum(1 for r in inconsistent if r["kind"] == "edge"),
        "inconsistentSplitCount": sum(1 for r in inconsistent if r["split"]),
        "inconsistentInImageCount": sum(1 for r in inconsistent if not r["split"]),
        "inconsistentNames": [r["name"] for r in inconsistent],
    }


def state_diff_indices(state_a: str, state_b: str) -> List[int]:
    if len(state_a) != len(state_b):
        return []
    return [i for i in range(len(state_a)) if state_a[i] != state_b[i]]


def index_to_face_position(index: int) -> str:
    face_idx, within = divmod(index, 9)
    row, col = divmod(within, 3)
    return f"{FACE_ORDER[face_idx]}[{row},{col}]"


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
    # broad_legal actually flipped from the recommended baseline).
    true_delta = state_diff_indices(cc_state, bl_state) if (cc_state and bl_state) else []

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
                "count": len(true_delta),
                "indices": true_delta,
                "facePositions": [index_to_face_position(i) for i in true_delta],
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
