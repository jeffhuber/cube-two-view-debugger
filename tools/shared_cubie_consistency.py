"""Shared cubie-consistency checks for two-view cube-state repair.

Photo A contributes the U/R/F faces and photo B contributes D/L/B after
the 180-degree flip. Some physical cubies therefore have stickers split
across the two images. This module centralizes the canonical cubie inventory
and the pure consistency checks used by diagnostics and guarded repair code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, FrozenSet, List, Sequence, Tuple

from rubik_recognizer.validation import (
    CORNER_COLORS,
    CORNER_FACELETS,
    EDGE_COLORS,
    EDGE_FACELETS,
    FACE_ORDER,
)


def image_for_facelet_index(index: int) -> str:
    """Return the source photo for a canonical WCA facelet index."""
    return "A" if index < 27 else "B"


def is_split_facelet_group(facelets: Sequence[int]) -> bool:
    """Whether a cubie has stickers visible in both photos."""
    return len({image_for_facelet_index(i) for i in facelets}) > 1


@dataclass(frozen=True)
class Cubie:
    name: str  # canonical face tuple as a string, e.g. "URF" or "UB"
    kind: str  # "corner" or "edge"
    facelets: Tuple[int, ...]
    expected_colorset: FrozenSet[str]
    split: bool

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["facelets"] = list(self.facelets)
        out["expected_colorset"] = sorted(self.expected_colorset)
        return out


def _build_cubies() -> List[Cubie]:
    out: List[Cubie] = []
    for colors, facelets in zip(CORNER_COLORS, CORNER_FACELETS):
        out.append(
            Cubie(
                name="".join(colors),
                kind="corner",
                facelets=tuple(facelets),
                expected_colorset=frozenset(colors),
                split=is_split_facelet_group(facelets),
            )
        )
    for colors, facelets in zip(EDGE_COLORS, EDGE_FACELETS):
        out.append(
            Cubie(
                name="".join(colors),
                kind="edge",
                facelets=tuple(facelets),
                expected_colorset=frozenset(colors),
                split=is_split_facelet_group(facelets),
            )
        )
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


def check_cubie(state: str, cubie: Cubie) -> Dict[str, Any]:
    """Check whether one cubie's observed colors form a valid WCA cubie set."""
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


def check_state_cubies(state: str) -> Dict[str, Any]:
    """Return per-cubie consistency plus aggregate split/in-image counts."""
    if len(state) != 54:
        raise ValueError(f"expected 54-char state, got {len(state)}")
    reports = [check_cubie(state, cubie) for cubie in ALL_CUBIES]
    inconsistent = [row for row in reports if not row["valid"]]
    return {
        "cubies": reports,
        "totalCubies": len(reports),
        "consistentCount": len(reports) - len(inconsistent),
        "inconsistentCount": len(inconsistent),
        "inconsistentCornerCount": sum(1 for row in inconsistent if row["kind"] == "corner"),
        "inconsistentEdgeCount": sum(1 for row in inconsistent if row["kind"] == "edge"),
        "inconsistentSplitCount": sum(1 for row in inconsistent if row["split"]),
        "inconsistentInImageCount": sum(1 for row in inconsistent if not row["split"]),
        "inconsistentNames": [row["name"] for row in inconsistent],
    }


def state_diff_indices(state_a: str, state_b: str) -> List[int]:
    if len(state_a) != len(state_b):
        return []
    return [index for index, (left, right) in enumerate(zip(state_a, state_b)) if left != right]


def index_to_face_position(index: int) -> str:
    face_idx, within = divmod(int(index), 9)
    row, col = divmod(within, 3)
    return f"{FACE_ORDER[face_idx]}[{row},{col}]"


# Backward-compatible names used by the Phase 1 diagnostic and tests.
check_state = check_state_cubies

