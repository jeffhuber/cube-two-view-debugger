"""Shared human corner and facelet conventions for two-view cube geometry.

This module is intentionally small and dependency-free. It names the
owner-approved convention for the visible trihedral vertices and the six
outer corners in image A and image B so labelers, reports, and tests do
not each reinvent the mapping.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Tuple


POINT_NAMES = (
    "vertex",
    "corner_0",
    "corner_1",
    "corner_2",
    "corner_3",
    "corner_4",
    "corner_5",
)

VERTEX_NAME_BY_SIDE = {
    "A": "Va",
    "B": "Vb",
}

YAW_SIDE_ORDER = ("F", "R", "B", "L")

# Face outlines are listed in view-slot order:
# the visible trihedral vertex followed by the three outer corners around
# that slot. Canonical WCA side faces depend on capture yaw, so the keys
# below deliberately say upper/right/front instead of U/R/F/D/B/L.
FACE_DEFS_BY_SIDE: Dict[str, Dict[str, Tuple[str, str, str, str]]] = {
    "A": {
        "upper": ("vertex", "corner_1", "corner_0", "corner_5"),
        "right": ("vertex", "corner_3", "corner_2", "corner_1"),
        "front": ("vertex", "corner_5", "corner_4", "corner_3"),
    },
    "B": {
        "upper": ("vertex", "corner_2", "corner_3", "corner_4"),
        "right": ("vertex", "corner_0", "corner_1", "corner_2"),
        "front": ("vertex", "corner_4", "corner_5", "corner_0"),
    },
}

CAPTURE_FACE_BY_SLOT_BY_SIDE = {
    "A": {
        "upper": "U",
        "right": "R",
        "front": "F",
    },
    "B": {
        "upper": "D",
        "right": "B",
        "front": "L",
    },
}

ONE_EDGE_CORNERS_BY_SIDE = {
    "A": ("corner_1", "corner_3", "corner_5"),
    "B": ("corner_0", "corner_2", "corner_4"),
}

FAR_CORNERS_BY_SIDE = {
    "A": ("corner_0", "corner_2", "corner_4"),
    "B": ("corner_1", "corner_3", "corner_5"),
}

# Each corner position's flattened-net facelets for yaw=0. Facelets use
# URFDLB order and row-major indices 1..9 within each face. Non-zero yaw
# remaps the side faces; do not use this table without checking yaw.
YAW0_CORNER_FACELETS = {
    "Va": ("U9", "R1", "F3"),
    "Vb": ("D7", "L7", "B9"),
    "corner_0": ("U1", "L1", "B3"),
    "corner_1": ("U3", "R3", "B1"),
    "corner_2": ("D9", "R9", "B7"),
    "corner_3": ("D3", "R7", "F9"),
    "corner_4": ("D1", "L9", "F7"),
    "corner_5": ("U7", "F1", "L3"),
}

_WCA_CORNER_FACELETS_BY_FACE_SET: Dict[FrozenSet[str], Tuple[str, str, str]] = {
    frozenset(facelet[0] for facelet in facelets): facelets
    for facelets in YAW0_CORNER_FACELETS.values()
}


def capture_to_wca_yaw_map(yaw_quarter_turns: int) -> Dict[str, str]:
    """Map capture-frame face labels to canonical WCA faces for a yaw."""
    yaw = yaw_quarter_turns % 4
    mapping = {"U": "U", "D": "D"}
    for index, capture_face in enumerate(YAW_SIDE_ORDER):
        mapping[capture_face] = YAW_SIDE_ORDER[(index + yaw) % len(YAW_SIDE_ORDER)]
    return mapping


def wca_face_by_slot(side: str, yaw_quarter_turns: int) -> Dict[str, str]:
    """Return canonical WCA face names for each visible view slot."""
    capture_to_wca = capture_to_wca_yaw_map(yaw_quarter_turns)
    return {
        slot: capture_to_wca[capture_face]
        for slot, capture_face in CAPTURE_FACE_BY_SLOT_BY_SIDE[side].items()
    }


def wca_facelets_for_point(point_name: str, yaw_quarter_turns: int) -> Tuple[str, str, str]:
    """Return canonical WCA facelets for a labeled physical corner.

    `YAW0_CORNER_FACELETS` names the capture-frame corner identities. Under
    non-zero yaw, the visible point keeps its human label, but its physical
    WCA corner changes. We first map the capture-frame faces through yaw, then
    look up the canonical WCA facelet indices for that resulting corner.
    """
    if point_name not in YAW0_CORNER_FACELETS:
        raise KeyError(f"unknown full-corner point: {point_name}")
    capture_to_wca = capture_to_wca_yaw_map(yaw_quarter_turns)
    wca_faces = frozenset(
        capture_to_wca[facelet[0]]
        for facelet in YAW0_CORNER_FACELETS[point_name]
    )
    return _WCA_CORNER_FACELETS_BY_FACE_SET[wca_faces]


def wca_facelets_by_point(yaw_quarter_turns: int) -> Dict[str, Tuple[str, str, str]]:
    """Return WCA facelets for `Va/Vb + corner_0..5` under capture yaw."""
    return {
        point_name: wca_facelets_for_point(point_name, yaw_quarter_turns)
        for point_name in (
            "Va", "Vb", "corner_0", "corner_1", "corner_2",
            "corner_3", "corner_4", "corner_5",
        )
    }


def wca_facelets_for_label(side: str, label: str, yaw_quarter_turns: int) -> Tuple[str, str, str]:
    """Return WCA facelets for a fixture label on side A/B.

    Full-corner fixture rows use the generic key `vertex`; humans call that
    point `Va` on side A and `Vb` on side B. Numbered corners are shared.
    """
    point_name = VERTEX_NAME_BY_SIDE[side] if label == "vertex" else label
    return wca_facelets_for_point(point_name, yaw_quarter_turns)
