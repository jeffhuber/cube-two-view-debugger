"""Direct slot/yaw assembly helpers for the hull-label path."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from tools.corner_conventions import FACE_DEFS_BY_SIDE, wca_face_by_slot, wca_facelets_for_label
from tools.sample_stickers_from_hull import apply_orientation, canonical_corner_order


SLOT_TO_LEGACY_FACE = {
    "upper": "face_xz",
    "right": "face_yz",
    "front": "face_xy",
}

SLOT_ORDER = ("upper", "right", "front")


def convention_orientation_for_slot(
    *,
    side: str,
    slot: str,
    yaw_quarter_turns: int,
    wca_face: str,
    quad: Sequence[Sequence[float]],
) -> Optional[Tuple[bool, int]]:
    """Return the raw-grid orientation that maps a slot to WCA row-major order."""
    labels_by_point = list(zip(FACE_DEFS_BY_SIDE[side][slot], quad))
    canonical_points = canonical_corner_order([(float(x), float(y)) for x, y in quad])
    raw_corner_indices = (0, 2, 8, 6)
    raw_to_wca: Dict[int, int] = {}
    for raw_index, point in zip(raw_corner_indices, canonical_points):
        label = _nearest_label(point, labels_by_point)
        facelet = next(
            (
                item
                for item in wca_facelets_for_label(side, label, yaw_quarter_turns)
                if item.startswith(wca_face)
            ),
            None,
        )
        if facelet is None:
            return None
        raw_to_wca[raw_index] = int(facelet[1:]) - 1
    return _orientation_from_corner_map(raw_to_wca)


def oriented_slot_matrix(
    *,
    raw_matrix: Sequence[Sequence[Any]],
    side: str,
    slot: str,
    yaw_quarter_turns: int,
    quad: Sequence[Sequence[float]],
) -> Optional[Tuple[str, list[list[Any]], Dict[str, Any]]]:
    """Map a raw hull-label slot matrix to its WCA face and orientation."""
    assignments = wca_face_by_slot(side, yaw_quarter_turns)
    wca_face = assignments[slot]
    orientation = convention_orientation_for_slot(
        side=side,
        slot=slot,
        yaw_quarter_turns=yaw_quarter_turns,
        wca_face=wca_face,
        quad=quad,
    )
    if orientation is None:
        return None
    raw_flat = [item for row in raw_matrix for item in row]
    if len(raw_flat) != 9:
        return None
    oriented = apply_orientation(raw_flat, *orientation)
    matrix = [oriented[row * 3:(row + 1) * 3] for row in range(3)]
    return wca_face, matrix, {
        "slot": slot,
        "wcaFace": wca_face,
        "mirror": orientation[0],
        "rotQuarter": orientation[1],
    }


def slot_face_assignments(yaw_quarter_turns: int) -> Dict[str, Dict[str, str]]:
    """Return WCA face assignments for A/B hull-label slots."""
    return {
        "A": wca_face_by_slot("A", yaw_quarter_turns),
        "B": wca_face_by_slot("B", yaw_quarter_turns),
    }


def _orientation_from_corner_map(raw_corner_to_wca_index: Mapping[int, int]) -> Optional[Tuple[bool, int]]:
    raw_indices = list(range(9))
    for mirror in (False, True):
        for rot in range(4):
            oriented = apply_orientation(raw_indices, mirror, rot)
            if all(oriented[wca_index] == raw_index for raw_index, wca_index in raw_corner_to_wca_index.items()):
                return mirror, rot
    return None


def _nearest_label(
    point: Tuple[float, float],
    labeled_points: Sequence[Tuple[str, Sequence[float]]],
) -> str:
    label, _ = min(
        labeled_points,
        key=lambda item: (point[0] - float(item[1][0])) ** 2 + (point[1] - float(item[1][1])) ** 2,
    )
    return label
