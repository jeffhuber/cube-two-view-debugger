from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


FACE_ORDER = "URFDLB"

CORNER_FACELETS = [
    (8, 9, 20),    # URF
    (6, 18, 38),   # UFL
    (0, 36, 47),   # ULB
    (2, 45, 11),   # UBR
    (29, 26, 15),  # DFR
    (27, 44, 24),  # DLF
    (33, 53, 42),  # DBL
    (35, 17, 51),  # DRB
]

CORNER_COLORS = [
    ("U", "R", "F"),
    ("U", "F", "L"),
    ("U", "L", "B"),
    ("U", "B", "R"),
    ("D", "F", "R"),
    ("D", "L", "F"),
    ("D", "B", "L"),
    ("D", "R", "B"),
]

EDGE_FACELETS = [
    (5, 10),   # UR
    (7, 19),   # UF
    (3, 37),   # UL
    (1, 46),   # UB
    (32, 16),  # DR
    (28, 25),  # DF
    (30, 43),  # DL
    (34, 52),  # DB
    (23, 12),  # FR
    (21, 41),  # FL
    (50, 39),  # BL
    (48, 14),  # BR
]

EDGE_COLORS = [
    ("U", "R"),
    ("U", "F"),
    ("U", "L"),
    ("U", "B"),
    ("D", "R"),
    ("D", "F"),
    ("D", "L"),
    ("D", "B"),
    ("F", "R"),
    ("F", "L"),
    ("B", "L"),
    ("B", "R"),
]

CENTER_INDICES = {"U": 4, "R": 13, "F": 22, "D": 31, "L": 40, "B": 49}


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]


def validate_state(state: str) -> ValidationResult:
    errors: List[str] = []
    if len(state) != 54:
        return ValidationResult(False, ["state_length_not_54"])
    invalid = sorted(set(state) - set(FACE_ORDER))
    if invalid:
        errors.append("invalid_face_letters")
    for face in FACE_ORDER:
        if state.count(face) != 9:
            errors.append(f"{face}_count_not_9")
    for face, idx in CENTER_INDICES.items():
        if state[idx] != face:
            errors.append(f"{face}_center_invalid")
    if errors:
        return ValidationResult(False, errors)

    cp, co, corner_errors = _corner_cubies(state)
    ep, eo, edge_errors = _edge_cubies(state)
    errors.extend(corner_errors)
    errors.extend(edge_errors)
    if errors:
        return ValidationResult(False, errors)

    if sum(co) % 3 != 0:
        errors.append("corner_orientation_invalid")
    if sum(eo) % 2 != 0:
        errors.append("edge_orientation_invalid")
    if _parity(cp) != _parity(ep):
        errors.append("permutation_parity_invalid")
    return ValidationResult(not errors, errors)


def _corner_cubies(state: str):
    cp: List[Optional[int]] = [None] * 8
    co: List[int] = [0] * 8
    errors: List[str] = []
    used = set()
    for pos, facelets in enumerate(CORNER_FACELETS):
        colors = [state[idx] for idx in facelets]
        ori = next((idx for idx, color in enumerate(colors) if color in {"U", "D"}), None)
        if ori is None:
            errors.append(f"corner_{pos}_missing_ud_color")
            continue
        color1 = colors[(ori + 1) % 3]
        color2 = colors[(ori + 2) % 3]
        cubie = next((idx for idx, proto in enumerate(CORNER_COLORS) if proto[1] == color1 and proto[2] == color2), None)
        if cubie is None:
            errors.append(f"corner_{pos}_invalid_color_set")
            continue
        if cubie in used:
            errors.append(f"corner_{pos}_duplicate_cubie")
            continue
        used.add(cubie)
        cp[pos] = cubie
        co[pos] = ori % 3
    if len(used) != 8:
        errors.append("corner_cubie_count_invalid")
    return [int(v) for v in cp if v is not None], co, errors


def _edge_cubies(state: str):
    ep: List[Optional[int]] = [None] * 12
    eo: List[int] = [0] * 12
    errors: List[str] = []
    used = set()
    for pos, facelets in enumerate(EDGE_FACELETS):
        colors = (state[facelets[0]], state[facelets[1]])
        cubie = None
        orientation = 0
        for idx, proto in enumerate(EDGE_COLORS):
            if colors == proto:
                cubie = idx
                orientation = 0
                break
            if colors == (proto[1], proto[0]):
                cubie = idx
                orientation = 1
                break
        if cubie is None:
            errors.append(f"edge_{pos}_invalid_color_set")
            continue
        if cubie in used:
            errors.append(f"edge_{pos}_duplicate_cubie")
            continue
        used.add(cubie)
        ep[pos] = cubie
        eo[pos] = orientation
    if len(used) != 12:
        errors.append("edge_cubie_count_invalid")
    return [int(v) for v in ep if v is not None], eo, errors


def _parity(permutation: List[int]) -> int:
    inversions = 0
    for i in range(len(permutation)):
        for j in range(i + 1, len(permutation)):
            if permutation[i] > permutation[j]:
                inversions += 1
    return inversions % 2
