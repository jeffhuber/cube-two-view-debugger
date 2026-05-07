from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Point = Tuple[float, float]
Matrix3 = List[List[object]]

EDGE_NAMES = ("top", "right", "bottom", "left")


@dataclass(frozen=True)
class Transform:
    name: str
    canonical_to_observed: Tuple[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], ...]
    edge_map: Dict[str, str]

    def apply(self, observed: Sequence[Sequence[object]]) -> Matrix3:
        return [
            [observed[self.canonical_to_observed[r][c][0]][self.canonical_to_observed[r][c][1]] for c in range(3)]
            for r in range(3)
        ]


def build_transforms() -> List[Transform]:
    mappings = {
        "identity": lambda r, c: (r, c),
        "rot90": lambda r, c: (2 - c, r),
        "rot180": lambda r, c: (2 - r, 2 - c),
        "rot270": lambda r, c: (c, 2 - r),
        "flip_h": lambda r, c: (r, 2 - c),
        "flip_v": lambda r, c: (2 - r, c),
        "transpose": lambda r, c: (c, r),
        "anti_transpose": lambda r, c: (2 - c, 2 - r),
    }
    transforms = []
    for name, fn in mappings.items():
        mapping = tuple(tuple(fn(r, c) for c in range(3)) for r in range(3))
        transforms.append(Transform(name, mapping, _edge_map(mapping)))
    return transforms


def closest_edge(grid_a: Sequence[Sequence[Point]], grid_b: Sequence[Sequence[Point]]) -> str:
    b_points = [pt for row in grid_b for pt in row]
    scores: Dict[str, float] = {}
    for edge in EDGE_NAMES:
        a_points = edge_points(grid_a, edge)
        scores[edge] = min(_dist(a, b) for a in a_points for b in b_points)
    return min(scores, key=scores.get)


def edge_points(grid: Sequence[Sequence[Point]], edge: str) -> List[Point]:
    if edge == "top":
        return list(grid[0])
    if edge == "right":
        return [grid[r][2] for r in range(3)]
    if edge == "bottom":
        return list(grid[2])
    if edge == "left":
        return [grid[r][0] for r in range(3)]
    raise ValueError(edge)


def possible_transforms(required: Dict[str, str]) -> List[Transform]:
    """Find transforms where canonical edge -> observed edge constraints hold."""
    matches = []
    for transform in TRANSFORMS:
        if all(transform.edge_map.get(canonical) == observed for canonical, observed in required.items()):
            matches.append(transform)
    return matches


def _edge_map(mapping: Sequence[Sequence[Tuple[int, int]]]) -> Dict[str, str]:
    edges = {
        "top": [mapping[0][c] for c in range(3)],
        "right": [mapping[r][2] for r in range(3)],
        "bottom": [mapping[2][c] for c in range(3)],
        "left": [mapping[r][0] for r in range(3)],
    }
    return {canonical: _observed_edge(points) for canonical, points in edges.items()}


def _observed_edge(points: Sequence[Tuple[int, int]]) -> str:
    rows = {p[0] for p in points}
    cols = {p[1] for p in points}
    if rows == {0}:
        return "top"
    if rows == {2}:
        return "bottom"
    if cols == {0}:
        return "left"
    if cols == {2}:
        return "right"
    raise ValueError(f"Not an edge: {points}")


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


TRANSFORMS = build_transforms()
