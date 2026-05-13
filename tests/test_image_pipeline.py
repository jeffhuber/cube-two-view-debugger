import math

import numpy as np

from rubik_recognizer.image_pipeline import (
    Sticker,
    _binary_dilate_square,
    _candidate_component_overlap,
    _candidate_key,
    _candidate_matched_set,
    _filter_tiny_white_components,
    _nearest_available_point,
    _score_grid_centers,
)
from rubik_recognizer.colors import ColorMatch


def naive_square_dilate(mask, size):
    radius = size // 2
    result = np.zeros(mask.shape, dtype=bool)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            y0 = max(0, y - radius)
            y1 = min(mask.shape[0], y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(mask.shape[1], x + radius + 1)
            result[y, x] = bool(mask[y0:y1, x0:x1].any())
    return result


def test_binary_dilate_square_matches_naive_window():
    mask = np.zeros((7, 8), dtype=bool)
    mask[1, 1] = True
    mask[3, 5] = True
    mask[6, 7] = True

    np.testing.assert_array_equal(_binary_dilate_square(mask, 3), naive_square_dilate(mask, 3))
    np.testing.assert_array_equal(_binary_dilate_square(mask, 5), naive_square_dilate(mask, 5))


def test_nearest_available_point_uses_tolerance_boundary():
    points = np.array([(0.0, 0.0), (3.0, 4.0), (10.0, 0.0)], dtype=float)

    assert _nearest_available_point(points, {1, 2}, 0.0, 0.0, 5.0) == (None, None)
    assert _nearest_available_point(points, {1, 2}, 0.0, 0.0, 5.1) == (1, 5.0)


def test_score_grid_centers_matches_nearest_points_greedily():
    points = np.array(
        [
            (0.0, 0.0),
            (10.0, 0.0),
            (20.0, 0.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (20.0, 10.0),
            (0.0, 20.0),
            (10.0, 20.0),
            (20.0, 20.0),
        ],
        dtype=float,
    )
    centers = [[(c * 10.0 + 0.2, r * 10.0 + 0.1) for c in range(3)] for r in range(3)]

    result = _score_grid_centers(points, list(range(9)), centers, spacing=25.0)

    assert result["matched"] == list(range(9))
    assert result["matched_count"] == 9
    assert result["cell_matches"] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert math.isclose(result["error"], math.hypot(0.2, 0.1), rel_tol=1e-12)


def test_candidate_key_and_overlap_reuse_cached_matched_collections():
    first = {"matched": [3, 1, 2]}
    second = {"matched": [2, 4]}
    third = {"matched": [9]}

    assert _candidate_key(first) == (1, 2, 3)
    assert _candidate_key(first) is first["_candidate_key"]
    assert _candidate_matched_set(first) == {1, 2, 3}
    assert _candidate_matched_set(first) is first["_matched_set"]
    assert _candidate_component_overlap([first, second, third]) == 1
    assert second["_matched_set"] == {2, 4}
    assert third["_matched_set"] == {9}


def test_tiny_white_components_are_removed_when_colored_stickers_anchor_scale():
    def sticker(index, color, area):
        face = {"white": "U", "red": "R", "orange": "L", "yellow": "D", "green": "F", "blue": "B"}[color]
        return Sticker(
            id=index,
            center=(float(index * 10), 0.0),
            bbox=(index * 10, 0, index * 10 + 8, 8),
            rgb=(255, 255, 255),
            match=ColorMatch(color, face, 0.8, 0.0, [(color, 0.0)]),
            area=area,
        )

    colored = [
        sticker(index, color, 5000)
        for index, color in enumerate(["red", "orange", "yellow", "green", "blue", "red", "orange", "yellow"])
    ]
    real_white = [sticker(100, "white", 4200)]
    tiny_white_noise = [sticker(200 + index, "white", 80) for index in range(18)]

    filtered = _filter_tiny_white_components([*colored, *real_white, *tiny_white_noise])

    assert real_white[0] in filtered
    assert not any(sticker in filtered for sticker in tiny_white_noise)
    assert len(filtered) == len(colored) + 1


def test_tiny_white_component_filter_requires_many_white_candidates():
    white = [
        Sticker(
            id=index,
            center=(float(index), 0.0),
            bbox=(index, 0, index + 1, 1),
            rgb=(255, 255, 255),
            match=ColorMatch("white", "U", 0.8, 0.0, [("white", 0.0)]),
            area=80,
        )
        for index in range(4)
    ]

    assert _filter_tiny_white_components(white) is white
