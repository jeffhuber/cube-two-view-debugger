import math

import numpy as np

from rubik_recognizer.image_pipeline import _binary_dilate_square, _nearest_available_point, _score_grid_centers


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
