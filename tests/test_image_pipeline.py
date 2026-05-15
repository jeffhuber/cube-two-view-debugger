import math

import numpy as np

from rubik_recognizer.image_pipeline import (
    Sticker,
    _binary_dilate_square,
    _candidate_component_overlap,
    _candidate_key,
    _candidate_matched_set,
    _filter_tiny_white_components,
    _find_cube_roi,
    _find_cube_roi_at_threshold,
    _nearest_available_point,
    _roi_covers_full_frame,
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


def _synthetic_image(height: int, width: int, *, background_sat: float, cube_box=None) -> np.ndarray:
    """Build a synthetic RGB array with a controllable background
    saturation level, optionally with a bright cube-shaped patch.

    Used to test ROI detection without depending on photo fixtures.
    Background = an HSV color with sat=background_sat, val=0.7,
    converted to RGB. Cube patch (if given) = highly saturated
    bright orange-ish pixels that mimic a Rubik's cube's
    saturation footprint.
    """
    # Background: hue=30° (warm), val=0.7, sat=background_sat.
    # Compute the RGB equivalent manually (HSV -> RGB for V=0.7,
    # H=30°): the conversion gives R=V, G=V*(1-S*(1-frac)), B=V*(1-S),
    # rounded to 8-bit. For H=30° in the [0,60) sector, frac=0.5.
    v = 0.7
    s = background_sat
    bg_r = int(round(v * 255))
    bg_g = int(round(v * (1 - s * 0.5) * 255))
    bg_b = int(round(v * (1 - s) * 255))
    arr = np.full((height, width, 3), [bg_r, bg_g, bg_b], dtype=np.uint8)
    if cube_box is not None:
        x0, y0, x1, y1 = cube_box
        # Bright orange (sat ≈ 1.0, val ≈ 1.0): obviously cube-like.
        arr[y0:y1, x0:x1] = [255, 128, 0]
    return arr


def test_find_cube_roi_low_saturation_background_isolates_cube():
    """Baseline: when the background is barely saturated (sat≈0.05,
    similar to Set 15's marble café table at 13.5% saturated pixels),
    ROI detection isolates the bright cube patch cleanly with the
    default threshold. No retry needed."""
    arr = _synthetic_image(1000, 800, background_sat=0.05, cube_box=(200, 300, 600, 700))
    roi = _find_cube_roi(arr)
    x0, y0, x1, y1 = roi
    # The ROI should contain the cube patch with some padding, NOT cover the entire frame.
    assert x0 <= 200 and x1 >= 600, f"ROI should include cube x-range, got {roi}"
    assert y0 <= 300 and y1 >= 700, f"ROI should include cube y-range, got {roi}"
    assert not _roi_covers_full_frame(roi, 800, 1000), (
        f"Low-saturation background must not trigger frame-cover retry, got {roi}"
    )


def test_find_cube_roi_chromatic_background_triggers_retry():
    """Set 46 failure mode (2026-05-14): textured/chromatic backgrounds
    can register as saturated under the default threshold and merge with
    the cube into a single frame-spanning component. The retry at the
    stricter threshold must isolate the cube. This synthetic reproduces
    the failure: a sat=0.30 background (similar to Set 46's 60-70%
    saturated pixels) plus a bright cube patch."""
    arr = _synthetic_image(1000, 800, background_sat=0.30, cube_box=(200, 300, 600, 700))
    # The default threshold alone must fail (frame-spanning).
    default_roi = _find_cube_roi_at_threshold(arr, 0.23)
    assert _roi_covers_full_frame(default_roi, 800, 1000), (
        f"Test premise: chromatic background should produce frame-spanning ROI at default threshold, got {default_roi}"
    )
    # The retry at the stricter threshold must isolate the cube.
    final_roi = _find_cube_roi(arr)
    assert not _roi_covers_full_frame(final_roi, 800, 1000), (
        f"Retry should isolate cube from chromatic background, got {final_roi}"
    )
    x0, y0, x1, y1 = final_roi
    # The retry result should still encompass the cube patch.
    assert x0 <= 200 and x1 >= 600 and y0 <= 300 and y1 >= 700, (
        f"Retry ROI should still contain the cube, got {final_roi}"
    )


def test_find_cube_roi_falls_back_to_full_image_when_retry_also_fails():
    """If both thresholds produce a frame-spanning component (no
    isolable cube), preserve the historical full-image fallback so
    downstream stages have a chance to do something useful.

    Synthetic: high-saturation noise everywhere with no bright
    cube — even the retry won't isolate anything."""
    arr = _synthetic_image(1000, 800, background_sat=0.60, cube_box=None)
    final = _find_cube_roi(arr)
    assert _roi_covers_full_frame(final, 800, 1000), (
        f"Both-threshold failure should fall through to full frame, got {final}"
    )


def test_roi_covers_full_frame_requires_both_dimensions_high():
    """A tight-framed photo (cube fills viewport width but not height,
    e.g., a portrait-cropped close-up) can legitimately have one
    dimension >95% without being a background-merge failure. The
    retry must NOT fire in that case — only when BOTH dimensions
    span ~the entire frame."""
    width, height = 800, 1000

    # Both dimensions full → frame-cover failure
    assert _roi_covers_full_frame((0, 0, 800, 1000), width, height) is True

    # Width spans frame but height is tight (legitimate landscape crop) → not failure
    assert _roi_covers_full_frame((0, 200, 800, 600), width, height) is False

    # Height spans frame but width is tight → not failure
    assert _roi_covers_full_frame((100, 0, 500, 1000), width, height) is False

    # Both dimensions well under threshold → not failure
    assert _roi_covers_full_frame((100, 200, 500, 700), width, height) is False
