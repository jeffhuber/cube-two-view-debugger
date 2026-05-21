"""Unit tests for the two-view geometric consistency signal."""

from rubik_recognizer.colors import ColorMatch
from rubik_recognizer.image_pipeline import FaceGrid, ImageAnalysis, Sticker
from rubik_recognizer.recognizer import (
    _grid_median_spacing,
    _photo_median_spacing,
    _two_view_geometry_consistency,
)


def _sticker(idx: int, center: tuple) -> Sticker:
    return Sticker(
        id=idx,
        center=center,
        bbox=(center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5),
        rgb=(200, 200, 200),
        match=ColorMatch("white", "U", 0.5, 0.5, [("white", 0.0)]),
        area=100,
    )


def _grid_with_spacing(spacing: float, origin: tuple = (100.0, 100.0)) -> FaceGrid:
    """Build a 3x3 face grid with uniform pixel spacing between adjacent cells."""
    points = [
        [(origin[0] + c * spacing, origin[1] + r * spacing) for c in range(3)]
        for r in range(3)
    ]
    stickers = [
        [_sticker(r * 3 + c, points[r][c]) for c in range(3)]
        for r in range(3)
    ]
    return FaceGrid(id=0, stickers=stickers, points=points, matched_count=9, fit_error=0.0)


def _analysis_with_grids(grids):
    return ImageAnalysis(
        width=1000, height=1000, roi=(0, 0, 1000, 1000),
        stickers=[s for g in grids for row in g.stickers for s in row],
        grids=grids, overlay_data_url="", warnings=[],
    )


def test_grid_median_spacing_returns_uniform_spacing():
    grid = _grid_with_spacing(50.0)
    assert _grid_median_spacing(grid) == 50.0


def test_grid_median_spacing_handles_non_uniform_grid():
    """If row spacing differs from column spacing, median falls between."""
    # 3x3 grid with col spacing 50, row spacing 80
    pts = [
        [(100, 100), (150, 100), (200, 100)],
        [(100, 180), (150, 180), (200, 180)],
        [(100, 260), (150, 260), (200, 260)],
    ]
    grid = FaceGrid(
        id=0,
        stickers=[[_sticker(0, p) for p in row] for row in pts],
        points=pts,
        matched_count=9, fit_error=0.0,
    )
    # 6 col-adjacencies at 50 + 6 row-adjacencies at 80 = 12 values
    # median of [50,50,50,50,50,50,80,80,80,80,80,80] = (50+80)/2 = 65
    assert _grid_median_spacing(grid) == 65.0


def test_grid_median_spacing_returns_none_for_malformed_grid():
    grid = FaceGrid(id=0, stickers=[], points=[], matched_count=0, fit_error=0.0)
    assert _grid_median_spacing(grid) is None


def test_photo_median_spacing_aggregates_across_grids():
    a = _analysis_with_grids([
        _grid_with_spacing(50.0),
        _grid_with_spacing(48.0),
        _grid_with_spacing(52.0),
    ])
    assert _photo_median_spacing(a) == 50.0  # median of [48, 50, 52]


def test_photo_median_spacing_none_when_no_grids():
    a = _analysis_with_grids([])
    assert _photo_median_spacing(a) is None


def test_two_view_consistency_balanced_pair_marked_ok():
    """A and B with similar spacing → not flagged inconsistent."""
    a = _analysis_with_grids([_grid_with_spacing(50.0)])
    b = _analysis_with_grids([_grid_with_spacing(52.0)])
    result = _two_view_geometry_consistency(a, b)
    assert result["inconsistent"] is False
    assert result["reason"] == "ok"
    assert result["ratio"] is not None and result["ratio"] < 1.4


def test_two_view_consistency_extreme_ratio_marked_inconsistent():
    """A spacing 50, B spacing 150 → ratio 3.0 → inconsistent."""
    a = _analysis_with_grids([_grid_with_spacing(50.0)])
    b = _analysis_with_grids([_grid_with_spacing(150.0)])
    result = _two_view_geometry_consistency(a, b)
    assert result["inconsistent"] is True
    assert result["reason"] == "two_view_inconsistent"
    assert result["ratio"] == 3.0


def test_two_view_consistency_handles_missing_grids():
    """One photo has no grids → can't compute, but no crash."""
    a = _analysis_with_grids([_grid_with_spacing(50.0)])
    b = _analysis_with_grids([])
    result = _two_view_geometry_consistency(a, b)
    assert result["inconsistent"] is False
    assert result["reason"] == "missing_grids"
    assert result["spacingPxB"] is None


def test_two_view_consistency_tolerance_env_variable(monkeypatch):
    """Tolerance is configurable via env var."""
    monkeypatch.setenv("RUBIK_TWO_VIEW_RATIO_TOLERANCE", "1.1")
    a = _analysis_with_grids([_grid_with_spacing(50.0)])
    b = _analysis_with_grids([_grid_with_spacing(60.0)])  # ratio 1.2
    result = _two_view_geometry_consistency(a, b)
    assert result["inconsistent"] is True  # 1.2 > 1.1 with stricter tolerance
    assert result["toleranceRatio"] == 1.1


def test_two_view_consistency_invalid_env_falls_back_to_default(monkeypatch):
    """Invalid env value → use default 1.4, don't crash."""
    monkeypatch.setenv("RUBIK_TWO_VIEW_RATIO_TOLERANCE", "not-a-number")
    a = _analysis_with_grids([_grid_with_spacing(50.0)])
    b = _analysis_with_grids([_grid_with_spacing(55.0)])
    result = _two_view_geometry_consistency(a, b)
    assert result["toleranceRatio"] == 1.4
    assert result["inconsistent"] is False
