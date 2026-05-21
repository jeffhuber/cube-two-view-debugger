"""Unit tests for the per-sticker confidence signal."""

from rubik_recognizer.colors import ColorMatch
from rubik_recognizer.image_pipeline import FaceGrid, ImageAnalysis, Sticker
from rubik_recognizer.recognizer import (
    _collect_sticker_confidences,
    _confidence_stats,
    _per_sticker_confidence_signal,
)


def _sticker(idx: int, confidence: float) -> Sticker:
    return Sticker(
        id=idx,
        center=(float(idx), 0.0),
        bbox=(idx, 0, idx + 1, 1),
        rgb=(200, 200, 200),
        match=ColorMatch("white", "U", 0.5, confidence, [("white", 0.0)]),
        area=100,
    )


def _grid_with_confidences(confs: list) -> FaceGrid:
    """confs is a list of 9 confidence values for the 3x3 grid (row-major)."""
    assert len(confs) == 9
    stickers = [
        [_sticker(r * 3 + c, confs[r * 3 + c]) for c in range(3)]
        for r in range(3)
    ]
    points = [[(float(c), float(r)) for c in range(3)] for r in range(3)]
    return FaceGrid(id=0, stickers=stickers, points=points, matched_count=9, fit_error=0.0)


def _analysis(grids: list) -> ImageAnalysis:
    flat_stickers = [s for g in grids for row in g.stickers for s in row]
    return ImageAnalysis(
        width=100, height=100, roi=(0, 0, 100, 100),
        stickers=flat_stickers, grids=grids, overlay_data_url="", warnings=[],
    )


def test_confidence_stats_empty_returns_nulls():
    s = _confidence_stats([])
    assert s["count"] == 0
    assert s["min"] is None
    assert s["median"] is None
    assert s["max"] is None


def test_confidence_stats_returns_correct_aggregates():
    s = _confidence_stats([0.5, 1.0, 1.5, 2.0, 2.5])
    assert s["count"] == 5
    assert s["min"] == 0.5
    assert s["max"] == 2.5
    assert s["median"] == 1.5


def test_confidence_stats_handles_even_count_median():
    s = _confidence_stats([1.0, 2.0, 3.0, 4.0])
    assert s["median"] == 2.5  # (2 + 3) / 2


def test_collect_sticker_confidences_flattens_grid():
    grid = _grid_with_confidences([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    a = _analysis([grid])
    confs = _collect_sticker_confidences(a)
    assert len(confs) == 9
    assert confs[0] == 1.0
    assert confs[-1] == 5.0


def test_collect_sticker_confidences_uses_only_first_three_grids():
    """cv-local emits "supplemental" grids too; the signal scopes to the
    primary 3 selected face grids to avoid double-counting."""
    grid = _grid_with_confidences([1.5] * 9)
    a = _analysis([grid, grid, grid, grid, grid])
    confs = _collect_sticker_confidences(a)
    assert len(confs) == 27  # 3 * 9, not 5 * 9


def test_per_sticker_confidence_signal_well_formed_pair():
    """ColorMatch confidence is in [0, 1]. High-confidence pair → 0 below threshold."""
    a = _analysis([_grid_with_confidences([0.5] * 9) for _ in range(3)])
    b = _analysis([_grid_with_confidences([0.7] * 9) for _ in range(3)])
    s = _per_sticker_confidence_signal(a, b)
    assert s["perPhoto"]["imageA"]["count"] == 27
    assert s["perPhoto"]["imageB"]["count"] == 27
    assert s["perPair"]["count"] == 54
    assert s["perPhoto"]["imageA"]["min"] == 0.5
    assert s["perPhoto"]["imageB"]["min"] == 0.7
    assert s["perPair"]["min"] == 0.5  # min of both
    assert s["belowThresholdCount"] == 0  # all >= 0.10 threshold
    assert s["threshold"] == 0.10


def test_per_sticker_confidence_signal_below_threshold_counted():
    """Confidences below threshold (default 0.10) are counted."""
    a = _analysis([
        _grid_with_confidences([0.02, 0.05, 0.08, 0.09, 0.099, 0.20, 0.30, 0.40, 0.50]),
        _grid_with_confidences([0.50] * 9),
        _grid_with_confidences([0.50] * 9),
    ])
    b = _analysis([_grid_with_confidences([0.50] * 9) for _ in range(3)])
    s = _per_sticker_confidence_signal(a, b)
    assert s["belowThresholdCount"] == 5  # 5 stickers below 0.10
    assert s["perPair"]["min"] == 0.02


def test_per_sticker_confidence_threshold_env_override(monkeypatch):
    """RUBIK_PER_STICKER_CONFIDENCE_THRESHOLD env var overrides default."""
    monkeypatch.setenv("RUBIK_PER_STICKER_CONFIDENCE_THRESHOLD", "0.30")
    a = _analysis([_grid_with_confidences([0.20] * 9) for _ in range(3)])
    b = _analysis([_grid_with_confidences([0.50] * 9) for _ in range(3)])
    s = _per_sticker_confidence_signal(a, b)
    assert s["threshold"] == 0.30
    assert s["belowThresholdCount"] == 27  # A's 27 stickers all < 0.30


def test_per_sticker_confidence_invalid_env_falls_back(monkeypatch):
    """Invalid env value falls back to default without crashing."""
    monkeypatch.setenv("RUBIK_PER_STICKER_CONFIDENCE_THRESHOLD", "not-a-number")
    a = _analysis([_grid_with_confidences([0.50] * 9) for _ in range(3)])
    b = _analysis([_grid_with_confidences([0.50] * 9) for _ in range(3)])
    s = _per_sticker_confidence_signal(a, b)
    assert s["threshold"] == 0.10


def test_per_sticker_confidence_signal_empty_analysis_no_crash():
    """If an analysis has no grids, signal returns empty/null stats but
    doesn't crash. Important for the "rejected at grid-fit stage" failure
    path that recognition_signals must still serialize."""
    a = _analysis([])
    b = _analysis([])
    s = _per_sticker_confidence_signal(a, b)
    assert s["perPhoto"]["imageA"]["count"] == 0
    assert s["perPhoto"]["imageA"]["min"] is None
    assert s["perPair"]["count"] == 0
    assert s["belowThresholdCount"] == 0
