import json

from tools.label_geometry_baseline import latest_label_paths, normalize_set_id, summarize_sets


def test_latest_label_paths_selects_newest_per_set_side(tmp_path):
    old_label = _write_label(tmp_path / "old-set-46-a-geometry-label.json", "Set 46", "A", "2026-05-17T01:00:00Z")
    new_label = _write_label(tmp_path / "new-set-46-a-geometry-label.json", "Set 46", "A", "2026-05-17T02:00:00Z")
    side_b = _write_label(tmp_path / "set-46-b-geometry-label.json", "Set 46", "B", "2026-05-17T01:30:00Z")
    _write_label(tmp_path / "set-47-a-geometry-label.json", "Set 47", "A", "2026-05-17T03:00:00Z")

    assert latest_label_paths(tmp_path, ["46"]) == [new_label, side_b]
    assert old_label not in latest_label_paths(tmp_path, ["46"])


def test_normalize_set_id_accepts_common_spellings():
    assert normalize_set_id("Set 46") == "46"
    assert normalize_set_id("set-46") == "46"
    assert normalize_set_id(46) == "46"


def test_summarize_sets_flags_missing_sides_and_aggregates_outliers():
    summary = summarize_sets(
        [
            {
                "setId": "Set 46",
                "imageSide": "A",
                "stickersOutsideHull": 0,
                "selectedGridCellsOutsideHull": 6,
                "topVisibleTripleCellsOutsideHull": 2,
                "hullIou": 0.74,
                "imageSha256Matches": True,
            },
            {
                "setId": "Set 46",
                "imageSide": "B",
                "stickersOutsideHull": 1,
                "selectedGridCellsOutsideHull": 0,
                "topVisibleTripleCellsOutsideHull": 0,
                "hullIou": 0.72,
                "imageSha256Matches": True,
            },
            {
                "setId": "Set 47",
                "imageSide": "A",
                "stickersOutsideHull": 0,
                "selectedGridCellsOutsideHull": 0,
                "topVisibleTripleCellsOutsideHull": 0,
                "hullIou": 0.7,
                "imageSha256Matches": False,
            },
        ]
    )

    assert summary[0] == {
        "setId": "Set 46",
        "sides": ["A", "B"],
        "missingSides": [],
        "stickersOutsideHull": 1,
        "selectedGridCellsOutsideHull": 6,
        "topVisibleTripleCellsOutsideHull": 2,
        "minHullIou": 0.72,
        "imageSha256Matches": True,
    }
    assert summary[1]["missingSides"] == ["B"]
    assert summary[1]["imageSha256Matches"] is False


def _write_label(path, set_id, side, saved_at):
    payload = {
        "setId": set_id,
        "imageSide": side,
        "savedAt": saved_at,
        "image": {"name": f"{set_id} {side}.jpg"},
        "labels": {"cubeHull": [[0, 0], [1, 0], [1, 1]], "faceQuads": {}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
