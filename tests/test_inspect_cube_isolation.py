from types import SimpleNamespace

from PIL import Image

from tools.inspect_cube_isolation import (
    convex_hull,
    expand_polygon,
    isolation_diagnostics_for_analysis,
    point_in_polygon,
    write_overlay,
)


def test_convex_hull_drops_interior_points():
    hull = convex_hull([(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])

    assert hull == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_expand_polygon_and_point_in_polygon_include_boundary():
    expanded = expand_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], 0.2)

    assert point_in_polygon((0, 0), expanded)
    assert point_in_polygon((5, 5), expanded)
    assert not point_in_polygon((20, 20), expanded)


def test_isolation_diagnostics_classifies_stickers_outside_selected_grid_hull():
    inside = _sticker(1, (30, 30), "U")
    outside = _sticker(2, (180, 180), "B")
    analysis = SimpleNamespace(
        width=220,
        height=220,
        roi=(0, 0, 220, 220),
        warnings=[],
        stickers=[inside, outside],
        grids=[
            _grid(1, "U", 20, 20),
            _grid(2, "F", 40, 40),
            _grid(3, "R", 60, 60),
        ],
    )

    diagnostics = isolation_diagnostics_for_analysis(analysis, anchor="U", padding_fraction=0.1)

    assert diagnostics["anchorUsed"] == "U"
    assert diagnostics["analysis"]["coordinateSpace"] == "processingImage"
    assert diagnostics["analysis"]["processingWidth"] == 220
    assert diagnostics["analysis"]["processingHeight"] == 220
    assert diagnostics["proposedCubeRegion"]["hullSource"] == "selectedGridPoints"
    assert diagnostics["classificationSummary"]["kept"] == 1
    assert diagnostics["classificationSummary"]["dropped"] == 1
    assert diagnostics["stickers"][0]["proposedAction"] == "keep"
    assert diagnostics["stickers"][1]["proposedAction"] == "drop"


def test_write_overlay_uses_processing_image_coordinate_space(tmp_path):
    image_path = tmp_path / "large-input.jpg"
    output_path = tmp_path / "overlay.png"
    Image.new("RGB", (3024, 4032), "white").save(image_path)
    diagnostics = {
        "analysis": {
            "roi": [0, 0, 862, 1150],
        },
        "proposedCubeRegion": {
            "hull": [[20, 20], [842, 20], [842, 1130], [20, 1130]],
            "paddedHull": [[10, 10], [852, 10], [852, 1140], [10, 1140]],
        },
        "stickers": [
            {
                "center": [861, 1149],
                "face": "U",
                "source": "component",
                "insideProposedCubeHull": True,
            }
        ],
    }

    write_overlay(image_path, diagnostics, output_path)

    with Image.open(output_path) as overlay:
        assert overlay.size == (862, 1150)


def _sticker(sticker_id, center, face, source="component"):
    return SimpleNamespace(
        id=sticker_id,
        center=center,
        source=source,
        rgb=(230, 230, 230),
        shape_angle=0.0,
        match=SimpleNamespace(face=face, color="white", confidence=0.9),
    )


def _grid(grid_id, face, x, y):
    stickers = [
        [_sticker(grid_id * 10 + row * 3 + col, (x + col * 10, y + row * 10), face) for col in range(3)]
        for row in range(3)
    ]
    return SimpleNamespace(
        id=grid_id,
        center_face=face,
        center_sticker=stickers[1][1],
        matched_count=9,
        fit_error=1.0,
        points=[[(x + col * 10, y + row * 10) for col in range(3)] for row in range(3)],
        stickers=stickers,
    )
