from __future__ import annotations

import math

import numpy as np

from tools.global_cube_model_v0 import (
    fit_projected_cube_model,
    model_mask,
    serialize_model,
    subdivide_face_quad,
)
from tools.interior_bezel_detection import InteriorBezelDetection
from tools.render_global_cube_model_v0_overlays import render_report, summarize_rows


def test_subdivide_face_quad_returns_row_major_nine_cells():
    cells = subdivide_face_quad([(0, 0), (90, 0), (90, 90), (0, 90)])

    assert len(cells) == 9
    assert cells[0]["row"] == 0
    assert cells[0]["col"] == 0
    assert cells[0]["quad"] == [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
    assert cells[-1]["row"] == 2
    assert cells[-1]["col"] == 2
    assert cells[-1]["quad"] == [(60.0, 60.0), (90.0, 60.0), (90.0, 90.0), (60.0, 90.0)]


def test_fit_projected_cube_model_yields_coherent_face_and_cell_quads():
    mask = np.zeros((220, 220), dtype=bool)
    mask[20:200, 20:200] = True
    detection = InteriorBezelDetection(
        cube_center=(80.0, 80.0),
        boundary_angles=[0.0, math.pi / 2.0, math.pi / 4.0],
        boundary_lines=[
            ((80.0, 80.0), (170.0, 80.0)),
            ((80.0, 80.0), (80.0, 170.0)),
            ((80.0, 80.0), (155.0, 155.0)),
        ],
        line_qualities=[0.8, 0.7, 0.6],
        signal_quality=0.7,
    )

    result = fit_projected_cube_model(detection, mask, edge_steps=8)

    assert result.model is not None
    assert result.status in {"ok", "low_iou", "low_inside_ratio", "low_cell_inside"}
    assert len(result.model.faces) == 3
    assert len(result.model.all_cell_quads()) == 27
    assert result.model.score_components["insideRatio"] > 0.75
    raster = model_mask(result.model, mask.shape)
    assert raster.sum() > 0
    serialized = serialize_model(result.model)
    assert serialized["edgeLength"] > 0
    assert [face["cellCount"] for face in serialized["faces"]] == [9, 9, 9]


def test_fit_projected_cube_model_abstains_on_missing_axes():
    mask = np.ones((80, 80), dtype=bool)
    detection = InteriorBezelDetection(cube_center=(40.0, 40.0), boundary_angles=[0.0])

    result = fit_projected_cube_model(detection, mask)

    assert result.model is None
    assert result.status == "missing_axes"


def test_report_summary_keeps_empty_artifacts_visible():
    rows = [
        {"setId": "45", "side": "A", "status": "ok", "model": {"score": 1.0, "scoreComponents": {}}},
        {"setId": "45", "side": "B", "status": "image_missing"},
        {"setId": "61", "side": "A", "status": "low_iou", "model": {"score": 0.2, "scoreComponents": {}}},
    ]

    summary = summarize_rows(rows)

    assert summary["requestedPairCount"] == 2
    assert summary["imageRowCount"] == 3
    assert summary["fittedRowCount"] == 2
    assert summary["okRowCount"] == 1
    assert summary["lowIouRowCount"] == 1
    assert summary["lowCellInsideRowCount"] == 0
    assert summary["errorRowCount"] == 1
    report = render_report({"summary": summary, "rows": rows})
    assert "Global Cube Model V0 Diagnostics" in report
    assert "`image_missing`" in report
