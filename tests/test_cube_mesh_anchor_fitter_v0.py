from __future__ import annotations

import math

import numpy as np

from tools.cube_mesh_anchor_fitter_v0 import (
    ANCHOR_NAMES,
    fit_cube_mesh_anchor_candidates,
    serialize_anchor_mesh,
)
from tools.interior_bezel_detection import InteriorBezelDetection
from tools.render_cube_mesh_anchor_overlays import render_report, summarize_rows


def test_fit_cube_mesh_anchor_candidates_returns_named_seven_anchor_mesh():
    mask = np.zeros((240, 240), dtype=bool)
    mask[25:220, 25:220] = True
    detection = InteriorBezelDetection(
        cube_center=(80.0, 80.0),
        boundary_angles=[0.0, math.pi / 2.0, math.pi / 4.0],
        boundary_lines=[
            ((80.0, 80.0), (180.0, 80.0)),
            ((80.0, 80.0), (80.0, 180.0)),
            ((80.0, 80.0), (160.0, 160.0)),
        ],
        line_qualities=[0.8, 0.7, 0.6],
        signal_quality=0.7,
        debug={"centroid_seed": [82.0, 82.0]},
    )

    result = fit_cube_mesh_anchor_candidates(
        detection,
        mask,
        edge_steps=4,
        center_offsets=((0.0, 0.0), (24.0, 24.0)),
        top_vertex_candidates=2,
        top_meshes=2,
    )

    assert result.top_mesh is not None
    assert result.status in {
        "ok",
        "low_iou",
        "low_inside_ratio",
        "low_cell_inside",
        "unbalanced_face_area",
        "weak_axis_separation",
    }
    assert len(result.meshes) == 2
    assert tuple(result.top_mesh.anchors.keys()) == ANCHOR_NAMES
    assert result.diagnostics["probeVersion"] == "cube-mesh-anchor-v0"
    assert result.diagnostics["fitMethod"] == "weak_projected_7_anchor_v0"
    assert result.diagnostics["pnpStatus"] == "not_run_no_calibrated_camera_or_human_anchor_labels"

    serialized = serialize_anchor_mesh(result.top_mesh, 1)
    assert serialized["rank"] == 1
    assert serialized["fitMethod"] == "weak_projected_7_anchor_v0"
    assert list(serialized["anchors"].keys()) == list(ANCHOR_NAMES)
    assert "anchorNearSilhouetteRatio" in serialized["scoreComponents"]


def test_fit_cube_mesh_anchor_candidates_abstains_with_vertex_error():
    mask = np.ones((80, 80), dtype=bool)
    detection = InteriorBezelDetection(cube_center=None, boundary_angles=[0.0, 1.0, 2.0])

    result = fit_cube_mesh_anchor_candidates(detection, mask)

    assert result.meshes == ()
    assert result.status == "missing_center"
    assert result.diagnostics["error"] == "no vertex candidates"


def test_cube_mesh_anchor_report_summary_keeps_easy_weak_rows_visible():
    rows = [
        {
            "setId": "15",
            "side": "A",
            "evaluationTier": "easy_corpus",
            "status": "ok",
            "diagnosticDisposition": "model_ok",
            "anchorMeshes": [
                {
                    "sourceVertexRank": 1,
                    "sourceVertexCandidate": {"source": "center_refine"},
                    "score": 2.0,
                    "scoreComponents": {
                        "silhouetteIoU": 0.8,
                        "anchorNearSilhouetteRatio": 1.0,
                        "faceAreaBalance": 0.9,
                        "axisAngleSeparationScore": 1.0,
                    },
                }
            ],
        },
        {
            "setId": "26",
            "side": "B",
            "evaluationTier": "easy_corpus",
            "status": "low_iou",
            "diagnosticDisposition": "model_iteration_needed",
            "anchorMeshes": [
                {
                    "sourceVertexRank": 1,
                    "sourceVertexCandidate": {"source": "silhouette_centroid_seed"},
                    "score": 1.0,
                    "scoreComponents": {},
                }
            ],
        },
    ]

    summary = summarize_rows(rows)

    assert summary["requestedPairCount"] == 2
    assert summary["imageRowCount"] == 2
    assert summary["fittedRowCount"] == 2
    assert summary["okRowCount"] == 1
    assert summary["weakRowCount"] == 1
    assert summary["easyOkRowCount"] == 1
    assert summary["easyWeakRowCount"] == 1
    assert summary["modelIterationNeededRowCount"] == 1
    assert summary["lowAnchorSupportWarningRowCount"] == 0

    report = render_report({"summary": summary, "rows": rows})
    assert "Cube Mesh Anchor V0 Diagnostics" in report
    assert "Easy-corpus weak rows" in report
    assert "`low_iou`" in report
