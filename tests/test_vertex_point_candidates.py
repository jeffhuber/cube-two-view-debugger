from __future__ import annotations

import math

import numpy as np

from tools.interior_bezel_detection import InteriorBezelDetection
from tools.render_vertex_point_candidates import render_report, summarize_rows
from tools.vertex_point_candidates import (
    fit_result_from_vertex_candidate,
    rank_vertex_point_candidates,
    serialize_vertex_candidate,
)


def test_rank_vertex_point_candidates_returns_top_n_schema():
    mask = np.zeros((220, 220), dtype=bool)
    mask[30:205, 30:205] = True
    detection = InteriorBezelDetection(
        cube_center=(70.0, 70.0),
        boundary_angles=[0.0, math.pi / 2.0, math.pi / 4.0],
        boundary_lines=[
            ((70.0, 70.0), (170.0, 70.0)),
            ((70.0, 70.0), (70.0, 170.0)),
            ((70.0, 70.0), (150.0, 150.0)),
        ],
        line_qualities=[0.8, 0.7, 0.6],
        signal_quality=0.7,
        debug={"centroid_seed": [68.0, 68.0]},
    )

    result = rank_vertex_point_candidates(
        detection,
        mask,
        edge_steps=4,
        center_offsets=((0.0, 0.0), (24.0, 24.0)),
        top_n=2,
    )

    assert result.top_candidate is not None
    assert result.status in {"ok", "low_iou", "low_inside_ratio", "low_cell_inside"}
    assert len(result.candidates) == 2
    assert result.diagnostics["probeVersion"] == "vertex-point-candidates-v0"
    assert result.diagnostics["rankingPolicy"] == "prefer_ok_then_model_score"
    assert result.diagnostics["candidatePointCount"] == 3
    assert result.diagnostics["returnedCandidateCount"] == 2
    assert result.diagnostics["evaluatedModels"] == 96
    serialized = serialize_vertex_candidate(result.candidates[0], 1)
    assert serialized["rank"] == 1
    assert len(serialized["vertexPoint"]) == 2
    assert "silhouetteIoU" in serialized["scoreComponents"]
    fit = fit_result_from_vertex_candidate(result.candidates[0])
    assert fit.model is result.candidates[0].model
    assert fit.diagnostics["fitVersion"] == "vertex-point-candidates-v0"


def test_rank_vertex_point_candidates_abstains_on_missing_center():
    mask = np.ones((80, 80), dtype=bool)
    detection = InteriorBezelDetection(cube_center=None, boundary_angles=[0.0, 1.0, 2.0])

    result = rank_vertex_point_candidates(detection, mask)

    assert result.candidates == ()
    assert result.status == "missing_center"


def test_vertex_point_candidate_report_summary_keeps_manual_review_visible():
    rows = [
        {
            "setId": "15",
            "side": "A",
            "evaluationTier": "easy_corpus",
            "status": "ok",
            "topCandidates": [{"source": "center_refine", "modelScore": 1.0, "scoreComponents": {}}],
            "candidateDiagnostics": {"detectorSignalQuality": 0.5},
            "manualReview": {"status": "unlabeled"},
        },
        {
            "setId": "26",
            "side": "B",
            "evaluationTier": "easy_corpus",
            "status": "low_iou",
            "topCandidates": [{"source": "bezel_detector", "modelScore": 0.5, "scoreComponents": {}}],
            "candidateDiagnostics": {"detectorSignalQuality": 0.1},
            "manualReview": {"status": "unlabeled"},
        },
        {
            "setId": "99",
            "side": "A",
            "evaluationTier": "unknown",
            "status": "image_missing",
        },
    ]

    summary = summarize_rows(rows)

    assert summary["requestedPairCount"] == 3
    assert summary["imageRowCount"] == 3
    assert summary["candidateRowCount"] == 2
    assert summary["topOkRowCount"] == 1
    assert summary["easyTopOkRowCount"] == 1
    assert summary["easyTopWeakRowCount"] == 1
    assert summary["errorRowCount"] == 1
    assert summary["unlabeledManualReviewRowCount"] == 2
    report = render_report({"summary": summary, "rows": rows})
    assert "Vertex Point Candidate Diagnostics" in report
    assert "Unlabeled manual-review rows" in report
    assert "`low_iou`" in report
