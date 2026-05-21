from __future__ import annotations

import math

import numpy as np

from tools.interior_bezel_detection import InteriorBezelDetection
from tools.vertex_candidate_source_probe import (
    _bezel_axis_ray_candidates,
    _bezel_line_intersection_candidates,
    _evaluate_source,
    generate_source_probe_artifacts,
    render_report,
    summarize_source_rows,
)


def test_bezel_line_intersection_candidates_use_line_equations():
    mask = np.ones((100, 100), dtype=bool)
    detection = InteriorBezelDetection(
        cube_center=(50.0, 50.0),
        line_equations=[
            (1.0, 0.0, -40.0),
            (0.0, 1.0, -30.0),
            (1.0, -1.0, 0.0),
        ],
        line_qualities=[0.8, 0.6, 0.4],
    )

    candidates = _bezel_line_intersection_candidates(detection, mask)

    assert candidates
    points = [candidate["point"] for candidate in candidates]
    assert any(math.isclose(x, 40.0) and math.isclose(y, 30.0) for x, y in points)
    assert candidates[0]["source"] == "bezel_line_intersection"


def test_bezel_axis_ray_candidates_step_along_detected_axes():
    mask = np.zeros((180, 180), dtype=bool)
    mask[20:160, 20:160] = True
    detection = InteriorBezelDetection(
        cube_center=(90.0, 90.0),
        boundary_angles=[0.0, math.pi / 2.0, math.pi / 4.0],
        line_qualities=[0.9, 0.5, 0.1],
    )

    candidates = _bezel_axis_ray_candidates(detection, mask)

    assert candidates
    assert all(candidate["source"] == "bezel_axis_ray" for candidate in candidates)
    assert any(candidate["details"]["lineIndex"] == 0 for candidate in candidates)
    assert any(candidate["details"]["distanceFromDetectorPx"] == 32.0 for candidate in candidates)


def test_evaluate_source_reports_ranked_and_oracle_hits():
    candidates = [
        {"point": (50.0, 50.0), "score": 3.0, "details": {}},
        {"point": (100.0, 100.0), "score": 2.0, "details": {}},
        {"point": (11.0, 11.0), "score": 1.0, "details": {}},
    ]

    result = _evaluate_source("example", candidates, (10.0, 10.0), (10.0, 20.0))

    assert result["candidateCount"] == 3
    assert result["top1Within10px"] is False
    assert result["top3Within10px"] is True
    assert result["oracleWithin10px"] is True
    assert result["bestCandidate"]["rank"] == 3


def test_source_probe_artifact_and_report_can_render_without_images():
    feedback = {
        "rows": [
            {
                "setId": "15",
                "side": "A",
                "evaluationTier": "easy_corpus",
                "imagePath": "/tmp/does-not-exist.jpg",
                "candidateStatus": "ok",
                "status": "labeled",
                "humanVertexPoint": [10.0, 10.0],
                "topCandidates": [
                    {
                        "rank": 1,
                        "source": "center_refine",
                        "vertexPoint": [12.0, 11.0],
                        "modelScore": 2.0,
                    }
                ],
            }
        ]
    }

    document = generate_source_probe_artifacts(feedback)
    summary = summarize_source_rows(document["rows"])
    report = render_report({**document, "summary": summary})

    assert document["probe"] == "vertex_candidate_source_probe_v0"
    assert summary["labeledRowCount"] == 1
    assert summary["sourceSummaries"]["model_ranked"]["top1HitCount@10px"] == 1
    assert "Vertex Candidate Source Probe" in report
    assert "`model_ranked`" in report
