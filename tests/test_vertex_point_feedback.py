from __future__ import annotations

from tools.vertex_point_feedback import (
    build_feedback_scaffold,
    evaluate_feedback,
    render_report,
)


def test_build_feedback_scaffold_preserves_candidates_for_labeling():
    candidate_document = {
        "rows": [
            {
                "setId": "15",
                "side": "A",
                "evaluationTier": "easy_corpus",
                "imagePath": "/tmp/image.jpg",
                "overlayPath": "/tmp/overlay.png",
                "status": "ok",
                "candidateDiagnostics": {"topSameStatusScoreGap": 0.2},
                "topCandidates": [
                    {
                        "rank": 1,
                        "source": "center_refine",
                        "vertexPoint": [10.0, 20.0],
                        "modelStatus": "ok",
                        "modelScore": 1.2,
                        "scoreComponents": {"silhouetteIoU": 0.8},
                    }
                ],
            }
        ]
    }

    scaffold = build_feedback_scaffold(candidate_document, source_candidate_summary="candidate.json")

    assert scaffold["probe"] == "vertex_point_human_feedback_v0"
    assert scaffold["sourceCandidateSummary"] == "candidate.json"
    row = scaffold["rows"][0]
    assert row["status"] == "unlabeled"
    assert row["humanVertexPoint"] is None
    assert row["topCandidates"][0]["vertexPoint"] == [10.0, 20.0]
    assert "10px" in row["target"]


def test_evaluate_feedback_reports_top1_and_top3_recall():
    feedback = {
        "distanceThresholdPx": 10.0,
        "sourceCandidateSummary": "candidate.json",
        "rows": [
            {
                "setId": "15",
                "side": "A",
                "evaluationTier": "easy_corpus",
                "candidateStatus": "ok",
                "status": "labeled",
                "humanVertexPoint": [100.0, 100.0],
                "topCandidates": [
                    {"rank": 1, "source": "a", "vertexPoint": [103.0, 104.0]},
                    {"rank": 2, "source": "b", "vertexPoint": [140.0, 140.0]},
                ],
            },
            {
                "setId": "15",
                "side": "B",
                "evaluationTier": "easy_corpus",
                "candidateStatus": "ok",
                "status": "labeled",
                "humanVertexPoint": [200.0, 200.0],
                "topCandidates": [
                    {"rank": 1, "source": "a", "vertexPoint": [230.0, 230.0]},
                    {"rank": 2, "source": "b", "vertexPoint": [204.0, 204.0]},
                ],
            },
            {
                "setId": "26",
                "side": "B",
                "evaluationTier": "easy_corpus",
                "candidateStatus": "low_iou",
                "status": "unlabeled",
                "topCandidates": [],
            },
        ],
    }

    evaluation = evaluate_feedback(feedback)

    assert evaluation["summary"]["rowCount"] == 3
    assert evaluation["summary"]["labeledRowCount"] == 2
    assert evaluation["summary"]["unlabeledRowCount"] == 1
    assert evaluation["summary"]["top1HitCount"] == 1
    assert evaluation["summary"]["top3HitCount"] == 2
    assert evaluation["summary"]["top1HitRate"] == 0.5
    assert evaluation["summary"]["top3HitRate"] == 1.0
    assert evaluation["rows"][0]["top1WithinThreshold"] is True
    assert evaluation["rows"][1]["top1WithinThreshold"] is False
    assert evaluation["rows"][1]["top3ContainsTruth"] is True

    report = render_report(feedback, evaluation)
    assert "Vertex Point Human Feedback" in report
    assert "Top-1 hit rate: 50.0%" in report
    assert "`unlabeled`" in report
