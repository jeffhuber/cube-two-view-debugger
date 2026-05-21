from __future__ import annotations

import math

import numpy as np

from tools.interior_bezel_detection import InteriorBezelDetection
from tools.vertex_fitter_assisted_ranker_v0 import (
    _fit_candidate_pool,
    _rank_fitted_by_model_score,
    generate_fitter_assisted_artifacts,
    render_report,
    summarize_fitter_rows,
)


def test_fit_candidate_pool_adds_projected_model_scores():
    mask = np.zeros((160, 160), dtype=bool)
    mask[20:145, 20:145] = True
    detection = InteriorBezelDetection(
        cube_center=(52.0, 52.0),
        boundary_angles=[0.0, math.pi / 2.0, math.pi / 4.0],
        line_qualities=[0.8, 0.7, 0.6],
        signal_quality=0.75,
    )
    candidates = [
        {
            "source": "model_local_grid",
            "sourceRank": 1,
            "point": (52.0, 52.0),
            "sourceScore": 1.0,
            "details": {"offset": [0.0, 0.0], "seedRank": 1},
        },
        {
            "source": "model_local_grid",
            "sourceRank": 2,
            "point": (118.0, 118.0),
            "sourceScore": 0.5,
            "details": {"offset": [48.0, 48.0], "seedRank": 4},
        },
    ]

    fitted = _fit_candidate_pool(
        candidates,
        detection,
        mask,
        edge_steps=3,
        scoring_max_dim=160,
    )

    assert len(fitted) == 2
    assert all("modelScore" in item for item in fitted)
    assert all("fitterScore" in item for item in fitted)
    assert all("silhouetteIoU" in item["scoreComponents"] for item in fitted)
    assert fitted[0]["_fitScale"] == 1.0


def test_rank_fitted_by_model_score_prefers_ok_then_score():
    candidates = [
        {"fitStatus": "low_iou", "modelScore": 10.0, "sourceRank": 1},
        {"fitStatus": "ok", "modelScore": 1.0, "sourceRank": 3},
        {"fitStatus": "ok", "modelScore": 2.0, "sourceRank": 2},
    ]

    ranked = _rank_fitted_by_model_score(candidates)

    assert ranked[0]["fitStatus"] == "ok"
    assert ranked[0]["modelScore"] == 2.0
    assert ranked[-1]["fitStatus"] == "low_iou"


def test_generate_fitter_artifact_handles_missing_images():
    feedback = {
        "rows": [
            {
                "setId": "15",
                "side": "A",
                "evaluationTier": "easy_corpus",
                "imagePath": "/tmp/does-not-exist.jpg",
                "status": "labeled",
                "humanVertexPoint": [10.0, 10.0],
            }
        ]
    }

    document = generate_fitter_assisted_artifacts(feedback, output_dir=None)
    report = render_report(document)

    assert document["probe"] == "vertex_fitter_assisted_ranker_v0"
    assert document["summary"]["rowCount"] == 1
    assert document["summary"]["labeledRowCount"] == 0
    assert document["summary"]["errorRowCount"] == 1
    assert "Vertex Fitter-Assisted Ranker V0" in report


def test_summarize_fitter_rows_counts_policy_hits():
    rows = [
        {
            "evaluationStatus": "labeled",
            "candidatePoolSize": 2,
            "fittedCandidateCount": 1,
            "policyResults": {
                "fitter_assisted_v0": {
                    "top1Hit@10px": True,
                    "top3Hit@10px": True,
                    "top5Hit@10px": True,
                    "oracleHit@10px": True,
                    "top1Hit@20px": True,
                    "top3Hit@20px": True,
                    "top5Hit@20px": True,
                    "oracleHit@20px": True,
                },
                "combined_oracle": {
                    "top1Hit@10px": True,
                    "top3Hit@10px": True,
                    "top5Hit@10px": True,
                    "oracleHit@10px": True,
                    "top1Hit@20px": True,
                    "top3Hit@20px": True,
                    "top5Hit@20px": True,
                    "oracleHit@20px": True,
                },
            },
        }
    ]

    summary = summarize_fitter_rows(rows)

    assert summary["labeledRowCount"] == 1
    assert summary["meanCandidatePoolSize"] == 2.0
    assert summary["meanFittedCandidateCount"] == 1.0
    assert summary["policySummaries"]["fitter_assisted_v0"]["top1HitCount@10px"] == 1
