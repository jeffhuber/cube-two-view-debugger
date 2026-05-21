from __future__ import annotations

from tools.vertex_candidate_ranker_v0 import (
    LabeledCandidateRow,
    _evaluate_ranker_row,
    _flatten_sources,
    _source_heuristic_score,
    generate_ranker_artifacts,
    render_report,
    summarize_ranker_rows,
)


def test_flatten_sources_keeps_source_rank_and_details():
    sources = {
        "model_ranked": [
            {"point": (10.0, 20.0), "score": 2.0, "details": {"baselineRank": 1}},
        ],
        "model_local_grid": [
            {"point": (12.0, 22.0), "score": 1.0, "details": {"offset": [0, 16]}},
        ],
    }

    flattened = _flatten_sources(sources)

    assert [item["source"] for item in flattened] == ["model_ranked", "model_local_grid"]
    assert flattened[0]["sourceRank"] == 1
    assert flattened[1]["details"]["offset"] == [0, 16]


def test_source_heuristic_prefers_reasonable_local_grid_candidate():
    near_seed = {
        "source": "model_local_grid",
        "sourceRank": 1,
        "details": {"offset": [0.0, 16.0], "seedRank": 1},
    }
    far_seed = {
        "source": "model_local_grid",
        "sourceRank": 2,
        "details": {"offset": [48.0, 48.0], "seedRank": 5},
    }

    assert _source_heuristic_score(near_seed) > _source_heuristic_score(far_seed)


def test_evaluate_ranker_row_reports_policy_metrics_without_label_leakage():
    rows = [
        LabeledCandidateRow(
            set_id="1",
            side="A",
            evaluation_tier="easy_corpus",
            image_path="/tmp/a.jpg",
            human_vertex=(10.0, 10.0),
            candidates=(
                {
                    "source": "model_ranked",
                    "sourceRank": 1,
                    "point": (40.0, 40.0),
                    "sourceScore": 1.0,
                    "details": {"baselineRank": 1, "modelScore": 1.0},
                },
                {
                    "source": "model_local_grid",
                    "sourceRank": 1,
                    "point": (12.0, 11.0),
                    "sourceScore": 1.0,
                    "details": {"offset": [0.0, 16.0], "seedRank": 1},
                },
            ),
        ),
        LabeledCandidateRow(
            set_id="2",
            side="A",
            evaluation_tier="easy_corpus",
            image_path="/tmp/b.jpg",
            human_vertex=(100.0, 100.0),
            candidates=(
                {
                    "source": "model_ranked",
                    "sourceRank": 1,
                    "point": (100.0, 101.0),
                    "sourceScore": 1.0,
                    "details": {"baselineRank": 1, "modelScore": 1.0},
                },
            ),
        ),
    ]

    result = _evaluate_ranker_row(rows, 0, (10.0, 20.0))

    assert result["candidatePoolSize"] == 2
    assert result["policyResults"]["combined_oracle"]["top1Hit@10px"] is True
    assert "leave_one_out_feature_prior_v0" in result["policyResults"]


def test_generate_ranker_artifact_handles_missing_images_with_baseline():
    feedback = {
        "rows": [
            {
                "setId": "15",
                "side": "A",
                "evaluationTier": "easy_corpus",
                "imagePath": "/tmp/does-not-exist.jpg",
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

    document = generate_ranker_artifacts(feedback)
    summary = summarize_ranker_rows(document["rows"])
    report = render_report({**document, "summary": summary})

    assert document["probe"] == "vertex_candidate_ranker_v0"
    assert summary["rowCount"] == 1
    assert summary["policySummaries"]["baseline_model_ranked"]["top1HitCount@10px"] == 1
    assert "Vertex Candidate Ranker V0" in report
