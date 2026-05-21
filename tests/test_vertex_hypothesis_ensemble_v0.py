from __future__ import annotations

import json
from pathlib import Path

from tools.vertex_hypothesis_ensemble_v0 import (
    build_canonical_feedback,
    generate_vertex_hypothesis_ensemble,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "tests" / "fixtures" / "vertex_hypothesis_ensemble_v0_summary.json"


def test_easy_lane_does_not_use_label_oracle_best_candidate():
    source_probe = {
        "rows": [
            {
                "setId": "1",
                "side": "A",
                "evaluationStatus": "labeled",
                "humanVertexPoint": [10.0, 10.0],
                "sources": [
                    {
                        "source": "model_ranked",
                        "bestCandidate": {
                            "point": [10.0, 10.0],
                            "rank": 99,
                            "score": 0.0,
                        },
                        "topCandidates": [
                            {
                                "point": [90.0, 90.0],
                                "rank": 1,
                                "score": 1.0,
                                "details": {},
                            }
                        ],
                    }
                ],
            }
        ]
    }

    feedback = build_canonical_feedback(
        geometry_document={"rows": []},
        source_probe_document=source_probe,
    )
    row = feedback["lanes"]["easy_processing"]["rows"][0]

    assert row["candidateCount"] == 1
    assert row["oracleBestCandidate"]["point"] == [90.0, 90.0]
    assert row["oracleBestCandidate"]["status"] == "false_confident"


def test_generate_vertex_hypothesis_ensemble_with_minimal_fixtures(tmp_path: Path):
    geometry_fixture = tmp_path / "geometry.json"
    source_probe_fixture = tmp_path / "source_probe.json"
    geometry_fixture.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "key": "1_A",
                        "setId": "1",
                        "side": "A",
                        "trueVertex": [0.0, 0.0],
                        "bestSourceByVertexError": "sam3",
                        "bestSourceStatus": "split_ready_strict",
                        "sources": {
                            "rembg": {
                                "vertex": [40.0, 0.0],
                                "fitQuality": 0.5,
                                "fitResidualRmsPx": 40.0,
                                "status": "vertex_error_blocked",
                                "nondegenerate": True,
                            },
                            "sam3": {
                                "vertex": [10.0, 0.0],
                                "fitQuality": 0.9,
                                "fitResidualRmsPx": 10.0,
                                "status": "split_ready_strict",
                                "nondegenerate": True,
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source_probe_fixture.write_text(json.dumps({"rows": []}), encoding="utf-8")

    feedback, summary = generate_vertex_hypothesis_ensemble(
        geometry_fixture_path=geometry_fixture,
        source_probe_fixture_path=source_probe_fixture,
    )

    assert feedback["lanes"]["gcm_fullres"]["rows"][0]["candidateCount"] == 5
    assert summary["laneResults"]["gcm_fullres"]["rowCount"] == 1
    assert (
        summary["laneResults"]["gcm_fullres"]["policySummaries"]["source_priority_top1_v0"][
            "strictReadyCount"
        ]
        == 1
    )


def test_committed_summary_preserves_no_wiring_conclusion():
    document = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert document["conclusion"]["productionWiringRecommendation"] == "wait"
    assert document["laneResults"]["gcm_fullres"]["rowCount"] == 23
    assert document["laneResults"]["easy_processing"]["rowCount"] == 16

    gcm = document["laneResults"]["gcm_fullres"]
    easy = document["laneResults"]["easy_processing"]

    assert gcm["oracleSummary"]["plausibleReachableCount"] == 12
    assert gcm["policySummaries"]["agreement_cluster_v0"]["falseConfidentCount"] == 15
    assert gcm["policySummaries"]["strict_agreement_cluster_v0"]["falseConfidentCount"] == 16

    assert easy["oracleSummary"]["strictReachableCount"] == 7
    assert easy["policySummaries"]["agreement_cluster_v0"]["falseConfidentCount"] == 5
    assert easy["policySummaries"]["strict_agreement_cluster_v0"]["falseConfidentCount"] == 2
