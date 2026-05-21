from __future__ import annotations

import json
from pathlib import Path

from tools.expanded_vertex_localizer_v0 import (
    DEFAULT_SUMMARY,
    build_expanded_feedback,
    render_report,
)


def test_expanded_feedback_keeps_labeled_rows_and_prefixes_keys(tmp_path: Path):
    canonical = tmp_path / "canonical.json"
    active = tmp_path / "active.json"
    canonical.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "key": "1_A",
                        "status": "labeled",
                        "humanVertexPoint": [10.0, 20.0],
                        "humanAxisEndpoints": [[20.0, 20.0], [10.0, 30.0], [30.0, 30.0]],
                    },
                    {
                        "key": "1_B",
                        "status": "unlabeled",
                        "humanVertexPoint": None,
                        "humanAxisEndpoints": [None, None, None],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    active.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "key": "2_A",
                        "status": "labeled",
                        "humanVertexPoint": [40.0, 50.0],
                        "humanAxisEndpoints": [[50.0, 50.0], [40.0, 60.0], [60.0, 60.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    document = build_expanded_feedback(
        canonical_feedback_path=canonical,
        active_feedback_path=active,
    )

    assert [row["key"] for row in document["rows"]] == ["canonical:1_A", "active:2_A"]
    assert document["rows"][0]["sourceFeedbackLane"] == "canonical"
    assert document["rows"][1]["sourceRowKey"] == "2_A"


def test_render_report_includes_no_wire_recommendation():
    report = render_report(
        {
            "feedbackSummary": {"rowCount": 2, "laneCounts": {"canonical": 1, "active": 1}},
            "benchmarks": [
                {
                    "name": "synthetic",
                    "summary": {
                        "evaluatedRowCount": 2,
                        "baselineStrictCount": 0,
                        "baselinePlausibleCount": 0,
                        "candidateOracleStrictCount": 2,
                        "candidateOraclePlausibleCount": 2,
                        "top1StrictCount": 1,
                        "top1PlausibleCount": 1,
                        "gatedStrictCount": 1,
                        "gatedPlausibleCount": 1,
                        "gatedAcceptedCount": 1,
                        "top1ImprovedRowCount": 1,
                        "top1WorsenedRowCount": 1,
                        "gatedImprovedRowCount": 1,
                        "gatedWorsenedRowCount": 1,
                        "meanBaselineVertexErrorPx": 100.0,
                        "meanCandidateOracleVertexErrorPx": 5.0,
                        "meanTop1VertexErrorPx": 40.0,
                        "meanGatedVertexErrorPx": 40.0,
                    },
                }
            ],
            "conclusion": {
                "bestOracleBenchmark": "synthetic",
                "bestOracleStrictCount": 2,
                "bestTop1Benchmark": "synthetic",
                "bestTop1StrictCount": 1,
                "bestGatedBenchmark": "synthetic",
                "bestGatedStrictCount": 1,
                "productionWiringRecommendation": "do_not_wire",
                "reason": "synthetic reason",
            },
        }
    )

    assert "Production wiring recommendation: `do_not_wire`" in report


def test_committed_expanded_vertex_localizer_summary_is_strict_json():
    text = DEFAULT_SUMMARY.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    parsed = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert parsed["probe"] == "expanded_vertex_localizer_v0"
