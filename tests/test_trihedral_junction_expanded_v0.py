from __future__ import annotations

import json
from pathlib import Path

from tools.trihedral_junction_expanded_v0 import (
    DEFAULT_SUMMARY,
    generate_trihedral_junction_expanded_summary,
    render_report,
)


def test_expanded_junction_uses_labeled_rows_and_reports_lanes(tmp_path: Path):
    canonical = tmp_path / "canonical.json"
    active = tmp_path / "active.json"
    image_path = tmp_path / "missing.png"
    row_template = {
        "setId": "synthetic",
        "side": "A",
        "imagePath": str(image_path),
        "status": "labeled",
        "humanVertexPoint": [60.0, 60.0],
        "humanAxisEndpoints": [[30.0, 60.0], [60.0, 95.0], [95.0, 95.0]],
        "currentModel": {
            "status": "ok",
            "vertexPoint": [60.0, 60.0],
            "axes": [
                {"status": "ok", "vector": [-30.0, 0.0]},
                {"status": "ok", "vector": [0.0, 35.0]},
                {"status": "ok", "vector": [35.0, 35.0]},
            ],
        },
    }
    canonical.write_text(
        json.dumps({"rows": [{**row_template, "key": "1_A"}]}),
        encoding="utf-8",
    )
    active.write_text(
        json.dumps({"rows": [{**row_template, "key": "2_A"}]}),
        encoding="utf-8",
    )

    document = generate_trihedral_junction_expanded_summary(
        canonical_feedback_path=canonical,
        active_feedback_path=active,
    )

    assert document["feedbackSummary"]["rowCount"] == 2
    assert document["feedbackSummary"]["laneCounts"] == {"active": 1, "canonical": 1}
    assert [row["key"] for row in document["rows"]] == ["canonical:1_A", "active:2_A"]
    assert document["summary"]["evaluatedRowCount"] == 0
    assert document["rows"][0]["evaluationStatus"] == "missing_image"


def test_expanded_junction_report_includes_do_not_wire_recommendation():
    report = render_report(
        {
            "feedbackSummary": {"rowCount": 2, "laneCounts": {"canonical": 1, "active": 1}},
            "summary": {
                "evaluatedRowCount": 2,
                "axisGoodRowCount": 1,
                "axisBlockedRowCount": 1,
                "baselineStrictCount": 1,
                "baselinePlausibleCount": 1,
                "modelJunctionStrictCount": 0,
                "modelJunctionPlausibleCount": 0,
                "modelJunctionGatedStrictCount": 0,
                "modelJunctionGatedPlausibleCount": 0,
                "humanAxisOracleStrictCount": 0,
                "humanAxisOraclePlausibleCount": 0,
                "humanAxisOracleGatedStrictCount": 0,
                "humanAxisOracleGatedPlausibleCount": 0,
                "axisGoodBaselineStrictCount": 1,
                "axisGoodModelGatedStrictCount": 0,
                "modelJunctionGatedAcceptedCount": 1,
                "modelJunctionGatedImprovedRowCount": 0,
                "modelJunctionGatedWorsenedRowCount": 1,
                "humanAxisOracleGatedImprovedRowCount": 0,
                "humanAxisOracleGatedWorsenedRowCount": 1,
                "meanBaselineVertexErrorPx": 10.0,
                "meanModelJunctionGatedVertexErrorPx": 20.0,
                "meanHumanAxisOracleGatedVertexErrorPx": 20.0,
                "medianBaselineVertexErrorPx": 10.0,
                "medianModelJunctionGatedVertexErrorPx": 20.0,
                "medianHumanAxisOracleGatedVertexErrorPx": 20.0,
                "modelJunctionThresholdSweep": {
                    "bestNonEmptyLowWorsen": None,
                    "bestNonEmptyAtMostTwoWorsens": None,
                },
            },
            "conclusion": {
                "productionWiringRecommendation": "do_not_wire",
                "reason": "synthetic reason",
            },
            "rows": [],
        }
    )

    assert "Production wiring recommendation: `do_not_wire`" in report
    assert "strong negative" in report


def test_committed_trihedral_junction_expanded_summary_is_strict_json():
    text = DEFAULT_SUMMARY.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    parsed = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert parsed["probe"] == "trihedral_junction_expanded_v0"
