from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from tools.learned_vertex_localizer_v0 import LearnedVertexConfig
from tools.raw_patch_vertex_localizer_v0 import (
    DEFAULT_SUMMARY,
    RawPatchVertexConfig,
    generate_raw_patch_vertex_localizer_summary,
    render_report,
)


def test_raw_patch_vertex_localizer_handles_synthetic_feedback(tmp_path: Path):
    rows = []
    for idx, offset in enumerate((0, 14, 28, 42)):
        image_path = tmp_path / f"trihedral_{idx}.png"
        image = Image.new("RGB", (280, 280), "white")
        draw = ImageDraw.Draw(image)
        vertex = (118.0 + offset * 0.2, 122.0 + offset * 0.1)
        endpoints = [
            [45.0 + offset * 0.2, 122.0 + offset * 0.1],
            [118.0 + offset * 0.2, 224.0 + offset * 0.1],
            [210.0 + offset * 0.2, 204.0 + offset * 0.1],
        ]
        for endpoint in endpoints:
            draw.line([vertex, tuple(endpoint)], fill="black", width=9)
        image.save(image_path)
        rows.append(
            {
                "key": f"synthetic_{idx}",
                "setId": "synthetic",
                "side": "A",
                "imagePath": str(image_path),
                "status": "labeled",
                "humanVertexPoint": list(vertex),
                "humanAxisEndpoints": endpoints,
                "currentModel": {
                    "status": "ok",
                    "fitQuality": 0.8,
                    "debug": {"fitResidualRmsPx": 25.0},
                    "vertexPoint": [vertex[0] + 34.0, vertex[1] - 6.0],
                    "axes": [
                        {"status": "ok", "vector": [-73.0, 0.0]},
                        {"status": "ok", "vector": [0.0, 102.0]},
                        {"status": "ok", "vector": [92.0, 82.0]},
                    ],
                },
            }
        )

    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    document = generate_raw_patch_vertex_localizer_summary(
        feedback_path=feedback_path,
        config=RawPatchVertexConfig(
            patch_radius_px=24.0,
            patch_size=7,
            candidate_config=LearnedVertexConfig(search_radius_px=72, search_step_px=8),
        ),
    )

    assert document["summary"]["evaluatedRowCount"] == 4
    assert document["summary"]["candidateOracleStrictCount"] == 4
    assert document["rows"][0]["featureCount"] > 80
    assert document["summary"]["rawPatchTop1PlausibleCount"] >= 1


def test_raw_patch_render_report_includes_training_language():
    report = render_report(
        {
            "summary": {
                "rowCount": 2,
                "evaluatedRowCount": 2,
                "axisGoodRowCount": 1,
                "axisBlockedRowCount": 1,
                "baselineStrictCount": 0,
                "baselinePlausibleCount": 0,
                "candidateOracleStrictCount": 2,
                "candidateOraclePlausibleCount": 2,
                "rawPatchTop1StrictCount": 1,
                "rawPatchTop1PlausibleCount": 1,
                "rawPatchGatedStrictCount": 1,
                "rawPatchGatedPlausibleCount": 1,
                "axisGoodBaselineStrictCount": 0,
                "axisGoodRawPatchGatedStrictCount": 1,
                "rawPatchGatedAcceptedCount": 1,
                "rawPatchTop1ImprovedRowCount": 1,
                "rawPatchTop1WorsenedRowCount": 1,
                "rawPatchGatedImprovedRowCount": 1,
                "rawPatchGatedWorsenedRowCount": 1,
                "meanBaselineVertexErrorPx": 100.0,
                "meanCandidateOracleVertexErrorPx": 5.0,
                "meanRawPatchTop1VertexErrorPx": 40.0,
                "meanRawPatchGatedVertexErrorPx": 40.0,
                "medianBaselineVertexErrorPx": 100.0,
                "medianCandidateOracleVertexErrorPx": 5.0,
                "medianRawPatchTop1VertexErrorPx": 40.0,
                "medianRawPatchGatedVertexErrorPx": 40.0,
                "rawPatchThresholdSweep": {
                    "bestNonEmptyLowWorsen": None,
                    "bestNonEmptyZeroWorsen": None,
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
    assert "raw image-patch" in report


def test_committed_raw_patch_vertex_localizer_summary_is_strict_json():
    text = DEFAULT_SUMMARY.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    parsed = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert parsed["probe"] == "raw_patch_vertex_localizer_v0"
