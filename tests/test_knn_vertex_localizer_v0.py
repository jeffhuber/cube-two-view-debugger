from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from tools.knn_vertex_localizer_v0 import (
    DEFAULT_SUMMARY,
    KnnVertexConfig,
    generate_knn_vertex_localizer_summary,
)
from tools.learned_vertex_localizer_v0 import LearnedVertexConfig


def test_knn_vertex_localizer_handles_synthetic_feedback(tmp_path: Path):
    rows = []
    for idx, offset in enumerate((0, 14, 28, 42)):
        image_path = tmp_path / f"trihedral_{idx}.png"
        image = Image.new("RGB", (260, 260), "white")
        draw = ImageDraw.Draw(image)
        vertex = (115.0 + offset * 0.2, 118.0 + offset * 0.1)
        endpoints = [
            [45.0 + offset * 0.2, 118.0 + offset * 0.1],
            [115.0 + offset * 0.2, 214.0 + offset * 0.1],
            [201.0 + offset * 0.2, 198.0 + offset * 0.1],
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
                    "vertexPoint": [vertex[0] + 32.0, vertex[1] - 6.0],
                    "axes": [
                        {"status": "ok", "vector": [-70.0, 0.0]},
                        {"status": "ok", "vector": [0.0, 96.0]},
                        {"status": "ok", "vector": [86.0, 80.0]},
                    ],
                },
            }
        )

    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    document = generate_knn_vertex_localizer_summary(
        feedback_path=feedback_path,
        config=KnnVertexConfig(
            positive_error_px=30.0,
            negative_error_px=55.0,
            candidate_config=LearnedVertexConfig(search_radius_px=64, search_step_px=8),
        ),
    )

    assert document["summary"]["evaluatedRowCount"] == 4
    assert document["summary"]["candidateOracleStrictCount"] == 4
    assert document["summary"]["knnTop1PlausibleCount"] >= 1


def test_knn_vertex_localizer_reports_missing_image(tmp_path: Path):
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "key": "missing_A",
                        "setId": "missing",
                        "side": "A",
                        "imagePath": str(tmp_path / "missing.png"),
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
                ]
            }
        ),
        encoding="utf-8",
    )

    document = generate_knn_vertex_localizer_summary(feedback_path=feedback_path)

    assert document["rows"][0]["evaluationStatus"] == "missing_image"
    assert document["summary"]["evaluatedRowCount"] == 0


def test_committed_knn_vertex_localizer_summary_is_strict_json():
    text = DEFAULT_SUMMARY.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    parsed = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert parsed["probe"] == "knn_vertex_localizer_v0"
