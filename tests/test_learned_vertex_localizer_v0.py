from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from tools.learned_vertex_localizer_v0 import (
    DEFAULT_SUMMARY,
    LearnedVertexConfig,
    generate_learned_vertex_localizer_summary,
)


def test_learned_vertex_localizer_handles_synthetic_feedback(tmp_path: Path):
    rows = []
    for idx, offset in enumerate((0, 12, 24)):
        image_path = tmp_path / f"trihedral_{idx}.png"
        image = Image.new("RGB", (260, 260), "white")
        draw = ImageDraw.Draw(image)
        vertex = (110.0 + offset * 0.25, 115.0 + offset * 0.1)
        for endpoint in (
            (45.0 + offset * 0.25, 115.0 + offset * 0.1),
            (110.0 + offset * 0.25, 210.0 + offset * 0.1),
            (195.0 + offset * 0.25, 195.0 + offset * 0.1),
        ):
            draw.line([vertex, endpoint], fill="black", width=9)
        image.save(image_path)
        model_vertex = [vertex[0] + 32.0, vertex[1] - 6.0]
        rows.append(
            {
                "key": f"synthetic_{idx}",
                "setId": "synthetic",
                "side": "A",
                "imagePath": str(image_path),
                "status": "labeled",
                "humanVertexPoint": list(vertex),
                "humanAxisEndpoints": [
                    [45.0 + offset * 0.25, 115.0 + offset * 0.1],
                    [110.0 + offset * 0.25, 210.0 + offset * 0.1],
                    [195.0 + offset * 0.25, 195.0 + offset * 0.1],
                ],
                "currentModel": {
                    "status": "ok",
                    "fitQuality": 0.8,
                    "debug": {"fitResidualRmsPx": 25.0},
                    "vertexPoint": model_vertex,
                    "axes": [
                        {"status": "ok", "vector": [-65.0, 0.0]},
                        {"status": "ok", "vector": [0.0, 95.0]},
                        {"status": "ok", "vector": [85.0, 80.0]},
                    ],
                },
            }
        )

    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    document = generate_learned_vertex_localizer_summary(
        feedback_path=feedback_path,
        config=LearnedVertexConfig(search_radius_px=60, search_step_px=8),
    )

    assert document["summary"]["evaluatedRowCount"] == 3
    assert document["summary"]["candidateOracleStrictCount"] == 3
    assert document["summary"]["learnedTop1StrictCount"] >= 1


def test_learned_vertex_localizer_reports_missing_image(tmp_path: Path):
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

    document = generate_learned_vertex_localizer_summary(feedback_path=feedback_path)

    assert document["rows"][0]["evaluationStatus"] == "missing_image"
    assert document["summary"]["evaluatedRowCount"] == 0


def test_committed_learned_vertex_localizer_summary_is_strict_json():
    text = DEFAULT_SUMMARY.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    parsed = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert parsed["probe"] == "learned_vertex_localizer_v0"
