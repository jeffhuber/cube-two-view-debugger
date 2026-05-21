from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.ray_start_vertex_refinement_v0 import (
    RayStartConfig,
    generate_ray_start_vertex_refinement_summary,
    refine_vertex,
)


def test_ray_start_penalty_prefers_start_of_three_rays():
    darkness = np.zeros((180, 180), dtype=np.float32)
    vertex = (80, 80)
    for x in range(80, 150):
        darkness[80, x] = 1.0
    for y in range(80, 150):
        darkness[y, 80] = 1.0
    for offset in range(0, 60):
        darkness[80 - offset, 80 + offset] = 1.0

    result = refine_vertex(
        darkness,
        start=(105.0, 80.0),
        axes=[(1.0, 0.0), (0.0, 1.0), (1.0, -1.0)],
        config=RayStartConfig(search_radius_px=45, search_step_px=3, forward_max_px=55, backward_max_px=45),
    )

    assert abs(result["point"][0] - vertex[0]) <= 5.0
    assert abs(result["point"][1] - vertex[1]) <= 5.0
    assert result["components"]["meanStartness"] > 0.2


def test_ray_start_report_handles_synthetic_feedback(tmp_path: Path):
    image_path = tmp_path / "trihedral.png"
    image = Image.new("RGB", (240, 240), "white")
    draw = ImageDraw.Draw(image)
    vertex = (120.0, 120.0)
    for endpoint in ((200.0, 120.0), (120.0, 210.0), (205.0, 72.0)):
        draw.line([vertex, endpoint], fill="black", width=9)
    image.save(image_path)

    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "key": "synthetic_A",
                        "setId": "synthetic",
                        "side": "A",
                        "imagePath": str(image_path),
                        "status": "labeled",
                        "humanVertexPoint": list(vertex),
                        "humanAxisEndpoints": [[200.0, 120.0], [120.0, 210.0], [205.0, 72.0]],
                        "currentModel": {
                            "status": "ok",
                            "vertexPoint": [145.0, 120.0],
                            "axes": [
                                {"status": "ok", "vector": [80.0, 0.0]},
                                {"status": "ok", "vector": [0.0, 90.0]},
                                {"status": "ok", "vector": [85.0, -48.0]},
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    document = generate_ray_start_vertex_refinement_summary(
        feedback_path=feedback_path,
        config=RayStartConfig(search_radius_px=45, search_step_px=5, forward_max_px=70, backward_max_px=50),
    )

    row = document["rows"][0]
    assert row["evaluationStatus"] == "ok"
    assert row["modelAxisGatedAccepted"]
    assert row["modelAxisGatedVertexErrorPx"] <= 10.0
    assert document["summary"]["modelAxisGatedStrictCount"] == 1


def test_ray_start_gating_can_keep_baseline_when_score_gain_is_weak(tmp_path: Path):
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (120, 120), "white").save(image_path)
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "key": "blank_A",
                        "setId": "blank",
                        "side": "A",
                        "imagePath": str(image_path),
                        "status": "labeled",
                        "humanVertexPoint": [60.0, 60.0],
                        "humanAxisEndpoints": [[90.0, 60.0], [60.0, 90.0], [90.0, 30.0]],
                        "currentModel": {
                            "status": "ok",
                            "vertexPoint": [60.0, 60.0],
                            "axes": [
                                {"status": "ok", "vector": [30.0, 0.0]},
                                {"status": "ok", "vector": [0.0, 30.0]},
                                {"status": "ok", "vector": [30.0, -30.0]},
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    document = generate_ray_start_vertex_refinement_summary(
        feedback_path=feedback_path,
        config=RayStartConfig(search_radius_px=30, search_step_px=10, forward_max_px=35, backward_max_px=25),
    )

    row = document["rows"][0]
    assert row["modelAxisGatedAccepted"] is False
    assert row["modelAxisGatedVertexErrorPx"] == 0.0


def test_committed_ray_start_summary_is_strict_json():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ray_start_vertex_refinement_v0_summary.json"

    document = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )

    assert document["summary"]["rowCount"] == 28
    assert document["summary"]["modelAxisGatedStrictCount"] == 5
