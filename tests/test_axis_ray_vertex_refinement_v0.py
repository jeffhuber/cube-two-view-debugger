from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.axis_ray_vertex_refinement_v0 import (
    RefinementConfig,
    generate_axis_ray_vertex_refinement_summary,
)


def test_axis_ray_refinement_improves_synthetic_trihedral(tmp_path: Path):
    image_path = tmp_path / "trihedral.png"
    image = Image.new("RGB", (240, 240), "white")
    draw = ImageDraw.Draw(image)
    vertex = (120.0, 120.0)
    for endpoint in ((40.0, 120.0), (120.0, 210.0), (205.0, 72.0)):
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
                        "humanAxisEndpoints": [[40.0, 120.0], [120.0, 210.0], [205.0, 72.0]],
                        "currentModel": {
                            "status": "ok",
                            "vertexPoint": [150.0, 120.0],
                            "axes": [
                                {"status": "ok", "vector": [-80.0, 0.0]},
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

    document = generate_axis_ray_vertex_refinement_summary(
        feedback_path=feedback_path,
        config=RefinementConfig(search_radius_px=50, search_step_px=5, ray_max_px=70),
    )

    row = document["rows"][0]
    assert row["evaluationStatus"] == "ok"
    assert row["baselineVertexErrorPx"] == 30.0
    assert row["modelAxisRefinedVertexErrorPx"] <= 8.0
    assert row["modelAxisImprovementPx"] >= 22.0
    assert document["summary"]["modelAxisRefinedStrictCount"] == 1


def test_axis_ray_refinement_reports_missing_image(tmp_path: Path):
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
                        "humanVertexPoint": [50.0, 50.0],
                        "humanAxisEndpoints": [[20.0, 50.0], [50.0, 90.0], [90.0, 20.0]],
                        "currentModel": {
                            "status": "ok",
                            "vertexPoint": [50.0, 50.0],
                            "axes": [
                                {"status": "ok", "vector": [-30.0, 0.0]},
                                {"status": "ok", "vector": [0.0, 40.0]},
                                {"status": "ok", "vector": [40.0, -30.0]},
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    document = generate_axis_ray_vertex_refinement_summary(feedback_path=feedback_path)

    assert document["rows"][0]["evaluationStatus"] == "missing_image"
    assert document["summary"]["evaluatedRowCount"] == 0


def test_darkness_objective_prefers_intersection_over_bright_background():
    from tools.axis_ray_vertex_refinement_v0 import refine_vertex

    darkness = np.zeros((120, 120), dtype=np.float32)
    darkness[60, 20:101] = 1.0
    darkness[60:101, 60] = 1.0
    for idx in range(0, 41):
        darkness[60 - idx, 60 + idx] = 1.0

    result = refine_vertex(
        darkness,
        start=(80.0, 60.0),
        axes=[(-1.0, 0.0), (0.0, 1.0), (1.0, -1.0)],
        config=RefinementConfig(search_radius_px=30, search_step_px=2, ray_max_px=40),
    )

    assert abs(result["point"][0] - 60.0) <= 4.0
    assert abs(result["point"][1] - 60.0) <= 4.0
