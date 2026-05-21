from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.trihedral_junction_extraction_v0 import (
    DEFAULT_SUMMARY,
    JunctionConfig,
    extract_junction,
    generate_trihedral_junction_extraction_summary,
)


def test_line_junction_extraction_recovers_synthetic_intersection():
    darkness = np.zeros((220, 220), dtype=np.float32)
    vertex = (110.0, 110.0)
    for x in range(35, 186):
        darkness[110, x] = 1.0
    for y in range(35, 186):
        darkness[y, 110] = 1.0
    for offset in range(-65, 66):
        darkness[110 + offset, 110 + offset] = 1.0

    result = extract_junction(
        darkness,
        start=(136.0, 104.0),
        axes=[(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        config=JunctionConfig(
            angle_search_deg=(-4.0, 0.0, 4.0),
            offset_radius_px=55,
            offset_step_px=5,
            line_extent_px=95,
            line_sample_count=39,
            min_line_score=0.25,
        ),
    )

    assert result["point"] is not None
    assert abs(result["point"][0] - vertex[0]) <= 5.0
    assert abs(result["point"][1] - vertex[1]) <= 5.0
    assert result["diagnostics"]["intersectionSpreadPx"] <= 8.0


def test_trihedral_junction_summary_handles_synthetic_feedback(tmp_path: Path):
    image_path = tmp_path / "trihedral.png"
    image = Image.new("RGB", (240, 240), "white")
    draw = ImageDraw.Draw(image)
    vertex = (120.0, 120.0)
    for endpoint in ((30.0, 120.0), (120.0, 210.0), (205.0, 205.0)):
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
                        "humanAxisEndpoints": [[30.0, 120.0], [120.0, 210.0], [205.0, 205.0]],
                        "currentModel": {
                            "status": "ok",
                            "vertexPoint": [145.0, 118.0],
                            "axes": [
                                {"status": "ok", "vector": [-90.0, 0.0]},
                                {"status": "ok", "vector": [0.0, 90.0]},
                                {"status": "ok", "vector": [85.0, 85.0]},
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    document = generate_trihedral_junction_extraction_summary(
        feedback_path=feedback_path,
        config=JunctionConfig(
            angle_search_deg=(-4.0, 0.0, 4.0),
            offset_radius_px=55,
            offset_step_px=5,
            line_extent_px=85,
            line_sample_count=37,
            min_line_score=0.30,
            min_line_contrast=-0.05,
        ),
    )

    row = document["rows"][0]
    assert row["evaluationStatus"] == "ok"
    assert row["modelJunctionVertexErrorPx"] <= 12.0
    assert row["modelJunctionGatedAccepted"]
    assert document["summary"]["modelJunctionGatedStrictCount"] == 1


def test_trihedral_junction_summary_reports_missing_image(tmp_path: Path):
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

    document = generate_trihedral_junction_extraction_summary(feedback_path=feedback_path)

    assert document["rows"][0]["evaluationStatus"] == "missing_image"
    assert document["summary"]["evaluatedRowCount"] == 0


def test_committed_trihedral_junction_summary_is_strict_json():
    text = DEFAULT_SUMMARY.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    parsed = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert parsed["probe"] == "trihedral_junction_extraction_v0"
