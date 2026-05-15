from __future__ import annotations

import json
from types import SimpleNamespace

from PIL import Image

from tools.audit_recognition_pair import file_sha256
from tools.evaluate_geometry_labels import (
    geometry_metrics_for_analysis,
    image_candidates,
    polygon_iou,
    resolve_image_path,
    scaled_label_geometry,
    write_geometry_overlay,
)


def test_scaled_label_geometry_converts_browser_natural_to_processing_space():
    document = _label_document(width=1000, height=500)

    geometry = scaled_label_geometry(document, 500, 250)

    assert geometry["scaleX"] == 0.5
    assert geometry["scaleY"] == 0.5
    assert geometry["faceQuads"]["U"] == [
        (5.0, 10.0),
        (55.0, 10.0),
        (55.0, 60.0),
        (5.0, 60.0),
    ]
    assert geometry["cubeHull"] == [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 100.0),
        (0.0, 100.0),
    ]


def test_geometry_metrics_count_stickers_against_labeled_cube_and_faces():
    analysis = SimpleNamespace(
        width=200,
        height=200,
        roi=(0, 0, 200, 200),
        warnings=[],
        stickers=[
            _sticker(1, (20, 20), "U"),
            _sticker(2, (130, 130), "F"),
            _sticker(3, (180, 180), "B"),
        ],
        grids=[
            _grid(1, "U", 10, 10),
            _grid(2, "F", 40, 40),
            _grid(3, "R", 70, 70),
        ],
    )

    document = _label_document(width=200, height=200)
    document["labels"]["cubeHull"] = [
        {"x": 0, "y": 0},
        {"x": 150, "y": 0},
        {"x": 150, "y": 150},
        {"x": 0, "y": 150},
    ]

    metrics = geometry_metrics_for_analysis(
        document,
        analysis,
        image_path=None,
        label_path=None,
    )

    sticker_metrics = metrics["metrics"]["stickers"]
    assert sticker_metrics["detected"] == 3
    assert sticker_metrics["insideLabeledCubeHull"] == 2
    assert sticker_metrics["outsideLabeledCubeHull"] == 1
    assert sticker_metrics["insideAnyFaceQuad"] == 1
    assert sticker_metrics["outsideLabeledCubeByDetectedFace"] == {"B": 1}
    assert metrics["metrics"]["faceCoverage"]["U"]["detectedCenters"] == 1
    assert metrics["metrics"]["faceCoverage"]["U"]["coverageVsNine"] == 0.1111
    assert metrics["metrics"]["roi"]["containsAllLabelHullVertices"]
    assert metrics["imageSha256"]["matches"]


def test_polygon_iou_uses_rasterized_overlap():
    first = [(0, 0), (10, 0), (10, 10), (0, 10)]
    second = [(5, 0), (15, 0), (15, 10), (5, 10)]

    assert polygon_iou(first, second, 20, 20) == 0.375


def test_resolve_image_path_prefers_manifest_candidate_matching_label_sha(tmp_path):
    image_path = tmp_path / "Set 999 - A.jpg"
    Image.new("RGB", (20, 20), "white").save(image_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "setId": "999",
                        "imageAPath": str(image_path),
                        "imageBPath": str(tmp_path / "missing.jpg"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    document = _label_document(width=20, height=20)
    document["setId"] = "Set 999"
    document["imageSide"] = "A"
    document["image"]["name"] = image_path.name
    document["image"]["sha256"] = file_sha256(str(image_path))

    resolved = resolve_image_path(document, tmp_path / "label.json", [manifest], [])

    assert resolved == image_path
    assert image_candidates(document, tmp_path / "label.json", [manifest], []) == [image_path]


def test_write_geometry_overlay_uses_processing_image_size(tmp_path):
    image_path = tmp_path / "photo.jpg"
    output_path = tmp_path / "overlay.png"
    Image.new("RGB", (3024, 4032), "white").save(image_path)
    metrics = {
        "analysis": {"roi": [0, 0, 862, 1150]},
        "labels": {
            "cubeHull": [[20, 20], [842, 20], [842, 1130], [20, 1130]],
            "faceQuads": {"U": [[100, 100], [300, 100], [300, 300], [100, 300]]},
        },
        "stickers": [
            {
                "center": [200, 200],
                "source": "component",
                "insideAnyFaceQuad": True,
                "insideLabeledCubeHull": True,
            }
        ],
    }

    write_geometry_overlay(image_path, metrics, output_path)

    with Image.open(output_path) as overlay:
        assert overlay.size == (862, 1150)


def _label_document(width=200, height=200):
    return {
        "schemaVersion": 1,
        "labelType": "cube_geometry",
        "coordinateSpace": "browser_image_natural",
        "setId": "Set 1",
        "imageSide": "A",
        "image": {
            "name": "Set 1 - A.jpg",
            "sha256": None,
            "width": width,
            "height": height,
        },
        "labels": {
            "faceQuads": {
                "U": [
                    {"x": 10, "y": 20},
                    {"x": 110, "y": 20},
                    {"x": 110, "y": 120},
                    {"x": 10, "y": 120},
                ]
            },
            "cubeHull": [
                {"x": 0, "y": 0},
                {"x": 200, "y": 0},
                {"x": 200, "y": 200},
                {"x": 0, "y": 200},
            ],
        },
    }


def _sticker(sticker_id, center, face, source="component"):
    return SimpleNamespace(
        id=sticker_id,
        center=center,
        source=source,
        rgb=(230, 230, 230),
        shape_angle=0.0,
        match=SimpleNamespace(face=face, color="white", confidence=0.9),
    )


def _grid(grid_id, face, x, y):
    stickers = [
        [_sticker(grid_id * 10 + row * 3 + col, (x + col * 10, y + row * 10), face) for col in range(3)]
        for row in range(3)
    ]
    return SimpleNamespace(
        id=grid_id,
        center_face=face,
        center_sticker=stickers[1][1],
        matched_count=9,
        fit_error=1.0,
        points=[[(x + col * 10, y + row * 10) for col in range(3)] for row in range(3)],
        stickers=stickers,
    )
