from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from tools import kaggle_curated_eval as curated
from tools import triage_kaggle_cube_corpus as triage


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 240), (230, 230, 225))
    draw = ImageDraw.Draw(image)
    draw.polygon([(100, 70), (160, 45), (220, 70), (160, 100)], fill=(240, 220, 40))
    draw.polygon([(100, 70), (160, 100), (160, 170), (100, 140)], fill=(30, 90, 220))
    draw.polygon([(160, 100), (220, 70), (220, 140), (160, 170)], fill=(220, 45, 45))
    image.save(path, "JPEG", quality=90)


def _triage_entry(
    rel_path: str,
    *,
    buckets: list[str],
    scores: dict[str, float],
    label: str = "solved",
    burst_id: str | None = None,
) -> dict:
    return {
        "relativePath": rel_path,
        "label": label,
        "width": 320,
        "height": 240,
        "fileSize": 1234,
        "timestamp": None,
        "burstId": burst_id or rel_path,
        "burstSize": 1,
        "features": {"colorBBoxTouchesFrame": False},
        "bucketScores": {
            "three_face_candidate": 0.0,
            "single_face": 0.0,
            "hand_occluded": 0.0,
            "cropped_close": 0.0,
            "table_isometric": 0.0,
            "retake_negative": 0.0,
            **scores,
        },
        "bucketReasons": {},
        "buckets": buckets,
    }


def _triage_manifest(corpus_root: Path) -> dict:
    return {
        "schema": "ctvd.kaggleCubeCorpusTriage.v1",
        "sourceRoot": str(corpus_root),
        "images": [
            _triage_entry(
                "solved/three.jpg",
                buckets=["three_face_candidate", "single_face"],
                scores={"three_face_candidate": 0.72, "single_face": 0.58},
            ),
            _triage_entry(
                "solved/table.jpg",
                buckets=["three_face_candidate", "single_face", "table_isometric"],
                scores={"three_face_candidate": 0.72, "single_face": 0.58, "table_isometric": 0.70},
            ),
            _triage_entry(
                "solved/single.jpg",
                buckets=["three_face_candidate", "single_face", "retake_negative"],
                scores={"three_face_candidate": 0.72, "single_face": 0.72, "retake_negative": 0.70},
            ),
            _triage_entry(
                "solved/hand.jpg",
                buckets=["hand_occluded", "cropped_close", "retake_negative"],
                scores={"hand_occluded": 1.0, "cropped_close": 0.82, "retake_negative": 0.92},
            ),
            _triage_entry(
                "solved/crop.jpg",
                buckets=["cropped_close", "retake_negative"],
                scores={"cropped_close": 0.82, "retake_negative": 0.75},
            ),
            _triage_entry(
                "unsolved/interesting.jpg",
                buckets=["three_face_candidate"],
                scores={"three_face_candidate": 0.72},
                label="unsolved",
            ),
            _triage_entry(
                "unsolved/interesting2.jpg",
                buckets=["three_face_candidate"],
                scores={"three_face_candidate": 0.72},
                label="unsolved",
            ),
        ],
    }


def test_build_curated_manifest_writes_contact_sheets(tmp_path):
    corpus = tmp_path / "kaggle"
    for rel_path in [
        "solved/three.jpg",
        "solved/table.jpg",
        "solved/single.jpg",
        "solved/hand.jpg",
        "solved/crop.jpg",
        "unsolved/interesting.jpg",
        "unsolved/interesting2.jpg",
    ]:
        _write_image(corpus / rel_path)
    triage_path = tmp_path / "triage.json"
    triage = _triage_manifest(corpus)
    triage_path.write_text(json.dumps(triage))
    output_dir = tmp_path / "curated"

    manifest = curated.build_curated_manifest(
        triage,
        triage_manifest_path=triage_path,
        output_dir=output_dir,
        quotas={
            "usable_three_face": 1,
            "usable_table_isometric": 1,
            "single_face_negative": 1,
            "hand_occluded_negative": 1,
            "cropped_close_negative": 1,
            "noncanonical_interesting": 1,
        },
    )

    assert manifest["schema"] == "ctvd.kaggleCubeCuratedEval.v1"
    assert manifest["summary"]["totalImages"] == 6
    assert manifest["summary"]["categories"]["hand_occluded_negative"] == 1
    assert {entry["id"] for entry in manifest["images"]} == {f"kaggle-{index:03d}" for index in range(1, 7)}
    for relative_path in manifest["summary"]["contactSheets"].values():
        assert (output_dir / relative_path).exists()


def test_defaults_chain_from_triage_default_output():
    assert curated.DEFAULT_TRIAGE_MANIFEST == triage.DEFAULT_OUTPUT_DIR / "manifest.json"


def test_table_isometric_entries_are_reserved_for_table_quota(tmp_path):
    corpus = tmp_path / "kaggle"
    _write_image(corpus / "solved" / "three.jpg")
    _write_image(corpus / "solved" / "table.jpg")
    triage_manifest = {
        "schema": "ctvd.kaggleCubeCorpusTriage.v1",
        "sourceRoot": str(corpus),
        "images": [
            _triage_entry(
                "solved/three.jpg",
                buckets=["three_face_candidate"],
                scores={"three_face_candidate": 0.72},
            ),
            _triage_entry(
                "solved/table.jpg",
                buckets=["three_face_candidate", "table_isometric"],
                scores={"three_face_candidate": 0.72, "table_isometric": 0.70},
            ),
        ],
    }

    manifest = curated.build_curated_manifest(
        triage_manifest,
        triage_manifest_path=tmp_path / "triage.json",
        output_dir=tmp_path / "curated",
        quotas={
            "usable_three_face": 1,
            "usable_table_isometric": 1,
            "single_face_negative": 0,
            "hand_occluded_negative": 0,
            "cropped_close_negative": 0,
            "noncanonical_interesting": 0,
        },
    )

    by_category = {entry["category"]: entry["relativePath"] for entry in manifest["images"]}
    assert by_category == {
        "usable_three_face": "solved/three.jpg",
        "usable_table_isometric": "solved/table.jpg",
    }


def test_single_face_negative_requires_retake_signal(tmp_path):
    corpus = tmp_path / "kaggle"
    for rel_path in ["solved/positive.jpg", "solved/table.jpg", "solved/negative.jpg"]:
        _write_image(corpus / rel_path)
    triage_manifest = {
        "schema": "ctvd.kaggleCubeCorpusTriage.v1",
        "sourceRoot": str(corpus),
        "images": [
            _triage_entry(
                "solved/positive.jpg",
                buckets=["three_face_candidate", "single_face"],
                scores={"three_face_candidate": 0.72, "single_face": 0.72},
            ),
            _triage_entry(
                "solved/table.jpg",
                buckets=["single_face", "table_isometric", "retake_negative"],
                scores={"single_face": 0.72, "table_isometric": 0.70, "retake_negative": 0.70},
            ),
            _triage_entry(
                "solved/negative.jpg",
                buckets=["three_face_candidate", "single_face", "retake_negative"],
                scores={"three_face_candidate": 0.72, "single_face": 0.72, "retake_negative": 0.70},
            ),
        ],
    }

    manifest = curated.build_curated_manifest(
        triage_manifest,
        triage_manifest_path=tmp_path / "triage.json",
        output_dir=tmp_path / "curated",
        quotas={
            "usable_three_face": 0,
            "usable_table_isometric": 0,
            "single_face_negative": 1,
            "hand_occluded_negative": 0,
            "cropped_close_negative": 0,
            "noncanonical_interesting": 0,
        },
    )

    assert [(entry["category"], entry["relativePath"]) for entry in manifest["images"]] == [
        ("single_face_negative", "solved/negative.jpg")
    ]


def test_analyze_curated_manifest_summarizes_failures(tmp_path):
    manifest = {
        "schema": "ctvd.kaggleCubeCuratedEval.v1",
        "sourceCorpusRoot": str(tmp_path),
        "images": [
            {"id": "kaggle-001", "relativePath": "a.jpg", "category": "usable_three_face", "sourceLabel": "solved"},
            {"id": "kaggle-002", "relativePath": "b.jpg", "category": "cropped_close_negative", "sourceLabel": "unsolved"},
        ],
    }

    def fake_analyzer(path: Path) -> dict:
        if path.name == "a.jpg":
            return {"stickers": 27, "gridCount": 3, "gridCenterFaces": ["U", "F", "R"], "warnings": []}
        return {"stickers": 3, "gridCount": 0, "gridCenterFaces": [], "warnings": ["Few sticker candidates detected."]}

    report = curated.analyze_curated_manifest(manifest, analyzer_fn=fake_analyzer)

    assert report["schema"] == "ctvd.kaggleCubeCuratedAnalysis.v1"
    assert report["summary"]["totalImages"] == 2
    assert report["summary"]["threeGridImages"] == 1
    assert report["summary"]["failureStages"] == {"three_grids_clean": 1, "sticker_detection_low": 1}
    markdown = curated.markdown_report(report)
    assert "Kaggle Curated Eval Analysis" in markdown
    assert "`cropped_close_negative`" in markdown


def test_main_build_manifest(tmp_path, capsys):
    corpus = tmp_path / "kaggle"
    _write_image(corpus / "solved" / "three.jpg")
    triage = {
        **_triage_manifest(corpus),
        "images": [
            _triage_entry(
                "solved/three.jpg",
                buckets=["three_face_candidate", "single_face"],
                scores={"three_face_candidate": 0.72, "single_face": 0.58},
            )
        ],
    }
    triage_path = tmp_path / "triage.json"
    triage_path.write_text(json.dumps(triage))
    output_dir = tmp_path / "curated"

    rc = curated.main(
        [
            "build-manifest",
            "--triage-manifest",
            str(triage_path),
            "--output-dir",
            str(output_dir),
            "--quota",
            "usable_three_face=1",
            "--quota",
            "usable_table_isometric=0",
            "--quota",
            "single_face_negative=0",
            "--quota",
            "hand_occluded_negative=0",
            "--quota",
            "cropped_close_negative=0",
            "--quota",
            "noncanonical_interesting=0",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "wrote" in captured.out
    saved = json.loads((output_dir / "curated_manifest.json").read_text())
    assert saved["summary"]["totalImages"] == 1
    assert saved["summary"]["categories"]["usable_three_face"] == 1
    assert saved["summary"]["categories"]["single_face_negative"] == 0
