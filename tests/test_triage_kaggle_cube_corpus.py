from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from tools import triage_kaggle_cube_corpus as triage


def _write_image(path: Path, *, variant: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (400, 300), (224, 224, 218))
    draw = ImageDraw.Draw(image)
    if variant == "framed_cube":
        draw.polygon([(150, 95), (220, 70), (290, 102), (220, 132)], fill=(240, 220, 40))
        draw.polygon([(150, 95), (220, 132), (220, 215), (150, 175)], fill=(30, 90, 220))
        draw.polygon([(220, 132), (290, 102), (290, 185), (220, 215)], fill=(220, 45, 45))
    elif variant == "cropped_close":
        draw.rectangle((0, 20, 385, 285), fill=(25, 90, 220))
        draw.rectangle((140, 40, 390, 230), fill=(240, 230, 35))
    elif variant == "hand_occluded":
        draw.rectangle((170, 105, 250, 190), fill=(30, 90, 220))
        draw.rectangle((95, 80, 210, 220), fill=(205, 140, 92))
    elif variant == "plain_negative":
        draw.rectangle((190, 140, 194, 144), fill=(40, 80, 220))
    else:  # pragma: no cover - test helper guard
        raise ValueError(variant)
    image.save(path, "JPEG", quality=90)


def test_parse_filename_timestamp():
    parsed = triage.parse_filename_timestamp(Path("IMG_20260530_120102.jpg"))

    assert parsed == datetime(2026, 5, 30, 12, 1, 2, tzinfo=timezone.utc)


def test_build_manifest_buckets_and_writes_contact_sheets(tmp_path):
    corpus = tmp_path / "kaggle-cube-data"
    _write_image(corpus / "solved" / "IMG_20260530_120000.jpg", variant="framed_cube")
    _write_image(corpus / "solved" / "IMG_20260530_120005.jpg", variant="plain_negative")
    _write_image(corpus / "unsolved" / "IMG_20260530_121000.jpg", variant="cropped_close")
    _write_image(corpus / "unsolved" / "IMG_20260530_121006.jpg", variant="hand_occluded")
    _write_image(corpus / "_triage" / "contact_sheets" / "ignored.jpg", variant="framed_cube")
    output_dir = tmp_path / "triage"

    manifest = triage.build_manifest(
        corpus,
        output_dir=output_dir,
        analyze_limit_per_label=0,
        burst_gap_seconds=12,
        sheet_size=4,
        seed=1,
    )
    triage.write_manifest(output_dir / "manifest.json", manifest)

    assert manifest["schema"] == "ctvd.kaggleCubeCorpusTriage.v1"
    assert manifest["summary"]["totalImages"] == 4
    assert manifest["summary"]["labels"] == {"solved": 2, "unsolved": 2}
    assert manifest["summary"]["bucketCounts"]["three_face_candidate"] >= 1
    assert manifest["summary"]["bucketCounts"]["cropped_close"] >= 1
    assert manifest["summary"]["bucketCounts"]["hand_occluded"] >= 1

    by_path = {entry["relativePath"]: entry for entry in manifest["images"]}
    assert "three_face_candidate" in by_path["solved/IMG_20260530_120000.jpg"]["buckets"]
    assert "cropped_close" in by_path["unsolved/IMG_20260530_121000.jpg"]["buckets"]
    assert "hand_occluded" in by_path["unsolved/IMG_20260530_121006.jpg"]["buckets"]
    assert by_path["solved/IMG_20260530_120000.jpg"]["burstId"] == "solved-0001"
    assert by_path["solved/IMG_20260530_120005.jpg"]["burstSize"] == 2

    for relative_sheet in manifest["contactSheets"].values():
        assert (output_dir / relative_sheet).exists()

    saved_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert saved_manifest["summary"]["totalImages"] == 4


def test_main_writes_manifest(tmp_path, capsys):
    corpus = tmp_path / "kaggle-cube-data"
    _write_image(corpus / "solved" / "IMG_20260530_120000.jpg", variant="framed_cube")
    output_dir = tmp_path / "triage"

    rc = triage.main(
        [
            "--input-root",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--analyze-limit-per-label",
            "0",
            "--sheet-size",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "wrote" in captured.out
    assert (output_dir / "manifest.json").exists()
