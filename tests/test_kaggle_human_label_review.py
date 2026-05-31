from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from tools import kaggle_human_label_review as labels


def _manifest(tmp_path: Path) -> dict:
    return {
        "schema": "ctvd.kaggleCubeCuratedEval.v1",
        "sourceCorpusRoot": str(tmp_path / "corpus"),
        "images": [
            {
                "id": "kaggle-001",
                "relativePath": "solved/three.jpg",
                "category": "usable_three_face",
                "sourceLabel": "solved",
                "tags": ["usable_three_face", "three_face_candidate"],
            },
            {
                "id": "kaggle-002",
                "relativePath": "unsolved/crop.jpg",
                "category": "cropped_close_negative",
                "sourceLabel": "unsolved",
                "tags": ["cropped_close_negative", "cropped_close"],
            },
        ],
    }


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "curated_manifest.json"
    path.write_text(json.dumps(_manifest(tmp_path)))
    return path


def test_role_definitions_cover_all_role_choices():
    assert set(labels.ROLE_DEFINITIONS) == {
        "three_face_positive",
        "table_isometric",
        "single_face_negative",
        "hand_occluded",
        "cropped_close",
        "noncanonical_interesting",
        "junk",
    }
    markdown = labels.role_definitions_markdown()
    for role in labels.ROLE_DEFINITIONS:
        assert f"`{role}`" in markdown


def test_write_template_prefills_suggested_roles(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    labels_path = tmp_path / "human_labels.csv"

    rows = labels.write_template(manifest_path, labels_path)

    assert [row["id"] for row in rows] == ["kaggle-001", "kaggle-002"]
    assert rows[0]["role"] == "three_face_positive"
    assert rows[0]["visible_faces"] == "3+"
    assert rows[0]["expected_detector_result"] == "should_find_grid"
    assert rows[1]["role"] == "cropped_close"
    assert rows[1]["retake_reason"] == "cropped"
    with labels_path.open(newline="") as handle:
        assert csv.DictReader(handle).fieldnames == labels.LABEL_HEADERS


def test_write_template_preserves_existing_human_edits_by_relative_path(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    labels_path = tmp_path / "human_labels.csv"
    labels.write_label_csv(
        labels_path,
        [
            {
                "id": "old-ordinal-id",
                "relativePath": "solved/three.jpg",
                "keep": "yes",
                "role": "junk",
                "visible_faces": "0",
                "quality": "bad",
                "expected_detector_result": "should_reject",
                "retake_reason": "not_cube",
                "notes": "not actually useful",
            }
        ],
    )

    rows = labels.write_template(manifest_path, labels_path)

    assert rows[0]["id"] == "kaggle-001"
    assert rows[0]["relativePath"] == "solved/three.jpg"
    assert rows[0]["keep"] == "yes"
    assert rows[0]["role"] == "junk"
    assert rows[0]["notes"] == "not actually useful"
    assert rows[1]["role"] == "cropped_close"


def test_write_template_resets_labels_when_ordinal_id_path_changes(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    labels_path = tmp_path / "human_labels.csv"
    labels.write_label_csv(
        labels_path,
        [
            {
                "id": "kaggle-001",
                "relativePath": "old/path.jpg",
                "keep": "yes",
                "role": "junk",
                "visible_faces": "0",
                "quality": "bad",
                "expected_detector_result": "should_reject",
                "retake_reason": "not_cube",
                "notes": "do not carry this forward",
            }
        ],
    )

    rows = labels.write_template(manifest_path, labels_path)

    assert rows[0]["id"] == "kaggle-001"
    assert rows[0]["relativePath"] == "solved/three.jpg"
    assert rows[0]["keep"] == "maybe"
    assert rows[0]["role"] == "three_face_positive"
    assert rows[0]["notes"] == ""


def test_validate_label_row_rejects_unknown_role():
    row = labels.default_label_row(
        {
            "id": "kaggle-001",
            "relativePath": "solved/three.jpg",
            "category": "usable_three_face",
        }
    )
    row["role"] = "almost_good"

    with pytest.raises(ValueError, match="invalid role"):
        labels.validate_label_row(row)


def test_server_config_writes_template_and_embeds_review_payload(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    labels_path = tmp_path / "human_labels.csv"

    config = labels.server_config(manifest_path, labels_path)
    html = labels.build_review_html(config)

    assert labels_path.exists()
    assert "Kaggle Cube Human Label Review" in html
    assert "three_face_positive" in html
    assert "/image/" in html
    assert "&quot;" not in html
    assert "meta.innerHTML" not in html
    assert "div.innerHTML" not in html
    assert "current.visible_faces = expected[2]" in html
    payload = _payload_from_html(html)
    assert list(payload["roleDefinitions"]) == list(labels.ROLE_DEFINITIONS)


def test_server_config_refreshes_rows_from_csv(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    labels_path = tmp_path / "human_labels.csv"
    config = labels.server_config(manifest_path, labels_path)
    rows = labels.read_label_csv(labels_path)
    rows[0]["keep"] = "yes"
    rows[0]["notes"] = "keeper"
    labels.write_label_csv(labels_path, rows)

    latest = labels.config_with_latest_rows(config)

    assert latest["rows"][0]["keep"] == "yes"
    assert latest["rows"][0]["notes"] == "keeper"


def test_resolve_manifest_image_path_blocks_traversal(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    assert labels.resolve_manifest_image_path(corpus, "solved/three.jpg") == (corpus / "solved" / "three.jpg").resolve()
    with pytest.raises(ValueError, match="escapes corpus root"):
        labels.resolve_manifest_image_path(corpus, "../secret.jpg")


def test_label_review_server_has_csv_lock(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    labels_path = tmp_path / "human_labels.csv"
    config = labels.server_config(manifest_path, labels_path)
    server = labels.LabelReviewServer(("127.0.0.1", 0), config)
    try:
        assert server.csv_lock
    finally:
        server.server_close()


def test_validate_label_payload_rejects_stale_or_truncated_rows(tmp_path):
    manifest = _manifest(tmp_path)
    entries = manifest["images"]
    rows = labels.starter_label_rows(manifest)

    labels.validate_label_payload(rows, entries)
    with pytest.raises(ValueError, match="row count"):
        labels.validate_label_payload(rows[:1], entries)
    stale = [dict(row) for row in rows]
    stale[0]["relativePath"] = "old/path.jpg"
    with pytest.raises(ValueError, match="does not match current manifest"):
        labels.validate_label_payload(stale, entries)


def _payload_from_html(body: str) -> dict:
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', body, re.S)
    assert match
    return json.loads(match.group(1))
