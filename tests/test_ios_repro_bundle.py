from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

from tools.ios_repro_bundle import (
    build_multipart_body,
    find_shared_pasteboard_bundles,
    select_image_pair,
    unpack_bundle,
)


def _bundle(tmp_path: Path, *, corrupt_bytes: bool = False) -> Path:
    image_a = b"\xff\xd8image-a\xff\xd9"
    image_b = b"\xff\xd8image-b\xff\xd9"
    payload = {
        "formatVersion": 1,
        "generatedAt": "2026-06-01T05:40:48Z",
        "status": "failed",
        "appVersion": "0.1.0(2)",
        "buildConfiguration": "Debug",
        "device": "iPhone",
        "system": "iOS 26.5",
        "requestedSizes": [1024],
        "displayedMessage": "Try another capture.",
        "attempts": [
            {
                "edgePx": 1024,
                "bytesUploaded": len(image_a) + len(image_b),
                "latencyMs": 1234,
                "totalMs": 2345,
                "status": "recognizer failed",
                "recognitionCategory": "reject_retake",
                "failedChecks": ["image_a_U_anchor_missing"],
                "detail": "Image A must contain the white/U center face.",
            }
        ],
        "images": [
            {
                "role": "imageA",
                "edgePx": 1024,
                "contentType": "image/jpeg",
                "bytes": len(image_a) + (1 if corrupt_bytes else 0),
                "base64": base64.b64encode(image_a).decode(),
            },
            {
                "role": "imageB",
                "edgePx": 1024,
                "contentType": "image/jpeg",
                "bytes": len(image_b),
                "base64": base64.b64encode(image_b).decode(),
            },
        ],
    }
    path = tmp_path / "cubesnap-repro.json"
    path.write_text(json.dumps(payload))
    return path


def test_unpack_bundle_decodes_images_and_writes_base64_free_manifest(tmp_path: Path):
    out_dir = tmp_path / "out"

    target, manifest, decoded = unpack_bundle(_bundle(tmp_path), out_dir=out_dir)

    assert target == out_dir
    assert (out_dir / "bundle.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "summary.md").exists()
    assert [image.role for image in decoded] == ["imageA", "imageB"]
    assert (out_dir / "images" / "imageA_1024.jpg").read_bytes() == b"\xff\xd8image-a\xff\xd9"
    assert (out_dir / "images" / "imageB_1024.jpg").read_bytes() == b"\xff\xd8image-b\xff\xd9"
    assert manifest["manifestSchema"] == "ctvd.iosReproBundleManifest.v1"
    assert "base64" not in manifest["images"][0]
    assert manifest["images"][0]["path"] == "images/imageA_1024.jpg"
    assert len(manifest["images"][0]["sha256"]) == 64
    summary = (out_dir / "summary.md").read_text()
    assert "image_a_U_anchor_missing" in summary
    assert "Image A must contain the white/U center face." in summary


def test_unpack_bundle_defaults_to_durable_corpus_root(tmp_path: Path):
    target, _manifest, _decoded = unpack_bundle(_bundle(tmp_path), out_root=tmp_path / "corpus")

    assert target == tmp_path / "corpus" / "cubesnap-repro"
    assert (target / "bundle.json").exists()


def test_unpack_bundle_collapses_dot_only_filename_stem(tmp_path: Path):
    bundle = _bundle(tmp_path)
    dot_bundle = tmp_path / "...json"
    dot_bundle.write_text(bundle.read_text())

    target, _manifest, _decoded = unpack_bundle(dot_bundle, out_root=tmp_path / "corpus")

    assert target == tmp_path / "corpus" / "ios-repro-bundle"
    assert target.is_relative_to(tmp_path / "corpus")


def test_unpack_bundle_rejects_image_byte_mismatch(tmp_path: Path):
    with pytest.raises(ValueError, match="decoded to .* expected"):
        unpack_bundle(_bundle(tmp_path, corrupt_bytes=True), out_dir=tmp_path / "out")


def test_select_image_pair_requires_matching_roles(tmp_path: Path):
    _, _, decoded = unpack_bundle(_bundle(tmp_path), out_dir=tmp_path / "out")

    image_a, image_b = select_image_pair(decoded, edge_px=1024)

    assert image_a.role == "imageA"
    assert image_b.role == "imageB"
    with pytest.raises(ValueError, match="does not contain"):
        select_image_pair(decoded, edge_px=512)


def test_find_shared_pasteboard_bundles_discovers_recent_repro_json(tmp_path: Path):
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old = old_dir / "cubesnap-repro-old.json"
    new = new_dir / "cubesnap-repro-new.json"
    unrelated = new_dir / "not-a-repro.json"
    old.write_text("{}")
    new.write_text("{}")
    unrelated.write_text("{}")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    assert find_shared_pasteboard_bundles(tmp_path) == [new, old]


def test_build_multipart_body_matches_ios_metadata_shape():
    body, content_type = build_multipart_body(
        fields={
            "clientSource": "cube-snap-ios",
            "clientEntryPoint": "ios-phase0",
            "clientAppVersion": "0.1.0(2)",
            "clientAttemptIndex": "0",
            "clientAttemptTotal": "1",
            "clientAttemptOrder": "imageA-imageB",
        },
        files={
            "imageA": ("imageA_1024.jpg", b"a", "image/jpeg"),
            "imageB": ("imageB_1024.jpg", b"b", "image/jpeg"),
        },
        boundary="BoundaryForTest",
    )

    assert content_type == "multipart/form-data; boundary=BoundaryForTest"
    assert b'name="clientSource"\r\n\r\ncube-snap-ios' in body
    assert b'name="clientEntryPoint"\r\n\r\nios-phase0' in body
    assert b'name="imageA"; filename="imageA_1024.jpg"' in body
    assert b"Content-Type: image/jpeg\r\n\r\na\r\n" in body
    assert body.endswith(b"--BoundaryForTest--\r\n")


def test_build_multipart_body_sanitizes_header_parameters():
    body, _content_type = build_multipart_body(
        fields={'client"\r\nInjected: bad': "value"},
        files={'imageA\r\nBad: x': ('evil"\r\nX-Bad: yes.jpg', b"a", "image/jpeg\r\nX-Bad: yes")},
        boundary="BoundaryForTest",
    )

    assert b"\r\nInjected:" not in body
    assert b"\r\nX-Bad:" not in body
    assert b'name="client_Injected: bad"' in body
    assert b'name="imageABad: x"; filename="evil_X-Bad: yes.jpg"' in body
