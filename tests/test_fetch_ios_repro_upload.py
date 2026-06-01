from __future__ import annotations

import base64
import json

import pytest

from tools import fetch_ios_repro_upload


def _bundle_bytes() -> bytes:
    image_a = b"\xff\xd8image-a\xff\xd9"
    image_b = b"\xff\xd8image-b\xff\xd9"
    return json.dumps(
        {
            "formatVersion": 1,
            "generatedAt": "2026-06-01T18:00:00Z",
            "status": "failed",
            "appVersion": "0.1.0(3)",
            "buildConfiguration": "Debug",
            "device": "iPhone",
            "system": "iOS 26.5",
            "requestedSizes": [1024],
            "displayedMessage": "Try another capture.",
            "attempts": [
                {
                    "edgePx": 1024,
                    "bytesUploaded": len(image_a) + len(image_b),
                    "latencyMs": 123,
                    "totalMs": 456,
                    "status": "recognizer failed",
                    "recognitionCategory": "reject_retake",
                    "failedChecks": ["hull_label_no_accepted_threshold_a"],
                    "detail": "Could not find a cube-like silhouette in image A.",
                }
            ],
            "images": [
                {
                    "role": "imageA",
                    "edgePx": 1024,
                    "contentType": "image/jpeg",
                    "bytes": len(image_a),
                    "base64": base64.b64encode(image_a).decode("ascii"),
                },
                {
                    "role": "imageB",
                    "edgePx": 1024,
                    "contentType": "image/jpeg",
                    "bytes": len(image_b),
                    "base64": base64.b64encode(image_b).decode("ascii"),
                },
            ],
        }
    ).encode("utf-8")


def test_upload_id_from_reference_accepts_id_path_and_url():
    assert fetch_ios_repro_upload.upload_id_from_reference("upload-123") == ("upload-123", None)
    assert fetch_ios_repro_upload.upload_id_from_reference(
        "/runs/ios-repro-uploads/upload-123/summary.md"
    ) == ("upload-123", None)
    assert fetch_ios_repro_upload.upload_id_from_reference(
        "https://api.test/runs/ios-repro-uploads/upload-123/bundle.json"
    ) == ("upload-123", "https://api.test")


@pytest.mark.parametrize("reference", ["..", "../escape", r"..\escape", "bad/id"])
def test_upload_id_from_reference_rejects_unsafe_ids(reference):
    with pytest.raises(ValueError):
        fetch_ios_repro_upload.upload_id_from_reference(reference)


def test_fetch_upload_downloads_bundle_and_imports_manifest(tmp_path, monkeypatch):
    captured = {}

    def fake_download_bundle(url, *, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return _bundle_bytes()

    monkeypatch.setattr(fetch_ios_repro_upload, "download_bundle", fake_download_bundle)

    target = fetch_ios_repro_upload.fetch_upload(
        "upload-123",
        base_url="https://api.test",
        out_root=tmp_path / "corpus",
        timeout=3.0,
    )

    assert target == tmp_path / "corpus" / "upload-123"
    assert captured == {
        "url": "https://api.test/runs/ios-repro-uploads/upload-123/bundle.json",
        "timeout": 3.0,
    }
    assert (target / "bundle.json").exists()
    assert (target / "manifest.json").exists()
    assert (target / "images" / "imageA_1024.jpg").read_bytes() == b"\xff\xd8image-a\xff\xd9"


def test_download_bundle_rejects_oversized_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

        def read(self, size):
            assert size == 4
            return b"abcd"

    monkeypatch.setattr(fetch_ios_repro_upload, "MAX_BUNDLE_BYTES", 3)
    monkeypatch.setattr(
        fetch_ios_repro_upload.urllib.request,
        "urlopen",
        lambda _request, timeout: FakeResponse(),
    )

    with pytest.raises(ValueError, match="exceeds"):
        fetch_ios_repro_upload.download_bundle("https://api.test/bundle.json", timeout=3.0)
