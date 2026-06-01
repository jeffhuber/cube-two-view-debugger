from __future__ import annotations

import json

from tools import smoke_ios_repro_upload


def test_build_smoke_bundle_has_valid_minimal_shape():
    bundle = smoke_ios_repro_upload.build_smoke_bundle("2026-06-01T00:00:00Z")

    assert bundle["formatVersion"] == 1
    assert bundle["generatedAt"] == "2026-06-01T00:00:00Z"
    assert len(bundle["images"]) == 2
    assert {image["role"] for image in bundle["images"]} == {"imageA", "imageB"}
    assert all(image["contentType"] == "image/jpeg" for image in bundle["images"])
    assert "base64" in bundle["images"][0]


def test_post_bundle_sends_json_and_debug_token(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

        def read(self):
            return b'{"status":"ok","uploadId":"smoke-123"}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["body"] = request.data
        return FakeResponse()

    monkeypatch.setattr(smoke_ios_repro_upload.urllib.request, "urlopen", fake_urlopen)

    response = smoke_ios_repro_upload.post_bundle(
        "https://api.test.local/api/ios-repro-bundles",
        "secret",
        {"formatVersion": 1},
        timeout=3.0,
    )

    assert response["uploadId"] == "smoke-123"
    assert captured["url"] == "https://api.test.local/api/ios-repro-bundles"
    assert captured["timeout"] == 3.0
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["headers"]["x-cubesnap-debug-upload-token"] == "secret"
    assert json.loads(captured["body"].decode("utf-8")) == {"formatVersion": 1}
