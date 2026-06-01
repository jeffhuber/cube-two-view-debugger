#!/usr/bin/env python3
"""Smoke-test the protected CubeSnap iOS repro upload endpoint.

This posts a tiny synthetic repro bundle with two minimal JPEG byte payloads.
It is intended for operator checks after rotating tokens or changing Railway /
Xcode Cloud configuration; it prints the returned upload id, never the token.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Mapping


DEFAULT_ENDPOINT = "https://api.cubesnap.app/api/ios-repro-bundles"


def build_smoke_bundle(generated_at: str) -> dict[str, Any]:
    image_a = b"\xff\xd8smoke-image-a\xff\xd9"
    image_b = b"\xff\xd8smoke-image-b\xff\xd9"
    return {
        "formatVersion": 1,
        "generatedAt": generated_at,
        "status": "failed",
        "appVersion": "smoke",
        "buildConfiguration": "Smoke",
        "device": "operator",
        "system": "local",
        "requestedSizes": [16],
        "displayedMessage": "Smoke upload.",
        "attempts": [
            {
                "edgePx": 16,
                "bytesUploaded": len(image_a) + len(image_b),
                "latencyMs": 0,
                "totalMs": 0,
                "status": "smoke",
                "recognitionCategory": "reject_retake",
                "recognitionCategoryReason": "operator_smoke",
                "failedChecks": ["operator_smoke"],
                "detail": "Synthetic operator smoke bundle.",
            }
        ],
        "images": [
            {
                "role": "imageA",
                "edgePx": 16,
                "contentType": "image/jpeg",
                "bytes": len(image_a),
                "base64": base64.b64encode(image_a).decode("ascii"),
            },
            {
                "role": "imageB",
                "edgePx": 16,
                "contentType": "image/jpeg",
                "bytes": len(image_b),
                "base64": base64.b64encode(image_b).decode("ascii"),
            },
        ],
    }


def post_bundle(endpoint: str, token: str, bundle: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(bundle, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-CubeSnap-Debug-Upload-Token": token,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "ok":
        raise RuntimeError(f"upload endpoint returned non-ok status: {payload}")
    return payload


def _token_from_args(args: argparse.Namespace) -> str:
    if args.token:
        return args.token
    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise SystemExit(
            f"missing token: pass --token or set {args.token_env}; token will not be printed"
        )
    return token


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("CUBESNAP_DEBUG_UPLOAD_URL", DEFAULT_ENDPOINT))
    parser.add_argument("--token", help="Debug upload token. Prefer --token-env for shell history safety.")
    parser.add_argument("--token-env", default="CUBE_IOS_REPRO_UPLOAD_TOKEN")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--generated-at",
        default=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    args = parser.parse_args(argv)

    token = _token_from_args(args)
    bundle = build_smoke_bundle(args.generated_at)
    try:
        response = post_bundle(args.endpoint, token, bundle, args.timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"upload smoke failed: HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"upload smoke failed: {exc}", file=sys.stderr)
        return 1

    print("upload smoke ok")
    print(f"uploadId: {response.get('uploadId')}")
    if response.get("summaryPath"):
        print(f"summaryPath: {response.get('summaryPath')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
