"""Unpack and optionally replay CubeSnap iOS recognition repro bundles.

The iOS app's "Share photos + report" button exports a JSON bundle containing
the exact JPEG crops that were uploaded to the recognizer plus timing and
failure metadata. This tool turns that opaque share payload into a local run
folder with decoded images, a base64-free manifest, and a short Markdown
summary. Network replay is explicit via ``--replay``.
"""
from __future__ import annotations

import argparse
import base64
import shutil
import hashlib
import json
import mimetypes
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


DEFAULT_ENDPOINT = "https://api.cubesnap.app/api/recognize?slim=1&hullLabelTier1=constrained"
DEFAULT_CORPUS_ROOT = Path.home() / "cube-corpus" / "ios-repro-bundles"
DEFAULT_SHARED_PASTEBOARD_ROOT = (
    Path.home()
    / "Library"
    / "Group Containers"
    / "group.com.apple.coreservices.useractivityd"
    / "shared-pasteboard"
    / "items"
)


@dataclass(frozen=True)
class DecodedImage:
    role: str
    edge_px: int
    content_type: str
    bytes: int
    sha256: str
    path: Path


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def load_bundle(path: Path) -> Mapping[str, Any]:
    bundle = _mapping(json.loads(path.read_text()), label="bundle")
    if bundle.get("formatVersion") != 1:
        raise ValueError(f"unsupported iOS repro bundle formatVersion: {bundle.get('formatVersion')!r}")
    return bundle


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem).strip("-")
    if stem in {".", ".."}:
        stem = ""
    return stem or "ios-repro-bundle"


def default_output_dir(bundle_path: Path, *, out_root: Path = DEFAULT_CORPUS_ROOT) -> Path:
    return out_root / _safe_stem(bundle_path)


def _image_suffix(content_type: str) -> str:
    guessed = mimetypes.guess_extension(content_type) or ".bin"
    return ".jpg" if guessed == ".jpe" else guessed


def decode_images(bundle: Mapping[str, Any], out_dir: Path) -> list[DecodedImage]:
    raw_images = bundle.get("images")
    if not isinstance(raw_images, Sequence) or isinstance(raw_images, (str, bytes)):
        raise ValueError("bundle.images must be an array")

    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    decoded: list[DecodedImage] = []
    for index, raw in enumerate(raw_images):
        image = _mapping(raw, label=f"bundle.images[{index}]")
        role = image.get("role")
        edge_px = image.get("edgePx")
        content_type = image.get("contentType") or "application/octet-stream"
        expected_bytes = image.get("bytes")
        encoded = image.get("base64")
        if role not in {"imageA", "imageB"}:
            raise ValueError(f"bundle.images[{index}].role must be imageA or imageB")
        if not isinstance(edge_px, int):
            raise ValueError(f"bundle.images[{index}].edgePx must be an integer")
        if not isinstance(expected_bytes, int):
            raise ValueError(f"bundle.images[{index}].bytes must be an integer")
        if not isinstance(encoded, str):
            raise ValueError(f"bundle.images[{index}].base64 must be a string")
        data = base64.b64decode(encoded, validate=True)
        if len(data) != expected_bytes:
            raise ValueError(
                f"bundle.images[{index}] decoded to {len(data)} bytes, expected {expected_bytes}"
            )
        suffix = _image_suffix(str(content_type))
        path = image_dir / f"{role}_{edge_px}{suffix}"
        path.write_bytes(data)
        decoded.append(
            DecodedImage(
                role=str(role),
                edge_px=edge_px,
                content_type=str(content_type),
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                path=path,
            )
        )
    return decoded


def _without_base64_images(bundle: Mapping[str, Any], decoded: Sequence[DecodedImage], out_dir: Path) -> dict[str, Any]:
    by_key = {(image.role, image.edge_px): image for image in decoded}
    images = []
    for raw in bundle.get("images") or []:
        image = dict(_mapping(raw, label="bundle image"))
        image.pop("base64", None)
        decoded_image = by_key.get((image.get("role"), image.get("edgePx")))
        if decoded_image is not None:
            image["path"] = str(decoded_image.path.relative_to(out_dir))
            image["sha256"] = decoded_image.sha256
        images.append(image)
    manifest = {key: value for key, value in bundle.items() if key != "images"}
    manifest["images"] = images
    manifest["sourceSchema"] = "cubesnap.iosRecognitionReproBundle.v1"
    manifest["manifestSchema"] = "ctvd.iosReproBundleManifest.v1"
    return manifest


def select_image_pair(decoded: Sequence[DecodedImage], edge_px: Optional[int] = None) -> tuple[DecodedImage, DecodedImage]:
    sizes = [edge_px] if edge_px is not None else sorted({image.edge_px for image in decoded}, reverse=True)
    for size in sizes:
        image_a = next((image for image in decoded if image.role == "imageA" and image.edge_px == size), None)
        image_b = next((image for image in decoded if image.role == "imageB" and image.edge_px == size), None)
        if image_a is not None and image_b is not None:
            return image_a, image_b
    requested = f" at {edge_px}px" if edge_px is not None else ""
    raise ValueError(f"bundle does not contain an imageA/imageB pair{requested}")


def _header_parameter_value(value: str) -> str:
    return str(value).replace("\r", "").replace("\n", "").replace('"', "_")


def _header_line_value(value: str) -> str:
    return str(value).replace("\r", "").replace("\n", "")


def render_summary(bundle: Mapping[str, Any], decoded: Sequence[DecodedImage], out_dir: Path) -> str:
    lines = [
        "# CubeSnap iOS Repro Bundle",
        "",
        f"- Generated: `{bundle.get('generatedAt')}`",
        f"- Status: `{bundle.get('status')}`",
        f"- App: `{bundle.get('appVersion')}` `{bundle.get('buildConfiguration')}`",
        f"- Device: `{bundle.get('device')}` `{bundle.get('system')}`",
    ]
    if bundle.get("recognizedState"):
        lines.append(f"- Recognized state: `{bundle.get('recognizedState')}`")
    if bundle.get("displayedMessage"):
        lines.append(f"- Displayed message: {bundle.get('displayedMessage')}")
    lines.append("")
    lines.append("## Attempts")
    for attempt in bundle.get("attempts") or []:
        if not isinstance(attempt, Mapping):
            continue
        checks = attempt.get("failedChecks") or []
        lines.extend(
            [
                "",
                f"- Edge: `{attempt.get('edgePx')}` px",
                f"- Status: `{attempt.get('status')}` / `{attempt.get('recognitionCategory')}`",
                f"- Uploaded bytes: `{attempt.get('bytesUploaded')}`",
                f"- Latency: `{attempt.get('latencyMs')}` ms, total `{attempt.get('totalMs')}` ms",
                f"- Failed checks: `{','.join(str(check) for check in checks)}`",
            ]
        )
        if attempt.get("detail"):
            lines.append(f"- Detail: {attempt.get('detail')}")
    lines.append("")
    lines.append("## Images")
    for image in decoded:
        rel = image.path.relative_to(out_dir)
        lines.append(
            f"- `{image.role}` `{image.edge_px}` px: `{rel}` "
            f"({image.bytes} bytes, `{image.sha256}`)"
        )
    lines.append("")
    return "\n".join(lines)


def unpack_bundle(
    bundle_path: Path,
    out_dir: Optional[Path] = None,
    *,
    out_root: Path = DEFAULT_CORPUS_ROOT,
    preserve_raw_bundle: bool = True,
) -> tuple[Path, dict[str, Any], list[DecodedImage]]:
    bundle = load_bundle(bundle_path)
    target = out_dir or default_output_dir(bundle_path, out_root=out_root)
    target.mkdir(parents=True, exist_ok=True)
    if preserve_raw_bundle:
        shutil.copy2(bundle_path, target / "bundle.json")
    decoded = decode_images(bundle, target)
    manifest = _without_base64_images(bundle, decoded, target)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (target / "summary.md").write_text(render_summary(bundle, decoded, target))
    return target, manifest, decoded


def find_shared_pasteboard_bundles(root: Path = DEFAULT_SHARED_PASTEBOARD_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        root.glob("*/cubesnap-repro-*.json"),
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )


def build_multipart_body(
    *,
    fields: Mapping[str, str],
    files: Mapping[str, tuple[str, bytes, str]],
    boundary: str = "----CubeSnapIOSReproBoundary",
) -> tuple[bytes, str]:
    parts: list[bytes] = []
    for name, value in fields.items():
        safe_name = _header_parameter_value(name)
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{safe_name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    for name, (filename, data, content_type) in files.items():
        safe_name = _header_parameter_value(name)
        safe_filename = _header_parameter_value(filename)
        safe_content_type = _header_line_value(content_type)
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{safe_name}"; '
                    f'filename="{safe_filename}"\r\n'
                ).encode(),
                f"Content-Type: {safe_content_type}\r\n\r\n".encode(),
                data,
                b"\r\n",
            ]
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def replay_bundle(
    *,
    bundle: Mapping[str, Any],
    decoded: Sequence[DecodedImage],
    out_dir: Path,
    endpoint: str,
    edge_px: Optional[int],
    timeout: float,
) -> dict[str, Any]:
    image_a, image_b = select_image_pair(decoded, edge_px=edge_px)
    fields = {
        "clientSource": "cube-snap-ios",
        "clientEntryPoint": "ios-phase0",
        "clientAppVersion": str(bundle.get("appVersion") or ""),
        "clientAttemptIndex": "0",
        "clientAttemptTotal": "1",
        "clientAttemptOrder": "imageA-imageB",
    }
    body, content_type = build_multipart_body(
        fields=fields,
        files={
            "imageA": (image_a.path.name, image_a.path.read_bytes(), image_a.content_type),
            "imageB": (image_b.path.name, image_b.path.read_bytes(), image_b.content_type),
        },
    )
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": content_type,
            "X-CubeSnap-Client": "ios-phase0-spike",
            "X-CubeSnap-Source": "cube-snap-ios",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
            status = int(response.status)
            headers = dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        raw_body = exc.read()
        status = int(exc.code)
        headers = dict(exc.headers.items())

    text = raw_body.decode("utf-8", errors="replace")
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    result = {
        "schema": "ctvd.iosReproBundleReplay.v1",
        "endpoint": endpoint,
        "edgePx": image_a.edge_px,
        "status": status,
        "headers": headers,
        "json": parsed,
        "text": None if parsed is not None else text,
    }
    (out_dir / "recognize_response.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path, help="path to a cubesnap-repro-*.json file")
    parser.add_argument("--out-dir", type=Path, help="directory for decoded images and manifest")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
        help="root directory for default outputs and shared-pasteboard imports",
    )
    parser.add_argument(
        "--import-shared-pasteboard",
        action="store_true",
        help="scan macOS shared-pasteboard items for fresh cubesnap-repro JSON files and unpack them",
    )
    parser.add_argument(
        "--shared-pasteboard-root",
        type=Path,
        default=DEFAULT_SHARED_PASTEBOARD_ROOT,
        help="override the macOS shared-pasteboard items directory",
    )
    parser.add_argument("--replay", action="store_true", help="POST decoded images to the recognizer endpoint")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="recognizer endpoint for --replay")
    parser.add_argument("--edge-px", type=int, help="specific edge size to replay")
    parser.add_argument("--timeout", type=float, default=30.0, help="network timeout in seconds for --replay")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        if args.import_shared_pasteboard:
            bundles = find_shared_pasteboard_bundles(args.shared_pasteboard_root)
            if not bundles:
                print(f"no cubesnap-repro bundles found under {args.shared_pasteboard_root}")
                return 1
            for bundle_path in bundles:
                out_dir, manifest, decoded = unpack_bundle(
                    bundle_path,
                    out_root=args.out_root,
                )
                print(f"imported {bundle_path}")
                print(f"wrote {out_dir / 'bundle.json'}")
                print(f"wrote {out_dir / 'manifest.json'}")
                print(f"wrote {out_dir / 'summary.md'}")
                if args.replay:
                    result = replay_bundle(
                        bundle=manifest,
                        decoded=decoded,
                        out_dir=out_dir,
                        endpoint=args.endpoint,
                        edge_px=args.edge_px,
                        timeout=args.timeout,
                    )
                    body = result.get("json") if isinstance(result.get("json"), Mapping) else {}
                    print(f"wrote {out_dir / 'recognize_response.json'}")
                    print(f"replay HTTP {result['status']}: {body.get('status') or 'non-json response'}")
            return 0

        if args.bundle is None:
            raise ValueError("bundle path is required unless --import-shared-pasteboard is set")
        out_dir, _manifest, decoded = unpack_bundle(args.bundle, args.out_dir, out_root=args.out_root)
        print(f"wrote {out_dir / 'bundle.json'}")
        print(f"wrote {out_dir / 'manifest.json'}")
        print(f"wrote {out_dir / 'summary.md'}")
        for image in decoded:
            print(f"wrote {image.path} ({image.bytes} bytes)")
        if args.replay:
            result = replay_bundle(
                bundle=load_bundle(args.bundle),
                decoded=decoded,
                out_dir=out_dir,
                endpoint=args.endpoint,
                edge_px=args.edge_px,
                timeout=args.timeout,
            )
            print(f"wrote {out_dir / 'recognize_response.json'}")
            body = result.get("json") if isinstance(result.get("json"), Mapping) else {}
            print(f"replay HTTP {result['status']}: {body.get('status') or 'non-json response'}")
    except Exception as exc:  # noqa: BLE001
        print(f"ios_repro_bundle: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
