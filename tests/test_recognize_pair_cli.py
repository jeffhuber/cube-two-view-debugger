"""
Tests for tools/recognize_pair.py — the CLI runner that mirrors the
HTTP /api/recognize path.

The recognizer can't produce a legal cube from a tiny synthetic JPEG,
so these tests assert on the rejection path: the CLI should still
exit zero (recognition rejection is not a CLI failure), still emit a
runId, still persist a run dir, and still respect --json-output /
--quiet / --set-id.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _solid_jpeg(color=(255, 255, 255)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="JPEG", quality=70)
    return buf.getvalue()


@pytest.fixture
def synthetic_pair(tmp_path: Path):
    a = tmp_path / "imageA.jpg"
    b = tmp_path / "imageB.jpg"
    a.write_bytes(_solid_jpeg((255, 255, 255)))
    b.write_bytes(_solid_jpeg((255, 255, 0)))
    return a, b


def _run_cli(*args, cwd=None):
    """Invoke the CLI as a subprocess. Use the project's venv if
    available so library versions match the test environment."""
    python = sys.executable
    cmd = [python, str(ROOT / "tools" / "recognize_pair.py"), *args]
    env = os.environ.copy()
    return subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_cli_help_succeeds():
    """--help should always succeed without importing the recognizer
    body — argparse should print and exit zero."""
    proc = _run_cli("--help")
    assert proc.returncode == 0
    assert "usage:" in proc.stdout
    assert "image_a" in proc.stdout
    assert "image_b" in proc.stdout
    assert "--set-id" in proc.stdout
    assert "--json-output" in proc.stdout


def test_cli_missing_image_returns_error():
    """Passing a path that doesn't exist should produce a non-zero
    exit + a descriptive stderr message. Important so scripted
    callers can rely on the exit code."""
    proc = _run_cli("/tmp/does-not-exist-A.jpg", "/tmp/does-not-exist-B.jpg")
    assert proc.returncode != 0
    assert "not found" in proc.stderr.lower()


def test_cli_writes_json_output_and_summary(synthetic_pair, tmp_path):
    """End-to-end: tiny synthetic JPEGs should reject (no legal cube
    state), the CLI should still write a complete JSON payload, and
    the summary line should be emitted on stdout."""
    image_a, image_b = synthetic_pair
    json_out = tmp_path / "cli-out.json"
    set_id = "Set 8888 (cli test)"

    proc = _run_cli(
        str(image_a),
        str(image_b),
        "--set-id",
        set_id,
        "--json-output",
        str(json_out),
    )

    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"

    # Summary line on stdout includes set + run id + status.
    assert "set=Set 8888 (cli test)" in proc.stdout
    assert "run=" in proc.stdout
    assert "status=" in proc.stdout

    # JSON file is present and has the expected fields.
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["setId"] == set_id
    run_id = payload["runId"]
    assert "set-8888" in run_id
    # Tiny synthetic image can't form a legal cube; reject is fine.
    assert payload["status"] in {"success", "rejected"}
    assert "artifacts" in payload
    assert "imageA" in payload["artifacts"]
    assert "imageB" in payload["artifacts"]

    # Persisted run dir matches the runId.
    run_dir = ROOT / "runs" / "pairs" / run_id
    assert run_dir.exists(), f"saved run dir missing: {run_dir}"


def test_cli_quiet_suppresses_stdout_summary(synthetic_pair, tmp_path):
    """--quiet should produce empty stdout but still write the JSON
    file. Important for scripted use where stdout is being captured
    for something else."""
    image_a, image_b = synthetic_pair
    json_out = tmp_path / "quiet.json"
    proc = _run_cli(
        str(image_a),
        str(image_b),
        "--set-id",
        "Set 7777 (cli quiet)",
        "--json-output",
        str(json_out),
        "--quiet",
    )
    assert proc.returncode == 0
    assert proc.stdout == ""
    assert json_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["setId"] == "Set 7777 (cli quiet)"
