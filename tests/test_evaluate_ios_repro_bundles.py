from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.evaluate_ios_repro_bundles import (
    build_summary,
    expected_state_from_manifest,
    find_manifests,
    image_paths,
    render_markdown,
    summarize_payload,
)


SOLVED = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"


def _manifest(root: Path, name: str = "case") -> Path:
    bundle = root / name
    image_dir = bundle / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "imageA_1024.jpg").write_bytes(b"a")
    (image_dir / "imageB_1024.jpg").write_bytes(b"b")
    path = bundle / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "manifestSchema": "ctvd.iosReproBundleManifest.v1",
                "generatedAt": "2026-06-01T05:40:48Z",
                "status": "failed",
                "recognizedState": SOLVED,
                "images": [
                    {"role": "imageA", "edgePx": 1024, "path": "images/imageA_1024.jpg"},
                    {"role": "imageB", "edgePx": 1024, "path": "images/imageB_1024.jpg"},
                ],
            }
        )
    )
    return path


def test_find_manifests_accepts_files_bundle_dirs_and_roots(tmp_path: Path):
    manifest = _manifest(tmp_path)

    assert find_manifests([manifest]) == [manifest]
    assert find_manifests([manifest.parent]) == [manifest]
    assert find_manifests([tmp_path]) == [manifest]


def test_script_help_runs_when_invoked_by_path():
    result = subprocess.run(
        [sys.executable, "tools/evaluate_ios_repro_bundles.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Replay imported CubeSnap iOS repro bundles" in result.stdout


def test_image_paths_selects_matching_pair(tmp_path: Path):
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())

    path_a, path_b = image_paths(manifest_path, manifest, edge_px=1024)

    assert path_a.read_bytes() == b"a"
    assert path_b.read_bytes() == b"b"


def test_expected_state_uses_success_state_from_manifest(tmp_path: Path):
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())

    assert expected_state_from_manifest(manifest) == SOLVED


def test_summarize_payload_extracts_rejection_quality_fields(tmp_path: Path):
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    payload = {
        "status": "rejected",
        "state": None,
        "recognitionCategory": "reject_retake",
        "recognitionCategoryReason": "constrained_fast_reject",
        "reason": "CubeSnap could not separate the cube from the background.",
        "failedChecks": ["non_cube_image_fast_reject", "hull_label_no_accepted_threshold_a"],
        "recognitionSignals": {
            "constrainedInference": {
                "status": "fast_reject",
                "fastReject": {
                    "source": "hull_label_threshold_acceptance",
                    "reason": "no_accepted_hull_label_threshold",
                    "qualityIssue": "cube_mask_not_accepted",
                },
            }
        },
    }

    row = summarize_payload(
        manifest_path=manifest_path,
        manifest=manifest,
        payload=payload,
        elapsed_ms=12.345,
        expected_state=SOLVED,
    )

    assert row["replayStatus"] == "rejected"
    assert row["fastRejectReason"] == "no_accepted_hull_label_threshold"
    assert row["fastRejectQualityIssue"] == "cube_mask_not_accepted"
    assert row["expectedAvailable"] is True
    assert row["exact"] is False


def test_build_summary_and_markdown_count_statuses_and_checks():
    rows = [
        {
            "manifest": "/tmp/case-a/manifest.json",
            "originalStatus": "failed",
            "replayStatus": "success",
            "failedChecks": [],
            "exact": True,
            "expectedAvailable": True,
            "hamming": 0,
        },
        {
            "manifest": "/tmp/case-b/manifest.json",
            "originalStatus": "failed",
            "replayStatus": "rejected",
            "failedChecks": ["non_cube_image_fast_reject"],
            "exact": False,
            "expectedAvailable": True,
            "hamming": None,
        },
    ]

    summary = build_summary(rows)
    markdown = render_markdown(summary)

    assert summary["statusCounts"] == {"rejected": 1, "success": 1}
    assert summary["failedCheckCounts"] == {"non_cube_image_fast_reject": 1}
    assert summary["expectedStateExact"] == {"available": 2, "exact": 1}
    assert "`case-a`" in markdown
    assert "`non_cube_image_fast_reject`" in markdown
