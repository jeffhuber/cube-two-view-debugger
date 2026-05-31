from __future__ import annotations

from pathlib import Path

from tools import check_validator_parity_fixture_sync as sync


def _write_fixture(repo: Path, relative: Path, data: bytes) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_main_accepts_byte_identical_fixtures(tmp_path, capsys):
    cube_snap = tmp_path / "cube-snap"
    ctvd = tmp_path / "cube-two-view-debugger"
    data = b'{\n  "schema": "cube.validatorParityCases.v1",\n  "cases": []\n}\n'
    _write_fixture(cube_snap, sync.CUBE_SNAP_FIXTURE, data)
    _write_fixture(ctvd, sync.CTVD_FIXTURE, data)

    rc = sync.main(["--cube-snap", str(cube_snap), "--ctvd", str(ctvd)])

    captured = capsys.readouterr()
    assert rc == 0
    assert "validator parity fixtures match" in captured.out
    assert "sha256=" in captured.out


def test_main_rejects_byte_mismatch(tmp_path, capsys):
    cube_snap = tmp_path / "cube-snap"
    ctvd = tmp_path / "cube-two-view-debugger"
    _write_fixture(cube_snap, sync.CUBE_SNAP_FIXTURE, b'{"cases":[]}\n')
    _write_fixture(ctvd, sync.CTVD_FIXTURE, b'{"cases": []}\n')

    rc = sync.main(["--cube-snap", str(cube_snap), "--ctvd", str(ctvd)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "validator parity fixtures differ" in captured.err
    assert "sha256=" in captured.err


def test_main_reports_missing_fixture(tmp_path, capsys):
    cube_snap = tmp_path / "cube-snap"
    ctvd = tmp_path / "cube-two-view-debugger"
    _write_fixture(cube_snap, sync.CUBE_SNAP_FIXTURE, b'{"cases":[]}\n')

    rc = sync.main(["--cube-snap", str(cube_snap), "--ctvd", str(ctvd)])

    captured = capsys.readouterr()
    assert rc == 2
    assert str(ctvd / sync.CTVD_FIXTURE) in captured.err
