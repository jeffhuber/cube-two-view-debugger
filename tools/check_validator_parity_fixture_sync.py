#!/usr/bin/env python3
"""Verify the shared validator parity fixture is byte-identical in both repos."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


CUBE_SNAP_FIXTURE = Path("src/fixtures/validator_parity_cases.json")
CTVD_FIXTURE = Path("tests/fixtures/validator_parity_cases.json")
CUBE_SNAP_ENV = "CUBE_SNAP_PATH"
CTVD_ENV = "CUBE_TWO_VIEW_DEBUGGER_PATH"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _path_arg(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _default_repo_path(repo_root: Path, fixture: Path, sibling_name: str) -> Path:
    if (repo_root / fixture).is_file():
        return repo_root
    return (repo_root.parent / sibling_name).resolve()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cube-snap's src/fixtures/validator_parity_cases.json "
            "with cube-two-view-debugger's tests/fixtures/validator_parity_cases.json."
        )
    )
    parser.add_argument(
        "--cube-snap",
        type=Path,
        default=_path_arg(os.environ.get(CUBE_SNAP_ENV)),
        help=f"path to the cube-snap checkout (default: ${CUBE_SNAP_ENV} or sibling checkout)",
    )
    parser.add_argument(
        "--ctvd",
        type=Path,
        default=_path_arg(os.environ.get(CTVD_ENV)),
        help=(
            "path to the cube-two-view-debugger checkout "
            f"(default: ${CTVD_ENV} or sibling checkout)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = _repo_root()

    cube_snap_root = (args.cube_snap.expanduser().resolve() if args.cube_snap else None) or (
        _default_repo_path(repo_root, CUBE_SNAP_FIXTURE, "cube-snap")
    )
    ctvd_root = (args.ctvd.expanduser().resolve() if args.ctvd else None) or (
        _default_repo_path(repo_root, CTVD_FIXTURE, "cube-two-view-debugger")
    )

    cube_snap_fixture = cube_snap_root / CUBE_SNAP_FIXTURE
    ctvd_fixture = ctvd_root / CTVD_FIXTURE
    missing = [path for path in (cube_snap_fixture, ctvd_fixture) if not path.is_file()]
    if missing:
        for path in missing:
            print(f"missing validator parity fixture: {path}", file=sys.stderr)
        return 2

    cube_snap_bytes = cube_snap_fixture.read_bytes()
    ctvd_bytes = ctvd_fixture.read_bytes()
    cube_snap_sha = hashlib.sha256(cube_snap_bytes).hexdigest()
    ctvd_sha = hashlib.sha256(ctvd_bytes).hexdigest()

    if cube_snap_bytes != ctvd_bytes:
        print("validator parity fixtures differ", file=sys.stderr)
        print(f"cube-snap: {cube_snap_fixture} sha256={cube_snap_sha}", file=sys.stderr)
        print(f"ctvd:      {ctvd_fixture} sha256={ctvd_sha}", file=sys.stderr)
        return 1

    print("validator parity fixtures match")
    print(f"cube-snap: {cube_snap_fixture} sha256={cube_snap_sha}")
    print(f"ctvd:      {ctvd_fixture} sha256={ctvd_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
