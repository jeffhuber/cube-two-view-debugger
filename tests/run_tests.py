from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import List


def _dependencies_available() -> bool:
    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _candidate_runtimes(root: Path) -> List[Path]:
    candidates = []
    env_python = os.environ.get("CUBE_PYTHON")
    if env_python:
        candidates.append(Path(env_python))
    candidates.append(root / ".venv" / "bin" / "python")
    candidates.append(Path("/Users/jhuber/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"))
    return candidates


def _rerun_with_dependency_runtime(root: Path) -> None:
    if _dependencies_available():
        return

    current = Path(sys.executable).resolve()
    for candidate in _candidate_runtimes(root):
        if not candidate.exists():
            continue
        try:
            if candidate.resolve() == current:
                continue
        except OSError:
            continue
        os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve())])

    print(
        "Missing Python dependencies: NumPy and/or Pillow.\n"
        "Create the project environment with:\n"
        "  python3 -m venv .venv\n"
        "  .venv/bin/python -m pip install -r requirements.txt\n"
        "Then run:\n"
        "  .venv/bin/python tests/run_tests.py",
        file=sys.stderr,
    )
    raise SystemExit(2)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    _rerun_with_dependency_runtime(root)
    sys.path.insert(0, str(root))
    failures = 0
    tests = 0
    for path in sorted((root / "tests").glob("test_*.py")):
        module = importlib.import_module(f"tests.{path.stem}")
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            tests += 1
            try:
                fn()
                print(f"PASS {path.stem}.{name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {path.stem}.{name}: {exc}")
    print(f"{tests - failures}/{tests} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
