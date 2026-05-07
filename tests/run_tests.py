from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
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
