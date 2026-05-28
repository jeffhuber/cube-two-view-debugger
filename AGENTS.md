# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

This is a single-process Python web server (stdlib `http.server`) that recognizes a Rubik's cube state from two isometric photos. No database, no external services, no Node.js/frontend build step.

### Development environment

- **Python >= 3.11** required (uses `int.bit_count()`). The VM ships Python 3.12.
- Virtual environment lives at `.venv/`; all commands use `.venv/bin/python`.
- Dependencies: `requirements.txt` (Pillow, NumPy, scikit-learn). Install with `.venv/bin/pip install -r requirements.txt`.
- `pytest` is needed for running the test suite but is not in `requirements.txt`; the update script installs it.

### Running the server

```bash
.venv/bin/python app.py --host 0.0.0.0 --port 8080
```

Serves the web UI at `http://localhost:8080/` and API at `/api/*`. Verify with `curl -s http://localhost:8080/api/diag`.

### Running tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Note: 8 tests in `test_white_up_rules.py` and `test_diagnose_fit_stage_transitions.py` fail due to pre-existing mock/signature mismatches in the repo (not environment issues). All other ~1190 tests pass.

### Linting

No dedicated linter configuration exists in this repo. The code uses standard Python without mypy/ruff/flake8 config files.

### Key gotchas

- The `python3.12-venv` system package must be installed for `python3 -m venv` to work on Ubuntu.
- NumPy version matters for recognition accuracy (see README "Pinned dependencies" section). The update script pins compatible versions via `requirements.txt`.
- The corpus probe and hard-case probe reference local macOS photo paths (`~/Downloads/...`) that don't exist in the Cloud VM — those tools will skip gracefully with "missing local files" messages.
- The custom test runner (`tests/run_tests.py`) does not support pytest fixtures; always use `pytest` directly.
