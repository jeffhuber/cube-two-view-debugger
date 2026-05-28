from __future__ import annotations

import json
from pathlib import Path

import app


ROOT = Path(__file__).resolve().parents[1]


def test_railway_start_command_uses_port_wrapper():
    config = json.loads((ROOT / "railway.json").read_text())

    assert config["deploy"]["startCommand"] == "python railway_start.py"
    assert config["deploy"]["healthcheckPath"] == "/api/diag"
    assert (ROOT / "Procfile").read_text().strip() == "web: python railway_start.py"


def test_railway_deploy_guard_uses_clean_worktree():
    gitignore_lines = (ROOT / ".gitignore").read_text().splitlines()
    script = (ROOT / "tools" / "deploy_railway_ctvd_recognizer.sh").read_text()

    assert ".worktrees/" in gitignore_lines
    assert "git -C \"$repo_root\" worktree add --detach \"$deploy_dir\" \"$REF\"" in script
    assert "cd \"$deploy_dir\"" in script
    assert "railway up" in script
    assert "--project \"$PROJECT_ID\"" in script
    assert "--service \"$SERVICE\"" in script
    assert "--environment \"$ENVIRONMENT\"" in script
    assert "configFile=/railway.json" in script


def test_runtime_requirements_pin_validated_numpy_minor():
    requirements = (ROOT / "requirements.txt").read_text()

    assert "numpy>=2.3.5,<2.4" in requirements
    assert "rembg>=2.0.75,<2.1" in requirements
    assert "onnxruntime>=1.26.0,<1.27" in requirements


def test_public_cube_snap_domains_are_cors_allowed():
    assert app._origin_is_allowed("https://jeffhuber.github.io")
    assert app._origin_is_allowed("https://cubesnap.app")
    assert app._origin_is_allowed("https://www.cubesnap.app")
    assert not app._origin_is_allowed("https://example.com")
