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
