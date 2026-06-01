from __future__ import annotations

import subprocess
from pathlib import Path

from tools import codex_audit_env_preflight as preflight


def _executable(path: Path) -> str:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def test_resolve_required_setting_requires_explicit_env_or_arg():
    result = preflight.resolve_required_setting(
        cli_value=None,
        env={},
        env_name="CODEX_AUDIT_PYTHON",
        label="controlled Python",
    )

    assert not result.ok
    assert "CODEX_AUDIT_PYTHON is not set" in result.detail
    assert "source tools/codex_audit_env.sh" in result.detail


def test_resolve_required_setting_accepts_executable_arg(tmp_path):
    tool = _executable(tmp_path / "python")
    result = preflight.resolve_required_setting(
        cli_value=tool,
        env={},
        env_name="CODEX_AUDIT_PYTHON",
        label="controlled Python",
    )

    assert result.ok
    assert result.detail == tool


def test_run_preflight_detects_tls_failure_before_long_audit():
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["/usr/bin/python3", "-c"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed",
            )
        return subprocess.CompletedProcess(command, 0, stdout="codex-cli 0.test\n", stderr="")

    results = preflight.run_preflight(
        python_path="/usr/bin/python3",
        codex_cli_path="/Applications/Codex.app/Contents/Resources/codex",
        runner=fake_run,
    )

    assert not results[0].ok
    assert "CERTIFICATE_VERIFY_FAILED" in results[0].detail
    assert results[1].ok
    assert all(call[1]["stdin"] == subprocess.DEVNULL for call in calls)


def test_github_tls_probe_treats_http_error_as_successful_tls():
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="github TLS ok\n", stderr="")

    result = preflight.check_python_github_tls(
        python_path="/usr/bin/python3",
        github_url="https://api.github.com/definitely-missing",
        timeout=10,
        runner=fake_run,
    )

    assert result.ok
    assert "urllib.error.HTTPError" in calls[0][0][2]


def test_run_preflight_reports_subprocess_timeout_instead_of_raising():
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    results = preflight.run_preflight(
        python_path="/usr/bin/python3",
        codex_cli_path="/Applications/Codex.app/Contents/Resources/codex",
        runner=fake_run,
    )

    assert not any(result.ok for result in results)
    assert all("timed out after" in result.detail for result in results)


def test_run_preflight_passes_python_tls_and_codex_cli():
    def fake_run(command, **kwargs):
        if command[:2] == ["/usr/bin/python3", "-c"]:
            return subprocess.CompletedProcess(command, 0, stdout="github TLS ok\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="codex-cli 0.test\n", stderr="")

    results = preflight.run_preflight(
        python_path="/usr/bin/python3",
        codex_cli_path="/Applications/Codex.app/Contents/Resources/codex",
        runner=fake_run,
    )

    assert all(result.ok for result in results)
    assert results[0].detail == "github TLS ok"
    assert results[1].detail == "codex-cli 0.test"
