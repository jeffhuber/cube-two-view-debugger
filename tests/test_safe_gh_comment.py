from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools import safe_gh_comment


def test_post_comment_passes_markdown_as_json_stdin():
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="")

    body = "Use `Path(...).expanduser()` and do not run $(echo boom)."

    safe_gh_comment.post_comment("jeffhuber/cube-two-view-debugger", 251, body, run=fake_run)

    assert calls == [
        (
            [
                "gh",
                "api",
                "-X",
                "POST",
                "repos/jeffhuber/cube-two-view-debugger/issues/251/comments",
                "--input",
                "-",
            ],
            {
                "input": json.dumps({"body": body}),
                "check": True,
                "text": True,
                "capture_output": True,
            },
        )
    ]


def test_edit_comment_passes_markdown_as_json_stdin():
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="")

    body = "Validation: `.venv/bin/python -m pytest tests` -> 769 passed."

    safe_gh_comment.edit_comment("jeffhuber/cube-two-view-debugger", 12345, body, run=fake_run)

    args, kwargs = calls[0]
    assert args[:5] == [
        "gh",
        "api",
        "-X",
        "PATCH",
        "repos/jeffhuber/cube-two-view-debugger/issues/comments/12345",
    ]
    assert args[-2:] == ["--input", "-"]
    assert json.loads(kwargs["input"]) == {"body": body}
    assert kwargs["capture_output"] is True


def test_read_body_accepts_file(tmp_path: Path):
    body_file = tmp_path / "comment.md"
    body_file.write_text("literal `backticks` stay literal\n", encoding="utf-8")

    assert safe_gh_comment._read_body(body_file=body_file) == "literal `backticks` stay literal\n"


def test_main_reports_clean_error_for_missing_body_file(capsys):
    status = safe_gh_comment.main(
        [
            "--repo",
            "jeffhuber/cube-two-view-debugger",
            "--issue",
            "251",
            "--body-file",
            "/tmp/definitely-missing-comment.md",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert captured.err.startswith("error: ")


def test_module_imports_under_pep585_incompatible_python():
    """Regression: `RunFn = Callable[..., subprocess.CompletedProcess[str]]`
    was a module-level assignment evaluated at import time. PEP 585 generic
    subscript on `subprocess.CompletedProcess` requires Python 3.9+, but
    macOS often ships an older interpreter (e.g. anaconda 3.7) as ambient
    `python3` — so the assignment failed at import and the helper was
    unusable from any cwd without a 3.9+ venv on PATH.

    Note that `from __future__ import annotations` (at the top of
    safe_gh_comment.py) only affects annotations IN function signatures
    and class attributes — NOT module-level assignments. So the alias
    must be a forward-reference string to stay sub-3.9-importable.

    This test enforces that RunFn remains a string. If a future refactor
    turns it back into a runtime-evaluated Callable, this test fails fast
    rather than waiting for an operator to hit the import error from
    ambient python3.
    """
    assert isinstance(safe_gh_comment.RunFn, str), (
        f"RunFn must remain a forward-reference string for sub-3.9 import "
        f"compatibility; got {type(safe_gh_comment.RunFn).__name__}: "
        f"{safe_gh_comment.RunFn!r}"
    )
    # Spot-check content so a future refactor that swaps it for an
    # unrelated string (e.g. accidentally "Any") gets flagged.
    assert "subprocess.CompletedProcess" in safe_gh_comment.RunFn, (
        f"RunFn forward-ref must still describe the CompletedProcess "
        f"return type; got {safe_gh_comment.RunFn!r}"
    )
