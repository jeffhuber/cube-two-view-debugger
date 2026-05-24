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
            {"input": json.dumps({"body": body}), "check": True, "text": True},
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


def test_read_body_accepts_file(tmp_path: Path):
    body_file = tmp_path / "comment.md"
    body_file.write_text("literal `backticks` stay literal\n", encoding="utf-8")

    assert safe_gh_comment._read_body(body_file=body_file) == "literal `backticks` stay literal\n"
