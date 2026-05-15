import json
from pathlib import Path


SETTINGS = Path(__file__).resolve().parents[1] / ".claude" / "settings.json"


def test_github_markdown_writes_require_body_files():
    permissions = json.loads(SETTINGS.read_text(encoding="utf-8"))["permissions"]
    allow = set(permissions["allow"])
    deny = set(permissions["deny"])

    unsafe_broad_allows = {
        "Bash(gh pr create:*)",
        "Bash(gh pr comment:*)",
        "Bash(gh pr review:*)",
        "Bash(gh pr edit:*)",
        "Bash(gh issue create:*)",
        "Bash(gh issue comment:*)",
        "Bash(gh issue edit:*)",
    }
    assert not (allow & unsafe_broad_allows)

    safe_body_file_allows = {
        "Bash(gh pr create --body-file:*)",
        "Bash(gh pr comment --body-file:*)",
        "Bash(gh pr review --body-file:*)",
        "Bash(gh pr edit --body-file:*)",
        "Bash(gh issue create --body-file:*)",
        "Bash(gh issue comment --body-file:*)",
        "Bash(gh issue edit --body-file:*)",
    }
    assert safe_body_file_allows <= allow

    common_inline_body_denies = {
        "Bash(gh pr create --body:*)",
        "Bash(gh pr comment --body:*)",
        "Bash(gh pr review --body:*)",
        "Bash(gh pr edit --body:*)",
        "Bash(gh issue create --body:*)",
        "Bash(gh issue comment --body:*)",
        "Bash(gh issue edit --body:*)",
    }
    assert common_inline_body_denies <= deny
