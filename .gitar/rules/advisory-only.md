# Advisory-only review mode

Operate in advisory, comment-only mode on this repository.

- Do not push commits or branches.
- Do not apply or commit fixes.
- Do not approve pull requests.
- Do not merge pull requests.
- Do not block merges or request changes.

Post review findings as comments only. Humans are the sole merge authority.

This rule is the version-controlled backstop for the same advisory-only
posture configured in the Gitar dashboard (Block merges: Never; Auto-approve:
off; autofix/auto-merge disabled). It exists so the constraint survives even if
a dashboard setting is changed, and so the intent is auditable in-repo.
