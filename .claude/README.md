# `.claude/` — Claude Code per-project config

## Files

- **`settings.json`** — committed team-wide allow/deny baseline. Every
  contributor with a checkout gets these rules.
- **`settings.local.json`** — personal host-specific additions or
  overrides. Gitignored globally via `~/.config/git/ignore`. Both files
  are loaded and their allow/deny rules are unioned.

## Rule format and a critical limitation

Each entry is `Bash(<prefix>:*)`. The `:*` suffix matches commands
starting with `<prefix>` followed by any arguments. For shell
pipelines (e.g. `gh pr diff 59 | sed -n '250,400p'`) **every segment
must be allowed independently** — that's why the basic text-tool
block (`sed`, `awk`, `cat`, `head`, `tail`, `jq`, ...) is in the
baseline.

**Critical limitation: prefix matching cannot catch flags at arbitrary
positions in a command.** For example:

- A deny rule `Bash(git push --force:*)` catches `git push --force origin branch`
  (the common form), but NOT `git push origin branch --force` — that command
  starts with `git push origin branch`, which doesn't match the deny prefix.
- A rule `Bash(gh pr merge:*)` allows ALL forms of `gh pr merge`, including
  `gh pr merge 103 --admin` (branch-protection bypass).
- A rule `Bash(python3:*)` allows `python3 -c "import os; os.system('rm -rf /')"`,
  bypassing the `rm` and `sudo` deny rules.

The baseline below mitigates this with an explicit operating envelope:
routine implementation/review loops are pre-approved, while destructive
commands, package installs, broad interpreters, inline GitHub markdown
bodies, and full force-push remain gated. The deny list still catches the
*common* dangerous forms as a tripwire for accidents.

## Operating envelope

The committed baseline is intentionally broad enough for an agent to run a
full PR loop without repeated human approval:

- edit/create repo files, Markdown docs, tests, JSON fixtures, and temp
  body files under `/tmp` or `/private/tmp`
- create branches, stage, commit, push explicit `origin` branches, and
  force-push with lease when rebasing a PR branch
- create/edit/comment/review GitHub PRs and issues, using body files for
  Markdown
- merge PRs after the expected review loop has completed, including
  admin-merge when the repository's branch-protection model requires it
- run repo-local pytest, probes, profiles, app smoke tests, and syntax checks
- inspect, restart, or stop local dev processes

Agents should still pause for human approval before destructive cleanup,
dependency installs, broad interpreter one-liners, credential/config writes,
or any action outside the two active repos / Codex worktrees / temp dirs.

## Rule groups (rough order in `settings.json`)

1. **Read-only git** — `status`, `log`, `diff`, `show`, `branch`,
   `rev-parse`, `fetch`, `worktree`, `reflog`, `blame`.

2. **State-mutating git (routine PR-loop variants)** — `add`, `commit`,
   `checkout`, `switch`, `pull --ff-only`, `cherry-pick`, `stash`,
   `restore`, `merge --no-ff`, `merge --ff-only`,
   explicit `git push -u origin ...`, explicit `git push origin ...`,
   and `push --force-with-lease`.
   - Full `git push --force` and `git push -f` are explicitly denied
     (catches the common form). When rewriting a PR branch, use
     `--force-with-lease`, which refuses if the remote has moved since
     your last fetch.

3. **Read-only gh** — `pr view/list/diff/checks/checkout`, `issue
   view/list`, `repo view`, `release view`, `workflow`, `run`, `api`.

4. **gh PR/issue lifecycle and merge** — body-writing
   commands are pre-approved only in `--body-file` form. Markdown
   passed through inline `--body "..."` is not allowed because shell
   backticks are command substitution. Create/comment/edit/review
   bodies must be written to a temp file first, then submitted with
   `gh ... --body-file /tmp/<name>.md`.
   - `gh pr merge` is inside the operating envelope so the Devin-review
     loop can proceed without a human approving every merge. Use it only
     after review comments are addressed or explicitly non-blocking, tests
     are appropriate for the PR, and the PR is merge-clean. Prefer squash
     merge + branch deletion for these short-lived agent branches.
   - Because permission matching is prefix-based, safe commands should
     put `--body-file` immediately after the subcommand, for example
     `gh pr create --body-file /tmp/pr.md --repo ...`. Reordering flags
     can fall out of the pre-approved path and should require manual
     confirmation rather than silently permitting an unsafe inline body.
     Broad lifecycle commands are also allowed for practical GitHub CLI
     flag ordering, but agents must still use body files for Markdown.

5. **Text tools & pipes** — `rg`, `grep`, `ls`, `find`, `sed`, `awk`,
   `cat`, `head`, `tail`, `jq`, `diff`, `stat`, `file`, `date`, `nl`,
   `realpath`, etc.

6. **Dev loop (narrow forms)** — `npm test`, `npm run
   build/dev/bench/lint/typecheck`, `npx vitest`, `npx tsc`,
   `npx tsx api/`, `npx tsx scripts/`, `npx tsx eval/`.
   - **`npm install` / `npm ci` are NOT pre-approved** — package
     installs run arbitrary install scripts and should stay
     confirmation-gated.

7. **Python (narrow forms only)** — `.venv/bin/python -m
   pytest/cProfile/py_compile/json.tool`, `.venv/bin/python tools/<script>`,
   `.venv/bin/python tests/<script>`, `.venv/bin/python app.py`,
   `.venv/bin/pip list/show` (read-only pip).
   - **`python3 -c` / `python3 <arbitrary>` / `.venv/bin/python -c` are
     NOT pre-approved.** A pre-approved `python3:*` would let
     `python3 -c "import os; os.system(\"rm -rf /\")"` bypass every
     other deny rule. Quick one-liners require a confirmation each
     time — annoying but the safety margin is worth it.

8. **Node (narrow forms only)** — `node scripts/<file>` and
   `node --check <file>`. Note:
   `node -e` and `node -p` (inline-code eval) are NOT pre-approved
   for the same reason as `python3 -c`.

9. **Process management** — `ps`, `kill`, `pkill`, `lsof`, `xargs`.

10. **File scopes** — Read/Edit/Write limited to the two project
    roots, Codex worktrees under `/Users/jhuber/Documents/Codex`,
    `/Users/jhuber/Downloads`, `/tmp`, and `/private/tmp`.

11. **Claude Code primitives** — `TodoWrite`, `TaskOutput`,
    `TaskStop`, `Monitor`, `ToolSearch`.

## Deny rules (best-effort tripwires)

These catch the **common** form of dangerous commands. As noted in
the limitation section above, deliberate flag reordering can slip
past prefix matching — these aren't perfect, but they're a real
backstop for accidents:

- `git push --force`, `git push -f` (catches `git push --force ...`;
  does NOT catch `git push ... --force`). The mitigation is that agents
  should use explicit `git push origin ...` or `git push -u origin ...`
  for normal pushes and `git push --force-with-lease ...` for rebased
  PR branches; full force-push is never part of the workflow.
- `git reset --hard`, `git clean -fd`, `git filter-branch`,
  `git update-ref -d`.
- `rm`, `rm -rf`.
- `curl`, `wget` (also blocks the obvious network-exfiltration
  paths; specific URLs can be whitelisted per-host in
  `settings.local.json`).
- `sudo`.
- Common inline GitHub body forms such as `gh pr create --body ...`
  and `gh issue comment --body ...`. These are denied because markdown
  backticks inside shell arguments execute as command substitutions.
  Prefix matching cannot catch every possible flag ordering, so this is
  a tripwire, not a substitute for the operating rule: always use
  `--body-file` for Markdown.
- `Write` / `Edit` into `~/.ssh`, `~/.aws`, `~/.config`.

## Personal overrides

If you need to whitelist a specific `curl` URL, an `npm install`,
an MCP server tool, a `node -e` one-liner you trust, or a
host-specific tool path, add it to `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(curl -s https://api.example.com/health:*)",
      "Bash(npm install --save-dev:*)"
    ]
  }
}
```

That file is host-private (gitignored) so each contributor's
additions stay local. Treat broadening of `git push:*`, `python3:*`,
or `node:*` as something you'd only add to your *personal* file after
thinking through the bypass paths.

## Why this exists

Earlier we accumulated ~370 ad-hoc allow rules in
`.claude/settings.local.json` via approve-as-you-go, with
inconsistent syntax (`Bash(gh pr *)` vs `Bash(gh pr view:*)`) and
zero deny rules. That made the file hard to audit, easy to
accidentally over-grant, and worse-than-useless for sharing across
contributors. The committed baseline above replaces it with an auditable
operating envelope organised by purpose, plus an explicit deny list.

This baseline started from the cube-snap repo's `.claude/settings.json`
(jeffhuber/cube-snap#103), then was broadened for the Devin/Codex PR
review loop. The important narrowing remains around destructive git,
full force-push, package installs, broad interpreters, and inline
Markdown bodies.
