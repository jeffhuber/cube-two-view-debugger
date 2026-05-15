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

The baseline below mitigates this by **only granting narrow allow rules**
for the highest-risk command families (push, merge, interpreters), and
relying on per-invocation confirmation for the riskier forms. The deny
list still catches the *common* `--force`-first patterns as a tripwire
for accidents.

## Rule groups (rough order in `settings.json`)

1. **Read-only git** — `status`, `log`, `diff`, `show`, `branch`,
   `rev-parse`, `fetch`, `worktree`, `reflog`, `blame`.

2. **State-mutating git (safe variants only)** — `add`, `commit`,
   `checkout`, `switch`, `pull --ff-only`, `cherry-pick`, `stash`,
   `restore`, `merge --no-ff`, `merge --ff-only`,
   `push --force-with-lease`.
   - **Note:** plain `git push:*` is NOT pre-approved. Every regular push
     requires confirmation. `--force-with-lease` is pre-approved
     because it refuses if the remote has moved since your last
     fetch. Full `git push --force` and `git push -f` are explicitly
     denied (catches the common form).

3. **Read-only gh** — `pr view/list/diff/checks/checkout`, `issue
   view/list`, `repo view`, `release view`, `workflow`, `run`, `api`.

4. **gh PR/issue lifecycle (excluding merge)** — body-writing
   commands are pre-approved only in `--body-file` form. Markdown
   passed through inline `--body "..."` is not allowed because shell
   backticks are command substitution. Create/comment/edit/review
   bodies must be written to a temp file first, then submitted with
   `gh ... --body-file /tmp/<name>.md`.
   - **`gh pr merge` is NOT pre-approved.** Every PR merge requires
     confirmation. The reason: `gh pr merge --admin` (branch-protection
     bypass) is indistinguishable from a normal merge under prefix
     matching, and bypassing branch protection should always be an
     explicit human gesture.
   - Because permission matching is prefix-based, safe commands should
     put `--body-file` immediately after the subcommand, for example
     `gh pr create --body-file /tmp/pr.md --repo ...`. Reordering flags
     can fall out of the pre-approved path and should require manual
     confirmation rather than silently permitting an unsafe inline body.

5. **Text tools & pipes** — `grep`, `ls`, `find`, `sed`, `awk`,
   `cat`, `head`, `tail`, `jq`, `diff`, etc.

6. **Dev loop (narrow forms)** — `npm test`, `npm run
   build/dev/bench/lint/typecheck`, `npx vitest`, `npx tsc`,
   `npx tsx api/`, `npx tsx scripts/`, `npx tsx eval/`.
   - **`npm install` / `npm ci` are NOT pre-approved** — package
     installs run arbitrary install scripts and should stay
     confirmation-gated.

7. **Python (narrow forms only)** — `.venv/bin/python -m
   pytest/py_compile/json.tool`, `.venv/bin/python tools/<script>`,
   `.venv/bin/python tests/<script>`, `.venv/bin/python app.py`,
   `.venv/bin/pip list/show` (read-only pip).
   - **`python3 -c` / `python3 <arbitrary>` / `.venv/bin/python -c` are
     NOT pre-approved.** A pre-approved `python3:*` would let
     `python3 -c "import os; os.system(\"rm -rf /\")"` bypass every
     other deny rule. Quick one-liners require a confirmation each
     time — annoying but the safety margin is worth it.

8. **Node (narrow forms only)** — `node scripts/<file>`. Note:
   `node -e` and `node -p` (inline-code eval) are NOT pre-approved
   for the same reason as `python3 -c`.

9. **Process management** — `ps`, `kill`, `pkill`, `lsof`, `xargs`.

10. **File scopes** — Read/Edit/Write limited to the two project
    roots plus `/Users/jhuber/Downloads` and `/tmp`.

11. **Claude Code primitives** — `TodoWrite`, `TaskOutput`,
    `TaskStop`, `Monitor`, `ToolSearch`.

## Deny rules (best-effort tripwires)

These catch the **common** form of dangerous commands. As noted in
the limitation section above, deliberate flag reordering can slip
past prefix matching — these aren't perfect, but they're a real
backstop for accidents:

- `git push --force`, `git push -f` (catches `git push --force ...`;
  does NOT catch `git push ... --force`). The mitigation is that
  plain `git push:*` is also not in the allow list, so any push
  requires confirmation, at which point a `--force` in the args
  is visible to the human approver.
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
  Prefix matching cannot catch every possible flag ordering, so the
  allow list also avoids broad `gh pr create:*` / `gh issue comment:*`
  grants and only pre-approves `--body-file` forms.
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
additions stay local. Treat broadening of `git push:*`, `gh pr
merge:*`, `python3:*`, or `node:*` as something you'd only add
to your *personal* file after thinking through the bypass paths.

## Why this exists

Earlier we accumulated ~370 ad-hoc allow rules in
`.claude/settings.local.json` via approve-as-you-go, with
inconsistent syntax (`Bash(gh pr *)` vs `Bash(gh pr view:*)`) and
zero deny rules. That made the file hard to audit, easy to
accidentally over-grant, and worse-than-useless for sharing across
contributors. The committed baseline above replaces it with ~95
clean rules organised by purpose, plus an explicit deny list.

This baseline is ported from the cube-snap repo's
`.claude/settings.json` (jeffhuber/cube-snap#103). An earlier draft
there was over-broad on `gh pr merge:*`, `git push:*`, `python3:*`,
and `node:*` — each of which let bypass paths through. The version
below incorporates the post-review narrowing.
