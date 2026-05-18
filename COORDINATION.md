# COORDINATION.md

Living document for Claude + Codex coordination on this repo. Two AI
agents working in parallel on overlapping code is a real merge-conflict
risk. This doc is the cheapest, lowest-friction mitigation: a shared
status board + clear lanes + a pre-PR checklist.

**Both agents read at session start, update at PR open and merge.**

The file lives at the repo root so it's discoverable; both agents can
edit any section. To minimize edit conflicts, keep entries terse
(1-2 lines each) and use simple markdown that's easy to merge.

---

## Lanes

Strict ownership boundaries. Avoid touching anything outside your lane
without coordinating in this doc first.

### Codex owns (production recognizer)

- `rubik_recognizer/*` — production color classifier, recognizer, image pipeline
- `tests/test_white_up_rules.py`, `tests/test_hard_cases.py`, `tests/test_recognizer.py` and other production-recognizer tests
- `tests/fixtures/corpus_manifest.json`, `tests/fixtures/hard_case_manifest.json` — corpus/hard-case baselines
- `tools/probe_corpus.py`, `tools/probe_hard_cases.py` — production contract probes
- `app.py` — the Rubik Two-View Recognizer Flask server

### Claude owns (parallel-track tooling)

- `tools/sample_stickers_from_hull.py`, `tools/extract_clean_dataset.py` — clean-label dataset extractor (PR #126)
- `tools/propose_geometry_labels.py`, `tools/evaluate_auto_geometry.py`, `tools/diagnose_grid_rejection.py` — auto-geometry framework (PR #127, #133, #137)
- `tools/rectify_faces.py` — face rectification (PR #136)
- `tools/render_synthetic_cube.py` — synthetic corpus renderer (PR #132)
- `tools/evaluate_mask_pipeline.py` — end-to-end mask-path evaluator (in flight)
- `tools/equalize_faces.py`, `tools/recognizer_mask.py` — future
- All `tools/*.md` docs that ship alongside the above

### Shared (touch with care)

- `tools/extract_color_samples.py` — Claude's, but Codex added the `white[- ]up` regex fix in #135. **Coordinate before editing.**
- `tests/test_auto_geometry_metrics.py` — Claude's, but a growing surface. **Coordinate before adding tests that interact with discovery/geometry.**
- `CLAUDE.md` — repo guidance. Either can update; mention here.
- `COORDINATION.md` — this file. Either can edit any section.

### Off limits unless explicitly coordinated

- Each other's in-flight branches
- Each other's open PRs (read freely, comment freely, do not commit)
- Git history rewrites on shared branches

---

## In Flight

Update when opening a PR; clear when merged. Keep this current — it's the primary collision avoidance mechanism.

| Owner | Branch | PR | What | Touches | ETA |
|---|---|---|---|---|---|
| Codex | `codex/grid-extrapolation-guard` | pending | Conservative default grid-extrapolation scoring/gating | `rubik_recognizer/*`, focused tests, probes | In progress |

*(Codex: please populate your row when you start something.)*

---

## Recently Shipped

Last 5 per side. Newest first. One line + PR # + the takeaway.

### Claude

- **#140** — Equalize-faces experiment. Negative result; confirms color equalization alone is not enough without better geometry.
- **#139** — End-to-end mask-pipeline evaluator. Mask path currently averages 61.5% sticker accuracy; useful evaluator, not production path yet.
- **#137** — RANSAC/Nelder-Mead hexagon fit + sticker-center error + classification accuracy. Face IoU 0.666 mean, 12% pass at ≥0.85 — geometry plateau confirmed.
- **#136** — Face rectification + per-sticker color extraction. Flat 300×300 face images; trivial pixel slicing for color sampling.
- **#133** — Rembg foundation-model proposers + grid-rejection diagnostic. Cube hull is solved (100% pass at hullIoU≥0.85, mean 0.962).

### Codex

- **#143** — Clean-label color evaluator recomputes runtime classifier modes. Prevents stale JSONL predictions from hiding KNN regressions.
- **#141** — Opt-in rembg hull guard for grid ranking. Penalizes selected grids with <7/9 centers inside the U2-Net cube hull; default behavior unchanged.
- **#135** — Discover hyphenated `white-up` photos. Sets 57/58/61/62 now seen by all tools.
- **#134** — Downgrade skewed repair false positives. Confident wrong → retake/review.
- **#131** — Balanced visible color assignment diagnostics.

---

## Decision Log

Newest first. Each entry: date, decision, one-line why.

- **2026-05-18** — Architecture direction is *rembg → optimized hexagon → rectify → classify*, with a parallel `recognizer_mask.py` path behind an env switch. Promote to production only when end-to-end mask-path evaluator shows it beats current recognizer on corpus + hard cases + has safe fallback policy. (Synthesis from Devin + Codex reviews on #137.)
- **2026-05-18** — Mask-path next steps in order: (1) end-to-end evaluator [Claude], (2) equalize-faces experiment [Claude], (3) learned vertex regressor only if (1)+(2) don't close the gap [Claude], (4) synthetic corpus v2 as parallel investment [Claude].
- **2026-05-18** — Codex next steps in order: (1) Sets 57/58/61/62 confident-false-legal-repair fix, (2) geometry-first ranking gate using `rembg_u2net_hull`, (3) KNN5 Phase 2 broader-scope rollout.
- **2026-05-17** — Clean-label color classifier ceiling is ~97.95% (RF-200 from #126 bake-off). All 11 candidates cluster 96-98%. Classifier algorithm is not the bottleneck.
- **2026-05-17** — KNN5 shipped as red/orange-override only (#128), not full classifier replacement. Phase 2 (broader scope) deferred.
- **2026-05-17** — Joint A+B multiset face-ID enforces {U,R,F,D,L,B} invariant (PR #126 fix from Codex review). Per-side center-classification is too fragile.

---

## Pre-PR Checklist

Before opening any PR or requesting Devin audit:

- [ ] `git fetch origin main && git rebase origin/main` — guarantees `mergeable: MERGEABLE`. Force-push with `--force-with-lease` after rebase.
- [ ] `gh pr list --state open` — check no other in-flight PR touches the same files. If overlap, coordinate via this doc.
- [ ] Full `pytest` green locally. New tests for new behavior.
- [ ] If touching anything in **Shared** above: mention in the PR description.
- [ ] If the PR's results depend on a long-running sweep: post the full `runs/*_summary.txt` as a PR comment so reviewers don't need to re-run.
- [ ] For Claude: tools-only, no `rubik_recognizer/*` edits. For Codex: production-only, no edits to Claude-owned auto-geometry / rectify / mask-path tooling listed above unless coordinated.
- [ ] Update **In Flight** in this doc when opening the PR.
- [ ] Update **Recently Shipped** when the PR merges.

---

## Merge Conventions

- **Squash-merge** is the default for tooling PRs (cleaner history). The pre-PR rebase requirement mitigates the "squash deletes downstream branches" problem that bit PRs #136/#137.
- **`--delete-branch`** is fine when no worktree references the branch. If a worktree exists (rare; usually only the PR author's local one), leave the branch alone — the author will clean up.
- **Force-push only with `--force-with-lease`**, never plain `--force`.
- **Hooks**: never skip with `--no-verify` or `--no-gpg-sign` unless the user explicitly says so.

---

## Sweep / Eval Logging Pattern

Long-running tools (>5 min) should:

1. Print one progress line per (set, proposer) iteration to stderr with `flush=True`.
2. Pipe to a log file with correct redirection order: `python tool.py > /tmp/sweep.log 2>&1` (NOT `2>&1 > file` which loses stderr).
3. On the PR: post the final `runs/*_summary.txt` as a PR comment for the audit record.
4. Provide a single command in the tool's docstring to reproduce.

---

## Updating This Doc

Either agent can edit any section. To minimize edit conflicts:

- Keep entries terse (1-2 lines each).
- When adding to a list, add to the top (newest first).
- When the In Flight table grows past 6 rows or Recently Shipped past 10 entries, prune the oldest.
- If you make a structural change (rearrange sections, add new section), mention in the Decision Log + commit message.

Update this doc as part of the PR that warrants the update, not in a separate PR. That keeps the doc honest about current state.
