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

**Current posture (2026-05-22):** Post-strategic-shift (see
`tools/STATE_OF_THE_WORLD.md`), Claude's geometry research is now
positioned as **scaffolding around** Codex's production `cv-local`,
not a replacement for it. The role split reflects that:

- **Claude**: geometry research + diagnostics + benchmark harness + docs.
- **Codex**: production recognizer + guardrail experiments + corpus/hard-case probes + API/server behavior.
- **Shared**: first-principles docs, labeled fixtures, the regression-gate harness.

### Codex owns (production recognizer)

- `rubik_recognizer/*` — production color classifier, recognizer, image pipeline
- `tests/test_white_up_rules.py`, `tests/test_hard_cases.py`, `tests/test_recognizer.py` and other production-recognizer tests
- `tests/fixtures/corpus_manifest.json`, `tests/fixtures/hard_case_manifest.json` — corpus/hard-case baselines
- `tools/probe_corpus.py`, `tools/probe_hard_cases.py` — production contract probes
- `app.py` — the Rubik Two-View Recognizer HTTP server (stdlib `ThreadingHTTPServer`)
- **Production guardrail behavior** (Phase 3 onward): retake/manual-fixer routing, low-trust abstention

### Claude owns (geometry research + tooling)

- `tools/global_cube_model.py` and adjacent scripts (`interior_bezel_detection.py`, `render_global_cube_model_v0_overlays.py`, etc.)
- `tools/baseline_post_218.py` — **the regression-gate harness.** Re-generate the committed snapshot when the global model changes.
- `tools/sample_stickers_from_hull.py`, `tools/extract_clean_dataset.py` — clean-label dataset extractor (PR #126)
- `tools/propose_geometry_labels.py`, `tools/evaluate_auto_geometry.py`, `tools/diagnose_grid_rejection.py` — auto-geometry framework
- `tools/rectify_faces.py` — face rectification helper
- `tools/render_synthetic_cube.py` — synthetic corpus renderer
- `tools/evaluate_mask_pipeline.py`, `tools/equalize_faces.py` — mask-path experiments
- **`tools/README.md`** — tool inventory + status tags.
- All `tools/*_v0.py` localizer/refiner experiments (negative results kept as institutional memory).

### Shared (touch with care, coordinate first)

- `COORDINATION.md` — this file. Either can edit any section.
- `CLAUDE.md` (both repos) — operating envelope. Either can update; mention here.
- `tools/FIRST_PRINCIPLES_RECOGNIZER_DESIGN.md` — north-star design.
- `tools/STATE_OF_THE_WORLD.md` — entry point / current architecture map.
- `tools/FAILURE_TAXONOMY.md` — failure-mode reference.
- `tools/BENCHMARK_INDEX.md` — fixture/report/script lookup.
- `tools/POST_218_BASELINE_AND_TAXONOMY.md` — the decision spine. Updates happen only via committed re-baseline runs of `baseline_post_218.py`.
- `tools/FULL_CORNER_LABELING.md` / `tools/corner_conventions.py` / `tests/fixtures/full_corner_ground_truth.json` — canonical `Va/Vb + 0..5` corner convention, A/B face outlines, flattened facelet mapping, and seed full-corner truth. Update these before touching any downstream geometry convention.
- `tests/fixtures/gcm_axis_ground_truth.json` — legacy user-labeled axis fixture. Treat as provisional until audited/migrated against full-corner labels; do not assume `near_x/near_y/near_z` are one-edge truth.
- `tests/fixtures/post_218_baseline.json` — legacy accuracy snapshot derived from `gcm_axis_ground_truth.json`. Regenerate only after the fixture semantics are made explicit.
- `tools/extract_color_samples.py` — Claude's, but Codex added the `white[- ]up` regex fix in #135. **Coordinate before editing.**
- `tests/test_auto_geometry_metrics.py` — Claude's, but a growing surface. **Coordinate before adding tests that interact with discovery/geometry.**

### Off limits unless explicitly coordinated

- Each other's in-flight branches
- Each other's open PRs (read freely, comment freely, do not commit)
- Git history rewrites on shared branches

---

## In Flight

Update when opening a PR; clear when merged. Keep this current — it's the primary collision avoidance mechanism.

| Owner | Branch | PR | What | Touches | ETA |
|---|---|---|---|---|---|
| Claude | `claude/phase4-corpus-expansion` | TBD | Phase 4 v1.1: corpus expansion to 70 cases (6 new sets: 20, 38, 40, 41, 43, 45) + regenerated Phase 2B matrix + re-fit ranker. **Honest finding: data lever doesn't move the needle** — random_forest's 18% FPR at v1 was overfit to the 58-case sample; with 33 catastrophic instead of 20, generalizes to 36%. The feature lever (`two_view_consistency` from #243) is the higher-leverage next step. | `tests/fixtures/{hard_case_manifest, gcm_axis_ground_truth, phase2b_recomputed_signals, phase4_trust_ranker_v1}.json`, `tools/PHASE_4_TRUST_RANKER_V1.md`, `tests/test_phase4_trust_ranker.py`. No production changes. | this week |

### Proposed for Codex (please pick up or push back)

- **Phase 4 v2: wire two-view consistency as the 7th feature.** Math primitive shipped in #243 (R_Y(180°)); axis-of-rotation fix shipped in #245 (single 180° around camera X, the correct convention). Integration is **NOT** straightforward direct-feeding — Codex's audit of #245 confirmed that raw per-image `model.axis_x_2d/y/z` fed into the metric produces median 173° residual on 35 human-labeled GOOD pairs (signature of an A↔B axis-frame mismatch). The follow-up PR must therefore land an A/B axis canonicalization step FIRST, validate it against `tests/fixtures/gcm_axis_ground_truth.json` (target: GOOD-pair median ≤25°), then capture/inject the metric. See `tools/TWO_VIEW_CONSISTENCY.md` "Integration plan" → "Step 0" for the canonicalization investigation. (Could also be Claude's lane — flag here for visibility; whoever picks it up first.)

*(Either side: populate your row when you start something.)*

---

## Recently Shipped

Last 5 per side. Newest first. One line + PR # + the takeaway.

### Claude

- **#225** — Phase 1: cv-local baseline on the 58-case axis-labeled gallery. New `tools/baseline_cv_local.py` + committed snapshot `tests/fixtures/cv_local_baseline.json` (schema uniform with `post_218_baseline.json` so `--diff` works cross-side). **Headline finding: 90% of cv-local cases fail structural consistency** — its 3 face-quads are independently extrapolated and don't share a trihedral vertex on most cases. Falls out: face-quad structural consistency is itself a candidate Phase 2 trust signal.
- **#224** — Doc fixups flagged by Codex on the Phase 0 plan: marked Phase 0 done in STATE_OF_THE_WORLD, replaced Phase 2 correlation success criterion with product-shaped recall/false-retake target, cleared stale In Flight row, corrected `app.py` description.
- **#223** — Hygiene: gitignore `.DS_Store` + `.claude/worktrees/`, allow `curl /tmp/sam2_checkpoints/` permissions.
- **#222** — Phase 0 consolidation: new `STATE_OF_THE_WORLD.md`, `FAILURE_TAXONOMY.md`, `BENCHMARK_INDEX.md`, `README.md`; rewritten `FIRST_PRINCIPLES_RECOGNIZER_DESIGN.md`; refreshed `COORDINATION.md` with new role split + regression-gate; downstream `baseline_post_218.py` field rename.
- **#221** — Mechanical rename `chirality_*` → `phase_*` / `_resolve_near_far_phase`. Names stripped of misnomer; legitimate true-chirality reference (CCW/CW hexagon ordering) kept with clarifying comment.
- **#220** — Post-#218 baseline + failure taxonomy (decision spine). Committed `tests/fixtures/post_218_baseline.json` regression-gate snapshot, taxonomy categories, `--diff` mode, 6-step recommended next sequence per Codex+Devin.
- **#218** — Run phase check AFTER vertex ensemble. +33pp end-to-end accuracy (45.7% → 79.3%) from a single block reorder; demonstrates the detector is sensitive to vertex-offset noise.
- **#213** — Enable chirality (now phase) auto-correction with empirical polarity. `sep<0` ≡ correct, opposite of naive bezel-darkness reasoning; ~82% detector agreement with position truth on non-ambiguous runs.
- **#210** — Chirality detection diagnostic (diagnostic-only). Initial framework for per-corner darkness sampling; landed as visibility-only before #213 enabled correction.
- **#182** — Global cube model, ground-truth-validated pipeline. PnP/mean3/refinement prototype and durable `gcm_vertex_ground_truth.json`; diagnostics-only with median vertex error ~72 px.
- **#178** — Iterative interior-bezel refinement + per-line quality + slot/cell join helper.
- **#177** — Interior bezel-line detection probe + human-review fixture.
- **#176** — Hex-fitter failure taxonomy + walkthrough generator (diagnostics-only).

### Infra (Devin-authored, mirrored across both repos)

- **ctvd#144 / cube-snap#135** — Audit watchdog: 5-min `schedule` cron in `devin-audit-bridge.yml` re-scans open PRs with `needs-devin-audit`; head-SHA dedupe prevents re-pinging Devin. Catches missed event-driven dispatches.

### Codex

- **#200** — Two-view geometry consistency signal (diagnostics-only). Adds CV-local provenance diagnostics in the recognizer and evaluator; no production promotion path yet.
- **#199** — Vertex hypothesis ensemble diagnostics. Canonicalizes vertex feedback and expands hypothesis pools, but agreement policies still make false-confident selections; do not wire.
- **#198** — Vertex/axis source-selection confidence diagnostics. Existing fit-quality selection picks the lower-error source 17/23 times but still makes 15 false-confident selections; confidence/source selection remains the blocker.
- **#197** — Geometry-first face split diagnostics. Generated face quads/cells are nondegenerate on paired rows; upstream vertex/axis confidence remains the blocker.
- **#196** — SAM3 whole-cube silhouette bakeoff diagnostics. Whole-cube masks beat rembg on mean/median vertex error but have a regression tail; use as alternate hypothesis/cross-check only.
- **#195** — SAM3 box-guided prompt bakeoff diagnostics. 7-anchor geometry boxes make SAM3 produce face masks, but vertex recall remains 0/16 top-3; do not wire box-guided face masks.
- **#194** — SAM3 current-prompt bakeoff diagnostics. Plain text face prompts produce masks/candidates on all easy rows but 0/16 top-3 vertex recall; do not wire current face prompts.
- **#193** — MLX SAM3 mask export diagnostics. Preserves Claude's cached whole-cube masks, proves the MLX bridge can run on this Mac without re-downloading weights, and keeps SAM3 outputs in the #192 external-mask schema.
- **#175** — Overlay-feedback ingest + cell-discontinuity diagnostics. Ships structured per-slot human labels (`hard_case_visual_feedback.json`) + stdlib xlsx parser + probe.
- **#174** — Repair-backfill behavior experiment. Probes unstable standard repairs only as a manual-review path; Set 61 improves 33/54 -> 34/54 and stays manual.
- **#173** — Grid purity manual-review guard. Promotes no states; routes Set 30-style top-visible grid impurity to manual review only.
- **#170** — Grid purity guard diagnostics. Adds diagnostics-only selected-grid purity/top-visible overlap tags; no recognizer behavior changes.
- **#169** — Repair backfill opportunity diagnostics. Adds diagnostics-only tags for Set 61-style skipped conflict-backfill opportunities; no recognizer behavior changes.

---

## Decision Log

Newest first. Each entry: date, decision, one-line why.

- **2026-05-22** — **Phase 1 finding (#225): cv-local face-quads are not geometrically consistent on 90% of the 58-case gallery.** cv-local extrapolates each 3×3 sticker grid to a face-quad independently with no shared-corner constraint, so the 3 face-quads on most cases don't form a single coherent projected cube. Direct implication: per-axis bearing error isn't the right cross-system metric. Higher-leverage implication: **the consistency check itself is a candidate Phase 2 trust signal** ("if the 3 face-quads don't share a vertex within X px, route to retake"). Falls directly out of the snapshot in `tests/fixtures/cv_local_baseline.json`. See `tools/PHASE_1_CV_LOCAL_BASELINE.md`.
- **2026-05-22** — **Strategic shift: first-principles geometry is scaffolding around `cv-local`, not a replacement.** Per the Codex+Devin synthesis ("a plausible cube model is cheap; a trustworthy cube model is hard"), the global cube model is no longer on a near-term path to replace production. Its role: (a) trust layer / guardrails around cv-local, (b) labeled-training-data source for a future learned ranker, (c) benchmark harness. Three policy bars now gate all first-principles work — see `tools/FIRST_PRINCIPLES_RECOGNIZER_DESIGN.md`. Phased roadmap Phase 0–5 in `tools/STATE_OF_THE_WORLD.md`. Decision spine: `tools/POST_218_BASELINE_AND_TAXONOMY.md`.
- **2026-05-22** — Chirality → near_far_phase rename (#221) — "chirality" is a misnomer for what is actually a 60° body-diagonal rotational degeneracy. See `tools/NEAR_FAR_PHASE_REPORT.md`.
- **2026-05-22** — Standing in-thread merge delegation for Claude (#140 cube-snap, #219 ctvd): "Keep going" / "continue" / "proceed" authorize merge of any Claude-owned PR that is `devin-audit-done` + CLEAN. Does NOT cover Codex's PRs, missing labels, `--admin` overrides, or topic redirects.
- **2026-05-21** — Phase auto-correction enabled in production (#213/#218). Empirically validated polarity (`sep<0` ≡ correct) + vertex-ensemble-first ordering yields 77.6% non-catastrophic vs 45.7% pre-#213. 95%+ of remaining catastrophic failures are phase-decision miscalls; vertex precision is the dominant remaining lever.
- **2026-05-20** — Bezel+discontinuity cell join remains diagnostics-only. On #175's 270 human-reviewed overlay cells, default `line_q>=0.40 && distance<=30px && discontinuity` hit 32 human-bad cells and 0 human-good cells, but 54 human-bad cells were both-miss and labels are slot-level; use as guard evidence, not behavior.
- **2026-05-19** — Human overlay feedback is now structured supervision for hybrid geometry. The first 5 reviewed sets have 28/30 bad slots, led by B:L and A:F wrong-source/bad-quad failures; use this to guide diagnostics, not production behavior.
- **2026-05-19** — Cell-discontinuity scoring remains diagnostics-only. On the 30 human-reviewed overlay slots, human-bad rows have much higher mean score than the 2 human-good rows, but the sample is too small/skewed to become a guard.
- **2026-05-19** — Repair backfill behavior may probe unstable standard repairs only under the Set 61 diagnostic shape. Full corpus stayed contract-clean and hard cases stayed target-clean; Set 61 moved 33/54 -> 34/54 and remains manual-review, so this is not a promotion path.
- **2026-05-19** — Grid-purity production behavior is manual-review only. Combined mining found no success-row false positives: the current grid-purity tag only hits hard Set 30, so the safe behavior experiment is to demote otherwise clean/high-confidence outcomes to manual review, never promote or reject.
- **2026-05-19** — Candidate grid-purity guard remains diagnostics-only. Set 30 has the distinct current signature of high top-visible component overlap plus low expected-face purity across the top-visible triple; surface it as manual-review evidence before considering any guard behavior.
- **2026-05-19** — Candidate repair-backfill opportunity remains diagnostics-only. Set 61 is a standard-repair/manual-review miss where red/orange conflict-backfill would apply but is skipped because standard repair candidates exist; Set 62 is the positive control where backfill runs and succeeds. Surface the opportunity in probes before any behavior experiment.
- **2026-05-19** — Candidate grid-span guard remains diagnostics-only. Mining #165 probe outputs found zero-current-FP candidates (`shape spread >=29.952 + sampled cells >=15`, `nearest-grid ratio >=1.284 + unsupported cells >=5`, and high span score), but the sample is only 30 rows and this is guard/manual-review evidence, not a promotion signal.
- **2026-05-19** — Set 21 direct-legal tie-break inspection found no safe promotion signal yet. The two legal states use the same selected side-pair geometry, the top raw merged-score gap is only 0.02 on a ~1416 score, and balanced facelet-variant cost is identical (93.8762 vs 93.8762). Keep Set 21 manual-review; surface raw-score and variant-cost margins in probes before considering any tie-break rule.
- **2026-05-19** — Direct-legal ambiguity diagnostics are now the next production trust lever. After #159, full corpus + hard-case mining found no remaining corpus exact-54 manual-review rows; the only exact-54 manual-review row is hard Set 21, whose direct legal candidates are effectively tied (2 states, top/second confidence both 0.8332, rounded gap 0.0). Do not promote Set 21 without a new tie-breaker signal; prefer surfacing legal-candidate margin in probes first.
- **2026-05-19** — Late KNN after canonical geometry selection is a negative production experiment. Codex tried three opt-in variants: (1) KNN as extra facelet repair/rebalance alternatives, (2) KNN-derived state variants after geometry selection, and (3) capped/cached balanced KNN-primary state variants for top merged candidates. Targeted corpus sets 12/14/24/27/28 and OOD hard sets 57/58/61/62 showed **zero score/category/candidate-count deltas** versus current main; the uncapped state-variant form was computationally unacceptable. Do not pursue this bolt-on path without a new scoring hypothesis. KNN remains useful for clean rectified samples and should be revisited when geometry is precise.
- **2026-05-18 (post-#152)** — Hybrid pipeline experiment (rectify-on-existing-recognizer-quads + knn5_lab_full) **does NOT close the end-to-end gap.** Per-sticker accuracy: canonical 66.11%, knn5_lab_full 65.32% — both far below existing recognizer's 82.7%. Per-face accuracy distribution is sharply BIMODAL (46% perfect faces + 46% near-random faces, almost nothing in the 0.5-0.9 middle), confirming the failure mechanism: `analyze_image` returns some 3×3 grids whose 9 "stickers" span across multiple physical cube faces; the resulting rectification produces garbage on those faces. Hull-guard attempt (mirror of Codex #141 as a hard reject in evaluator) doesn't fix it — the cube hull encloses all 3 visible faces, so multi-face grids pass the inside-count check trivially. **Two viable forward paths**: (B1) Codex applies `knn5_lab_full` to the existing recognizer's direct sticker samples — leverages production's good grid selection, expected +1-2pp on 82.7% [Codex's Track B item 8]; (B2) Claude builds a learned face-quad regressor with richer CNN features on rembg mask + RGB crop, trained on 68 real hull labels with harsh leave-one-set-out CV [Tier 2 item 12]. Both paths can run in parallel. Synthetic corpus v2 deferred until learned-regressor-on-real-labels saturates. **Negative-result PR** (hull-guard attempt + bimodal-distribution analysis + grid-geometry diagnosis) coming; documents the experiment for future readers.
- **2026-05-18 (end of day)** — Mask-path next-steps update: steps (1), (2), (3) all SHIPPED (#139, #140, #142). (3) — learned vertex regressor — shipped despite the original "only if (1)+(2) don't close the gap" framing because the negative result on (2) [equalize] confirmed classification is near-ceiling on rectified-from-human-quads (97.28%) and the regressor experiment itself was the right way to learn whether sklearn-on-15D could close the geometry gap. It cannot — Ridge smooths toward the mean. Only step **(4) synthetic corpus v2** remains as Claude's active mask-path investment; the binding constraint is the labeled-data budget (n=68), and synthetic v2 is the highest-leverage way to expand it. Meanwhile Codex's lane is closing the gap from the production-recognizer side (#141 hull guard, #145 grid-extrapolation guard).
- **2026-05-18 (end of day)** — Audit chain hardened with Devin-authored watchdog (ctvd#144 / cube-snap#135). The event-driven dispatch (`pull_request_target` + `issue_comment`) had a stuck-state failure mode: PRs labeled `needs-devin-audit` could sit indefinitely if the `labeled` event failed to fire (manual labeling outside the `gh pr edit --add-label` path, workflow errors, races). The 5-min `schedule` cron + `scheduled_pull_requests()` scan converges these. `devin_already_reviewed_sha()` dedupe prevents re-pinging Devin for the same head SHA, so the watchdog doesn't become a billing problem.
- **2026-05-18** — Learned vertex regressor (Ridge on 68 hull labels, 15-D cheap-hexagon features → 24-D face-quad coords) is a *mixed* result: better mean face IoU (0.728 vs 0.669), 3× lower sticker-center error (35.9 vs 106.4 px), +4.1pp classification accuracy — but 0% pass at face≥0.85 (vs 12% RANSAC) and lower gridsAccepted (62.5% vs 85.7%). Suggests Ridge is smoothing toward the mean and the 15-D feature is too sparse. Path forward: richer features (per-side training, mask-CNN features) and/or offset-from-hexagon targets, but a bigger labeled set (synthetic corpus v2) is likely the binding constraint at n=68. Land as a checkpoint, not yet a production proposer.
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
- [ ] **Geometry regression gate.** If the PR touches `tools/global_cube_model.py` or any downstream behavior (cv-local recognition pipeline, mask-path, rectify), run `tools/baseline_post_218.py --diff tests/fixtures/post_218_baseline.json /tmp/your_new_baseline.json` and paste the row-level diff into the PR body. Aggregate metrics alone are insufficient — per Devin, "some changes improve averages while worsening critical rows." A PR that regresses any case from GOOD → catastrophic without offsetting wins is a merge blocker.
- [ ] **Stacked PRs.** If your PR depends on an unmerged PR, branch from the parent's branch (not main) and document the stack in the PR body ("Stacks on #NNN"). Merge order is then forced and reviewers can audit the diff against the parent rather than against main.
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
