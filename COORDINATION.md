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

**Current posture (2026-05-26):** Hull-labels Tier 1 (the
convention-aware silhouette-corner labeling pipeline from PRs
#282/#284/#286/#288/#289/#291) is now the **foundation** of the
rectification + repair stack — no longer the next deliverable. The
shadow-trace analyzer (#292) landed; the current corpus run accepts
the whole 70-row axis corpus with no hard failures, with a small
number of accepted-with-warnings rows clustered on
`vertex_cloud_spread_px` and `sticker_score_total`. For current
numbers (acceptance, warnings, gate distributions) read
`tools/SHADOW_TRACE_ANALYSIS.md` directly — it's the source of truth
and regenerates with the corpus. Operator handoff:
`tools/HULL_LABEL_TIER1_WIRING.md`.

Active focus has shifted to **constrained cube-state inference** as
the architectural frame: LLMs are one evidence source in a constrained
solver, not the source of truth. See both repos' `CLAUDE.md` for the
deeper write-up (codified in ctvd #327 / cube-snap #188).

Recent additions on top of the rectification foundation:

- **Hull-label mask threshold selector (#322)** — production-shaped
  threshold picker (Codex); replaces the prior single-threshold
  heuristic in the mask path.
- **Color repair API (#324)** — deterministic color-conflict repair
  surface on the rectified hull-label samples. Foundation for the
  scoreboard and legality probes below.
- **Post-repair scoreboard (#325)** + **legality repair probes
  (#326, #329)** — 46-pair shadow corpus evaluated across multiple
  repair variants (canonical-count → conservative-legal → guarded-broad
  → broad). The progression is the calibration spine for graduating
  variants. `broad_legal_repaired` is a **diagnostic upper-bound**,
  NOT the recognition number; `guarded_broad_legal_repaired` adds a
  no-ground-truth cost/changes gate as the first defensible
  promotion-candidate slice. Read
  `tools/HULL_LABEL_LEGAL_REPAIR_DIAGNOSTIC.md` for current exact-match
  counts and per-row hamming.
- **Current scoreboard / promotion gate** — the corpus is now 71 GT pairs.
  `tools/CURRENT_HULL_LABEL_SCOREBOARD.md` summarizes the latest numbers:
  guarded pair-threshold selection is 71/71 exact, and
  `tools/CONSTRAINED_INFERENCE_PROMOTION_GATE.md` applies a GT-free
  production-shaped gate that accepts 71/71 of those candidates. The hidden
  `/api/recognize?hullLabelTier1=constrained` path now validates the same
  candidate at the recognizer boundary: 71/71 exact/legal versus 24/71 exact
  for the unchanged legacy/default path. This is the current default-recognizer
  shadow candidate, not a default flip by itself.

Open levers, approximate descending leverage:

1. **Real-traffic shadow rollout** for the hidden constrained recognizer mode:
   run `/api/recognize?hullLabelTier1=constrained-shadow` beside the legacy
   default. The endpoint now appends compact JSONL events to
   `runs/constrained_inference_shadow.jsonl` (override with
   `CUBE_CONSTRAINED_SHADOW_LOG`, disable with `off`); compare accept/reject,
   method, threshold, and yaw distributions to the 71-pair corpus before
   replacing the legacy default.
2. **Lab + LLM ensemble** for per-sticker color reads.
3. **Confidence-gated auto-merge** of repair variants based on
   inter-variant agreement.
4. **Graduating deterministic repair to the default recognizer** once
   confidence calibration is settled.
5. **Targeted cubie-consistency reclassification** only if new failures
   appear that guarded legal repair cannot reach. The first two-view pull
   is no longer open: the shared cubie map/checker and a guarded
   `two_view_consistency_repaired` candidate codify the 12 split cubies
   (6 corners + 6 edges) visible across A/B and expose that evidence in the
   repair payload. It is a selector over an already legal candidate, not a
   new color reclassification search.

The 2026-05-22 strategic shift still holds: Claude's geometry
research is **scaffolding around** Codex's production `cv-local`,
not a replacement for it. The role split:

- **Claude**: geometry research + diagnostics + benchmark harness + docs. Hull-labels candidate path (`rectify_via_hull_labels`, acceptance gates, shadow-trace analyzer) is Claude's.
- **Codex**: production recognizer + guardrail experiments + corpus/hard-case probes + API/server behavior. Tier 1 wiring into `fit_global_cube_model` is Codex's.
- **Shared**: first-principles docs, labeled fixtures, regression-gate harness, the Tier 1 wiring surface itself.

### Codex owns (production recognizer)

- `rubik_recognizer/*` — production color classifier, recognizer, image pipeline
- `tests/test_white_up_rules.py`, `tests/test_hard_cases.py`, `tests/test_recognizer.py` and other production-recognizer tests
- `tests/fixtures/corpus_manifest.json`, `tests/fixtures/hard_case_manifest.json` — corpus/hard-case baselines
- `tools/probe_corpus.py`, `tools/probe_hard_cases.py` — production contract probes
- `app.py` — the Rubik Two-View Recognizer HTTP server (stdlib `ThreadingHTTPServer`)
- **Production guardrail behavior** (Phase 3 onward): retake/manual-fixer routing, low-trust abstention

### Claude owns (geometry research + tooling)

- `tools/global_cube_model.py` adjacent scripts (`interior_bezel_detection.py`, `render_global_cube_model_v0_overlays.py`, etc.) — but the function itself is now SHARED via the Tier 1 wiring (Codex's #291); coordinate before changing its signature.
- `tools/baseline_post_218.py` — **the regression-gate harness.** Re-generate the committed snapshot when the global model changes.
- **Hull-labels lane** (active focus, 2026-05-23 onwards):
  - `tools/rectify_via_hull_labels.py` — convention-aware silhouette-corner labeling + per-side `SILHOUETTE_TO_CORNER`, hybrid affine/projective vertex selection (#289)
  - `tools/hull_label_acceptance.py` — production-shaped acceptance gates (sticker score, vertex spread, projective residual)
  - `tools/projective_vertex.py` — vanishing-point construction for perspective-heavy cases (#288)
  - `tools/measure_hull_labels_corpus.py` — direct rectification corpus evaluation
  - `tools/analyze_shadow_traces.py` — Tier 1 acceptance-gate corpus evaluation (#292)
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
- `tools/HULL_LABEL_TIER1_WIRING.md` — operator handoff for the Tier 1 feature-flagged candidate path. Modes / gates / trace surface. Owned jointly: Codex wired it (#291), Claude maintains the candidate pipeline + analyzer.
- `tools/SHADOW_TRACE_ANALYSIS.md` — markdown report from `tools/analyze_shadow_traces.py`. Auto-regenerated; commit on substantive corpus changes. (Landed in PR #292.)
- `tools/FULL_CORNER_LABELING.md` / `tools/corner_conventions.py` / `tests/fixtures/full_corner_ground_truth.json` — canonical `Va/Vb + 0..5` corner convention, A/B face outlines, flattened facelet mapping, and seed full-corner truth. Update these before touching any downstream geometry convention. **Schema rename 2026-05-23 (#286):** `near_x/y/z` → `axis_x/y/z` with backward-compat read shim — same FAR-corner positions, naming clarified.
- `tests/fixtures/gcm_axis_ground_truth.json` — user-labeled axis fixture. The label fields now read `axis_x/y/z` (was `near_x/y/z`). Sits at FAR-corner positions per side-specific convention.
- `tests/fixtures/post_218_baseline.json` — legacy accuracy snapshot derived from the axis fixture. Regenerate when global model semantics change.
- `tests/fixtures/hard_case_manifest.json` + `tests/fixtures/corpus_manifest.json` — paired manifests. Both are needed by tools that iterate the 70-row axis-truth corpus: corpus_manifest covers more setIds than the axis truth uses (e.g. it includes sets 63-68 which have no axis labels), hard_case_manifest covers the hard-case sets the axis truth specifically labels (17/21/22/25/30/39/44/46-49/57-58/61-62). Set 44 appears in both; tools should resolve collisions deterministically (e.g. `dict.setdefault` with corpus_manifest first). Tools that consume the axis truth should load both manifests and let `_resolve_image_path`'s pattern-search fallback handle any remaining gaps. (The shadow-trace analyzer is the first tool to follow this pattern explicitly — see PR #292.)
- `tools/extract_color_samples.py` — Claude's, but Codex added the `white[- ]up` regex fix in #135. **Coordinate before editing.**
- `tests/test_auto_geometry_metrics.py` — Claude's, but a growing surface. **Coordinate before adding tests that interact with discovery/geometry.**
- `tools/global_cube_model.py` — was previously Claude-only; PR #291 wired Tier 1 hull-label paths into it. Now joint: Claude owns the hull-label candidate code; Codex owns the dispatcher / feature-flag wiring. Coordinate before changing the public signature of `fit_global_cube_model`.

### Off limits unless explicitly coordinated

- Each other's in-flight branches
- Each other's open PRs (read freely, comment freely, do not commit)
- Git history rewrites on shared branches

---

## In Flight

Update when opening a PR; clear when merged. Keep this current — it's the primary collision avoidance mechanism.

| Owner | Branch | PR | What | Touches | ETA |
|---|---|---|---|---|---|
| Claude | `claude/coordination-refresh` / `claude/readme-current-state` | TBD / TBD | This PR pair: COORDINATION.md current-state refresh (ctvd) + README.md v0.2 description + iOS phase update (cube-snap), catching up to 2026-05-26 reality after the color-repair / legality / scoreboard / threshold-selector wave. Doc-only. | `COORDINATION.md`, `README.md` | open |

### Proposed for Codex (please pick up or push back)

- **Gate calibration in shadow mode against real traffic.** The 70-row corpus shadow-trace analyzer (#292) is the calibration baseline; current acceptance + accepted-with-warnings counts and gate distributions live in `tools/SHADOW_TRACE_ANALYSIS.md` (regenerates with the corpus — read it for current numbers rather than relying on this prose). Before flipping the default from `shadow` → `prefer`: (1) collect shadow traces from production traffic to confirm distribution matches corpus, (2) validate gate accepts ARE the right rows (cross-reference vs ground-truth rectification quality), (3) decide whether the unexercised hard thresholds on `projective_residual_norm` and `sticker_score_total` need calibration against synthetic bad cases or wait for real-traffic data.

*(Either side: populate your row when you start something.)*

---

## Recently Shipped

Last 5 per side. Newest first. One line + PR # + the takeaway.

### Claude

- **#328** — Use flipped B-side prefill in full-corner labeling gallery. Side B prefill now matches the corner order a human labeler sees in the 180°-rotated image (corner_0 at bottom-of-image, etc.); Side A defaults preserved. Pure labeling-tool ergonomics; production recognizer untouched.
- **#327 / cube-snap #188** — Document audit-wrapper loud-failure contract (defensive comments at `post_pr_comment` call sites) + new CLAUDE.md sections in both repos codifying the 2026-05-26 constrained-cube-state-inference shift. The wrapper change is mirror-invariant; the CLAUDE.md updates are repo-specific.
- **#326** — Legality repair probe on hull-label corpus. Conservative variant at 43/46 exact; broad variant at 46/46 exact (diagnostic upper-bound, NOT the production recognition number). Probe-only — no production wire-up.
- **#325** — Post-repair scoreboard on 46-pair shadow corpus across 5 repair variants. Establishes the 0/46 → 20/46 → 42/46 → 43/46 → 46/46 progression as the calibration spine for graduating variants.
- **#324** — Expose hull-label color repair draft as a deterministic API surface on the rectified samples. Foundation for #325 scoreboard + #326 legality probe.
- **#323 / cube-snap #186** — Fix `run_codex_audit_pr.sh` silently swallowing Python failures. Wrapper's `finish_lock` was logging `status="completed"` on every run because `local rc; rc=$?` captures the exit of `local`, not the prior command. Structural fix; #327/#188 added defensive comments at the Python call sites.
- **#321** — Claude session-start queue sweep hook (mirrored as cube-snap #185). On session resume, sweep both repos for `needs-claude-review` PRs and re-arm the audit-log monitor; closes the "monitor died, missed PR" failure mode.
- **#316** — Label rectified LLM cells by WCA facelet. Center-color yaw inference + per-cell WCA labels in the rectified contact sheet handed to the cloud LLM.
- **#315** — Harden rectified LLM yaw prep. Edge cases in the yaw inference path that previously silently produced mis-oriented contact sheets now raise or fall back deterministically.
- **#292** — Shadow-trace analyzer for Tier 1 hull-label gates. `tools/analyze_shadow_traces.py` + `tools/SHADOW_TRACE_ANALYSIS.md` baseline + `tests/fixtures/shadow_trace_corpus.json` artifact. Headline: 69/70 accept, 1/70 reject (30_A bad-hull, caught by `projective_residual_norm`).
- **#289** — Hybrid affine/projective vertex switch in `rectify_via_hull_labels` (normalized threshold + projective_residual_norm bad-input gate). Resolution-independent: `vertex_cloud_spread_norm > 0.26` switches to projective vertex. 69/70 corpus rectifications clean (was 68/70); 37_B recovered; 30_A still flagged via new `projective_residual_norm` hard gate (correct — bad hull input).
- **#288** — Projective vertex via vanishing-point construction (diagnostic). `tools/projective_vertex.py` computes the analytically-correct cube-vertex from parallel-edge intersections; preserved as diagnostic before the #289 hybrid wiring decided when to use it.
- **#286** — Schema rename `near_x/y/z` → `axis_x/y/z` in axis-truth + doc. Reader shims in 4 places preserve backward-compat; positions unchanged. Removes the misnomer that "near" labels actually sit at FAR-corner positions per side convention.
- **#284** — Lazy rembg session init in `measure_hull_labels_corpus.main`. Avoids ImportError on clean installs when no rows resolve. Codex P1.
- **#282** — Hull-labels rectification: 70-row corpus validation. 68/70 essentially-oracle-quality rectifications; failure taxonomy (mask vs label vs vertex-cloud vs sticker-score). Sets up the gate-design work that became `tools/hull_label_acceptance.py` (PR #287).
- **#277/#278** — Hull-labels rectification standalone pipeline (`tools/rectify_via_hull_labels.py`): convention-aware silhouette → 6 corners → 3 face quads with per-side `SILHOUETTE_TO_CORNER` mapping derived from `FACE_DEFS_BY_SIDE`. Initial 12/12 oracle-quality on the full-corner corpus.
- **#275** — Production-vs-oracle contact sheet for 12 oracle rows. Visual diagnostic that motivated the hull-labels pivot.
- **#271** — Bezel-alignment second-signal chirality disambiguator (V4 with post-ensemble scoring). Production fix for Mode A near_far_phase failures.

### Infra (Devin-authored, mirrored across both repos)

- **ctvd#144 / cube-snap#135** — Audit watchdog: 5-min `schedule` cron in `devin-audit-bridge.yml` re-scans open PRs with `needs-devin-audit`; head-SHA dedupe prevents re-pinging Devin. Catches missed event-driven dispatches.

### Codex

- **#200** — Two-view geometry consistency signal (diagnostics-only). Adds CV-local provenance diagnostics in the recognizer and evaluator. Superseded for production accuracy by the later split-cubie repair selector (`two_view_consistency_repaired`), which is now part of constrained inference.
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

- **2026-05-27** — **Default-recognizer question reframed as a promotion gate.** `tools/CONSTRAINED_INFERENCE_PROMOTION_GATE.md` evaluates the guarded pair-threshold constrained-inference candidate with no GT in the decision and GT only for scoring: 71/71 candidates accepted, legal, and exact on the current GT corpus. This is strong evidence for a recognizer shadow/candidate mode, but production default should wait for the same gate to run in `/api/recognize` with explicit fallback behavior when it rejects.

- **2026-05-26** — **Architectural reframe: constrained cube-state inference, not LLM-as-oracle.** Codified in both repos' `CLAUDE.md` (ctvd #327 / cube-snap #188). The recognizer's job is to combine evidence sources (LLM color reads + hull-label rectified samples + deterministic color/legality repair + two-view split-cubie consistency) under cube-state constraints — not to clean up an LLM oracle. Empirical state on the 46-pair shadow corpus: a graduated set of repair variants (canonical-count → conservative-legal → guarded-broad → broad) covers the climb from a stuck baseline up to a `broad_legal_repaired` upper-bound. The upper-bound is a diagnostic ceiling, NOT the recognition number; the guarded slice is the first defensible promotion candidate. See `tools/HULL_LABEL_LEGAL_REPAIR_DIAGNOSTIC.md` for current exact-match counts. The next open accuracy levers are Lab+LLM ensemble, confidence-gated auto-merge, and targeted cubie-consistency reclassification for failures that the selector can localize but not repair.

- **2026-05-26** — **Color-repair / legality / scoreboard wave landed.** #324 exposes the deterministic color-conflict repair API on rectified hull-label samples; #325 establishes the multi-variant scoreboard as the calibration spine; #326 + #329 add the legality and guarded-broad probes; Codex #322 lands the production-shape mask threshold selector. Production recognizer default unchanged; these are the building blocks behind the constrained-inference reframe. Current scoreboard exact-match counts live in `tools/HULL_LABEL_LEGAL_REPAIR_DIAGNOSTIC.md`.

- **2026-05-25** — **Hull-labels Tier 1 candidate path landed feature-flagged in `fit_global_cube_model` (Codex #291, default off).** Three modes (`off` / `shadow` / `prefer`) controlled by `CUBE_RECOGNIZER_HULL_LABEL_TIER1` env var. Operator handoff in `tools/HULL_LABEL_TIER1_WIRING.md`. The candidate path uses Claude's `tools/rectify_via_hull_labels.py` + acceptance gates; the dispatcher / feature flag is Codex's. **Empirical floor numbers pending PR #292** (shadow-trace analyzer, open at time of writing — first run shows 69/70 accept + 1/70 reject on 30_A bad-hull; will be cited here once #292 merges). **Next: gate calibration in shadow mode against real traffic before flipping `shadow → prefer` default.**

- **2026-05-25** — **iOS scoping doc landed (cube-snap PR #158)** — `IOS_APP_SCOPING.md` captures Q1-Q5 answers and 4-lane multi-agent build plan. v1: native SwiftUI, cloud-only recognizer, SceneKit, $0.99 paid, iPhone-only. v1.5: iPad + Mac Catalyst. v2: on-device recognizer + RealityKit/ARKit re-evaluate. 12-17 weeks for full v1 vs 3-6 weeks for ultra-MVP TestFlight learning vehicle. **Not yet started** — scoping captured so implementation can begin cleanly when prioritized.

- **2026-05-25** — **Fixer Tier 1 trace UI design doc (cube-snap PR #159)** — `FIXER_TIER1_TRACE_DESIGN.md` captures speculative design for cube-snap-side Fixer surfacing of the Tier 1 hull-label shadow trace. 4-phase rollout (shadow-only read → side-by-side previews → gate explainability → guided-capture handoff). Design only; no implementation. First concrete next-action is verifying whether cv-local actually calls `fit_global_cube_model` with `hull_label_mode` set.

- **2026-05-24** — **Hybrid affine/projective vertex switch (#289)** — `vertex_cloud_spread / hexagon_diameter > 0.26` switches to projective vertex; below threshold, affine averaging wins. Resolution-independent normalization (was raw px, was scale-dependent in prior #285 version). Hull-labels corpus: 69/70 clean (was 68/70 pure affine, was 38/70 regression on pure projective). 37_B recovered (vertex_err 80 → 38 px). 30_A still flagged, but via the NEW `projective_residual_norm` bad-input gate (residual 0.0315 > 0.025 hard).

- **2026-05-24** — **Schema rename `near_x/y/z` → `axis_x/y/z` (#286)** — the 70-row labels never actually sat at NEAR-corner positions; they sit at FAR-corner positions per side-specific convention. Rename removes the misnomer; backward-compat read shim in 4 readers (`baseline_cv_local`, `baseline_post_218`, `measure_hull_labels_corpus`, `evaluate_axis_ground_truth`) preserves legacy fixtures.

- **2026-05-23** — **Hull-labels rectification corpus floor: 12/12 oracle-quality, 68/70 on the larger axis corpus (#282).** First-principles approach: extract 6 silhouette extrema, label by image-position convention per side, complete parallelograms to get vertex + 3 face quads. Bypasses Procrustes/PnP/chirality entirely on the cases where it works. Failure taxonomy: mask_failure / label_failure / vertex_cloud_high_spread / axis_misfit_high / sticker_score_high / rectified_clean.

- **2026-05-24** — **Paid review lanes are final-confirmation gates, not iterative defaults.** Codex + Claude do normal iterative cross-review; apply `needs-greptile-audit` / `needs-devin-audit` only when a PR is stable and risk justifies paid review. Greptile is repo-configured to review only PRs carrying `needs-greptile-audit`.
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

Before opening any PR or requesting paid audit:

- [ ] `git fetch origin main && git rebase origin/main` — guarantees `mergeable: MERGEABLE`. Force-push with `--force-with-lease` after rebase.
- [ ] `gh pr list --state open` — check no other in-flight PR touches the same files. If overlap, coordinate via this doc.
- [ ] Full `pytest` green locally. New tests for new behavior.
- [ ] If touching anything in **Shared** above: mention in the PR description.
- [ ] If the PR's results depend on a long-running sweep: post the full `runs/*_summary.txt` as a PR comment so reviewers don't need to re-run.
- [ ] For Claude: tools-only, no `rubik_recognizer/*` edits. For Codex: production-only, no edits to Claude-owned auto-geometry / rectify / mask-path tooling listed above unless coordinated.
- [ ] **Geometry regression gate.** If the PR touches `tools/global_cube_model.py` or any downstream behavior (cv-local recognition pipeline, mask-path, rectify), run `tools/baseline_post_218.py --diff tests/fixtures/post_218_baseline.json /tmp/your_new_baseline.json` and paste the row-level diff into the PR body. Aggregate metrics alone are insufficient — per Devin, "some changes improve averages while worsening critical rows." A PR that regresses any case from GOOD → catastrophic without offsetting wins is a merge blocker.
- [ ] **Stacked PRs.** If your PR depends on an unmerged PR, branch from the parent's branch (not main) and document the stack in the PR body ("Stacks on #NNN"). Merge order is then forced and reviewers can audit the diff against the parent rather than against main.
- [ ] **Paid review gate.** Do not apply `needs-devin-audit` or `needs-greptile-audit` during normal iteration. First get the PR stable with local validation plus Codex/Claude cross-review. Then apply a paid-review label only if the PR is production-risky, behavior-changing, cross-repo infrastructure, security/privacy-sensitive, or otherwise worth a final external check. Tiny docs/comment-only PRs normally need no paid review.
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
