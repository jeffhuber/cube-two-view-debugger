# State of the world (2026-05-22)

> One-page map of the project. Start here.

## What this project is

Real-camera Rubik's cube state recognition from **two isometric
photos** (A: white-up showing U+R+F; B: yellow-up showing D+L+B after
180° rotation). Two sister repos:

- **`jeffhuber/cube-snap`** — the web app + production recognizer
  pipeline.
- **`jeffhuber/cube-two-view-debugger`** (this repo) — the research /
  evaluation / diagnostics workbench. Hosts experimental geometry
  tooling, hard-case probes, and the labeled ground-truth fixtures.

## Production architecture (current main, 2026-05-22)

The end-to-end production path lives in cube-snap:

```text
two photos → cv-local recognizer (rubik_recognizer/*) → 54-char state → solver
```

`cv-local` is the **primary recognizer**. It uses face-quad detection
+ rectification + per-sticker color classification, with hull
validation and grid-extrapolation guards. End-to-end per-sticker
accuracy on the labeled corpus is ~82–83%.

The research pipeline in this repo runs in parallel — it does NOT
replace cv-local. Its current role is scaffolding around cv-local:

```text
two photos → cv-local (primary) → state
                ↓
           global cube model (research) → trust signals, future learned ranker
```

See [`FIRST_PRINCIPLES_RECOGNIZER_DESIGN.md`](FIRST_PRINCIPLES_RECOGNIZER_DESIGN.md)
for the north-star architecture and the policy bar that gates first-
principles work.

## Where to find things

### If you want to understand…

| …then read | …because |
|---|---|
| **The current architecture + roadmap** | `STATE_OF_THE_WORLD.md` (this file) |
| **The target architecture** | [`FIRST_PRINCIPLES_RECOGNIZER_DESIGN.md`](FIRST_PRINCIPLES_RECOGNIZER_DESIGN.md) |
| **What works and what doesn't, with numbers** | [`POST_218_BASELINE_AND_TAXONOMY.md`](POST_218_BASELINE_AND_TAXONOMY.md) |
| **The categorization of failure modes** | [`FAILURE_TAXONOMY.md`](FAILURE_TAXONOMY.md) |
| **Which fixture/report answers which question** | [`BENCHMARK_INDEX.md`](BENCHMARK_INDEX.md) |
| **The "chirality" / near-far phase issue** | [`NEAR_FAR_PHASE_REPORT.md`](NEAR_FAR_PHASE_REPORT.md) |
| **The global cube model implementation** | [`GLOBAL_CUBE_MODEL.md`](GLOBAL_CUBE_MODEL.md) |
| **Coordination between Claude and Codex** | `../COORDINATION.md` |
| **Status of each script under `tools/`** | [`README.md`](README.md) |

### Key durable assets

| Asset | Path | What it is |
|---|---|---|
| **Axis-labeled ground truth** | `tests/fixtures/gcm_axis_ground_truth.json` | 58 user-labeled photos: vertex + 3 near corners per photo. THE eval set + future training data. |
| **Post-#218 baseline snapshot** | `tests/fixtures/post_218_baseline.json` | Current-main accuracy snapshot. Regression gate for geometry-sensitive PRs. |
| **Benchmark harness** | `tools/baseline_post_218.py` | Runs the eval, supports `--diff` mode for row-level deltas. |
| **Production recognizer (cv-local)** | `cube-snap` repo, `rubik_recognizer/*` | The primary system. |
| **Global cube model (research)** | `tools/global_cube_model.py` | Scaffolding around cv-local. NOT a replacement. |

## Phased roadmap

| Phase | Status | What it produces | Success criterion |
|---|---|---|---|
| **0 — Consolidate** | **In flight (this PR)** | Updated docs: this file, FIRST_PRINCIPLES_RECOGNIZER_DESIGN, FAILURE_TAXONOMY, BENCHMARK_INDEX, tools/README, COORDINATION. | All 6 docs land; no code changes. |
| **1 — Re-baseline both sides** | Pending | Post-#218 global model snapshot (done; `tests/fixtures/post_218_baseline.json`) + matching cv-local snapshot. Documented gap between them. | Two committed JSON snapshots, both runnable via `--diff`. |
| **2 — Trust policy diagnostics** | Pending | New `model.debug` fields: phase confidence, axis-vs-bezel agreement, two-view consistency. Diagnostics-only, no behavior change. | Per-case "would-have-routed-to-retake" prediction logged. Diagnostic confidence correlates >0.6 with position-truth on 58 cases. |
| **3 — Guardrail experiment** | Pending | Production behavior change: low-trust cases route to retake/manual-fixer. First phase where production behavior changes. | Confident-wrong rate drops without abstention >15% (or agreed budget). Tracked in `--diff`. |
| **4 — Learned geometry** | Pending | Trained vertex/axis/phase ranker on 58+ labels with held-out splits + calibrated abstention. | Held-out test accuracy + abstention curves match or beat Phase-3 heuristic. |
| **5 — Better capture / UX** | Pending | Phase-3 diagnostics drive retake instructions / manual fixer rather than forcing repair. | Reduction in user-reported confident-wrong rate (production metric). |

## How to contribute a change

Before opening a geometry-sensitive PR (anything that could affect
`tools/global_cube_model.py` outputs, or the cv-local equivalent):

1. Run the regression gate:
   ```bash
   .venv/bin/python tools/baseline_post_218.py \
     --out /tmp/baseline_my_branch.json --report /dev/null
   .venv/bin/python tools/baseline_post_218.py \
     --diff tests/fixtures/post_218_baseline.json /tmp/baseline_my_branch.json
   ```
2. Paste the diff summary into the PR body.
3. Confirm no GOOD → catastrophic regressions without offsetting wins.

See [`POST_218_BASELINE_AND_TAXONOMY.md`](POST_218_BASELINE_AND_TAXONOMY.md)
"How to use this as a regression gate" for details.

## Recently shipped (chirality / phase work, May 2026)

| PR | What |
|---|---|
| #210 | Chirality detection diagnostic (diagnostic-only). |
| #213 | Enabled auto-correction with empirically-validated polarity. |
| #218 | Reordered: vertex ensemble BEFORE phase check. +33pp accuracy. |
| #220 | Post-#218 baseline + failure taxonomy (decision spine). |
| #221 | Mechanical rename: chirality → near_far_phase. |
| #140, #219 | Documented standing in-thread merge delegation. |

## Open questions that need user judgment (not Claude/Codex)

These are deliberately surfaced here so they don't get lost:

- **Labeling cadence**: when do we extend the 58-case set? Active-
  label queues (based on current-model disagreement) need user time.
- **Abstention budget**: what fraction of solves can route to retake
  before UX degrades unacceptably? This sets Phase 3's success bar.
- **Capture-flow product changes**: Phase 5 implies UX work (retake
  prompts, manual-fixer flow). Out of scope for first-principles
  geometry work alone.
