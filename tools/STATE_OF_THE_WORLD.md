# State of the world (2026-05-30)

> One-page map of the project. Start here.

## What this project is

Real-camera Rubik's cube state recognition from **two isometric
photos** (A: white-up showing U+R+F; B: yellow-up showing D+L+B after
180° rotation). Two sister repos:

- **`jeffhuber/cube-snap`** — the web app (live at
  [cubesnap.app](https://cubesnap.app/)) + React frontend + iOS app
  scaffolding.
- **`jeffhuber/cube-two-view-debugger`** (this repo) — the Python
  recognizer backend (deployed on Railway at `api.cubesnap.app`),
  plus the research / evaluation / diagnostics workbench, ground-truth
  corpus, and diagnostic tools.

## Production architecture (2026-05-29)

The end-to-end production path:

```text
two photos → constrained cube-state inference (this repo, app.py on Railway)
           → 54-char URFDLB state → Kociemba solver (cube-snap, client-side)
```

The **constrained recognizer** is the production default. It uses
hull-label rectification → rembg silhouette masking → per-side
threshold selection → deterministic color/count/legality repair →
guarded pair-threshold search → solver validation. The pipeline
treats LLM color reads and Lab-distance classification as evidence
sources in a constrained solver, not as the source of truth.

**Accuracy**: 71/71 exact on the 71-pair ground-truth corpus.
Legacy cv-local on the same corpus: 24/71 exact.

**Latency**: server-side p50 ~2.1s, p90 ~2.2s (Railway deployment).
Bottleneck is rembg (background removal via U2Net), which accounts
for ~76% of p50.

The legacy cv-local path (`rubik_recognizer/*`) is retained for
compatibility and debugging but is no longer the default production
path. The Vercel LLM endpoint is an availability fallback.

See both repos' `CLAUDE.md` for the constrained-inference architecture
write-up and `COORDINATION.md` for the decision log.

## Where to find things

### If you want to understand…

| …then read | …because |
|---|---|
| **The current architecture + roadmap** | `STATE_OF_THE_WORLD.md` (this file) |
| **The target architecture** | [`FIRST_PRINCIPLES_RECOGNIZER_DESIGN.md`](FIRST_PRINCIPLES_RECOGNIZER_DESIGN.md) |
| **What works and what doesn't, with numbers (global model side)** | [`POST_218_BASELINE_AND_TAXONOMY.md`](POST_218_BASELINE_AND_TAXONOMY.md) |
| **What works and what doesn't, with numbers (cv-local side)** | [`PHASE_1_CV_LOCAL_BASELINE.md`](PHASE_1_CV_LOCAL_BASELINE.md) |
| **The categorization of failure modes** | [`FAILURE_TAXONOMY.md`](FAILURE_TAXONOMY.md) |
| **Which fixture/report answers which question** | [`BENCHMARK_INDEX.md`](BENCHMARK_INDEX.md) |
| **The "chirality" / near-far phase issue** | [`NEAR_FAR_PHASE_REPORT.md`](NEAR_FAR_PHASE_REPORT.md) |
| **The explicit corner/facelet convention** | [`FULL_CORNER_LABELING.md`](FULL_CORNER_LABELING.md) |
| **The global cube model implementation** | [`GLOBAL_CUBE_MODEL.md`](GLOBAL_CUBE_MODEL.md) |
| **Coordination between Claude and Codex** | `../COORDINATION.md` |
| **Status of each script under `tools/`** | [`README.md`](README.md) |

### Key durable assets

| Asset | Path | What it is |
|---|---|---|
| **Ground-truth corpus manifest** | `tests/fixtures/corpus_manifest.json` | 71-pair corpus with image paths, ground-truth states, and per-set metadata. |
| **Hard-case manifest** | `tests/fixtures/hard_case_manifest.json` | Hard-case subset for focused regression testing. |
| **Full-corner convention** | `tools/FULL_CORNER_LABELING.md` | Canonical `Va/Vb + 0..5` convention, A/B face outlines, flattened facelet mapping. |
| **Full-corner truth** | `tests/fixtures/full_corner_ground_truth.json` | Canonical full-corner fixture covering sets 20, 38, 40, 41, 43, 45 plus tail/stress sets 63-73. |
| **Hull-label scoreboard** | `tools/CURRENT_HULL_LABEL_SCOREBOARD.md` | Current constrained-inference accuracy snapshot: 71/71 exact. |
| **Promotion gate** | `tools/CONSTRAINED_INFERENCE_PROMOTION_GATE.md` | GT-free production-shaped gate evaluation: 71/71 accepted. |
| **Latency plan** | `tools/CONSTRAINED_RECOGNIZER_LATENCY_PLAN.md` | Deployed bottleneck analysis and optimization roadmap. |
| **Production recognizer** | `app.py` (this repo, Railway) | Constrained inference endpoint at `api.cubesnap.app`. |
| **Legacy cv-local** | `rubik_recognizer/*` | Retained for compatibility; no longer the default. |
| **Global cube model** | `tools/global_cube_model.py` | Hull-label pipeline foundation; now wired into constrained inference. |

## Phased roadmap

| Phase | Status | What it produces | Success criterion |
|---|---|---|---|
| **0 — Consolidate** | ✅ Done (#222) | Docs: this file, FIRST_PRINCIPLES_RECOGNIZER_DESIGN, FAILURE_TAXONOMY, BENCHMARK_INDEX, tools/README, COORDINATION. | All 6 docs landed; no code changes. |
| **1 — Re-baseline both sides** | ✅ Done. | Legacy global/cv-local snapshots remain historical. Canonical seed global-model snapshot now uses `tests/fixtures/full_corner_ground_truth.json`. | Baselines established. |
| **2 — Trust policy diagnostics** | ✅ Done. | Hand-tuned rules couldn't simultaneously clear both recall and FPR bars. Pivoted to constrained inference architecture instead. | Completed; superseded by constrained inference. |
| **3 — Guardrail experiment** | Superseded. | Constrained inference achieves 71/71 exact, eliminating the need for guardrail-based routing. | N/A — superseded. |
| **4 — Constrained cube-state inference** | ✅ **Shipped as production default (2026-05-29).** | Hull-label rectification + deterministic repair + guarded pair-threshold selection. Deployed on Railway at `api.cubesnap.app`. | 71/71 exact on GT corpus. Server p50 ~2.1s. |
| **5 — Latency optimization** | **In progress.** | rembg cost reduction, guarded-pair tail capping, client image pre-resize. | Target: sub-2s server p50. See `CONSTRAINED_RECOGNIZER_LATENCY_PLAN.md`. |

## How to contribute a change

Before opening a geometry-sensitive PR (anything that could affect
`tools/global_cube_model.py` outputs, or the cv-local equivalent):

1. Prefer the canonical seed gate when the touched behavior can be scored on
   full-corner truth:
   ```bash
   .venv/bin/python tools/baseline_full_corner_global_model.py \
     --out /tmp/full_corner_baseline_my_branch.json \
     --report /tmp/full_corner_baseline_my_branch.md
   ```
2. For legacy 58-case comparisons, run the provisional gate:
   ```bash
   .venv/bin/python tools/baseline_post_218.py \
     --out /tmp/baseline_my_branch.json --report /dev/null
   .venv/bin/python tools/baseline_post_218.py \
     --diff tests/fixtures/post_218_baseline.json /tmp/baseline_my_branch.json
   ```
3. Paste the diff summary into the PR body.
4. Confirm no GOOD → catastrophic regressions without offsetting wins.

See [`POST_218_BASELINE_AND_TAXONOMY.md`](POST_218_BASELINE_AND_TAXONOMY.md)
"How to use this as a regression gate" for details.

## Recently shipped (May 2026)

| PR | What |
|---|---|
| #396 | Parallelize constrained hull fitting (latency). |
| #395 | Refresh recognizer architecture docs. |
| #393 | Durable recognition event logging (Railway production). |
| #392 | Fast-reject obvious non-cube constrained inputs. |
| #388 | Short-circuit guarded pair canonical search (latency). |
| #385 | Guarded Railway deploy script. |
| #384 | Speed constrained recognize path. |
| #380 | Railway deployment configuration. |
| #355 | Parity check speedup (O(n) from O(n²)). |

## Open questions that need user judgment (not Claude/Codex)

These are deliberately surfaced here so they don't get lost:

- **rembg replacement**: Is a cheaper silhouette estimator worth
  the risk on edge cases? Could cut p50 from ~2.1s to sub-1s.
- **Real-traffic telemetry review**: The durable event log exists;
  when to start systematic review of real-user failure patterns?
- **Hard-case corpus expansion**: When to add more sets beyond the
  current 71-pair corpus? Edge cases from real traffic may reveal
  gaps.
- **Capture-flow product changes**: Better retake instructions,
  guided-capture UX. Out of scope for first-principles
  geometry work alone.
