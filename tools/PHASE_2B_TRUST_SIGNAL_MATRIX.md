# Phase 2B: trust-signal matrix

**Status: diagnostics-only.** No production behavior change. This report evaluates candidate trust rules against the Phase 2 bar:

- catastrophic recall ≥ **80%**
- GOOD false-retake ≤ **10%**

> **Legacy-data caution (2026-05-23):** this report joins fixtures derived
> from the legacy `near_*` axis labels. Treat row-level phase/chirality
> categories as provisional until regenerated from `Va/Vb + 0..5`
> full-corner truth.

## Dataset

- 116 per-case-per-run rows across 58 cases
- Outcome breakdown: **76 GOOD**, **16 MARGINAL**, **24 CATASTROPHIC**

Joined from `tests/fixtures/post_218_baseline.json` (per-run phase_sep + outcome category) and `tests/fixtures/cv_local_baseline.json` (per-case cv-local face-quad structural status).

## Candidate rule evaluation

| rule | description | recall | GOOD FPR | MARGINAL routed | meets bar? |
|---|---|---|---|---|---|
| `phase_sep_alone_T11.7` | Retake when |phase_sep| < 11.7 (Phase 2A operating point). | 45.8% (11/24) | 9.2% (7/76) | 37.5% (6/16) | ❌ |
| `phase_sep_alone_T0.5` | Retake when |phase_sep| < 0.5. | 0.0% (0/24) | 0.0% (0/76) | 0.0% (0/16) | ❌ |
| `phase_sep_alone_T2.0` | Retake when |phase_sep| < 2.0. | 0.0% (0/24) | 0.0% (0/76) | 12.5% (2/16) | ❌ |
| `phase_sep_alone_T5.0` | Retake when |phase_sep| < 5.0. | 12.5% (3/24) | 1.3% (1/76) | 18.8% (3/16) | ❌ |
| `phase_sep_alone_T8.0` | Retake when |phase_sep| < 8.0. | 29.2% (7/24) | 5.3% (4/76) | 25.0% (4/16) | ❌ |
| `phase_sep_alone_T15.0` | Retake when |phase_sep| < 15.0. | 50.0% (12/24) | 19.7% (15/76) | 50.0% (8/16) | ❌ |
| `phase_sep_alone_T20.0` | Retake when |phase_sep| < 20.0. | 66.7% (16/24) | 30.3% (23/76) | 56.2% (9/16) | ❌ |
| `cv_local_alone` | Retake when cv-local face-quad fit is NOT structurally consistent (status != 'ok'). | 100.0% (24/24) | 85.5% (65/76) | 93.8% (15/16) | ❌ |
| `phase_or_cv_T8.0` | Retake when |phase_sep| < 8.0 OR cv-local NOT consistent. | 100.0% (24/24) | 85.5% (65/76) | 100.0% (16/16) | ❌ |
| `phase_or_cv_T11.7` | Retake when |phase_sep| < 11.7 OR cv-local NOT consistent. | 100.0% (24/24) | 85.5% (65/76) | 100.0% (16/16) | ❌ |
| `phase_or_cv_T15.0` | Retake when |phase_sep| < 15.0 OR cv-local NOT consistent. | 100.0% (24/24) | 86.8% (66/76) | 100.0% (16/16) | ❌ |
| `phase_and_cv_T8.0` | Retake when |phase_sep| < 8.0 AND cv-local NOT consistent. | 29.2% (7/24) | 5.3% (4/76) | 18.8% (3/16) | ❌ |
| `phase_and_cv_T11.7` | Retake when |phase_sep| < 11.7 AND cv-local NOT consistent. | 45.8% (11/24) | 9.2% (7/76) | 31.2% (5/16) | ❌ |
| `phase_and_cv_T15.0` | Retake when |phase_sep| < 15.0 AND cv-local NOT consistent. | 50.0% (12/24) | 18.4% (14/76) | 43.8% (7/16) | ❌ |
| `cv_severe_alone` | Retake when cv-local status is `fewer_than_3_face_quads` (the more severe failure mode — geometry couldn't even find 3 faces). | 33.3% (8/24) | 18.4% (14/76) | 12.5% (2/16) | ❌ |
| `phase_or_cv_severe_T8.0` | Retake when |phase_sep| < 8.0 OR cv-local is `fewer_than_3_face_quads`. | 45.8% (11/24) | 23.7% (18/76) | 31.2% (5/16) | ❌ |
| `phase_or_cv_severe_T11.7` | Retake when |phase_sep| < 11.7 OR cv-local is `fewer_than_3_face_quads`. | 54.2% (13/24) | 25.0% (19/76) | 43.8% (7/16) | ❌ |
| `phase_or_cv_severe_T15.0` | Retake when |phase_sep| < 15.0 OR cv-local is `fewer_than_3_face_quads`. | 54.2% (13/24) | 31.6% (24/76) | 50.0% (8/16) | ❌ |

## Headline finding

**No rule over the currently-available signals (phase_sep + cv-local structural status, alone or combined) meets the Phase 2 bar.**

Closest-to-bar rule: `phase_sep_alone_T20.0`
- catastrophic recall: 66.7% (bar: 80%; shortfall 13.3%)
- GOOD false-retake:   30.3% (bar: 10%; excess 20.3%)

## Implications

1. **cv-local structural consistency alone is too aggressive.** It catches 100% of catastrophic but trips on 85% of GOOD cases too (Phase 1 already found 90% structural-fit-fail rate; that result weakens cv-local as a retake gate — most PRs would be retaken).

2. **phase_sep alone confirms Phase 2A's ceiling.** No threshold sweep over phase_sep clears the bar; pushing recall up to ~70% drives GOOD FPR past 30%.

3. **OR-compounds inherit cv-local's noise**; AND-compounds inherit phase_sep's weakness. Neither composition direction with these two signals alone clears the bar.

## Next signals to add (per Codex's spec)

This PR's matrix is signal-light by design — it tests whether the cheap-to-compute existing signals suffice. They don't. To make Phase 2B's verdict actionable, the next iteration should extend the matrix with:

- **`fit_residual_rms_px`** (continuous, per-run): the global model's affine/PnP residual is already stored in `model.debug['fit_residual_rms_px']`. Re-running the global model across the 58-case axis-labeled gallery captures this.
- **`vertex_ensemble_stddev_px`** (continuous, per-run): disagreement across the 3-vertex ensemble (hexagon-PnP, bezel-detection, image refinement). Currently aggregated into a mean inside the global model; needs exposing per-component.
- **`two_view_consistency_deg`** (continuous, per-set): pair-wise A↔B orientation agreement. Requires running both photos of a set through the model and comparing the inferred axis bearings. Two-view consistency is the architectural lever the cube-snap product UI relies on.

Implementation hook: `tools/phase2b_trust_matrix.py --recompute-global-model` is reserved for this extension. The matrix schema is already shaped for the additional fields (currently null).

## Conditional pivot (per Codex)

> If Phase 2B finds a rule that meets the bar, then Phase 3 becomes straightforward: wire it as a conservative guardrail. If it does not, we will have clean evidence to pivot to learned geometry or capture/UX instead of hand-tuning another scalar.

The current matrix (existing signals only) is insufficient. Before triggering the pivot to learned-geometry or capture-UX, extend with `--recompute-global-model` to fold in the three signals above. If those still don't clear the bar with any rule composition, the pivot decision becomes evidence-backed.

## See also

- `tools/PHASE_2A_PHASE_CONFIDENCE_CALIBRATION.md` — solo phase_sep calibration (this PR's starting point).
- `tools/PHASE_1_CV_LOCAL_BASELINE.md` — cv-local structural-fit baseline (this PR's second signal).
- `tools/POST_218_BASELINE_AND_TAXONOMY.md` — outcome categorization (the labels this PR predicts against).
- `tools/STATE_OF_THE_WORLD.md` — phased roadmap (Phase 0 → Phase 5).
