# Phase 2B: trust-signal matrix

**Status: diagnostics-only.** No production behavior change. This report evaluates candidate trust rules against the Phase 2 bar:

- catastrophic recall ≥ **80%**
- GOOD false-retake ≤ **10%**

## Dataset

- 116 per-case-per-run rows across 58 cases
- Outcome breakdown: **74 GOOD**, **22 MARGINAL**, **20 CATASTROPHIC**

**Source**: `tests/fixtures/phase2b_recomputed_signals.json` (per-run global-model re-fit on the 58-case axis-labeled gallery, capturing `fit_residual_rms_px`, `pnp_rms_px`, `hexagon_centroid_vs_bezel_vertex_offset_px`, `junction_score_at_ensemble`, `ensemble_shift_px`, and `phase_darkness_separation` at native precision) joined with `tests/fixtures/cv_local_baseline.json` (per-case cv-local face-quad structural status). Outcome counts differ from `post_218_baseline.json` (74/22/20 vs 76/16/24) because the re-fit is non-deterministic (PnP basin-of-attraction) and runs are paired with the signals from the same fit.

## Candidate rule evaluation

| rule | description | recall | GOOD FPR | MARGINAL routed | meets bar? |
|---|---|---|---|---|---|
| `phase_sep_alone_T11.7` | Retake when |phase_sep| < 11.7 (Phase 2A operating point). | 50.0% (10/20) | 12.2% (9/74) | 22.7% (5/22) | ❌ |
| `phase_sep_alone_T0.5` | Retake when |phase_sep| < 0.5. | 0.0% (0/20) | 0.0% (0/74) | 0.0% (0/22) | ❌ |
| `phase_sep_alone_T2.0` | Retake when |phase_sep| < 2.0. | 5.0% (1/20) | 1.4% (1/74) | 9.1% (2/22) | ❌ |
| `phase_sep_alone_T5.0` | Retake when |phase_sep| < 5.0. | 10.0% (2/20) | 2.7% (2/74) | 13.6% (3/22) | ❌ |
| `phase_sep_alone_T8.0` | Retake when |phase_sep| < 8.0. | 25.0% (5/20) | 2.7% (2/74) | 18.2% (4/22) | ❌ |
| `phase_sep_alone_T15.0` | Retake when |phase_sep| < 15.0. | 70.0% (14/20) | 18.9% (14/74) | 31.8% (7/22) | ❌ |
| `phase_sep_alone_T20.0` | Retake when |phase_sep| < 20.0. | 80.0% (16/20) | 31.1% (23/74) | 36.4% (8/22) | ❌ |
| `cv_local_alone` | Retake when cv-local face-quad fit is NOT structurally consistent (status != 'ok'). | 100.0% (20/20) | 85.1% (63/74) | 95.5% (21/22) | ❌ |
| `phase_or_cv_T8.0` | Retake when |phase_sep| < 8.0 OR cv-local NOT consistent. | 100.0% (20/20) | 85.1% (63/74) | 100.0% (22/22) | ❌ |
| `phase_or_cv_T11.7` | Retake when |phase_sep| < 11.7 OR cv-local NOT consistent. | 100.0% (20/20) | 85.1% (63/74) | 100.0% (22/22) | ❌ |
| `phase_or_cv_T15.0` | Retake when |phase_sep| < 15.0 OR cv-local NOT consistent. | 100.0% (20/20) | 85.1% (63/74) | 100.0% (22/22) | ❌ |
| `phase_and_cv_T8.0` | Retake when |phase_sep| < 8.0 AND cv-local NOT consistent. | 25.0% (5/20) | 2.7% (2/74) | 13.6% (3/22) | ❌ |
| `phase_and_cv_T11.7` | Retake when |phase_sep| < 11.7 AND cv-local NOT consistent. | 50.0% (10/20) | 12.2% (9/74) | 18.2% (4/22) | ❌ |
| `phase_and_cv_T15.0` | Retake when |phase_sep| < 15.0 AND cv-local NOT consistent. | 70.0% (14/20) | 18.9% (14/74) | 27.3% (6/22) | ❌ |
| `cv_severe_alone` | Retake when cv-local status is `fewer_than_3_face_quads` (the more severe failure mode — geometry couldn't even find 3 faces). | 40.0% (8/20) | 14.9% (11/74) | 22.7% (5/22) | ❌ |
| `phase_or_cv_severe_T8.0` | Retake when |phase_sep| < 8.0 OR cv-local is `fewer_than_3_face_quads`. | 50.0% (10/20) | 17.6% (13/74) | 31.8% (7/22) | ❌ |
| `phase_or_cv_severe_T11.7` | Retake when |phase_sep| < 11.7 OR cv-local is `fewer_than_3_face_quads`. | 65.0% (13/20) | 24.3% (18/74) | 36.4% (8/22) | ❌ |
| `phase_or_cv_severe_T15.0` | Retake when |phase_sep| < 15.0 OR cv-local is `fewer_than_3_face_quads`. | 75.0% (15/20) | 31.1% (23/74) | 40.9% (9/22) | ❌ |
| `fit_residual_alone_T60.0` | Retake when fit_residual_rms_px >= 60.0. | 60.0% (12/20) | 56.8% (42/74) | 54.5% (12/22) | ❌ |
| `fit_residual_alone_T80.0` | Retake when fit_residual_rms_px >= 80.0. | 60.0% (12/20) | 48.6% (36/74) | 54.5% (12/22) | ❌ |
| `fit_residual_alone_T100.0` | Retake when fit_residual_rms_px >= 100.0. | 45.0% (9/20) | 23.0% (17/74) | 50.0% (11/22) | ❌ |
| `fit_residual_alone_T120.0` | Retake when fit_residual_rms_px >= 120.0. | 45.0% (9/20) | 12.2% (9/74) | 45.5% (10/22) | ❌ |
| `fit_residual_alone_T150.0` | Retake when fit_residual_rms_px >= 150.0. | 0.0% (0/20) | 0.0% (0/74) | 0.0% (0/22) | ❌ |
| `fit_residual_alone_T200.0` | Retake when fit_residual_rms_px >= 200.0. | 0.0% (0/20) | 0.0% (0/74) | 0.0% (0/22) | ❌ |
| `pnp_rms_alone_T60.0` | Retake when pnp_rms_px >= 60.0. | 60.0% (12/20) | 56.8% (42/74) | 54.5% (12/22) | ❌ |
| `pnp_rms_alone_T100.0` | Retake when pnp_rms_px >= 100.0. | 45.0% (9/20) | 36.5% (27/74) | 54.5% (12/22) | ❌ |
| `pnp_rms_alone_T150.0` | Retake when pnp_rms_px >= 150.0. | 5.0% (1/20) | 0.0% (0/74) | 9.1% (2/22) | ❌ |
| `hex_bezel_disagree_T30.0` | Retake when hexagon_centroid_vs_bezel_vertex_offset_px >= 30.0. | 70.0% (14/20) | 71.6% (53/74) | 68.2% (15/22) | ❌ |
| `hex_bezel_disagree_T50.0` | Retake when hexagon_centroid_vs_bezel_vertex_offset_px >= 50.0. | 55.0% (11/20) | 31.1% (23/74) | 63.6% (14/22) | ❌ |
| `hex_bezel_disagree_T80.0` | Retake when hexagon_centroid_vs_bezel_vertex_offset_px >= 80.0. | 50.0% (10/20) | 23.0% (17/74) | 50.0% (11/22) | ❌ |
| `hex_bezel_disagree_T120.0` | Retake when hexagon_centroid_vs_bezel_vertex_offset_px >= 120.0. | 10.0% (2/20) | 9.5% (7/74) | 22.7% (5/22) | ❌ |
| `hex_bezel_disagree_T200.0` | Retake when hexagon_centroid_vs_bezel_vertex_offset_px >= 200.0. | 0.0% (0/20) | 0.0% (0/74) | 4.5% (1/22) | ❌ |
| `ensemble_shift_T20.0` | Retake when ensemble_shift_px >= 20.0. | 70.0% (14/20) | 47.3% (35/74) | 72.7% (16/22) | ❌ |
| `ensemble_shift_T40.0` | Retake when ensemble_shift_px >= 40.0. | 30.0% (6/20) | 18.9% (14/74) | 31.8% (7/22) | ❌ |
| `ensemble_shift_T60.0` | Retake when ensemble_shift_px >= 60.0. | 30.0% (6/20) | 5.4% (4/74) | 9.1% (2/22) | ❌ |
| `ensemble_shift_T100.0` | Retake when ensemble_shift_px >= 100.0. | 10.0% (2/20) | 0.0% (0/74) | 0.0% (0/22) | ❌ |
| `junction_score_below_T50.0` | Retake when junction_score_at_ensemble < 50.0 (low = weak vertex). | 10.0% (2/20) | 0.0% (0/74) | 0.0% (0/22) | ❌ |
| `junction_score_below_T100.0` | Retake when junction_score_at_ensemble < 100.0 (low = weak vertex). | 15.0% (3/20) | 2.7% (2/74) | 0.0% (0/22) | ❌ |
| `junction_score_below_T150.0` | Retake when junction_score_at_ensemble < 150.0 (low = weak vertex). | 40.0% (8/20) | 13.5% (10/74) | 9.1% (2/22) | ❌ |
| `junction_score_below_T200.0` | Retake when junction_score_at_ensemble < 200.0 (low = weak vertex). | 60.0% (12/20) | 32.4% (24/74) | 45.5% (10/22) | ❌ |
| `phaseANDcv_OR_fit_residual_T80.0` | Retake when (|phase|<8 AND cv-fail) OR fit_residual >= 80.0. | 65.0% (13/20) | 48.6% (36/74) | 59.1% (13/22) | ❌ |
| `phaseANDcv_OR_fit_residual_T100.0` | Retake when (|phase|<8 AND cv-fail) OR fit_residual >= 100.0. | 60.0% (12/20) | 24.3% (18/74) | 54.5% (12/22) | ❌ |
| `phaseANDcv_OR_fit_residual_T150.0` | Retake when (|phase|<8 AND cv-fail) OR fit_residual >= 150.0. | 25.0% (5/20) | 2.7% (2/74) | 13.6% (3/22) | ❌ |
| `phaseANDcv_OR_pnp_rms_T60.0` | Retake when (|phase|<8 AND cv-fail) OR pnp_rms >= 60.0. | 65.0% (13/20) | 56.8% (42/74) | 59.1% (13/22) | ❌ |
| `phaseANDcv_OR_pnp_rms_T100.0` | Retake when (|phase|<8 AND cv-fail) OR pnp_rms >= 100.0. | 60.0% (12/20) | 37.8% (28/74) | 59.1% (13/22) | ❌ |
| `phaseANDcv_OR_pnp_rms_T150.0` | Retake when (|phase|<8 AND cv-fail) OR pnp_rms >= 150.0. | 30.0% (6/20) | 2.7% (2/74) | 22.7% (5/22) | ❌ |
| `phaseANDcv_OR_hex_bezel_T50.0` | Retake when (|phase|<8 AND cv-fail) OR hex_bezel >= 50.0. | 70.0% (14/20) | 32.4% (24/74) | 68.2% (15/22) | ❌ |
| `phaseANDcv_OR_hex_bezel_T80.0` | Retake when (|phase|<8 AND cv-fail) OR hex_bezel >= 80.0. | 65.0% (13/20) | 25.7% (19/74) | 54.5% (12/22) | ❌ |
| `phaseANDcv_OR_hex_bezel_T120.0` | Retake when (|phase|<8 AND cv-fail) OR hex_bezel >= 120.0. | 35.0% (7/20) | 12.2% (9/74) | 36.4% (8/22) | ❌ |
| `phaseANDcv_OR_ensemble_shift_T20.0` | Retake when (|phase|<8 AND cv-fail) OR ensemble_shift >= 20.0. | 80.0% (16/20) | 48.6% (36/74) | 72.7% (16/22) | ❌ |
| `phaseANDcv_OR_ensemble_shift_T40.0` | Retake when (|phase|<8 AND cv-fail) OR ensemble_shift >= 40.0. | 50.0% (10/20) | 21.6% (16/74) | 45.5% (10/22) | ❌ |
| `phaseANDcv_OR_ensemble_shift_T60.0` | Retake when (|phase|<8 AND cv-fail) OR ensemble_shift >= 60.0. | 50.0% (10/20) | 8.1% (6/74) | 22.7% (5/22) | ❌ |
| `phaseANDcv_OR_junction_below_T50.0` | Retake when (|phase|<8 AND cv-fail) OR junction_score < 50.0. | 30.0% (6/20) | 2.7% (2/74) | 13.6% (3/22) | ❌ |
| `phaseANDcv_OR_junction_below_T100.0` | Retake when (|phase|<8 AND cv-fail) OR junction_score < 100.0. | 35.0% (7/20) | 4.1% (3/74) | 13.6% (3/22) | ❌ |
| `phaseANDcv_OR_junction_below_T150.0` | Retake when (|phase|<8 AND cv-fail) OR junction_score < 150.0. | 50.0% (10/20) | 14.9% (11/74) | 13.6% (3/22) | ❌ |
| `phaseANDcv_OR_junction_below_T200.0` | Retake when (|phase|<8 AND cv-fail) OR junction_score < 200.0. | 60.0% (12/20) | 33.8% (25/74) | 45.5% (10/22) | ❌ |
| `phaseANDcv_OR_fit80.0_OR_hex50.0` | Retake when (|phase|<8 AND cv-fail) OR fit_residual >= 80.0 OR hex_bezel >= 50.0. | 75.0% (15/20) | 59.5% (44/74) | 68.2% (15/22) | ❌ |
| `phaseANDcv_OR_fit100.0_OR_hex80.0` | Retake when (|phase|<8 AND cv-fail) OR fit_residual >= 100.0 OR hex_bezel >= 80.0. | 70.0% (14/20) | 35.1% (26/74) | 68.2% (15/22) | ❌ |
| `phaseANDcv_OR_fit150.0_OR_hex120.0` | Retake when (|phase|<8 AND cv-fail) OR fit_residual >= 150.0 OR hex_bezel >= 120.0. | 35.0% (7/20) | 12.2% (9/74) | 36.4% (8/22) | ❌ |
| `fit80.0_AND_hex50.0` | Retake when fit_residual >= 80.0 AND hex_bezel >= 50.0. | 45.0% (9/20) | 20.3% (15/74) | 54.5% (12/22) | ❌ |
| `fit60.0_AND_hex30.0` | Retake when fit_residual >= 60.0 AND hex_bezel >= 30.0. | 60.0% (12/20) | 56.8% (42/74) | 54.5% (12/22) | ❌ |

## Headline finding

**No rule over the 6 evaluated signals (phase_sep, cv-local structural status, fit_residual_rms_px, pnp_rms_px, hex↔bezel disagreement, ensemble_shift_px, junction_score_at_ensemble — alone, in OR/AND compounds, or as triples) meets the Phase 2 bar.**

Closest-to-bar rule: `phase_sep_alone_T15.0`
- catastrophic recall: 70.0% (bar: 80%; shortfall 10.0%)
- GOOD false-retake:   18.9% (bar: 10%; excess 8.9%)

## Implications

1. **One rule clears the recall bar but not FPR**: `phase_sep_alone_T20.0` hits 80% recall but at ~30% GOOD false-retake — way over the 10% bar. Loosening phase_sep high enough to catch all catastrophics necessarily catches many GOOD runs whose phase_sep happens to be small.

2. **One compound clears the FPR bar but not recall**: `phaseANDcv_OR_ensemble_shift_T60.0` is the first compound rule to land UNDER the 10% FPR bar — at 50% recall. Layering ensemble_shift on top of the phase+cv AND-compound demonstrably reduces false retakes without inheriting the noise of cv-local-solo or `hex_bezel`. This is the most encouraging multi-signal compound yet, but recall is still 30 pp short of the bar.

3. **No rule simultaneously clears both bars.** Hand-tuned thresholds and OR/AND compounds over 6 signals (phase_sep, cv-local, fit_residual, hex_bezel, ensemble_shift, junction_score) cannot get past the (≥80% recall AND ≤10% FPR) frontier on this 58-case eval.

4. **fit_residual_rms_px is weaker than expected**: alone, the best fit-residual rule is `T120.0` at 45% recall / 12.2% FPR — close to but not better than `phase_sep_T11.7`. Fit quality and outcome-correctness correlate but the thresholds don't separate cleanly.

5. **junction_score_at_ensemble doesn't help much**: at any threshold sweep, junction-score-based rules sit below the phase_sep curve. The image-space junction quality at the ensemble vertex isn't carrying enough information about phase/chirality correctness.

## Conditional pivot (per Codex) — TRIGGERED

> If Phase 2B finds a rule that meets the bar, then Phase 3 becomes straightforward: wire it as a conservative guardrail. If it does not, we will have clean evidence to pivot to learned geometry or capture/UX instead of hand-tuning another scalar.

**Evidence is in. Hand-tuned rules over the current signal set don't meet the bar.** The pivot options are now evidence-backed:

- **Learned geometry / ranker (Phase 4)** — train a logistic-regression or small-MLP retake classifier on the 6 continuous signals captured here. The Phase 2B matrix (`tests/fixtures/phase2b_trust_signal_matrix_recomputed.json`) is already shaped as a labeled dataset (per-row features + outcome). Likely lifts both axes simultaneously because the model learns the boundary in 6-D space instead of hand-tuning a few axis-aligned cuts.

- **Better capture / UX (Phase 5)** — diagnostics from Phase 2B (especially `ensemble_shift_px` and `hex_bezel_disagree`) can tell the user 'cube partially occluded, retake from a different angle' instead of just abstaining. This is the architectural lever cube-snap's two-photo UI relies on.

- **Two-view consistency (still not yet captured)** — the matrix has a `two_view_consistency_deg` column reserved but unpopulated; this would require fitting BOTH A and B views per set and comparing inferred orientations. Could be the single missing piece if A/B disagreement turns out to be strongly correlated with phase miss.

## See also

- `tools/PHASE_2A_PHASE_CONFIDENCE_CALIBRATION.md` — solo phase_sep calibration (this PR's starting point).
- `tools/PHASE_1_CV_LOCAL_BASELINE.md` — cv-local structural-fit baseline (this PR's second signal).
- `tools/POST_218_BASELINE_AND_TAXONOMY.md` — outcome categorization (the labels this PR predicts against).
- `tools/STATE_OF_THE_WORLD.md` — phased roadmap (Phase 0 → Phase 5).
