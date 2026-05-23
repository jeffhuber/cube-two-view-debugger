# Phase 2B: trust-signal matrix

**Status: diagnostics-only.** No production behavior change. This report evaluates candidate trust rules against the Phase 2 bar:

- catastrophic recall ≥ **80%**
- GOOD false-retake ≤ **10%**

## Dataset

- 140 per-case-per-run rows across 70 cases
- Outcome breakdown: **91 GOOD**, **16 MARGINAL**, **33 CATASTROPHIC**

**Source**: `tests/fixtures/phase2b_recomputed_signals.json` (per-run global-model re-fit on the 58-case axis-labeled gallery, capturing `fit_residual_rms_px`, `pnp_rms_px`, `hexagon_centroid_vs_bezel_vertex_offset_px`, `junction_score_at_ensemble`, `ensemble_shift_px`, and `phase_darkness_separation` at native precision) joined with `tests/fixtures/cv_local_baseline.json` (per-case cv-local face-quad structural status). Outcome counts differ from `post_218_baseline.json` (74/22/20 vs 76/16/24) because the re-fit is non-deterministic (PnP basin-of-attraction) and runs are paired with the signals from the same fit.

## Candidate rule evaluation

| rule | description | recall | GOOD FPR | MARGINAL routed | meets bar? |
|---|---|---|---|---|---|
| `phase_sep_alone_T11.7` | Retake when |phase_sep| < 11.7 (Phase 2A operating point). | 45.5% (15/33) | 14.3% (13/91) | 37.5% (6/16) | ❌ |
| `phase_sep_alone_T0.5` | Retake when |phase_sep| < 0.5. | 0.0% (0/33) | 0.0% (0/91) | 0.0% (0/16) | ❌ |
| `phase_sep_alone_T2.0` | Retake when |phase_sep| < 2.0. | 15.2% (5/33) | 5.5% (5/91) | 0.0% (0/16) | ❌ |
| `phase_sep_alone_T5.0` | Retake when |phase_sep| < 5.0. | 18.2% (6/33) | 8.8% (8/91) | 0.0% (0/16) | ❌ |
| `phase_sep_alone_T8.0` | Retake when |phase_sep| < 8.0. | 33.3% (11/33) | 11.0% (10/91) | 12.5% (2/16) | ❌ |
| `phase_sep_alone_T15.0` | Retake when |phase_sep| < 15.0. | 63.6% (21/33) | 19.8% (18/91) | 43.8% (7/16) | ❌ |
| `phase_sep_alone_T20.0` | Retake when |phase_sep| < 20.0. | 75.8% (25/33) | 30.8% (28/91) | 56.2% (9/16) | ❌ |
| `cv_local_alone` | Retake when cv-local face-quad fit is NOT structurally consistent (status != 'ok'). | 100.0% (33/33) | 89.0% (81/91) | 87.5% (14/16) | ❌ |
| `phase_or_cv_T8.0` | Retake when |phase_sep| < 8.0 OR cv-local NOT consistent. | 100.0% (33/33) | 90.1% (82/91) | 93.8% (15/16) | ❌ |
| `phase_or_cv_T11.7` | Retake when |phase_sep| < 11.7 OR cv-local NOT consistent. | 100.0% (33/33) | 90.1% (82/91) | 100.0% (16/16) | ❌ |
| `phase_or_cv_T15.0` | Retake when |phase_sep| < 15.0 OR cv-local NOT consistent. | 100.0% (33/33) | 91.2% (83/91) | 100.0% (16/16) | ❌ |
| `phase_and_cv_T8.0` | Retake when |phase_sep| < 8.0 AND cv-local NOT consistent. | 33.3% (11/33) | 9.9% (9/91) | 6.2% (1/16) | ❌ |
| `phase_and_cv_T11.7` | Retake when |phase_sep| < 11.7 AND cv-local NOT consistent. | 45.5% (15/33) | 13.2% (12/91) | 25.0% (4/16) | ❌ |
| `phase_and_cv_T15.0` | Retake when |phase_sep| < 15.0 AND cv-local NOT consistent. | 63.6% (21/33) | 17.6% (16/91) | 31.2% (5/16) | ❌ |
| `cv_severe_alone` | Retake when cv-local status is `fewer_than_3_face_quads` (the more severe failure mode — geometry couldn't even find 3 faces). | 21.2% (7/33) | 15.4% (14/91) | 18.8% (3/16) | ❌ |
| `phase_or_cv_severe_T8.0` | Retake when |phase_sep| < 8.0 OR cv-local is `fewer_than_3_face_quads`. | 48.5% (16/33) | 24.2% (22/91) | 25.0% (4/16) | ❌ |
| `phase_or_cv_severe_T11.7` | Retake when |phase_sep| < 11.7 OR cv-local is `fewer_than_3_face_quads`. | 57.6% (19/33) | 27.5% (25/91) | 37.5% (6/16) | ❌ |
| `phase_or_cv_severe_T15.0` | Retake when |phase_sep| < 15.0 OR cv-local is `fewer_than_3_face_quads`. | 69.7% (23/33) | 33.0% (30/91) | 43.8% (7/16) | ❌ |
| `fit_residual_alone_T60.0` | Retake when fit_residual_rms_px >= 60.0. | 66.7% (22/33) | 48.4% (44/91) | 50.0% (8/16) | ❌ |
| `fit_residual_alone_T80.0` | Retake when fit_residual_rms_px >= 80.0. | 63.6% (21/33) | 37.4% (34/91) | 43.8% (7/16) | ❌ |
| `fit_residual_alone_T100.0` | Retake when fit_residual_rms_px >= 100.0. | 42.4% (14/33) | 17.6% (16/91) | 31.2% (5/16) | ❌ |
| `fit_residual_alone_T120.0` | Retake when fit_residual_rms_px >= 120.0. | 33.3% (11/33) | 13.2% (12/91) | 18.8% (3/16) | ❌ |
| `fit_residual_alone_T150.0` | Retake when fit_residual_rms_px >= 150.0. | 3.0% (1/33) | 0.0% (0/91) | 0.0% (0/16) | ❌ |
| `fit_residual_alone_T200.0` | Retake when fit_residual_rms_px >= 200.0. | 0.0% (0/33) | 0.0% (0/91) | 0.0% (0/16) | ❌ |
| `pnp_rms_alone_T60.0` | Retake when pnp_rms_px >= 60.0. | 66.7% (22/33) | 48.4% (44/91) | 50.0% (8/16) | ❌ |
| `pnp_rms_alone_T100.0` | Retake when pnp_rms_px >= 100.0. | 42.4% (14/33) | 28.6% (26/91) | 50.0% (8/16) | ❌ |
| `pnp_rms_alone_T150.0` | Retake when pnp_rms_px >= 150.0. | 15.2% (5/33) | 1.1% (1/91) | 6.2% (1/16) | ❌ |
| `hex_bezel_disagree_T30.0` | Retake when hexagon_centroid_vs_bezel_vertex_offset_px >= 30.0. | 63.6% (21/33) | 68.1% (62/91) | 56.2% (9/16) | ❌ |
| `hex_bezel_disagree_T50.0` | Retake when hexagon_centroid_vs_bezel_vertex_offset_px >= 50.0. | 57.6% (19/33) | 39.6% (36/91) | 37.5% (6/16) | ❌ |
| `hex_bezel_disagree_T80.0` | Retake when hexagon_centroid_vs_bezel_vertex_offset_px >= 80.0. | 45.5% (15/33) | 36.3% (33/91) | 25.0% (4/16) | ❌ |
| `hex_bezel_disagree_T120.0` | Retake when hexagon_centroid_vs_bezel_vertex_offset_px >= 120.0. | 12.1% (4/33) | 15.4% (14/91) | 6.2% (1/16) | ❌ |
| `hex_bezel_disagree_T200.0` | Retake when hexagon_centroid_vs_bezel_vertex_offset_px >= 200.0. | 0.0% (0/33) | 1.1% (1/91) | 0.0% (0/16) | ❌ |
| `ensemble_shift_T20.0` | Retake when ensemble_shift_px >= 20.0. | 63.6% (21/33) | 64.8% (59/91) | 62.5% (10/16) | ❌ |
| `ensemble_shift_T40.0` | Retake when ensemble_shift_px >= 40.0. | 33.3% (11/33) | 39.6% (36/91) | 31.2% (5/16) | ❌ |
| `ensemble_shift_T60.0` | Retake when ensemble_shift_px >= 60.0. | 15.2% (5/33) | 24.2% (22/91) | 25.0% (4/16) | ❌ |
| `ensemble_shift_T100.0` | Retake when ensemble_shift_px >= 100.0. | 12.1% (4/33) | 7.7% (7/91) | 6.2% (1/16) | ❌ |
| `junction_score_below_T50.0` | Retake when junction_score_at_ensemble < 50.0 (low = weak vertex). | 12.1% (4/33) | 1.1% (1/91) | 0.0% (0/16) | ❌ |
| `junction_score_below_T100.0` | Retake when junction_score_at_ensemble < 100.0 (low = weak vertex). | 18.2% (6/33) | 4.4% (4/91) | 6.2% (1/16) | ❌ |
| `junction_score_below_T150.0` | Retake when junction_score_at_ensemble < 150.0 (low = weak vertex). | 36.4% (12/33) | 13.2% (12/91) | 18.8% (3/16) | ❌ |
| `junction_score_below_T200.0` | Retake when junction_score_at_ensemble < 200.0 (low = weak vertex). | 72.7% (24/33) | 31.9% (29/91) | 50.0% (8/16) | ❌ |
| `phaseANDcv_OR_fit_residual_T80.0` | Retake when (|phase|<8 AND cv-fail) OR fit_residual >= 80.0. | 69.7% (23/33) | 44.0% (40/91) | 43.8% (7/16) | ❌ |
| `phaseANDcv_OR_fit_residual_T100.0` | Retake when (|phase|<8 AND cv-fail) OR fit_residual >= 100.0. | 57.6% (19/33) | 25.3% (23/91) | 37.5% (6/16) | ❌ |
| `phaseANDcv_OR_fit_residual_T150.0` | Retake when (|phase|<8 AND cv-fail) OR fit_residual >= 150.0. | 36.4% (12/33) | 9.9% (9/91) | 6.2% (1/16) | ❌ |
| `phaseANDcv_OR_pnp_rms_T60.0` | Retake when (|phase|<8 AND cv-fail) OR pnp_rms >= 60.0. | 72.7% (24/33) | 53.8% (49/91) | 50.0% (8/16) | ❌ |
| `phaseANDcv_OR_pnp_rms_T100.0` | Retake when (|phase|<8 AND cv-fail) OR pnp_rms >= 100.0. | 57.6% (19/33) | 36.3% (33/91) | 50.0% (8/16) | ❌ |
| `phaseANDcv_OR_pnp_rms_T150.0` | Retake when (|phase|<8 AND cv-fail) OR pnp_rms >= 150.0. | 42.4% (14/33) | 11.0% (10/91) | 12.5% (2/16) | ❌ |
| `phaseANDcv_OR_hex_bezel_T50.0` | Retake when (|phase|<8 AND cv-fail) OR hex_bezel >= 50.0. | 72.7% (24/33) | 46.2% (42/91) | 43.8% (7/16) | ❌ |
| `phaseANDcv_OR_hex_bezel_T80.0` | Retake when (|phase|<8 AND cv-fail) OR hex_bezel >= 80.0. | 66.7% (22/33) | 42.9% (39/91) | 31.2% (5/16) | ❌ |
| `phaseANDcv_OR_hex_bezel_T120.0` | Retake when (|phase|<8 AND cv-fail) OR hex_bezel >= 120.0. | 42.4% (14/33) | 25.3% (23/91) | 12.5% (2/16) | ❌ |
| `phaseANDcv_OR_ensemble_shift_T20.0` | Retake when (|phase|<8 AND cv-fail) OR ensemble_shift >= 20.0. | 75.8% (25/33) | 70.3% (64/91) | 68.8% (11/16) | ❌ |
| `phaseANDcv_OR_ensemble_shift_T40.0` | Retake when (|phase|<8 AND cv-fail) OR ensemble_shift >= 40.0. | 57.6% (19/33) | 48.4% (44/91) | 37.5% (6/16) | ❌ |
| `phaseANDcv_OR_ensemble_shift_T60.0` | Retake when (|phase|<8 AND cv-fail) OR ensemble_shift >= 60.0. | 45.5% (15/33) | 33.0% (30/91) | 31.2% (5/16) | ❌ |
| `phaseANDcv_OR_junction_below_T50.0` | Retake when (|phase|<8 AND cv-fail) OR junction_score < 50.0. | 45.5% (15/33) | 11.0% (10/91) | 6.2% (1/16) | ❌ |
| `phaseANDcv_OR_junction_below_T100.0` | Retake when (|phase|<8 AND cv-fail) OR junction_score < 100.0. | 48.5% (16/33) | 13.2% (12/91) | 12.5% (2/16) | ❌ |
| `phaseANDcv_OR_junction_below_T150.0` | Retake when (|phase|<8 AND cv-fail) OR junction_score < 150.0. | 66.7% (22/33) | 19.8% (18/91) | 18.8% (3/16) | ❌ |
| `phaseANDcv_OR_junction_below_T200.0` | Retake when (|phase|<8 AND cv-fail) OR junction_score < 200.0. | 90.9% (30/33) | 35.2% (32/91) | 50.0% (8/16) | ❌ |
| `phaseANDcv_OR_fit80.0_OR_hex50.0` | Retake when (|phase|<8 AND cv-fail) OR fit_residual >= 80.0 OR hex_bezel >= 50.0. | 84.8% (28/33) | 60.4% (55/91) | 50.0% (8/16) | ❌ |
| `phaseANDcv_OR_fit100.0_OR_hex80.0` | Retake when (|phase|<8 AND cv-fail) OR fit_residual >= 100.0 OR hex_bezel >= 80.0. | 72.7% (24/33) | 46.2% (42/91) | 43.8% (7/16) | ❌ |
| `phaseANDcv_OR_fit150.0_OR_hex120.0` | Retake when (|phase|<8 AND cv-fail) OR fit_residual >= 150.0 OR hex_bezel >= 120.0. | 45.5% (15/33) | 25.3% (23/91) | 12.5% (2/16) | ❌ |
| `fit80.0_AND_hex50.0` | Retake when fit_residual >= 80.0 AND hex_bezel >= 50.0. | 42.4% (14/33) | 22.0% (20/91) | 31.2% (5/16) | ❌ |
| `fit60.0_AND_hex30.0` | Retake when fit_residual >= 60.0 AND hex_bezel >= 30.0. | 48.5% (16/33) | 46.2% (42/91) | 50.0% (8/16) | ❌ |

## Headline finding

**No rule over the 6 evaluated signals (phase_sep, cv-local structural status, fit_residual_rms_px, pnp_rms_px, hex↔bezel disagreement, ensemble_shift_px, junction_score_at_ensemble — alone, in OR/AND compounds, or as triples) meets the Phase 2 bar.**

Closest-to-bar rule: `phaseANDcv_OR_junction_below_T150.0`
- catastrophic recall: 66.7% (bar: 80%; shortfall 13.3%)
- GOOD false-retake:   19.8% (bar: 10%; excess 9.8%)

## Implications

1. **6 rule(s) clear the ≥80% recall bar, none also clear the ≤10% FPR bar.** Best (by recall − FPR margin): `phaseANDcv_OR_junction_below_T200.0` at 90.9% recall / 35.2% FPR. Loosening retake thresholds high enough to catch all catastrophics necessarily also retakes many GOOD runs.

2. **12 rule(s) clear the ≤10% FPR bar, none also clear the ≥80% recall bar.** Best (by recall − FPR margin): `phaseANDcv_OR_fit_residual_T150.0` at 36.4% recall / 9.9% FPR. These are predominantly OR-compounds of the phase+cv AND-rule with a high-threshold continuous signal — they hold FPR down by being narrow but pay for it on recall.

3. **No rule simultaneously clears both bars.** Hand-tuned thresholds and OR/AND compounds over 6 signals (phase_sep, cv-local, fit_residual, hex_bezel, ensemble_shift, junction_score, pnp_rms) cannot get past the (≥80% recall AND ≤10% FPR) frontier on this 58-case eval.

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
