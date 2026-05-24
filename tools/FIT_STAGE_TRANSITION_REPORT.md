# Fit stage transition diagnostic

This diagnostic traces axis correctness through the production global-cube fit stages. It is diagnostic-only and exists to locate where canonical-good initial correspondence becomes a broken final axis model.

Axis state buckets: `usable` <= 30 deg total axis misfit; `broken` >= 150 deg; otherwise `marginal`.

## Source

- Tool: `tools/diagnose_fit_stage_transitions.py`
- Commit: `3237af802c5d4b75980d6da5454db57ead7b65b1`
- Generated: `2026-05-24T22:07:54.841658+00:00`
- Truth: `tests/fixtures/full_corner_ground_truth.json`
- Manifest: `tests/fixtures/corpus_manifest.json`
- Max image dim: `1600`
- Run selection: single deterministic run per row

## Aggregate

- Rows traced: 12 / 12
- First broken stage counts by path: `{'corr_false': {'affine_selected': 9, 'never_broken': 3}, 'corr_true': {'affine_selected': 9, 'never_broken': 1, 'corr_true_phase_check': 2}}`
- Phase correction effect counts: `{'keeps_broken': 7, 'fixes_broken_to_usable': 2, 'keeps_usable': 1, 'breaks_usable_to_broken': 2}`

## Headline Findings

- `affine_selected` reproduces production's strict first-minimum residual tie behavior. It does not use human truth or category tie-breaking.
- On this run, most broken final axes are already broken at `affine_selected`; PnP, mean3, and vertex refinement mostly preserve the selected axis state rather than creating the near-180-degree misfit later.
- This should be read alongside the Procrustes correspondence diagnostic as: canonical-good assignments exist in the search space, but production's residual-only tie behavior can still select the wrong 3-fold phase.
- Phase correction is mixed: it can rescue a broken assignment, but it can also flip a usable assignment into the broken phase.

### Stage Summary

| Stage | n | usable | marginal | broken | median total axis misfit deg |
|---|---:|---:|---:|---:|---:|
| `affine_selected` | 12 | 3 | 0 | 9 | 177.6 |
| `template_fit_pnp_or_affine` | 12 | 3 | 0 | 9 | 177.7 |
| `mean3_vertex` | 12 | 3 | 0 | 9 | 177.7 |
| `corr_false_phase_check` | 12 | 3 | 0 | 9 | 177.7 |
| `corr_false_final_refined` | 12 | 3 | 0 | 9 | 177.7 |
| `corr_true_phase_check` | 12 | 3 | 0 | 9 | 178.1 |
| `corr_true_final_refined` | 12 | 3 | 0 | 9 | 178.1 |

## Per-row Stage Trace

| Row | Stage | axis state | total misfit deg | delta vs previous | vertex err px | phase/refine |
|---|---|---|---:|---:|---:|---|
| `20_A` | `affine_selected` | broken | 177.4 | None | 46.9 | approach=affine_selected |
| `20_A` | `template_fit_pnp_or_affine` | broken | 177.4 | 0.0 | 43.2 | approach=perspective_pnp |
| `20_A` | `mean3_vertex` | broken | 177.4 | 0.0 | 44.2 | approach=procrustes_template_fit+mean3_vertex; ensemble_shift=3.0 |
| `20_A` | `corr_false_phase_check` | broken | 177.4 | 0.0 | 44.2 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; ensemble_shift=3.0 |
| `20_A` | `corr_false_final_refined` | broken | 177.4 | 0.0 | 44.2 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; refinement=skipped_high_base_score; ensemble_shift=3.0 |
| `20_A` | `corr_true_phase_check` | broken | 177.4 | 0.0 | 44.2 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; ensemble_shift=3.0 |
| `20_A` | `corr_true_final_refined` | broken | 177.4 | 0.0 | 44.2 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; refinement=skipped_high_base_score; ensemble_shift=3.0 |
| `20_B` | `affine_selected` | broken | 179.8 | None | 78.9 | approach=affine_selected |
| `20_B` | `template_fit_pnp_or_affine` | broken | 179.6 | -0.2 | 81.0 | approach=perspective_pnp |
| `20_B` | `mean3_vertex` | broken | 179.6 | 0.0 | 77.6 | approach=procrustes_template_fit+mean3_vertex; ensemble_shift=3.4 |
| `20_B` | `corr_false_phase_check` | broken | 179.6 | 0.0 | 77.6 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; ensemble_shift=3.4 |
| `20_B` | `corr_false_final_refined` | broken | 179.6 | 0.0 | 77.6 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; refinement=skipped_high_base_score; ensemble_shift=3.4 |
| `20_B` | `corr_true_phase_check` | broken | 179.6 | 0.0 | 77.6 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; ensemble_shift=3.4 |
| `20_B` | `corr_true_final_refined` | broken | 179.6 | 0.0 | 77.6 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; refinement=skipped_high_base_score; ensemble_shift=3.4 |
| `38_A` | `affine_selected` | broken | 179.2 | None | 73.2 | approach=affine_selected |
| `38_A` | `template_fit_pnp_or_affine` | broken | 179.4 | 0.2 | 90.1 | approach=perspective_pnp |
| `38_A` | `mean3_vertex` | broken | 179.4 | 0.0 | 79.8 | approach=procrustes_template_fit+mean3_vertex; ensemble_shift=10.4 |
| `38_A` | `corr_false_phase_check` | broken | 179.4 | 0.0 | 79.8 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; ensemble_shift=10.4 |
| `38_A` | `corr_false_final_refined` | broken | 179.4 | 0.0 | 79.8 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; refinement=skipped_high_base_score; ensemble_shift=10.4 |
| `38_A` | `corr_true_phase_check` | broken | 179.4 | 0.0 | 79.8 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; ensemble_shift=10.4 |
| `38_A` | `corr_true_final_refined` | broken | 179.4 | 0.0 | 79.8 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; refinement=skipped_high_base_score; ensemble_shift=10.4 |
| `38_B` | `affine_selected` | broken | 175.2 | None | 42.5 | approach=affine_selected |
| `38_B` | `template_fit_pnp_or_affine` | broken | 175.0 | -0.2 | 55.3 | approach=perspective_pnp |
| `38_B` | `mean3_vertex` | broken | 175.0 | 0.0 | 97.5 | approach=procrustes_template_fit+mean3_vertex; ensemble_shift=56.6 |
| `38_B` | `corr_false_phase_check` | broken | 175.0 | 0.0 | 97.5 | approach=procrustes_template_fit+mean3_vertex; phase_check=flip_suggested_diagnostic_only; ensemble_shift=56.6 |
| `38_B` | `corr_false_final_refined` | broken | 175.0 | 0.0 | 86.0 | approach=procrustes_template_fit+mean3_vertex; phase_check=flip_suggested_diagnostic_only; refinement=applied; ensemble_shift=56.6; refine_move=55.6 |
| `38_B` | `corr_true_phase_check` | usable | 8.9 | -166.1 | 97.5 | approach=procrustes_template_fit+mean3_vertex; phase_check=corrected_60deg_flip; ensemble_shift=56.6 |
| `38_B` | `corr_true_final_refined` | usable | 8.9 | 0.0 | 83.2 | approach=procrustes_template_fit+mean3_vertex; phase_check=corrected_60deg_flip; refinement=applied; ensemble_shift=56.6; refine_move=55.6 |
| `40_A` | `affine_selected` | broken | 178.8 | None | 106.9 | approach=affine_selected |
| `40_A` | `template_fit_pnp_or_affine` | broken | 178.8 | 0.0 | 106.9 | approach=affine_fallback |
| `40_A` | `mean3_vertex` | broken | 178.8 | 0.0 | 104.9 | approach=procrustes_template_fit+mean3_vertex; ensemble_shift=3.6 |
| `40_A` | `corr_false_phase_check` | broken | 178.8 | 0.0 | 104.9 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; ensemble_shift=3.6 |
| `40_A` | `corr_false_final_refined` | broken | 178.8 | 0.0 | 78.6 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; refinement=applied; ensemble_shift=3.6; refine_move=31.8 |
| `40_A` | `corr_true_phase_check` | broken | 178.8 | 0.0 | 104.9 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; ensemble_shift=3.6 |
| `40_A` | `corr_true_final_refined` | broken | 178.8 | 0.0 | 78.6 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; refinement=applied; ensemble_shift=3.6; refine_move=31.8 |
| `40_B` | `affine_selected` | broken | 178.7 | None | 95.1 | approach=affine_selected |
| `40_B` | `template_fit_pnp_or_affine` | broken | 178.2 | -0.5 | 136.2 | approach=perspective_pnp |
| `40_B` | `mean3_vertex` | broken | 178.2 | 0.0 | 149.7 | approach=procrustes_template_fit+mean3_vertex; ensemble_shift=48.5 |
| `40_B` | `corr_false_phase_check` | broken | 178.2 | 0.0 | 149.7 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; ensemble_shift=48.5 |
| `40_B` | `corr_false_final_refined` | broken | 178.2 | 0.0 | 184.9 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; refinement=applied; ensemble_shift=48.5; refine_move=39.6 |
| `40_B` | `corr_true_phase_check` | broken | 178.2 | 0.0 | 149.7 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; ensemble_shift=48.5 |
| `40_B` | `corr_true_final_refined` | broken | 178.2 | 0.0 | 184.9 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; refinement=applied; ensemble_shift=48.5; refine_move=39.6 |
| `41_A` | `affine_selected` | usable | 12.9 | None | 92.2 | approach=affine_selected |
| `41_A` | `template_fit_pnp_or_affine` | usable | 12.9 | 0.0 | 92.2 | approach=affine_fallback |
| `41_A` | `mean3_vertex` | usable | 12.9 | 0.0 | 91.5 | approach=procrustes_template_fit+mean3_vertex; ensemble_shift=2.0 |
| `41_A` | `corr_false_phase_check` | usable | 12.9 | 0.0 | 91.5 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; ensemble_shift=2.0 |
| `41_A` | `corr_false_final_refined` | usable | 12.9 | 0.0 | 101.8 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; refinement=applied; ensemble_shift=2.0; refine_move=33.2 |
| `41_A` | `corr_true_phase_check` | usable | 12.9 | 0.0 | 91.5 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; ensemble_shift=2.0 |
| `41_A` | `corr_true_final_refined` | usable | 12.9 | 0.0 | 101.8 | approach=procrustes_template_fit+mean3_vertex; phase_check=correct; refinement=applied; ensemble_shift=2.0; refine_move=33.2 |
| `41_B` | `affine_selected` | broken | 179.8 | None | 55.2 | approach=affine_selected |
| `41_B` | `template_fit_pnp_or_affine` | broken | 179.8 | 0.0 | 78.0 | approach=perspective_pnp |
| `41_B` | `mean3_vertex` | broken | 179.8 | 0.0 | 75.8 | approach=procrustes_template_fit+mean3_vertex; ensemble_shift=41.1 |
| `41_B` | `corr_false_phase_check` | broken | 179.8 | 0.0 | 75.8 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; ensemble_shift=41.1 |
| `41_B` | `corr_false_final_refined` | broken | 179.8 | 0.0 | 37.3 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; refinement=applied; ensemble_shift=41.1; refine_move=43.6 |
| `41_B` | `corr_true_phase_check` | broken | 179.8 | 0.0 | 75.8 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; ensemble_shift=41.1 |
| `41_B` | `corr_true_final_refined` | broken | 179.8 | 0.0 | 37.3 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; refinement=applied; ensemble_shift=41.1; refine_move=43.6 |
| `43_A` | `affine_selected` | usable | 9.3 | None | 45.4 | approach=affine_selected |
| `43_A` | `template_fit_pnp_or_affine` | usable | 5.7 | -3.6 | 22.2 | approach=perspective_pnp |
| `43_A` | `mean3_vertex` | usable | 5.7 | 0.0 | 27.0 | approach=procrustes_template_fit+mean3_vertex; ensemble_shift=38.6 |
| `43_A` | `corr_false_phase_check` | usable | 5.7 | 0.0 | 27.0 | approach=procrustes_template_fit+mean3_vertex; phase_check=flip_suggested_diagnostic_only; ensemble_shift=38.6 |
| `43_A` | `corr_false_final_refined` | usable | 5.7 | 0.0 | 27.0 | approach=procrustes_template_fit+mean3_vertex; phase_check=flip_suggested_diagnostic_only; refinement=skipped_high_base_score; ensemble_shift=38.6 |
| `43_A` | `corr_true_phase_check` | broken | 179.5 | 173.8 | 27.0 | approach=procrustes_template_fit+mean3_vertex; phase_check=corrected_60deg_flip; ensemble_shift=38.6 |
| `43_A` | `corr_true_final_refined` | broken | 179.5 | 0.0 | 27.0 | approach=procrustes_template_fit+mean3_vertex; phase_check=corrected_60deg_flip; refinement=skipped_high_base_score; ensemble_shift=38.6 |
| `43_B` | `affine_selected` | broken | 177.8 | None | 73.1 | approach=affine_selected |
| `43_B` | `template_fit_pnp_or_affine` | broken | 177.9 | 0.1 | 89.7 | approach=perspective_pnp |
| `43_B` | `mean3_vertex` | broken | 177.9 | 0.0 | 78.1 | approach=procrustes_template_fit+mean3_vertex; ensemble_shift=13.9 |
| `43_B` | `corr_false_phase_check` | broken | 177.9 | 0.0 | 78.1 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; ensemble_shift=13.9 |
| `43_B` | `corr_false_final_refined` | broken | 177.9 | 0.0 | 78.1 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; refinement=skipped_high_base_score; ensemble_shift=13.9 |
| `43_B` | `corr_true_phase_check` | broken | 177.9 | 0.0 | 78.1 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; ensemble_shift=13.9 |
| `43_B` | `corr_true_final_refined` | broken | 177.9 | 0.0 | 78.1 | approach=procrustes_template_fit+mean3_vertex; phase_check=ambiguous_no_correction; refinement=skipped_high_base_score; ensemble_shift=13.9 |
| `45_A` | `affine_selected` | usable | 13.5 | None | 88.7 | approach=affine_selected |
| `45_A` | `template_fit_pnp_or_affine` | usable | 5.9 | -7.6 | 17.2 | approach=perspective_pnp |
| `45_A` | `mean3_vertex` | usable | 5.9 | 0.0 | 68.0 | approach=procrustes_template_fit+mean3_vertex; ensemble_shift=51.1 |
| `45_A` | `corr_false_phase_check` | usable | 5.9 | 0.0 | 68.0 | approach=procrustes_template_fit+mean3_vertex; phase_check=flip_suggested_diagnostic_only; ensemble_shift=51.1 |
| `45_A` | `corr_false_final_refined` | usable | 5.9 | 0.0 | 41.6 | approach=procrustes_template_fit+mean3_vertex; phase_check=flip_suggested_diagnostic_only; refinement=applied; ensemble_shift=51.1; refine_move=56.2 |
| `45_A` | `corr_true_phase_check` | broken | 178.1 | 172.2 | 68.0 | approach=procrustes_template_fit+mean3_vertex; phase_check=corrected_60deg_flip; ensemble_shift=51.1 |
| `45_A` | `corr_true_final_refined` | broken | 178.1 | 0.0 | 41.5 | approach=procrustes_template_fit+mean3_vertex; phase_check=corrected_60deg_flip; refinement=applied; ensemble_shift=51.1; refine_move=53.4 |
| `45_B` | `affine_selected` | broken | 177.3 | None | 115.7 | approach=affine_selected |
| `45_B` | `template_fit_pnp_or_affine` | broken | 176.6 | -0.7 | 149.0 | approach=perspective_pnp |
| `45_B` | `mean3_vertex` | broken | 176.6 | 0.0 | 129.8 | approach=procrustes_template_fit+mean3_vertex; ensemble_shift=20.3 |
| `45_B` | `corr_false_phase_check` | broken | 176.6 | 0.0 | 129.8 | approach=procrustes_template_fit+mean3_vertex; phase_check=flip_suggested_diagnostic_only; ensemble_shift=20.3 |
| `45_B` | `corr_false_final_refined` | broken | 176.6 | 0.0 | 114.7 | approach=procrustes_template_fit+mean3_vertex; phase_check=flip_suggested_diagnostic_only; refinement=applied; ensemble_shift=20.3; refine_move=55.3 |
| `45_B` | `corr_true_phase_check` | usable | 24.8 | -151.8 | 129.8 | approach=procrustes_template_fit+mean3_vertex; phase_check=corrected_60deg_flip; ensemble_shift=20.3 |
| `45_B` | `corr_true_final_refined` | usable | 24.8 | 0.0 | 116.0 | approach=procrustes_template_fit+mean3_vertex; phase_check=corrected_60deg_flip; refinement=applied; ensemble_shift=20.3; refine_move=56.7 |

## Interpretation

- If `affine_selected` is usable but `template_fit_pnp_or_affine` is broken, PnP / affine fallback selection is the first bad stage.
- If `template_fit_pnp_or_affine` is usable but `mean3_vertex` is broken, the stage trace is inconsistent because mean3 should not change axis vectors; inspect scoring or geometry translation.
- If `mean3_vertex` is usable but a `corr_*_phase_check` stage is broken, phase correction is the first bad stage.
- If a phase-check stage is usable but the matching final-refined stage is broken, image-junction vertex refinement is the first bad stage.
