# Vertex + Axis Active-Learning Queue V0

Diagnostics/data-only artifact. This does not alter recognizer behavior.

This queue selects additional manifest image rows for human visible-trihedral labeling after the KNN V0 result: candidate generation is strong, but ranking/confidence needs more supervision.

## Summary

- Queue rows: 30
- Already completed canonical trihedral labels: 28
- Queue rows with current model attached: 30
- Queue rows already labeled: 0
- Priority buckets: `tier1_easy_unlabeled` 6, `tier2_pair_completion` 6, `tier3_manifest_stress` 17, `tier4_retake_or_model_boundary` 1
- Evaluation tiers: `corpus_stress` 4, `easy_corpus` 6, `hard_case_stress` 20
- Source model statuses: `low_cell_inside` 1, `ok` 29
- Active reasons: `corpus_stress_unlabeled` 4, `easy_corpus_unlabeled` 6, `hard_case_unlabeled` 20, `low_cell_inside` 3, `low_silhouette_iou` 9, `paired_with_existing_label` 8, `weak_geometry_or_retake_boundary` 1

## Labeling Command

```bash
.venv/bin/python tools/vertex_axis_label_server.py --feedback tests/fixtures/vertex_axis_active_learning_feedback_v0.json --report tools/VERTEX_AXIS_ACTIVE_LEARNING_FEEDBACK_V0_REPORT.md --port 8778
```

## Labeling Instructions

- Start with `tier1_easy_unlabeled` rows. These are the cleanest way to make the localizer boringly consistent before hard backgrounds.
- For each row, click the visible trihedral vertex first, then one endpoint on each outgoing cube-edge ray. Axis order does not matter.
- Use `ambiguous` or `not_visible` if the vertex/rays cannot be marked honestly.
- Keep notes short when the model overlay is misleading, the cube is occluded, or the background should be treated as a retake candidate.

## Rows

| Row | Priority | Tier | Source status | Score | Reasons | Model score | IoU | Cell inside |
|---|---|---|---|---:|---|---:|---:|---:|
| `42_A` | `tier1_easy_unlabeled` | `easy_corpus` | `ok` | 131.42 | easy_corpus_unlabeled, low_silhouette_iou, paired_with_existing_label | 1.898 | 0.642 | 1.000 |
| `36_A` | `tier1_easy_unlabeled` | `easy_corpus` | `ok` | 130.96 | easy_corpus_unlabeled, low_silhouette_iou, paired_with_existing_label | 1.886 | 0.596 | 1.000 |
| `23_B` | `tier1_easy_unlabeled` | `easy_corpus` | `ok` | 113.69 | easy_corpus_unlabeled | 2.224 | 0.869 | 1.000 |
| `37_A` | `tier1_easy_unlabeled` | `easy_corpus` | `ok` | 113.18 | easy_corpus_unlabeled | 2.155 | 0.818 | 1.000 |
| `23_A` | `tier1_easy_unlabeled` | `easy_corpus` | `ok` | 113.06 | easy_corpus_unlabeled | 2.146 | 0.806 | 1.000 |
| `37_B` | `tier1_easy_unlabeled` | `easy_corpus` | `ok` | 111.60 | easy_corpus_unlabeled, low_silhouette_iou | 1.934 | 0.660 | 1.000 |
| `12_A` | `tier2_pair_completion` | `corpus_stress` | `ok` | 93.67 | corpus_stress_unlabeled, paired_with_existing_label | 2.224 | 0.868 | 1.000 |
| `27_A` | `tier2_pair_completion` | `corpus_stress` | `ok` | 93.50 | corpus_stress_unlabeled, paired_with_existing_label | 2.277 | 0.850 | 1.000 |
| `24_B` | `tier2_pair_completion` | `corpus_stress` | `ok` | 93.14 | corpus_stress_unlabeled, paired_with_existing_label | 2.116 | 0.814 | 1.000 |
| `14_A` | `tier2_pair_completion` | `corpus_stress` | `ok` | 91.27 | corpus_stress_unlabeled, low_silhouette_iou, low_cell_inside, paired_with_existing_label | 1.954 | 0.664 | 0.926 |
| `17_A` | `tier2_pair_completion` | `hard_case_stress` | `ok` | 73.07 | hard_case_unlabeled, paired_with_existing_label | 2.144 | 0.826 | 0.963 |
| `57_B` | `tier2_pair_completion` | `hard_case_stress` | `ok` | 70.17 | hard_case_unlabeled, low_silhouette_iou, low_cell_inside, paired_with_existing_label | 1.675 | 0.554 | 0.926 |
| `62_A` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 53.96 | hard_case_unlabeled | 2.272 | 0.896 | 1.000 |
| `62_B` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 53.77 | hard_case_unlabeled | 2.226 | 0.877 | 1.000 |
| `22_B` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 53.62 | hard_case_unlabeled | 2.239 | 0.862 | 1.000 |
| `58_A` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 53.52 | hard_case_unlabeled | 2.221 | 0.852 | 1.000 |
| `58_B` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 53.50 | hard_case_unlabeled | 2.187 | 0.850 | 1.000 |
| `39_A` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 53.44 | hard_case_unlabeled | 2.189 | 0.844 | 1.000 |
| `47_A` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 53.43 | hard_case_unlabeled | 2.209 | 0.862 | 0.963 |
| `47_B` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 53.40 | hard_case_unlabeled | 2.205 | 0.840 | 1.000 |
| `49_A` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 53.39 | hard_case_unlabeled | 2.196 | 0.857 | 0.963 |
| `22_A` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 53.22 | hard_case_unlabeled | 2.189 | 0.840 | 0.963 |
| `49_B` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 53.02 | hard_case_unlabeled | 2.123 | 0.802 | 1.000 |
| `48_B` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 52.60 | hard_case_unlabeled | 2.082 | 0.760 | 1.000 |
| `39_B` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 52.12 | hard_case_unlabeled | 1.995 | 0.712 | 1.000 |
| `46_B` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 52.09 | hard_case_unlabeled | 2.013 | 0.728 | 0.963 |
| `25_B` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 51.17 | hard_case_unlabeled, low_silhouette_iou | 1.944 | 0.636 | 0.963 |
| `48_A` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 50.86 | hard_case_unlabeled, low_silhouette_iou | 1.895 | 0.586 | 1.000 |
| `46_A` | `tier3_manifest_stress` | `hard_case_stress` | `ok` | 50.52 | hard_case_unlabeled, low_silhouette_iou | 1.753 | 0.570 | 0.963 |
| `25_A` | `tier4_retake_or_model_boundary` | `hard_case_stress` | `low_cell_inside` | 24.94 | hard_case_unlabeled, weak_geometry_or_retake_boundary, low_silhouette_iou, low_cell_inside | 1.639 | 0.568 | 0.852 |

## Interpretation

- This is step #1 for the next learned-localizer loop: expand the human vertex+axis labels without relabeling the completed 28-row fixture.
- Step #2 should train/evaluate the richer localizer only after a meaningful portion of this queue is labeled.
- Until then, KNN V0 remains the best diagnostics baseline and should not be production-wired.
