# Vertex Point Candidate Diagnostics

Diagnostics-only first-principles scaffold. This does not alter recognition behavior.

The vertex point is the visible trihedral corner where the three visible cube faces meet. This report ranks candidate vertex points before any production wiring.

## Summary

- Requested pairs: 8
- Image rows: 16
- Rows with candidates: 16
- Top candidate OK rows: 15
- Top candidate weak rows: 1
- Easy-corpus top OK rows: 15 / 16
- Easy-corpus top weak rows: 1
- Error/missing rows: 0
- Unlabeled manual-review rows: 16

## Readout

| Set | Side | Tier | Status | Top source | Top point | Score | IoU | Inside | Cell inside | Same-status gap | Detector signal | Candidates | Overlay |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 15 | A | `easy_corpus` | `ok` | `silhouette_centroid_seed` | `[430.4, 571.4]` | 2.2162 | 0.8379 | 0.9605 | 1.0 | 0.2356 | 0.3125 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_15_A_vertex_point_candidates.png` |
| 15 | B | `easy_corpus` | `ok` | `center_refine` | `[460.42, 579.14]` | 2.2085 | 0.8628 | 0.9448 | 1.0 | 0.0084 | 0.1208 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_15_B_vertex_point_candidates.png` |
| 23 | A | `easy_corpus` | `ok` | `center_refine` | `[382.86, 529.44]` | 2.1455 | 0.806 | 0.9623 | 1.0 | 0.0062 | 0.1024 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_23_A_vertex_point_candidates.png` |
| 23 | B | `easy_corpus` | `ok` | `center_refine` | `[400.4, 545.71]` | 2.2238 | 0.8686 | 0.931 | 1.0 | 0.0004 | 0.2341 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_23_B_vertex_point_candidates.png` |
| 26 | A | `easy_corpus` | `ok` | `silhouette_centroid_seed` | `[432.4, 549.7]` | 2.0533 | 0.7024 | 0.9414 | 0.963 | 0.1646 | 0.5282 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_26_A_vertex_point_candidates.png` |
| 26 | B | `easy_corpus` | `low_iou` | `silhouette_centroid_seed` | `[413.5, 577.9]` | 1.7354 | 0.4907 | 0.958 | 1.0 | 0.0727 | 0.0472 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_26_B_vertex_point_candidates.png` |
| 29 | A | `easy_corpus` | `ok` | `bezel_detector` | `[456.68, 604.45]` | 2.2628 | 0.8513 | 0.9626 | 1.0 | 0.0105 | 0.5 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_29_A_vertex_point_candidates.png` |
| 29 | B | `easy_corpus` | `ok` | `bezel_detector` | `[398.3, 555.58]` | 2.2017 | 0.8596 | 0.9676 | 1.0 | 0.0057 | 0.0 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_29_B_vertex_point_candidates.png` |
| 32 | A | `easy_corpus` | `ok` | `center_refine` | `[433.74, 543.01]` | 2.2317 | 0.863 | 0.9497 | 1.0 | 0.0011 | 0.2527 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_32_A_vertex_point_candidates.png` |
| 32 | B | `easy_corpus` | `ok` | `bezel_detector` | `[399.17, 514.92]` | 2.1896 | 0.8485 | 0.9637 | 1.0 | 0.0067 | 0.0308 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_32_B_vertex_point_candidates.png` |
| 36 | A | `easy_corpus` | `ok` | `silhouette_centroid_seed` | `[444.0, 514.3]` | 2.2268 | 0.8431 | 0.983 | 1.0 | 0.3408 | 0.2339 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_36_A_vertex_point_candidates.png` |
| 36 | B | `easy_corpus` | `ok` | `center_refine` | `[360.87, 509.12]` | 2.1929 | 0.8471 | 0.9582 | 1.0 | 0.0085 | 0.0904 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_36_B_vertex_point_candidates.png` |
| 37 | A | `easy_corpus` | `ok` | `center_refine` | `[399.14, 525.16]` | 2.1552 | 0.8182 | 0.9715 | 1.0 | 0.0028 | 0.0207 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_37_A_vertex_point_candidates.png` |
| 37 | B | `easy_corpus` | `ok` | `center_refine` | `[363.39, 557.3]` | 1.9337 | 0.6599 | 0.9436 | 1.0 | 0.0241 | 0.0188 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_37_B_vertex_point_candidates.png` |
| 42 | A | `easy_corpus` | `ok` | `center_refine` | `[512.55, 480.86]` | 1.8981 | 0.6424 | 0.9213 | 1.0 | 0.0266 | 0.0445 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_42_A_vertex_point_candidates.png` |
| 42 | B | `easy_corpus` | `ok` | `center_refine` | `[440.31, 520.65]` | 2.1786 | 0.8408 | 0.9548 | 1.0 | 0.0196 | 0.0637 | 5 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_42_B_vertex_point_candidates.png` |

## Interpretation

- Ranking prefers candidates whose coherent projected cube model clears the existing geometry thresholds, then sorts by model score.
- Top-5 candidates are preserved so human labels can measure top-1 precision and top-3 recall instead of forcing a single unreviewed answer.
- A weak top candidate on an easy-corpus row means the vertex point is still a geometry-model iteration target, not a production fallback target.
- The next useful human input is marking the true vertex point on these overlays and recording whether top-1 is within 10 px and top-3 contains the correct point.
