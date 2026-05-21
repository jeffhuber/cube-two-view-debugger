# Vertex Point Human Feedback

Diagnostics/data-only artifact. This does not alter recognition behavior.

The purpose is to validate whether the ranked vertex-point candidates actually hit the human-visible trihedral corner.

## Summary

- Distance threshold: 10px
- Rows: 16
- Labeled rows: 16
- Unlabeled rows: 0
- Ambiguous rows: 0
- Not-visible rows: 0
- Invalid-label rows: 0
- Top-1 hits: 1
- Top-3 hits: 2
- Top-3 misses: 14
- Top-1 hit rate: 6.2%
- Top-3 hit rate: 12.5%

## Readout

| Set | Side | Tier | Label | Candidate status | Top-1 dist | Top-1 hit | Top-3 hit | Best rank | Best dist | Overlay |
|---:|---|---|---|---|---:|---|---|---:|---:|---|
| 15 | A | `easy_corpus` | `labeled` | `ok` | 30.69 | no | no | 1 | 30.69 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_15_A_vertex_point_candidates.png` |
| 15 | B | `easy_corpus` | `labeled` | `ok` | 18.37 | no | no | 4 | 12.87 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_15_B_vertex_point_candidates.png` |
| 23 | A | `easy_corpus` | `labeled` | `ok` | 20.7 | no | no | 4 | 13.33 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_23_A_vertex_point_candidates.png` |
| 23 | B | `easy_corpus` | `labeled` | `ok` | 31.06 | no | no | 3 | 22.78 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_23_B_vertex_point_candidates.png` |
| 26 | A | `easy_corpus` | `labeled` | `ok` | 48.38 | no | no | 1 | 48.38 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_26_A_vertex_point_candidates.png` |
| 26 | B | `easy_corpus` | `labeled` | `low_iou` | 21.7 | no | no | 1 | 21.7 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_26_B_vertex_point_candidates.png` |
| 29 | A | `easy_corpus` | `labeled` | `ok` | 37.13 | no | yes | 3 | 6.05 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_29_A_vertex_point_candidates.png` |
| 29 | B | `easy_corpus` | `labeled` | `ok` | 10.01 | no | no | 1 | 10.01 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_29_B_vertex_point_candidates.png` |
| 32 | A | `easy_corpus` | `labeled` | `ok` | 29.2 | no | no | 5 | 6.88 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_32_A_vertex_point_candidates.png` |
| 32 | B | `easy_corpus` | `labeled` | `ok` | 29.51 | no | no | 4 | 8.59 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_32_B_vertex_point_candidates.png` |
| 36 | A | `easy_corpus` | `labeled` | `ok` | 63.68 | no | no | 1 | 63.68 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_36_A_vertex_point_candidates.png` |
| 36 | B | `easy_corpus` | `labeled` | `ok` | 49.94 | no | no | 4 | 27.59 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_36_B_vertex_point_candidates.png` |
| 37 | A | `easy_corpus` | `labeled` | `ok` | 81.4 | no | no | 3 | 58.0 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_37_A_vertex_point_candidates.png` |
| 37 | B | `easy_corpus` | `labeled` | `ok` | 114.58 | no | no | 2 | 93.46 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_37_B_vertex_point_candidates.png` |
| 42 | A | `easy_corpus` | `labeled` | `ok` | 86.75 | no | no | 3 | 47.13 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_42_A_vertex_point_candidates.png` |
| 42 | B | `easy_corpus` | `labeled` | `ok` | 6.84 | yes | yes | 1 | 6.84 | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_42_B_vertex_point_candidates.png` |

## Interpretation

- Source candidate summary: `tests/fixtures/vertex_point_candidates_easy_corpus_summary.json`
- Rows begin unlabeled by design. Human feedback should only add label fields; candidate coordinates should remain generated data.
- Top-1 precision tells us whether the automatic first choice is trustworthy enough to seed downstream geometry.
- Top-3 recall tells us whether the correct vertex is at least present in the candidate set for a later fitter or manual review.
- A top-3 miss on an easy-corpus row is a geometry-init failure, not a color-classification problem.
