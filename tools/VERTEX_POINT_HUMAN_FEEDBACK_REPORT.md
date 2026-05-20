# Vertex Point Human Feedback

Diagnostics/data-only artifact. This does not alter recognition behavior.

The purpose is to validate whether the ranked vertex-point candidates actually hit the human-visible trihedral corner.

## Summary

- Distance threshold: 10px
- Rows: 16
- Labeled rows: 0
- Unlabeled rows: 16
- Ambiguous rows: 0
- Not-visible rows: 0
- Invalid-label rows: 0
- Top-1 hits: 0
- Top-3 hits: 0
- Top-3 misses: 0
- Top-1 hit rate: n/a
- Top-3 hit rate: n/a

## Readout

| Set | Side | Tier | Label | Candidate status | Top-1 dist | Top-1 hit | Top-3 hit | Best rank | Best dist | Overlay |
|---:|---|---|---|---|---:|---|---|---:|---:|---|
| 15 | A | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_15_A_vertex_point_candidates.png` |
| 15 | B | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_15_B_vertex_point_candidates.png` |
| 23 | A | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_23_A_vertex_point_candidates.png` |
| 23 | B | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_23_B_vertex_point_candidates.png` |
| 26 | A | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_26_A_vertex_point_candidates.png` |
| 26 | B | `easy_corpus` | `unlabeled` | `low_iou` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_26_B_vertex_point_candidates.png` |
| 29 | A | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_29_A_vertex_point_candidates.png` |
| 29 | B | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_29_B_vertex_point_candidates.png` |
| 32 | A | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_32_A_vertex_point_candidates.png` |
| 32 | B | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_32_B_vertex_point_candidates.png` |
| 36 | A | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_36_A_vertex_point_candidates.png` |
| 36 | B | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_36_B_vertex_point_candidates.png` |
| 37 | A | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_37_A_vertex_point_candidates.png` |
| 37 | B | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_37_B_vertex_point_candidates.png` |
| 42 | A | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_42_A_vertex_point_candidates.png` |
| 42 | B | `easy_corpus` | `unlabeled` | `ok` |  |  |  |  |  | `/tmp/vertex_point_candidate_easy_corpus_overlays/set_42_B_vertex_point_candidates.png` |

## Interpretation

- Source candidate summary: `tests/fixtures/vertex_point_candidates_easy_corpus_summary.json`
- Rows begin unlabeled by design. Human feedback should only add label fields; candidate coordinates should remain generated data.
- Top-1 precision tells us whether the automatic first choice is trustworthy enough to seed downstream geometry.
- Top-3 recall tells us whether the correct vertex is at least present in the candidate set for a later fitter or manual review.
- A top-3 miss on an easy-corpus row is a geometry-init failure, not a color-classification problem.
