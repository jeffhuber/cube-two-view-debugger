# Cube Mesh Anchor V0 Diagnostics

Diagnostics-only first-principles scaffold. This does not alter recognition behavior.

This probe fits a weak-projection 7-anchor cube mesh from the ranked visible trihedral vertex candidates.

## Summary

- Requested pairs: 8
- Image rows: 16
- Fitted rows: 16
- OK rows: 15
- Weak rows: 1
- Low-IoU rows: 1
- Low-anchor-support warning rows: 16
- Error/missing rows: 0
- Easy-corpus OK rows: 15 / 16
- Easy-corpus weak rows: 1
- Model-iteration-needed rows: 1

## Readout

| Set | Side | Tier | Status | Source vertex | Score | IoU | Anchor support | Face balance | Axis sep | Overlay |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
| 15 | A | `easy_corpus` | `ok` | `silhouette_centroid_seed` r1 | 2.6884 | 0.8379 | 0.5714 | 0.8609 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_15_A_cube_mesh_anchor_v0.png` |
| 15 | B | `easy_corpus` | `ok` | `center_refine` r1 | 2.681 | 0.8628 | 0.5714 | 0.8626 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_15_B_cube_mesh_anchor_v0.png` |
| 23 | A | `easy_corpus` | `ok` | `center_refine` r2 | 2.6218 | 0.8205 | 0.7143 | 0.6625 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_23_A_cube_mesh_anchor_v0.png` |
| 23 | B | `easy_corpus` | `ok` | `center_refine` r1 | 2.6963 | 0.8686 | 0.5714 | 0.8626 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_23_B_cube_mesh_anchor_v0.png` |
| 26 | A | `easy_corpus` | `ok` | `silhouette_centroid_seed` r1 | 2.4325 | 0.7024 | 0.4286 | 0.6463 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_26_A_cube_mesh_anchor_v0.png` |
| 26 | B | `easy_corpus` | `low_iou` | `silhouette_centroid_seed` r1 | 2.2263 | 0.4907 | 0.8571 | 0.5257 | 0.8571 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_26_B_cube_mesh_anchor_v0.png` |
| 29 | A | `easy_corpus` | `ok` | `center_refine` r5 | 2.7279 | 0.8425 | 0.7143 | 0.8015 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_29_A_cube_mesh_anchor_v0.png` |
| 29 | B | `easy_corpus` | `ok` | `bezel_detector` r1 | 2.6788 | 0.8596 | 0.5714 | 0.8856 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_29_B_cube_mesh_anchor_v0.png` |
| 32 | A | `easy_corpus` | `ok` | `center_refine` r3 | 2.7545 | 0.8477 | 0.7143 | 0.8856 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_32_A_cube_mesh_anchor_v0.png` |
| 32 | B | `easy_corpus` | `ok` | `center_refine` r3 | 2.6979 | 0.8628 | 0.7143 | 0.8262 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_32_B_cube_mesh_anchor_v0.png` |
| 36 | A | `easy_corpus` | `ok` | `silhouette_centroid_seed` r1 | 2.7445 | 0.8431 | 0.7143 | 0.8385 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_36_A_cube_mesh_anchor_v0.png` |
| 36 | B | `easy_corpus` | `ok` | `center_refine` r1 | 2.704 | 0.8471 | 0.7143 | 0.8055 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_36_B_cube_mesh_anchor_v0.png` |
| 37 | A | `easy_corpus` | `ok` | `center_refine` r1 | 2.6631 | 0.8182 | 0.7143 | 0.7895 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_37_A_cube_mesh_anchor_v0.png` |
| 37 | B | `easy_corpus` | `ok` | `center_refine` r1 | 2.4087 | 0.6599 | 0.7143 | 0.6252 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_37_B_cube_mesh_anchor_v0.png` |
| 42 | A | `easy_corpus` | `ok` | `silhouette_centroid_seed` r3 | 2.3154 | 0.5989 | 0.8571 | 0.5003 | 0.8571 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_42_A_cube_mesh_anchor_v0.png` |
| 42 | B | `easy_corpus` | `ok` | `center_refine` r1 | 2.6665 | 0.8408 | 0.7143 | 0.6896 | 1.0 | `/tmp/cube_mesh_anchor_v0_easy_corpus_overlays/set_42_B_cube_mesh_anchor_v0.png` |

## Interpretation

- `V` is the visible trihedral vertex point; `X/Y/Z` are one shared edge length away; `XY/YZ/ZX` close the three visible face parallelograms.
- This is a weak projected-image mesh, not a calibrated 3D pose solve. The report records `pnpStatus=not_run_no_calibrated_camera_or_human_anchor_labels` in row diagnostics.
- Anchor-near-silhouette support is reported as a warning metric only; it is not a V0 status gate because hull-edge anchors sit on noisy rembg boundaries.
- Easy-corpus weak rows should drive model iteration. Hard-background weak rows remain retake/segmentation candidates rather than forced recognizer wins.
- The companion vertex human-feedback artifact should decide whether these meshes are anchored on the correct visible vertex before any downstream color sampling is considered.
