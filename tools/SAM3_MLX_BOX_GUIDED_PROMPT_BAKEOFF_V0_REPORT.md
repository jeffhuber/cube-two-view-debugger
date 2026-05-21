# SAM3 MLX Box-Guided Prompt Bakeoff V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This report captures a live MLX SAM3 sweep using geometry-guided box prompts derived from the current 7-anchor cube mesh face boxes. It follows #194, which showed that plain text prompts for `top_face`, `left_face`, and `right_face` are not usable vertex signals.

## Setup

- Runtime: `/Users/jhuber/sam3_artifacts/sam3_probe/mlx_sam3/.venv/bin/python`
- MLX SAM3 source: `/Users/jhuber/sam3_artifacts/sam3_probe/mlx_sam3`
- Box source: `tests/fixtures/cube_mesh_anchor_v0_easy_corpus_summary.json`
- External masks: `/Users/jhuber/sam3_artifacts/foundation_masks_mlx_box_guided_probe`
- Fixture: `tests/fixtures/sam3_mlx_box_guided_prompt_bakeoff_v0_easy_summary.json`

Each row uses the top 7-anchor mesh. Its three face parallelograms become axis-aligned positive box prompts:

- `top_face`: `V-X-XY-Y`
- `left_face`: `V-Y-YZ-Z`
- `right_face`: `V-Z-ZX-X`

Two policies were tested:

- `box_visual`: positive geometric box prompt only.
- `box_face_text`: text prompt `visible face of the Rubik cube`, then a positive geometric box prompt.

## Result

| Policy | Rows | Rows with masks | Rows with 3 face masks | Candidate rows | Top-3 @10 | Oracle @20 | Mean best distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| `box_visual` | 16 | 16 | 16 | 16 | 0 | 0 | 150.50 px |
| `box_face_text` | 16 | 16 | 16 | 16 | 0 | 0 | 133.03 px |

Box guidance improved the distance scale compared with plain text prompts, but it still failed the vertex-selection threshold completely. Neither policy produced a single top-3 hit within 10 px or oracle hit within 20 px.

Closest rows:

| Policy | Best rows by distance |
|---|---|
| `box_visual` | Set 42 A 69.95 px; Set 26 A 97.36 px; Set 32 A 113.67 px; Set 26 B 117.43 px; Set 29 B 119.49 px |
| `box_face_text` | Set 42 A 91.78 px; Set 36 A 112.73 px; Set 26 B 113.80 px; Set 32 A 114.32 px; Set 29 B 116.60 px |

## Interpretation

- Current 7-anchor geometry boxes are not sufficient SAM3 guidance for reliable face-plane masks.
- Adding a generic face text prompt to the boxes helps mean distance but does not produce usable vertex recall.
- Do not wire SAM3 box-guided face masks into vertex selection or rectification.
- The next SAM3 path, if any, should either use stronger interactive prompts tied to a trusted vertex/hull edge, or shift to whole-cube silhouette comparison and geometry-first face splitting.
