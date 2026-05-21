# SAM3 MLX Current Prompt Bakeoff V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This report captures a live MLX SAM3 sweep using the current foundation-segmentation prompt schema from #192/#193 over the 16 human-labeled easy-corpus vertex rows.

## Setup

- Runtime: `/Users/jhuber/sam3_artifacts/sam3_probe/mlx_sam3/.venv/bin/python`
- MLX SAM3 source: `/Users/jhuber/sam3_artifacts/sam3_probe/mlx_sam3`
- External masks: `/Users/jhuber/sam3_artifacts/foundation_masks_mlx_current`
- Bakeoff overlays: `/tmp/sam3_mlx_current_bakeoff_overlays`
- Fixture: `tests/fixtures/sam3_mlx_current_prompt_bakeoff_v0_easy_summary.json`

## Result

| Provider | Rows | Rows with masks | Rows with 3 face masks | Candidate rows | Top-3 @10 | Oracle @20 |
|---|---:|---:|---:|---:|---:|---:|
| `sam3` | 16 | 16 | 16 | 16 | 0 | 0 |
| `falcon` | 16 | 0 | 0 | 0 | 0 | 0 |

The current SAM3 prompt set is a negative vertex-selection result. It produces masks for all easy rows and technically gives the bakeoff three face-mask inputs for every row, but the resulting vertex candidates are nowhere near the human-labeled visible trihedral vertex.

## Distance Readout

- Mean best candidate distance: 229.91 px
- Mean top-3 distance: 262.16 px
- Mean top-5 distance: 251.52 px
- Best rows were still misses: Set 29 B at 104.07 px, Set 23 B at 109.36 px, Set 23 A at 115.25 px, Set 15 A at 120.66 px, Set 15 B at 122.86 px.

## Mask Shape Signal

Average area fraction by prompt:

| Prompt | Mean area fraction |
|---|---:|
| `whole_cube` | 0.266322 |
| `top_face` | 0.211939 |
| `left_face` | 0.102825 |
| `right_face` | 0.189063 |
| `stickers` | 0.267291 |

The coarse failure pattern is that `top_face`, `stickers`, and `whole_cube` are often similar large regions, while `left_face` and `right_face` are inconsistent subregions. The bakeoff dutifully generates face-boundary intersections from those masks, but they are intersections of prompt artifacts rather than the cube's true visible face planes.

## Interpretation

- Do not wire current text-prompt SAM3 face masks into vertex selection or rectification.
- Whole-cube SAM3 masks remain useful as a silhouette cross-check candidate.
- The next useful SAM3 experiment should be a prompt/adapter search that targets the three actual visible face planes, not the generic `top visible face`, `left visible face`, and `right visible face` prompts.
- A better next shape is either interactive point/box-prompting from the detected cube hull/vertex candidates, or a mask-postprocessing step that splits a whole-cube/sticker mask into three planar regions using geometry.
