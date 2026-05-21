# Foundation Segmentation Bakeoff V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This harness evaluates whether SAM3/Falcon-style masks can provide face-plane or sticker/grid evidence for the visible trihedral vertex and future rectification.

## Current Status

`sam3` status `package_missing_external_masks_required`, top-3 @10px 0 / 16; `falcon` status `package_missing_external_masks_required`, top-3 @10px 0 / 16. Direct model dependencies are intentionally optional; current committed output is a harness/scaffold unless external masks are supplied.

## Provider Availability

| Provider | Package installed | External mask files | Status | Notes |
|---|---:|---:|---|---|
| `sam3` | False | 0 | `package_missing_external_masks_required` | Promptable segmentation candidate; direct adapter intentionally not wired until local package/API is stable. |
| `falcon` | False | 0 | `package_missing_external_masks_required` | Open-vocabulary detection/segmentation candidate; external masks keep the repo dependency-free. |

## Prompt Matrix

| Key | Prompt | Role |
|---|---|---|
| `whole_cube` | Rubik's cube | `cube_silhouette` |
| `top_face` | top visible face of the Rubik's cube | `visible_face` |
| `left_face` | left visible face of the Rubik's cube | `visible_face` |
| `right_face` | right visible face of the Rubik's cube | `visible_face` |
| `stickers` | colored sticker squares on the Rubik's cube | `stickers` |
| `black_grid_lines` | black plastic grid lines between stickers | `grid_or_bezel` |

## Metrics

| Provider | Rows | Rows with masks | Rows with 3 face masks | Top-1 @10 | Top-3 @10 | Top-5 @10 | Oracle @10 | Top-3 @20 | Oracle @20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sam3` | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `falcon` | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Per-Row Readout

| Set | Side | Provider | Mask prompts | Candidates | Best distance | Top-3 distance | Overlay | Notes |
|---:|---|---|---|---:|---:|---:|---|---|
| 15 | A | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 15 | A | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 15 | B | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 15 | B | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 23 | A | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 23 | A | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 23 | B | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 23 | B | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 26 | A | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 26 | A | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 26 | B | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 26 | B | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 29 | A | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 29 | A | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 29 | B | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 29 | B | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 32 | A | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 32 | A | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 32 | B | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 32 | B | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 36 | A | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 36 | A | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 36 | B | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 36 | B | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 37 | A | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 37 | A | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 37 | B | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 37 | B | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 42 | A | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 42 | A | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 42 | B | `sam3` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |
| 42 | B | `falcon` |  | 0 | None | None |  | direct model inference is not wired; provide external masks to score this provider |

## External Mask Schema

Place masks at:

```text
<mask-dir>/<provider>/set_<SET>_<SIDE>_<prompt>.png
```

Example:

```text
/tmp/foundation_masks/sam3/set_15_A_top_face.png
/tmp/foundation_masks/sam3/set_15_A_left_face.png
/tmp/foundation_masks/sam3/set_15_A_right_face.png
```

Mask pixels are read from alpha or grayscale values greater than 128.

## Interpretation

- The most important first metric is whether three visible-face masks generate vertex candidates near the human-labeled visible trihedral vertex.
- Whole-cube masks are useful as a silhouette replacement/cross-check, but face masks are the key signal for rectification and vertex selection.
- A future direct SAM3/Falcon adapter should write exactly this external-mask schema first, then graduate to in-process inference only if dependency/runtime cost is acceptable.
