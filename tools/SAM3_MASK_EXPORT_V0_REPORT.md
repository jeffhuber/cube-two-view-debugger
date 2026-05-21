# SAM3 Mask Export V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This report records whether the local machine can export MLX SAM3 masks into the foundation segmentation bakeoff schema.

## Status

- Status: `cached_import_completed`
- Blocked reason: `python_lt_3_13,mlx_package_missing,mlx_sam3_package_missing`
- Rows considered: 16
- Exported masks: 28
- Cached whole-cube masks imported: 28

## Environment

- Python: 3.12.13 (meets >=3.13: False)
- Platform: Darwin arm64 (Apple Silicon Mac: True)
- MLX installed: False
- Torch installed: True (2.12.0)
- Torch meets >=2.9: True
- CUDA available: False (None)
- CUDA required: False
- MPS available: True
- MLX SAM3 package installed: False
- HF token required: False

## Mask Schema

When prerequisites are available, masks are written to:

```text
<mask-dir>/sam3/set_<SET>_<SIDE>_<prompt>.png
```

These masks can then be scored with:

```bash
.venv/bin/python tools/foundation_segmentation_bakeoff_v0.py --external-mask-dir <mask-dir>
```

## Interpretation

- This exporter targets the community `Deekshith-Dade/mlx_sam3` Apple-Silicon port, not Meta's CUDA/HF-gated official package.
- The exporter is useful because it defines the exact bridge from a capable MLX SAM3 environment into the repo's dependency-free bakeoff harness.
- Cached `.npy` whole-cube masks can be imported without a SAM3 runtime; they prove the interchange path and provide silhouette coverage, but they do not by themselves provide the three face masks needed for vertex candidate scoring.
- Once masks exist, the next useful metric is whether three visible-face prompts improve top-3 vertex recall over the current 3/16 source heuristic and 11/16 source-pool oracle ceiling.

## Cached Whole-Cube Import

| Set | Side | Status | Source shape | Output shape | Resized | Mask pixels | Output |
|---:|---|---|---|---|---:|---:|---|
| 12 | B | `imported` | [4032, 3024] | [4032, 3024] | False | 2960377 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_12_B_whole_cube.png` |
| 14 | B | `imported` | [4032, 3024] | [4032, 3024] | False | 3558288 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_14_B_whole_cube.png` |
| 15 | A | `imported` | [4032, 3024] | [1150, 862] | True | 270882 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_15_A_whole_cube.png` |
| 15 | B | `imported` | [4032, 3024] | [1150, 862] | True | 279874 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_15_B_whole_cube.png` |
| 17 | B | `imported` | [4032, 3024] | [4032, 3024] | False | 3248482 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_17_B_whole_cube.png` |
| 21 | A | `imported` | [4032, 3024] | [4032, 3024] | False | 3429101 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_21_A_whole_cube.png` |
| 21 | B | `imported` | [4032, 3024] | [4032, 3024] | False | 3528631 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_21_B_whole_cube.png` |
| 24 | A | `imported` | [4032, 3024] | [4032, 3024] | False | 3162234 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_24_A_whole_cube.png` |
| 26 | A | `imported` | [4032, 3024] | [1150, 862] | True | 240835 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_26_A_whole_cube.png` |
| 26 | B | `imported` | [4032, 3024] | [1150, 862] | True | 264542 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_26_B_whole_cube.png` |
| 27 | B | `imported` | [4032, 3024] | [4032, 3024] | False | 2907429 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_27_B_whole_cube.png` |
| 28 | A | `imported` | [4032, 3024] | [4032, 3024] | False | 3404254 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_28_A_whole_cube.png` |
| 28 | B | `imported` | [4032, 3024] | [4032, 3024] | False | 3067367 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_28_B_whole_cube.png` |
| 29 | A | `imported` | [4032, 3024] | [1150, 862] | True | 277974 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_29_A_whole_cube.png` |
| 29 | B | `imported` | [4032, 3024] | [1150, 862] | True | 241608 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_29_B_whole_cube.png` |
| 30 | A | `imported` | [4032, 3024] | [4032, 3024] | False | 2055271 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_30_A_whole_cube.png` |
| 30 | B | `imported` | [4032, 3024] | [4032, 3024] | False | 2177665 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_30_B_whole_cube.png` |
| 31 | A | `imported` | [4032, 3024] | [4032, 3024] | False | 2329042 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_31_A_whole_cube.png` |
| 31 | B | `imported` | [4032, 3024] | [4032, 3024] | False | 2549153 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_31_B_whole_cube.png` |
| 32 | A | `imported` | [4032, 3024] | [1150, 862] | True | 243185 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_32_A_whole_cube.png` |
| 32 | B | `imported` | [4032, 3024] | [1150, 862] | True | 236264 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_32_B_whole_cube.png` |
| 36 | B | `imported` | [4032, 3024] | [1150, 862] | True | 222741 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_36_B_whole_cube.png` |
| 42 | B | `imported` | [4032, 3024] | [1150, 862] | True | 350974 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_42_B_whole_cube.png` |
| 44 | A | `imported` | [4032, 3024] | [4032, 3024] | False | 3907979 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_44_A_whole_cube.png` |
| 44 | B | `imported` | [4032, 3024] | [4032, 3024] | False | 3748777 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_44_B_whole_cube.png` |
| 57 | A | `imported` | [4032, 3024] | [4032, 3024] | False | 3040752 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_57_A_whole_cube.png` |
| 61 | A | `imported` | [4032, 3024] | [4032, 3024] | False | 3475553 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_61_A_whole_cube.png` |
| 61 | B | `imported` | [4032, 3024] | [4032, 3024] | False | 3110035 | `/Users/jhuber/sam3_artifacts/foundation_masks/sam3/set_61_B_whole_cube.png` |
