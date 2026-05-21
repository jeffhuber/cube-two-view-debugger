# SAM3 Mask Export V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This report records whether the local machine can export SAM3 masks into the foundation segmentation bakeoff schema.

## Status

- Status: `blocked_prerequisites`
- Blocked reason: `cuda_12_6_unavailable,sam3_package_missing,hf_token_missing`
- Rows considered: 16
- Exported masks: 0

## Environment

- Python: 3.12.13 (meets >=3.12: True)
- Torch installed: True (2.12.0)
- Torch meets >=2.7: True
- CUDA available: False (None)
- CUDA meets >=12.6: False
- MPS available: True
- sam3 package installed: False
- HF token present: False

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

- This machine cannot run official SAM3 image inference as-is because the required CUDA/Hugging Face/SAM3 prerequisites are not all present.
- The exporter is still useful: it defines the exact bridge from a capable SAM3 environment into the repo's dependency-free bakeoff harness.
- Once masks exist, the next useful metric is whether three visible-face prompts improve top-3 vertex recall over the current 3/16 source heuristic and 11/16 source-pool oracle ceiling.
