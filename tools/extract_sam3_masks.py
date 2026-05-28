"""Pre-compute SAM 3 silhouette masks for the global cube model corpus.

SAM 3 (Meta, late 2026) supports open-vocabulary concept prompts —
"rubik's cube" returns a tight mask of the cube. On the 27-case
ground-truth corpus this gives meaningfully better silhouettes than
rembg on several cases (median vertex-error 57 px vs 72 px after the
full pipeline; <50 px count 9 vs 5; recovers 44_B from 118 → 15 px).
But it regresses on 5 cases (worst: 42_B 89 → 186 px), so we keep
rembg as the production default and treat SAM 3 as a diagnostics /
server-side option behind `--silhouette-source sam3` in
``run_global_cube_model.py``.

Setup
-----

SAM 3 itself requires CUDA. On Apple Silicon, use the community MLX
port (https://github.com/Deekshith-Dade/mlx_sam3):

    git clone https://github.com/Deekshith-Dade/mlx_sam3.git ~/sam3
    cd ~/sam3
    uv sync
    .venv/bin/pip install rembg onnxruntime  # this script doesn't need
                                              # rembg, but cube-snap deps
                                              # do for cross-comparison

Then run this script in that env:

    cd ~/sam3
    .venv/bin/python /path/to/this/extract_sam3_masks.py \\
        --out /tmp/sam3_masks \\
        --sets 12 14 15 17 21 23 24 26 27 28 29 30 31 32 36 37 42 44 47 57 58 61

Note: requires Python 3.13+ and ~3.5 GB checkpoint download on first run.
Each image takes ~7s on M1.

Outputs:
    <out>/set_<id>_<side>_sam3.npy   bool array, shape (H, W)

After extraction, pass to the main pipeline via:

    .venv/bin/python tools/run_global_cube_model.py \\
        --sets ... --silhouette-source sam3 --sam3-mask-dir /tmp/sam3_masks
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_manifests() -> list:
    out = []
    for fname in ("hard_case_manifest.json", "corpus_manifest.json"):
        with (REPO_ROOT / "tests" / "fixtures" / fname).open() as f:
            out.append(json.load(f))
    return out


def _resolve_pair(manifests: list, set_id: str):
    for manifest in manifests:
        for entry in manifest["pairs"]:
            if entry["setId"] == set_id:
                return Path(entry["imageAPath"]), Path(entry["imageBPath"])
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", nargs="+", required=True,
                        help="Set IDs to extract masks for")
    parser.add_argument("--sides", nargs="+", choices=["A", "B"],
                        default=["A", "B"])
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for .npy mask files")
    parser.add_argument("--prompt", default="rubik's cube",
                        help="Text prompt for SAM 3 (default: 'rubik's cube')")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    try:
        from sam3 import build_sam3_image_model  # type: ignore
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
    except ImportError as e:
        sys.exit(
            f"SAM 3 not installed: {e}\n"
            "Install the MLX port:\n"
            "  git clone https://github.com/Deekshith-Dade/mlx_sam3.git\n"
            "  cd mlx_sam3 && uv sync\n"
            "Then run this script in that env (Python 3.13+ required)."
        )

    print("Loading SAM 3 model...", file=sys.stderr)
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)

    manifests = _load_manifests()

    for set_id in args.sets:
        path_a, path_b = _resolve_pair(manifests, set_id)
        if path_a is None:
            print(f"[set {set_id}] not in any manifest", file=sys.stderr)
            continue
        for side, path in (("A", path_a), ("B", path_b)):
            if side not in args.sides:
                continue
            if not path.exists():
                print(f"[set {set_id} {side}] SKIP: {path}", file=sys.stderr)
                continue
            out_path = args.out / f"set_{set_id}_{side}_sam3.npy"
            if out_path.exists():
                print(f"[set {set_id} {side}] cached", file=sys.stderr)
                continue

            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            t0 = time.time()
            state = processor.set_image(img)
            state = processor.set_text_prompt(args.prompt, state)
            dt = time.time() - t0

            scores = state.get("scores", [])
            if len(scores) == 0:
                print(
                    f"[set {set_id} {side}] NO MASK from SAM 3 ({dt:.1f}s)",
                    file=sys.stderr,
                )
                continue
            score = float(scores[0])
            mask = np.asarray(state["masks"][0]).squeeze().astype(bool)
            np.save(out_path, mask)
            print(
                f"[set {set_id} {side}] score={score:.3f} area={int(mask.sum())} "
                f"({dt:.1f}s) -> {out_path.name}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
