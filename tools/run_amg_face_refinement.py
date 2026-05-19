#!/usr/bin/env python3
"""Run the AMG face-quad refiner on specific corpus pairs and compare
to the baseline geometry proposer.

Research tool — NOT wired into the production recognizer or the
evaluator. Use to reproduce / explore the AMG-refinement findings
documented in `docs/amg_face_refinement_findings.md`.

Usage:
  .venv/bin/python tools/run_amg_face_refinement.py --sets 44 47 17
  .venv/bin/python tools/run_amg_face_refinement.py --all  # full corpus sweep

Prerequisites:
  .venv/bin/pip install samv2
  mkdir -p /tmp/sam2_checkpoints
  curl -sSL -o /tmp/sam2_checkpoints/sam2_hiera_tiny.pt \\
      https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.extract_color_samples import (  # noqa: E402
    discover_additional_tasks,
    load_corpus_tasks,
    parse_ground_truth,
)


def install_amg_proposer_wrapper() -> None:
    """Monkey-patch `_proposer_face_quads` to apply AMG refinement to
    slots where the topology fallback was rejected by the hull guard.

    The wrapper:
      1. Calls the original proposer
      2. Identifies slots with topologyFallbackRejectedByHullGuard=True
      3. If any such slots exist, runs AMG refinement on them
      4. Returns the refined quads (other slots pass through unchanged)

    This is the per-slot gated path that produced the +7.4pp Set 44 win.
    """
    import tools.evaluate_hybrid_pipeline as ehp
    from tools.amg_face_refiner import amg_refine_quads_per_slot
    from tools.evaluate_hybrid_pipeline import _load_processing_image

    _orig_proposer = ehp._proposer_face_quads

    def _amg_proposer(image_path, side, hull_guard=True, slot_src_filter=False,
                      fit_error_fallback=False, fit_error_threshold=4.0,
                      processing_image=None):
        if processing_image is None:
            processing_image, _ = _load_processing_image(image_path)
        orig_quads, orig_debug = _orig_proposer(
            image_path, side,
            hull_guard=hull_guard, slot_src_filter=slot_src_filter,
            fit_error_fallback=fit_error_fallback,
            fit_error_threshold=fit_error_threshold,
            processing_image=processing_image,
        )
        if not orig_quads:
            return orig_quads, orig_debug
        slots_to_refine = [
            slot for slot, meta in orig_debug.get("selectedPerFace", {}).items()
            if meta.get("topologyFallbackRejectedByHullGuard")
        ]
        if not slots_to_refine:
            return orig_quads, {**orig_debug, "amg_refine_skipped": {
                "reason": "no slots had topology fallback rejected by hull guard",
            }}
        refined, refine_debug = amg_refine_quads_per_slot(
            processing_image, orig_quads, slots_to_refine,
            snap_tolerance_px=25.0,
        )
        return refined, {**orig_debug, "amg_refine_per_slot": refine_debug}

    ehp._proposer_face_quads = _amg_proposer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sets", nargs="*", default=None,
                   help="Set IDs to run (e.g. --sets 44 17 47)")
    g.add_argument("--all", action="store_true",
                   help="Run on full corpus (~25-30 min)")
    ap.add_argument("--report", default=None,
                    help="JSON report output path")
    ap.add_argument("--summary", default=None,
                    help="text summary output path")
    args = ap.parse_args()

    install_amg_proposer_wrapper()

    # Reuse evaluate_hybrid_pipeline.main by manipulating sys.argv
    import tools.evaluate_hybrid_pipeline as ehp
    os.environ.setdefault("CUBE_RECOGNIZER_CLASSIFIER", "knn5_lab_full")
    new_argv = ["run_amg_face_refinement.py", "--fit-error-fallback"]
    if args.sets:
        new_argv.extend(["--only-sets"] + list(args.sets))
    if args.report:
        new_argv.extend(["--report", args.report])
    if args.summary:
        new_argv.extend(["--summary", args.summary])
    sys.argv = new_argv
    return ehp.main()


if __name__ == "__main__":
    sys.exit(main())
