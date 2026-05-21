#!/usr/bin/env python3
"""Export SAM3 masks into the foundation-segmentation bakeoff schema.

Diagnostics/data-only. This module does not alter recognizer behavior.

PR #192 established the stable external-mask layout consumed by
``foundation_segmentation_bakeoff_v0.py``:

    <mask-dir>/sam3/set_<SET>_<SIDE>_<prompt>.png

This exporter is the thin SAM3 adapter for that layout. It intentionally keeps
SAM3 optional: when the package/checkpoint/GPU prerequisites are unavailable it
writes a structured environment report instead of failing midway through the
diagnostic workflow.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.foundation_segmentation_bakeoff_v0 import PROMPT_SPECS  # noqa: E402


DEFAULT_FEEDBACK = ROOT / "tests" / "fixtures" / "vertex_point_human_feedback.json"
DEFAULT_MASK_DIR = Path("/tmp/foundation_masks")
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "sam3_mask_export_v0_environment_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "SAM3_MASK_EXPORT_V0_REPORT.md"
DEFAULT_SCORE_THRESHOLD = 0.0
DEFAULT_MAX_INSTANCES = 3


@dataclass(frozen=True)
class ExportConfig:
    feedback_path: Path = DEFAULT_FEEDBACK
    mask_dir: Path = DEFAULT_MASK_DIR
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    max_instances: int = DEFAULT_MAX_INSTANCES
    limit_rows: Optional[int] = None
    prompts: Tuple[str, ...] = tuple(prompt.key for prompt in PROMPT_SPECS)


def generate_sam3_export_artifacts(
    config: ExportConfig,
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """Check SAM3 environment and export masks when possible."""
    environment = sam3_environment_status()
    feedback = _read_json(config.feedback_path)
    rows = _feedback_rows(feedback, limit_rows=config.limit_rows)
    document: Dict[str, Any] = {
        "schemaVersion": 1,
        "probe": "sam3_mask_export_v0",
        "description": (
            "Diagnostics-only SAM3 external-mask exporter for the foundation "
            "segmentation bakeoff schema."
        ),
        "sourceFeedback": str(config.feedback_path),
        "config": {
            "maskDir": str(config.mask_dir),
            "scoreThreshold": float(config.score_threshold),
            "maxInstances": int(config.max_instances),
            "limitRows": config.limit_rows,
            "prompts": list(config.prompts),
        },
        "environment": environment,
        "summary": {
            "status": "blocked_prerequisites",
            "rowCount": len(rows),
            "exportedMaskCount": 0,
            "blockedReason": environment["blockedReason"],
        },
        "rows": [],
    }
    if not environment["canAttemptInference"]:
        if strict:
            document["summary"]["strictExitCode"] = 2
        return document

    exporter = Sam3MaskExporter(config)
    exported_rows: List[Dict[str, Any]] = []
    exported_count = 0
    for row in rows:
        exported = exporter.export_row(row)
        exported_rows.append(exported)
        exported_count += int(exported.get("exportedMaskCount") or 0)
    document["summary"] = {
        "status": "completed",
        "rowCount": len(exported_rows),
        "exportedMaskCount": exported_count,
        "blockedReason": None,
    }
    document["rows"] = exported_rows
    return document


def sam3_environment_status() -> Dict[str, Any]:
    """Return a machine-readable SAM3 prerequisite report."""
    status: Dict[str, Any] = {
        "pythonVersion": platform.python_version(),
        "pythonMeetsRequirement": sys.version_info >= (3, 12),
        "sam3PackageInstalled": importlib.util.find_spec("sam3") is not None,
        "hfTokenPresent": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "torchInstalled": False,
        "torchVersion": None,
        "torchMeetsRequirement": False,
        "cudaAvailable": False,
        "cudaVersion": None,
        "cudaMeetsRequirement": False,
        "mpsAvailable": False,
        "blockedReason": None,
        "canAttemptInference": False,
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        status["blockedReason"] = f"torch_unavailable:{exc.__class__.__name__}"
        return status

    status["torchInstalled"] = True
    status["torchVersion"] = str(getattr(torch, "__version__", ""))
    status["torchMeetsRequirement"] = _version_at_least(status["torchVersion"], (2, 7))
    status["cudaAvailable"] = bool(torch.cuda.is_available())
    status["cudaVersion"] = getattr(torch.version, "cuda", None)
    status["cudaMeetsRequirement"] = (
        status["cudaAvailable"]
        and status["cudaVersion"] is not None
        and _version_at_least(str(status["cudaVersion"]), (12, 6))
    )
    status["mpsAvailable"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    blockers = []
    if not status["pythonMeetsRequirement"]:
        blockers.append("python_lt_3_12")
    if not status["torchMeetsRequirement"]:
        blockers.append("torch_lt_2_7")
    if not status["cudaMeetsRequirement"]:
        blockers.append("cuda_12_6_unavailable")
    if not status["sam3PackageInstalled"]:
        blockers.append("sam3_package_missing")
    if not status["hfTokenPresent"]:
        blockers.append("hf_token_missing")
    status["blockedReason"] = ",".join(blockers) if blockers else None
    status["canAttemptInference"] = not blockers
    return status


class Sam3MaskExporter:
    """Lazy SAM3 image-prompt exporter."""

    def __init__(self, config: ExportConfig):
        self.config = config
        self._processor = None

    @property
    def processor(self):
        if self._processor is None:
            from sam3.model_builder import build_sam3_image_model  # type: ignore
            from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore

            model = build_sam3_image_model()
            self._processor = Sam3Processor(model)
        return self._processor

    def export_row(self, feedback_row: Dict[str, Any]) -> Dict[str, Any]:
        set_id = str(feedback_row.get("setId"))
        side = str(feedback_row.get("side"))
        image_path = Path(str(feedback_row.get("imagePath") or ""))
        base = {
            "setId": set_id,
            "side": side,
            "imagePath": str(image_path),
            "status": "pending",
            "exportedMaskCount": 0,
            "prompts": [],
        }
        if not image_path.exists():
            return {**base, "status": "image_missing"}

        image = Image.open(image_path).convert("RGB")
        state = self.processor.set_image(image)
        prompt_rows = []
        exported_count = 0
        for prompt_spec in PROMPT_SPECS:
            if prompt_spec.key not in self.config.prompts:
                continue
            output = self.processor.set_text_prompt(state=state, prompt=prompt_spec.prompt)
            mask, stats = select_mask_from_sam3_output(
                output,
                score_threshold=self.config.score_threshold,
                max_instances=self.config.max_instances,
            )
            prompt_row = {
                "key": prompt_spec.key,
                "prompt": prompt_spec.prompt,
                "role": prompt_spec.role,
                "status": "no_mask",
                "scores": stats.get("scores", []),
                "selectedInstanceCount": stats.get("selectedInstanceCount", 0),
                "outputPath": None,
            }
            if mask is not None:
                output_path = self.config.mask_dir / "sam3" / f"set_{set_id}_{side}_{prompt_spec.key}.png"
                _write_mask(output_path, mask)
                exported_count += 1
                prompt_row.update({
                    "status": "exported",
                    "outputPath": str(output_path),
                    "maskPixels": int(mask.sum()),
                })
            prompt_rows.append(prompt_row)
        return {
            **base,
            "status": "completed",
            "exportedMaskCount": exported_count,
            "prompts": prompt_rows,
        }


def select_mask_from_sam3_output(
    output: Dict[str, Any],
    *,
    score_threshold: float,
    max_instances: int,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Select/union SAM3 masks into one prompt mask for the bakeoff schema."""
    masks = _to_numpy(output.get("masks"))
    scores = _to_numpy(output.get("scores"))
    if masks is None or masks.size == 0:
        return None, {"scores": [], "selectedInstanceCount": 0}
    masks = _normalize_mask_array(masks)
    if scores is None or scores.size == 0:
        scores = np.ones((masks.shape[0],), dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = np.argsort(-scores)[: max(1, max_instances)]
    selected = [
        idx for idx in order
        if float(scores[idx]) >= score_threshold
    ]
    if not selected and len(order):
        selected = [int(order[0])]
    if not selected:
        return None, {
            "scores": [round(float(value), 4) for value in scores.tolist()],
            "selectedInstanceCount": 0,
        }
    union = np.any(masks[selected].astype(bool), axis=0)
    return union, {
        "scores": [round(float(value), 4) for value in scores.tolist()],
        "selectedInstanceCount": len(selected),
        "selectedIndexes": [int(idx) for idx in selected],
    }


def render_report(document: Dict[str, Any]) -> str:
    environment = document["environment"]
    summary = document["summary"]
    lines = [
        "# SAM3 Mask Export V0",
        "",
        "Diagnostics/data-only artifact. This does not alter recognition behavior.",
        "",
        "This report records whether the local machine can export SAM3 masks into the foundation segmentation bakeoff schema.",
        "",
        "## Status",
        "",
        f"- Status: `{summary['status']}`",
        f"- Blocked reason: `{summary.get('blockedReason')}`",
        f"- Rows considered: {summary['rowCount']}",
        f"- Exported masks: {summary['exportedMaskCount']}",
        "",
        "## Environment",
        "",
        f"- Python: {environment['pythonVersion']} (meets >=3.12: {environment['pythonMeetsRequirement']})",
        f"- Torch installed: {environment['torchInstalled']} ({environment['torchVersion']})",
        f"- Torch meets >=2.7: {environment['torchMeetsRequirement']}",
        f"- CUDA available: {environment['cudaAvailable']} ({environment['cudaVersion']})",
        f"- CUDA meets >=12.6: {environment['cudaMeetsRequirement']}",
        f"- MPS available: {environment['mpsAvailable']}",
        f"- sam3 package installed: {environment['sam3PackageInstalled']}",
        f"- HF token present: {environment['hfTokenPresent']}",
        "",
        "## Mask Schema",
        "",
        "When prerequisites are available, masks are written to:",
        "",
        "```text",
        "<mask-dir>/sam3/set_<SET>_<SIDE>_<prompt>.png",
        "```",
        "",
        "These masks can then be scored with:",
        "",
        "```bash",
        ".venv/bin/python tools/foundation_segmentation_bakeoff_v0.py --external-mask-dir <mask-dir>",
        "```",
        "",
        "## Interpretation",
        "",
        "- This machine cannot run official SAM3 image inference as-is because the required CUDA/Hugging Face/SAM3 prerequisites are not all present.",
        "- The exporter is still useful: it defines the exact bridge from a capable SAM3 environment into the repo's dependency-free bakeoff harness.",
        "- Once masks exist, the next useful metric is whether three visible-face prompts improve top-3 vertex recall over the current 3/16 source heuristic and 11/16 source-pool oracle ceiling.",
        "",
    ]
    if document.get("rows"):
        lines.extend([
            "## Exported Rows",
            "",
            "| Set | Side | Status | Exported masks |",
            "|---:|---|---|---:|",
        ])
        for row in document["rows"]:
            lines.append(
                f"| {row.get('setId')} | {row.get('side')} | `{row.get('status')}` | "
                f"{row.get('exportedMaskCount', 0)} |"
            )
    return "\n".join(lines)


def _feedback_rows(feedback: Dict[str, Any], *, limit_rows: Optional[int]) -> List[Dict[str, Any]]:
    rows = [row for row in feedback.get("rows", []) if row.get("status") == "labeled"]
    if limit_rows is not None:
        return rows[:limit_rows]
    return rows


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_mask_array(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, :, :]
    if masks.ndim != 3:
        raise ValueError(f"expected SAM3 masks shaped [N,H,W] or [N,1,H,W], got {masks.shape}")
    return masks > 0.5


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def _version_at_least(value: str, required: Tuple[int, int]) -> bool:
    pieces = []
    for token in value.replace("+", ".").split("."):
        if token.isdigit():
            pieces.append(int(token))
        else:
            break
    while len(pieces) < 2:
        pieces.append(0)
    return tuple(pieces[:2]) >= required


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--max-instances", type=int, default=DEFAULT_MAX_INSTANCES)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[prompt.key for prompt in PROMPT_SPECS],
        choices=[prompt.key for prompt in PROMPT_SPECS],
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 when prerequisites block inference after writing report/summary.",
    )
    args = parser.parse_args(argv)

    config = ExportConfig(
        feedback_path=args.feedback,
        mask_dir=args.mask_dir,
        score_threshold=args.score_threshold,
        max_instances=args.max_instances,
        limit_rows=args.limit_rows if args.limit_rows > 0 else None,
        prompts=tuple(args.prompts),
    )
    document = generate_sam3_export_artifacts(config, strict=args.strict)
    _write_json(args.summary, document)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary}")
    print(f"wrote {args.report}")
    if document["summary"]["status"] == "completed":
        print(f"wrote masks to {args.mask_dir / 'sam3'}")
        return 0
    print(f"blocked: {document['summary']['blockedReason']}", file=sys.stderr)
    return 2 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
