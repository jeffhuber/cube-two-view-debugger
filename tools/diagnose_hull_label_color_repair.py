#!/usr/bin/env python3
"""Diagnose deterministic hull-label panel color/count repair.

This tool is intentionally diagnostic-only. It starts from the accepted
hull-label Tier 1 geometry, assigns rectified panels to WCA faces with the
shared slot/yaw convention, then compares deterministic color repair stages:

* `canonical`: nearest canonical Rubik color per sticker.
* `canonical_center_forced`: same, but WCA centers are forced to U/R/F/D/L/B.
* `canonical_count_repaired`: greedy count repair to exactly 9 stickers per
  WCA face color, with centers fixed.
* `adaptive_center_forced`: adaptive Lab palette anchored by the six known
  WCA centers.
* `adaptive_count_repaired`: count repair using that adaptive palette.

Sets 69-73 exposed the main failure mode clearly: geometry often gives usable
panels, while raw LLM/color reads duplicate or miss colors. This diagnostic now
runs on the full manifest corpus so the repair layer can be compared against
the broader hull-label shadow/prefer baseline. It intentionally uses the same
1600px image-prep and threshold-selector geometry path as the Fixer
``/api/llm-rectified-input`` endpoint; older 1150px diagnostic geometry can
change the selected hull and misstate the real Fixer behavior.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import (  # noqa: E402
    CANONICAL_RGB,
    CLASSIFIER_CANONICAL,
    COLOR_ORDER,
    COLOR_TO_FACE,
    FACE_TO_COLOR,
    build_adaptive_palette,
    classify_rgb_with_mode,
)
from rubik_recognizer.validation import FACE_ORDER, validate_state  # noqa: E402
from tools.audit_recognition_pair import parse_ground_truth as parse_pair_ground_truth  # noqa: E402
from tools.diagnose_slot_yaw_assignment import (  # noqa: E402
    DEFAULT_MANIFEST,
    SLOT_TO_LEGACY_FACE,
    _hull_label_center_yaw_source,
    manifest_yaw_source,
    slot_face_assignments,
)
from tools.extract_color_samples import PairTask, load_corpus_tasks  # noqa: E402
from tools.global_cube_model import _slot_center_faces_from_rectified  # noqa: E402
from tools.hull_label_color_repair import repair_from_hull_label_fits  # noqa: E402
from tools.hull_label_assembly import convention_orientation_for_slot  # noqa: E402
from tools.rectify_faces import DEFAULT_FACE_SIZE, extract_stickers_from_rectified, rectify_face  # noqa: E402
from tools.rectify_via_hull_labels import select_hull_label_threshold_fit  # noqa: E402
from tools.sample_stickers_from_hull import apply_orientation  # noqa: E402


DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "hull_label_color_repair_diagnostic.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "HULL_LABEL_COLOR_REPAIR_DIAGNOSTIC.md"
FIXER_MAX_SIDE = 1600


@dataclass(frozen=True)
class StickerObservation:
    index: int
    side: str
    slot: str
    wca_face: str
    face_index: int
    rgb: Tuple[int, int, int]
    raw_color: str

    @property
    def is_center(self) -> bool:
        return self.face_index == 4


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _manifest_by_set(manifest_path: Path) -> Dict[str, Mapping[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        str(pair["setId"]): pair
        for pair in manifest.get("pairs", [])
    }


def _ground_truth_yaw_source(gt_path: Path) -> Optional[Dict[str, Any]]:
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    row = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(row, Mapping):
        return None
    yaw = row.get("captureYawQuarterTurns")
    if isinstance(yaw, int):
        return {
            "source": "ground_truth_captureYaw",
            "yawQuarterTurns": yaw % 4,
            "status": row.get("canonicalizationSource") or "ground_truth",
        }
    return None


def _metadata_yaw_source(task: PairTask, manifest_pair: Mapping[str, Any]) -> Dict[str, Any]:
    for source in (_ground_truth_yaw_source(task.ground_truth), manifest_yaw_source(manifest_pair)):
        if source is not None and isinstance(source.get("yawQuarterTurns"), int):
            return dict(source)
    return {
        "source": "white_up_default",
        "yawQuarterTurns": 0,
        "status": "assumed_from_capture_protocol",
    }


def _load_fixer_processing_image(image_path: Path) -> Image.Image:
    """Load image with the same max-side geometry scale used by Fixer."""
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    natural_max = max(image.size)
    if natural_max > FIXER_MAX_SIDE:
        scale = FIXER_MAX_SIDE / float(natural_max)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return image


def _fit_hull_side(image_path: Path, side: str, sess: Any) -> Dict[str, Any]:
    """Fit one side through the same rembg + threshold-selector path as Fixer."""
    from rembg import remove  # noqa: E402

    image = _load_fixer_processing_image(image_path)
    rgba = remove(image, session=sess).convert("RGBA")
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    selection = select_hull_label_threshold_fit(
        image,
        alpha,
        side,
        face_size_px=DEFAULT_FACE_SIZE,
    )
    fit = selection.fit
    trace = dict(selection.trace)
    if fit is not None:
        trace["slot_center_faces"] = _slot_center_faces_from_rectified(fit.rectified_faces)
    return {
        "image": image,
        "fit": fit,
        "model": fit,
        "trace": trace,
    }


def _hamming(actual: Optional[str], expected: str) -> Optional[int]:
    if actual is None or len(actual) != len(expected):
        return None
    return sum(1 for got, want in zip(actual, expected) if got != want)


def _assembled_state(face_chunks: Mapping[str, Sequence[str]]) -> Optional[str]:
    if set(face_chunks) != set(FACE_ORDER):
        return None
    return "".join("".join(face_chunks[face]) for face in FACE_ORDER)


def _face_index(wca_face: str, face_index: int) -> int:
    return FACE_ORDER.index(wca_face) * 9 + face_index


def _sample_observations(
    *,
    side_fits: Mapping[str, Mapping[str, Any]],
    yaw_quarter_turns: int,
) -> Tuple[List[StickerObservation], Dict[str, Any]]:
    observations: List[StickerObservation] = []
    per_panel: List[Dict[str, Any]] = []
    assignments = slot_face_assignments(yaw_quarter_turns)
    for side in ("A", "B"):
        model = side_fits[side].get("model")
        if model is None:
            raise ValueError(f"{side} hull-label model unavailable")
        for slot, wca_face in assignments[side].items():
            legacy_face = SLOT_TO_LEGACY_FACE[slot]
            quad = model.face_quads[legacy_face]
            orientation = convention_orientation_for_slot(
                side=side,
                slot=slot,
                yaw_quarter_turns=yaw_quarter_turns,
                wca_face=wca_face,
                quad=quad,
            )
            if orientation is None:
                raise ValueError(f"could not orient {side} {slot} as {wca_face}")
            rectified = rectify_face(side_fits[side]["image"], quad, output_size=DEFAULT_FACE_SIZE)
            samples = extract_stickers_from_rectified(rectified)
            raw_flat = [sample for row in samples for sample in row]
            oriented = apply_orientation(raw_flat, *orientation)
            panel_faces: List[str] = []
            panel_rgbs: List[Tuple[int, int, int]] = []
            for face_index, sample in enumerate(oriented):
                index = _face_index(wca_face, face_index)
                rgb = tuple(int(value) for value in sample.rgb)
                observations.append(
                    StickerObservation(
                        index=index,
                        side=side,
                        slot=slot,
                        wca_face=wca_face,
                        face_index=face_index,
                        rgb=rgb,
                        raw_color=sample.classified_color,
                    )
                )
                panel_rgbs.append(rgb)
                panel_faces.append(COLOR_TO_FACE[sample.classified_color])
            per_panel.append({
                "side": side,
                "slot": slot,
                "wcaFace": wca_face,
                "orientation": {"mirror": orientation[0], "rotQuarter": orientation[1]},
                "rawFaces": "".join(panel_faces),
                "centerRgb": panel_rgbs[4],
                "centerFace": panel_faces[4],
            })
    observations.sort(key=lambda item: item.index)
    if [item.index for item in observations] != list(range(54)):
        raise ValueError("observations did not cover all 54 WCA sticker indices")
    return observations, {
        "slotFaceAssignments": assignments,
        "panels": per_panel,
    }


def _costs_for_palette(
    observations: Sequence[StickerObservation],
    palette: Mapping[str, Tuple[int, int, int]],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for obs in observations:
        match = classify_rgb_with_mode(obs.rgb, CLASSIFIER_CANONICAL, prototypes=palette)
        by_color = dict(match.alternatives)
        rows.append({
            COLOR_TO_FACE[color]: float(by_color[color])
            for color in COLOR_ORDER
        })
    return rows


def _best_faces_from_costs(costs: Sequence[Mapping[str, float]]) -> List[str]:
    return [
        min(FACE_ORDER, key=lambda face: (row[face], face))
        for row in costs
    ]


def _force_centers(faces: Sequence[str]) -> List[str]:
    out = list(faces)
    for face in FACE_ORDER:
        out[_face_index(face, 4)] = face
    return out


def greedy_count_repair(
    faces: Sequence[str],
    costs: Sequence[Mapping[str, float]],
    *,
    fixed_indices: Iterable[int],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Repair a 54-face assignment to exact WCA counts with fixed centers.

    This is a deterministic local-cost repair, not a cube legality solver. It
    fixes the obvious class of panel-read mistakes where one color is overfull
    and another is underfull by moving the least expensive surplus stickers
    into deficit colors.
    """
    out = list(faces)
    fixed = set(fixed_indices)
    moves: List[Dict[str, Any]] = []
    while True:
        counts = Counter(out)
        deficits = {face: 9 - counts.get(face, 0) for face in FACE_ORDER if counts.get(face, 0) < 9}
        surplus = {face: counts.get(face, 0) - 9 for face in FACE_ORDER if counts.get(face, 0) > 9}
        if not deficits:
            break
        candidates = []
        for index, current in enumerate(out):
            if index in fixed or surplus.get(current, 0) <= 0:
                continue
            for target, needed in deficits.items():
                if needed <= 0 or target == current:
                    continue
                delta = costs[index][target] - costs[index][current]
                candidates.append((delta, index, current, target))
        if not candidates:
            break
        delta, index, source, target = min(candidates, key=lambda item: (item[0], item[1], item[3]))
        out[index] = target
        moves.append({
            "index": index,
            "from": source,
            "to": target,
            "delta": round(float(delta), 4),
        })
    return out, moves


def _state_eval(state: Optional[str], gt_state: str) -> Dict[str, Any]:
    validation = validate_state(state) if state else None
    hamming = _hamming(state, gt_state)
    return {
        "state": state,
        "hamming": hamming,
        "stickersCorrect": (54 - hamming) if hamming is not None else None,
        "exactMatch": state == gt_state if state else False,
        "validState": bool(validation and validation.valid),
        "validationErrors": list(validation.errors) if validation else ["not_assembled"],
        "counts": dict(sorted(Counter(state or "").items())),
    }


def _state_from_faces(faces: Sequence[str]) -> str:
    return "".join(faces)


def _evaluate_palette(
    *,
    observations: Sequence[StickerObservation],
    palette: Mapping[str, Tuple[int, int, int]],
    gt_state: str,
    prefix: str,
) -> Dict[str, Dict[str, Any]]:
    costs = _costs_for_palette(observations, palette)
    raw_faces = _best_faces_from_costs(costs)
    center_faces = _force_centers(raw_faces)
    fixed_centers = [_face_index(face, 4) for face in FACE_ORDER]
    repaired_faces, moves = greedy_count_repair(center_faces, costs, fixed_indices=fixed_centers)

    return {
        prefix: _state_eval(_state_from_faces(raw_faces), gt_state),
        f"{prefix}_center_forced": _state_eval(_state_from_faces(center_faces), gt_state),
        f"{prefix}_count_repaired": {
            **_state_eval(_state_from_faces(repaired_faces), gt_state),
            "repairMoves": moves,
            "repairMoveCount": len(moves),
        },
    }


def _adaptive_palette(observations: Sequence[StickerObservation]) -> Dict[str, Tuple[int, int, int]]:
    anchors: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for obs in observations:
        if obs.is_center:
            anchors[FACE_TO_COLOR[obs.wca_face]].append(obs.rgb)
    return build_adaptive_palette([obs.rgb for obs in observations], anchors=anchors)


def _evaluate_yaw_source(
    *,
    source: Mapping[str, Any],
    gt_state: str,
    side_fits: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    yaw = source.get("yawQuarterTurns")
    if not isinstance(yaw, int):
        return {
            "source": source.get("source"),
            "status": "unavailable",
            "yawQuarterTurns": None,
        }
    try:
        repair_payload = repair_from_hull_label_fits(
            side_fits=side_fits,
            yaw_quarter_turns=yaw,
            gt_state=gt_state,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "source": source.get("source"),
            "status": "sample_failed",
            "yawQuarterTurns": yaw,
            "error": f"{type(exc).__name__}: {exc}",
            "sideTraces": {side: side_fits[side].get("trace") for side in ("A", "B")},
        }

    return {
        "source": source.get("source"),
        "status": repair_payload.get("status"),
        "yawQuarterTurns": yaw,
        "sourceMeta": {key: value for key, value in source.items() if key not in {"source"}},
        "panelMeta": repair_payload.get("panelMeta"),
        "adaptivePalette": repair_payload.get("adaptivePalette"),
        "recommendedMethod": repair_payload.get("recommendedMethod"),
        "recommended": repair_payload.get("recommended"),
        "methods": repair_payload.get("methods", {}),
        "sideTraces": repair_payload.get("sideTraces", {}),
    }


def _evaluate_pair(task: PairTask, manifest_pair: Mapping[str, Any], sess: Any) -> Dict[str, Any]:
    _sha, raw_state, gt_state, canonicalized = parse_pair_ground_truth(str(task.ground_truth))
    side_fits = {
        "A": _fit_hull_side(task.image_a, "A", sess),
        "B": _fit_hull_side(task.image_b, "B", sess),
    }
    sources = [_metadata_yaw_source(task, manifest_pair)]
    center_source = _hull_label_center_yaw_source(side_fits)
    if isinstance(center_source.get("yawQuarterTurns"), int):
        sources.append(center_source)
    evaluations = [
        _evaluate_yaw_source(source=source, gt_state=gt_state, side_fits=side_fits)
        for source in sources
    ]
    return {
        "setId": task.set_id,
        "source": task.source,
        "images": {
            "A": _rel(task.image_a),
            "B": _rel(task.image_b),
            "groundTruth": _rel(task.ground_truth),
        },
        "groundTruthCanonicalized": canonicalized,
        "rawGroundTruthState": raw_state,
        "canonicalGroundTruthState": gt_state,
        "evaluations": {
            str(row["source"]): row
            for row in evaluations
        },
    }


def _method_summary(rows: Sequence[Mapping[str, Any]], yaw_source: str, method: str) -> Dict[str, Any]:
    evals = [
        row.get("evaluations", {}).get(yaw_source)
        for row in rows
        if row.get("evaluations", {}).get(yaw_source)
    ]
    assembled = [row for row in evals if row and row.get("status") == "assembled" and method in row.get("methods", {})]
    hamming = [row["methods"][method]["hamming"] for row in assembled if row["methods"][method]["hamming"] is not None]
    return {
        "assembled": len(assembled),
        "exact": sum(1 for row in assembled if row["methods"][method].get("exactMatch")),
        "legal": sum(1 for row in assembled if row["methods"][method].get("validState")),
        "meanStickersCorrect": round(statistics.mean(54 - value for value in hamming), 2) if hamming else None,
        "medianHamming": statistics.median(hamming) if hamming else None,
    }


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    yaw_sources = sorted({
        source
        for row in rows
        for source in row.get("evaluations", {})
    })
    methods = [
        "canonical",
        "canonical_center_forced",
        "canonical_count_repaired",
        "conservative_legal_repaired",
        "guarded_broad_legal_repaired",
        "adaptive",
        "adaptive_center_forced",
        "adaptive_count_repaired",
    ]
    return {
        "pairCount": len(rows),
        "yawSources": {
            source: {
                method: _method_summary(rows, source, method)
                for method in methods
            }
            for source in yaw_sources
        },
    }


def _best_method(row: Mapping[str, Any], source: str) -> Optional[Tuple[str, int]]:
    evaluation = row.get("evaluations", {}).get(source)
    if not evaluation or evaluation.get("status") != "assembled":
        return None
    scored = [
        (name, method.get("hamming"))
        for name, method in evaluation.get("methods", {}).items()
        if name != "broad_legal_repaired" and method.get("hamming") is not None
    ]
    if not scored:
        return None
    name, hamming = min(scored, key=lambda item: (item[1], item[0]))
    return name, int(hamming)


def _method_hamming_values(rows: Sequence[Mapping[str, Any]], source: str, method: str) -> List[int]:
    values: List[int] = []
    for row in rows:
        evaluation = row.get("evaluations", {}).get(source)
        if not evaluation or evaluation.get("status") != "assembled":
            continue
        hamming = evaluation.get("methods", {}).get(method, {}).get("hamming")
        if isinstance(hamming, int):
            values.append(hamming)
    return values


def _hamming_distribution(values: Sequence[int]) -> str:
    if not values:
        return "{}"
    counts = Counter(values)
    return "{" + ", ".join(f"{key}: {counts[key]}" for key in sorted(counts)) + "}"


def render_report(payload: Mapping[str, Any]) -> str:
    rows = payload["rows"]
    primary_source = "hull_label_center_colors"
    primary_methods = payload["summary"]["yawSources"].get(primary_source, {})
    lines = [
        "# Hull-Label Color Repair Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic asks whether deterministic color bookkeeping can clean",
        "up hull-label rectified panels before involving an LLM. It uses the",
        "same slot/yaw WCA assignment as the hull-label path, then compares",
        "plain Lab nearest-color classification, center forcing, exact",
        "9-per-color count repair, and guarded cubie-legality repair.",
        "",
        f"Git head: `{payload['source']['git_sha']}`",
        f"Generated: `{payload['source']['generated_at_utc']}`",
        "",
        "## Headline",
        "",
        "The production-like yaw source is `hull_label_center_colors`, because it",
        "does not use ground-truth yaw metadata. It is available for most pairs;",
        "rows without an accepted center-yaw inference are still shown with their",
        "metadata/default yaw fallback in the per-set table.",
        f"On the current {len(rows)}-pair GT corpus, count repair changes the story from",
        "mostly-correct panels to mostly-exact cubes:",
        "",
        "| Stage | Assembled | Exact | Legal | Mean stickers | Median hamming | Hamming distribution |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for method in (
        "canonical",
        "canonical_center_forced",
        "canonical_count_repaired",
        "conservative_legal_repaired",
        "guarded_broad_legal_repaired",
        "adaptive",
        "adaptive_count_repaired",
    ):
        summary = primary_methods.get(method, {})
        values = _method_hamming_values(rows, primary_source, method)
        lines.append(
            f"| `{method}` | {summary.get('assembled')} | {summary.get('exact')} | "
            f"{summary.get('legal')} | {summary.get('meanStickersCorrect')} | "
            f"{summary.get('medianHamming')} | `{_hamming_distribution(values)}` |"
        )

    canonical_summary = primary_methods.get("canonical", {})
    canonical_count_summary = primary_methods.get("canonical_count_repaired", {})
    guarded_legal_summary = primary_methods.get("guarded_broad_legal_repaired", {})
    adaptive_count_summary = primary_methods.get("adaptive_count_repaired", {})
    canonical_values = _method_hamming_values(rows, primary_source, "canonical")
    canonical_count_values = _method_hamming_values(rows, primary_source, "canonical_count_repaired")
    recommended_values = [
        evaluation.get("recommended", {}).get("hamming")
        for row in rows
        for evaluation in [row.get("evaluations", {}).get(primary_source, {})]
        if isinstance(evaluation.get("recommended", {}).get("hamming"), int)
    ]
    above_three = sum(value > 3 for value in canonical_count_values)
    above_three_label = "row" if above_three == 1 else "rows"
    lines.extend([
        "",
        "`canonical_count_repaired` is the stable deterministic baseline:",
        f"{canonical_count_summary.get('exact')}/{canonical_count_summary.get('assembled')} exact/legal, "
        f"{sum(value <= 3 for value in canonical_count_values)}/{len(canonical_count_values)} within 3 stickers, "
        f"and only {above_three} {above_three_label} above 3 stickers.",
        f"The payload's recommended-method selector is now "
        f"{sum(value == 0 for value in recommended_values)}/{len(recommended_values)} exact "
        f"with hamming distribution `{_hamming_distribution(recommended_values)}`.",
        "",
        "## Full Summary By Yaw Source",
        "",
        "| Yaw source | Method | Assembled | Exact | Legal | Mean stickers | Median hamming |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for source, methods in payload["summary"]["yawSources"].items():
        for method in (
            "canonical",
            "canonical_center_forced",
            "canonical_count_repaired",
            "conservative_legal_repaired",
            "guarded_broad_legal_repaired",
            "adaptive",
            "adaptive_center_forced",
            "adaptive_count_repaired",
        ):
            row = methods[method]
            lines.append(
                f"| `{source}` | `{method}` | {row['assembled']} | {row['exact']} | "
                f"{row['legal']} | {row['meanStickersCorrect']} | {row['medianHamming']} |"
            )

    lines.extend([
        "",
        "## Per-Set Snapshot",
        "",
        "| Set | Source | Recommended | Best safe method | Best hamming | Canonical | Canonical+count | Guarded legal | Adaptive+count | Status |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        source = primary_source if primary_source in row.get("evaluations", {}) else next(iter(row.get("evaluations", {})), "")
        evaluation = row.get("evaluations", {}).get(source, {})
        best = _best_method(row, source)
        canonical = evaluation.get("methods", {}).get("canonical", {}).get("hamming")
        canonical_count = evaluation.get("methods", {}).get("canonical_count_repaired", {}).get("hamming")
        guarded_legal = evaluation.get("methods", {}).get("guarded_broad_legal_repaired", {}).get("hamming")
        adaptive_count = evaluation.get("methods", {}).get("adaptive_count_repaired", {}).get("hamming")
        lines.append(
            f"| {row['setId']} | `{source}` | `{evaluation.get('recommendedMethod', 'n/a')}` | "
            f"`{best[0] if best else 'n/a'}` | {best[1] if best else 'n/a'} | "
            f"{canonical} | {canonical_count} | {guarded_legal} | {adaptive_count} | "
            f"`{evaluation.get('status', 'missing')}` |"
        )

    lines.extend([
        "",
        "## Current Run Notes",
        "",
        f"- The raw `canonical` classifier is already close: "
        f"{canonical_summary.get('exact')}/{canonical_summary.get('assembled')} exact, "
        f"{sum(value <= 3 for value in canonical_values)}/{len(canonical_values)}",
        "  within 3 stickers. The dominant issue is duplicated/missing color",
        "  counts, not WCA face assignment.",
        f"- Greedy count repair is a large deterministic jump: "
        f"{canonical_count_summary.get('exact')}/{canonical_count_summary.get('assembled')} exact/legal",
        "  with the production-like yaw source. This supersedes the older",
        "  20/46 exact headline for raw hull-label `prefer` panels.",
        "- Guarded cubie-legality repair is now part of the color-repair payload:",
        f"  it is {guarded_legal_summary.get('exact')}/{guarded_legal_summary.get('assembled')} exact here and exposes",
        "  conservative and guarded-broad legal candidates, while the",
        "  ungated broad legal candidate remains diagnostic-only.",
        "- Canonical Lab count repair beats the adaptive-palette count repair in",
        f"  this run ({canonical_count_summary.get('exact')}/{canonical_count_summary.get('assembled')} exact versus "
        f"{adaptive_count_summary.get('exact')}/{adaptive_count_summary.get('assembled')}). Adaptive palettes should stay",
        "  diagnostic or gated; do not blindly prefer them.",
        "- Sets 69-73 remain useful stress cases. With the Fixer-equivalent 1600px",
        "  geometry path, Sets 70-73 are exact after count repair; Set 69 still",
        "  needs the conservative legal layer to resolve a 3-sticker count-repair",
        "  ambiguity. This replaces the older lower-res diagnostic read where",
        "  Sets 70 and 72 looked like remaining tails.",
        "- The muddy side-face panels in these rows are photometric failures,",
        "  not rembg failures. rembg supplies the silhouette mask; the rectified",
        "  panels sample the original RGB image. Grazing side faces stretch shadow,",
        "  black bevels, reflections, and sticker texture into a square, so humans",
        "  can still read the colors while static Lab distance struggles.",
        "- Set 70 should be inspected with yaw-aware panel labels. Its current",
        "  yaw=2 Image B slots map to D/F/R; older no-yaw D/L/B contact sheets",
        "  are useful visually but misleading for face identity.",
    ])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `center_forced` is a cheap sanity step: the center sticker of each WCA",
        "  face is known once slot/yaw assignment succeeds, so a center-color",
        "  miss should not be allowed to poison the state.",
        "- `count_repaired` is deliberately not a legality solver. It only enforces",
        "  the physical requirement that each WCA face color appears exactly nine",
        "  times. This catches duplicated/missing color reads while preserving the",
        "  sampled geometry.",
        "- `conservative_legal_repaired` and `guarded_broad_legal_repaired` add the",
        "  next constraint layer: cubie legality. The guarded broad method uses",
        "  the same no-ground-truth cost/state-delta gate as the legal-repair diagnostic;",
        "  the raw `broad_legal_repaired` method is emitted only for traceability.",
        "- The adaptive palette uses the six known center samples as anchors. It is",
        "  still deterministic and local to the two input photos; no GT colors or",
        "  LLM output are used.",
        "- Rows that remain high-hamming after adaptive count repair are likely",
        "  geometry/panel-quality failures rather than cube-count failures. Those",
        "  should be handled by hull-label acceptance gates or a visual repair UI.",
    ])
    return "\n".join(lines) + "\n"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--only-sets", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    manifest_pairs = _manifest_by_set(args.manifest)
    tasks = load_corpus_tasks(args.manifest)
    only = {str(item) for item in args.only_sets} if args.only_sets else None
    if only is not None:
        tasks = [task for task in tasks if task.set_id in only]

    try:
        from rembg import new_session
    except Exception as exc:  # noqa: BLE001
        print(f"failed to import rembg: {exc}", file=sys.stderr)
        return 2

    sess = new_session("u2net")
    rows: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] set {task.set_id}", flush=True)
        try:
            rows.append(_evaluate_pair(task, manifest_pairs.get(task.set_id, {}), sess))
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "setId": task.set_id,
                "source": task.source,
                "images": {
                    "A": _rel(task.image_a),
                    "B": _rel(task.image_b),
                    "groundTruth": _rel(task.ground_truth),
                },
                "error": f"{type(exc).__name__}: {exc}",
                "evaluations": {},
            })

    payload = {
        "schema": "hull_label_color_repair_diagnostic_v1",
        "source": {
            "tool": "tools/diagnose_hull_label_color_repair.py",
            "manifest": _rel(args.manifest),
            "git_sha": _git_head_sha(),
            "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "summary": build_summary(rows),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
