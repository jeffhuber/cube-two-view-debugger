"""Deterministic color/count repair for hull-label rectified panels.

The hull-label path gives us geometry-clean, WCA-assigned 3x3 panels. This
module owns the deterministic color bookkeeping that should happen before an
LLM is treated as the source of truth:

* classify each oriented sticker in CIELAB space against canonical Rubik colors;
* force the six known WCA centers;
* greedily repair duplicated/missing colors to exactly nine stickers per WCA
  face color;
* optionally run cubie-legality repair over the same Lab evidence with guarded
  cost/change thresholds;
* expose the trace as a JSON-safe payload that Fixer can use as a draft.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from rubik_recognizer.colors import (
    CANONICAL_RGB,
    CLASSIFIER_CANONICAL,
    COLOR_ORDER,
    COLOR_TO_FACE,
    FACE_TO_COLOR,
    build_adaptive_palette,
    classify_rgb_with_mode,
)
from rubik_recognizer.image_pipeline import Sticker
from rubik_recognizer.recognizer import _legal_repaired_state_from_faces
from rubik_recognizer.validation import FACE_ORDER, validate_state
from tools.corner_conventions import wca_face_by_slot
from tools.hull_label_assembly import convention_orientation_for_slot
from tools.rectify_faces import DEFAULT_FACE_SIZE, extract_stickers_from_rectified, rectify_face
from tools.sample_stickers_from_hull import apply_orientation


SLOT_TO_LEGACY_FACE = {
    "upper": "face_xz",
    "right": "face_yz",
    "front": "face_xy",
}
GUARDED_BROAD_MAX_REPAIR_COST = 20.0
GUARDED_BROAD_MAX_REPAIR_CHANGES = 4


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


def face_index(wca_face: str, face_index_in_face: int) -> int:
    return FACE_ORDER.index(wca_face) * 9 + face_index_in_face


# Backward-compatible private name used by existing diagnostics/tests.
_face_index = face_index


def _json_rgb(rgb: Sequence[int]) -> List[int]:
    return [int(value) for value in rgb]


def _json_palette(palette: Mapping[str, Sequence[int]]) -> Dict[str, List[int]]:
    return {color: _json_rgb(rgb) for color, rgb in sorted(palette.items())}


def _slot_assignments(yaw_quarter_turns: int) -> Dict[str, Dict[str, str]]:
    return {
        "A": wca_face_by_slot("A", yaw_quarter_turns),
        "B": wca_face_by_slot("B", yaw_quarter_turns),
    }


def _fit_for_side(entry: Any) -> Any:
    if isinstance(entry, Mapping):
        return entry.get("fit") or entry.get("model")
    return entry


def _image_for_side(entry: Any) -> Image.Image:
    if isinstance(entry, Mapping):
        image = entry.get("image")
    else:
        image = getattr(entry, "image", None)
    if image is None:
        raise ValueError("side fit entry missing image")
    return image


def _quad_for_slot(model: Any, slot: str) -> Sequence[Tuple[float, float]]:
    face_quads = getattr(model, "face_quads", None)
    if not isinstance(face_quads, Mapping):
        raise ValueError("hull-label model missing face_quads")
    if slot in face_quads:
        return face_quads[slot]
    legacy_face = SLOT_TO_LEGACY_FACE[slot]
    if legacy_face in face_quads:
        return face_quads[legacy_face]
    raise ValueError(f"hull-label model missing face quad for slot {slot!r}")


def sample_observations(
    *,
    side_fits: Mapping[str, Any],
    yaw_quarter_turns: int,
) -> Tuple[List[StickerObservation], Dict[str, Any]]:
    observations: List[StickerObservation] = []
    per_panel: List[Dict[str, Any]] = []
    assignments = _slot_assignments(yaw_quarter_turns)
    for side in ("A", "B"):
        if side not in side_fits:
            raise ValueError(f"missing side {side} hull-label fit")
        entry = side_fits[side]
        model = _fit_for_side(entry)
        if model is None:
            raise ValueError(f"{side} hull-label model unavailable")
        image = _image_for_side(entry)
        for slot, wca_face in assignments[side].items():
            quad = _quad_for_slot(model, slot)
            orientation = convention_orientation_for_slot(
                side=side,
                slot=slot,
                yaw_quarter_turns=yaw_quarter_turns,
                wca_face=wca_face,
                quad=quad,
            )
            if orientation is None:
                raise ValueError(f"could not orient {side} {slot} as {wca_face}")
            rectified = rectify_face(image, quad, output_size=DEFAULT_FACE_SIZE)
            samples = extract_stickers_from_rectified(rectified)
            raw_flat = [sample for row in samples for sample in row]
            oriented = apply_orientation(raw_flat, *orientation)
            panel_faces: List[str] = []
            panel_rgbs: List[Tuple[int, int, int]] = []
            for face_idx, sample in enumerate(oriented):
                index = face_index(wca_face, face_idx)
                rgb = tuple(int(value) for value in sample.rgb)
                observations.append(
                    StickerObservation(
                        index=index,
                        side=side,
                        slot=slot,
                        wca_face=wca_face,
                        face_index=face_idx,
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
                "orientation": {"mirror": bool(orientation[0]), "rotQuarter": int(orientation[1])},
                "rawFaces": "".join(panel_faces),
                "centerRgb": _json_rgb(panel_rgbs[4]),
                "centerFace": panel_faces[4],
            })
    observations.sort(key=lambda item: item.index)
    if [item.index for item in observations] != list(range(54)):
        raise ValueError("observations did not cover all 54 WCA sticker indices")
    return observations, {
        "slotFaceAssignments": assignments,
        "panels": per_panel,
    }


# Backward-compatible private name used by the existing diagnostic.
_sample_observations = sample_observations


def costs_for_palette(
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
        out[face_index(face, 4)] = face
    return out


def greedy_count_repair(
    faces: Sequence[str],
    costs: Sequence[Mapping[str, float]],
    *,
    fixed_indices: Iterable[int],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Repair a 54-face assignment to exact WCA counts with fixed centers."""
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
            "index": int(index),
            "from": source,
            "to": target,
            "delta": round(float(delta), 4),
        })
    return out, moves


def state_from_faces(faces: Sequence[str]) -> str:
    return "".join(faces)


_state_from_faces = state_from_faces


def state_payload(state: Optional[str], *, repair_moves: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    validation = validate_state(state) if state else None
    counts = Counter(state or "")
    payload: Dict[str, Any] = {
        "state": state,
        "validState": bool(validation and validation.valid),
        "validationErrors": list(validation.errors) if validation else ["not_assembled"],
        "counts": {face: int(counts.get(face, 0)) for face in FACE_ORDER},
        "countBalanced": all(counts.get(face, 0) == 9 for face in FACE_ORDER),
    }
    if repair_moves is not None:
        payload["repairMoves"] = list(repair_moves)
        payload["repairMoveCount"] = len(repair_moves)
    return payload


def _state_eval(state: Optional[str], gt_state: str) -> Dict[str, Any]:
    hamming = None
    if state is not None and len(state) == len(gt_state):
        hamming = sum(1 for got, want in zip(state, gt_state) if got != want)
    return {
        **state_payload(state),
        "hamming": hamming,
        "stickersCorrect": (54 - hamming) if hamming is not None else None,
        "exactMatch": state == gt_state if state else False,
    }


def evaluate_palette(
    *,
    observations: Sequence[StickerObservation],
    palette: Mapping[str, Tuple[int, int, int]],
    prefix: str,
    gt_state: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    costs = costs_for_palette(observations, palette)
    raw_faces = _best_faces_from_costs(costs)
    center_faces = _force_centers(raw_faces)
    fixed_centers = [face_index(face, 4) for face in FACE_ORDER]
    repaired_faces, moves = greedy_count_repair(center_faces, costs, fixed_indices=fixed_centers)

    def payload(faces: Sequence[str], *, repair_moves: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
        state = state_from_faces(faces)
        if gt_state is not None:
            out = _state_eval(state, gt_state)
            if repair_moves is not None:
                out["repairMoves"] = list(repair_moves)
                out["repairMoveCount"] = len(repair_moves)
            return out
        return state_payload(state, repair_moves=repair_moves)

    return {
        prefix: payload(raw_faces),
        f"{prefix}_center_forced": payload(center_faces),
        f"{prefix}_count_repaired": payload(repaired_faces, repair_moves=moves),
    }


# Backward-compatible private name used by the existing diagnostic.
_evaluate_palette = evaluate_palette


def _payload_for_state(state: Optional[str], gt_state: Optional[str]) -> Dict[str, Any]:
    if gt_state is not None:
        return _state_eval(state, gt_state)
    return state_payload(state)


def _observation_by_index(observations: Sequence[StickerObservation]) -> Dict[int, StickerObservation]:
    return {int(obs.index): obs for obs in observations}


def _faces_for_legal_repair(
    observations: Sequence[StickerObservation],
    *,
    source: str,
) -> Dict[str, List[List[Any]]]:
    by_index = _observation_by_index(observations)
    faces: Dict[str, List[List[Any]]] = {}
    for face in FACE_ORDER:
        matrix: List[List[Any]] = []
        for row in range(3):
            matrix_row: List[Any] = []
            for col in range(3):
                index_in_face = row * 3 + col
                index = face_index(face, index_in_face)
                if index_in_face == 4:
                    matrix_row.append(face)
                    continue
                obs = by_index[index]
                match = classify_rgb_with_mode(obs.rgb, CLASSIFIER_CANONICAL, prototypes=CANONICAL_RGB)
                matrix_row.append(
                    Sticker(
                        id=index,
                        center=(0.0, 0.0),
                        bbox=(0.0, 0.0, 1.0, 1.0),
                        rgb=obs.rgb,
                        match=match,
                        area=1,
                        source=source,
                    )
                )
            matrix.append(matrix_row)
        faces[face] = matrix
    return faces


def _legal_repair_payload(
    observations: Sequence[StickerObservation],
    *,
    gt_state: Optional[str],
    source: str,
    baseline_state: Optional[str],
) -> Dict[str, Any]:
    if baseline_state and validate_state(baseline_state).valid:
        return {
            "status": "already_valid_count_repair",
            **_payload_for_state(baseline_state, gt_state),
            "repairCost": 0.0,
            "repairChanges": 0,
            "sourceMode": source,
        }
    faces = _faces_for_legal_repair(observations, source=source)
    result = _legal_repaired_state_from_faces(faces)
    if result is None:
        return {
            "status": "no_legal_repair",
            **_payload_for_state(None, gt_state),
            "repairCost": None,
            "repairChanges": None,
            "sourceMode": source,
        }
    state, cost, changes = result
    return {
        "status": "legal_repair_found",
        **_payload_for_state(state, gt_state),
        "repairCost": round(float(cost), 4),
        "repairChanges": int(changes),
        "sourceMode": source,
    }


def _guarded_broad_payload(broad_payload: Mapping[str, Any], *, gt_state: Optional[str]) -> Dict[str, Any]:
    cost = broad_payload.get("repairCost")
    changes = broad_payload.get("repairChanges")
    accepted = (
        broad_payload.get("validState") is True
        and isinstance(cost, (int, float))
        and isinstance(changes, int)
        and float(cost) <= GUARDED_BROAD_MAX_REPAIR_COST
        and int(changes) <= GUARDED_BROAD_MAX_REPAIR_CHANGES
    )
    gate = {
        "maxRepairCost": GUARDED_BROAD_MAX_REPAIR_COST,
        "maxRepairChanges": GUARDED_BROAD_MAX_REPAIR_CHANGES,
        "accepted": accepted,
    }
    if accepted:
        out = dict(broad_payload)
        out["status"] = "accepted_guarded_broad_legal_repair"
        out["gate"] = gate
        return out
    return {
        "status": "rejected_guarded_broad_legal_repair",
        **_payload_for_state(None, gt_state),
        "repairCost": None,
        "repairChanges": None,
        "rejectedRepairCost": cost,
        "rejectedRepairChanges": changes,
        "sourceMode": broad_payload.get("sourceMode"),
        "gate": gate,
    }


def evaluate_legal_repair_methods(
    *,
    observations: Sequence[StickerObservation],
    baseline_state: Optional[str],
    gt_state: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    broad = _legal_repair_payload(
        observations,
        gt_state=gt_state,
        source="grid_sample",
        baseline_state=baseline_state,
    )
    guarded = _guarded_broad_payload(broad, gt_state=gt_state)
    broad = dict(broad)
    broad["diagnosticOnly"] = True
    return {
        "conservative_legal_repaired": _legal_repair_payload(
            observations,
            gt_state=gt_state,
            source="hull_label_sample",
            baseline_state=baseline_state,
        ),
        "guarded_broad_legal_repaired": guarded,
        "broad_legal_repaired": broad,
    }


def adaptive_palette(observations: Sequence[StickerObservation]) -> Dict[str, Tuple[int, int, int]]:
    anchors: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for obs in observations:
        if obs.is_center:
            anchors[FACE_TO_COLOR[obs.wca_face]].append(obs.rgb)
    return build_adaptive_palette([obs.rgb for obs in observations], anchors=anchors)


_adaptive_palette = adaptive_palette


def _sticker_observation_payload(obs: StickerObservation) -> Dict[str, Any]:
    return {
        "index": int(obs.index),
        "side": obs.side,
        "slot": obs.slot,
        "wcaFace": obs.wca_face,
        "faceIndex": int(obs.face_index),
        "rgb": _json_rgb(obs.rgb),
        "rawColor": obs.raw_color,
        "rawFace": COLOR_TO_FACE[obs.raw_color],
    }


def _confidence_for_method(method: Mapping[str, Any]) -> str:
    move_count = int(method.get("repairMoveCount") or 0)
    if method.get("validState") and move_count <= 6:
        return "high"
    if method.get("validState"):
        return "medium"
    if method.get("countBalanced") and move_count <= 10:
        return "medium"
    return "low"


def choose_recommended_method(methods: Mapping[str, Mapping[str, Any]]) -> str:
    preferred = [
        "canonical_count_repaired",
        "conservative_legal_repaired",
        "guarded_broad_legal_repaired",
        "adaptive_count_repaired",
        "canonical_center_forced",
        "adaptive_center_forced",
        "canonical",
        "adaptive",
    ]
    legal = [name for name in preferred if methods.get(name, {}).get("validState")]
    if legal:
        return legal[0]
    balanced = [name for name in preferred if methods.get(name, {}).get("countBalanced")]
    if balanced:
        return balanced[0]
    return next((name for name in preferred if name in methods), "")


def assemble_color_repair_payload(
    *,
    observations: Sequence[StickerObservation],
    panel_meta: Mapping[str, Any],
    yaw_quarter_turns: int,
    side_traces: Optional[Mapping[str, Any]] = None,
    gt_state: Optional[str] = None,
) -> Dict[str, Any]:
    canonical = evaluate_palette(
        observations=observations,
        palette=CANONICAL_RGB,
        prefix="canonical",
        gt_state=gt_state,
    )
    adaptive = adaptive_palette(observations)
    adaptive_methods = evaluate_palette(
        observations=observations,
        palette=adaptive,
        prefix="adaptive",
        gt_state=gt_state,
    )
    methods = {**canonical, **adaptive_methods}
    methods.update(
        evaluate_legal_repair_methods(
            observations=observations,
            baseline_state=canonical["canonical_count_repaired"].get("state"),
            gt_state=gt_state,
        )
    )
    recommended_name = choose_recommended_method(methods)
    recommended = dict(methods[recommended_name]) if recommended_name else {}
    if recommended:
        recommended["method"] = recommended_name
        recommended["confidence"] = _confidence_for_method(recommended)
    return {
        "schema": "hull_label_color_repair_v1",
        "status": "assembled",
        "yawQuarterTurns": int(yaw_quarter_turns) % 4,
        "recommendedMethod": recommended_name,
        "recommended": recommended,
        "methods": methods,
        "panelMeta": panel_meta,
        "adaptivePalette": _json_palette(adaptive),
        "stickerObservations": [_sticker_observation_payload(obs) for obs in observations],
        "sideTraces": dict(side_traces or {}),
    }


def repair_from_hull_label_fits(
    *,
    side_fits: Mapping[str, Any],
    yaw_quarter_turns: int,
    gt_state: Optional[str] = None,
) -> Dict[str, Any]:
    observations, panel_meta = sample_observations(
        side_fits=side_fits,
        yaw_quarter_turns=yaw_quarter_turns,
    )
    side_traces: Dict[str, Any] = {}
    for side, entry in side_fits.items():
        if isinstance(entry, Mapping) and "trace" in entry:
            side_traces[side] = entry["trace"]
    return assemble_color_repair_payload(
        observations=observations,
        panel_meta=panel_meta,
        yaw_quarter_turns=yaw_quarter_turns,
        side_traces=side_traces,
        gt_state=gt_state,
    )
