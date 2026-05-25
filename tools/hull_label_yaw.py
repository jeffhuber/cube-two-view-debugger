"""Production-shaped yaw inference for the hull-label path."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from tools.corner_conventions import wca_face_by_slot


MIN_ACCEPTED_CENTER_MATCHES = 5
MIN_ACCEPTED_CENTER_MARGIN = 2

ObservedCenter = Tuple[str, str, str]


def score_yaw_candidates(
    observed_centers: Sequence[ObservedCenter],
    *,
    min_matches: int = MIN_ACCEPTED_CENTER_MATCHES,
    min_margin: int = MIN_ACCEPTED_CENTER_MARGIN,
) -> Dict[str, Any]:
    """Score yaw 0..3 by center-face agreement with the slot convention."""
    candidates = []
    for yaw in range(4):
        assignments = {
            "A": wca_face_by_slot("A", yaw),
            "B": wca_face_by_slot("B", yaw),
        }
        matches = []
        mismatches = []
        for side, slot, observed in observed_centers:
            expected = assignments[side][slot]
            item = {
                "side": side,
                "slot": slot,
                "observed": observed,
                "expected": expected,
            }
            if observed == expected:
                matches.append(item)
            else:
                mismatches.append(item)
        candidates.append({
            "yawQuarterTurns": yaw,
            "score": len(matches),
            "matches": matches,
            "mismatches": mismatches,
        })
    candidates.sort(key=lambda item: (-int(item["score"]), int(item["yawQuarterTurns"])))
    best = candidates[0]
    second = candidates[1]
    margin = int(best["score"]) - int(second["score"])
    accepted = int(best["score"]) >= min_matches and margin >= min_margin
    return {
        "accepted": accepted,
        "yawQuarterTurns": int(best["yawQuarterTurns"]) if accepted else None,
        "bestYawQuarterTurns": int(best["yawQuarterTurns"]),
        "bestScore": int(best["score"]),
        "secondScore": int(second["score"]),
        "margin": margin,
        "minMatches": min_matches,
        "minMargin": min_margin,
        "candidates": candidates,
    }


def observed_centers_from_side_traces(
    image_a_trace: Mapping[str, Any],
    image_b_trace: Mapping[str, Any],
) -> Tuple[ObservedCenter, ...]:
    """Return observed center faces from Tier 1 per-side traces."""
    observed = []
    for side, trace in (("A", image_a_trace), ("B", image_b_trace)):
        slot_centers = trace.get("slot_center_faces")
        if not isinstance(slot_centers, Mapping):
            continue
        for slot in ("upper", "right", "front"):
            item = slot_centers.get(slot)
            if not isinstance(item, Mapping):
                continue
            center_face = item.get("face")
            if isinstance(center_face, str) and len(center_face) == 1:
                observed.append((side, slot, center_face))
    return tuple(observed)


def infer_yaw_from_side_traces(
    image_a_trace: Mapping[str, Any],
    image_b_trace: Mapping[str, Any],
    *,
    min_matches: int = MIN_ACCEPTED_CENTER_MATCHES,
    min_margin: int = MIN_ACCEPTED_CENTER_MARGIN,
) -> Dict[str, Any]:
    """Infer capture yaw from the six hull-label slot center faces."""
    observed = observed_centers_from_side_traces(image_a_trace, image_b_trace)
    if len(observed) != 6:
        return {
            "source": "hull_label_center_colors",
            "status": "unavailable",
            "accepted": False,
            "yawQuarterTurns": None,
            "reason": "need six slot center observations",
            "observedCenters": [
                {"side": side, "slot": slot, "centerFace": center}
                for side, slot, center in observed
            ],
        }
    result = score_yaw_candidates(
        observed,
        min_matches=min_matches,
        min_margin=min_margin,
    )
    result.update({
        "source": "hull_label_center_colors",
        "status": "accepted" if result["accepted"] else "ambiguous",
        "observedCenters": [
            {"side": side, "slot": slot, "centerFace": center}
            for side, slot, center in observed
        ],
    })
    return result
