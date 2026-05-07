from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


RGB = Tuple[int, int, int]

COLOR_TO_FACE = {
    "white": "U",
    "red": "R",
    "green": "F",
    "yellow": "D",
    "orange": "L",
    "blue": "B",
}

FACE_TO_COLOR = {face: color for color, face in COLOR_TO_FACE.items()}
COLOR_ORDER = ("white", "yellow", "red", "orange", "green", "blue")

CANONICAL_RGB: Dict[str, RGB] = {
    "white": (238, 238, 232),
    "yellow": (230, 210, 42),
    "red": (190, 48, 36),
    "orange": (218, 112, 42),
    "green": (58, 145, 82),
    "blue": (62, 86, 150),
}


@dataclass(frozen=True)
class ColorMatch:
    color: str
    face: str
    distance: float
    confidence: float
    alternatives: List[Tuple[str, float]]


def rgb_to_hsv(rgb: RGB) -> Tuple[float, float, float]:
    r, g, b = [v / 255.0 for v in rgb]
    return colorsys.rgb_to_hsv(r, g, b)


def rgb_to_lab(rgb: RGB) -> Tuple[float, float, float]:
    r, g, b = [_srgb_to_linear(v / 255.0) for v in rgb]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    x /= 0.95047
    z /= 1.08883
    fx, fy, fz = _lab_f(x), _lab_f(y), _lab_f(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def classify_rgb(rgb: RGB, prototypes: Mapping[str, RGB] | None = None) -> ColorMatch:
    palette = prototypes or CANONICAL_RGB
    hsv_hint = _rubik_hsv_hint(rgb)
    lab = rgb_to_lab(rgb)
    distances = []
    for color, proto in palette.items():
        proto_lab = rgb_to_lab(proto)
        distance = _rubik_lab_distance(lab, proto_lab, color) if prototypes is not None else _distance(lab, proto_lab)
        distances.append((color, distance))
    distances.sort(key=lambda item: item[1])
    use_hsv_hint = hsv_hint is not None and (prototypes is None or hsv_hint not in {"red", "orange"})
    if use_hsv_hint:
        distance_by_color = dict(distances)
        best = hsv_hint
        best_dist = distance_by_color[best]
        distances = [(best, best_dist)] + [item for item in distances if item[0] != best]
        confidence = max(0.45, _hsv_hint_confidence(rgb, best))
    else:
        best, best_dist = distances[0]
        second_dist = distances[1][1] if len(distances) > 1 else best_dist + 1.0
        confidence = max(0.0, min(1.0, (second_dist - best_dist) / max(second_dist, 1.0)))
    return ColorMatch(best, COLOR_TO_FACE[best], best_dist, confidence, distances)


def build_adaptive_palette(
    samples: Iterable[RGB],
    anchors: Mapping[str, Sequence[RGB]] | None = None,
    iterations: int = 7,
) -> Dict[str, RGB]:
    sample_list = [_clamp_rgb(sample) for sample in samples]
    sample_list = [sample for sample in sample_list if _usable_calibration_sample(sample)]
    if len(sample_list) < 12:
        return dict(CANONICAL_RGB)

    anchor_map = {
        color: [_clamp_rgb(sample) for sample in (anchors or {}).get(color, []) if _usable_calibration_sample(sample)]
        for color in COLOR_ORDER
    }
    high_confidence = _high_confidence_sample_groups(sample_list)
    prototypes = dict(CANONICAL_RGB)
    for color in COLOR_ORDER:
        seeds = anchor_map[color] or high_confidence.get(color, [])
        if seeds:
            prototypes[color] = median_rgb(seeds)

    for _ in range(max(1, iterations)):
        buckets: Dict[str, List[RGB]] = {color: list(anchor_map[color]) * 4 for color in COLOR_ORDER}
        for sample in sample_list:
            match = classify_rgb(sample, prototypes)
            buckets[match.color].append(sample)

        next_prototypes = dict(prototypes)
        for color in COLOR_ORDER:
            if buckets[color]:
                next_prototypes[color] = median_rgb(buckets[color])
        if next_prototypes == prototypes:
            break
        prototypes = next_prototypes
    return prototypes


def _high_confidence_sample_groups(samples: Sequence[RGB]) -> Dict[str, List[RGB]]:
    grouped: Dict[str, List[RGB]] = {color: [] for color in COLOR_ORDER}
    for sample in samples:
        match = classify_rgb(sample)
        second = match.alternatives[1][1] if len(match.alternatives) > 1 else match.distance + 1.0
        margin = second - match.distance
        if match.confidence >= 0.62 or margin >= 10.0:
            grouped[match.color].append(sample)
    return grouped


def _usable_calibration_sample(rgb: RGB) -> bool:
    _, saturation, value = rgb_to_hsv(rgb)
    return value >= 0.20 and not (saturation <= 0.08 and value <= 0.45)


def _clamp_rgb(rgb: RGB) -> RGB:
    return tuple(max(0, min(255, int(round(channel)))) for channel in rgb)  # type: ignore[return-value]


def _rubik_hsv_hint(rgb: RGB) -> str | None:
    hue, saturation, value = rgb_to_hsv(rgb)
    if value < 0.18:
        return None
    if saturation <= 0.22 and value >= 0.42:
        return "white"

    # In dim or weakly saturated samples the hue angle becomes noisy; this is
    # where wooden desks, shadows, and black plastic can look orange-ish in HSV.
    # Let Lab/adaptive palette distance win for those cases.
    if saturation < 0.36 or value < 0.38:
        return None
    if hue < 0.035 or hue >= 0.94:
        return "red"
    if hue < 0.105:
        return "orange"
    if hue < 0.195:
        return "yellow"
    if hue < 0.47:
        return "green"
    if hue < 0.75:
        return "blue"
    return "red"


def _hsv_hint_confidence(rgb: RGB, color: str) -> float:
    hue, saturation, value = rgb_to_hsv(rgb)
    if color == "white":
        saturation_margin = max(0.0, 0.30 - saturation) / 0.30
        value_margin = max(0.0, value - 0.42) / 0.58
        return min(1.0, 0.35 + saturation_margin * 0.35 + value_margin * 0.30)
    ranges = {
        "red": (0.0, 0.035),
        "orange": (0.035, 0.105),
        "yellow": (0.105, 0.195),
        "green": (0.195, 0.47),
        "blue": (0.47, 0.75),
    }
    low, high = ranges.get(color, (0.0, 1.0))
    if color == "red" and hue >= 0.94:
        hue_margin = min(hue - 0.94, 1.0 - hue) / 0.06
    else:
        hue_margin = min(max(0.0, hue - low), max(0.0, high - hue)) / max(high - low, 1e-6)
    saturation_margin = min(1.0, max(0.0, saturation - 0.18) / 0.45)
    return min(1.0, 0.45 + hue_margin * 0.30 + saturation_margin * 0.25)


def median_rgb(values: Iterable[RGB]) -> RGB:
    rows = list(values)
    if not rows:
        return 0, 0, 0
    channels = list(zip(*rows))
    return tuple(int(sorted(ch)[len(ch) // 2]) for ch in channels)  # type: ignore[return-value]


def _srgb_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _lab_f(value: float) -> float:
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    if value > epsilon:
        return value ** (1.0 / 3.0)
    return (kappa * value + 16.0) / 116.0


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((av - bv) ** 2 for av, bv in zip(a, b)))


def _rubik_lab_distance(a: Tuple[float, float, float], b: Tuple[float, float, float], color: str) -> float:
    lightness_weight = 0.78 if color == "white" else 0.55
    return math.sqrt(((a[0] - b[0]) * lightness_weight) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
