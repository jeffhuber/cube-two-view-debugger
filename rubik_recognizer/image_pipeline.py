from __future__ import annotations

import base64
import io
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .colors import ColorMatch, RGB, classify_rgb, rgb_to_hsv


Point = Tuple[float, float]

FACE_GRID_ANCHORS = ("U", "D")
FACE_GRID_SIDE_PAIRS = (("F", "R"), ("R", "B"), ("B", "L"), ("L", "F"))
MAX_RESCUE_GRID_COMPONENT_OVERLAP = 3
MAX_RESCUE_FACE_CANDIDATES = 48
MAX_RESCUE_TRIPLES = 12
MIN_COLORED_STICKERS_FOR_WHITE_NOISE_FILTER = 8
MIN_WHITE_STICKERS_FOR_WHITE_NOISE_FILTER = 18
TINY_WHITE_COMPONENT_AREA_FRACTION = 0.05

# ROI detection tuning. The default saturation threshold (0.23) handles
# most natural photo backgrounds (marble, light cloth, sky). The retry
# threshold (0.40) drops weakly-saturated chromatic/textured backgrounds
# where large saturated regions can merge with the cube into a single
# frame-spanning component. The retry-trigger threshold (0.95 of both dimensions)
# guards against false-triggering on legitimately tight-framed cubes.
# See `_find_cube_roi` for the failure mode this addresses.
_ROI_SATURATION_MIN_DEFAULT = 0.23
_ROI_SATURATION_MIN_RETRY = 0.40
_ROI_FRAME_COVERAGE_THRESHOLD = 0.95

REMBG_GRID_GUARD_ENV = "CUBE_RECOGNIZER_REMBG_GRID_GUARD"
REMBG_GRID_GUARD_SOURCE = "rembg_u2net_hull"
REMBG_GRID_INSIDE_MIN = 7
REMBG_GRID_OUTSIDE_BASE_PENALTY = 160.0
REMBG_GRID_OUTSIDE_EXTRA_PENALTY = 40.0
REMBG_HULL_PADDING_FRACTION = 0.025
REMBG_HULL_MIN_AREA_FRACTION = 0.015
REMBG_HULL_MAX_AREA_FRACTION = 0.80
REMBG_HULL_POINT_TOLERANCE = 2.0

_REMBG_SESSIONS: Dict[str, object] = {}


@dataclass
class Sticker:
    id: int
    center: Point
    bbox: Tuple[float, float, float, float]
    rgb: RGB
    match: ColorMatch
    area: int
    source: str = "component"
    shape_angle: Optional[float] = None

    @property
    def face(self) -> str:
        return self.match.face


@dataclass
class FaceGrid:
    id: int
    stickers: List[List[Sticker]]
    points: List[List[Point]]
    matched_count: int
    fit_error: float
    cube_hull_inside_count: Optional[int] = None
    cube_hull_outside_count: Optional[int] = None
    cube_hull_source: Optional[str] = None

    @property
    def center_sticker(self) -> Sticker:
        return self.stickers[1][1]

    @property
    def center_face(self) -> str:
        return self.center_sticker.face


@dataclass
class ImageAnalysis:
    width: int
    height: int
    roi: Tuple[int, int, int, int]
    stickers: List[Sticker]
    grids: List[FaceGrid]
    overlay_data_url: str
    warnings: List[str]
    hull_label_tier1: Optional[Dict[str, Any]] = None

    def summary(self) -> Dict:
        grids = []
        for grid in self.grids:
            row = {
                "id": grid.id,
                "centerFace": grid.center_face,
                "centerColor": grid.center_sticker.match.color,
                "matchedCount": grid.matched_count,
                "fitError": round(grid.fit_error, 2),
            }
            if grid.cube_hull_inside_count is not None:
                row.update(
                    {
                        "cubeHullInsideCount": grid.cube_hull_inside_count,
                        "cubeHullOutsideCount": grid.cube_hull_outside_count,
                        "cubeHullSource": grid.cube_hull_source,
                    }
                )
            grids.append(row)
        summary = {
            "width": self.width,
            "height": self.height,
            "roi": self.roi,
            "stickers": len(self.stickers),
            "grids": grids,
            "warnings": self.warnings,
        }
        if self.hull_label_tier1 is not None:
            summary["hullLabelTier1"] = self.hull_label_tier1
        return summary


def analyze_image(
    image_bytes: bytes,
    *,
    hull_label_side: Optional[str] = None,
    hull_label_mode: str = "off",
) -> ImageAnalysis:
    image = _load_image(image_bytes)
    process_image, scale = _resize_for_processing(image, max_side=1150)
    arr = np.asarray(process_image).astype(np.uint8)
    roi = _find_cube_roi(arr)
    stickers = _find_stickers(arr, roi)
    cube_hull, cube_hull_warning = _rembg_cube_hull_if_enabled(process_image)
    grids = _fit_face_grids(stickers, arr, scale, cube_hull=cube_hull)
    hull_label_tier1 = _hull_label_tier1_if_enabled(
        process_image,
        arr,
        side=hull_label_side,
        mode=hull_label_mode,
    )
    if hull_label_tier1 and hull_label_tier1.get("selected"):
        grids = _hull_label_tier1_grids(hull_label_tier1, arr)
    warnings = []
    if cube_hull_warning:
        warnings.append(cube_hull_warning)
    if hull_label_tier1 and hull_label_tier1.get("status") == "error":
        warnings.append("Hull-label Tier 1 failed before producing candidate grids.")
    if len(stickers) < 18:
        warnings.append("Few sticker candidates detected.")
    if len(grids) < 3:
        warnings.append("Could not fit three visible 3x3 face grids.")
    overlay = _make_overlay(process_image, roi, stickers, grids)
    return ImageAnalysis(
        width=image.width,
        height=image.height,
        roi=roi,
        stickers=stickers,
        grids=grids,
        overlay_data_url=overlay,
        warnings=warnings,
        hull_label_tier1=hull_label_tier1,
    )


def _load_image(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _resize_for_processing(image: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    side = max(image.size)
    if side <= max_side:
        return image.copy(), 1.0
    scale = max_side / float(side)
    size = (int(image.width * scale), int(image.height * scale))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def _rembg_cube_hull_if_enabled(image: Image.Image) -> Tuple[Optional[List[Point]], Optional[str]]:
    if not _rembg_grid_guard_enabled():
        return None, None
    try:
        from rembg import new_session, remove
    except ImportError:
        return None, "rembg grid guard requested but rembg is not installed."

    try:
        session = _REMBG_SESSIONS.get("u2net")
        if session is None:
            session = new_session("u2net")
            _REMBG_SESSIONS["u2net"] = session
        rgba = remove(image, session=session)
    except Exception as exc:
        return None, f"rembg grid guard failed: {exc.__class__.__name__}."

    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
    mask = alpha > 128
    if not mask.any():
        return None, "rembg grid guard skipped: empty cube mask."

    area_fraction = float(mask.sum()) / float(mask.size)
    if area_fraction < REMBG_HULL_MIN_AREA_FRACTION or area_fraction > REMBG_HULL_MAX_AREA_FRACTION:
        return None, "rembg grid guard skipped: implausible cube mask area."

    hull = _hull_from_mask(mask)
    if len(hull) < 3:
        return None, "rembg grid guard skipped: cube hull too small."
    return _expand_polygon(hull, REMBG_HULL_PADDING_FRACTION), None


def _rembg_grid_guard_enabled() -> bool:
    value = os.environ.get(REMBG_GRID_GUARD_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on", "u2net", REMBG_GRID_GUARD_SOURCE}


def _hull_label_tier1_if_enabled(
    image: Image.Image,
    arr: np.ndarray,
    *,
    side: Optional[str],
    mode: str,
) -> Optional[Dict[str, Any]]:
    """Run the feature-flagged hull-label candidate path for one photo.

    Shadow mode records the trace while preserving legacy grids. Prefer mode
    marks accepted candidate grids as selected; the recognizer still applies a
    pair-level legal-state guard before returning a prefer result.
    """
    resolved_mode = (mode or "off").strip().lower()
    if resolved_mode in {"", "0", "false", "no", "off", "legacy"}:
        return None
    if resolved_mode in {"trace", "diagnostic", "dry-run", "dry_run"}:
        resolved_mode = "shadow"
    if resolved_mode not in {"shadow", "prefer"}:
        resolved_mode = "off"
    if resolved_mode == "off":
        return None

    if side not in {"A", "B"}:
        return {
            "mode": resolved_mode,
            "side": side,
            "status": "skipped_missing_or_unsupported_side",
            "accepted": False,
            "selected": False,
            "hard_failures": ["hull_label_side must be 'A' or 'B'"],
        }

    try:
        from rembg import remove
        from tools.global_cube_model import _fit_hull_label_tier1_model
    except ImportError as exc:
        return {
            "mode": resolved_mode,
            "side": side,
            "status": "error",
            "accepted": False,
            "selected": False,
            "error": f"ImportError: {exc}",
        }

    try:
        session = _REMBG_SESSIONS.get("u2net")
        if session is None:
            from rembg import new_session

            session = new_session("u2net")
            _REMBG_SESSIONS["u2net"] = session
        rgba = remove(image, session=session)
        if rgba.mode != "RGBA":
            rgba = rgba.convert("RGBA")
        alpha = np.asarray(rgba.split()[-1], dtype=np.uint8)
        mask = alpha > 128
        if not mask.any():
            return {
                "mode": resolved_mode,
                "side": side,
                "status": "rejected",
                "accepted": False,
                "selected": False,
                "hard_failures": ["empty cube mask"],
            }
        model, trace = _fit_hull_label_tier1_model(
            arr,
            mask,
            side=side,
            mode=resolved_mode,
        )
        trace = dict(trace or {})
        trace.setdefault("mode", resolved_mode)
        trace.setdefault("side", side)
        trace["selected"] = bool(
            resolved_mode == "prefer"
            and model is not None
            and trace.get("status") == "accepted"
        )
        if trace["selected"] and model is not None:
            trace["_model"] = model
        return trace
    except Exception as exc:  # noqa: BLE001
        return {
            "mode": resolved_mode,
            "side": side,
            "status": "error",
            "accepted": False,
            "selected": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _hull_label_tier1_grids(trace: Mapping[str, Any], arr: np.ndarray) -> List[FaceGrid]:
    model = trace.get("_model")
    if model is None:
        return []
    grids: List[FaceGrid] = []
    face_keys = ("face_xz", "face_yz", "face_xy")
    for offset, face_key in enumerate(face_keys):
        cells = getattr(model, "sticker_cells", {}).get(face_key)
        if not cells or len(cells) != 9:
            continue
        grids.append(_face_grid_from_hull_label_cells(10_000 + offset, cells, arr))
    trace.pop("_model", None)
    return grids


def _face_grid_from_hull_label_cells(
    grid_id: int,
    cells: Sequence[Sequence[Point]],
    arr: np.ndarray,
) -> FaceGrid:
    points: List[List[Point]] = []
    stickers: List[List[Sticker]] = []
    spacing = _median_cell_spacing(cells)
    sample_half = max(3, min(18, int(round(spacing * 0.14))))
    for r in range(3):
        point_row: List[Point] = []
        sticker_row: List[Sticker] = []
        for c in range(3):
            cell = cells[r * 3 + c]
            center = _cell_center(cell)
            bbox = _cell_bbox(cell)
            rgb = _sample_rgb_square(arr, center[0], center[1], sample_half)
            sticker = Sticker(
                id=grid_id * 100 + r * 3 + c,
                center=center,
                bbox=bbox,
                rgb=rgb,
                match=classify_rgb(rgb),
                area=(2 * sample_half + 1) ** 2,
                source="grid_sample",
            )
            sticker.grid_spacing = spacing  # type: ignore[attr-defined]
            point_row.append(center)
            sticker_row.append(sticker)
        points.append(point_row)
        stickers.append(sticker_row)
    return FaceGrid(
        id=grid_id,
        stickers=stickers,
        points=points,
        matched_count=9,
        fit_error=0.0,
        cube_hull_inside_count=9,
        cube_hull_outside_count=0,
        cube_hull_source="hull_label_tier1",
    )


def _cell_center(cell: Sequence[Point]) -> Point:
    return (
        sum(float(point[0]) for point in cell) / len(cell),
        sum(float(point[1]) for point in cell) / len(cell),
    )


def _cell_bbox(cell: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [float(point[0]) for point in cell]
    ys = [float(point[1]) for point in cell]
    return min(xs), min(ys), max(xs), max(ys)


def _median_cell_spacing(cells: Sequence[Sequence[Point]]) -> float:
    centers = [_cell_center(cell) for cell in cells]
    distances: List[float] = []
    for r in range(3):
        for c in range(2):
            a, b = centers[r * 3 + c], centers[r * 3 + c + 1]
            distances.append(math.hypot(a[0] - b[0], a[1] - b[1]))
    for r in range(2):
        for c in range(3):
            a, b = centers[r * 3 + c], centers[(r + 1) * 3 + c]
            distances.append(math.hypot(a[0] - b[0], a[1] - b[1]))
    if not distances:
        return 32.0
    distances.sort()
    return distances[len(distances) // 2]


def _sample_rgb_square(arr: np.ndarray, x: float, y: float, half: int) -> RGB:
    h, w = arr.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)
    if x0 >= x1 or y0 >= y1:
        return (0, 0, 0)
    patch = arr[y0:y1, x0:x1].reshape(-1, 3)
    return tuple(int(np.median(patch[:, channel])) for channel in range(3))  # type: ignore[return-value]


def _find_cube_roi(arr: np.ndarray) -> Tuple[int, int, int, int]:
    """Locate the cube's bounding box by color-segmentation + connected
    components on a saturation-thresholded mask.

    The default saturation threshold (0.23) handles most natural
    backgrounds. When the resulting top component covers ≥95% of BOTH
    image dimensions, the cube has merged with a chromatically loud or
    textured background — the Set 46 regression fixture had 60–70% of pixels
    registering as saturated, and after dilation everything became one giant
    frame-spanning blob.

    In that case, retry with a stricter saturation threshold (0.40) so
    weakly-saturated backgrounds drop out while the cube's bright
    stickers (orange/red/yellow/green/blue typically have saturation
    well above 0.40) survive. If the retry also produces a
    frame-spanning component, fall through to the original — preserves
    the historical "return full image" fallback.

    The "both dimensions" criterion is important: a tightly-framed
    cube can legitimately span one dimension at >95%; only the
    background-merge failure spans both.
    """
    height, width = arr.shape[:2]
    roi = _find_cube_roi_at_threshold(arr, _ROI_SATURATION_MIN_DEFAULT)
    if _roi_covers_full_frame(roi, width, height):
        retry = _find_cube_roi_at_threshold(arr, _ROI_SATURATION_MIN_RETRY)
        if not _roi_covers_full_frame(retry, width, height):
            return retry
    return roi


def _find_cube_roi_at_threshold(arr: np.ndarray, sat_min: float) -> Tuple[int, int, int, int]:
    """Single-pass ROI detection at the given saturation threshold.
    Returns the full image as ROI when no usable signal is found
    (matches the pre-refactor fallback)."""
    hsv = _rgb_to_hsv_arrays(arr)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    saturated = (sat > sat_min) & (val > 0.20)
    if saturated.sum() < 200:
        return 0, 0, arr.shape[1], arr.shape[0]

    join = max(21, int(max(arr.shape[:2]) * 0.055) | 1)
    joined = _binary_dilate_square(saturated, join)
    comps = _connected_components(joined, min_area=250)
    if not comps:
        return 0, 0, arr.shape[1], arr.shape[0]

    height, width = arr.shape[:2]
    best = max(comps, key=lambda comp: _roi_score(comp, saturated, width, height))
    x0, y0, x1, y1 = best["bbox"]
    pad = int(max(x1 - x0, y1 - y0) * 0.12)
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)


def _roi_covers_full_frame(roi: Tuple[int, int, int, int], width: int, height: int) -> bool:
    """True if the ROI spans ≥95% of BOTH image dimensions.

    Both-dimensions logic: a legitimately tight-framed photo can span
    one dimension at high coverage (cube fills viewport width, say),
    but only a background-merge failure spans both. Catches the Set 46
    textured-background failure (100%×100%) without false-triggering on
    Set 15 (78%×61%) or any other corpus row.
    """
    x0, y0, x1, y1 = roi
    return (
        (x1 - x0) / width >= _ROI_FRAME_COVERAGE_THRESHOLD
        and (y1 - y0) / height >= _ROI_FRAME_COVERAGE_THRESHOLD
    )


def _binary_dilate_square(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return mask.astype(bool, copy=True)
    if size % 2 == 0:
        raise ValueError("Square dilation size must be odd.")

    radius = size // 2
    source = mask.astype(bool, copy=False)
    padded_x = np.pad(source, ((0, 0), (radius, radius)), mode="constant", constant_values=False)
    horizontal = np.lib.stride_tricks.sliding_window_view(padded_x, size, axis=1).any(axis=-1)
    padded_y = np.pad(horizontal, ((radius, radius), (0, 0)), mode="constant", constant_values=False)
    return np.lib.stride_tricks.sliding_window_view(padded_y, size, axis=0).any(axis=-1)


def _roi_score(comp: Dict, saturated: np.ndarray, width: int, height: int) -> float:
    x0, y0, x1, y1 = comp["bbox"]
    colored = saturated[y0:y1, x0:x1].sum()
    cx = ((x0 + x1) / 2.0) / width
    cy = ((y0 + y1) / 2.0) / height
    center_bonus = 1.0 - min(1.0, math.hypot(cx - 0.55, cy - 0.5))
    return float(colored) * (1.0 + center_bonus)


def _find_stickers(arr: np.ndarray, roi: Tuple[int, int, int, int]) -> List[Sticker]:
    x0, y0, x1, y1 = roi
    hsv = _rgb_to_hsv_arrays(arr)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    dark = val < 0.18
    colored = (sat > 0.16) & (val > 0.26)
    white_like = (sat < 0.20) & (val > 0.62)
    mask = (colored | white_like) & ~dark

    roi_mask = np.zeros(mask.shape, dtype=bool)
    roi_mask[y0:y1, x0:x1] = True
    mask &= roi_mask

    comps = _connected_components(mask, min_area=max(30, int(arr.shape[0] * arr.shape[1] * 0.000035)))
    stickers: List[Sticker] = []
    max_dim = max(arr.shape[:2])
    min_size = max(5, max_dim * 0.006)
    max_size = max_dim * 0.24

    for comp in comps:
        bx0, by0, bx1, by1 = comp["bbox"]
        width = bx1 - bx0
        height = by1 - by0
        if width < min_size or height < min_size or width > max_size or height > max_size:
            continue
        if bx0 <= x0 + 2 or by0 <= y0 + 2 or bx1 >= x1 - 2 or by1 >= y1 - 2:
            continue
        aspect = width / float(max(height, 1))
        if aspect < 0.25 or aspect > 4.0:
            continue
        fill = comp["area"] / float(max(width * height, 1))
        if fill < 0.20:
            continue

        pixels = arr[comp["ys"], comp["xs"]]
        rgb = tuple(int(v) for v in np.median(pixels, axis=0))  # type: ignore[assignment]
        match = classify_rgb(rgb)
        # Large white table/background fragments tend to be low-saturation and high-fill.
        if match.color == "white" and comp["area"] > arr.shape[0] * arr.shape[1] * 0.018:
            continue
        stickers.append(
            Sticker(
                id=len(stickers),
                center=((bx0 + bx1) / 2.0, (by0 + by1) / 2.0),
                bbox=(bx0, by0, bx1, by1),
                rgb=rgb,
                match=match,
                area=comp["area"],
                shape_angle=_component_shape_angle(comp),
            )
        )

    stickers = _filter_tiny_white_components(stickers)
    return _filter_cube_sticker_cluster(_dedupe_stickers(stickers))


def _dedupe_stickers(stickers: List[Sticker]) -> List[Sticker]:
    kept: List[Sticker] = []
    for sticker in sorted(stickers, key=lambda s: s.area, reverse=True):
        sx0, sy0, sx1, sy1 = sticker.bbox
        sw = sx1 - sx0
        sh = sy1 - sy0
        if any(math.hypot(sticker.center[0] - k.center[0], sticker.center[1] - k.center[1]) < 0.35 * max(sw, sh) for k in kept):
            continue
        sticker.id = len(kept)
        kept.append(sticker)
    return sorted(kept, key=lambda s: (s.center[1], s.center[0]))


def _filter_tiny_white_components(stickers: List[Sticker]) -> List[Sticker]:
    white_stickers = [sticker for sticker in stickers if sticker.match.color == "white"]
    colored_areas = [sticker.area for sticker in stickers if sticker.match.color != "white"]
    if (
        len(white_stickers) < MIN_WHITE_STICKERS_FOR_WHITE_NOISE_FILTER
        or len(colored_areas) < MIN_COLORED_STICKERS_FOR_WHITE_NOISE_FILTER
    ):
        return stickers

    area_floor = float(np.median(colored_areas)) * TINY_WHITE_COMPONENT_AREA_FRACTION
    filtered = [sticker for sticker in stickers if sticker.match.color != "white" or sticker.area >= area_floor]
    for index, sticker in enumerate(filtered):
        sticker.id = index
    return filtered


def _filter_cube_sticker_cluster(stickers: List[Sticker]) -> List[Sticker]:
    if len(stickers) < 18:
        return stickers

    spacing = _estimate_spacing(np.array([sticker.center for sticker in stickers], dtype=float))
    if spacing <= 0:
        return stickers
    threshold = max(40.0, spacing * 3.1)

    components: List[List[int]] = []
    seen = set()
    for index, sticker in enumerate(stickers):
        if index in seen:
            continue
        stack = [index]
        seen.add(index)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            cx, cy = stickers[current].center
            for other, candidate in enumerate(stickers):
                if other in seen:
                    continue
                ox, oy = candidate.center
                if math.hypot(cx - ox, cy - oy) <= threshold:
                    seen.add(other)
                    stack.append(other)
        components.append(component)

    if len(components) <= 1:
        return stickers

    largest = max(components, key=len)
    if len(largest) < max(12, int(len(stickers) * 0.55)):
        return stickers

    filtered = [stickers[index] for index in sorted(largest)]
    for index, sticker in enumerate(filtered):
        sticker.id = index
    return filtered


def _component_shape_angle(comp: Dict) -> Optional[float]:
    xs = comp.get("xs")
    ys = comp.get("ys")
    if xs is None or ys is None or len(xs) < 5:
        return None
    points = np.column_stack([xs, ys]).astype(float)
    points -= points.mean(axis=0)
    try:
        covariance = np.cov(points.T)
        values, vectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return None
    axis = vectors[:, int(np.argmax(values))]
    return math.degrees(math.atan2(float(axis[1]), float(axis[0]))) % 180.0


def _fit_face_grids(
    stickers: List[Sticker],
    arr: np.ndarray,
    scale: float,
    *,
    cube_hull: Optional[Sequence[Point]] = None,
) -> List[FaceGrid]:
    if len(stickers) < 9:
        return []
    points = np.array([sticker.center for sticker in stickers], dtype=float)
    spacing = _estimate_spacing(points)
    if spacing <= 0:
        return []

    candidates = _grid_candidates(points, list(range(len(stickers))), spacing, arr, stickers)
    _annotate_candidates_with_cube_hull(candidates, cube_hull)
    selected = _add_supplemental_grids(_select_grid_combo(candidates), candidates, max_grids=24)
    component_hull = _convex_hull([sticker.center for sticker in stickers])
    grids: List[FaceGrid] = []
    for grid_id, candidate in enumerate(selected):
        matched_indices = set(candidate["matched"])
        sampled = _sample_grid_stickers(
            arr,
            candidate["centers"],
            candidate.get("cell_matches"),
            stickers,
            matched_indices,
            scale,
            component_hull=component_hull,
        )
        grids.append(
            FaceGrid(
                id=grid_id,
                stickers=sampled,
                points=candidate["centers"],
                matched_count=candidate["matched_count"],
                fit_error=candidate["error"],
                cube_hull_inside_count=_optional_int(candidate.get("cube_hull_inside_count")),
                cube_hull_outside_count=_optional_int(candidate.get("cube_hull_outside_count")),
                cube_hull_source=candidate.get("cube_hull_source"),
            )
        )
    return grids


def _estimate_spacing(points: np.ndarray) -> float:
    nearest = []
    for i, point in enumerate(points):
        distances = [float(np.linalg.norm(point - other)) for j, other in enumerate(points) if i != j]
        if distances:
            nearest.append(min(distances))
    if not nearest:
        return 0.0
    return float(np.median(nearest))


def _best_grid_for_indices(points: np.ndarray, indices: List[int], spacing: float) -> Optional[Dict]:
    candidates = _grid_candidates(points, indices, spacing, arr=None, source_stickers=None)
    return candidates[0] if candidates else None


def _grid_candidates(
    points: np.ndarray,
    indices: List[int],
    spacing: float,
    arr: Optional[np.ndarray],
    source_stickers: Optional[Sequence[Sticker]] = None,
) -> List[Dict]:
    seen: Dict[Tuple[int, ...], Dict] = {}
    min_len = spacing * 0.55
    max_len = spacing * 1.85
    candidates = indices[:]
    attempts = 0
    max_attempts = 14000
    for anchor_idx in candidates:
        anchor = points[anchor_idx]
        neighbor_vectors = []
        for other_idx in candidates:
            if other_idx == anchor_idx:
                continue
            vec = points[other_idx] - anchor
            length = float(np.linalg.norm(vec))
            if min_len <= length <= max_len:
                neighbor_vectors.append((length, vec))
        neighbor_vectors = [vec for _, vec in sorted(neighbor_vectors, key=lambda item: item[0])[:10]]
        for u in neighbor_vectors:
            for v in neighbor_vectors:
                attempts += 1
                if attempts > max_attempts:
                    return best
                if np.allclose(u, v):
                    continue
                cos = abs(float(np.dot(u, v)) / max(np.linalg.norm(u) * np.linalg.norm(v), 1.0))
                if cos > 0.86:
                    continue
                for ar in range(3):
                    for ac in range(3):
                        origin = anchor - ar * u - ac * v
                        result = _score_grid(points, indices, origin, u, v, spacing)
                        if result["matched_count"] < 5:
                            continue
                        _annotate_candidate(result, arr, points, source_stickers)
                        key = tuple(sorted(result["matched"]))
                        current = seen.get(key)
                        if current is None or _candidate_rank(result) > _candidate_rank(current):
                            seen[key] = result
    return sorted(seen.values(), key=_candidate_rank, reverse=True)[:220]


def _annotate_candidate(candidate: Dict, arr: Optional[np.ndarray], points: np.ndarray, source_stickers: Optional[Sequence[Sticker]]) -> None:
    candidate["shape_spread"] = _candidate_shape_spread(candidate, source_stickers)
    centers = candidate["centers"]
    if arr is None:
        candidate["center_face"] = "?"
        return
    center_idx = candidate.get("cell_matches", [[None] * 3 for _ in range(3)])[1][1]
    if center_idx is None:
        candidate["center_face"] = "?"
        candidate["center_confidence"] = 0.0
        return
    cx, cy = points[center_idx]
    spacing = _grid_spacing(centers)
    rgb = _sample_patch(arr, cx, cy, max(3, int(spacing * 0.16)), avoid_core=True)
    match = classify_rgb(rgb)
    if source_stickers and 0 <= center_idx < len(source_stickers):
        rgb, match = _preserve_white_component_center(source_stickers[center_idx], rgb, match)
    candidate["center_rgb"] = rgb
    candidate["center_face"] = match.face
    candidate["center_confidence"] = match.confidence


def _candidate_rank(candidate: Dict) -> float:
    spread = min(65.0, float(candidate.get("shape_spread", 45.0)))
    return float(candidate["matched_count"]) * 18.0 - float(candidate["error"]) * 1.25 - spread * 1.25


def _candidate_shape_spread(candidate: Dict, source_stickers: Optional[Sequence[Sticker]]) -> float:
    if not source_stickers:
        return 45.0
    angles = []
    for idx in candidate.get("matched", []):
        if 0 <= idx < len(source_stickers):
            angle = source_stickers[idx].shape_angle
            if angle is not None:
                angles.append(angle)
    if len(angles) < 4:
        return 45.0
    return _undirected_angle_spread(angles)


def _undirected_angle_spread(angles: Sequence[float]) -> float:
    if not angles:
        return 45.0
    doubled = [math.radians(angle * 2.0) for angle in angles]
    x = sum(math.cos(angle) for angle in doubled) / len(doubled)
    y = sum(math.sin(angle) for angle in doubled) / len(doubled)
    concentration = max(1e-6, min(1.0, math.hypot(x, y)))
    circular_std = math.sqrt(max(0.0, -2.0 * math.log(concentration)))
    return math.degrees(circular_std) / 2.0


def _annotate_candidates_with_cube_hull(candidates: Sequence[Dict], cube_hull: Optional[Sequence[Point]]) -> None:
    if not cube_hull or len(cube_hull) < 3:
        return
    for candidate in candidates:
        centers = candidate.get("centers") or []
        inside = 0
        total = 0
        for row in centers:
            for point in row:
                total += 1
                if _outside_polygon_distance(point, cube_hull) <= REMBG_HULL_POINT_TOLERANCE:
                    inside += 1
        if total == 0:
            continue
        candidate["cube_hull_inside_count"] = inside
        candidate["cube_hull_outside_count"] = max(0, total - inside)
        candidate["cube_hull_source"] = REMBG_GRID_GUARD_SOURCE


def cube_hull_grid_penalty(inside_count: object) -> float:
    inside = _optional_int(inside_count)
    if inside is None or inside >= REMBG_GRID_INSIDE_MIN:
        return 0.0
    missing = REMBG_GRID_INSIDE_MIN - max(0, inside)
    return REMBG_GRID_OUTSIDE_BASE_PENALTY + missing * REMBG_GRID_OUTSIDE_EXTRA_PENALTY


def _cube_hull_candidate_penalty(candidate: Dict) -> float:
    inside = _optional_int(candidate.get("cube_hull_inside_count"))
    # The rembg hull is excellent at rejecting severe off-cube grids, but its
    # boundary can trim one row/column of a real face. Keep early grid-combo
    # selection conservative; later recognizer scoring can combine the hull
    # signal with sampled-cell extrapolation evidence.
    if inside is None or inside >= REMBG_GRID_INSIDE_MIN - 1:
        return 0.0
    return cube_hull_grid_penalty(inside)


def _select_grid_combo(candidates: List[Dict]) -> List[Dict]:
    if len(candidates) <= 3:
        return candidates
    best_combo: Optional[Tuple[Dict, Dict, Dict]] = None
    best_score = float("-inf")
    pool = candidates[:70]
    for i in range(len(pool)):
        a = pool[i]
        a_set = _candidate_matched_set(a)
        for j in range(i + 1, len(pool)):
            b = pool[j]
            b_set = _candidate_matched_set(b)
            ab_overlap = len(a_set & b_set)
            if ab_overlap > 5:
                continue
            for k in range(j + 1, len(pool)):
                c = pool[k]
                c_set = _candidate_matched_set(c)
                overlap = ab_overlap + len(a_set & c_set) + len(b_set & c_set)
                if overlap > 7:
                    continue
                combo = (a, b, c)
                score = _combo_score(combo, overlap)
                if score > best_score:
                    best_score = score
                    best_combo = combo
    if best_combo is None:
        return candidates[:3]
    return sorted(best_combo, key=lambda item: item["centers"][1][1])


def _add_supplemental_grids(selected: List[Dict], candidates: List[Dict], max_grids: int) -> List[Dict]:
    enriched = list(selected)
    selected_keys = {_candidate_key(candidate) for candidate in enriched}
    enriched = _add_low_overlap_triple_grids(enriched, candidates, max_grids)
    selected_faces = {candidate.get("center_face") for candidate in enriched}
    selected_keys = {_candidate_key(candidate) for candidate in enriched}
    for candidate in candidates:
        if len(enriched) >= max_grids:
            break
        face = candidate.get("center_face")
        key = _candidate_key(candidate)
        if key in selected_keys or face in selected_faces or face not in {"U", "R", "F", "D", "L", "B"}:
            continue
        matched_set = _candidate_matched_set(candidate)
        if any(len(matched_set & _candidate_matched_set(existing)) > 6 for existing in enriched):
            continue
        enriched.append(candidate)
        selected_faces.add(face)
        selected_keys.add(key)
    for candidate in candidates:
        if len(enriched) >= max_grids:
            break
        face = candidate.get("center_face")
        key = _candidate_key(candidate)
        if key in selected_keys or face not in {"U", "R", "F", "D", "L", "B"}:
            continue
        matched_set = _candidate_matched_set(candidate)
        if any(len(matched_set & _candidate_matched_set(existing)) > 7 for existing in enriched):
            continue
        enriched.append(candidate)
        selected_keys.add(key)
    return sorted(enriched, key=lambda item: item["centers"][1][1])


def _add_low_overlap_triple_grids(enriched: List[Dict], candidates: List[Dict], max_grids: int) -> List[Dict]:
    if len(enriched) >= max_grids:
        return enriched

    grouped: Dict[str, List[Dict]] = {face: [] for face in {"U", "R", "F", "D", "L", "B"}}
    for candidate in candidates:
        face = candidate.get("center_face")
        if face in grouped:
            grouped[face].append(candidate)

    triples: List[Tuple[float, Tuple[Dict, Dict, Dict]]] = []
    for anchor_face in FACE_GRID_ANCHORS:
        for side_a, side_b in FACE_GRID_SIDE_PAIRS:
            for anchor in grouped[anchor_face][:MAX_RESCUE_FACE_CANDIDATES]:
                for first in grouped[side_a][:MAX_RESCUE_FACE_CANDIDATES]:
                    for second in grouped[side_b][:MAX_RESCUE_FACE_CANDIDATES]:
                        triple = (anchor, first, second)
                        if len({_candidate_key(candidate) for candidate in triple}) < 3:
                            continue
                        overlap = _candidate_component_overlap(triple)
                        if overlap > MAX_RESCUE_GRID_COMPONENT_OVERLAP:
                            continue
                        score = _combo_score(triple, overlap) + sum(_candidate_rank(candidate) for candidate in triple) * 0.15
                        triples.append((score, triple))

    if not triples:
        return enriched

    selected_keys = {_candidate_key(candidate) for candidate in enriched}
    selected_triples = sorted(triples, key=lambda item: item[0], reverse=True)[:MAX_RESCUE_TRIPLES]
    for _, triple in selected_triples:
        for candidate in sorted(triple, key=_candidate_rank, reverse=True):
            if len(enriched) >= max_grids:
                return enriched
            key = _candidate_key(candidate)
            if key in selected_keys:
                continue
            enriched.append(candidate)
            selected_keys.add(key)
    return enriched


def _candidate_key(candidate: Dict) -> Tuple[int, ...]:
    cached = candidate.get("_candidate_key")
    if isinstance(cached, tuple):
        return cached
    key = tuple(sorted(candidate.get("matched", [])))
    candidate["_candidate_key"] = key
    return key


def _candidate_matched_set(candidate: Dict) -> set[int]:
    cached = candidate.get("_matched_set")
    if isinstance(cached, set):
        return cached
    matched = set(candidate.get("matched", []))
    candidate["_matched_set"] = matched
    return matched


def _candidate_component_overlap(candidates: Sequence[Dict]) -> int:
    matched_sets = [_candidate_matched_set(candidate) for candidate in candidates]
    overlap = 0
    for i, first in enumerate(matched_sets):
        for second in matched_sets[i + 1 :]:
            overlap += len(first & second)
    return overlap


def _combo_score(combo: Sequence[Dict], overlap: int) -> float:
    matched = sum(float(candidate["matched_count"]) for candidate in combo)
    error = sum(float(candidate["error"]) for candidate in combo)
    shape_spread = sum(min(65.0, float(candidate.get("shape_spread", 45.0))) for candidate in combo)
    center_faces = [candidate.get("center_face", "?") for candidate in combo]
    distinct_centers = len(set(center_faces))
    anchor_bonus = 70.0 * len({face for face in center_faces if face in {"U", "D"}})
    side_bonus = 8.0 * len({face for face in center_faces if face in {"R", "F", "L", "B"}})
    geometry_penalty = sum(_cube_hull_candidate_penalty(candidate) for candidate in combo)
    return (
        matched * 36.0
        - error * 1.7
        - shape_spread * 0.9
        + distinct_centers * 24.0
        + anchor_bonus
        + side_bonus
        - overlap * 18.0
        - geometry_penalty
    )


def _score_grid(points: np.ndarray, indices: List[int], origin: np.ndarray, u: np.ndarray, v: np.ndarray, spacing: float) -> Dict:
    centers: List[List[Point]] = []
    cell_matches: List[List[Optional[int]]] = []
    matched = []
    errors = []
    tolerance = max(12.0, spacing * 0.44)
    available = set(indices)
    point_xs = points[:, 0]
    point_ys = points[:, 1]
    for r in range(3):
        row = []
        match_row: List[Optional[int]] = []
        for c in range(3):
            predicted = origin + r * u + c * v
            px = float(predicted[0])
            py = float(predicted[1])
            row.append((px, py))
            best_idx, best_dist = _nearest_available_point(
                points,
                available,
                px,
                py,
                tolerance,
                point_xs=point_xs,
                point_ys=point_ys,
            )
            if best_idx is not None:
                matched.append(best_idx)
                errors.append(float(best_dist))
                available.remove(best_idx)
            match_row.append(best_idx)
        centers.append(row)
        cell_matches.append(match_row)
    result = {
        "centers": centers,
        "cell_matches": cell_matches,
        "matched": matched,
        "matched_count": len(matched),
        "error": float(np.mean(errors) if errors else 9999.0),
    }
    refined = _refine_grid_homography(points, indices, result, spacing)
    if refined and _candidate_rank(refined) > _candidate_rank(result):
        return refined
    return result


def _refine_grid_homography(points: np.ndarray, indices: List[int], candidate: Dict, spacing: float) -> Optional[Dict]:
    src = []
    dst = []
    for r, row in enumerate(candidate["cell_matches"]):
        for c, idx in enumerate(row):
            if idx is None:
                continue
            src.append((float(c), float(r)))
            dst.append(tuple(float(v) for v in points[idx]))
    if len(src) < 4 or _canonical_points_degenerate(src):
        return None
    homography = _fit_homography(np.array(src, dtype=float), np.array(dst, dtype=float))
    if homography is None:
        return None

    centers: List[List[Point]] = []
    for r in range(3):
        row = []
        for c in range(3):
            projected = homography @ np.array([float(c), float(r), 1.0])
            if abs(projected[2]) < 1e-6:
                return None
            row.append((float(projected[0] / projected[2]), float(projected[1] / projected[2])))
        centers.append(row)
    return _score_grid_centers(points, indices, centers, spacing)


def _score_grid_centers(points: np.ndarray, indices: List[int], centers: List[List[Point]], spacing: float) -> Dict:
    tolerance = max(12.0, spacing * 0.46)
    available = set(indices)
    matched = []
    errors = []
    cell_matches: List[List[Optional[int]]] = []
    point_xs = points[:, 0]
    point_ys = points[:, 1]
    for row in centers:
        match_row: List[Optional[int]] = []
        for center in row:
            best_idx, best_dist = _nearest_available_point(
                points,
                available,
                float(center[0]),
                float(center[1]),
                tolerance,
                point_xs=point_xs,
                point_ys=point_ys,
            )
            if best_idx is not None:
                matched.append(best_idx)
                errors.append(float(best_dist))
                available.remove(best_idx)
            match_row.append(best_idx)
        cell_matches.append(match_row)
    return {
        "centers": centers,
        "cell_matches": cell_matches,
        "matched": matched,
        "matched_count": len(matched),
        "error": float(np.mean(errors) if errors else 9999.0),
    }


def _nearest_available_point(
    points: np.ndarray,
    available: Iterable[int],
    px: float,
    py: float,
    tolerance: float,
    *,
    point_xs: Optional[np.ndarray] = None,
    point_ys: Optional[np.ndarray] = None,
) -> Tuple[Optional[int], Optional[float]]:
    best_idx = None
    best_dist_sq = tolerance * tolerance
    if point_xs is not None and point_ys is not None:
        for idx in available:
            dx = float(point_xs[idx]) - px
            dy = float(point_ys[idx]) - py
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_idx = idx
                best_dist_sq = dist_sq
    else:
        for idx in available:
            dx = float(points[idx, 0]) - px
            dy = float(points[idx, 1]) - py
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_idx = idx
                best_dist_sq = dist_sq
    if best_idx is None:
        return None, None
    return best_idx, math.sqrt(best_dist_sq)


def _fit_homography(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    matrix = np.array(rows, dtype=float)
    try:
        _, _, vh = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        return None
    homography = vh[-1].reshape(3, 3)
    if abs(homography[2, 2]) > 1e-9:
        homography = homography / homography[2, 2]
    if not np.isfinite(homography).all():
        return None
    return homography


def _canonical_points_degenerate(points: Sequence[Tuple[float, float]]) -> bool:
    xs = {point[0] for point in points}
    ys = {point[1] for point in points}
    return len(xs) < 2 or len(ys) < 2


def _sample_grid_stickers(
    arr: np.ndarray,
    centers: List[List[Point]],
    cell_matches: Optional[List[List[Optional[int]]]],
    source_stickers: List[Sticker],
    matched_indices: Iterable[int],
    scale: float,
    *,
    component_hull: Sequence[Point],
) -> List[List[Sticker]]:
    spacing = _grid_spacing(centers)
    radius = max(3, int(spacing * 0.16))
    matched_set = set(matched_indices)
    source_centers = [sticker.center for sticker in source_stickers]
    matched_source_centers = [
        source_stickers[index].center
        for index in matched_set
        if 0 <= index < len(source_stickers)
    ]
    matched_component_hull = _convex_hull(matched_source_centers)
    rows: List[List[Sticker]] = []
    next_id = 1000
    for r in range(3):
        row: List[Sticker] = []
        for c in range(3):
            cx, cy = centers[r][c]
            matched_idx = cell_matches[r][c] if cell_matches and cell_matches[r][c] in matched_set else None
            if matched_idx is not None:
                source = source_stickers[matched_idx]
                sample_x, sample_y = source.center
                rgb = _sample_patch(arr, sample_x, sample_y, radius, avoid_core=(r == 1 and c == 1))
                match = classify_rgb(rgb)
                if r == 1 and c == 1:
                    rgb, match = _preserve_white_component_center(source, rgb, match)
                sticker = Sticker(source.id, source.center, source.bbox, rgb, match, source.area, source.source, source.shape_angle)
            else:
                rgb = _sample_patch(arr, cx, cy, radius, avoid_core=(r == 1 and c == 1))
                match = classify_rgb(rgb)
                bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
                sticker = Sticker(next_id, (cx, cy), bbox, rgb, match, radius * radius, "grid_sample")
                sticker.grid_spacing = spacing  # type: ignore[attr-defined]
                sticker.nearest_component_distance = _nearest_point_distance(  # type: ignore[attr-defined]
                    (cx, cy),
                    source_centers,
                )
                sticker.outside_component_hull_distance = _outside_polygon_distance(  # type: ignore[attr-defined]
                    (cx, cy),
                    component_hull,
                )
                sticker.nearest_grid_component_distance = _nearest_point_distance(  # type: ignore[attr-defined]
                    (cx, cy),
                    matched_source_centers,
                )
                sticker.outside_grid_component_hull_distance = _outside_polygon_distance(  # type: ignore[attr-defined]
                    (cx, cy),
                    matched_component_hull,
                )
                next_id += 1
            row.append(sticker)
        rows.append(row)
    return rows


def _preserve_white_component_center(source: Sticker, sampled_rgb: RGB, sampled_match: ColorMatch) -> Tuple[RGB, ColorMatch]:
    source_match = source.match
    if source_match.color != "white" or source_match.confidence < 0.45:
        return sampled_rgb, sampled_match
    _, _, source_value = rgb_to_hsv(source.rgb)
    if source_value < 0.86:
        return sampled_rgb, sampled_match
    if sampled_match.color == "white":
        return sampled_rgb, sampled_match

    _, sampled_saturation, sampled_value = rgb_to_hsv(sampled_rgb)
    if sampled_value < 0.58 or (sampled_saturation < 0.18 and sampled_match.confidence < 0.35):
        return source.rgb, source_match
    return sampled_rgb, sampled_match


def _sample_patch(arr: np.ndarray, cx: float, cy: float, radius: int, avoid_core: bool) -> RGB:
    height, width = arr.shape[:2]
    x0, x1 = max(0, int(cx - radius)), min(width, int(cx + radius + 1))
    y0, y1 = max(0, int(cy - radius)), min(height, int(cy + radius + 1))
    patch = arr[y0:y1, x0:x1]
    if patch.size == 0:
        return 0, 0, 0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = dist <= radius
    if avoid_core:
        mask &= dist >= radius * 0.38
    pixels = patch[mask]
    if len(pixels) == 0:
        pixels = patch.reshape(-1, 3)
    return tuple(int(v) for v in np.median(pixels, axis=0))  # type: ignore[return-value]


def _grid_spacing(centers: List[List[Point]]) -> float:
    distances = []
    for r in range(3):
        for c in range(2):
            distances.append(math.hypot(centers[r][c][0] - centers[r][c + 1][0], centers[r][c][1] - centers[r][c + 1][1]))
    for r in range(2):
        for c in range(3):
            distances.append(math.hypot(centers[r][c][0] - centers[r + 1][c][0], centers[r][c][1] - centers[r + 1][c][1]))
    return float(np.median(distances)) if distances else 12.0


def _nearest_point_distance(point: Point, points: Sequence[Point]) -> float:
    if not points:
        return 0.0
    return min(math.hypot(point[0] - other[0], point[1] - other[1]) for other in points)


def _outside_polygon_distance(point: Point, polygon: Sequence[Point]) -> float:
    if len(polygon) < 3 or _point_in_polygon(point, polygon):
        return 0.0
    return min(
        _point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        denom = yj - yi
        if abs(denom) < 1e-9:
            denom = 1e-9
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / denom + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def _hull_from_mask(mask: np.ndarray) -> List[Point]:
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    ys, xs = np.where(boundary)
    if len(xs) < 3:
        ys, xs = np.where(mask)
    if len(xs) < 3:
        return []
    points = [(float(x), float(y)) for x, y in zip(xs, ys)]
    return _convex_hull(points)


def _expand_polygon(poly: Sequence[Point], padding_fraction: float) -> List[Point]:
    if len(poly) < 3:
        return [(float(x), float(y)) for x, y in poly]
    cx = sum(x for x, _ in poly) / len(poly)
    cy = sum(y for _, y in poly) / len(poly)
    return [
        (
            float(cx + (x - cx) * (1.0 + padding_fraction)),
            float(cy + (y - cy) * (1.0 + padding_fraction)),
        )
        for x, y in poly
    ]


def _convex_hull(points: Sequence[Point]) -> List[Point]:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 1:
        return list(unique)

    def cross(origin: Point, first: Point, second: Point) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: List[Point] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: List[Point] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rgb_to_hsv_arrays(arr: np.ndarray) -> np.ndarray:
    rgb = arr.astype(float) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    delta = maxc - minc
    sat = np.where(maxc == 0, 0, delta / np.maximum(maxc, 1e-6))
    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-6
    safe_delta = np.where(nonzero, delta, 1.0)
    hue = np.where((maxc == r) & nonzero, ((g - b) / safe_delta) % 6, hue)
    hue = np.where((maxc == g) & nonzero, ((b - r) / safe_delta) + 2, hue)
    hue = np.where((maxc == b) & nonzero, ((r - g) / safe_delta) + 4, hue)
    hue /= 6.0
    return np.dstack([hue, sat, maxc])


def _connected_components(mask: np.ndarray, min_area: int) -> List[Dict]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    comps: List[Dict] = []
    true_points = np.argwhere(mask)
    for sy, sx in true_points:
        if visited[sy, sx]:
            continue
        queue = deque([(int(sx), int(sy))])
        visited[sy, sx] = True
        xs = []
        ys = []
        while queue:
            x, y = queue.popleft()
            xs.append(x)
            ys.append(y)
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        if len(xs) >= min_area:
            xs_arr = np.array(xs, dtype=int)
            ys_arr = np.array(ys, dtype=int)
            comps.append(
                {
                    "area": len(xs),
                    "bbox": (int(xs_arr.min()), int(ys_arr.min()), int(xs_arr.max() + 1), int(ys_arr.max() + 1)),
                    "xs": xs_arr,
                    "ys": ys_arr,
                }
            )
    return comps


def _make_overlay(image: Image.Image, roi: Tuple[int, int, int, int], stickers: List[Sticker], grids: List[FaceGrid]) -> str:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rectangle(roi, outline=(48, 120, 255, 210), width=3)
    for sticker in stickers:
        color = _face_draw_color(sticker.face)
        draw.rectangle(sticker.bbox, outline=color, width=2)
        draw.text((sticker.center[0] + 3, sticker.center[1] + 3), f"{sticker.id}:{sticker.face}", fill=(0, 0, 0, 230))
    for grid in grids:
        grid_color = _face_draw_color(grid.center_face)
        for r in range(3):
            for c in range(3):
                cx, cy = grid.points[r][c]
                draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=grid_color)
                draw.text((cx + 6, cy - 7), f"{grid.id}{r}{c}", fill=(0, 0, 0, 235))
    buf = io.BytesIO()
    overlay.thumbnail((1200, 1200))
    overlay.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _face_draw_color(face: str) -> Tuple[int, int, int, int]:
    return {
        "U": (255, 255, 255, 240),
        "R": (220, 35, 35, 240),
        "F": (20, 165, 80, 240),
        "D": (245, 220, 35, 240),
        "L": (245, 130, 35, 240),
        "B": (50, 90, 220, 240),
    }.get(face, (255, 0, 255, 240))
