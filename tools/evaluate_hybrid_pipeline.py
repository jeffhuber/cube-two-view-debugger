#!/usr/bin/env python3
"""End-to-end hybrid pipeline evaluator.

The hypothesis under test: take the existing recognizer's face-quad
detection (which delivers 82.7% per-sticker via WhiteUpRecognizer
with rejection) and pipe its quads through rectification +
knn5_lab_full classification (which delivers 99.29% on
rectified-from-human-quads per PR #150). Does the classification-side
improvement transfer to the end-to-end auto pipeline?

Pipeline:

  A+B images
    → analyze_image() per side  (existing recognizer's geometry)
    → grids grouped by center_face → best per face
    → 3x3 grid centers → 4-point face quad via homography
    → rectify each face to a 300x300 square (PR #136)
    → 9 sticker samples per face → classify_rgb (env-selected mode)
    → joint A+B multiset face-ID (PR #126)
    → assemble 54-state in URFDLB order
    → compare to GT

Run twice — once with `CUBE_RECOGNIZER_CLASSIFIER=canonical`, once
with `=knn5_lab_full` — to isolate the classification-side lift.

NO production-recognizer changes. Tooling-only.

Per `COORDINATION.md` sweep-logging convention: per-pair progress to
stderr with flush=True; log file should be redirected with
`> log 2>&1` (not `2>&1 > log`).

Usage:
  CUBE_RECOGNIZER_CLASSIFIER=knn5_lab_full \\
    .venv/bin/python tools/evaluate_hybrid_pipeline.py
  .venv/bin/python tools/evaluate_hybrid_pipeline.py --only-sets 46 47 61 62
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import COLOR_TO_FACE  # noqa: E402
from rubik_recognizer.image_pipeline import analyze_image  # noqa: E402
from rubik_recognizer.validation import FACE_ORDER  # noqa: E402
from tools.extract_color_samples import (  # noqa: E402
    discover_additional_tasks,
    face_colors_from_state,
    load_corpus_tasks,
    parse_ground_truth,
)
from tools.inspect_cube_isolation import point_in_polygon  # noqa: E402
from tools.propose_geometry_labels import (  # noqa: E402
    _face_quad_from_grid_centers,
    _face_quads_from_hexagon,
    _fit_hexagon_to_hull,
    _get_rembg_session,
    _hull_from_mask,
)
from tools.rectify_faces import (  # noqa: E402
    DEFAULT_FACE_SIZE,
    extract_stickers_from_rectified,
    rectify_face,
)
from tools.sample_stickers_from_hull import (  # noqa: E402
    apply_orientation,
    discover_orientation,
    identify_faces_jointly,
)

CORPUS_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_REPORT = REPO_ROOT / "runs" / "hybrid_pipeline_report.json"
DEFAULT_SUMMARY = REPO_ROOT / "runs" / "hybrid_pipeline_summary.txt"
PROCESSING_MAX = 1150

EXPECTED_FACES_BY_SIDE = {"A": ("U", "R", "F"), "B": ("D", "L", "B")}
OOD_SETS = {"57", "58", "61", "62"}

# Hull-guard threshold (matches Codex's REMBG_GRID_INSIDE_MIN in PR #141).
# Any analyze_image grid with fewer than this many sticker-center points
# inside the rembg cube hull is rejected as spatially incoherent.
HULL_GUARD_INSIDE_MIN = 7


def _load_processing_image(image_path: Path) -> Tuple[Image.Image, np.ndarray]:
    """EXIF-correct + resize to max 1150, same as the rest of the tooling
    AND same as analyze_image's internal pipeline (so coordinates are
    directly comparable)."""
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    natural_max = max(image.size)
    if natural_max > PROCESSING_MAX:
        scale = PROCESSING_MAX / float(natural_max)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return image, np.asarray(image)


def _rembg_cube_hull(processing_image: Image.Image) -> List[Tuple[float, float]]:
    """Compute rembg u2net cube hull in processing-image coords. Returns
    [] on rembg failure (e.g. mask empty). This is the same hull that
    Codex's PR #141 uses to guard production-recognizer grid ranking.

    Cached on the underlying image object's id() because each pair runs
    this twice (proposer + verification) and rembg is the slow step."""
    cache = _rembg_cube_hull._cache  # type: ignore[attr-defined]
    key = id(processing_image)
    if key in cache:
        return cache[key]
    try:
        from rembg import remove
    except ImportError:
        cache[key] = []
        return cache[key]
    try:
        rgba = remove(processing_image, session=_get_rembg_session("u2net"))
    except Exception:
        cache[key] = []
        return cache[key]
    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    mask = alpha > 128
    if not mask.any():
        cache[key] = []
        return cache[key]
    hull = list(_hull_from_mask(mask))
    cache[key] = hull
    return hull


_rembg_cube_hull._cache = {}  # type: ignore[attr-defined]


def _grid_inside_hull_count(grid, hull: List[Tuple[float, float]]) -> int:
    """Count how many of a FaceGrid's 9 sticker centers fall inside the
    cube hull. Mirrors Codex's `cube_hull_inside_count` semantics from
    rubik_recognizer/image_pipeline.py."""
    if not hull or len(hull) < 3:
        return 9  # no hull → don't apply guard (degrade gracefully)
    inside = 0
    for row in grid.points:
        for (x, y) in row:
            if point_in_polygon((float(x), float(y)), hull):
                inside += 1
    return inside


SLOT_SRC_EXPECTED_BY_SIDE = {"A": frozenset({"U", "R", "F"}), "B": frozenset({"D", "L", "B"})}

# Fit-error threshold for the trust-fit-or-derive hybrid (proposal A).
# When a chosen grid's fit_error exceeds this, treat analyze_image's
# 3x3 as spatially incoherent (e.g. spans multiple physical faces) and
# replace its extrapolated face quad with one derived geometrically
# from the rembg hexagon hull via _face_quads_from_hexagon. Calibrated
# against the Set 17 A diagnostic where U=0.34, R=2.01, B=6.50 — so
# threshold around 3-5 cleanly separates the spatially-coherent grids
# from the multi-face spans.
FIT_ERROR_FALLBACK_THRESHOLD = 4.0


def _rembg_hexagon(processing_image: Image.Image) -> Optional[List[Tuple[float, float]]]:
    """rembg → hull → 6-vertex hexagon (CW from top). Returns None on failure."""
    try:
        from rembg import remove
    except ImportError:
        return None
    try:
        rgba = remove(processing_image, session=_get_rembg_session("u2net"))
    except Exception:
        return None
    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    mask = alpha > 128
    if not mask.any():
        return None
    hull = _hull_from_mask(mask)
    if len(hull) < 6:
        return None
    hexagon = _fit_hexagon_to_hull(hull)
    if hexagon is None or len(hexagon) != 6:
        return None
    return [(float(x), float(y)) for x, y in hexagon]


def _template_face_quads_from_image(
    processing_image: Image.Image,
) -> Optional[Dict[str, List[Tuple[float, float]]]]:
    """Rembg hull → 6-vertex hexagon → 3 anonymous face quads
    (top/right/left) via the Geometry Labeler's template formula.
    Returns None if rembg/hexagon fitting fails."""
    hexagon = _rembg_hexagon(processing_image)
    if hexagon is None:
        return None
    return _face_quads_from_hexagon(hexagon)


def _clip_to_hull(point: Tuple[float, float],
                  hull: List[Tuple[float, float]]) -> Tuple[float, float]:
    """If `point` is outside the cube hull (rembg silhouette polygon),
    project to the nearest hull EDGE point. If inside, return as-is.

    Why: even "trusted" face quads from analyze_image have extrapolated
    corners that can extend 30-60 px past the actual face boundary
    (see Set 17 A R.S at (426, 1012) — well below the cube). When
    those corners propagate into topology-derived neighbors via shared
    vertices, the bad corner pulls the neighbor's quad outside the
    cube too, dropping sample positions onto the table/background.
    """
    if not hull or len(hull) < 3:
        return point
    if point_in_polygon(point, hull):
        return point
    # Project to nearest point on hull edge (segment-by-segment).
    best_d2 = float("inf")
    best_pt = point
    px, py = point
    for i in range(len(hull)):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % len(hull)]
        # Closest point on segment AB to P
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-9:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        d2 = (px - proj_x) ** 2 + (py - proj_y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_pt = (proj_x, proj_y)
    return best_pt


def _hull_vertex_in_direction(
    hull: List[Tuple[float, float]],
    center: Tuple[float, float],
    angle_radians: float,
    angle_tolerance_radians: float = 0.5,  # ~28°
) -> Optional[Tuple[float, float]]:
    """Find the hull vertex farthest from `center` in the direction
    `angle_radians` (in standard image coords: 0=east, π/2=south,
    -π/2=north). Returns None if no hull vertex within tolerance.

    Used for unique-to-face vertices (h0=top, h2=right-mid, h4=left-mid)
    when the 6-vertex hexagon fit is degenerate. The full hull (51+
    vertices on a typical cube) gives more precise placement than the
    angular-sector hexagon, which can collapse adjacent vertices.
    """
    if not hull:
        return None
    import math as _math
    best_dist = -1.0
    best_pt = None
    for hp in hull:
        dx = hp[0] - center[0]
        dy = hp[1] - center[1]
        if dx == 0 and dy == 0:
            continue
        angle = _math.atan2(dy, dx)
        # Angular difference, wrapped to [0, π]
        diff = abs(angle - angle_radians)
        if diff > _math.pi:
            diff = 2 * _math.pi - diff
        if diff > angle_tolerance_radians:
            continue
        dist = _math.hypot(dx, dy)
        if dist > best_dist:
            best_dist = dist
            best_pt = hp
    return best_pt


def _cardinal_corners(quad: List[Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Identify a 4-corner face quad's corners by cardinal position
    (N=min-y, E=max-x, S=max-y, W=min-x). This works for parallelograms
    aligned with image axes (iso-projected cube faces are approximately
    so) and is invariant to canonical_corner_order's start-index
    ambiguity.

    Why this matters: `canonical_corner_order` sorts CW from the
    smallest-CW-from-N angle. Which corner ends up at index 0 depends
    on the quad's specific geometry — for U on Set 17 A, index 0 was
    the EAST corner (h1); for U on a different photo it could be the
    NORTH corner (h0). Treating index 0 as "always h0" gives wrong
    shared-corner identification across the cube's faces.
    """
    by_y = sorted(quad, key=lambda p: p[1])  # ascending y → north first
    by_x = sorted(quad, key=lambda p: p[0])  # ascending x → west first
    return {
        "N": by_y[0],   # smallest y
        "S": by_y[-1],  # largest y
        "W": by_x[0],   # smallest x
        "E": by_x[-1],  # largest x
    }


def _derive_face_quad_topology_aware(
    slot_position: str,
    trusted_quads_by_position: Dict[str, List[Tuple[float, float]]],
    hexagon: List[Tuple[float, float]],
    cube_hull: Optional[List[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    """Derive a face quad's 4 corners using the cube-face TOPOLOGY:

      * The 3 visible faces of a cube in iso-projection share specific
        vertices. Every pair of adjacent faces shares an EDGE, and all
        3 faces meet at a single cube-center vertex in image space.
      * When 1-2 neighbor faces are accurately fitted (their quads
        come from a low-fit-error analyze_image grid), the SHARED
        corners of the untrusted face are already known with the
        precision of those trusted quads — no need to approximate
        them from the rembg hexagon.
      * Only corners that are unique to the untrusted face (not shared
        with any trusted neighbor) need to come from the hexagon.

    Hexagon convention (6 vertices CW from top, indices 0-5):
      h0=top, h1=upper-right, h2=right-middle, h3=bottom,
      h4=left-middle, h5=upper-left.

    Each face's 4 corners in canonical CW-from-N order:
      top   (U/D): [h0, h1, center, h5]
      right (R/L): [h1, h2, h3, center]
      left  (F/B): [h5, center, h3, h4]

    Shared vertices:
      h1 = top ∩ right
      h3 = right ∩ left
      h5 = top ∩ left
      center = top ∩ right ∩ left (the cube center vertex)

    Best-case (e.g. Set 17 A, derive 'left' with both top and right
    trusted): 3 of 4 corners come from precision sources (top quad
    contributes h5 + center; right quad contributes h3 + center
    consistency check); only h4 from hexagon approximation.

    Args:
      slot_position: "top", "right", or "left"
      trusted_quads_by_position: dict with subset of those keys
      hexagon: 6 hexagon vertices CW from top
    """
    h0, h1, h2, h3, h4, h5 = hexagon

    # Cardinal-position corner extraction. For each trusted quad, get the
    # 4 corners keyed by cardinal direction (N/E/S/W). The mapping
    # from cardinal direction to hexagon-named vertex is fixed by the
    # cube-face topology and independent of canonical_corner_order
    # ordering ambiguity:
    #
    #   top   (U): N=h0, E=h1, S=center, W=h5
    #   right (R): N=h1, E=h2, S=h3,    W=center
    #   left  (F): N=h5, E=center, S=h3, W=h4
    #
    # So shared vertices:
    #   h1 = top.E = right.N
    #   h3 = right.S = left.S
    #   h5 = top.W = left.N
    #   center = top.S = right.W = left.E
    cardinals_by_position = {
        pos: _cardinal_corners(quad)
        for pos, quad in trusted_quads_by_position.items()
    }

    def from_neighbor(position: str, direction: str) -> Optional[Tuple[float, float]]:
        c = cardinals_by_position.get(position)
        corner = c.get(direction) if c else None
        if corner is None:
            return None
        # Clip to cube hull: extrapolated corners from
        # _face_quad_from_grid_centers can extend past the cube outline
        # (e.g., Set 17 A's R.S at (426, 1012) — well below the cube
        # hull's bottom). Clipping bounds the error to the hull boundary.
        if cube_hull is not None:
            return _clip_to_hull(corner, cube_hull)
        return corner

    def avg_points(*pts: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        valid = [p for p in pts if p is not None]
        if not valid:
            return None
        return (sum(p[0] for p in valid) / len(valid),
                sum(p[1] for p in valid) / len(valid))

    # Cube center: shared by all 3 faces. Best estimate is consensus from
    # trusted neighbors' shared-center corners. Fall back to hexagon
    # centroid only if no trusted neighbors.
    center_from_top = from_neighbor("top", "S")
    center_from_right = from_neighbor("right", "W")
    center_from_left = from_neighbor("left", "E")
    center = avg_points(center_from_top, center_from_right, center_from_left)
    if center is None:
        center = (sum(p[0] for p in hexagon) / 6.0,
                  sum(p[1] for p in hexagon) / 6.0)

    # Shared hexagon vertices via cardinal-position lookup
    h1_shared = from_neighbor("top", "E") or from_neighbor("right", "N")
    h3_shared = from_neighbor("right", "S") or from_neighbor("left", "S")
    h5_shared = from_neighbor("top", "W") or from_neighbor("left", "N")

    # Cube faces in iso projection are PARALLELOGRAMS, so their diagonals
    # bisect each other: for corners [A, B, C, D] in order around the
    # perimeter, A + C = B + D. When 3 corners are known precisely from
    # trusted neighbors, the 4th can be derived EXACTLY via parallelogram
    # geometry instead of from the rembg hexagon approximation. This
    # matters especially when the hexagon is degenerate (collapsed/missing
    # vertices), which happens on yawed cubes — Set 17 A's `h2` (would-be
    # right-middle vertex) ended up at the bottom of the image because the
    # angular-sector hexagon fitter couldn't find a clear right-side vertex.

    def parallelogram_4th(a, b, c):
        """Given 3 corners of a parallelogram in perimeter order, return
        the 4th. Uses A+C = B+D → D = A + C - B (when a/b/c are the 3
        consecutive corners and D is the 4th to close the loop)."""
        if a is None or b is None or c is None:
            return None
        return (a[0] - b[0] + c[0], a[1] - b[1] + c[1])

    # Compose the 4-corner quad in CANONICAL CW-from-N order so the
    # rectify step gets a consistent corner sequence. The canonical
    # order for each slot position is:
    #   top   (CW from N): [h0, h1, center, h5]
    #   right (CW from N): [h1, h2, h3, center]
    #   left  (CW from N): [h5, center, h3, h4]
    # but the calling code feeds the result through canonical_corner_order
    # anyway, so any order that returns the correct 4 corners is fine.
    # Unique-to-face vertex derivation. Priority order:
    #   1. Parallelogram (when 3 other corners trusted) — geometrically exact
    #      for orthographic iso, approximate for real photos with perspective
    #   2. Full-hull angular lookup (precise hull point in expected direction)
    #   3. 6-vertex hexagon (often degenerate on yawed cubes — last resort)
    import math as _math
    # Angle from cube center to each unique vertex (image coords:
    # 0=east, π/2=south, -π/2=north). Based on iso projection geometry.
    UNIQUE_VERTEX_ANGLES = {
        "h0_top": -_math.pi / 2,         # straight up (north)
        "h2_right_mid": _math.pi / 6,    # east-southeast (~30° down from east)
        "h4_left_mid": 5 * _math.pi / 6, # west-southeast (~30° down from west)
    }

    def unique_vertex(parallelogram_value, hexagon_value, angle_key):
        """Pick best estimate for a unique-to-face vertex: parallelogram
        derivation > full-hull lookup > degenerate hexagon fallback."""
        if parallelogram_value is not None:
            # Clip to hull (parallelogram can also overshoot)
            if cube_hull is not None:
                return _clip_to_hull(parallelogram_value, cube_hull)
            return parallelogram_value
        if cube_hull is not None and center is not None:
            hull_pt = _hull_vertex_in_direction(
                cube_hull, center, UNIQUE_VERTEX_ANGLES[angle_key],
            )
            if hull_pt is not None:
                return hull_pt
        return hexagon_value

    if slot_position == "top":
        # corners: [h0, h1, center, h5]
        ch1 = h1_shared or h1
        ch5 = h5_shared or h5
        para_h0 = None
        if h1_shared is not None and h5_shared is not None and center is not None:
            para_h0 = parallelogram_4th(ch1, center, ch5)
        ch0 = unique_vertex(para_h0, h0, "h0_top")
        return [ch0, ch1, center, ch5]
    elif slot_position == "right":
        # corners: [h1, h2, h3, center]
        ch1 = h1_shared or h1
        ch3 = h3_shared or h3
        para_h2 = None
        if h1_shared is not None and h3_shared is not None and center is not None:
            para_h2 = parallelogram_4th(ch1, center, ch3)
        ch2 = unique_vertex(para_h2, h2, "h2_right_mid")
        return [ch1, ch2, ch3, center]
    elif slot_position == "left":
        # corners: [h5, center, h3, h4]
        ch5 = h5_shared or h5
        ch3 = h3_shared or h3
        para_h4 = None
        if h5_shared is not None and h3_shared is not None and center is not None:
            para_h4 = parallelogram_4th(ch5, center, ch3)
        ch4 = unique_vertex(para_h4, h4, "h4_left_mid")
        return [ch5, center, ch3, ch4]
    else:
        raise ValueError(f"unknown slot_position: {slot_position!r}")


def _slot_to_template_position(
    slot_label: str,
    quads_by_slot: Dict[str, List[Tuple[float, float]]],
    side: str,
) -> Optional[str]:
    """Map a slot (U/R/F or D/L/B) to the template's positional key
    (top/left/right) given the OTHER slots' current quads. The anchor
    slot (U or D) always maps to 'top'. For side-face slots: use the
    centroid x of each currently-assigned slot to determine left vs
    right. If the target slot is unassigned, infer its position from
    the centroid of the OTHER side-face slot (it gets the OTHER
    position).

    Why centroid x: the cube's iso projection puts the 'left' face
    geometrically on image-left and 'right' face on image-right.
    Even when yaw rotates which color is on which side, the
    LEFT/RIGHT image positions stay constant — so quads are matched
    to template positions by their CURRENT image-x location."""
    anchor = "U" if side == "A" else "D"
    if slot_label == anchor:
        return "top"

    side_face_labels = ("R", "F") if side == "A" else ("L", "B")
    other_slot = next(s for s in side_face_labels if s != slot_label)

    # Compute centroid x of both side-face slots if assigned
    def centroid_x(slot: str) -> Optional[float]:
        q = quads_by_slot.get(slot)
        if q is None or len(q) != 4:
            return None
        return sum(p[0] for p in q) / 4.0

    self_cx = centroid_x(slot_label)
    other_cx = centroid_x(other_slot)

    if self_cx is not None and other_cx is not None:
        # Both assigned → simple comparison
        return "left" if self_cx < other_cx else "right"
    if self_cx is not None:
        # Only this slot assigned; infer from image-center heuristic
        # (this case is rare — if the other slot wasn't even assigned
        # we'd have <3 faces total)
        return "left"  # default guess; rarely matters in practice
    if other_cx is not None:
        # This slot UNassigned; infer position as opposite of other slot
        # Determine other's position first, give us the opposite
        # Heuristic: other's centroid relative to image midline
        return "right"  # default; refined by caller if needed
    return None


def _proposer_face_quads(
    image_path: Path,
    side: str,
    hull_guard: bool = True,
    slot_src_filter: bool = False,
    fit_error_fallback: bool = False,
    fit_error_threshold: float = FIT_ERROR_FALLBACK_THRESHOLD,
    processing_image: Optional[Image.Image] = None,
) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict]:
    """Run analyze_image on raw bytes, pick best FaceGrid per center_face,
    convert 3x3 sticker centers → 4-point face quads, then RE-KEY the
    output to match the geometry-labeler convention that
    `identify_faces_jointly` expects (U/R/F on side A, D/L/B on side B).

    Why re-keying is necessary (Devin #152 audit caught this):
    `identify_faces_jointly` internally hardcodes
    `expected_a = ["R", "F"]` and `expected_b = ["L", "B"]` and treats
    those as literal quad-dict keys. Quads whose key isn't in that set
    are silently dropped at `_sample_multisets` (`if label not in
    quads: continue`). Without re-keying, any side A quad that
    analyze_image classified as L/B/D (Set 23-style yaw2 photos, or
    orange↔red center confusion from PR #150's diagnostic) gets
    dropped — making the eval understate the geometry that's actually
    available.

    Re-key strategy: trust analyze_image to identify the U/D anchor
    (white/yellow centers are visually distinctive and rarely confused
    by the canonical classifier). Take the best non-anchor grids by
    quality and re-key them as the geometry-labeler's side-face
    placeholders (R/F on A, L/B on B) in arbitrary order; joint
    face-ID's 16-config yaw enumeration figures out which physical
    face each really is.

    Returns (face_quads_by_label, debug_info). The label keys after
    re-key are a subset of {U, R, F} for side A or {D, L, B} for B.
    """
    assert side in ("A", "B")
    anchor_label = "U" if side == "A" else "D"
    side_face_labels = ("R", "F") if side == "A" else ("L", "B")

    image_bytes = image_path.read_bytes()
    analysis = analyze_image(image_bytes)

    # Hull guard: compute rembg cube hull, reject grids whose sticker
    # centers don't sit (mostly) within it. This addresses the
    # "catastrophic-grid" failure mode identified post-#152: analyze_image
    # sometimes returns 3x3 grids that span multiple physical faces of
    # the cube (or include non-sticker positions), producing valid-looking
    # FaceGrids whose extrapolated face quads cover the wrong region.
    # Rejecting these before rectification eliminates the bimodal
    # "perfect-or-garbage" per-face distribution seen on the pre-guard
    # eval (see PR description for the data).
    hull: List[Tuple[float, float]] = []
    grids_rejected_by_hull: List[Dict] = []
    if hull_guard and processing_image is not None:
        hull = _rembg_cube_hull(processing_image)

    def _grid_passes_guard(grid) -> Tuple[bool, int]:
        if not hull_guard or not hull:
            return True, 9
        inside = _grid_inside_hull_count(grid, hull)
        return inside >= HULL_GUARD_INSIDE_MIN, inside

    # Slot/src filter: reject any grid whose center_face is incompatible
    # with the side's expected face set (the observation from the
    # /tmp/hybrid_overlays/ visual review: bad rectifications correlate
    # strongly with `slot != src` — i.e. `analyze_image` classified the
    # grid's center as a color that shouldn't be visible on this side).
    # Canonical-yaw cubes only show {U, R, F} on side A and {D, L, B} on
    # side B. A grid with center_face outside that set is either:
    #   (a) a multi-face span where analyze_image's center-color
    #       classifier got fooled by a wrong-side sticker that
    #       happened to land on the 3x3 center
    #   (b) a yaw-rotated photo (rare; Set 23 was the canonical example).
    # Hard-rejecting these costs the yaw-rotated case for the (likely)
    # win on catastrophic-grid filtering. Opt-in via the flag.
    grids_rejected_by_slot_src: List[Dict] = []
    expected_for_side = SLOT_SRC_EXPECTED_BY_SIDE[side]

    # First pass: partition grids into "preferred" (slot/src matches expected
    # set on this side, when filter is enabled) and "deferred" (slot/src
    # mismatches — would be rejected by the hard filter, but kept as a
    # fallback pool in case the preferred set leaves a side with <3
    # faces, which happens on yaw-rotated photos like Set 23 where the
    # "wrong-side" grids ARE the correct ones).
    grids_by_face: Dict[str, list] = {}
    deferred_by_face: Dict[str, list] = {}
    for grid in analysis.grids:
        # Apply hull-guard first (independent of slot/src logic)
        ok, inside = _grid_passes_guard(grid)
        if not ok:
            grids_rejected_by_hull.append({
                "centerFace": grid.center_face,
                "matchedCount": grid.matched_count,
                "fitError": round(grid.fit_error, 2),
                "insideCount": inside,
            })
            continue
        # Then slot/src partition
        if slot_src_filter and grid.center_face not in expected_for_side:
            deferred_by_face.setdefault(grid.center_face, []).append(grid)
            continue
        grids_by_face.setdefault(grid.center_face, []).append(grid)

    # Fallback: if slot/src filter left this side with fewer than 3 distinct
    # face buckets (i.e. we'd struggle to fill anchor + 2 side faces),
    # promote the best deferred grids back into the pool until we have 3.
    # This preserves the soft-filter benefit (prefer slot==src when
    # plentiful) while gracefully handling yaw-rotated cases.
    promoted_from_deferred: List[Dict] = []  # exposed in proposer_debug
    if slot_src_filter:
        # Sort deferred buckets by their best grid's quality (matched desc, fit asc)
        deferred_sorted = sorted(
            deferred_by_face.items(),
            key=lambda kv: (-max(g.matched_count for g in kv[1]),
                            min(g.fit_error for g in kv[1])),
        )
        for face, candidates in deferred_sorted:
            if len(grids_by_face) >= 3:
                break
            best = min(candidates, key=lambda g: (-g.matched_count, g.fit_error))
            grids_by_face.setdefault(face, []).append(best)
            promoted_from_deferred.append({
                "centerFace": face,
                "matchedCount": best.matched_count,
                "fitError": round(best.fit_error, 2),
                "reason": "preferred_pool_<3_faces",
            })
        grids_rejected_by_slot_src = [
            {
                "centerFace": face,
                "matchedCount": g.matched_count,
                "fitError": round(g.fit_error, 2),
                "expectedSet": sorted(expected_for_side),
            }
            for face, candidates in deferred_by_face.items()
            for g in candidates
            if face not in grids_by_face
        ]

    # Best grid per center_face (analyze_image's color classification),
    # AFTER hull guard filtering
    best_per_face: List[Tuple[str, object]] = []
    for face, candidates in grids_by_face.items():
        best = min(candidates, key=lambda g: (-g.matched_count, g.fit_error))
        best_per_face.append((face, best))

    face_quads: Dict[str, List[Tuple[float, float]]] = {}
    selected_metrics: Dict[str, Dict] = {}

    # Anchor (U on side A, D on side B): take the grid whose center_face
    # matches the anchor letter if any exist. If none, the side degrades
    # to <3 faces and joint face-ID handles the missing anchor.
    anchor_grids = [(f, g) for f, g in best_per_face if f == anchor_label]
    if anchor_grids:
        face, grid = min(anchor_grids, key=lambda fg: (-fg[1].matched_count, fg[1].fit_error))
        quad = _face_quad_from_grid_centers(grid.points)
        if quad is not None:
            face_quads[anchor_label] = [(float(x), float(y)) for (x, y) in quad]
            selected_metrics[anchor_label] = {
                "sourceCenterFace": face,
                "matchedCount": grid.matched_count,
                "fitError": round(grid.fit_error, 2),
                "cubeHullInside": grid.cube_hull_inside_count,
            }

    # Side faces: take the best 2 non-anchor grids by quality, re-key as
    # the placeholder side-face labels. The assignment to R vs F (or L vs B)
    # is arbitrary because joint face-ID's yaw enumeration resolves both.
    non_anchor = sorted(
        [(f, g) for f, g in best_per_face if f != anchor_label],
        key=lambda fg: (-fg[1].matched_count, fg[1].fit_error),
    )
    for slot_idx, (face, grid) in enumerate(non_anchor[:2]):
        quad = _face_quad_from_grid_centers(grid.points)
        if quad is None:
            continue
        slot_label = side_face_labels[slot_idx]
        face_quads[slot_label] = [(float(x), float(y)) for (x, y) in quad]
        selected_metrics[slot_label] = {
            "sourceCenterFace": face,
            "matchedCount": grid.matched_count,
            "fitError": round(grid.fit_error, 2),
            "cubeHullInside": grid.cube_hull_inside_count,
            "rekeyedFrom": face,
        }

    # Fit-error fallback (topology-aware): for any slot whose
    # underlying grid has fit_error > threshold, replace its face quad
    # with one derived from the cube-face TOPOLOGY using:
    #   * Shared corners from trusted neighbor faces (precision-preserving)
    #   * Only corners unique to this face come from the rembg hexagon
    #
    # The geometric topology: cube faces share specific corners (cube
    # center vertex + 1 hexagon vertex per adjacent face). When 2 of 3
    # face quads are trusted, only 1 of the bad face's 4 corners needs
    # to come from the hexagon approximation — the other 3 come from
    # precise trusted-quad corners.
    fallback_actions: List[Dict] = []
    if fit_error_fallback and processing_image is not None:
        hexagon = _rembg_hexagon(processing_image)
        # Also fetch the full rembg cube hull (51+ vertices) for clip-to-hull
        # and for full-hull angular lookup of unique-to-face vertices when
        # the 6-vertex hexagon is degenerate (collapsed/missing vertices on
        # yawed cubes — see Set 17 A diagnosis).
        try:
            from rembg import remove
            rgba = remove(processing_image, session=_get_rembg_session("u2net"))
            mask_arr = np.array(rgba.split()[-1], dtype=np.uint8) > 128
            full_cube_hull = _hull_from_mask(mask_arr) if mask_arr.any() else None
        except Exception:
            full_cube_hull = None
        if hexagon is not None:
            # Determine slot positions (top/left/right) by centroid x
            # of currently-assigned quads. Anchor slot always = top.
            slot_positions: Dict[str, str] = {anchor_label: "top"}
            non_anchor_slots = [s for s in face_quads if s != anchor_label]
            if len(non_anchor_slots) == 2:
                a, b = non_anchor_slots
                cx_a = sum(p[0] for p in face_quads[a]) / 4.0
                cx_b = sum(p[0] for p in face_quads[b]) / 4.0
                if cx_a < cx_b:
                    slot_positions[a] = "left"
                    slot_positions[b] = "right"
                else:
                    slot_positions[a] = "right"
                    slot_positions[b] = "left"
            elif len(non_anchor_slots) == 1:
                # Only 1 side-face slot assigned; use image-midline heuristic
                slot = non_anchor_slots[0]
                cx = sum(p[0] for p in face_quads[slot]) / 4.0
                img_mid = processing_image.width / 2.0
                slot_positions[slot] = "left" if cx < img_mid else "right"

            # Identify trusted slots: assigned, position-mapped, fit_error ok
            trusted_quads_by_position: Dict[str, List[Tuple[float, float]]] = {}
            for slot, position in slot_positions.items():
                if slot not in face_quads:
                    continue
                meta = selected_metrics.get(slot, {})
                fit_err = meta.get("fitError")
                if fit_err is not None and fit_err <= fit_error_threshold:
                    trusted_quads_by_position[position] = face_quads[slot]

            # For each untrusted slot (fit_err > threshold), derive its
            # quad from the trusted neighbors + hexagon
            for slot_label in list(face_quads.keys()):
                meta = selected_metrics.get(slot_label, {})
                fit_err = meta.get("fitError")
                if fit_err is None or fit_err <= fit_error_threshold:
                    continue
                slot_position = slot_positions.get(slot_label)
                if slot_position is None:
                    continue
                derived = _derive_face_quad_topology_aware(
                    slot_position, trusted_quads_by_position, hexagon,
                    cube_hull=full_cube_hull,
                )
                if derived is None or any(p is None for p in derived):
                    continue
                derived = [(float(x), float(y)) for (x, y) in derived]
                fallback_actions.append({
                    "slot": slot_label,
                    "originalSourceCenterFace": meta.get("sourceCenterFace"),
                    "originalFitError": fit_err,
                    "fallbackPosition": slot_position,
                    "trustedNeighborsAvailable": sorted(trusted_quads_by_position.keys()),
                    "cornersFromTrustedNeighbors": sum(
                        1 for p_name in {
                            "top": ["h1_via_right", "center", "h5_via_left"],
                            "right": ["h1_via_top", "h3_via_left", "center"],
                            "left": ["h5_via_top", "h3_via_right", "center"],
                        }[slot_position]
                        if (p_name.endswith("_via_top") and "top" in trusted_quads_by_position)
                        or (p_name.endswith("_via_right") and "right" in trusted_quads_by_position)
                        or (p_name.endswith("_via_left") and "left" in trusted_quads_by_position)
                        or (p_name == "center" and trusted_quads_by_position)
                    ),
                    "thresholdUsed": fit_error_threshold,
                })
                face_quads[slot_label] = derived
                selected_metrics[slot_label] = {
                    **meta,
                    "replacedByTopologyFallback": True,
                    "fallbackPosition": slot_position,
                }

    return face_quads, {
        "stickerCount": len(analysis.stickers),
        "gridCount": len(analysis.grids),
        "side": side,
        "anchorFound": anchor_label in face_quads,
        "selectedPerFace": selected_metrics,
        "facesProposedAfterRekey": sorted(face_quads.keys()),
        "hullGuardEnabled": hull_guard and bool(hull),
        "gridsRejectedByHullGuard": grids_rejected_by_hull,
        "slotSrcFilterEnabled": slot_src_filter,
        "gridsRejectedBySlotSrc": grids_rejected_by_slot_src,
        "gridsPromotedFromDeferred": promoted_from_deferred,
        "fitErrorFallbackEnabled": fit_error_fallback,
        "fitErrorFallbackThreshold": fit_error_threshold if fit_error_fallback else None,
        "fitErrorFallbackActions": fallback_actions,
        "warnings": list(analysis.warnings),
    }


def _classify_face_aligned(face_img: Image.Image, gt_colors: List[str]):
    """Sample 9 stickers from rectified face, classify with the env-selected
    classifier mode, align via discover_orientation against gt_colors.
    Returns (correct_count, aligned_classified, rgbs_aligned)."""
    samples = extract_stickers_from_rectified(face_img)
    rgbs = [s.rgb for row in samples for s in row]
    classified = [s.classified_color for row in samples for s in row]
    mirror, rot, _ = discover_orientation(rgbs, gt_colors)
    aligned = apply_orientation(classified, mirror, rot)
    rgbs_aligned = apply_orientation(rgbs, mirror, rot)
    correct = sum(1 for c, g in zip(aligned, gt_colors) if c == g)
    return correct, aligned, rgbs_aligned


def evaluate_pair(
    set_id: str, image_a: Path, image_b: Path, gt_state: str,
    hull_guard: bool = True,
    slot_src_filter: bool = False,
    fit_error_fallback: bool = False,
    fit_error_threshold: float = FIT_ERROR_FALLBACK_THRESHOLD,
) -> Dict:
    """One pair: analyze_image-quads → rectify → classify → joint face-ID
    → assemble 54-state → compare to GT."""
    images: Dict[str, Image.Image] = {}
    arrs: Dict[str, np.ndarray] = {}
    proposer_quads: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    proposer_debug: Dict[str, Dict] = {}
    for side, image_path in (("A", image_a), ("B", image_b)):
        try:
            img, arr = _load_processing_image(image_path)
            images[side] = img
            arrs[side] = arr
        except Exception as e:
            return {"setId": set_id, "error": f"load {side}: {type(e).__name__}: {e}"}
        try:
            quads, debug = _proposer_face_quads(
                image_path, side,
                hull_guard=hull_guard,
                slot_src_filter=slot_src_filter,
                fit_error_fallback=fit_error_fallback,
                fit_error_threshold=fit_error_threshold,
                processing_image=img,
            )
        except Exception as e:
            return {"setId": set_id, "error": f"proposer {side}: {type(e).__name__}: {e}"}
        proposer_quads[side] = quads
        proposer_debug[side] = debug

    # Joint A+B face-ID using analyze_image's auto-proposed quads.
    # The function takes "expected" faces per side (URF for A, DLB for B)
    # so it can multiset-match against the GT's 6 face centers.
    prepared = {
        side: {
            "arr": arrs[side],
            "quads": proposer_quads[side],
            "expected": EXPECTED_FACES_BY_SIDE[side],
        }
        for side in ("A", "B")
    }
    label_to_true, joint_score, joint_status = identify_faces_jointly(
        prepared, gt_state, inset=0.20
    )

    per_face_aligned: Dict[str, List[str]] = {}
    per_face_metrics: List[Dict] = []
    stickers_sampled = 0
    stickers_correct = 0
    for side in ("A", "B"):
        mapping = label_to_true.get(side, {})
        for label_face, true_face in mapping.items():
            quad = proposer_quads[side].get(label_face)
            if quad is None or len(quad) != 4:
                continue
            gt_colors = face_colors_from_state(gt_state, true_face)
            try:
                rectified = rectify_face(images[side], quad,
                                         output_size=DEFAULT_FACE_SIZE)
            except Exception as e:
                per_face_metrics.append({
                    "side": side, "labelFace": label_face,
                    "trueFace": true_face,
                    "error": f"rectify: {type(e).__name__}: {e}",
                })
                continue
            try:
                correct, aligned, _ = _classify_face_aligned(rectified, gt_colors)
            except Exception as e:
                per_face_metrics.append({
                    "side": side, "labelFace": label_face,
                    "trueFace": true_face,
                    "error": f"classify: {type(e).__name__}: {e}",
                })
                continue
            stickers_sampled += 9
            stickers_correct += correct
            per_face_aligned[true_face] = aligned
            per_face_metrics.append({
                "side": side, "labelFace": label_face,
                "trueFace": true_face,
                "correct": correct, "ofTotal": 9,
            })

    # Assemble 54-state in URFDLB order
    assembled: Optional[str] = None
    if all(face in per_face_aligned for face in FACE_ORDER):
        chunks: List[str] = []
        for face in FACE_ORDER:
            colors = per_face_aligned[face]
            chunks.append("".join(COLOR_TO_FACE[c] for c in colors))
        assembled = "".join(chunks)

    exact_match = (assembled is not None and assembled == gt_state)
    sticker_matches_assembled = None
    if assembled is not None and len(gt_state) == 54:
        sticker_matches_assembled = sum(
            1 for a, g in zip(assembled, gt_state) if a == g
        )

    return {
        "setId": set_id,
        "isOOD": set_id in OOD_SETS,
        "stickersSampled": stickers_sampled,
        "stickersCorrect": stickers_correct,
        "perStickerAccuracy":
            round(stickers_correct / stickers_sampled, 4)
            if stickers_sampled else None,
        "facesRecovered": len(per_face_aligned),
        "facesExpected": 6,
        "facesProposedA": proposer_debug["A"]["facesProposedAfterRekey"],
        "facesProposedB": proposer_debug["B"]["facesProposedAfterRekey"],
        "anchorFoundA": proposer_debug["A"]["anchorFound"],
        "anchorFoundB": proposer_debug["B"]["anchorFound"],
        "hullGuardA": {
            "enabled": proposer_debug["A"].get("hullGuardEnabled"),
            "gridsRejected": len(proposer_debug["A"].get("gridsRejectedByHullGuard", [])),
            "gridsAccepted": proposer_debug["A"].get("gridCount", 0)
                - len(proposer_debug["A"].get("gridsRejectedByHullGuard", [])),
        },
        "hullGuardB": {
            "enabled": proposer_debug["B"].get("hullGuardEnabled"),
            "gridsRejected": len(proposer_debug["B"].get("gridsRejectedByHullGuard", [])),
            "gridsAccepted": proposer_debug["B"].get("gridCount", 0)
                - len(proposer_debug["B"].get("gridsRejectedByHullGuard", [])),
        },
        "slotSrcFilterA": {
            "enabled": proposer_debug["A"].get("slotSrcFilterEnabled"),
            "gridsRejected": len(proposer_debug["A"].get("gridsRejectedBySlotSrc", [])),
            "gridsPromotedFromDeferred": len(proposer_debug["A"].get("gridsPromotedFromDeferred", [])),
            "rejectedDetails": proposer_debug["A"].get("gridsRejectedBySlotSrc", []),
            "promotedDetails": proposer_debug["A"].get("gridsPromotedFromDeferred", []),
        },
        "slotSrcFilterB": {
            "enabled": proposer_debug["B"].get("slotSrcFilterEnabled"),
            "gridsRejected": len(proposer_debug["B"].get("gridsRejectedBySlotSrc", [])),
            "gridsPromotedFromDeferred": len(proposer_debug["B"].get("gridsPromotedFromDeferred", [])),
            "rejectedDetails": proposer_debug["B"].get("gridsRejectedBySlotSrc", []),
            "promotedDetails": proposer_debug["B"].get("gridsPromotedFromDeferred", []),
        },
        "jointStatus": joint_status,
        "jointScore": round(joint_score, 4) if joint_score is not None else None,
        "assembledState": assembled,
        "exactMatch": exact_match,
        "perStickerMatchesAssembled": sticker_matches_assembled,
        "perFace": per_face_metrics,
    }


def discover_pairs() -> List[Tuple[str, Path, Path, str]]:
    tasks = load_corpus_tasks(CORPUS_MANIFEST)
    tasks.extend(discover_additional_tasks({t.set_id for t in tasks}))
    out: List[Tuple[str, Path, Path, str]] = []
    for task in tasks:
        if not (task.image_a.exists() and task.image_b.exists()):
            continue
        try:
            gt = parse_ground_truth(task.ground_truth)
        except Exception:
            continue
        out.append((task.set_id, task.image_a, task.image_b, gt))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only-sets", nargs="*", default=None)
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument(
        "--no-hull-guard", action="store_true",
        help="Disable the rembg-cube-hull validation that rejects "
             "analyze_image grids whose sticker centers fall outside "
             "the cube silhouette (default: guard enabled). Useful for "
             "A/B comparison against the pre-guard baseline.",
    )
    ap.add_argument(
        "--fit-error-fallback", action="store_true",
        help="Trust-fit-or-derive hybrid (proposal A): for any slot whose "
             "underlying analyze_image grid has fit_error > threshold, "
             "replace the extrapolated face quad with one derived "
             "geometrically from the rembg hexagon hull (via the Geometry "
             "Labeler's template formula). Targets the catastrophic-grid "
             "failure mode on cubes where 1-2 of 3 faces fit cleanly but the "
             "third doesn't.",
    )
    ap.add_argument(
        "--fit-error-threshold", type=float, default=FIT_ERROR_FALLBACK_THRESHOLD,
        help=f"Fit-error threshold for --fit-error-fallback (default: "
             f"{FIT_ERROR_FALLBACK_THRESHOLD}). Calibrated from Set 17 A "
             f"diagnostic where good grids had fit_error ≤ 2 and the "
             f"catastrophic one had 6.5.",
    )
    ap.add_argument(
        "--slot-src-filter", action="store_true",
        help="Reject any grid whose analyze_image center_face is "
             "outside the side's canonical expected set ({U,R,F} on A, "
             "{D,L,B} on B). Targets the catastrophic-grid failure mode "
             "where analyze_image's center classifier got fooled by a "
             "wrong-side sticker on a multi-face span. Costs the rare "
             "yaw-rotated case (Set 23-style).",
    )
    args = ap.parse_args()

    pairs = discover_pairs()
    if args.only_sets:
        wanted = set(args.only_sets)
        pairs = [p for p in pairs if p[0] in wanted]

    hull_guard = not args.no_hull_guard
    slot_src_filter = args.slot_src_filter
    fit_error_fallback = args.fit_error_fallback
    fit_error_threshold = args.fit_error_threshold
    classifier_mode = os.environ.get("CUBE_RECOGNIZER_CLASSIFIER", "canonical")
    print(f"evaluating hybrid pipeline on {len(pairs)} pairs "
          f"(classifier={classifier_mode}, hull_guard={hull_guard}, "
          f"slot_src_filter={slot_src_filter}, "
          f"fit_error_fallback={fit_error_fallback}"
          f"{f' threshold={fit_error_threshold}' if fit_error_fallback else ''})",
          file=sys.stderr)
    print("", file=sys.stderr)

    rows: List[Dict] = []
    for i, (set_id, image_a, image_b, gt) in enumerate(pairs, 1):
        try:
            row = evaluate_pair(
                set_id, image_a, image_b, gt,
                hull_guard=hull_guard,
                slot_src_filter=slot_src_filter,
                fit_error_fallback=fit_error_fallback,
                fit_error_threshold=fit_error_threshold,
            )
        except Exception as e:
            row = {"setId": set_id, "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        if "error" in row:
            print(f"  [{i:>2}/{len(pairs)}] set {set_id}: ERROR {row['error']}",
                  file=sys.stderr, flush=True)
        else:
            flag = " [OOD]" if row.get("isOOD") else ""
            exact = "EXACT" if row["exactMatch"] else "diff"
            psa = row["perStickerAccuracy"]
            stk = row.get("perStickerMatchesAssembled")
            stk_str = f"{stk}/54" if stk is not None else "n/a"
            print(
                f"  [{i:>2}/{len(pairs)}] set {set_id}: "
                f"faces={row['facesRecovered']}/6  "
                f"perSticker(rect)={psa}  state={exact} ({stk_str}){flag}",
                file=sys.stderr, flush=True,
            )

    valid = [r for r in rows if "error" not in r]
    total_sampled = sum(r["stickersSampled"] for r in valid)
    total_correct = sum(r["stickersCorrect"] for r in valid)
    rect_accuracy = total_correct / max(1, total_sampled)

    pairs_assembled = [r for r in valid if r["assembledState"] is not None]
    pairs_exact = [r for r in valid if r["exactMatch"]]
    pairs_failed_to_assemble = [r for r in valid if r["assembledState"] is None]

    # Per-sticker accuracy MEASURED ON ASSEMBLED 54-STATE (not just on
    # the rectified-face samples). This catches cases where joint
    # face-ID picked the wrong mapping but per-face classification
    # was internally consistent.
    assembled_stickers_total = sum(1 for r in pairs_assembled) * 54
    assembled_stickers_correct = sum(r["perStickerMatchesAssembled"]
                                     for r in pairs_assembled)
    assembled_accuracy = (
        assembled_stickers_correct / max(1, assembled_stickers_total)
    )

    ood = [r for r in valid if r.get("isOOD")]
    ood_assembled = [r for r in ood if r["assembledState"] is not None]
    ood_exact = sum(1 for r in ood if r["exactMatch"])
    ood_sticker_total = sum(1 for r in ood_assembled) * 54
    ood_sticker_correct = sum(r["perStickerMatchesAssembled"]
                              for r in ood_assembled)

    non_ood = [r for r in valid if not r.get("isOOD")]
    non_ood_assembled = [r for r in non_ood if r["assembledState"] is not None]
    non_ood_exact = sum(1 for r in non_ood if r["exactMatch"])
    non_ood_sticker_total = sum(1 for r in non_ood_assembled) * 54
    non_ood_sticker_correct = sum(r["perStickerMatchesAssembled"]
                                  for r in non_ood_assembled)

    summary_lines: List[str] = []
    summary_lines.append(
        f"Hybrid pipeline evaluation: {len(pairs)} pairs "
        f"(classifier={classifier_mode})"
    )
    summary_lines.append("")
    summary_lines.append("Stages: analyze_image → grids → face_quads → "
                         "rectify → knn5_lab_full → joint face-ID → assemble")
    summary_lines.append("")
    summary_lines.append("Pair outcomes:")
    summary_lines.append(f"  total:                {len(valid)}")
    summary_lines.append(f"  exact 54-state:       {len(pairs_exact)}  "
                         f"({len(pairs_exact)*100/max(1,len(valid)):.1f}%)")
    summary_lines.append(f"  assembled (not exact):"
                         f" {len(pairs_assembled) - len(pairs_exact)}")
    summary_lines.append(f"  failed to assemble:   {len(pairs_failed_to_assemble)}")
    summary_lines.append("")
    summary_lines.append("Per-sticker accuracy:")
    summary_lines.append(
        f"  on rectified faces (only): {rect_accuracy:.4f} "
        f"({total_correct}/{total_sampled})"
    )
    summary_lines.append(
        f"  on assembled 54-state:     {assembled_accuracy:.4f} "
        f"({assembled_stickers_correct}/{assembled_stickers_total})"
    )
    summary_lines.append("")
    summary_lines.append("OOD-set breakdown (Sets 57/58/61/62):")
    if ood:
        ood_rect_acc = sum(r["stickersCorrect"] for r in ood) / max(
            1, sum(r["stickersSampled"] for r in ood)
        )
        ood_assembled_acc = (
            ood_sticker_correct / max(1, ood_sticker_total)
            if ood_sticker_total else None
        )
        summary_lines.append(
            f"  pairs: {len(ood)}, exact: {ood_exact}, "
            f"assembled: {len(ood_assembled)}"
        )
        summary_lines.append(
            f"  rect accuracy:      {ood_rect_acc:.4f}"
        )
        if ood_assembled_acc is not None:
            summary_lines.append(
                f"  assembled accuracy: {ood_assembled_acc:.4f}"
            )
    summary_lines.append("")
    summary_lines.append("Non-OOD breakdown (28 pairs):")
    if non_ood:
        non_ood_rect_acc = sum(r["stickersCorrect"] for r in non_ood) / max(
            1, sum(r["stickersSampled"] for r in non_ood)
        )
        non_ood_assembled_acc = (
            non_ood_sticker_correct / max(1, non_ood_sticker_total)
            if non_ood_sticker_total else None
        )
        summary_lines.append(
            f"  pairs: {len(non_ood)}, exact: {non_ood_exact}, "
            f"assembled: {len(non_ood_assembled)}"
        )
        summary_lines.append(
            f"  rect accuracy:      {non_ood_rect_acc:.4f}"
        )
        if non_ood_assembled_acc is not None:
            summary_lines.append(
                f"  assembled accuracy: {non_ood_assembled_acc:.4f}"
            )
    summary_lines.append("")

    summary_lines.append("Comparison to known baselines:")
    summary_lines.append(
        "  rectified-from-human-quads + knn5_lab_full (PR #150 A/B): "
        "0.9929 per-sticker"
    )
    summary_lines.append(
        "  existing recognizer (WhiteUpRecognizer):                   "
        "~0.827 per-sticker (from #139 sweep)"
    )
    summary_lines.append(
        "  mask pipeline (rembg → optimized hexagon):                 "
        "~0.615 per-sticker (from #139 sweep)"
    )
    summary_lines.append("")

    # Worst pairs by per-sticker (rect) accuracy
    sorted_pairs = sorted(
        valid, key=lambda r: r.get("perStickerAccuracy") or 0
    )
    summary_lines.append("Worst 10 pairs by rectified per-sticker accuracy:")
    for r in sorted_pairs[:10]:
        flag = " [OOD]" if r.get("isOOD") else ""
        psa = r.get("perStickerAccuracy")
        faces = r["facesRecovered"]
        exact = "EXACT" if r["exactMatch"] else "."
        summary_lines.append(
            f"  set {r['setId']}: rect={psa} faces={faces}/6 "
            f"state={exact}{flag}"
        )

    # Failed-to-assemble pairs deserve a callout — what went wrong?
    if pairs_failed_to_assemble:
        summary_lines.append("")
        summary_lines.append(
            f"Pairs that failed to assemble (joint face-ID couldn't "
            f"recover 6 faces): {len(pairs_failed_to_assemble)}"
        )
        for r in pairs_failed_to_assemble[:10]:
            flag = " [OOD]" if r.get("isOOD") else ""
            summary_lines.append(
                f"  set {r['setId']}: faces={r['facesRecovered']}/6, "
                f"proposedA={r['facesProposedA']}, "
                f"proposedB={r['facesProposedB']}, "
                f"joint={r['jointStatus']}{flag}"
            )

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rows, indent=2))
    Path(args.summary).write_text("\n".join(summary_lines) + "\n")

    print("", file=sys.stderr)
    print("\n".join(summary_lines), file=sys.stderr)
    print(f"\nwrote {args.report}", file=sys.stderr)
    print(f"wrote {args.summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
