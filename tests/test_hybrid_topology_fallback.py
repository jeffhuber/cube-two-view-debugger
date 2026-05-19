"""Unit tests for the topology-aware fit-error fallback in
evaluate_hybrid_pipeline.

The fallback replaces a face quad whose underlying analyze_image grid
has fit_error > threshold with a quad derived from the cube-face
TOPOLOGY:

  * The 3 visible faces in iso-projection share specific corners
    (every pair shares 1 hexagon vertex; all 3 share the cube center).
  * When 1-2 neighbor faces are trustworthy, their shared corners are
    already known with the precision of those trusted quads.
  * Only corners unique to the untrusted face need approximation, and
    even those use the full rembg hull (51+ vertices) rather than the
    often-degenerate 6-vertex hexagon fit.

Calibration: Set 17 A diagnostic — U fit_error=0.34, R fit_error=2.01,
B fit_error=6.50. Threshold around 3-5 cleanly separates the spatially
coherent grids from the multi-face spans.

Outcome on Set 17 A:
    Baseline                   : 30/54
    Cardinal + parallelogram   : 34/54
    + clip-to-hull + full-hull : 37/54 (side A perfect 27/27)

Outcome on the full 33-pair corpus:
    Baseline (hull-guard only)    : 0.6532 assembled per-sticker
    Topology fallback (this PR)   : 0.7970 assembled per-sticker (+14.4pp)
    Exact 54-state pairs          : 0 → 2 (first ever non-zero)
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from tools import evaluate_hybrid_pipeline  # noqa: E402
from tools.evaluate_hybrid_pipeline import (  # noqa: E402
    _cardinal_corners,
    _clip_to_hull,
    _derive_face_quad_topology_aware,
    _hull_vertex_in_direction,
)


# ---------------------------------------------------------------------------
# _cardinal_corners — corner identification by image-space cardinal direction
# ---------------------------------------------------------------------------

def test_cardinal_corners_basic_parallelogram():
    """A parallelogram aligned roughly with image axes: each cardinal
    direction picks out the extreme corner in that direction."""
    # Quad in CW order from top: top → right → bottom → left
    quad = [(100, 50), (180, 100), (130, 180), (50, 130)]
    cardinals = _cardinal_corners(quad)
    assert cardinals["N"] == (100, 50)
    assert cardinals["E"] == (180, 100)
    assert cardinals["S"] == (130, 180)
    assert cardinals["W"] == (50, 130)


def test_cardinal_corners_invariant_to_input_order():
    """The mapping depends only on coordinates, not on input order.
    This is the whole point: canonical_corner_order's start-index is
    ambiguous, so cardinal-position lookup is the disambiguator."""
    quad_cw = [(100, 50), (180, 100), (130, 180), (50, 130)]
    quad_ccw = list(reversed(quad_cw))
    quad_shuffled = [quad_cw[2], quad_cw[0], quad_cw[3], quad_cw[1]]
    assert (
        _cardinal_corners(quad_cw)
        == _cardinal_corners(quad_ccw)
        == _cardinal_corners(quad_shuffled)
    )


# ---------------------------------------------------------------------------
# _clip_to_hull — projects out-of-hull points to nearest hull edge
# ---------------------------------------------------------------------------

def test_clip_to_hull_inside_passes_through():
    """Points inside the hull are returned unchanged."""
    hull = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert _clip_to_hull((50, 50), hull) == (50, 50)
    assert _clip_to_hull((10, 90), hull) == (10, 90)


def test_clip_to_hull_outside_clipped_to_edge():
    """Points outside the hull are projected to the nearest edge POINT
    (not the nearest vertex). For a square hull, a point directly to
    the right of the middle clips to the right-edge midpoint."""
    hull = [(0, 0), (100, 0), (100, 100), (0, 100)]
    out = _clip_to_hull((150, 50), hull)
    assert abs(out[0] - 100) < 1e-6
    assert abs(out[1] - 50) < 1e-6


def test_clip_to_hull_outside_corner_clipped_to_nearest_vertex():
    """A point beyond a corner clips to that corner."""
    hull = [(0, 0), (100, 0), (100, 100), (0, 100)]
    out = _clip_to_hull((150, 150), hull)
    assert abs(out[0] - 100) < 1e-6
    assert abs(out[1] - 100) < 1e-6


def test_clip_to_hull_empty_hull_returns_input():
    """Defensive: empty or degenerate hulls return the point unchanged
    rather than crashing the pipeline."""
    assert _clip_to_hull((50, 50), []) == (50, 50)
    assert _clip_to_hull((50, 50), [(10, 10)]) == (50, 50)
    assert _clip_to_hull((50, 50), [(10, 10), (20, 20)]) == (50, 50)


# ---------------------------------------------------------------------------
# _hull_vertex_in_direction — angular lookup on the full hull
# ---------------------------------------------------------------------------

def test_hull_vertex_in_direction_finds_north():
    """Image coords: -π/2 is straight up. Should return the top vertex."""
    hull = [(50, 10), (90, 50), (50, 90), (10, 50)]
    north = _hull_vertex_in_direction(hull, (50, 50), -math.pi / 2)
    assert north == (50, 10)


def test_hull_vertex_in_direction_finds_east():
    """Image coords: 0 is right (east)."""
    hull = [(50, 10), (90, 50), (50, 90), (10, 50)]
    east = _hull_vertex_in_direction(hull, (50, 50), 0.0)
    assert east == (90, 50)


def test_hull_vertex_in_direction_returns_none_outside_tolerance():
    """No vertex within the angular tolerance → None. Caller falls back
    to the degenerate hexagon vertex."""
    hull = [(50, 10), (90, 50), (50, 90)]  # missing west vertex
    west = _hull_vertex_in_direction(
        hull, (50, 50), math.pi, angle_tolerance_radians=0.1,
    )
    assert west is None


def test_hull_vertex_in_direction_picks_farthest_in_sector():
    """When multiple vertices fall in the angular tolerance window,
    the one farthest from center wins — this is what makes the lookup
    pick true extreme points instead of nearby interior bumps."""
    # Two vertices roughly north of (50, 50); the farther one wins.
    hull = [(50, 20), (50, 10), (90, 50)]
    north = _hull_vertex_in_direction(hull, (50, 50), -math.pi / 2)
    assert north == (50, 10)


# ---------------------------------------------------------------------------
# _derive_face_quad_topology_aware — the core fallback logic
# ---------------------------------------------------------------------------

def _make_quad_with_cardinals(n, e, s, w):
    """Construct a 4-corner quad in canonical CW-from-N order where
    each input corner is at the expected cardinal position. Inputs
    must satisfy n.y < e.y, w.y, s.y; s.y > others; etc. — i.e.,
    they ARE the strict-extreme corners of the quad in each direction.

    This lets tests bypass the symmetric-iso ambiguity where center
    and h3 share an axis: the test constructs each cardinal corner at
    a specific image-space coordinate and the function's cardinal
    lookup unambiguously returns those coordinates back.
    """
    return [n, e, s, w]


# Common fixture: hexagon + cube-center vertex + per-face h3 estimates.
# Key invariant: h3_left.x < center_vertex.x < h3_right.x. This is what
# makes right.W and left.E both unambiguously return the cube center
# vertex rather than tying with h3. In real photos, each face's grid
# fits its h3 corner to slightly different image coordinates anyway,
# so per-quad h3 values are a faithful model of the actual data.
HEX_H0 = (500, 200)   # top
HEX_H1 = (700, 320)   # upper-right
HEX_H2 = (740, 480)   # right-mid
HEX_H4 = (260, 480)   # left-mid
HEX_H5 = (300, 320)   # upper-left
HEX_H3_RIGHT = (520, 700)  # bottom corner as fitted by right face
HEX_H3_LEFT = (480, 700)   # bottom corner as fitted by left face
# The function's hexagon arg is the SINGLE rembg-derived hexagon — its h3
# is a third estimate (and is what the function falls back to when
# parallelogram derivation isn't possible). For these tests, the
# hexagon's h3 doesn't matter to the parallelogram path; pick any.
HEX_FOR_FALLBACK = [
    HEX_H0, HEX_H1, HEX_H2,
    (500, 700),  # h3
    HEX_H4, HEX_H5,
]
CENTER_VERTEX = (500, 470)


def test_derive_top_with_two_trusted_neighbors_uses_parallelogram():
    """When 'right' and 'left' face quads are trusted, the 'top' face's
    derivation pulls h1 (=right.N) and h5 (=left.N) as shared corners,
    and the cube center as the average of right.W and left.E (both
    estimates of the cube-center vertex). The h0 corner (unique to top)
    is then derived via the parallelogram identity h0 = h1 + h5 - center.
    """
    # Right quad cardinals: N=h1, E=h2, S=h3_right, W=center
    right_quad = _make_quad_with_cardinals(
        HEX_H1, HEX_H2, HEX_H3_RIGHT, CENTER_VERTEX,
    )
    # Left quad cardinals: N=h5, E=center, S=h3_left, W=h4
    left_quad = _make_quad_with_cardinals(
        HEX_H5, CENTER_VERTEX, HEX_H3_LEFT, HEX_H4,
    )
    trusted = {"right": right_quad, "left": left_quad}
    derived = _derive_face_quad_topology_aware(
        "top", trusted, HEX_FOR_FALLBACK, cube_hull=None,
    )
    # Returned as [h0, h1_shared, center, h5_shared] in canonical
    # CW-from-N order
    assert len(derived) == 4
    derived_h0, derived_h1, derived_center, derived_h5 = derived
    assert derived_h1 == HEX_H1
    assert derived_h5 == HEX_H5
    # Center is the average of right.W and left.E — both equal CENTER_VERTEX
    # so the averaged center matches exactly
    assert math.isclose(derived_center[0], CENTER_VERTEX[0], abs_tol=1e-6)
    assert math.isclose(derived_center[1], CENTER_VERTEX[1], abs_tol=1e-6)
    # Parallelogram: h0 = h1 + h5 - center
    expected_h0 = (HEX_H1[0] + HEX_H5[0] - CENTER_VERTEX[0],
                   HEX_H1[1] + HEX_H5[1] - CENTER_VERTEX[1])
    assert math.isclose(derived_h0[0], expected_h0[0], abs_tol=1e-6)
    assert math.isclose(derived_h0[1], expected_h0[1], abs_tol=1e-6)


def test_derive_right_with_two_trusted_neighbors_uses_parallelogram():
    """Symmetric case for 'right' slot: h2 is unique-to-face and
    derived from h1 + h3 - center, with h1 from top.E and h3 from
    left.S (since right is not trusted). Center comes from top.S
    and left.E (both estimates of the cube-center vertex)."""
    # Top quad cardinals: N=h0, E=h1, S=center, W=h5
    top_quad = _make_quad_with_cardinals(
        HEX_H0, HEX_H1, CENTER_VERTEX, HEX_H5,
    )
    # Left quad cardinals: N=h5, E=center, S=h3_left, W=h4
    left_quad = _make_quad_with_cardinals(
        HEX_H5, CENTER_VERTEX, HEX_H3_LEFT, HEX_H4,
    )
    trusted = {"top": top_quad, "left": left_quad}
    derived = _derive_face_quad_topology_aware(
        "right", trusted, HEX_FOR_FALLBACK, cube_hull=None,
    )
    # Right canonical: [h1, h2, h3, center]
    assert len(derived) == 4
    derived_h1, derived_h2, derived_h3, derived_center = derived
    assert derived_h1 == HEX_H1
    # right.S not available (right not trusted) → falls back to left.S
    assert derived_h3 == HEX_H3_LEFT
    assert math.isclose(derived_center[0], CENTER_VERTEX[0], abs_tol=1e-6)
    assert math.isclose(derived_center[1], CENTER_VERTEX[1], abs_tol=1e-6)
    expected_h2 = (HEX_H1[0] + HEX_H3_LEFT[0] - CENTER_VERTEX[0],
                   HEX_H1[1] + HEX_H3_LEFT[1] - CENTER_VERTEX[1])
    assert math.isclose(derived_h2[0], expected_h2[0], abs_tol=1e-6)
    assert math.isclose(derived_h2[1], expected_h2[1], abs_tol=1e-6)


def test_derive_left_with_two_trusted_neighbors_uses_parallelogram():
    """Symmetric case for 'left' slot: h4 is unique-to-face and
    derived from h5 + h3 - center."""
    top_quad = _make_quad_with_cardinals(
        HEX_H0, HEX_H1, CENTER_VERTEX, HEX_H5,
    )
    right_quad = _make_quad_with_cardinals(
        HEX_H1, HEX_H2, HEX_H3_RIGHT, CENTER_VERTEX,
    )
    trusted = {"top": top_quad, "right": right_quad}
    derived = _derive_face_quad_topology_aware(
        "left", trusted, HEX_FOR_FALLBACK, cube_hull=None,
    )
    # Left canonical: [h5, center, h3, h4]
    assert len(derived) == 4
    derived_h5, derived_center, derived_h3, derived_h4 = derived
    assert derived_h5 == HEX_H5
    # h3_shared = right.S (first non-None) = HEX_H3_RIGHT
    assert derived_h3 == HEX_H3_RIGHT
    assert math.isclose(derived_center[0], CENTER_VERTEX[0], abs_tol=1e-6)
    assert math.isclose(derived_center[1], CENTER_VERTEX[1], abs_tol=1e-6)
    expected_h4 = (HEX_H5[0] + HEX_H3_RIGHT[0] - CENTER_VERTEX[0],
                   HEX_H5[1] + HEX_H3_RIGHT[1] - CENTER_VERTEX[1])
    assert math.isclose(derived_h4[0], expected_h4[0], abs_tol=1e-6)
    assert math.isclose(derived_h4[1], expected_h4[1], abs_tol=1e-6)


def test_derive_with_no_trusted_neighbors_falls_back_to_hexagon():
    """When no neighbors are trusted, the derivation must still return
    a valid 4-corner quad — pulled directly from the hexagon. This is
    the worst case but should not crash."""
    hexagon = [
        (500, 200), (700, 320), (740, 480),
        (500, 700), (260, 480), (300, 320),
    ]
    derived = _derive_face_quad_topology_aware(
        "top", {}, hexagon, cube_hull=None,
    )
    assert len(derived) == 4
    for p in derived:
        assert p is not None


def test_derive_clip_to_hull_bounds_extrapolated_corners():
    """When a 'trusted' neighbor's corner extends past the cube hull
    (the Set 17 A R.S = (426, 1012) below-the-cube case), clip-to-hull
    bounds the shared corner to the hull boundary before it propagates
    into the derived face quad. Tests that the function actually
    invokes _clip_to_hull on cardinal-position outputs when a cube
    hull is provided."""
    hexagon = [
        (500, 200), (700, 320), (740, 480),
        (500, 700), (260, 480), (300, 320),
    ]
    h0, h1, h2, h3, h4, h5 = hexagon
    center_vertex = (500, 470)
    # Set 17 A's actual bug pattern: R quad's south corner extrapolated
    # to (426, 1012) — well below the cube. Simulate with a bad_h3
    # value 500px below the true h3 position. Topology-aware derivation
    # of 'left' uses right.S as the h3 candidate; with clip-to-hull
    # this should snap back inside the hull rather than dragging the
    # derived left quad below the cube.
    bad_h3 = (h3[0], h3[1] + 500)
    top_quad = _make_quad_with_cardinals(h0, h1, center_vertex, h5)
    right_quad = _make_quad_with_cardinals(h1, h2, bad_h3, center_vertex)
    # The hull is the synthetic hexagon outline
    hull = list(hexagon)
    trusted = {"top": top_quad, "right": right_quad}

    derived_no_hull = _derive_face_quad_topology_aware(
        "left", trusted, hexagon, cube_hull=None,
    )
    derived_with_hull = _derive_face_quad_topology_aware(
        "left", trusted, hexagon, cube_hull=hull,
    )

    # Without hull: derived h3 corner inherits the bad extrapolation
    # (y > hull's bottom). With hull: clipped back to the hull boundary.
    # Left canonical: [h5, center, h3, h4]; h3 is at index 2.
    no_hull_h3 = derived_no_hull[2]
    with_hull_h3 = derived_with_hull[2]
    assert no_hull_h3[1] > 900, (
        f"expected unclipped derived h3.y to inherit bad extrapolation "
        f"(~1200), got {no_hull_h3[1]}"
    )
    assert with_hull_h3[1] <= 700.001, (
        f"expected hull-clipped derived h3.y to snap to hexagon bottom "
        f"(~700), got {with_hull_h3[1]}"
    )


def test_derive_invalid_slot_position_raises():
    """Defensive: an unknown slot_position is a programming error and
    should fail loudly."""
    hexagon = [
        (500, 200), (700, 320), (740, 480),
        (500, 700), (260, 480), (300, 320),
    ]
    with pytest.raises(ValueError, match="unknown slot_position"):
        _derive_face_quad_topology_aware(
            "back", {}, hexagon, cube_hull=None,
        )


# ---------------------------------------------------------------------------
# Module-level constant pin: threshold must be stable for sweep reproducibility
# ---------------------------------------------------------------------------

def test_fit_error_threshold_default_is_pinned():
    """The default threshold = 4.0 is calibrated from Set 17 A's
    fit_error breakdown (U=0.34, R=2.01, B=6.50). Changing it without
    re-running the sweep would silently shift accuracy numbers.
    Pin it so a future tweak forces an explicit decision."""
    assert evaluate_hybrid_pipeline.FIT_ERROR_FALLBACK_THRESHOLD == 4.0
