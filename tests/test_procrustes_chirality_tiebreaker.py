"""Tests for the Procrustes chirality tiebreaker added to
`fit_cube_template_to_anchors` (PR follow-up to #268 / #270).

The tiebreaker uses `_score_phase_separation` — the same `mean_near -
mean_far` darkness signal as `_resolve_near_far_phase` — to break
residual ties at the Procrustes layer, before downstream PnP and
phase correction. See `tools/PROCRUSTES_TIEBREAKER_REPORT.md` for the
12-row before/after.

Empirical caveat to know about while reading these tests: the
tiebreaker fixes the near/far axis of the chirality ambiguity (1 bit),
but it does NOT distinguish mirror chirality (the other bit). On 9/12
oracle rows the tiebreaker correctly identifies the negative-separation
set but the lex-first pick within that set is sometimes mirror-flipped.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.global_cube_model import (  # noqa: E402
    GlobalCubeModel,
    _score_phase_separation,
    derive_geometry,
    fit_cube_template_to_anchors,
)


# ---------------- _score_phase_separation unit tests ----------------


def test_score_phase_separation_returns_none_without_visible_corners():
    """Defensive: caller may pass a bare GlobalCubeModel without having
    called `derive_geometry`. Tiebreaker code interprets None as "no
    signal, keep default tiebreak", so the function must return None
    rather than raise."""
    m = GlobalCubeModel(cube_center_screen=(100.0, 100.0))
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    assert _score_phase_separation(m, rgb) is None


def test_score_phase_separation_negative_on_near_lighter_image():
    """Empirical polarity (per `tools/NEAR_FAR_PHASE_REPORT.md`):
    NEGATIVE `mean_near - mean_far` = GOOD (model's labeled near is
    LIGHTER on average than the labeled far). Build a synthetic image
    where the model's near corners sit on a bright background and far
    corners sit on a dark background; the helper must return negative.
    """
    m = GlobalCubeModel(
        cube_center_screen=(100.0, 100.0),
        axis_x_2d=(-50.0, -25.0),   # near LEFT-UP
        axis_y_2d=(0.0, 50.0),      # near DOWN
        axis_z_2d=(50.0, -25.0),    # near RIGHT-UP
    )
    derive_geometry(m)
    rgb = np.full((300, 300, 3), 200, dtype=np.uint8)  # bright
    # Paint dark stripes along the 3 vertex→far lines (h_xy/xz/yz).
    for k in ("h_xy", "h_xz", "h_yz"):
        target = m.visible_corners[k]
        for t in np.linspace(0.2, 0.8, 30):
            x = int(100.0 + (target[0] - 100.0) * t)
            y = int(100.0 + (target[1] - 100.0) * t)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if 0 <= y + dy < 300 and 0 <= x + dx < 300:
                        rgb[y + dy, x + dx] = (20, 20, 20)
    sep = _score_phase_separation(m, rgb)
    assert sep is not None
    assert sep < 0, (
        f"Expected negative separation (mean_near is LIGHTER than mean_far) "
        f"on a synthetic image with dark far-lines; got {sep}"
    )


def test_score_phase_separation_positive_on_swapped_image():
    """Mirror of the previous test: paint the near lines dark and far
    lines bright → separation should be positive (model's labeled near
    is DARKER, the phase-swapped signal)."""
    m = GlobalCubeModel(
        cube_center_screen=(100.0, 100.0),
        axis_x_2d=(-50.0, -25.0),
        axis_y_2d=(0.0, 50.0),
        axis_z_2d=(50.0, -25.0),
    )
    derive_geometry(m)
    rgb = np.full((300, 300, 3), 200, dtype=np.uint8)
    for k in ("h_x", "h_y", "h_z"):
        target = m.visible_corners[k]
        for t in np.linspace(0.2, 0.8, 30):
            x = int(100.0 + (target[0] - 100.0) * t)
            y = int(100.0 + (target[1] - 100.0) * t)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if 0 <= y + dy < 300 and 0 <= x + dx < 300:
                        rgb[y + dy, x + dx] = (20, 20, 20)
    sep = _score_phase_separation(m, rgb)
    assert sep is not None
    assert sep > 0


# ---------------- fit_cube_template_to_anchors integration ----------------


def _make_clean_iso_hexagon(center=(300.0, 300.0), radius=100.0):
    """6 hexagon vertices around `center` at `radius`, CCW from h_z (30°
    on math angle convention). Matching the template's CCW ordering."""
    # Math angles in degrees (CCW): h_z=30, h_xz=90, h_x=150, h_xy=210,
    # h_y=270, h_yz=330. Image y points DOWN, so flip sign of sin.
    angles = [30, 90, 150, 210, 270, 330]
    return [
        (
            center[0] + radius * math.cos(math.radians(a)),
            center[1] - radius * math.sin(math.radians(a)),
        )
        for a in angles
    ]


def test_tiebreaker_preserves_behavior_without_image():
    """When image_rgb is omitted, the function must behave EXACTLY as it
    did pre-tiebreaker (first tied perm by iteration order wins). Pin
    this so future refactors of the tiebreaker can't accidentally break
    no-image callers."""
    hex6 = _make_clean_iso_hexagon()
    bezels = [math.radians(150), math.radians(270), math.radians(30)]
    model = fit_cube_template_to_anchors(
        (300.0, 300.0), hex6, bezels, image_size=(600, 600),
    )
    assert model is not None
    # The pre-tiebreaker contract: a successful fit returns a model with
    # 3 axes + 7 visible corners. We don't assert on specific perm choice
    # (it's iteration-order-dependent) — just that the no-image path
    # still works.
    assert model.cube_center_screen is not None
    assert len(model.visible_corners) == 7


def test_tiebreaker_records_debug_when_image_provided():
    """When image_rgb is provided and there are tied perms, the debug
    dict must record:
      - procrustes_n_tied: how many perms tied at the floor
      - procrustes_tiebreaker: 'phase_separation' (not 'iteration_order')
      - procrustes_tiebreaker_chosen_separation: float
      - procrustes_tiebreaker_all_separations: list of all tied seps

    This is what makes the behavior auditable — without these fields we
    couldn't tell whether the tiebreaker fired."""
    hex6 = _make_clean_iso_hexagon()
    bezels = [math.radians(150), math.radians(270), math.radians(30)]
    # Uniform gray image — separation will be ~0, but the field must
    # still be recorded so callers know the tiebreaker ran.
    rgb = np.full((600, 600, 3), 128, dtype=np.uint8)
    model = fit_cube_template_to_anchors(
        (300.0, 300.0), hex6, bezels,
        image_size=(600, 600),
        image_rgb=rgb,
    )
    assert model is not None
    debug = model.debug
    assert debug.get("procrustes_n_tied", 0) > 1, (
        "Clean iso hexagon must produce multiple tied perms via the "
        "cube's 3-fold body-diagonal symmetry"
    )
    assert debug.get("procrustes_tiebreaker") == "phase_separation"
    assert isinstance(
        debug.get("procrustes_tiebreaker_chosen_separation"), float,
    )
    seps = debug.get("procrustes_tiebreaker_all_separations")
    assert isinstance(seps, list)
    assert len(seps) == debug["procrustes_n_tied"]


def test_tiebreaker_records_iteration_order_when_no_image():
    """Inverse: no image → debug.procrustes_tiebreaker = 'iteration_order'
    (the documented no-tiebreaker mode), with procrustes_n_tied still
    reported (so callers can see ties exist even if they're not broken)."""
    hex6 = _make_clean_iso_hexagon()
    bezels = [math.radians(150), math.radians(270), math.radians(30)]
    model = fit_cube_template_to_anchors(
        (300.0, 300.0), hex6, bezels, image_size=(600, 600),
        # image_rgb left unset
    )
    assert model is not None
    assert model.debug.get("procrustes_tiebreaker") == "iteration_order"
    assert model.debug.get("procrustes_n_tied", 0) >= 1


def test_tiebreaker_returns_none_on_invalid_hexagon():
    """Existing failure mode preserved: <6 vertices → return None."""
    model = fit_cube_template_to_anchors(
        (300.0, 300.0),
        [(100.0, 100.0), (200.0, 100.0)],  # only 2 vertices
        [0.0, 0.0, 0.0],
        image_size=(600, 600),
    )
    assert model is None
