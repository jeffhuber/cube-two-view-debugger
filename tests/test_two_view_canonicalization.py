"""Tests for `tools/two_view_canonicalization.py`.

Coverage:
- The 24-element CUBE_ROTATIONS tuple has expected properties
  (all det=+1, no duplicates, includes the identity).
- Round-trip: applying a cube rotation to perfectly consistent axes
  still recovers ~0° residual under the min-over-24 search.
- Empirical validation against the 35 human-labeled GOOD pairs in
  `tests/fixtures/gcm_axis_ground_truth.json` — the contract for
  v2 integration is GOOD-pair median ≤25° (achieved: ~15°).
"""

from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path

import numpy as np
import pytest

from tools import two_view_canonicalization as tvcanon
from tools import two_view_consistency as tvc


GROUND_TRUTH_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "gcm_axis_ground_truth.json"
)


def _axes_from_label(rec):
    vx, vy = rec["vertex"]
    return (
        (rec["near_x"][0] - vx, rec["near_x"][1] - vy),
        (rec["near_y"][0] - vx, rec["near_y"][1] - vy),
        (rec["near_z"][0] - vx, rec["near_z"][1] - vy),
    )


def _load_good_pairs():
    with open(GROUND_TRUTH_PATH) as f:
        data = json.load(f)
    sets = sorted({k.split("_")[0] for k in data})
    pairs = []
    for s in sets:
        ka, kb = f"{s}_A", f"{s}_B"
        if ka in data and kb in data:
            if data[ka].get("approved") and data[kb].get("approved"):
                pairs.append((s, _axes_from_label(data[ka]),
                               _axes_from_label(data[kb])))
    return pairs


# ----- ALL_AXIS_TRANSFORMS / CUBE_ROTATIONS shape -----


def test_all_axis_transforms_count():
    """All 6 permutations × 8 sign-flip patterns = 48 signed axis
    permutations."""
    assert len(tvcanon.ALL_AXIS_TRANSFORMS) == 48


def test_all_axis_transforms_distinct():
    """No duplicates in the precomputed search space."""
    assert len(set(tvcanon.ALL_AXIS_TRANSFORMS)) == 48


def test_cube_rotations_subset_count():
    """The chiral octahedral group has exactly 24 elements. Retained
    as a subset for diagnostic comparisons (not used in the metric)."""
    assert len(tvcanon.CUBE_ROTATIONS) == 24


def test_cube_rotations_all_det_plus_one():
    """Every cube rotation is a proper rotation (det = +1)."""
    for perm, signs in tvcanon.CUBE_ROTATIONS:
        assert tvcanon._transform_det(perm, signs) == 1


def test_all_axis_transforms_includes_identity():
    """Identity (perm (0,1,2), signs (+1,+1,+1)) must be present."""
    assert ((0, 1, 2), (1, 1, 1)) in tvcanon.ALL_AXIS_TRANSFORMS


def test_all_axis_transforms_half_are_reflections():
    """24 of the 48 transforms are reflections (det = -1). Codex P2
    on PR #246: these are empirically required to align ~one-third
    of human-labeled GOOD pairs; the metric must search both halves."""
    reflections = [t for t in tvcanon.ALL_AXIS_TRANSFORMS
                   if tvcanon._transform_det(*t) == -1]
    assert len(reflections) == 24


# ----- Round-trip with known axes -----


def test_round_trip_identity_pair_after_flip():
    """A pair related by the canonical R_FLIP returns ~0° regardless
    of which signed axis permutation is applied to B (canonicalization
    absorbs it)."""
    R_A = np.eye(3)
    R_B = tvc.R_FLIP @ R_A
    axes_A = _project_axes(R_A)
    axes_B = _project_axes(R_B)
    deg = tvcanon.canonicalized_two_view_consistency_deg(axes_A, axes_B)
    assert deg < 1e-3, f"identity-then-flip should give ~0°, got {deg:.4f}°"


def test_round_trip_under_arbitrary_b_signed_permutation():
    """Apply each of the 48 signed axis permutations to B's raw axes;
    the canonicalized metric should still return ~0° (within numerical
    tolerance) because the canonicalization absorbs the transform."""
    R_A = _euler_to_R(45.0, -35.26, 0.0)
    R_B = tvc.R_FLIP @ R_A
    axes_A = _project_axes(R_A)
    axes_B_raw = _project_axes(R_B)
    for perm, signs in tvcanon.ALL_AXIS_TRANSFORMS:
        axes_B_perm = tvcanon._apply_transform(axes_B_raw, perm, signs)
        deg = tvcanon.canonicalized_two_view_consistency_deg(
            axes_A, axes_B_perm
        )
        assert deg < 1.0, (
            f"canonicalization failed for perm={perm} signs={signs}: "
            f"residual {deg:.2f}°"
        )


# ----- best_canonicalization diagnostic API -----


def test_best_canonicalization_returns_winning_transform():
    R_A = _euler_to_R(45.0, -35.26, 0.0)
    R_B = tvc.R_FLIP @ R_A
    axes_A = _project_axes(R_A)
    axes_B = _project_axes(R_B)
    deg, perm, signs = tvcanon.best_canonicalization(axes_A, axes_B)
    assert deg < 1e-3
    assert (perm, signs) in tvcanon.ALL_AXIS_TRANSFORMS


# ----- Degenerate-input contract (Codex P2 on PR #246) -----


def test_canonicalized_raises_on_zero_axes_A():
    """If axes_A is all-zeros (e.g., a default-initialized
    GlobalCubeModel that never received a fit), the function must
    raise — silently returning inf would leak a non-finite trust
    feature into the matrix."""
    zero = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    R = np.eye(3)
    good_axes = _project_axes(R)
    with pytest.raises(ValueError, match="degenerate"):
        tvcanon.canonicalized_two_view_consistency_deg(zero, good_axes)


def test_canonicalized_raises_on_zero_axes_B():
    """If axes_B_raw is all-zeros, every signed permutation is
    degenerate; the function must raise rather than fall through to
    inf (Codex P2 on PR #246)."""
    zero = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    R = np.eye(3)
    good_axes = _project_axes(R)
    with pytest.raises(ValueError, match="degenerate"):
        tvcanon.canonicalized_two_view_consistency_deg(good_axes, zero)


def test_best_canonicalization_raises_on_zero_axes_B():
    """Same degenerate-input contract for the diagnostic API."""
    zero = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    R = np.eye(3)
    good_axes = _project_axes(R)
    with pytest.raises(ValueError, match="degenerate"):
        tvcanon.best_canonicalization(good_axes, zero)


# ----- Empirical validation contract -----


def test_canonicalization_baseline_raw_axes_fails():
    """Sanity: without canonicalization, raw axes give ~173° median
    on the GOOD fixture pairs (the empirical 'before' state Codex
    flagged on PR #245). If this test starts passing under 25°
    something has shifted — investigate before trusting downstream
    metrics."""
    pairs = _load_good_pairs()
    raw = []
    for _, ax_A, ax_B in pairs:
        try:
            d = tvc.two_view_consistency_deg(ax_A, ax_B)
            raw.append(d)
        except ValueError:
            continue
    assert len(raw) >= 30
    median = statistics.median(raw)
    assert median > 100, (
        f"raw-axes baseline median should be ~173° (the empirical "
        f"'before' state from Codex's audit), got {median:.2f}°"
    )


def test_canonicalization_collapses_good_pair_median():
    """The v2 integration contract: GOOD-pair median ≤25° on the
    35-pair human-labeled fixture. Empirical with the 48-transform
    search: ~10.66°."""
    pairs = _load_good_pairs()
    assert len(pairs) >= 30, (
        f"expected ≥30 GOOD pairs in fixture, got {len(pairs)}"
    )

    residuals = []
    for _, ax_A, ax_B in pairs:
        d = tvcanon.canonicalized_two_view_consistency_deg(ax_A, ax_B)
        residuals.append(d)

    median = statistics.median(residuals)
    assert median <= 25.0, (
        f"canonicalized GOOD-pair median should be ≤25° "
        f"(integration target; v1 measured ~10.66°), got {median:.2f}°"
    )


def test_canonicalization_all_good_pairs_under_25():
    """ALL GOOD pairs (35/35) should fall under the 25° threshold
    after the full 48-transform canonicalization. This is the stronger
    contract enabled by Codex's P2 fix (include reflections). v0
    measured only 23/35 with the det=+1 restriction; v1 measures
    35/35 with all 48 transforms."""
    pairs = _load_good_pairs()
    under_25 = sum(
        1 for _, ax_A, ax_B in pairs
        if tvcanon.canonicalized_two_view_consistency_deg(ax_A, ax_B) < 25.0
    )
    assert under_25 == len(pairs), (
        f"expected ALL {len(pairs)} GOOD pairs under 25° (Codex P2 fix), "
        f"got {under_25}/{len(pairs)}"
    )


def test_canonicalization_max_good_residual_bounded():
    """Max residual on the GOOD fixture should be ≤25°. Empirical: ~23.41°."""
    pairs = _load_good_pairs()
    max_res = max(
        tvcanon.canonicalized_two_view_consistency_deg(ax_A, ax_B)
        for _, ax_A, ax_B in pairs
    )
    assert max_res <= 25.0, (
        f"max GOOD-pair residual should be ≤25°, got {max_res:.2f}°"
    )


def test_unflipped_same_pose_pair_canonicalized_is_low():
    """Codex P1 documentation: the canonicalized metric ALONE cannot
    distinguish an unflipped/same-pose pair from a legitimate consistent
    pair. This test pins the empirical behavior so it doesn't
    silently change — same-pose pairs DO score under 25° because
    the search includes ((0,1,2),(1,-1,-1)) = R_FLIP itself.

    The v2 integration MUST use `consistency_features()` and consume
    the raw value alongside the canonicalized one — see
    `test_unflipped_same_pose_pair_raw_is_180_exactly`.
    """
    pairs = _load_good_pairs()
    same_pose_canon = []
    for _, ax_A, _ in pairs:
        # Substitute axes_B with axes_A — simulates the user duplicating
        # photo A instead of flipping the cube.
        d = tvcanon.canonicalized_two_view_consistency_deg(ax_A, ax_A)
        same_pose_canon.append(d)
    median = statistics.median(same_pose_canon)
    assert median < 25.0, (
        f"documented limitation: same-pose pairs score low on "
        f"canonicalized alone, expected median <25°, got {median:.2f}°"
    )


def test_unflipped_same_pose_pair_raw_is_180_exactly():
    """The mitigation for Codex P1: the RAW (un-canonicalized) metric
    DOES detect same-pose pairs — it returns exactly 180° because
    R_A = R_B exactly, so the residual is R_FLIP itself.

    v2 integration must include raw_deg as a feature so the trust
    ranker can flag the same-pose mode."""
    pairs = _load_good_pairs()
    for s, ax_A, _ in pairs:
        feats = tvcanon.consistency_features(ax_A, ax_A)
        # 1e-4 tolerance accounts for arccos numerical sensitivity
        # near trace = −1 (where dθ/d(trace) → ∞). Same numerical
        # behavior as the candidate-set 180° check in
        # test_two_view_orientation_consistency.py.
        assert feats["raw_deg"] == pytest.approx(180.0, abs=1e-4), (
            f"same-pose pair for set {s}: raw_deg should be ~180° "
            f"(the angle of R_FLIP), got {feats['raw_deg']:.4f}°"
        )


def test_consistency_features_returns_expected_keys():
    """Contract: consistency_features() returns a dict with the 3
    expected keys for v2 integration."""
    pairs = _load_good_pairs()
    s, ax_A, ax_B = pairs[0]
    feats = tvcanon.consistency_features(ax_A, ax_B)
    assert set(feats.keys()) == {
        "canonicalized_deg", "raw_deg", "canon_gap_deg"
    }
    assert 0.0 <= feats["canonicalized_deg"] <= 180.0
    assert 0.0 <= feats["raw_deg"] <= 180.0
    assert feats["canon_gap_deg"] == pytest.approx(
        feats["raw_deg"] - feats["canonicalized_deg"]
    )


def test_consistency_features_raises_on_degenerate():
    """Same degenerate-input contract as the bare canonicalization."""
    zero = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    R = np.eye(3)
    good_axes = _project_axes(R)
    with pytest.raises(ValueError, match="degenerate"):
        tvcanon.consistency_features(zero, good_axes)
    with pytest.raises(ValueError, match="degenerate"):
        tvcanon.consistency_features(good_axes, zero)


def test_canonicalized_value_bounded_by_180():
    """The metric always returns a value in [0, 180] since it's a
    rotation angle."""
    pairs = _load_good_pairs()
    for s, ax_A, ax_B in pairs:
        d = tvcanon.canonicalized_two_view_consistency_deg(ax_A, ax_B)
        assert 0.0 <= d <= 180.0, (
            f"residual out of [0, 180] for set {s}: {d:.2f}°"
        )


# ----- Test helpers (mirror the math test module's pattern) -----


def _project_axes(R: np.ndarray, s: float = 100.0):
    M = s * R[:2, :3]
    return (
        (float(M[0, 0]), float(M[1, 0])),
        (float(M[0, 1]), float(M[1, 1])),
        (float(M[0, 2]), float(M[1, 2])),
    )


def _euler_to_R(yaw_deg, pitch_deg, roll_deg):
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0],
                   [np.sin(r), np.cos(r), 0],
                   [0, 0, 1]])
    return Rz @ Rx @ Ry
