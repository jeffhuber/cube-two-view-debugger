"""Unit tests for the analyze_image quad re-key in evaluate_hybrid_pipeline.

Pins the behavior Devin's PR #152 audit caught: when analyze_image
classifies a side-A center under a non-canonical letter (L, B, or D —
common for yawed photos like Set 23 or after orange↔red center
confusion from PR #150), the corresponding quad MUST NOT be silently
dropped by the geometry-labeler convention enforced inside
identify_faces_jointly.

Tests use a fake analyze_image to avoid coupling to real corpus
files and the macOS Full Disk Access requirement on ~/Downloads.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from tools import evaluate_hybrid_pipeline  # noqa: E402


def test_classifier_mode_label_uses_shared_runtime_default(monkeypatch):
    monkeypatch.delenv(evaluate_hybrid_pipeline.CLASSIFIER_MODE_ENV, raising=False)

    assert evaluate_hybrid_pipeline._classifier_mode_label() == evaluate_hybrid_pipeline.DEFAULT_CLASSIFIER_MODE


def test_classifier_mode_label_preserves_explicit_canonical(monkeypatch):
    monkeypatch.setenv(evaluate_hybrid_pipeline.CLASSIFIER_MODE_ENV, "canonical")

    assert evaluate_hybrid_pipeline._classifier_mode_label() == "canonical"


def _fake_grid(center_face: str, matched_count: int, fit_error: float, points):
    """Minimal FaceGrid stand-in for the re-key test. Only the
    attributes actually read by _proposer_face_quads are populated."""
    return types.SimpleNamespace(
        center_face=center_face,
        matched_count=matched_count,
        fit_error=fit_error,
        points=points,
        cube_hull_inside_count=9,
    )


def _square_3x3(cx: float, cy: float, spacing: float = 30.0):
    """A 3x3 grid of sticker centers around (cx, cy), spacing apart.
    Lays out row-major, top-left to bottom-right (matching how
    analyze_image's FaceGrid.points is structured)."""
    return [
        [(cx + (c - 1) * spacing, cy + (r - 1) * spacing) for c in range(3)]
        for r in range(3)
    ]


def _patch_analyze(monkeypatch, grids):
    """Replace analyze_image with a stub that returns the supplied grids
    on every call. Tests that call _proposer_face_quads for one side at
    a time should pass the grids appropriate for that side."""
    def fake_analyze(_image_bytes):
        return types.SimpleNamespace(
            stickers=[],
            grids=grids,
            warnings=[],
        )
    monkeypatch.setattr(evaluate_hybrid_pipeline, "analyze_image", fake_analyze)


def test_side_a_yaw2_quads_with_L_and_B_centers_are_not_dropped(monkeypatch, tmp_path):
    """Set 23-style yaw2: side A's visible faces are U + L + B (NOT U + R + F).
    analyze_image correctly classifies them as U, L, B centers. The
    pre-fix _proposer_face_quads keyed the output as {U, L, B}, which
    identify_faces_jointly would then silently drop down to just {U}
    because L and B aren't in expected_a = ["R", "F"]. After the fix,
    L and B must be re-keyed as R and F (in arbitrary order) so all
    three quads survive."""
    side_a_grids = [
        _fake_grid("U", matched_count=9, fit_error=2.0, points=_square_3x3(500, 200)),
        _fake_grid("L", matched_count=9, fit_error=2.5, points=_square_3x3(300, 500)),
        _fake_grid("B", matched_count=9, fit_error=3.0, points=_square_3x3(700, 500)),
    ]
    _patch_analyze(monkeypatch, side_a_grids)

    fake_path = tmp_path / "fake_side_a.jpg"
    fake_path.write_bytes(b"unused")

    quads, debug = evaluate_hybrid_pipeline._proposer_face_quads(fake_path, "A")

    # All three quads must survive; keys must match the geometry-labeler convention
    assert set(quads.keys()) == {"U", "R", "F"}, \
        f"expected re-key to {{U, R, F}}, got {set(quads.keys())}"
    # Anchor flag should report that U was found
    assert debug["anchorFound"] is True
    # Both side-face slots should record which original analyze_image label
    # they were re-keyed from (one L, one B, in arbitrary order)
    rekeyed_sources = {
        debug["selectedPerFace"]["R"].get("rekeyedFrom"),
        debug["selectedPerFace"]["F"].get("rekeyedFrom"),
    }
    assert rekeyed_sources == {"L", "B"}, \
        f"expected R and F to be rekeyed from {{L, B}}, got {rekeyed_sources}"


def test_side_a_orange_red_center_confusion_does_not_drop_quad(monkeypatch, tmp_path):
    """PR #150's diagnostic showed orange↔red center confusion is a real
    failure mode of the canonical classifier. If analyze_image's center
    classifier mistakes a side-A face's red center for orange, the
    corresponding grid arrives keyed as "L" (orange). Without re-key
    that quad would be dropped by joint face-ID."""
    side_a_grids = [
        _fake_grid("U", matched_count=9, fit_error=2.0, points=_square_3x3(500, 200)),
        _fake_grid("R", matched_count=9, fit_error=2.2, points=_square_3x3(300, 500)),
        # This face is really R (red center) but analyze_image mis-classified
        # it as L (orange) — simulating the dim-red→orange confusion
        _fake_grid("L", matched_count=9, fit_error=2.5, points=_square_3x3(700, 500)),
    ]
    _patch_analyze(monkeypatch, side_a_grids)

    fake_path = tmp_path / "fake_orange_red.jpg"
    fake_path.write_bytes(b"unused")

    quads, debug = evaluate_hybrid_pipeline._proposer_face_quads(fake_path, "A")
    assert set(quads.keys()) == {"U", "R", "F"}


def test_side_b_with_L_and_R_centers_rekeyed_to_L_and_B(monkeypatch, tmp_path):
    """Side B's expected labels are D/L/B. If analyze_image classifies a
    side-B center as "R" (e.g. orange→red confusion on side B), the
    re-key must restore it under the L or B slot so it survives."""
    side_b_grids = [
        _fake_grid("D", matched_count=9, fit_error=1.5, points=_square_3x3(500, 200)),
        _fake_grid("L", matched_count=9, fit_error=2.0, points=_square_3x3(300, 500)),
        # Side-B face misclassified as R
        _fake_grid("R", matched_count=9, fit_error=2.2, points=_square_3x3(700, 500)),
    ]
    _patch_analyze(monkeypatch, side_b_grids)

    fake_path = tmp_path / "fake_side_b.jpg"
    fake_path.write_bytes(b"unused")

    quads, debug = evaluate_hybrid_pipeline._proposer_face_quads(fake_path, "B")
    assert set(quads.keys()) == {"D", "L", "B"}
    rekeyed_sources = {
        debug["selectedPerFace"]["L"].get("rekeyedFrom"),
        debug["selectedPerFace"]["B"].get("rekeyedFrom"),
    }
    assert rekeyed_sources == {"L", "R"}


def test_missing_anchor_degrades_gracefully(monkeypatch, tmp_path):
    """If analyze_image fails to find a white (U) center on side A,
    the anchor slot is empty and we only return the 2 side-face slots.
    Pair-level evaluation will then have <3 faces for side A; joint
    face-ID handles this as 'missing_side' upstream."""
    side_a_grids = [
        _fake_grid("R", matched_count=9, fit_error=2.0, points=_square_3x3(300, 500)),
        _fake_grid("F", matched_count=9, fit_error=2.5, points=_square_3x3(700, 500)),
    ]
    _patch_analyze(monkeypatch, side_a_grids)

    fake_path = tmp_path / "fake_no_anchor.jpg"
    fake_path.write_bytes(b"unused")

    quads, debug = evaluate_hybrid_pipeline._proposer_face_quads(fake_path, "A")
    assert "U" not in quads
    assert debug["anchorFound"] is False
    # Side-face quads still present
    assert set(quads.keys()).issubset({"R", "F"})


def test_excess_proposer_grids_picks_best_two_by_quality(monkeypatch, tmp_path):
    """analyze_image can return many candidate grids per face (different
    fits for the same physical face). We should pick the BEST candidate
    per analyze_image center_face, then take the top 2 non-anchor by
    quality. This test pins that ordering: higher matched_count + lower
    fit_error wins."""
    side_a_grids = [
        _fake_grid("U", matched_count=9, fit_error=1.0, points=_square_3x3(500, 200)),
        _fake_grid("R", matched_count=5, fit_error=10.0, points=_square_3x3(300, 500)),  # weak
        _fake_grid("F", matched_count=9, fit_error=2.0, points=_square_3x3(700, 500)),  # strong
        _fake_grid("L", matched_count=9, fit_error=1.5, points=_square_3x3(400, 600)),  # strongest non-anchor
    ]
    _patch_analyze(monkeypatch, side_a_grids)

    fake_path = tmp_path / "fake_excess.jpg"
    fake_path.write_bytes(b"unused")

    quads, debug = evaluate_hybrid_pipeline._proposer_face_quads(fake_path, "A")
    # The weak R grid should be dropped in favor of L+F (the two highest-quality
    # non-anchor grids), re-keyed onto R/F slots
    rekeyed_sources = {
        debug["selectedPerFace"]["R"].get("rekeyedFrom"),
        debug["selectedPerFace"]["F"].get("rekeyedFrom"),
    }
    assert rekeyed_sources == {"L", "F"}
