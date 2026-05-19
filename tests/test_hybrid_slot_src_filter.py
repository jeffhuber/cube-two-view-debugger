"""Unit tests for the slot/src filter in evaluate_hybrid_pipeline.

The slot/src filter rejects analyze_image grids whose center_face is
incompatible with the side's expected canonical set ({U,R,F} on A,
{D,L,B} on B). It exists as experimental infrastructure documenting
a NEGATIVE deployment result: the underlying signal is real (Set 61
went from 0.3519 → 0.6296 per-sticker with the filter enabled, +27.8pp)
but the deployment shape doesn't beat the existing recognizer:

  Hull-guard only        : 0.6532 per-sticker (rect faces)
  Slot/src hard filter   : 0.6911 per-sticker  but 18/33 fail to assemble
  Slot/src + fallback    : 0.6609 per-sticker  4/33 fail to assemble
  Existing recognizer    : 0.8270 per-sticker  (direct sampling, no rectify)

These tests pin the filter behavior so the experimental flag stays
verifiable for future iterations (e.g. a "soft penalty in ranking"
variant that doesn't hard-reject).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from tools import evaluate_hybrid_pipeline  # noqa: E402


def _grid(center_face: str, matched_count: int, fit_error: float,
          points=None):
    """Minimal FaceGrid stand-in. Default points form a tiny square at
    (100, 100) — enough for the filter logic; geometry irrelevant to
    these tests because we monkeypatch the hull guard."""
    if points is None:
        points = [
            [(100, 100), (110, 100), (120, 100)],
            [(100, 110), (110, 110), (120, 110)],
            [(100, 120), (110, 120), (120, 120)],
        ]
    return types.SimpleNamespace(
        center_face=center_face,
        matched_count=matched_count,
        fit_error=fit_error,
        points=points,
        cube_hull_inside_count=9,
    )


def _patch_analyze(monkeypatch, grids):
    def fake_analyze(_image_bytes):
        return types.SimpleNamespace(stickers=[], grids=grids, warnings=[])
    monkeypatch.setattr(evaluate_hybrid_pipeline, "analyze_image", fake_analyze)


def _patch_hull_off(monkeypatch):
    """Bypass rembg hull computation — return empty hull so the inside-count
    check degrades to 'no validation, accept everything' (returns 9)."""
    monkeypatch.setattr(evaluate_hybrid_pipeline, "_rembg_cube_hull",
                        lambda image: [])


def test_filter_keeps_expected_side_a_centers(monkeypatch, tmp_path):
    """Side A with {U, R, F} centers — none should be rejected by slot/src."""
    _patch_analyze(monkeypatch, [
        _grid("U", 9, 1.0),
        _grid("R", 9, 1.5),
        _grid("F", 9, 2.0),
    ])
    _patch_hull_off(monkeypatch)

    image_path = tmp_path / "fake.jpg"
    image_path.write_bytes(b"unused")
    from PIL import Image
    fake_img = Image.new("RGB", (600, 600), (128, 128, 128))

    quads, debug = evaluate_hybrid_pipeline._proposer_face_quads(
        image_path, "A", hull_guard=False, slot_src_filter=True,
        processing_image=fake_img,
    )
    assert set(quads.keys()) == {"U", "R", "F"}
    assert debug["slotSrcFilterEnabled"] is True
    assert debug["gridsRejectedBySlotSrc"] == []


def test_filter_rejects_wrong_side_centers_when_3_valid_present(monkeypatch, tmp_path):
    """Side A with valid {U, R, F} + bonus wrong-side {L, B} grids: the
    filter promotes only the 3 valid grids; the wrong-side grids are in
    the deferred pool but not promoted (since 3 faces already filled)."""
    _patch_analyze(monkeypatch, [
        _grid("U", 9, 1.0),
        _grid("R", 9, 1.5),
        _grid("F", 9, 2.0),
        # Bonus wrong-side grids (L and B shouldn't be visible on A in canonical yaw)
        _grid("L", 9, 0.5),  # high quality — would be promoted by fallback if needed
        _grid("B", 9, 0.5),
    ])
    _patch_hull_off(monkeypatch)

    image_path = tmp_path / "fake.jpg"
    image_path.write_bytes(b"unused")
    from PIL import Image
    fake_img = Image.new("RGB", (600, 600), (128, 128, 128))

    quads, debug = evaluate_hybrid_pipeline._proposer_face_quads(
        image_path, "A", hull_guard=False, slot_src_filter=True,
        processing_image=fake_img,
    )
    # Only U/R/F selected; L/B stay in deferred (not promoted)
    assert set(quads.keys()) == {"U", "R", "F"}
    # L and B should be tracked as rejected (in the deferred pool but not promoted)
    rejected_faces = {r["centerFace"] for r in debug["gridsRejectedBySlotSrc"]}
    assert rejected_faces == {"L", "B"}


def test_fallback_promotes_deferred_when_lt3_valid(monkeypatch, tmp_path):
    """Yaw-rotated-style case: side A has only U + L + B from analyze_image
    (no R, no F — analyze_image classified all side-faces as B-side colors).
    Hard filter would drop L and B → only 1 face. Fallback should promote
    L and B back into the pool to maintain 3 faces total."""
    _patch_analyze(monkeypatch, [
        _grid("U", 9, 1.0),
        _grid("L", 9, 1.5),
        _grid("B", 9, 2.0),
    ])
    _patch_hull_off(monkeypatch)

    image_path = tmp_path / "fake.jpg"
    image_path.write_bytes(b"unused")
    from PIL import Image
    fake_img = Image.new("RGB", (600, 600), (128, 128, 128))

    quads, debug = evaluate_hybrid_pipeline._proposer_face_quads(
        image_path, "A", hull_guard=False, slot_src_filter=True,
        processing_image=fake_img,
    )
    # Fallback should re-key all 3 to the U/R/F convention:
    # U keeps its slot (anchor), L and B get re-keyed to R/F (arbitrary order)
    assert set(quads.keys()) == {"U", "R", "F"}
    # The R and F slots should have "rekeyedFrom" pointing to L or B
    rekeyed = {
        debug["selectedPerFace"]["R"].get("sourceCenterFace"),
        debug["selectedPerFace"]["F"].get("sourceCenterFace"),
    }
    assert rekeyed == {"L", "B"}


def test_filter_off_default_behavior_unchanged(monkeypatch, tmp_path):
    """With slot_src_filter=False (default), the proposer behaves identically
    to the pre-filter version — any non-anchor grid can fill the side-face
    slots, regardless of analyze_image's center_face classification."""
    _patch_analyze(monkeypatch, [
        _grid("U", 9, 1.0),
        _grid("L", 9, 1.5),  # wrong-side label
        _grid("B", 9, 2.0),  # wrong-side label
    ])
    _patch_hull_off(monkeypatch)

    image_path = tmp_path / "fake.jpg"
    image_path.write_bytes(b"unused")
    from PIL import Image
    fake_img = Image.new("RGB", (600, 600), (128, 128, 128))

    quads, debug = evaluate_hybrid_pipeline._proposer_face_quads(
        image_path, "A", hull_guard=False, slot_src_filter=False,
        processing_image=fake_img,
    )
    # All 3 grids accepted as-is, just re-keyed to U/R/F slot convention
    assert set(quads.keys()) == {"U", "R", "F"}
    assert debug["slotSrcFilterEnabled"] is False
    assert debug["gridsRejectedBySlotSrc"] == []


def test_filter_works_for_side_b(monkeypatch, tmp_path):
    """Side B: expected set is {D, L, B}. A wrong-side R or U or F should
    end up in the deferred pool (and stay there if D/L/B has enough)."""
    _patch_analyze(monkeypatch, [
        _grid("D", 9, 1.0),
        _grid("L", 9, 1.5),
        _grid("B", 9, 2.0),
        _grid("R", 9, 0.5),  # wrong side
    ])
    _patch_hull_off(monkeypatch)

    image_path = tmp_path / "fake.jpg"
    image_path.write_bytes(b"unused")
    from PIL import Image
    fake_img = Image.new("RGB", (600, 600), (128, 128, 128))

    quads, debug = evaluate_hybrid_pipeline._proposer_face_quads(
        image_path, "B", hull_guard=False, slot_src_filter=True,
        processing_image=fake_img,
    )
    assert set(quads.keys()) == {"D", "L", "B"}
    rejected_faces = {r["centerFace"] for r in debug["gridsRejectedBySlotSrc"]}
    assert rejected_faces == {"R"}


def test_expected_sets_are_correct():
    """Pin the expected-by-side mapping so it can't drift silently."""
    assert evaluate_hybrid_pipeline.SLOT_SRC_EXPECTED_BY_SIDE["A"] == frozenset({"U", "R", "F"})
    assert evaluate_hybrid_pipeline.SLOT_SRC_EXPECTED_BY_SIDE["B"] == frozenset({"D", "L", "B"})


def test_debug_surfaces_promoted_from_deferred_count(monkeypatch, tmp_path):
    """Pins the debug-output contract that Codex flagged on PR #157 review:
    when the slot/src fallback promotes deferred grids back into the pool
    (yaw-rotated case), `gridsPromotedFromDeferred` must list them so the
    per-pair JSON report can surface promoted/fallback counts.

    Without this, the experimental flag can't be A/B-analyzed
    productively (you can't tell whether a sweep's results reflect
    'filter held tight' vs 'filter degenerated to fallback')."""
    _patch_analyze(monkeypatch, [
        _grid("U", 9, 1.0),
        _grid("L", 9, 1.5),  # promoted via fallback
        _grid("B", 9, 2.0),  # promoted via fallback
    ])
    _patch_hull_off(monkeypatch)

    image_path = tmp_path / "fake.jpg"
    image_path.write_bytes(b"unused")
    from PIL import Image
    fake_img = Image.new("RGB", (600, 600), (128, 128, 128))

    quads, debug = evaluate_hybrid_pipeline._proposer_face_quads(
        image_path, "A", hull_guard=False, slot_src_filter=True,
        processing_image=fake_img,
    )
    promoted = debug["gridsPromotedFromDeferred"]
    promoted_faces = {entry["centerFace"] for entry in promoted}
    assert promoted_faces == {"L", "B"}
    # And the truly-rejected set should be empty (everything promoted)
    assert debug["gridsRejectedBySlotSrc"] == []


def test_debug_promoted_empty_when_filter_off(monkeypatch, tmp_path):
    """When the filter is off, no grids are deferred → promotion list
    is empty regardless of what grids analyze_image returns."""
    _patch_analyze(monkeypatch, [
        _grid("U", 9, 1.0),
        _grid("L", 9, 1.5),
        _grid("B", 9, 2.0),
    ])
    _patch_hull_off(monkeypatch)

    image_path = tmp_path / "fake.jpg"
    image_path.write_bytes(b"unused")
    from PIL import Image
    fake_img = Image.new("RGB", (600, 600), (128, 128, 128))

    _, debug = evaluate_hybrid_pipeline._proposer_face_quads(
        image_path, "A", hull_guard=False, slot_src_filter=False,
        processing_image=fake_img,
    )
    assert debug["gridsPromotedFromDeferred"] == []
