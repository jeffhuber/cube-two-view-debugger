"""Unit tests for `tools/measure_axis_correctness.py`.

Pure-function tests of the angle math, axis-matching, ground-truth
derivation, and report rendering. The pipeline invocation
(`evaluate_one_row`) is not unit-tested here because it depends on
rembg + bezel + global cube model fit; the canonical end-to-end
validation is the committed `tests/fixtures/axis_correctness_trace.json`
which captures real-data output on the 12 oracle rows.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import measure_axis_correctness as m  # noqa: E402


# ---------------- angle math ----------------


def test_angle_between_parallel_vectors_is_zero():
    assert m._angle_between((1.0, 0.0), (5.0, 0.0)) == pytest.approx(0.0)


def test_angle_between_perpendicular_vectors_is_90():
    assert m._angle_between((1.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)


def test_angle_between_antiparallel_vectors_is_180():
    assert m._angle_between((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(180.0)


def test_angle_between_handles_zero_vector_safely():
    """Degenerate input (zero-length vector) returns NaN, not crash."""
    result = m._angle_between((0.0, 0.0), (1.0, 0.0))
    assert math.isnan(result)


def test_angle_between_clamps_floating_point_overflow():
    """Two near-identical vectors should give ~0° even if the cos
    computation produces 1.0000000001 due to floating point."""
    a = (1.0, 0.0)
    b = (1.0, 1e-15)
    assert 0.0 <= m._angle_between(a, b) < 1.0


def test_length_uses_euclidean_norm():
    assert m._length((3.0, 4.0)) == pytest.approx(5.0)
    assert m._length((0.0, 0.0)) == pytest.approx(0.0)


# ---------------- axis matching ----------------


def test_match_axes_identical_input_returns_zero_misfit():
    """If predicted axes == ground truth, total misfit is 0 and
    assignment is identity."""
    axes = [(100.0, 0.0), (0.0, 100.0), (-50.0, -50.0)]
    result = m._match_axes_to_ground_truth(axes, axes)
    assert result["total_misfit_deg"] == pytest.approx(0.0, abs=1e-6)
    assert result["assignment"] == [0, 1, 2]
    for err in result["per_axis_angle_errors_deg"]:
        assert err == pytest.approx(0.0, abs=1e-6)


def test_match_axes_finds_best_permutation_under_reordering():
    """Predicted axes reordered should be matched back via the best
    permutation; final misfit still 0."""
    gt = [(100.0, 0.0), (0.0, 100.0), (-50.0, -50.0)]
    # Predicted in reverse order:
    predicted = [gt[2], gt[1], gt[0]]
    result = m._match_axes_to_ground_truth(predicted, gt)
    assert result["total_misfit_deg"] == pytest.approx(0.0, abs=1e-6)
    # assignment[i] = which GT axis predicted[i] matches.
    assert result["assignment"] == [2, 1, 0]


def test_match_axes_returns_gt_lengths_in_matched_order():
    """Codex P2 on PR #268 head 70f90ca: when the best assignment is
    not identity, `gt_axis_lengths_px` must be reordered to align with
    `predicted_axis_lengths_px` so the report's "compare predicted vs
    GT axis length" guidance lines up the right pairs.

    Setup: predicted axes deliberately ordered so the best assignment
    is [2, 1, 0] (reverse). GT has DISTINCT lengths per axis so the
    reorder is observable.
    """
    # GT in original order: lengths 10, 20, 30.
    gt = [(10.0, 0.0), (0.0, 20.0), (-30.0, 0.0)]
    # Predicted matches GT[2], GT[1], GT[0] in that order — same
    # directions, so the best assignment must be [2, 1, 0].
    predicted = [(-30.0, 0.0), (0.0, 20.0), (10.0, 0.0)]
    result = m._match_axes_to_ground_truth(predicted, gt)
    assert result["assignment"] == [2, 1, 0]
    # Predicted lengths in predicted order: 30, 20, 10.
    assert result["predicted_axis_lengths_px"] == [30.0, 20.0, 10.0]
    # GT lengths MATCHED to predicted-axis order: also 30, 20, 10
    # (because predicted[i] matches gt[assignment[i]]).
    assert result["gt_axis_lengths_px"] == [30.0, 20.0, 10.0], (
        "gt_axis_lengths_px must be reordered to align with the matched "
        "predicted axes, not left in raw GT order."
    )
    # The raw-GT-order field preserves the original [10, 20, 30] for
    # callers that need it.
    assert result["gt_axis_lengths_px_raw_order"] == [10.0, 20.0, 30.0]


def test_match_axes_chirality_flip_signature_around_180():
    """The 60° body-diagonal flip in the chirality detector produces
    axes that are systematically off by ~50-70° each. The TOTAL
    misfit across 3 axes after best permutation matching is ~178°
    — the signature pattern the diagnostic surfaces. This pins
    that interpretation so the threshold (~30° for clean, ~175° for
    broken) in the report stays meaningful."""
    # GT: axes at 90°, 210°, 330° (standard iso layout, pointing
    # down/up-left/up-right).
    gt = [
        (math.cos(math.radians(90.0)) * 100, math.sin(math.radians(90.0)) * 100),
        (math.cos(math.radians(210.0)) * 100, math.sin(math.radians(210.0)) * 100),
        (math.cos(math.radians(330.0)) * 100, math.sin(math.radians(330.0)) * 100),
    ]
    # Predicted: rotated 60° (the body-diagonal flip signature) — same
    # length but pointing at the OPPOSITE 3 hexagon corners.
    pred = [
        (math.cos(math.radians(150.0)) * 100, math.sin(math.radians(150.0)) * 100),
        (math.cos(math.radians(270.0)) * 100, math.sin(math.radians(270.0)) * 100),
        (math.cos(math.radians(30.0)) * 100, math.sin(math.radians(30.0)) * 100),
    ]
    result = m._match_axes_to_ground_truth(pred, gt)
    # Best permutation finds the minimum sum of 3 angles. Each
    # predicted axis is 60° from its nearest GT neighbor → total ~180°.
    assert 170.0 <= result["total_misfit_deg"] <= 190.0


# ---------------- ground-truth axes derivation ----------------


def test_ground_truth_axes_uses_one_edge_corners_for_side_a():
    """Per `ONE_EDGE_CORNERS_BY_SIDE['A']`, the single-axis hex
    corners on side A are corner_1, corner_3, corner_5. The derived
    GT axes must point from the vertex at those 3 corners."""
    truth_row = {
        "vertex": [100.0, 100.0],
        "corner_0": [999.0, 999.0],  # FAR corner; should NOT be used
        "corner_1": [200.0, 100.0],  # ONE_EDGE; should be used
        "corner_2": [999.0, 999.0],
        "corner_3": [100.0, 200.0],  # ONE_EDGE; should be used
        "corner_4": [999.0, 999.0],
        "corner_5": [50.0, 50.0],    # ONE_EDGE; should be used
    }
    vertex, axes = m._ground_truth_axes(truth_row, "A", scale=1.0)
    assert vertex == (100.0, 100.0)
    assert axes == [
        (100.0, 0.0),   # corner_1 - vertex
        (0.0, 100.0),   # corner_3 - vertex
        (-50.0, -50.0), # corner_5 - vertex
    ]


def test_ground_truth_axes_uses_one_edge_corners_for_side_b():
    """Side B's ONE_EDGE corners are corner_0, corner_2, corner_4."""
    truth_row = {
        "vertex": [0.0, 0.0],
        "corner_0": [10.0, 0.0],
        "corner_1": [999.0, 999.0],
        "corner_2": [0.0, 10.0],
        "corner_3": [999.0, 999.0],
        "corner_4": [-5.0, -5.0],
        "corner_5": [999.0, 999.0],
    }
    vertex, axes = m._ground_truth_axes(truth_row, "B", scale=1.0)
    assert vertex == (0.0, 0.0)
    assert axes == [
        (10.0, 0.0),
        (0.0, 10.0),
        (-5.0, -5.0),
    ]


def test_ground_truth_axes_applies_scale():
    """The processing-resolution image is downscaled from full-res by
    `scale`. GT corners come from full-res coords; tool must scale them
    down to match the model's processing-resolution output."""
    truth_row = {
        "vertex": [100.0, 100.0],
        "corner_0": [0.0, 0.0],
        "corner_1": [200.0, 100.0],
        "corner_2": [0.0, 0.0],
        "corner_3": [100.0, 200.0],
        "corner_4": [0.0, 0.0],
        "corner_5": [50.0, 50.0],
    }
    vertex_full, _ = m._ground_truth_axes(truth_row, "A", scale=1.0)
    vertex_half, axes_half = m._ground_truth_axes(truth_row, "A", scale=0.5)
    assert vertex_full == (100.0, 100.0)
    assert vertex_half == (50.0, 50.0)
    # Axes are differences, so they also scale.
    assert axes_half[0] == (50.0, 0.0)


# ---------------- visual-quality classification ----------------


def test_classify_face_for_row_returns_unknown_when_no_labels():
    """Rows with no labels in `_VISUAL_QUALITY_SAMPLES` classify as
    'unknown' (the un-eyeballed bucket)."""
    assert m._classify_face_for_row_visual("99_X", "corr_true") == "unknown"


def test_classify_face_for_row_returns_broken_if_any_face_broken():
    """The aggregation rule: 'broken' if any of 3 faces is broken,
    'clean' if all clean/decent, 'marginal' otherwise. Verifies the
    20_A corr_true row classifies as 'broken' since all 3 face labels
    in the committed sample are 'broken'."""
    assert m._classify_face_for_row_visual("20_A", "corr_true") == "broken"


def test_classify_face_for_row_returns_unknown_for_partial_non_broken_labels(monkeypatch):
    """A single decent/clean visual label should not promote the whole
    three-face row to clean."""
    monkeypatch.setitem(
        m._VISUAL_QUALITY_SAMPLES,
        "99_A:corr_true:face_yz",
        "decent",
    )
    assert m._classify_face_for_row_visual("99_A", "corr_true") == "unknown"


def test_classify_face_for_row_returns_clean_when_all_clean_or_decent():
    """41_A corr_false has labels {decent, clean, clean} → clean."""
    assert m._classify_face_for_row_visual("41_A", "corr_false") == "clean"


def test_classify_face_for_row_returns_marginal_when_mixed():
    """41_A corr_true has labels {decent, decent, marginal} →
    marginal (mixed clean/decent + marginal, no broken)."""
    assert m._classify_face_for_row_visual("41_A", "corr_true") == "marginal"


# ---------------- image path resolution / output safety ----------------


def test_resolve_image_path_falls_back_to_corpus_root(tmp_path):
    corpus_root = tmp_path / "cube-corpus"
    corpus_root.mkdir()
    image = corpus_root / "Set 20 - A - white up IMG_9999.JPG"
    image.write_bytes(b"fake image")

    stale_downloads_path = (
        "/Users/someone/Downloads/Set 20 - A - white up IMG_9999.JPG"
    )

    assert (
        m._resolve_image_path(stale_downloads_path, "20", "A", [corpus_root])
        == image
    )


def test_resolve_image_path_finds_sha_matching_candidate(tmp_path):
    """Codex P2 on PR #268 head 9a12e97: when the first-resolved path
    exists but has the WRONG content (different SHA), the resolver
    must check later candidates for the expected SHA before giving
    up. The pre-fix flow skipped immediately on the first mismatch,
    which broke canonical regeneration on setups where a stale
    same-named copy lived in `/Users/jhuber/Downloads` while the
    canonical bytes were in `~/cube-corpus`."""
    import hashlib
    canonical_payload = b"correct canonical image bytes"
    stale_payload = b"wrong/stale image bytes with different content"
    expected_sha = hashlib.sha256(canonical_payload).hexdigest()

    downloads_root = tmp_path / "Downloads"
    downloads_root.mkdir()
    corpus_root = tmp_path / "cube-corpus"
    corpus_root.mkdir()
    # First candidate (the raw path from the manifest) exists but has
    # the WRONG content.
    raw = downloads_root / "Set 20 - A - white up IMG_9999.JPG"
    raw.write_bytes(stale_payload)
    # Fallback candidate in corpus_root has the CORRECT content.
    canonical = corpus_root / "Set 20 - A - white up IMG_9999.JPG"
    canonical.write_bytes(canonical_payload)

    # Without expected SHA: returns first existing (the stale one) —
    # legacy behavior preserved.
    assert (
        m._resolve_image_path(str(raw), "20", "A", [downloads_root, corpus_root])
        == raw
    )
    # With expected SHA: skips the stale one, returns the canonical
    # one from corpus_root.
    assert (
        m._resolve_image_path(
            str(raw), "20", "A", [downloads_root, corpus_root],
            expected_sha256=expected_sha,
        )
        == canonical
    )


def test_resolve_image_path_walks_all_same_pattern_matches_in_one_root(tmp_path):
    """Codex P2 on PR #268 head a384965: when a single corpus root
    contains MULTIPLE files matching the same `Set N - SIDE -*`
    pattern (e.g. a stale duplicate sitting lex-before the canonical
    one), the resolver used to return only the lex-first via
    `_find_corpus_side`. With expected_sha256 set, the SHA check then
    rejected that lex-first file and skipped the row even though a
    valid file sat in the same directory.

    The fix uses `_find_corpus_sides` (plural) so all same-pattern
    matches participate in the SHA filter."""
    import hashlib
    canonical_payload = b"correct canonical bytes"
    stale_payload = b"wrong stale bytes"
    expected_sha = hashlib.sha256(canonical_payload).hexdigest()

    root = tmp_path / "Downloads"
    root.mkdir()
    # Two same-pattern files in the SAME directory. Lex-first is the
    # stale one (alphabetical: "001" < "999").
    stale = root / "Set 20 - A - white up IMG_001.JPG"
    stale.write_bytes(stale_payload)
    canonical = root / "Set 20 - A - white up IMG_999.JPG"
    canonical.write_bytes(canonical_payload)

    raw = "/Users/nobody/missing/Set 20 - A - white up IMG_999.JPG"
    # Without SHA: returns lex-first by name match (legacy behavior).
    # The raw path's filename matches `IMG_999`, so by_name=root/IMG_999
    # exists and gets returned first.
    assert (
        m._resolve_image_path(raw, "20", "A", [root])
        == canonical
    )
    # With expected SHA matching the canonical: still returns canonical.
    assert (
        m._resolve_image_path(raw, "20", "A", [root], expected_sha256=expected_sha)
        == canonical
    )
    # Now flip: raw points at the STALE file's name, so by_name would
    # return the stale one first. With expected_sha set, the resolver
    # must walk past it to the canonical match within the same root.
    raw_stale = "/Users/nobody/missing/Set 20 - A - white up IMG_001.JPG"
    assert (
        m._resolve_image_path(
            raw_stale, "20", "A", [root], expected_sha256=expected_sha,
        )
        == canonical
    )


def test_resolve_image_path_returns_none_when_no_sha_match(tmp_path):
    """If no candidate matches the expected SHA, resolver returns None
    (caller skips the row instead of tracing wrong pixels)."""
    import hashlib
    expected_sha = hashlib.sha256(b"never appears anywhere").hexdigest()

    downloads_root = tmp_path / "Downloads"
    downloads_root.mkdir()
    raw = downloads_root / "Set 20 - A.JPG"
    raw.write_bytes(b"wrong content")

    result = m._resolve_image_path(
        str(raw), "20", "A", [downloads_root],
        expected_sha256=expected_sha,
    )
    assert result is None


def test_default_output_blocker_rejects_empty_or_partial_trace():
    assert m._default_output_blocker({
        "per_row": [],
        "skipped": [],
    }) == "no rows were traced"
    assert m._default_output_blocker({
        "per_row": [{"status": "traced"}],
        "skipped": [{"key": "20_A", "reason": "missing"}],
    }) == "1 row(s) were skipped"
    assert m._default_output_blocker({
        "per_row": [{"status": "error"}],
        "skipped": [],
    }) == "no rows were traced"


def test_default_output_blocker_catches_hypothesis_level_errors():
    """Codex P2 #1 on PR #268 head 808fc10: when row-level status is
    'traced' but a hypothesis-level fit raised or returned None,
    `evaluate_one_row` records `corr_true.error` / `corr_false.error`
    while leaving the row status 'traced'. Default regeneration would
    happily overwrite the committed trace with missing axis metrics for
    the failed hypothesis. The blocker must surface this."""
    msg = m._default_output_blocker({
        "per_row": [
            {
                "key": "20_A",
                "status": "traced",
                "corr_true": {"axis_match": {"total_misfit_deg": 12.3}},
                "corr_false": {"error": "fit returned None"},
            },
        ],
        "skipped": [],
    })
    assert msg is not None
    assert "hypothesis fit" in msg
    assert "20_A:corr_false" in msg


def test_default_output_blocker_accepts_clean_traced_row_both_hypotheses():
    """Negative case: a row where BOTH hypotheses have axis_match and
    no error must NOT trip the blocker — that's the canonical clean
    case the diagnostic was built to produce."""
    msg = m._default_output_blocker({
        "per_row": [
            {
                "key": "20_A",
                "status": "traced",
                "corr_true": {"axis_match": {"total_misfit_deg": 12.3}},
                "corr_false": {"axis_match": {"total_misfit_deg": 14.1}},
            },
        ],
        "skipped": [],
    })
    assert msg is None


def test_default_output_blocker_rejects_non_default_truth(
    tmp_path, monkeypatch,
):
    """Codex P2 #1 on PR #268 head 988715b: a non-default `--truth`
    fixture (subset, superset, or replacement) must block writes to
    default committed outputs regardless of how successfully it traces
    — the canonical artifact is tied to a SPECIFIC truth fixture, and
    overwriting with any other source corrupts its provenance."""
    canonical_truth = tmp_path / "canonical_truth.json"
    canonical_truth.write_text(
        json.dumps({f"{i}_A": {"approved": True} for i in range(12)}),
        encoding="utf-8",
    )
    canonical_manifest = tmp_path / "canonical_manifest.json"
    canonical_manifest.write_text(json.dumps({"pairs": []}), encoding="utf-8")
    monkeypatch.setattr(m, "DEFAULT_TRUTH", canonical_truth)
    monkeypatch.setattr(m, "DEFAULT_MANIFEST", canonical_manifest)

    other_truth = tmp_path / "other_truth.json"
    other_truth.write_text(
        # Even a SUPERSET (more rows than canonical) must be rejected,
        # because the trace would still encode a different set of rows
        # under the canonical schema label.
        json.dumps({f"{i}_A": {"approved": True} for i in range(20)}),
        encoding="utf-8",
    )
    happy_payload = {
        "per_row": [
            {
                "key": f"{i}_A",
                "status": "traced",
                "corr_true": {"axis_match": {"total_misfit_deg": 12.3}},
                "corr_false": {"axis_match": {"total_misfit_deg": 14.1}},
            }
            for i in range(20)
        ],
        "skipped": [],
    }

    # Non-default truth with happy 20-row trace → still blocked.
    msg = m._default_output_blocker(
        happy_payload,
        truth_path=other_truth,
        manifest_path=canonical_manifest,
    )
    assert msg is not None
    assert "non-default --truth" in msg

    # Canonical truth + canonical manifest + happy trace → not blocked.
    happy_payload_canonical = {
        "per_row": happy_payload["per_row"][:12],
        "skipped": [],
    }
    msg2 = m._default_output_blocker(
        happy_payload_canonical,
        truth_path=canonical_truth,
        manifest_path=canonical_manifest,
    )
    assert msg2 is None


def test_default_output_blocker_rejects_non_default_max_image_dim(
    tmp_path, monkeypatch,
):
    """Codex P2 #1 on PR #268 head 7d90d30: vertex/axis coordinates in
    the trace are in processing-resolution pixels, which scale with
    `--max-image-dim`. A trace at dim=800 has half-coordinates relative
    to dim=1600, but the canonical schema is tied to dim=1600. The
    guard must reject any non-default dim from default outputs."""
    canonical_truth = tmp_path / "canonical_truth.json"
    canonical_truth.write_text(json.dumps({}), encoding="utf-8")
    canonical_manifest = tmp_path / "canonical_manifest.json"
    canonical_manifest.write_text(json.dumps({"pairs": []}), encoding="utf-8")
    monkeypatch.setattr(m, "DEFAULT_TRUTH", canonical_truth)
    monkeypatch.setattr(m, "DEFAULT_MANIFEST", canonical_manifest)
    monkeypatch.setattr(m, "DEFAULT_MAX_IMAGE_DIM", 1600)

    happy_payload = {
        "per_row": [
            {
                "key": "20_A",
                "status": "traced",
                "corr_true": {"axis_match": {"total_misfit_deg": 12.3}},
                "corr_false": {"axis_match": {"total_misfit_deg": 14.1}},
            },
        ],
        "skipped": [],
    }
    # Default dim → no block.
    msg_ok = m._default_output_blocker(
        happy_payload,
        truth_path=canonical_truth,
        manifest_path=canonical_manifest,
        max_image_dim=1600,
    )
    assert msg_ok is None
    # Non-default dim → block.
    msg_block = m._default_output_blocker(
        happy_payload,
        truth_path=canonical_truth,
        manifest_path=canonical_manifest,
        max_image_dim=800,
    )
    assert msg_block is not None
    assert "non-default --max-image-dim" in msg_block
    assert "800" in msg_block


def test_file_sha256_round_trips_on_known_content(tmp_path):
    """`_file_sha256` must compute the same SHA-256 as `hashlib.sha256`
    on the same bytes. Used by `run_all` to verify resolved image files
    match the manifest's expected SHA before tracing (Codex P2 #2 on
    PR #268 head 7d90d30 — without this check the fuzzy path resolver
    could land on a same-named file from a different corpus and the
    trace would contain measurements from the wrong pixels under the
    canonical schema)."""
    import hashlib
    payload = b"the quick brown fox jumps over the lazy dog\n" * 100
    p = tmp_path / "thing.bin"
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert m._file_sha256(p) == expected


def test_file_sha256_returns_none_on_missing_file(tmp_path):
    """Defensive: missing file → None (caller skips the row instead of
    crashing the whole run)."""
    assert m._file_sha256(tmp_path / "does-not-exist.bin") is None


def test_default_output_blocker_rejects_non_default_manifest(
    tmp_path, monkeypatch,
):
    """Codex P2 #2 on PR #268 head 988715b: even with canonical truth,
    a non-default `--manifest` (e.g. one pointing at /tmp images
    instead of the canonical corpus) must block writes to default
    committed outputs — the trace's images then come from a non-
    canonical source even though the row count and IDs match."""
    canonical_truth = tmp_path / "canonical_truth.json"
    canonical_truth.write_text(json.dumps({}), encoding="utf-8")
    canonical_manifest = tmp_path / "canonical_manifest.json"
    canonical_manifest.write_text(json.dumps({"pairs": []}), encoding="utf-8")
    monkeypatch.setattr(m, "DEFAULT_TRUTH", canonical_truth)
    monkeypatch.setattr(m, "DEFAULT_MANIFEST", canonical_manifest)

    other_manifest = tmp_path / "other_manifest.json"
    other_manifest.write_text(json.dumps({"pairs": []}), encoding="utf-8")

    happy_payload = {
        "per_row": [
            {
                "key": "20_A",
                "status": "traced",
                "corr_true": {"axis_match": {"total_misfit_deg": 12.3}},
                "corr_false": {"axis_match": {"total_misfit_deg": 14.1}},
            },
        ],
        "skipped": [],
    }
    msg = m._default_output_blocker(
        happy_payload,
        truth_path=canonical_truth,
        manifest_path=other_manifest,
    )
    assert msg is not None
    assert "non-default --manifest" in msg


def test_main_per_path_guard_protects_asymmetric_default(monkeypatch, tmp_path):
    """Codex P2 #2 on PR #268 head 808fc10: if `--out-md /tmp/x.md`
    (explicit) is passed but `--out-json` is left at default, the
    pre-fix check `_using_default_outputs` returned False because the
    pair isn't both defaults, and the default JSON still got clobbered.
    The per-path guard must protect EACH default path independently."""
    truth_path = tmp_path / "truth.json"
    manifest_path = tmp_path / "manifest.json"
    truth_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text('{"pairs": []}', encoding="utf-8")

    default_out_json = tmp_path / "committed_trace.json"
    default_out_md = tmp_path / "committed_report.md"
    default_out_json.write_text('{"sentinel": "do not overwrite"}', encoding="utf-8")
    default_out_md.write_text("sentinel report — do not overwrite", encoding="utf-8")
    monkeypatch.setattr(m, "DEFAULT_OUT_JSON", default_out_json)
    monkeypatch.setattr(m, "DEFAULT_OUT_MD", default_out_md)

    explicit_md = tmp_path / "exploratory.md"
    monkeypatch.setattr(
        m,
        "run_all",
        lambda *a, **k: {
            "schema": "axis_correctness_v1",
            "source": {},
            "per_row": [],
            "skipped": [{"key": "20_A", "reason": "missing"}],
        },
    )

    rc = m.main([
        "--truth", str(truth_path),
        "--manifest", str(manifest_path),
        "--out-md", str(explicit_md),
        # NOTE: --out-json LEFT AT DEFAULT
    ])
    # Default JSON path must be protected even though --out-md is
    # explicit. Exit code 2.
    assert rc == 2
    # Sentinel must be preserved.
    assert default_out_json.read_text(encoding="utf-8") == (
        '{"sentinel": "do not overwrite"}'
    )
    # Explicit md path must NOT have been written either (we exit before
    # writing anything when any default-path guard fires).
    assert not explicit_md.exists()
    assert default_out_md.read_text(encoding="utf-8") == (
        "sentinel report — do not overwrite"
    )


def test_main_per_path_guard_does_not_block_fully_explicit_paths(
    monkeypatch, tmp_path,
):
    """Mirror of the asymmetric case: when BOTH --out-json and --out-md
    are explicit (non-default), the guard must let the (potentially
    partial) write through. This is the documented exploratory escape
    hatch."""
    truth_path = tmp_path / "truth.json"
    manifest_path = tmp_path / "manifest.json"
    truth_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text('{"pairs": []}', encoding="utf-8")

    # Make the defaults exist + carry sentinels, but USE explicit paths
    # for both outputs.
    default_out_json = tmp_path / "committed_trace.json"
    default_out_md = tmp_path / "committed_report.md"
    default_out_json.write_text("{}", encoding="utf-8")
    default_out_md.write_text("", encoding="utf-8")
    monkeypatch.setattr(m, "DEFAULT_OUT_JSON", default_out_json)
    monkeypatch.setattr(m, "DEFAULT_OUT_MD", default_out_md)

    explicit_json = tmp_path / "explore.json"
    explicit_md = tmp_path / "explore.md"
    monkeypatch.setattr(
        m,
        "run_all",
        lambda *a, **k: {
            "schema": "axis_correctness_v1",
            "source": {},
            "per_row": [],
            "skipped": [{"key": "20_A", "reason": "missing"}],
        },
    )

    rc = m.main([
        "--truth", str(truth_path),
        "--manifest", str(manifest_path),
        "--out-json", str(explicit_json),
        "--out-md", str(explicit_md),
    ])
    assert rc == 0
    assert explicit_json.exists()
    assert explicit_md.exists()


def test_main_refuses_default_output_when_trace_is_incomplete(monkeypatch, tmp_path):
    truth_path = tmp_path / "truth.json"
    manifest_path = tmp_path / "manifest.json"
    truth_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text('{"pairs": []}', encoding="utf-8")

    out_json = tmp_path / "axis.json"
    out_md = tmp_path / "axis.md"
    monkeypatch.setattr(m, "DEFAULT_OUT_JSON", out_json)
    monkeypatch.setattr(m, "DEFAULT_OUT_MD", out_md)
    monkeypatch.setattr(
        m,
        "run_all",
        lambda *args, **kwargs: {
            "schema": "axis_correctness_v1",
            "source": {},
            "per_row": [],
            "skipped": [{"key": "20_A", "reason": "missing"}],
        },
    )

    rc = m.main(["--truth", str(truth_path), "--manifest", str(manifest_path)])
    assert rc == 2
    assert not out_json.exists()
    assert not out_md.exists()


# ---------------- report rendering ----------------


def test_render_report_includes_per_row_table_and_cross_reference():
    """End-to-end render: feed a minimal payload with one traced row
    + one untraced row, verify the markdown structure."""
    payload = {
        "source": {
            "tool": "tools/measure_axis_correctness.py",
            "truth": "tests/fixtures/full_corner_ground_truth.json",
            "manifest": "tests/fixtures/corpus_manifest.json",
            "max_image_dim": 1600,
            "run_selection": "single deterministic run per row/hypothesis",
        },
        "per_row": [
            {
                "key": "20_A",
                "side": "A",
                "yaw_quarter_turns": 0,
                "status": "traced",
                "corr_true": {
                    "vertex_error_processing_px": 44.2,
                    "flip_applied": False,
                    "axis_match": {
                        "total_misfit_deg": 177.4,
                        "per_axis_angle_errors_deg": [63.6, 56.3, 57.5],
                        "predicted_axis_lengths_px": [408.9, 411.4, 410.8],
                        "gt_axis_lengths_px": [502.2, 590.4, 547.0],
                    },
                },
                "corr_false": {
                    "vertex_error_processing_px": 44.2,
                    "flip_applied": False,
                    "axis_match": {
                        "total_misfit_deg": 177.4,
                        "per_axis_angle_errors_deg": [63.6, 56.3, 57.5],
                        "predicted_axis_lengths_px": [408.9, 411.4, 410.8],
                        "gt_axis_lengths_px": [502.2, 590.4, 547.0],
                    },
                },
            },
            {
                "key": "99_X",
                "status": "error",
                "error": "rembg failed: synthetic test error",
            },
        ],
    }
    md = m.render_report(payload)
    assert "# Axis-correctness diagnostic" in md
    assert "`20_A`" in md
    assert "corr_true" in md
    assert "## Source" in md
    assert "the the" not in md
    # Untraced row error message surfaces.
    assert "ERR" in md and "synthetic test error" in md
    # Cross-reference section exists and includes the broken bucket
    # (since 20_A corr_true has _VISUAL_QUALITY_SAMPLES entries
    # tagged 'broken').
    assert "Cross-reference" in md


# ---------------- committed trace sanity ----------------


def test_committed_trace_matches_expected_shape():
    """The trace JSON committed under tests/fixtures/ is the canonical
    post-#267 measurement. Pin its structure so a future refactor
    that breaks the schema fails loudly here."""
    import json
    trace_path = (
        REPO_ROOT / "tests" / "fixtures" / "axis_correctness_trace.json"
    )
    if not trace_path.exists():
        pytest.skip(
            "axis_correctness_trace.json not committed — run "
            "`python tools/measure_axis_correctness.py` to regenerate."
        )
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload.get("schema") == "axis_correctness_v1"
    assert payload.get("source", {}).get("tool") == "tools/measure_axis_correctness.py"
    assert (
        payload.get("source", {}).get("truth")
        == "tests/fixtures/full_corner_ground_truth.json"
    )
    assert payload.get("source", {}).get("max_image_dim") == 1600
    assert isinstance(payload.get("per_row"), list)
    assert len(payload["per_row"]) >= 10, (
        "Expected ~12 rows (all approved full-corner-ground-truth rows); "
        f"got {len(payload['per_row'])}"
    )
    # Spot-check: at least one row should have both hypotheses
    # populated.
    has_both = any(
        r.get("status") == "traced"
        and "corr_true" in r
        and "corr_false" in r
        for r in payload["per_row"]
    )
    assert has_both, (
        "no row has both corr_true and corr_false populated — trace "
        "appears malformed"
    )


# ---------------- provenance helpers ----------------


def test_git_head_sha_returns_a_sha_in_a_checkout():
    """When the tool runs in a git checkout, the source block should
    record the current commit SHA so a future reader can locate the
    exact code that produced the trace. Skips if `git rev-parse HEAD`
    can't run for any reason (e.g. running outside any checkout)."""
    sha = m._git_head_sha()
    if sha is None:
        pytest.skip("git rev-parse HEAD unavailable (no checkout?)")
    # 40-char hex SHA — short SHAs would be at least 7. Pin shape so
    # an accidental string substitution catches.
    assert isinstance(sha, str) and len(sha) >= 7
    assert all(c in "0123456789abcdef" for c in sha.lower())


def test_now_utc_iso_returns_iso_8601_utc_string():
    """The generated_at_utc field is an ISO-8601 timestamp anchored at
    UTC. Pin the shape so consumers can rely on parsing it."""
    ts = m._now_utc_iso()
    import datetime as _dt
    # round-trips via fromisoformat — proves it's valid ISO 8601 and
    # confirms the timezone is UTC (not local).
    parsed = _dt.datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == _dt.timedelta(0)
