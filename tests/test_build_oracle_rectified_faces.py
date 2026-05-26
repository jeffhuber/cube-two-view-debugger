"""Unit tests for `tools/build_oracle_rectified_faces.py`.

The load-bearing piece is the (side, slot, yaw) -> `[TL, TR, BR, BL]`
rectification quad order derivation. The 6 yaw=0 cases below are pinned
against the worked tables in `tools/ORACLE_RECTIFIED_FACES_DESIGN.md`,
and 2 yaw=1 cases pin the rotation behavior the design doc calls out.

Sticker numbering, output-path uniqueness, the color-output contract
(reuse `rgb_to_hsv` / `rgb_to_lab` / `classify_rgb` from
`rubik_recognizer.colors`), a perspective round-trip, and an
end-to-end smoke test cover the rest of the test plan.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import build_oracle_rectified_faces as orf  # noqa: E402


# --------------- yaw=0 pinning (6 cases) ---------------


@pytest.mark.parametrize(
    "side,slot,expected",
    [
        # Image A's 3 faces at yaw=0 — design doc table.
        ("A", "upper", ("corner_0", "corner_1", "vertex", "corner_5")),
        ("A", "right", ("vertex", "corner_1", "corner_2", "corner_3")),
        ("A", "front", ("corner_5", "vertex", "corner_3", "corner_4")),
        # Image B's 3 faces at yaw=0 — design doc table.
        ("B", "upper", ("corner_4", "corner_3", "corner_2", "vertex")),
        ("B", "right", ("corner_1", "corner_0", "vertex", "corner_2")),
        ("B", "front", ("corner_0", "corner_5", "corner_4", "vertex")),
    ],
)
def test_rectification_quad_labels_yaw_zero(side: str, slot: str, expected):
    """Pins the (side, slot) -> [TL, TR, BR, BL] mapping at yaw=0
    against the worked tables in tools/ORACLE_RECTIFIED_FACES_DESIGN.md."""
    got = orf._rectification_quad_labels_for(side, slot, 0)
    assert got == expected


# --------------- yaw integration (2+ cases) ---------------


def test_rectification_quad_labels_yaw_one_rotates_upper_A():
    """Design doc: at A/upper yaw=1, the outline labels map to
    (U3, U1, U7, U9), so the rectification order shifts to
    (corner_1, vertex, corner_5, corner_0)."""
    got = orf._rectification_quad_labels_for("A", "upper", 1)
    assert got == ("corner_1", "vertex", "corner_5", "corner_0")


def test_rectification_quad_labels_yaw_one_upper_B_rotates_corner_labels():
    """For B/upper at yaw=1: the slot's WCA face name is still D (the
    face NAME is yaw-invariant for top/bottom), BUT the physical
    corner labels touching D's TL/TR/BR/BL positions rotate one
    quarter turn around the cube. At yaw=0 the order is
    (corner_4, corner_3, corner_2, vertex); at yaw=1 it shifts to
    (vertex, corner_4, corner_3, corner_2) — a clockwise rotation
    of the same 4 labels, matching the cube's physical rotation."""
    yaw0 = orf._rectification_quad_labels_for("B", "upper", 0)
    yaw1 = orf._rectification_quad_labels_for("B", "upper", 1)
    assert yaw0 == ("corner_4", "corner_3", "corner_2", "vertex")
    assert yaw1 == ("vertex", "corner_4", "corner_3", "corner_2"), (
        "B/upper yaw=1 should rotate the corner labels one position "
        "clockwise from yaw=0, reflecting the cube's physical rotation."
    )
    # The labels are the same set, just rotated.
    assert set(yaw0) == set(yaw1)


def test_yaw_changes_facelet_ids_even_when_quad_labels_match():
    """Yaw doesn't just rename output files. For a yaw-invariant face
    like D (B/upper), the LABEL order is identical to yaw=0, but the
    9 sticker facelet IDs are unchanged because the face IS D at any
    yaw. Pin: yaw rotates which face name a side-face slot shows
    (R -> B -> L -> F -> R), but U and D stay put."""
    a_upper_yaw0 = orf._facelet_ids_for_slot("A", "upper", 0)
    a_upper_yaw1 = orf._facelet_ids_for_slot("A", "upper", 1)
    # U is yaw-invariant — A/upper is always U.
    assert a_upper_yaw0 == [f"U{i}" for i in range(1, 10)]
    assert a_upper_yaw1 == [f"U{i}" for i in range(1, 10)]
    # But A/right rotates: R -> B -> L -> F per yaw=0/1/2/3.
    a_right_yaw0 = orf._facelet_ids_for_slot("A", "right", 0)
    a_right_yaw1 = orf._facelet_ids_for_slot("A", "right", 1)
    assert a_right_yaw0[0].startswith("R")
    assert a_right_yaw1[0].startswith("B"), (
        "yaw=1 should rotate A/right slot to WCA face B, not R"
    )


# --------------- sticker numbering ---------------


@pytest.mark.parametrize(
    "row,col,expected",
    [
        (0, 0, 1), (0, 1, 2), (0, 2, 3),
        (1, 0, 4), (1, 1, 5), (1, 2, 6),
        (2, 0, 7), (2, 1, 8), (2, 2, 9),
    ],
)
def test_sticker_id_row_col_urfdlb(row: int, col: int, expected: int):
    """Standard URFDLB: 1 2 3 / 4 5 6 / 7 8 9 row-major."""
    assert orf.sticker_id_from_row_col(row, col) == expected


# --------------- color output contract ---------------


def test_color_output_imports_from_rubik_recognizer_colors():
    """Per design doc: tool must reuse `rgb_to_hsv`, `rgb_to_lab`,
    `classify_rgb` from `rubik_recognizer.colors` rather than
    duplicate constants/math."""
    import inspect
    import rubik_recognizer.colors as colors

    # The tool's symbols must be the canonical implementations.
    assert orf.rgb_to_hsv is colors.rgb_to_hsv, (
        "tool must reuse rgb_to_hsv from rubik_recognizer.colors"
    )
    assert orf.rgb_to_lab is colors.rgb_to_lab, (
        "tool must reuse rgb_to_lab from rubik_recognizer.colors"
    )
    assert orf.classify_rgb is colors.classify_rgb, (
        "tool must reuse classify_rgb from rubik_recognizer.colors"
    )

    # Sanity check on the imported functions' shape — guards against
    # accidental upstream API changes that would break our usage.
    sig_hsv = inspect.signature(orf.rgb_to_hsv)
    assert "rgb" in sig_hsv.parameters


# --------------- perspective round-trip ---------------


def test_rectify_face_oracle_round_trips_on_axis_aligned_square():
    """Warping a face whose source quad IS already a square should
    return the same pixels (identity mapping)."""
    color = (200, 50, 100)
    src = Image.new("RGB", (60, 60), color)
    quad = [(0, 0), (60, 0), (60, 60), (0, 60)]
    out = orf.rectify_face_oracle(src, quad, output_size=60)
    arr = np.asarray(out)
    # Allow a 1-channel slack from bicubic interpolation at the edge.
    median = tuple(int(np.median(arr.reshape(-1, 3)[:, ch])) for ch in range(3))
    assert all(abs(median[ch] - color[ch]) <= 2 for ch in range(3))


def test_sample_stickers_oracle_recovers_face_colors_on_synthetic_grid():
    """Build a synthetic 3x3 face with distinct colors per cell. After
    sampling, the median RGB per cell should match the painted color."""
    size = 300
    cell = size // 3
    palette = [
        [(255, 255, 255), (255, 255, 0), (255, 0, 0)],
        [(255, 128, 0), (0, 255, 0), (0, 0, 255)],
        [(120, 120, 120), (200, 200, 200), (50, 50, 50)],
    ]
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(3):
        for c in range(3):
            arr[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = palette[r][c]
    face = Image.fromarray(arr)
    # A/upper at yaw=0 -> facelet IDs U1..U9 row-major.
    samples = orf.sample_stickers_oracle(face, "A", "upper", 0)
    assert len(samples) == 9
    for sample in samples:
        # rgb should closely match the synthetic palette.
        expected = palette[sample.row][sample.col]
        for ch in range(3):
            assert abs(sample.rgb[ch] - expected[ch]) <= 2, (
                f"sticker ({sample.row},{sample.col}) ch{ch}: "
                f"got {sample.rgb}, expected {expected}"
            )
        # facelet_id matches A/upper yaw=0 -> U{sticker_id}.
        assert sample.facelet_id == f"U{sample.sticker_id}"


# --------------- output-path uniqueness ---------------


def test_observation_id_is_globally_unique_across_fixture_observations():
    """The full-corner rows x 3 faces x 9 stickers observations must NOT
    collapse to 54 distinct facelet IDs in the output paths — repeated
    observations of the same WCA facelet across rows MUST have unique
    paths or they overwrite each other. Pin the `key_facelet` shape."""
    seen = set()
    truth = json.loads(orf.DEFAULT_TRUTH.read_text(encoding="utf-8"))
    truth_keys = sorted(key for key, row in truth.items() if row.get("approved"))
    for key in truth_keys:
        _, side = orf._row_key_set_side(key)
        for slot in orf.SLOTS:
            # yaw doesn't matter for this uniqueness check — pick 0.
            facelet_ids = orf._facelet_ids_for_slot(side, slot, 0)
            for facelet_id in facelet_ids:
                observation_id = f"{key}_{facelet_id}"
                assert observation_id not in seen, (
                    f"duplicate observation_id {observation_id!r}"
                )
                seen.add(observation_id)
    assert len(seen) == len(truth_keys) * 27


# --------------- end-to-end smoke ---------------


def _make_synthetic_truth_and_image(tmp_path: Path):
    """Build a tiny synthetic image + truth fixture for end-to-end smoke
    test (avoids the rembg dep and avoids needing real photos)."""
    img_size = 600
    image = Image.new("RGB", (img_size, img_size), (40, 40, 40))
    # Paint a fake "face" centered, 200x200, distinct colors.
    arr = np.asarray(image).copy()
    fx0, fy0 = 200, 200
    cell = 200 // 3
    palette = [
        [(255, 255, 255), (255, 255, 0), (255, 0, 0)],
        [(255, 128, 0), (0, 255, 0), (0, 0, 255)],
        [(255, 255, 255), (255, 255, 0), (255, 0, 0)],
    ]
    for r in range(3):
        for c in range(3):
            y0 = fy0 + r * cell
            x0 = fx0 + c * cell
            arr[y0:y0 + cell, x0:x0 + cell] = palette[r][c]
    Image.fromarray(arr).save(tmp_path / "fake.jpg", quality=95)
    # Construct an A-side row where the "upper" face occupies the
    # painted square. To do this, we need to fill in vertex,
    # corner_0..5 such that FACE_DEFS_BY_SIDE A/upper -> 4 corners
    # of the painted square in rectification order. We'll set every
    # point to one of the 4 corners of the painted square, and just
    # rely on the per-face quad lookup to pick the right ones.
    # Rectification labels at A/upper yaw=0 = (corner_0, corner_1,
    # vertex, corner_5) in [TL, TR, BR, BL].
    truth = {
        "99_A": {
            "approved": True,
            "yaw_quarter_turns": 0,
            "vertex": [fx0 + 200, fy0 + 200],     # BR of painted square
            "corner_0": [fx0, fy0],                # TL of painted square
            "corner_1": [fx0 + 200, fy0],          # TR of painted square
            "corner_2": [fx0 + 200, fy0 + 200],
            "corner_3": [fx0 + 200, fy0 + 200],
            "corner_4": [fx0, fy0 + 200],
            "corner_5": [fx0, fy0 + 200],          # BL of painted square
        }
    }
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth))
    manifest = {
        "pairs": [
            {"setId": "99", "imageAPath": str(tmp_path / "fake.jpg")},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return truth_path, manifest_path


def test_build_all_end_to_end_smoke(tmp_path: Path):
    """Run the full pipeline on a synthetic 1-row fixture (no rembg
    dep, no real photo). Verifies that:
    - index.json schema is well-formed
    - face PNG written + readable
    - per-sticker PNGs written + observation_id-unique paths
    - colors round-trip for the A/upper face
    """
    truth_path, manifest_path = _make_synthetic_truth_and_image(tmp_path)
    out_dir = tmp_path / "out"
    index = orf.build_all(
        truth_path=truth_path,
        manifest_path=manifest_path,
        out_root=out_dir,
        face_size=120,
        patch_fraction=0.4,
        yaw_overrides={},
        save_patches=True,
        rows_glob="99_A",
    )
    assert index["schema"] == "oracle_rectified_faces_v1"
    assert len(index["rows"]) == 1
    row_record = index["rows"][0]
    assert row_record["key"] == "99_A"
    assert row_record["yaw_quarter_turns"] == 0
    assert len(row_record["faces"]) == 3
    upper_face = next(f for f in row_record["faces"] if f["slot"] == "upper")
    assert upper_face["wca_face"] == "U"
    # All 9 sticker PNGs + face PNG should exist on disk.
    face_png = out_dir / upper_face["face_png"]
    assert face_png.exists()
    Image.open(face_png).close()  # readable
    for sticker in upper_face["stickers"]:
        assert (out_dir / sticker["sticker_png"]).exists()
        assert (out_dir / sticker["patch_png"]).exists()
        # Grouped comparison view exists.
        grouped_png = (
            out_dir / "by_facelet" / sticker["facelet_id"] / "99_A.png"
        )
        assert grouped_png.exists()
        # observation_id is row+facelet (the load-bearing uniqueness key).
        assert sticker["observation_id"] == f"99_A_{sticker['facelet_id']}"
    # Top-level index + gallery written.
    assert (out_dir / "index.json").exists()
    assert (out_dir / "gallery.html").exists()
    # Colors on the upper face should approximately match the synthetic
    # palette (allowing perspective-warp + quantization slack).
    expected_first_row = [
        (255, 255, 255), (255, 255, 0), (255, 0, 0),
    ]
    for sticker, expected in zip(
        upper_face["stickers"][:3], expected_first_row
    ):
        rgb = tuple(sticker["rgb"])
        for ch in range(3):
            assert abs(rgb[ch] - expected[ch]) <= 12, (
                f"sticker {sticker['facelet_id']} ch{ch}: "
                f"got {rgb}, expected ~{expected}"
            )


# --------------- yaw overrides parsing ---------------


def test_parse_yaw_overrides_basic():
    assert orf._parse_yaw_overrides("") == {}
    assert orf._parse_yaw_overrides("20:1") == {"20": 1}
    assert orf._parse_yaw_overrides("20:1,38:0") == {"20": 1, "38": 0}
    # Whitespace forgiving.
    assert orf._parse_yaw_overrides(" 20 : 1 , 38 : 0 ") == {"20": 1, "38": 0}


def test_parse_yaw_overrides_rejects_malformed():
    with pytest.raises(ValueError):
        orf._parse_yaw_overrides("20=1")


def test_row_yaw_fallback_to_zero_when_field_missing():
    """If neither override nor `yaw_quarter_turns` is present, default
    to 0 with the assumed-zero flag set."""
    yaw, assumed = orf._row_yaw({}, "99", {})
    assert (yaw, assumed) == (0, True)


def test_row_yaw_uses_override_in_preference_to_fixture():
    yaw, assumed = orf._row_yaw(
        {"yaw_quarter_turns": 2}, "20", {"20": 1}
    )
    assert (yaw, assumed) == (1, False)


def test_row_yaw_uses_fixture_when_no_override():
    yaw, assumed = orf._row_yaw({"yaw_quarter_turns": 3}, "20", {})
    assert (yaw, assumed) == (3, False)


# --------------- Codex P2 follow-ups on PR #259 ---------------


def test_sticker_png_is_full_cell_crop_not_central_patch(tmp_path: Path):
    """by_observation/{key}/{facelet_id}.png must be the FULL sticker
    cell (cell-sized, with surrounding bezel context), NOT the central
    sampling patch. Conflating them — as the first version of this
    tool did — leaves consumers following `sticker_png` without the
    per-sticker context the field name implies. Codex P2 on PR #259.

    Test strategy: at face_size=120 and patch_fraction=0.40, the cell
    is 40px square and the central patch is ~16px square. The two
    sizes are far enough apart that a typo regression would be obvious.
    """
    truth_path, manifest_path = _make_synthetic_truth_and_image(tmp_path)
    out_dir = tmp_path / "out"
    orf.build_all(
        truth_path=truth_path,
        manifest_path=manifest_path,
        out_root=out_dir,
        face_size=120,
        patch_fraction=0.4,
        yaw_overrides={},
        save_patches=True,
        rows_glob="99_A",
    )
    sticker_png = out_dir / "by_observation" / "99_A" / "U1.png"
    patch_png = out_dir / "patch_png" / "99_A_U1.png"
    assert sticker_png.exists()
    assert patch_png.exists()
    with Image.open(sticker_png) as img:
        sticker_size = img.size
    with Image.open(patch_png) as img:
        patch_size = img.size
    # Cell is face_size/3 = 40 px; sticker_png must be ~that size.
    assert sticker_size[0] >= 35, (
        f"sticker_png should be the full cell (~40 px at face_size=120); "
        f"got {sticker_size}. Regression: are by_observation and "
        f"patch_png both using the central patch?"
    )
    # Patch is patch_fraction * cell = 16 px; patch_png must be smaller.
    assert patch_size[0] < sticker_size[0], (
        f"patch_png ({patch_size}) must be SMALLER than sticker_png "
        f"({sticker_size}); they should not be the same image."
    )


def test_clean_output_root_removes_owned_subdirs_only(tmp_path: Path):
    """clean_output_root should wipe by_row, by_observation, by_facelet,
    patch_png, index.json, gallery.html — and leave any unrelated
    sibling content alone (so the default /tmp/<...>/ root is safe to
    wipe without nuking unrelated directories someone might have left
    there). Codex P2 on PR #259."""
    out_root = tmp_path / "out"
    out_root.mkdir()
    # Owned content the tool wrote on a previous run.
    for sub in ("by_row", "by_observation", "by_facelet", "patch_png"):
        (out_root / sub).mkdir()
        (out_root / sub / "stale.png").write_bytes(b"stale")
    (out_root / "index.json").write_text("{}")
    (out_root / "gallery.html").write_text("<html></html>")
    # Unrelated sibling content the tool must NOT touch.
    (out_root / "notes.md").write_text("hand-written notes")
    (out_root / "sibling").mkdir()
    (out_root / "sibling" / "important.txt").write_text("important")
    orf.clean_output_root(out_root)
    # Owned content gone.
    for sub in ("by_row", "by_observation", "by_facelet", "patch_png"):
        assert not (out_root / sub).exists(), (
            f"{sub}/ should have been removed but still exists"
        )
    assert not (out_root / "index.json").exists()
    assert not (out_root / "gallery.html").exists()
    # Unrelated content preserved.
    assert (out_root / "notes.md").exists()
    assert (out_root / "sibling" / "important.txt").exists()


def test_clean_output_root_is_idempotent_when_root_does_not_exist(tmp_path: Path):
    """Calling clean_output_root on a non-existent root must not raise."""
    nowhere = tmp_path / "does-not-exist"
    orf.clean_output_root(nowhere)  # no error
    assert not nowhere.exists()


def test_build_all_does_not_leave_stale_artifacts_from_prior_wider_run(
    tmp_path: Path,
):
    """End-to-end: run with --rows-glob '*', then rerun with a narrower
    glob; the narrower run must not leave stale by_row/ entries from
    the wider run. Codex P2 on PR #259."""
    truth_path, manifest_path = _make_synthetic_truth_and_image(tmp_path)
    # Add a 2nd row to the truth so the "wider" run has something the
    # "narrower" run won't include.
    truth = json.loads(truth_path.read_text())
    # 99_B at the same coordinates (synthetic).
    truth["99_B"] = dict(truth["99_A"])
    truth_path.write_text(json.dumps(truth))
    manifest = json.loads(manifest_path.read_text())
    manifest["pairs"][0]["imageBPath"] = manifest["pairs"][0]["imageAPath"]
    manifest_path.write_text(json.dumps(manifest))
    out_dir = tmp_path / "out"
    # Wider first run: both rows.
    orf.build_all(
        truth_path=truth_path, manifest_path=manifest_path,
        out_root=out_dir, face_size=60, patch_fraction=0.4,
        yaw_overrides={}, save_patches=True, rows_glob="*",
    )
    assert (out_dir / "by_row" / "99_A").exists()
    assert (out_dir / "by_row" / "99_B").exists()
    # Narrower rerun: only 99_A. 99_B's stale artifacts MUST be gone.
    orf.build_all(
        truth_path=truth_path, manifest_path=manifest_path,
        out_root=out_dir, face_size=60, patch_fraction=0.4,
        yaw_overrides={}, save_patches=True, rows_glob="99_A",
    )
    assert (out_dir / "by_row" / "99_A").exists()
    assert not (out_dir / "by_row" / "99_B").exists(), (
        "99_B/ from the wider run should have been wiped before the "
        "narrower rerun wrote 99_A."
    )


def test_build_all_skips_rows_when_manifest_image_hash_mismatches(
    tmp_path: Path,
):
    """If the manifest pins an expected image hash, the oracle tool must
    not apply human corner labels to drifted/replaced pixels."""
    truth_path, manifest_path = _make_synthetic_truth_and_image(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["pairs"][0]["imageA_sha256_expected"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    out_dir = tmp_path / "out"
    index = orf.build_all(
        truth_path=truth_path,
        manifest_path=manifest_path,
        out_root=out_dir,
        face_size=60,
        patch_fraction=0.4,
        yaw_overrides={},
        save_patches=True,
        rows_glob="99_A",
    )
    assert index["rows"] == []
    assert len(index["skipped"]) == 1
    assert index["skipped"][0]["key"] == "99_A"
    assert "image hash mismatch" in index["skipped"][0]["reason"]
    assert not (out_dir / "by_row" / "99_A").exists()
    assert not (out_dir / "by_observation" / "99_A").exists()
    assert not list((out_dir / "patch_png").glob("99_A_*.png"))


def test_build_all_cleans_partial_artifacts_for_skipped_rows(tmp_path: Path):
    """If a row errors after writing an earlier face, no current-run
    filesystem artifacts for that skipped key should survive outside
    index.json."""
    truth_path, manifest_path = _make_synthetic_truth_and_image(tmp_path)
    truth = json.loads(truth_path.read_text())
    # A/upper can still write; A/right then needs corner_2 and fails.
    del truth["99_A"]["corner_2"]
    truth_path.write_text(json.dumps(truth))
    out_dir = tmp_path / "out"
    index = orf.build_all(
        truth_path=truth_path,
        manifest_path=manifest_path,
        out_root=out_dir,
        face_size=60,
        patch_fraction=0.4,
        yaw_overrides={},
        save_patches=True,
        rows_glob="99_A",
    )
    assert index["rows"] == []
    assert len(index["skipped"]) == 1
    assert index["skipped"][0]["key"] == "99_A"
    assert not (out_dir / "by_row" / "99_A").exists()
    assert not (out_dir / "by_observation" / "99_A").exists()
    assert not list((out_dir / "patch_png").glob("99_A_*.png"))
    by_facelet = out_dir / "by_facelet"
    if by_facelet.exists():
        assert not list(by_facelet.glob("*/99_A.png"))
