from collections import Counter
from types import SimpleNamespace

from tools.diagnose_background import (
    collect_rows,
    dominant_grid_center_face,
    filter_preview,
    hull_viability,
    ordered_face_counts,
    roi_fraction,
)


def test_roi_fraction_uses_processing_dimensions():
    assert roi_fraction((10, 20, 60, 120), 100, 200) == 0.25


def test_dominant_grid_center_face_uses_face_order_for_ties():
    analysis = SimpleNamespace(grids=[_grid("R"), _grid("U"), _grid("R"), _grid("U")])

    summary = dominant_grid_center_face(analysis)

    assert summary == {
        "face": "U",
        "count": 2,
        "counts": {"U": 2, "R": 2},
    }


def test_filter_preview_flags_anchor_missing_and_weak_hulls(monkeypatch):
    monkeypatch.setattr(
        "tools.diagnose_background.selected_anchor_detail",
        lambda analysis, anchor: {"present": False},
    )

    preview = filter_preview({"kept": 12, "dropped": 18}, SimpleNamespace(), "U")

    assert preview == {
        "wouldKeep": 12,
        "wouldDrop": 18,
        "keptFraction": 0.4,
        "viability": "anchor_missing",
    }


def test_hull_viability_requires_anchor_and_enough_kept_stickers():
    assert hull_viability(22, 3, anchor_present=True) == "candidate"
    assert hull_viability(22, 0, anchor_present=True) == "no_partition"
    assert hull_viability(12, 18, anchor_present=True) == "too_few_kept"
    assert hull_viability(22, 3, anchor_present=False) == "anchor_missing"


def test_ordered_face_counts_preserves_cube_face_order_then_unknowns():
    counts = Counter({"Z": 2, "B": 1, "U": 3})

    assert ordered_face_counts(counts) == {"U": 3, "B": 1, "Z": 2}


def test_collect_rows_defaults_to_background_noise_and_adds_controls(tmp_path):
    hard_manifest = tmp_path / "hard.json"
    corpus_manifest = tmp_path / "corpus.json"
    hard_manifest.write_text(
        """
        {
          "pairs": [
            {"setId": "46", "failureClass": "background_sticker_noise"},
            {"setId": "17", "failureClass": "image_b_visible_face_evidence_weak"}
          ]
        }
        """,
        encoding="utf-8",
    )
    corpus_manifest.write_text(
        """
        {
          "pairs": [
            {"setId": "15"},
            {"setId": "24"}
          ]
        }
        """,
        encoding="utf-8",
    )

    rows = collect_rows(
        hard_case_manifest=hard_manifest,
        corpus_manifest=corpus_manifest,
        set_ids=[],
        control_set_ids=["15"],
    )

    assert [(row["setId"], source, manifest) for row, manifest, source in rows] == [
        ("46", "hard-case", hard_manifest),
        ("15", "control", corpus_manifest),
    ]


def _grid(face):
    return SimpleNamespace(center_face=face)
