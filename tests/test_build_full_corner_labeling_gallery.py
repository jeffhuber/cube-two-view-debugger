from __future__ import annotations

import json
from pathlib import Path

from tools.build_full_corner_labeling_gallery import (
    HTML_TEMPLATE,
    _extract_cases_json,
    _full_image_display,
    _initial_full_corner_prefill,
    _resolve_pair_paths,
)
from tools.corner_conventions import (
    CAPTURE_FACE_BY_SLOT_BY_SIDE,
    FACE_DEFS_BY_SIDE,
    FAR_CORNERS_BY_SIDE,
    ONE_EDGE_CORNERS_BY_SIDE,
    POINT_NAMES,
    VERTEX_NAME_BY_SIDE,
    YAW0_CORNER_FACELETS,
    capture_to_wca_yaw_map,
    wca_face_by_slot,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FULL_CORNER_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "full_corner_ground_truth.json"


def test_face_defs_pin_image_a_human_corner_convention():
    assert FACE_DEFS_BY_SIDE["A"] == {
        "upper": ("vertex", "corner_1", "corner_0", "corner_5"),
        "right": ("vertex", "corner_3", "corner_2", "corner_1"),
        "front": ("vertex", "corner_5", "corner_4", "corner_3"),
    }


def test_face_defs_pin_image_b_human_corner_convention_after_flip():
    assert FACE_DEFS_BY_SIDE["B"] == {
        "upper": ("vertex", "corner_2", "corner_3", "corner_4"),
        "right": ("vertex", "corner_0", "corner_1", "corner_2"),
        "front": ("vertex", "corner_4", "corner_5", "corner_0"),
    }


def test_point_names_are_explicit_full_corner_schema():
    assert POINT_NAMES == (
        "vertex",
        "corner_0",
        "corner_1",
        "corner_2",
        "corner_3",
        "corner_4",
        "corner_5",
    )


def test_vertex_names_are_side_specific():
    assert VERTEX_NAME_BY_SIDE == {"A": "Va", "B": "Vb"}


def test_one_edge_and_far_triplets_are_side_specific():
    assert ONE_EDGE_CORNERS_BY_SIDE == {
        "A": ("corner_1", "corner_3", "corner_5"),
        "B": ("corner_0", "corner_2", "corner_4"),
    }
    assert FAR_CORNERS_BY_SIDE == {
        "A": ("corner_0", "corner_2", "corner_4"),
        "B": ("corner_1", "corner_3", "corner_5"),
    }


def test_yaw0_corner_facelets_pin_flattened_urf_dlb_net_mapping():
    assert YAW0_CORNER_FACELETS == {
        "Va": ("U9", "R1", "F3"),
        "Vb": ("D7", "L7", "B9"),
        "corner_0": ("U1", "L1", "B3"),
        "corner_1": ("U3", "R3", "B1"),
        "corner_2": ("D9", "R9", "B7"),
        "corner_3": ("D3", "R7", "F9"),
        "corner_4": ("D1", "L9", "F7"),
        "corner_5": ("U7", "F1", "L3"),
    }


def test_capture_slots_are_distinct_from_wca_faces_under_yaw():
    assert CAPTURE_FACE_BY_SLOT_BY_SIDE == {
        "A": {"upper": "U", "right": "R", "front": "F"},
        "B": {"upper": "D", "right": "B", "front": "L"},
    }
    assert capture_to_wca_yaw_map(0) == {
        "U": "U",
        "D": "D",
        "F": "F",
        "R": "R",
        "B": "B",
        "L": "L",
    }
    assert capture_to_wca_yaw_map(1) == {
        "U": "U",
        "D": "D",
        "F": "R",
        "R": "B",
        "B": "L",
        "L": "F",
    }
    assert wca_face_by_slot("A", 1) == {"upper": "U", "right": "B", "front": "R"}
    assert wca_face_by_slot("B", 1) == {"upper": "D", "right": "L", "front": "F"}


def test_full_image_display_never_crops():
    crop_box, scale, display_size = _full_image_display((3024, 4032))

    assert crop_box == (0, 0, 3024, 4032)
    assert scale == 1.0
    assert display_size == (3024, 4032)


def test_initial_prefill_contains_all_points_inside_image():
    prefill = _initial_full_corner_prefill((3000, 4000))

    assert set(prefill) == set(POINT_NAMES)
    for point in prefill.values():
        assert 0 <= point[0] <= 3000
        assert 0 <= point[1] <= 4000


def test_resolve_pair_paths_expands_user_home_from_manifest():
    manifests = [
        {
            "pairs": [
                {
                    "setId": 999,
                    "imageAPath": "~/cube-corpus/Set 999 - A - white up.JPG",
                    "imageBPath": "~/cube-corpus/Set 999 - B - yellow up.JPG",
                }
            ]
        }
    ]

    path_a, path_b = _resolve_pair_paths(manifests, "999")

    assert str(path_a).startswith(str(Path.home()))
    assert str(path_b).startswith(str(Path.home()))


def test_html_template_uses_full_corner_export_schema():
    assert "near_x" not in HTML_TEMPLATE
    assert "near_y" not in HTML_TEMPLATE
    assert "near_z" not in HTML_TEMPLATE
    assert "Va/Vb" in HTML_TEMPLATE
    assert "A slots: upper=Va+1,0,5; right=Va+3,2,1; front=Va+5,4,3" in HTML_TEMPLATE
    assert "B slots: upper=Vb+2,3,4; right=Vb+0,1,2; front=Vb+4,5,0" in HTML_TEMPLATE
    assert "full_corner_ground_truth.json" in HTML_TEMPLATE


def test_extract_cases_json_round_trips_embedded_cases():
    cases = [
        {
            "key": "20_A",
            "prefill": _initial_full_corner_prefill((1000, 800)),
        }
    ]
    html = HTML_TEMPLATE.replace("__CASES_JSON__", json.dumps(cases))

    assert _extract_cases_json(html) == cases


def test_full_corner_truth_fixture_is_schema_clean():
    data = json.loads(FULL_CORNER_FIXTURE.read_text(encoding="utf-8"))

    assert set(data) == {
        "20_A",
        "20_B",
        "38_A",
        "38_B",
        "40_A",
        "40_B",
        "41_A",
        "41_B",
        "43_A",
        "43_B",
        "45_A",
        "45_B",
    }
    for key, row in data.items():
        assert key.endswith(("_A", "_B"))
        assert row["approved"] is True
        for name in POINT_NAMES:
            assert name in row
            assert len(row[name]) == 2
            assert all(isinstance(value, (int, float)) for value in row[name])
