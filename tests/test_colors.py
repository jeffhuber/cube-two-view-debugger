from rubik_recognizer.colors import build_adaptive_palette, classify_rgb
from rubik_recognizer.recognizer import _facelet_options, _facelet_repair_options
from rubik_recognizer.validation import FACE_ORDER


def test_classify_dim_side_lit_yellow_as_yellow():
    match = classify_rgb((164, 162, 93))

    assert match.color == "yellow"
    assert match.face == "D"


def test_classify_shadowed_green_as_green():
    match = classify_rgb((74, 129, 94))

    assert match.color == "green"
    assert match.face == "F"


def test_low_light_weak_saturation_does_not_force_orange_hsv_hint():
    match = classify_rgb((75, 65, 59))

    assert match.color == "blue"
    assert match.face == "B"


def test_warm_low_saturation_background_does_not_force_orange_hsv_hint():
    match = classify_rgb((233, 198, 158))

    assert match.color == "white"
    assert match.face == "U"


def test_adaptive_palette_learns_warm_red_orange_boundary():
    samples = [
        (226, 226, 222),
        (215, 204, 52),
        (187, 82, 52),
        (194, 78, 48),
        (222, 126, 50),
        (230, 132, 56),
        (62, 142, 88),
        (55, 136, 82),
        (57, 86, 153),
        (64, 92, 160),
        (184, 78, 50),
        (226, 122, 48),
    ]
    anchors = {
        "white": [(226, 226, 222)],
        "yellow": [(215, 204, 52)],
        "red": [(187, 82, 52), (194, 78, 48)],
        "orange": [(222, 126, 50), (230, 132, 56)],
        "green": [(62, 142, 88)],
        "blue": [(57, 86, 153)],
    }

    palette = build_adaptive_palette(samples, anchors)

    assert classify_rgb((190, 80, 50), palette).color == "red"
    assert classify_rgb((226, 126, 52), palette).color == "orange"


def test_repair_options_exclude_non_ambiguous_green_to_yellow_flip():
    sticker = type("Sticker", (), {"match": classify_rgb((79, 176, 108))})()

    faces = [face for face, _, _ in _facelet_repair_options(sticker)]

    assert "F" in faces
    assert "D" not in faces


def test_suspect_grid_sample_keeps_far_rebalance_alternatives():
    rgb = (233, 198, 158)
    sticker = type("Sticker", (), {"source": "grid_sample", "rgb": rgb, "match": classify_rgb(rgb)})()

    faces = [face for face, _ in _facelet_options(sticker)]

    assert "U" in faces
    assert "R" in faces
    assert "B" in faces


def test_grid_samples_keep_full_palette_available_for_legal_repair():
    rgb = (74, 166, 122)
    sticker = type("Sticker", (), {"source": "grid_sample", "rgb": rgb, "match": classify_rgb(rgb)})()

    faces = {face for face, _, _ in _facelet_repair_options(sticker)}

    assert faces == set(FACE_ORDER)


def test_low_confidence_components_can_repair_to_far_colors():
    rgb = (75, 65, 59)
    sticker = type("Sticker", (), {"source": "component", "rgb": rgb, "match": classify_rgb(rgb)})()

    faces = {face for face, _, _ in _facelet_repair_options(sticker)}

    assert "B" in faces
    assert "D" in faces


def test_weak_grid_context_components_can_repair_to_far_colors():
    rgb = (178, 109, 75)
    sticker = type("Sticker", (), {"source": "component", "rgb": rgb, "match": classify_rgb(rgb), "grid_repair_flex": 1.2})()

    faces = {face for face, _, _ in _facelet_repair_options(sticker)}

    assert "L" in faces
    assert "D" in faces
