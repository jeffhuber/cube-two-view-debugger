import pytest

from rubik_recognizer import colors as color_module
from rubik_recognizer.colors import (
    CANONICAL_RGB,
    CLASSIFIER_CANONICAL,
    CLASSIFIER_KNN5_LAB,
    CLASSIFIER_KNN5_LAB_FULL,
    CLASSIFIER_MODE_ENV,
    COLOR_TO_FACE,
    ColorMatch,
    build_adaptive_palette,
    classify_rgb,
    classify_rgb_with_mode,
    _palette_normalized_rgb,
)
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


def test_build_adaptive_palette_ignores_runtime_classifier_switch(monkeypatch):
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

    canonical_palette = build_adaptive_palette(samples, anchors)
    monkeypatch.setenv(CLASSIFIER_MODE_ENV, CLASSIFIER_KNN5_LAB)

    assert build_adaptive_palette(samples, anchors) == canonical_palette


def test_knn5_classifier_mode_changes_red_orange_boundary():
    rgb = (144, 72, 49)

    assert classify_rgb_with_mode(rgb, CLASSIFIER_CANONICAL).color == "orange"
    assert classify_rgb_with_mode(rgb, CLASSIFIER_KNN5_LAB).color == "red"


def test_classify_rgb_env_switch_uses_knn5(monkeypatch):
    monkeypatch.setenv(CLASSIFIER_MODE_ENV, CLASSIFIER_KNN5_LAB)

    match = classify_rgb((144, 72, 49))

    assert match.color == "red"
    assert match.face == "R"


def test_full_knn5_mode_uses_knn_for_all_colors(monkeypatch):
    calls = []

    def fake_raw_knn(rgb, prototypes=None):
        calls.append((rgb, prototypes))
        return _match("yellow", confidence=0.91)

    monkeypatch.setattr(color_module, "_raw_knn5_lab_match", fake_raw_knn)

    match = classify_rgb_with_mode((226, 226, 222), CLASSIFIER_KNN5_LAB_FULL)

    assert match.color == "yellow"
    assert match.face == "D"
    assert calls == [((226, 226, 222), None)]


def test_classify_rgb_env_switch_uses_full_knn5(monkeypatch):
    monkeypatch.setenv(CLASSIFIER_MODE_ENV, CLASSIFIER_KNN5_LAB_FULL)
    monkeypatch.setattr(
        color_module,
        "_raw_knn5_lab_match",
        lambda rgb, prototypes=None: _match("green", confidence=0.88),
    )

    match = classify_rgb((226, 226, 222))

    assert match.color == "green"
    assert match.face == "F"


def test_knn5_mode_skips_confident_canonical_red_orange(monkeypatch):
    calls = []

    def fake_raw_knn(rgb, prototypes=None):
        calls.append(rgb)
        return _match("orange", confidence=1.0)

    monkeypatch.setattr(color_module, "_raw_knn5_lab_match", fake_raw_knn)

    match = classify_rgb_with_mode((190, 48, 36), CLASSIFIER_KNN5_LAB)

    assert match.color == "red"
    assert calls == []


def test_knn5_mode_rejects_low_confidence_override(monkeypatch):
    def fake_raw_knn(rgb, prototypes=None):
        return _match("red", confidence=0.63)

    monkeypatch.setattr(color_module, "_raw_knn5_lab_match", fake_raw_knn)

    match = classify_rgb_with_mode((144, 72, 49), CLASSIFIER_KNN5_LAB)

    assert match.color == "orange"


def test_knn5_mode_skips_non_red_orange_canonical(monkeypatch):
    calls = []

    def fake_raw_knn(rgb, prototypes=None):
        calls.append(rgb)
        return _match("yellow", confidence=1.0)

    monkeypatch.setattr(color_module, "_raw_knn5_lab_match", fake_raw_knn)

    match = classify_rgb_with_mode((226, 226, 222), CLASSIFIER_KNN5_LAB)

    assert match.color == "white"
    assert calls == []


def test_palette_normalized_rgb_maps_adaptive_palette_to_canonical_space():
    assert _palette_normalized_rgb(CANONICAL_RGB["red"], CANONICAL_RGB) == CANONICAL_RGB["red"]

    shifted = {
        "white": (228, 218, 208),
        "yellow": (220, 190, 28),
        "red": (180, 42, 30),
        "orange": (208, 94, 32),
        "green": (48, 122, 70),
        "blue": (52, 72, 132),
    }

    normalized = _palette_normalized_rgb(shifted["red"], shifted)

    raw_error = sum(abs(shifted["red"][index] - CANONICAL_RGB["red"][index]) for index in range(3))
    normalized_error = sum(abs(normalized[index] - CANONICAL_RGB["red"][index]) for index in range(3))
    assert normalized_error < raw_error


def test_classify_rgb_rejects_unknown_env_switch(monkeypatch):
    monkeypatch.setenv(CLASSIFIER_MODE_ENV, "mystery")

    with pytest.raises(ValueError, match="CUBE_RECOGNIZER_CLASSIFIER"):
        classify_rgb((190, 80, 50))


def _match(color: str, confidence: float) -> ColorMatch:
    return ColorMatch(color, COLOR_TO_FACE[color], 0.0, confidence, [(color, 0.0)])


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
