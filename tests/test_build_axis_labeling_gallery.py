from __future__ import annotations

from tools.build_axis_labeling_gallery import (
    HTML_TEMPLATE,
    _full_image_display,
    _resolve_pair_paths,
)


def _display_point(crop_box, scale: float, point):
    return ((point[0] - crop_box[0]) * scale, (point[1] - crop_box[1]) * scale)


def test_full_image_display_does_not_crop_label_targets():
    crop_box, scale, (display_w, display_h) = _full_image_display((2400, 1800))

    assert crop_box == (0, 0, 2400, 1800)
    assert scale == 1.0
    assert (display_w, display_h) == (2400, 1800)
    for point in [(0, 0), (2399, 1799), (1200, 900)]:
        dx, dy = _display_point(crop_box, scale, point)
        assert 0 <= dx <= display_w
        assert 0 <= dy <= display_h


def test_gallery_template_uses_single_viewport_safe_case_canvas():
    assert 'id="case-host"' in HTML_TEMPLATE
    assert 'id="caseCanvas"' in HTML_TEMPLATE
    assert "max-height: 100%" in HTML_TEMPLATE
    assert "markerRadiusForCanvas" in HTML_TEMPLATE
    assert HTML_TEMPLATE.count("judgments[c.key] = {") == 1
    assert 'id="cases"' not in HTML_TEMPLATE


def test_resolve_pair_paths_falls_back_to_corpus_root(tmp_path):
    corpus = tmp_path / "cube-corpus"
    corpus.mkdir()
    manifest_anchor_a = corpus / "Set 1 - A - white up.JPG"
    manifest_anchor_b = corpus / "Set 1 - B - white up.JPG"
    manifest_anchor_a.write_bytes(b"anchor a")
    manifest_anchor_b.write_bytes(b"anchor b")
    expected_a = corpus / "Set 20 - A - white up.JPG"
    expected_b = corpus / "Set 20 - B - white up.JPG"
    expected_a.write_bytes(b"a")
    expected_b.write_bytes(b"b")
    manifests = [
        {
            "pairs": [
                {
                    "setId": "1",
                    "imageAPath": str(manifest_anchor_a),
                    "imageBPath": str(manifest_anchor_b),
                }
            ]
        }
    ]

    assert _resolve_pair_paths(manifests, "20") == (expected_a, expected_b)
