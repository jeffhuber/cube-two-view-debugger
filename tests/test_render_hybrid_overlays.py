from pathlib import Path

from PIL import Image

from tools import render_hybrid_overlays


def test_render_pair_forwards_hybrid_pipeline_flags(monkeypatch, tmp_path):
    calls = []

    def fake_load_processing_image(_path: Path):
        return Image.new("RGB", (120, 120), (240, 240, 240)), None

    def fake_proposer(image_path, side, *, hull_guard, fit_error_fallback, processing_image):
        calls.append(
            {
                "image_path": image_path,
                "side": side,
                "hull_guard": hull_guard,
                "fit_error_fallback": fit_error_fallback,
                "processing_image_size": processing_image.size,
            }
        )
        quads = {
            face: [(10, 10), (50, 10), (50, 50), (10, 50)]
            for face in render_hybrid_overlays.EXPECTED_FACES_BY_SIDE[side]
        }
        debug = {
            "selectedPerFace": {
                face: {"sourceCenterFace": face}
                for face in render_hybrid_overlays.EXPECTED_FACES_BY_SIDE[side]
            }
        }
        return quads, debug

    def fake_rectify_face(_img, _quad, *, output_size):
        return Image.new("RGB", (output_size, output_size), (220, 220, 220))

    monkeypatch.setattr(render_hybrid_overlays, "_load_processing_image", fake_load_processing_image)
    monkeypatch.setattr(render_hybrid_overlays, "_proposer_face_quads", fake_proposer)
    monkeypatch.setattr(render_hybrid_overlays, "rectify_face", fake_rectify_face)

    written = render_hybrid_overlays.render_pair(
        "99",
        tmp_path / "A.jpg",
        tmp_path / "B.jpg",
        tmp_path,
        hull_guard=True,
        fit_error_fallback=True,
    )

    assert [call["side"] for call in calls] == ["A", "B"]
    assert all(call["hull_guard"] is True for call in calls)
    assert all(call["fit_error_fallback"] is True for call in calls)
    assert all(call["processing_image_size"] == (120, 120) for call in calls)
    assert sorted(written) == ["A_quads", "A_rectified", "B_quads", "B_rectified"]
