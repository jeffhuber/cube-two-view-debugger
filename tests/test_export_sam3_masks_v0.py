from __future__ import annotations

import numpy as np
from PIL import Image

from tools.export_sam3_masks_v0 import (
    ExportConfig,
    import_cached_whole_cube_masks,
    _normalize_mask_array,
    _version_at_least,
    generate_sam3_export_artifacts,
    render_report,
    select_mask_from_sam3_output,
)


def test_select_mask_from_sam3_output_unions_top_scored_instances():
    masks = np.zeros((3, 12, 12), dtype=np.float32)
    masks[0, 1:4, 1:4] = 1.0
    masks[1, 6:9, 6:9] = 1.0
    masks[2, 9:11, 9:11] = 1.0
    output = {
        "masks": masks,
        "scores": np.array([0.9, 0.7, 0.1], dtype=np.float32),
    }

    mask, stats = select_mask_from_sam3_output(
        output,
        score_threshold=0.5,
        max_instances=3,
    )

    assert mask is not None
    assert int(mask.sum()) == 18
    assert stats["selectedIndexes"] == [0, 1]
    assert stats["selectedInstanceCount"] == 2


def test_normalize_mask_array_accepts_common_sam_shapes():
    hw = np.ones((5, 6), dtype=np.float32)
    nhw = np.ones((2, 5, 6), dtype=np.float32)
    n1hw = np.ones((2, 1, 5, 6), dtype=np.float32)

    assert _normalize_mask_array(hw).shape == (1, 5, 6)
    assert _normalize_mask_array(nhw).shape == (2, 5, 6)
    assert _normalize_mask_array(n1hw).shape == (2, 5, 6)


def test_generate_artifacts_writes_blocked_environment_report(tmp_path):
    feedback = {
        "rows": [
            {
                "setId": "1",
                "side": "A",
                "status": "labeled",
                "imagePath": "/tmp/does-not-exist.jpg",
            }
        ]
    }
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(__import__("json").dumps(feedback), encoding="utf-8")

    document = generate_sam3_export_artifacts(
        ExportConfig(feedback_path=feedback_path, mask_dir=tmp_path / "masks")
    )
    report = render_report(document)

    assert document["probe"] == "sam3_mask_export_v0"
    assert document["summary"]["status"] in {"blocked_prerequisites", "completed"}
    assert "SAM3 Mask Export V0" in report
    if document["summary"]["status"] == "blocked_prerequisites":
        assert document["summary"]["blockedReason"]


def test_import_cached_whole_cube_npy_masks(tmp_path):
    source_dir = tmp_path / "cached"
    source_dir.mkdir()
    mask = np.zeros((8, 9), dtype=bool)
    mask[2:5, 3:7] = True
    np.save(source_dir / "set_15_A_sam3.npy", mask)

    output_dir = tmp_path / "foundation_masks"
    result = import_cached_whole_cube_masks(
        source_dir=source_dir,
        mask_dir=output_dir,
    )

    output_path = output_dir / "sam3" / "set_15_A_whole_cube.png"
    assert result["importedMaskCount"] == 1
    assert output_path.exists()
    assert np.asarray(Image.open(output_path).convert("L")).sum() == int(mask.sum()) * 255


def test_generate_artifacts_can_import_cached_masks_without_runtime(tmp_path):
    source_dir = tmp_path / "cached"
    source_dir.mkdir()
    np.save(source_dir / "set_26_B_sam3.npy", np.ones((4, 5), dtype=bool))
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (10, 8), "white").save(image_path)
    feedback = {
        "rows": [
            {
                "setId": "26",
                "side": "B",
                "status": "labeled",
                "imagePath": str(image_path),
            }
        ]
    }
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(__import__("json").dumps(feedback), encoding="utf-8")

    document = generate_sam3_export_artifacts(
        ExportConfig(
            feedback_path=feedback_path,
            mask_dir=tmp_path / "masks",
            import_whole_cube_npy_dir=source_dir,
        )
    )

    assert document["cachedImport"]["importedMaskCount"] == 1
    assert document["summary"]["cachedWholeCubeMaskCount"] == 1
    output = tmp_path / "masks" / "sam3" / "set_26_B_whole_cube.png"
    assert output.exists()
    assert Image.open(output).size == (10, 8)


def test_version_at_least_handles_local_version_suffixes():
    assert _version_at_least("2.12.0+cpu", (2, 7))
    assert _version_at_least("12.6", (12, 6))
    assert not _version_at_least("2.6.9", (2, 7))
