from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tools.foundation_segmentation_bakeoff_v0 import (
    ProviderSpec,
    _face_boundary_vertex_candidates,
    _load_external_masks,
    _provider_status,
    generate_segmentation_bakeoff_artifacts,
    render_report,
)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def test_face_boundary_candidates_find_shared_trihedral_corner():
    top = np.zeros((80, 80), dtype=bool)
    left = np.zeros((80, 80), dtype=bool)
    right = np.zeros((80, 80), dtype=bool)
    top[8:31, 8:54] = True
    left[30:72, 8:31] = True
    right[30:72, 30:72] = True

    candidates = _face_boundary_vertex_candidates(
        {"top_face": top, "left_face": left, "right_face": right},
    )

    assert candidates
    x, y = candidates[0]["point"]
    assert abs(x - 30.0) <= 2.0
    assert abs(y - 30.0) <= 2.0
    assert candidates[0]["details"]["faceBoundaryHitCount"] == 3


def test_external_mask_loader_reads_provider_prompt_files(tmp_path):
    mask = np.zeros((12, 14), dtype=bool)
    mask[2:9, 3:10] = True
    _write_mask(tmp_path / "sam3" / "set_15_A_top_face.png", mask)

    masks = _load_external_masks(tmp_path, "sam3", "15", "A")

    assert list(masks) == ["top_face"]
    assert masks["top_face"].shape == (12, 14)
    assert int(masks["top_face"].sum()) == int(mask.sum())


def test_generate_artifacts_scores_external_face_masks(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (80, 80), (240, 240, 240)).save(image_path)
    masks_root = tmp_path / "masks"
    top = np.zeros((80, 80), dtype=bool)
    left = np.zeros((80, 80), dtype=bool)
    right = np.zeros((80, 80), dtype=bool)
    top[8:31, 8:54] = True
    left[30:72, 8:31] = True
    right[30:72, 30:72] = True
    _write_mask(masks_root / "sam3" / "set_1_A_top_face.png", top)
    _write_mask(masks_root / "sam3" / "set_1_A_left_face.png", left)
    _write_mask(masks_root / "sam3" / "set_1_A_right_face.png", right)

    feedback = {
        "rows": [
            {
                "setId": "1",
                "side": "A",
                "evaluationTier": "easy_corpus",
                "imagePath": str(image_path),
                "status": "labeled",
                "humanVertexPoint": [30.0, 30.0],
            }
        ]
    }
    provider = ProviderSpec(
        key="sam3",
        display_name="SAM3 test",
        package_names=("definitely_missing_sam3_test_pkg",),
        notes="test provider",
    )

    document = generate_segmentation_bakeoff_artifacts(
        feedback,
        external_mask_dir=masks_root,
        output_dir=tmp_path / "overlays",
        provider_specs=(provider,),
    )
    report = render_report(document)

    summary = document["summary"]["providerSummaries"]["sam3"]
    assert summary["rowsWithAnyMask"] == 1
    assert summary["rowsWithThreeFaceMasks"] == 1
    assert summary["top1HitCount@10px"] == 1
    assert document["rows"][0]["providers"]["sam3"]["overlayPath"]
    assert "Foundation Segmentation Bakeoff V0" in report


def test_provider_status_reports_missing_package_without_masks(tmp_path):
    provider = ProviderSpec(
        key="sam3",
        display_name="SAM3 test",
        package_names=("definitely_missing_sam3_test_pkg",),
        notes="test provider",
    )

    status = _provider_status(provider, tmp_path)

    assert status["packageInstalled"] is False
    assert status["externalMaskFileCount"] == 0
    assert status["status"] == "package_missing_external_masks_required"
