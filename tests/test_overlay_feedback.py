from PIL import Image, ImageDraw

from tools.overlay_feedback import (
    ingest_overlay_feedback_rows,
    parse_rectified_label,
)
from tools.probe_overlay_discontinuity import cell_discontinuity_metrics


def test_parse_rectified_label_accepts_src_and_srd_typos():
    assert parse_rectified_label("src = U; good") == {"sourceFace": "U", "quality": "good"}
    assert parse_rectified_label("srd = B; bad") == {"sourceFace": "B", "quality": "bad"}


def test_ingest_overlay_feedback_rows_ranks_bad_slots():
    rows = [
        [None, "A quads", "A quads", "A quads", "A rectified", "A rectified", "A rectified"],
        [
            "Image Set",
            "red outline",
            "green outline",
            "blue outline",
            "U face rectified",
            "R face rectified",
            "F face rectified",
            "purple outline",
            "orange outline",
            "yellow outline",
            "D face rectified",
            "L face rectified",
            "B face rectified",
        ],
        [
            "17",
            "good",
            "good",
            "bad",
            "src = U; good",
            "src = R; good",
            "src = B; bad",
            "bad",
            "bad",
            "bad",
            "src = D; bad",
            "srd = B; bad",
            "src = B; bad",
        ],
    ]

    document = ingest_overlay_feedback_rows(rows, source_workbook="feedback.xlsx")
    set_17 = document["sets"][0]
    a_f = next(slot for slot in set_17["slots"] if slot["side"] == "A" and slot["slot"] == "F")
    b_l = next(slot for slot in set_17["slots"] if slot["side"] == "B" and slot["slot"] == "L")

    assert set_17["badSlotCount"] == 4
    assert a_f["failureModes"] == ["bad_quad", "bad_rectified", "wrong_source_face"]
    assert b_l["rectifiedSourceFace"] == "B"
    assert document["summary"]["failureModeCounts"]["bad_quad"] == 4
    assert document["summary"]["rankedSlots"][0]["badSignalCount"] >= 2


def test_cell_discontinuity_scores_split_cell_above_solid_cell():
    solid = Image.new("RGB", (300, 300), (220, 20, 20))
    split = Image.new("RGB", (300, 300), (220, 20, 20))
    draw = ImageDraw.Draw(split)
    draw.rectangle((50, 0, 100, 300), fill=(20, 20, 220))

    solid_metrics = cell_discontinuity_metrics(solid)
    split_metrics = cell_discontinuity_metrics(split)

    assert solid_metrics["score"] < 0.1
    assert split_metrics["score"] > solid_metrics["score"]
    assert split_metrics["maxHalfDelta"] > 100
