from tools.hull_label_assembly import oriented_slot_matrix, slot_face_assignments


RAW_MATRIX = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
]

SLOT_QUADS = {
    "upper": [(0, 0), (10, 0), (10, 10), (0, 10)],
    "right": [(20, 0), (30, 0), (30, 10), (20, 10)],
    "front": [(-20, 0), (-10, 0), (-10, 10), (-20, 10)],
}


def test_slot_face_assignments_follow_shared_yaw_convention():
    assert slot_face_assignments(2) == {
        "A": {"upper": "U", "right": "L", "front": "B"},
        "B": {"upper": "D", "right": "F", "front": "R"},
    }


def test_oriented_slot_matrix_preserves_center_and_assigns_wca_face():
    wca_face, matrix, orientation = oriented_slot_matrix(
        raw_matrix=RAW_MATRIX,
        side="B",
        slot="front",
        yaw_quarter_turns=2,
        quad=SLOT_QUADS["front"],
    )

    assert wca_face == "R"
    assert matrix[1][1] == 4
    assert orientation == {
        "slot": "front",
        "wcaFace": "R",
        "mirror": True,
        "rotQuarter": 3,
    }


def test_oriented_slot_matrix_covers_all_sides_yaws_and_slots():
    for side in ("A", "B"):
        for yaw_quarter_turns in range(4):
            assignments = slot_face_assignments(yaw_quarter_turns)[side]
            for slot, quad in SLOT_QUADS.items():
                result = oriented_slot_matrix(
                    raw_matrix=RAW_MATRIX,
                    side=side,
                    slot=slot,
                    yaw_quarter_turns=yaw_quarter_turns,
                    quad=quad,
                )

                assert result is not None, (side, yaw_quarter_turns, slot)
                wca_face, matrix, orientation = result
                assert wca_face == assignments[slot]
                assert matrix[1][1] == 4
                assert orientation["slot"] == slot
                assert orientation["wcaFace"] == wca_face
