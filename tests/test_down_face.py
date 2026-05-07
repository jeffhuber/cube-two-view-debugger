from rubik_recognizer.down_face import complete_down_face


SOLVED = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9


def test_complete_down_face_solves_hidden_solved_down_face():
    chars = list(SOLVED)
    for idx in (27, 28, 29, 30, 32, 33, 34, 35):
        chars[idx] = "?"

    assert complete_down_face("".join(chars), limit=2) == [SOLVED]


def test_complete_down_face_rejects_bad_center():
    chars = list(SOLVED)
    chars[31] = "U"

    assert complete_down_face("".join(chars)) == []
