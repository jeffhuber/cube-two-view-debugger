from rubik_recognizer.validation import is_valid_state, validate_state


SOLVED = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9


def test_validate_state_accepts_solved_cube():
    assert validate_state(SOLVED).valid
    assert is_valid_state(SOLVED)


def test_validate_state_rejects_wrong_length():
    result = validate_state(SOLVED[:-1])

    assert not result.valid
    assert not is_valid_state(SOLVED[:-1])
    assert "state_length_not_54" in result.errors


def test_validate_state_rejects_bad_counts():
    state = "U" + SOLVED[1:-1] + "U"
    result = validate_state(state)

    assert not result.valid
    assert not is_valid_state(state)
    assert "U_count_not_9" in result.errors
    assert "B_count_not_9" in result.errors


def test_validate_state_rejects_invalid_piece_even_with_counts():
    chars = list(SOLVED)
    chars[9] = "F"
    chars[18] = "R"
    result = validate_state("".join(chars))

    assert not result.valid
    assert not is_valid_state("".join(chars))
    assert any(error.startswith("corner_") or error.startswith("edge_") for error in result.errors)
