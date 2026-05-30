import json
from itertools import permutations
from pathlib import Path

from rubik_recognizer.validation import _parity, is_valid_state, validate_state


SOLVED = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
VALIDATOR_PARITY_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "validator_parity_cases.json").read_text()
)["cases"]


def test_validate_state_matches_shared_parity_fixtures():
    for case in VALIDATOR_PARITY_CASES:
        result = validate_state(case["state"])
        assert result.valid is case["valid"], case["name"]
        assert result.errors == case["errors"], case["name"]
        assert is_valid_state(case["state"]) is case["valid"], case["name"]


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


def test_validate_state_rejects_ud_corner_sticker_swap():
    # Swap the U facelet of the URF corner (idx 8) with the D facelet of the
    # DFR corner (idx 29). Counts and centers stay valid, but neither corner is
    # a valid oriented corner anymore (wrong U/D color). The earlier lenient
    # lookup (accepting either U or D in the twist slot) validated this
    # unreachable state as solvable.
    chars = list(SOLVED)
    chars[8], chars[29] = chars[29], chars[8]
    state = "".join(chars)
    result = validate_state(state)

    assert not result.valid
    assert not is_valid_state(state)
    assert "corner_0_invalid_color_set" in result.errors
    assert "corner_4_invalid_color_set" in result.errors
    assert not any(error.endswith("_count_not_9") for error in result.errors)


def test_validate_state_rejects_mirrored_corner():
    # Swap the R and F facelets of the URF corner (idx 9 and 20): same color
    # set {U, R, F} but reversed cyclic order, which no legal move can produce.
    # Already rejected by the ordered lookup; pinned here for parity with
    # cube-snap's validateCubeState.ts coverage.
    chars = list(SOLVED)
    chars[9], chars[20] = chars[20], chars[9]
    state = "".join(chars)
    result = validate_state(state)

    assert not result.valid
    assert not is_valid_state(state)
    assert "corner_0_invalid_color_set" in result.errors


def test_parity_matches_inversion_count_for_permutations():
    for permutation in permutations(range(5)):
        inversions = 0
        for i in range(len(permutation)):
            for j in range(i + 1, len(permutation)):
                if permutation[i] > permutation[j]:
                    inversions += 1

        assert _parity(list(permutation)) == inversions % 2
