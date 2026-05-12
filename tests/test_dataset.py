import json

import pytest

from rubik_recognizer.dataset import (
    ImageUpload,
    _image_marker,
    evaluate_state,
    pair_image_uploads,
    parse_ground_truth,
)


@pytest.mark.parametrize(
    "filename,expected",
    [
        # Standalone A/B token with \s_- delimiters.
        ("Set 15 - A - white up.JPG", "A"),
        ("Set 15 - B - white up.JPG", "B"),
        ("Set_15_A_white_up.JPG", "A"),
        ("Set_15_B_white_up.JPG", "B"),
        ("a.jpg", "A"),
        ("b.jpg", "B"),
        ("Pair-A.jpg", "A"),
        ("Pair-B.jpg", "B"),
        # imageA / imageB / image A / image B — the case Codex caught
        # the JS detectABMarker missing on PR #44. Pin both server and
        # JS behavior here so the next time someone touches either
        # side they see what the contract is.
        ("imageA.jpg", "A"),
        ("imageB.jpg", "B"),
        ("IMAGE-A.jpg", "A"),
        ("imageb.jpg", "B"),
        ("image A.JPG", "A"),
        ("image B.JPG", "B"),
        # Negative cases: A/B in the middle of a word should NOT match.
        ("alpha.jpg", None),
        ("beta.jpg", None),
        ("ASCAR.jpg", None),
        ("foobar.jpg", None),
        ("photo.jpg", None),
    ],
)
def test_image_marker_patterns_match_js_detector(filename, expected):
    """Pins the server-side _image_marker() patterns. The single-pair
    drop zone in static/app.js has a parallel JS implementation
    (detectABMarker) that must match these expectations exactly —
    otherwise filenames the server understands silently swap labels
    in the UI. If you add a pattern on either side, add the
    equivalent cases here.
    """
    result = _image_marker(filename)
    if expected is None:
        assert result is None, f"expected no marker for {filename!r}, got {result!r}"
    else:
        assert result is not None, f"expected marker {expected!r} for {filename!r}, got None"
        # _image_marker returns (set_id, side); we only care about side here.
        _, side = result
        assert side == expected, f"expected side {expected!r} for {filename!r}, got {side!r}"


def test_pair_image_uploads_handles_reversed_image_a_b_filenames():
    """Regression for the bug Codex caught on PR #44: a user dropping
    imageB.jpg before imageA.jpg would previously fall through to
    drop-order pairing (in the JS) and silently swap A/B. The
    server-side _image_marker patterns DO cover the imageA/imageB
    case, so pair_image_uploads correctly assigns A=imageA.jpg,
    B=imageB.jpg regardless of drop order. PR #44 also updates the
    JS detectABMarker to match this server-side behavior."""
    pairs, unpaired = pair_image_uploads(
        [
            ImageUpload("imageB.jpg", b"this-is-actually-B"),
            ImageUpload("imageA.jpg", b"this-is-actually-A"),
        ]
    )
    assert unpaired == []
    assert len(pairs) == 1
    assert pairs[0].image_a.data == b"this-is-actually-A"
    assert pairs[0].image_b.data == b"this-is-actually-B"


def test_pair_image_uploads_matches_set_a_b_names():
    pairs, unpaired = pair_image_uploads(
        [
            ImageUpload("Set 15 - B - white up IMG_6708.JPG", b"b"),
            ImageUpload("Set 15 - A - white up IMG_6707.JPG", b"a"),
        ]
    )

    assert unpaired == []
    assert len(pairs) == 1
    assert pairs[0].set_id == "set-15"
    assert pairs[0].image_a.data == b"a"
    assert pairs[0].image_b.data == b"b"


def test_parse_ground_truth_expected_state_csv():
    state = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
    truth = parse_ground_truth(f"set_id,expected_state\nSet 1,{state}\n".encode("utf-8"))

    assert truth["set-1"] == state


def test_parse_ground_truth_json_export_uses_corrected_state():
    corrected = "DRULUDDUFLFRDRRFDFLBUFFLBUDULRUDBLRRRBFFLUBRLBDBBBLUFD"
    provider_guess = "LDRDUDBUFDFURRRDFDLBLFFLBULURBUDRLLFLRBULFFBRDFUDBBRUB"
    payload = [
        {
            "setName": "Set 15",
            "recognitions": [{"state": provider_guess, "provider": "gemini-3.1-pro"}],
            "corrected": corrected,
        }
    ]

    truth = parse_ground_truth(json.dumps(payload).encode("utf-8"), "ground-truth.json")

    assert truth["set-15"] == corrected


def test_parse_ground_truth_json_canonicalizes_unique_legal_net_export():
    exported = "LBFRUDRUDFBRUFLBRRBLRULRLLLDBULDBFRBFDDDBFUFBUUDDRFUFL"
    canonical = "RRLUUBDDFUUDDRFUFLFBRUFLBRRUBBBDRDLFBLRULRLLLFDDDBFUFB"
    payload = [{"setName": "Set 12", "corrected": exported}]

    truth = parse_ground_truth(json.dumps(payload).encode("utf-8"), "ground-truth.json")

    assert truth["set-12"] == canonical


def test_parse_ground_truth_fast_paths_already_canonical_state():
    """Ground truth captured post-PR-#20/#95 saves canonical WCA URFDLB
    in the `corrected` field directly (capture frame preserved separately
    in recognitionSignals.captureFrameState). The parser should return
    the canonical state unchanged — no 4096-rotation search needed.

    This is the expected path for all new corpus entries and the
    preferred path going forward. The legacy capture-frame fallback in
    `_canonical_ground_truth_state` remains for older saved files but
    is documented as deprecated.
    """
    # Set 42's actual canonical WCA URFDLB from the post-PR-#95 capture
    # (cube-snap PR #95). This is already legal, so the parser should
    # return it byte-for-byte.
    canonical = "UDDDURDFLFDBLRRRFUFUDRFBUBBFUUUDLLUFLBRLLBBLLRFBFBDRRD"
    payload = [{"setName": "Set 42", "corrected": canonical}]

    truth = parse_ground_truth(json.dumps(payload).encode("utf-8"), "ground-truth.json")

    assert truth["set-42"] == canonical


def test_parse_ground_truth_post_pr95_set12_round_trips_without_canonicalization():
    """Regression case for Set 12 after the post-PR-#95 Fixer capture.
    The new file's `corrected` is canonical WCA; parse_ground_truth
    should return it unchanged (NOT regroup-and-rotate it). Catches a
    future regression where the canonicalizer accidentally rewrites an
    already-valid state.
    """
    canonical = "RRLUUBDDFUUDDRFUFLFBRUFLBRRUBBBDRDLFBLRULRLLLFDDDBFUFB"
    payload = [{"setName": "Set 12", "corrected": canonical}]

    truth = parse_ground_truth(json.dumps(payload).encode("utf-8"), "ground-truth.json")

    assert truth["set-12"] == canonical


def test_evaluate_state_hamming_distance():
    expected = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
    actual = "R" + expected[1:]

    evaluation = evaluate_state(actual, expected)

    assert evaluation["available"]
    assert not evaluation["exact"]
    assert evaluation["hamming"] == 1
