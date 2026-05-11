import json

from rubik_recognizer.dataset import ImageUpload, evaluate_state, pair_image_uploads, parse_ground_truth


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
