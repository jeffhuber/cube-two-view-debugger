from rubik_recognizer.recognizer import RecognitionResult, _prefer_calibrated_result, _white_up_checks


class StubGrid:
    def __init__(self, center_face, y=0, matched_count=9, fit_error=0.0, rgb=None):
        self.center_face = center_face
        self.matched_count = matched_count
        self.fit_error = fit_error
        if rgb is None:
            rgb = {
                "U": (230, 232, 235),
                "R": (190, 45, 35),
                "F": (60, 145, 85),
                "D": (230, 220, 45),
                "L": (220, 120, 45),
                "B": (60, 90, 170),
            }.get(center_face, (0, 0, 0))
        self.center_sticker = type("CenterSticker", (), {"center": (0, y), "rgb": rgb})()


class StubAnalysis:
    def __init__(self, centers):
        self.grids = [StubGrid(center, y=index * 10) for index, center in enumerate(centers)]


def test_white_up_checks_accept_complementary_side_centers():
    a = StubAnalysis(["U", "R", "F"])
    b = StubAnalysis(["D", "L", "B"])

    assert _white_up_checks(a, b) == []


def test_white_up_checks_accept_logo_corrupted_white_center_by_assumption():
    a = StubAnalysis(["D", "R", "F"])
    a.grids[0].center_sticker.rgb = (221, 225, 230)
    b = StubAnalysis(["D", "L", "B"])

    assert _white_up_checks(a, b) == []


def test_white_up_checks_reject_missing_credible_white_up_face():
    a = StubAnalysis(["R", "L", "F"])
    b = StubAnalysis(["D", "L", "B"])

    assert "image_a_U_anchor_missing" in _white_up_checks(a, b)


def test_white_up_checks_reject_too_similar_views():
    a = StubAnalysis(["U", "R", "F"])
    b = StubAnalysis(["U", "R", "F"])

    assert "missing_side_face_coverage" in _white_up_checks(a, b)
    assert "image_b_D_anchor_missing" in _white_up_checks(a, b)


def test_white_up_checks_ignores_extra_opposite_candidate_when_side_coverage_exists():
    a = StubAnalysis(["U", "R", "D", "F"])
    b = StubAnalysis(["D", "L", "B"])

    assert _white_up_checks(a, b) == []


def test_white_up_checks_allows_yellowish_sample_on_image_b_anchor():
    a = StubAnalysis(["U", "R", "F"])
    b = StubAnalysis(["R", "L", "B"])
    b.grids[0].center_sticker.rgb = (224, 215, 55)

    assert _white_up_checks(a, b) == []


def test_white_up_checks_allows_whiteish_logo_sample_on_image_a_anchor():
    a = StubAnalysis(["D", "R", "F"])
    a.grids[0].center_sticker.rgb = (225, 226, 220)
    b = StubAnalysis(["D", "L", "B"])

    assert _white_up_checks(a, b) == []


def test_calibrated_unique_result_preferred_over_raw_repair():
    raw = RecognitionResult(status="success", confidence=0.80, reason="Recognized a legal white-up cube state after cubie-level color repair.")
    calibrated = RecognitionResult(status="success", confidence=0.70, reason="Recognized a unique legal white-up cube state.")

    assert _prefer_calibrated_result(calibrated, raw)


def test_same_tier_calibrated_result_needs_clear_margin():
    raw = RecognitionResult(status="success", confidence=0.80, reason="Recognized a legal white-up cube state after cubie-level color repair.")
    calibrated = RecognitionResult(status="success", confidence=0.82, reason="Recognized a legal white-up cube state after cubie-level color repair.")

    assert not _prefer_calibrated_result(calibrated, raw)
