import json
from pathlib import Path

import rubik_recognizer.recognizer as recognizer
from rubik_recognizer.recognizer import (
    PIECE_CONFLICT_KEYS,
    IMAGE_B_VISIBLE_FACE_EVIDENCE_WEAK_CHECK,
    RED_ORANGE_PAIR_CALIBRATION_SUSPECTED_CHECK,
    BACKGROUND_STICKER_NOISE_CHECK,
    RecognitionResult,
    RecognitionWorkset,
    WhiteUpRecognizer,
    _anchor_grid_candidates,
    _assigned_grid_by_face,
    _attach_failed_pair_color_calibration_signal,
    _capture_yaw_state_to_wca,
    _balanced_color_assignment_score_penalty,
    _balanced_color_assignment_summary,
    _corner_assignment,
    _direct_legal_candidate_summary,
    _failed_checks_with_context,
    _grid_signal_summary,
    _image_b_visible_face_evidence_weak,
    _grid_matrix_for_orientation,
    _merge_faces,
    _merged_face_candidates,
    _pair_calibration_anchors,
    _oriented_options_for_grid_map,
    _pair_color_calibration_signal,
    _prefer_calibrated_result,
    _recognition_category_payload,
    _recognition_signals_with_failed_checks,
    _repair_details_with_orientation_selection_scores,
    _repair_backfill_applies,
    _repair_orientation_rerank_applies,
    _repair_ranking_penalty,
    _side_pair_complement,
    _u_logo_anchor_grid_candidate,
    _validation_failed_checks,
    _selected_faces_by_image,
    _selected_sides_by_image,
    _state_to_capture_yaw,
    _white_up_checks,
)


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


def test_u_anchor_candidates_include_low_confidence_logo_center():
    from rubik_recognizer.colors import ColorMatch

    center_match = ColorMatch(
        color="green",
        face="F",
        distance=36.0,
        confidence=0.35,
        alternatives=[("green", 36.0), ("white", 56.0), ("blue", 58.0)],
    )
    center = type(
        "Sticker",
        (),
        {"source": "component", "rgb": (92, 98, 67), "match": center_match, "shape_angle": 0.0},
    )()
    grid = type(
        "Grid",
        (),
        {
            "center_face": "F",
            "center_sticker": center,
            "matched_count": 8,
            "fit_error": 1.5,
            "stickers": [[center for _ in range(3)] for _ in range(3)],
        },
    )()

    assert _u_logo_anchor_grid_candidate(grid)
    assert _anchor_grid_candidates([grid], "U") == [grid]


def test_u_logo_anchor_candidates_reject_confident_side_center():
    from rubik_recognizer.colors import ColorMatch

    center_match = ColorMatch(
        color="green",
        face="F",
        distance=20.0,
        confidence=0.78,
        alternatives=[("green", 20.0), ("white", 35.0)],
    )
    center = type(
        "Sticker",
        (),
        {"source": "component", "rgb": (70, 145, 90), "match": center_match, "shape_angle": 0.0},
    )()
    grid = type(
        "Grid",
        (),
        {
            "center_face": "F",
            "center_sticker": center,
            "matched_count": 9,
            "fit_error": 1.0,
            "stickers": [[center for _ in range(3)] for _ in range(3)],
        },
    )()

    assert not _u_logo_anchor_grid_candidate(grid)
    assert _anchor_grid_candidates([grid], "U") == []


def test_assigned_grid_by_face_does_not_reuse_assumed_anchor_as_side_face():
    from rubik_recognizer.colors import ColorMatch

    def grid(grid_id, face, rgb, matched_count):
        match = ColorMatch("white" if face == "U" else "green", face, 0.0, 0.9, [("white", 0.0)])
        center = type(
            "Sticker",
            (),
            {"source": "component", "rgb": rgb, "match": match, "shape_angle": 0.0},
        )()
        return type(
            "Grid",
            (),
            {
                "id": grid_id,
                "center_face": face,
                "center_sticker": center,
                "matched_count": matched_count,
                "fit_error": 1.0,
                "stickers": [[center for _ in range(3)] for _ in range(3)],
            },
        )()

    assumed_u = grid(1, "F", (225, 225, 225), 9)
    fallback_f = grid(2, "F", (80, 150, 90), 8)
    analysis = type("Analysis", (), {"grids": [assumed_u, fallback_f]})()

    assigned = _assigned_grid_by_face(analysis, "U")

    assert assigned["U"] is assumed_u
    assert assigned["F"] is fallback_f


def test_pair_calibration_skips_non_white_logo_anchor_seed():
    from rubik_recognizer.colors import ColorMatch

    def grid(grid_id, face, rgb, match):
        center = type(
            "Sticker",
            (),
            {"source": "component", "rgb": rgb, "match": match, "shape_angle": 0.0},
        )()
        return type(
            "Grid",
            (),
            {
                "id": grid_id,
                "center_face": face,
                "center_sticker": center,
                "matched_count": 8,
                "fit_error": 1.5,
                "stickers": [[center for _ in range(3)] for _ in range(3)],
            },
        )()

    logo_match = ColorMatch(
        color="green",
        face="F",
        distance=36.0,
        confidence=0.35,
        alternatives=[("green", 36.0), ("white", 56.0)],
    )
    yellow_match = ColorMatch("yellow", "D", 0.0, 0.9, [("yellow", 0.0)])
    logo_u = grid(1, "F", (92, 98, 67), logo_match)
    direct_d = grid(2, "D", (230, 220, 45), yellow_match)
    analysis_a = type("Analysis", (), {"grids": [logo_u]})()
    analysis_b = type("Analysis", (), {"grids": [direct_d]})()

    anchors = _pair_calibration_anchors(analysis_a, analysis_b)

    assert anchors["white"] == []
    assert anchors["yellow"] == [(230, 220, 45)]


def test_pair_calibration_keeps_whiteish_assumed_anchor_seed():
    from rubik_recognizer.colors import ColorMatch

    white_match = ColorMatch("white", "U", 0.0, 0.9, [("white", 0.0)])
    center = type(
        "Sticker",
        (),
        {"source": "component", "rgb": (225, 225, 225), "match": white_match, "shape_angle": 0.0},
    )()
    assumed_u = type(
        "Grid",
        (),
        {
            "id": 1,
            "center_face": "F",
            "center_sticker": center,
            "matched_count": 8,
            "fit_error": 1.5,
            "stickers": [[center for _ in range(3)] for _ in range(3)],
        },
    )()
    analysis = type("Analysis", (), {"grids": [assumed_u]})()

    anchors = _pair_calibration_anchors(analysis, type("Analysis", (), {"grids": []})())

    assert anchors["white"] == [(225, 225, 225)]


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


def test_white_up_checks_rejects_weak_image_b_down_anchor():
    a = StubAnalysis(["U", "R", "F"])
    b = StubAnalysis(["D", "L", "B"])
    b.grids[0].matched_count = recognizer.MIN_IMAGE_B_D_ANCHOR_MATCHED_COUNT - 1

    checks = _white_up_checks(a, b)

    assert "image_b_D_anchor_weak" in checks
    assert recognizer._reason_for_checks(checks) == "Image B contains a weak yellow/D center grid; retake with a clearer yellow-up face."


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


def test_recognize_from_analyses_reuses_workset_for_direct_and_repair(monkeypatch):
    workset = RecognitionWorkset(options_a=[], options_b=[], merged_candidates=[])
    calls = {"workset": 0, "direct": 0, "repair": 0}

    def fake_workset(analysis_a, analysis_b, **kwargs):
        calls["workset"] += 1
        return workset

    def fake_state_candidates(self, candidate_workset):
        calls["direct"] += 1
        assert candidate_workset is workset
        return []

    def fake_repair_details(self, candidate_workset, *, release_merged_candidates=False):
        calls["repair"] += 1
        assert candidate_workset is workset
        assert release_merged_candidates is True
        return []

    monkeypatch.setattr(recognizer, "_base_recognition_signals", lambda analysis_a, analysis_b: {})
    monkeypatch.setattr(recognizer, "_white_up_checks", lambda analysis_a, analysis_b, **kwargs: [])
    monkeypatch.setattr(recognizer, "_recognition_workset", fake_workset)
    monkeypatch.setattr(WhiteUpRecognizer, "_state_candidates_from_workset", fake_state_candidates)
    monkeypatch.setattr(WhiteUpRecognizer, "_legal_repair_candidate_details_from_workset", fake_repair_details)
    monkeypatch.setattr(recognizer, "REPAIR_SKIP_DIRECT_CANDIDATE_THRESHOLD", 0)

    result = WhiteUpRecognizer()._recognize_from_analyses(object(), object())

    assert result.status == "rejected"
    assert calls == {"workset": 1, "direct": 1, "repair": 1}


def test_recognize_from_analyses_skips_repair_for_low_direct_candidate_count(monkeypatch):
    workset = RecognitionWorkset(options_a=[], options_b=[], merged_candidates=[])
    calls = {"workset": 0, "direct": 0, "repair": 0}

    def fake_workset(analysis_a, analysis_b, **kwargs):
        calls["workset"] += 1
        return workset

    def fake_state_candidates(self, candidate_workset):
        calls["direct"] += 1
        assert candidate_workset is workset
        return []

    def fake_repair_details(self, candidate_workset, *, release_merged_candidates=False):
        calls["repair"] += 1
        raise AssertionError("low-candidate direct rejects should not enter repair")

    monkeypatch.setattr(recognizer, "_base_recognition_signals", lambda analysis_a, analysis_b: {})
    monkeypatch.setattr(recognizer, "_white_up_checks", lambda analysis_a, analysis_b, **kwargs: [])
    monkeypatch.setattr(recognizer, "_recognition_workset", fake_workset)
    monkeypatch.setattr(WhiteUpRecognizer, "_state_candidates_from_workset", fake_state_candidates)
    monkeypatch.setattr(WhiteUpRecognizer, "_legal_repair_candidate_details_from_workset", fake_repair_details)

    result = WhiteUpRecognizer()._recognize_from_analyses(object(), object())

    assert result.status == "rejected"
    assert result.failed_checks == ["no_legal_state"]
    assert result.candidates == 0
    assert result.recognition_signals["repairCandidateCount"] == 0
    assert result.recognition_signals["topRepairCandidates"] == []
    assert calls == {"workset": 1, "direct": 1, "repair": 0}


def test_direct_legal_candidate_summary_reports_margin_and_top_candidates():
    top_state = "U" * 54
    second_state = "R" * 54
    unique = {
        second_state: 0.8123,
        top_state: 0.8332,
    }
    details = {
        top_state: {
            "sidePairA": "B/R",
            "sidePairB": "F/L",
            "orderedSidePairA": "R/B",
            "orderedSidePairB": "F/L",
            "rawMergedScore": 1416.029713,
            "variantCost": 93.876222,
        },
        second_state: {
            "sidePairA": "F/R",
            "sidePairB": "B/L",
            "orderedSidePairA": "F/R",
            "orderedSidePairB": "B/L",
            "rawMergedScore": 1416.009713,
            "variantCost": 93.876222,
        },
    }

    summary = _direct_legal_candidate_summary(unique, details)

    assert summary["status"] == "separated"
    assert summary["stateCount"] == 2
    assert summary["topConfidence"] == 0.8332
    assert summary["secondConfidence"] == 0.8123
    assert summary["confidenceGap"] == 0.0209
    assert summary["topTieCount"] == 1
    assert summary["topRawMergedScore"] == 1416.0297
    assert summary["secondRawMergedScore"] == 1416.0097
    assert summary["rawMergedScoreGap"] == 0.02
    assert summary["topVariantCost"] == 93.8762
    assert summary["secondVariantCost"] == 93.8762
    assert summary["variantCostGap"] == 0.0
    assert summary["topCandidates"][0]["state"] == top_state
    assert summary["topCandidates"][0]["rawMergedScore"] == 1416.0297
    assert summary["topCandidates"][0]["variantCost"] == 93.8762
    assert summary["topCandidates"][0]["selectedFacesByImage"] == {
        "imageA": ["B", "R", "U"],
        "imageB": ["D", "F", "L"],
    }


def test_recognize_from_analyses_attaches_direct_legal_candidate_summary(monkeypatch):
    workset = RecognitionWorkset(options_a=[], options_b=[], merged_candidates=[])
    top_state = "U" * 54
    second_state = "R" * 54

    monkeypatch.setattr(recognizer, "_base_recognition_signals", lambda analysis_a, analysis_b: {})
    monkeypatch.setattr(recognizer, "_white_up_checks", lambda analysis_a, analysis_b, **kwargs: [])
    monkeypatch.setattr(recognizer, "_recognition_workset", lambda analysis_a, analysis_b, **kwargs: workset)
    monkeypatch.setattr(
        WhiteUpRecognizer,
        "_state_candidates_from_workset",
        lambda self, candidate_workset: [
            (
                top_state,
                0.83,
                {
                    "sidePairA": "B/R",
                    "sidePairB": "F/L",
                    "orderedSidePairA": "R/B",
                    "orderedSidePairB": "F/L",
                    "rawMergedScore": 1400.0,
                    "variantCost": 12.5,
                },
            ),
            (
                second_state,
                0.81,
                {
                    "sidePairA": "F/R",
                    "sidePairB": "B/L",
                    "orderedSidePairA": "F/R",
                    "orderedSidePairB": "B/L",
                    "rawMergedScore": 1390.0,
                    "variantCost": 14.0,
                },
            ),
        ],
    )
    monkeypatch.setattr(recognizer, "validate_state", lambda state: type("Validation", (), {"valid": True, "errors": []})())

    result = WhiteUpRecognizer()._recognize_from_analyses(object(), object())

    assert result.status == "success"
    assert result.reason == "Recognized the highest-scoring legal white-up cube state."
    summary = result.recognition_signals["directLegalCandidates"]
    assert summary["status"] == "separated"
    assert summary["stateCount"] == 2
    assert summary["confidenceGap"] == 0.02
    assert summary["rawMergedScoreGap"] == 10.0
    assert summary["variantCostGap"] == 1.5
    assert summary["topCandidates"][0]["state"] == top_state


def test_recognize_from_analyses_reports_empty_backfill_attempt(monkeypatch):
    workset = RecognitionWorkset(options_a=[], options_b=[], merged_candidates=[])
    calls = {"standard_repair": 0, "backfill": 0, "backfill_repair": 0}

    def fake_standard_repair(self, candidate_workset, *, release_merged_candidates=False):
        calls["standard_repair"] += 1
        assert candidate_workset is workset
        assert release_merged_candidates is True
        return []

    def fake_backfill(analysis_a, candidate_workset):
        calls["backfill"] += 1
        assert candidate_workset is workset
        return []

    def fake_backfill_repair(self, candidate_workset, repair_merges, *, repair_source="standard"):
        calls["backfill_repair"] += 1
        assert candidate_workset is workset
        assert repair_merges == []
        assert repair_source == "conflict_backfill"
        return []

    monkeypatch.setattr(recognizer, "_base_recognition_signals", lambda analysis_a, analysis_b: {})
    monkeypatch.setattr(recognizer, "_white_up_checks", lambda analysis_a, analysis_b, **kwargs: [])
    monkeypatch.setattr(recognizer, "_recognition_workset", lambda analysis_a, analysis_b, **kwargs: workset)
    monkeypatch.setattr(WhiteUpRecognizer, "_state_candidates_from_workset", lambda self, candidate_workset: [])
    monkeypatch.setattr(WhiteUpRecognizer, "_legal_repair_candidate_details_from_workset", fake_standard_repair)
    monkeypatch.setattr(WhiteUpRecognizer, "_legal_repair_candidate_details_from_merges", fake_backfill_repair)
    monkeypatch.setattr(recognizer, "_repair_backfill_applies", lambda analysis_a, analysis_b: True)
    monkeypatch.setattr(recognizer, "_repair_backfill_merged_face_candidates", fake_backfill)
    monkeypatch.setattr(recognizer, "REPAIR_SKIP_DIRECT_CANDIDATE_THRESHOLD", 0)

    result = WhiteUpRecognizer()._recognize_from_analyses(object(), object())

    assert result.status == "rejected"
    assert result.recognition_signals["repairBackfillAttempted"] is True
    assert result.recognition_signals["repairBackfillEvaluatedMerges"] == 0
    assert result.recognition_signals["repairBackfillUsed"] is False
    assert calls == {"standard_repair": 1, "backfill": 1, "backfill_repair": 1}


def test_recognize_from_analyses_uses_conflict_backfill_repair_source(monkeypatch):
    solved = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    workset = RecognitionWorkset(options_a=[], options_b=[], merged_candidates=[])
    backfill_merge = (123.0, {"_score": 123.0})
    calls = {"standard_repair": 0, "backfill": 0, "backfill_repair": 0}

    def fake_standard_repair(self, candidate_workset, *, release_merged_candidates=False):
        calls["standard_repair"] += 1
        assert candidate_workset is workset
        assert release_merged_candidates is True
        return []

    def fake_backfill(analysis_a, candidate_workset):
        calls["backfill"] += 1
        assert candidate_workset is workset
        return [backfill_merge]

    def fake_backfill_repair(self, candidate_workset, repair_merges, *, repair_source="standard"):
        calls["backfill_repair"] += 1
        assert candidate_workset is workset
        assert repair_merges == [backfill_merge]
        assert repair_source == "conflict_backfill"
        return [
            {
                "state": solved,
                "confidence": recognizer.REPAIRED_HIGH_CONFIDENCE_THRESHOLD,
                "repairRankingPenalty": 0.0,
                "repairSource": repair_source,
            }
        ]

    monkeypatch.setattr(recognizer, "_base_recognition_signals", lambda analysis_a, analysis_b: {})
    monkeypatch.setattr(recognizer, "_white_up_checks", lambda analysis_a, analysis_b, **kwargs: [])
    monkeypatch.setattr(recognizer, "_recognition_workset", lambda analysis_a, analysis_b, **kwargs: workset)
    monkeypatch.setattr(WhiteUpRecognizer, "_state_candidates_from_workset", lambda self, candidate_workset: [])
    monkeypatch.setattr(WhiteUpRecognizer, "_legal_repair_candidate_details_from_workset", fake_standard_repair)
    monkeypatch.setattr(WhiteUpRecognizer, "_legal_repair_candidate_details_from_merges", fake_backfill_repair)
    monkeypatch.setattr(recognizer, "_repair_backfill_applies", lambda analysis_a, analysis_b: True)
    monkeypatch.setattr(recognizer, "_repair_backfill_merged_face_candidates", fake_backfill)
    monkeypatch.setattr(recognizer, "REPAIR_SKIP_DIRECT_CANDIDATE_THRESHOLD", 0)

    result = WhiteUpRecognizer()._recognize_from_analyses(object(), object())

    assert result.status == "success"
    assert result.state == solved
    assert result.recognition_signals["repairBackfillAttempted"] is True
    assert result.recognition_signals["repairBackfillEvaluatedMerges"] == 1
    assert result.recognition_signals["repairBackfillUsed"] is True
    assert result.recognition_signals["repairBackfillProbeReason"] == "no_standard_repair"
    assert result.recognition_signals["selectedRepairCandidate"]["repairSource"] == "conflict_backfill"
    assert result.recognition_signals["topRepairCandidates"][0]["repairSource"] == "conflict_backfill"
    assert calls == {"standard_repair": 1, "backfill": 1, "backfill_repair": 1}


def test_recognize_from_analyses_tries_backfill_for_unstable_standard_repair(monkeypatch):
    solved = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    workset = RecognitionWorkset(options_a=[], options_b=[], merged_candidates=[])
    backfill_merge = (123.0, {"_score": 123.0})
    calls = {"standard_repair": 0, "backfill": 0, "backfill_repair": 0}

    def fake_standard_repair(self, candidate_workset, *, release_merged_candidates=False):
        calls["standard_repair"] += 1
        assert candidate_workset is workset
        assert release_merged_candidates is True
        return [
            {
                "state": solved,
                "confidence": 0.5486,
                "repairRankingPenalty": recognizer.MAX_REPAIR_RANKING_PENALTY,
                "repairChanges": 10,
                "preRepairFaceCounts": {face: 9 for face in "URFDLB"},
                "preRepairConflicts": {"totalConflicts": 8, "validCorners": 5},
                "repairSource": "standard",
            }
        ]

    def fake_backfill(analysis_a, candidate_workset):
        calls["backfill"] += 1
        assert candidate_workset is workset
        return [backfill_merge]

    def fake_backfill_repair(self, candidate_workset, repair_merges, *, repair_source="standard"):
        calls["backfill_repair"] += 1
        assert candidate_workset is workset
        assert repair_merges == [backfill_merge]
        assert repair_source == "conflict_backfill"
        return [
            {
                "state": solved,
                "confidence": 0.631,
                "repairRankingPenalty": 0.08,
                "repairChanges": 4,
                "preRepairFaceCounts": {face: 9 for face in "URFDLB"},
                "preRepairConflicts": {"totalConflicts": 2, "validCorners": 8},
                "repairSource": repair_source,
            }
        ]

    monkeypatch.setattr(recognizer, "_base_recognition_signals", lambda analysis_a, analysis_b: {})
    monkeypatch.setattr(recognizer, "_white_up_checks", lambda analysis_a, analysis_b, **kwargs: [])
    monkeypatch.setattr(recognizer, "_recognition_workset", lambda analysis_a, analysis_b, **kwargs: workset)
    monkeypatch.setattr(WhiteUpRecognizer, "_state_candidates_from_workset", lambda self, candidate_workset: [])
    monkeypatch.setattr(WhiteUpRecognizer, "_legal_repair_candidate_details_from_workset", fake_standard_repair)
    monkeypatch.setattr(WhiteUpRecognizer, "_legal_repair_candidate_details_from_merges", fake_backfill_repair)
    monkeypatch.setattr(recognizer, "_repair_backfill_applies", lambda analysis_a, analysis_b: True)
    monkeypatch.setattr(recognizer, "_repair_backfill_merged_face_candidates", fake_backfill)
    monkeypatch.setattr(recognizer, "REPAIR_SKIP_DIRECT_CANDIDATE_THRESHOLD", 0)

    result = WhiteUpRecognizer()._recognize_from_analyses(object(), object())

    assert result.status == "success"
    assert result.state == solved
    assert result.confidence == 0.631
    assert result.recognition_signals["repairBackfillAttempted"] is True
    assert result.recognition_signals["repairBackfillEvaluatedMerges"] == 1
    assert result.recognition_signals["repairBackfillUsed"] is True
    assert result.recognition_signals["repairBackfillProbeReason"] == "unstable_standard_repair"
    assert result.recognition_signals["selectedRepairCandidate"]["repairSource"] == "conflict_backfill"
    monkeypatch.setattr(recognizer, "REPAIR_RETAKE_MIN_CANDIDATES", 0)
    category = _recognition_category_payload(result)
    assert category["category"] == "needs_manual_review"
    assert category["reason"] == "repair_backfill_from_unstable_standard_repair"
    assert calls == {"standard_repair": 1, "backfill": 1, "backfill_repair": 1}


def test_recognize_from_analyses_keeps_backfill_skipped_for_pre_count_skew(monkeypatch):
    solved = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    workset = RecognitionWorkset(options_a=[], options_b=[], merged_candidates=[])
    calls = {"standard_repair": 0, "backfill": 0, "backfill_repair": 0}

    def fake_standard_repair(self, candidate_workset, *, release_merged_candidates=False):
        calls["standard_repair"] += 1
        return [
            {
                "state": solved,
                "confidence": 0.5486,
                "repairRankingPenalty": recognizer.MAX_REPAIR_RANKING_PENALTY,
                "repairChanges": 10,
                "preRepairFaceCounts": {
                    "U": 9,
                    "R": 7,
                    "F": 8,
                    "D": 9,
                    "L": 10,
                    "B": 11,
                },
                "preRepairConflicts": {"totalConflicts": 8, "validCorners": 5},
                "repairSource": "standard",
            }
        ]

    def fake_backfill(analysis_a, candidate_workset):
        calls["backfill"] += 1
        return []

    def fake_backfill_repair(self, candidate_workset, repair_merges, *, repair_source="standard"):
        calls["backfill_repair"] += 1
        return []

    monkeypatch.setattr(recognizer, "_base_recognition_signals", lambda analysis_a, analysis_b: {})
    monkeypatch.setattr(recognizer, "_white_up_checks", lambda analysis_a, analysis_b, **kwargs: [])
    monkeypatch.setattr(recognizer, "_recognition_workset", lambda analysis_a, analysis_b, **kwargs: workset)
    monkeypatch.setattr(WhiteUpRecognizer, "_state_candidates_from_workset", lambda self, candidate_workset: [])
    monkeypatch.setattr(WhiteUpRecognizer, "_legal_repair_candidate_details_from_workset", fake_standard_repair)
    monkeypatch.setattr(WhiteUpRecognizer, "_legal_repair_candidate_details_from_merges", fake_backfill_repair)
    monkeypatch.setattr(recognizer, "_repair_backfill_applies", lambda analysis_a, analysis_b: True)
    monkeypatch.setattr(recognizer, "_repair_backfill_merged_face_candidates", fake_backfill)
    monkeypatch.setattr(recognizer, "REPAIR_SKIP_DIRECT_CANDIDATE_THRESHOLD", 0)

    result = WhiteUpRecognizer()._recognize_from_analyses(object(), object())

    assert result.status == "success"
    assert result.recognition_signals["selectedRepairCandidate"]["repairSource"] == "standard"
    assert "repairBackfillAttempted" not in result.recognition_signals
    assert calls == {"standard_repair": 1, "backfill": 0, "backfill_repair": 0}


def test_repair_orientation_rerank_targets_low_confidence_rank_capped_repairs():
    low_ranked = [
        {
            "confidence": 0.508,
            "repairRankingPenalty": recognizer.MAX_REPAIR_RANKING_PENALTY,
            "orientationRankA": 2,
            "orientationRankB": 12,
        }
    ]
    high_confidence = [{**low_ranked[0], "confidence": recognizer.REPAIRED_HIGH_CONFIDENCE_THRESHOLD}]
    high_orientation = [{**low_ranked[0], "orientationRankA": 2, "orientationRankB": 3}]

    assert _repair_orientation_rerank_applies(low_ranked)
    assert not _repair_orientation_rerank_applies(high_confidence)
    assert not _repair_orientation_rerank_applies(high_orientation)


def test_repair_orientation_rerank_promotes_better_ranked_candidate():
    details = [
        {
            "state": "old",
            "confidence": 0.5078,
            "repairRankingPenalty": recognizer.MAX_REPAIR_RANKING_PENALTY,
            "orientationRankA": 2,
            "orientationRankB": 12,
        },
        {
            "state": "better",
            "confidence": 0.5,
            "repairRankingPenalty": recognizer.MAX_REPAIR_RANKING_PENALTY,
            "orientationRankA": 5,
            "orientationRankB": 1,
        },
    ]

    annotated = _repair_details_with_orientation_selection_scores(details)

    assert [item["state"] for item in annotated] == ["better", "old"]
    assert annotated[0]["preRerankConfidence"] == 0.5
    assert annotated[0]["repairOrientationRerankBonus"] == 0.024
    assert annotated[0]["repairSelectionScore"] == 0.524
    assert annotated[0]["confidence"] == 0.524
    assert annotated[1]["repairSelectionScore"] == 0.5078
    assert annotated[1].get("repairOrientationRerankBonus") is None


def test_repair_orientation_rerank_preserves_order_when_gate_does_not_apply():
    details = [
        {
            "state": "first",
            "confidence": recognizer.REPAIRED_HIGH_CONFIDENCE_THRESHOLD,
            "repairRankingPenalty": recognizer.MAX_REPAIR_RANKING_PENALTY,
            "orientationRankA": 2,
            "orientationRankB": 12,
        },
        {
            "state": "second",
            "confidence": 0.5,
            "repairRankingPenalty": recognizer.MAX_REPAIR_RANKING_PENALTY,
            "orientationRankA": 1,
            "orientationRankB": 1,
        },
    ]

    annotated = _repair_details_with_orientation_selection_scores(details)

    assert [item["state"] for item in annotated] == ["first", "second"]
    assert [item["repairSelectionScore"] for item in annotated] == [
        recognizer.REPAIRED_HIGH_CONFIDENCE_THRESHOLD,
        0.5,
    ]
    assert all("preRerankConfidence" not in item for item in annotated)
    assert all("repairOrientationRerankBonus" not in item for item in annotated)


def test_validation_failed_checks_tags_opposing_red_orange_skew():
    a = StubAnalysis(["R", "R", "R", "R", "R", "L", "L"])
    b = StubAnalysis(["L", "L", "L", "L", "L", "R", "R"])

    checks = _validation_failed_checks(["R_count_not_9"], a, b)

    assert checks == ["R_count_not_9", RED_ORANGE_PAIR_CALIBRATION_SUSPECTED_CHECK]


def test_validation_failed_checks_tags_weak_image_b_visible_face_evidence():
    a = StubAnalysis(["R", "R", "R", "R", "R", "L", "L"])
    b = type(
        "Analysis",
        (),
        {
            "grids": [
                _stub_face_grid("D", matched_count=7, grid_samples=1),
                *[_stub_face_grid("L", matched_count=5, grid_samples=4) for _ in range(5)],
                _stub_face_grid("R", matched_count=6, grid_samples=3),
            ],
            "stickers": [],
        },
    )()

    checks = _validation_failed_checks(["R_count_not_9"], a, b)

    assert checks == [
        "R_count_not_9",
        RED_ORANGE_PAIR_CALIBRATION_SUSPECTED_CHECK,
        IMAGE_B_VISIBLE_FACE_EVIDENCE_WEAK_CHECK,
    ]
    assert _image_b_visible_face_evidence_weak(b)


def test_validation_failed_checks_tags_background_anchor_evidence_collapse():
    collapsed_u = ["B", "D", "F", "L", "U", "R", "B", "D", "F"]
    collapsed_d = ["B", "U", "F", "L", "D", "R", "B", "U", "F"]
    a = type(
        "Analysis",
        (),
        {
            "grids": [_stub_face_grid("U", cell_faces=collapsed_u)],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()
    b = type(
        "Analysis",
        (),
        {
            "grids": [_stub_face_grid("D", cell_faces=collapsed_d)],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()

    checks = _validation_failed_checks([f"{face}_count_not_9" for face in recognizer.FACE_ORDER], a, b)

    assert checks[-1] == BACKGROUND_STICKER_NOISE_CHECK
    signal = _recognition_signals_with_failed_checks({}, checks, a, b)["backgroundStickerNoise"]
    assert signal["reason"] == "all_face_counts_failed_with_anchor_evidence_collapse"
    assert signal["images"]["imageA"]["selectedAnchor"]["cellFaceCounts"]["U"] == 1
    assert signal["images"]["imageB"]["selectedAnchor"]["cellFaceCounts"]["D"] == 1


def test_validation_failed_checks_do_not_tag_set17_style_anchor_evidence():
    a = type(
        "Analysis",
        (),
        {
            "grids": [_stub_face_grid("U", cell_faces=["U", "U", "U", "B", "D", "F", "L", "R", "B"])],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()
    b = type(
        "Analysis",
        (),
        {
            "grids": [_stub_face_grid("D", cell_faces=["B", "U", "F", "L", "D", "R", "B", "U", "F"])],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()

    checks = _validation_failed_checks([f"{face}_count_not_9" for face in recognizer.FACE_ORDER], a, b)

    assert BACKGROUND_STICKER_NOISE_CHECK not in checks


def test_failed_checks_tag_background_blue_dominance_when_u_anchor_missing():
    a = type(
        "Analysis",
        (),
        {
            "grids": [
                *[_stub_face_grid("B", grid_id=index) for index in range(14)],
                _stub_face_grid("F", grid_id=20),
            ],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()
    b = type(
        "Analysis",
        (),
        {
            "grids": [_stub_face_grid("D")],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()

    checks = _failed_checks_with_context(["image_a_U_anchor_missing"], a, b)

    assert checks == ["image_a_U_anchor_missing", BACKGROUND_STICKER_NOISE_CHECK]
    signal = _recognition_signals_with_failed_checks({}, checks, a, b)["backgroundStickerNoise"]
    assert signal["reason"] == "image_a_u_anchor_missing_with_blue_grid_dominance"
    assert signal["images"]["imageA"]["dominantGridCenterFace"] == "B"
    assert signal["images"]["imageA"]["dominantGridCenterCount"] == 14
    assert signal["images"]["imageA"]["selectedAnchor"] is None


def test_failed_checks_tag_background_blue_dominance_after_anchor_recovered():
    a = type(
        "Analysis",
        (),
        {
            "grids": [
                *[_stub_face_grid("B", grid_id=index) for index in range(14)],
                _stub_face_grid("U", grid_id=30),
            ],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()
    b = type(
        "Analysis",
        (),
        {
            "grids": [_stub_face_grid("D")],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()

    checks = _failed_checks_with_context(["no_legal_state"], a, b)

    assert checks == ["no_legal_state", BACKGROUND_STICKER_NOISE_CHECK]
    signal = _recognition_signals_with_failed_checks({}, checks, a, b)["backgroundStickerNoise"]
    assert signal["reason"] == "no_legal_state_with_blue_grid_dominance"
    assert signal["images"]["imageA"]["dominantGridCenterFace"] == "B"
    assert signal["images"]["imageA"]["selectedAnchor"]["centerFace"] == "U"


def test_failed_checks_tag_background_when_image_a_triple_collapses_anchor_evidence():
    a = type(
        "Analysis",
        (),
        {
            "grids": [_stub_face_grid("U", cell_faces=["U", "B", "D", "F", "L", "L", "L", "R", "B"])],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()
    b = type(
        "Analysis",
        (),
        {
            "grids": [_stub_face_grid("D", cell_faces=["D", "B", "F", "L", "R", "R", "U", "U", "B"])],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()

    checks = _failed_checks_with_context(["image_a_face_triple_overlap_low_quality"], a, b)

    assert checks == ["image_a_face_triple_overlap_low_quality", BACKGROUND_STICKER_NOISE_CHECK]
    signal = _recognition_signals_with_failed_checks({}, checks, a, b)["backgroundStickerNoise"]
    assert signal["reason"] == "image_a_face_triple_low_quality_with_anchor_evidence_collapse"
    assert signal["images"]["imageA"]["selectedAnchor"]["cellFaceCounts"]["U"] == 1
    assert signal["images"]["imageB"]["selectedAnchor"]["cellFaceCounts"]["D"] == 1


def test_failed_checks_tag_background_when_no_legal_state_has_collapsed_anchor_evidence():
    a = type(
        "Analysis",
        (),
        {
            "grids": [_stub_face_grid("U", cell_faces=["U", "B", "D", "F", "L", "L", "L", "R", "B"])],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()
    b = type(
        "Analysis",
        (),
        {
            "grids": [_stub_face_grid("D", cell_faces=["D", "B", "F", "L", "R", "R", "U", "U", "B"])],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()

    checks = _failed_checks_with_context(["no_legal_state"], a, b)

    assert checks == ["no_legal_state", BACKGROUND_STICKER_NOISE_CHECK]
    signal = _recognition_signals_with_failed_checks({}, checks, a, b)["backgroundStickerNoise"]
    assert signal["reason"] == "no_legal_state_with_anchor_evidence_collapse"


def test_failed_checks_tag_background_when_many_face_counts_fail_with_collapsed_anchor_evidence():
    a = type(
        "Analysis",
        (),
        {
            "grids": [_stub_face_grid("U", cell_faces=["U", "B", "D", "F", "L", "L", "L", "R", "B"])],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()
    b = type(
        "Analysis",
        (),
        {
            "grids": [_stub_face_grid("D", cell_faces=["D", "B", "F", "L", "R", "R", "U", "U", "B"])],
            "stickers": [],
            "roi": (0, 100, 800, 1100),
        },
    )()
    checks = [
        "B_count_not_9",
        "D_count_not_9",
        "F_count_not_9",
        "R_count_not_9",
        "U_count_not_9",
        "piece_legality_invalid",
    ]

    contextual = _failed_checks_with_context(checks, a, b)

    assert contextual == [*checks, BACKGROUND_STICKER_NOISE_CHECK]
    signal = _recognition_signals_with_failed_checks({}, contextual, a, b)["backgroundStickerNoise"]
    assert signal["reason"] == "multi_face_count_failure_with_anchor_evidence_collapse"


def test_visible_face_color_count_imbalance_requires_collapsed_anchor_evidence(monkeypatch):
    monkeypatch.setattr(
        recognizer,
        "_top_visible_face_pair_color_counts",
        lambda analysis_a, analysis_b: {"U": 8, "R": 7, "F": 8, "D": 6, "L": 10, "B": 15},
    )
    monkeypatch.setattr(recognizer, "_anchor_evidence_collapsed", lambda analysis_a, analysis_b: True)

    assert recognizer._visible_face_color_count_imbalance_suspected(object(), object())

    monkeypatch.setattr(recognizer, "_anchor_evidence_collapsed", lambda analysis_a, analysis_b: False)

    assert not recognizer._visible_face_color_count_imbalance_suspected(object(), object())


def test_visible_face_color_count_imbalance_allows_success_control_counts(monkeypatch):
    monkeypatch.setattr(
        recognizer,
        "_top_visible_face_pair_color_counts",
        lambda analysis_a, analysis_b: {"U": 10, "R": 6, "F": 10, "D": 8, "L": 12, "B": 8},
    )
    monkeypatch.setattr(recognizer, "_anchor_evidence_collapsed", lambda analysis_a, analysis_b: True)

    assert not recognizer._visible_face_color_count_imbalance_suspected(object(), object())


def test_balanced_color_assignment_uses_cheapest_surplus_change():
    faces = _solved_facelet_matrices()
    faces["R"][0][0] = _facelet_with_alternatives("U", [("white", 0.0), ("red", 3.0)])

    summary = _balanced_color_assignment_summary(faces, include_changes=True)

    assert summary["status"] == "assigned"
    assert summary["primaryCounts"]["U"] == 10
    assert summary["primaryCounts"]["R"] == 8
    assert summary["assignedCounts"] == {face: 9 for face in recognizer.FACE_ORDER}
    assert summary["requiredChanges"] == 1
    assert summary["changes"] == [{"cell": "R00", "from": "U", "to": "R", "cost": 3.4}]
    assert _balanced_color_assignment_score_penalty(summary) > 0.0


def test_balanced_color_assignment_reports_unassignable_when_no_alternative_can_fill_deficit():
    faces = _solved_facelet_matrices()
    faces["R"][0][0] = _facelet_with_alternatives("U", [("white", 0.0)])

    summary = _balanced_color_assignment_summary(faces, include_changes=True)

    assert summary["status"] == "unassignable"
    assert summary["requiredChanges"] == 1
    assert summary["cost"] is None
    assert _balanced_color_assignment_score_penalty(summary) > 0.0


def test_merge_faces_applies_opt_in_balanced_assignment_score_penalty(monkeypatch):
    monkeypatch.setenv(recognizer.BALANCED_COLOR_SCORING_ENV, "1")
    faces = _solved_facelet_matrices()
    faces["R"][0][0] = _facelet_with_alternatives("U", [("white", 0.0), ("red", 3.0)])
    image_a = {face: faces[face] for face in ("U", "R", "F")}
    image_a["_score"] = 100.0
    image_a["_visible_color_counts"] = recognizer._primary_face_counts(image_a)
    image_b = {face: faces[face] for face in ("D", "L", "B")}
    image_b["_score"] = 50.0
    image_b["_visible_color_counts"] = recognizer._primary_face_counts(image_b)

    merged = _merge_faces(image_a, image_b)

    assert merged is not None
    assert merged["_score"] < 150.0
    assert merged["_balanced_color_assignment"]["status"] == "primary_count_scored"
    assert merged["_balanced_color_assignment_score_penalty"] > 0.0


def test_merge_faces_leaves_balanced_assignment_scoring_off_by_default():
    faces = _solved_facelet_matrices()
    faces["R"][0][0] = _facelet_with_alternatives("U", [("white", 0.0), ("red", 3.0)])
    image_a = {face: faces[face] for face in ("U", "R", "F")}
    image_a["_score"] = 100.0
    image_b = {face: faces[face] for face in ("D", "L", "B")}
    image_b["_score"] = 50.0

    merged = _merge_faces(image_a, image_b)

    assert merged is not None
    assert merged["_score"] == 150.0
    assert "_balanced_color_assignment" not in merged
    assert "_balanced_color_assignment_score_penalty" not in merged


def _solved_facelet_matrices():
    return {
        face: [[_facelet_with_alternatives(face, [(recognizer.FACE_TO_CENTER_COLOR[face], 0.0)]) for _ in range(3)] for _ in range(3)]
        for face in recognizer.FACE_ORDER
    }


def _facelet_with_alternatives(face, alternatives):
    from rubik_recognizer.colors import COLOR_TO_FACE, ColorMatch

    color = recognizer.FACE_TO_CENTER_COLOR[face]
    first_color, first_distance = alternatives[0]
    match = ColorMatch(
        first_color,
        COLOR_TO_FACE[first_color],
        first_distance,
        0.8,
        alternatives,
    )
    return type(
        "Facelet",
        (),
        {
            "source": "component",
            "rgb": (230, 230, 230),
            "shape_angle": None,
            "match": match,
            "face": face,
            "color": color,
        },
    )()


def _stub_face_grid(face, *, matched_count=9, fit_error=1.0, grid_samples=0, cell_faces=None, grid_id=None):
    from rubik_recognizer.colors import ColorMatch

    rgbs = {
        "U": (230, 232, 235),
        "R": (190, 45, 35),
        "F": (60, 145, 85),
        "D": (230, 220, 45),
        "L": (220, 120, 45),
        "B": (60, 90, 170),
    }
    faces = list(cell_faces or [face] * 9)
    stickers = []
    for index, sticker_face in enumerate(faces):
        color = recognizer.FACE_TO_CENTER_COLOR[sticker_face]
        match = ColorMatch(color, sticker_face, 0.0, 0.8, [(color, 0.0)])
        stickers.append(
            type(
                "Sticker",
                (),
                {
                    "source": "grid_sample" if index < grid_samples else "component",
                    "rgb": rgbs[sticker_face],
                    "match": match,
                    "shape_angle": None,
                },
            )()
        )
    return type(
        "Grid",
        (),
        {
            "id": grid_id if grid_id is not None else id(stickers[4]),
            "center_face": face,
            "center_sticker": stickers[4],
            "matched_count": matched_count,
            "fit_error": fit_error,
            "stickers": [stickers[0:3], stickers[3:6], stickers[6:9]],
        },
    )()


def test_corner_assignment_preserves_legacy_ud_agnostic_side_order():
    assert _corner_assignment(("D", "F", "R")) == (4, 0)
    assert _corner_assignment(("U", "F", "R")) == (4, 0)


def test_validation_failed_checks_ignores_one_sided_red_orange_skew():
    a = StubAnalysis(["R", "R", "R", "L", "L"])
    b = StubAnalysis(["B", "F", "U", "D"])

    checks = _validation_failed_checks(["R_count_not_9"], a, b)

    assert checks == ["R_count_not_9"]


def test_repair_backfill_applies_to_opposing_red_orange_skew():
    a = StubAnalysis(["R", "R", "R", "R", "R", "L", "L"])
    b = StubAnalysis(["L", "L", "L", "L", "L", "R", "R"])

    assert _repair_backfill_applies(a, b)


def test_repair_backfill_skips_one_sided_red_orange_skew():
    a = StubAnalysis(["R", "R", "R", "L", "L"])
    b = StubAnalysis(["B", "F", "U", "D"])

    assert not _repair_backfill_applies(a, b)


def test_side_pair_complement_returns_opposite_visible_pair():
    assert _side_pair_complement(("B", "L")) == "F/R"
    assert _side_pair_complement("F/R") == "B/L"
    assert _side_pair_complement(("F", "B")) == "L/R"
    assert _side_pair_complement(("F",)) == ""


def test_grid_signal_summary_reports_cell_face_and_source_counts():
    def facelet(face, source="component"):
        return type(
            "Facelet",
            (),
            {
                "source": source,
                "rgb": (230, 230, 230),
                "shape_angle": None,
                "match": type("Match", (), {"face": face, "color": "white", "confidence": 0.8})(),
            },
        )()

    grid = type(
        "Grid",
        (),
        {
            "id": 4,
            "center_face": "U",
            "center_sticker": facelet("U"),
            "matched_count": 7,
            "fit_error": 1.25,
            "cube_hull_inside_count": 6,
            "cube_hull_outside_count": 3,
            "cube_hull_source": "rembg_u2net_hull",
            "stickers": [
                [facelet("U"), facelet("U", "grid_sample"), facelet("R")],
                [facelet("D"), facelet("U"), facelet("B")],
                [facelet("L", "grid_sample"), facelet("F"), facelet("U")],
            ],
        },
    )()

    summary = _grid_signal_summary(grid)

    assert summary["cellFaceCounts"] == {"U": 4, "R": 1, "F": 1, "D": 1, "L": 1, "B": 1}
    assert summary["cellSourceCounts"] == {"component": 7, "grid_sample": 2}
    assert summary["extrapolatedSamples"] == 0
    assert summary["extrapolatedSampleScore"] == 0.0
    assert summary["unsupportedSamples"] == 0
    assert summary["unsupportedSampleScore"] == 0.0
    assert summary["gridExtrapolationPenalty"] == 0.0
    assert summary["cubeHullInsideCount"] == 6
    assert summary["cubeHullOutsideCount"] == 3
    assert summary["cubeHullSource"] == "rembg_u2net_hull"
    assert summary["cubeHullPenalty"] == 0.0
    assert summary["rawCubeHullPenalty"] == 200.0


def test_grid_signal_summary_reports_span_contamination_diagnostics():
    def facelet(face, source="component", shape_angle=None, outside=0.0, nearest=0.0, spacing=20.0):
        item = type("Facelet", (), {})()
        item.source = source
        item.rgb = (240, 240, 240)
        item.shape_angle = shape_angle
        item.match = type("Match", (), {"face": face, "color": "white", "confidence": 0.3})()
        if source == "grid_sample":
            item.grid_spacing = spacing
            item.outside_grid_component_hull_distance = outside
            item.nearest_grid_component_distance = nearest
        return item

    grid = type(
        "Grid",
        (),
        {
            "id": 12,
            "center_face": "U",
            "center_sticker": facelet("U"),
            "matched_count": 5,
            "fit_error": 1.0,
            "cube_hull_inside_count": 7,
            "cube_hull_outside_count": 2,
            "cube_hull_source": "rembg_u2net_hull",
            "stickers": [
                [facelet("U", shape_angle=0), facelet("R", shape_angle=5), facelet("F", "grid_sample", outside=20, nearest=40)],
                [facelet("D", shape_angle=10), facelet("L", shape_angle=70), facelet("B", "grid_sample")],
                [facelet("U", shape_angle=75), facelet("R", "grid_sample"), facelet("F", "grid_sample")],
            ],
        },
    )()

    span = _grid_signal_summary(grid)["gridSpanContamination"]

    assert span["componentShapeAngleCount"] == 5
    assert span["componentShapeSpread"] > 25.0
    assert span["sampledCellCount"] == 4
    assert span["sampleCellsOutsideGridComponentHull"] == 1
    assert span["sampleCellsFarFromGridComponents"] == 1
    assert span["maxOutsideGridComponentHullRatio"] == 1.0
    assert span["maxNearestGridComponentRatio"] == 2.0
    assert span["extrapolatedCellCount"] == 1
    assert span["score"] > 0.0


def test_cube_hull_grid_penalty_requires_extrapolation_signal():
    def facelet(source, outside=0.0, nearest=70.0):
        return type(
            "Facelet",
            (),
            {
                "source": source,
                "rgb": (230, 230, 230),
                "shape_angle": None,
                "match": type("Match", (), {"face": "U", "color": "white", "confidence": 0.82})(),
                "grid_spacing": 70.0,
                "outside_component_hull_distance": outside,
                "nearest_component_distance": nearest,
                "outside_grid_component_hull_distance": outside,
                "nearest_grid_component_distance": nearest,
            },
        )()

    def grid(stickers):
        return type(
            "Grid",
            (),
            {
                "cube_hull_inside_count": 6,
                "matched_count": 6,
                "stickers": stickers,
            },
        )()

    clean = grid([[facelet("component") for _ in range(3)] for _ in range(3)])
    extrapolated = grid(
        [
            [facelet("component"), facelet("component"), facelet("component")],
            [facelet("component"), facelet("component"), facelet("component")],
            [
                facelet("grid_sample", outside=105.0, nearest=170.0),
                facelet("grid_sample", outside=105.0, nearest=170.0),
                facelet("grid_sample", outside=105.0, nearest=170.0),
            ],
        ]
    )

    assert recognizer._grid_cube_hull_penalty(clean) == 0.0
    assert recognizer._grid_cube_hull_penalty(extrapolated) == 200.0


def test_unsupported_grid_sample_score_measures_white_extrapolation():
    def sample(
        source,
        *,
        outside=0.0,
        nearest=70.0,
        spacing=70.0,
        grid_outside=None,
        grid_nearest=None,
    ):
        facelet = type(
            "Facelet",
            (),
            {
                "source": source,
                "rgb": (230, 230, 230),
                "shape_angle": None,
                "match": type(
                    "Match",
                    (),
                    {
                        "face": "U",
                        "color": "white",
                        "confidence": 0.82,
                        "alternatives": [("white", 0.0)],
                    },
                )(),
                "grid_spacing": spacing,
                "outside_component_hull_distance": outside,
                "nearest_component_distance": nearest,
                "outside_grid_component_hull_distance": outside if grid_outside is None else grid_outside,
                "nearest_grid_component_distance": nearest if grid_nearest is None else grid_nearest,
            },
        )()
        return facelet

    supported = sample("grid_sample", outside=0.0, nearest=70.0)
    unsupported = sample("grid_sample", outside=70.0, nearest=120.0)
    grid_local_unsupported = sample(
        "grid_sample",
        outside=0.0,
        nearest=70.0,
        grid_outside=70.0,
        grid_nearest=120.0,
    )

    assert (
        recognizer._unsupported_grid_sample_score(unsupported)
        > recognizer._unsupported_grid_sample_score(supported) + 2.0
    )
    assert recognizer._unsupported_grid_sample_score(grid_local_unsupported) > 2.0
    assert recognizer._unsupported_grid_sample_score(sample("component", outside=70.0, nearest=120.0)) == 0.0
    zero_spacing_sample = sample(
        "grid_sample",
        outside=70.0,
        nearest=120.0,
        spacing=0.0,
    )
    assert recognizer._unsupported_grid_sample_score(zero_spacing_sample) == 0.0


def test_grid_extrapolation_penalty_targets_sparse_unsupported_white_samples():
    def sample(source, color="white", *, outside=0.0, nearest=70.0, spacing=70.0):
        return type(
            "Facelet",
            (),
            {
                "source": source,
                "rgb": (230, 230, 230),
                "shape_angle": None,
                "match": type(
                    "Match",
                    (),
                    {
                        "face": "U" if color == "white" else "F",
                        "color": color,
                        "confidence": 0.82,
                        "alternatives": [(color, 0.0)],
                    },
                )(),
                "grid_spacing": spacing,
                "outside_grid_component_hull_distance": outside,
                "nearest_grid_component_distance": nearest,
            },
        )()

    unsupported = sample("grid_sample", outside=90.0, nearest=140.0)
    supported = sample("grid_sample", outside=0.0, nearest=70.0)
    component = sample("component", outside=90.0, nearest=140.0)
    sparse_grid = type(
        "Grid",
        (),
        {
            "matched_count": 5,
            "stickers": [
                [unsupported, unsupported, supported],
                [component, unsupported, component],
                [component, component, component],
            ],
        },
    )()
    strong_grid = type(
        "Grid",
        (),
        {
            "matched_count": 7,
            "stickers": sparse_grid.stickers,
        },
    )()

    assert recognizer._extrapolated_grid_sample_score(sample("grid_sample", "green", outside=90.0)) > 0.0
    assert recognizer._grid_extrapolated_sample_count(sparse_grid) == 3
    assert recognizer._grid_extrapolation_penalty(sparse_grid) > 0.0
    assert recognizer._grid_extrapolation_penalty(strong_grid) == 0.0


def test_pair_color_calibration_signal_reports_red_orange_counts():
    from rubik_recognizer.colors import ColorMatch

    def sticker(face, rgb):
        return type(
            "Sticker",
            (),
            {
                "id": id(rgb),
                "center": (0, 0),
                "rgb": rgb,
                "match": ColorMatch(
                    {"R": "red", "L": "orange", "U": "white"}[face],
                    face,
                    0.0,
                    1.0,
                    [({"R": "red", "L": "orange", "U": "white"}[face], 0.0)],
                ),
            },
        )()

    raw_a = StubAnalysis(["U", "R", "R", "R", "R", "R", "L"])
    raw_b = StubAnalysis(["D", "L", "L", "L", "L", "L", "R"])
    calibrated_a = StubAnalysis(["U", "R", "R", "R", "L", "L", "L"])
    calibrated_b = StubAnalysis(["D", "L", "L", "L", "R", "R", "R"])
    raw_a.stickers = [sticker("R", (180, 55, 45)) for _ in range(5)] + [sticker("L", (220, 115, 45))]
    raw_b.stickers = [sticker("L", (220, 115, 45)) for _ in range(5)] + [sticker("R", (180, 55, 45))]
    calibrated_a.stickers = [sticker("R", (180, 55, 45)) for _ in range(3)] + [sticker("L", (220, 115, 45)) for _ in range(3)]
    calibrated_b.stickers = [sticker("L", (220, 115, 45)) for _ in range(3)] + [sticker("R", (180, 55, 45)) for _ in range(3)]
    raw_result = RecognitionResult(
        status="rejected",
        failed_checks=["R_count_not_9", RED_ORANGE_PAIR_CALIBRATION_SUSPECTED_CHECK],
    )
    calibrated_result = RecognitionResult(
        status="rejected",
        failed_checks=["piece_legality_invalid", RED_ORANGE_PAIR_CALIBRATION_SUSPECTED_CHECK],
    )

    signal = _pair_color_calibration_signal(raw_a, raw_b, calibrated_a, calibrated_b, raw_result, calibrated_result)

    assert signal["rawFailedChecks"] == ["R_count_not_9", RED_ORANGE_PAIR_CALIBRATION_SUSPECTED_CHECK]
    assert signal["calibratedFailedChecks"] == ["piece_legality_invalid", RED_ORANGE_PAIR_CALIBRATION_SUSPECTED_CHECK]
    assert signal["anchorCounts"]["red"] == 2
    assert signal["anchorCounts"]["orange"] == 2
    assert signal["images"]["imageA"]["rawRedOrangeSkew"] == {
        "redCount": 5,
        "orangeCount": 1,
        "gap": 4,
        "dominantFace": "R",
    }
    assert signal["images"]["imageB"]["rawRedOrangeSkew"]["dominantFace"] == "L"
    raw_face_evidence = signal["images"]["imageA"]["selectedFaceEvidence"]["raw"]
    assert raw_face_evidence["R"]["centerColor"] == "red"
    assert raw_face_evidence["R"]["centerDistances"]["red"] < raw_face_evidence["R"]["centerDistances"]["orange"]
    assert raw_face_evidence["R"]["cellFaceEvidence"] == {"U": 0, "R": 0, "L": 0}


def test_pair_color_calibration_signal_attaches_only_to_final_red_orange_check():
    result = RecognitionResult(
        status="rejected",
        failed_checks=["R_count_not_9"],
        recognition_signals={},
    )
    calibrated_result = RecognitionResult(
        status="rejected",
        failed_checks=[RED_ORANGE_PAIR_CALIBRATION_SUSPECTED_CHECK],
    )

    _attach_failed_pair_color_calibration_signal(
        result,
        calibrated_result,
        StubAnalysis(["U", "R", "L"]),
        StubAnalysis(["D", "R", "L"]),
        StubAnalysis(["U", "R", "L"]),
        StubAnalysis(["D", "R", "L"]),
    )

    assert "pairColorCalibration" not in result.recognition_signals


def test_state_candidates_reuse_facelet_options_cache(monkeypatch):
    facelet = object()
    merged = {
        face: [[facelet for _ in range(3)] for _ in range(3)]
        for face in recognizer.FACE_ORDER
    }
    merged["_score"] = 100.0
    workset = RecognitionWorkset(
        options_a=[],
        options_b=[],
        merged_candidates=[(100.0, merged), (99.0, merged)],
    )
    calls = {"options": 0}

    def fake_facelet_options(value):
        calls["options"] += 1
        assert value is facelet
        return [("U", 0.0)]

    monkeypatch.setattr(recognizer, "_facelet_options", fake_facelet_options)

    candidates = WhiteUpRecognizer()._state_candidates_from_workset(workset)

    assert len(candidates) == 2
    assert calls == {"options": 1}
    assert len(workset.facelet_options_by_key) == 1


def test_candidate_face_count_diagnostics_reuses_facelet_options_cache(monkeypatch):
    facelet = object()
    merged = {
        face: [[facelet for _ in range(3)] for _ in range(3)]
        for face in recognizer.FACE_ORDER
    }
    cache = {}
    calls = {"options": 0}

    def fake_facelet_options(value):
        calls["options"] += 1
        assert value is facelet
        return [("U", 0.0)]

    monkeypatch.setattr(recognizer, "_facelet_options", fake_facelet_options)

    recognizer._candidate_face_count_diagnostics(
        [(100.0, merged), (99.0, merged)],
        facelet_options_cache=cache,
    )

    assert calls == {"options": 1}
    assert len(cache) == 1


def test_merged_face_candidates_precomputes_option_signatures(monkeypatch):
    options_a = [
        {
            "U": [["U"] * 3 for _ in range(3)],
            "R": [["R"] * 3 for _ in range(3)],
            "F": [["F"] * 3 for _ in range(3)],
            "_score": 10.0,
        },
        {
            "U": [["U"] * 3 for _ in range(3)],
            "R": [["R"] * 3 for _ in range(3)],
            "F": [["B"] * 3 for _ in range(3)],
            "_score": 9.0,
        },
    ]
    options_b = [
        {
            "D": [["D"] * 3 for _ in range(3)],
            "L": [["L"] * 3 for _ in range(3)],
            "B": [["B"] * 3 for _ in range(3)],
            "_score": 8.0,
        },
        {
            "D": [["D"] * 3 for _ in range(3)],
            "L": [["F"] * 3 for _ in range(3)],
            "B": [["B"] * 3 for _ in range(3)],
            "_score": 7.0,
        },
        {
            "D": [["U"] * 3 for _ in range(3)],
            "L": [["L"] * 3 for _ in range(3)],
            "B": [["B"] * 3 for _ in range(3)],
            "_score": 6.0,
        },
    ]
    calls = {"signatures": 0}
    original_face_signature = recognizer._face_signature

    def counting_face_signature(faces):
        calls["signatures"] += 1
        return original_face_signature(faces)

    monkeypatch.setattr(recognizer, "_face_signature", counting_face_signature)

    merged = _merged_face_candidates(options_a, options_b)

    assert len(merged) == len(options_a) * len(options_b)
    assert calls == {"signatures": len(options_a) + len(options_b)}
    assert all("_option_signature_a" in faces for _, faces in merged)
    assert all("_option_signature_b" in faces for _, faces in merged)


def test_grid_matrix_for_orientation_uses_provided_context_flex(monkeypatch):
    sticker = type("Sticker", (), {"source": "component"})()
    grid = type(
        "Grid",
        (),
        {
            "stickers": [[sticker for _ in range(3)] for _ in range(3)],
        },
    )()

    def fail_score(candidate):
        raise AssertionError("provided flex should skip recomputing grid context score")

    monkeypatch.setattr(recognizer, "_grid_context_repair_score", fail_score)
    monkeypatch.setattr(
        recognizer,
        "_grid_contextual_facelet",
        lambda sticker, candidate, flex: (_ for _ in ()).throw(
            AssertionError("below-threshold flex should keep original stickers directly")
        ),
    )

    matrix = _grid_matrix_for_orientation(grid, flex=0.0)

    assert matrix == grid.stickers
    assert matrix is not grid.stickers


def test_grid_contextual_facelet_copies_sticker_without_generic_copy(monkeypatch):
    from rubik_recognizer.colors import ColorMatch
    from rubik_recognizer.image_pipeline import Sticker

    match = ColorMatch("white", "U", 0.0, 1.0, [("white", 0.0)])
    sticker = Sticker(
        id=7,
        center=(1.0, 2.0),
        bbox=(0.0, 1.0, 2.0, 3.0),
        rgb=(230, 230, 230),
        match=match,
        area=42,
        shape_angle=12.5,
    )
    grid = type("Grid", (), {"id": 99})()

    monkeypatch.setattr(
        recognizer.copy,
        "copy",
        lambda value: (_ for _ in ()).throw(AssertionError("Sticker path should not use generic copy.copy")),
    )

    contextual = recognizer._grid_contextual_facelet(sticker, grid, 1.25)

    assert contextual is not sticker
    assert contextual.id == sticker.id
    assert contextual.center == sticker.center
    assert contextual.bbox == sticker.bbox
    assert contextual.rgb == sticker.rgb
    assert contextual.match is sticker.match
    assert contextual.area == sticker.area
    assert contextual.source == sticker.source
    assert contextual.shape_angle == sticker.shape_angle
    assert contextual.grid_repair_flex == 1.25
    assert contextual.grid_context_id == 99


def test_oriented_options_cache_grid_context_flex_per_grid(monkeypatch):
    def grid(grid_id, x_offset):
        sticker = type("Sticker", (), {"source": "component"})()
        points = [[(x_offset + c * 10, r * 10) for c in range(3)] for r in range(3)]
        return type(
            "Grid",
            (),
            {
                "id": grid_id,
                "points": points,
                "stickers": [[sticker for _ in range(3)] for _ in range(3)],
            },
        )()

    grids = {"U": grid(1, 0), "F": grid(2, 40), "R": grid(3, 80)}
    calls = {"context": 0}

    def fake_context_score(candidate):
        calls["context"] += 1
        return 0.0

    monkeypatch.setattr(recognizer, "_grid_context_repair_score", fake_context_score)
    monkeypatch.setattr(recognizer, "_ranked_transforms", lambda requirements, weights: recognizer.TRANSFORMS[:2])
    monkeypatch.setattr(recognizer, "_visible_piece_plausibility_score", lambda oriented: 0.0)

    options = _oriented_options_for_grid_map(grids, "U")

    assert options
    assert calls == {"context": 3}


def test_oriented_options_reuses_contextual_transformed_matrices(monkeypatch):
    def grid(grid_id, x_offset):
        sticker = type("Sticker", (), {"source": "component"})()
        points = [[(x_offset + c * 10, r * 10) for c in range(3)] for r in range(3)]
        return type(
            "Grid",
            (),
            {
                "id": grid_id,
                "points": points,
                "stickers": [[sticker for _ in range(3)] for _ in range(3)],
            },
        )()

    grids = {"U": grid(1, 0), "F": grid(2, 40), "R": grid(3, 80)}
    calls = {"matrix": 0}
    original_matrix = recognizer._grid_matrix_for_orientation

    def counting_matrix(candidate, *, flex=None):
        calls["matrix"] += 1
        return original_matrix(candidate, flex=flex)

    monkeypatch.setattr(recognizer, "_grid_context_repair_score", lambda candidate: 1.25)
    monkeypatch.setattr(recognizer, "_ranked_transforms", lambda requirements, weights: recognizer.TRANSFORMS[:2])
    monkeypatch.setattr(recognizer, "_visible_piece_plausibility_score", lambda oriented: 0.0)
    monkeypatch.setattr(recognizer, "_grid_matrix_for_orientation", counting_matrix)

    options = _oriented_options_for_grid_map(grids, "U")

    assert options
    assert calls == {"matrix": 3}


def test_ranked_visible_face_triples_rescues_small_overlap_when_strict_empty():
    def grid(grid_id, component_ids, x_offset):
        stickers = []
        ids = list(component_ids)
        for row in range(3):
            sticker_row = []
            for col in range(3):
                component_id = ids[(row * 3 + col) % len(ids)]
                sticker_row.append(type("Sticker", (), {"id": component_id, "source": "component", "shape_angle": None})())
            stickers.append(sticker_row)
        points = [[(x_offset + col * 10, row * 10) for col in range(3)] for row in range(3)]
        return type(
            "Grid",
            (),
            {
                "id": grid_id,
                "matched_count": len(set(component_ids)),
                "fit_error": 0.0,
                "points": points,
                "stickers": stickers,
            },
        )()

    grids_by_face = {
        "U": [grid(1, range(9), 0)],
        "B": [grid(2, (0, 1, 2, 20, 21, 22, 23), 10)],
        "L": [grid(3, (3, 4, 5, 30, 31, 32, 33), 20)],
    }

    triples = recognizer._ranked_visible_face_triples(grids_by_face, "U")

    assert len(triples) == 1
    assert set(triples[0][1]) == {"U", "B", "L"}
    assert recognizer._triple_overlap_count(triples[0][1].values()) == 6


def test_grid_usable_for_triple_rejects_low_match_suspect_samples(monkeypatch):
    grid = type("Grid", (), {"matched_count": 5})()

    monkeypatch.setattr(recognizer, "_grid_suspect_sample_score", lambda candidate: 3.1)
    monkeypatch.setattr(recognizer, "_grid_bad_sample_count", lambda candidate: 0)

    assert not recognizer._grid_usable_for_triple(grid)


def test_triple_rejects_contaminated_side_when_anchor_evidence_collapses(monkeypatch):
    anchor_grid = type("Grid", (), {"matched_count": 9})()
    contaminated_grid = type("Grid", (), {"matched_count": 6})()
    clean_grid = type("Grid", (), {"matched_count": 9})()

    monkeypatch.setattr(recognizer, "_grid_cell_face_counts", lambda candidate: {"U": 2})
    monkeypatch.setattr(
        recognizer,
        "_grid_suspect_sample_score",
        lambda candidate: 3.1 if candidate is contaminated_grid else 0.0,
    )
    monkeypatch.setattr(
        recognizer,
        "_grid_bad_sample_count",
        lambda candidate: 2 if candidate is contaminated_grid else 0,
    )

    assert recognizer._triple_has_collapsed_anchor_contamination(
        anchor_grid,
        contaminated_grid,
        clean_grid,
        "U",
    )


def test_triple_keeps_contaminated_side_when_anchor_evidence_is_solid(monkeypatch):
    anchor_grid = type("Grid", (), {"matched_count": 9})()
    contaminated_grid = type("Grid", (), {"matched_count": 6})()
    clean_grid = type("Grid", (), {"matched_count": 9})()

    monkeypatch.setattr(recognizer, "_grid_cell_face_counts", lambda candidate: {"U": 4})
    monkeypatch.setattr(recognizer, "_grid_suspect_sample_score", lambda candidate: 3.1)
    monkeypatch.setattr(recognizer, "_grid_bad_sample_count", lambda candidate: 2)

    assert not recognizer._triple_has_collapsed_anchor_contamination(
        anchor_grid,
        contaminated_grid,
        clean_grid,
        "U",
    )


def test_face_triple_failure_check_reports_low_quality_overlap_rescue():
    def grid(grid_id, component_ids, x_offset, matched_count=6, fit_error=0.0):
        stickers = []
        ids = list(component_ids)
        for row in range(3):
            sticker_row = []
            for col in range(3):
                component_id = ids[(row * 3 + col) % len(ids)]
                sticker_row.append(type("Sticker", (), {"id": component_id, "source": "component", "shape_angle": None})())
            stickers.append(sticker_row)
        points = [[(x_offset + col * 10, row * 10) for col in range(3)] for row in range(3)]
        return type(
            "Grid",
            (),
            {
                "id": grid_id,
                "matched_count": matched_count,
                "fit_error": fit_error,
                "points": points,
                "stickers": stickers,
            },
        )()

    grids_by_face = {
        "D": [grid(1, range(9), 0, matched_count=8)],
        "L": [grid(2, (0, 1, 2, 20, 21, 22), 10, fit_error=20.0)],
        "F": [grid(3, (3, 4, 5, 30, 31, 32), 20, fit_error=20.0)],
    }

    assert recognizer._ranked_visible_face_triples(grids_by_face, "D") == []
    assert (
        recognizer._face_triple_failure_check("image_b", grids_by_face, "D")
        == "image_b_face_triple_overlap_low_quality"
    )
    assert (
        recognizer._reason_for_checks(["image_b_face_triple_overlap_low_quality"])
        == "Image B only produced overlapping or low-quality three-face grids; retake with clearer face separation."
    )


def test_repair_details_memoizes_signature_stable_work(monkeypatch):
    solved = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

    def merged_candidate(side_pair_a):
        merged = {
            face: [[face for _ in range(3)] for _ in range(3)]
            for face in recognizer.FACE_ORDER
        }
        merged.update(
            {
                "_score": 100.0,
                "_score_a": 50.0,
                "_score_b": 50.0,
                "_side_pair_a": side_pair_a,
                "_side_pair_b": ("L", "B"),
                "_ordered_side_pair_a": side_pair_a,
                "_ordered_side_pair_b": ("L", "B"),
                "_orientation_rank_a": 0,
                "_orientation_rank_b": 0,
            }
        )
        return merged

    first = merged_candidate(("F", "R"))
    second = merged_candidate(("R", "B"))
    workset = RecognitionWorkset(
        options_a=[],
        options_b=[],
        merged_candidates=[(100.0, first), (99.0, second)],
    )
    calls = {"legal": 0, "conflicts": 0, "counts": 0, "penalty": 0}
    conflicts = {key: 0 for key in recognizer.PIECE_CONFLICT_KEYS}
    counts = {face: 9 for face in recognizer.FACE_ORDER}

    def fake_legal_repair(faces):
        calls["legal"] += 1
        return solved, 10.0, 2

    def fake_conflicts(faces):
        calls["conflicts"] += 1
        return conflicts

    def fake_counts(faces):
        calls["counts"] += 1
        return counts

    def fake_penalty(conflict_summary, faces, *, repair_cost, repair_changes, face_counts=None):
        calls["penalty"] += 1
        assert conflict_summary is conflicts
        assert face_counts is counts
        return 0.0

    monkeypatch.setattr(recognizer, "_legal_repaired_state_from_faces", fake_legal_repair)
    monkeypatch.setattr(recognizer, "_piece_conflict_summary", fake_conflicts)
    monkeypatch.setattr(recognizer, "_primary_face_counts", fake_counts)
    monkeypatch.setattr(recognizer, "_repair_ranking_penalty", fake_penalty)

    details = WhiteUpRecognizer()._legal_repair_candidate_details_from_workset(workset)

    assert len(details) == 1
    assert details[0]["state"] == solved
    assert calls == {"legal": 1, "conflicts": 1, "counts": 1, "penalty": 2}
    assert len(workset.repaired_state_by_signature) == 1
    assert len(workset.conflicts_by_signature) == 1
    assert len(workset.face_counts_by_signature) == 1


def test_recognition_result_exposes_additive_signals():
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        recognition_signals={
            "repairPathUsed": True,
            "repairCandidateCount": 1,
            "topRepairCandidates": [{"state": "U" * 54, "repairCost": 1.25}],
        },
    )

    payload = result.to_api_dict(include_overlays=False)

    assert payload["recognitionSignals"]["repairPathUsed"] is True
    assert payload["recognitionSignals"]["topRepairCandidates"][0]["repairCost"] == 1.25
    assert payload["recognitionCategory"] == "reject_retake"


def test_recognition_signals_support_versioned_repair_candidate_conflicts():
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        recognition_signals={
            "schemaVersion": 1,
            "repairPathUsed": True,
            "repairCandidateCount": 1,
            "topRepairCandidates": [
                {
                    "state": "U" * 54,
                    "repairCost": 1.25,
                    "repairChanges": 2,
                    "preRepairConflicts": {"invalidCorners": 0, "invalidEdges": 1, "totalConflicts": 1},
                }
            ],
        },
    )

    signals = result.to_api_dict(include_overlays=False)["recognitionSignals"]

    assert signals["schemaVersion"] == 1
    assert signals["topRepairCandidates"][0]["preRepairConflicts"]["invalidCorners"] == 0
    assert signals["topRepairCandidates"][0]["preRepairConflicts"]["invalidEdges"] == 1


def test_selected_faces_by_image_uses_winning_side_pairs():
    faces = _selected_faces_by_image("B/L", "F/R")

    assert faces == {
        "imageA": ["B", "L", "U"],
        "imageB": ["D", "F", "R"],
    }


def test_selected_sides_by_image_preserves_photo_order():
    sides = _selected_sides_by_image("L/B", "F/R")

    assert sides == {
        "imageA": {"left": "L", "right": "B"},
        "imageB": {"left": "F", "right": "R"},
    }


def test_selected_faces_signal_reports_standard_capture_yaw():
    from rubik_recognizer.recognizer import _selected_faces_signal

    signal = _selected_faces_signal({"orderedSidePairA": "F/R", "orderedSidePairB": "L/B"})

    assert signal["captureYaw"]["status"] == "standard"
    assert signal["captureYaw"]["quarterTurns"] == 0
    assert signal["captureYaw"]["degrees"] == 0


def test_selected_faces_signal_reports_nonstandard_capture_yaw():
    from rubik_recognizer.recognizer import _selected_faces_signal

    state = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    signal = _selected_faces_signal({"orderedSidePairA": "R/B", "orderedSidePairB": "F/L"}, state=state)

    assert signal["captureYaw"]["status"] == "nonstandard"
    assert signal["captureYaw"]["quarterTurns"] == 1
    assert signal["captureYaw"]["requiresNormalization"] is True
    assert signal["captureYaw"]["normalizationApplied"] is True
    assert signal["captureYaw"]["stateFrame"] == "wca"
    assert signal["captureYaw"]["captureFrameState"] == _state_to_capture_yaw(state, 1)


def test_capture_yaw_signal_supports_all_white_up_yaws():
    from rubik_recognizer.recognizer import _selected_faces_signal

    state = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    cases = [
        (0, "F/R", "L/B", "standard"),
        (1, "R/B", "F/L", "nonstandard"),
        (2, "B/L", "R/F", "nonstandard"),
        (3, "L/F", "B/R", "nonstandard"),
    ]

    for yaw, side_pair_a, side_pair_b, status in cases:
        signal = _selected_faces_signal(
            {"orderedSidePairA": side_pair_a, "orderedSidePairB": side_pair_b},
            state=state,
        )
        capture_yaw = signal["captureYaw"]

        assert capture_yaw["status"] == status
        assert capture_yaw["quarterTurns"] == yaw
        assert capture_yaw["degrees"] == yaw * 90
        assert capture_yaw["requiresNormalization"] is (yaw != 0)
        assert capture_yaw["normalizationApplied"] is (yaw != 0)
        assert capture_yaw["stateFrame"] == "wca"
        assert capture_yaw["captureFrameState"] == _state_to_capture_yaw(state, yaw)
        assert _capture_yaw_state_to_wca(capture_yaw["captureFrameState"], yaw) == state


def test_recognition_category_accepts_normalized_nonstandard_capture_yaw():
    signals = {
        "repairPathUsed": False,
        "captureYaw": {
            "status": "nonstandard",
            "quarterTurns": 1,
            "degrees": 90,
            "requiresNormalization": True,
            "normalizationApplied": True,
        },
        "selectedGridQuality": {
            "imageA": {
                "U": {"matchedCount": 9, "fitError": 0.5, "quality": 100, "badSamples": 0, "suspectSamples": 0},
                "R": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
                "B": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
            },
            "imageB": {
                "D": {"matchedCount": 9, "fitError": 0.5, "quality": 100, "badSamples": 0, "suspectSamples": 0},
                "F": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
                "L": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
            },
        },
    }
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        confidence=0.847,
        reason="Recognized a unique legal white-up cube state.",
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_clean"


def test_recognition_category_demotes_unnormalized_nonstandard_capture_yaw():
    signals = {
        "repairPathUsed": False,
        "captureYaw": {
            "status": "nonstandard",
            "quarterTurns": 1,
            "degrees": 90,
            "requiresNormalization": True,
            "normalizationApplied": False,
        },
    }
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        confidence=0.847,
        reason="Recognized a unique legal white-up cube state.",
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "needs_manual_review"
    assert category["reason"] == "nonstandard_capture_yaw_without_normalization"


def test_capture_yaw_state_transform_matches_saved_capture_frame_examples():
    set_32_raw = "DBDRUFUFLULBBBDBLURDFDRULFLFRURDUDFRBDFLFBRUDRRLLLBBUF"
    set_32_wca = "DFLBUFDRURDFDRULFLBDFLFBRUDDRFFDRRUURRLLLBBUFULBBBDBLU"
    set_12_raw = "LBFRUDRUDFBRUFLBRRBLRULRLLLDBULDBFRBFDDDBFUFBUUDDRFUFL"
    set_12_wca = "RRLUUBDDFUUDDRFUFLFBRUFLBRRUBBBDRDLFBLRULRLLLFDDDBFUFB"

    assert _state_to_capture_yaw(set_32_wca, 1) == set_32_raw
    assert _capture_yaw_state_to_wca(set_32_raw, 1) == set_32_wca
    assert _state_to_capture_yaw(set_12_wca, 3) == set_12_raw
    assert _capture_yaw_state_to_wca(set_12_raw, 3) == set_12_wca


def test_recognition_signal_sample_fixtures_have_stable_shape():
    fixture_dir = Path(__file__).parent / "fixtures"
    direct = json.loads((fixture_dir / "recognition_signals_direct.json").read_text())
    repair = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())

    direct_signals = direct["recognitionSignals"]
    repair_signals = repair["recognitionSignals"]

    assert direct["recognitionCategory"] == "success_clean"
    assert repair["recognitionCategory"] == "success_repaired_high_confidence"

    assert direct_signals["schemaVersion"] == 1
    assert direct_signals["repairPathUsed"] is False
    assert "topRepairCandidates" not in direct_signals
    assert "selectedRepairCandidate" not in direct_signals

    assert repair_signals["schemaVersion"] == 1
    assert repair_signals["repairPathUsed"] is True
    assert repair_signals["topRepairCandidates"]
    assert repair_signals["selectedRepairCandidate"]["state"] == repair_signals["topRepairCandidates"][0]["state"]

    conflicts = repair_signals["topRepairCandidates"][0]["preRepairConflicts"]
    for key in PIECE_CONFLICT_KEYS:
        assert key in conflicts

    assert repair_signals["topRepairCandidates"][0]["baseConfidence"] > repair_signals["topRepairCandidates"][0]["confidence"]
    assert repair_signals["topRepairCandidates"][0]["repairRankingPenalty"] > 0


def test_recognition_category_marks_rejected_as_retake():
    result = RecognitionResult(status="rejected", reason="No legal cube state matched the detected stickers.")

    category = _recognition_category_payload(result)

    assert category["category"] == "reject_retake"
    assert category["reason"] == "recognizer_rejected"


def _set_30_style_top_visible_triple_quality(component_overlap=6):
    return {
        "imageA": {
            "componentOverlap": component_overlap,
            "sidePair": "B/L",
            "grids": {
                "B": {"gridId": 22, "cellFaceCounts": {"B": 2, "L": 5, "U": 1, "D": 1}},
                "L": {"gridId": 5, "cellFaceCounts": {"L": 2, "F": 3, "U": 2, "D": 1, "R": 1}},
                "U": {"gridId": 12, "cellFaceCounts": {"U": 1, "B": 4, "L": 2, "F": 1, "R": 1}},
            },
        },
        "imageB": {
            "componentOverlap": 1,
            "sidePair": "F/L",
            "grids": {
                "D": {"gridId": 13, "cellFaceCounts": {"D": 2, "B": 4, "U": 2, "R": 1}},
                "F": {"gridId": 5, "cellFaceCounts": {"F": 3, "R": 3, "B": 1, "D": 1, "L": 1}},
                "L": {"gridId": 20, "cellFaceCounts": {"L": 2, "B": 2, "D": 2, "R": 1, "U": 2}},
            },
        },
    }


def test_recognition_category_marks_direct_unique_as_clean():
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_direct.json").read_text())
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=payload["confidence"],
        reason=payload["reason"],
        recognition_signals=payload["recognitionSignals"],
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_clean"


def test_recognition_category_filters_artifact_grids_by_selected_faces():
    signals = {
        "repairPathUsed": False,
        "selectedFacesByImage": {
            "imageA": ["F", "R", "U"],
            "imageB": ["B", "D", "L"],
        },
        "selectedGridQuality": {
            "imageA": {
                "U": {"matchedCount": 9, "fitError": 0.5, "quality": 100, "badSamples": 0, "suspectSamples": 0},
                "F": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
                "R": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
                "L": {"matchedCount": 8, "fitError": 16.4, "quality": 72, "badSamples": 0, "suspectSamples": 0},
            },
            "imageB": {
                "D": {"matchedCount": 9, "fitError": 0.5, "quality": 100, "badSamples": 0, "suspectSamples": 0},
                "B": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
                "L": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
                "R": {"matchedCount": 8, "fitError": 16.4, "quality": 72, "badSamples": 0, "suspectSamples": 0},
            },
        },
    }
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        confidence=0.838,
        reason="Recognized a unique legal white-up cube state.",
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_clean"


def test_recognition_category_demotes_weak_selected_visible_grid():
    signals = {
        "repairPathUsed": False,
        "selectedFacesByImage": {"imageA": ["F", "R", "U"], "imageB": ["B", "D", "L"]},
        "selectedGridQuality": {
            "imageA": {
                "U": {"matchedCount": 9, "fitError": 0.5, "quality": 100, "badSamples": 0, "suspectSamples": 0},
                "F": {"matchedCount": 8, "fitError": 16.4, "quality": 72, "badSamples": 0, "suspectSamples": 0},
                "R": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
            },
            "imageB": {
                "D": {"matchedCount": 9, "fitError": 0.5, "quality": 100, "badSamples": 0, "suspectSamples": 0},
                "B": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
                "L": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
            },
        },
    }
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        confidence=0.838,
        reason="Recognized a unique legal white-up cube state.",
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "needs_manual_review"


def test_recognition_category_accepts_marginal_but_supported_selected_grids():
    signals = {
        "repairPathUsed": False,
        "selectedFacesByImage": {"imageA": ["F", "R", "U"], "imageB": ["B", "D", "L"]},
        "selectedGridQuality": {
            "imageA": {
                "U": {"matchedCount": 7, "fitError": 0.768, "quality": 116.0, "badSamples": 0, "suspectSamples": 0.0},
                "F": {"matchedCount": 7, "fitError": 1.415, "quality": 115.661, "badSamples": 0, "suspectSamples": 0.0},
                "R": {"matchedCount": 6, "fitError": 2.16, "quality": 69.544, "badSamples": 0, "suspectSamples": 1.4},
            },
            "imageB": {
                "B": {"matchedCount": 8, "fitError": 1.167, "quality": 135.915, "badSamples": 0, "suspectSamples": 0.0},
                "D": {"matchedCount": 9, "fitError": 0.953, "quality": 156.349, "badSamples": 0, "suspectSamples": 0.0},
                "L": {"matchedCount": 5, "fitError": 6.455, "quality": 61.641, "badSamples": 0, "suspectSamples": 0.0},
            },
        },
    }
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        confidence=0.8124,
        reason="Recognized a unique legal white-up cube state.",
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_clean"


def test_recognition_category_filters_by_dynamic_yaw():
    signals = {
        "repairPathUsed": False,
        "selectedFacesByImage": {"imageA": ["B", "L", "U"], "imageB": ["D", "F", "R"]},
        "selectedGridQuality": {
            "imageA": {
                "U": {"matchedCount": 9, "fitError": 0.5, "quality": 100, "badSamples": 0, "suspectSamples": 0},
                "L": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
                "B": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
                "R": {"matchedCount": 8, "fitError": 16.4, "quality": 72, "badSamples": 0, "suspectSamples": 0},
            },
            "imageB": {
                "D": {"matchedCount": 9, "fitError": 0.5, "quality": 100, "badSamples": 0, "suspectSamples": 0},
                "F": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
                "R": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
                "B": {"matchedCount": 8, "fitError": 16.4, "quality": 72, "badSamples": 0, "suspectSamples": 0},
            },
        },
    }
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        confidence=0.838,
        reason="Recognized a unique legal white-up cube state.",
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_clean"


def test_recognition_category_counts_all_grids_when_selected_faces_missing():
    signals = {
        "repairPathUsed": False,
        "selectedGridQuality": {
            "imageA": {
                "U": {"matchedCount": 9, "fitError": 0.5, "quality": 100, "badSamples": 0, "suspectSamples": 0},
                "F": {"matchedCount": 8, "fitError": 2.0, "quality": 96, "badSamples": 0, "suspectSamples": 0},
                "L": {"matchedCount": 8, "fitError": 16.4, "quality": 72, "badSamples": 0, "suspectSamples": 0},
            }
        },
    }
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        confidence=0.838,
        reason="Recognized a unique legal white-up cube state.",
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "needs_manual_review"


def test_recognition_category_grid_purity_guard_demotes_direct_unique_to_manual_review():
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        confidence=0.847,
        reason="Recognized a unique legal white-up cube state.",
        recognition_signals={
            "repairPathUsed": False,
            "topVisibleTripleQuality": _set_30_style_top_visible_triple_quality(),
        },
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "needs_manual_review"
    assert category["reason"] == "grid_purity_guard"


def test_recognition_category_grid_purity_guard_requires_high_component_overlap():
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        confidence=0.847,
        reason="Recognized a unique legal white-up cube state.",
        recognition_signals={
            "repairPathUsed": False,
            "topVisibleTripleQuality": _set_30_style_top_visible_triple_quality(component_overlap=5),
        },
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_clean"


def test_recognition_category_marks_low_penalty_repair_as_high_confidence():
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=payload["confidence"],
        reason=payload["reason"],
        recognition_signals=payload["recognitionSignals"],
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_repaired_high_confidence"


def test_recognition_category_grid_purity_guard_demotes_high_confidence_repair_to_manual_review():
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())
    signals = json.loads(json.dumps(payload["recognitionSignals"]))
    signals["topVisibleTripleQuality"] = _set_30_style_top_visible_triple_quality()
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=payload["confidence"],
        reason=payload["reason"],
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "needs_manual_review"
    assert category["reason"] == "grid_purity_guard"


def test_recognition_category_keeps_unstable_standard_backfill_manual_review():
    signals = {
        "repairPathUsed": True,
        "repairCandidateCount": 100_000,
        "repairBackfillAttempted": True,
        "repairBackfillEvaluatedMerges": 24,
        "repairBackfillUsed": True,
        "repairBackfillProbeReason": "unstable_standard_repair",
        "selectedRepairCandidate": {
            "confidence": 0.631,
            "repairRankingPenalty": 0.08,
            "repairSource": "conflict_backfill",
            "preRepairFaceCounts": {face: 9 for face in "URFDLB"},
            "preRepairConflicts": {"totalConflicts": 2, "validCorners": 8},
        },
    }
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        confidence=0.631,
        reason="Recognized the highest-scoring legal cube state after cubie-level color repair.",
        candidates=100_000,
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "needs_manual_review"
    assert category["reason"] == "repair_backfill_from_unstable_standard_repair"


def test_recognition_category_allows_no_standard_backfill_high_confidence_repair():
    signals = {
        "repairPathUsed": True,
        "repairCandidateCount": 100_000,
        "repairBackfillAttempted": True,
        "repairBackfillEvaluatedMerges": 40,
        "repairBackfillUsed": True,
        "repairBackfillProbeReason": "no_standard_repair",
        "selectedRepairCandidate": {
            "confidence": 0.631,
            "repairRankingPenalty": 0.08,
            "repairSource": "conflict_backfill",
            "preRepairFaceCounts": {face: 9 for face in "URFDLB"},
            "preRepairConflicts": {"totalConflicts": 2, "validCorners": 8},
        },
    }
    result = RecognitionResult(
        status="success",
        state="U" * 54,
        confidence=0.631,
        reason="Recognized the highest-scoring legal cube state after cubie-level color repair.",
        candidates=100_000,
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_repaired_high_confidence"
    assert category["reason"] == "repair_path_high_confidence_low_penalty"


def test_recognition_category_marks_moderate_repair_as_high_confidence():
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())
    signals = json.loads(json.dumps(payload["recognitionSignals"]))
    selected = signals["selectedRepairCandidate"]
    selected["confidence"] = 0.655
    selected["repairRankingPenalty"] = 0.131
    signals["topRepairCandidates"][0] = selected
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=0.655,
        reason=payload["reason"],
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_repaired_high_confidence"


def test_recognition_category_downgrades_repair_with_paired_pre_count_skew():
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())
    signals = json.loads(json.dumps(payload["recognitionSignals"]))
    selected = signals["selectedRepairCandidate"]
    selected["confidence"] = 0.687
    selected["repairRankingPenalty"] = 0.105
    selected["repairChanges"] = 3
    selected["preRepairFaceCounts"] = {
        "U": 9,
        "R": 7,
        "F": 8,
        "D": 9,
        "L": 10,
        "B": 11,
    }
    signals["topRepairCandidates"][0] = selected
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=0.687,
        reason=payload["reason"],
        candidates=102_858,
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "needs_manual_review"
    assert category["reason"] == "repair_path_pre_repair_color_count_skew"


def test_recognition_category_downgrades_repair_with_unstable_pre_repair_piece_evidence():
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())
    signals = json.loads(json.dumps(payload["recognitionSignals"]))
    selected = signals["selectedRepairCandidate"]
    selected["confidence"] = 0.647
    selected["repairRankingPenalty"] = 0.137
    selected["repairChanges"] = 4
    selected["preRepairConflicts"] = {
        "missingCorners": 0,
        "duplicateColorCorners": 1,
        "missingUdCorners": 1,
        "invalidCorners": 3,
        "missingEdges": 0,
        "duplicateColorEdges": 0,
        "invalidEdges": 0,
        "duplicateCornerCubies": 0,
        "duplicateEdgeCubies": 1,
        "validCorners": 5,
        "validEdges": 12,
        "totalConflicts": 6,
    }
    signals["topRepairCandidates"][0] = selected
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=0.647,
        reason=payload["reason"],
        candidates=134_208,
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "needs_manual_review"
    assert category["reason"] == "repair_path_unstable_pre_repair_piece_evidence"


def test_recognition_category_allows_one_sided_pre_count_skew():
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())
    signals = json.loads(json.dumps(payload["recognitionSignals"]))
    selected = signals["selectedRepairCandidate"]
    selected["confidence"] = 0.637
    selected["repairRankingPenalty"] = 0.148
    selected["preRepairFaceCounts"] = {
        "U": 9,
        "R": 8,
        "F": 11,
        "D": 8,
        "L": 10,
        "B": 8,
    }
    signals["topRepairCandidates"][0] = selected
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=0.637,
        reason=payload["reason"],
        candidates=134_208,
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_repaired_high_confidence"


def test_recognition_category_uses_evaluated_candidate_count_for_repair_retake_gate():
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())
    signals = json.loads(json.dumps(payload["recognitionSignals"]))
    selected = signals["selectedRepairCandidate"]
    selected["confidence"] = 0.655
    selected["repairRankingPenalty"] = 0.131
    # recognitionSignals.repairCandidateCount is the capped public repair
    # details list, not the full evaluated candidate population.
    signals["repairCandidateCount"] = 8
    signals["topRepairCandidates"][0] = selected
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=0.655,
        reason=payload["reason"],
        candidates=134_208,
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_repaired_high_confidence"


def test_recognition_category_downgrades_high_penalty_repair_to_manual_review():
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())
    signals = json.loads(json.dumps(payload["recognitionSignals"]))
    selected = signals["selectedRepairCandidate"]
    selected["repairRankingPenalty"] = 0.18
    signals["repairCandidateCount"] = 100_000
    signals["topRepairCandidates"][0] = selected
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=0.61,
        reason=payload["reason"],
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "needs_manual_review"
    assert category["reason"] == "repair_path_low_confidence_or_high_conflict"


def test_recognition_category_marks_floor_confidence_repair_as_retake():
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())
    signals = json.loads(json.dumps(payload["recognitionSignals"]))
    signals["repairCandidateCount"] = 12_101
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=0.50,
        reason=payload["reason"],
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "reject_retake"
    assert category["reason"] == "repair_path_floor_confidence_or_too_few_candidates"


def test_repair_ranking_penalty_prefers_cleaner_pre_repair_pieces():
    clean = {key: 0 for key in PIECE_CONFLICT_KEYS}
    clean["validCorners"] = 8
    clean["validEdges"] = 12
    clean["totalConflicts"] = 0
    conflicted = dict(clean)
    conflicted.update(
        {
            "invalidCorners": 2,
            "invalidEdges": 1,
            "duplicateCornerCubies": 1,
            "validCorners": 6,
            "validEdges": 11,
            "totalConflicts": 4,
        }
    )
    faces = {"_orientation_rank_a": 0, "_orientation_rank_b": 0}

    assert _repair_ranking_penalty(conflicted, faces, repair_cost=10.0, repair_changes=2) > _repair_ranking_penalty(
        clean,
        faces,
        repair_cost=10.0,
        repair_changes=2,
    )


def test_repair_ranking_penalty_is_continuous_not_a_hard_reject():
    conflicts = {key: 3 for key in PIECE_CONFLICT_KEYS}
    conflicts["validCorners"] = 0
    conflicts["validEdges"] = 0
    faces = {"_orientation_rank_a": 10, "_orientation_rank_b": 10}

    penalty = _repair_ranking_penalty(conflicts, faces, repair_cost=95.0, repair_changes=12)

    assert 0 < penalty <= 0.18
