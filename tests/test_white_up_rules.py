import json
from pathlib import Path

from rubik_recognizer.recognizer import (
    PIECE_CONFLICT_KEYS,
    RecognitionResult,
    _prefer_calibrated_result,
    _recognition_category_payload,
    _repair_ranking_penalty,
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


def test_recognition_category_marks_low_penalty_repair_as_high_confidence():
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=payload["confidence"],
        reason=payload["reason"],
        candidates=payload["candidates"],
        recognition_signals=payload["recognitionSignals"],
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_repaired_high_confidence"


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
        candidates=payload["candidates"],
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
    signals["topRepairCandidates"][0] = selected
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=0.61,
        reason=payload["reason"],
        candidates=payload["candidates"],
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "needs_manual_review"
    assert category["reason"] == "repair_path_low_confidence_or_high_conflict"


def test_recognition_category_marks_floor_confidence_repair_as_retake():
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())
    signals = json.loads(json.dumps(payload["recognitionSignals"]))
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        confidence=0.50,
        reason=payload["reason"],
        # Match Set 31's heavily-pruned candidate pool; the gate fires
        # via the confidence floor as well, but pinning candidates
        # documents the second clause too.
        candidates=12_101,
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "reject_retake"
    assert category["reason"] == "repair_path_floor_confidence_or_too_few_candidates"


def test_recognition_category_marks_low_total_candidate_count_as_retake():
    """Set 31 fingerprint: heavily-pruned upstream candidate pool with
    confidence above the floor. The retake gate should fire on candidate
    count alone, even when confidence is healthy enough to clear 0.50."""
    fixture_dir = Path(__file__).parent / "fixtures"
    payload = json.loads((fixture_dir / "recognition_signals_repair.json").read_text())
    signals = json.loads(json.dumps(payload["recognitionSignals"]))
    result = RecognitionResult(
        status=payload["status"],
        state=payload["state"],
        # Above the 0.50 floor — only the candidate-count clause fires.
        confidence=0.55,
        reason=payload["reason"],
        candidates=12_101,
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "reject_retake"
    assert category["reason"] == "repair_path_floor_confidence_or_too_few_candidates"


def test_recognition_category_ignores_artifact_grids_in_weak_count():
    """A clean direct-legal recognition (Set 32 fingerprint) should land
    in success_clean even when selectedGridQuality reports weak fitError
    or low quality on grid slots that AREN'T part of the visible-face
    triple. Image A exposes U/R/F; the recognizer also records candidate
    grids for B/L slots as artifacts (back faces glimpsed through the
    silhouette, glints, etc.) — those should not count toward the
    weak-grid gate."""
    signals = {
        "schemaVersion": 1,
        "repairPathUsed": False,
        "repairCandidateCount": 0,
        "selectedGridQuality": {
            "imageA": {
                "U": {"gridId": 1, "centerFace": "U", "matchedCount": 9, "fitError": 0.5,
                      "quality": 150.0, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
                "R": {"gridId": 2, "centerFace": "R", "matchedCount": 9, "fitError": 0.5,
                      "quality": 150.0, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
                "F": {"gridId": 3, "centerFace": "F", "matchedCount": 9, "fitError": 0.5,
                      "quality": 150.0, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
                # Artifact grid in non-visible slot — should NOT count as weak.
                "L": {"gridId": 4, "centerFace": "L", "matchedCount": 9, "fitError": 16.4,
                      "quality": 96.5, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
            },
            "imageB": {
                "D": {"gridId": 5, "centerFace": "D", "matchedCount": 9, "fitError": 0.5,
                      "quality": 150.0, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
                "L": {"gridId": 6, "centerFace": "L", "matchedCount": 9, "fitError": 0.5,
                      "quality": 150.0, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
                "B": {"gridId": 7, "centerFace": "B", "matchedCount": 9, "fitError": 0.5,
                      "quality": 150.0, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
                # Another artifact in a non-visible slot.
                "F": {"gridId": 8, "centerFace": "F", "matchedCount": 6, "fitError": 0.3,
                      "quality": 10.9, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
            },
        },
    }
    result = RecognitionResult(
        status="success",
        state="U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9,
        confidence=0.838,
        reason="Recognized a unique legal white-up cube state.",
        candidates=30_784,
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "success_clean"
    assert category["reason"] == "direct_unique_legal_high_confidence"


def test_recognition_category_demotes_when_visible_face_grid_is_weak():
    """Counterpart to the artifact-grid test: if the WEAK grid is in a
    visible face slot (i.e. genuinely part of the recognition), the
    success_clean gate should still fire its weak-grid demotion."""
    signals = {
        "schemaVersion": 1,
        "repairPathUsed": False,
        "repairCandidateCount": 0,
        "selectedGridQuality": {
            "imageA": {
                "U": {"gridId": 1, "centerFace": "U", "matchedCount": 9, "fitError": 0.5,
                      "quality": 150.0, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
                "R": {"gridId": 2, "centerFace": "R", "matchedCount": 9, "fitError": 0.5,
                      "quality": 150.0, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
                # Weak grid IS in a visible slot — should count.
                "F": {"gridId": 3, "centerFace": "F", "matchedCount": 9, "fitError": 13.5,
                      "quality": 80.0, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
            },
            "imageB": {
                "D": {"gridId": 5, "centerFace": "D", "matchedCount": 9, "fitError": 0.5,
                      "quality": 150.0, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
                "L": {"gridId": 6, "centerFace": "L", "matchedCount": 9, "fitError": 0.5,
                      "quality": 150.0, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
                "B": {"gridId": 7, "centerFace": "B", "matchedCount": 9, "fitError": 0.5,
                      "quality": 150.0, "gridSamples": 9, "badSamples": 0, "suspectSamples": 0.0},
            },
        },
    }
    result = RecognitionResult(
        status="success",
        state="U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9,
        confidence=0.838,
        reason="Recognized a unique legal white-up cube state.",
        candidates=30_784,
        recognition_signals=signals,
    )

    category = _recognition_category_payload(result)

    assert category["category"] == "needs_manual_review"
    assert category["reason"] == "direct_legal_but_low_margin_or_grid_quality"


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
