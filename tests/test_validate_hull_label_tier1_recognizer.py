from __future__ import annotations


def _mode(
    *,
    hamming: int | None,
    status: str = "success",
    exact: bool = False,
    selected: bool = False,
    failed_checks=None,
    tier1=None,
    yaw=None,
):
    return {
        "status": status,
        "category": "needs_manual_review" if status == "success" else "reject_retake",
        "categoryReason": "test",
        "reason": "test",
        "failedChecks": list(failed_checks or []),
        "confidence": 0.8,
        "candidateCount": 1,
        "hamming": hamming,
        "stickersCorrect": (54 - hamming) if hamming is not None else 0,
        "exactMatch": exact,
        "validState": status == "success",
        "validationErrors": [],
        "hullLabelTier1": tier1,
        "hullLabelTier1Yaw": yaw,
        "hullLabelSelected": selected,
        "hullLabelTier1Prefer": None,
    }


def test_build_summary_tracks_effective_prefer_selection_and_delta():
    from tools.validate_hull_label_tier1_recognizer import build_summary

    tier1 = {
        "images": {
            "imageA": {"status": "accepted", "accepted": True, "selected": True},
            "imageB": {"status": "accepted", "accepted": True, "selected": True},
        }
    }
    yaw = {"status": "accepted", "bestYawQuarterTurns": 0}
    prefer = _mode(hamming=0, exact=True, selected=True, tier1=tier1, yaw=yaw)
    prefer["hullLabelTier1Prefer"] = {
        "selected": True,
        "fallbackToLegacy": False,
        "candidateCategory": "needs_manual_review",
        "candidateFailedChecks": [],
    }
    rows = [
        {
            "setId": "1",
            "modes": {
                "off": _mode(hamming=5),
                "shadow": _mode(hamming=5, tier1=tier1, yaw=yaw),
                "prefer_candidate": _mode(hamming=0, exact=True, selected=True, tier1=tier1, yaw=yaw),
                "prefer_effective": prefer,
            },
        }
    ]

    summary = build_summary(rows)

    assert summary["byMode"]["prefer_effective"]["exact"] == 1
    assert summary["preferVsOff"]["improved"] == [
        {
            "setId": "1",
            "offHamming": 5,
            "preferHamming": 0,
            "offExact": False,
            "preferExact": True,
            "selected": True,
        }
    ]
    assert summary["preferFallback"]["selectedSetIds"] == ["1"]
    assert summary["preferCandidateTrace"]["acceptedSides"] == 2
    assert summary["preferCandidateTrace"]["selectedSides"] == 2
    assert summary["preferCandidateTrace"]["yawStatusCounts"] == {"accepted": 1}


def test_capture_guidance_buckets_candidate_checks_and_trace_warnings():
    from tools.validate_hull_label_tier1_recognizer import build_summary

    tier1 = {
        "images": {
            "imageA": {
                "status": "rejected",
                "accepted": False,
                "selected": False,
                "hard_failures": ["projective_residual_norm=0.0300; max 0.0250"],
                "warnings": ["vertex_cloud_spread_px=260.0; warning 240.0"],
            },
            "imageB": {
                "status": "accepted",
                "accepted": True,
                "selected": True,
                "hard_failures": [],
                "warnings": ["sticker_score_total=720.0; warning 700.0"],
            },
        }
    }
    prefer = _mode(
        hamming=10,
        status="success",
        tier1=tier1,
        failed_checks=["missing_side_face_coverage", "image_a_U_anchor_missing"],
    )
    prefer["hullLabelTier1Prefer"] = {
        "selected": False,
        "fallbackToLegacy": True,
        "candidateCategory": "reject_retake",
        "candidateFailedChecks": ["missing_side_face_coverage", "image_a_U_anchor_missing"],
    }
    rows = [
        {
            "setId": "1",
            "modes": {
                "off": _mode(hamming=10),
                "shadow": _mode(hamming=10, tier1=tier1),
                "prefer_candidate": _mode(
                    hamming=None,
                    status="rejected",
                    tier1=tier1,
                    failed_checks=["missing_side_face_coverage", "image_a_U_anchor_missing"],
                ),
                "prefer_effective": prefer,
            },
        }
    ]

    guidance = build_summary(rows)["preferFallback"]["captureGuidanceCounts"]

    assert guidance["side-center coverage collapsed; inspect yaw/slot-to-WCA assignment"] == 1
    assert guidance["U/D anchor not reliable; ensure A is white-up and B is yellow-up"] == 1
    assert guidance["hull/projective residual high; avoid background edges and steep tilt"] == 1
    assert guidance["vertex estimates disagree; reduce perspective tilt"] == 1
    assert guidance["rectified sticker score high; improve lighting/focus/glare"] == 1
