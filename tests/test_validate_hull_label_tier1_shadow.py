from __future__ import annotations


def test_face_quads_by_label_uses_capture_side_labels():
    from tools.validate_hull_label_tier1_shadow import _face_quads_by_label

    class Model:
        face_quads = {
            "face_xz": [(0, 0), (1, 0), (1, 1), (0, 1)],
            "face_yz": [(2, 0), (3, 0), (3, 1), (2, 1)],
            "face_xy": [(4, 0), (5, 0), (5, 1), (4, 1)],
        }

    assert set(_face_quads_by_label(Model(), "A")) == {"U", "R", "F"}
    assert set(_face_quads_by_label(Model(), "B")) == {"D", "B", "L"}
    assert _face_quads_by_label(Model(), "A")["R"] == [
        (2.0, 0.0),
        (3.0, 0.0),
        (3.0, 1.0),
        (2.0, 1.0),
    ]
    assert _face_quads_by_label(Model(), "B")["L"] == [
        (4.0, 0.0),
        (5.0, 0.0),
        (5.0, 1.0),
        (4.0, 1.0),
    ]


def test_build_summary_tracks_prefer_selection_and_delta():
    from tools.validate_hull_label_tier1_shadow import build_summary

    rows = [
        {
            "setId": "1",
            "modes": {
                "off": {
                    "assembledState": "x" * 54,
                    "exactMatch": False,
                    "validState": False,
                    "hamming": 5,
                    "stickersSampled": 54,
                    "stickersCorrect": 49,
                    "perStickerMatchesAssembled": 49,
                },
                "shadow": {
                    "assembledState": "x" * 54,
                    "exactMatch": False,
                    "validState": False,
                    "hamming": 5,
                    "stickersSampled": 54,
                    "stickersCorrect": 49,
                    "perStickerMatchesAssembled": 49,
                    "sideTraces": {
                        "A": {"status": "accepted", "accepted": True, "selected": False},
                        "B": {"status": "rejected", "hard_failures": ["bad"], "selected": False},
                    },
                },
                "prefer": {
                    "assembledState": "x" * 54,
                    "exactMatch": False,
                    "validState": True,
                    "hamming": 2,
                    "stickersSampled": 54,
                    "stickersCorrect": 52,
                    "perStickerMatchesAssembled": 52,
                    "sideTraces": {
                        "A": {"status": "accepted", "accepted": True, "selected": True},
                        "B": {"status": "rejected", "hard_failures": ["bad"], "selected": False},
                    },
                },
            },
        }
    ]

    summary = build_summary(rows)

    assert summary["byMode"]["prefer"]["legal"] == 1
    assert summary["byMode"]["prefer"]["assembledStickerAccuracy"] == round(52 / 54, 6)
    assert summary["shadowTrace"]["acceptedSides"] == 1
    assert summary["preferTrace"]["selectedSides"] == 1
    assert summary["preferTrace"]["hardFailureCounts"] == {"bad": 1}
    assert summary["preferVsOff"]["improved"] == [
        {
            "setId": "1",
            "offHamming": 5,
            "preferHamming": 2,
            "offExact": False,
            "preferExact": False,
        }
    ]
