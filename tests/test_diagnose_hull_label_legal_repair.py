from __future__ import annotations

from tools.diagnose_hull_label_legal_repair import build_summary, render_report


def test_build_summary_counts_legal_repair_methods():
    rows = [{
        "setId": "99",
        "methods": {
            "canonical_count_repaired": {
                "hamming": 2,
                "exactMatch": False,
                "validState": False,
            },
            "conservative_legal_repaired": {
                "hamming": 0,
                "exactMatch": True,
                "validState": True,
                "repairCost": 0.75,
                "repairChanges": 1,
            },
            "guarded_broad_legal_repaired": {
                "hamming": 0,
                "exactMatch": True,
                "validState": True,
                "repairCost": 2.0,
                "repairChanges": 2,
            },
            "broad_legal_repaired": {
                "hamming": 0,
                "exactMatch": True,
                "validState": True,
                "repairCost": 3.25,
                "repairChanges": 4,
            },
        },
    }]

    summary = build_summary(rows)

    conservative = summary["methods"]["conservative_legal_repaired"]
    assert conservative["assembled"] == 1
    assert conservative["legal"] == 1
    assert conservative["exact"] == 1
    assert conservative["medianRepairCost"] == 0.75
    assert conservative["medianRepairChanges"] == 1
    assert summary["methods"]["guarded_broad_legal_repaired"]["exact"] == 1


def test_render_report_marks_broad_repair_as_diagnostic_only():
    rows = [{
        "setId": "99",
        "methods": {
            "canonical_count_repaired": {
                "hamming": 2,
                "exactMatch": False,
                "validState": False,
            },
            "conservative_legal_repaired": {
                "status": "no_legal_repair",
                "hamming": None,
                "exactMatch": False,
                "validState": False,
                "repairCost": None,
                "repairChanges": None,
            },
            "guarded_broad_legal_repaired": {
                "status": "rejected_guarded_broad_legal_repair",
                "hamming": None,
                "exactMatch": False,
                "validState": False,
                "repairCost": None,
                "repairChanges": None,
                "rejectedRepairCost": 22.0,
                "rejectedRepairChanges": 8,
                "rejectedStateDeltaFromCanonical": {
                    "available": True,
                    "count": 5,
                    "indices": [0, 1, 2, 3, 4],
                    "facePositions": ["U[0,0]", "U[0,1]", "U[0,2]", "U[1,0]", "U[1,1]"],
                },
            },
            "broad_legal_repaired": {
                "status": "legal_repair_found",
                "hamming": 0,
                "exactMatch": True,
                "validState": True,
                "repairCost": 2.5,
                "repairChanges": 3,
                "stateDeltaFromCanonical": {
                    "available": True,
                    "count": 5,
                    "indices": [0, 1, 2, 3, 4],
                    "facePositions": ["U[0,0]", "U[0,1]", "U[0,2]", "U[1,0]", "U[1,1]"],
                },
            },
        },
    }]
    payload = {
        "source": {"git_sha": "abc123", "generated_at_utc": "2026-05-26T00:00:00+00:00"},
        "summary": build_summary(rows),
        "rows": rows,
    }

    report = render_report(payload)

    assert "Broad cubie repair is diagnostic-only" in report
    assert "Guarded broad repair applies a no-ground-truth gate" in report
    assert "state delta from `canonical_count_repaired` <= 4" in report
    assert "| 99 | 2 | None | `no_legal_repair` | None | `rejected_guarded_broad_legal_repair` | 0 | 2.5 | 3 | 5 |" in report
