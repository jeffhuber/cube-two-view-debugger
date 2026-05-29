from __future__ import annotations

from tools.score_deployed_recognizer import _constrained_performance, _multipart_body, _summary


def test_constrained_performance_flattens_stage_timings():
    perf = _constrained_performance({
        "performance": {
            "schema": "constrained_recognize_performance_v1",
            "rectifiedInputPerformanceSchema": "llm_rectified_input_performance_v1",
            "contactSheetsIncluded": False,
            "hullFitMode": "threaded",
            "stageTimingsMs": {
                "recognizeTotal": 2600.55,
                "prepareTotal": 2600.36,
                "imports": 0.04,
                "rembgSession": 0.01,
            },
        }
    })

    assert perf["performanceSchema"] == "constrained_recognize_performance_v1"
    assert perf["rectifiedInputPerformanceSchema"] == "llm_rectified_input_performance_v1"
    assert perf["contactSheetsIncluded"] is False
    assert perf["hullFitMode"] == "threaded"
    assert perf["recognizeTotalMs"] == 2600.55
    assert perf["prepareTotalMs"] == 2600.36
    assert perf["importsMs"] == 0.04
    assert perf["rembgSessionMs"] == 0.01


def test_summary_includes_performance_distributions():
    summary = _summary([
        {
            "status": "success",
            "exactMatch": True,
            "hamming": 0,
            "recommendedMethod": "canonical_count_repaired",
            "performanceSchema": "constrained_recognize_performance_v1",
            "contactSheetsIncluded": False,
            "hullFitMode": "threaded",
            "clientSource": "codex-deployed-score",
            "clientTimingSchema": "deployed_score_client_timing_v1",
            "latencyMs": 3100,
            "clientWallOverheadMs": 500.0,
            "clientHttpToHeadersMs": 3050.0,
            "clientResponseReadMs": 2.0,
            "recognizeTotalMs": 2600.0,
            "importsMs": 0.04,
            "rembgSessionMs": 0.01,
        },
        {
            "status": "success",
            "exactMatch": True,
            "hamming": 0,
            "recommendedMethod": "canonical_count_repaired",
            "performanceSchema": "constrained_recognize_performance_v1",
            "contactSheetsIncluded": False,
            "hullFitMode": "threaded",
            "clientSource": "codex-deployed-score",
            "clientTimingSchema": "deployed_score_client_timing_v1",
            "latencyMs": 39000,
            "clientWallOverheadMs": 400.0,
            "clientHttpToHeadersMs": 38900.0,
            "clientResponseReadMs": 2.5,
            "recognizeTotalMs": 38600.0,
            "importsMs": 32300.0,
            "rembgSessionMs": 3270.0,
        },
    ])

    assert summary["exactCount"] == 2
    assert summary["performanceSchemaCounts"] == {"constrained_recognize_performance_v1": 2}
    assert summary["contactSheetsIncludedCounts"] == {"False": 2}
    assert summary["hullFitModeCounts"] == {"threaded": 2}
    assert summary["clientSourceCounts"] == {"codex-deployed-score": 2}
    assert summary["clientTimingSchemaCounts"] == {"deployed_score_client_timing_v1": 2}
    assert summary["timings"]["latencyMs"]["max"] == 39000.0
    assert summary["timings"]["clientWallOverheadMs"]["max"] == 500.0
    assert summary["timings"]["clientHttpToHeadersMs"]["max"] == 38900.0
    assert summary["timings"]["importsMs"]["max"] == 32300.0


def test_multipart_body_allows_text_fields_without_filename():
    body, boundary = _multipart_body([
        ("imageA", "a.jpg", b"abc", "image/jpeg"),
        ("clientSource", None, b"codex-deployed-score", "text/plain"),
    ])

    assert boundary.encode("ascii") in body
    assert b'name="imageA"; filename="a.jpg"' in body
    assert b'name="clientSource"; filename=' not in body
    assert b'name="clientSource"' in body
