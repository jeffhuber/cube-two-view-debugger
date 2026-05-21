from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "tests" / "fixtures" / "sam3_mlx_current_prompt_bakeoff_v0_easy_summary.json"


def test_current_mlx_sam3_prompts_are_negative_vertex_signal():
    document = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary = document["summary"]["providerSummaries"]["sam3"]

    assert summary["rowCount"] == 16
    assert summary["rowsWithAnyMask"] == 16
    assert summary["rowsWithThreeFaceMasks"] == 16
    assert summary["rowsWithCandidates"] == 16
    assert summary["top3HitCount@10px"] == 0
    assert summary["oracleHitCount@20px"] == 0


def test_current_mlx_sam3_fixture_has_expected_prompts_per_row():
    document = json.loads(SUMMARY.read_text(encoding="utf-8"))
    expected_prompts = {
        "whole_cube",
        "top_face",
        "left_face",
        "right_face",
        "stickers",
    }

    for row in document["rows"]:
        sam3 = row["providers"]["sam3"]
        assert set(sam3["maskPrompts"]) == expected_prompts
        assert sam3["candidateCount"] == 20
