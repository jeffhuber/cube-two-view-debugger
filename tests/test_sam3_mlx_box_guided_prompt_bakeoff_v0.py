from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "tests" / "fixtures" / "sam3_mlx_box_guided_prompt_bakeoff_v0_easy_summary.json"


def test_box_guided_sam3_prompts_remain_negative_vertex_signal():
    document = json.loads(SUMMARY.read_text(encoding="utf-8"))

    for policy in ("box_visual", "box_face_text"):
        summary = document["policies"][policy]["bakeoff"]["summary"]["providerSummaries"]["sam3"]
        assert summary["rowCount"] == 16
        assert summary["rowsWithAnyMask"] == 16
        assert summary["rowsWithThreeFaceMasks"] == 16
        assert summary["rowsWithCandidates"] == 16
        assert summary["top3HitCount@10px"] == 0
        assert summary["oracleHitCount@20px"] == 0


def test_box_guided_fixture_has_expected_policy_shape():
    document = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert set(document["policies"]) == {"box_visual", "box_face_text"}

    for policy in document["policies"].values():
        assert policy["exportSummary"]["exported"] == 48
        assert policy["exportSummary"]["noMask"] == 0
        assert len(policy["bakeoff"]["rows"]) == 16
