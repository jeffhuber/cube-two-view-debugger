import json
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "constrained_recognize_mode_validation.json"


def test_constrained_recognize_validation_summary_matches_rows():
    payload = json.loads(FIXTURE.read_text())
    rows = payload["rows"]
    summary = payload["summary"]

    assert payload["schema"] == "constrained_recognize_mode_validation_v1"
    assert summary["pairCount"] == len(rows)
    assert summary["byMode"]["legacy"]["pairs"] == len(rows)
    assert summary["byMode"]["constrained"]["pairs"] == len(rows)

    constrained_exact = sum(1 for row in rows if row["modes"]["constrained"]["exactMatch"])
    constrained_legal = sum(1 for row in rows if row["modes"]["constrained"]["validState"])
    legacy_exact = sum(1 for row in rows if row["modes"]["legacy"]["exactMatch"])

    assert summary["byMode"]["constrained"]["exact"] == constrained_exact
    assert summary["byMode"]["constrained"]["legal"] == constrained_legal
    assert summary["byMode"]["legacy"]["exact"] == legacy_exact


def test_constrained_recognize_validation_quality_floor():
    payload = json.loads(FIXTURE.read_text())
    summary = payload["summary"]
    pair_count = summary["pairCount"]

    constrained = summary["byMode"]["constrained"]
    legacy = summary["byMode"]["legacy"]
    delta = summary["constrainedVsLegacy"]
    signals = summary["constrainedSignals"]

    assert constrained["exact"] >= legacy["exact"]
    assert constrained["legal"] >= legacy["legal"]
    assert constrained["exact"] >= int(pair_count * 0.95)
    assert delta["regressed"] == []
    assert len(signals["selectedSetIds"]) == constrained["success"]
    assert len(signals["gateAcceptedSetIds"]) == constrained["success"]


def test_constrained_recognize_validation_covers_recent_gan_yaw_rows():
    payload = json.loads(FIXTURE.read_text())
    rows_by_set = {row["setId"]: row for row in payload["rows"]}

    for set_id in ("74", "75", "76", "77", "78"):
        row = rows_by_set[set_id]
        assert row["modes"]["constrained"]["exactMatch"] is True
        assert row["modes"]["constrained"]["constrainedSignal"]["selected"] is True
