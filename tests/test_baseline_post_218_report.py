from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import baseline_post_218  # noqa: E402


def test_render_markdown_preserves_legacy_convention_warning():
    summary = {
        "n_runs": 0,
        "n_cases": 0,
        "category_counts": {},
        "error_distribution": {},
        "stable_good_cases": 0,
        "stable_bad_cases": 0,
        "mixed_cases": 0,
    }

    report = baseline_post_218._render_markdown(summary, {})

    assert "legacy decision spine" in report
    assert "2026-05-23 convention caution" in report
    # Wording updated when axis_x/y/z became the canonical schema name
    # (legacy near_x/y/z still accepted by readers); the FAR/double-axis
    # finding the warning preserves is what matters, not the verbatim
    # wording.
    assert "FAR / double-axis" in report
    assert "`A -> 0,2,4`" in report
    assert "`B -> 1,3,5`" in report
    assert "canonical until the baseline is regenerated from full-corner truth" in report
    assert "The `near_*` target semantics are provisional" in report
