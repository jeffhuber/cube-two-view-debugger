"""Unit tests for `tools/diagnose_center_color_phase_gate.py`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import CANONICAL_RGB  # noqa: E402
from tools import diagnose_center_color_phase_gate as d  # noqa: E402
from tools.global_cube_model import GlobalCubeModel  # noqa: E402


def _synthetic_image_and_model(
    slot_colors: dict[str, str],
) -> tuple[Image.Image, GlobalCubeModel]:
    image = Image.new("RGB", (420, 160), (40, 40, 40))
    draw = ImageDraw.Draw(image)
    quads = {
        "upper": [(20, 20), (120, 20), (120, 120), (20, 120)],
        "right": [(160, 20), (260, 20), (260, 120), (160, 120)],
        "front": [(300, 20), (400, 20), (400, 120), (300, 120)],
    }
    for slot, quad in quads.items():
        color_name = slot_colors[slot]
        draw.polygon(quad, fill=CANONICAL_RGB[color_name])

    model = GlobalCubeModel()
    model.face_quads = {
        d.MODEL_FACE_BY_SLOT[slot]: tuple(quad)
        for slot, quad in quads.items()
    }
    return image, model


def test_score_model_center_colors_identity_wins_for_a_yaw0():
    image, model = _synthetic_image_and_model({
        "upper": "white",
        "right": "red",
        "front": "green",
    })
    score = d.score_model_center_colors(image, model, "A", 0)
    assert score["identity_assignment"] == ["U", "R", "F"]
    assert score["winning_hypothesis"] == "identity"
    assert score["identity_score"] < 1.0
    assert score["margin"] > 100.0


def test_score_model_center_colors_detects_cyclic_relabeling():
    image, model = _synthetic_image_and_model({
        "upper": "red",
        "right": "green",
        "front": "white",
    })
    score = d.score_model_center_colors(image, model, "A", 0)
    assert score["identity_assignment"] == ["U", "R", "F"]
    assert score["winning_hypothesis"] == "cyclic_120"
    assert score["hypothesis_assignments"]["cyclic_120"] == ["R", "F", "U"]
    assert score["hypothesis_scores"]["cyclic_120"] < 1.0


def test_choose_by_center_identity_score_picks_lower_score():
    unflipped = {"identity_score": 15.0}
    forced_flip = {"identity_score": 40.0}
    assert (
        d.choose_by_center_identity_score(unflipped, forced_flip)["choice"]
        == "unflipped"
    )
    assert (
        d.choose_by_center_identity_score(forced_flip, unflipped)["choice"]
        == "forced_flip"
    )


def test_choose_by_center_identity_score_ties_inside_margin():
    result = d.choose_by_center_identity_score(
        {"identity_score": 10.0},
        {"identity_score": 10.5},
        min_margin=1.0,
    )
    assert result["choice"] == "tie"


def test_force_phase_flip_model_uses_far_corners_as_new_axes():
    from tools.global_cube_model import derive_geometry

    original = GlobalCubeModel(
        cube_center_screen=(100.0, 100.0),
        axis_x_2d=(10.0, 0.0),
        axis_y_2d=(0.0, 20.0),
        axis_z_2d=(-5.0, -15.0),
    )
    derive_geometry(original)
    flipped = d.force_phase_flip_model(original)
    assert flipped.axis_x_2d == (10.0, 20.0)  # h_xy - vertex
    assert flipped.axis_y_2d == (5.0, -15.0)  # h_xz - vertex
    assert flipped.axis_z_2d == (-5.0, 5.0)  # h_yz - vertex
    assert flipped.debug["phase_check"] == "forced_60deg_flip_diagnostic"


def test_force_phase_flip_model_does_not_inherit_unflipped_fit_quality():
    """Greptile P2 on PR #265: the forced-flip model is analytically
    constructed (no optimization loop). Inheriting the unflipped
    model's `fit_loss` / `fit_quality` would mislead any future
    best-of-N picker into treating the synthetic hypothesis as
    equivalently well-fit. Default sentinel values (fit_loss=inf,
    fit_quality=0) are the honest signal that no fit ran."""
    from tools.global_cube_model import derive_geometry

    original = GlobalCubeModel(
        cube_center_screen=(100.0, 100.0),
        axis_x_2d=(10.0, 0.0),
        axis_y_2d=(0.0, 20.0),
        axis_z_2d=(-5.0, -15.0),
        fit_loss=0.5,
        fit_quality=0.95,
    )
    derive_geometry(original)
    flipped = d.force_phase_flip_model(original)
    # The forced-flip model MUST NOT inherit the unflipped's fit
    # quality — those numbers came from a Procrustes optimization
    # that didn't run for this synthetic hypothesis. Pinning to the
    # GlobalCubeModel defaults (inf / 0.0) so the "no fit ran"
    # signal is unambiguous.
    assert flipped.fit_loss == float("inf"), (
        f"forced_flip.fit_loss must be the inf sentinel, not the "
        f"unflipped value 0.5; got {flipped.fit_loss}"
    )
    assert flipped.fit_quality == 0.0, (
        f"forced_flip.fit_quality must be 0.0 sentinel, not the "
        f"unflipped value 0.95; got {flipped.fit_quality}"
    )


def test_force_phase_flip_model_does_not_inherit_unflipped_debug_signals():
    """Greptile P2 on PR #265: copying the unflipped model's full
    `debug` into the forced-flip model carried stale detector signals
    like `phase_darkness_separation` (which was computed for the
    UNFLIPPED model, not for the synthetic forced-flip hypothesis).
    Trace consumers must see absent/None for those signals on the
    forced-flip side — that's the honest "no detector ran" signal."""
    from tools.global_cube_model import derive_geometry

    original = GlobalCubeModel(
        cube_center_screen=(100.0, 100.0),
        axis_x_2d=(10.0, 0.0),
        axis_y_2d=(0.0, 20.0),
        axis_z_2d=(-5.0, -15.0),
    )
    derive_geometry(original)
    # Stuff the unflipped model's debug with detector signals that
    # would NOT be valid for the forced-flip hypothesis if blindly
    # copied over.
    original.debug["phase_check"] = "corrected_60deg_flip"
    original.debug["phase_darkness_separation"] = 42.5
    original.debug["phase_mean_near_darkness"] = 100.0
    flipped = d.force_phase_flip_model(original)
    # phase_check on the forced-flip MUST identify it as synthetic,
    # not carry over the unflipped's detector verdict.
    assert flipped.debug["phase_check"] == "forced_60deg_flip_diagnostic"
    # Stale detector-derived numerics must NOT be present.
    assert "phase_darkness_separation" not in flipped.debug, (
        "forced_flip must not carry over the unflipped model's "
        "phase_darkness_separation — that signal applies to a "
        "different hypothesis"
    )
    assert "phase_mean_near_darkness" not in flipped.debug


def test_render_report_handles_not_traced_rows():
    payload = {
        "schema": "center_color_phase_gate_trace_v1",
        "summary": {
            "n_total_rows": 1,
            "n_traced_rows": 0,
            "n_fully_stable_rows": 0,
            "center_choice_modal_counts": {},
            "effect_vs_production_modal_counts": {},
            "production_geometry_category_modal_counts": {},
            "selected_geometry_category_modal_counts": {},
        },
        "per_row": [{
            "key": "20_A",
            "summary": {"status": "all_runs_untraced"},
        }],
    }
    report = d.render_report(payload)
    assert "# Center-color phase gate diagnostic" in report
    assert "`20_A`" in report
    assert "all_runs_untraced" in report


def test_main_writes_outputs_with_monkeypatched_runner(tmp_path, monkeypatch):
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps({"20_A": {"approved": True}}))
    out_json = tmp_path / "trace.json"
    out_md = tmp_path / "report.md"

    def fake_run_diagnostic(*args, **kwargs):
        return {
            "schema": "center_color_phase_gate_trace_v1",
            "summary": {
                "n_total_rows": 0,
                "n_traced_rows": 0,
                "n_fully_stable_rows": 0,
                "center_choice_modal_counts": {},
                "effect_vs_production_modal_counts": {},
                "production_geometry_category_modal_counts": {},
                "selected_geometry_category_modal_counts": {},
            },
            "per_row": [],
        }

    monkeypatch.setattr(d, "run_diagnostic", fake_run_diagnostic)
    rc = d.main([
        "--truth", str(truth_path),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
        "--n-runs", "1",
    ])
    assert rc == 0
    assert json.loads(out_json.read_text())["schema"] == (
        "center_color_phase_gate_trace_v1"
    )
    assert "Center-color phase gate diagnostic" in out_md.read_text()
