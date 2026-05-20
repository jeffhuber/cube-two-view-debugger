from tools.probe_bezel_discontinuity_join import (
    cross_tab_axis,
    discontinuity_cell_flag,
    main,
    subdivide_face_quad,
    threshold_sweep,
)


def test_subdivide_face_quad_returns_row_major_cell_quads():
    face = [(0, 0), (90, 0), (90, 90), (0, 90)]

    cells = subdivide_face_quad(face)

    assert len(cells) == 9
    assert cells[0]["row"] == 0
    assert cells[0]["col"] == 0
    assert cells[0]["quad"] == [
        (0.0, 0.0),
        (30.0, 0.0),
        (30.0, 30.0),
        (0.0, 30.0),
    ]
    assert cells[-1]["row"] == 2
    assert cells[-1]["col"] == 2
    assert cells[-1]["quad"] == [
        (60.0, 60.0),
        (90.0, 60.0),
        (90.0, 90.0),
        (60.0, 90.0),
    ]


def test_discontinuity_cell_flag_uses_existing_probe_thresholds():
    assert discontinuity_cell_flag({"internalStd": 34.9, "maxHalfDelta": 44.9}) is False
    assert discontinuity_cell_flag({"internalStd": 35.0, "maxHalfDelta": 1.0}) is True
    assert discontinuity_cell_flag({"internalStd": 1.0, "maxHalfDelta": 45.0}) is True


def test_cross_tab_axis_names_candidate_join_states():
    assert cross_tab_axis(True, True) == "both_hit"
    assert cross_tab_axis(True, False) == "bezel_only"
    assert cross_tab_axis(False, True) == "discontinuity_only"
    assert cross_tab_axis(False, False) == "both_miss"


def test_threshold_sweep_recomputes_bezel_flags_from_raw_per_line():
    rows = [
        {
            "humanBad": True,
            "discontinuity": {"flag": True},
            "bezel": {
                "per_line": [
                    {
                        "crosses_cell": True,
                        "quality": 0.5,
                        "distance_from_centroid_px": 25.0,
                    }
                ]
            },
        },
        {
            "humanBad": False,
            "discontinuity": {"flag": True},
            "bezel": {
                "per_line": [
                    {
                        "crosses_cell": True,
                        "quality": 0.35,
                        "distance_from_centroid_px": 25.0,
                    }
                ]
            },
        },
    ]

    sweep = threshold_sweep(rows)
    q04_d30 = next(
        row for row in sweep
        if row["lineQuality"] == 0.4 and row["distancePx"] == 30.0
    )
    q03_d30 = next(
        row for row in sweep
        if row["lineQuality"] == 0.3 and row["distancePx"] == 30.0
    )

    assert q04_d30["bothHitHumanBadCells"] == 1
    assert q04_d30["bothHitHumanGoodCells"] == 0
    assert q03_d30["bothHitHumanBadCells"] == 1
    assert q03_d30["bothHitHumanGoodCells"] == 1


def test_main_refuses_to_write_when_required_optional_deps_missing(
    monkeypatch,
    tmp_path,
    capsys,
):
    import tools.probe_bezel_discontinuity_join as mod

    output = tmp_path / "out.json"
    report = tmp_path / "report.md"
    monkeypatch.setattr(
        mod,
        "missing_required_optional_dependencies",
        lambda: ["rembg", "scipy"],
    )

    rc = main(["--output", str(output), "--report", str(report)])

    captured = capsys.readouterr()
    assert rc == 2
    assert "requires optional diagnostic dependencies" in captured.err
    assert "rembg, scipy" in captured.err
    assert not output.exists()
    assert not report.exists()
