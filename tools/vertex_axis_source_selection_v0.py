#!/usr/bin/env python3
"""Evaluate vertex/axis source-selection confidence policies.

Diagnostics/data-only. This module does not alter recognizer behavior.

The source pool is the paired rembg/SAM3 whole-cube global-model outputs from
the geometry-first face split probe. Each policy may select one source or
abstain. The goal is to test whether existing fit-quality signals can choose a
trusted visible trihedral vertex and axis model before rectified color reads.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY_FIXTURE = ROOT / "tests" / "fixtures" / "geometry_first_face_split_v0_summary.json"
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "vertex_axis_source_selection_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "VERTEX_AXIS_SOURCE_SELECTION_V0_REPORT.md"
STRICT_VERTEX_PX = 30.0
PLAUSIBLE_VERTEX_PX = 50.0

PolicyFn = Callable[[Dict[str, Any]], Tuple[Optional[str], Dict[str, Any]]]


def generate_source_selection_bakeoff(
    *,
    geometry_fixture_path: Path = DEFAULT_GEOMETRY_FIXTURE,
) -> Dict[str, Any]:
    geometry = _read_json(geometry_fixture_path)
    rows = list(geometry.get("rows") or [])
    policies: Dict[str, PolicyFn] = {
        "always_sam3": _always_sam3,
        "always_rembg": _always_rembg,
        "global_model_score_v0": _global_model_score_v0,
        "strict_residual_margin_confidence_v0": _strict_residual_margin_confidence_v0,
        "high_fit_quality_margin_confidence_v0": _high_fit_quality_margin_confidence_v0,
        "oracle_best_source": _oracle_best_source,
    }
    policy_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in policies}
    for row in rows:
        for name, policy in policies.items():
            selection, details = policy(row)
            policy_rows[name].append(_evaluate_selection(row, name, selection, details))

    return {
        "schemaVersion": 1,
        "probe": "vertex_axis_source_selection_v0",
        "description": (
            "Diagnostics/data-only bakeoff for choosing between rembg/SAM3 "
            "whole-cube global-model hypotheses, with abstention when "
            "confidence is low."
        ),
        "sourceGeometryFixture": str(geometry_fixture_path),
        "config": {
            "strictVertexPx": STRICT_VERTEX_PX,
            "plausibleVertexPx": PLAUSIBLE_VERTEX_PX,
            "strictResidualMarginConfidenceV0": {
                "maxSelectedResidualRmsPx": 18.0,
                "minResidualMarginPx": 20.0,
            },
            "highFitQualityMarginConfidenceV0": {
                "minSelectedFitQuality": 0.90,
                "minFitQualityMargin": 0.08,
            },
        },
        "summary": {
            "rowCount": len(rows),
            "policySummaries": {
                name: _summarize_policy(entries)
                for name, entries in policy_rows.items()
            },
        },
        "rows": [
            {
                "key": row.get("key"),
                "bestSourceByVertexError": row.get("bestSourceByVertexError"),
                "trueVertex": row.get("trueVertex"),
                "sourceErrors": {
                    source: row.get("sources", {}).get(source, {}).get("vertexErrorPx")
                    for source in ("rembg", "sam3")
                },
                "sourceDiagnostics": {
                    source: _source_diagnostics(row.get("sources", {}).get(source, {}))
                    for source in ("rembg", "sam3")
                },
                "policyResults": {
                    name: policy_rows[name][idx]
                    for name in policies
                },
            }
            for idx, row in enumerate(rows)
        ],
    }


def render_report(document: Dict[str, Any]) -> str:
    summaries = document["summary"]["policySummaries"]
    lines = [
        "# Vertex/Axis Source Selection V0",
        "",
        "Diagnostics/data-only artifact. This does not alter recognition behavior.",
        "",
        "This report tests whether existing global-model fit signals can choose between rembg- and SAM3-driven vertex/axis hypotheses, and abstain when confidence is low.",
        "",
        "## Summary",
        "",
        f"- Rows: {document['summary']['rowCount']}",
        f"- Strict threshold: {STRICT_VERTEX_PX:.0f} px",
        f"- Plausible threshold: {PLAUSIBLE_VERTEX_PX:.0f} px",
        "",
        "| Policy | Selected | Abstained | Best-source correct | Strict-ready | Plausible | False-confident >50px | Mean selected error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "always_rembg",
        "always_sam3",
        "global_model_score_v0",
        "strict_residual_margin_confidence_v0",
        "high_fit_quality_margin_confidence_v0",
        "oracle_best_source",
    ):
        summary = summaries[name]
        mean_error = summary["meanSelectedErrorPx"]
        mean_text = "" if mean_error is None else f"{mean_error:.1f} px"
        lines.append(
            f"| `{name}` | {summary['selectedCount']} | {summary['abstainCount']} | "
            f"{summary['bestSourceCorrectCount']} | {summary['strictReadyCount']} | "
            f"{summary['plausibleCount']} | {summary['falseConfidentCount']} | {mean_text} |"
        )

    lines.extend([
        "",
        "## Per-Row Readout",
        "",
        "| Row | Best source | rembg err | SAM3 err | Score policy | Strict confidence |",
        "|---|---|---:|---:|---|---|",
    ])
    for row in document["rows"]:
        score = row["policyResults"]["global_model_score_v0"]
        strict = row["policyResults"]["strict_residual_margin_confidence_v0"]
        lines.append(
            f"| `{row['key']}` | `{row['bestSourceByVertexError']}` | "
            f"{row['sourceErrors']['rembg']:.0f} | {row['sourceErrors']['sam3']:.0f} | "
            f"`{score['selection']}` / {score.get('selectedErrorPx', '')} | "
            f"`{strict['selection']}` / {strict.get('selectedErrorPx', '')} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The broad global-model score policy improves source choice versus a fixed source, but it still selects too many bad vertices to trust.",
        "- The conservative residual-margin confidence policy can abstain its way to zero false-confident rows here, but it selects only one row. That is evidence of signal, not a usable recognizer path.",
        "- The oracle across rembg/SAM3 remains far better than the deployable policies, so the missing piece is confidence/source selection, not deterministic face splitting.",
        "- Production wiring should continue to wait. The next useful move is adding richer confidence features or more labeled rows, then re-running this exact source-selection report.",
        "",
    ])
    return "\n".join(lines)


def _evaluate_selection(
    row: Dict[str, Any],
    policy_name: str,
    selection: Optional[str],
    details: Dict[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "policy": policy_name,
        "selection": selection or "abstain",
        "details": details,
    }
    if selection is None:
        result["status"] = "abstained"
        return result
    source = row["sources"][selection]
    error = float(source["vertexErrorPx"])
    result.update({
        "status": _status_from_error(error),
        "selectedErrorPx": round(error, 2),
        "bestSourceCorrect": selection == row.get("bestSourceByVertexError"),
        "fitResidualRmsPx": source.get("fitResidualRmsPx"),
        "fitQuality": source.get("fitQuality"),
    })
    return result


def _summarize_policy(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    selected = [entry for entry in entries if entry["selection"] != "abstain"]
    errors = [float(entry["selectedErrorPx"]) for entry in selected]
    return {
        "rowCount": len(entries),
        "selectedCount": len(selected),
        "abstainCount": len(entries) - len(selected),
        "bestSourceCorrectCount": sum(1 for entry in selected if entry.get("bestSourceCorrect")),
        "strictReadyCount": sum(1 for entry in selected if float(entry["selectedErrorPx"]) <= STRICT_VERTEX_PX),
        "plausibleCount": sum(1 for entry in selected if float(entry["selectedErrorPx"]) <= PLAUSIBLE_VERTEX_PX),
        "falseConfidentCount": sum(1 for entry in selected if float(entry["selectedErrorPx"]) > PLAUSIBLE_VERTEX_PX),
        "meanSelectedErrorPx": round(float(statistics.fmean(errors)), 4) if errors else None,
        "medianSelectedErrorPx": round(float(statistics.median(errors)), 4) if errors else None,
    }


def _always_sam3(row: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    return "sam3", {"reason": "fixed_source"}


def _always_rembg(row: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    return "rembg", {"reason": "fixed_source"}


def _oracle_best_source(row: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    return str(row.get("bestSourceByVertexError")), {"reason": "label_oracle"}


def _global_model_score_v0(row: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    rembg = row["sources"]["rembg"]
    sam3 = row["sources"]["sam3"]
    if float(sam3["fitResidualRmsPx"]) < float(rembg["fitResidualRmsPx"]):
        selection = "sam3"
    else:
        selection = "rembg"
    return selection, {
        "reason": "lower_fit_residual_rms",
        "residualMarginPx": round(abs(float(sam3["fitResidualRmsPx"]) - float(rembg["fitResidualRmsPx"])), 2),
    }


def _strict_residual_margin_confidence_v0(row: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    selection, details = _global_model_score_v0(row)
    if selection is None:
        return None, details
    selected = row["sources"][selection]
    other_source = "rembg" if selection == "sam3" else "sam3"
    other = row["sources"][other_source]
    selected_residual = float(selected["fitResidualRmsPx"])
    other_residual = float(other["fitResidualRmsPx"])
    margin = other_residual - selected_residual
    details = {
        "reason": "lower_residual_with_strict_margin",
        "selectedResidualRmsPx": round(selected_residual, 2),
        "otherResidualRmsPx": round(other_residual, 2),
        "residualMarginPx": round(margin, 2),
    }
    if selected_residual <= 18.0 and margin >= 20.0:
        return selection, details
    return None, details


def _high_fit_quality_margin_confidence_v0(row: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    rembg = row["sources"]["rembg"]
    sam3 = row["sources"]["sam3"]
    rembg_quality = float(rembg["fitQuality"])
    sam3_quality = float(sam3["fitQuality"])
    if sam3_quality > rembg_quality:
        selection = "sam3"
        margin = sam3_quality - rembg_quality
        selected_quality = sam3_quality
        other_quality = rembg_quality
    else:
        selection = "rembg"
        margin = rembg_quality - sam3_quality
        selected_quality = rembg_quality
        other_quality = sam3_quality
    details = {
        "reason": "higher_fit_quality_with_strict_margin",
        "selectedFitQuality": round(selected_quality, 4),
        "otherFitQuality": round(other_quality, 4),
        "fitQualityMargin": round(margin, 4),
    }
    if selected_quality >= 0.90 and margin >= 0.08:
        return selection, details
    return None, details


def _status_from_error(error: float) -> str:
    if error <= STRICT_VERTEX_PX:
        return "strict_ready"
    if error <= PLAUSIBLE_VERTEX_PX:
        return "plausible"
    return "false_confident"


def _source_diagnostics(source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": source.get("status"),
        "vertex": source.get("vertex"),
        "vertexErrorPx": source.get("vertexErrorPx"),
        "fitResidualRmsPx": source.get("fitResidualRmsPx"),
        "fitQuality": source.get("fitQuality"),
        "refinement": source.get("refinement"),
        "cubeCenterSource": source.get("cubeCenterSource"),
        "nondegenerate": source.get("nondegenerate"),
    }


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-fixture", type=Path, default=DEFAULT_GEOMETRY_FIXTURE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    document = generate_source_selection_bakeoff(
        geometry_fixture_path=args.geometry_fixture,
    )
    _write_json(args.summary, document)
    _write_text(args.report, render_report(document))
    print(f"wrote {args.summary}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
