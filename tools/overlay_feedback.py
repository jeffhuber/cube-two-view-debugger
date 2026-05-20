#!/usr/bin/env python3
"""Ingest human visual feedback from hybrid overlay review workbooks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = Path.home() / "Downloads" / "overlay tool visual feedback  (1).xlsx"
DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "hard_case_visual_feedback.json"
DEFAULT_REPORT = ROOT / "tools" / "OVERLAY_FEEDBACK_REPORT.md"

SLOT_LAYOUT = {
    "imageA": [
        {"slot": "U", "outline": "red"},
        {"slot": "R", "outline": "green"},
        {"slot": "F", "outline": "blue"},
    ],
    "imageB": [
        {"slot": "D", "outline": "purple"},
        {"slot": "L", "outline": "orange"},
        {"slot": "B", "outline": "yellow"},
    ],
}


def read_xlsx_first_sheet(path: Path) -> List[List[Any]]:
    """Read the first worksheet values using only stdlib xlsx XML parsing."""
    with zipfile.ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet_name = _first_sheet_path(zf)
        root = ElementTree.fromstring(zf.read(sheet_name))

    rows: Dict[int, Dict[int, Any]] = defaultdict(dict)
    max_col = 0
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    for cell in root.findall(".//x:c", ns):
        ref = cell.attrib.get("r", "")
        row_index, col_index = _cell_ref_indices(ref)
        max_col = max(max_col, col_index)
        value = _cell_value(cell, shared_strings)
        rows[row_index][col_index] = value

    matrix: List[List[Any]] = []
    for row_index in range(1, max(rows.keys(), default=0) + 1):
        matrix.append([rows[row_index].get(col_index) for col_index in range(1, max_col + 1)])
    return matrix


def ingest_overlay_feedback_rows(
    rows: Sequence[Sequence[Any]],
    *,
    source_workbook: str = "",
    source_sheet: str = "Sheet1",
) -> Dict[str, Any]:
    """Normalize the current overlay-feedback workbook layout into slot labels."""
    data_rows = []
    for row in rows:
        if not row:
            continue
        first = _string_or_none(row[0])
        if not first or first.lower() == "image set":
            continue
        if not first.isdigit():
            continue
        data_rows.append(row)

    sets = []
    for row in data_rows:
        set_id = str(row[0]).strip()
        slots = []
        for image_key, offset in (("imageA", 1), ("imageB", 7)):
            rectified_offset = offset + 3
            for index, layout in enumerate(SLOT_LAYOUT[image_key]):
                quad_quality = _quality(row[offset + index] if offset + index < len(row) else None)
                rectified_raw = _string_or_none(
                    row[rectified_offset + index] if rectified_offset + index < len(row) else None
                )
                rectified = parse_rectified_label(rectified_raw)
                slot = {
                    "image": image_key,
                    "side": "A" if image_key == "imageA" else "B",
                    "slot": layout["slot"],
                    "outline": layout["outline"],
                    "quadQuality": quad_quality,
                    "rectifiedQuality": rectified["quality"],
                    "rectifiedSourceFace": rectified["sourceFace"],
                    "rawRectifiedLabel": rectified_raw,
                }
                slot["failureModes"] = slot_failure_modes(slot)
                slots.append(slot)
        sets.append(
            {
                "setId": set_id,
                "slots": slots,
                "badSlotCount": sum(1 for slot in slots if _slot_is_bad(slot)),
            }
        )

    document = {
        "schemaVersion": 1,
        "source": {
            "workbook": source_workbook,
            "sheet": source_sheet,
            "description": "Human visual review of hybrid overlay quads and rectified slot images.",
        },
        "slotLayout": SLOT_LAYOUT,
        "sets": sets,
    }
    document["summary"] = visual_feedback_summary(document)
    return document


def parse_rectified_label(raw: Optional[str]) -> Dict[str, Optional[str]]:
    text = (raw or "").strip()
    source_match = re.search(r"\bsr[cd]\s*=\s*([URFDLB])\b", text, flags=re.IGNORECASE)
    quality = _quality(text)
    return {
        "sourceFace": source_match.group(1).upper() if source_match else None,
        "quality": quality,
    }


def slot_failure_modes(slot: Dict[str, Any]) -> List[str]:
    modes = []
    slot_face = slot.get("slot")
    source_face = slot.get("rectifiedSourceFace")
    quad_bad = slot.get("quadQuality") == "bad"
    rectified_bad = slot.get("rectifiedQuality") == "bad"
    if quad_bad:
        modes.append("bad_quad")
    if rectified_bad:
        modes.append("bad_rectified")
    if source_face and source_face != slot_face:
        modes.append("wrong_source_face")
    if slot.get("quadQuality") == "good" and rectified_bad:
        modes.append("rectification_bad_despite_good_quad")
    if quad_bad and slot.get("rectifiedQuality") == "good":
        modes.append("rectification_survives_bad_quad")
    if not modes:
        modes.append("ok")
    return modes


def visual_feedback_summary(document: Dict[str, Any]) -> Dict[str, Any]:
    slots = [slot for item in document.get("sets", []) for slot in item.get("slots", [])]
    mode_counts = Counter(mode for slot in slots for mode in slot.get("failureModes", []) if mode != "ok")
    slot_counts: Dict[str, Counter] = defaultdict(Counter)
    image_counts: Dict[str, Counter] = defaultdict(Counter)
    for slot in slots:
        slot_key = f"{slot.get('side')}:{slot.get('slot')}"
        for mode in slot.get("failureModes", []):
            if mode == "ok":
                continue
            slot_counts[slot_key][mode] += 1
            image_counts[str(slot.get("image"))][mode] += 1

    ranked_slots = []
    for slot_key, counts in slot_counts.items():
        ranked_slots.append(
            {
                "slot": slot_key,
                "badSignalCount": sum(counts.values()),
                "failureModes": dict(sorted(counts.items())),
            }
        )
    ranked_slots.sort(key=lambda item: (-item["badSignalCount"], item["slot"]))

    return {
        "setCount": len(document.get("sets", [])),
        "slotCount": len(slots),
        "badSlotCount": sum(1 for slot in slots if _slot_is_bad(slot)),
        "failureModeCounts": dict(sorted(mode_counts.items())),
        "failureModesByImage": {key: dict(sorted(value.items())) for key, value in sorted(image_counts.items())},
        "rankedSlots": ranked_slots,
    }


def render_visual_feedback_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Overlay Feedback Visual Labels",
        "",
        "Diagnostics/data-only artifact generated from the human overlay feedback workbook.",
        "It records whether each hybrid overlay slot's quad and rectified face looked good or bad.",
        "",
        "## Summary",
        "",
        f"- Sets reviewed: {summary['setCount']}",
        f"- Slots reviewed: {summary['slotCount']}",
        f"- Slots with at least one bad signal: {summary['badSlotCount']}",
        "",
        "## Failure Modes",
        "",
        "| Failure mode | Count |",
        "|---|---:|",
    ]
    for mode, count in summary["failureModeCounts"].items():
        lines.append(f"| `{mode}` | {count} |")

    lines.extend(
        [
            "",
            "## Ranked Slots",
            "",
            "| Slot | Bad signals | Modes |",
            "|---|---:|---|",
        ]
    )
    for item in summary["rankedSlots"]:
        modes = ", ".join(f"`{mode}`={count}" for mode, count in item["failureModes"].items())
        lines.append(f"| `{item['slot']}` | {item['badSignalCount']} | {modes} |")

    lines.extend(
        [
            "",
            "## Per-Set Labels",
            "",
            "| Set | Bad slots | Slot labels |",
            "|---|---:|---|",
        ]
    )
    for item in document["sets"]:
        labels = []
        for slot in item["slots"]:
            modes = [mode for mode in slot["failureModes"] if mode != "ok"]
            if not modes:
                continue
            source = slot.get("rectifiedSourceFace") or "?"
            labels.append(
                f"`{slot['side']}:{slot['slot']}` quad={slot['quadQuality']} "
                f"rect={slot['rectifiedQuality']} src={source} ({', '.join(modes)})"
            )
        lines.append(f"| {item['setId']} | {item['badSlotCount']} | {'<br>'.join(labels)} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `wrong_source_face` means the rectified face was sourced from a different detected face than the slot label.",
            "- `rectification_bad_despite_good_quad` is especially useful for sampling/rectification debugging.",
            "- This report is not a recognizer behavior change; it is supervision for future diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(data)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings = []
    for item in root.findall("x:si", ns):
        parts = [text.text or "" for text in item.findall(".//x:t", ns)]
        strings.append("".join(parts))
    return strings


def _first_sheet_path(zf: zipfile.ZipFile) -> str:
    workbook = ElementTree.fromstring(zf.read("xl/workbook.xml"))
    ns = {
        "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    sheet = workbook.find(".//x:sheet", ns)
    if sheet is None:
        raise ValueError("Workbook has no sheets.")
    relationship_id = sheet.attrib[f"{{{ns['r']}}}id"]

    rels = ElementTree.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    for rel in rels.findall("r:Relationship", rel_ns):
        if rel.attrib.get("Id") == relationship_id:
            target = rel.attrib["Target"].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError(f"Could not resolve sheet relationship {relationship_id}.")


def _cell_value(cell: ElementTree.Element, shared_strings: Sequence[str]) -> Any:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    value_type = cell.attrib.get("t")
    value = cell.find("x:v", ns)
    if value_type == "inlineStr":
        parts = [text.text or "" for text in cell.findall(".//x:t", ns)]
        return "".join(parts)
    if value is None or value.text is None:
        return None
    if value_type == "s":
        return shared_strings[int(value.text)]
    try:
        number = float(value.text)
    except ValueError:
        return value.text
    return int(number) if number.is_integer() else number


def _cell_ref_indices(ref: str) -> tuple[int, int]:
    match = re.match(r"([A-Z]+)([0-9]+)", ref)
    if not match:
        raise ValueError(f"Invalid cell reference: {ref}")
    col = 0
    for char in match.group(1):
        col = col * 26 + (ord(char) - ord("A") + 1)
    return int(match.group(2)), col


def _quality(value: Any) -> Optional[str]:
    text = (_string_or_none(value) or "").lower()
    if "bad" in text:
        return "bad"
    if "good" in text:
        return "good"
    return None


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _slot_is_bad(slot: Dict[str, Any]) -> bool:
    return any(mode != "ok" for mode in slot.get("failureModes", []))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    rows = read_xlsx_first_sheet(args.workbook)
    document = ingest_overlay_feedback_rows(
        rows,
        source_workbook=args.workbook.name,
    )
    write_json(args.output, document)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_visual_feedback_report(document), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
