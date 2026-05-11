from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .validation import FACE_ORDER, validate_state


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


@dataclass(frozen=True)
class ImageUpload:
    name: str
    data: bytes


@dataclass(frozen=True)
class ImagePair:
    set_id: str
    image_a: ImageUpload
    image_b: ImageUpload


def pair_image_uploads(uploads: Iterable[ImageUpload]) -> Tuple[List[ImagePair], List[str]]:
    buckets: Dict[str, Dict[str, ImageUpload]] = {}
    unpaired: List[str] = []
    fallback: List[ImageUpload] = []

    for upload in sorted(uploads, key=lambda item: item.name.lower()):
        marker = _image_marker(upload.name)
        if marker is None:
            fallback.append(upload)
            continue
        set_id, side = marker
        bucket = buckets.setdefault(set_id, {})
        if side in bucket:
            unpaired.append(upload.name)
            continue
        bucket[side] = upload

    pairs: List[ImagePair] = []
    for set_id, bucket in sorted(buckets.items()):
        if "A" in bucket and "B" in bucket:
            pairs.append(ImagePair(set_id=set_id, image_a=bucket["A"], image_b=bucket["B"]))
        else:
            unpaired.extend(upload.name for upload in bucket.values())

    if fallback:
        if len(fallback) % 2:
            unpaired.append(fallback[-1].name)
            fallback = fallback[:-1]
        for index in range(0, len(fallback), 2):
            first = fallback[index]
            second = fallback[index + 1]
            set_id = _fallback_set_id(first.name, second.name, index // 2 + 1)
            pairs.append(ImagePair(set_id=set_id, image_a=first, image_b=second))

    return pairs, sorted(unpaired)


def load_image_uploads_from_dir(path: Path) -> List[ImageUpload]:
    return [
        ImageUpload(item.name, item.read_bytes())
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]


def parse_ground_truth(data: bytes, filename: str = "ground_truth.csv") -> Dict[str, str]:
    text = data.decode("utf-8-sig", errors="replace")
    if filename.lower().endswith(".json") or text.lstrip().startswith(("{", "[")):
        parsed = _parse_ground_truth_json(text)
        if parsed:
            return parsed

    sample = text[:2048]
    delimiter = "\t" if filename.lower().endswith(".tsv") else _sniff_delimiter(sample)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    truth: Dict[str, str] = {}
    for row in reader:
        normalized = {str(key or "").strip().lower(): (value or "").strip() for key, value in row.items()}
        set_id = normalized.get("set_id") or normalized.get("set") or normalized.get("id") or normalized.get("name")
        if not set_id:
            continue
        state = normalized.get("expected_state") or normalized.get("state") or normalized.get("urfdlb")
        if not state and all(face.lower() in normalized for face in FACE_ORDER):
            state = "".join(normalized[face.lower()] for face in FACE_ORDER)
        if state:
            truth[normalize_set_id(set_id)] = state.strip().upper()
    return truth


def _parse_ground_truth_json(text: str) -> Dict[str, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    truth: Dict[str, str] = {}
    _collect_ground_truth_json(payload, truth)
    return truth


def _collect_ground_truth_json(value: Any, truth: Dict[str, str], inherited_set_id: Optional[str] = None) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_ground_truth_json(item, truth, inherited_set_id)
        return

    if not isinstance(value, dict):
        return

    normalized = {str(key).strip().lower(): item for key, item in value.items()}
    set_id = _first_json_value(normalized, "set_id", "set", "id", "name", "setname", "set_name") or inherited_set_id
    state = _json_state_candidate(normalized)
    if set_id and state:
        truth[normalize_set_id(str(set_id))] = state

    container_keys = {
        "sets",
        "items",
        "results",
        "groundtruth",
        "ground_truth",
        "truth",
        "entries",
        "cubes",
    }
    state_keys = {
        "corrected",
        "correctedstate",
        "corrected_state",
        "expected",
        "expectedstate",
        "expected_state",
        "state",
        "urfdlb",
    }

    for key, item in value.items():
        normalized_key = str(key).strip().lower()
        if normalized_key in state_keys:
            continue
        if isinstance(item, str):
            item_state = _normalize_state_candidate(item)
            if item_state:
                truth[normalize_set_id(str(key))] = item_state
            continue
        if normalized_key in container_keys:
            _collect_ground_truth_json(item, truth)
        elif isinstance(item, dict):
            _collect_ground_truth_json(item, truth, str(key))


def _first_json_value(items: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        value = items.get(key)
        if value not in (None, ""):
            return value
    return None


def _json_state_candidate(items: Dict[str, Any]) -> Optional[str]:
    value = _first_json_value(
        items,
        "corrected",
        "correctedstate",
        "corrected_state",
        "expected",
        "expectedstate",
        "expected_state",
        "state",
        "urfdlb",
    )
    state = _normalize_state_candidate(value)
    if state:
        return state
    if all(face.lower() in items for face in FACE_ORDER):
        return _normalize_state_candidate("".join(str(items[face.lower()]) for face in FACE_ORDER))
    return None


def _normalize_state_candidate(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    state = re.sub(r"\s+", "", value).upper()
    if len(state) != 54:
        return None
    if any(face not in FACE_ORDER for face in state):
        return None
    return _canonical_ground_truth_state(state)


def _canonical_ground_truth_state(state: str) -> str:
    """Return a canonical WCA URFDLB state for the given (possibly
    capture-frame) state.

    Two paths:

    1. **Fast-path / canonical input (preferred)**. If `state` already
       validates as a legal WCA URFDLB cube, return it unchanged. This
       is the expected case for ground truth captured against post-
       cube-two-view-debugger-PR-#20 cv-local + post-cube-snap-PR-#95
       Fixer: the saved `corrected` field is already canonical solver-
       ready URFDLB, with `recognitionSignals.captureFrameState`
       preserved separately for the photo-frame projection. New corpus
       entries (Set 42, the refreshed Set 12/Set 32 files) take this
       path.

    2. **Legacy / capture-frame input (fallback, deprecated)**. If the
       state isn't a valid WCA cube but its 6 centers are the 6 face
       letters, try all 4096 per-face rotation combinations to find a
       single legal URFDLB. This is the only path that used to handle
       Set 12 and Set 32's old saved ground truths (which were
       inadvertently saved in capture frame because the pre-PR-#95
       Fixer rendered diamonds in canonical frame regardless of yaw,
       so a "no edits needed" save inherited the canonical state but
       was interpreted by the user as photo-frame). When more than one
       legal candidate exists, the input is returned unchanged (ambiguous
       capture frames shouldn't be silently canonicalized).

    Deprecation intent: once the corpus manifest no longer references
    legacy capture-frame ground-truth files (tracked in issue #21), the
    legacy path can be removed and this function becomes the trivial
    `validate_state(state).valid` check inlined at the call site.
    """
    if validate_state(state).valid:
        return state

    chunks = [state[index * 9 : (index + 1) * 9] for index in range(6)]
    centers = [chunk[4] for chunk in chunks]
    if set(centers) != set(FACE_ORDER) or len(set(centers)) != len(FACE_ORDER):
        return state

    by_center = {chunk[4]: chunk for chunk in chunks}
    rotations = [_face_rotations(by_center[face]) for face in FACE_ORDER]
    legal = []
    for indices in product(range(4), repeat=6):
        candidate = "".join(rotations[face_index][rotation] for face_index, rotation in enumerate(indices))
        if validate_state(candidate).valid:
            legal.append(candidate)
            if len(legal) > 1:
                return state
    return legal[0] if legal else state


def _face_rotations(face: str) -> List[str]:
    rotations = [face]
    for _ in range(3):
        rotations.append(_rotate_face_clockwise(rotations[-1]))
    return rotations


def _rotate_face_clockwise(face: str) -> str:
    return "".join(face[index] for index in (6, 3, 0, 7, 4, 1, 8, 5, 2))


def parse_manifest_pairs(path: Path) -> Tuple[List[ImagePair], Dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else _sniff_delimiter(path.read_text(encoding="utf-8-sig", errors="replace")[:2048])
    truth: Dict[str, str] = {}
    pairs: List[ImagePair] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            normalized = {str(key or "").strip().lower(): (value or "").strip() for key, value in row.items()}
            set_id = normalized.get("set_id") or normalized.get("set") or normalized.get("id") or normalized.get("name")
            image_a = normalized.get("image_a") or normalized.get("a")
            image_b = normalized.get("image_b") or normalized.get("b")
            if not set_id or not image_a or not image_b:
                continue
            a_path = _resolve_manifest_path(path, image_a)
            b_path = _resolve_manifest_path(path, image_b)
            pairs.append(
                ImagePair(
                    set_id=set_id,
                    image_a=ImageUpload(a_path.name, a_path.read_bytes()),
                    image_b=ImageUpload(b_path.name, b_path.read_bytes()),
                )
            )
            expected = normalized.get("expected_state") or normalized.get("state") or normalized.get("urfdlb")
            if not expected and all(face.lower() in normalized for face in FACE_ORDER):
                expected = "".join(normalized[face.lower()] for face in FACE_ORDER)
            if expected:
                truth[normalize_set_id(set_id)] = expected.upper()
    return pairs, truth


def evaluate_state(actual: Optional[str], expected: Optional[str]) -> Dict[str, object]:
    if not expected:
        return {"available": False}
    expected = expected.strip().upper()
    actual = (actual or "").strip().upper()
    expected_validation = validate_state(expected)
    hamming = sum(1 for left, right in zip(actual, expected) if left != right) + abs(len(actual) - len(expected))
    return {
        "available": True,
        "expectedState": expected,
        "expectedValid": expected_validation.valid,
        "expectedErrors": expected_validation.errors,
        "exact": actual == expected,
        "hamming": hamming,
    }


def normalize_set_id(value: str) -> str:
    value = Path(value).stem
    value = re.sub(r"(?i)\bimage\s*[ab]\b", " ", value)
    value = re.sub(r"(?i)(?:^|[\s_-])(?:a|b)(?:[\s_-]|$)", " ", value)
    value = re.sub(r"(?i)\bimg[_\s-]*\d+\b", " ", value)
    value = re.sub(r"(?i)\bwhite\s*up\b", " ", value)
    value = re.sub(r"(?i)\b(?:jpg|jpeg|png|webp|heic|heif)\b", " ", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or "set"


def _image_marker(filename: str) -> Optional[Tuple[str, str]]:
    stem = Path(filename).stem
    patterns = (
        r"(?i)(?:^|[\s_-])([ab])(?:[\s_-]|$)",
        r"(?i)\bimage\s*([ab])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, stem)
        if not match:
            continue
        side = match.group(1).upper()
        set_id = normalize_set_id(stem[: match.start()] + " " + stem[match.end() :])
        return set_id, side
    return None


def _fallback_set_id(first: str, second: str, number: int) -> str:
    first_id = normalize_set_id(first)
    second_id = normalize_set_id(second)
    return first_id if first_id == second_id else f"pair-{number:03d}"


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        return ","


def _resolve_manifest_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return path
