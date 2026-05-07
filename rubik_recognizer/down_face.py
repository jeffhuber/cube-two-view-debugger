from __future__ import annotations

from collections import Counter
from itertools import permutations
from typing import Iterable, List, Sequence

from .validation import FACE_ORDER, validate_state


D_UNKNOWN_INDICES = (27, 28, 29, 30, 32, 33, 34, 35)


def complete_down_face(partial_state: str, limit: int = 2) -> List[str]:
    """Return legal completions for a 54-char state with '?' in hidden D slots.

    The D center must already be set to D. The search is quota-constrained first, then
    checked with the cubie validator. Returning at most two candidates lets the caller
    distinguish unique success from ambiguity.
    """
    if len(partial_state) != 54:
        return []
    chars = list(partial_state)
    if chars[31] != "D":
        return []
    unknowns = [idx for idx in D_UNKNOWN_INDICES if chars[idx] == "?"]
    if any(chars[idx] == "?" for idx in range(54) if idx not in D_UNKNOWN_INDICES):
        return []

    counts = Counter(c for c in chars if c != "?")
    missing: List[str] = []
    for face in FACE_ORDER:
        needed = 9 - counts[face]
        if needed < 0:
            return []
        missing.extend(face for _ in range(needed))
    if len(missing) != len(unknowns):
        return []

    completions: List[str] = []
    seen = set()
    for candidate_values in unique_permutations(missing):
        for idx, face in zip(unknowns, candidate_values):
            chars[idx] = face
        candidate = "".join(chars)
        result = validate_state(candidate)
        if result.valid:
            completions.append(candidate)
            if len(completions) >= limit:
                break
        seen.add(candidate_values)
    for idx in unknowns:
        chars[idx] = "?"
    return completions


def unique_permutations(values: Sequence[str]) -> Iterable[Sequence[str]]:
    # itertools.permutations is fine for eight hidden slots, but duplicate colors are common.
    yielded = set()
    for item in permutations(values):
        if item in yielded:
            continue
        yielded.add(item)
        yield item
