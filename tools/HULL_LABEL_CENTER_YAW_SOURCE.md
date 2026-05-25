# Hull-Label Center-Color Yaw Source Diagnostic

## Purpose

This diagnostic evaluates a production-shaped yaw source for the
hull-label path: infer capture yaw from the six rectified slot center
stickers, then map `upper` / `right` / `front` slots to WCA faces via
`tools.corner_conventions.wca_face_by_slot()`.

The inference accepts a yaw only when the best candidate has at least
5/6 center matches and a margin of at least
2 over the runner-up.

Git head: `e1c6457ed1da783ae46c15cba01fcc2ba8794906`
Generated: `2026-05-25T16:13:20.081306+00:00`
Input trace: `tests/fixtures/hull_label_slot_yaw_assignment.json`

## Summary

- Rows evaluated: 41
- Accepted yaw inference: 40 / 41
- Rejected: 1 (`{'fit_failed': 1}`)
- Agreement with manifest/human-known yaw: 14 / 14
- Agreement with legacy `captureYaw` when available: 27 / 27
- Rows inferred where legacy `captureYaw` was missing: 13
- Best-score buckets: `{'5': 13, '6': 27}`
- Margin buckets: `{'2': 12, '4': 28}`

## Recommendation

Use center-color yaw inference as the production hull-label yaw source.
It is available from the hull-label rectified faces themselves, so it
survives rows where the legacy recognizer rejects before emitting
`captureYaw`. If CubeSnap later has explicit capture-yaw metadata, pass
it as an optional hint/override and cross-check it against center-color
inference; a conflict should fall back rather than force a yaw.

Do not depend on the legacy `captureYaw` signal for the hull-label path.
It is useful diagnostic metadata, but it is missing exactly in the class
of reject rows where the hull-label path is supposed to help.

## Per-Pair Snapshot

| Set | Inferred yaw | Score | Margin | Manifest yaw | Legacy detected yaw | Result |
|---|---:|---:|---:|---:|---:|---|
| 12 | 3 | 6 | 4 | 3 | 3 | ok |
| 14 | 0 | 5 | 2 | None | 0 | ok |
| 15 | 0 | 6 | 4 | None | 0 | ok |
| 23 | 2 | 6 | 4 | 2 | 2 | ok |
| 24 | 0 | 5 | 2 | None | 0 | ok |
| 26 | 0 | 5 | 2 | None | 0 | ok |
| 27 | 0 | 5 | 2 | None | 0 | ok |
| 28 | 0 | 5 | 2 | None | None | inferred_no_legacy_yaw |
| 29 | 0 | 5 | 2 | None | 0 | ok |
| 31 | 1 | 6 | 4 | None | None | inferred_no_legacy_yaw |
| 32 | 1 | 6 | 4 | 1 | 1 | ok |
| 36 | 2 | 6 | 4 | 2 | 2 | ok |
| 37 | 1 | 6 | 4 | 1 | 1 | ok |
| 42 | 1 | 6 | 4 | 1 | 1 | ok |
| 44 | 0 | 6 | 4 | None | None | inferred_no_legacy_yaw |
| 20 | 0 | 6 | 4 | None | 0 | ok |
| 38 | 1 | 6 | 4 | None | 1 | ok |
| 40 | 0 | 6 | 4 | None | 0 | ok |
| 41 | 1 | 6 | 4 | None | 1 | ok |
| 43 | 1 | 6 | 4 | None | 1 | ok |
| 45 | 0 | 6 | 4 | None | 0 | ok |
| 17 | 1 | 6 | 4 | None | None | inferred_no_legacy_yaw |
| 21 | 1 | 6 | 4 | None | 1 | ok |
| 22 | 2 | 6 | 4 | None | 2 | ok |
| 25 | 0 | 5 | 2 | None | None | inferred_no_legacy_yaw |
| 30 | None | None | None | None | 1 | fit_failed |
| 39 | 1 | 6 | 4 | None | None | inferred_no_legacy_yaw |
| 46 | 0 | 5 | 2 | 0 | None | inferred_no_legacy_yaw |
| 47 | 1 | 6 | 4 | 1 | None | inferred_no_legacy_yaw |
| 48 | 2 | 6 | 4 | 2 | None | inferred_no_legacy_yaw |
| 49 | 3 | 5 | 2 | 3 | None | inferred_no_legacy_yaw |
| 57 | 1 | 5 | 4 | None | 1 | ok |
| 58 | 2 | 6 | 4 | None | 2 | ok |
| 61 | 0 | 6 | 4 | None | 0 | ok |
| 62 | 1 | 5 | 2 | None | 1 | ok |
| 63 | 2 | 6 | 4 | 2 | 2 | ok |
| 64 | 0 | 5 | 2 | 0 | None | inferred_no_legacy_yaw |
| 65 | 1 | 6 | 4 | None | None | inferred_no_legacy_yaw |
| 66 | 2 | 6 | 4 | None | None | inferred_no_legacy_yaw |
| 67 | 3 | 5 | 2 | 3 | 3 | ok |
| 68 | 0 | 6 | 4 | 0 | 0 | ok |
