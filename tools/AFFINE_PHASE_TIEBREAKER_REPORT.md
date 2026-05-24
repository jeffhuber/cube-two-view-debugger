# Affine phase tie-breaker audit

This diagnostic enumerates the 720 affine correspondence candidates and audits production-available secondary tie-breakers. Human full-corner truth is used only to label outcomes as usable or broken.

## Source

- Tool: `tools/diagnose_affine_phase_tiebreakers.py`
- Commit: `ad352898dca2c042d21545bff5e7f3c8bb77783c`
- Generated: `2026-05-24T23:32:29.494126+00:00`
- Truth: `tests/fixtures/full_corner_ground_truth.json`
- Manifest: `tests/fixtures/corpus_manifest.json`
- Max image dim: `1600`
- Exact tie RMS epsilon: `0.001`
- Near tie RMS epsilon: `0.25`
- Mask path: rembg.remove(...).alpha channel, matching production baselines
- Human truth usage: evaluation only, never selector input

## Aggregate

- Rows traced: 12 / 12
- Selector axis-state counts: `{'production': {'broken': 9, 'usable': 3}, 'exact_bezel': {'usable': 8, 'broken': 4}, 'exact_center_to_bezel': {'usable': 8, 'broken': 4}, 'exact_bezel_then_center': {'usable': 8, 'broken': 4}, 'near_bezel': {'usable': 8, 'broken': 4}, 'near_center_to_bezel': {'usable': 8, 'broken': 4}, 'near_bezel_then_center': {'usable': 8, 'broken': 4}}`
- Selector effects vs production: `{'production': {'baseline': 12}, 'exact_bezel': {'fixes_broken_to_usable': 6, 'keeps_broken': 3, 'breaks_usable_to_broken': 1, 'keeps_usable': 2}, 'exact_center_to_bezel': {'fixes_broken_to_usable': 6, 'keeps_broken': 3, 'breaks_usable_to_broken': 1, 'keeps_usable': 2}, 'exact_bezel_then_center': {'fixes_broken_to_usable': 6, 'keeps_broken': 3, 'breaks_usable_to_broken': 1, 'keeps_usable': 2}, 'near_bezel': {'fixes_broken_to_usable': 6, 'keeps_broken': 3, 'breaks_usable_to_broken': 1, 'keeps_usable': 2}, 'near_center_to_bezel': {'fixes_broken_to_usable': 6, 'keeps_broken': 3, 'breaks_usable_to_broken': 1, 'keeps_usable': 2}, 'near_bezel_then_center': {'fixes_broken_to_usable': 6, 'keeps_broken': 3, 'breaks_usable_to_broken': 1, 'keeps_usable': 2}}`
- Exact tie rows with a usable candidate: `12`
- Near tie rows with a usable candidate: `12`
- Median exact / near group size: `12.0` / `12.0`
- Exact rows with nonzero bezel / center metric range: `0` / `0`

## Headline Findings

- Every row has a 12-candidate exact residual tie group with 6 usable and 6 broken phases.
- Simple geometric secondary metrics are degenerate in this group: bezel-axis alignment and center proximity do not vary across exact ties. Apparent selector wins/losses from these metrics are therefore fallback-order effects, not reliable evidence.
- The next production candidate needs a signal that breaks the 3-fold phase symmetry, such as directed face/color evidence or a stronger convention-aware correspondence constraint.

## Per-row Summary

| Row | prod | exact n/u | exact metric ranges | exact bezel | near n/u | near bezel | best oracle |
|---|---|---:|---|---|---:|---|---|
| `20_A` | broken (177.4) | 12 / 6 | bezel=0.0; center=0.0 | usable (8.4) | 12 / 6 | usable (8.4) | usable (8.4) |
| `20_B` | broken (179.8) | 12 / 6 | bezel=0.0; center=0.0 | usable (10.5) | 12 / 6 | usable (10.5) | usable (10.5) |
| `38_A` | broken (179.2) | 12 / 6 | bezel=0.0; center=0.0 | usable (11.2) | 12 / 6 | usable (11.2) | usable (11.2) |
| `38_B` | broken (175.2) | 12 / 6 | bezel=0.0; center=0.0 | usable (8.3) | 12 / 6 | usable (8.3) | usable (8.3) |
| `40_A` | broken (178.8) | 12 / 6 | bezel=0.0; center=0.0 | broken (178.8) | 12 / 6 | broken (178.8) | usable (11.7) |
| `40_B` | broken (178.7) | 12 / 6 | bezel=0.0; center=0.0 | broken (178.7) | 12 / 6 | broken (178.7) | usable (7.4) |
| `41_A` | usable (12.9) | 12 / 6 | bezel=0.0; center=0.0 | broken (179.4) | 12 / 6 | broken (179.4) | usable (12.9) |
| `41_B` | broken (179.8) | 12 / 6 | bezel=0.0; center=0.0 | broken (179.8) | 12 / 6 | broken (179.8) | usable (7.0) |
| `43_A` | usable (9.3) | 12 / 6 | bezel=0.0; center=0.0 | usable (9.3) | 12 / 6 | usable (9.3) | usable (9.3) |
| `43_B` | broken (177.8) | 12 / 6 | bezel=0.0; center=0.0 | usable (13.4) | 12 / 6 | usable (13.4) | usable (10.5) |
| `45_A` | usable (13.5) | 12 / 6 | bezel=0.0; center=0.0 | usable (13.5) | 12 / 6 | usable (13.5) | usable (11.6) |
| `45_B` | broken (177.3) | 12 / 6 | bezel=0.0; center=0.0 | usable (18.8) | 12 / 6 | usable (18.8) | usable (7.8) |

## Interpretation

- `production` is strict residual then permutation order, matching the current affine selector.
- `exact_*` selectors only choose among candidates whose residual RMS is tied with the minimum.
- `near_*` selectors choose among candidates within the near-residual band; these are diagnostic only unless a future production prototype defines a safe band.
- If a selector repeatedly fixes `production` broken rows without breaking usable rows AND has a nonzero metric range inside the tied group, it is a candidate production tie-breaker.
