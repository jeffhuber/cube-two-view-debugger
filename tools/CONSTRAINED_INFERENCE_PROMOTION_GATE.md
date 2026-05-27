# Constrained-Inference Promotion Gate

Diagnostic-only. This report evaluates a production-shaped gate for
using the hull-label constrained-inference candidate outside Fixer.
Ground truth is used only for scoring the gate, never for the gate
decision itself.

Git head: `1d8a28a75fc1189ff812c68bb82b6265dbd1c159`
Generated: `2026-05-27T17:34:30.243987+00:00`
Input: `tests/fixtures/pair_threshold_repair_diagnostic.json`

## Gate

A candidate may auto-return only when all of these production-available
conditions hold:

- the guarded pair-threshold selection assembled both sides;
- the yaw inference is accepted when present;
- both selected side thresholds were accepted and have no hard failures;
- the recommended repair is valid, count-balanced, and not low confidence;
- the recommended method is one of `['canonical_count_repaired', 'conservative_legal_repaired', 'guarded_broad_legal_repaired', 'two_view_consistency_repaired']`;
- canonical count repair uses at most `10` moves;
- legal repair methods stay within state delta `4`,
  repair cost `20.0`, and repair changes `6`;
- pair-threshold switching is allowed only when the current per-side
  threshold repair was invalid.

The ungated `broad_legal_repaired` method is intentionally excluded.

## Summary

| Pairs | Gate accepted | Gate rejected | Accepted exact | Accepted legal | Accepted <=3 | Threshold switches |
|---:|---:|---:|---:|---:|---:|---:|
| 71 | 71 | 0 | 71 | 71 | 71 | 1 |

Accepted hamming distribution: `{'0': 71}`

Accepted method counts:

- `canonical_count_repaired`: `67`
- `conservative_legal_repaired`: `2`
- `guarded_broad_legal_repaired`: `1`
- `two_view_consistency_repaired`: `1`

Selection reason counts:

- `kept_current_valid_repair`: `70`
- `current_invalid_selected_best_pair`: `1`

## Rejected Rows

_None._

## Interpretation

- Passing this diagnostic means the constrained-inference candidate is
  coherent enough to run in a default-recognizer shadow lane or an
  explicit candidate mode. It does not by itself flip `/api/recognize`.
- The production flip should require the same gate to hold on shadow
  traffic, plus a confidence policy for what to do when the gate rejects:
  fall back to legacy, return manual-review, or ask for a retake.
- The pair-threshold switch count is important: a switch is permitted
  only as a rescue when current deterministic repair is invalid, which is
  the Set 14 shape documented in the pair-threshold report.
