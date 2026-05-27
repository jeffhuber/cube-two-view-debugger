# Current Hull-Label Scoreboard

Diagnostic-only snapshot of the current hull-label constrained-inference path
on the 71-pair ground-truth corpus.

Git head: `0078609caa58f3455805650d576991f655ad2586`

## Headline

| Path | Exact cubes | Legal cubes | Hamming distribution |
|---|---:|---:|---|
| Raw canonical Lab classification | 29/71 | 29/71 | `{0: 29, 1: 7, 2: 11, 3: 5, 4: 3, 5: 6, 6: 3, 7: 2, 8: 1, 9: 1, 10: 2, 12: 1}` |
| Canonical count repair | 66/71 | 66/71 | `{0: 66, 2: 3, 3: 1, 4: 1}` |
| Current recommended repair selector | 70/71 | 70/71 | `{0: 70, 4: 1}` |
| Guarded pair-threshold selector | 71/71 | 71/71 | `{0: 71}` |
| Constrained-inference promotion gate | 71/71 accepted | 71/71 accepted | accepted `{0: 71}` |
| Hidden `/api/recognize?hullLabelTier1=constrained` mode | 71/71 | 71/71 | `{0: 71}` |

The current per-side threshold selector leaves one row, Set 14, at hamming 4.
The guarded pair-threshold selector switches Set 14 from thresholds
`{'A': 160, 'B': 160}` to `{'A': 64, 'B': 192}` and reaches hamming 0 without
regressing any row in the corpus.

## Reports

- `tools/HULL_LABEL_COLOR_REPAIR_DIAGNOSTIC.md` reruns deterministic color,
  count, two-view, and legality repair on the 71-pair corpus.
- `tools/PAIR_THRESHOLD_REPAIR_DIAGNOSTIC.md` compares current per-side
  threshold selection, aggressive pair-threshold selection, guarded
  pair-threshold selection, and oracle best threshold pairs.
- `tools/CURRENT_SCOREBOARD_FAILURE_GALLERY.md` renders the remaining
  current-selector failure as a large visual walkthrough.
- `tools/CONSTRAINED_INFERENCE_PROMOTION_GATE.md` applies a GT-free
  production-shaped auto-return gate to the guarded pair-threshold candidate.
- `tools/CONSTRAINED_RECOGNIZE_MODE_VALIDATION.md` runs the same gate at the
  recognizer boundary through the hidden constrained mode and compares it with
  the unchanged legacy/default recognizer path.

## Interpretation

The remaining miss is no longer a broad color-reading failure. Set 14's
per-side thresholds produce a plausible but invalid repaired state; evaluating
thresholds at the pair level chooses a different B-side mask threshold and
the same deterministic repair machinery becomes exact.

That makes the guarded pair-threshold path the current production-shaped
candidate. The promotion-gate diagnostic accepts 71/71 corpus rows and scores
71/71 exact using ground truth, and the hidden constrained `/api/recognize`
mode now reproduces that 71/71 exact result at the recognizer boundary while
the unchanged legacy path is 24/71 exact on the same corpus.

This is a strong shadow/default-candidate signal, but it is not yet a default
flip: the next production step should run the hidden constrained mode in shadow
on real traffic and explicitly preserve fallback/manual-review behavior when
the gate rejects.
