# Pipeline phase-parity failure modes

Diagnostic-only. Walks each human-validated full-corner row through the recognizer pipeline and answers Codex's 4 questions (see `tools/diagnose_pipeline_phase_parity.py` docstring).

Rows: 12 traced / 12 total. Stability across 5 runs per row:

- **8/12 post-category-stable** (all runs agree on the FINAL outcome — the modal post-category is reliable as a row-level summary)
- **0/12 delta-class-stable** (all runs agree on the path the pipeline took — `flip_helped` vs `flip_hurt` vs `no_flip`)
- **0/12 fully stable** (both)

> ⚠️ **Pipeline non-determinism note (Codex P1 on PR #255).** The recognizer is stochastic across runs (likely ONNX thread ordering in rembg + vertex-refinement non-determinism). This diagnostic runs each row N times and reports DISTRIBUTIONS + modal values. Headline counts below quote MODAL per-row outcomes; row-level claims should be read against the per-row distribution table to know whether they are stable.

> ⚠️ **`phase_rewound` ≠ true pre-correction state (Codex P2 on PR #255).** The pre-flip model is reconstructed from the post-pipeline output by mathematically inverting the phase flip. But vertex refinement runs AFTER phase correction in production, so the reconstruction preserves vertex refinement that was tuned to the post-flip axes. This is a useful inverse-axis probe but is NOT the same as an `apply_phase_correction=False` pipeline run. See the module docstring for the semantic caveat.

## Aggregate (modal per-row outcomes)

Each row contributes its MODAL outcome across `n_runs` trials. Stable rows have one unanimous modal value; unstable rows pick the most-common run outcome (ties broken by sort order).

**Post-pipeline canonical category (production outcome):**

| Category | Rows (modal) |
|---|---:|
| `PHASE_SWAPPED` | 7 |
| `GOOD` | 4 |
| `MARGINAL` | 1 |

**Phase-rewound canonical category** (caveat above):

| Category | Rows (modal) |
|---|---:|
| `GOOD` | 6 |
| `PHASE_SWAPPED` | 6 |

**`phase_check` distribution (modal):**

| phase_check | Rows (modal) |
|---|---:|
| `corrected_60deg_flip` | 5 |
| `ambiguous_no_correction` | 4 |
| `correct` | 3 |

**Score-delta classification** (modal; did the flip help, hurt, or no-op?):

| Class | Rows (modal) | Meaning |
|---|---:|---|
| `no_flip` | 7 | phase-correction made no change |
| `flip_hurt` | 3 | flip moved category toward worse |
| `flip_helped` | 2 | flip moved category toward better |

## Per-row trace (distributions over N runs)

| Key | Stable? | Post-category dist | phase_check dist | Score-delta dist |
|---|---|---|---|---|
| `20_A` | ✓ | PHASE_SWAPPED:5 | corrected_60deg_flip:4, correct:1 | flip_hurt:4, no_flip:1 |
| `20_B` | ✓ | PHASE_SWAPPED:5 | correct:3, corrected_60deg_flip:2 | no_flip:3, flip_hurt:2 |
| `38_A` | ✓ | PHASE_SWAPPED:5 | corrected_60deg_flip:4, correct:1 | flip_hurt:4, no_flip:1 |
| `38_B` | ✓ | GOOD:5 | corrected_60deg_flip:3, correct:2 | flip_helped:3, no_flip:2 |
| `40_A` | ✓ | GOOD:5 | ambiguous_no_correction:4, corrected_60deg_flip:1 | no_flip:4, flip_helped:1 |
| `40_B` | ✓ | PHASE_SWAPPED:5 | corrected_60deg_flip:2, ambiguous_no_correction:2, correct:1 | no_flip:3, flip_hurt:2 |
| `41_A` | **✗ post-varies** | GOOD:4, MARGINAL:1 | correct:3, corrected_60deg_flip:2 | no_flip:3, flip_helped:2 |
| `41_B` | ✓ | GOOD:5 | ambiguous_no_correction:2, corrected_60deg_flip:2, correct:1 | no_flip:3, flip_helped:2 |
| `43_A` | ✓ | PHASE_SWAPPED:5 | correct:3, corrected_60deg_flip:2 | no_flip:3, flip_hurt:2 |
| `43_B` | **✗ post-varies** | PHASE_SWAPPED:4, GOOD:1 | corrected_60deg_flip:3, ambiguous_no_correction:2 | flip_hurt:3, no_flip:2 |
| `45_A` | **✗ post-varies** | PHASE_SWAPPED:3, GOOD:1, MARGINAL:1 | ambiguous_no_correction:3, corrected_60deg_flip:2 | no_flip:3, flip_hurt:1, flip_helped:1 |
| `45_B` | **✗ post-varies** | MARGINAL:3, GOOD:2 | corrected_60deg_flip:3, correct:2 | flip_helped:3, no_flip:2 |

## Q4: Which evidence would have selected the right parity?

Each row contributes its MEDIAN `phase_darkness_separation` across runs. Grouped by row's modal `post_canonical_category`.

| Post canonical (modal) | n rows | sep median | sep min | sep max |
|---|---:|---:|---:|---:|
| `GOOD` | 4 | -0.2 | -30.9 | 17.4 |
| `MARGINAL` | 1 | 29.4 | 29.4 | 29.4 |
| `PHASE_SWAPPED` | 7 | 9.1 | -42.6 | 61.7 |

## Findings & implications

1. **Phase-rewound = PHASE_SWAPPED on 6/12 rows (modal).** After mathematically inverting any flip from the post-pipeline output, these rows still land in PHASE_SWAPPED. This is an approximate lower bound on "how many rows the upstream correspondence + vertex-refinement landed in the wrong parity" — but see the phase-rewound caveat above; the true upstream-correspondence error rate would require a clean apply_phase_correction=False pipeline run. (8/12 rows are post-category-stable across runs; 4 rows had run-to-run disagreement on the final outcome).

2. **Phase-correction's impact on canonical score (modal):** helped 2 row(s), hurt 3 row(s), no flip applied on 7 row(s). If `flip_hurt > flip_helped`, the detector is on net actively creating PHASE_SWAPPED outcomes that the initial correspondence got right. (8/12 rows are post-category-stable across runs; 4 rows had run-to-run disagreement on the final outcome).

3. **End-state PHASE_SWAPPED count (modal): 7/12.** Compare to phase-rewound count above — if they're similar, phase-correction isn't making net progress.

## Next-step recommendations

Based on the row-level evidence above, the fix surface is scoped per failure mode:

- **Rows where pre=PHASE_SWAPPED, post=PHASE_SWAPPED, phase_check=`correct` or `ambiguous_no_correction`**: the correspondence picked FAR and the detector did not catch it. Fix surface: either (a) make correspondence pick ONE_EDGE more reliably, or (b) strengthen the detector to catch this subset. Look at `phase_darkness_separation` distribution for this subset to scope (b).
- **Rows where pre=PHASE_SWAPPED, post=GOOD/MARGINAL, phase_check=`corrected_60deg_flip`**: the detector correctly caught and corrected an upstream FAR pick. This is the detector working as intended — preserve.
- **Rows where pre=GOOD/MARGINAL, post=PHASE_SWAPPED, phase_check=`corrected_60deg_flip`**: the detector flipped a correct fit into a wrong one. This is the inverted-polarity wrong-call mode from PR #250's diagnostic. Fix surface: gate the detector's polarity rule on a meta-signal that predicts when its assumption holds (PR #250 suggested `junction_score_at_ensemble`, but its categorization was provisional — re-evaluate under canonical truth here).

Candidate fix paths from Codex's #250 review (in priority order, with the color-anchor caveat: do not use sticker colors sampled from already-wrong geometry as hard truth):

1. **Carry both phase hypotheses forward and score**: instead of mutating one model post-hoc, run the correspondence with both phase parities, score each against orthogonal evidence (center-color consistency, white-up/yellow-up A/B convention, two-view A↔B flip constraint), pick the better-scoring one. Avoids the polarity-rule inversion risk entirely.
2. **Center-color consistency check**: even if interior stickers are sampled from wrong geometry, the visible-face CENTER stickers should still hit the same color across phase hypotheses for the correct one. Use this as a tie-breaker, not a primary signal.
3. **Two-view A↔B flip constraint**: if A and B fits both succeed under the documented 180°-camera-X flip convention, the relative parity between the two views is constrained. (Requires the canonicalization helper from #249.)
