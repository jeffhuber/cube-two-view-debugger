# Pipeline phase-parity failure modes

Diagnostic-only. Walks each human-validated full-corner row through the recognizer pipeline and answers Codex's 4 questions (see `tools/diagnose_pipeline_phase_parity.py` docstring).

Rows: 12 traced / 12 total

## Aggregate

**Post-pipeline canonical category (production outcome):**

| Category | Rows |
|---|---:|
| `PHASE_SWAPPED` | 7 |
| `GOOD` | 4 |
| `GEOMETRY_FAIL` | 1 |

**Pre-correction canonical category (initial correspondence output):**

| Category | Rows |
|---|---:|
| `GOOD` | 8 |
| `PHASE_SWAPPED` | 4 |

**`phase_check` distribution:**

| phase_check | Rows |
|---|---:|
| `corrected_60deg_flip` | 6 |
| `correct` | 5 |
| `ambiguous_no_correction` | 1 |

**Score-delta classification (did the flip improve the canonical score?):**

| Class | Rows | Meaning |
|---|---:|---|
| `no_flip` | 6 | phase-correction made no change (pre == post) |
| `flip_hurt` | 5 | flip moved category toward worse (e.g. GOOD → PHASE_SWAPPED) |
| `flip_helped` | 1 | flip moved category toward better (e.g. PHASE_SWAPPED → GOOD) |

## Per-row trace

| Key | Q1 pre→ (correspondence) | Q2 phase_check | Q3 delta | post final |
|---|---|---|---|---|
| `20_A` | ONE_EDGE ✓ (`GOOD`) | `corrected_60deg_flip` | `flip_hurt` | `PHASE_SWAPPED` |
| `20_B` | FAR ✗ (`PHASE_SWAPPED`) | `correct` | `no_flip` | `PHASE_SWAPPED` |
| `38_A` | ONE_EDGE ✓ (`GOOD`) | `corrected_60deg_flip` | `flip_hurt` | `PHASE_SWAPPED` |
| `38_B` | ONE_EDGE ✓ (`GOOD`) | `correct` | `no_flip` | `GOOD` |
| `40_A` | ONE_EDGE ✓ (`GOOD`) | `ambiguous_no_correction` | `no_flip` | `GOOD` |
| `40_B` | FAR ✗ (`PHASE_SWAPPED`) | `correct` | `no_flip` | `PHASE_SWAPPED` |
| `41_A` | FAR ✗ (`PHASE_SWAPPED`) | `corrected_60deg_flip` | `flip_helped` | `GOOD` |
| `41_B` | ONE_EDGE ✓ (`GOOD`) | `corrected_60deg_flip` | `flip_hurt` | `GEOMETRY_FAIL` |
| `43_A` | FAR ✗ (`PHASE_SWAPPED`) | `correct` | `no_flip` | `PHASE_SWAPPED` |
| `43_B` | ONE_EDGE ✓ (`GOOD`) | `corrected_60deg_flip` | `flip_hurt` | `PHASE_SWAPPED` |
| `45_A` | ONE_EDGE ✓ (`GOOD`) | `corrected_60deg_flip` | `flip_hurt` | `PHASE_SWAPPED` |
| `45_B` | ONE_EDGE ✓ (`GOOD`) | `correct` | `no_flip` | `GOOD` |

## Q4: Which evidence would have selected the right parity?

Comparing `phase_darkness_separation` across rows by canonical outcome (post-pipeline):

| Post canonical | n | sep median | sep min | sep max |
|---|---:|---:|---:|---:|
| `GEOMETRY_FAIL` | 1 | 11.4 | 11.4 | 11.4 |
| `GOOD` | 4 | -10.0 | -22.3 | 31.0 |
| `PHASE_SWAPPED` | 7 | 15.0 | -48.2 | 70.6 |

## Findings & implications

1. **Initial correspondence picks FAR (PHASE_SWAPPED) on 4/12 rows** — i.e., the Procrustes/template fit assigns axis_x/y/z to far-corner positions before `_resolve_near_far_phase` is even called. This is the upstream bug Codex flagged: "the whole correspondence + phase-correction pathway," not just the detector.

2. **Phase-correction's impact on canonical score:** helped 1 row(s), hurt 5 row(s), no category change on 0 row(s), no flip applied on 6 row(s). If `flip_hurt > 0`, the detector is actively creating PHASE_SWAPPED outcomes that the initial correspondence got right.

3. **End-state (post-pipeline) PHASE_SWAPPED count: 7/12.** Compare to pre count above — if they're similar, phase-correction isn't making net progress.

## Next-step recommendations

Based on the row-level evidence above, the fix surface is scoped per failure mode:

- **Rows where pre=PHASE_SWAPPED, post=PHASE_SWAPPED, phase_check=`correct` or `ambiguous_no_correction`**: the correspondence picked FAR and the detector did not catch it. Fix surface: either (a) make correspondence pick ONE_EDGE more reliably, or (b) strengthen the detector to catch this subset. Look at `phase_darkness_separation` distribution for this subset to scope (b).
- **Rows where pre=PHASE_SWAPPED, post=GOOD/MARGINAL, phase_check=`corrected_60deg_flip`**: the detector correctly caught and corrected an upstream FAR pick. This is the detector working as intended — preserve.
- **Rows where pre=GOOD/MARGINAL, post=PHASE_SWAPPED, phase_check=`corrected_60deg_flip`**: the detector flipped a correct fit into a wrong one. This is the inverted-polarity wrong-call mode from PR #250's diagnostic. Fix surface: gate the detector's polarity rule on a meta-signal that predicts when its assumption holds (PR #250 suggested `junction_score_at_ensemble`, but its categorization was provisional — re-evaluate under canonical truth here).

Candidate fix paths from Codex's #250 review (in priority order, with the color-anchor caveat: do not use sticker colors sampled from already-wrong geometry as hard truth):

1. **Carry both phase hypotheses forward and score**: instead of mutating one model post-hoc, run the correspondence with both phase parities, score each against orthogonal evidence (center-color consistency, white-up/yellow-up A/B convention, two-view A↔B flip constraint), pick the better-scoring one. Avoids the polarity-rule inversion risk entirely.
2. **Center-color consistency check**: even if interior stickers are sampled from wrong geometry, the visible-face CENTER stickers should still hit the same color across phase hypotheses for the correct one. Use this as a tie-breaker, not a primary signal.
3. **Two-view A↔B flip constraint**: if A and B fits both succeed under the documented 180°-camera-X flip convention, the relative parity between the two views is constrained. (Requires the canonicalization helper from #249.)
