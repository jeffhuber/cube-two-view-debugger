# Split-cubie consistency diagnostic (Phase 1)

Diagnostic-only. Phase 1 of the "split-cubie consistency" lever
discussed in `COORDINATION.md`. Asks whether the cross-image
cubie-validity constraint catches recognition failures earlier or
more locally than the existing legal-repair layer.

> ⚠ **Verdict update (post-review)**: an earlier draft of this
> report concluded "yield zero, don't build Phase 2" based on the
> 20-row fresh corpus alone. That was wrong — sample of N=2 cc-failures
> drew an over-strong conclusion. Re-running against the 46-pair
> labeled corpus surfaced 3/3 cc-failure rows with split-cubie
> inconsistency. Combined corpus is 4/5 (80%) caught by whole-cube
> per-cubie consistency, 3/5 (60%) caught by split-cubie specifically.
> Revised verdict + state-delta gate finding below.

## What's a split cubie?

Photo A captures U+R+F faces. Photo B (after a 180° flip) captures
D+L+B faces. Of the 26 cube pieces, **12 have stickers in BOTH
photos**:

- **6 split corners:** UFL, ULB, UBR, DFR, DLF, DRB
- **6 split edges:** UB, UL, FL, BR, DF, DR

(The remaining 2 corners — URF and DBL — are fully in one image.
Same for the remaining 6 edges.)

For each split cubie, the observed colors on its stickers must
form a real cubie face-set (one of the 8 valid corner triples or
12 valid edge pairs). If not, at least one sticker is misclassified.

## The corrected empirical headline

Combined across both corpora (66 rows total — 46 labeled + 20 fresh):

| Corpus | Rows | cc-failures | Caught by split-cubie | Caught by whole-cube |
|---|---:|---:|---:|---:|
| Labeled (46) | 46 | 3 | **3** (Sets 14, 65, 69) | **3** (same) |
| Fresh (20) | 20 | 2 | 0 | 1 (Set 11) |
| **Combined** | **66** | **5** | **3 (60%)** | **4 (80%)** |

**Whole-cube per-cubie consistency catches 4/5 cc-failure cases on
the current 66-row corpus.** Split-cubie specifically catches 3/5.
The 1 case neither catches (Set 59) is a parity/twist failure
already handled in production by pair-threshold search (#340).

## Per-row evidence

| Set | corpus | cc_ham | inval_cubies | which | bl_ham | reported_changes | **state_delta(cc→bl)** |
|---:|---|---:|---:|---|---:|---:|---:|
| 14 | labeled (DANGER) | 4 | 2 split | UBR, DR | 4 | 5 | **6** |
| 65 | labeled (rescue) | 2 | 2 split | UBR, DRB | 0 | 4 | **2** |
| 69 | labeled (rescue) | 3 | 2 split | UFL, UL | 0 | 1 | **3** |
| 11 | fresh (rescue) | 2 | 2 in-image | URF, DBL | 0 | 6 | **2** |
| 59 | fresh (parity) | 2 | 0 | — | 0 | 5 | 2 |

## The `repairChanges` semantic surprise → state_delta gate finding

The legal-repair helper reports `repairChanges` against RAW
observations (pre-count-repair), not against the count-repaired
baseline. This causes systematic mismatches between
"reported_changes" and "true state_delta from cc → bl":

- Set 65: reported 4, true delta 2 (overcount)
- Set 69: reported 1, true delta 3 (**undercount**)
- Set 11: reported 6, true delta 2 (overcount)
- Set 14: reported 5, true delta 6 (slight undercount, but still
  the largest delta among these 4)

**A gate of `cost <= 20.0 AND state_delta <= 4` cleanly separates
rescue from danger on the current corpus:**

| Row | state_delta | Verdict under new gate |
|---:|---:|---|
| 14 (DANGER) | 6 | rejected (delta > 4) ✓ |
| 65 (rescue) | 2 | admitted ✓ |
| 69 (rescue) | 3 | admitted ✓ |
| 11 (rescue) | 2 | admitted ✓ |

This is a strictly cleaner gate than #345's recommended "bump
`GUARDED_BROAD_MAX_REPAIR_CHANGES` from 4 to 6". The new gate:

1. Measures what it's supposed to measure (divergence of broad-legal
   from the trusted recommended path).
2. Admits Set 11 without any threshold relaxation.
3. Still rejects Set 14 by the same margin (6 vs 4 → rejected by 2).
4. Doesn't depend on the legal-repair helper's accounting quirk.

**Recommendation: replace #345's "bump changes ceiling" with
"switch the gate to state_delta_from_canonical".** Codex owns the
gate code; Codex should evaluate this before landing either tune.

## Verdict on D Phase 2 (revised)

**Still don't build Phase 2 yet, but for a different reason than
the original draft.**

- Original draft (wrong): "empirical yield is zero on the 20 fresh
  rows, so don't build it."
- Corrected (right): empirical yield on the combined 66-row corpus
  is real (4/5 cc-failures have cubie inconsistency). But **the
  state-delta gate already admits all 4 of those rows via the
  existing broad-legal path**. So Phase 2's marginal value over
  "gate-tune alone" is zero on the current corpus.

If the state-delta gate is adopted, Phase 2 (targeted re-classification
guided by per-cubie inconsistency) becomes redundant for these
specific failure modes. Phase 2's architectural value re-emerges
when failure modes appear that the gate alone can't reach (e.g.,
rows where broad-legal can't produce ANY valid cube from raw
observations, but a targeted re-classification would).

If the gate-tune is NOT adopted, Phase 2 would lift recognition
on the 4 cubie-inconsistent rescue rows — but that's a more
expensive build than the gate change.

**If you ever do build Phase 2: build it as whole-cube per-cubie
consistency, not split-cubie specifically.** Whole-cube catches a
strict superset of split-cubie on this corpus (4 vs 3). The split
distinction adds no localization value.

## Reproducer

```bash
# 1. Run legal-repair on the 20 fresh GT rows
.venv/bin/python tools/diagnose_hull_label_legal_repair.py \
  --only-sets 8 9 10 11 13 16 18 19 33 34 35 50 51 52 53 54 55 56 59 60 \
  --out-json /tmp/fresh_legal.json \
  --report /tmp/fresh_legal.md

# 2. Run the split-cubie diagnostic on those results
.venv/bin/python tools/diagnose_split_cubie_consistency.py \
  --input /tmp/fresh_legal.json \
  --out-json tests/fixtures/split_cubie_consistency_summary.json

# 3. Same diagnostic on the labeled corpus (uses the checked-in
#    legal-repair fixture, no fresh compute needed)
.venv/bin/python tools/diagnose_split_cubie_consistency.py \
  --input tests/fixtures/hull_label_legal_repair_diagnostic.json \
  --out-json tests/fixtures/split_cubie_consistency_labeled_corpus.json
```

Both fixtures are checked in. The combined-corpus headline (4/5)
comes from joining their `summary.withSplitCubieInconsistency` +
`summary.withInImageCubieInconsistencyOnly` lists across the two
files.
