# Split-cubie consistency diagnostic (Phase 1)

Diagnostic-only. Phase 1 of the "split-cubie consistency" lever
discussed in `COORDINATION.md`. Asks whether the cross-image
cubie-validity constraint catches recognition failures earlier or
more locally than the existing legal-repair layer.

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

## Phase 1 question

> When `canonical_count_repaired` fails to reach the GT state, is
> the failure visible in split-cubie inconsistency? Could a
> targeted re-classification step (Phase 2 of D) recover the row?

## Headline finding

On the 20 fresh-GT rows from #343 / #344:

| Failure mode | Rows |
|---|---|
| `canonical_count_repaired` already exact (no failure) | 18 / 20 |
| `cc` fails with split-cubie inconsistency | **0 / 20** |
| `cc` fails with in-image-only cubie inconsistency | 1 / 20 (Set 11) |
| `cc` fails with NO cubie inconsistency (parity/twist) | 1 / 20 (Set 59) |

**Split-cubie consistency would catch 0 of the 2 fresh-corpus
failures.** The lever exists as an architectural primitive, but its
empirical yield on the current corpus is zero.

## Set 11 — in-image cubie inconsistency

Recovered by broad-legal as the GT-exact state. Per-cubie check on
`canonical_count_repaired`:

- ✗ **URF** corner (in-image, all in photo A): observed `(U, R, D)`,
  not a valid corner triple. The D should be F.
- ✗ **DBL** corner (in-image, all in photo B): observed `(B, F, R)`,
  not a valid corner triple. The F should be D, etc.
- ✓ All 6 split corners, all 6 split edges, and all 6 in-image
  edges are consistent.

So Set 11's two failed cubies are both **non-split**. The split-cubie
diagnostic, by construction, only checks the 12 split cubies — it
doesn't see URF or DBL. **Whole-cube cubie consistency catches Set 11;
split-cubie consistency does not.**

Whole-cube cubie validity is a strict superset of split-cubie validity.
If we're considering Phase 2, the right primitive is **whole-cube
per-cubie consistency**, not split-cubie consistency. The "split"
distinction adds nothing here.

## Set 59 — parity/twist failure

`canonical_count_repaired` hamming = 2, **but every cubie is
internally consistent.** The two mismatched stickers swap with each
other in a way that keeps each cubie's colorset valid; the cube
state is invalid for parity-or-orientation reasons, not for
local-cubie reasons.

Set 59 is recovered in production today by the **pair-threshold
search** (#340) — picking different A/B mask thresholds produces
rectified samples where the count-repair finds the GT-exact state
without invoking legal repair. So Set 59 isn't a residual gap; it's
proof that cross-image *evidence* (alternative threshold pairs) is
already handling this failure mode in production. No additional
constraint layer needed.

## The `repairChanges` semantic surprise

While running this diagnostic I hit a side-finding worth recording.

Set 11's broad-legal repair reports `repairChanges = 6`. But the
**true state delta** from `canonical_count_repaired` → `broad_legal_repaired`
is **2 stickers** (indices 20 and 53, i.e. F[0,2] and B[2,2]).

The 6 vs 2 gap comes from how the legal-repair helper accumulates
`changes`. It sums per-piece changes against raw observations
(before count repair), not against the count-repaired baseline.
For URF (3 stickers), if the legal helper picks a different
corner color triple than the raw observation, every sticker in
that corner counts as a "change" — even if 2 of those 3 happened
to already match the count-repaired state.

**Implication for #345's guard-tune recommendation:** the "right"
gate semantic is arguably `state_delta_from_canonical_count <= N`
(directly the hamming distance between recommended and proposed
states) rather than `repairChanges <= N` (which measures the
algorithm's accounting). For Set 11:

| Measure | Value | Current gate result |
|---|---:|---|
| `repairChanges` (vs raw observation) | 6 | rejected (6 > 4) |
| `state_delta` (vs `canonical_count_repaired`) | 2 | would be accepted |

A gate of `cost <= 20.0 AND state_delta_from_canonical <= 4` would
admit Set 11 *without* changing the changes-against-observations
ceiling. It also has a cleaner semantic: "broad legal can change
at most N stickers from the recommended count-repair baseline."

This may be a better tune than #345's "bump
`GUARDED_BROAD_MAX_REPAIR_CHANGES` from 4 to 6" recommendation —
or a complementary one. Worth Codex's input before either lands.

## Verdict on D Phase 2

**Don't build Phase 2 yet.** The empirical yield on the current
corpus is zero new cases caught by split-cubie consistency
specifically. The 1 case it could catch (Set 11) is in non-split
cubies, addressable by the existing guard-tune work in #345 (or
the alternative gate semantic suggested above). The 1 case that
genuinely needs cross-image evidence (Set 59) is already handled
by the pair-threshold search in production (#340).

The architecture is sound — the 12 split cubies are real,
cross-view color agreement is a real constraint, and a Phase 2
implementation would be tractable. But there's no current
empirical signal that the cost of building it would be repaid.
Revisit once the corpus has more diverse failure modes (e.g.,
post-graduation traffic on the default `/api/recognize`).

## What's left if Phase 2 ever does become valuable

Phase 2 would add `cubie_consistency_repaired` as a tier between
`conservative_legal_repaired` and `guarded_broad_legal_repaired`
in the `choose_recommended_method` preference list:

1. For each cubie that fails consistency, identify the lowest-confidence
   sticker as the suspect.
2. Re-classify that sticker using a tighter palette / alternative
   classifier.
3. Accept the re-classified state only if it produces a valid cube
   AND each repaired cubie is now consistent.

The split-cubie variant (vs whole-cube cubie consistency) would
only differ in which 12 of the 20 cubies are checked — likely not
worth the special-casing. **If we build it, build it as whole-cube
per-cubie consistency**, not split-cubie specifically.

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
```

Or, for Set 11 alone, replace `--only-sets ... 60` with just
`--only-sets 11` in step 1.
