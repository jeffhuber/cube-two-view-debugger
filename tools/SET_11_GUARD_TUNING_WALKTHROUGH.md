# Set 11 guard-tuning walkthrough

Diagnostic-only. Surgical analysis of the single remaining
recommended miss from `tools/FRESH_GT_CONSTRAINED_INFERENCE_REPORT.md`
(ctvd#344), with a concrete recommendation for safely tuning the
guarded-broad legal-repair gate.

Source artifacts:
- `tools/FRESH_GT_CONSTRAINED_INFERENCE_REPORT.md` — Codex #344, headline 19/20 exact.
- `tools/HULL_LABEL_LEGAL_REPAIR_DIAGNOSTIC.md` — the 46-pair labeled-corpus scoreboard.
- `tools/PAIR_THRESHOLD_REPAIR_DIAGNOSTIC.md` — pair-level threshold sweep (ctvd#342).
- Set 11 broad-legal probe re-run via `tools/diagnose_pair_threshold_repair.py --only-sets 11`.

## The miss

Set 11 is the only fresh-row recommended miss. Its broad-legal
repair contains the GT-exact state, but the guarded-broad gate
rejects it. Specifics:

| Method | hamming | validState | repair cost | repair changes | gate decision |
|---|---:|---:|---:|---:|---|
| `canonical_count_repaired` | 2 | false | n/a | 11 moves | (recommended winner; invalid) |
| `conservative_legal_repaired` | n/a | false | n/a | n/a | `no_legal_repair` |
| `broad_legal_repaired` | **0** | **true** | **10.15** | **6** | (diagnostic only) |
| `guarded_broad_legal_repaired` | n/a | false | n/a | n/a | **`rejected_guarded_broad_legal_repair`** — changes 6 > 4 |

So the GT-exact answer is reachable by deterministic repair —
the guard just won't let it through. The gate is currently
`cost <= 20.0 AND changes <= 4`. Set 11 fails the changes ceiling
(6 > 4) despite its very low cost (10.15, well under the 20.0
limit).

## Corpus-wide discriminator analysis

To know whether the gate can be safely tuned, the question is:
**among rows where broad-legal produces a valid state, which are
GT-correct (rescue rows) and which are legal-but-wrong (danger
rows)?** A safe tune must admit the rescues while rejecting the
dangers.

### Rescue rows (broad-legal exact, canonical-count invalid)

These are the rows where deterministic repair *can* recover the
GT state but only through the broad path:

| Set | Source | broad cost | broad changes | canonical_count hamming |
|---:|---|---:|---:|---:|
| 65 | labeled corpus | 19.58 | 4 | 2 |
| 69 | labeled corpus |  3.15 | 1 | 3 |
| **11** | **fresh corpus** | **10.15** | **6** | **2** |

### Danger rows (broad-legal valid but NOT exact)

These are the rows where broad-legal produces a legal cube state
that does NOT match GT. Admitting these would surface a
plausible-but-wrong recognition result to the user:

| Set | Source | broad cost | broad changes | broad hamming |
|---:|---|---:|---:|---:|
| 14 | labeled corpus | **26.69** | 5 | 4 |

Set 14 is the **only** legal-but-wrong row across the 46-pair
labeled corpus + the 20 fresh-corpus rows (n=66 total). That's
the entire danger surface for this gate decision.

## The key observation

| | Set 11 (want admit) | Set 14 (must reject) |
|---|---:|---:|
| broad cost | **10.15** | **26.69** |
| broad changes | 6 | **5** |

**`changes` is inverted as a discriminator.** Set 14 (must-reject)
has *fewer* changes than Set 11 (want-admit). Any relaxation of
the changes ceiling that admits Set 11 (≥ 6) would also admit
Set 14 (changes = 5). The current `changes <= 4` ceiling rejects
both, but as a tuning lever it can't distinguish them.

**`cost` discriminates cleanly.** Set 11 cost (10.15) is well
below Set 14 cost (26.69). The current `cost <= 20.0` gate
already rejects Set 14 by cost alone; it would do the same job
without help from the changes ceiling.

## Tuning sweep

| Gate | Rescues admitted | Dangers admitted | Safe? |
|---|---|---|---|
| `cost <= 20.0 AND changes <= 4` (current) | 65, 69 | (none) | ✓ — but misses Set 11 |
| `cost <= 20.0 AND changes <= 6` | 65, 69, **11** | (none) | ✓ — admits Set 11 |
| `cost <= 20.0 AND changes <= 8` | 65, 69, **11** | (none) | ✓ |
| `cost <= 20.0` (no changes ceiling) | 65, 69, **11** | (none) | ✓ |
| `cost <= 15.0` (no changes ceiling) | 69, **11** | (none) | ✓ — but drops Set 65 |
| `cost <= 12.0` (no changes ceiling) | 69, **11** | (none) | ✓ — but drops Set 65 |

The danger column stays empty across all relaxations of the
changes ceiling because Set 14's *cost* (26.69) is the gating
constraint — the changes ceiling never had to discriminate it.

## Recommended tune

**Change `GUARDED_BROAD_MAX_REPAIR_CHANGES` from `4` to `6`.**
Leave `GUARDED_BROAD_MAX_REPAIR_COST` at `20.0` unchanged.

```python
# tools/hull_label_color_repair.py
# tools/diagnose_hull_label_legal_repair.py
GUARDED_BROAD_MAX_REPAIR_COST = 20.0
GUARDED_BROAD_MAX_REPAIR_CHANGES = 6  # was 4 — see SET_11_GUARD_TUNING_WALKTHROUGH.md
```

Expected impact on the current corpus (46 labeled + 20 fresh = 66 rows):

- **Admits Set 11**: 19/20 → 20/20 exact on the fresh corpus.
- **No regressions**: zero legal-but-wrong rows fall in the
  `4 < changes <= 6` window, so the gate's "no false positives"
  property is preserved.
- **Other admits unchanged**: Sets 65, 69 continue to admit
  (they're under both the old and new ceilings); Set 14 continues
  to reject (its cost 26.69 still exceeds the 20.0 cost gate).
- **Reaches headline 100% exact** across both the labeled corpus
  (was 46/46, stays 46/46) and the fresh corpus (was 19/20,
  becomes 20/20).

This mirrors the discipline of the original `cost <= 16 → 20`
tune in #335 (admitted Set 65). Same shape: identify the
specific boundary row, verify no danger rows fall in the
admitted window, then ship a focused threshold change.

## A more aggressive alternative

If you prefer a bigger architectural move: **drop the
`GUARDED_BROAD_MAX_REPAIR_CHANGES` ceiling entirely** (set to
`math.inf` or simply remove the check) and rely on cost alone.
The corpus analysis above shows this is also safe today. Pro: simpler gate, one fewer hyperparameter to tune. Con: if a
future bad row has cost < 20 but absurd changes (e.g. 30+), it'd
slip through. Keeping a sane upper bound (6, or even 8) preserves
a safety net without costing any current rescue rows.

I'd recommend the conservative `changes <= 6` tune as the next
step, with the option to relax further (or drop the ceiling) once
more corpus data accumulates.

## Caveats

- **n=66 is small.** Three rescue rows and one danger row is a
  thin distribution to tune against. The recommendation is the
  right *direction*; the specific threshold value should re-tune
  as the corpus grows.
- **The "danger" definition assumes GT is correct.** If labeling
  errors exist in the fresh GT for Set 11, the analysis is wrong.
  Worth a visual spot-check of Set 11's images vs the GT JSON.
- **The cost-axis discriminator is empirical, not theoretical.**
  Cost reflects classifier-confidence-weighted Lab distance for
  the changed stickers. It's a "how confident is the change?"
  signal. Set 14's cost (26.69) being so far above the rescue
  rows is the data we get to lean on; it's not guaranteed by
  the architecture that this gap holds on novel corpora.
- **Out of scope here**: the actual code change. This walkthrough
  is the *analysis* establishing that the tune is safe.
  Production change is a separate PR (lift pattern: same as
  #335).

## Reproducer

```bash
# 1. Get Set 11's broad-legal repair details
.venv/bin/python tools/diagnose_pair_threshold_repair.py \
  --only-sets 11 \
  --out-json /tmp/set11_probe.json \
  --report /tmp/set11_probe.md

# 2. Read /tmp/set11_probe.json → rows[0].pairSelected.summary.broadLegal
#    Shows: { hamming: 0, repairCost: 10.1513, repairChanges: 6, validState: true }

# 3. Cross-reference against the labeled corpus scoreboard
.venv/bin/python -c "
import json
p = json.loads(open('tests/fixtures/hull_label_legal_repair_diagnostic.json').read())
for r in p['rows']:
    m = r['methods'].get('broad_legal_repaired', {})
    if m.get('validState') and m.get('hamming', 99) > 0:
        print('DANGER:', r['setId'], m.get('repairCost'), m.get('repairChanges'), 'hamming=', m.get('hamming'))
"
# → Only Set 14 prints. (cost 26.69, changes 5, hamming 4)
```
