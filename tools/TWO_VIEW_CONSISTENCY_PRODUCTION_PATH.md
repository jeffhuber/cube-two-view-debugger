# Two-view consistency production signal

Status: production-path observability.

The constrained `/api/recognize?hullLabelTier1=constrained` path already
evaluates `two_view_consistency_repaired` inside the deterministic color repair
payload. That method promotes a legal repair only when:

- the count-repaired baseline has split-cubie inconsistency;
- the legal candidate clears cubie inconsistency;
- the repair stays under the same cost/state-delta bounds as guarded broad
  repair.

This signal is now exposed in the compact `recognitionSignals` block returned
to CubeSnap:

```json
{
  "constrainedInference": {
    "twoViewConsistencyRepair": {
      "status": "accepted_two_view_consistency_repair",
      "gate": {
        "accepted": true,
        "reasons": [],
        "baselineCubieConsistency": {
          "inconsistentSplitCount": 1,
          "inconsistentNames": ["UFL"]
        },
        "candidateCubieConsistency": {
          "inconsistentCount": 0
        }
      }
    }
  }
}
```

Why this matters:

- User-facing CubeSnap debug exports now show whether the two-view lever helped
  or was rejected, without returning the heavy per-sticker repair payload.
- Remaining failures can be triaged by gate reason:
  `no_split_cubie_inconsistency`, `candidate_cubies_still_inconsistent`,
  `state_delta_out_of_range`, or `repair_cost_out_of_range`.
- This keeps the lever observable in the public/default Solve path while the
  next recovery step remains scoped: targeted reclassification only when the
  gate localizes a split-cubie inconsistency.
