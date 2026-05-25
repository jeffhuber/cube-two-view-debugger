# Hull-Label Acceptance Gates

This is the production-shaped gate contract for a future feature-flagged
`rectify_via_hull_labels` path. It does not wire production behavior; it defines
when the hull-label candidate is trusted and when callers must fall back to the
existing Procrustes/PnP path.

Reference implementation: `tools/hull_label_acceptance.py`.

## Inputs

A caller can evaluate these gates with production-available signals only:

- `side`: capture side (`A` or `B` today)
- `hexagon_corner_count`: result count from `detect_hexagon_anchors(mask)`
- `vertex_estimates`: the three parallelogram-completion vertex estimates
- `rectified_face_slots`: slots produced by rectification
- `sticker_score_total`: canonical CIELAB distance over 27 sampled sticker centers
- `sticker_score_per_face`: canonical CIELAB distance per rectified face

Do not use ground-truth-only metrics such as `axis_total_misfit_deg` in
production acceptance. They remain useful for corpus reports and threshold
tuning, but a live recognizer cannot rely on them.

## Hard Fallback Gates

Fallback to the existing path if any hard gate fails:

| Gate | Hard fallback threshold | Why |
|---|---:|---|
| Side convention | `side` must have both `SILHOUETTE_TO_CORNER` and `FACE_DEFS_BY_SIDE` entries | Prevents silently applying A/B geometry to unsupported capture conventions |
| Hexagon | exactly 6 hull corners | Hull-labels assumes one labeled point for each silhouette extremum |
| Vertex estimates | exactly 3 estimates | One estimate per visible face is required for cloud-spread confidence |
| Face slots | exactly `upper`, `right`, `front` | Current downstream stitching expects these three visible faces |
| Per-face scores | score present for all three slots | Missing scores mean the color/rectification gate is blind |
| Vertex cloud spread | `<= 300 px` | Above the observed 70-row corpus max (267.6 px), so treat as out-of-distribution |
| Sticker score total | `<= 900` | Above the observed 70-row corpus max (730.3), below the old off-cube diagnostic threshold (1500) |
| Worst face score | `<= 450` | Above the observed worst face score (414.7), catches one badly sampled face even if total score is diluted |

The feature flag should return the legacy fit on any hard failure and record the
failure reasons.

## Warning Gates

Warning gates do not force fallback by themselves. They are a shadow-mode
signal for rows near the edge of current evidence:

| Gate | Warning threshold |
|---|---:|
| Vertex cloud spread | `> 240 px` |
| Sticker score total | `> 700` |
| Worst face score | `> 350` |

For initial rollout, use warnings to populate logs/traces and visual review
queues. If a product-facing flag needs to be extra conservative, it can choose
to fallback on warnings too, but the default helper only treats them as
warnings.

On the committed 70-row trace, the default hard gates accept all 70 rows and
emit warnings on 9 rows. This is intentional: the hard gates are
out-of-distribution/fallback gates, not a ground-truth accuracy oracle.

## What These Gates Do Not Catch Yet

The 70-row corpus report found two borderline rows (`30_A`, `37_B`) whose
ground-truth axis misfit is just above the diagnostic threshold. Production
cannot directly compute that metric. Their available production metrics are not
catastrophic:

- no mask failure
- valid 6-corner hull
- plausible vertex-cloud spread
- plausible sticker scores

With the default warning thresholds, `37_B` is warning-level because its
vertex-cloud spread is high. `30_A` is not warning-level from the current
production-available signals, which is exactly why pair-level and shadow-mode
validation remain required before default-on rollout.

So the gate contract is intentionally honest: these gates catch structural
failures and out-of-distribution rows, but they do not prove geometric perfection
on every row. Before making hull-labels default-on, add at least one of:

- pair-level A/B consistency checks
- old-path vs hull-path disagreement diagnostics
- visual gallery review for warning rows
- larger-corpus validation with more yaw/perspective variation

## Feature-Flag Rollout Recommendation

1. **Shadow mode:** run hull-labels and the legacy path; never choose
   hull-labels. Log `HullLabelGateDecision` plus old/new fit summaries.
2. **Candidate mode:** choose hull-labels only when hard gates pass; fallback
   on hard failures. Keep warnings in telemetry.
3. **Strict candidate mode:** choose hull-labels only when hard gates pass and
   warnings are empty. Use this for sensitive demos or early public testing.
4. **Default-on:** only after larger-corpus and A/B pair-level validation show
   warning rows are safe or the warning thresholds have been retuned.

## Fallback Result Contract

When the feature flag falls back, return enough metadata to explain why:

```json
{
  "fit_source": "legacy_procrustes",
  "hull_label_attempted": true,
  "hull_label_accepted": false,
  "hull_label_hard_failures": ["..."],
  "hull_label_warnings": ["..."]
}
```

When it accepts:

```json
{
  "fit_source": "hull_labels",
  "hull_label_attempted": true,
  "hull_label_accepted": true,
  "hull_label_hard_failures": [],
  "hull_label_warnings": ["..."]
}
```

Warnings should remain visible even on accepted rows.
