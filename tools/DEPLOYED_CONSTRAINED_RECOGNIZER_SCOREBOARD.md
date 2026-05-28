# Deployed constrained recognizer scoreboard

Generated: `2026-05-28T17:09:54.376725+00:00`
Endpoint: `https://ctvd-recognizer-production.up.railway.app/api/recognize?slim=1&hullLabelTier1=constrained`
Manifest: `/Users/jhuber/cube-two-view-debugger/.worktrees/codex-two-view-consistency-production/tests/fixtures/corpus_manifest.json`

## Run Notes

- Initial full pass used concurrency=2 and timeout=90s: 53/71 exact, 18 request exceptions.
- The user lost internet during the late rows; Sets 70-78 failed with local DNS errors in the initial pass.
- Serial recovery reruns used timeout=180s for the failed set list and timeout=360s for Set 67.
- After recovery reruns, every scored row returned an exact 54/54 state; the remaining issue is serving latency, not recognizer accuracy on this corpus.
- `twoViewStatusCounts` is `none` because the live Railway endpoint did not yet include this PR's compact two-view signal. The recommended method still shows one row selected `two_view_consistency_repaired`.

## Summary

| Rows | Scored | Exact | Within 3 | Rejected | Missing local files |
|---:|---:|---:|---:|---:|---:|
| 71 | 71 | 71 | 71 | 0 | 0 |

Recommended methods:

- `canonical_count_repaired`: `67`
- `conservative_legal_repaired`: `2`
- `guarded_broad_legal_repaired`: `1`
- `two_view_consistency_repaired`: `1`

Two-view repair statuses:

- `none`: `71`

## Non-exact Rows

_None._
