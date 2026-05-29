# Constrained recognizer latency plan

Updated: 2026-05-29

## Goal

Improve hosted CubeSnap solve latency without trading away recognition quality. The
constraint is important: the constrained recognizer is currently the product path
because it is exact on the local corpus, while the legacy recognizer still rejects
or misreads many hard cases.

## Current bottlenecks

The deployed scoreboard from 2026-05-29 shows `prepareConstrainedInput` as the
dominant stage:

- `prepareConstrainedInputMs`: p50 2360.63 ms, p90 3278.48 ms
- `rembgAMs`: p50 866.61 ms, p90 1315.98 ms
- `rembgBMs`: p50 699.28 ms, p90 1103.59 ms
- `hullFitAMs`: p50 203.72 ms, p90 276.76 ms
- `hullFitBMs`: p50 208.78 ms, p90 287.71 ms
- `selectGuardedPairMs`: p50 32.63 ms, p90 34.30 ms, max 4360.40 ms

This makes the practical strategy:

1. Keep rembg serial for now. Local testing showed concurrent rembg calls against
   one session contend badly and increase wall-clock latency.
2. Parallelize independent deterministic work after rembg. A-side and B-side
   hull-threshold fitting are independent once the alpha masks exist.
3. Keep measuring deployed stage timings after every change. Local timings are
   useful for direction, but Railway CPU behavior is the launch truth.
4. Attack remaining tails in order: rembg cost first, then rare
   `selectGuardedPair` outliers, then client upload/image-size overhead.

## This PR

This PR parallelizes A/B hull-threshold fitting after serial rembg. It preserves
the same inputs, thresholds, fit selection, pair evaluation, and production
output. It also adds `hullFitWall` to stage timings so deployed scoring can show
whether the parallel section is saving real wall-clock time.

Local Set 41 timing, after warmup:

- Before: median `prepareTotal` about 745 ms
- After: median `prepareTotal` about 668 ms

Validation:

- Focused tests: 50 passed
- Set 41 constrained validation: 1/1 exact
- Full constrained validation: 71/71 exact, 71/71 within 3, 0 regressions

Post-deploy Railway score, after merging #396:

- Full deployed corpus: 71/71 exact, 71/71 within 3, 0 rejected
- `prepareConstrainedInputMs`: p50 2058.88 ms, p90 2220.45 ms
- `hullFitWall`: p50 743.94 ms, p90 807.27 ms
- `selectGuardedPairMs`: p50 192.24 ms, p90 209.49 ms, max 4938.70 ms

Compared with the previous deployed scoreboard, total
`prepareConstrainedInputMs` p50 was lower in this run: 2360.63 ms to 2058.88 ms.
Treat this as a post-deploy health/timing snapshot, not a controlled proof of
the parallelization benefit, because rembg timings also shifted materially
between runs. The next measurement step is a same-instance A/B of serial vs.
parallel hull fitting. The remaining observed latency problem is concentrated
in guarded-pair tail cases, especially sets 11, 14, 59, 65, and 69.

## Next candidates

- Controlled benchmark: use
  `tools/benchmark_constrained_recognizer.py --only-sets 11 14 59 65 69`
  to compare serial/threaded hull fitting and guarded-pair tails in one local
  process before interpreting deploy-to-deploy timing deltas. Add
  `--max-sides 1600 1200` to compare rembg pre-resize policies on the same
  code path.
- Rembg cost: evaluate smaller `max_side` or pre-rembg resize policies against
  the full corpus, because rembg dominates p50 and p90.
- 2026-05-29 resize check: 1200/1400/1500 were rejected as production
  defaults because they misread Set 11 by 15 stickers. 1560 looked promising on
  the tail-set smoke, but the benefit is too fiddly relative to launch risk.
  Keep production at 1600 unless a future model/segmentation change gives a
  broad, full-corpus win.
- Operational feedback: use
  `/api/recognition-events/report?sinceHours=24&recentLimit=20` for a
  metadata-only production summary of success/reject rates, latency, source,
  failure reason, and recommended method without needing SSH access to the
  Railway volume. Use `tools/report_recognition_events.py --endpoint ...` for
  daily/adhoc Markdown snapshots.
- Validation traffic: deployed scorer requests send `clientSource` metadata, so
  operational reports can separate Codex score/smoke runs from CubeSnap user
  traffic.
- Wall-clock overhead: deployed scoreboards now include client read/multipart,
  HTTP-to-headers, response read, JSON parse, and `clientWallOverheadMs`
  distributions alongside server recognizer stage timings.
- Rembg replacement path: test whether a deterministic cube-silhouette proposer
  can bypass rembg on easy cases while falling back to rembg on uncertainty.
- Guarded-pair tail: the repair DP now prunes each bucket incrementally instead
  of after bucket growth. This preserves the same per-key top-solution limit
  while reducing repair-heavy pair-search CPU. Keep watching
  `selectGuardedPair` max in deployed scoreboards.
- Frontend input size: confirm CubeSnap is not uploading larger-than-needed
  images before the backend resizes them.
