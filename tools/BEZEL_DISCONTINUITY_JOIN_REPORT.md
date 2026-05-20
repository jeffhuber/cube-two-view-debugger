# Bezel + Discontinuity Join Probe

Diagnostics-only slot/cell mining over the same hybrid overlay quads that received human visual feedback in #175.
The candidate guard shape under inspection is `crosses_high_quality_bezel AND discontinuity_flag`; neither signal alone is proposed for behavior wiring.

## Summary

- Cells scored: 270 / 270
- Slots scored: 30 / 30
- Human-bad cells: 252 (slot label repeated per cell)
- Human-good cells: 18 (slot label repeated per cell)
- Bezel thresholds: line quality >= 0.4, distance <= 30.0 px
- Discontinuity thresholds: internal std >= 35.0 or half delta >= 45.0

## Cross-Tab Axes

| Axis | Count | Human-bad | Human-good |
|---|---:|---:|---:|
| `both_hit` | 32 | 32 | 0 |
| `bezel_only` | 10 | 10 | 0 |
| `discontinuity_only` | 162 | 156 | 6 |
| `both_miss` | 66 | 54 | 12 |

## Guard-Candidate Readout

- Both-hit cells: 32
- Both-hit human-bad cells: 32
- Both-hit human-good cells: 0
- Both-miss human-bad cells: 54

## Threshold Sweep

| Rank | Line q | Distance px | Bezel-hit cells | Both-hit cells | Both-hit bad | Both-hit good |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3 | 60.0 | 66 | 52 | 52 | 0 |
| 2 | 0.4 | 60.0 | 63 | 49 | 49 | 0 |
| 3 | 0.3 | 40.0 | 52 | 41 | 41 | 0 |
| 4 | 0.4 | 40.0 | 48 | 37 | 37 | 0 |
| 5 | 0.5 | 60.0 | 46 | 36 | 36 | 0 |
| 6 | 0.3 | 30.0 | 43 | 33 | 33 | 0 |
| 7 | 0.6 | 60.0 | 43 | 33 | 33 | 0 |
| 8 | 0.4 | 30.0 | 42 | 32 | 32 | 0 |
| 9 | 0.5 | 40.0 | 33 | 26 | 26 | 0 |
| 10 | 0.3 | 20.0 | 32 | 23 | 23 | 0 |
| 11 | 0.4 | 20.0 | 31 | 22 | 22 | 0 |
| 12 | 0.5 | 30.0 | 28 | 22 | 22 | 0 |

## Both-Hit Human-Good Cells

These are the potential false-positive rows for the conjunction.

_None._

## Both-Hit Human-Bad Cells

These are the rows the conjunction explains.

| Set | Slot | Cell | Axis | Std | Half delta | Best crossing line | Failure modes |
|---:|---|---|---|---:|---:|---|---|
| 17 | `B:B` | 1,1 | `both_hit` | 161.094 | 55.505 | q=0.454, d=13.2px, angle=86.0 | `bad_quad`, `bad_rectified` |
| 47 | `A:F` | 2,0 | `both_hit` | 154.994 | 278.419 | q=0.436, d=17.6px, angle=92.0 | `bad_quad`, `bad_rectified`, `wrong_source_face` |
| 47 | `B:B` | 1,1 | `both_hit` | 148.185 | 35.374 | q=0.607, d=12.4px, angle=152.0 | `bad_rectified`, `wrong_source_face`, `rectification_bad_despite_good_quad` |
| 49 | `A:F` | 2,0 | `both_hit` | 145.264 | 88.738 | q=0.729, d=24.8px, angle=86.0 | `bad_quad`, `bad_rectified` |
| 17 | `B:B` | 2,0 | `both_hit` | 134.945 | 238.736 | q=0.454, d=28.5px, angle=86.0 | `bad_quad`, `bad_rectified` |
| 49 | `A:U` | 1,1 | `both_hit` | 132.417 | 55.562 | q=0.729, d=7.0px, angle=86.0 | `bad_quad`, `bad_rectified` |
| 17 | `B:D` | 1,0 | `both_hit` | 128.056 | 258.635 | q=0.454, d=4.7px, angle=86.0 | `bad_quad`, `bad_rectified` |
| 47 | `A:R` | 1,1 | `both_hit` | 124.123 | 25.393 | q=0.436, d=8.6px, angle=92.0 | `bad_quad`, `wrong_source_face`, `rectification_survives_bad_quad` |
| 49 | `A:U` | 1,2 | `both_hit` | 123.14 | 118.226 | q=0.729, d=20.9px, angle=86.0 | `bad_quad`, `bad_rectified` |
| 17 | `B:D` | 1,2 | `both_hit` | 121.266 | 31.477 | q=0.454, d=6.3px, angle=86.0 | `bad_quad`, `bad_rectified` |
| 47 | `A:U` | 1,0 | `both_hit` | 117.493 | 138.257 | q=0.436, d=18.3px, angle=92.0 | `bad_rectified`, `rectification_bad_despite_good_quad` |
| 47 | `B:B` | 0,1 | `both_hit` | 112.199 | 41.97 | q=0.607, d=14.6px, angle=152.0 | `bad_rectified`, `wrong_source_face`, `rectification_bad_despite_good_quad` |

## Both-Miss Human-Bad Cells

These are known-bad slot cells neither signal explains.

| Set | Slot | Cell | Axis | Std | Half delta | Best crossing line | Failure modes |
|---:|---|---|---|---:|---:|---|---|
| 47 | `A:R` | 0,1 | `both_miss` | 34.894 | 12.619 | q=0.436, d=91.8px, angle=92.0 | `bad_quad`, `wrong_source_face`, `rectification_survives_bad_quad` |
| 17 | `B:D` | 0,0 | `both_miss` | 34.675 | 16.419 |  | `bad_quad`, `bad_rectified` |
| 61 | `A:F` | 0,2 | `both_miss` | 34.312 | 19.465 | q=0.88, d=52.6px, angle=26.0 | `bad_quad`, `bad_rectified`, `wrong_source_face` |
| 21 | `A:F` | 0,0 | `both_miss` | 34.06 | 26.737 |  | `bad_quad`, `bad_rectified`, `wrong_source_face` |
| 17 | `A:F` | 1,2 | `both_miss` | 32.074 | 14.239 | q=0.129, d=48.7px, angle=38.0 | `bad_quad`, `bad_rectified`, `wrong_source_face` |
| 47 | `A:R` | 0,2 | `both_miss` | 31.478 | 17.214 | q=0.075, d=38.4px, angle=152.0 | `bad_quad`, `wrong_source_face`, `rectification_survives_bad_quad` |
| 21 | `A:U` | 1,0 | `both_miss` | 31.263 | 21.011 |  | `bad_rectified`, `rectification_bad_despite_good_quad` |
| 61 | `B:D` | 0,0 | `both_miss` | 30.704 | 3.87 | q=1.0, d=48.3px, angle=92.0 | `bad_quad`, `bad_rectified` |
| 61 | `A:R` | 0,2 | `both_miss` | 30.511 | 26.653 |  | `bad_quad`, `wrong_source_face`, `rectification_survives_bad_quad` |
| 47 | `B:D` | 0,1 | `both_miss` | 30.511 | 15.406 | q=0.092, d=80.7px, angle=96.0 | `bad_quad`, `rectification_survives_bad_quad` |
| 49 | `A:R` | 0,2 | `both_miss` | 29.983 | 36.788 |  | `bad_quad`, `bad_rectified`, `wrong_source_face` |
| 47 | `B:D` | 2,0 | `both_miss` | 28.956 | 27.519 | q=0.228, d=50.6px, angle=24.0 | `bad_quad`, `rectification_survives_bad_quad` |

## Interpretation

- This report is not a production guard and does not change recognizer behavior.
- The human label is slot-level and repeated across that slot's 9 cells; cell counts should be read as diagnostic evidence, not independent labels.
- The canonical quad source is the #175 hybrid overlay quads, not production recognizer quads. Production selected-grid transfer is a later check.
- The only plausible future behavior shape remains a conservative manual-review guard where independent bezel and discontinuity signals agree, after broader zero-FP mining.
