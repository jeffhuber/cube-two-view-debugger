# Hull-Label Tier 1 E2E Bench

Date: 2026-05-25

This report records the first end-to-end cube-snap bench run against the hidden
Hull-Label Tier 1 API path added in CTVD PR #296 and cube-snap PR #162.

## Setup

- CTVD server: `/Users/jhuber/cube-two-view-debugger` at `7c9dfe1`
- cube-snap bench client: `/tmp/cube-snap-tier1` at `ce72976`
- Local API: `http://localhost:8085/api/recognize`
- Corpus: 41 two-image pairs generated from
  `tests/fixtures/corpus_manifest.json` into `/tmp/cube-snap-e2e-corpus`
- Result files:
  - `/tmp/cube-snap-e2e-results/off.json`
  - `/tmp/cube-snap-e2e-results/shadow.json`
  - `/tmp/cube-snap-e2e-results/prefer.json`

The cube-snap `cv-local` bench adapter currently reports `0ms` latency for all
rows, so latency was not evaluated here.

## Commands

```bash
npm run bench -- \
  --corpus /tmp/cube-snap-e2e-corpus \
  --providers cv-local \
  --cv-api http://localhost:8085/api/recognize \
  --concurrency 2 \
  --out /tmp/cube-snap-e2e-results/off.json

npm run bench -- \
  --corpus /tmp/cube-snap-e2e-corpus \
  --providers cv-local \
  --cv-api http://localhost:8085/api/recognize \
  --cv-hull-label-tier1 shadow \
  --concurrency 2 \
  --out /tmp/cube-snap-e2e-results/shadow.json

npm run bench -- \
  --corpus /tmp/cube-snap-e2e-corpus \
  --providers cv-local \
  --cv-api http://localhost:8085/api/recognize \
  --cv-hull-label-tier1 prefer \
  --concurrency 2 \
  --out /tmp/cube-snap-e2e-results/prefer.json
```

## Headline Results

| Mode | Pairs | Mean stickers | Exact 54/54 | Failures |
| --- | ---: | ---: | ---: | ---: |
| off | 41 | 34.5/54 | 19 | 12 |
| shadow | 41 | 34.5/54 | 19 | 12 |
| prefer | 41 | 35.8/54 | 20 | 11 |

Shadow mode was output-identical to the default path for `ok`,
`stickersCorrect`, and `validityError` on all 41 rows.

Prefer mode selected the hull-label candidate on 4/41 rows:

| Pair | Default | Prefer | Delta | Vertex source |
| --- | ---: | ---: | ---: | --- |
| set-32 | 54/54 | 54/54 | 0 | A affine, B affine |
| set-39 | 0/54 retake | 54/54 | +54 | A affine, B affine |
| set-42 | 54/54 | 54/54 | 0 | A affine, B affine |
| set-45 | 54/54 | 54/54 | 0 | A projective, B projective |

No row regressed in `prefer` during this run. The single positive delta was
`set-39`, which moved from "No legal cube state matched the detected stickers"
to a unique legal `54/54` result.

## Gate Behavior

The prefer candidate was rejected and fell back to legacy on 37/41 rows.

Candidate categories:

| Category | Count |
| --- | ---: |
| `reject_retake` | 37 |
| `needs_manual_review` | 4 |

Top rejection reasons:

| Reason | Count |
| --- | ---: |
| Image A did not contain a reliable non-overlapping three-face grid. | 15 |
| Image A must contain the white/U center face. | 10 |
| Image B did not contain a reliable non-overlapping three-face grid. | 7 |
| Image B must contain the yellow/D center face after the flip. | 2 |
| The two flip photos do not expose all four side face centers. | 2 |
| No legal cube state matched the detected stickers. | 1 |

Frequent failed checks:

| Check | Count |
| --- | ---: |
| `missing_side_face_coverage` | 30 |
| `image_b_no_reliable_face_triple` | 20 |
| `image_a_no_reliable_face_triple` | 15 |
| `image_a_U_anchor_missing` | 10 |
| `image_b_D_anchor_missing` | 5 |
| `no_legal_state` | 1 |

## Candidate Diagnostic Follow-Up

After PR #299 added `candidateDiagnostics` under `hullLabelTier1Prefer`, the
prefer bench was rerun on merged main at `b1f4b2a`:

```bash
npm run bench -- \
  --corpus /tmp/cube-snap-e2e-corpus \
  --providers cv-local \
  --cv-api http://localhost:8085/api/recognize \
  --cv-hull-label-tier1 prefer \
  --concurrency 2 \
  --out /tmp/cube-snap-e2e-results/prefer-diagnostics-main.json
```

The headline result was unchanged: 41 pairs, 35.8/54 mean stickers, 11
failures, and the same 4 selected hull-label candidates.

The new diagnostic payload changed the interpretation of the low selection
rate:

- Per-image hull geometry was accepted for both images on 40/41 pairs.
- The only per-image geometry rejection was `set-30` image A, rejected by the
  projective-residual hard gate.
- Most fallbacks therefore happen after geometry, during face identity and
  pair-level recognition checks.
- In fallback rows, candidate face assignments often collapse to too few
  distinct center-face labels. For example, candidate assignment keys contain
  `B` on image A in 36/37 fallback rows, while `R`, `F`, and `L` appear far
  less often.
- `missing_side_face_coverage` appears on 30/37 fallback rows because the
  candidate grids often do not expose all four side center labels after
  center-color classification.

This argues against simply loosening the hull geometry gates. The bottleneck is
now face-slot to WCA-face identity under the hull-label grids, especially U/D
anchor recognition and side-face coverage after center color classification.

## Interpretation

This is a good first production-shaped result, but not enough to default to
prefer mode yet.

The good news:

- Shadow mode is non-invasive.
- Prefer mode produced no regressions across this 41-pair run.
- Prefer recovered one real default-path failure (`set-39`) to `54/54`.
- Projective vertex selection was exercised successfully on one selected pair
  (`set-45`).

The caution:

- The gates are still very conservative: only 4/41 pairs selected the candidate.
- All selected rows are `needs_manual_review`, not clean-pass rows.
- Most fallbacks are due to face-grid coverage or U/D anchor checks, so the next
  useful work is improving coverage and explaining whether those rejections are
  correct conservatism or overly strict gating.

Recommended next step: keep the API hidden/default-off, leave shadow mode
available, and run one focused gate-coverage follow-up before considering any
cube-snap-facing prefer-mode wiring.

Post-diagnostics refinement: that follow-up should focus on assigning WCA faces
from known capture slots, yaw, and color evidence rather than treating
center-sticker color classification as the only face-identity authority. The
hull-label geometry is usually passing; the face identity layer is where the
coverage collapses.
