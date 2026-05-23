# Current-main solvable baseline

## Purpose

This is the production-recognizer solvable-rate baseline generated from
`tools/probe_corpus.py` on current `main`. It covers the current
`corpus_manifest.json` (27 rows) and `hard_case_manifest.json` (15 rows), and
keeps the per-sticker metric comparable to the older #139 solvable-rate
number.

Git head: `fabbac936247cfe3d78e8b6975bdc48da60db843`

## Headline

- Corpus-only per-sticker accuracy: 1128 / (27 * 54) = **77.4%**.
- Hard-case per-sticker accuracy: 285 / (14 * 54) = **37.7%**.
- Combined scored-row per-sticker accuracy: 1413 / (41 * 54) = **63.8%**.
- Combined all-row denominator, counting skipped rows as zero: **62.3%** (1 skipped row).
- Confident-wrong count: **0**.

## Summary

| Slice | Rows | Scored | Skipped | Score sum | Per-sticker | Exact match | Legal state | Confident solve | Confident wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corpus | 27 | 27 | 0 | 1128 | 77.4% | 16/27 (59.3%) | 22/27 (81.5%) | 13/27 (48.1%) | 0/27 (0.0%) |
| hard | 15 | 14 | 1 | 285 | 37.7% | 3/14 (21.4%) | 7/14 (50.0%) | 2/14 (14.3%) | 0/14 (0.0%) |
| overall | 42 | 41 | 1 | 1413 | 63.8% | 19/41 (46.3%) | 29/41 (70.7%) | 15/41 (36.6%) | 0/41 (0.0%) |

## Notes

- `perStickerAccuracy` excludes skipped rows because no recognizer score exists.
- The all-row denominator is included to make skipped rows visible (1 in this snapshot).
- `legalState` means the recognizer emitted a 54-sticker success state; manual-review successes still count as legal states.
- `confidentSolve` includes `success_clean` and `success_repaired_high_confidence`; it excludes `needs_manual_review` and `reject_retake`.
- `confidentWrong` is the Phase 3 guardrail target: a confident solve with hamming > 0.

## Rows

| Group | Set | Status | Category | Score | Hamming | Exact | Legal | Confident | Confident wrong |
|---|---:|---|---|---:|---:|---|---|---|---|
| corpus | 12 | success | needs_manual_review | 52 | 2 | no | yes | no | no |
| corpus | 14 | success | needs_manual_review | 50 | 4 | no | yes | no | no |
| corpus | 15 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| corpus | 23 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| corpus | 24 | success | success_repaired_high_confidence | 54 | 0 | yes | yes | yes | no |
| corpus | 26 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| corpus | 27 | success | needs_manual_review | 42 | 12 | no | yes | no | no |
| corpus | 28 | success | needs_manual_review | 30 | 24 | no | yes | no | no |
| corpus | 29 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| corpus | 31 | rejected | reject_retake | 0 | 54 | no | no | no | no |
| corpus | 32 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| corpus | 36 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| corpus | 37 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| corpus | 42 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| corpus | 44 | rejected | reject_retake | 0 | 54 | no | no | no | no |
| corpus | 20 | success | needs_manual_review | 54 | 0 | yes | yes | no | no |
| corpus | 38 | success | needs_manual_review | 48 | 6 | no | yes | no | no |
| corpus | 40 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| corpus | 41 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| corpus | 43 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| corpus | 45 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| corpus | 63 | success | needs_manual_review | 54 | 0 | yes | yes | no | no |
| corpus | 64 | rejected | reject_retake | 0 | 54 | no | no | no | no |
| corpus | 65 | rejected | reject_retake | 0 | 54 | no | no | no | no |
| corpus | 66 | rejected | reject_retake | 0 | 54 | no | no | no | no |
| corpus | 67 | success | needs_manual_review | 42 | 12 | no | yes | no | no |
| corpus | 68 | success | needs_manual_review | 54 | 0 | yes | yes | no | no |
| hard | 17 | rejected | reject_retake | 0 | 54 | no | no | no | no |
| hard | 21 | success | needs_manual_review | 54 | 0 | yes | yes | no | no |
| hard | 22 | success | success_clean | 54 | 0 | yes | yes | yes | no |
| hard | 25 | skipped | missing_files |  |  | no | no | no | no |
| hard | 30 | success | needs_manual_review | 18 | 36 | no | yes | no | no |
| hard | 39 | rejected | reject_retake | 0 | 54 | no | no | no | no |
| hard | 44 | rejected | reject_retake | 0 | 54 | no | no | no | no |
| hard | 46 | rejected | reject_retake | 0 | 54 | no | no | no | no |
| hard | 47 | rejected | reject_retake | 0 | 54 | no | no | no | no |
| hard | 48 | rejected | reject_retake | 0 | 54 | no | no | no | no |
| hard | 49 | rejected | reject_retake | 0 | 54 | no | no | no | no |
| hard | 57 | success | needs_manual_review | 29 | 25 | no | yes | no | no |
| hard | 58 | success | needs_manual_review | 42 | 12 | no | yes | no | no |
| hard | 61 | success | needs_manual_review | 34 | 20 | no | yes | no | no |
| hard | 62 | success | success_repaired_high_confidence | 54 | 0 | yes | yes | yes | no |

## Reproducing

```bash
.venv/bin/python tools/probe_corpus.py \
  --manifest tests/fixtures/corpus_manifest.json \
  --json-output /tmp/current_main_corpus_probe.json \
  --quiet
.venv/bin/python tools/probe_corpus.py \
  --manifest tests/fixtures/hard_case_manifest.json \
  --json-output /tmp/current_main_hard_probe.json \
  --quiet
.venv/bin/python tools/main_solvable_baseline.py \
  --corpus-json /tmp/current_main_corpus_probe.json \
  --hard-json /tmp/current_main_hard_probe.json
```
