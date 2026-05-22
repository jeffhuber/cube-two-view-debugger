# Current-main solvable baseline

## Purpose

This is the production-recognizer solvable-rate baseline generated from
`tools/probe_corpus.py` on current `main`. It covers the 15-row
`corpus_manifest.json` and the 15-row `hard_case_manifest.json`, and
keeps the per-sticker metric comparable to the older #139 solvable-rate
number.

Git head: `e2f9fc04e74d33a7fceb2f40ff38e5be9ddd6c39`

## Headline

- Corpus-only per-sticker accuracy: 660 / (15 * 54) = **81.5%**.
- Hard-case per-sticker accuracy: 285 / (14 * 54) = **37.7%**.
- Combined scored-row per-sticker accuracy: 945 / (29 * 54) = **60.3%**.
- Combined all-row denominator, counting skipped rows as zero: **58.3%** (1 skipped row).
- Confident-wrong count: **0**.

## Summary

| Slice | Rows | Scored | Skipped | Score sum | Per-sticker | Exact match | Legal state | Confident solve | Confident wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corpus | 15 | 15 | 0 | 660 | 81.5% | 9/15 (60.0%) | 13/15 (86.7%) | 9/15 (60.0%) | 0/15 (0.0%) |
| hard | 15 | 14 | 1 | 285 | 37.7% | 3/14 (21.4%) | 7/14 (50.0%) | 2/14 (14.3%) | 0/14 (0.0%) |
| overall | 30 | 29 | 1 | 945 | 60.3% | 12/29 (41.4%) | 20/29 (69.0%) | 11/29 (37.9%) | 0/29 (0.0%) |

## Notes

- `perStickerAccuracy` excludes skipped rows because no recognizer score exists.
- The all-row denominator is included only to make the missing Set 25 hard-case row visible.
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
