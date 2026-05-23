# Corpus sets 63-68 ingest report

## Summary

Six new A/B image pairs and their ground-truth JSON files were added from
`/Users/jhuber/cube-corpus` to `tests/fixtures/corpus_manifest.json`.
The corresponding uploaded files in `/Users/jhuber/Downloads` are byte-identical
to the canonical corpus JSON files.

All six rows now have pinned image/ground-truth SHA-256 hashes and calibrated
current-recognizer contracts from `tools/probe_corpus.py`.

## Current Recognizer Results

| Set | Ground-truth provider | Ground-truth source category | Source diff | Source yaw | Source canonicalization | Current status | Current category | Score | Hamming |
|---:|---|---|---:|---:|---|---|---|---:|---:|
| 63 | cv-local | needs_manual_review | 0 | 2 | explicit-yaw | success | needs_manual_review | 54/54 | 0 |
| 64 | gemini-3.1-pro | n/a | 31 | n/a | n/a | rejected | reject_retake | 0/54 | 54 |
| 65 | claude-sonnet | n/a | 36 | 1 | center-inference | rejected | reject_retake | 0/54 | 54 |
| 66 | claude-sonnet | n/a | 43 | 2 | center-inference | rejected | reject_retake | 0/54 | 54 |
| 67 | cv-local | needs_manual_review | 12 | 3 | explicit-yaw | success | needs_manual_review | 42/54 | 12 |
| 68 | cv-local | needs_manual_review | 0 | n/a | n/a | success | needs_manual_review | 54/54 | 0 |

## Expanded Baseline Impact

The committed solvable-rate snapshot now covers:

- Corpus manifest: 27 rows, 1128/1458 stickers correct = 77.4%.
- Hard-case manifest: 15 rows, 14 scored, 285/756 stickers correct = 37.7%.
- Combined: 42 rows, 41 scored, 1413/2214 stickers correct = 63.8%.
- Confident-wrong rows: 0.

See `tools/MAIN_SOLVABLE_BASELINE.md` for the full row-level table.

## Verification

```bash
.venv/bin/python tools/probe_corpus.py \
  --manifest tests/fixtures/corpus_manifest.json \
  --json-output /tmp/current_main_corpus_probe_with_63_68.json \
  --quiet \
  --fail-on-contract

.venv/bin/python tools/probe_corpus.py \
  --manifest tests/fixtures/hard_case_manifest.json \
  --json-output /tmp/current_main_hard_probe_with_63_68.json \
  --quiet \
  --fail-on-contract

.venv/bin/python tools/main_solvable_baseline.py \
  --corpus-json /tmp/current_main_corpus_probe_with_63_68.json \
  --hard-json /tmp/current_main_hard_probe_with_63_68.json
```
