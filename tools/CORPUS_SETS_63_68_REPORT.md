# Corpus sets 63-68 ingest report

## Summary

Six new A/B image pairs and their ground-truth JSON files were added from
`/Users/jhuber/cube-corpus` to `tests/fixtures/corpus_manifest.json`.
The corresponding uploaded files in `/Users/jhuber/Downloads` are byte-identical
to the canonical corpus JSON files.

All six rows now have pinned image/ground-truth SHA-256 hashes and calibrated
current-recognizer contracts from `tools/probe_corpus.py`.

Important provenance note: the seed-provider fields embedded in these
ground-truth JSONs describe the machine attempt that started the Fixer session,
not the authority of the final label. Sets 64-66 started from LLM-generated
seed states after cv-local failed to produce a useful result; those seed states
were extensively corrected in Fixer. Treat the final `corrected` state as the
human-corrected ground truth and the provider/diff fields as debugging
provenance.

All six sets share the same final `corrected` cube state:

```text
LBBBUUUBRULLRRLFFRFRBFFFDFDLLRUDUDBUDDLRLDBRFUUBDBDFLR
```

That makes Sets 63-68 a controlled repeated-capture/yaw cohort: the cube state
is constant while capture/yaw/geometry evidence varies.

## Current Recognizer Results

| Set | Seed provider | Seed category | Seed diff before Fixer | Recorded yaw | Canonicalization source | Current status | Current category | Score | Hamming |
|---:|---|---|---:|---:|---|---|---|---:|---:|
| 63 | cv-local | needs_manual_review | 0 | 2 | explicit-yaw | success | needs_manual_review | 54/54 | 0 |
| 64 | gemini-3.1-pro | n/a | 31 | 0 (human-confirmed) | n/a | rejected | reject_retake | 0/54 | 54 |
| 65 | claude-sonnet | n/a | 36 | 1 | center-inference | rejected | reject_retake | 0/54 | 54 |
| 66 | claude-sonnet | n/a | 43 | 2 | center-inference | rejected | reject_retake | 0/54 | 54 |
| 67 | cv-local | needs_manual_review | 12 | 3 | explicit-yaw | success | needs_manual_review | 42/54 | 12 |
| 68 | cv-local | needs_manual_review | 0 | 0 (human-confirmed) | n/a | success | needs_manual_review | 54/54 | 0 |

Manifest yaw contracts are intentionally limited to rows where the current
recognizer emits `recognitionSignals.captureYaw`: Sets 63, 67, and 68 now carry
`expectedYaw`; Sets 64-66 are rejected before that signal is available, so their
yaw remains documented in prose/metadata rather than enforced as a probe
contract.

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
