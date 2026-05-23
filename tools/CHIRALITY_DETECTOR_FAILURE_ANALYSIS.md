# Chirality detector failure analysis

Source matrix: `tests/fixtures/phase2b_recomputed_signals.json`
Total rows: 140 (70 cases × ~2 runs each)

## Per-category row counts

| Category | Rows |
|---|---|
| `GOOD` | 91 |
| `CHIRALITY_MISS` | 20 |
| `MARGINAL` | 16 |
| `CHIRALITY_FALSE_FLIP` | 12 |
| `TRUE_GEOMETRY_FAIL` | 1 |

## Chirality-row failure modes

Total chirality-failure rows: 32

| Failure mode | Rows | % |
|---|---:|---:|
| `DETECTOR_WRONG_CALL` | 19 | 59.4% |
| `DETECTOR_AMBIGUOUS` | 13 | 40.6% |

### What each failure mode means

- `DETECTOR_AMBIGUOUS`: the darkness-separation discriminator saw `|sep| < 10` and declined to make a decision. The detector needs a stronger or different signal to resolve these rows.
- `DETECTOR_WRONG_CALL`: the detector confidently chose a phase (`correct` or `corrected_60deg_flip`) but the resulting axes are still wrong (err_near ≥ 25°). The signal exists and is strong, but the decision is inverted or the threshold/polarity is mis-calibrated for these rows.
- `FLIP_SUGGESTED_NOT_APPLIED`: detector identified a flip but the recompute pipeline ran with `apply_correction=False`. This is a pipeline-configuration finding, not a detector bug.
- `PIPELINE_BUG`: `phase_check` value not in the expected set — warrants direct investigation.

## Separation distribution by category × phase_check

| Category × phase_check | n | median sep | min | max |
|---|---:|---:|---:|---:|
| CHIRALITY_FALSE_FLIP \| corrected_60deg_flip | 12 | 15.7 | 11.3 | 40.9 |
| CHIRALITY_MISS \| ambiguous_no_correction | 13 | -1.2 | -8.4 | 8.6 |
| CHIRALITY_MISS \| correct | 7 | -20.0 | -49.3 | -10.5 |

## Key findings

1. **The detector's polarity rule is being correctly applied — but to the wrong sign on a specific subset of rows (19 `DETECTOR_WRONG_CALL` rows).** Looking at the per-row patterns:

   - `corrected_60deg_flip` (12 rows): sep median 15.7 (all same sign); err_near median 58.2° after the detector's decision. The detector confidently chose this branch and got it wrong on these rows.
   - `correct` (7 rows): sep median -20.0 (all same sign); err_near median 59.3° after the detector's decision. The detector confidently chose this branch and got it wrong on these rows.

   These rows are NOT random noise — the sep signal is unambiguous and the detector's polarity rule is firing as documented. The rule's underlying assumption (NEG sep = correct, POS sep = needs flip) **simply does not hold on this subset**. Something about these specific images (lighting? sticker color? bezel contrast? vertex offset?) inverts the polarity.

2. **DETECTOR_AMBIGUOUS (13 rows, 41% of chirality failures) need a stronger discriminator.** The `|sep| < 10` band leaves these rows undecided. They have valid geometry except for the 60° phase ambiguity — solving them would directly reduce the `CHIRALITY_MISS` rate.

## Recommended next experiments (out of scope for this diagnostic PR)

- **Look for a meta-signal that predicts polarity inversion.** The matrix already records other per-row signals (fit_residual_rms_px, vertex_ensemble_stddev_px, junction_score_at_ensemble, ensemble_shift_px, etc). Do any correlate with the `DETECTOR_WRONG_CALL` rows above? If yes, the detector could gate its polarity rule on the meta-signal.
- **Visual inspection of the 19 wrong-call rows.** What is physically different about Sets 12, 25, 30, 40, 41, 44, 45, 46, 47, 49, 57, 62 that inverts the darkness polarity? Bezel reflectivity, lighting angle, sticker color saturation are the candidate variables.
- **Strengthen the ambiguous-band discriminator.** Candidate signals: per-line darkness variance (not just mean), color-saturation along the lines, edge-gradient orientation, or multiple lines per corner (not just vertex→corner).

## Meta-signal candidate for the wrong-call subset

Live compare of per-row feature distributions between two groups in `tests/fixtures/phase2b_recomputed_signals.json`:

- **RIGHT-call rows** (n=91): detector decided (`phase_check ∈ {correct, corrected_60deg_flip}`) AND the row outcome is GOOD or MARGINAL
- **WRONG-call rows** (n=19): same `phase_check` values but the row is `CHIRALITY_MISS` or `CHIRALITY_FALSE_FLIP`

| Feature | RIGHT median | RIGHT IQR | WRONG median | WRONG IQR | Verdict |
|---|---:|---|---:|---|---|
| `junction_score_at_ensemble` | 216.4 | [195.75, 230.75] | 124.2 | [81.1, 178.75] | **IQR-DISJOINT** |
| `bezel_vs_fit_cube_center_offset_px` | 11.3 | [0.0, 61.15] | 28.3 | [0.0, 52.25] | IQR-overlap |
| `hexagon_centroid_vs_bezel_vertex_offset_px` | 41.8 | [18.9, 88.85] | 53.0 | [15.85, 93.4] | IQR-overlap |
| `phase_darkness_separation` | 17.3 | [-35.0, 37.45] | 13.6 | [-14.55, 17.5] | IQR-overlap |
| `fit_residual_rms_px` | 68.27 | [30.63, 96.72] | 61.51 | [35.58, 98.03] | IQR-overlap |
| `ensemble_shift_px` | 30.1 | [14.8, 62.85] | 28.4 | [17.3, 47.7] | IQR-overlap |
| `pnp_rms_px` | 70.27 | [30.63, 107.33] | 72.4 | [35.58, 106.79] | IQR-overlap |
| `ensemble_n_candidates` | 3.0 | [3.0, 3.0] | 3.0 | [3.0, 3.0] | IQR-overlap |

**Interpretation (live):** `junction_score_at_ensemble` has IQR-disjoint separation between the right-call and wrong-call groups — RIGHT IQR [195.75, 230.75] lies entirely above WRONG IQR [81.1, 178.75]. **The full ranges also overlap** (RIGHT min/max 38.4/248.8; WRONG min/max 15.5/241.1), so the IQR-only separation does NOT cleanly partition the populations.

**Candidate-threshold trade-off at `junction_score_at_ensemble < 187.2`** (midpoint of the two IQRs):

- **16/19** WRONG-call rows would be gated to `ambiguous_no_correction` (benefit: those rows avoid the bad polarity decision).
- **19/91** RIGHT-call rows would ALSO be gated (cost: those rows lose the correct polarity decision; some may flip back to `ambiguous` and remain correct anyway, others may regress).

**Actionable hypothesis for next fix PR:** gate the polarity rule on `junction_score_at_ensemble`. The IQR-midpoint threshold `187.2` is the starting point; the actual operating point should be calibrated on a held-out split with the FP/FN trade-off above made explicit. Likely the right approach is a soft confidence rather than a hard binary gate — i.e., treat low-feature rows (the WRONG-call territory) as ambiguous.

## Per-row detail (chirality failures only)

| Case | Run | Category | phase_check | sep | err_near | err_far | Failure mode |
|---|---:|---|---|---:|---:|---:|---|
| `12_A` | 0 | `CHIRALITY_MISS` | `ambiguous_no_correction` | -8.4 | 50.0 | 10.4 | DETECTOR_AMBIGUOUS |
| `12_A` | 1 | `CHIRALITY_FALSE_FLIP` | `corrected_60deg_flip` | 12.7 | 49.8 | 11.1 | DETECTOR_WRONG_CALL |
| `21_B` | 1 | `CHIRALITY_MISS` | `ambiguous_no_correction` | 7.4 | 51.1 | 8.9 | DETECTOR_AMBIGUOUS |
| `22_B` | 0 | `CHIRALITY_MISS` | `ambiguous_no_correction` | 8.6 | 50.9 | 9.1 | DETECTOR_AMBIGUOUS |
| `25_A` | 0 | `CHIRALITY_MISS` | `ambiguous_no_correction` | -5.7 | 50.3 | 10.5 | DETECTOR_AMBIGUOUS |
| `25_A` | 1 | `CHIRALITY_MISS` | `ambiguous_no_correction` | -6.4 | 50.5 | 10.5 | DETECTOR_AMBIGUOUS |
| `25_B` | 0 | `CHIRALITY_MISS` | `correct` | -10.5 | 58.7 | 4.9 | DETECTOR_WRONG_CALL |
| `26_B` | 1 | `CHIRALITY_MISS` | `ambiguous_no_correction` | -2.2 | 57.1 | 3.7 | DETECTOR_AMBIGUOUS |
| `30_B` | 0 | `CHIRALITY_MISS` | `correct` | -16.4 | 55.8 | 4.2 | DETECTOR_WRONG_CALL |
| `37_A` | 1 | `CHIRALITY_MISS` | `ambiguous_no_correction` | -0.6 | 49.7 | 14.6 | DETECTOR_AMBIGUOUS |
| `39_A` | 0 | `CHIRALITY_MISS` | `ambiguous_no_correction` | 1.9 | 57.0 | 7.8 | DETECTOR_AMBIGUOUS |
| `39_A` | 1 | `CHIRALITY_MISS` | `ambiguous_no_correction` | -7.4 | 57.3 | 7.7 | DETECTOR_AMBIGUOUS |
| `40_A` | 0 | `CHIRALITY_MISS` | `ambiguous_no_correction` | 1.4 | 59.5 | 10.9 | DETECTOR_AMBIGUOUS |
| `40_A` | 1 | `CHIRALITY_MISS` | `correct` | -12.7 | 59.5 | 10.8 | DETECTOR_WRONG_CALL |
| `41_A` | 0 | `CHIRALITY_FALSE_FLIP` | `corrected_60deg_flip` | 13.7 | 58.8 | 21.5 | DETECTOR_WRONG_CALL |
| `41_A` | 1 | `CHIRALITY_MISS` | `correct` | -33.2 | 59.7 | 9.7 | DETECTOR_WRONG_CALL |
| `43_B` | 1 | `CHIRALITY_MISS` | `ambiguous_no_correction` | 0.9 | 60.0 | 5.1 | DETECTOR_AMBIGUOUS |
| `44_A` | 0 | `CHIRALITY_MISS` | `correct` | -42.2 | 52.3 | 9.7 | DETECTOR_WRONG_CALL |
| `44_A` | 1 | `CHIRALITY_FALSE_FLIP` | `corrected_60deg_flip` | 24.0 | 57.2 | 6.1 | DETECTOR_WRONG_CALL |
| `45_A` | 1 | `CHIRALITY_FALSE_FLIP` | `corrected_60deg_flip` | 13.6 | 57.7 | 15.6 | DETECTOR_WRONG_CALL |
| `45_B` | 0 | `CHIRALITY_MISS` | `correct` | -20.0 | 59.3 | 3.0 | DETECTOR_WRONG_CALL |
| `45_B` | 1 | `CHIRALITY_FALSE_FLIP` | `corrected_60deg_flip` | 40.9 | 59.9 | 18.4 | DETECTOR_WRONG_CALL |
| `46_B` | 0 | `CHIRALITY_FALSE_FLIP` | `corrected_60deg_flip` | 16.3 | 58.8 | 14.9 | DETECTOR_WRONG_CALL |
| `46_B` | 1 | `CHIRALITY_FALSE_FLIP` | `corrected_60deg_flip` | 15.1 | 59.8 | 14.8 | DETECTOR_WRONG_CALL |
| `47_B` | 0 | `CHIRALITY_FALSE_FLIP` | `corrected_60deg_flip` | 39.7 | 60.0 | 16.3 | DETECTOR_WRONG_CALL |
| `47_B` | 1 | `CHIRALITY_MISS` | `correct` | -49.3 | 59.4 | 3.6 | DETECTOR_WRONG_CALL |
| `49_A` | 0 | `CHIRALITY_FALSE_FLIP` | `corrected_60deg_flip` | 11.3 | 49.9 | 13.9 | DETECTOR_WRONG_CALL |
| `49_A` | 1 | `CHIRALITY_FALSE_FLIP` | `corrected_60deg_flip` | 35.3 | 50.3 | 15.3 | DETECTOR_WRONG_CALL |
| `57_A` | 1 | `CHIRALITY_FALSE_FLIP` | `corrected_60deg_flip` | 18.7 | 56.4 | 5.4 | DETECTOR_WRONG_CALL |
| `58_A` | 1 | `CHIRALITY_MISS` | `ambiguous_no_correction` | -1.2 | 56.2 | 7.4 | DETECTOR_AMBIGUOUS |
| `62_B` | 0 | `CHIRALITY_FALSE_FLIP` | `corrected_60deg_flip` | 13.7 | 59.6 | 9.9 | DETECTOR_WRONG_CALL |
| `62_B` | 1 | `CHIRALITY_MISS` | `ambiguous_no_correction` | -7.1 | 59.4 | 6.8 | DETECTOR_AMBIGUOUS |
