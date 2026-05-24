# Oracle rectified faces tool — design (for Codex review BEFORE implementation)

**Status: design-only, no code yet.** Per the user's request, posting this for
Codex design review before I write the tool. Comments / pushback welcome on
any of the decisions below.

## Purpose

Use `tests/fixtures/full_corner_ground_truth.json` (+ `corner_conventions.py`)
to produce **oracle-quality flat rectified face crops and per-sticker patches**
with full WCA facelet labels. Isolates color classification from geometry
failure: every sample uses human-validated face corners + canonical
conventions, so any color-classification mistake is purely a color problem,
not geometry contamination.

Diagnostic-only. Not a production-recognizer change.

## Inputs

- `tests/fixtures/full_corner_ground_truth.json` — 12 rows × {vertex,
  corner_0..corner_5, yaw_quarter_turns, approved} (yaw added in #256;
  this design assumes that PR lands first).
- `tools/corner_conventions.py` — `FACE_DEFS_BY_SIDE`, `YAW0_CORNER_FACELETS`,
  `CAPTURE_FACE_BY_SLOT_BY_SIDE`, `wca_face_by_slot()`.
- `tests/fixtures/corpus_manifest.json` — image paths.
- Raw cube photos.

## Outputs

```
/tmp/oracle_rectified_faces_v1/        (ephemeral, not committed)
  ├─ by_row/{set_id}_{side}/{wca_face}.png    (36 face PNGs: 12 rows × 3 faces)
  ├─ by_facelet/{wca_facelet_id}.png          (324 sticker PNGs: 36 × 9)
  ├─ patch_png/{wca_facelet_id}.png           (324 patch PNGs — the central
  │                                            40% raw pixel patch per sticker)
  ├─ index.json                                (full metadata; see schema below)
  └─ gallery.html                              (visual inspection grid)

tools/ORACLE_RECTIFIED_FACES.md              (writeup; yaw assumptions; suspicious-row flags)
```

### `index.json` schema

```json
{
  "schema": "oracle_rectified_faces_v1",
  "source": {
    "truth_path": "tests/fixtures/full_corner_ground_truth.json",
    "face_size_px": 300,
    "sticker_patch_fraction": 0.40
  },
  "rows": [
    {
      "key": "20_A",
      "side": "A",
      "yaw_quarter_turns": 0,
      "image_path": "...",
      "faces": [
        {
          "slot": "upper",                  // upper | right | front
          "wca_face": "U",                  // depends on side + yaw
          "face_png": "by_row/20_A/U.png",
          "quad_image_px": [[vx,vy],[c1x,c1y],[c0x,c0y],[c5x,c5y]],
          "stickers": [
            {
              "row": 0, "col": 0,
              "facelet_id": "U1",
              "patch_png": "patch_png/U1.png",
              "patch_pixel_center_in_face": [50, 50],
              "rgb": [r, g, b],
              "hsv": [h, s, v],          // colorsys output (h,s,v in [0,1])
              "lab": [L, a, b],          // CIELAB (L* in [0,100], a*/b* approx [-128,128])
              "classify_rgb": "white",   // production classifier color verdict
              "classify_confidence": 0.92
            },
            // ... 8 more stickers per face ...
          ]
        },
        // ... 2 more faces per row ...
      ]
    },
    // ... 11 more rows ...
  ]
}
```

## API

```
tools/build_oracle_rectified_faces.py

  --truth          tests/fixtures/full_corner_ground_truth.json   (default)
  --out            /tmp/oracle_rectified_faces_v1/                (default; ephemeral)
  --face-size      300                                            (pixels per side)
  --sticker-patch  0.40                                           (cell-width fraction)
  --yaw-overrides  ""                                             (e.g. "20:1,38:0"; default uses fixture's yaw)
  --no-patches     (flag)                                         (skip per-sticker patch PNGs to save disk)
```

`--yaw-overrides` lets a user temporarily try a different yaw without editing
the fixture (e.g., suspect a label is wrong, want to test before committing).
Default uses each row's `yaw_quarter_turns` from the fixture; falls back to 0
with a `yaw_assumed_zero` flag in the per-row metadata if the field is missing.

## Sticker geometry math (recap from the user-facing scoping)

The perspective homography removes ALL projective distortion. Once
rectified to a flat `face_size × face_size` square, the 3×3 sticker
grid is provably regular:

- Each cell = `face_size / 3` per side
- Sticker (row=r, col=c) center pixel = `((c+0.5)*cell_size, (r+0.5)*cell_size)`
- Standard URFDLB sticker numbering: `sticker_id = 3*row + col + 1`
- Color sample = median RGB over a central `patch_fraction × cell_size` patch
  (default 40% of cell width, leaving 30 px of margin on each side at face_size=300
  — comfortable distance from bezels and sticker borders)

`_perspective_coeffs(src_quad, dst_size)` in the existing
`tools/rectify_faces.py` does the 8-coefficient solve (4 corner pairs → 8
linear equations in 8 unknowns). I'll reuse that function; everything else
is re-implemented for canonical-convention awareness.

## Convention mapping (load-bearing — needs pinning tests)

For each `(image_side, slot)` — 6 cases total — the canonical quad order
from `FACE_DEFS_BY_SIDE[side][slot]` maps to specific facelet corner IDs
via `YAW0_CORNER_FACELETS`. Worked tables below.

### Image A's 3 faces at yaw=0

| Slot | WCA face | Quad order (from FACE_DEFS) | Facelet corner IDs | Target square corner mapping |
|---|---|---|---|---|
| upper | U | (vertex, corner_1, corner_0, corner_5) | (U9, U3, U1, U7) | Va→BR, c_1→TR, c_0→TL, c_5→BL |
| right | R | (vertex, corner_3, corner_2, corner_1) | (R1, R7, R9, R3) | Va→TL, c_3→BL, c_2→BR, c_1→TR |
| front | F | (vertex, corner_5, corner_4, corner_3) | (F3, F1, F7, F9) | Va→TR, c_5→TL, c_4→BL, c_3→BR |

### Image B's 3 faces at yaw=0

| Slot | WCA face | Quad order | Facelet corner IDs | Target square corner mapping |
|---|---|---|---|---|
| upper | D | (vertex, corner_2, corner_3, corner_4) | (D7, D9, D3, D1) | Vb→BL, c_2→BR, c_3→TR, c_4→TL |
| right | B | (vertex, corner_0, corner_1, corner_2) | (B9, B3, B1, B7) | Vb→BR, c_0→TR, c_1→TL, c_2→BL |
| front | L | (vertex, corner_4, corner_5, corner_0) | (L7, L9, L3, L1) | Vb→BL, c_4→BR, c_5→TR, c_0→TL |

(I hand-derived these during the #251 review; will re-verify during
implementation. Sticker numbering within face after rectification follows
standard URFDLB `1 2 3 / 4 5 6 / 7 8 9` row-major.)

### Non-zero yaw

For yaw ≠ 0, the WCA face name at each slot shifts per
`wca_face_by_slot(side, yaw)`. The rectified pixel content + sticker grid
positions are identical — only the OUTPUT FILENAME changes (e.g., the right
slot at A/yaw=1 saves to `B.png` instead of `R.png`, and its stickers are
named `B1..B9` instead of `R1..R9`). The convention mapping above is the
yaw=0 base; everything else is a relabel.

## Decisions / answers to the open scoping questions

1. **Yaw**: take values from the fixture's new `yaw_quarter_turns` field
   (added in PR #256). Default to 0 with a warning flag if missing.
   `--yaw-overrides` for ad-hoc testing.

2. **Reuse `rectify_faces.py` primitives**: only `_perspective_coeffs`
   (pure math, no convention coupling). Re-implement `rectify_face` and
   the sticker sampler inline because:
   - `rectify_faces.rectify_face` calls `canonical_corner_order()` with a
     "CW-from-N" ordering that doesn't match `FACE_DEFS_BY_SIDE` order
     — bypassing it is cleaner than adapting.
   - `extract_stickers_from_rectified` returns single-rgb + single classified
     color. We want raw RGB + HSV + Lab + classify_rgb + patch PNGs.
   - Net new code: ~150-200 LOC (most of it is convention tables +
     color-space conversions, not new pipeline logic).

3. **Color output**: emit ALL of:
   - `rgb`: median RGB over the central patch (raw device color)
   - `hsv`: `colorsys.rgb_to_hsv` output — repo's `rubik_recognizer/colors.py`
     already uses HSV with a `_rubik_hsv_hint(rgb)` for the 6 cube colors;
     including HSV makes the tool's output directly consumable by that
     classifier without re-conversion
   - `lab`: CIELAB — perceptually uniform; ΔE differences match human
     perception; industry standard for "is this red or orange?" decisions
     where the hue band is narrow. Computed via sRGB → XYZ → Lab closed-form
     math (~15 LOC, no scipy dependency).
   - `classify_rgb`: the production classifier's verdict, for direct
     comparison against the raw sample

   Also save the raw 41×41 patch PNG per sticker (controlled by `--no-patches`
   flag, default ON), so downstream color tools can re-derive any statistic
   (median / mean / mode / trimmed mean / percentile etc.) without re-running
   the rectifier. Cost: ~324 small PNGs (~5-10 KB each = ~3 MB), ephemeral
   in `/tmp/`.

## Test plan

- 6 pinning tests for `(side, slot)` → facelet-corner-ID mapping (one per face)
- 1 pinning test for `(row, col)` → URFDLB sticker number
- 1 yaw-integration test (e.g., `wca_face_by_slot("A", 1)` → right slot is B,
  facelet IDs become B1..B9)
- 1 sRGB → Lab round-trip test against a published reference value
  (e.g., pure red sRGB (255,0,0) → Lab (53.24, 80.09, 67.20) within 0.1)
- 1 perspective round-trip test: warp a known-colored synthetic face, sample,
  verify rgb matches input
- 1 end-to-end smoke test on 1 row with a synthetic image (no rembg dep
  needed — the rectifier doesn't call rembg)

## Sequencing

1. PR #256 (yaw fixture) lands first — design depends on that field.
2. Implement on a new branch `claude/oracle-rectified-faces` (this design
   branch can be the same one if you'd rather avoid a second PR).
3. Codex pre-push review on convention math + Lab conversion.
4. Open PR with `needs-codex-audit`.

## Open design questions for Codex

- Sticker numbering convention: I assumed standard URFDLB `1 2 3 / 4 5 6 / 7 8 9`
  row-major. I'll verify against `src/cube.ts` `PLACEMENTS` before pinning, but
  if the project uses a different convention let me know.
- Per-sticker PNG count (324 files): ephemeral so disk cost is low. Codex —
  any concern with PNG count or preference for a single packed image atlas?
- Patch fraction default of 0.40 (40% of cell width central): matches existing
  `rectify_faces.extract_stickers_from_rectified` default. Adequate for color
  sampling without bezel contamination, but is there a more conservative
  number Codex has used elsewhere?
- Lab values rounded to: `[L: 0.1°, a: 0.1°, b: 0.1°]` precision in the JSON
  (perceptually meaningful threshold ~1 ΔE)? Or just full float precision and
  let the consumer round?

## Why this design now

The user's strategic note (earlier in the conversation): the next chirality
fix path most likely runs both phase hypotheses forward and picks the better
one using **orthogonal evidence — center-color consistency** being a top
candidate. The oracle tool provides the foundation for that probe by
producing clean color samples uncontaminated by geometry uncertainty. It
also enables direct color-classifier tuning under known-correct geometry.

Both downstream uses depend on the convention math being correct. Hence
this design review before implementation.

---

*Design author: Claude. Reviewer: Codex (please review the convention tables,
the Lab decision, the file-count strategy, and any other choices that aren't
obviously right).*
