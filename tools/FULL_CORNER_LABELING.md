# Full Corner Labeling

Status: active convention-reset labeling tool. The first canonical fixture is
`tests/fixtures/full_corner_ground_truth.json` (sets 20, 38, 40, 41, 43, and
45; both A/B photos).

Seed fixture review on 2026-05-23:

- 12/12 rows approved and schema-clean.
- All points are inside the source image bounds.
- The labeled A/B face outlines visually align with the cube faces in the
  contact-sheet audit.
- The one-edge distance ratio across the seed rows is 1.151 / 1.222 / 1.419
  (min / median / max), which is plausible under the perspective range in the
  sample.

The full-corner label format is the source of truth for visible cube geometry.
It labels the seven human-visible points directly and avoids model-axis names
such as `h_x`, `h_y`, `h_z`, `near_x`, `near_y`, and `near_z`.

## Human Convention

Each photo has one visible trihedral vertex and six outer visible corners.
Use `Va` for image A and `Vb` for image B.

Image A is the white-up view. Its visible trihedral vertex is `Va`, where
the upper, right-slot, and front-slot faces meet:

```text
          0
     5         1

          Va

     4         2
          3
```

```text
upper slot = Va + 1,0,5
right slot = Va + 3,2,1
front slot = Va + 5,4,3
```

Image B is the yellow-up view after the single 180-degree flip around
image-horizontal / camera-X. Its visible trihedral vertex is `Vb`, where
the upper, right-slot, and front-slot faces meet:

```text
          3
     4         2

          Vb

     5         1
          0
```

```text
upper slot = Vb + 2,3,4
right slot = Vb + 0,1,2
front slot = Vb + 4,5,0
```

The one-edge-away vs far/double-axis triplet depends on side:

```text
Image A one-edge corners from Va   = 1,3,5
Image A far/double-axis corners    = 0,2,4

Image B one-edge corners from Vb   = 0,2,4
Image B far/double-axis corners    = 1,3,5
```

The numbering is a human visual convention. It does not change when the cube
has capture yaw. Downstream code may convert these points into model-axis names
or WCA face names, but that conversion must be explicit and tested.

## Capture Yaw

The slot labels above are view-local. Canonical WCA face names depend on
capture yaw around the `U/D` axis. For a standard Rubik's color scheme,
white/yellow stay on the upper slots, while the four side faces rotate:

| yaw | A front slot | A right slot | B front slot | B right slot |
|---:|---|---|---|---|
| 0 | `F` / green | `R` / red | `L` / orange | `B` / blue |
| 1 | `R` / red | `B` / blue | `F` / green | `L` / orange |
| 2 | `B` / blue | `L` / orange | `R` / red | `F` / green |
| 3 | `L` / orange | `F` / green | `B` / blue | `R` / red |

Example: if image A's right-slot face has a blue center, that is yaw `1`.
The corner labels are still the same; only the mapping from slots to WCA
faces changes.

## Facelet Mapping

Flattened cube states use `URFDLB` face order, with each face row-major:

```text
1 2 3
4 5 6
7 8 9
```

At yaw `0`, corner positions map to flattened-net facelets as follows:

| Corner position | Facelets |
|---|---|
| `Va` | `U9 / R1 / F3` |
| `Vb` | `D7 / L7 / B9` |
| `0` | `U1 / L1 / B3` |
| `1` | `U3 / R3 / B1` |
| `2` | `D9 / R9 / B7` |
| `3` | `D3 / R7 / F9` |
| `4` | `D1 / L9 / F7` |
| `5` | `U7 / F1 / L3` |

Do not describe `Va` as a sticker. It is a physical corner position. At yaw
`0`, it is the `URF` corner where `U9`, `R1`, and `F3` meet. Likewise, at
yaw `0`, `Vb` is the `DBL` corner where `D7`, `L7`, and `B9` meet. For
non-zero yaw, derive the WCA facelet mapping from the yaw table above. Code
should use `tools.corner_conventions.wca_facelets_for_label(...)` rather than
hand-replacing face letters; yaw changes the physical corner identity, including
the U/D facelet index.

## JSON Schema

```json
{
  "20_A": {
    "vertex": [1495.9, 1870.8],
    "corner_0": [0.0, 0.0],
    "corner_1": [0.0, 0.0],
    "corner_2": [0.0, 0.0],
    "corner_3": [0.0, 0.0],
    "corner_4": [0.0, 0.0],
    "corner_5": [0.0, 0.0],
    "approved": true
  }
}
```

## Build The Gallery

```bash
.venv/bin/python tools/build_full_corner_labeling_gallery.py \
  --out /tmp/full_corner_labeling_v1 \
  --sets 20 38 40 41 43 45
```

Open:

```text
file:///private/tmp/full_corner_labeling_v1/gallery.html
```

The gallery is file-based. It copies EXIF-corrected full images into the output
directory and lets the browser fit them to the viewport. It does not run rembg,
global-model prefill, or any geometry model.

## Do Not Conflate With Older Fixtures

Older axis fixtures may contain fields named `near_x`, `near_y`, and `near_z`.
Those names are now known to be ambiguous in practice. Do not assume they mean
the one-edge triplet unless a conversion against full-corner truth proves it.

Initial audit result on the 12 full-corner seed rows: the legacy
`tests/fixtures/gcm_axis_ground_truth.json` `near_*` points match the
full-corner **far/double-axis** triplet, not the one-edge triplet:

```text
Image A: legacy near_* -> 0,2,4
Image B: legacy near_* -> 1,3,5
```

The median nearest-corner offset in that audit is about 20 px, so the legacy
labels are usable as historical pixel annotations, but their field names are
not safe as semantic truth.

Use full-corner labels to audit and migrate those older fixtures.
