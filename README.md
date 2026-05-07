# Two-View Rubik's Cube Recognizer

A small local web app for recognizing a Rubik's cube from two isometric photos:
image A starts with the white face up, then image B shows the cube after the flip
with the opposite yellow face visible.

The recognizer is intentionally strict. It only returns a 54-character URFDLB state when the
two images provide enough evidence to build a legal cube state. Otherwise it returns a rejection
reason and annotated overlays for debugging or retaking the photos.

Batch mode can compare results against CSV, TSV, or JSON ground-truth files. JSON exports with
`setName` and `corrected` fields are supported directly, including unique legal net exports that
need to be canonicalized into standard URFDLB order.

## Run

Use the bundled Codex Python runtime because it includes Pillow and NumPy:

```sh
/Users/jhuber/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 app.py
```

Then open:

```text
http://127.0.0.1:8080/
```

## CLI Probe

You can also analyze a pair directly:

```sh
/Users/jhuber/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 app.py --analyze \
  "/Users/jhuber/Downloads/Set 16 - A - white up IMG_6709.JPG" \
  "/Users/jhuber/Downloads/Set 16 - B - white up IMG_6710.JPG"
```

## Recognition Contract

- `U=white`, `D=yellow`, `R=red`, `L=orange`, `F=green`, `B=blue`.
- Center opposites are fixed: white/yellow, red/orange, green/blue.
- Image A must show the `U/white` center face; the Rubik's logo is allowed.
- Image B must show the flipped `D/yellow` center face.
- The two images must provide all four side-face centers across the pair.
- All six face grids are assembled from the two visible 3-face views.
- Successful states are exactly 54 characters, contain 9 of each URFDLB letter, and pass cubie legality checks.

## Notes

This first implementation is built to be debuggable and calibration-friendly. It avoids silent
guesses and exposes overlays so the detection thresholds can be tuned against more real sample
pairs.
