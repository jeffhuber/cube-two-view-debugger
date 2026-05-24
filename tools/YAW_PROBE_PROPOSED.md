# Yaw probe — proposed values for full_corner_ground_truth.json

Sampled the right-slot face center sticker in each of the 12 
full-corner-truth rows and mapped its color → WCA face → yaw via 
`tools/corner_conventions.wca_face_by_slot`.

**For user confirmation before committing.** Spot-check rows where 
the classified color seems off (lighting / sticker glare can 
confuse the classifier).

| Key | Image | Sampled RGB | Classified | Proposed yaw | Center px |
|---|---|---|---|---:|---|
| `20_A` | `Set 20 - A - white up.JPG` | `(96, 48, 40)` | `red` | 0 | `(1965, 2271)` |
| `20_B` | `Set 20 - B - white up.JPG` | `(61, 70, 86)` | `blue` | 0 | `(1984, 2386)` |
| `38_A` | `Set 38 - A - white up IMG_7155.JPG` | `(46, 65, 108)` | `blue` | 1 | `(1955, 2177)` |
| `38_B` | `Set 38 - B - white up IMG_7156.JPG` | `(158, 79, 38)` | `orange` | 1 | `(1941, 2005)` |
| `40_A` | `Set 40 - A - white up IMG_7159.JPG` | `(127, 38, 30)` | `red` | 0 | `(1918, 2261)` |
| `40_B` | `Set 40 - B - white up IMG_7160.JPG` | `(46, 65, 115)` | `blue` | 0 | `(2000, 2076)` |
| `41_A` | `Set 41 - A - white up IMG_7161.JPG` | `(35, 54, 112)` | `blue` | 1 | `(2052, 2250)` |
| `41_B` | `Set 41 - B - white up IMG_7162.JPG` | `(176, 94, 48)` | `orange` | 1 | `(2048, 2427)` |
| `43_A` | `Set 43 - A - white up IMG_7165.JPG` | `(58, 84, 141)` | `blue` | 1 | `(1989, 1983)` |
| `43_B` | `Set 43 - B - white up IMG_7166.JPG` | `(193, 104, 56)` | `orange` | 1 | `(2032, 2219)` |
| `45_A` | `Set 45 - A - white up IMG_7169.JPG` | `(153, 76, 65)` | `red` | 0 | `(1947, 2303)` |
| `45_B` | `Set 45 - B - white up IMG_7170.JPG` | `(70, 92, 138)` | `blue` | 0 | `(2095, 2248)` |

## Mapping reference

Right-slot WCA face at each yaw_quarter_turns:

| Yaw | Image A right slot | Image B right slot |
|---:|---|---|
| 0 | `R` (red) | `B` (blue) |
| 1 | `B` (blue) | `L` (orange) |
| 2 | `L` (orange) | `F` (green) |
| 3 | `F` (green) | `R` (red) |
