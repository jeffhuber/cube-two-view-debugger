# `tools/` index

> Status-tagged inventory of every script and report in this directory.
> Start at [`STATE_OF_THE_WORLD.md`](STATE_OF_THE_WORLD.md), then come
> here when you need a specific tool or want to know whether a path is
> still active.

## Status legend

| Tag | Meaning |
|---|---|
| 🟢 **active** | Current production / canonical eval / actively maintained |
| 🟡 **historical** | Documents a negative result or superseded experiment. Don't delete — institutional memory prevents re-discovery. |
| 🔴 **deprecated** | Replaced by something newer. Don't use; left in place to avoid breaking external links. |
| 🔵 **infra** | Bot / automation / CI plumbing |

## Documentation (Markdown)

### 🟢 Active — read these first

| File | Purpose |
|---|---|
| [`STATE_OF_THE_WORLD.md`](STATE_OF_THE_WORLD.md) | Entry point. Current architecture map + phased roadmap. |
| [`FIRST_PRINCIPLES_RECOGNIZER_DESIGN.md`](FIRST_PRINCIPLES_RECOGNIZER_DESIGN.md) | Target architecture + policy bar for first-principles work. |
| [`POST_218_BASELINE_AND_TAXONOMY.md`](POST_218_BASELINE_AND_TAXONOMY.md) | Decision spine. Numbers, taxonomy, recommended next sequence. (Global-model side.) |
| [`PHASE_1_CV_LOCAL_BASELINE.md`](PHASE_1_CV_LOCAL_BASELINE.md) | Companion baseline on cv-local side. Headline: cv-local face-quads are not geometrically consistent (90% structural fit-fail). |
| [`FAILURE_TAXONOMY.md`](FAILURE_TAXONOMY.md) | Single source of truth for failure-mode categories. |
| [`BENCHMARK_INDEX.md`](BENCHMARK_INDEX.md) | Which fixture/report/script answers which question. |
| [`README.md`](README.md) | This file. |

### 🟢 Active — system / pipeline specs

| File | What it documents |
|---|---|
| [`GLOBAL_CUBE_MODEL.md`](GLOBAL_CUBE_MODEL.md) | Global cube model implementation spec. |
| [`NEAR_FAR_PHASE_REPORT.md`](NEAR_FAR_PHASE_REPORT.md) | Near/far phase ambiguity framing + current detector. (formerly CHIRALITY_DETECTION_REPORT.md) |
| [`INTERIOR_BEZEL_DETECTION.md`](INTERIOR_BEZEL_DETECTION.md) | Interior bezel detector (initializer for global model). |
| [`RECTIFY_FACES.md`](RECTIFY_FACES.md) | Per-face rectification — when it's safe to use. |
| [`AUTO_GEOMETRY_PIPELINE.md`](AUTO_GEOMETRY_PIPELINE.md) | Auto-geometry mask-path overview. |
| [`HYBRID_PIPELINE_GEOMETRY.md`](HYBRID_PIPELINE_GEOMETRY.md) | Hybrid pipeline architecture. |
| [`AMG_FACE_REFINER.md`](AMG_FACE_REFINER.md) | AMG face refiner. |
| [`CLEAN_LABEL_PIPELINE.md`](CLEAN_LABEL_PIPELINE.md) | Clean-label color sample pipeline. |
| [`REMBG_PROPOSERS.md`](REMBG_PROPOSERS.md) | rembg-based proposers (silhouette → anchors). |
| [`CV_LOCAL_IMPROVEMENTS.md`](CV_LOCAL_IMPROVEMENTS.md) | cv-local production-side improvements roadmap. |
| [`SYNTHETIC_CORPUS.md`](SYNTHETIC_CORPUS.md) | Synthetic-corpus rendering for training/eval. |

### 🟡 Historical — negative results & superseded experiments

| File | Outcome |
|---|---|
| `AXIS_RAY_VERTEX_REFINEMENT_V0_REPORT.md` | Hand-tuned axis-ray refinement — not safe to enable. |
| `BEZEL_DISCONTINUITY_JOIN_REPORT.md` | Bezel-line joining via discontinuity — superseded. |
| `CUBE_MESH_ANCHOR_V0_EASY_CORPUS_REPORT.md` | Early cube-mesh-anchor fitter — superseded by global model. |
| `EXPANDED_VERTEX_LOCALIZER_V0_REPORT.md` | Patch-based vertex localizer variant — narrow improvement, didn't graduate. |
| `FOUNDATION_SEGMENTATION_BAKEOFF_V0_REPORT.md` | SAM3 vs rembg silhouette bakeoff — SAM3 didn't materially beat rembg. |
| `GEOMETRY_FIRST_FACE_SPLIT_V0_REPORT.md` | Geometry-first face split experiment — superseded. |
| `GLOBAL_CUBE_MODEL_V0_REPORT.md` | First global-model snapshot. Superseded by `POST_218_BASELINE_AND_TAXONOMY.md`. |
| `GLOBAL_CUBE_MODEL_V0_EASY_CORPUS_REPORT.md` | V0 on the easy corpus. Superseded. |
| `GLOBAL_CUBE_MODEL_V01_EASY_CORPUS_REPORT.md` | V0.1 on the easy corpus. Superseded. |
| `GLOBAL_CUBE_MODEL_V01_EASY_WEAK_REPORT.md` | V0.1 on the easy-weak subset. Superseded. |
| `HEX_FITTER_FAILURE_TAXONOMY.md` | Earlier hex-fitter failure modes — partly subsumed by `FAILURE_TAXONOMY.md`. |
| `KNN_VERTEX_LOCALIZER_V0_REPORT.md` | KNN-based vertex localizer — didn't graduate at safe coverage. |
| `LEARNED_VERTEX_LOCALIZER_V0_REPORT.md` | Sklearn Ridge-on-15D vertex regressor — smoothed toward mean. |
| `OVERLAY_DISCONTINUITY_REPORT.md` | Overlay discontinuity analysis. |
| `OVERLAY_FEEDBACK_REPORT.md` | Overlay-feedback corpus analysis. |
| `PATCH_JUNCTION_VERTEX_LOCALIZER_V0_REPORT.md` | Patch+junction vertex localizer — narrow pocket, didn't graduate. |
| `RAW_PATCH_VERTEX_LOCALIZER_V0_REPORT.md` | Raw-patch vertex localizer — negative. |
| `RAY_START_VERTEX_REFINEMENT_V0_REPORT.md` | Ray-start refinement variant — negative. |
| `SAM3_MASK_EXPORT_V0_REPORT.md` | SAM3 mask export pipeline. Bakeoff group — off the table for more iteration. |
| `SAM3_MLX_BOX_GUIDED_PROMPT_BAKEOFF_V0_REPORT.md` | SAM3 box-guided prompt bakeoff. 0/16 top-3 vertex recall — do not wire. |
| `SAM3_MLX_CURRENT_PROMPT_BAKEOFF_V0_REPORT.md` | SAM3 text-prompt bakeoff. Same negative result on vertex recall. |
| `SAM3_WHOLE_CUBE_SILHOUETTE_BAKEOFF_V0_REPORT.md` | SAM3 whole-cube silhouette vs rembg. Mean/median wins but regression tail — alternate hypothesis only. |
| `TRIHEDRAL_AXIS_FIT_V0_REPORT.md` | Trihedral axis fit — negative. |
| `TRIHEDRAL_JUNCTION_EXTRACTION_V0_REPORT.md` | Explicit junction extraction — negative. |
| `TRIHEDRAL_JUNCTION_EXPANDED_V0_REPORT.md` | Expanded junction-extraction variant — also negative. |
| `VERTEX_AXIS_HUMAN_FEEDBACK_V0_REPORT.md` | Earlier vertex/axis human-feedback round. |
| `VERTEX_AXIS_SOURCE_SELECTION_V0_REPORT.md` | Vertex/axis source-selection probe — confidence is the blocker. |
| `VERTEX_AXIS_ACTIVE_LEARNING_FEEDBACK_V0_REPORT.md` | Active-learning feedback iteration on vertex/axis labels. |
| `VERTEX_AXIS_ACTIVE_LEARNING_QUEUE_V0_REPORT.md` | Active-learning queue design — picks next photos to label based on model disagreement. |
| `VERTEX_CANDIDATE_RANKER_V0_REPORT.md` | Multi-candidate vertex ranker — top-1 OK but calibrated abstention is the bottleneck. |
| `VERTEX_FITTER_ASSISTED_RANKER_V0_REPORT.md` | Fitter-assisted variant of the ranker — same conclusion. |
| `VERTEX_HYPOTHESIS_ENSEMBLE_V0_REPORT.md` | Ensemble over vertex hypotheses — no safe lift. |
| `VERTEX_CANDIDATE_SOURCE_PROBE_REPORT.md` | Candidate-source probe — useful for future learned ranker. |
| `VERTEX_POINT_CANDIDATES_EASY_CORPUS_REPORT.md` | Vertex-point candidate proposers on easy corpus — early exploration. |
| `VERTEX_POINT_HUMAN_FEEDBACK_REPORT.md` | Human feedback on vertex-point candidates — superseded by axis-labeled gallery. |

## Python scripts

### 🟢 Active — benchmark / regression-gate

| Script | What it does |
|---|---|
| **`baseline_post_218.py`** | **THE global-model benchmark.** Runs global model on 58-case labeled gallery, categorizes, emits JSON + report. Supports `--diff` for row-level regression checks. |
| **`baseline_cv_local.py`** | **THE cv-local benchmark.** Same 58 cases; derives (vertex, 3 near, 3 far) from cv-local's face-quads via union-find clustering. JSON schema uniform with `baseline_post_218.py` so `--diff` works across both sides. Headline: 90% structural fit-fail. |
| `evaluate_axis_ground_truth.py` | Per-axis bearing/length error against a candidate model output. |
| `evaluate_hybrid_pipeline.py` | End-to-end production-recognizer accuracy on hard-case corpus. |
| `evaluate_color_classifier_modes.py` | Color classifier mode comparison. |
| `evaluate_two_view_consistency.py` | A+B center consistency check. |
| `evaluate_per_sticker_confidence.py` | Per-sticker confidence calibration. |
| `evaluate_mask_pipeline.py` | Mask-path end-to-end eval. |

### 🟢 Active — global cube model core

| Script | What it does |
|---|---|
| **`global_cube_model.py`** | Global cube model fitter. Phase detection + auto-correction. |
| `interior_bezel_detection.py` | Interior bezel detector (initializer for global model). |
| `render_global_cube_model_v0_overlays.py` | Visualize global model output on a photo. |

### 🟢 Active — labels and labeling

| Script | What it does |
|---|---|
| `build_axis_labeling_gallery.py` | Generate the gallery HTML for user vertex+axis labeling. |
| `active_vertex_axis_label_queue_v0.py` | Active-learning queue for picking next photos to label. |
| `vertex_axis_label_server.py` | Server for the labeling UI. |
| `vertex_axis_feedback.py` | Per-label feedback collection. |
| `label_geometry_baseline.py` | Baseline geometry labeling tool. |
| `propose_geometry_labels.py` | Propose geometry labels for human review. |

### 🟢 Active — production recognizer support

| Script | What it does |
|---|---|
| `recognize_pair.py` | Recognize an A+B pair (production-style). |
| `audit_recognition_pair.py` | Audit production output against ground truth. |
| `extract_color_samples.py` | Extract per-sticker color samples for training. |
| `extract_clean_dataset.py` | Curate a clean labeled dataset. |
| `rectify_faces.py` | Per-face rectification helper. |
| `sample_stickers_from_hull.py` | Sample sticker colors from hexagon hull. |
| `train_color_classifier.py` | Train the color classifier. |
| `regenerate_knn_color_data.py` | Regenerate KNN color-data cache. |

### 🔵 Infra

| Script | What it does |
|---|---|
| `devin_audit_bridge.py` | Bridge from PR events → Devin audit dispatch. **Mirrored byte-identical with `cube-snap/tools/`.** |
| `devin_audit_labeler.py` | Apply `devin-audit-done` / `devin-audit-blocked` labels based on Devin's audit trailer. **Mirrored byte-identical with `cube-snap/tools/`.** |
| `view_photo.py` | EXIF-correct + view a photo (works around Read tool's raw-pixel quirk). |

### 🟡 Historical — vertex-localizer probes (negative results, kept for memory)

| Script | What we learned |
|---|---|
| `vertex_candidate_ranker_v0.py` | Multi-candidate ranker — top-1 is okay but calibrated abstention is the bottleneck. |
| `vertex_fitter_assisted_ranker_v0.py` | Fitter-assisted variant — same conclusion. |
| `vertex_hypothesis_ensemble_v0.py` | Ensemble of hypotheses — no safe lift. |
| `vertex_candidate_source_probe.py` | Probe over candidate sources — useful for future learned ranker. |
| `vertex_axis_source_selection_v0.py` | Source-selection over labeled feedback. |
| `vertex_point_candidates.py` | Earlier vertex-point candidate proposer. |
| `vertex_point_feedback.py` | Per-candidate vertex-point human feedback collection. |
| `axis_ray_vertex_refinement_v0.py` | Hand-tuned ray-darkness refinement — not safe. |
| `ray_start_vertex_refinement_v0.py` | Ray-start refinement — negative. |
| `learned_vertex_localizer_v0.py` | Sklearn-Ridge-on-15D — smooths toward mean. |
| `knn_vertex_localizer_v0.py` | KNN vertex localizer — narrow pocket. |
| `expanded_vertex_localizer_v0.py` | Expanded patch features — narrow. |
| `raw_patch_vertex_localizer_v0.py` | Raw image patch ranker — negative. |
| `patch_junction_vertex_localizer_v0.py` | Patch+junction combination — narrow. |
| `trihedral_junction_extraction_v0.py` | Explicit junction extractor — negative. |
| `trihedral_junction_expanded_v0.py` | Expanded junction-extraction variant — also negative. |
| `trihedral_axis_fit_v0.py` | Trihedral axis fitting — superseded by global model. |
| `cube_mesh_anchor_fitter_v0.py` | Early mesh-anchor fitter — superseded. |
| `global_cube_model_v0.py` | First global-model implementation. Superseded by `global_cube_model.py`. |
| `train_vertex_regressor.py` | Trained vertex regressor — superseded by labeled-data approach. |

### 🟡 Historical — segmentation bakeoffs (off the table)

| Script | What we learned |
|---|---|
| `foundation_segmentation_bakeoff_v0.py` | SAM3 vs rembg — SAM3 didn't beat. |
| `sam3_whole_cube_silhouette_bakeoff_v0.py` | SAM3 silhouette bakeoff. |
| `export_sam3_masks_v0.py` | SAM3 mask export utility (v0). |
| `extract_sam3_masks.py` | SAM3 mask extraction utility. |
| `geometry_first_face_split_v0.py` | Geometry-first face split — superseded by hybrid pipeline. |

### 🟡 Historical — diagnostic probes (one-shot)

| Script | What it answered |
|---|---|
| `probe_corpus.py` / `probe_hard_cases.py` / `probe_candidate_guards.py` | Corpus probes for failure-mode analysis. |
| `probe_bezel_discontinuity_join.py` / `probe_overlay_discontinuity.py` | Bezel/overlay discontinuity probes. |
| `diagnose_background.py` / `diagnose_grid_rejection.py` / `diagnose_per_sticker_errors.py` | One-shot diagnostics. |
| `inspect_cube_isolation.py` | Cube isolation inspection. |
| `equalize_faces.py` / `evaluate_equalize_lift.py` | Color equalization experiment — no lift. |
| `evaluate_auto_geometry.py` | Auto-geometry pipeline eval (early). |
| `evaluate_geometry_labels.py` | Geometry-label round-trip eval. |
| `prepare_label_calibration.py` / `score_label_calibration.py` | Label calibration utilities. |
| `generate_hex_fitter_walkthroughs.py` | Hex-fitter visual walkthroughs (May-18 review). |
| `overlay_feedback.py` | Overlay feedback collection. |
| `vertex_label_server.py` | Earlier vertex-only labeling server (superseded by `vertex_axis_label_server.py`). |

### 🟢 Active — rendering / overlays

| Script | What it does |
|---|---|
| `render_synthetic_cube.py` | Render a synthetic cube image. |
| `render_hybrid_overlays.py` | Render hybrid-pipeline overlays. |
| `render_cube_mesh_anchor_overlays.py` | Render cube-mesh-anchor overlays. |
| `render_vertex_point_candidates.py` | Render vertex-candidate overlays. |
| `run_amg_face_refinement.py` | Run AMG face refinement. |
| `amg_face_refiner.py` | AMG refiner implementation. |

### Tests

| Script | What it tests |
|---|---|
| `test_global_cube_model.py` | Global cube model unit tests. |
| `test_interior_bezel.py` | Interior bezel detector unit tests. |
| (under `tests/`) | Project-wide test suite. |

## How to keep this index complete

Every `tools/*.py` and `tools/*.md` (except this README) must appear
in this file. Quick check:

```bash
python3 -c "from pathlib import Path; r = Path('tools/README.md').read_text(); \
  miss = [f.name for f in sorted(Path('tools').glob('*.md')) + sorted(Path('tools').glob('*.py')) \
          if f.name != 'README.md' and f.name not in r]; \
  print('missing:', miss or 'none')"
```

When adding a new script or report:
- 🟢 active → put it in the relevant active section.
- 🟡 historical (negative result / superseded experiment) → drop it in
  the matching historical section. Don't delete; institutional memory.
- 🔵 infra → add to the infra section.

Filenames must appear verbatim — don't use slash-grouped suffix forms
(e.g., `FOO_V0_REPORT.md / _V01_REPORT.md`) because a substring check
won't recognize the grouped variant as "present".

## Conventions

- **Don't delete historical scripts/docs.** Even when a path is
  superseded, the negative-result documentation prevents
  re-discovery. Move to 🟡 historical here instead.
- **`_v0.py` / `_V0_REPORT.md` suffix** = experiment iteration. After
  multiple iterations, consider whether the latest deserves a non-v0
  name; otherwise keep the suffix to make the "this is exploratory"
  status visible.
- **Cross-link from one doc to another** when answering the same
  question from different angles. Most readers find the right doc by
  following links, not by directory listing.
- **`probe_*.py`** = one-shot diagnostic, often outdated. Don't run
  these for current decisions without checking the date in the
  associated report.
- **`evaluate_*.py`** = canonical eval. Stable interface, runnable on
  current main.
- **`render_*.py`** = visualization. Output goes to `runs/` or
  `/tmp/`.
