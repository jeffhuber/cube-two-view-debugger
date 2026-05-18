// Rubik Two-View Recognizer — debugger UI.
//
// Three surfaces:
//   1. Single Pair  — unified drop zone + optional manual A/B override + optional set name.
//   2. Batch        — multi-pair drop zone with ground-truth CSV/TSV/JSON.
//   3. Recent Runs  — list of saved runs; click to reload without re-running.
//
// Pairing for the Single Pair drop zone mirrors the server-side
// pair_image_uploads() in rubik_recognizer/dataset.py: filenames with
// an "A"/"B" marker auto-detect; otherwise the order they were
// added.

const APP_VERSION = "0.0.1";

const form = document.querySelector("#recognizeForm");
const batchForm = document.querySelector("#batchForm");
const statusEl = document.querySelector("#status");
const stateEl = document.querySelector("#state");
const detailsEl = document.querySelector("#details");
const overlaysEl = document.querySelector("#overlays");
const copyButton = document.querySelector("#copyState");
const batchImages = document.querySelector("#batchImages");
const groundTruth = document.querySelector("#groundTruth");
const dropZone = document.querySelector("#dropZone");
const batchRows = document.querySelector("#batchRows");
const batchReport = document.querySelector("#batchReport");
const singleDropZone = document.querySelector("#singleDropZone");
const singleImages = document.querySelector("#singleImages");
const singleDropZoneFiles = document.querySelector("#singleDropZoneFiles");
const singleSetId = document.querySelector("#singleSetId");
const manualImageA = document.querySelector("#imageA");
const manualImageB = document.querySelector("#imageB");
const recentRunsRows = document.querySelector("#recentRunsRows");
const recentRunsRefresh = document.querySelector("#recentRunsRefresh");
const buildFooter = document.querySelector("#buildFooter");

// Detect an "A" or "B" marker in a filename — same logic the Python
// `_image_marker()` uses but kept locally so we can label the drop
// zone before the server sees the files. Keep IN SYNC with the
// server-side patterns — if the server adds a recognition rule, add
// it here too, or single-pair-form auto-pairing will silently swap
// labels for filenames the server understands but the UI doesn't.
// Parity is pinned by tests/test_dataset.py::test_image_marker_patterns_match_js_detector.
// Server uses Path(filename).stem (drops trailing extension); we
// strip the last dot-extension to match.
function detectABMarker(name) {
  const stem = name.replace(/\.[^./]+$/, "");
  // Pattern 1: standalone A/B token surrounded by \s_- (or string edges).
  //   server: (?i)(?:^|[\s_-])([ab])(?:[\s_-]|$)
  const m1 = /(?:^|[\s_-])([ab])(?:[\s_-]|$)/i.exec(stem);
  if (m1) return m1[1].toUpperCase();
  // Pattern 2: imageA / imageB / image A / image B.
  //   server: (?i)\bimage\s*([ab])\b
  const m2 = /\bimage\s*([ab])\b/i.exec(stem);
  if (m2) return m2[1].toUpperCase();
  return null;
}

// Order [imageA, imageB] given 1-2 files. Falls back to drop order if
// markers are missing or ambiguous, and returns null if a single file
// is provided (caller is responsible for prompting for the second).
function pairForRecognize(files) {
  const arr = Array.from(files || []).filter((f) => f.type.startsWith("image/"));
  if (arr.length === 0) return { imageA: null, imageB: null, autoPaired: false };
  if (arr.length === 1) return { imageA: arr[0], imageB: null, autoPaired: false };

  // Take the first two if more than 2 were dropped — same convention
  // as Single Pair's intent. Multi-pair belongs in the Batch form.
  const [first, second] = arr;
  const firstMarker = detectABMarker(first.name);
  const secondMarker = detectABMarker(second.name);

  if (firstMarker === "A" && secondMarker === "B") {
    return { imageA: first, imageB: second, autoPaired: true };
  }
  if (firstMarker === "B" && secondMarker === "A") {
    return { imageA: second, imageB: first, autoPaired: true };
  }
  // Markers absent or ambiguous: pair by drop order.
  return { imageA: first, imageB: second, autoPaired: false };
}

// Render the file labels under the drop zone so the user sees what
// the form is going to send.
function renderSingleDropZoneFiles(files) {
  singleDropZoneFiles.innerHTML = "";
  const { imageA, imageB, autoPaired } = pairForRecognize(files);
  if (!imageA && !imageB) return;
  const lines = [];
  if (imageA) lines.push(`A: ${imageA.name}`);
  if (imageB) lines.push(`B: ${imageB.name}`);
  if (autoPaired) lines.push("(auto-paired from filename markers)");
  else if (imageA && imageB) lines.push("(paired by drop order)");
  for (const line of lines) {
    const div = document.createElement("div");
    div.textContent = line;
    singleDropZoneFiles.append(div);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  // Manual override wins if both manual fields are populated. Otherwise
  // use the unified drop zone, pairing as needed.
  let imageA = manualImageA.files[0] || null;
  let imageB = manualImageB.files[0] || null;
  if (!imageA || !imageB) {
    const paired = pairForRecognize(singleImages.files);
    imageA = imageA || paired.imageA;
    imageB = imageB || paired.imageB;
  }
  if (!imageA || !imageB) {
    renderMessage("Drop 2 photos (or use the manual A/B override) before recognizing.");
    return;
  }

  const body = new FormData();
  body.append("imageA", imageA);
  body.append("imageB", imageB);
  const setId = singleSetId.value.trim();
  if (setId) body.append("setId", setId);

  renderMessage("Processing images...");
  stateEl.textContent = "";
  overlaysEl.innerHTML = "";

  const response = await fetch("/api/recognize", { method: "POST", body });
  const payload = await response.json();
  renderResult(payload);
  // Refresh recent runs so the new one shows up immediately.
  fetchRecentRuns().catch(() => {});
});

batchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const files = Array.from(batchImages.files || []);
  if (!files.length) {
    renderMessage("Select a batch of images first.");
    return;
  }

  const body = new FormData();
  for (const file of files) body.append("images", file);
  if (groundTruth.files[0]) body.append("groundTruth", groundTruth.files[0]);

  renderMessage(`Processing ${files.length} images...`);
  batchRows.innerHTML = "";
  batchReport.hidden = true;

  const response = await fetch("/api/recognize-batch", { method: "POST", body });
  const payload = await response.json();
  renderBatch(payload);
  fetchRecentRuns().catch(() => {});
});

copyButton.addEventListener("click", async () => {
  const state = stateEl.textContent.trim();
  if (state.length === 54) {
    await navigator.clipboard.writeText(state);
    copyButton.textContent = "Copied";
    setTimeout(() => (copyButton.textContent = "Copy"), 900);
  }
});

// Wire up the unified Single Pair drop zone + the existing Batch drop zone
// with identical drag/drop visual feedback.
function setupDropZone(zone, fileInput, onFilesChanged) {
  for (const name of ["dragenter", "dragover"]) {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add("is-dragging");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.remove("is-dragging");
    });
  }
  zone.addEventListener("drop", (event) => {
    const dropped = Array.from(event.dataTransfer.files || []).filter((file) => file.type.startsWith("image/"));
    if (!dropped.length) return;
    const dataTransfer = new DataTransfer();
    for (const file of dropped) dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
    if (onFilesChanged) onFilesChanged(fileInput.files);
  });
  // Also re-render when the user picks files via the native chooser.
  fileInput.addEventListener("change", () => {
    if (onFilesChanged) onFilesChanged(fileInput.files);
  });
}

setupDropZone(singleDropZone, singleImages, renderSingleDropZoneFiles);
setupDropZone(dropZone, batchImages, null);

// Convenience: a set-name typed into the form is persistent so the
// user can switch between sets without losing the label. Doesn't
// persist across page reloads (intentionally — different debug
// sessions should start clean).

function renderMessage(message) {
  statusEl.textContent = message;
  detailsEl.textContent = "";
  copyButton.disabled = true;
}

// Parse the set-id portion out of a run id like "20260512-184522-...-set-21-set-21"
// or "20260512-184522-...-set-21". The set id is the trailing dash-separated
// chunk after the trailing 6-digit microsecond field.
function setIdFromRunId(runId) {
  if (!runId) return null;
  // Strip the leading timestamp tokens: YYYYMMDD-HHMMSS-<usec>
  const m = /^\d{8}-\d{6}-\d+-(.*)$/.exec(runId);
  if (!m) return runId;
  // The set id is sometimes duplicated (set-21-set-21); collapse.
  const tail = m[1];
  const halves = tail.split("-");
  if (halves.length >= 2 && halves.length % 2 === 0) {
    const half = halves.length / 2;
    const a = halves.slice(0, half).join("-");
    const b = halves.slice(half).join("-");
    if (a === b) return a;
  }
  return tail;
}

function basenameFromPath(path) {
  if (!path) return null;
  const i = path.lastIndexOf("/");
  return i >= 0 ? path.slice(i + 1) : path;
}

function renderResult(payload) {
  statusEl.textContent = payload.status === "success" ? "Recognized" : "Rejected";
  stateEl.textContent = payload.state || "";
  copyButton.disabled = !payload.state;
  detailsEl.textContent = JSON.stringify(
    {
      runId: payload.runId,
      reason: payload.reason,
      confidence: payload.confidence,
      failedChecks: payload.failedChecks,
      candidates: payload.candidates,
      evaluation: payload.evaluation,
      artifacts: payload.artifacts,
      imageAAssignments: payload.imageAAssignments,
      imageBAssignments: payload.imageBAssignments,
      imageA: payload.imageA,
      imageB: payload.imageB,
    },
    null,
    2
  );

  // Surface set id + original filename in the overlay captions so cut-
  // and-paste images and side-by-side debugging stay unambiguous.
  // payload.runId encodes the set id; artifacts.imageA / artifacts.imageB
  // carry the original uploaded filenames in their server paths.
  const setId = setIdFromRunId(payload.runId) || "";
  const artifacts = payload.artifacts || {};
  overlaysEl.innerHTML = "";
  for (const [label, src] of Object.entries(payload.overlays || {})) {
    if (!src) continue;
    const figure = document.createElement("figure");
    const caption = document.createElement("figcaption");
    const img = document.createElement("img");
    const filename = basenameFromPath(artifacts[label]);
    const parts = [];
    if (setId) parts.push(setId);
    parts.push(label);
    if (filename) parts.push(filename);
    caption.textContent = parts.join(" · ");
    img.src = src;
    img.alt = `${parts.join(" ")} recognition overlay`;
    figure.append(caption, img);
    overlaysEl.append(figure);
  }
}

function renderBatch(payload) {
  statusEl.textContent = `${payload.successes || 0}/${payload.totalPairs || 0} recognized`;
  detailsEl.textContent = JSON.stringify(
    {
      batchId: payload.batchId,
      batchUrl: payload.batchUrl,
      totalPairs: payload.totalPairs,
      successes: payload.successes,
      rejections: payload.rejections,
      truthCount: payload.truthCount,
      exactMatches: payload.exactMatches,
      unpaired: payload.unpaired,
    },
    null,
    2
  );

  if (payload.batchUrl) {
    batchReport.href = payload.batchUrl;
    batchReport.hidden = false;
  }

  batchRows.innerHTML = "";
  for (const item of payload.results || []) {
    const row = document.createElement("tr");
    const evaluation = item.evaluation || {};
    const exact = evaluation.available ? (evaluation.exact ? "yes" : `no (${evaluation.hamming})`) : "";
    row.append(
      cell(item.setId || ""),
      cell(item.status || ""),
      cell(exact),
      cell(item.confidence ?? ""),
      codeCell(item.state || ""),
      linkCell("run", item.runUrl)
    );
    batchRows.append(row);
  }
}

// Recent runs panel — populated on page load and refreshed after every
// successful recognize. Clicking a row reloads that run's saved JSON
// into the result section without re-running the recognizer.
async function fetchRecentRuns() {
  const response = await fetch("/api/runs");
  if (!response.ok) return;
  const payload = await response.json();
  renderRecentRuns(payload.runs || []);
}

function renderRecentRuns(runs) {
  recentRunsRows.innerHTML = "";
  for (const run of runs) {
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    tr.setAttribute("role", "button");
    tr.classList.add("recentRunsRow");
    const score = run.evaluation && run.evaluation.available
      ? `${run.evaluation.matched}/54`
      : "";
    const status = run.status || "";
    tr.append(
      cell(run.setId || ""),
      cell(status),
      cell(run.recognitionCategory || ""),
      cell(score),
      cell(formatCreatedAt(run.createdAt)),
      linkCell("open", run.runUrl)
    );
    const loader = () => loadRunIntoResult(run);
    tr.addEventListener("click", (event) => {
      // Don't intercept clicks on the link cell so the user can still
      // open the raw summary.json in a new tab.
      if (event.target.tagName === "A") return;
      loader();
    });
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        loader();
      }
    });
    recentRunsRows.append(tr);
  }
  if (runs.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.textContent = "No runs yet — run a recognition above.";
    td.style.color = "var(--muted)";
    td.style.fontStyle = "italic";
    td.style.padding = "10px 8px";
    tr.append(td);
    recentRunsRows.append(tr);
  }
}

function formatCreatedAt(iso) {
  if (!iso) return "";
  const dt = new Date(iso);
  if (Number.isNaN(dt.getTime())) return iso;
  return dt.toLocaleString();
}

function shortSha(sha) {
  if (!sha) return "unknown";
  return String(sha).slice(0, 7);
}

function renderBuildFooter(diag) {
  if (!buildFooter) return;
  const git = (diag && diag.git) || {};
  const python = (diag && diag.python) || {};
  const libraries = (diag && diag.libraries) || {};
  const warnings = Array.isArray(diag && diag.warnings) ? diag.warnings : [];
  const sha = `${shortSha(git.sha)}${git.dirty === true ? "-dirty" : ""}`;
  const branch = git.branch || "detached";
  const parts = [
    `Rubik Two-View Recognizer v${APP_VERSION}`,
    `build ${sha} (${branch})`,
    `Python ${python.version || "unknown"}`,
    `Pillow ${libraries.pillow || "unknown"}`,
    `NumPy ${libraries.numpy || "unknown"}`,
    new Date().toLocaleString(),
  ];
  for (const warning of warnings) parts.push(warning);

  buildFooter.textContent = parts.join(" · ");
  buildFooter.title = [
    `cwd: ${git.cwd || "unknown"}`,
    `python: ${python.executable || "unknown"}`,
    `dirty scope: ${git.dirtyScope || "unknown"}`,
  ].join("\n");
  const footer = buildFooter.closest(".appFooter");
  if (footer) footer.classList.toggle("is-warning", warnings.length > 0);
}

async function fetchBuildFooter() {
  if (!buildFooter) return;
  try {
    const response = await fetch("/api/diag", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const diag = await response.json();
    renderBuildFooter(diag);
  } catch (err) {
    buildFooter.textContent = [
      `Rubik Two-View Recognizer v${APP_VERSION}`,
      "build unavailable",
      new Date().toLocaleString(),
    ].join(" · ");
    buildFooter.title = err && err.message ? err.message : String(err);
    const footer = buildFooter.closest(".appFooter");
    if (footer) footer.classList.add("is-warning");
  }
}

// Load a saved run by fetching its result.json + summary.json + overlay
// files. Mirrors renderResult shape so the experience is identical to
// a fresh recognition.
async function loadRunIntoResult(run) {
  if (!run.runId) return;
  renderMessage(`Loading saved run ${run.runId}...`);
  stateEl.textContent = "";
  overlaysEl.innerHTML = "";
  try {
    const resultResponse = await fetch(`/runs/pairs/${encodeURIComponent(run.runId)}/result.json`);
    if (!resultResponse.ok) {
      renderMessage(`Could not load saved run: HTTP ${resultResponse.status}`);
      return;
    }
    const payload = await resultResponse.json();
    // Saved result.json strips overlays (they live as separate PNG files);
    // patch them back in via the summary's artifact paths so the
    // overlays section still renders.
    const artifacts = payload.artifacts || (run.artifacts || {});
    const overlayPaths = (artifacts && artifacts.overlays) || {};
    payload.overlays = {};
    for (const [label, path] of Object.entries(overlayPaths)) {
      payload.overlays[label] = path;
    }
    payload.runId = payload.runId || run.runId;
    payload.artifacts = artifacts;
    renderResult(payload);
  } catch (err) {
    renderMessage(`Could not load saved run: ${err && err.message ? err.message : err}`);
  }
}

if (recentRunsRefresh) {
  recentRunsRefresh.addEventListener("click", () => {
    fetchRecentRuns().catch(() => {});
  });
}

function cell(value) {
  const td = document.createElement("td");
  td.textContent = value;
  return td;
}

function codeCell(value) {
  const td = document.createElement("td");
  const code = document.createElement("code");
  code.textContent = value;
  td.append(code);
  return td;
}

function linkCell(label, href) {
  const td = document.createElement("td");
  if (href) {
    const link = document.createElement("a");
    link.href = href;
    link.textContent = label;
    td.append(link);
  }
  return td;
}

// Kick off initial recent-runs fetch on page load.
fetchRecentRuns().catch(() => {});
fetchBuildFooter().catch(() => {});
setInterval(() => {
  fetchBuildFooter().catch(() => {});
}, 60000);
