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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const imageA = document.querySelector("#imageA").files[0];
  const imageB = document.querySelector("#imageB").files[0];
  if (!imageA || !imageB) {
    renderMessage("Upload both images before recognizing.");
    return;
  }

  const body = new FormData();
  body.append("imageA", imageA);
  body.append("imageB", imageB);

  renderMessage("Processing images...");
  stateEl.textContent = "";
  overlaysEl.innerHTML = "";

  const response = await fetch("/api/recognize", { method: "POST", body });
  const payload = await response.json();
  renderResult(payload);
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
});

copyButton.addEventListener("click", async () => {
  const state = stateEl.textContent.trim();
  if (state.length === 54) {
    await navigator.clipboard.writeText(state);
    copyButton.textContent = "Copied";
    setTimeout(() => (copyButton.textContent = "Copy"), 900);
  }
});

for (const name of ["dragenter", "dragover"]) {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
}

for (const name of ["dragleave", "drop"]) {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
}

dropZone.addEventListener("drop", (event) => {
  const files = Array.from(event.dataTransfer.files || []).filter((file) => file.type.startsWith("image/"));
  if (!files.length) return;
  const dataTransfer = new DataTransfer();
  for (const file of files) dataTransfer.items.add(file);
  batchImages.files = dataTransfer.files;
});

function renderMessage(message) {
  statusEl.textContent = message;
  detailsEl.textContent = "";
  copyButton.disabled = true;
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

  overlaysEl.innerHTML = "";
  for (const [label, src] of Object.entries(payload.overlays || {})) {
    if (!src) continue;
    const figure = document.createElement("figure");
    const caption = document.createElement("figcaption");
    const img = document.createElement("img");
    caption.textContent = label;
    img.src = src;
    img.alt = `${label} recognition overlay`;
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
