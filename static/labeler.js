// Geometry Labeler — dev-mode annotation surface for cube/body labels.
//
// Saves browser-image-natural coordinates. The server persists the JSON
// under runs/labels/; no recognizer behavior depends on these labels yet.

const appRoot = document.querySelector(".app");
const viewTabs = Array.from(document.querySelectorAll("[data-view]"));
const labelImageInput = document.querySelector("#labelImage");
const labelDropZone = document.querySelector("#labelDropZone");
const labelDropZoneFiles = document.querySelector("#labelDropZoneFiles");
const labelSetId = document.querySelector("#labelSetId");
const labelImageSide = document.querySelector("#labelImageSide");
const labelCanvas = document.querySelector("#labelCanvas");
const labelImageTitle = document.querySelector("#labelImageTitle");
const labelStatus = document.querySelector("#labelStatus");
const labelNotes = document.querySelector("#labelNotes");
const labelJson = document.querySelector("#labelJson");
const labelCopyJson = document.querySelector("#labelCopyJson");
const labelSave = document.querySelector("#labelSave");
const labelSavedLink = document.querySelector("#labelSavedLink");
const labelRows = document.querySelector("#labelRows");
const labelRefresh = document.querySelector("#labelRefresh");
const labelModeButtons = Array.from(document.querySelectorAll("[data-label-mode]"));
const labelFaceButtons = Array.from(document.querySelectorAll("[data-face]"));
const labelFaceGuidance = document.querySelector("#labelFaceGuidance");
const labelUndo = document.querySelector("#labelUndo");
const labelClearActive = document.querySelector("#labelClearActive");
const labelClearAll = document.querySelector("#labelClearAll");

const faceOrder = ["U", "R", "F", "D", "L", "B"];
const facesByImageSide = {
  A: ["U", "R", "F"],
  B: ["D", "L", "B"],
  single: faceOrder,
};
const faceGuidanceBySide = {
  A: "Image A: label the visible faces as U, R, and F.",
  B: "Image B: label the visible faces as D, L, and B.",
  single: "Single image: use the canonical WCA face labels visible in the photo.",
};
const faceColors = {
  U: "#f8f8f2",
  R: "#d83b31",
  F: "#1f9d62",
  D: "#e5c529",
  L: "#ed7d23",
  B: "#2d6cdf",
};

let activeMode = "face";
let activeFace = "U";
let currentImage = null;
let pendingFacePoints = [];
let labels = emptyLabels();

function emptyLabels() {
  return {
    faceQuads: {},
    cubeHull: [],
  };
}

function setView(view) {
  if (!appRoot) return;
  appRoot.dataset.activeView = view;
  for (const button of viewTabs) {
    const active = button.dataset.view === view;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }
}

for (const button of viewTabs) {
  button.addEventListener("click", () => setView(button.dataset.view));
}

function setupLabelDropZone() {
  if (!labelDropZone || !labelImageInput) return;
  for (const name of ["dragenter", "dragover"]) {
    labelDropZone.addEventListener(name, (event) => {
      event.preventDefault();
      labelDropZone.classList.add("is-dragging");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    labelDropZone.addEventListener(name, (event) => {
      event.preventDefault();
      labelDropZone.classList.remove("is-dragging");
    });
  }
  labelDropZone.addEventListener("drop", (event) => {
    const file = Array.from(event.dataTransfer.files || []).find((item) => item.type.startsWith("image/"));
    if (!file) return;
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    labelImageInput.files = dataTransfer.files;
    loadLabelImage(file);
  });
  labelImageInput.addEventListener("change", () => {
    const file = labelImageInput.files && labelImageInput.files[0];
    if (file) loadLabelImage(file);
  });
}

async function loadLabelImage(file) {
  labels = emptyLabels();
  pendingFacePoints = [];
  labelSavedLink.hidden = true;
  labelSavedLink.href = "#";
  setLabelStatus("Loading image...");
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.onload = async () => {
    URL.revokeObjectURL(url);
    currentImage = {
      file,
      element: image,
      name: file.name,
      type: file.type,
      bytes: file.size,
      width: image.naturalWidth,
      height: image.naturalHeight,
      lastModified: file.lastModified,
      sha256: await sha256Hex(file),
    };
    inferSetFields(file.name);
    labelCanvas.width = currentImage.width;
    labelCanvas.height = currentImage.height;
    labelImageTitle.textContent = file.name;
    labelDropZoneFiles.textContent = `${file.name} (${currentImage.width} x ${currentImage.height})`;
    setLabelStatus("Ready");
    drawLabels();
    updateLabelJson();
  };
  image.onerror = () => {
    URL.revokeObjectURL(url);
    setLabelStatus("Could not load image.");
  };
  image.src = url;
}

async function sha256Hex(file) {
  if (!window.crypto || !window.crypto.subtle) return null;
  const digest = await window.crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function inferSetFields(name) {
  if (!labelSetId.value.trim()) {
    const setMatch = /\bset\s*([0-9]+)\b/i.exec(name);
    if (setMatch) labelSetId.value = `Set ${setMatch[1]}`;
  }
  const sideMatch = /(?:^|[\s_-])([ab])(?:[\s_-]|$)/i.exec(name.replace(/\.[^./]+$/, ""));
  if (sideMatch) labelImageSide.value = sideMatch[1].toUpperCase();
  updateFaceControls();
}

function setActiveMode(mode) {
  activeMode = mode;
  pendingFacePoints = [];
  for (const button of labelModeButtons) {
    button.classList.toggle("is-active", button.dataset.labelMode === mode);
  }
  updateFaceControls();
  drawLabels();
  updateLabelJson();
}

function setActiveFace(face) {
  if (!allowedFacesForCurrentSide().includes(face)) return;
  activeFace = face;
  activeMode = "face";
  pendingFacePoints = [];
  updateFaceControls();
  for (const button of labelModeButtons) {
    button.classList.toggle("is-active", button.dataset.labelMode === "face");
  }
  drawLabels();
  updateLabelJson();
}

for (const button of labelModeButtons) {
  button.addEventListener("click", () => setActiveMode(button.dataset.labelMode));
}

for (const button of labelFaceButtons) {
  button.addEventListener("click", () => setActiveFace(button.dataset.face));
}

function allowedFacesForCurrentSide() {
  return facesByImageSide[labelImageSide.value] || faceOrder;
}

function faceQuadsForCurrentSide() {
  const allowedFaces = allowedFacesForCurrentSide();
  const out = {};
  for (const face of allowedFaces) {
    if (labels.faceQuads[face]) out[face] = labels.faceQuads[face];
  }
  return out;
}

function pruneFaceQuadsForCurrentSide() {
  const allowedFaces = allowedFacesForCurrentSide();
  let removed = false;
  for (const face of Object.keys(labels.faceQuads)) {
    if (!allowedFaces.includes(face)) {
      delete labels.faceQuads[face];
      removed = true;
    }
  }
  return removed;
}

function updateFaceControls() {
  const allowedFaces = allowedFacesForCurrentSide();
  if (!allowedFaces.includes(activeFace)) {
    activeFace = allowedFaces[0] || "U";
    pendingFacePoints = [];
  }
  for (const button of labelFaceButtons) {
    const face = button.dataset.face;
    const enabled = activeMode === "face" && allowedFaces.includes(face);
    button.disabled = !enabled;
    button.classList.toggle("is-muted", !enabled);
    button.classList.toggle("is-active", activeMode === "face" && face === activeFace);
  }
  if (labelFaceGuidance) {
    labelFaceGuidance.textContent = activeMode === "hull"
      ? "Cube Hull mode: face labels are ignored; click the outer cube silhouette."
      : (faceGuidanceBySide[labelImageSide.value] || faceGuidanceBySide.single);
  }
}

labelCanvas.addEventListener("click", (event) => {
  if (!currentImage) {
    setLabelStatus("Add an image first.");
    return;
  }
  const point = canvasPoint(event);
  if (activeMode === "face") {
    pendingFacePoints.push(point);
    if (pendingFacePoints.length === 4) {
      labels.faceQuads[activeFace] = pendingFacePoints;
      pendingFacePoints = [];
      setLabelStatus(`${activeFace} face saved.`);
    }
  } else {
    labels.cubeHull.push(point);
    setLabelStatus(`${labels.cubeHull.length} hull point${labels.cubeHull.length === 1 ? "" : "s"}.`);
  }
  drawLabels();
  updateLabelJson();
});

function canvasPoint(event) {
  const rect = labelCanvas.getBoundingClientRect();
  const scaleX = labelCanvas.width / rect.width;
  const scaleY = labelCanvas.height / rect.height;
  return {
    x: roundPoint((event.clientX - rect.left) * scaleX),
    y: roundPoint((event.clientY - rect.top) * scaleY),
  };
}

function roundPoint(value) {
  return Math.round(value * 10) / 10;
}

labelUndo.addEventListener("click", () => {
  if (activeMode === "face" && pendingFacePoints.length) {
    pendingFacePoints.pop();
  } else if (activeMode === "hull" && labels.cubeHull.length) {
    labels.cubeHull.pop();
  } else if (activeMode === "face" && labels.faceQuads[activeFace]) {
    delete labels.faceQuads[activeFace];
  }
  drawLabels();
  updateLabelJson();
});

labelClearActive.addEventListener("click", () => {
  if (activeMode === "face") {
    pendingFacePoints = [];
    delete labels.faceQuads[activeFace];
  } else {
    labels.cubeHull = [];
  }
  drawLabels();
  updateLabelJson();
});

labelClearAll.addEventListener("click", () => {
  labels = emptyLabels();
  pendingFacePoints = [];
  drawLabels();
  updateLabelJson();
});

labelCopyJson.addEventListener("click", async () => {
  await navigator.clipboard.writeText(labelJson.textContent || "");
  labelCopyJson.textContent = "Copied";
  setTimeout(() => (labelCopyJson.textContent = "Copy"), 900);
});

labelSave.addEventListener("click", async () => {
  if (!currentImage) {
    setLabelStatus("Add an image first.");
    return;
  }
  const payload = buildLabelPayload();
  if (!Object.keys(payload.labels.faceQuads).length && !payload.labels.cubeHull.length) {
    setLabelStatus("Add at least one face or hull.");
    return;
  }
  setLabelStatus("Saving...");
  const response = await fetch("/api/labels", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const saved = await response.json();
  if (!response.ok) {
    setLabelStatus(saved.reason || "Save failed.");
    return;
  }
  labelSavedLink.href = saved.labelUrl;
  labelSavedLink.hidden = false;
  setLabelStatus(`Saved ${saved.labelId}.`);
  fetchSavedLabels().catch(() => {});
});

if (labelRefresh) {
  labelRefresh.addEventListener("click", () => fetchSavedLabels().catch(() => {}));
}

for (const input of [labelSetId, labelImageSide, labelNotes]) {
  input.addEventListener("input", updateLabelJson);
}
labelImageSide.addEventListener("change", () => {
  pendingFacePoints = [];
  const removed = pruneFaceQuadsForCurrentSide();
  updateFaceControls();
  drawLabels();
  updateLabelJson();
  if (removed) setLabelStatus(`Removed face labels not valid for side ${labelImageSide.value}.`);
});

document.addEventListener("keydown", (event) => {
  const tag = event.target && event.target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  const key = event.key.toUpperCase();
  if (faceOrder.includes(key) && allowedFacesForCurrentSide().includes(key)) setActiveFace(key);
  if (event.key === "Escape") labelUndo.click();
});

function buildLabelPayload() {
  const faceQuads = faceQuadsForCurrentSide();
  return {
    schemaVersion: 1,
    labelType: "cube_geometry",
    tool: "rubik-two-view-recognizer-geometry-labeler",
    coordinateSpace: "browser_image_natural",
    createdAt: new Date().toISOString(),
    setId: labelSetId.value.trim() || null,
    imageSide: labelImageSide.value,
    image: currentImage
      ? {
          name: currentImage.name,
          type: currentImage.type,
          bytes: currentImage.bytes,
          sha256: currentImage.sha256,
          width: currentImage.width,
          height: currentImage.height,
          lastModified: currentImage.lastModified,
        }
      : null,
    labels: {
      faceQuads: sortedFaceQuads(faceQuads),
      cubeHull: labels.cubeHull,
    },
    counts: {
      faceQuads: Object.keys(faceQuads).length,
      cubeHullPoints: labels.cubeHull.length,
    },
    notes: labelNotes.value.trim() || null,
  };
}

function sortedFaceQuads(faceQuads) {
  const out = {};
  for (const face of faceOrder) {
    if (faceQuads[face]) out[face] = faceQuads[face];
  }
  return out;
}

function updateLabelJson() {
  labelJson.textContent = JSON.stringify(buildLabelPayload(), null, 2);
}

function drawLabels() {
  const ctx = labelCanvas.getContext("2d");
  ctx.clearRect(0, 0, labelCanvas.width || 1, labelCanvas.height || 1);
  if (!currentImage) {
    labelJson.textContent = "";
    return;
  }
  ctx.drawImage(currentImage.element, 0, 0, labelCanvas.width, labelCanvas.height);
  drawHull(ctx, labels.cubeHull);
  for (const face of faceOrder) {
    if (labels.faceQuads[face]) drawFaceQuad(ctx, face, labels.faceQuads[face], false);
  }
  if (activeMode === "face" && pendingFacePoints.length) {
    drawFaceQuad(ctx, activeFace, pendingFacePoints, true);
  }
}

function drawHull(ctx, points) {
  if (!points.length) return;
  ctx.save();
  ctx.strokeStyle = "#111820";
  ctx.fillStyle = "rgba(17, 24, 32, 0.08)";
  ctx.lineWidth = 5;
  ctx.setLineDash([18, 12]);
  drawPath(ctx, points, points.length >= 3);
  if (points.length >= 3) ctx.fill();
  ctx.stroke();
  ctx.setLineDash([]);
  drawPoints(ctx, points, "#111820");
  ctx.restore();
}

function drawFaceQuad(ctx, face, points, pending) {
  const color = faceColors[face] || "#ff00ff";
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = pending ? "rgba(255,255,255,0.02)" : hexToRgba(color, 0.16);
  ctx.lineWidth = pending ? 6 : 5;
  if (pending) ctx.setLineDash([16, 10]);
  drawPath(ctx, points, points.length === 4);
  if (points.length === 4) {
    ctx.fill();
    drawGrid(ctx, points);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  drawPoints(ctx, points, color);
  if (points.length) drawFaceLabel(ctx, face, points[0], color);
  ctx.restore();
}

function drawPath(ctx, points, closed) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  if (closed) ctx.closePath();
}

function drawGrid(ctx, points) {
  const [p0, p1, p2, p3] = points;
  ctx.save();
  ctx.lineWidth = 3;
  ctx.globalAlpha = 0.82;
  for (const t of [1 / 3, 2 / 3]) {
    const top = lerpPoint(p0, p1, t);
    const bottom = lerpPoint(p3, p2, t);
    const left = lerpPoint(p0, p3, t);
    const right = lerpPoint(p1, p2, t);
    line(ctx, top, bottom);
    line(ctx, left, right);
  }
  ctx.restore();
}

function drawPoints(ctx, points, color) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = "#111820";
  ctx.lineWidth = 3;
  points.forEach((point, index) => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#111820";
    ctx.font = "bold 24px system-ui, sans-serif";
    ctx.fillText(String(index + 1), point.x + 14, point.y - 12);
    ctx.fillStyle = color;
  });
  ctx.restore();
}

function drawFaceLabel(ctx, face, point, color) {
  ctx.save();
  ctx.fillStyle = "#111820";
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.font = "bold 38px system-ui, sans-serif";
  ctx.strokeText(face, point.x + 24, point.y + 34);
  ctx.fillText(face, point.x + 24, point.y + 34);
  ctx.restore();
}

function line(ctx, a, b) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function lerpPoint(a, b, t) {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
  };
}

function hexToRgba(hex, alpha) {
  const value = hex.replace("#", "");
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function setLabelStatus(message) {
  labelStatus.textContent = message;
}

async function fetchSavedLabels() {
  const response = await fetch("/api/labels");
  if (!response.ok) return;
  const payload = await response.json();
  renderSavedLabels(payload.labels || []);
}

function renderSavedLabels(items) {
  labelRows.innerHTML = "";
  for (const item of items) {
    const row = document.createElement("tr");
    row.append(
      tableCell(item.setId || ""),
      tableCell(item.imageSide || ""),
      tableCell((item.faceLabels || []).join("") || ""),
      tableCell(formatLabelDate(item.savedAt)),
      labelLinkCell("open", item.labelUrl)
    );
    labelRows.append(row);
  }
  if (!items.length) {
    const row = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = "No labels saved yet.";
    td.style.color = "var(--muted)";
    td.style.fontStyle = "italic";
    row.append(td);
    labelRows.append(row);
  }
}

function tableCell(value) {
  const td = document.createElement("td");
  td.textContent = value;
  return td;
}

function labelLinkCell(label, href) {
  const td = document.createElement("td");
  if (href) {
    const link = document.createElement("a");
    link.href = href;
    link.textContent = label;
    td.append(link);
  }
  return td;
}

function formatLabelDate(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

setupLabelDropZone();
updateFaceControls();
updateLabelJson();
fetchSavedLabels().catch(() => {});
