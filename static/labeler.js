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
const labelTemplate = document.querySelector("#labelTemplate");

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
let activeSide = labelImageSide ? labelImageSide.value : "A";
let currentImage = null;
let pendingFacePoints = [];
let labels = emptyLabels();
let templateAnchors = null;
let templateDrag = null;
let suppressNextTemplateClick = false;
const labelStates = {
  A: emptyLabelState(),
  B: emptyLabelState(),
  single: emptyLabelState(),
};

function emptyLabels() {
  return {
    faceQuads: {},
    cubeHull: [],
  };
}

function emptyLabelState() {
  return {
    image: null,
    labels: emptyLabels(),
    pendingFacePoints: [],
    templateAnchors: null,
    savedUrl: null,
  };
}

function stateForSide(side) {
  if (!labelStates[side]) labelStates[side] = emptyLabelState();
  return labelStates[side];
}

function setLabels(nextLabels) {
  labels = nextLabels;
  stateForSide(activeSide).labels = nextLabels;
}

function setPendingFacePoints(points) {
  pendingFacePoints = points;
  stateForSide(activeSide).pendingFacePoints = points;
}

function setTemplateAnchors(anchors) {
  templateAnchors = anchors;
  stateForSide(activeSide).templateAnchors = anchors;
}

function bindActiveSide(side) {
  activeSide = side;
  const state = stateForSide(side);
  currentImage = state.image;
  labels = state.labels;
  pendingFacePoints = state.pendingFacePoints;
  templateAnchors = state.templateAnchors;
  templateDrag = null;
  updateImageChrome();
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
    const files = Array.from(event.dataTransfer.files || []).filter((item) => item.type.startsWith("image/"));
    if (!files.length) return;
    const dataTransfer = new DataTransfer();
    for (const file of files.slice(0, 2)) dataTransfer.items.add(file);
    labelImageInput.files = dataTransfer.files;
    loadLabelImages(labelImageInput.files);
  });
  labelImageInput.addEventListener("change", () => {
    if (labelImageInput.files && labelImageInput.files.length) loadLabelImages(labelImageInput.files);
  });
}

async function loadLabelImages(fileList) {
  const files = Array.from(fileList || []).filter((file) => file.type.startsWith("image/")).slice(0, 2);
  if (!files.length) return;
  setLabelStatus(`Loading ${files.length} image${files.length === 1 ? "" : "s"}...`);
  const pairing = pairLabelFiles(files);
  const loadedSides = [];
  for (const [side, file] of Object.entries(pairing)) {
    if (!file) continue;
    const imageRecord = await readLabelImage(file);
    const state = stateForSide(side);
    state.image = imageRecord;
    state.labels = emptyLabels();
    state.pendingFacePoints = [];
    state.templateAnchors = null;
    state.savedUrl = null;
    loadedSides.push(side);
    inferSetFields(file.name, { updateSide: files.length === 1 });
  }
  const nextSide = files.length > 1 && pairing.A ? "A" : loadedSides[0] || activeSide;
  if (labelImageSide.value !== nextSide) labelImageSide.value = nextSide;
  bindActiveSide(labelImageSide.value);
  updateFaceControls();
  drawLabels();
  updateLabelJson();
  setLabelStatus(`Loaded ${loadedSides.join("/")}.`);
}

function pairLabelFiles(files) {
  const [first, second] = files;
  if (!second) {
    const marker = detectLabelABMarker(first.name);
    const side = marker || (labelImageSide.value === "B" ? "B" : labelImageSide.value === "single" ? "single" : "A");
    return { [side]: first };
  }

  const firstMarker = detectLabelABMarker(first.name);
  const secondMarker = detectLabelABMarker(second.name);
  if (firstMarker === "A" && secondMarker === "B") return { A: first, B: second };
  if (firstMarker === "B" && secondMarker === "A") return { A: second, B: first };
  return { A: first, B: second };
}

function detectLabelABMarker(name) {
  const stem = name.replace(/\.[^./]+$/, "");
  const token = /(?:^|[\s_-])([ab])(?:[\s_-]|$)/i.exec(stem);
  if (token) return token[1].toUpperCase();
  const imageToken = /\bimage\s*([ab])\b/i.exec(stem);
  if (imageToken) return imageToken[1].toUpperCase();
  return null;
}

async function readLabelImage(file) {
  const url = URL.createObjectURL(file);
  const image = new Image();
  return new Promise((resolve, reject) => {
    image.onload = async () => {
      URL.revokeObjectURL(url);
      resolve({
        file,
        element: image,
        name: file.name,
        type: file.type,
        bytes: file.size,
        width: image.naturalWidth,
        height: image.naturalHeight,
        lastModified: file.lastModified,
        sha256: await sha256Hex(file),
      });
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      setLabelStatus("Could not load image.");
      reject(new Error(`Could not load ${file.name}`));
    };
    image.src = url;
  });
}

function updateImageChrome() {
  const activeState = stateForSide(activeSide);
  const image = activeState.image;
  labelSavedLink.href = activeState.savedUrl || "#";
  labelSavedLink.hidden = !activeState.savedUrl;
  if (image) {
    labelCanvas.width = image.width;
    labelCanvas.height = image.height;
    labelImageTitle.textContent = `${activeSide}: ${image.name}`;
  } else {
    labelCanvas.width = 0;
    labelCanvas.height = 0;
    labelImageTitle.textContent = `${activeSide}: Image`;
  }
  renderLabelDropZoneFiles();
}

function renderLabelDropZoneFiles() {
  labelDropZoneFiles.innerHTML = "";
  for (const side of ["A", "B"]) {
    const image = stateForSide(side).image;
    if (!image) continue;
    const div = document.createElement("div");
    div.textContent = `${side}: ${image.name} (${image.width} x ${image.height})`;
    labelDropZoneFiles.append(div);
  }
  const singleImage = stateForSide("single").image;
  if (singleImage && !stateForSide("A").image && !stateForSide("B").image) {
    const div = document.createElement("div");
    div.textContent = `single: ${singleImage.name} (${singleImage.width} x ${singleImage.height})`;
    labelDropZoneFiles.append(div);
  }
}

async function sha256Hex(file) {
  if (!window.crypto || !window.crypto.subtle) return null;
  const digest = await window.crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function inferSetFields(name, { updateSide = true } = {}) {
  if (!labelSetId.value.trim()) {
    const setMatch = /\bset\s*([0-9]+)\b/i.exec(name);
    if (setMatch) labelSetId.value = `Set ${setMatch[1]}`;
  }
  const sideMatch = /(?:^|[\s_-])([ab])(?:[\s_-]|$)/i.exec(name.replace(/\.[^./]+$/, ""));
  if (updateSide && sideMatch) labelImageSide.value = sideMatch[1].toUpperCase();
  updateFaceControls();
}

function setActiveMode(mode) {
  activeMode = mode;
  setPendingFacePoints([]);
  for (const button of labelModeButtons) {
    button.classList.toggle("is-active", button.dataset.labelMode === mode);
  }
  labelCanvas.dataset.mode = mode;
  updateFaceControls();
  drawLabels();
  updateLabelJson();
}

function setActiveFace(face) {
  if (!allowedFacesForCurrentSide().includes(face)) return;
  activeFace = face;
  activeMode = "face";
  setPendingFacePoints([]);
  labelCanvas.dataset.mode = "face";
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

function updateFaceControls() {
  const allowedFaces = allowedFacesForCurrentSide();
  if (!allowedFaces.includes(activeFace)) {
    activeFace = allowedFaces[0] || "U";
    setPendingFacePoints([]);
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
      : activeMode === "template"
        ? "Template mode: seven anchors derive the hull and all three visible face quads."
        : (faceGuidanceBySide[labelImageSide.value] || faceGuidanceBySide.single);
  }
}

labelCanvas.addEventListener("click", (event) => {
  if (!currentImage) {
    setLabelStatus("Add an image first.");
    return;
  }
  const point = canvasPoint(event);
  if (activeMode === "template") {
    if (suppressNextTemplateClick) {
      suppressNextTemplateClick = false;
      return;
    }
    placeTemplate(point);
    return;
  }
  if (activeMode === "face") {
    pendingFacePoints.push(point);
    if (pendingFacePoints.length === 4) {
      labels.faceQuads[activeFace] = pendingFacePoints;
      setPendingFacePoints([]);
      setLabelStatus(`${activeFace} face saved.`);
    }
  } else {
    labels.cubeHull.push(point);
    setLabelStatus(`${labels.cubeHull.length} hull point${labels.cubeHull.length === 1 ? "" : "s"}.`);
  }
  drawLabels();
  updateLabelJson();
});

labelCanvas.addEventListener("pointerdown", (event) => {
  if (!currentImage || activeMode !== "template" || !templateAnchors) return;
  const point = canvasPoint(event);
  const hit = hitTemplateAnchor(point);
  if (!hit) return;
  templateDrag = {
    pointerId: event.pointerId,
    hit,
    startPoint: point,
    startAnchors: cloneTemplateAnchors(templateAnchors),
    moved: false,
  };
  suppressNextTemplateClick = true;
  labelCanvas.setPointerCapture(event.pointerId);
  event.preventDefault();
});

labelCanvas.addEventListener("pointermove", (event) => {
  if (!templateDrag || templateDrag.pointerId !== event.pointerId) return;
  moveTemplateAnchor(templateDrag, canvasPoint(event));
  templateDrag.moved = true;
  suppressNextTemplateClick = true;
  drawLabels();
  updateLabelJson();
});

labelCanvas.addEventListener("pointerup", (event) => {
  if (!templateDrag || templateDrag.pointerId !== event.pointerId) return;
  if (labelCanvas.hasPointerCapture(event.pointerId)) labelCanvas.releasePointerCapture(event.pointerId);
  setLabelStatus("Template updated.");
  templateDrag = null;
});

labelCanvas.addEventListener("pointercancel", (event) => {
  if (!templateDrag || templateDrag.pointerId !== event.pointerId) return;
  if (labelCanvas.hasPointerCapture(event.pointerId)) labelCanvas.releasePointerCapture(event.pointerId);
  templateDrag = null;
  suppressNextTemplateClick = false;
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

function defaultTemplateCenter() {
  return {
    x: roundPoint(currentImage.width * 0.47),
    y: roundPoint(currentImage.height * 0.52),
  };
}

function placeTemplate(center) {
  const nextAnchors = templateAnchors
    ? translateTemplate(templateAnchors, center)
    : makeDefaultTemplate(center);
  setTemplateAnchors(nextAnchors);
  applyTemplateLabels();
  drawLabels();
  updateLabelJson();
  setLabelStatus("Template placed.");
}

function makeDefaultTemplate(center) {
  const w = currentImage.width;
  const h = currentImage.height;
  const hull = [
    { x: center.x + w * 0.0, y: center.y - h * 0.3 },
    { x: center.x + w * 0.34, y: center.y - h * 0.18 },
    { x: center.x + w * 0.3, y: center.y + h * 0.08 },
    { x: center.x + w * 0.02, y: center.y + h * 0.32 },
    { x: center.x - w * 0.28, y: center.y + h * 0.08 },
    { x: center.x - w * 0.34, y: center.y - h * 0.18 },
  ].map(clampPointToImage);
  return {
    center: clampPointToImage(center),
    hull,
  };
}

function translateTemplate(anchors, nextCenter) {
  return translateTemplateByDelta(anchors, nextCenter.x - anchors.center.x, nextCenter.y - anchors.center.y);
}

function translateTemplateByDelta(anchors, dx, dy) {
  const points = [anchors.center, ...anchors.hull];
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const boundedDx = Math.max(-minX, Math.min(currentImage.width - maxX, dx));
  const boundedDy = Math.max(-minY, Math.min(currentImage.height - maxY, dy));
  return {
    center: clampPointToImage({ x: anchors.center.x + boundedDx, y: anchors.center.y + boundedDy }),
    hull: anchors.hull.map((point) => clampPointToImage({ x: point.x + boundedDx, y: point.y + boundedDy })),
  };
}

function cloneTemplateAnchors(anchors) {
  return {
    center: { ...anchors.center },
    hull: anchors.hull.map((point) => ({ ...point })),
  };
}

function clampPointToImage(point) {
  return {
    x: roundPoint(Math.max(0, Math.min(currentImage.width, point.x))),
    y: roundPoint(Math.max(0, Math.min(currentImage.height, point.y))),
  };
}

function moveTemplateAnchor(drag, point) {
  if (!templateAnchors) return;
  let next;
  if (drag.hit.kind === "center" || drag.hit.kind === "body") {
    next = translateTemplateByDelta(
      drag.startAnchors,
      point.x - drag.startPoint.x,
      point.y - drag.startPoint.y,
    );
  } else {
    next = cloneTemplateAnchors(templateAnchors);
    next.hull[drag.hit.index] = clampPointToImage(point);
  }
  setTemplateAnchors(next);
  applyTemplateLabels();
}

function hitTemplateAnchor(point) {
  if (!templateAnchors) return null;
  const hullThreshold = templateHitThreshold();
  const centerThreshold = templateCenterHitThreshold();
  const anchors = [{ kind: "center", index: -1, point: templateAnchors.center, threshold: centerThreshold }];
  templateAnchors.hull.forEach((hullPoint, index) => anchors.push({ kind: "hull", index, point: hullPoint }));
  let best = null;
  for (const anchor of anchors) {
    const distance = Math.hypot(anchor.point.x - point.x, anchor.point.y - point.y);
    const threshold = anchor.threshold || hullThreshold;
    if (distance <= threshold && (!best || distance < best.distance)) best = { ...anchor, distance };
  }
  if (best) return { kind: best.kind, index: best.index };
  return pointInTemplateHull(point) ? { kind: "body", index: -1 } : null;
}

function templateHitThreshold() {
  const scale = templateVisualScale();
  return Math.max(24, 18 * scale);
}

function templateCenterHitThreshold() {
  const scale = templateVisualScale();
  return Math.max(44, 42 * scale);
}

function templateVisualScale() {
  const rect = labelCanvas.getBoundingClientRect();
  return labelCanvas.width / Math.max(1, rect.width);
}

function pointInTemplateHull(point) {
  if (!templateAnchors || templateAnchors.hull.length < 3) return false;
  let inside = false;
  for (let i = 0, j = templateAnchors.hull.length - 1; i < templateAnchors.hull.length; j = i, i += 1) {
    const a = templateAnchors.hull[i];
    const b = templateAnchors.hull[j];
    const crosses = (a.y > point.y) !== (b.y > point.y)
      && point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x;
    if (crosses) inside = !inside;
  }
  return inside;
}

function applyTemplateLabels() {
  if (!templateAnchors) return;
  const [topFace, rightFace, leftFace] = templateFacesForCurrentSide();
  const hull = templateAnchors.hull.map((point) => ({ ...point }));
  const center = { ...templateAnchors.center };
  setLabels({
    faceQuads: {
      [topFace]: [hull[0], hull[1], center, hull[5]],
      [rightFace]: [hull[1], hull[2], hull[3], center],
      [leftFace]: [hull[5], center, hull[3], hull[4]],
    },
    cubeHull: hull,
  });
  setPendingFacePoints([]);
}

function templateFacesForCurrentSide() {
  if (labelImageSide.value === "B") return ["D", "L", "B"];
  return ["U", "R", "F"];
}

function clearTemplateLabels() {
  setTemplateAnchors(null);
  setLabels(emptyLabels());
  setPendingFacePoints([]);
  setLabelStatus("Template cleared.");
}

labelUndo.addEventListener("click", () => {
  if (activeMode === "face" && pendingFacePoints.length) {
    pendingFacePoints.pop();
  } else if (activeMode === "hull" && labels.cubeHull.length) {
    labels.cubeHull.pop();
  } else if (activeMode === "template" && templateAnchors) {
    clearTemplateLabels();
  } else if (activeMode === "face" && labels.faceQuads[activeFace]) {
    delete labels.faceQuads[activeFace];
  }
  drawLabels();
  updateLabelJson();
});

labelClearActive.addEventListener("click", () => {
  if (activeMode === "face") {
    setPendingFacePoints([]);
    delete labels.faceQuads[activeFace];
  } else if (activeMode === "hull") {
    labels.cubeHull = [];
  } else {
    clearTemplateLabels();
  }
  drawLabels();
  updateLabelJson();
});

labelClearAll.addEventListener("click", () => {
  setLabels(emptyLabels());
  setPendingFacePoints([]);
  setTemplateAnchors(null);
  drawLabels();
  updateLabelJson();
});

if (labelTemplate) {
  labelTemplate.addEventListener("click", () => {
    if (!currentImage) {
      setLabelStatus("Add an image first.");
      return;
    }
    setActiveMode("template");
    placeTemplate(defaultTemplateCenter());
  });
}

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
  let saved;
  try {
    const response = await fetch("/api/labels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    saved = await response.json();
    if (!response.ok) {
      setLabelStatus(saved.reason || "Save failed.");
      return;
    }
  } catch (error) {
    setLabelStatus("Save failed.");
    return;
  }
  labelSavedLink.href = saved.labelUrl;
  labelSavedLink.hidden = false;
  stateForSide(activeSide).savedUrl = saved.labelUrl;
  try {
    await fetchSavedLabels();
    setLabelStatus(`Saved ${saved.labelId}. Saved Labels updated.`);
  } catch (error) {
    setLabelStatus(`Saved ${saved.labelId}. Use Refresh if the list did not update.`);
  }
});

if (labelRefresh) {
  labelRefresh.addEventListener("click", async () => {
    setLabelStatus("Refreshing saved labels...");
    try {
      await fetchSavedLabels();
      setLabelStatus("Saved Labels refreshed.");
    } catch (error) {
      setLabelStatus("Could not refresh Saved Labels.");
    }
  });
}

for (const input of [labelSetId, labelImageSide, labelNotes]) {
  input.addEventListener("input", updateLabelJson);
}
labelImageSide.addEventListener("change", () => {
  bindActiveSide(labelImageSide.value);
  setPendingFacePoints([]);
  updateFaceControls();
  drawLabels();
  updateLabelJson();
  setLabelStatus(currentImage ? `Showing ${labelImageSide.value}.` : `No ${labelImageSide.value} image loaded.`);
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
  const templateEditing = activeMode === "template" && Boolean(templateAnchors);
  drawHull(ctx, labels.cubeHull, { showPoints: !templateEditing });
  for (const face of faceOrder) {
    if (labels.faceQuads[face]) drawFaceQuad(ctx, face, labels.faceQuads[face], false, { showPoints: !templateEditing });
  }
  if (activeMode === "face" && pendingFacePoints.length) {
    drawFaceQuad(ctx, activeFace, pendingFacePoints, true);
  }
  if (templateAnchors) drawTemplateAnchors(ctx, templateAnchors);
}

function drawHull(ctx, points, { showPoints = true } = {}) {
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
  if (showPoints) drawPoints(ctx, points, "#111820");
  ctx.restore();
}

function drawFaceQuad(ctx, face, points, pending, { showPoints = true } = {}) {
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
  if (showPoints) drawPoints(ctx, points, color);
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

function drawTemplateAnchors(ctx, anchors) {
  ctx.save();
  const scale = templateVisualScale();
  const center = anchors.center;

  ctx.lineWidth = 3 * scale;
  ctx.strokeStyle = "rgba(17,24,32,0.55)";
  ctx.fillStyle = "rgba(255,255,255,0.1)";
  ctx.beginPath();
  ctx.arc(center.x, center.y, 34 * scale, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.strokeStyle = "rgba(255,77,61,0.78)";
  ctx.lineWidth = 3 * scale;
  ctx.beginPath();
  ctx.arc(center.x, center.y, 8 * scale, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(center.x - 42 * scale, center.y);
  ctx.lineTo(center.x - 12 * scale, center.y);
  ctx.moveTo(center.x + 12 * scale, center.y);
  ctx.lineTo(center.x + 42 * scale, center.y);
  ctx.moveTo(center.x, center.y - 42 * scale);
  ctx.lineTo(center.x, center.y - 12 * scale);
  ctx.moveTo(center.x, center.y + 12 * scale);
  ctx.lineTo(center.x, center.y + 42 * scale);
  ctx.stroke();

  drawAnchorText(ctx, "C", { x: center.x, y: center.y - 24 * scale }, "rgba(17,24,32,0.5)", 17 * scale);
  anchors.hull.forEach((point, index) => {
    const half = 15 * scale;
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    ctx.strokeStyle = "rgba(17,24,32,0.58)";
    ctx.lineWidth = 3 * scale;
    ctx.beginPath();
    ctx.rect(point.x - half, point.y - half, half * 2, half * 2);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = "rgba(255,77,61,0.66)";
    ctx.lineWidth = 2 * scale;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5 * scale, 0, Math.PI * 2);
    ctx.stroke();
    drawAnchorText(ctx, String(index + 1), anchorLabelPoint(point, scale), "rgba(17,24,32,0.55)", 14 * scale);
  });
  ctx.restore();
}

function anchorLabelPoint(point, scale) {
  return {
    x: Math.max(16 * scale, Math.min(labelCanvas.width - 16 * scale, point.x + 22 * scale)),
    y: Math.max(16 * scale, Math.min(labelCanvas.height - 16 * scale, point.y - 22 * scale)),
  };
}

function drawAnchorText(ctx, text, point, color, fontSize = 24) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.font = `bold ${fontSize}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, point.x, point.y);
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
  const response = await fetch("/api/labels", { cache: "no-store" });
  if (!response.ok) throw new Error("Could not load saved labels.");
  const payload = await response.json();
  renderSavedLabels(payload.labels || []);
  return payload.labels || [];
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
bindActiveSide(activeSide);
labelCanvas.dataset.mode = activeMode;
updateFaceControls();
updateLabelJson();
fetchSavedLabels().catch(() => {});
