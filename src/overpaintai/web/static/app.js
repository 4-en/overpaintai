"use strict";

const $ = (id) => document.getElementById(id);
const MASK_MAX_SIDE = 1024; // mask is edited at up to this resolution (server resizes it to the canvas)
const MASK_COLOR = "255, 90, 120";

const state = {
  imageFile: null,
  maskCanvas: null, // offscreen: white with alpha = importance
  hasMask: false,
  brushMode: "paint",
  drawing: false,
  lastPoint: null,
  job: null,
  events: null,
  lastDone: null, // last finished painting job (continue / video export start from it)
  videoEvents: null,
  exportEvents: null,
  plan: null,
  startTime: 0,
  timer: null,
};

// ---------------------------------------------------------------------------------------------------
// Settings (persisted per browser)
// ---------------------------------------------------------------------------------------------------
const SETTING_IDS = ["size-preset", "size-custom", "whole-steps", "tiles", "rounds", "max-tiles", "min-tile",
  "threshold", "margin", "fade", "importance-gain", "num-stamps"];

function saveSettings() {
  const values = {};
  for (const id of SETTING_IDS) {
    const el = $(id);
    values[id] = el.type === "checkbox" ? el.checked : el.value;
  }
  values.checkpoint = $("checkpoint").value;
  try { localStorage.setItem("overpaint-settings", JSON.stringify(values)); } catch (e) { /* private mode */ }
}

function loadSettings() {
  let values = {};
  try { values = JSON.parse(localStorage.getItem("overpaint-settings") || "{}"); } catch (e) { values = {}; }
  for (const id of SETTING_IDS) {
    if (!(id in values)) continue;
    const el = $(id);
    if (el.type === "checkbox") el.checked = values[id]; else el.value = values[id];
  }
  return values;
}

function collectSettings() {
  const preset = $("size-preset").value;
  const num = (id) => ($(id).value === "" ? null : Number($(id).value));
  return {
    checkpoint: $("checkpoint").value,
    size: preset === "custom" ? Number($("size-custom").value) : Number(preset),
    whole_steps: num("whole-steps"),
    tiles: $("tiles").checked,
    rounds: num("rounds"),
    max_tiles: num("max-tiles"),
    min_tile: num("min-tile"),
    threshold: num("threshold"),
    margin: num("margin"),
    fade: $("fade").checked,
    importance_gain: num("importance-gain"),
    num_stamps: num("num-stamps"),
  };
}

function updateSettingVisibility() {
  $("size-custom").hidden = $("size-preset").value !== "custom";
  $("tile-settings").style.opacity = $("tiles").checked ? 1 : 0.4;
  $("tile-settings").querySelectorAll("input, select").forEach((el) => { el.disabled = !$("tiles").checked; });
}

// ---------------------------------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------------------------------
async function loadCheckpoints(saved) {
  const list = await (await fetch("/api/checkpoints")).json();
  const select = $("checkpoint");
  select.innerHTML = "";
  for (const c of list) {
    const opt = document.createElement("option");
    opt.value = c.path;
    const psnr = c.val_psnr != null ? ` · val ${c.val_psnr.toFixed(2)} dB` : "";
    const epoch = c.epoch != null ? ` · ep ${c.epoch}` : "";
    opt.textContent = `${c.run}/${c.name}${epoch}${psnr}`;
    select.appendChild(opt);
  }
  if (!list.length) {
    setMessage("No checkpoints found in the runs folder.", true);
  } else if (saved.checkpoint && list.some((c) => c.path === saved.checkpoint)) {
    select.value = saved.checkpoint;
  }
  await loadDefaults();
}

async function loadDefaults() {
  const path = $("checkpoint").value;
  if (!path) return;
  const res = await fetch(`/api/checkpoints/defaults?path=${encodeURIComponent(path)}`);
  if (!res.ok) return;
  const d = await res.json();
  $("whole-steps").placeholder = `as trained (${d.whole_steps})`;
  $("steps-hint").textContent = `trained with ${d.whole_steps}`;
  const data = d.data ? ` · data: ${d.data}` : "";
  $("ckpt-info").textContent = `view ${d.view_res}px · trained at ${d.train_image_size ?? "?"}px${data}`;
}

// ---------------------------------------------------------------------------------------------------
// Target image + mask editor
// ---------------------------------------------------------------------------------------------------
function setImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  state.imageFile = file;
  const url = URL.createObjectURL(file);
  const img = $("target-img");
  img.onload = () => {
    const scale = Math.min(1, MASK_MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
    const mask = document.createElement("canvas");
    mask.width = Math.round(img.naturalWidth * scale);
    mask.height = Math.round(img.naturalHeight * scale);
    state.maskCanvas = mask;
    state.hasMask = false;
    const overlay = $("mask-overlay");
    overlay.width = mask.width;
    overlay.height = mask.height;
    $("target-stack").hidden = false;
    $("target-stage").classList.remove("empty");
    $("compare-target").src = url;
    $("paint").disabled = !$("checkpoint").value;
    redrawOverlay();
  };
  img.src = url;
}

function redrawOverlay() {
  const overlay = $("mask-overlay");
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!state.maskCanvas) return;
  ctx.globalAlpha = Number($("mask-opacity").value) / 100;
  ctx.drawImage(state.maskCanvas, 0, 0);
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = `rgb(${MASK_COLOR})`;
  ctx.fillRect(0, 0, overlay.width, overlay.height);
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
}

function maskPoint(evt) {
  const overlay = $("mask-overlay");
  const rect = overlay.getBoundingClientRect();
  return {
    x: ((evt.clientX - rect.left) / rect.width) * overlay.width,
    y: ((evt.clientY - rect.top) / rect.height) * overlay.height,
  };
}

function brushRadius() {
  const m = state.maskCanvas;
  return (Number($("brush-size").value) / 100) * Math.max(m.width, m.height) * 0.5;
}

function strokeTo(p) {
  const ctx = state.maskCanvas.getContext("2d");
  const r = brushRadius();
  ctx.globalCompositeOperation = state.brushMode === "erase" ? "destination-out" : "source-over";
  ctx.strokeStyle = ctx.fillStyle = "#fff";
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.lineWidth = 2 * r;
  const from = state.lastPoint || p;
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  ctx.globalCompositeOperation = "source-over";
  state.lastPoint = p;
  if (state.brushMode === "paint") state.hasMask = true;
  requestAnimationFrame(redrawOverlay);
}

function setupMaskEditor() {
  const overlay = $("mask-overlay");
  overlay.addEventListener("pointerdown", (e) => {
    if (!state.maskCanvas) return;
    overlay.setPointerCapture(e.pointerId);
    state.drawing = true;
    state.lastPoint = null;
    strokeTo(maskPoint(e));
  });
  overlay.addEventListener("pointermove", (e) => { if (state.drawing) strokeTo(maskPoint(e)); });
  const stop = () => { state.drawing = false; state.lastPoint = null; };
  overlay.addEventListener("pointerup", stop);
  overlay.addEventListener("pointercancel", stop);

  document.querySelectorAll("#mask-tools [data-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.brushMode = btn.dataset.mode;
      document.querySelectorAll("#mask-tools [data-mode]").forEach((b) => b.classList.toggle("active", b === btn));
    });
  });
  $("mask-opacity").addEventListener("input", redrawOverlay);
  $("mask-clear").addEventListener("click", () => {
    if (!state.maskCanvas) return;
    state.maskCanvas.getContext("2d").clearRect(0, 0, state.maskCanvas.width, state.maskCanvas.height);
    state.hasMask = false;
    redrawOverlay();
  });
  $("mask-file").addEventListener("change", async (e) => {
    if (e.target.files[0]) await loadMaskImage(e.target.files[0]);
    e.target.value = "";
  });
  $("gen-subject").addEventListener("click", () => generateMask("subject"));
  $("gen-face").addEventListener("click", () => generateMask("face"));
}

async function loadMaskImage(blob) {
  if (!state.maskCanvas) return setMessage("Open an image first.", true);
  const bitmap = await createImageBitmap(blob);
  const m = state.maskCanvas;
  const tmp = document.createElement("canvas");
  tmp.width = m.width;
  tmp.height = m.height;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(bitmap, 0, 0, m.width, m.height);
  const data = tctx.getImageData(0, 0, m.width, m.height);
  const px = data.data;
  for (let i = 0; i < px.length; i += 4) { // luminance -> alpha of a white mask
    const lum = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
    px[i] = px[i + 1] = px[i + 2] = 255;
    px[i + 3] = lum * (px[i + 3] / 255);
  }
  m.getContext("2d").putImageData(data, 0, 0);
  state.hasMask = true;
  redrawOverlay();
}

async function generateMask(mode) {
  if (!state.imageFile) return setMessage("Open an image first.", true);
  const form = new FormData();
  form.append("image", state.imageFile);
  form.append("mode", mode);
  const res = await fetch("/api/mask/auto", { method: "POST", body: form });
  if (!res.ok) return setMessage(`Mask generation failed: ${await res.text()}`, true);
  await loadMaskImage(await res.blob());
  setMessage(`Generated a ${mode} mask (placeholder ellipse; a real detector comes later).`);
}

function maskBlob() {
  return new Promise((resolve) => {
    if (!state.hasMask || !state.maskCanvas) return resolve(null);
    const m = state.maskCanvas;
    const out = document.createElement("canvas");
    out.width = m.width;
    out.height = m.height;
    const ctx = out.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(m, 0, 0);
    out.toBlob(resolve, "image/png");
  });
}

function setupDropZone() {
  const stage = $("target-stage");
  ["dragenter", "dragover"].forEach((t) => stage.addEventListener(t, (e) => {
    e.preventDefault();
    stage.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach((t) => stage.addEventListener(t, (e) => {
    e.preventDefault();
    stage.classList.remove("dragover");
  }));
  stage.addEventListener("drop", (e) => setImage(e.dataTransfer.files[0]));
  $("image-file").addEventListener("change", (e) => setImage(e.target.files[0]));
}

// ---------------------------------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------------------------------
function setMessage(text, isError = false) {
  const el = $("message");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function formatTime(s) {
  return s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

function setRunning(running) {
  $("paint").disabled = running || !state.imageFile;
  $("cancel").hidden = !running;
  $("continue").disabled = running;
  $("export-video").disabled = running;
  $("export-png").disabled = running;
}

function stageKey(name) {
  return `${state.job}|${name}`;
}

function buildStageTable(plan) {
  const body = document.querySelector("#stages tbody");
  if (plan.continues) {
    const sep = document.createElement("tr");
    sep.className = "separator";
    sep.innerHTML = `<td colspan="5">continued</td>`;
    body.appendChild(sep);
  } else {
    body.innerHTML = "";
  }
  const names = [...(plan.whole_steps > 0 ? ["whole image"] : []), ...plan.levels.map((l) => `tiles ${l}`)];
  for (const name of names) {
    const tr = document.createElement("tr");
    tr.className = "pending";
    tr.dataset.stage = stageKey(name);
    tr.innerHTML = `<td>${name}</td><td>–</td><td>–</td><td>–</td><td>–</td>`;
    body.appendChild(tr);
  }
}

function markActiveStage(name) {
  document.querySelectorAll("#stages tbody tr").forEach((tr) => {
    tr.classList.toggle("active", tr.dataset.stage === stageKey(name) && tr.classList.contains("pending"));
  });
}

function fillStage(stat) {
  let tr = document.querySelector(`#stages tbody tr[data-stage="${CSS.escape(stageKey(stat.stage))}"]`);
  if (!tr) {
    tr = document.createElement("tr");
    tr.dataset.stage = stageKey(stat.stage);
    document.querySelector("#stages tbody").appendChild(tr);
  }
  tr.className = "";
  tr.innerHTML = `<td>${stat.stage}</td><td>${stat.strokes}</td><td>${stat.tiles || "–"}</td>` +
    `<td>${stat.psnr.toFixed(2)} dB</td><td>${formatTime(stat.seconds)}</td>`;
}

function showPreview(version) {
  const img = $("preview-img");
  const next = new Image();
  next.onload = () => {
    img.src = next.src;
    $("paint-stack").hidden = false;
    $("paint-stage").classList.remove("empty");
  };
  next.src = `/api/jobs/${state.job}/preview.jpg?v=${version}`;
}

function handleEvent(ev) {
  if (ev.type === "plan") {
    state.plan = ev;
    buildStageTable(ev);
  } else if (ev.type === "progress") {
    $("progress-bar").style.width = `${(100 * ev.fraction).toFixed(1)}%`;
    const detail = ev.kind === "whole" ? `step ${ev.step}/${ev.steps}` : `round ${ev.round}/${ev.rounds} · ${ev.tiles} tiles`;
    $("stage-label").textContent = `${ev.stage} — ${detail}`;
    $("stats-label").textContent = `${ev.strokes} strokes · ${formatTime(ev.seconds)}`;
    markActiveStage(ev.stage);
    if (ev.preview) showPreview(ev.preview);
  } else if (ev.type === "stage") {
    fillStage(ev);
  } else if (ev.type === "preview") {
    showPreview(ev.preview);
  } else if (ev.type === "status") {
    if (ev.status === "queued") {
      $("stage-label").textContent = ev.position > 0 ? `queued (${ev.position} ahead)` : "starting…";
    } else if (ev.status === "running") {
      $("stage-label").textContent = "loading model…";
    } else {
      finishJob(ev);
    }
  }
}

function finishJob(ev) {
  if (state.events) state.events.close();
  state.events = null;
  clearInterval(state.timer);
  setRunning(false);
  if (ev.status === "done") {
    $("progress-bar").style.width = "100%";
    const last = ev.stats && ev.stats[ev.stats.length - 1];
    $("stage-label").textContent = "done";
    const total = ev.total_strokes || (last && last.strokes);
    if (last) $("stats-label").textContent = `${total} strokes · PSNR ${last.psnr.toFixed(2)} dB · ${formatTime(last.seconds)}`;
    const base = `/api/jobs/${state.job}`;
    $("dl-png").href = `${base}/result.png`;
    $("dl-strokes").href = `${base}/strokes.pt`;
    $("dl-stats").href = `${base}/stats.json`;
    $("downloads").hidden = false;
    state.lastDone = state.job;
    $("continue-bar").hidden = false;
    $("video-bar").hidden = false;
    $("export-bar").hidden = false;
    $("dl-video").hidden = true;
    $("dl-export").hidden = true;
    $("video-status").textContent = "";
    $("export-status").textContent = "";
    setMessage("");
  } else if (ev.status === "cancelled") {
    $("stage-label").textContent = "cancelled";
  } else if (ev.status === "error") {
    $("stage-label").textContent = "failed";
    setMessage(ev.error || "Painting failed.", true);
  }
}

async function startJob(parent = null) {
  if (!parent && !state.imageFile) return;
  saveSettings();
  const form = new FormData();
  const settings = collectSettings();
  if (parent) {
    // Continue the last painting: same image and resolution (server side), the bar's step settings
    const rounds = Number($("cont-rounds").value);
    settings.whole_steps = Number($("cont-whole").value);
    settings.tiles = rounds > 0;
    settings.rounds = Math.max(1, rounds);
    settings.min_tile = $("cont-min-tile").value === "" ? null : Number($("cont-min-tile").value);
    if (settings.whole_steps === 0 && !settings.tiles) return setMessage("Nothing to do: add steps or tile rounds.", true);
    form.append("parent", parent);
  } else {
    form.append("image", state.imageFile);
    $("continue-bar").hidden = true;
    $("video-bar").hidden = true;
    $("export-bar").hidden = true;
  }
  form.append("settings", JSON.stringify(settings));
  const mask = await maskBlob();
  if (mask) form.append("mask", mask, "mask.png");

  setRunning(true);
  setMessage("");
  $("downloads").hidden = true;
  $("progress-bar").style.width = "0";
  $("stage-label").textContent = "submitting…";
  $("stats-label").textContent = "";

  const res = await fetch("/api/jobs", { method: "POST", body: form });
  if (!res.ok) {
    setRunning(false);
    $("stage-label").textContent = "idle";
    let detail = await res.text();
    try { detail = JSON.parse(detail).detail; } catch (e) { /* plain text */ }
    return setMessage(detail, true);
  }
  const job = await res.json();
  state.job = job.id;
  state.startTime = Date.now();
  state.events = new EventSource(`/api/jobs/${job.id}/events`);
  state.events.onmessage = (msg) => handleEvent(JSON.parse(msg.data));
  state.events.onerror = () => { /* the browser reconnects automatically; finishJob closes it */ };
}

async function cancelJob() {
  if (!state.job) return;
  await fetch(`/api/jobs/${state.job}/cancel`, { method: "POST" });
}

// Exports (video, high-res PNG): a job derived from the last painting, with its own progress line
async function runExport({ endpoint, fields, button, status, link, file, eventsKey }) {
  if (!state.lastDone) return;
  const form = new FormData();
  for (const [k, v] of Object.entries(fields)) form.append(k, v);
  const res = await fetch(`/api/jobs/${state.lastDone}/${endpoint}`, { method: "POST", body: form });
  if (!res.ok) return setMessage(`Export failed: ${await res.text()}`, true);
  const job = await res.json();
  $(button).disabled = true;
  $(link).hidden = true;
  $(status).textContent = "queued…";
  if (state[eventsKey]) state[eventsKey].close();
  const events = new EventSource(`/api/jobs/${job.id}/events`);
  state[eventsKey] = events;
  events.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === "plan") {
      $(status).textContent = `rendering ${ev.strokes} strokes…`;
    } else if (ev.type === "progress") {
      $(status).textContent = `rendering ${Math.round(100 * ev.fraction)}% · ${formatTime(ev.seconds)}`;
    } else if (ev.type === "status" && ev.status === "running") {
      $(status).textContent = "starting…";
    } else if (ev.type === "status" && ["done", "error", "cancelled"].includes(ev.status)) {
      events.close();
      state[eventsKey] = null;
      $(button).disabled = false;
      if (ev.status === "done") {
        $(status).textContent = "ready";
        $(link).href = `/api/jobs/${job.id}/${file}`;
        $(link).hidden = false;
      } else {
        $(status).textContent = ev.status === "error" ? "failed" : "cancelled";
        if (ev.error) setMessage(ev.error, true);
      }
    }
  };
}

function exportVideo() {
  return runExport({ endpoint: "video", fields: { duration: $("video-duration").value }, button: "export-video",
    status: "video-status", link: "dl-video", file: "video.mp4", eventsKey: "videoEvents" });
}

function exportPng() {
  return runExport({ endpoint: "export", fields: { side: $("export-side").value }, button: "export-png",
    status: "export-status", link: "dl-export", file: "export.png", eventsKey: "exportEvents" });
}

// ---------------------------------------------------------------------------------------------------
function setupCompare() {
  const apply = () => {
    const v = Number($("compare").value);
    $("preview-img").style.clipPath = `inset(0 ${100 - v}% 0 0)`;
  };
  $("compare").addEventListener("input", apply);
  apply();
}

async function init() {
  const saved = loadSettings();
  updateSettingVisibility();
  SETTING_IDS.forEach((id) => $(id).addEventListener("change", () => { updateSettingVisibility(); saveSettings(); }));
  $("checkpoint").addEventListener("change", () => { saveSettings(); loadDefaults(); });
  $("paint").addEventListener("click", () => startJob());
  $("continue").addEventListener("click", () => startJob(state.lastDone));
  $("export-video").addEventListener("click", exportVideo);
  $("export-png").addEventListener("click", exportPng);
  $("cancel").addEventListener("click", cancelJob);
  setupDropZone();
  setupMaskEditor();
  setupCompare();
  try {
    const info = await (await fetch("/api/info")).json();
    $("device").textContent = info.device_name;
  } catch (e) { /* offline */ }
  await loadCheckpoints(saved);
}

init();
