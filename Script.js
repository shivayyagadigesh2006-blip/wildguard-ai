/* ==========================================================
   WildGuard AI — script.js (vanilla JS, no dependencies)

   BACKEND HOOKS
   -------------
   The same-origin FastAPI server provides:
     GET  {API_BASE}/api/health   -> per-capability status badges
     GET  {API_BASE}/api/dashboard -> recorded dashboard metrics
     GET  {API_BASE}/api/history  -> real analysis history
     POST {API_BASE}/api/analyze  -> multipart media and optional browser upload coordinates,
        per-animal boxes, crop classifications, and a note.
     GET  {API_BASE}/api/locations -> browser locations from locally recorded analyses.
   The interface never substitutes fabricated results when analysis is unavailable.

   PHOTOGRAPHS are <img> tags in index.html (local /images file first,
   then the online copy). Fallbacks are handled by wgImg in the page <head>.
   ========================================================== */

const configuredApiBase =
  window.WILDGUARD_API_BASE ||
  document.querySelector('meta[name="wildguard-api-base"]')?.content ||
  "";

const CONFIG = {
  API_BASE: configuredApiBase.replace(/\/+$/, ""),
  HEALTH_PATH: "/api/health",
  ANALYZE_PATH: "/api/analyze",
  TIMEOUT_MS: 4000,
  ANALYSIS_TIMEOUT_MS: 300000,
  MAP_TILE_URL: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  MAP_REFRESH_MS: 30000,
};

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const pct = (n) => Math.round(n * 100);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---------- Navigation ---------- */
(function nav() {
  const nav = $("#nav"), toggle = $("#navToggle"), links = $("#navLinks"), bar = $("#progress");
  let ticking = false;
  const update = () => {
    nav.classList.toggle("scrolled", window.scrollY > 40);
    const max = document.documentElement.scrollHeight - window.innerHeight;
    if (bar && max > 0) bar.style.transform = `scaleX(${Math.min(window.scrollY / max, 1)})`;
    ticking = false;
  };
  update();
  window.addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });

  const setOpen = (open) => {
    toggle.setAttribute("aria-expanded", String(open));
    links.classList.toggle("open", open);
    document.body.style.overflow = open ? "hidden" : "";
  };
  toggle.addEventListener("click", () => setOpen(toggle.getAttribute("aria-expanded") !== "true"));
  links.addEventListener("click", (e) => { if (e.target.closest("a")) setOpen(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setOpen(false); });
  window.matchMedia("(min-width: 861px)").addEventListener("change", () => setOpen(false));

  // highlight the current section in the nav
  const map = new Map($$('.nav-links a[href^="#"]:not(.btn)').map((a) => [a.getAttribute("href").slice(1), a]));
  const sections = ["overview", "monitoring", "species", "conservation"].map((id) => document.getElementById(id)).filter(Boolean);
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => entries.forEach((en) => {
      if (!en.isIntersecting) return;
      map.forEach((a) => { a.classList.remove("active"); a.removeAttribute("aria-current"); });
      const a = map.get(en.target.id);
      if (a) { a.classList.add("active"); a.setAttribute("aria-current", "true"); }
    }), { rootMargin: "-45% 0px -50% 0px" });
    sections.forEach((s) => io.observe(s));
  }
})();

/* ---------- Scroll reveals ---------- */
(function reveals() {
  const items = $$(".reveal");
  $$(".caps, .themes, .gallery").forEach((group) =>
    $$(".reveal", group).forEach((el, i) => el.style.setProperty("--rd", `${Math.min(i, 5) * 0.08}s`)));
  if (!("IntersectionObserver" in window)) { items.forEach((el) => el.classList.add("in")); return; }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
  }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  items.forEach((el) => io.observe(el));
})();

/* ---------- Capabilities: tap to expand on touch screens ---------- */
(function caps() {
  const cards = $$(".cap");
  cards.forEach((c) => c.addEventListener("click", () => {
    if (!window.matchMedia("(hover: none)").matches) return;
    const was = c.classList.contains("open");
    cards.forEach((x) => x.classList.remove("open"));
    c.classList.toggle("open", !was);
  }));
})();

/* ---------- Backend health check (drives every status label) ---------- */
const backend = { online: false, caps: {} };
(async function health() {
  const state = $("#apiState"), chip = $("#connChip");
  const set = (online, caps = {}) => {
    backend.online = online; backend.caps = caps;
    const anyLive = online && Object.values(caps).some(Boolean);
    state.className = "api-state " + (online ? "online" : "offline");
    $("span", state).textContent = !online ? "No backend connected — AI features are not running yet"
      : anyLive ? "Backend online — some AI features are running" : "Backend online — no analysis models are configured";
    chip.textContent = online ? "Backend online" : "Backend offline";
    chip.classList.toggle("online", online);
    $$(".cap").forEach((card) => {
      const b = $("[data-badge]", card), live = online && !!caps[card.dataset.cap];
      b.textContent = live ? "Live" : online ? "Not enabled on backend" : "Backend pending";
      b.classList.toggle("live", live);
    });
  };
  try {
    const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), CONFIG.TIMEOUT_MS);
    let res;
    try {
      res = await fetch(CONFIG.API_BASE + CONFIG.HEALTH_PATH, { signal: ctl.signal });
    } finally {
      clearTimeout(t);
    }
    if (!res.ok) return set(false);
    const info = await res.json().catch(() => ({}));
    set(true, info.capabilities || {});
    loadDashboard();
  } catch (error) {
    set(false);
    state.className = "api-state offline";
    $("span", state).textContent = error.name === "AbortError" ? "Backend health check timed out" : "Could not connect to the backend";
  }
})();

async function loadDashboard() {
  try {
    const [dashboardResponse, historyResponse, locationsResponse] = await Promise.all([
      fetch(CONFIG.API_BASE + "/api/dashboard"),
      fetch(CONFIG.API_BASE + "/api/history?limit=6"),
      fetch(CONFIG.API_BASE + "/api/locations?limit=1000"),
    ]);
    if (!dashboardResponse.ok || !historyResponse.ok || !locationsResponse.ok) throw new Error("Dashboard data could not be loaded.");
    const dashboard = await dashboardResponse.json();
    $("#totalPredictions").textContent = Number(dashboard.total_predictions || 0).toLocaleString();
    $("#speciesIdentified").textContent = Number(dashboard.species_identified || 0).toLocaleString();
    $("#analysisSessions").textContent = Number(dashboard.analysis_sessions || 0).toLocaleString();
    $("#averageConfidence").textContent = Number.isFinite(dashboard.average_confidence)
      ? `${Math.round(dashboard.average_confidence * 100)}%` : "—";
    renderHistory(await historyResponse.json());
    renderLocations(await locationsResponse.json());
  } catch (error) {
    console.error("Could not load dashboard data:", error);
    $("#history").innerHTML = `<tr><td colspan="5" class="empty-state">Analysis history is currently unavailable.</td></tr>`;
    $("#locationCount").textContent = "Unavailable";
    $("#locationRecords").innerHTML = `<li class="empty-state">Location data is currently unavailable.</li>`;
  }
}

/* ---------- Result panels ---------- */
const fillBars = (root) => requestAnimationFrame(() => requestAnimationFrame(() => $$(".bar i", root).forEach((b) => (b.style.width = b.dataset.w + "%"))));

function renderResults(list) {
  const ul = $("#results");
  ul.innerHTML = list.length ? list.map((r) => `
    <li><div class="res-top"><div><div class="res-name">${esc(r.species)}</div><div class="res-sci">${Number(r.count) || 0} confident detection${Number(r.count) === 1 ? "" : "s"}</div></div>
    <span class="res-pct">${pct(r.confidence)}%</span></div>
    <div class="bar" role="img" aria-label="Confidence ${pct(r.confidence)} percent"><i data-w="${pct(r.confidence)}"></i></div></li>`).join("")
    : `<li class="empty-state">No classification candidates are available.</li>`;
  fillBars(ul);
  $("#resultTag").textContent = list.length ? "Model candidates" : "No result";
}
function renderDetections(detections, mediaType, imageSize) {
  const list = $("#detectedAnimals"), media = $("#previewMedia");
  list.innerHTML = detections.length ? detections.slice(0, 60).map((d) => {
    const frame = mediaType === "video"
      ? `Frame ${Number(d.frame_index) + 1}${Number.isFinite(d.timestamp_seconds) ? ` · ${Number(d.timestamp_seconds).toFixed(1)} s` : ""}`
      : "Image";
    const bbox = d.bbox || {};
    const box = [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
      .map((value) => Number.isFinite(value) ? `${Math.round(value * 100)}%` : "—").join(", ");
    return `<li><div class="detection-row"><strong>${esc(d.species || "Uncertain")}</strong><span>${pct(d.confidence)}%</span></div>
      <span class="detection-meta">${esc(frame)} · animal ${pct(d.detector_confidence)}% · box ${esc(box)}${d.needs_review ? ` · <b class="detection-review">review</b>` : ""}</span></li>`;
  }).join("") + (detections.length > 60 ? `<li class="detection-meta">Showing 60 of ${detections.length} detected instances.</li>` : "")
    : `<li class="empty-state">No animals detected in the analyzed media.</li>`;

  const priorOverlay = $(".detection-overlay", media);
  if (priorOverlay) priorOverlay.remove();
  if (mediaType !== "image" || !detections.length || !imageSize?.[0] || !imageSize?.[1] || !$("img", media)) return;
  const [width, height] = imageSize;
  const boxes = detections.map((d) => {
    const b = d.bbox;
    if (!b) return "";
    const x = b.x_min * width, y = b.y_min * height;
    return `<g><rect x="${x}" y="${y}" width="${(b.x_max - b.x_min) * width}" height="${(b.y_max - b.y_min) * height}"/>
      <text x="${x + 5}" y="${Math.max(18, y + 18)}">${esc(d.species)} ${pct(d.confidence)}%</text></g>`;
  }).join("");
  media.insertAdjacentHTML("beforeend", `<svg class="detection-overlay" viewBox="0 0 ${width} ${height}" role="img" aria-label="Animal detection bounding boxes">${boxes}</svg>`);
}

let locationsMap = null;
let locationsLayer = null;
let locationSignature = null;

function initializeLocationsMap() {
  const host = $("#analysisMap");
  if (!host) return false;
  if (!window.L) {
    host.innerHTML = `<p class="empty-state">The interactive map library did not load. Check your internet connection.</p>`;
    $("#locationCount").textContent = "Unavailable";
    return false;
  }
  if (locationsMap) return true;

  locationsMap = window.L.map(host, { scrollWheelZoom: false }).setView([20, 0], 2);
  window.L.tileLayer(CONFIG.MAP_TILE_URL, {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(locationsMap);
  locationsLayer = window.L.featureGroup().addTo(locationsMap);
  requestAnimationFrame(() => locationsMap.invalidateSize());
  return true;
}

function renderLocations(rows) {
  const records = $("#locationRecords"), tag = $("#locationCount");
  if (!initializeLocationsMap()) return;

  const validRows = rows.filter((row) => {
    const latitude = Number(row.latitude), longitude = Number(row.longitude);
    return Number.isFinite(latitude) && Number.isFinite(longitude)
      && Math.abs(latitude) <= 90 && Math.abs(longitude) <= 180;
  });
  tag.textContent = validRows.length ? `${validRows.length} observations` : "No locations";

  const signature = validRows.map((row) => `${row.id}:${row.latitude}:${row.longitude}`).join("|");
  if (signature === locationSignature) return;
  locationSignature = signature;
  locationsLayer.clearLayers();

  if (!validRows.length) {
    records.innerHTML = `<li class="empty-state">No geotagged upload observations yet.</li>`;
    locationsMap.setView([20, 0], 2);
    return;
  }

  const grouped = new Map();
  validRows.forEach((row) => {
    const latitude = Number(row.latitude), longitude = Number(row.longitude);
    const key = `${latitude.toFixed(5)},${longitude.toFixed(5)}`;
    if (!grouped.has(key)) grouped.set(key, { latitude, longitude, rows: [] });
    grouped.get(key).rows.push(row);
  });

  const markers = [];
  const recordItems = [];
  for (const point of grouped.values()) {
    const entries = point.rows.map((row) => {
      const species = (row.predictions || []).map((prediction) =>
        `${esc(prediction.species)} ×${Number(prediction.count) || 0}`
      ).join(", ") || "No confident species label";
      const timestamp = new Date(row.created_at).toLocaleString();
      return `<li><strong>${esc(row.name)}</strong><br>${species}<br><small>${esc(timestamp)}</small></li>`;
    }).join("");
    const marker = window.L.circleMarker([point.latitude, point.longitude], {
      radius: 8,
      color: "#fff",
      weight: 2,
      fillColor: "#526B45",
      fillOpacity: 0.95,
    }).bindPopup(
      `<strong>${point.latitude.toFixed(5)}, ${point.longitude.toFixed(5)}</strong><ul>${entries}</ul>`
    ).addTo(locationsLayer);
    markers.push(marker);
    recordItems.push(
      `<li class="location-coordinates"><strong>${point.latitude.toFixed(6)}, ${point.longitude.toFixed(6)}</strong><span>${point.rows.length} upload observation${point.rows.length === 1 ? "" : "s"}</span></li>`
      + point.rows.map((row) => {
        const species = (row.predictions || []).map((prediction) =>
          `${esc(prediction.species)} ×${Number(prediction.count) || 0}`
        ).join(", ") || "No confident species label";
        return `<li><strong>${esc(row.name)}</strong><span>${esc(species)} · ${esc(row.source || "browser upload location")} · ${esc(new Date(row.created_at).toLocaleString())}</span></li>`;
      }).join("")
    );
  }
  records.innerHTML = recordItems.join("");

  const bounds = window.L.featureGroup(markers).getBounds();
  if (markers.length === 1) locationsMap.setView(markers[0].getLatLng(), 12);
  else locationsMap.fitBounds(bounds.pad(0.15), { maxZoom: 12 });
  requestAnimationFrame(() => locationsMap.invalidateSize());
}

function refreshLocations() {
  return fetch(CONFIG.API_BASE + "/api/locations?limit=1000")
    .then((response) => {
      if (!response.ok) throw new Error(`Location refresh failed (${response.status}).`);
      return response.json();
    })
    .then(renderLocations);
}

initializeLocationsMap();
window.setInterval(() => {
  if (!backend.online) return;
  refreshLocations().catch((error) => console.error("Could not refresh map locations:", error));
}, CONFIG.MAP_REFRESH_MS);
function renderCalls(list) {
  $("#calls").innerHTML = list.length ? list.map((c) => `<li><span>${esc(c.species)} <em>${esc(c.scientific || "")}</em></span><span>${esc(c.time || "")} · ${pct(c.confidence)}%</span></li>`).join("")
    : `<li class="empty-state">No audio classifier is configured.</li>`;
}
function renderHistory(rows) {
  const label = { ok: "Reviewed", review: "Needs review", queued: "Queued" };
  $("#history").innerHTML = rows.length ? rows.map((h) => `
    <tr><td>${esc(h.name)}</td><td>${esc(h.type)}</td>
    <td>${esc(h.top)}${h.sci ? `<span class="sci">${esc(h.sci)}</span>` : ""}</td>
    <td>${h.conf ? h.conf + "%" : "—"}</td><td><span class="status ${h.status}">${label[h.status] || ""}</span></td></tr>`).join("")
    : `<tr><td colspan="5" class="empty-state">No analyses recorded yet.</td></tr>`;
  const tag = $(".p-history .tag"); if (tag) tag.textContent = rows.length ? "Recorded data" : "No data";
}

/* ---------- Upload + analysis flow (ready for FastAPI) ---------- */
(function upload() {
  const drop = $("#drop"), input = $("#fileInput"), idle = $("#dropIdle"), prev = $("#dropPreview");
  const media = $("#previewMedia"), nameEl = $("#fileName"), btn = $("#analyzeBtn"), note = $("#analyzeNote");
  let file = null, objectUrl = null;
  const MAX_MB = 200;
  const imageExtensions = new Set([".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"]);
  const videoExtensions = new Set([".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"]);

  const setFile = (f) => {
    if (!f) return;
    const extension = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    const kind = imageExtensions.has(extension) ? "image" : videoExtensions.has(extension) ? "video" : "";
    if (!kind) {
      clear(true);
      note.textContent = "Unsupported file. Use JPG, PNG, BMP, TIFF, WebP, MP4, MOV, AVI, MKV, M4V, MPEG, MPG, or WebM.";
      return;
    }
    if (f.size > MAX_MB * 1048576) {
      clear(true);
      note.textContent = `That file is larger than ${MAX_MB} MB. Choose a smaller file.`;
      return;
    }
    clear(true); file = f; objectUrl = URL.createObjectURL(f);
    media.innerHTML = kind === "image" ? `<img alt="Preview of ${esc(f.name)}" src="${objectUrl}">`
      : kind === "video" ? `<video src="${objectUrl}" muted playsinline loop autoplay></video>`
      : `<div class="audio-ph">Video file ready</div>`;
    nameEl.textContent = f.name;
    idle.hidden = true; prev.hidden = false; btn.disabled = false;
    note.textContent = !backend.online ? "Backend unavailable. Start the Python server before submitting."
      : backend.caps.detect && backend.caps.species ? "Ready to submit for analysis."
      : "Backend is online, but animal detection models are not configured.";
  };
  function clear(silent) {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    file = null; objectUrl = null; media.innerHTML = ""; input.value = "";
    idle.hidden = false; prev.hidden = true; btn.disabled = true; drop.classList.remove("loading");
    if (!silent) note.textContent = "Choose a file to begin.";
  }
  function getBrowserLocation() {
    if (!navigator.geolocation) {
      return Promise.resolve({
        coordinates: null,
        warning: "This browser does not support location. Analysis will continue without a map spot.",
      });
    }
    return new Promise((resolve) => {
      try {
        navigator.geolocation.getCurrentPosition(
          ({ coords }) => resolve({
            coordinates: { latitude: coords.latitude, longitude: coords.longitude },
            warning: "",
          }),
          (error) => {
            const message = {
              1: "Location permission was denied.",
              2: "Your current location is unavailable.",
              3: "The location request timed out.",
            }[error.code] || "The browser location could not be read.";
            resolve({
              coordinates: null,
              warning: `${message} Analysis will continue without a map spot.`,
            });
          },
          { enableHighAccuracy: false, maximumAge: 0, timeout: 15000 },
        );
      } catch {
        resolve({
          coordinates: null,
          warning: "Browser location is unavailable in this page context. Analysis will continue without a map spot.",
        });
      }
    });
  }

  drop.addEventListener("click", (e) => { if (!e.target.closest("#clearFile")) input.click(); });
  drop.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && e.target === drop) { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => setFile(input.files[0]));
  $("#clearFile").addEventListener("click", (e) => { e.stopPropagation(); clear(); });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); if (ev === "dragleave" && drop.contains(e.relatedTarget)) return; drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  btn.addEventListener("click", async () => {
    if (!file) return;
    btn.disabled = true; btn.innerHTML = `<span class="spin"></span> Analyzing…`; drop.classList.add("loading");
    note.textContent = "Requesting browser location…";
    let result = null, reason = "";
    let locationWarning = "";
    try {
      const { coordinates, warning } = await getBrowserLocation();
      locationWarning = warning;
      note.textContent = coordinates ? "Location received. Analyzing…" : "Location unavailable. Analyzing…";
      const fd = new FormData(); fd.append("file", file);
      if (coordinates) {
        fd.append("browser_latitude", String(coordinates.latitude));
        fd.append("browser_longitude", String(coordinates.longitude));
      }
      const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), CONFIG.ANALYSIS_TIMEOUT_MS);
      try {
        const res = await fetch(CONFIG.API_BASE + CONFIG.ANALYZE_PATH, { method: "POST", body: fd, signal: ctl.signal });
        if (!res.ok) { const j = await res.json().catch(() => ({})); throw new Error(j.detail || `Request failed (${res.status})`); }
        result = await res.json();
      } finally {
        clearTimeout(t);
      }
    } catch (err) {
      reason = err.name === "AbortError" ? "Analysis request timed out."
        : err instanceof TypeError
          ? "Could not reach the analysis API. Run `python app.py` and open http://127.0.0.1:8001."
          : String(err.message || "Analysis failed.");
    }
    drop.classList.remove("loading");
    btn.textContent = "Analyze file"; btn.disabled = false;
    if (result) {
      renderResults(result.candidates || []);
      renderDetections(result.detections || [], result.media_type, result.image_size);
      renderCalls(result.audio || []);
      note.textContent = [result.note || "Analysis complete.", locationWarning].filter(Boolean).join(" ");
      loadDashboard();
    } else {
      $("#resultTag").textContent = "No result";
      $("#results").innerHTML = `<li class="empty-state">No analysis result is available.</li>`;
      $("#detectedAnimals").innerHTML = `<li class="empty-state">No analysis result is available.</li>`;
      $("#calls").innerHTML = `<li class="empty-state">No analysis result is available.</li>`;
      note.textContent = reason || "Analysis could not be completed.";
    }
    $(".p-species").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest" });
  });
})();

/* ==========================================================
   MOTION + 3D LAYER
   ========================================================== */
const canHover = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

/* ---------- Kinetic hero headline (letters rise in 3D) ---------- */
(function splitTitle() {
  const t = $(".hero-title");
  if (!t || reduceMotion) return;
  t.setAttribute("aria-label", t.textContent.replace(/\s+/g, " ").trim());
  t.classList.remove("fade"); t.style.opacity = 1; t.style.transform = "none";
  let n = 0;
  const walk = (node) => {
    [...node.childNodes].forEach((c) => {
      if (c.nodeType === 3) {
        const frag = document.createDocumentFragment();
        c.textContent.split(/(\s+)/).forEach((part) => {
          if (!part) return;
          if (/^\s+$/.test(part)) { frag.appendChild(document.createTextNode(" ")); return; }
          const w = document.createElement("span"); w.className = "word"; w.setAttribute("aria-hidden", "true");
          [...part].forEach((ch) => { const s = document.createElement("span"); s.className = "ch"; s.style.setProperty("--i", n++); s.textContent = ch; w.appendChild(s); });
          frag.appendChild(w);
        });
        c.replaceWith(frag);
      } else if (c.nodeType === 1 && c.tagName !== "BR") walk(c);
    });
  };
  walk(t);
})();

/* ---------- Ticker of scientific names ---------- */
(function ticker() {
  const names = ["Panthera tigris", "Elephas maximus", "Panthera pardus", "Axis axis", "Buceros bicornis", "Rusa unicolor", "Ursus thibetanus", "Pavo cristatus", "Cuon alpinus", "Hoolock hoolock"];
  const html = names.map((n) => `<span>${n}</span>`).join("");
  const el = $("#tickerTrack"); if (el) el.innerHTML = html + html;
})();

/* ---------- Hero: drifting light motes ---------- */
(function particles() {
  const canvas = $("#particles");
  if (!canvas || reduceMotion) return;
  const ctx = canvas.getContext("2d"), hero = canvas.parentElement;
  let w, h, dpr, motes = [], running = true, raf;
  const spawn = (anywhere) => ({ x: Math.random() * w, y: anywhere ? Math.random() * h : h + 10, r: Math.random() * 1.7 + 0.4,
    vx: (Math.random() - 0.3) * 0.18, vy: -(Math.random() * 0.24 + 0.05), a: Math.random() * 0.5 + 0.15, ph: Math.random() * 6.28, gold: Math.random() > 0.5 });
  const resize = () => {
    dpr = Math.min(window.devicePixelRatio || 1, 2); w = hero.clientWidth; h = hero.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    motes = Array.from({ length: Math.round(Math.min(60, (w * h) / 26000)) }, () => spawn(true));
  };
  const tick = (t) => {
    if (!running) return;
    ctx.clearRect(0, 0, w, h);
    for (const m of motes) {
      m.x += m.vx + Math.sin(t / 2600 + m.ph) * 0.14; m.y += m.vy;
      if (m.y < -10 || m.x > w + 10 || m.x < -10) Object.assign(m, spawn(false));
      const tw = 0.55 + 0.45 * Math.sin(t / 1200 + m.ph);
      ctx.beginPath(); ctx.fillStyle = m.gold ? `rgba(240,205,140,${m.a * tw})` : `rgba(243,240,229,${m.a * tw * 0.8})`;
      ctx.shadowBlur = 8; ctx.shadowColor = m.gold ? "rgba(214,172,99,.8)" : "rgba(243,240,229,.6)";
      ctx.arc(m.x, m.y, m.r, 0, 6.283); ctx.fill();
    }
    raf = requestAnimationFrame(tick);
  };
  const setRun = (on) => { if (on && !running) { running = true; raf = requestAnimationFrame(tick); } else if (!on) { running = false; cancelAnimationFrame(raf); } };
  new IntersectionObserver(([e]) => setRun(e.isIntersecting), { threshold: 0 }).observe(hero);
  document.addEventListener("visibilitychange", () => setRun(!document.hidden));
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(resize, 200); });
  resize(); raf = requestAnimationFrame(tick);
})();

/* ---------- Scroll parallax inside photos ---------- */
(function photoParallax() {
  if (reduceMotion) return;
  const items = $$(".g, .impact-photo-wrap");
  let ticking = false;
  const update = () => {
    const vh = window.innerHeight;
    items.forEach((el) => {
      const r = el.getBoundingClientRect();
      if (r.bottom < -80 || r.top > vh + 80) return;
      const off = (r.top + r.height / 2 - vh / 2) / vh;       // -1 .. 1
      el.style.setProperty("--py", `${Math.max(-1, Math.min(1, off)) * -26}px`);
    });
    ticking = false;
  };
  window.addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
  update();
})();

/* ---------- Cursor-driven 3D: card tilt, hero depth, magnetic buttons ---------- */
(function depth3d() {
  if (reduceMotion || !canHover) return;
  $$(".g").forEach((el) => {
    const max = 8; let raf = 0;
    el.addEventListener("pointerenter", () => el.classList.add("tilting"));
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        el.style.setProperty("--ry", ((x - 0.5) * 2 * max).toFixed(2) + "deg");
        el.style.setProperty("--rx", ((0.5 - y) * 2 * max).toFixed(2) + "deg");
        el.style.setProperty("--tx", ((x - 0.5) * 2).toFixed(3)); el.style.setProperty("--ty", ((y - 0.5) * 2).toFixed(3));
        el.style.setProperty("--gx", (x * 100).toFixed(1) + "%"); el.style.setProperty("--gy", (y * 100).toFixed(1) + "%");
      });
    });
    el.addEventListener("pointerleave", () => {
      cancelAnimationFrame(raf); el.classList.remove("tilting");
      ["--rx", "--ry", "--tx", "--ty"].forEach((p) => el.style.removeProperty(p));
    });
  });

  const hero = $(".hero");
  if (hero) {
    let tx = 0, ty = 0, cx = 0, cy = 0, raf = 0;
    const loop = () => {
      cx += (tx - cx) * 0.07; cy += (ty - cy) * 0.07;
      hero.style.setProperty("--hx", cx.toFixed(4)); hero.style.setProperty("--hy", cy.toFixed(4));
      raf = (Math.abs(tx - cx) > 0.001 || Math.abs(ty - cy) > 0.001) ? requestAnimationFrame(loop) : 0;
    };
    hero.addEventListener("pointermove", (e) => {
      const r = hero.getBoundingClientRect();
      tx = ((e.clientX - r.left) / r.width - 0.5) * 2; ty = ((e.clientY - r.top) / r.height - 0.5) * 2;
      if (!raf) raf = requestAnimationFrame(loop);
    });
    hero.addEventListener("pointerleave", () => { tx = 0; ty = 0; if (!raf) raf = requestAnimationFrame(loop); });
  }

  $$(".hero-actions .btn, .nav .btn").forEach((b) => {
    b.addEventListener("pointermove", (e) => {
      const r = b.getBoundingClientRect();
      b.style.translate = `${((e.clientX - r.left) / r.width - 0.5) * 14}px ${((e.clientY - r.top) / r.height - 0.5) * 10}px`;
    });
    b.addEventListener("pointerleave", () => { b.style.translate = ""; });
  });
})();

/* ---------- 3D ring carousel of real photographs ---------- */
(function ring() {
  const stage = $("#ringStage"), ringEl = $("#ring");
  if (!stage || !ringEl) return;
  const cards = $$(".ring-card", ringEl), n = cards.length, step = 360 / n;
  const info = $("#ringInfo"), nameEl = $("#ringName"), sciEl = $("#ringSci"), factEl = $("#ringFact"), dotsEl = $("#ringDots");
  let angle = 0, vel = 0, target = null, dragging = false, lastX = 0, hover = false, raf = 0, visible = false, R = 330, active = -1;

  const dots = cards.map((c, i) => {
    const b = document.createElement("button"); b.type = "button"; b.setAttribute("aria-label", `Show ${c.dataset.name}`);
    b.addEventListener("click", () => focusCard(i)); dotsEl.appendChild(b); return b;
  });
  const snap = (a) => Math.round(a / step) * step;
  function focusCard(i) { const base = -i * step; target = base + Math.round((angle - base) / 360) * 360; vel = 0; }
  const layout = () => {
    const w = cards[0].offsetWidth || 290;
    R = Math.round(w / 2 / Math.tan(Math.PI / n) + (window.innerWidth < 640 ? 20 : 54));
    cards.forEach((c, i) => { c.style.transform = `rotateY(${i * step}deg) translateZ(${R}px)`; });
  };
  const setInfo = (i) => {
    if (i === active) return; active = i;
    info.classList.add("swap");
    setTimeout(() => {
      const c = cards[i];
      nameEl.textContent = c.dataset.name; sciEl.textContent = c.dataset.sci; factEl.textContent = c.dataset.fact;
      info.classList.remove("swap");
    }, active === -1 ? 0 : 180);
    dots.forEach((d, k) => d.classList.toggle("on", k === i));
  };
  const render = () => {
    ringEl.style.transform = `translateZ(${-R}px) rotateY(${angle}deg)`;
    cards.forEach((c, i) => {
      let rel = ((i * step + angle) % 360 + 540) % 360 - 180;          // -180..180, 0 = facing viewer
      c.style.setProperty("--f", Math.max(0, Math.cos(rel * Math.PI / 180)).toFixed(3));
    });
    setInfo((((Math.round(-angle / step)) % n) + n) % n);
  };
  const tick = () => {
    if (!visible) { raf = 0; return; }
    if (target !== null) { angle += (target - angle) * 0.12; if (Math.abs(target - angle) < 0.05) { angle = target; target = null; } }
    else if (!dragging) {
      angle += vel; vel *= 0.94; if (Math.abs(vel) < 0.02) vel = 0;
      if (!hover && !reduceMotion && vel === 0) angle -= 0.12;
    }
    render(); raf = requestAnimationFrame(tick);
  };

  stage.addEventListener("pointerdown", (e) => { dragging = true; target = null; lastX = e.clientX; vel = 0; stage.setPointerCapture(e.pointerId); });
  stage.addEventListener("pointermove", (e) => { if (!dragging) return; const dx = e.clientX - lastX; lastX = e.clientX; angle += dx * 0.32; vel = dx * 0.32; });
  const end = () => { dragging = false; };
  stage.addEventListener("pointerup", end); stage.addEventListener("pointercancel", end);
  stage.addEventListener("pointerenter", () => { hover = true; }); stage.addEventListener("pointerleave", () => { hover = false; });
  stage.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") { e.preventDefault(); target = snap(angle) - step; }
    if (e.key === "ArrowLeft") { e.preventDefault(); target = snap(angle) + step; }
  });
  $("#ringNext").addEventListener("click", () => { target = snap(target ?? angle) - step; });
  $("#ringPrev").addEventListener("click", () => { target = snap(target ?? angle) + step; });
  cards.forEach((c, i) => c.addEventListener("click", () => { if (Math.abs(vel) < 0.5) focusCard(i); }));

  layout(); render();
  window.addEventListener("resize", () => { layout(); render(); });
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible && !raf) raf = requestAnimationFrame(tick); }, { threshold: 0.1 }).observe(stage);
})();

/* ---------- Tell visitors if the 3D engine could not load ---------- */
setTimeout(() => {
  if (window.__studioReady) return;
  const m = $("#studioMsg");
  if (m) { m.classList.add("error"); m.innerHTML = "<p>The 3D studio could not start. It needs an internet connection and a web server such as VS Code Live Server (not a file:// address).</p>"; }
}, 12000);

/* ---------- Footer year ---------- */
$("#year").textContent = new Date().getFullYear();