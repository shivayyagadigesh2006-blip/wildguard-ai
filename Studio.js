/* ==========================================================
   WildGuard AI — studio.js  (3D animal studio, three.js)

   Loads animated 3D animals and lets visitors orbit them.
   Needs a web server (VS Code Live Server is fine) and an internet
   connection for three.js. It does not run from a file:// address.

   ADD YOUR OWN ANIMALS
   --------------------
   Put a .glb file in a /models folder (for example models/tiger.glb) and add
   an entry to MODELS below. Local files are tried first, then the CDN.
   ========================================================== */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

window.__studioReady = true;

const CDN = "https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/gltf/";
const MODELS = [
  { id: "flamingo", name: "Flamingo", sci: "Family Phoenicopteridae", lift: 0.55, fly: true,
    desc: "Flamingos feed by filtering tiny organisms from shallow water. The model includes a looping wing-flap animation.",
    urls: ["models/Flamingo.glb", CDN + "Flamingo.glb"] },
  { id: "parrot", name: "Parrot", sci: "Order Psittaciformes", lift: 0.55, fly: true,
    desc: "Parrots are among the most intelligent birds, and many species are threatened by habitat loss.",
    urls: ["models/Parrot.glb", CDN + "Parrot.glb"] },
  { id: "stork", name: "Stork", sci: "Family Ciconiidae", lift: 0.55, fly: true,
    desc: "Many storks migrate long distances and depend on healthy wetlands along the way.",
    urls: ["models/Stork.glb", CDN + "Stork.glb"] },
  { id: "horse", name: "Horse", sci: "Equus ferus caballus", lift: 0, fly: false,
    desc: "The model includes a looping gallop animation. Add a local wildlife model to include another species.",
    urls: ["models/Horse.glb", CDN + "Horse.glb"] },
];

const $ = (s) => document.querySelector(s);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const host = $("#studioCanvas"), canvas = $("#studioGL"), msg = $("#studioMsg"), hint = $("#studioHint");
const tabsEl = $("#studioTabs"), autoBox = $("#autoRotate");
if (!host || !canvas) throw new Error("Studio markup not found");

function showMsg(text, error = false) {
  msg.hidden = false; msg.classList.toggle("error", error);
  msg.innerHTML = error ? `<p>${text}</p>` : `<span class="spin-lg"></span><p>${text}</p>`;
}
const hideMsg = () => { msg.hidden = true; };

/* ---------- Renderer, scene, lights ---------- */
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
} catch (e) {
  showMsg("Your browser could not start WebGL, so the 3D studio is unavailable.", true);
  throw e;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 60);
camera.position.set(3.6, 1.9, 5.2);

scene.add(new THREE.HemisphereLight(0xf3f0e5, 0x16352a, 1.15));
const key = new THREE.DirectionalLight(0xffe0a8, 3.0);
key.position.set(4, 6.5, 3.5); key.castShadow = true;
key.shadow.mapSize.set(1024, 1024); key.shadow.bias = -0.0004;
Object.assign(key.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 1, far: 18 });
scene.add(key);
const fill = new THREE.DirectionalLight(0x9fc4a0, 0.9); fill.position.set(-4, 2.2, -3); scene.add(fill);

// Round stage with a thin gold rim
const stage = new THREE.Mesh(new THREE.CircleGeometry(3.1, 80), new THREE.MeshStandardMaterial({ color: 0x1b4030, roughness: 0.95 }));
stage.rotation.x = -Math.PI / 2; stage.receiveShadow = true; scene.add(stage);
const rim = new THREE.Mesh(new THREE.RingGeometry(3.1, 3.14, 96), new THREE.MeshBasicMaterial({ color: 0xd6ac63, transparent: true, opacity: 0.7, side: THREE.DoubleSide }));
rim.rotation.x = -Math.PI / 2; rim.position.y = 0.002; scene.add(rim);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.06;
controls.enablePan = false; controls.enableZoom = false;       // keep page scrolling smooth
controls.minPolarAngle = 0.35; controls.maxPolarAngle = Math.PI * 0.49;
controls.autoRotate = !reduceMotion; controls.autoRotateSpeed = 1.4;
controls.addEventListener("start", () => hint.classList.add("gone"));
autoBox.checked = controls.autoRotate;
autoBox.addEventListener("change", () => { controls.autoRotate = autoBox.checked; });

/* ---------- Sizing ---------- */
function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host); resize();

/* ---------- Model loading ---------- */
const loader = new GLTFLoader();
const cache = new Map();      // id -> { holder, mixer, def }
let current = null, token = 0;

function loadUrl(urls) {
  return new Promise((resolve, reject) => {
    const list = [...urls];
    const next = () => {
      const url = list.shift();
      if (!url) return reject(new Error("All sources failed"));
      loader.load(url, resolve, undefined, next);
    };
    next();
  });
}

function fit(def, gltf) {
  const holder = new THREE.Group();
  const obj = gltf.scene;
  obj.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = false; } });
  holder.add(obj);
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  obj.scale.multiplyScalar(2.3 / Math.max(size.x, size.y, size.z));
  box.setFromObject(obj);
  const c = box.getCenter(new THREE.Vector3());
  obj.position.x -= c.x; obj.position.z -= c.z; obj.position.y -= box.min.y;
  holder.position.y = def.lift;
  holder.userData.height = box.max.y - box.min.y;
  let mixer = null;
  if (gltf.animations && gltf.animations.length) {
    mixer = new THREE.AnimationMixer(obj);
    const action = mixer.clipAction(gltf.animations[0]);
    action.play();
    if (reduceMotion) mixer.timeScale = 0;
  }
  holder.visible = false; scene.add(holder);
  return { holder, mixer, def };
}

async function show(def) {
  const my = ++token;
  $("#studioName").textContent = def.name;
  $("#studioSci").textContent = def.sci;
  $("#studioDesc").textContent = def.desc;
  [...tabsEl.children].forEach((b) => b.setAttribute("aria-selected", String(b.dataset.id === def.id)));
  if (current) current.holder.visible = false;
  if (!cache.has(def.id)) {
    showMsg(`Loading ${def.name}…`);
    try { cache.set(def.id, fit(def, await loadUrl(def.urls))); }
    catch { if (my === token) showMsg("The 3D model could not be loaded. Check your internet connection and refresh.", true); return; }
  }
  if (my !== token) return;
  current = cache.get(def.id); current.holder.visible = true;
  controls.target.set(0, current.holder.position.y + current.holder.userData.height * 0.5, 0);
  hideMsg();
}

MODELS.forEach((def) => {
  const b = document.createElement("button");
  b.type = "button"; b.className = "tab"; b.setAttribute("role", "tab"); b.dataset.id = def.id; b.textContent = def.name;
  b.setAttribute("aria-selected", "false");
  b.addEventListener("click", () => show(def));
  tabsEl.appendChild(b);
});

/* ---------- Render loop (runs only while the studio is on screen) ---------- */
const clock = new THREE.Clock();
let running = false, started = false, onScreen = false, raf = 0;
function frame() {
  if (!running) return;
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
  if (current) {
    if (current.mixer) current.mixer.update(dt);
    if (current.def.fly && !reduceMotion) current.holder.position.y = current.def.lift + Math.sin(t * 1.6) * 0.07;
  }
  controls.update();
  renderer.render(scene, camera);
  raf = requestAnimationFrame(frame);
}
function setRunning(on) {
  if (on && !running) { running = true; clock.getDelta(); raf = requestAnimationFrame(frame); }
  else if (!on && running) { running = false; cancelAnimationFrame(raf); }
}
new IntersectionObserver(([e]) => {
  onScreen = e.isIntersecting;
  if (onScreen && !started) { started = true; show(MODELS[0]); }
  setRunning(onScreen && !document.hidden);
}, { threshold: 0.1 }).observe(host);
document.addEventListener("visibilitychange", () => setRunning(onScreen && !document.hidden));