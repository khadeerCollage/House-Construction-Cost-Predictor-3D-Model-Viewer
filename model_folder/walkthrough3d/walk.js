// =============================================================================
// Walk-through 3D — the generated plan as a house you can walk inside
// Scene JSON comes from floorplan_renderer/walkthrough.py (everything measured from the plan).
// Three.js is vendored (./vendor) — no network, no CDN, no AI service.
// Plan frame: metres, x right, y up (road side at the top). World: X = x, Y = height, Z = -y.
// =============================================================================
import * as THREE from "./vendor/three.module.min.js";

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const DEG = Math.PI / 180;
function rng(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const ZW = { N: "North", NE: "North-East", E: "East", SE: "South-East", S: "South", SW: "South-West", W: "West",
  NW: "North-West", CENTER: "Centre" };

// ───────────────────────────── Streamlit bridge (component protocol v1, no npm lib)
const ST = {
  send(type, data) { try { window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type }, data || {}), "*"); } catch (e) { /* standalone */ } },
  ready() { this.send("streamlit:componentReady", { apiVersion: 1 }); },
  height(h) { this.send("streamlit:setFrameHeight", { height: h }); },
  value(v) { this.send("streamlit:setComponentValue", { value: v, dataType: "json" }); },
};

// ───────────────────────────── renderer
const canvas = $("c");
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
} catch (e) {
  $("msg").textContent = "This 3D view needs WebGL. Please open the app in Chrome, Edge or Firefox.";
  throw e;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
const MAX_ANISO = Math.min(8, renderer.capabilities.getMaxAnisotropy());

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(62, 1, 0.05, 700);
const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
const sun = new THREE.DirectionalLight(0xffffff, 2.5);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.03;
const night = new THREE.AmbientLight(0xFFD9A0, 0);
scene.add(hemi, sun, sun.target, night);

// sky dome for the real-colours look (follows the camera)
const skyMat = new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false, fog: false,
  uniforms: { top: { value: new THREE.Color(0x4F8FD0) }, hor: { value: new THREE.Color(0xDCEBF5) }, bot: { value: new THREE.Color(0x9C927E) } },
  vertexShader: "varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
  fragmentShader: "uniform vec3 top; uniform vec3 hor; uniform vec3 bot; varying vec3 vP;" +
    "void main(){ float h = vP.y; vec3 c = h > 0.0 ? mix(hor, top, pow(h, 0.55)) : mix(hor, bot, pow(-h, 0.4));" +
    " gl_FragColor = vec4(c, 1.0);\n#include <tonemapping_fragment>\n#include <colorspace_fragment>\n}",
});
const sky = new THREE.Mesh(new THREE.SphereGeometry(500, 32, 16), skyMat);
sky.frustumCulled = false;
scene.add(sky);

// ───────────────────────────── procedural textures (canvas, deterministic)
function shade(hex, k) { const c = new THREE.Color(hex); c.offsetHSL(0, 0, k); return "#" + c.getHexString(); }
function canvasTex(size, draw) {
  const c = document.createElement("canvas"); c.width = c.height = size;
  draw(c.getContext("2d"), size);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = MAX_ANISO;
  return t;
}
function tileTex(base, grout, n, vary, seed) {
  return canvasTex(512, (g, s) => {
    const r = rng(seed), st = s / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      g.fillStyle = shade(base, (r() - 0.5) * vary); g.fillRect(i * st, j * st, st, st);
      g.globalAlpha = 0.22;
      for (let q = 0; q < 90; q++) { g.fillStyle = shade(base, (r() - 0.5) * vary * 3); g.fillRect(i * st + r() * st, j * st + r() * st, 3, 3); }
      g.globalAlpha = 1;
    }
    g.strokeStyle = grout; g.lineWidth = 3;
    for (let i = 0; i <= n; i++) { g.beginPath(); g.moveTo(i * st, 0); g.lineTo(i * st, s); g.stroke(); g.beginPath(); g.moveTo(0, i * st); g.lineTo(s, i * st); g.stroke(); }
  });
}
function speckleTex(base, vary, seed, dots = 2600) {
  return canvasTex(256, (g, s) => {
    const r = rng(seed); g.fillStyle = base; g.fillRect(0, 0, s, s);
    for (let q = 0; q < dots; q++) { g.globalAlpha = 0.35; g.fillStyle = shade(base, (r() - 0.5) * vary * 2); g.fillRect(r() * s, r() * s, 1 + r() * 2.5, 1 + r() * 2.5); }
    g.globalAlpha = 1;
  });
}
function gridTex(bg, minor, major, span, perM) {
  return canvasTex(512, (g, s) => {
    g.fillStyle = bg; g.fillRect(0, 0, s, s);
    const pxm = s / span;
    g.strokeStyle = minor; g.lineWidth = 1.5;
    for (let k = 0; k <= span * perM; k++) { const p = k * pxm / perM; g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.stroke(); g.beginPath(); g.moveTo(0, p); g.lineTo(s, p); g.stroke(); }
    g.strokeStyle = major; g.lineWidth = 3;
    for (let k = 0; k <= span; k++) { const p = k * pxm; g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.stroke(); g.beginPath(); g.moveTo(0, p); g.lineTo(s, p); g.stroke(); }
  });
}
function woodTex(base, seed, panels = false) {
  return canvasTex(512, (g, s) => {
    const r = rng(seed); g.fillStyle = base; g.fillRect(0, 0, s, s);
    for (let k = 0; k < 140; k++) {
      const x = r() * s; g.strokeStyle = shade(base, (r() - 0.5) * 0.12); g.globalAlpha = 0.5; g.lineWidth = 1 + r() * 3;
      g.beginPath(); g.moveTo(x, 0); for (let y = 0; y <= s; y += 32) g.lineTo(x + Math.sin(y * 0.02 + k) * 4, y); g.stroke();
    }
    g.globalAlpha = 1;
    if (panels) { g.strokeStyle = shade(base, -0.12); g.lineWidth = 6; g.strokeRect(s * 0.15, s * 0.08, s * 0.7, s * 0.36); g.strokeRect(s * 0.15, s * 0.56, s * 0.7, s * 0.36); }
  });
}
function marbleTex() {
  return canvasTex(512, (g, s) => {
    const r = rng(31); g.fillStyle = "#F3F1EC"; g.fillRect(0, 0, s, s);
    g.strokeStyle = "rgba(120,120,125,0.35)";
    for (let k = 0; k < 18; k++) { g.lineWidth = 0.6 + r() * 1.6; g.beginPath(); let x = r() * s, y = 0; g.moveTo(x, y); while (y < s) { x += (r() - 0.5) * 40; y += 20 + r() * 30; g.lineTo(x, y); } g.stroke(); }
    g.strokeStyle = "#C9C5BC"; g.lineWidth = 3; g.strokeRect(0, 0, s, s); g.beginPath(); g.moveTo(s / 2, 0); g.lineTo(s / 2, s); g.moveTo(0, s / 2); g.lineTo(s, s / 2); g.stroke();
  });
}
function grassTex() {
  return canvasTex(256, (g, s) => {
    const r = rng(41); g.fillStyle = "#5C8A3A"; g.fillRect(0, 0, s, s);
    for (let q = 0; q < 5000; q++) { g.fillStyle = shade("#5C8A3A", (r() - 0.5) * 0.22); g.fillRect(r() * s, r() * s, 1, 2 + r() * 3); }
  });
}
function paverTex() {
  return canvasTex(512, (g, s) => {
    const r = rng(51); const bw = s / 4, bh = s / 8;
    for (let j = 0; j < 8; j++) for (let i = -1; i < 5; i++) {
      const x = i * bw + (j % 2) * bw / 2; g.fillStyle = shade("#9A8F84", (r() - 0.5) * 0.08); g.fillRect(x, j * bh, bw, bh);
      g.strokeStyle = "#6E655C"; g.lineWidth = 3; g.strokeRect(x, j * bh, bw, bh);
    }
  });
}
const TEXDEF = {
  bp: () => gridTex("#0A2342", "#123A63", "#1C5A92", 2, 4),
  bpGround: () => gridTex("#061A31", "#0A2846", "#103A63", 4, 1),
  tile: () => tileTex("#E9E3D8", "#BDB3A3", 2, 0.035, 11),
  tileWarm: () => tileTex("#E4D4BC", "#B9A78D", 2, 0.04, 12),
  wet: () => tileTex("#C6D1D5", "#8D9CA2", 4, 0.05, 13),
  bathwall: () => tileTex("#EAF1F3", "#BFCBCF", 4, 0.02, 14),
  marble: marbleTex,
  granite: () => speckleTex("#3C3C3F", 0.2, 15),
  stone: () => speckleTex("#5E5A55", 0.14, 16),
  plaster: () => speckleTex("#D9CEB9", 0.05, 17),
  yard: () => speckleTex("#C8BBA1", 0.08, 18),
  soil: () => speckleTex("#A8956F", 0.12, 19),
  grass: grassTex,
  asphalt: () => speckleTex("#3B3E42", 0.12, 20),
  paver: paverTex,
  teak: () => woodTex("#7A4A25", 21, true),
  woodMed: () => woodTex("#A47A50", 22),
  woodDark: () => woodTex("#5B3920", 23),
  lam: () => woodTex("#CDB38F", 24),
};
const TEX = {};
const tex = (k) => TEX[k] || (TEX[k] = TEXDEF[k]());

// ───────────────────────────── materials: every surface has a role with a blueprint and a real look
const NAVY = 0x1E4570, FILL = 0x123A63;
const MAT_DEF = {
  wall_out: [{ c: 0x22507E }, { c: 0xEDE3D1, r: 0.95 }],
  wall_in: [{ c: NAVY }, { c: 0xF4F1EA }],
  in_living: [{ c: NAVY }, { c: 0xF3E9D8 }],
  in_master: [{ c: NAVY }, { c: 0xE8D9C4 }],
  in_child: [{ c: NAVY }, { c: 0xD8E8F3 }],
  in_bed: [{ c: NAVY }, { c: 0xE0ECDB }],
  in_kitchen: [{ c: NAVY }, { c: 0xF6F1E6 }],
  in_bath: [{ c: NAVY }, { map: "bathwall", uv: 1.2, r: 0.4 }],
  in_pooja: [{ c: NAVY }, { c: 0xFFEFC8 }],
  plinth: [{ c: 0x0F2C4D }, { map: "stone", uv: 1.5, r: 0.9 }],
  slab: [{ c: 0x1B3A5C }, { c: 0xDAD4C9 }],
  ceiling: [{ c: 0x0E2A4A }, { c: 0xF7F7F4 }],
  parapet: [{ c: 0x1B3A5C }, { c: 0xE6D9C3 }],
  compound: [{ c: 0x173556 }, { map: "plaster", uv: 2, r: 0.95 }],
  pillar: [{ c: 0x1B3A5C }, { c: 0xB9764F }],
  column: [{ c: 0x1B3A5C }, { c: 0xCBC5BA }],
  fl_tile: [{ map: "bp", uv: 2 }, { map: "tile", uv: 1.2, r: 0.35 }],
  fl_bed: [{ map: "bp", uv: 2 }, { map: "tileWarm", uv: 1.2, r: 0.4 }],
  fl_wet: [{ map: "bp", uv: 2 }, { map: "wet", uv: 1.2, r: 0.7 }],
  fl_marble: [{ map: "bp", uv: 2 }, { map: "marble", uv: 1.2, r: 0.25 }],
  stair: [{ c: 0x1B3A5C }, { map: "granite", uv: 1, r: 0.45 }],
  rail: [{ c: 0x00DDFF, e: 0x004455 }, { c: 0x9AA4AE, m: 0.8, r: 0.3 }],
  glass: [{ c: 0x00BFFF, o: 0.16 }, { c: 0xBFD9E6, o: 0.24, r: 0.05 }],
  frost: [{ c: 0x00BFFF, o: 0.3 }, { c: 0xE6EDF1, o: 0.72, r: 0.3 }],
  frame: [{ c: 0x00DDFF, e: 0x003344 }, { c: 0xF2F2F2, r: 0.5 }],
  door_main: [{ c: 0x0E4A3A, o: 0.92 }, { map: "teak", uv: 1, r: 0.55 }],
  door_int: [{ c: 0x0E4A3A, o: 0.85 }, { map: "lam", uv: 1, r: 0.6 }],
  door_bath: [{ c: 0x0E4A3A, o: 0.85 }, { c: 0xE5E0D6, r: 0.5 }],
  door_steel: [{ c: 0x0E4A3A, o: 0.9 }, { c: 0xA9B2BA, m: 0.7, r: 0.35 }],
  dframe: [{ c: 0x0B3D30 }, { map: "teak", uv: 1, r: 0.6 }],
  handle: [{ c: 0x00FF88 }, { c: 0xC9A43B, m: 0.9, r: 0.3 }],
  gate: [{ c: 0x00C87A, e: 0x003322 }, { c: 0x2A2C2E, m: 0.6, r: 0.45 }],
  wood: [{ c: FILL }, { map: "woodMed", uv: 1, r: 0.6 }],
  wood_dark: [{ c: 0x0D2A4A }, { map: "woodDark", uv: 1, r: 0.55 }],
  laminate: [{ c: FILL }, { c: 0xEAE2D4, r: 0.6 }],
  fabric: [{ c: 0x12406B }, { c: 0x6E7E8E, r: 0.95 }],
  fabric2: [{ c: 0x0F3A60 }, { c: 0x9C6B4E, r: 0.95 }],
  mattress: [{ c: 0x1A4C7A }, { c: 0xF4F1EA, r: 0.95 }],
  pillow: [{ c: 0x2A6496 }, { c: 0xFFFFFF, r: 0.95 }],
  blanket: [{ c: 0x1E5A8C }, { c: 0x3E6E9C, r: 0.95 }],
  white: [{ c: 0x2A5C8C }, { c: 0xF7F7F7, r: 0.2 }],
  steel: [{ c: 0x2A5C8C }, { c: 0xB8C0C8, m: 0.85, r: 0.3 }],
  black: [{ c: 0x061426 }, { c: 0x16181B, r: 0.3 }],
  screen: [{ c: 0x02101F, e: 0x00334F }, { c: 0x0B0D10, r: 0.15, e: 0x05090F }],
  counter: [{ c: FILL }, { c: 0x22272A, r: 0.25 }],
  cabinet: [{ c: FILL }, { c: 0xEFE8DC, r: 0.5 }],
  brass: [{ c: 0xFFC857, e: 0x332200 }, { c: 0xC9A43B, m: 0.9, r: 0.3 }],
  flame: [{ c: 0xFFB000, e: 0xFF9000 }, { c: 0xFFB000, e: 0xFF8A00 }],
  mirror: [{ c: 0x2A6496 }, { c: 0xDCE6EE, m: 1.0, r: 0.05 }],
  plant: [{ c: 0x1E7A5A }, { c: 0x3C7D3A, r: 0.9 }],
  pot: [{ c: 0x1B3A5C }, { c: 0xA85A32, r: 0.9 }],
  books1: [{ c: 0x1E5A8C }, { c: 0x8E3B2E, r: 0.8 }],
  books2: [{ c: 0x2A6496 }, { c: 0x2F5D7C, r: 0.8 }],
  fan: [{ c: 0x2A6496 }, { c: 0xF0EDE6, r: 0.5 }],
  lamp: [{ c: 0xFFE9A8, e: 0x221A08 }, { c: 0xFFF7E0, e: 0x000000 }],
  land: [{ map: "bpGround", uv: 4 }, { map: "soil", uv: 6, r: 1 }],
  yard: [{ map: "bp", uv: 2 }, { map: "yard", uv: 3, r: 0.95 }],
  grass: [{ c: 0x0F4A35, o: 0.9 }, { map: "grass", uv: 3, r: 1 }],
  road: [{ c: 0x061629 }, { map: "asphalt", uv: 4, r: 0.95 }],
  verge: [{ c: 0x081E38 }, { c: 0xB8AD97, r: 1 }],
  roadline: [{ c: 0x00FFFF, e: 0x006666 }, { c: 0xF4F4F4, r: 0.6 }],
  paver: [{ c: 0x0C2A4D }, { map: "paver", uv: 1.2, r: 0.9 }],
  concrete: [{ c: 0x1B3A5C }, { c: 0xBDB8AE, r: 0.95 }],
  water: [{ c: 0x0A6BA8, o: 0.82, e: 0x002233 }, { c: 0x3BA7D9, o: 0.86, r: 0.08 }],
  genset: [{ c: 0x5C4A10 }, { c: 0x2F6B3A, r: 0.6 }],
  cabin: [{ c: 0x2E1F5C }, { c: 0xE8E0D0, r: 0.9 }],
  leaf: [{ c: 0x1E7A5A, o: 0.95 }, { c: 0x3F7F35, r: 0.9 }],
  leaf2: [{ c: 0x1A6B50 }, { c: 0x5A8F3C, r: 0.9 }],
  trunk: [{ c: 0x1B3A5C }, { c: 0x6B4A2E, r: 1 }],
  tyre: [{ c: 0x061426 }, { c: 0x1A1A1A, r: 0.9 }],
  car_glass: [{ c: 0x0B2238 }, { c: 0x1C2833, r: 0.1, m: 0.3 }],
  car0: [{ c: 0x2B5C8A }, { c: 0xF2F2F0, r: 0.35, m: 0.2 }],
  car1: [{ c: 0x2B5C8A }, { c: 0xA9B0B6, r: 0.3, m: 0.5 }],
  car2: [{ c: 0x2B5C8A }, { c: 0x9E1B1B, r: 0.35, m: 0.2 }],
  car3: [{ c: 0x2B5C8A }, { c: 0x1F3F7A, r: 0.35, m: 0.2 }],
  ghost: [{ c: 0x1B3A5C, o: 0.22 }, { c: 0xF2EDE4, o: 0.38 }],
  manhole: [{ c: 0x061426 }, { c: 0x4A4A4A, m: 0.6, r: 0.5 }],
  arrow: [{ c: 0x00FFFF, e: 0x006666 }, { c: 0xC0392B, r: 0.6 }],
  lift_car: [{ c: 0x1E4570 }, { c: 0xC9CED3, m: 0.75, r: 0.3 }],
  lift_floor: [{ c: 0x0E2A4A }, { map: "granite", uv: 1, r: 0.4 }],
  lift_light: [{ c: 0xE6FFFF, e: 0x66CCCC }, { c: 0xFFFFFF, e: 0xFFF2D6 }],
  lift_frame: [{ c: 0x00DDFF, e: 0x003344 }, { c: 0x9AA4AE, m: 0.85, r: 0.25 }],
  lift_door: [{ c: 0x15466F }, { c: 0xB4BCC4, m: 0.6, r: 0.45 }],
  lift_tower: [{ c: 0x24588A }, { c: 0xD3D6DA, r: 0.9 }],
  machine: [{ c: 0x2A5C8C }, { c: 0x3A4048, m: 0.6, r: 0.5 }],
  cwt: [{ c: 0x1B5A8C }, { c: 0x5A5F66, m: 0.4, r: 0.7 }],
  buffer: [{ c: 0xFFC857, e: 0x332200 }, { c: 0xE0B000, r: 0.5 }],
};
const NO_SHADOW = new Set(["glass", "frost", "water", "ghost", "grass", "land", "yard", "road", "verge", "roadline", "paver", "fl_tile",
  "fl_bed", "fl_wet", "fl_marble", "lamp", "flame", "arrow", "ceiling", "lift_light"]);
const MATS = { bp: {}, real: {} };
function makeMat(spec, theme) {
  const p = { color: spec.map ? 0xffffff : spec.c, roughness: spec.r ?? (theme === "bp" ? 0.92 : 0.85), metalness: spec.m ?? 0 };
  if (spec.map) { const t = tex(spec.map).clone(); t.needsUpdate = true; t.repeat.set(1 / spec.uv, 1 / spec.uv); p.map = t; }
  if (spec.o !== undefined) { p.transparent = true; p.opacity = spec.o; p.depthWrite = spec.o > 0.6; }
  if (spec.e !== undefined) { p.emissive = new THREE.Color(spec.e); p.emissiveIntensity = 1; }
  const m = new THREE.MeshStandardMaterial(p);
  if (spec.o !== undefined && spec.o < 0.6) m.side = THREE.DoubleSide;
  return m;
}
function mat(role, theme = state.theme) {
  const store = MATS[theme];
  if (!store[role]) { const d = MAT_DEF[role] || MAT_DEF.wall_in; store[role] = makeMat(theme === "bp" ? d[0] : d[1], theme); }
  return store[role];
}
const LINE_DEF = { wall: [0x00FFFF, 0.8], furn: [0x6FA8DC, 0.75], door: [0x00FF88, 0.95], win: [0x00DDFF, 0.9],
  site: [0x3E9CD8, 0.7], stair: [0x00E5FF, 0.85], ghost: [0x2E7FB8, 0.55], gate: [0x00FF88, 0.8] };
const LINE_MAT = {};
const lineMat = (k) => LINE_MAT[k] || (LINE_MAT[k] = new THREE.LineBasicMaterial({ color: LINE_DEF[k][0], transparent: true, opacity: LINE_DEF[k][1] }));

// ───────────────────────────── geometry batching (one mesh per surface role)
class Batch {
  constructor() { this.p = []; this.n = []; this.t = []; }
  _uv(x, y, z, nx, ny, nz) {
    if (Math.abs(ny) > 0.5) this.t.push(x, -z);
    else if (Math.abs(nx) > 0.5) this.t.push(-z * Math.sign(nx), y);
    else this.t.push(x * Math.sign(nz || 1), y);
  }
  quad(a, b, c, d, n) {
    for (const v of [a, b, c, a, c, d]) { this.p.push(v[0], v[1], v[2]); this.n.push(n[0], n[1], n[2]); this._uv(v[0], v[1], v[2], n[0], n[1], n[2]); }
  }
  geom(g) {
    const gg = g.index ? g.toNonIndexed() : g;
    const P = gg.attributes.position.array, N = gg.attributes.normal.array;
    for (let i = 0; i < P.length; i += 3) {
      this.p.push(P[i], P[i + 1], P[i + 2]); this.n.push(N[i], N[i + 1], N[i + 2]);
      this._uv(P[i], P[i + 1], P[i + 2], N[i], N[i + 1], N[i + 2]);
    }
    g.dispose(); if (gg !== g) gg.dispose();
  }
  get empty() { return this.p.length === 0; }
  build() {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.p, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(this.n, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(this.t, 2));
    g.computeBoundingSphere();
    return g;
  }
}
class Lines {
  constructor() { this.p = []; }
  seg(a, b) { this.p.push(a[0], a[1], a[2], b[0], b[1], b[2]); }
  box(b) {
    const [x0, y0, x1, y1, z0, z1] = b, X = [x0, x1], Y = [z0, z1], Z = [-y1, -y0];
    for (const y of Y) for (const z of Z) this.seg([X[0], y, z], [X[1], y, z]);
    for (const x of X) for (const z of Z) this.seg([x, Y[0], z], [x, Y[1], z]);
    for (const x of X) for (const y of Y) this.seg([x, y, Z[0]], [x, y, Z[1]]);
  }
  build(kind) {
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(this.p, 3));
    const l = new THREE.LineSegments(g, lineMat(kind)); l.userData.edge = true; return l;
  }
}
// faces of a plan box [x0, y0, x1, y1, z0, z1]; pn = plan-frame normal, c = plan centre of the face
function boxFaces(b) {
  const [x0, y0, x1, y1, z0, z1] = b, ax = x0, bx = x1, ay = z0, by = z1, az = -y1, bz = -y0, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  return [
    { n: [1, 0, 0], v: [[bx, ay, bz], [bx, ay, az], [bx, by, az], [bx, by, bz]], c: [x1, cy], pn: [1, 0] },
    { n: [-1, 0, 0], v: [[ax, ay, az], [ax, ay, bz], [ax, by, bz], [ax, by, az]], c: [x0, cy], pn: [-1, 0] },
    { n: [0, 1, 0], v: [[ax, by, bz], [bx, by, bz], [bx, by, az], [ax, by, az]], c: [cx, cy], top: true },
    { n: [0, -1, 0], v: [[ax, ay, az], [bx, ay, az], [bx, ay, bz], [ax, ay, bz]], c: [cx, cy], bottom: true },
    { n: [0, 0, 1], v: [[ax, ay, bz], [bx, ay, bz], [bx, by, bz], [ax, by, bz]], c: [cx, y0], pn: [0, -1] },
    { n: [0, 0, -1], v: [[bx, ay, az], [ax, ay, az], [ax, by, az], [bx, by, az]], c: [cx, y1], pn: [0, 1] },
  ];
}
class Builder {           // batches by role + edge lines by kind, emitted into one group
  constructor() { this.b = {}; this.l = {}; }
  B(role) { return this.b[role] || (this.b[role] = new Batch()); }
  L(kind) { return this.l[kind] || (this.l[kind] = new Lines()); }
  box(b, role, edge = null, route = null) {
    if (b[2] - b[0] < 1e-4 || b[3] - b[1] < 1e-4 || b[5] - b[4] < 1e-4) return;
    for (const f of boxFaces(b)) {
      const r = route ? route(f) : role;
      if (r) this.B(r).quad(f.v[0], f.v[1], f.v[2], f.v[3], f.n);
    }
    if (edge) this.L(edge).box(b);
  }
  wall(b, edge, route, cuts) {        // like box(), but side faces are split where rooms change
    if (b[2] - b[0] < 1e-4 || b[3] - b[1] < 1e-4 || b[5] - b[4] < 1e-4) return;
    const faces = boxFaces(b);
    faces.forEach((f, k) => {
      if (f.top || f.bottom || !f.pn) { const r = route(f); if (r) this.B(r).quad(f.v[0], f.v[1], f.v[2], f.v[3], f.n); return; }
      const alongX = f.pn[1] !== 0, lo = alongX ? b[0] : b[1], hi = alongX ? b[2] : b[3];
      const pts = [lo, ...(alongX ? cuts.x : cuts.y).filter((c) => c > lo + 0.02 && c < hi - 0.02), hi];
      for (let i = 0; i + 1 < pts.length; i++) {
        const sb = alongX ? [pts[i], b[1], pts[i + 1], b[3], b[4], b[5]] : [b[0], pts[i], b[2], pts[i + 1], b[4], b[5]];
        const sf = boxFaces(sb)[k], r = route(sf);
        if (r) this.B(r).quad(sf.v[0], sf.v[1], sf.v[2], sf.v[3], sf.n);
      }
    });
    if (edge) this.L(edge).box(b);
  }
  quadUp(x0, y0, x1, y1, z, role) {   // horizontal rectangle facing up
    this.B(role).quad([x0, z, -y0], [x1, z, -y0], [x1, z, -y1], [x0, z, -y1], [0, 1, 0]);
  }
  quadDown(x0, y0, x1, y1, z, role) {
    this.B(role).quad([x0, z, -y1], [x1, z, -y1], [x1, z, -y0], [x0, z, -y0], [0, -1, 0]);
  }
  cyl(x, y, r, z0, z1, role, seg = 16, rTop = null) {
    const g = new THREE.CylinderGeometry(rTop ?? r, r, z1 - z0, seg); g.translate(x, (z0 + z1) / 2, -y); this.B(role).geom(g);
  }
  emit(group, pickable = null) {
    for (const [role, b] of Object.entries(this.b)) {
      if (b.empty) continue;
      const m = new THREE.Mesh(b.build(), mat(role));
      m.userData.role = role; m.castShadow = !NO_SHADOW.has(role); m.receiveShadow = true;
      if (role === "ceiling") m.castShadow = true;
      group.add(m); registry.meshes.push(m);
      if (pickable && pickable.has(role)) registry.pick.push(m);
    }
    for (const [kind, l] of Object.entries(this.l)) {
      if (!l.p.length) continue;
      const ls = l.build(kind); group.add(ls); registry.edges.push(ls);
    }
  }
}

// ───────────────────────────── state
const state = {
  theme: "bp", mode: "outside", furn: true, vastu: false, sunOpen: false, view: 0, explode: false,
  sunMin: 540, day: dayOfYear(new Date()), voice: true, started: false, xray: false, sound: true,
};
const registry = { meshes: [], edges: [], pick: [], doors: [], gates: [], fans: [], labels: [], lamps: [] };
let S = null;            // scene data
let SID = null;
let canWish = false;
let G = {};              // groups
let nav = null, vis = null, navs = { g0: null, typ: null, top: null };   // walking grids by kind of level
let levelGroups = [];    // [{L, root, ceil, furn}] — one per level with rooms (and the stilt)
let segCells = null;     // per walking cell: -1 off the stairs, 0 lower half of a flight, 1 upper half
const GAP = 2.8;         // exploded view: extra space between floors
const walker = { x: 0, y: 0, yaw: 0, pitch: -0.05, eye: 1.6, h: 2.0, path: [], lastLook: 0, room: null, level: 0, seg: null };
const orbit = { target: new THREE.Vector3(), r: 20, theta: 0, phi: 1.0, auto: false };
const trans = { on: false, t: 0, dur: 1, fromP: new THREE.Vector3(), fromT: new THREE.Vector3(), toP: new THREE.Vector3(), toT: new THREE.Vector3(), done: null };
const keys = new Set();
let dirtyUntil = 0;
const poke = (ms = 400) => { dirtyUntil = Math.max(dirtyUntil, performance.now() + ms); };

function dayOfYear(d) { return Math.floor((d - new Date(d.getFullYear(), 0, 0)) / 86400000); }

// ───────────────────────────── build the house from the scene JSON
function clearScene() {
  const seen = new Set();
  const drop = (g) => { g.traverse((o) => { if (o.geometry && !seen.has(o.geometry)) { seen.add(o.geometry); o.geometry.dispose(); } }); scene.remove(g); };
  for (const k of Object.keys(G)) drop(G[k]);
  for (const lg of levelGroups) drop(lg.root);
  levelGroups = [];
  registry.meshes = []; registry.edges = []; registry.pick = []; registry.doors = []; registry.gates = []; registry.fans = []; registry.lamps = [];
  for (const l of registry.labels) l.el.remove();
  registry.labels = [];
}

const planOf = (L) => (S && S.levels_list && S.levels_list[L] ? (S.levels_list[L].plan || 0) : 0);
const PL = (p) => (p === 1 && S.upper ? S.upper : S);          // the floor plan a level shows (a duplex has two)
function roomAt(x, y, p = planOf(walker.level)) {
  if (!S) return null;
  for (const r of PL(p).rooms) { const [x0, y0, x1, y1] = r.r; if (x > x0 && x < x1 && y > y0 && y < y1) return r; }
  return null;
}
const ORD = (n) => n + (["th", "st", "nd", "rd"][(n % 100 > 10 && n % 100 < 14) ? 0 : Math.min(n % 10, 4) % 4] || "th");
const homeName = (L) => (L <= 0 ? "Ground-floor home" : `${ORD(L)}-floor home`);
const houseKind = () => (S && S.apartment && S.apartment.kind === "house");
function inFootprint(x, y, pad = 0) {
  const [fx, fy, fw, fd] = S.footprint;
  return x > fx - pad && x < fx + fw + pad && y > fy - pad && y < fy + fd + pad;
}
const PAINT = { LIVING: "in_living", DINING: "in_living", CORRIDOR: "in_living", FOYER: "in_living", STUDY: "in_child",
  KITCHEN: "in_kitchen", UTILITY: "in_kitchen", BATHROOM: "in_bath", POOJA: "in_pooja", STORE: "wall_in", SERVANT: "in_bed",
  STAIRS: "wall_in", LIFT: "wall_in" };
function paintFor(room) {
  if (!room) return "wall_in";
  if (room.cat === "BEDROOM") return room.role === "MASTER_BEDROOM" ? "in_master" : room.role === "CHILD_BEDROOM" ? "in_child" : "in_bed";
  return PAINT[room.cat] || "wall_in";
}
const FLOOR = { BEDROOM: "fl_bed", BATHROOM: "fl_wet", UTILITY: "fl_wet", POOJA: "fl_marble", LIFT: "black" };

// level helpers (levels: [stilt,] floors with rooms, terrace)
const R0 = () => (S && S.r0) || 0;
const TOP = () => (S && S.top !== undefined ? S.top : 1);
const zf = (L) => (S.levels_list && S.levels_list[L] ? S.levels_list[L].z : S.levels.ffl + L * S.levels.storey);
const levelKind = (L) => (S.levels_list && S.levels_list[L] ? S.levels_list[L].kind : (L === 0 ? "rooms" : "terrace"));
const levelLabel = (L) => (S.levels_list && S.levels_list[L] ? S.levels_list[L].label : (L === 0 ? "Ground floor" : "Terrace"));
const hasRooms = (L) => levelKind(L) === "rooms";
function activeLevel() {
  if (state.mode === "walk") return walker.level;
  if (state.mode === "outside" && !state.explode) return R0();
  return state.view;
}

function build(data) {
  clearScene();
  S = data;
  const lv = S.levels, ffl = lv.ffl, H = lv.storey, r0 = R0(), top = TOP();
  G = { site: new THREE.Group(), roof: new THREE.Group(), dyn: new THREE.Group(), dyn2: new THREE.Group(),
    vastu: new THREE.Group(), lift: new THREE.Group(), ghost: new THREE.Group(), core: new THREE.Group() };
  liftReset();
  for (const g of Object.values(G)) scene.add(g);
  levelGroups = [];
  const pickRoles = new Set(["fl_tile", "fl_bed", "fl_wet", "fl_marble", "black", "yard", "grass", "paver", "road", "verge", "land", "stair", "concrete"]);

  // ════ the designed floor, built once at level r0, then repeated on every floor with rooms (a duplex: two plans)
  const makeTemplate = (P, p) => {
    const body = new Builder(), stairsB = new Builder(), furnB = new Builder(), ceilB = new Builder();
    const groundX = new Builder(), upperX = new Builder();
    const route = (f) => {
      if (f.bottom) return null;
      if (f.top) return "wall_in";
      const px = f.c[0] + f.pn[0] * 0.2, py = f.c[1] + f.pn[1] * 0.2;
      if (!inFootprint(px, py, -0.01)) return "wall_out";
      return paintFor(roomAt(px, py, p));
    };
    const cuts = { x: [...new Set(P.rooms.flatMap((r) => [r.r[0], r.r[2]]))].sort((a, b) => a - b),
      y: [...new Set(P.rooms.flatMap((r) => [r.r[1], r.r[3]]))].sort((a, b) => a - b) };
    for (const k of ["ext", "int", "post"]) for (const b of P.walls[k] || []) body.wall(b, "wall", route, cuts);
    for (const b of P.walls.parapet || []) body.box(b, "parapet", "wall");
    for (const f of P.floors) {
      if (f.cat === "LIFT" && S.lift) continue;             // the car is the lift's floor
      const [x0, y0, x1, y1] = f.r;
      (f.ground_only ? groundX : body).quadUp(x0, y0, x1, y1, ffl + 0.002, FLOOR[f.cat] || "fl_tile");
    }
    for (const st of S.stairs) {
      if (st.external) continue;
      stairParts(stairsB, st);
    }
    for (const w of P.windows) windowParts(body, w);
    for (const d of P.doors) {
      if (d.upper === "wall" || d.upper === "parapet") { doorFrame(groundX, d, ffl); upperFill(upperX, d, ffl); }
      else doorFrame(body, d, ffl);
    }
    if (p === 0) for (const b of S.balconies || []) balcony(upperX, b, ffl);
    for (const it of P.furniture) { try { furniture(furnB, it, ffl); } catch (e) { console.warn("furniture", it.kind, e); } }
    const sz = S.shell.slab_z;
    for (const r of S.shell.slab) ceilB.box([r[0], r[1], r[2], r[3], sz[0], sz[1]], null, "wall", (f) => (f.bottom ? "ceiling" : "slab"));
    for (const r of P.rooms) {
      if (r.cat === "STAIRS" || r.cat === "LIFT" || r.name.includes("Duct")) continue;
      const [x0, y0, x1, y1] = r.r, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      const fan = hasFan(r), off = fan ? Math.min(0.9, Math.max(x1 - x0, y1 - y0) / 4) : 0;
      const lx = (x1 - x0) >= (y1 - y0) ? cx + off : cx, ly = (x1 - x0) >= (y1 - y0) ? cy : cy + off;
      ceilB.cyl(lx, ly, 0.16, ffl + lv.ceiling - 0.035, ffl + lv.ceiling - 0.005, "lamp", 20);
    }
    const tpl = new THREE.Group();
    const part = (name, bld, pick) => { const g = new THREE.Group(); g.name = name; bld.emit(g, pick); tpl.add(g); return g; };
    part("body", body, pickRoles); part("stairs", stairsB, pickRoles); part("furn", furnB, null); part("ceil", ceilB, new Set(["slab"]));
    const gx = new THREE.Group(); groundX.emit(gx, pickRoles);
    const ux = new THREE.Group(); upperX.emit(ux, pickRoles);
    for (const o of [tpl, gx, ux]) o.traverse((c) => { if (c.isMesh && registry.pick.includes(c)) c.userData.pick = true; });
    return { tpl, gx, ux, used: false, gxUsed: false, uxUsed: false };
  };
  const T = [makeTemplate(S, 0)];
  if (S.upper) T.push(makeTemplate(S.upper, 1));
  const registerClone = (o, L) => o.traverse((c) => {
    c.userData = Object.assign({}, c.userData, { level: L });
    if (c.isMesh) { registry.meshes.push(c); if (c.userData.pick) registry.pick.push(c); }
    if (c.isLineSegments) registry.edges.push(c);
  });
  const useOnce = (t, key, o, L) => {                    // the first use takes the original, later ones a clone
    const c = t[key] ? o.clone(true) : o;
    if (t[key]) registerClone(c, L); else c.traverse((q) => { q.userData = Object.assign({}, q.userData, { level: L }); });
    t[key] = true;
    return c;
  };
  for (let L = 0; L < top; L++) {
    const root = new THREE.Group(); root.userData.level = L;
    if (hasRooms(L)) {
      const t = T[Math.min(planOf(L), T.length - 1)];
      const fl = useOnce(t, "used", t.tpl, L);
      fl.position.y = (L - r0) * H; root.add(fl);
      const extra = L === 0 ? useOnce(t, "gxUsed", t.gx, L) : useOnce(t, "uxUsed", t.ux, L);
      extra.position.y = (L - r0) * H; root.add(extra);
      levelGroups.push({ L, root, ceil: fl.getObjectByName("ceil"), furn: fl.getObjectByName("furn") });
    } else {                                             // stilt: columns, the lift shaft, the stair up
      const sb = new Builder();
      const o = S.shell.outer;
      sb.box([o[0], o[1], o[2], o[3], 0, 0.15], "concrete", "site");
      for (const c of (S.stilt && S.stilt.columns) || []) sb.box([c[0], c[1], c[2], c[3], 0.15, ffl - 0.15], "column", "wall");
      if (!S.lift) for (const lr of S.lifts) sb.box([lr[0], lr[1], lr[2], lr[3], 0.15, ffl - 0.15], "wall_out", "wall");
      for (const r of S.shell.slab) sb.box([r[0], r[1], r[2], r[3], ffl - 0.15, ffl], null, "wall", (f) => (f.bottom ? "ceiling" : "slab"));
      const sg = new THREE.Group(); sb.emit(sg, pickRoles); root.add(sg);
      const fl = T[0].tpl.getObjectByName("stairs").clone(true); registerClone(fl, L);
      fl.position.y = -H; root.add(fl);
      levelGroups.push({ L, root, ceil: null, furn: null });
    }
    scene.add(root);
  }

  // ════ roof: parapet (open where an outside stair arrives) and the stair head-room
  const roofB = new Builder();
  const roofZ = S.shell.roof_z;
  for (const b of S.shell.parapet || perimeter(S.shell.outer, 0.15, roofZ, roofZ + lv.parapet)) roofB.box(b, "parapet", "wall");
  if (S.shell.mumty_walls) {
    for (const b of S.shell.mumty_walls) roofB.box(b, "wall_out", "wall");
    roofB.box(S.shell.mumty_roof, "slab", "wall");
  }
  roofB.emit(G.roof);

  // ════ moving parts follow the floor you look at: door leaves, ceiling fans, labels, Vastu grid
  for (const d of S.doors) if (!(d.kind === "lift" && S.lift)) doorLeaf(d, ffl);
  for (const r of S.rooms) if (hasFan(r)) fanAt(r, ffl + lv.ceiling);
  if (S.upper) {                                         // a duplex's first floor: its own doors and fans
    for (const d of S.upper.doors) if (d.kind !== "lift") doorLeaf(d, ffl, G.dyn2, 1);
    for (const r of S.upper.rooms) if (hasFan(r)) fanAt(r, ffl + lv.ceiling, G.dyn2);
  }

  // ════ site: ground, plinth, entrance steps, road, yard, amenities, cars, trees, gates, outside stair
  buildSite(pickRoles);
  buildDrive();
  buildVastu(ffl);
  buildGhosts();
  buildLift();
  $("btnLift").style.display = lift.ok ? "" : "none";
  if (!lift.ok) state.xray = false;

  // ════ labels
  const flatOf = (name) => (name.includes(" · ") ? name.split(" · ")[0] : "");
  for (const p of S.upper ? [0, 1] : [0]) {
    for (const r of PL(p).rooms) {
      if (r.cat === "LIFT" || r.name.includes("Duct")) continue;
      const [x0, y0, x1, y1] = r.r;
      const nm = r.name.includes(" · ") ? r.name.split(" · ")[1] : r.name;
      addLabel(`${esc(nm)}<small>${r.size} · ${r.ft}</small>`, [(x0 + x1) / 2, (y0 + y1) / 2, ffl + 0.3], "lbl",
        () => ["dollhouse", "top"].includes(state.mode) && !state.explode && hasRooms(activeLevel()) && planOf(activeLevel()) === p,
        { lvl: true, plan: p });
    }
  }
  if (S.apartment) {
    S.apartment.slots.forEach((sl, i) => {
      const lab = S.apartment.labels[i] || "";
      addLabel("", [sl[0] + sl[2] / 2, sl[1] + sl[3] / 2, ffl + 2.0], "lbl flat",
        () => state.mode !== "walk" && hasRooms(activeLevel()) && !state.explode, { lvl: true, flat: lab });
    });
  }
  // the stair and lift run through every floor: in the pulled-apart view they bridge the gaps
  const core = [...S.stairs.filter((q) => !q.external).map((q) => ["stair", q.r]), ...(S.lift ? [["lift", S.lift.room]] : [])];
  for (let L = 0; L + 1 < top && core.length; L++) {
    for (const [kind, rc] of core) {
      const g = new THREE.BoxGeometry(rc[2] - rc[0], GAP, rc[3] - rc[1]);
      g.translate((rc[0] + rc[2]) / 2, zf(L + 1) + L * GAP + GAP / 2, -(rc[1] + rc[3]) / 2);
      const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: kind === "lift" ? 0xFFC857 : 0x00E5FF, transparent: true,
        opacity: 0.14, depthWrite: false, side: THREE.DoubleSide }));
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color: kind === "lift" ? 0xFFC857 : 0x00E5FF,
        transparent: true, opacity: 0.6 }));
      G.core.add(m, e);
    }
  }
  if (core.length && top >= 2) {
    const [, rc] = core[0], mid = Math.max(0, Math.floor((top - 1) / 2));
    addLabel(S.lift ? "🪜 Stair + 🛗 lift<small>the same on every floor</small>" : "🪜 Stair<small>the same on every floor</small>",
      [(rc[0] + rc[2]) / 2, (rc[1] + rc[3]) / 2, zf(mid + 1) + mid * GAP + GAP / 2], "lbl xr",
      () => state.explode && state.mode !== "walk" && !state.xray);
  }
  for (const p of S.upper ? [0, 1] : [0]) {
    for (const d of PL(p).doors) {
      if (d.kind === "lift") continue;
      for (const side of [1, -1]) {
        const to = side === 1 ? d.other : d.room;     // the room side reads where the door goes
        if (!to) continue;
        const px = d.mid[0] + d.n[0] * side * 0.12, py = d.mid[1] + d.n[1] * side * 0.12;
        const shown = to === "Outside" ? "Outside" : (to.includes(" · ") ? to.split(" · ")[1] : to);
        addLabel(`→ ${esc(shown)}`, [px, py, ffl + 2.32], "sign",
          () => state.mode === "walk" && hasRooms(walker.level) && planOf(walker.level) === p && !lift.rider &&
            !(walker.level > 0 && d.upper && d.upper !== "balcony"),
          { door: d, side, lvl: true, to, flatTo: flatOf(to), plan: p });
      }
    }
  }
  for (const g of S.mandala) addLabel(`${g.word}`, [(g.r[0] + g.r[2]) / 2, (g.r[1] + g.r[3]) / 2, ffl + 0.05], "lbl zone",
    () => state.vastu && state.mode !== "outside" && hasRooms(activeLevel()), { lvl: true });
  const nx = S.north[0], ny = S.north[1], ax = 1.0, ay = S.plot.h - 1.0;
  addLabel("N", [ax + nx * 1.25, ay + ny * 1.25, 0.15], "north", () => state.mode !== "walk");
  for (const l of S.levels_list || []) {
    addLabel(`${esc(l.label)}`, [S.shell.outer[0] - 0.5, S.shell.outer[3] + 0.3, l.z + (l.kind === "terrace" ? 0.6 : 1.3)], "lbl floor",
      () => state.mode !== "walk" && (state.explode || state.xray), { floorLabel: l.i });
  }

  navs = { g0: decodeNav(S.nav), typ: S.nav_typ ? decodeNav(S.nav_typ) : null, top: S.nav1 ? decodeNav(S.nav1) : null };
  nav = navs.g0;
  segCells = new Int8Array(nav.nx * nav.ny).fill(-1);       // which half of a flight each stair cell is on
  for (const st of S.stairs) {
    const w = st.wr || st.r;
    const i0 = Math.max(0, Math.floor((w[0] - nav.x0) / nav.cell)), i1 = Math.min(nav.nx - 1, Math.floor((w[2] - nav.x0) / nav.cell));
    const j0 = Math.max(0, Math.floor((w[1] - nav.y0) / nav.cell)), j1 = Math.min(nav.ny - 1, Math.floor((w[3] - nav.y0) / nav.cell));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
      const b = stairBoxAt(nav.x0 + (i + 0.5) * nav.cell, nav.y0 + (j + 0.5) * nav.cell);
      if (b) segCells[j * nav.nx + i] = b.seg;
    }
  }
  walker.level = clamp(walker.level, 0, top);
  state.view = clamp(state.view || r0, 0, top);
  vis = buildVis();
  drawMinimapBase();
  buildFloorBar();
  applyTheme();
  applySun();
  setVastu(state.vastu);
  setFurniture(state.furn);
  applyLevelView();
  $("startMeta").textContent = `${S.meta.title} · ${S.meta.cents} cents · carpet ${S.meta.carpet} m² · Vastu ${S.meta.vastu}/100`;
  drawLangs();
  $("msg").style.display = "none";
  poke(1500);
}

function stairParts(bld, st) {
  for (const b of st.boxes) bld.box(b.slice(0, 6), "stair", "stair");
  for (const rail of st.rails) {
    for (let i = 0; i + 1 < rail.length; i++) rod(bld, rail[i], rail[i + 1], 0.022, "rail");
    for (const p of rail) rod(bld, [p[0], p[1], p[2] - 0.9], p, 0.014, "rail");
  }
}
function doorSpan(d) {
  const ca = d.closed * DEG, ex = d.hinge[0] + Math.round(Math.cos(ca)) * d.w, ey = d.hinge[1] + Math.round(Math.sin(ca)) * d.w;
  return d.horiz ? [Math.min(d.hinge[0], ex), Math.max(d.hinge[0], ex), d.hinge[1]] : [Math.min(d.hinge[1], ey), Math.max(d.hinge[1], ey), d.hinge[0]];
}
function upperFill(bld, d, ffl) {        // above the ground the building's entrance is closed: wall + window, or parapet
  const [a, b, line] = doorSpan(d), t = d.t / 2;
  const box = (z0, z1, role, edge) => bld.box(d.horiz ? [a, line - t, b, line + t, z0, z1] : [line - t, a, line + t, b, z0, z1], role, edge);
  if (d.upper === "parapet") { box(ffl, ffl + S.levels.parapet, "parapet", "wall"); return; }
  box(ffl, ffl + 0.9, "wall_out", "wall"); box(ffl + 2.1, ffl + S.levels.ceiling, "wall_out", "wall");
  box(ffl + 0.9, ffl + 2.1, "glass", "win");
}
function balcony(bld, b, ffl) {          // upper-floor balcony where the ground floor has an outside door
  const [x0, y0, x1, y1] = b;
  bld.box([x0, y0, x1, y1, ffl - 0.15, ffl], "slab", "wall", (f) => (f.top ? "fl_tile" : f.bottom ? "ceiling" : "slab"));
  const z = ffl + 1.0, c = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  for (let i = 0; i < 4; i++) {
    const p = c[i], q = c[(i + 1) % 4];
    if (inFootprint((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, 0.05)) continue;     // the house wall side
    rod(bld, [p[0], p[1], z], [q[0], q[1], z], 0.025, "rail");
    const L = Math.hypot(q[0] - p[0], q[1] - p[1]), n = Math.max(2, Math.round(L / 0.15));
    for (let k = 0; k <= n; k++) { const f = k / n, x = p[0] + (q[0] - p[0]) * f, y = p[1] + (q[1] - p[1]) * f; rod(bld, [x, y, ffl], [x, y, z], 0.01, "rail"); }
  }
}

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function perimeter(o, t, z0, z1) {   // four wall boxes on the inside of rect o
  return [[o[0], o[1], o[2], o[1] + t, z0, z1], [o[0], o[3] - t, o[2], o[3], z0, z1],
    [o[0], o[1] + t, o[0] + t, o[3] - t, z0, z1], [o[2] - t, o[1] + t, o[2], o[3] - t, z0, z1]];
}
function rod(bld, a, b, r, role) {          // a, b = plan points [x, y, z]
  const A = new THREE.Vector3(a[0], a[2], -a[1]), B = new THREE.Vector3(b[0], b[2], -b[1]);
  const len = A.distanceTo(B); if (len < 1e-3) return;
  const g = new THREE.CylinderGeometry(r, r, len, 8);
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), B.clone().sub(A).normalize());
  g.applyQuaternion(q); g.translate((A.x + B.x) / 2, (A.y + B.y) / 2, (A.z + B.z) / 2);
  bld.B(role).geom(g);
}
function hasFan(r) { const a = (r.r[2] - r.r[0]) * (r.r[3] - r.r[1]); return ["BEDROOM", "LIVING", "DINING"].includes(r.cat) && a >= 7.5; }

function windowParts(bld, w) {
  const horiz = Math.abs(w.a[1] - w.b[1]) < 1e-6;
  const a = horiz ? w.a[0] : w.a[1], b = horiz ? w.b[0] : w.b[1], line = horiz ? w.a[1] : w.a[0];
  const out = horiz ? w.out[1] : w.out[0];
  const box = (u0, u1, v0, v1, z0, z1, role, edge) => {   // u along the wall, v across (+ = outside)
    const p = horiz ? [u0, line + v0 * out, u1, line + v1 * out] : [line + v0 * out, u0, line + v1 * out, u1];
    bld.box([Math.min(p[0], p[2]), Math.min(p[1], p[3]), Math.max(p[0], p[2]), Math.max(p[1], p[3]), z0, z1], role, edge);
  };
  const fw = 0.05, ft = 0.07;
  box(a, a + fw, -ft / 2, ft / 2, w.z0, w.z1, "frame", "win");
  box(b - fw, b, -ft / 2, ft / 2, w.z0, w.z1, "frame", "win");
  box(a, b, -ft / 2, ft / 2, w.z0, w.z0 + fw, "frame", "win");
  box(a, b, -ft / 2, ft / 2, w.z1 - fw, w.z1, "frame", "win");
  if (w.vent) {
    box(a + fw, b - fw, -0.005, 0.005, w.z0 + fw, w.z1 - fw, "frost", null);
  } else {
    const mid = (a + b) / 2;
    box(mid - 0.025, mid + 0.025, -ft / 2, ft / 2, w.z0, w.z1, "frame", "win");
    box(a + fw, mid + 0.03, 0.012, 0.022, w.z0 + fw, w.z1 - fw, "glass", null);
    box(mid - 0.03, b - fw, -0.022, -0.012, w.z0 + fw, w.z1 - fw, "glass", null);
    // external sill and the sunshade (chajja) every Indian window wears
    box(a - 0.05, b + 0.05, w.t / 2 - 0.02, w.t / 2 + 0.06, w.z0 - 0.05, w.z0, "slab", "wall");
    box(a - 0.3, b + 0.3, w.t / 2, w.t / 2 + 0.45, w.z1 + 0.1, w.z1 + 0.18, "slab", "wall");
  }
}

function doorFrame(bld, d, ffl) {
  if (d.kind === "lift") return;
  const ca = d.closed * DEG, ux = Math.round(Math.cos(ca)), uy = Math.round(Math.sin(ca));
  const a = [d.hinge[0], d.hinge[1]], b = [d.hinge[0] + ux * d.w, d.hinge[1] + uy * d.w];
  const horiz = d.horiz, t = d.t + 0.02, jw = 0.06;
  const jamb = (p) => horiz ? [p[0] - jw / 2, p[1] - t / 2, p[0] + jw / 2, p[1] + t / 2] : [p[0] - t / 2, p[1] - jw / 2, p[0] + t / 2, p[1] + jw / 2];
  for (const p of [a, b]) { const r = jamb(p); bld.box([r[0], r[1], r[2], r[3], ffl, ffl + d.h], "dframe", "door"); }
  const lo = horiz ? Math.min(a[0], b[0]) : Math.min(a[1], b[1]), hi = horiz ? Math.max(a[0], b[0]) : Math.max(a[1], b[1]);
  const line = horiz ? a[1] : a[0];
  const head = horiz ? [lo - jw / 2, line - t / 2, hi + jw / 2, line + t / 2] : [line - t / 2, lo - jw / 2, line + t / 2, hi + jw / 2];
  bld.box([head[0], head[1], head[2], head[3], ffl + d.h - 0.06, ffl + d.h], "dframe", "door");
}

function doorLeaf(d, ffl, grp = G.dyn, plan = 0) {
  const role = { main: "door_main", flatdoor: "door_main", bath: "door_bath", lift: "door_steel", service: "door_steel" }[d.kind] || "door_int";
  const pivot = new THREE.Group();
  pivot.position.set(d.hinge[0], 0, -d.hinge[1]);
  pivot.rotation.y = d.closed * DEG;
  const w = d.w - 0.06, h = d.h - 0.03, th = d.kind === "main" ? 0.045 : 0.035;
  const geo = new THREE.BoxGeometry(w, h, th); geo.translate(0.03 + w / 2, ffl + 0.015 + h / 2, 0);
  const leaf = new THREE.Mesh(geo, mat(role)); leaf.userData.role = role; leaf.castShadow = true; leaf.receiveShadow = true;
  pivot.add(leaf); registry.meshes.push(leaf);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), lineMat("door")); edges.userData.edge = true;
  pivot.add(edges); registry.edges.push(edges);
  if (d.kind !== "lift") {
    for (const s of [1, -1]) {
      const hg = new THREE.BoxGeometry(0.12, 0.03, 0.04); hg.translate(w - 0.06, ffl + 1.0, s * (th / 2 + 0.02));
      const hm = new THREE.Mesh(hg, mat("handle")); hm.userData.role = "handle"; pivot.add(hm); registry.meshes.push(hm);
    }
  }
  grp.add(pivot);
  let delta = ((d.open - d.closed + 540) % 360) - 180;
  registry.doors.push({ d, pivot, cur: 0, target: 0, delta: delta * DEG, still: d.kind === "lift",
    groundOnly: d.upper === "wall" || d.upper === "parapet", plan });
}

// ───────────────────────────── furniture (the drawing's own pieces, given height)
function furniture(bld, it, ffl) {
  const k = it.kind;
  if (k === "floor_trap") return;
  if (k === "planter") {
    bld.cyl(it.x, it.y, it.r * 0.8, ffl, ffl + 0.38, "pot", 14, it.r);
    const g = new THREE.IcosahedronGeometry(0.32, 1); g.scale(1, 1.25, 1); g.translate(it.x, ffl + 0.78, -it.y); bld.B("plant").geom(g);
    return;
  }
  const o = it.o, u = it.u, v = it.v;
  const P = (uu, vv) => [o[0] + u[0] * uu + v[0] * vv, o[1] + u[1] * uu + v[1] * vv];
  const box = (u0, u1, v0, v1, h0, h1, role, edge = "furn") => {
    const a = P(u0, v0), b = P(u1, v1);
    bld.box([Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1]), ffl + h0, ffl + h1], role, edge);
  };
  const cyl = (uc, vc, r, h0, h1, role, seg = 16, rTop = null) => { const p = P(uc, vc); bld.cyl(p[0], p[1], r, ffl + h0, ffl + h1, role, seg, rTop); };
  const ell = (uc, vc, ru, rv, h0, h1, role) => {
    const p = P(uc, vc), g = new THREE.CylinderGeometry(1, 1, h1 - h0, 20);
    const alongX = Math.abs(u[0]) > 0.5;
    g.scale(alongX ? ru : rv, 1, alongX ? rv : ru); g.translate(p[0], ffl + (h0 + h1) / 2, -p[1]); bld.B(role).geom(g);
  };
  const W = it.W ?? 0.5, D = it.D ?? 0.5;
  switch (k) {
    case "bed": {
      box(0, W, 0, D, 0, 0.3, "wood");
      box(0.03, W - 0.03, 0.08, D - 0.03, 0.3, 0.5, "mattress");
      box(0, W, 0, 0.07, 0, 1.05, "wood_dark");
      if (it.double) { box(0.12, W / 2 - 0.06, 0.12, 0.46, 0.5, 0.62, "pillow"); box(W / 2 + 0.06, W - 0.12, 0.12, 0.46, 0.5, 0.62, "pillow"); }
      else box(0.14, W - 0.14, 0.12, 0.44, 0.5, 0.62, "pillow");
      box(0.02, W - 0.02, 0.62, D - 0.02, 0.5, 0.55, "blanket");
      break;
    }
    case "side_table":
      box(0, W, 0, D, 0, 0.5, "wood");
      cyl(W / 2, D / 2, 0.05, 0.5, 0.72, "brass", 10); cyl(W / 2, D / 2, 0.13, 0.72, 0.9, "lamp", 16, 0.09);
      break;
    case "wardrobe": {
      box(0, W, 0, D, 0, 2.1, "laminate");
      const n = Math.max(1, Math.round(W / 0.45));
      for (let i = 0; i < n; i++) { const uc = (i + 0.5) * W / n; box(uc - 0.01, uc + 0.01, D, D + 0.025, 0.95, 1.25, "steel", null); }
      for (let i = 1; i < n; i++) box(i * W / n - 0.004, i * W / n + 0.004, D, D + 0.005, 0.02, 2.08, "wood_dark", null);
      break;
    }
    case "sofa": {
      const arm = 0.18, seats = it.seats || 1;
      box(0, W, 0, D, 0.08, 0.42, "fabric");
      box(0, W, 0, 0.2, 0.42, 0.85, "fabric");
      box(0, arm, 0.2, D, 0.42, 0.62, "fabric"); box(W - arm, W, 0.2, D, 0.42, 0.62, "fabric");
      const cw = (W - 2 * arm) / seats;
      for (let i = 0; i < seats; i++) box(arm + i * cw + 0.01, arm + (i + 1) * cw - 0.01, 0.2, D - 0.02, 0.42, 0.52, "fabric2", null);
      box(0.05, 0.1, 0.05, 0.1, 0, 0.08, "wood_dark", null); box(W - 0.1, W - 0.05, D - 0.1, D - 0.05, 0, 0.08, "wood_dark", null);
      break;
    }
    case "coffee_table":
      box(0, W, 0, D, 0.38, 0.42, "wood");
      for (const [a, b] of [[0.04, 0.04], [W - 0.08, 0.04], [0.04, D - 0.08], [W - 0.08, D - 0.08]]) box(a, a + 0.04, b, b + 0.04, 0, 0.38, "wood_dark", null);
      break;
    case "tv_unit": {
      box(0, W, 0, D, 0, 0.45, "wood_dark");
      const tw = Math.min(1.4, W * 0.85), th = tw * 9 / 16;
      box((W - tw) / 2, (W + tw) / 2, 0.02, 0.07, 0.95, 0.95 + th, "screen");
      break;
    }
    case "fridge":
      box(0.02, W - 0.02, 0.0, D - 0.04, 0, 1.75, "steel");
      box(W - 0.12, W - 0.09, D - 0.04, D - 0.0, 0.9, 1.4, "black", null);
      break;
    case "counter":
      box(it.a, it.b, 0, 0.58, 0, 0.84, "cabinet");
      box(it.a, it.b, 0, 0.6, 0.84, 0.88, "counter");
      box(it.a, it.b, 0, 0.02, 0.88, 1.5, "in_bath", null);       // dado tiles above the counter
      break;
    case "sink": {
      const sw = it.double ? 0.8 : 0.45;
      box(it.a, it.a + sw, 0.09, 0.5, 0.86, 0.885, "steel", null);
      cyl(it.a + sw / 2, 0.06, 0.018, 0.88, 1.12, "steel", 8);
      break;
    }
    case "hob": {
      box(it.a, it.a + 0.6, 0.06, 0.54, 0.88, 0.9, "black", null);
      for (const [cu, cv] of [[0.15, 0.18], [0.45, 0.18], [0.15, 0.42], [0.45, 0.42]]) cyl(it.a + cu, cv, 0.07, 0.9, 0.915, "steel", 14);
      box(it.a - 0.05, it.a + 0.65, 0, 0.5, 1.6, 1.85, "steel");                      // chimney hood
      box(it.a + 0.18, it.a + 0.42, 0, 0.24, 1.85, 2.3, "steel", null);
      break;
    }
    case "wc":
      box(0.04, W - 0.04, 0, 0.18, 0.38, 0.75, "white");
      ell(W / 2, 0.44, 0.18, 0.26, 0, 0.4, "white");
      ell(W / 2, 0.44, 0.19, 0.27, 0.4, 0.43, "white");
      break;
    case "basin":
      box(0, W, 0, D, 0.74, 0.86, "white");
      box(W / 2 - 0.08, W / 2 + 0.08, 0.04, 0.2, 0, 0.74, "white", null);
      cyl(W / 2, 0.05, 0.015, 0.86, 1.02, "steel", 8);
      box(0.03, W - 0.03, -0.01, 0.0, 1.15, 1.75, "mirror", null);
      break;
    case "shower":
      box(0, W, 0, D, 0, 0.02, "fl_wet", null);
      box(0, W * 0.65, D - 0.01, D, 0.02, 2.0, "glass", null);
      cyl(W / 2, 0.05, 0.015, 1.0, 2.05, "steel", 8);
      cyl(W / 2, 0.2, 0.09, 2.0, 2.03, "steel", 16);
      break;
    case "washer":
      box(0.02, W - 0.02, 0, D, 0, 0.85, "white");
      box(W / 2 - 0.16, W / 2 + 0.16, D, D + 0.01, 0.25, 0.57, "black", null);
      break;
    case "utility_sink":
      box(0, W, 0, D, 0, 0.75, "concrete");
      box(0.06, W - 0.06, 0.08, D - 0.05, 0.75, 0.77, "steel", null);
      cyl(W / 2, 0.04, 0.015, 0.75, 1.0, "steel", 8);
      break;
    case "desk": {
      box(0, W, 0, D, 0.72, 0.75, "wood");
      box(0, 0.03, 0, D, 0, 0.72, "wood"); box(W - 0.03, W, 0, D, 0, 0.72, "wood");
      const c = W / 2;
      box(c - 0.22, c + 0.22, D + 0.08, D + 0.5, 0.42, 0.47, "fabric");
      box(c - 0.22, c + 0.22, D + 0.46, D + 0.5, 0.47, 0.9, "fabric");
      cyl(c, D + 0.29, 0.03, 0.05, 0.42, "steel", 8);
      break;
    }
    case "bookshelf":
    case "shelves": {
      const Ht = k === "bookshelf" ? 1.8 : 2.0, r = rng(Math.round(o[0] * 100 + o[1] * 7));
      box(0, W, 0, 0.02, 0, Ht, "wood"); box(0, 0.02, 0, D, 0, Ht, "wood"); box(W - 0.02, W, 0, D, 0, Ht, "wood");
      for (let s = 0; s <= 4; s++) {
        const h = s * Ht / 4; box(0, W, 0, D, h, h + 0.02, "wood", null);
        if (s < 4) { let uu = 0.04; while (uu < W - 0.08) { const bw = 0.03 + r() * 0.05, bh = 0.18 + r() * 0.16; box(uu, uu + bw, 0.04, D - 0.05, h + 0.02, h + 0.02 + Math.min(bh, Ht / 4 - 0.05), r() > 0.5 ? "books1" : "books2", null); uu += bw + 0.01 + (r() > 0.85 ? 0.08 : 0); } }
      }
      break;
    }
    case "pooja": {
      box(0, W, 0, D, 0, 0.75, "wood_dark");
      box(0, W, 0, 0.05, 0.75, 1.65, "wood_dark");
      box(0.03, W - 0.03, 0.05, D * 0.6, 0.75, 0.78, "brass", null);
      const p = P(W / 2, D * 0.3);
      const cone = new THREE.ConeGeometry(Math.min(0.22, W / 3), 0.35, 4); cone.rotateY(Math.PI / 4); cone.translate(p[0], ffl + 1.83, -p[1]); bld.B("brass").geom(cone);
      box(0.05, W - 0.05, 0.05, D * 0.6, 1.62, 1.66, "brass", null);
      cyl(W / 2, D * 0.78, 0.035, 0.75, 0.79, "brass", 12);
      const f = P(W / 2, D * 0.78); const fl = new THREE.SphereGeometry(0.018, 8, 6); fl.scale(1, 1.8, 1); fl.translate(f[0], ffl + 0.82, -f[1]); bld.B("flame").geom(fl);
      break;
    }
    case "shoe_rack":
      box(0, W, 0, D, 0, 0.9, "wood");
      break;
    case "dining": {
      const L = W, Tw = D, seats = it.seats || 4, n = Math.max(1, Math.floor(seats / 2));
      box(0, L, 0, Tw, 0.72, 0.76, "wood");
      for (const [a, b] of [[0.06, 0.06], [L - 0.1, 0.06], [0.06, Tw - 0.1], [L - 0.1, Tw - 0.1]]) box(a, a + 0.05, b, b + 0.05, 0, 0.72, "wood_dark", null);
      for (let i = 0; i < n; i++) {
        const uc = (i + 0.5) * L / n;
        box(uc - 0.21, uc + 0.21, -0.52, -0.1, 0.43, 0.47, "fabric2"); box(uc - 0.21, uc + 0.21, -0.52, -0.48, 0.47, 0.92, "wood_dark");
        box(uc - 0.21, uc + 0.21, Tw + 0.1, Tw + 0.52, 0.43, 0.47, "fabric2"); box(uc - 0.21, uc + 0.21, Tw + 0.48, Tw + 0.52, 0.47, 0.92, "wood_dark");
        for (const vv of [-0.5, -0.14, Tw + 0.12, Tw + 0.48]) for (const du of [-0.19, 0.17]) box(uc + du, uc + du + 0.03, vv, vv + 0.03, 0, 0.43, "wood_dark", null);
      }
      break;
    }
    default:
      box(0, W, 0, D, 0, 0.6, "wood");
  }
}

function fanAt(r, ceilZ, grp = G.dyn) {
  const [x0, y0, x1, y1] = r.r, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const g = new THREE.Group(); g.position.set(cx, ceilZ - 0.4, -cy);
  const add = (geo, role) => { const m = new THREE.Mesh(geo, mat(role)); m.userData.role = role; m.castShadow = true; g.add(m); registry.meshes.push(m); return m; };
  add(new THREE.CylinderGeometry(0.012, 0.012, 0.38, 6).translate(0, 0.21, 0), "fan");
  add(new THREE.CylinderGeometry(0.11, 0.09, 0.09, 18), "fan");
  for (let i = 0; i < 3; i++) {
    const b = new THREE.BoxGeometry(0.55, 0.012, 0.11); b.translate(0.36, -0.01, 0); b.rotateY(i * 2 * Math.PI / 3); add(b, "fan");
  }
  grp.add(g); registry.fans.push(g);
}

// ───────────────────────────── site
function buildSite(pickRoles) {
  const bld = new Builder(), W = S.plot.w, H = S.plot.h, rd = S.road;
  for (const b of S.walls.cmp || []) {
    bld.box(b, "compound", "site");
    bld.box([b[0] - 0.04, b[1] - 0.04, b[2] + 0.04, b[3] + 0.04, b[5], b[5] + 0.06], "pillar", null);
  }
  for (const b of S.walls.pillar || []) bld.box(b, "pillar", "site");
  if (!S.stilt) { const o = S.shell.outer; bld.box([o[0] - 0.08, o[1] - 0.08, o[2] + 0.08, o[3] + 0.08, 0, S.levels.ffl], "plinth", "wall"); }
  for (const b of S.steps) bld.box(b, "stair", "stair");
  for (const st of S.stairs) if (st.external) stairParts(bld, st);
  bld.quadUp(-60, -60, W + 60, H + 60, -0.02, "land");
  bld.quadUp(0, 0, W, H, 0.0, "yard");
  bld.quadUp(-60, H, W + 60, rd.y0, 0.003, "verge");
  bld.quadUp(-60, rd.y0, W + 60, rd.y1, 0.004, "road");
  const mid = (rd.y0 + rd.y1) / 2;
  for (let x = -60; x < W + 60; x += 3.0) bld.quadUp(x, mid - 0.06, x + 1.5, mid + 0.06, 0.006, "roadline");
  for (const s of S.site) {
    const [x0, y0, x1, y1] = s.r;
    if (s.cat === "PARKING" || s.cat === "BIKE_PARKING") {
      bld.quadUp(x0, y0, x1, y1, 0.006, "paver");
      for (const [a, b, c, d] of [[x0, y0, x1, y0 + 0.06], [x0, y1 - 0.06, x1, y1], [x0, y0, x0 + 0.06, y1], [x1 - 0.06, y0, x1, y1]]) bld.quadUp(a, b, c, d, 0.008, "roadline");
    } else if (s.cat === "GARDEN") {
      bld.quadUp(x0, y0, x1, y1, 0.007, "grass");
    } else if (s.cat === "POOL") {
      const c = 0.3;
      for (const b of [[x0, y0, x1, y0 + c], [x0, y1 - c, x1, y1], [x0, y0 + c, x0 + c, y1 - c], [x1 - c, y0 + c, x1, y1 - c]]) bld.box([...b, 0, 0.1], "concrete", "site");
      bld.quadUp(x0 + c, y0 + c, x1 - c, y1 - c, 0.05, "water");
    } else if (s.cat === "WATER_TANK") {
      bld.box([x0, y0, x1, y1, 0, 0.06], "concrete", "site");
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      bld.box([cx - 0.3, cy - 0.3, cx + 0.3, cy + 0.3, 0.06, 0.09], "manhole", "site");
    } else if (s.cat === "GENERATOR") {
      bld.box([x0 + 0.05, y0 + 0.05, x1 - 0.05, y1 - 0.05, 0, 0.1], "concrete", "site");
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, gw = Math.min(1.3, x1 - x0 - 0.3) / 2, gd = Math.min(0.75, y1 - y0 - 0.3) / 2;
      bld.box([cx - gw, cy - gd, cx + gw, cy + gd, 0.1, 1.15], "genset", "site");
    } else if (s.cat === "SECURITY") {
      bld.box([x0, y0, x1, y1, 0, 2.4], "cabin", "site");
      bld.box([x0 - 0.15, y0 - 0.15, x1 + 0.15, y1 + 0.15, 2.4, 2.52], "slab", "site");
      bld.box([x0 + 0.2, y1, x1 - 0.2, y1 + 0.02, 1.0, 1.8], "glass", null);
      const cx = (x0 + x1) / 2; bld.box([cx - 0.4, y0 - 0.02, cx + 0.4, y0, 0, 2.0], "door_int", "door");
    }
  }
  const stiltBase = S.stilt ? 0.15 : 0;
  for (const c of S.cars) if (!c.drive) car(bld, c, c.stilt ? stiltBase : 0.0);
  for (const b of S.bikes) bike(bld, b);
  for (const t of S.trees) tree(bld, t);
  // walkway pavers along the guided path from the gate to the steps
  const door = (S.tour || []).find((s) => s.kind === "door");
  if (door) {
    const pts = [(S.tour[0] && S.tour[0].path[0]) || door.path[0], ...door.path];
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = pts[i], b = pts[i + 1];
      if (a[1] > H && b[1] > H) continue;
      const A = [a[0], Math.min(a[1], H)], B = [b[0], Math.min(b[1], H)];
      const dx = B[0] - A[0], dy = B[1] - A[1], L = Math.hypot(dx, dy); if (L < 0.05) continue;
      const nx = -dy / L * 0.6, ny = dx / L * 0.6, ex = dx / L * 0.3, ey = dy / L * 0.3;
      const p = [[A[0] - ex + nx, A[1] - ey + ny], [B[0] + ex + nx, B[1] + ey + ny], [B[0] + ex - nx, B[1] + ey - ny], [A[0] - ex - nx, A[1] - ey - ny]];
      const nrm = (dx * ny - dy * nx) > 0 ? [0, 1, 0] : [0, 1, 0];
      const q = p.map((pp) => [pp[0], 0.009, -pp[1]]);
      // keep CCW seen from above
      const cross = (q[1][0] - q[0][0]) * (q[2][2] - q[0][2]) - (q[1][2] - q[0][2]) * (q[2][0] - q[0][0]);
      if (cross < 0) bld.B("paver").quad(q[0], q[1], q[2], q[3], nrm); else bld.B("paver").quad(q[3], q[2], q[1], q[0], nrm);
    }
  }
  // north arrow on the ground
  const ax = 1.0, ay = H - 1.0, nx = S.north[0], ny = S.north[1], px = -ny, py = nx;
  const tip = [ax + nx * 0.9, ay + ny * 0.9], l = [ax - nx * 0.5 + px * 0.35, ay - ny * 0.5 + py * 0.35], r = [ax - nx * 0.5 - px * 0.35, ay - ny * 0.5 - py * 0.35];
  const tri = [[tip[0], 0.02, -tip[1]], [l[0], 0.02, -l[1]], [r[0], 0.02, -r[1]]];
  const cr = (tri[1][0] - tri[0][0]) * (tri[2][2] - tri[0][2]) - (tri[1][2] - tri[0][2]) * (tri[2][0] - tri[0][0]);
  const bt = bld.B("arrow"); const order = cr < 0 ? [0, 1, 2] : [0, 2, 1];
  for (const i of order) { bt.p.push(...tri[i]); bt.n.push(0, 1, 0); bt.t.push(tri[i][0], -tri[i][2]); }
  bld.emit(G.site, pickRoles);
  for (const g of S.gates) gate(g);
}

function car(bld, c, base) {
  const along = c.along === "x", L = c.L, Wc = c.W, [cx, cy] = c.c;
  const P = (uu, vv) => (along ? [cx + uu, cy + vv] : [cx + vv, cy + uu]);
  const box = (u0, u1, v0, v1, h0, h1, role, edge = "site") => {
    const a = P(u0, v0), b = P(u1, v1);
    bld.box([Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1]), base + h0, base + h1], role, edge);
  };
  const col = "car" + (c.i % 4);
  box(-L / 2, L / 2, -Wc / 2, Wc / 2, 0.3, 0.8, col);
  box(-L * 0.3, L * 0.2, -Wc * 0.45, Wc * 0.45, 0.8, 1.22, "car_glass");
  box(-L * 0.28, L * 0.18, -Wc * 0.43, Wc * 0.43, 1.22, 1.3, col);
  box(L / 2 - 0.02, L / 2, -Wc * 0.42, -Wc * 0.25, 0.6, 0.7, "lamp", null); box(L / 2 - 0.02, L / 2, Wc * 0.25, Wc * 0.42, 0.6, 0.7, "lamp", null);
  for (const su of [-0.33, 0.33]) for (const sv of [-1, 1]) {
    const p = P(su * L, sv * (Wc / 2 - 0.1)), g = new THREE.CylinderGeometry(0.31, 0.31, 0.2, 16);
    if (along) g.rotateX(Math.PI / 2); else g.rotateZ(Math.PI / 2);
    g.translate(p[0], base + 0.31, -p[1]); bld.B("tyre").geom(g);
  }
}
function bike(bld, b) {
  const along = b.along === "x", [cx, cy] = b.c, L = Math.min(b.L, 1.8);
  const P = (uu, vv) => (along ? [cx + uu, cy + vv] : [cx + vv, cy + uu]);
  const box = (u0, u1, v0, v1, h0, h1, role) => { const a = P(u0, v0), c = P(u1, v1); bld.box([Math.min(a[0], c[0]), Math.min(a[1], c[1]), Math.max(a[0], c[0]), Math.max(a[1], c[1]), h0, h1], role, "site"); };
  box(-L * 0.3, L * 0.3, -0.12, 0.12, 0.35, 0.72, "car" + ((b.i + 2) % 4));
  box(-L * 0.25, L * 0.05, -0.13, 0.13, 0.72, 0.8, "black");
  box(L * 0.24, L * 0.28, -0.36, 0.36, 0.95, 0.99, "steel");
  for (const su of [-0.36, 0.36]) {
    const p = P(su * L, 0), g = new THREE.CylinderGeometry(0.27, 0.27, 0.08, 14);
    if (along) g.rotateX(Math.PI / 2); else g.rotateZ(Math.PI / 2);
    g.translate(p[0], 0.27, -p[1]); bld.B("tyre").geom(g);
  }
}
function tree(bld, t) {
  const [x, y, s] = t, r = rng(Math.round(x * 131 + y * 17));
  const h = 1.5 + s * 0.9;
  bld.cyl(x, y, 0.05 + s * 0.04, 0, h, "trunk", 8, 0.035 + s * 0.025);
  for (let k = 0; k < 3; k++) {
    const g = new THREE.IcosahedronGeometry(s * (0.4 + r() * 0.18), 1);
    g.translate(x + (r() - 0.5) * s * 0.6, h + s * (0.2 + r() * 0.5), -(y + (r() - 0.5) * s * 0.6));
    bld.B(k % 2 ? "leaf2" : "leaf").geom(g);
  }
}
// a gate leaf along local +x from 0 to len (MS frame, rails, pickets; a wicket door drawn in it)
function gateLeaf(len, h, wk = null) {
  const bld = new Builder(), L = len - 0.04;
  const bar = (x0, x1, z0, z1) => bld.box([x0, -0.02, x1, 0.02, z0, z1], "gate", "gate");
  bar(0.02, L, 0.08, 0.13); bar(0.02, L, h - 0.08, h - 0.03); bar(0.02, L, h / 2, h / 2 + 0.04);
  bar(0.02, 0.07, 0.08, h - 0.03); bar(L - 0.05, L, 0.08, h - 0.03);
  for (let x = 0.2; x < L - 0.08; x += 0.13) bar(x, x + 0.022, 0.13, h - 0.08);
  if (wk) { const [a, b] = wk; bld.box([a, 0.02, b, 0.04, 0.05, 1.95 < h ? 1.95 : h - 0.05], "frame", "gate"); }
  return bld;
}
function gate(g) {
  if (Array.isArray(g)) g = { x0: g[0], x1: g[1], h: 1.5, leaves: [] };
  const H = S.plot.h, width = g.x1 - g.x0, gh = g.h || 1.5;
  let leaves = (g.leaves || []).filter((lf) => lf.type !== "wicket");
  if (!leaves.length) leaves = width > 2.4 ? [{ type: "swing", hinge: g.x0, len: width / 2, dir: 1 }, { type: "swing", hinge: g.x1, len: width / 2, dir: -1 }]
    : [{ type: "swing", hinge: g.x0, len: width, dir: 1 }];
  const mid = g.wicket ? [(g.wicket[0] + g.wicket[1]) / 2, H] : [(g.x0 + g.x1) / 2, H];
  const rec = { cur: 0, target: 0, mid, gmid: [(g.x0 + g.x1) / 2, H], parts: [], g };
  for (const lf of leaves) {
    const h = lf.wicket ? Math.min(gh, 1.5) : gh;
    if (lf.type === "swing") {                       // hinge on the pillar, opens inward (never over the road)
      const pivot = new THREE.Group(); pivot.position.set(lf.hinge, 0, -H);
      const c = lf.dir > 0 ? 0 : Math.PI; pivot.rotation.y = c;
      gateLeaf(lf.len, h).emit(pivot); G.site.add(pivot);
      rec.parts.push((e) => { pivot.rotation.y = c - lf.dir * (Math.PI / 2) * e; });
    } else if (lf.type === "slide") {                // rolls on a ground track just inside the wall
      const grp = new THREE.Group(), z = -H + 0.16 + (rec.parts.length % 2) * 0.07;
      grp.position.set(lf.x0, 0, z);
      const len = lf.x1 - lf.x0, inner = g.wicket && g.vehicle && g.wicket[0] >= g.vehicle[0] - 1e-6 && g.wicket[1] <= g.vehicle[1] + 1e-6;
      const wk = inner && rec.parts.length === 0 ? [Math.max(0.1, g.wicket[0] - lf.x0), Math.min(len - 0.1, g.wicket[1] - lf.x0)] : null;
      gateLeaf(len, h, wk && wk[1] > wk[0] + 0.5 ? wk : null).emit(grp);
      G.site.add(grp);
      rec.parts.push((e) => { grp.position.x = lf.x0 + lf.to * lf.travel * e; });
    } else if (lf.type === "fold") {                 // bi-fold: two panels folding back against the pillar
      const pa = new THREE.Group(); pa.position.set(lf.hinge, 0, -H);
      const c = lf.dir > 0 ? 0 : Math.PI; pa.rotation.y = c;
      gateLeaf(lf.len, h).emit(pa);
      const pb = new THREE.Group(); pb.position.set(lf.len, 0, 0.05); gateLeaf(lf.len, h).emit(pb); pa.add(pb);
      G.site.add(pa);
      rec.parts.push((e) => { pa.rotation.y = c - lf.dir * (Math.PI / 2) * e; pb.rotation.y = lf.dir * Math.PI * 0.97 * e; });
    }
  }
  // the track and the wall the sliding leaf runs behind
  if (g.runback) {
    const tb = new Builder(), a = Math.min(g.x0, g.runback[0]), b = Math.max(g.x1, g.runback[1]);
    tb.box([a, H - 0.2, b, H - 0.12, 0, 0.03], "steel", "gate");
    tb.emit(G.site);
  }
  registry.gates.push(rec);
  const what = { walk: "walking gate", two_wheeler: "walk + bike gate", vehicle: "car gate", vehicle_wicket: "car gate + wicket" }[g.kind] || "gate";
  const op = { sliding: " · sliding", telescopic: " · telescopic", bifold: " · bi-fold" }[g.op] || "";
  if (g.tag) addLabel(`🚪 ${g.tag} · ${g.feet || width.toFixed(2) + " m"} ${what}${op}`, [(g.x0 + g.x1) / 2, H + 0.3, gh + 0.55], "lbl",
    () => state.mode === "outside" && !state.xray && !state.explode && !lift.rider);
}

// ───────────────────────────── drive in: the design car from the road, through its gate, to its bay
const drive = { on: false, s: 0, grp: null, cum: null, len: 0, trail: null, n: 0 };
function buildDrive() {
  drive.on = false; drive.grp = null;
  $("btnDrive").style.display = S.drive ? "" : "none";
  if (!S.drive) return;
  const d = S.drive.dims, bld = new Builder();
  car(bld, { c: [0, 0], along: "x", L: d.L, W: d.W, i: 1 }, 0);
  drive.grp = new THREE.Group(); bld.emit(drive.grp); G.site.add(drive.grp);
  const P = S.drive.path; drive.cum = [0];
  for (let i = 1; i < P.length; i++) drive.cum.push(drive.cum[i - 1] + Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]));
  drive.len = drive.cum[drive.cum.length - 1];
  // the swept path: the body's four corners traced on the ground (as engineers check a gate)
  const pos = [];
  const hl = d.L / 2, hw = d.W / 2;
  const corner = (p, su, sv) => [p[0] + su * hl * Math.cos(p[2]) - sv * hw * Math.sin(p[2]), p[1] + su * hl * Math.sin(p[2]) + sv * hw * Math.cos(p[2])];
  for (let i = 1; i < P.length; i++) for (const [su, sv] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const a = corner(P[i - 1], su, sv), b = corner(P[i], su, sv);
    pos.push(a[0], 0.03, -a[1], b[0], 0.03, -b[1]);
  }
  const geo = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  drive.trail = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0xFFB020, transparent: true, opacity: 0.9 }));
  drive.trail.visible = false; drive.n = (P.length - 1) * 8; G.site.add(drive.trail);
  drivePose(drive.len);                                 // parked in its bay until asked
}
function drivePose(s) {
  const P = S.drive.path, c = drive.cum;
  let i = 1; while (i < c.length - 1 && c[i] < s) i++;
  const t = clamp((s - c[i - 1]) / Math.max(1e-6, c[i] - c[i - 1]), 0, 1), a = P[i - 1], b = P[i];
  const x = a[0] + (b[0] - a[0]) * t, y = a[1] + (b[1] - a[1]) * t;
  let hd = b[2] - a[2]; while (hd > Math.PI) hd -= 2 * Math.PI; while (hd < -Math.PI) hd += 2 * Math.PI;
  const h = a[2] + hd * t;
  const f = S.footprint, under = S.drive.stilt && x > f[0] && x < f[0] + f[2] && y > f[1] && y < f[1] + f[3];
  drive.grp.position.set(x, under ? 0.15 : 0, -y); drive.grp.rotation.y = h;
  drive.at = [x, y];
  if (drive.trail) drive.trail.geometry.setDrawRange(0, Math.max(0, Math.round(i * 8)));
}
function startDrive() {
  if (!S || !S.drive) return;
  tourInterrupt(); hideStart();
  if (state.mode !== "outside" && state.mode !== "top") setMode("outside");
  const g = S.gates[S.drive.gate] || S.gates[0];
  if (g) { orbit.target.set((g.x0 + g.x1) / 2, 1.0, -(S.plot.h - 1.0)); orbit.r = 21; orbit.phi = 0.72; orbit.theta = frontTheta();
    flyTo(orbitPos(), orbit.target.clone(), 0.9); }
  drive.on = true; drive.s = 0; drive.trail.visible = true; drivePose(0);
  const wd = g ? `${g.feet || g.width.toFixed(2) + " m"} ${g.tag}` : "";
  toast(`🚗 A ${S.drive.label || "car"} turns in from the ${Math.round((S.road.w || (S.road.y1 - S.road.y0)) / 0.3048 / 5) * 5} ft road through the ${wd} gate — the orange lines trace its four corners.`, 7000);
}
function driveUpdate(dt) {
  if (!drive.on) return false;
  const H = S.plot.h, y = drive.at ? drive.at[1] : H + 5, left = drive.len - drive.s;
  const v = left < 3 ? Math.max(0.35, left * 0.7) : (Math.abs(y - H) < 3.5 ? 1.4 : 3.2);
  drive.s = Math.min(drive.len, drive.s + v * dt);
  drivePose(drive.s);
  if (drive.s >= drive.len - 1e-3) { drive.on = false; setTimeout(() => { if (!drive.on && drive.trail) { drive.trail.visible = false; poke(); } }, 6000); }
  return true;
}
$("btnDrive").addEventListener("click", startDrive);

// ───────────────────────────── Vastu overlay
const ZONE_COL = { NE: 0x2E86DE, E: 0xF6C343, SE: 0xE67E22, S: 0xC0392B, SW: 0x8E5B3A, W: 0x95A5A6, NW: 0xB8E0F6, N: 0x27AE60, CENTER: 0xF1C40F };
function buildVastu(ffl) {
  for (const g of S.mandala) {
    const [x0, y0, x1, y1] = g.r;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0 - 0.04, y1 - y0 - 0.04),
      new THREE.MeshBasicMaterial({ color: ZONE_COL[g.zone] || 0xffffff, transparent: true, opacity: 0.22, depthWrite: false }));
    m.rotation.x = -Math.PI / 2; m.position.set((x0 + x1) / 2, ffl + 0.012, -(y0 + y1) / 2); G.vastu.add(m);   // moves with the floor
  }
  if (S.brahmasthan) {
    const [x0, y0, x1, y1] = S.brahmasthan, l = new Lines(), z = ffl + 0.02;
    l.seg([x0, z, -y0], [x1, z, -y0]); l.seg([x1, z, -y0], [x1, z, -y1]); l.seg([x1, z, -y1], [x0, z, -y1]); l.seg([x0, z, -y1], [x0, z, -y0]);
    const ls = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(l.p, 3)),
      new THREE.LineBasicMaterial({ color: 0xFFD700 }));
    G.vastu.add(ls);
  }
}
function setVastu(on) { state.vastu = on; if (G.vastu) G.vastu.visible = on; $("btnVastu").classList.toggle("on", on); poke(); }

// ───────────────────────────── labels (HTML over the canvas)
function addLabel(html, p, cls, show, extra = {}) {
  const el = document.createElement("div"); el.className = cls; el.innerHTML = html; el.style.display = "none";
  $("labels").appendChild(el);
  registry.labels.push(Object.assign({ el, pos: new THREE.Vector3(p[0], p[2], -p[1]), plan: p, show, shown: false }, extra));
}
const _v = new THREE.Vector3();
function updateLabels(w, h) {
  const placed = [];
  const order = registry.labels.map((l) => [l, l.pos.distanceToSquared(camera.position)]).sort((a, b) => a[1] - b[1]).map((x) => x[0]);
  for (const l of order) {
    let ok = l.show();
    if (ok && l.door) {
      const d = l.door, dx = walker.x - d.mid[0], dy = walker.y - d.mid[1], dist = Math.hypot(dx, dy);
      ok = dist < 5 && dist > 0.35 && (dx * d.n[0] + dy * d.n[1]) * l.side > 0.05 && losClear(walker.x, walker.y, l.plan[0], l.plan[1]);
    }
    if (ok) {
      _v.copy(l.pos);
      if (l.lvl) _v.y += G.dyn.position.y;
      if (l.floorLabel !== undefined && state.explode) _v.y += l.floorLabel * GAP;
      _v.project(camera); ok = _v.z < 1 && Math.abs(_v.x) < 1.1 && Math.abs(_v.y) < 1.1;
    }
    let sx = 0, sy = 0;
    if (ok) {
      sx = (_v.x + 1) / 2 * w; sy = (1 - _v.y) / 2 * h;
      if (!l.size || !l.shown) { l.el.style.display = "block"; l.size = [l.el.offsetWidth, l.el.offsetHeight]; }
      const [bw, bh] = l.size, x0 = sx - bw / 2, y0 = l.door ? sy - bh : sy - bh / 2;
      if (placed.some((p) => x0 < p[2] && x0 + bw > p[0] && y0 < p[3] && y0 + bh > p[1])) ok = false;
      else placed.push([x0 - 3, y0 - 2, x0 + bw + 3, y0 + bh + 2]);
    }
    if (!ok) { if (l.shown || l.el.style.display !== "none") { l.el.style.display = "none"; l.shown = false; } continue; }
    l.shown = true; l.el.style.display = "block";
    l.el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px) translate(-50%, ${l.door ? "-100%" : "-50%"})`;
  }
}

// ───────────────────────────── walking grid (from Python), visibility raster, A*
function decodeNav(n) {
  const bin = atob(n.bits), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const free = new Uint8Array(n.nx * n.ny);
  for (let k = 0; k < free.length; k++) free[k] = (bytes[k >> 3] >> (7 - (k & 7))) & 1;
  return Object.assign({}, n, { free });
}
const cellOf = (x, y) => [Math.floor((x - nav.x0) / nav.cell), Math.floor((y - nav.y0) / nav.cell)];
const gridOf = (L) => (L <= 0 ? navs.g0 : L >= TOP() ? (navs.top || navs.g0) : (navs.typ || navs.g0));
const okCell = (i, j, L = walker.level) => {
  const g = gridOf(L);
  return i >= 0 && j >= 0 && i < g.nx && j < g.ny && g.free[j * g.nx + i] === 1;
};
const freeAt = (x, y, L = walker.level) => { const [i, j] = cellOf(x, y); return okCell(i, j, L); };
function nearestFree(x, y, within = 1.5, L = walker.level) {
  const [ci, cj] = cellOf(x, y); if (okCell(ci, cj, L)) return [ci, cj];
  const k = Math.ceil(within / nav.cell); let best = null;
  for (let dj = -k; dj <= k; dj++) for (let di = -k; di <= k; di++) {
    if (okCell(ci + di, cj + dj, L)) { const d = di * di + dj * dj; if (!best || d < best[0]) best = [d, [ci + di, cj + dj]]; }
  }
  return best ? best[1] : null;
}
const centreOf = (c) => [nav.x0 + (c[0] + 0.5) * nav.cell, nav.y0 + (c[1] + 0.5) * nav.cell];
function astar(sx, sy, gx, gy, L = walker.level) {
  const s = nearestFree(sx, sy, 1.0, L), g = nearestFree(gx, gy, 1.2, L);
  if (!s || !g) return null;
  const nx = nav.nx, N = nx * nav.ny, gs = new Float32Array(N).fill(Infinity), came = new Int32Array(N).fill(-1), closed = new Uint8Array(N);
  const si = s[1] * nx + s[0], gi = g[1] * nx + g[0];
  const hf = (i) => { const x = i % nx, y = (i / nx) | 0, dx = Math.abs(x - g[0]), dy = Math.abs(y - g[1]); return Math.max(dx, dy) + 0.414 * Math.min(dx, dy); };
  const heap = [[hf(si), si]]; gs[si] = 0;
  const push = (it) => { heap.push(it); let k = heap.length - 1; while (k > 0) { const p = (k - 1) >> 1; if (heap[p][0] <= heap[k][0]) break; [heap[p], heap[k]] = [heap[k], heap[p]]; k = p; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let k = 0; for (;;) { const l = 2 * k + 1, r = l + 1; let m = k; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === k) break; [heap[m], heap[k]] = [heap[k], heap[m]]; k = m; } } return top; };
  let found = false, n = 0;
  while (heap.length && n++ < 200000) {
    const [, i] = pop(); if (closed[i]) continue; closed[i] = 1;
    if (i === gi) { found = true; break; }
    const x = i % nx, y = (i / nx) | 0;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      if (!dx && !dy) continue;
      const a = x + dx, b = y + dy; if (!okCell(a, b, L)) continue;
      if (dx && dy && !(okCell(a, y, L) && okCell(x, b, L))) continue;
      const j = b * nx + a; if (closed[j]) continue;
      if (segCells && segCells[i] >= 0 && segCells[j] >= 0 && segCells[i] !== segCells[j]) continue;   // that step changes floor
      const ng = gs[i] + (dx && dy ? 1.414 : 1);
      if (ng < gs[j]) { gs[j] = ng; came[j] = i; push([ng + hf(j), j]); }
    }
  }
  if (!found) return null;
  const cells = []; for (let i = gi; i !== -1; i = came[i]) cells.push([i % nx, (i / nx) | 0]);
  cells.reverse();
  // string-pull
  const out = [cells[0]]; let k = 0;
  while (k < cells.length - 1) { let m = cells.length - 1; while (m > k + 1 && !losCells(cells[k], cells[m], L)) m--; out.push(cells[m]); k = m; }
  return out.map(centreOf);
}
function losCells(a, b, L = walker.level) {
  const n = Math.ceil(Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])) * 2) + 1;
  let half = -1;
  for (let k = 0; k <= n; k++) {
    const f = k / n, i = Math.round(a[0] + (b[0] - a[0]) * f), j = Math.round(a[1] + (b[1] - a[1]) * f);
    if (!okCell(i, j, L)) return false;
    const sg = segCells ? segCells[j * nav.nx + i] : -1;
    if (sg >= 0) { if (half >= 0 && sg !== half) return false; half = sg; }      // never cut across a flight's halves
  }
  return true;
}
function buildVis() {              // full-height walls only: what blocks the eye
  const v = { x0: nav.x0, y0: nav.y0, nx: nav.nx, ny: nav.ny, cell: nav.cell, m: new Uint8Array(nav.nx * nav.ny) };
  const ffl = S.levels.ffl;
  for (const k of ["ext", "int", "post"]) for (const b of S.walls[k] || []) {
    if (!(b[4] <= ffl + 1.0 && b[5] >= ffl + 2.0)) continue;
    const i0 = Math.floor((b[0] - v.x0) / v.cell), i1 = Math.floor((b[2] - v.x0) / v.cell), j0 = Math.floor((b[1] - v.y0) / v.cell), j1 = Math.floor((b[3] - v.y0) / v.cell);
    for (let j = Math.max(0, j0); j <= Math.min(v.ny - 1, j1); j++) for (let i = Math.max(0, i0); i <= Math.min(v.nx - 1, i1); i++) v.m[j * v.nx + i] = 1;
  }
  return v;
}
function losClear(x0, y0, x1, y1) {
  const L = Math.hypot(x1 - x0, y1 - y0), n = Math.ceil(L / 0.05);
  for (let k = 1; k < n; k++) {
    const f = k / n, i = Math.floor((x0 + (x1 - x0) * f - vis.x0) / vis.cell), j = Math.floor((y0 + (y1 - y0) * f - vis.y0) / vis.cell);
    if (i >= 0 && j >= 0 && i < vis.nx && j < vis.ny && vis.m[j * vis.nx + i]) return false;
  }
  return true;
}
function stairServes(st, L) { return st.external ? L <= 1 : L >= 0 && L <= TOP(); }
function stairBoxAt(x, y) {          // the step / landing box under (x, y), whatever floor it belongs to
  for (const st of S.stairs) {
    const w = st.wr || st.r; if (x < w[0] || x > w[2] || y < w[1] || y > w[3]) continue;
    for (const b of st.boxes) if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) return { st, b, seg: b[6] || 0 };
  }
  return null;
}
function stairAt(x, y, L = walker.level) {   // the step under the feet on level L (lower half: up from L; upper: into L)
  const r = stairBoxAt(x, y);
  if (!r || !stairServes(r.st, L)) return null;
  const f = r.seg === 0 ? L : L - 1;                              // the flight starts on level f
  if (f < 0 || f > (r.st.external ? 0 : TOP() - 1)) return null;  // no flight there
  return { h: r.st.low + (f - (r.st.ls || 0)) * S.levels.storey + (r.b[5] - r.st.low), st: r.st, seg: r.seg };
}
function groundAt(x, y, L = walker.level) {
  const s = stairAt(x, y, L); if (s) return s.h;
  if (L > 0) return zf(L);
  let z = 0;
  for (const g of S.ground) if (x >= g[0] && x <= g[2] && y >= g[1] && y <= g[3]) z = Math.max(z, g[4]);
  return z;
}
function updateLevel() {           // crossing from the lower half of a flight to the upper half changes the floor
  const s = stairBoxAt(walker.x, walker.y);
  if (!s) { walker.seg = null; return; }
  if (walker.seg === 0 && s.seg === 1 && walker.level < TOP()) walker.level++;
  else if (walker.seg === 1 && s.seg === 0 && walker.level > 0) walker.level--;
  walker.seg = s.seg;
  if (walker.level !== hudLevel) applyLevelView();
}

// ───────────────────────────── modes & camera
function fitRadius(span) { return span / 2 / Math.tan(camera.fov * DEG / 2) * 1.25; }
function frontTheta() { return Math.atan2(-0.45, -1); }      // camera on the road side, a little to the left
function setMode(m, opts = {}) {
  if (!S) return;
  const prev = state.mode; state.mode = m;
  document.querySelectorAll("#modes .b[data-mode]").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  if (m === "dollhouse" || m === "top") state.view = prev === "walk" ? walker.level : (state.view || R0());
  if (m === "dollhouse" && opts.all !== undefined) state.explode = opts.all;
  if (m === "top") state.explode = false;
  $("btnExplode").classList.toggle("on", state.explode);
  $("pad").style.display = m === "walk" ? "block" : "none";
  $("hud").style.display = m === "walk" ? "block" : "none";
  if (m !== "walk") closeWish();
  const [fx, fy, fw, fd] = S.footprint;
  applyLevelView();
  if (m === "dollhouse" && state.explode && !state.xray && !floorTip && multiFloor()) {
    floorTip = true;
    setTimeout(() => toast("Every floor, pulled apart so you can look into each one — the stair" + (S.lift ? " and lift run" : " runs") +
      " through all of them. Tap a floor on the left to open just that floor.", 7000), 900);
  }
  const vz = state.explode ? zf(activeLevel()) + activeLevel() * GAP : zf(activeLevel());
  const centre = new THREE.Vector3(fx + fw / 2, (m === "outside" ? S.levels.ffl : vz) + 0.5, -(fy + fd / 2));
  const bh = S.shell.roof_z + (state.explode ? TOP() * GAP : 0);
  showCop((m === "walk" && lift.rider) || state.xray);
  if (m === "walk") {
    if (lift.rider || lift.anim) { /* still in the lift car */ }
    else if (opts.at) { walker.x = opts.at[0]; walker.y = opts.at[1]; walker.level = opts.level ?? R0(); }
    else if (prev !== "walk") {
      const L = (prev === "dollhouse" || prev === "top") && hasRooms(state.view) ? state.view : R0();
      placeWalkerDefault(L); walker.level = L;
    }
    walker.seg = null;
    if (opts.yaw !== undefined) walker.yaw = opts.yaw;
    walker.pitch = -0.05; walker.h = (lift.rider || lift.anim ? lift.z : groundAt(walker.x, walker.y)) + walker.eye; walker.path = [];
    const toP = new THREE.Vector3(walker.x, walker.h, -walker.y);
    const toT = toP.clone().add(lookDir());
    flyTo(toP, toT, prev === "walk" ? 0.01 : 1.1);
  } else {
    orbit.target.copy(centre);
    const span = Math.max(S.plot.w, S.plot.h);
    if (m === "outside") { orbit.phi = 1.1; orbit.theta = frontTheta(); orbit.r = fitRadius(Math.max(span, bh * 1.3)) * 1.3;
      orbit.target.y = Math.max(1.5, bh * 0.4); orbit.target.z -= 1.5; }
    if (m === "dollhouse" && !state.explode) { orbit.phi = 0.72; orbit.theta = frontTheta(); orbit.r = fitRadius(Math.max(fw, fd)) * 1.35; }
    if (m === "dollhouse" && state.explode) {            // frame the whole stack
      const z0 = zf(0), z1 = zf(TOP()) + TOP() * GAP;
      orbit.target.set(fx + fw / 2, (z0 + z1) / 2, -(fy + fd / 2));
      orbit.phi = 1.0; orbit.theta = frontTheta(); orbit.r = fitRadius(Math.max(fw, fd, (z1 - z0) * 1.15)) * 1.2;
    }
    if (m === "top") { orbit.phi = 0.012; orbit.theta = 0; orbit.r = fitRadius(span) * 1.25; }
    const toP = orbitPos(), toT = orbit.target.clone();
    flyTo(toP, toT, opts.instant ? 0.01 : 1.0);
  }
  poke(1500);
}
let floorTip = false;
function placeWalkerDefault(L = R0()) {
  const main = S.doors.find((d) => d.kind === "main");
  let p = null;
  if (main) { const c = nearestFree(main.mid[0] + main.n[0] * 0.9, main.mid[1] + main.n[1] * 0.9, 1.5, L); if (c) p = centreOf(c); }
  if (!p) { const lr = S.rooms.find((r) => r.cat === "LIVING") || S.rooms[0]; const c = nearestFree((lr.r[0] + lr.r[2]) / 2, (lr.r[1] + lr.r[3]) / 2, 2, L); if (c) p = centreOf(c); }
  if (!p) p = [S.plot.w / 2, S.plot.h / 2];
  walker.x = p[0]; walker.y = p[1];
  walker.yaw = main ? Math.atan2(main.n[1], main.n[0]) : 0;
}
function lookDir() {
  const c = Math.cos(walker.pitch);
  return new THREE.Vector3(Math.cos(walker.yaw) * c, Math.sin(walker.pitch), -Math.sin(walker.yaw) * c);
}
function orbitPos() {
  const s = Math.sin(orbit.phi);
  return new THREE.Vector3(orbit.target.x + orbit.r * s * Math.sin(orbit.theta), orbit.target.y + orbit.r * Math.cos(orbit.phi), orbit.target.z + orbit.r * s * Math.cos(orbit.theta));
}
function flyTo(toP, toT, dur, done = null) {
  trans.fromP.copy(camera.position); const d = new THREE.Vector3(); camera.getWorldDirection(d);
  trans.fromT.copy(camera.position).add(d.multiplyScalar(5));
  trans.toP.copy(toP); trans.toT.copy(toT); trans.t = 0; trans.dur = dur; trans.on = true; trans.done = done;
}

// ───────────────────────────── input
const ptrs = new Map();
let drag = null;
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId); canvas.focus();
  ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  drag = { x: e.clientX, y: e.clientY, t: performance.now(), moved: 0, btn: e.button, shift: e.shiftKey };
  orbit.auto = false; poke();
});
canvas.addEventListener("pointermove", (e) => {
  if (!ptrs.has(e.pointerId)) return;
  const p = ptrs.get(e.pointerId), dx = e.clientX - p.x, dy = e.clientY - p.y;
  p.x = e.clientX; p.y = e.clientY;
  if (drag) drag.moved += Math.abs(dx) + Math.abs(dy);
  if (ptrs.size === 2) {                     // pinch zoom
    const [a, b] = [...ptrs.values()], d = Math.hypot(a.x - b.x, a.y - b.y);
    if (drag && drag.pinch) { const k = drag.pinch / Math.max(1, d); if (state.mode !== "walk") orbit.r = clamp(orbit.r * k, 3, 140); }
    if (drag) drag.pinch = d;
    poke(); return;
  }
  if (trans.on) return;
  if (state.mode === "walk") {
    walker.yaw += dx * 0.0042; walker.pitch = clamp(walker.pitch + dy * 0.0042, -1.35, 1.35); walker.lastLook = performance.now();
  } else if (drag && (drag.btn === 2 || drag.shift)) {
    const k = orbit.r * 0.0016, right = new THREE.Vector3(Math.cos(orbit.theta), 0, -Math.sin(orbit.theta)), fwd = new THREE.Vector3(-Math.sin(orbit.theta), 0, -Math.cos(orbit.theta));
    orbit.target.addScaledVector(right, -dx * k).addScaledVector(fwd, dy * k);
  } else {
    orbit.theta -= dx * 0.0055; orbit.phi = clamp(orbit.phi - dy * 0.0045, 0.012, 1.52);
  }
  poke();
});
const endPtr = (e) => {
  if (!ptrs.has(e.pointerId)) return;
  ptrs.delete(e.pointerId);
  if (drag && ptrs.size === 0) {
    if (drag.moved < 7 && performance.now() - drag.t < 450 && !drag.pinch) tap(e.clientX, e.clientY);
    drag = null;
  }
};
canvas.addEventListener("pointerup", endPtr);
canvas.addEventListener("pointercancel", endPtr);
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (state.mode === "walk") { stepWalker(-Math.sign(e.deltaY) * 0.35, 0); tourInterrupt(); }
  else orbit.r = clamp(orbit.r * (1 + Math.sign(e.deltaY) * 0.1), 3, 140);
  poke();
}, { passive: false });
window.addEventListener("keydown", (e) => {
  if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " "].includes(e.key)) e.preventDefault();
  keys.add(e.key.toLowerCase());
  if (state.mode === "walk" && /^(arrow|w|a|s|d)/.test(e.key.toLowerCase())) { tourInterrupt(); if (lift.auto && !lift.rider) lift.auto = null; }
  poke();
});
window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
window.addEventListener("blur", () => keys.clear());
document.querySelectorAll("#pad button[data-k]").forEach((b) => {
  const k = b.dataset.k;
  const on = (e) => { e.preventDefault(); keys.add("pad-" + k); tourInterrupt(); poke(); };
  const off = () => keys.delete("pad-" + k);
  b.addEventListener("pointerdown", on); b.addEventListener("pointerup", off); b.addEventListener("pointerleave", off); b.addEventListener("pointercancel", off);
});
$("eye").addEventListener("change", (e) => { walker.eye = parseFloat(e.target.value); poke(); });

const ray = new THREE.Raycaster();
function tap(cx, cy) {
  const rect = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((cx - rect.left) / rect.width) * 2 - 1, -((cy - rect.top) / rect.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hits = ray.intersectObjects(registry.pick.filter((m) => m.visible && m.parent && m.parent.visible), false)
    .filter((h) => !h.face || h.face.normal.y > 0.5);                 // floors, steps and the terrace — not ceilings
  if (lift.ok) {
    if (state.mode === "walk" && (lift.rider || lift.anim)) { if (lift.rider) stepOut(); return; }
    const shown = (o) => { for (let q = o; q; q = q.parent) if (!q.visible) return false; return true; };
    const lh = ray.intersectObjects(lift.hits.filter(shown), false)[0];
    if (lh && (!hits.length || lh.distance <= hits[0].distance + 0.05)) {
      const info = lh.object.userData.lift;
      if (state.mode !== "walk") { if (info.L !== undefined) { liftCall(stopOf(info.L), "remote"); toast(`🛗 Calling the lift to ${levelLabel(info.L)}…`, 1800); } return; }
      if (canBoard() && (info.kind === "car" || info.L === walker.level)) { boardLift(); return; }
      if (info.L === walker.level && Math.hypot(walker.x - S.lift.door.mid[0], walker.y - S.lift.door.mid[1]) < 3.5) {
        liftCall(stopOf(info.L), "hall"); toast("🔔 Lift called — it is coming.", 1800); return;
      }
    }
  }
  if (!hits.length) return;
  const p = hits[0].point, x = p.x, y = -p.z;
  let L = hits[0].object.userData.level ?? 0;
  if (hits[0].object.userData.role === "slab") L = Math.min(TOP(), L + 1);         // a slab top is the floor above
  if (hits[0].object.userData.role === "stair" && p.y > zf(L) + S.levels.storey / 2) L = Math.min(TOP(), L + 1);
  if (state.explode && state.mode !== "walk") L = clamp(Math.round((p.y - zf(0)) / (S.levels.storey + GAP)), 0, TOP());
  if (state.mode === "walk") { tourInterrupt(); walkTo(x, y, L); }
  else if (inFootprint(x, y) || L > 0) {
    const c = nearestFree(x, y, 1.2, L);
    if (c) { const q = centreOf(c); setMode("walk", { at: q, level: L, yaw: Math.atan2(S.footprint[1] + S.footprint[3] / 2 - q[1], S.footprint[0] + S.footprint[2] / 2 - q[0]) }); }
  }
}
function walkTo(x, y, L = walker.level) {
  let path = null;
  if (L === walker.level) path = astar(walker.x, walker.y, x, y, L);
  else {                            // another floor: up or down the stairs, one flight at a time
    let cur = [walker.x, walker.y], lvl = walker.level, pts = [cur];
    while (lvl !== L && pts) {
      const up = L > lvl;
      const st = S.stairs.find((q) => stairServes(q, lvl) && stairServes(q, up ? lvl + 1 : lvl - 1));
      if (!st) { pts = null; break; }
      const lowHalf = st.boxes.filter((b) => !(b[6] || 0)), highHalf = st.boxes.filter((b) => b[6]);
      const a = lowHalf.reduce((m, q) => (q[5] > m[5] ? q : m), lowHalf[0]), b = highHalf.reduce((m, q) => (q[5] < m[5] ? q : m), highHalf[0]);
      const ca = [(a[0] + a[2]) / 2, (a[1] + a[3]) / 2], cb = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
      const [p0, p1] = up ? [ca, cb] : [cb, ca];
      const leg = astar(cur[0], cur[1], p0[0], p0[1], lvl);
      if (!leg) { pts = null; break; }
      pts = pts.concat(leg.slice(1)); pts.push(p1);
      cur = p1; lvl += up ? 1 : -1;
    }
    if (pts) { const last = astar(cur[0], cur[1], x, y, L); path = last ? pts.concat(last.slice(1)) : null; }
  }
  if (path && path.length) { walker.path = path.slice(1); if (!walker.path.length) walker.path = [path[0]]; poke(); }
  else toast(L !== walker.level ? "No way up there from here." : "Can't walk there — that spot is inside a wall or furniture.");
}
function stepWalker(fwd, side) {
  const fx = Math.cos(walker.yaw), fy = Math.sin(walker.yaw), sx = Math.cos(walker.yaw - Math.PI / 2), sy = Math.sin(walker.yaw - Math.PI / 2);
  const nx = walker.x + fx * fwd + sx * side, ny = walker.y + fy * fwd + sy * side;
  if (freeAt(nx, ny)) { walker.x = nx; walker.y = ny; }
  else if (freeAt(nx, walker.y)) walker.x = nx;
  else if (freeAt(walker.x, ny)) walker.y = ny;
  updateLevel();
}

// ───────────────────────────── guided tour
const tour = { on: false, i: -1, phase: "", t: 0, paused: false, base: 0, spoken: true, capAt: 0, capWords: "", capMax: 8 };
// The guide speaks English (Indian), Telugu or Hindi. Its words come with every stop (st.talk[lang], written by
// narration.py); each sentence has a key k, and a neural voice prepared by the app has spoken it into
// voice/<voice>/<k>.wav. A sentence without a clip falls back to the browser's own (offline) voice for that
// language, else the caption is read in silence.
const LANGS = ["en", "te", "hi"];
const LANG_NAME = { en: "English", te: "తెలుగు", hi: "हिन्दी" };
const UI = {
  walking: { en: "Walking to: ", te: "వెళ్తున్నాం: ", hi: "चल रहे हैं: " },
  lift: { en: "🛗 Calling the lift — it takes you up, then the doors open.", te: "🛗 లిఫ్ట్ వస్తోంది — పైకి తీసుకెళ్తది, తర్వాత డోర్లు తెరుచుకుంటాయి.",
    hi: "🛗 लिफ़्ट आ रही है — ऊपर ले जाएगी, फिर दरवाज़े खुलेंगे।" },
  ready: { en: "voice ready", te: "వాయిస్ రెడీ", hi: "आवाज़ तैयार" },
  making: { en: "preparing the voice", te: "వాయిస్ తయారవుతోంది", hi: "आवाज़ तैयार हो रही है" },
  browser: { en: "the browser's voice", te: "బ్రౌజర్ వాయిస్", hi: "ब्राउज़र की आवाज़" },
  silent: { en: "words on screen (no voice for this language here)", te: "స్క్రీన్ మీద మాటలు మాత్రమే (ఈ భాషకి వాయిస్ లేదు)",
    hi: "सिर्फ़ स्क्रीन पर शब्द (इस भाषा की आवाज़ यहाँ नहीं है)" },
};
let VOICES = {};                                         // from the app: {lang: {url, label, neural, have, total, running}}
let argLang = null;
state.lang = "en";
const browserVoice = {};
let voice = null;
function pickVoice() {
  if (!("speechSynthesis" in window)) return;
  const vs = speechSynthesis.getVoices().filter((v) => v.localService !== false);   // offline voices only
  browserVoice.en = vs.find((v) => /en[-_]IN/i.test(v.lang)) || vs.find((v) => /^en/i.test(v.lang)) || null;
  browserVoice.te = vs.find((v) => /^te/i.test(v.lang)) || null;
  browserVoice.hi = vs.find((v) => /^hi/i.test(v.lang)) || null;
  voice = browserVoice.en;
}
if ("speechSynthesis" in window) { pickVoice(); speechSynthesis.onvoiceschanged = pickVoice; }
const say = { list: [], i: 0, audio: null, token: 0 };
function neuralOn() { const v = VOICES[state.lang]; return !!(v && v.neural); }
function stopSpeech() {
  say.token++;
  if (say.audio) { try { say.audio.pause(); } catch (e) { /* gone */ } say.audio = null; }
  if ("speechSynthesis" in window) speechSynthesis.cancel();
}
function pauseSpeech(p) {
  if (say.audio) { if (p) say.audio.pause(); else say.audio.play().catch(() => {}); }
  if ("speechSynthesis" in window) { if (p) speechSynthesis.pause(); else speechSynthesis.resume(); }
}
function browserSay(text, tok, done) {
  const bv = browserVoice[state.lang];
  if (!bv || !("speechSynthesis" in window)) {          // nobody here speaks this language: the time to read it
    setTimeout(done, Math.max(1200, text.split(/\s+/).length * 420));
    return;
  }
  try {
    const u = new SpeechSynthesisUtterance(text); u.voice = bv; u.lang = bv.lang; u.rate = 0.95;
    u.onend = done; u.onerror = done;
    speechSynthesis.speak(u);
  } catch (e) { done(); }
}
function playNext(tok) {
  if (tok !== say.token) return;
  if (say.i >= say.list.length) { tour.spoken = true; return; }
  const line = say.list[say.i++];
  let finished = false;
  const done = () => { if (finished) return; finished = true; say.audio = null; if (tok === say.token) setTimeout(() => playNext(tok), 140); };
  const v = VOICES[state.lang];
  if (v && v.neural && line.k) {
    const a = new Audio(`${v.url}/${line.k}.wav`);
    say.audio = a;
    a.onended = done;
    a.onerror = () => { if (tok === say.token && !finished) { say.audio = null; browserSay(line.t, tok, done); } };
    a.play().catch(() => { if (tok === say.token && !finished && say.audio === a) { say.audio = null; browserSay(line.t, tok, done); } });
    return;
  }
  browserSay(line.t, tok, done);
}
function speakLines(lines, text) {
  stopSpeech();
  tour.spoken = true;
  if (!state.voice) return;
  say.list = lines && lines.length ? lines : (text ? [{ k: "", t: text }] : []);
  say.i = 0;
  if (!say.list.length) return;
  tour.spoken = false;
  playNext(say.token);
}
// one line in the browser's voice, outside the guide's words (the lift calling out its floor)
function speak(text, lang = "en") {
  const bv = browserVoice[lang] || browserVoice.en;
  if (!bv || !("speechSynthesis" in window)) return;
  try { const u = new SpeechSynthesisUtterance(text); u.voice = bv; u.lang = bv.lang; u.rate = 0.95; speechSynthesis.speak(u); }
  catch (e) { /* no voice */ }
}
// a room's words when the owner asks (🔊 Tell me): the guide's stop for it, else the same room of the toured flat
function roomWords(name) {
  const rooms = (S && S.tour || []).filter((s) => s.kind === "room" && s.talk);
  const tail = (n) => String(n || "").split(" · ").pop();
  const st = rooms.find((s) => s.room === name) || rooms.find((s) => tail(s.room) === tail(name));
  return st ? words(st) : null;
}
// the stop's words in the chosen language (a stop after the floor choice has one version per floor / lift-or-stairs)
function words(st) {
  const tk = st.talk && (st.talk[state.lang] || st.talk.en);
  if (!tk) return { title: st.title, say: st.say, speech: null };
  let w = tk;
  if (tk.by) {
    const c = tour.choice || {};
    w = tk.by[`${c.L}|${c.via}`] || tk.by[`${c.L}`] || Object.values(tk.by)[0] || tk;
  }
  return { title: w.title || tk.title || st.title, say: w.say || st.say, speech: w.speech || null };
}
function setLang(l, fromApp = false) {
  if (!LANGS.includes(l)) return;
  const changed = state.lang !== l;
  state.lang = l;
  drawLangs();
  if (!fromApp && changed) ST.value({ id: `lang-${Date.now()}`, action: "lang", lang: l });   // the app prepares that voice
  if (changed && tour.on && tour.cur && ["wait", "pan", "end", "choice"].includes(tour.phase)) {
    caption(tour.cur);                                   // say this stop again, in the new language
    if (tour.phase === "choice") showChoice(tour.cur);
  }
}
function drawLangs() {
  for (const id of ["startLang", "capLang"]) {
    const box = $(id); if (!box) continue;
    box.innerHTML = "";
    for (const l of LANGS) {
      const b = document.createElement("button");
      b.className = "b" + (state.lang === l ? " on" : ""); b.textContent = LANG_NAME[l];
      b.onclick = () => setLang(l);
      box.appendChild(b);
    }
  }
  voiceLine();
}
function voiceLine() {
  const el = $("startVoice"); if (!el) return;
  const v = VOICES[state.lang];
  if (!v || !v.neural) {                                // the browser speaks, or nobody does: say which
    el.textContent = browserVoice[state.lang] ? `🗣️ ${UI.browser[state.lang]}` : `📝 ${UI.silent[state.lang]}`;
    return;
  }
  const n = v.have || 0, tot = v.total || 0;
  el.textContent = n >= tot && tot ? `🎙️ ${v.label} · ${UI.ready[state.lang]}` : `🎙️ ${v.label} · ${UI.making[state.lang]} ${n} / ${tot}`;
}
async function pollVoice() {                             // while the app speaks the sentences, count them here
  const v = VOICES[state.lang];
  if (!v || !v.neural || !v.running) return;
  try {
    const r = await fetch(`${v.url}/_status.json?t=${Date.now()}`, { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      if (typeof j.done === "number") { v.have = Math.max(v.have || 0, j.done); v.total = j.total || v.total; }
      if (j.state === "done" || j.state === "error") v.running = false;
    }
  } catch (e) { /* not yet */ }
  voiceLine();
}
setInterval(() => { if (!state.started || tour.on) pollVoice(); }, 5000);
function tourStart() {
  if (!S || !S.tour || !S.tour.length) { toast("No guided walk for this plan."); return; }
  hideStart(); tour.on = true; tour.paused = false; tour.i = -1;
  const ch = S.tour.find((s) => s.kind === "choice");
  tour.choice = ch ? { L: ch.default.L, via: ch.default.via } : null;
  showChoice(null);
  $("cap").style.display = "block"; $("tPause").textContent = "⏸ Pause";
  const st0 = S.tour[0];
  walker.level = S.tour_level || 0;
  setMode("walk", { at: st0.path[0], level: S.tour_level || 0, yaw: yawTo(st0.path[0], st0.look) });
  setTimeout(tourNext, 1150);
}
function yawTo(from, to) { return to ? Math.atan2(to[1] - from[1], to[0] - from[0]) : walker.yaw; }
// a building of floors: stops after the choice say 'chosen' for the floor; titles and words name it
function resolveStop(raw) {
  const st = Object.assign({}, raw), c = tour.choice;
  if (c) {
    if (st.level === "chosen") st.level = c.L;
    const opt = (S.tour.find((s) => s.kind === "choice") || { options: [] }).options.find((o) => o.L === c.L);
    const fill = (t) => String(t || "").replace(/\{floor\}/g, opt ? opt.label : levelLabel(c.L)).replace(/\{home\}/g, opt ? opt.home : "the home");
    if (st.kind === "ride" && st.say_by) st.say = st.say_by[c.via] || st.say;
    st.title = fill(st.title); st.say = fill(st.say);
  }
  return st;
}
function showChoice(st) {
  const box = $("capC"); box.innerHTML = ""; box.style.display = st ? "flex" : "none";
  if (!st) return;
  const vias = st.via || ["stairs"];
  const tk = (st.talk && (st.talk[state.lang] || st.talk.en)) || {};
  const row1 = document.createElement("div"); row1.className = "crow";
  for (const v of vias) {
    const b = document.createElement("button"); b.className = "b" + (tour.choice.via === v ? " on" : "");
    b.textContent = (tk.via && tk.via[v]) || (v === "lift" ? "🛗 By the lift" : "🪜 By the stairs");
    b.onclick = () => { tour.choice.via = v; showChoice(st); };
    row1.appendChild(b);
  }
  const row2 = document.createElement("div"); row2.className = "crow";
  for (const o of st.options) {
    const b = document.createElement("button"); b.className = "b go" + (tour.choice.L === o.L ? " on" : "");
    const lo = (tk.options || []).find((x) => x.L === o.L);
    b.textContent = lo ? lo.label : `${o.label} · ${o.home.replace(/^the /, "")}`;
    b.onclick = () => { tour.choice.L = o.L; showChoice(null); tourNext(); };
    row2.appendChild(b);
  }
  box.appendChild(row1); box.appendChild(row2);
}
function tourNext() {
  if (!tour.on) return;
  if (tour.phase === "choice") showChoice(null);
  tour.i++;
  if (tour.i >= S.tour.length) return tourEnd(true);
  const st = resolveStop(S.tour[tour.i]);
  tour.cur = st;
  $("capN").textContent = `${tour.i + 1} / ${S.tour.length}`;
  if (st.kind === "end") { caption(st); tour.phase = "end"; tour.t = 0; return; }
  if (st.kind === "choice") { caption(st); showChoice(st); tour.phase = "choice"; tour.t = 0; return; }
  if (st.kind === "ride" && st.level !== walker.level && tour.choice && tour.choice.via === "lift" && lift.ok &&
      stopOf(st.level) >= 0 && stopOf(walker.level) >= 0 && liftTrip(st.level, true)) {
    tour.phase = "ride"; tour.t = 0;
    $("capT").textContent = words(st).title; $("capS").textContent = UI.lift[state.lang];
    return;
  }
  if (st.level !== undefined && st.target) {           // upstairs: the walker plans the climb flight by flight
    walkTo(st.target[0], st.target[1], st.level);
    tour.phase = walker.path.length ? "walk" : "turn"; tour.t = 0; tour.from = walker.yaw; tour.to = walker.yaw;
    $("capT").textContent = UI.walking[state.lang] + words(st).title; $("capS").textContent = "";
    return;
  }
  if (st.kind === "start" || !st.path.length) { caption(st); tour.phase = "wait"; tour.t = 0; return; }
  walker.path = st.path.slice(1); tour.phase = "walk"; tour.t = 0;
  $("capT").textContent = UI.walking[state.lang] + words(st).title; $("capS").textContent = "";
}
function caption(st) {
  const w = words(st);
  $("capT").textContent = w.title; $("capS").textContent = w.say;
  tour.capWords = w.say || ""; tour.capAt = performance.now();
  const n = (w.say || "").split(/\s+/).length / 3.0;
  tour.capMax = state.voice && neuralOn() ? Math.max(30, n * 2.6 + 20) : Math.max(3.5, n) + 4;
  speakLines(w.speech, w.say);
}
function tourInterrupt() { if (tour.on) tourEnd(false); }
function tourEnd(finished) {
  tour.on = false; tour.phase = ""; walker.path = []; showChoice(null);
  if (lift.auto && lift.auto.tour) lift.auto = null;
  stopSpeech();
  $("cap").style.display = "none";
  if (finished) setMode("dollhouse");
}
function tourUpdate(dt) {
  if (!tour.on || tour.paused) return;
  const st = tour.cur; if (!st) return;
  tour.t += dt;
  if (tour.phase === "ride") {                       // out of the lift on the chosen floor: walk into the lobby
    if (!lift.auto && !lift.rider && !lift.anim && walker.level === st.level) {
      walkTo(st.target[0], st.target[1], st.level);
      tour.phase = walker.path.length ? "walk" : "turn"; tour.t = 0; tour.from = walker.yaw; tour.to = walker.yaw;
    } else if (!lift.auto && !lift.rider && !lift.anim && tour.t > 4) {   // the lift could not take us: the stairs
      walkTo(st.target[0], st.target[1], st.level); tour.phase = "walk"; tour.t = 0;
    }
    return;
  }
  if (tour.phase === "choice") return;               // waiting for the owner (⏭ Next takes the suggestion)
  const minRead = Math.max(3.5, (tour.capWords || "").split(" ").length / 3.2);
  // browser speech 'end' events are unreliable in some browsers: never wait longer than tour.capMax
  const readDone = () => {
    const el = (performance.now() - tour.capAt) / 1000;
    return el > tour.capMax || (el > (state.voice ? 2.0 : minRead) && tour.spoken);
  };
  if (tour.phase === "walk" && !walker.path.length) {
    tour.phase = "turn"; tour.t = 0; tour.from = walker.yaw;
    tour.to = st.look ? yawTo([walker.x, walker.y], st.look) : walker.yaw;
    tour.to = tour.from + ((((tour.to - tour.from) % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI) - Math.PI);
  } else if (tour.phase === "turn") {
    const k = Math.min(1, tour.t / 0.8); walker.yaw = lerp(tour.from, tour.to, ease(k)); walker.pitch = lerp(walker.pitch, -0.08, 0.1);
    if (k >= 1) { caption(st); tour.phase = st.pan ? "pan" : "wait"; tour.t = 0; tour.base = walker.yaw; }
  } else if (tour.phase === "pan") {
    const total = st.pan * DEG, dur = Math.max(3, st.pan / 38), k = Math.min(1, tour.t / dur);
    // look left, sweep right, come back: sin curve over the sweep
    walker.yaw = tour.base + Math.sin(k * Math.PI * 2) * total / 2 * (k < 1 ? 1 : 0);
    if (k >= 1) { walker.yaw = tour.base; tour.phase = "wait"; tour.t = 0; }
  } else if (tour.phase === "wait") {
    if (readDone()) tourNext();
  } else if (tour.phase === "end") {
    if (tour.t > 3 && (tour.spoken || tour.t > 9)) tourEnd(true);
  }
}
$("btnTour").addEventListener("click", tourStart);
$("tPause").addEventListener("click", () => {
  tour.paused = !tour.paused; $("tPause").textContent = tour.paused ? "▶ Resume" : "⏸ Pause";
  if (tour.paused) tour.pausedAt = performance.now(); else if (tour.pausedAt) tour.capAt += performance.now() - tour.pausedAt;
  pauseSpeech(tour.paused);
});
$("tNext").addEventListener("click", () => {
  if (!tour.on) return; stopSpeech(); tour.spoken = true;
  if (tour.phase === "ride") return;                 // let the lift finish the ride
  walker.path = []; tourNext();
});
$("tStop").addEventListener("click", () => tourEnd(false));
$("tVoice").addEventListener("click", () => {
  state.voice = !state.voice; $("tVoice").textContent = state.voice ? "🔊 Voice on" : "🔇 Voice off";
  if (!state.voice) { stopSpeech(); tour.spoken = true; }
});

// ───────────────────────────── sun & sky
function sunPos(min, day, lat) {
  const phi = lat * DEG, decl = 23.44 * DEG * Math.sin(2 * Math.PI * (284 + day) / 365), ha = (min / 60 - 12) * 15 * DEG;
  const el = Math.asin(Math.sin(phi) * Math.sin(decl) + Math.cos(phi) * Math.cos(decl) * Math.cos(ha));
  const az = Math.atan2(Math.sin(ha), Math.cos(ha) * Math.sin(phi) - Math.tan(decl) * Math.cos(phi)) + Math.PI;   // from North, clockwise
  return { el, az };
}
function compassWord(az) {
  const w = ["North", "North-North-East", "North-East", "East-North-East", "East", "East-South-East", "South-East", "South-South-East",
    "South", "South-South-West", "South-West", "West-South-West", "West", "West-North-West", "North-West", "North-North-West"];
  return w[Math.round(((az / DEG) % 360) / 22.5) % 16];
}
function applySun() {
  if (!S) return;
  const { el, az } = sunPos(state.sunMin, state.day, S.lat);
  const n = S.north, e = [n[1], -n[0]];                         // East = North turned clockwise
  const hx = n[0] * Math.cos(az) + e[0] * Math.sin(az), hy = n[1] * Math.cos(az) + e[1] * Math.sin(az);
  const dir = new THREE.Vector3(hx * Math.cos(el), Math.sin(el), -hy * Math.cos(el)).normalize();
  const W = S.plot.w, H = S.plot.h, ctr = new THREE.Vector3(W / 2, 0, -H / 2), half = Math.max(W, H) / 2 + 8;
  sun.target.position.copy(ctr); sun.position.copy(ctr).addScaledVector(dir, 80);
  const cam = sun.shadow.camera; cam.left = -half; cam.right = half; cam.top = half; cam.bottom = -half; cam.near = 1; cam.far = 200; cam.updateProjectionMatrix();
  const day = el > 0.01, low = clamp(Math.sin(Math.max(el, 0)) * 2.2, 0, 1), bp = state.theme === "bp";
  sun.visible = day;
  sun.intensity = day ? (bp ? 1.7 : 3.1) * (0.35 + 0.65 * low) : 0;
  sun.color.setHSL(0.09, low < 0.6 ? 0.85 : 0.3, low < 0.6 ? 0.72 : 0.96);
  hemi.intensity = bp ? (day ? 1.9 : 1.0) : (day ? 0.9 + 0.6 * low : 0.15);
  night.intensity = day ? 0 : (bp ? 0.6 : 1.1);
  for (const role of ["lamp"]) { const m = mat(role); m.emissive.set(day ? (bp ? 0x221A08 : 0x000000) : 0xFFE6B0); m.emissiveIntensity = day ? 1 : 2.2; }
  if (!bp) {
    const k = day ? low : 0;
    skyMat.uniforms.top.value.setRGB(lerp(0.02, 0.24, k), lerp(0.03, 0.45, k), lerp(0.08, 0.78, k)).convertSRGBToLinear();
    skyMat.uniforms.hor.value.setRGB(lerp(0.1, k > 0.35 ? 0.86 : 0.98, day ? 1 : 0.3), lerp(0.1, k > 0.35 ? 0.92 : 0.72, day ? 1 : 0.3), lerp(0.16, k > 0.35 ? 0.96 : 0.5, day ? 1 : 0.4)).convertSRGBToLinear();
    scene.fog.color.copy(skyMat.uniforms.hor.value);
  }
  // which rooms catch direct sun right now (windows facing the sun, not too high)
  const lit = new Set();
  if (day && el < 68 * DEG) for (const w of S.windows) if (!w.vent && w.out[0] * hx + w.out[1] * hy > 0.18) lit.add(w.room);
  const t = state.sunMin, hh = Math.floor(t / 60), mm = t % 60, ap = hh >= 12 ? "PM" : "AM";
  $("sunTime").textContent = `${((hh + 11) % 12) + 1}:${String(mm).padStart(2, "0")} ${ap}`;
  $("sunInfo").innerHTML = day
    ? `Sun from the <b>${compassWord(az)}</b>, ${Math.round(el / DEG)}° up.<br>` +
      (lit.size ? `Direct sunlight now enters: <b>${[...lit].map(esc).join(", ")}</b>.` : "No room gets direct sun at this moment (soft daylight only).")
    : "🌙 After sunset — the lights are on.";
  poke(600);
}
const sunRange = $("sunRange");
sunRange.addEventListener("input", () => { state.sunMin = parseInt(sunRange.value, 10); applySun(); });
document.querySelectorAll("#sunDays .b").forEach((b) => b.addEventListener("click", () => {
  state.day = b.dataset.day === "today" ? dayOfYear(new Date()) : parseInt(b.dataset.day, 10);
  document.querySelectorAll("#sunDays .b").forEach((x) => x.classList.toggle("on", x === b)); applySun();
}));
$("btnSun").addEventListener("click", () => { state.sunOpen = !state.sunOpen; $("sun").style.display = state.sunOpen ? "block" : "none"; $("btnSun").classList.toggle("on", state.sunOpen); });

// ───────────────────────────── theme
function applyTheme() {
  const bp = state.theme === "bp";
  for (const m of registry.meshes) if (m.userData.role) m.material = mat(m.userData.role);
  for (const e of registry.edges) e.visible = bp;
  sky.visible = !bp;
  scene.background = bp ? new THREE.Color(0x0A2342) : null;
  scene.fog = bp ? new THREE.Fog(0x0A2342, 45, 190) : new THREE.Fog(0xDCEBF5, 80, 420);
  hemi.color.set(bp ? 0x9CC8F0 : 0xDDEBFF); hemi.groundColor.set(bp ? 0x1B3A5C : 0xD8CFC0);   // floor bounce lights ceilings
  renderer.toneMappingExposure = bp ? 1.05 : 1.0;
  $("themeTx").textContent = bp ? "Blueprint" : "Real colours";
  $("btnTheme").classList.toggle("on", bp);
  applySun(); drawMinimapBase(mm.plan || 0); poke();
}
$("btnTheme").addEventListener("click", () => { state.theme = state.theme === "bp" ? "real" : "bp"; applyTheme(); });
$("btnVastu").addEventListener("click", () => setVastu(!state.vastu));
function setFurniture(on) {
  state.furn = on;
  for (const lg of levelGroups) if (lg.furn) lg.furn.visible = on;
  $("btnFurn").classList.toggle("on", on); poke();
}
$("btnFurn").addEventListener("click", () => setFurniture(!state.furn));
$("btnShot").addEventListener("click", () => {
  renderer.render(scene, camera);
  canvas.toBlob((b) => { const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "my-house-3d.png"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 4000); });
});
$("btnFull").addEventListener("click", () => {
  const el = $("app");
  if (document.fullscreenElement) document.exitFullscreen(); else if (el.requestFullscreen) el.requestFullscreen().catch(() => toast("Full screen is not allowed here."));
});
$("btnHelp").addEventListener("click", () => { $("help").style.display = "block"; });
$("helpClose").addEventListener("click", () => { $("help").style.display = "none"; });
document.querySelectorAll("#modes .b[data-mode]").forEach((b) => b.addEventListener("click", () => {
  hideStart(); tourInterrupt(); if (state.xray) { state.xray = false; $("btnLift").classList.remove("on"); }
  setMode(b.dataset.mode, b.dataset.mode === "dollhouse" && multiFloor() ? { all: true } : {});
}));
$("btnLift").addEventListener("click", () => setXray(!state.xray));

function hideStart() { if (!state.started) { state.started = true; $("start").style.display = "none"; orbit.auto = false; } }
$("sTour").addEventListener("click", tourStart);
$("sWalk").addEventListener("click", () => { hideStart(); setMode("walk"); setTimeout(() => toast("Drag to look around · tap the floor to walk there"), 1200); });
$("sDoll").addEventListener("click", () => { hideStart(); setMode("dollhouse", multiFloor() ? { all: true } : {}); });

// ───────────────────────────── HUD + "change this room" wishes → Smart Re-Plan
let hudRoom, hudLevel = -1;
const flatNo = (L, lab) => { const i = lab ? "ABCD".indexOf(lab) + 1 : 1; return L === 0 ? `G0${i}` : `${L}0${i}`; };
function nearLift() {
  if (!lift.ok || stopOf(walker.level) < 0) return false;
  return Math.hypot(walker.x - S.lift.door.mid[0], walker.y - S.lift.door.mid[1]) < 2.6;
}
function liftRow(L) {
  const st = S.lift.stops, here = stopOf(L), [lab, dir] = (lift.disp || "|0").split("|");
  const arrow = +dir > 0 ? "▲ " : +dir < 0 ? "▼ " : "";
  const openHere = lift.at === here && (lift.phase === "open" || lift.phase === "opening");
  const called = lift.queue.has(here) || (lift.auto && lift.auto.stage === "call");
  const btn = openHere ? '<button class="b go" data-lin="1">🚶 Step in</button>'
    : `<button class="b go" data-lcall="1">${called ? "⏳ Coming…" : "🔔 Call the lift"}</button>`;
  const to = st.map((s, i) => (i === here ? "" : `<button class="b" data-lift="${s.L}" title="${esc(s.name)}">${esc(s.label)}</button>`)).join("");
  const mv = S.lift.movable ? '<button class="b" data-lmove="1" title="Put the lift somewhere else">✏️ Move the lift</button>' : "";
  return `<div class="row">🛗 Lift at <b>${esc(arrow + lab)}</b> ${btn}</div><div class="row">Ride to: ${to} ${mv}</div>`;
}
function updateHud() {
  if (lift.anim) return;
  if (lift.rider) {
    const st = S.lift.stops, [lab, dir] = (lift.disp || "|0").split("|"), key = `ride|${lift.disp}|${lift.phase}|${lift.queue.size}`;
    if (key === hudRoom) return;
    hudRoom = key;
    const what = lift.phase === "moving" ? `Going to ${esc(st[lift.target].name)} · ${S.lift.speed.toFixed(1)} m/s`
      : lift.phase === "open" ? (lift.queue.size ? "Doors closing…" : "Press a floor on the panel — or walk out")
        : lift.phase === "closing" ? "Doors closing…" : lift.phase === "opening" ? "Doors opening…" : "";
    $("hud").innerHTML = `<b>🛗 In the lift · ${+dir > 0 ? "▲ " : +dir < 0 ? "▼ " : ""}${esc(lab)}</b><div class="sub">${what}</div>`;
    return;
  }
  const L = walker.level, nearL = nearLift();
  const rr = hasRooms(L) ? roomAt(walker.x, walker.y) : null, r = rr && rr.cat === "STAIRS" ? null : rr;
  const sa = r ? null : stairAt(walker.x, walker.y) || (rr ? { st: S.stairs.find((q) => q.name === rr.name) || S.stairs[0], h: 0 } : null);
  const key = `${L}|${r ? r.name : sa ? "stair:" + sa.st.name : inFootprint(walker.x, walker.y) ? "in" : "out"}|${nearL ? lift.disp + lift.phase + lift.at + lift.queue.size + (lift.auto ? lift.auto.stage : "") : 0}|${canWish}`;
  if (key === hudRoom) return;
  hudRoom = key;
  const lv = S.levels, ch = lv.ceiling, ft = Math.round(ch / 0.3048 * 12);
  const floorTxt = TOP() > 1 ? `<span class="fl">${esc(levelLabel(L))}</span> ` : "";
  const liftHtml = nearL ? liftRow(L) : "";
  let html;
  if (levelKind(L) === "terrace") {
    const z = zf(L), fz = Math.round(z / 0.3048);
    html = `<b>🏙️ On the terrace</b><div class="sub">${z.toFixed(2)} m (${fz} ft) above the ground · parapet ${lv.parapet.toFixed(1)} m high</div>`;
  } else if (levelKind(L) === "stilt") {
    html = `<b>🚗 Stilt parking</b><div class="sub">Under the building — the stair${S.lifts.length ? " and the lift" : ""} go up to the homes</div>`;
  } else if (!r && nearL && !sa) {
    html = `<b>🛗 Lift landing</b><div class="sub">${floorTxt}${esc(S.lift.words)}</div>`;
  } else if (!r) {
    html = sa ? `<b>🪜 ${esc(sa.st.name)}</b><div class="sub">${floorTxt}${sa.st.steps} steps of ${sa.st.riser_mm} mm, ` +
      `treads ${sa.st.tread_mm} mm · goes up to the ${levelKind(L + 1) === "terrace" || TOP() <= 1 ? "terrace" : "next floor"}</div>`
      : `<b>${inFootprint(walker.x, walker.y) ? "Inside" : "Outside the building"}</b><div class="sub">${floorTxt}</div>`;
  } else {
    const flat = r.name.includes(" · ") ? r.name.split(" · ")[0] : (S.apartment ? "" : null);
    const nm = r.name.includes(" · ") ? r.name.split(" · ")[1] : r.name;
    const flatTxt = S.apartment && flat !== null && r.role !== "COMMON"
      ? `<span class="fl">🏠 ${houseKind() ? homeName(L) : "Flat " + flatNo(L, flat)}</span> ` : "";
    const badge = { best: "🟢 best Vastu spot", good: "🟢 good Vastu", ok: "🟡 acceptable Vastu", poor: "🔴 not ideal for Vastu" }[r.vastu] || "";
    html = `<b>📍 ${esc(nm)}</b><div class="sub">${floorTxt}${flatTxt}${r.size} (${r.ft}) · ${r.area} m² · ${r.sqft} sq ft · ` +
      `${r.zone_word ? esc(r.zone_word) + " · " : ""}ceiling ${ch.toFixed(2)} m (${Math.floor(ft / 12)}'${ft % 12}")${badge ? " · " + badge : ""}</div>` +
      (canWish && r.key ? `<div class="row"><button class="b" id="hudWish">✏️ Change this room</button><button class="b" id="hudSay">🔊 Tell me</button></div>`
        : `<div class="row"><button class="b" id="hudSay">🔊 Tell me</button></div>`);
  }
  $("hud").innerHTML = html + liftHtml;
  const w = $("hudWish"); if (w) w.onclick = () => openWish(r);
  const sayB = $("hudSay");
  if (sayB) sayB.onclick = () => {
    const w = roomWords(r.name);
    const old = state.voice; state.voice = true;
    speakLines(w ? w.speech : null, w ? w.say : r.say);
    state.voice = old; toast(w ? w.say : r.say, 9000);
  };
  for (const b of $("hud").querySelectorAll("[data-lift]")) b.onclick = () => { if (!liftTrip(+b.dataset.lift)) toast("Can't reach the lift from here."); };
  const lc = $("hud").querySelector("[data-lcall]"); if (lc) lc.onclick = () => { liftCall(stopOf(walker.level), "hall"); hudRoom = undefined; };
  const li = $("hud").querySelector("[data-lin]"); if (li) li.onclick = () => boardLift();
  const lm = $("hud").querySelector("[data-lmove]"); if (lm) lm.onclick = openLiftMove;
}

// ───────────────────────────── floors: the bar, the view of one floor, floors apart, the lift
const multiFloor = () => !!S && TOP() - R0() >= 2;            // two or more floors with rooms
function buildFloorBar() {
  const el = $("floors"); el.innerHTML = "";
  const top = TOP();
  el.style.display = top >= 1 ? "flex" : "none";
  if (multiFloor()) {                                     // every floor at once
    const b = document.createElement("button"); b.className = "b"; b.dataset.l = "all";
    b.textContent = "🏢 All"; b.title = "Every floor, pulled apart";
    b.onclick = () => { hideStart(); tourInterrupt(); if (state.xray) setXray(false);
      setMode(state.mode === "walk" || state.mode === "top" ? "dollhouse" : state.mode === "outside" ? "dollhouse" : state.mode, { all: true }); };
    el.appendChild(b);
  }
  for (let L = top; L >= 0; L--) {
    const b = document.createElement("button"); b.className = "b"; b.dataset.l = L;
    b.textContent = levelKind(L) === "terrace" ? "🏙️ Top" : levelKind(L) === "stilt" ? "🚗 Stilt" : L === 0 ? "G" : String(L);
    b.title = levelLabel(L);
    b.onclick = () => goLevel(L);
    el.appendChild(b);
  }
  $("btnExplode").style.display = top >= 2 ? "" : "none";
}
function updateFloorBar() {
  const a = activeLevel(), all = state.explode && state.mode !== "walk";
  for (const b of $("floors").children) {
    b.classList.toggle("on", b.dataset.l === "all" ? all : (!all && +b.dataset.l === a && (state.mode !== "outside" || state.explode)));
  }
}
function applyLevelView() {
  if (!S) return;
  const m = state.mode, act = activeLevel(), ex = state.explode && m !== "walk", H = S.levels.storey, top = TOP(), r0 = R0();
  for (const lg of levelGroups) {
    const L = lg.L;
    lg.root.position.y = ex ? L * GAP : 0;
    let show = true, ceil = true;
    if (m === "dollhouse" || m === "top") { show = ex || L <= act; ceil = !ex && L < act; }
    else if (m === "outside") ceil = !ex;
    lg.root.visible = show;
    if (lg.ceil) lg.ceil.visible = ceil;
  }
  const xr = !!state.xray && lift.ok;
  if (xr) for (const lg of levelGroups) lg.root.visible = false;
  for (const rec of lift.lv) {
    if (!rec) continue;
    rec.g.position.y = ex ? rec.L * GAP : 0;
    rec.g.visible = xr || !(m === "dollhouse" || m === "top") || ex || rec.L <= act;
    if (rec.walls) rec.walls.visible = !xr;
  }
  if (lift.ok) {
    lift.x.visible = xr;
    lift.car.visible = xr || ex || !((m === "dollhouse" || m === "top") && liftLevelF(lift.z) > act + 0.01);
  }
  for (const g of G.ghost.children) g.visible = xr;
  G.core.visible = ex && !xr;
  G.roof.visible = !xr && (m === "walk" || m === "outside" || act >= top || ex);
  G.roof.position.y = ex ? top * GAP : 0;
  const dl = hasRooms(act) ? act : r0;
  const pl = planOf(dl);
  G.dyn.position.y = (dl - r0) * H + (ex ? dl * GAP : 0);
  G.dyn.visible = !xr && pl === 0 && (hasRooms(act) || m === "outside");
  G.dyn2.position.y = G.dyn.position.y;
  G.dyn2.visible = !xr && pl === 1 && hasRooms(act);
  if (S.upper && mm.plan !== pl) drawMinimapBase(pl);
  G.vastu.visible = state.vastu && !xr;
  if (lift.ok) liftPose(true);
  G.vastu.position.y = G.dyn.position.y;
  for (const o of registry.doors) o.pivot.visible = !(o.groundOnly && dl > 0);
  for (const f of registry.fans) f.visible = m === "walk";
  for (const l of registry.labels) if (l.flat !== undefined) l.el.innerHTML = houseKind() ? `🏠 ${homeName(dl)}` : `🏠 Flat ${flatNo(dl, l.flat)}`, l.size = null;
  hudLevel = walker.level; hudRoom = undefined;
  const cap = document.querySelector("#mini .cap");
  if (cap) cap.textContent = (top > 1 ? levelLabel(act) + " · " : "") + "Tap the map to walk there";
  updateFloorBar(); poke(600);
}
function goLevel(L) {
  hideStart(); tourInterrupt();
  if (state.mode === "walk") {
    if (L === walker.level || lift.anim) return;
    if (lift.rider) { const i = stopOf(L); if (i >= 0) liftCall(i, "car"); else toast("The lift does not go there."); return; }
    if (lift.ok && stopOf(L) >= 0 && stopOf(walker.level) >= 0 && liftTrip(L)) return;   // walk to the lift, ride
    walkTo(walker.x, walker.y, L);                              // by the stairs
    return;
  }
  if (state.xray) { const i = stopOf(L); if (i >= 0) liftCall(i, "remote"); return; }
  state.view = L;
  state.explode = false;                                  // one floor, opened up
  setMode(state.mode === "outside" ? "dollhouse" : state.mode, { all: false });
}
// ───────────────────────────── the lift (scene.lift, measured from the plan): landing doors, a call button and a
// floor display on every floor, a car with its button panel, counterweight, ropes, rails and motor — it rides at the
// lift's own speed (smooth start and stop), chimes, opens both doors together, and you walk in and press a floor.
const DOOR_OPEN_S = 1.4, DOOR_CLOSE_S = 1.6, DWELL_S = 4.0;
const lift = { ok: false };
function liftReset() {
  Object.assign(lift, { ok: false, z: 0, v: 0, at: 0, target: null, queue: new Set(), door: 0, phase: "idle", hold: 0,
    rider: false, anim: null, auto: null, fast: false, dir: 0, lv: [], car: null, carDoors: [], x: null, cw: null,
    ropes: null, ropeMat: null, hits: [], g: {}, open: -1, disp: "", cop: "", labels: {}, prev: null, last: 0 });
}
liftReset();
const liftDisp = { canvas: null, tex: null, mat: null };
const copPlate = { canvas: null, tex: null };
const LIFT_AX = () => { const n = S.lift.door.n; return { n, u: [n[1], -n[0]], theta: Math.atan2(-n[0], n[1]) }; };
function liftPt(u, v) { const D = S.lift.door, a = LIFT_AX(); return [D.mid[0] + a.u[0] * u + a.n[0] * v, D.mid[1] + a.u[1] * u + a.n[1] * v]; }
function lgeo(u0, u1, v0, v1, z0, z1) {
  const g = new THREE.BoxGeometry(Math.max(1e-3, u1 - u0), Math.max(1e-3, z1 - z0), Math.max(1e-3, v1 - v0));
  g.translate((u0 + u1) / 2, (z0 + z1) / 2, -(v0 + v1) / 2); return g;
}
function lgroup(parent) {
  const g = new THREE.Group(), D = S.lift.door; g.position.set(D.mid[0], 0, -D.mid[1]); g.rotation.y = LIFT_AX().theta;
  parent.add(g); return g;
}
function lmesh(g, geo, role, edge = "door", own = null) {
  const m = new THREE.Mesh(geo, own || mat(role)); m.castShadow = !NO_SHADOW.has(role); m.receiveShadow = true;
  if (!own) { m.userData.role = role; registry.meshes.push(m); }
  g.add(m);
  if (edge) { const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), lineMat(edge)); e.userData.edge = true; g.add(e); registry.edges.push(e); }
  return m;
}
const stopOf = (L) => (S && S.lift ? S.lift.stops.findIndex((s) => s.L === L) : -1);
function liftLevelF(z) {                 // fractional level for a car height (the exploded view moves the car with it)
  const st = S.lift.stops;
  if (z <= st[0].z) return st[0].L;
  for (let i = 0; i + 1 < st.length; i++) if (z <= st[i + 1].z) return st[i].L + (z - st[i].z) / (st[i + 1].z - st[i].z) * (st[i + 1].L - st[i].L);
  return st[st.length - 1].L;
}
function liftGeom() {                    // the lift's local layout: u along the door, v into the shaft
  const Lf = S.lift, D = Lf.door, a = LIFT_AX(), c = Lf.clear;
  let vMax = -9, uMin = 9, uMax = -9;
  for (const [x, y] of [[c[0], c[1]], [c[2], c[1]], [c[2], c[3]], [c[0], c[3]]]) {
    const dx = x - D.mid[0], dy = y - D.mid[1];
    vMax = Math.max(vMax, dx * a.n[0] + dy * a.n[1]);
    const u = dx * a.u[0] + dy * a.u[1]; uMin = Math.min(uMin, u); uMax = Math.max(uMax, u);
  }
  const t = D.t, vCar = t / 2 + 0.1, cw = Lf.car.w, cd = Math.min(Lf.car.d, vMax - vCar - 0.04);
  const rear = vMax - (vCar + cd) >= 0.17;
  const side = uMax - cw / 2 >= -uMin - cw / 2 ? 1 : -1;
  let cu = 0;
  if (!rear) cu = clamp(-side * 0.08, uMin + cw / 2 + 0.04, uMax - cw / 2 - 0.04);
  cu = clamp(cu, uMin + cw / 2 + 0.03, uMax - cw / 2 - 0.03);
  const cwU = rear ? [cu - 0.28, cu + 0.28] : side > 0 ? [uMax - 0.14, uMax - 0.03] : [uMin + 0.03, uMin + 0.14];
  const cwV = rear ? [vMax - 0.14, vMax - 0.03] : [vCar + cd / 2 - 0.3, vCar + cd / 2 + 0.3];
  return { t, w: D.w, dc: Lf.door_clear, vCar, cw, cd, ch: Lf.car.h, cu, uMin, uMax, vMax, cwU, cwV, rear,
    zMin: Lf.stops[0].z, zMax: Lf.stops[Lf.stops.length - 1].z };
}
function canvasPlane(w, h, canvas) {
  const tex = new THREE.CanvasTexture(canvas); tex.colorSpace = THREE.SRGBColorSpace;
  return { tex, mat: new THREE.MeshBasicMaterial({ map: tex, toneMapped: false }) };
}
function drawDisplay(label, dir) {
  const c = liftDisp.canvas, g = c.getContext("2d");
  g.fillStyle = "#06090D"; g.fillRect(0, 0, c.width, c.height);
  g.fillStyle = "#FFB000"; g.textAlign = "center"; g.textBaseline = "middle";
  g.font = "bold 46px system-ui, sans-serif"; g.fillText(label, c.width * 0.62, c.height / 2 + 2);
  if (dir) { g.font = "bold 34px system-ui, sans-serif"; g.fillText(dir > 0 ? "▲" : "▼", c.width * 0.24, c.height / 2 + 2); }
  liftDisp.tex.needsUpdate = true;
}
function drawPlate() {                   // the car's button panel, buttons lit while their floor is called
  const c = copPlate.canvas, g = c.getContext("2d"), st = S.lift.stops, n = st.length;
  g.fillStyle = "#AEB6BE"; g.fillRect(0, 0, c.width, c.height);
  g.fillStyle = "#06090D"; g.fillRect(14, 10, c.width - 28, 44);
  g.fillStyle = "#FFB000"; g.font = "bold 30px system-ui"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(lift.disp.split("|")[0] || "", c.width / 2, 33);
  const rows = Math.ceil(n / 2), top = 72, gap = Math.min(52, (c.height - top - 70) / Math.max(1, rows));
  st.forEach((s, i) => {
    const col = n > 3 ? i % 2 : 0.5, row = n > 3 ? Math.floor(i / 2) : i;
    const x = n > 3 ? 40 + col * 48 : c.width / 2, y = c.height - 70 - row * gap;
    const lit = lift.queue.has(i);
    g.beginPath(); g.arc(x, y, 17, 0, Math.PI * 2); g.fillStyle = lit ? "#FFB000" : "#D9DEE3"; g.fill();
    g.lineWidth = 3; g.strokeStyle = lit ? "#FFE29A" : "#6B737B"; g.stroke();
    g.fillStyle = "#1A1D21"; g.font = "bold 17px system-ui"; g.fillText(s.label, x, y + 1);
  });
  const y = c.height - 24, tri = (x, d) => { g.beginPath(); g.moveTo(x, y - 7); g.lineTo(x + d * 8, y); g.lineTo(x, y + 7); g.closePath(); g.fill(); };
  for (const [x, open] of [[30, true], [64, false]]) {                 // door open ◀|▶ and close ▶|◀
    g.beginPath(); g.arc(x, y, 14, 0, Math.PI * 2); g.fillStyle = "#D9DEE3"; g.fill(); g.strokeStyle = "#6B737B"; g.lineWidth = 2; g.stroke();
    g.fillStyle = "#1A1D21"; if (open) { tri(x - 3, -1); tri(x + 3, 1); } else { tri(x - 9, 1); tri(x + 9, -1); }
  }
  g.beginPath(); g.arc(98, y, 14, 0, Math.PI * 2); g.fillStyle = "#FFC400"; g.fill(); g.strokeStyle = "#8A6A00"; g.stroke();   // alarm
  g.fillStyle = "#1A1D21"; g.font = "bold 16px system-ui"; g.fillText("!", 98, y + 1);
  copPlate.tex.needsUpdate = true;
}

function buildLift() {
  const Lf = S.lift;
  lift.ok = !!(Lf && Lf.stops && Lf.stops.length > 1);
  if (!lift.ok) return;
  const k = liftGeom(); lift.k = k;
  const D = Lf.door, a = LIFT_AX(), top = TOP(), H = S.levels.storey;
  liftDisp.canvas = liftDisp.canvas || Object.assign(document.createElement("canvas"), { width: 128, height: 56 });
  if (!liftDisp.tex) { const p = canvasPlane(0.4, 0.17, liftDisp.canvas); liftDisp.tex = p.tex; liftDisp.mat = p.mat; }
  copPlate.canvas = copPlate.canvas || Object.assign(document.createElement("canvas"), { width: 128, height: 384 });
  if (!copPlate.tex) copPlate.tex = canvasPlane(0.2, 0.6, copPlate.canvas).tex;
  lift.disp = ""; lift.cop = "";

  // ── every floor: landing doors (the car door drives them), frame, floor display, call button; the outside tower's walls
  const ring = (o, c) => [[o[0], o[1], o[2], c[1]], [o[0], c[3], o[2], o[3]], [o[0], c[1], c[0], c[3]], [c[2], c[1], o[2], c[3]]];
  const shaftOuter = Lf.tower ? Lf.outer : [Lf.room[0] - 0.0575, Lf.room[1] - 0.0575, Lf.room[2] + 0.0575, Lf.room[3] + 0.0575];
  const wrole = Lf.tower ? "lift_tower" : "wall_out";      // an outside tower reads as its own RCC shaft
  const wallsFor = (bld, z0, z1, stopZ, role = wrole) => {   // ring of shaft walls, the door strip cut round the opening
    for (const s of ring(shaftOuter, Lf.clear)) {
      const inS = D.mid[0] >= s[0] - 0.02 && D.mid[0] <= s[2] + 0.02 && D.mid[1] >= s[1] - 0.02 && D.mid[1] <= s[3] + 0.02;
      if (!inS || stopZ === null) { bld.box([...s, z0, z1], role, "wall"); continue; }
      const alongX = s[2] - s[0] > s[3] - s[1], m = alongX ? D.mid[0] : D.mid[1], lo = m - D.w / 2, hi = m + D.w / 2;
      const piece = (p0, p1, za, zb) => { if (p1 - p0 < 1e-3 || zb - za < 1e-3) return; bld.box(alongX ? [p0, s[1], p1, s[3], za, zb] : [s[0], p0, s[2], p1, za, zb], role, "wall"); };
      const s0 = alongX ? s[0] : s[1], s1 = alongX ? s[2] : s[3];
      piece(s0, lo, z0, z1); piece(hi, s1, z0, z1); piece(lo, hi, stopZ + S.levels.door, z1); piece(lo, hi, z0, stopZ);
    }
  };
  for (let L = 0; L < top; L++) {
    const g = new THREE.Group(); g.userData.level = L; G.lift.add(g);
    const rec = { L, g, walls: null, doors: [], button: null, btnMat: null };
    const si = stopOf(L), stopZ = si >= 0 ? Lf.stops[si].z : null;
    if (Lf.tower || levelKind(L) === "stilt") {
      const wg = new THREE.Group(); g.add(wg); rec.walls = wg;
      const bld = new Builder();
      const z0 = L === 0 ? 0 : zf(L), z1 = L + 1 <= top ? zf(L + 1) : Lf.roof;
      wallsFor(bld, z0, levelKind(L) === "stilt" && !Lf.tower ? zf(L + 1) - 0.15 : z1, stopZ);
      bld.emit(wg);
    }
    if (si >= 0) {
      const lg = lgroup(g); lg.position.y = stopZ;
      const t = k.t, w = k.w, f = -t / 2 - 0.02;
      lmesh(lg, lgeo(-w / 2 - 0.07, -w / 2, f, -t / 2 + 0.005, 0, S.levels.door + 0.07), "lift_frame");
      lmesh(lg, lgeo(w / 2, w / 2 + 0.07, f, -t / 2 + 0.005, 0, S.levels.door + 0.07), "lift_frame");
      lmesh(lg, lgeo(-w / 2 - 0.07, w / 2 + 0.07, f, -t / 2 + 0.005, S.levels.door, S.levels.door + 0.07), "lift_frame");
      for (const sgn of [-1, 1]) {
        const p = lmesh(lg, lgeo(-w / 4 - 0.01, w / 4 + 0.01, t / 2 + 0.01, t / 2 + 0.04, 0.005, S.levels.door), "lift_door");
        p.position.x = sgn * w / 4; p.userData.lift = { kind: "door", L }; lift.hits.push(p);
        rec.doors.push({ m: p, sgn, w });
      }
      const disp = new THREE.Mesh(new THREE.PlaneGeometry(0.4, 0.17), liftDisp.mat);
      disp.position.set(0, S.levels.door + 0.22, t / 2 + 0.026); lg.add(disp);
      lmesh(lg, lgeo(w / 2 + 0.17, w / 2 + 0.31, f, -t / 2 + 0.005, 0.95, 1.27), "lift_frame");
      rec.btnMat = new THREE.MeshBasicMaterial({ color: 0x6B7680 });
      const bg = new THREE.CylinderGeometry(0.03, 0.03, 0.02, 20); bg.rotateX(Math.PI / 2);
      const btn = new THREE.Mesh(bg, rec.btnMat); btn.position.set(w / 2 + 0.24, 1.11, t / 2 + 0.03); lg.add(btn);
      btn.userData.lift = { kind: "call", L }; lift.hits.push(btn); rec.button = btn;
      const hit = new THREE.Mesh(lgeo(w / 2 + 0.12, w / 2 + 0.36, f - 0.1, f, 0.85, 1.37), new THREE.MeshBasicMaterial({ visible: false }));
      hit.userData.lift = { kind: "call", L }; lg.add(hit); lift.hits.push(hit);
    }
    lift.lv[L] = rec;
  }
  // the shaft's head above the roof (the motor's overrun), with the roof
  const hb = new Builder(), roofZ = Lf.roof;
  wallsFor(hb, roofZ, Lf.head, null, "lift_tower");
  addLabel("🛗 Lift", [(Lf.clear[0] + Lf.clear[2]) / 2, (Lf.clear[1] + Lf.clear[3]) / 2, Lf.head + 0.5], "lbl",
    () => state.mode === "outside" && !state.xray && !state.explode);
  hb.box([shaftOuter[0] - 0.08, shaftOuter[1] - 0.08, shaftOuter[2] + 0.08, shaftOuter[3] + 0.08, Lf.head, Lf.head + 0.15], "slab", "wall");
  hb.emit(G.roof);

  // ── the car: floor, panelled walls, mirror, handrail, light, button panel, display, doors, sling and safety gear
  const car = lgroup(G.lift); lift.car = car;
  const { vCar, cw, cd, ch, cu, dc } = k, w2 = cw / 2;
  const fl = lmesh(car, lgeo(cu - w2, cu + w2, vCar, vCar + cd, -0.12, 0), "lift_floor", "stair");
  fl.userData.lift = { kind: "car" }; lift.hits.push(fl);
  lmesh(car, lgeo(cu - w2, cu + w2, vCar + cd, vCar + cd + 0.03, 0, ch), "lift_car", "door");
  lmesh(car, lgeo(cu - w2 - 0.03, cu - w2, vCar, vCar + cd + 0.03, 0, ch), "lift_car", "door");
  lmesh(car, lgeo(cu + w2, cu + w2 + 0.03, vCar, vCar + cd + 0.03, 0, ch), "lift_car", "door");
  if (cu - w2 < -dc / 2 - 0.005) lmesh(car, lgeo(cu - w2, -dc / 2, vCar - 0.02, vCar + 0.01, 0, ch), "lift_car", "door");
  if (cu + w2 > dc / 2 + 0.005) lmesh(car, lgeo(dc / 2, cu + w2, vCar - 0.02, vCar + 0.01, 0, ch), "lift_car", "door");
  lmesh(car, lgeo(-dc / 2, dc / 2, vCar - 0.02, vCar + 0.01, S.levels.door, ch), "lift_car", "door");
  lmesh(car, lgeo(cu - w2 - 0.03, cu + w2 + 0.03, vCar - 0.02, vCar + cd + 0.03, ch, ch + 0.06), "lift_car", "door");
  lmesh(car, lgeo(cu - 0.28, cu + 0.28, vCar + cd / 2 - 0.3, vCar + cd / 2 + 0.3, ch - 0.012, ch), "lift_light", null);
  const bulb = new THREE.PointLight(0xFFF4E0, 1.1, 3.2, 1.6);        // the car's ceiling light lights the car
  bulb.position.set(cu, ch - 0.25, -(vCar + cd / 2)); car.add(bulb); lift.bulb = bulb;
  lmesh(car, lgeo(cu - w2 + 0.08, cu + w2 - 0.08, vCar + cd - 0.006, vCar + cd, 1.0, 2.0), "mirror", null);
  lmesh(car, lgeo(cu - w2 + 0.1, cu + w2 - 0.1, vCar + cd - 0.07, vCar + cd - 0.04, 0.88, 0.92), "rail", null);
  const plate = new THREE.Mesh(new THREE.PlaneGeometry(0.2, 0.6), new THREE.MeshBasicMaterial({ map: copPlate.tex, toneMapped: false }));
  plate.rotation.y = -Math.PI / 2; plate.position.set(cu + w2 - 0.004, 1.15, -(vCar + 0.2)); car.add(plate);
  const cdisp = new THREE.Mesh(new THREE.PlaneGeometry(0.3, 0.13), liftDisp.mat);
  cdisp.rotation.y = Math.PI; cdisp.position.set(0, S.levels.door + 0.08, -(vCar + 0.012)); car.add(cdisp);
  lift.carDoors = [];
  for (const sgn of [-1, 1]) {
    const p = lmesh(car, lgeo(-dc / 4 - 0.01, dc / 4 + 0.01, vCar - 0.045, vCar - 0.025, 0.005, S.levels.door), "lift_door");
    p.position.x = sgn * dc / 4; lift.carDoors.push({ m: p, sgn, w: dc });
  }
  lmesh(car, lgeo(cu - w2 - 0.07, cu + w2 + 0.07, vCar + cd / 2 - 0.05, vCar + cd / 2 + 0.05, ch + 0.06, ch + 0.22), "machine", "door");
  for (const sgn of [-1, 1]) {
    lmesh(car, lgeo(cu + sgn * (w2 + 0.035) - 0.025, cu + sgn * (w2 + 0.035) + 0.025, vCar + cd / 2 - 0.04, vCar + cd / 2 + 0.04, -0.2, ch + 0.22), "machine", null);
    lmesh(car, lgeo(cu + sgn * (w2 + 0.035) - 0.04, cu + sgn * (w2 + 0.035) + 0.04, vCar + cd / 2 - 0.06, vCar + cd / 2 + 0.06, -0.2, -0.08), "buffer", null);
  }

  // ── how it works (x-ray): rails, counterweight, ropes, motor and pulley, pit and buffers, the shaft's outline
  const x = new THREE.Group(); G.lift.add(x); lift.x = x;
  const xl = lgroup(x), zb = k.zMin - Lf.pit, zt = Lf.head;
  for (const sgn of [-1, 1]) {
    const ru = cu + sgn * (w2 + 0.085);
    lmesh(xl, lgeo(ru - 0.02, ru + 0.02, vCar + cd / 2 - 0.03, vCar + cd / 2 + 0.03, zb, zt - 0.3), "rail", null);
    const cu0 = k.rear ? (sgn < 0 ? k.cwU[0] - 0.04 : k.cwU[1] + 0.01) : (k.cwU[0] + k.cwU[1]) / 2 - 0.015;
    const cv0 = k.rear ? (k.cwV[0] + k.cwV[1]) / 2 - 0.015 : (sgn < 0 ? k.cwV[0] - 0.04 : k.cwV[1] + 0.01);
    lmesh(xl, lgeo(cu0, cu0 + 0.03, cv0, cv0 + 0.03, zb, zt - 0.3), "rail", null);
  }
  lmesh(xl, lgeo(k.uMin, k.uMax, k.t / 2, k.vMax, zb - 0.1, zb), "concrete", "site");
  lmesh(xl, lgeo(cu - 0.05, cu + 0.05, vCar + cd / 2 - 0.05, vCar + cd / 2 + 0.05, zb, zb + 0.45), "buffer", null);
  const cwc = [(k.cwU[0] + k.cwU[1]) / 2, (k.cwV[0] + k.cwV[1]) / 2];
  lmesh(xl, lgeo(cwc[0] - 0.05, cwc[0] + 0.05, cwc[1] - 0.05, cwc[1] + 0.05, zb, zb + 0.35), "buffer", null);
  const mu = (cu + cwc[0]) / 2, mv = (vCar + cd / 2 + cwc[1]) / 2;
  lmesh(xl, lgeo(mu - 0.28, mu + 0.28, mv - 0.17, mv + 0.17, zt - 0.62, zt - 0.3), "machine", "door");
  const sh = new THREE.CylinderGeometry(0.17, 0.17, 0.09, 24); sh.rotateZ(Math.PI / 2);
  const sheave = new THREE.Mesh(sh, mat("steel")); sheave.userData.role = "steel"; registry.meshes.push(sheave);
  const pv = new THREE.Group(), side = Math.abs(cwc[0] - cu) > Math.abs(cwc[1] - (vCar + cd / 2));
  if (side) pv.rotation.y = Math.PI / 2;                 // the pulley turns in the plane of the ropes
  pv.position.set(mu, zt - 0.8, -mv); pv.add(sheave); xl.add(pv); lift.sheave = sheave;
  const cwm = lmesh(xl, lgeo(k.cwU[0] - (k.cwU[0] + k.cwU[1]) / 2, k.cwU[1] - (k.cwU[0] + k.cwU[1]) / 2,
    k.cwV[0] - cwc[1], k.cwV[1] - cwc[1], 0, 1.3), "cwt", "door");
  cwm.position.x = cwc[0]; cwm.position.z = -cwc[1]; lift.cw = cwm;
  lift.ropeMat = new THREE.LineBasicMaterial({ color: 0x00E5FF });
  const rg = new THREE.BufferGeometry(); rg.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(8 * 3), 3));
  lift.ropes = new THREE.LineSegments(rg, lift.ropeMat); lift.ropes.frustumCulled = false; xl.add(lift.ropes);
  lift.xl = xl; lift.geo = { mu, mv, cwc };
  const ol = new Lines(), c = Lf.clear;
  ol.box([c[0], c[1], c[2], c[3], zb, zt]);
  const ob = ol.build("ghost"); x.add(ob);

  // ── labels for the x-ray view (they follow the moving parts)
  const lab = (key, html) => { addLabel(html, [0, 0, 0], "lbl xr", () => state.xray, { xr: key }); lift.labels[key] = registry.labels[registry.labels.length - 1]; };
  lab("motor", "⚙️ Motor + brake<small>no machine room needed</small>");
  lab("cw", "⚖️ Counterweight<small>car + half the load</small>");
  lab("rope", "🔗 Steel ropes");
  lab("rail", "🛤️ Guide rails<small>safety gear grips them</small>");
  lab("car", "🛗 Car");
  lab("pit", `🛡️ Buffers<small>pit ${Lf.pit.toFixed(1)} m deep</small>`);
  const place = (key, u, v, z) => { const p = liftPt(u, v), l = lift.labels[key]; l.pos.set(p[0], z, -p[1]); l.plan = [p[0], p[1], z]; };
  place("motor", mu, mv, zt - 0.2); place("rail", cu - w2 - 0.085, vCar + cd / 2, (k.zMin + k.zMax) / 2 + 1.2);
  place("pit", cu, vCar + cd / 2, zb + 0.2);
  lift.place = place;
  lift.z = k.zMin; lift.at = 0; lift.door = 0; lift.phase = "idle";
  liftPose(true);
  buildCopPanel();
}

function liftPose(force = false) {
  if (!lift.ok) return;
  const k = lift.k, ex = state.explode && state.mode !== "walk" && !state.xray;
  const zOff = ex ? liftLevelF(lift.z) * GAP : 0;
  lift.car.position.y = lift.z + zOff;
  const e = ease(lift.door), stopHere = lift.phase !== "moving" ? lift.at : -1;
  for (const d of lift.carDoors) d.m.position.x = d.sgn * (d.w / 4 + d.w / 2 * e);
  const openL = stopHere >= 0 ? S.lift.stops[stopHere].L : -1;
  for (const rec of lift.lv) {
    if (!rec || !rec.doors.length) continue;
    const oe = rec.L === openL ? e : 0;
    if (!force && rec.L !== openL && rec.L !== lift.open) continue;
    for (const d of rec.doors) d.m.position.x = d.sgn * (d.w / 4 + d.w / 2 * oe);
  }
  lift.open = openL;
  // the display on every floor and in the car: where the car is, which way it goes
  const st = S.lift.stops, near = st.reduce((b, s, i) => (Math.abs(s.z - lift.z) < Math.abs(st[b].z - lift.z) ? i : b), 0);
  const dir = lift.phase === "moving" ? Math.sign(st[lift.target].z - lift.z) : 0;
  const key = `${st[near].label}|${dir}`;
  if (key !== lift.disp) { lift.disp = key; drawDisplay(st[near].label, dir); lift.cop = ""; syncCop(); }
  if ($("cop").style.display === "block") $("copInfo").textContent = `${lift.v.toFixed(2)} m/s · ${lift.z.toFixed(2)} m above the ground`;
  if (lift.bulb) lift.bulb.intensity = lift.rider || lift.door > 0.01 || state.xray ? 1.1 : 0;   // (no shader rebuilds)
  const ck = [...lift.queue].sort().join(",") + "|" + key;
  if (ck !== lift.cop) { lift.cop = ck; drawPlate(); syncCop(); }
  for (const rec of lift.lv) if (rec && rec.btnMat) {
    const on = lift.queue.has(stopOf(rec.L)) || (lift.auto && lift.auto.stage === "call" && stopOf(rec.L) === stopOf(walker.level));
    rec.btnMat.color.set(on ? 0xFFB000 : (state.theme === "bp" ? 0x2E7FB8 : 0x6B7680));
  }
  if (state.xray || force) {             // counterweight goes down as the car goes up; ropes over the pulley
    const zb = k.zMin - S.lift.pit, zt = S.lift.head;
    const cwz = clamp(k.zMin + 0.35 + (k.zMax - lift.z), zb + 0.4, zt - 2.4);
    lift.cw.position.y = cwz;
    const { mu, mv, cwc } = lift.geo, top = lift.z + k.ch + 0.22, sz = zt - 0.8 + 0.17;
    const P = lift.ropes.geometry.attributes.position.array;
    const pts = [[k.cu - 0.04, k.vCar + k.cd / 2, top], [mu - 0.04, mv, sz], [k.cu + 0.04, k.vCar + k.cd / 2, top], [mu + 0.04, mv, sz],
      [cwc[0] - 0.04, cwc[1], cwz + 1.3], [mu - 0.04, mv, sz], [cwc[0] + 0.04, cwc[1], cwz + 1.3], [mu + 0.04, mv, sz]];
    pts.forEach((p, i) => { P[i * 3] = p[0]; P[i * 3 + 1] = p[2]; P[i * 3 + 2] = -p[1]; });
    lift.ropes.geometry.attributes.position.needsUpdate = true; lift.ropes.geometry.computeBoundingSphere();
    lift.ropeMat.color.set(state.theme === "bp" ? 0x00E5FF : 0x2B2F33);
    lift.sheave.rotation.x -= (lift.v || 0) * 0.1;
    if (lift.place) {
      lift.place("car", k.cu, k.vCar + k.cd / 2, lift.z + k.ch + 0.5);
      lift.place("cw", cwc[0], cwc[1], cwz + 1.6);
      lift.place("rope", (k.cu + mu) / 2, (k.vCar + k.cd / 2 + mv) / 2, (top + sz) / 2);
      const cl = lift.labels.car, txt = `🛗 Car ${k.cw.toFixed(2)} × ${k.cd.toFixed(2)} m<small>${lift.v.toFixed(2)} m/s · ${(lift.z).toFixed(1)} m up</small>`;
      if (cl.el.innerHTML !== txt) { cl.el.innerHTML = txt; cl.size = null; }
    }
  }
}

// ── sounds (made here, no files): arrival chime, motor hum, alarm bell
let actx = null, hum = null;
function audio() {
  if (!state.sound) return null;
  try { actx = actx || new (window.AudioContext || window.webkitAudioContext)(); if (actx.state === "suspended") actx.resume(); } catch (e) { actx = null; }
  return actx;
}
function tone(f, t0, dur, gain = 0.05, type = "sine") {
  const a = audio(); if (!a) return;
  const o = a.createOscillator(), g = a.createGain(), t = a.currentTime + t0;
  o.type = type; o.frequency.value = f; g.gain.setValueAtTime(0.0001, t); g.gain.linearRampToValueAtTime(gain, t + 0.012);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur); o.connect(g).connect(a.destination); o.start(t); o.stop(t + dur + 0.05);
}
const liftHeard = () => lift.rider || state.xray || (state.mode === "walk" && Math.hypot(walker.x - S.lift.door.mid[0], walker.y - S.lift.door.mid[1]) < 7);
function chime() { if (liftHeard()) { tone(1318.5, 0, 0.9); tone(1046.5, 0.3, 1.2); } }
function bell() { for (let i = 0; i < 8; i++) tone(1046.5, i * 0.11, 0.09, 0.04, "square"); }
function humOn(on) {
  if (on && !hum && liftHeard()) {
    const a = audio(); if (!a) return;
    const o = a.createOscillator(), f = a.createBiquadFilter(), g = a.createGain();
    o.type = "sawtooth"; o.frequency.value = 55; f.type = "lowpass"; f.frequency.value = 170;
    g.gain.setValueAtTime(0.0001, a.currentTime); g.gain.linearRampToValueAtTime(0.022, a.currentTime + 0.8);
    o.connect(f).connect(g).connect(a.destination); o.start(); hum = { o, g, a };
  } else if (!on && hum) {
    const h = hum; hum = null; h.g.gain.linearRampToValueAtTime(0.0001, h.a.currentTime + 0.5); h.o.stop(h.a.currentTime + 0.6);
  }
}

// ── the controller: calls from the landings (hall) and from the car; collective — keeps going the way it goes
function liftCall(i, from = "hall") {
  if (!lift.ok || i < 0 || i >= S.lift.stops.length) return;
  audio();
  if (i === lift.at && lift.phase !== "moving") {
    if (lift.phase === "closing" || lift.phase === "idle") lift.phase = "opening";
    lift.hold = Math.max(lift.hold, lift.rider ? 30 : DWELL_S);
  } else lift.queue.add(i);
  if (from === "car" && lift.phase === "open") lift.hold = Math.min(lift.hold, 0.9);
  poke(600);
}
function liftNext() {
  const q = [...lift.queue]; if (!q.length) return null;
  const ahead = q.filter((i) => (lift.dir > 0 ? i > lift.at : lift.dir < 0 ? i < lift.at : true));
  const pool = ahead.length ? ahead : q;
  return pool.reduce((b, i) => (Math.abs(i - lift.at) < Math.abs(b - lift.at) ? i : b), pool[0]);
}
function liftArrived() {
  const s = S.lift.stops[lift.at];
  chime();
  if (lift.rider) {
    walker.level = s.L; applyLevelView();
    if (state.voice && liftHeard()) speak(s.name);
    toast(`🛗 ${s.name}`, 1800);
  }
  hudRoom = undefined;
}
function liftUpdate(dt) {
  if (!lift.ok) return false;
  const st = S.lift.stops; let active = false;
  if (lift.phase === "idle" && lift.queue.size) {
    const n = liftNext();
    if (n === lift.at) { lift.queue.delete(n); lift.phase = "opening"; }
    else { lift.target = n; lift.dir = Math.sign(n - lift.at); lift.phase = "moving"; humOn(true); }
  }
  if (lift.phase === "moving") {
    const a = S.lift.acc * (lift.fast ? 3 : 1), vmax = S.lift.speed * (lift.fast ? 3 : 1);
    const s = Math.sign(st[lift.target].z - lift.z) || 1;
    for (const i of lift.queue) {           // a call ahead that the car can still stop for comes first
      const ahead = (st[i].z - lift.z) * s;
      if (ahead > 0.02 && ahead < (st[lift.target].z - lift.z) * s && ahead >= lift.v * lift.v / (2 * a) + 0.05) lift.target = i;
    }
    const d = (st[lift.target].z - lift.z) * s;
    lift.v = Math.min(vmax, Math.sqrt(2 * a * Math.max(0, d)), lift.v + a * dt);
    const step = Math.max(lift.v, 0.02) * dt;
    if (d <= step + 1e-4) {
      lift.z = st[lift.target].z; lift.v = 0; lift.at = lift.target; lift.queue.delete(lift.at);
      lift.phase = "opening"; humOn(false); liftArrived();
    } else lift.z += s * step;
    // which floor it is passing (for the dir of the next choice)
    const passed = st.reduce((b, q, i) => ((q.z - lift.z) * s <= 0.001 ? i : b), lift.at);
    if (s > 0) lift.at = Math.max(lift.at, passed); else lift.at = Math.min(lift.at, st.findIndex((q) => q.z >= lift.z - 0.001));
    active = true;
  } else if (lift.phase === "opening") {
    lift.door = Math.min(1, lift.door + dt / DOOR_OPEN_S);
    if (lift.door >= 1) { lift.phase = "open"; lift.hold = Math.max(lift.hold, lift.rider ? 30 : DWELL_S); }
    active = true;
  } else if (lift.phase === "open") {
    if (!lift.anim) {
      if (lift.rider && lift.queue.size) lift.hold = Math.min(lift.hold, 1.0);
      if (!(lift.rider && !lift.queue.size)) lift.hold -= dt;
      if (lift.hold <= 0) lift.phase = "closing";
    }
  } else if (lift.phase === "closing") {
    lift.door = Math.max(0, lift.door - dt / DOOR_CLOSE_S);
    if (lift.door <= 0) { lift.phase = "idle"; lift.hold = 0; if (!lift.queue.size) lift.dir = 0; }
    active = true;
  }
  if (active || state.xray) liftPose();
  return active;
}

// ── getting in and out
const angDiff = (a, b) => ((a - b + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
function liftFront() { return liftPt(0, -(S.lift.door.t / 2 + 0.75)); }
function canBoard() {
  return lift.ok && !lift.rider && !lift.anim && state.mode === "walk" && lift.at === stopOf(walker.level) &&
    (lift.phase === "open" || (lift.phase === "opening" && lift.door > 0.55));
}
function boardLift() {
  if (!canBoard()) return;
  const k = lift.k, D = S.lift.door;
  lift.anim = { kind: "in", t: 0, from: [walker.x, walker.y], to: liftPt(k.cu, k.vCar + k.cd * 0.55), yaw0: walker.yaw,
    yaw1: Math.atan2(-D.n[1], -D.n[0]) };
  lift.hold = 30; walker.path = []; if (!(lift.auto && lift.auto.tour)) tourInterrupt(); hudRoom = undefined;
}
function stepOut() {
  if (!lift.rider || lift.anim) return;
  if (lift.phase === "moving") { toast("The lift is moving — wait for the doors to open."); return; }
  if (lift.phase !== "open") { liftCall(lift.at, "car"); toast("Opening the doors…", 1500); return; }
  const out = liftFront();
  lift.anim = { kind: "out", t: 0, from: [walker.x, walker.y], to: out, yaw0: walker.yaw, yaw1: walker.yaw };
}
function riderUpdate(dt) {
  const A = lift.anim;
  if (A) {
    A.t += dt / 0.9; const k = ease(Math.min(1, A.t));
    walker.x = lerp(A.from[0], A.to[0], k); walker.y = lerp(A.from[1], A.to[1], k);
    walker.yaw = A.yaw0 + angDiff(A.yaw1, A.yaw0) * k; walker.pitch = lerp(walker.pitch, -0.04, Math.min(1, dt * 3));
    if (A.t >= 1) {
      lift.anim = null;
      if (A.kind === "in") { lift.rider = true; showCop(true); toast("Press a floor on the panel →", 2600); }
      else { lift.rider = false; walker.level = S.lift.stops[lift.at].L; lift.hold = 2.5; showCop(state.xray); applyLevelView(); }
      hudRoom = undefined;
    }
  } else if (lift.rider && lift.phase === "open" && (keys.has("w") || keys.has("arrowup") || keys.has("pad-fwd"))) stepOut();
  walker.h = lift.z + walker.eye;
  camera.position.set(walker.x, walker.h, -walker.y);
  camera.lookAt(camera.position.clone().add(lookDir()));
}
function liftTrip(L, forTour = false) {   // the floor bar / HUD / tour: walk to the lift, call, ride, step out
  const to = stopOf(L), from = stopOf(walker.level);
  if (to < 0 || from < 0) return false;
  if (!forTour) tourInterrupt();
  lift.auto = { to, stage: "walk", tour: forTour };
  const f = liftFront(), c = nearestFree(f[0], f[1], 1.2);
  if (c && Math.hypot(walker.x - f[0], walker.y - f[1]) > 0.6) {
    const p = centreOf(c), path = astar(walker.x, walker.y, p[0], p[1], walker.level);
    if (!path) { lift.auto = null; return false; }
    walker.path = path.slice(1);
  }
  toast(`🛗 Taking the lift to ${levelLabel(L)}…`, 2200);
  return true;
}
function liftAuto() {
  const A = lift.auto; if (!A) return;
  if (A.stage === "walk" && !walker.path.length) {
    const D = S.lift.door; walker.yaw = Math.atan2(D.n[1], D.n[0]);
    liftCall(stopOf(walker.level), "hall"); A.stage = "call";
  } else if (A.stage === "call" && canBoard()) { boardLift(); A.stage = "board"; }
  else if (A.stage === "board" && lift.rider && !lift.anim) { liftCall(A.to, "car"); A.stage = "ride"; }
  else if (A.stage === "ride" && lift.rider && lift.at === A.to && lift.phase === "open" && !lift.anim) { stepOut(); lift.auto = null; }
}

// ── the button panel on screen (in the car, or to drive the car in the x-ray view)
function buildCopPanel() {
  const el = $("copBtns"); if (!el) return;
  el.innerHTML = "";
  const st = S.lift.stops;
  for (let i = st.length - 1; i >= 0; i--) {
    const b = document.createElement("button"); b.className = "fb"; b.textContent = st[i].label; b.title = st[i].name; b.dataset.i = i;
    b.onclick = () => { liftCall(i, lift.rider ? "car" : "remote"); syncCop(); };
    el.appendChild(b);
  }
  const Lf = S.lift, how = $("copHow");
  if (how) how.innerHTML =
    `<b>How your lift works</b> · ${Lf.persons} persons · ${Lf.load} kg · ${Lf.speed.toFixed(1)} m/s · ${st.length} stops<br>` +
    "⚙️ A small motor at the top turns a grooved pulley — no machine room.<br>" +
    "🔗 Steel ropes carry the car on one side and ⚖️ a counterweight (the car + half the load) on the other, " +
    "so the motor only lifts the difference.<br>" +
    "🛤️ Rails guide both; if the car ever runs too fast, the safety gear grips the rails. 🛡️ Buffers wait in the pit.<br>" +
    "🚪 The car door drives the landing door — a landing door opens only when the car is there.<br>" +
    `♿ ${esc(Lf.words)}`;
  syncCop();
}
function syncCop() {
  if (!lift.ok) return;
  const [lab, dir] = (lift.disp || "|0").split("|");
  const d = $("copDisp"); if (d) d.textContent = (+dir > 0 ? "▲ " : +dir < 0 ? "▼ " : "") + lab;
  for (const b of $("copBtns").children) b.classList.toggle("lit", lift.queue.has(+b.dataset.i));
  const info = $("copInfo");
  if (info) info.textContent = `${lift.v.toFixed(2)} m/s · ${lift.z.toFixed(2)} m above the ground`;
  $("copOut").style.display = lift.rider ? "" : "none";
  $("copHow").style.display = state.xray ? "" : "none";
  $("copFast").classList.toggle("on", lift.fast);
  $("copSnd").textContent = state.sound ? "🔊" : "🔇";
}
function showCop(on) { $("cop").style.display = on && lift.ok ? "block" : "none"; syncCop(); }
document.querySelectorAll("#cop [data-op]").forEach((b) => b.addEventListener("click", () => {
  const op = b.dataset.op;
  if (op === "open" && lift.phase !== "moving") liftCall(lift.at, "car");
  if (op === "close" && lift.phase === "open" && !lift.anim) lift.hold = 0;
  if (op === "bell") { bell(); toast("🔔 Alarm — in a real lift this rings the building's alarm and the lift company's 24-hour line.", 5000); }
  poke();
}));
$("copOut").addEventListener("click", stepOut);
$("copFast").addEventListener("click", () => { lift.fast = !lift.fast; syncCop(); });
$("copSnd").addEventListener("click", () => { state.sound = !state.sound; if (!state.sound) humOn(false); syncCop(); });
$("copClose").addEventListener("click", () => { if (state.xray) setXray(false); else showCop(false); });

// ── x-ray: the building fades to outlines and the lift's machinery shows, with what each part does
function setXray(on) {
  if (!S || !lift.ok) return;
  state.xray = on; $("btnLift").classList.toggle("on", on);
  hideStart(); tourInterrupt();
  if (on) {
    state.explode = false; $("btnExplode").classList.remove("on");
    if (state.mode === "walk" || state.mode === "top") setMode("dollhouse", { instant: true });
    const c = S.lift.clear, D = S.lift.door, zb = lift.k.zMin - S.lift.pit, zt = S.lift.head;
    orbit.target.set((c[0] + c[2]) / 2, (zb + zt) / 2, -(c[1] + c[3]) / 2);
    const ang = Math.atan2(-D.n[1], -D.n[0]) + 0.6, dx = Math.cos(ang), dy = Math.sin(ang);
    orbit.theta = Math.atan2(dx, -dy); orbit.phi = 1.3; orbit.r = fitRadius((zt - zb) * 0.95);
    flyTo(orbitPos(), orbit.target.clone(), 1.0);
    toast("🛗 How the lift works — press a floor on the panel and watch the car and the counterweight.", 4200);
  }
  showCop(on || lift.rider);
  liftPose(true); applyLevelView(); poke(1500);
}

// ── ghosts: the floors above the one in view (and every floor in the x-ray) as outlines — the building stays whole
function buildGhosts() {
  const o = S.shell.outer, top = TOP();
  for (let L = 0; L <= top; L++) {
    const l = new Lines(), z = zf(L), z1 = L === top ? z + S.levels.parapet : z + S.levels.ceiling;
    l.box([o[0], o[1], o[2], o[3], z - (L ? 0.15 : 0), z1]);
    const ls = l.build("ghost"); ls.userData.level = L;
    const pl = new THREE.Mesh(new THREE.PlaneGeometry(o[2] - o[0], o[3] - o[1]),
      new THREE.MeshBasicMaterial({ color: 0x4FA3D9, transparent: true, opacity: 0.06, depthWrite: false, side: THREE.DoubleSide }));
    pl.rotation.x = -Math.PI / 2; pl.position.set((o[0] + o[2]) / 2, z, -(o[1] + o[3]) / 2); pl.userData.level = L;
    const g = new THREE.Group(); g.userData.level = L; g.add(ls, pl); G.ghost.add(g);
  }
}

$("btnExplode").addEventListener("click", () => {
  if (state.xray) { state.xray = false; $("btnLift").classList.remove("on"); showCop(lift.rider); }
  state.explode = !state.explode; $("btnExplode").classList.toggle("on", state.explode);
  hideStart(); tourInterrupt();
  setMode(state.mode === "walk" ? "outside" : state.mode);
});

function openLiftMove() {
  const cur = S.lift.position;
  for (const b of document.querySelectorAll("#liftMove [data-pos]")) b.classList.toggle("on", b.dataset.pos === cur);
  $("liftMove").style.display = "block";
}
document.querySelectorAll("#liftMove [data-pos]").forEach((b) => b.addEventListener("click", () => {
  $("liftMove").style.display = "none";
  if (b.dataset.pos === S.lift.position) { toast("The lift is already there."); return; }
  ST.value({ id: Date.now(), action: "lift", position: b.dataset.pos, sid: SID });
  toast(`🛗 Re-making your design with the lift ${b.dataset.pos === "inside" ? "inside the home" : b.dataset.pos === "entrance" ? "beside the main door" : "at the side"}…`, 6000);
}));
$("liftMoveClose").addEventListener("click", () => { $("liftMove").style.display = "none"; });

let wishRoom = null;
function openWish(r) { wishRoom = r; $("wishT").textContent = `Change the ${r.name}`; $("wish").style.display = "block"; }
function closeWish() { $("wish").style.display = "none"; wishRoom = null; }
$("wishClose").addEventListener("click", closeWish);
document.querySelectorAll("#wish [data-w]").forEach((b) => b.addEventListener("click", () => {
  if (!wishRoom) return;
  ST.value({ id: Date.now(), action: "wish", kind: b.dataset.w, room: wishRoom.key, name: wishRoom.name, sid: SID });
  toast(`✅ Wish sent: ${b.textContent.trim()} — ${wishRoom.name}. Scroll up to 🧠 Smart Re-Plan to see your 3 new designs.`, 6500);
  closeWish();
}));
let toastT = 0;
function toast(t, ms = 3800) { $("toast").textContent = t; $("toast").style.display = "block"; clearTimeout(toastT); toastT = setTimeout(() => { $("toast").style.display = "none"; }, ms); }

// ───────────────────────────── minimap
const mini = $("miniC"), mg = mini.getContext("2d");
const miniBase = document.createElement("canvas");     // the plan drawn once; each frame only adds 'you are here'
let mm = { s: 1, pad: 6, base: false };
function drawMinimapBase(p = 0) {
  if (!S) return;
  const P = PL(p);
  const W = S.plot.w, H = S.plot.h, max = window.innerWidth < 640 ? 130 : 190, s = max / Math.max(W, H);
  mini.width = miniBase.width = Math.round(W * s + 12); mini.height = miniBase.height = Math.round(H * s + 12); mm = { s, pad: 6, plan: p };
  const X = (x) => mm.pad + x * s, Y = (y) => mm.pad + (H - y) * s;
  mg.fillStyle = "#06182E"; mg.fillRect(0, 0, mini.width, mini.height);
  mg.strokeStyle = "#3E6E9C"; mg.setLineDash([3, 3]); mg.strokeRect(X(0), Y(H), W * s, H * s); mg.setLineDash([]);
  const col = { LIVING: "#1E5A92", DINING: "#1E5A92", BEDROOM: "#2C4F86", KITCHEN: "#6B4E1E", BATHROOM: "#1D5E66", POOJA: "#7A5B12", STAIRS: "#3A3A5A" };
  for (const r of P.rooms) { const [x0, y0, x1, y1] = r.r; mg.fillStyle = col[r.cat] || "#203C5E"; mg.globalAlpha = 0.75; mg.fillRect(X(x0), Y(y1), (x1 - x0) * s, (y1 - y0) * s); }
  mg.globalAlpha = 1;
  for (const st of S.site) { const [x0, y0, x1, y1] = st.r; mg.strokeStyle = st.cat === "GARDEN" ? "#22C55E" : st.cat === "POOL" ? "#38BDF8" : "#64748B"; mg.strokeRect(X(x0), Y(y1), (x1 - x0) * s, (y1 - y0) * s); }
  mg.fillStyle = "#00E5FF";
  for (const k of ["ext", "int", "post", "cmp"]) for (const b of (k === "cmp" ? S.walls[k] : P.walls[k]) || []) { if (k !== "cmp" && b[4] > S.levels.ffl + 0.5) continue; mg.fillRect(X(b[0]), Y(b[3]), Math.max(1, (b[2] - b[0]) * s), Math.max(1, (b[3] - b[1]) * s)); }
  mg.fillStyle = "#00FF88";
  for (const d of P.doors) { mg.beginPath(); mg.arc(X(d.mid[0]), Y(d.mid[1]), 2.2, 0, 7); mg.fill(); }
  // north
  const cx = mini.width - 16, cy = 16, n = S.north;
  mg.strokeStyle = "#fff"; mg.fillStyle = "#fff"; mg.lineWidth = 2; mg.beginPath(); mg.moveTo(cx - n[0] * 9, cy + n[1] * 9); mg.lineTo(cx + n[0] * 9, cy - n[1] * 9); mg.stroke();
  mg.font = "bold 10px sans-serif"; mg.textAlign = "center"; mg.fillText("N", cx + n[0] * 13, cy - n[1] * 13 + 3); mg.lineWidth = 1;
  miniBase.getContext("2d").drawImage(mini, 0, 0); mm.base = true;
}
function drawMinimap() {
  if (!S || !mm.base) return;
  mg.drawImage(miniBase, 0, 0);
  const H = S.plot.h, s = mm.s, X = (x) => mm.pad + x * s, Y = (y) => mm.pad + (H - y) * s;
  let px, py, yaw;
  if (state.mode === "walk") { px = walker.x; py = walker.y; yaw = walker.yaw; }
  else { px = camera.position.x; py = -camera.position.z; const d = new THREE.Vector3(); camera.getWorldDirection(d); yaw = Math.atan2(-d.z, d.x); }
  const x = X(px), y = Y(py);
  mg.fillStyle = "rgba(255,214,102,0.28)"; mg.beginPath(); mg.moveTo(x, y);
  mg.arc(x, y, 34, -yaw - 0.5, -yaw + 0.5); mg.closePath(); mg.fill();
  mg.fillStyle = "#FFD166"; mg.beginPath(); mg.arc(x, y, 4.5, 0, 7); mg.fill(); mg.strokeStyle = "#000"; mg.stroke();
}
mini.addEventListener("click", (e) => {
  if (!S) return;
  const r = mini.getBoundingClientRect(), x = (e.clientX - r.left - mm.pad) / mm.s, y = S.plot.h - (e.clientY - r.top - mm.pad) / mm.s;
  hideStart(); tourInterrupt();
  if (state.mode === "walk") walkTo(x, y);
  else { const c = nearestFree(x, y, 1.5); if (c) setMode("walk", { at: centreOf(c), yaw: walker.yaw }); }
});

// ───────────────────────────── frame loop
let lastT = performance.now();
let visible = true;
try { new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) poke(); }).observe($("app")); } catch (e) { /* old browser */ }
function resize() {
  const w = $("app").clientWidth, h = $("app").clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix(); poke();
}
new ResizeObserver(resize).observe($("app"));

function update(dt) {
  let active = false;
  if (trans.on) {
    trans.t += dt / trans.dur; const k = ease(Math.min(1, trans.t));
    camera.position.lerpVectors(trans.fromP, trans.toP, k);
    const tgt = new THREE.Vector3().lerpVectors(trans.fromT, trans.toT, k); camera.lookAt(tgt);
    if (trans.t >= 1) { trans.on = false; if (trans.done) trans.done(); }
    active = true;
  } else if (state.mode === "walk" && (lift.rider || lift.anim)) {
    riderUpdate(dt); liftAuto(); updateHud();
    active = true;
  } else if (state.mode === "walk") {
    const run = keys.has("shift") ? 2.4 : 1.4;
    let f = 0, s = 0, turn = 0;
    if (keys.has("w") || keys.has("arrowup") || keys.has("pad-fwd")) f += 1;
    if (keys.has("s") || keys.has("arrowdown") || keys.has("pad-back")) f -= 1;
    if (keys.has("a")) s -= 1;
    if (keys.has("d")) s += 1;
    if (keys.has("arrowleft") || keys.has("pad-left")) turn += 1;
    if (keys.has("arrowright") || keys.has("pad-right")) turn -= 1;
    if (f || s) { stepWalker(f * run * dt, s * run * 0.8 * dt); walker.path = []; active = true; }
    if (turn) { walker.yaw += turn * 1.7 * dt; active = true; }
    if (walker.path.length) {
      const [tx, ty] = walker.path[0], dx = tx - walker.x, dy = ty - walker.y, d = Math.hypot(dx, dy), step = 1.25 * dt;
      if (d <= step) { walker.x = tx; walker.y = ty; walker.path.shift(); }
      else { walker.x += dx / d * step; walker.y += dy / d * step; }
      updateLevel();
      if (performance.now() - walker.lastLook > 1500 && d > 0.05) {
        const want = Math.atan2(dy, dx), diff = ((want - walker.yaw + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
        walker.yaw += clamp(diff, -2.6 * dt, 2.6 * dt);
        walker.pitch = lerp(walker.pitch, -0.06, Math.min(1, dt * 2));
      }
      active = true;
    }
    tourUpdate(dt);
    if (tour.on) active = true;
    if (lift.ok) {
      liftAuto();
      if (canBoard() && !lift.auto && lift.prev) {               // walking into the open lift steps you in
        const D = S.lift.door, dx = walker.x - D.mid[0], dy = walker.y - D.mid[1];
        const along = dx * D.n[0] + dy * D.n[1], across = Math.abs(dx * D.n[1] - dy * D.n[0]);
        const mv = (walker.x - lift.prev[0]) * D.n[0] + (walker.y - lift.prev[1]) * D.n[1];
        if (along > -(D.t / 2 + 0.5) && across < D.w / 2 + 0.1 && mv > 0.0005) boardLift();
      }
      lift.prev = [walker.x, walker.y];
    }
    const gz = groundAt(walker.x, walker.y) + walker.eye;
    if (Math.abs(gz - walker.h) > 0.002) { walker.h += (gz - walker.h) * Math.min(1, dt * 7); active = true; }
    camera.position.set(walker.x, walker.h, -walker.y);
    camera.lookAt(camera.position.clone().add(lookDir()));
    updateHud();
  } else {
    if (orbit.auto) { orbit.theta += dt * 0.08; active = true; }
    camera.position.copy(orbitPos()); camera.lookAt(orbit.target);
  }
  if (liftUpdate(dt)) active = true;                  // the lift runs in every view
  // doors open as you come near (always open in the roof-off views), gates likewise
  for (const o of registry.doors) {
    if (o.still) {                                   // (a lift without scene.lift) the door slides away near it
      const open = state.mode === "walk" && Math.hypot(walker.x - o.d.mid[0], walker.y - o.d.mid[1]) < 1.2;
      if (o.pivot.visible === open) { o.pivot.visible = !open; active = true; }
      continue;
    }
    const near = hasRooms(walker.level) && planOf(walker.level) === (o.plan || 0) &&
      Math.hypot(walker.x - o.d.mid[0], walker.y - o.d.mid[1]) < 1.7;
    o.target = state.mode === "walk" ? (near ? 1 : 0) : (state.mode === "outside" ? 0 : 1);
    if (Math.abs(o.target - o.cur) > 1e-3) { o.cur += clamp(o.target - o.cur, -dt * 1.6, dt * 1.6); o.pivot.rotation.y = o.d.closed * DEG + o.delta * ease(o.cur); active = true; }
  }
  if (driveUpdate(dt)) active = true;
  for (const o of registry.gates) {
    const near = state.mode === "walk" && Math.hypot(walker.x - o.mid[0], walker.y - o.mid[1]) < 3.2;
    const car = drive.on && drive.at && Math.hypot(drive.at[0] - o.gmid[0], drive.at[1] - o.gmid[1]) < 8.5 &&
      S.gates[S.drive.gate] === o.g;
    o.target = near || tour.on || car ? 1 : 0;
    if (Math.abs(o.target - o.cur) > 1e-3) { o.cur += clamp(o.target - o.cur, -dt * 0.7, dt * 0.7); const e = ease(o.cur); for (const f of o.parts) f(e); active = true; }
  }
  if (state.mode === "walk" && registry.fans.length) { for (const f of registry.fans) f.rotation.y += dt * 4.2; active = true; }
  sky.position.copy(camera.position);
  return active;
}
function frame() {
  requestAnimationFrame(frame);
  const now = performance.now(), dt = Math.min(0.05, (now - lastT) / 1000); lastT = now;
  if (!S || !visible) return;
  const active = update(dt);
  if (active || performance.now() < dirtyUntil) {
    renderer.render(scene, camera);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    updateLabels(w, h); drawMinimap();
  }
}

// ───────────────────────────── load
function load(data, id) {
  const keepWalk = state.mode === "walk" && S;
  const at = keepWalk ? [walker.x, walker.y] : null, yaw = walker.yaw;
  build(data); SID = id;
  if (!state.started) { setMode("outside", { instant: true }); orbit.auto = true; }
  else if (keepWalk) {
    const c = nearestFree(at[0], at[1], 2.0);
    setMode("walk", { at: c ? centreOf(c) : undefined, yaw });
    toast("The 3D house now shows your new design.");
  } else setMode(state.mode, { instant: true });
}
window.addEventListener("message", (e) => {
  const d = e.data;
  if (!d || d.type !== "streamlit:render") return;
  const a = d.args || {};
  canWish = !!a.can_wish;
  if (a.height) ST.height(a.height);
  if (a.voices) { VOICES = a.voices; voiceLine(); }
  if (a.lang && a.lang !== argLang) { argLang = a.lang; setLang(a.lang, true); }
  if (a.sid && a.sid !== SID && a.scene) {
    try { load(typeof a.scene === "string" ? JSON.parse(a.scene) : a.scene, a.sid); }
    catch (err) { $("msg").style.display = "flex"; $("msg").textContent = "Could not build the 3D house: " + err.message; console.error(err); }
  }
});
ST.ready();
const qs = new URLSearchParams(location.search);
if (qs.get("scene")) {             // standalone preview: ?scene=path/to/scene.json
  fetch(qs.get("scene")).then((r) => r.json()).then((j) => { canWish = true; load(j, "dev"); }).catch((e) => { $("msg").textContent = "Scene load failed: " + e; });
}
resize();
requestAnimationFrame(frame);
// debugging handle (read-only use from the browser console)
window.__walk = { scene, camera, renderer, registry, state, walker, get S() { return S; }, walkTo, stairAt, groundAt,
  lift, liftCall, boardLift, stepOut, setXray, liftTrip, goLevel, orbit, drive, startDrive, setLang, words, say, tour,
  tick(n = 1, dt = 1 / 30) { for (let k = 0; k < n; k++) update(dt); renderer.render(scene, camera); updateLabels(canvas.clientWidth, canvas.clientHeight); drawMinimap(); },
  pick(x, y) {
    const r = canvas.getBoundingClientRect(); ray.setFromCamera(new THREE.Vector2((x / r.width) * 2 - 1, -(y / r.height) * 2 + 1), camera);
    return ray.intersectObjects(registry.meshes, false).slice(0, 4).map((h) => [h.object.userData.role, +h.distance.toFixed(2), [+h.point.x.toFixed(2), +(-h.point.z).toFixed(2), +h.point.y.toFixed(2)]]);
  } };
