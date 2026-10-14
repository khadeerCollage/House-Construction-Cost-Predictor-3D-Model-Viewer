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
};
const NO_SHADOW = new Set(["glass", "frost", "water", "ghost", "grass", "land", "yard", "road", "verge", "roadline", "paver", "fl_tile",
  "fl_bed", "fl_wet", "fl_marble", "lamp", "flame", "arrow", "ceiling"]);
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
  theme: "bp", mode: "outside", furn: true, vastu: false, sunOpen: false,
  sunMin: 540, day: dayOfYear(new Date()), voice: true, started: false,
};
const registry = { meshes: [], edges: [], pick: [], doors: [], gates: [], fans: [], labels: [], lamps: [] };
let S = null;            // scene data
let SID = null;
let canWish = false;
let G = {};              // groups
let nav = null, vis = null;
const walker = { x: 0, y: 0, yaw: 0, pitch: -0.05, eye: 1.6, h: 2.0, path: [], lastLook: 0, room: null };
const orbit = { target: new THREE.Vector3(), r: 20, theta: 0, phi: 1.0, auto: false };
const trans = { on: false, t: 0, dur: 1, fromP: new THREE.Vector3(), fromT: new THREE.Vector3(), toP: new THREE.Vector3(), toT: new THREE.Vector3(), done: null };
const keys = new Set();
let dirtyUntil = 0;
const poke = (ms = 400) => { dirtyUntil = Math.max(dirtyUntil, performance.now() + ms); };

function dayOfYear(d) { return Math.floor((d - new Date(d.getFullYear(), 0, 0)) / 86400000); }

// ───────────────────────────── build the house from the scene JSON
function clearScene() {
  for (const k of Object.keys(G)) {
    G[k].traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    scene.remove(G[k]);
  }
  registry.meshes = []; registry.edges = []; registry.pick = []; registry.doors = []; registry.gates = []; registry.fans = []; registry.lamps = [];
  for (const l of registry.labels) l.el.remove();
  registry.labels = [];
}

function roomAt(x, y) {
  if (!S) return null;
  for (const r of S.rooms) { const [x0, y0, x1, y1] = r.r; if (x > x0 && x < x1 && y > y0 && y < y1) return r; }
  return null;
}
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

function build(data) {
  clearScene();
  S = data;
  const lv = S.levels, ffl = lv.ffl;
  G = { shell: new THREE.Group(), ceil: new THREE.Group(), furn: new THREE.Group(), doors: new THREE.Group(),
    site: new THREE.Group(), vastu: new THREE.Group() };
  for (const g of Object.values(G)) scene.add(g);
  const pickRoles = new Set(["fl_tile", "fl_bed", "fl_wet", "fl_marble", "black", "yard", "grass", "paver", "road", "verge", "land", "stair"]);

  // —— walls (exterior faces painted outside, interior faces take the room's colour)
  const shell = new Builder();
  const route = (f) => {
    if (f.bottom) return null;
    if (f.top) return "wall_in";
    const px = f.c[0] + f.pn[0] * 0.2, py = f.c[1] + f.pn[1] * 0.2;
    if (!inFootprint(px, py, -0.01)) return "wall_out";
    return paintFor(roomAt(px, py));
  };
  const cuts = { x: [...new Set(S.rooms.flatMap((r) => [r.r[0], r.r[2]]))].sort((a, b) => a - b),
    y: [...new Set(S.rooms.flatMap((r) => [r.r[1], r.r[3]]))].sort((a, b) => a - b) };
  for (const k of ["ext", "int", "post"]) for (const b of S.walls[k] || []) shell.wall(b, "wall", route, cuts);
  for (const b of S.walls.cmp || []) {
    shell.box(b, "compound", "site");
    shell.box([b[0] - 0.04, b[1] - 0.04, b[2] + 0.04, b[3] + 0.04, b[5], b[5] + 0.06], "pillar", null);
  }
  for (const b of S.walls.pillar || []) shell.box(b, "pillar", "site");

  // —— plinth / stilt
  const [fx, fy, fw, fd] = S.footprint, o = S.shell.outer;
  if (S.stilt) {
    shell.box([o[0], o[1], o[2], o[3], 0, 0.15], "concrete", "site");
    shell.box([o[0], o[1], o[2], o[3], ffl - 0.15, ffl], "slab", "wall");
    for (const c of S.stilt.columns) shell.box([c[0], c[1], c[2], c[3], 0.15, ffl - 0.15], "column", "wall");
  } else {
    shell.box([o[0] - 0.08, o[1] - 0.08, o[2] + 0.08, o[3] + 0.08, 0, ffl], "plinth", "wall");
  }
  // —— floors
  for (const f of S.floors) {
    const [x0, y0, x1, y1] = f.r;
    shell.quadUp(x0, y0, x1, y1, ffl + 0.002, FLOOR[f.cat] || "fl_tile");
  }
  // —— stairs
  for (const st of S.stairs) {
    for (const b of st.boxes) shell.box(b, "stair", "stair");
    for (const rail of st.rails) {
      for (let i = 0; i + 1 < rail.length; i++) rod(shell, rail[i], rail[i + 1], 0.022, "rail");
      for (const p of rail) rod(shell, [p[0], p[1], p[2] - 0.9], p, 0.014, "rail");
    }
  }
  // —— entrance steps
  for (const b of S.steps) shell.box(b, "stair", "stair");
  // —— windows: frames, glass, external sill and sunshade (chajja)
  for (const w of S.windows) windowParts(shell, w);
  // —— door frames (fixed); leaves are animated groups
  for (const d of S.doors) doorFrame(shell, d, ffl);
  shell.emit(G.shell, pickRoles);
  for (const d of S.doors) doorLeaf(d, ffl);

  // —— ceiling slab, roof, parapet, upper floors, mumty
  const ceil = new Builder();
  const sz = S.shell.slab_z;
  for (const r of S.shell.slab) ceil.box([r[0], r[1], r[2], r[3], sz[0], sz[1]], null, "wall", (f) => (f.bottom ? "ceiling" : "slab"));
  const floors = S.shell.floors, H = lv.storey;
  for (let k = 1; k < floors; k++) {
    const z0 = ffl + k * H, z1 = z0 + lv.ceiling, t = 0.23;
    for (const b of perimeter(o, t, z0, z1)) ceil.box(b, "ghost", "ghost");
    ceil.box([o[0], o[1], o[2], o[3], z0 + lv.ceiling, z0 + H], "ghost", "ghost");
  }
  const roofZ = S.shell.roof_z;
  for (const b of perimeter(o, 0.15, roofZ, roofZ + lv.parapet)) ceil.box(b, "parapet", "wall");
  if (S.shell.mumty) {
    const m = S.shell.mumty, mh = 2.4, t = 0.115;
    const walls = perimeter([m[0] - t / 2, m[1] - t / 2, m[2] + t / 2, m[3] + t / 2], t, roofZ, roofZ + mh);
    walls.forEach((b, i) => { if (i !== 1) ceil.box(b, "wall_out", "wall"); });   // one side open: the terrace door
    ceil.box([m[0] - 0.2, m[1] - 0.2, m[2] + 0.2, m[3] + 0.2, roofZ + mh, roofZ + mh + 0.12], "slab", "wall");
  }
  // ceiling lights
  for (const r of S.rooms) {
    if (r.cat === "STAIRS" || r.cat === "LIFT" || r.name.includes("Duct")) continue;
    const [x0, y0, x1, y1] = r.r, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const fan = hasFan(r), off = fan ? Math.min(0.9, Math.max(x1 - x0, y1 - y0) / 4) : 0;
    const lx = (x1 - x0) >= (y1 - y0) ? cx + off : cx, ly = (x1 - x0) >= (y1 - y0) ? cy : cy + off;
    ceil.cyl(lx, ly, 0.16, ffl + lv.ceiling - 0.035, ffl + lv.ceiling - 0.005, "lamp", 20);
  }
  ceil.emit(G.ceil);

  // —— furniture + ceiling fans
  const furn = new Builder();
  for (const it of S.furniture) { try { furniture(furn, it, ffl); } catch (e) { console.warn("furniture", it.kind, e); } }
  furn.emit(G.furn);
  for (const r of S.rooms) if (hasFan(r)) fanAt(r, ffl + lv.ceiling);

  // —— site: ground, road, yard, amenities, cars, trees, gates, walkway
  buildSite(pickRoles);

  // —— Vastu mandala on the floor
  buildVastu(ffl);

  // —— labels
  for (const r of S.rooms) {
    if (r.cat === "LIFT" || r.name.includes("Duct")) continue;
    const [x0, y0, x1, y1] = r.r;
    addLabel(`${esc(r.name)}<small>${r.size} · ${r.ft}</small>`, [(x0 + x1) / 2, (y0 + y1) / 2, ffl + 0.3], "lbl", () => ["dollhouse", "top"].includes(state.mode));
  }
  for (const d of S.doors) {
    if (d.kind === "lift") continue;
    for (const side of [1, -1]) {
      const to = side === 1 ? d.other : d.room;     // the room side reads where the door goes
      if (!to) continue;
      const px = d.mid[0] + d.n[0] * side * 0.12, py = d.mid[1] + d.n[1] * side * 0.12;
      addLabel(`→ ${esc(to === "Outside" ? "Outside" : to)}`, [px, py, ffl + 2.32], "sign", () => state.mode === "walk", { door: d, side });
    }
  }
  for (const g of S.mandala) addLabel(`${g.word}`, [(g.r[0] + g.r[2]) / 2, (g.r[1] + g.r[3]) / 2, ffl + 0.05], "lbl zone",
    () => state.vastu && state.mode !== "outside");
  // north arrow on the ground (front-left corner of the plot)
  const nx = S.north[0], ny = S.north[1], ax = 1.0 + 0.0, ay = S.plot.h - 1.0;
  addLabel("N", [ax + nx * 1.25, ay + ny * 1.25, 0.15], "north", () => state.mode !== "walk");
  if (floors > 1) addLabel(`Upper floor${floors > 2 ? "s" : ""} — shell only (rooms not designed yet)`,
    [fx + fw / 2, fy + fd / 2, ffl + H * (floors - 0.4)], "lbl", () => state.mode === "outside");

  nav = decodeNav(S.nav);
  vis = buildVis();
  drawMinimapBase();
  applyTheme();
  applySun();
  setVastu(state.vastu);
  G.furn.visible = state.furn;
  $("startMeta").textContent = `${S.meta.title} · ${S.meta.cents} cents · carpet ${S.meta.carpet} m² · Vastu ${S.meta.vastu}/100`;
  $("msg").style.display = "none";
  poke(1500);
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

function doorLeaf(d, ffl) {
  const role = { main: "door_main", bath: "door_bath", lift: "door_steel", service: "door_steel" }[d.kind] || "door_int";
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
  G.doors.add(pivot);
  let delta = ((d.open - d.closed + 540) % 360) - 180;
  registry.doors.push({ d, pivot, cur: 0, target: 0, delta: delta * DEG, still: d.kind === "lift" });
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

function fanAt(r, ceilZ) {
  const [x0, y0, x1, y1] = r.r, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const g = new THREE.Group(); g.position.set(cx, ceilZ - 0.4, -cy);
  const add = (geo, role) => { const m = new THREE.Mesh(geo, mat(role)); m.userData.role = role; m.castShadow = true; g.add(m); registry.meshes.push(m); return m; };
  add(new THREE.CylinderGeometry(0.012, 0.012, 0.38, 6).translate(0, 0.21, 0), "fan");
  add(new THREE.CylinderGeometry(0.11, 0.09, 0.09, 18), "fan");
  for (let i = 0; i < 3; i++) {
    const b = new THREE.BoxGeometry(0.55, 0.012, 0.11); b.translate(0.36, -0.01, 0); b.rotateY(i * 2 * Math.PI / 3); add(b, "fan");
  }
  G.ceil.add(g); registry.fans.push(g);
}

// ───────────────────────────── site
function buildSite(pickRoles) {
  const bld = new Builder(), W = S.plot.w, H = S.plot.h, rd = S.road;
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
  for (const c of S.cars) car(bld, c, c.stilt ? stiltBase : 0.0);
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
function gate(g) {
  const [g0, g1] = g, H = S.plot.h, width = g1 - g0;
  const leaves = width > 2.4 ? [[g0, 0, -90, width / 2], [g1, 180, 90, width / 2]] : [[g0, 0, -90, width]];
  for (const [hx, closed, delta, len] of leaves) {
    const pivot = new THREE.Group(); pivot.position.set(hx, 0, -H); pivot.rotation.y = closed * DEG;
    const bld = new Builder(), L = len - 0.06;
    const bar = (x0, x1, z0, z1) => bld.box([x0, -0.02, x1, 0.02, z0, z1], "gate", "gate");
    bar(0.03, L, 0.08, 0.13); bar(0.03, L, 1.42, 1.47); bar(0.03, L, 0.75, 0.79); bar(0.03, 0.08, 0.08, 1.47); bar(L - 0.05, L, 0.08, 1.47);
    for (let x = 0.2; x < L - 0.08; x += 0.13) bar(x, x + 0.022, 0.13, 1.42);
    bld.emit(pivot);            // leaf built along local +x from its hinge
    G.site.add(pivot);
    registry.gates.push({ pivot, closed: closed * DEG, cur: 0, target: 0, delta: delta * DEG, mid: [(g0 + g1) / 2, H] });
  }
}

// ───────────────────────────── Vastu overlay
const ZONE_COL = { NE: 0x2E86DE, E: 0xF6C343, SE: 0xE67E22, S: 0xC0392B, SW: 0x8E5B3A, W: 0x95A5A6, NW: 0xB8E0F6, N: 0x27AE60, CENTER: 0xF1C40F };
function buildVastu(ffl) {
  for (const g of S.mandala) {
    const [x0, y0, x1, y1] = g.r;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0 - 0.04, y1 - y0 - 0.04),
      new THREE.MeshBasicMaterial({ color: ZONE_COL[g.zone] || 0xffffff, transparent: true, opacity: 0.22, depthWrite: false }));
    m.rotation.x = -Math.PI / 2; m.position.set((x0 + x1) / 2, ffl + 0.012, -(y0 + y1) / 2); G.vastu.add(m);
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
    if (ok) { _v.copy(l.pos).project(camera); ok = _v.z < 1 && Math.abs(_v.x) < 1.1 && Math.abs(_v.y) < 1.1; }
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
const okCell = (i, j) => i >= 0 && j >= 0 && i < nav.nx && j < nav.ny && nav.free[j * nav.nx + i] === 1;
const freeAt = (x, y) => { const [i, j] = cellOf(x, y); return okCell(i, j); };
function nearestFree(x, y, within = 1.5) {
  const [ci, cj] = cellOf(x, y); if (okCell(ci, cj)) return [ci, cj];
  const k = Math.ceil(within / nav.cell); let best = null;
  for (let dj = -k; dj <= k; dj++) for (let di = -k; di <= k; di++) {
    if (okCell(ci + di, cj + dj)) { const d = di * di + dj * dj; if (!best || d < best[0]) best = [d, [ci + di, cj + dj]]; }
  }
  return best ? best[1] : null;
}
const centreOf = (c) => [nav.x0 + (c[0] + 0.5) * nav.cell, nav.y0 + (c[1] + 0.5) * nav.cell];
function astar(sx, sy, gx, gy) {
  const s = nearestFree(sx, sy, 1.0), g = nearestFree(gx, gy, 1.2);
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
      const a = x + dx, b = y + dy; if (!okCell(a, b)) continue;
      if (dx && dy && !(okCell(a, y) && okCell(x, b))) continue;
      const j = b * nx + a; if (closed[j]) continue;
      const ng = gs[i] + (dx && dy ? 1.414 : 1);
      if (ng < gs[j]) { gs[j] = ng; came[j] = i; push([ng + hf(j), j]); }
    }
  }
  if (!found) return null;
  const cells = []; for (let i = gi; i !== -1; i = came[i]) cells.push([i % nx, (i / nx) | 0]);
  cells.reverse();
  // string-pull
  const out = [cells[0]]; let k = 0;
  while (k < cells.length - 1) { let m = cells.length - 1; while (m > k + 1 && !losCells(cells[k], cells[m])) m--; out.push(cells[m]); k = m; }
  return out.map(centreOf);
}
function losCells(a, b) {
  const n = Math.ceil(Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])) * 2) + 1;
  for (let k = 0; k <= n; k++) { const f = k / n; if (!okCell(Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f))) return false; }
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
function groundAt(x, y) {
  let z = 0;
  for (const g of S.ground) if (x >= g[0] && x <= g[2] && y >= g[1] && y <= g[3]) z = Math.max(z, g[4]);
  return z;
}

// ───────────────────────────── modes & camera
function fitRadius(span) { return span / 2 / Math.tan(camera.fov * DEG / 2) * 1.25; }
function frontTheta() { return Math.atan2(-0.45, -1); }      // camera on the road side, a little to the left
function setMode(m, opts = {}) {
  if (!S) return;
  const prev = state.mode; state.mode = m;
  document.querySelectorAll("#modes .b[data-mode]").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  G.ceil.visible = m === "walk" || m === "outside";
  $("pad").style.display = m === "walk" ? "block" : "none";
  $("hud").style.display = m === "walk" ? "block" : "none";
  if (m !== "walk") closeWish();
  const [fx, fy, fw, fd] = S.footprint;
  const centre = new THREE.Vector3(fx + fw / 2, S.levels.ffl + 0.5, -(fy + fd / 2));
  if (m === "walk") {
    if (opts.at) { walker.x = opts.at[0]; walker.y = opts.at[1]; }
    else if (prev !== "walk") placeWalkerDefault();
    if (opts.yaw !== undefined) walker.yaw = opts.yaw;
    walker.pitch = -0.05; walker.h = groundAt(walker.x, walker.y) + walker.eye; walker.path = [];
    const toP = new THREE.Vector3(walker.x, walker.h, -walker.y);
    const toT = toP.clone().add(lookDir());
    flyTo(toP, toT, prev === "walk" ? 0.01 : 1.1);
  } else {
    orbit.target.copy(centre);
    const span = Math.max(S.plot.w, S.plot.h);
    if (m === "outside") { orbit.phi = 1.1; orbit.theta = frontTheta(); orbit.r = fitRadius(span) * 1.3; orbit.target.y = 1.5;
      orbit.target.z -= 1.5; }
    if (m === "dollhouse") { orbit.phi = 0.72; orbit.theta = frontTheta(); orbit.r = fitRadius(Math.max(fw, fd)) * 1.35; }
    if (m === "top") { orbit.phi = 0.012; orbit.theta = 0; orbit.r = fitRadius(span) * 1.25; }
    const toP = orbitPos(), toT = orbit.target.clone();
    flyTo(toP, toT, opts.instant ? 0.01 : 1.0);
  }
  poke(1500);
}
function placeWalkerDefault() {
  const main = S.doors.find((d) => d.kind === "main");
  let p = null;
  if (main) { const c = nearestFree(main.mid[0] + main.n[0] * 0.9, main.mid[1] + main.n[1] * 0.9, 1.2); if (c) p = centreOf(c); }
  if (!p) { const lr = S.rooms.find((r) => r.cat === "LIVING") || S.rooms[0]; const c = nearestFree((lr.r[0] + lr.r[2]) / 2, (lr.r[1] + lr.r[3]) / 2, 2); if (c) p = centreOf(c); }
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
  keys.add(e.key.toLowerCase()); if (state.mode === "walk" && /^(arrow|w|a|s|d)/.test(e.key.toLowerCase())) tourInterrupt();
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
  const hits = ray.intersectObjects(registry.pick.filter((m) => m.visible && m.parent && m.parent.visible), false);
  if (!hits.length) return;
  const p = hits[0].point, x = p.x, y = -p.z;
  if (state.mode === "walk") { tourInterrupt(); walkTo(x, y); }
  else if (inFootprint(x, y)) { const c = nearestFree(x, y, 1.2); if (c) { const q = centreOf(c); setMode("walk", { at: q, yaw: Math.atan2(S.footprint[1] + S.footprint[3] / 2 - q[1], S.footprint[0] + S.footprint[2] / 2 - q[0]) }); } }
}
function walkTo(x, y) {
  const path = astar(walker.x, walker.y, x, y);
  if (path && path.length) { walker.path = path.slice(1); if (!walker.path.length) walker.path = [path[0]]; poke(); }
  else toast("Can't walk there — that spot is inside a wall or furniture.");
}
function stepWalker(fwd, side) {
  const fx = Math.cos(walker.yaw), fy = Math.sin(walker.yaw), sx = Math.cos(walker.yaw - Math.PI / 2), sy = Math.sin(walker.yaw - Math.PI / 2);
  const nx = walker.x + fx * fwd + sx * side, ny = walker.y + fy * fwd + sy * side;
  if (freeAt(nx, ny)) { walker.x = nx; walker.y = ny; }
  else if (freeAt(nx, walker.y)) walker.x = nx;
  else if (freeAt(walker.x, ny)) walker.y = ny;
}

// ───────────────────────────── guided tour
const tour = { on: false, i: -1, phase: "", t: 0, paused: false, base: 0, spoken: true, capAt: 0 };
let voice = null;
function pickVoice() {
  if (!("speechSynthesis" in window)) return;
  const vs = speechSynthesis.getVoices().filter((v) => v.localService !== false);   // offline voices only
  voice = vs.find((v) => /en[-_]IN/i.test(v.lang)) || vs.find((v) => /^en/i.test(v.lang)) || null;
}
if ("speechSynthesis" in window) { pickVoice(); speechSynthesis.onvoiceschanged = pickVoice; }
function speak(text) {
  tour.spoken = true;
  if (!state.voice || !voice || !("speechSynthesis" in window)) return;
  try {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text); u.voice = voice; u.lang = voice.lang; u.rate = 0.95;
    tour.spoken = false; u.onend = () => { tour.spoken = true; }; u.onerror = () => { tour.spoken = true; };
    speechSynthesis.speak(u);
  } catch (e) { tour.spoken = true; }
}
function tourStart() {
  if (!S || !S.tour || !S.tour.length) { toast("No guided walk for this plan."); return; }
  hideStart(); tour.on = true; tour.paused = false; tour.i = -1;
  $("cap").style.display = "block"; $("tPause").textContent = "⏸ Pause";
  const st0 = S.tour[0];
  setMode("walk", { at: st0.path[0], yaw: yawTo(st0.path[0], st0.look) });
  setTimeout(tourNext, 1150);
}
function yawTo(from, to) { return to ? Math.atan2(to[1] - from[1], to[0] - from[0]) : walker.yaw; }
function tourNext() {
  if (!tour.on) return;
  tour.i++;
  if (tour.i >= S.tour.length) return tourEnd(true);
  const st = S.tour[tour.i];
  $("capN").textContent = `${tour.i + 1} / ${S.tour.length}`;
  if (st.kind === "end") { caption(st); tour.phase = "end"; tour.t = 0; return; }
  if (st.kind === "start" || !st.path.length) { caption(st); tour.phase = "wait"; tour.t = 0; return; }
  walker.path = st.path.slice(1); tour.phase = "walk"; tour.t = 0;
  $("capT").textContent = `Walking to: ${st.title}`; $("capS").textContent = "";
}
function caption(st) { $("capT").textContent = st.title; $("capS").textContent = st.say; speak(st.say); tour.capAt = performance.now(); }
function tourInterrupt() { if (tour.on) tourEnd(false); }
function tourEnd(finished) {
  tour.on = false; tour.phase = ""; walker.path = [];
  if ("speechSynthesis" in window) speechSynthesis.cancel();
  $("cap").style.display = "none";
  if (finished) setMode("dollhouse");
}
function tourUpdate(dt) {
  if (!tour.on || tour.paused) return;
  const st = S.tour[tour.i]; if (!st) return;
  tour.t += dt;
  const minRead = Math.max(3.5, (st.say || "").split(" ").length / 3.2);
  // speech 'end' events are unreliable in some browsers: never wait longer than the reading time + 4 s
  const readDone = () => {
    const el = (performance.now() - tour.capAt) / 1000;
    return el > minRead + 4 || (el > (state.voice && voice ? 2.0 : minRead) && tour.spoken);
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
  if ("speechSynthesis" in window) { if (tour.paused) speechSynthesis.pause(); else speechSynthesis.resume(); }
});
$("tNext").addEventListener("click", () => { if (!tour.on) return; if ("speechSynthesis" in window) speechSynthesis.cancel(); tour.spoken = true; walker.path = []; tourNext(); });
$("tStop").addEventListener("click", () => tourEnd(false));
$("tVoice").addEventListener("click", () => {
  state.voice = !state.voice; $("tVoice").textContent = state.voice ? "🔊 Voice on" : "🔇 Voice off";
  if (!state.voice && "speechSynthesis" in window) { speechSynthesis.cancel(); tour.spoken = true; }
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
  applySun(); drawMinimapBase(); poke();
}
$("btnTheme").addEventListener("click", () => { state.theme = state.theme === "bp" ? "real" : "bp"; applyTheme(); });
$("btnVastu").addEventListener("click", () => setVastu(!state.vastu));
$("btnFurn").addEventListener("click", () => { state.furn = !state.furn; G.furn.visible = state.furn; $("btnFurn").classList.toggle("on", state.furn); poke(); });
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
document.querySelectorAll("#modes .b[data-mode]").forEach((b) => b.addEventListener("click", () => { hideStart(); tourInterrupt(); setMode(b.dataset.mode); }));

function hideStart() { if (!state.started) { state.started = true; $("start").style.display = "none"; orbit.auto = false; } }
$("sTour").addEventListener("click", tourStart);
$("sWalk").addEventListener("click", () => { hideStart(); setMode("walk"); setTimeout(() => toast("Drag to look around · tap the floor to walk there"), 1200); });
$("sDoll").addEventListener("click", () => { hideStart(); setMode("dollhouse"); });

// ───────────────────────────── HUD + "change this room" wishes → Smart Re-Plan
let hudRoom;
function updateHud() {
  const r = roomAt(walker.x, walker.y);
  if (r === hudRoom) return;
  hudRoom = r;
  const lv = S.levels, ch = lv.ceiling, ft = Math.round(ch / 0.3048 * 12);
  if (!r) { $("hud").innerHTML = `<b>${inFootprint(walker.x, walker.y) ? "Inside" : "Outside the house"}</b>`; return; }
  const badge = { best: "🟢 best Vastu spot", good: "🟢 good Vastu", ok: "🟡 acceptable Vastu", poor: "🔴 not ideal for Vastu" }[r.vastu] || "";
  $("hud").innerHTML = `<b>📍 ${esc(r.name)}</b><div class="sub">${r.size} (${r.ft}) · ${r.area} m² · ${r.sqft} sq ft · ` +
    `${r.zone_word ? esc(r.zone_word) + " · " : ""}ceiling ${ch.toFixed(2)} m (${Math.floor(ft / 12)}'${ft % 12}")${badge ? " · " + badge : ""}</div>` +
    (canWish && r.key ? `<div class="row"><button class="b" id="hudWish">✏️ Change this room</button><button class="b" id="hudSay">🔊 Tell me</button></div>`
      : `<div class="row"><button class="b" id="hudSay">🔊 Tell me</button></div>`);
  const w = $("hudWish"); if (w) w.onclick = () => openWish(r);
  $("hudSay").onclick = () => { const old = state.voice; state.voice = true; speak(r.say); state.voice = old; toast(r.say, 7000); };
}
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
function drawMinimapBase() {
  if (!S) return;
  const W = S.plot.w, H = S.plot.h, max = window.innerWidth < 640 ? 130 : 190, s = max / Math.max(W, H);
  mini.width = miniBase.width = Math.round(W * s + 12); mini.height = miniBase.height = Math.round(H * s + 12); mm = { s, pad: 6 };
  const X = (x) => mm.pad + x * s, Y = (y) => mm.pad + (H - y) * s;
  mg.fillStyle = "#06182E"; mg.fillRect(0, 0, mini.width, mini.height);
  mg.strokeStyle = "#3E6E9C"; mg.setLineDash([3, 3]); mg.strokeRect(X(0), Y(H), W * s, H * s); mg.setLineDash([]);
  const col = { LIVING: "#1E5A92", DINING: "#1E5A92", BEDROOM: "#2C4F86", KITCHEN: "#6B4E1E", BATHROOM: "#1D5E66", POOJA: "#7A5B12", STAIRS: "#3A3A5A" };
  for (const r of S.rooms) { const [x0, y0, x1, y1] = r.r; mg.fillStyle = col[r.cat] || "#203C5E"; mg.globalAlpha = 0.75; mg.fillRect(X(x0), Y(y1), (x1 - x0) * s, (y1 - y0) * s); }
  mg.globalAlpha = 1;
  for (const st of S.site) { const [x0, y0, x1, y1] = st.r; mg.strokeStyle = st.cat === "GARDEN" ? "#22C55E" : st.cat === "POOL" ? "#38BDF8" : "#64748B"; mg.strokeRect(X(x0), Y(y1), (x1 - x0) * s, (y1 - y0) * s); }
  mg.fillStyle = "#00E5FF";
  for (const k of ["ext", "int", "post", "cmp"]) for (const b of S.walls[k] || []) { if (k !== "cmp" && b[4] > S.levels.ffl + 0.5) continue; mg.fillRect(X(b[0]), Y(b[3]), Math.max(1, (b[2] - b[0]) * s), Math.max(1, (b[3] - b[1]) * s)); }
  mg.fillStyle = "#00FF88";
  for (const d of S.doors) { mg.beginPath(); mg.arc(X(d.mid[0]), Y(d.mid[1]), 2.2, 0, 7); mg.fill(); }
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
      if (performance.now() - walker.lastLook > 1500 && d > 0.05) {
        const want = Math.atan2(dy, dx), diff = ((want - walker.yaw + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
        walker.yaw += clamp(diff, -2.6 * dt, 2.6 * dt);
        walker.pitch = lerp(walker.pitch, -0.06, Math.min(1, dt * 2));
      }
      active = true;
    }
    tourUpdate(dt);
    if (tour.on) active = true;
    const gz = groundAt(walker.x, walker.y) + walker.eye;
    if (Math.abs(gz - walker.h) > 0.002) { walker.h += (gz - walker.h) * Math.min(1, dt * 7); active = true; }
    camera.position.set(walker.x, walker.h, -walker.y);
    camera.lookAt(camera.position.clone().add(lookDir()));
    updateHud();
  } else {
    if (orbit.auto) { orbit.theta += dt * 0.08; active = true; }
    camera.position.copy(orbitPos()); camera.lookAt(orbit.target);
  }
  // doors open as you come near (always open in the roof-off views), gates likewise
  for (const o of registry.doors) {
    if (o.still) continue;
    const near = Math.hypot(walker.x - o.d.mid[0], walker.y - o.d.mid[1]) < 1.7;
    o.target = state.mode === "walk" ? (near ? 1 : 0) : (state.mode === "outside" ? 0 : 1);
    if (Math.abs(o.target - o.cur) > 1e-3) { o.cur += clamp(o.target - o.cur, -dt * 1.6, dt * 1.6); o.pivot.rotation.y = o.d.closed * DEG + o.delta * ease(o.cur); active = true; }
  }
  for (const o of registry.gates) {
    const near = state.mode === "walk" && Math.hypot(walker.x - o.mid[0], walker.y - o.mid[1]) < 3.2;
    o.target = near || tour.on ? 1 : 0;
    if (Math.abs(o.target - o.cur) > 1e-3) { o.cur += clamp(o.target - o.cur, -dt * 0.9, dt * 0.9); o.pivot.rotation.y = o.closed + o.delta * ease(o.cur); active = true; }
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
window.__walk = { scene, camera, renderer, registry, state, walker, get S() { return S; },
  pick(x, y) {
    const r = canvas.getBoundingClientRect(); ray.setFromCamera(new THREE.Vector2((x / r.width) * 2 - 1, -(y / r.height) * 2 + 1), camera);
    return ray.intersectObjects(registry.meshes, false).slice(0, 4).map((h) => [h.object.userData.role, +h.distance.toFixed(2), [+h.point.x.toFixed(2), +(-h.point.z).toFixed(2), +h.point.y.toFixed(2)]]);
  } };
