// GENESIS WORLD — observatório 3D ao vivo. Tudo que aparece vem do estado real da simulação (protocolo genesis-live);
// overlays interpretativos são marcados. O viewer nunca altera ciência nem RNG: só observa.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { decode, asU16, asI16, asF32, asI64, f16 } from './msgpack.js';

const $ = (id) => document.getElementById(id);
const MOBILE = matchMedia('(max-width: 760px)').matches || /Mobi|Android/i.test(navigator.userAgent);
const hp = new URLSearchParams(location.hash.slice(1));
let TOKEN = hp.get('t') || sessionStorage.getItem('gtok') || '';
let ADMIN = hp.get('a') || sessionStorage.getItem('gadm') || '';
if (TOKEN) sessionStorage.setItem('gtok', TOKEN);
if (ADMIN) sessionStorage.setItem('gadm', ADMIN);
if (location.hash) history.replaceState(null, '', location.pathname + location.search);
const store = { get(k, d) { try { const v = localStorage.getItem('gw.' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
                set(k, v) { try { localStorage.setItem('gw.' + k, JSON.stringify(v)); } catch (e) { /* armazenamento opcional */ } } };

async function api(path, opts = {}) {
  const h = Object.assign({ 'X-Genesis-Token': TOKEN }, opts.headers || {});
  const r = await fetch(path, Object.assign({}, opts, { headers: h }));
  if (!r.ok) throw new Error(path + ' → ' + r.status);
  const ct = r.headers.get('content-type') || '';
  if (ct.includes('msgpack')) return decode(new Uint8Array(await r.arrayBuffer()));
  return r.json();
}
async function admin(action, body) {
  return api('/api/admin/' + action, { method: 'POST', body: JSON.stringify(body || {}), headers: { 'X-Genesis-Admin': ADMIN, 'Content-Type': 'application/json' } });
}

// ------------------------------------------------------------------ cena
const canvas = $('world');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: !MOBILE, powerPreference: 'high-performance', preserveDrawingBuffer: true });
let pixelRatio = Math.min(devicePixelRatio || 1, MOBILE ? 1.25 : 2);
renderer.setPixelRatio(pixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
const world = new THREE.Group(); scene.add(world);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 6000);
camera.position.set(0, 160, 190);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.maxPolarAngle = 1.47; controls.minDistance = 2; controls.maxDistance = 1800;
const hemi = new THREE.HemisphereLight(0xbfd8ff, 0x23301c, 0.75); scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff1dc, 1.3); sun.position.set(200, 300, 100); scene.add(sun);
const amb = new THREE.AmbientLight(0x404a66, 0.35); scene.add(amb);
scene.fog = new THREE.Fog(0x0a1020, 400, 2200);
const DAY = new THREE.Color(0x6f9ccc), NIGHT = new THREE.Color(0x03050c), DUSK = new THREE.Color(0x2b2141);
scene.background = DAY.clone();

const S = { W: 64, H: 64, step: 1, gw: 2, gh: 2, hs: 24, VS: 0.55, BH: 0.5, heights: null, sea: 0, base: false,
            tick: 0, frameT: 0, frameDt: 250, lastFrameWall: 0, mode: 'planet', sel: null, follow: false, replay: false,
            science: store.get('science', false), palette: store.get('palette', 0), layer: null, maxAgents: MOBILE ? 1500 : 6000,
            status: {}, lat: null, events: [], settlements: [], roads: [], dayPhase: 0.25, card: null, devFrames: null };

function hAt(x, y) {
  if (!S.heights) return 0;
  const gx = Math.min(S.gw - 1.001, Math.max(0, x / S.step)), gy = Math.min(S.gh - 1.001, Math.max(0, y / S.step));
  const x0 = Math.floor(gx), y0 = Math.floor(gy), fx = gx - x0, fy = gy - y0, w = S.gw, H = S.heights;
  const a = H[y0 * w + x0], b = H[y0 * w + x0 + 1], c = H[(y0 + 1) * w + x0], d = H[(y0 + 1) * w + x0 + 1];
  return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy) * S.VS;
}
const toX = (x) => x - S.W / 2, toZ = (y) => y - S.H / 2;

// ------------------------------------------------------------------ terreno, água, nuvens
let terrain = null, water = null, heatMesh = null, clouds = null, grid = null;
const heatTex = new THREE.DataTexture(new Uint8Array(64 * 64 * 4), 64, 64, THREE.RGBAFormat); heatTex.needsUpdate = true;
heatTex.magFilter = THREE.LinearFilter;
const stormTex = new THREE.DataTexture(new Uint8Array(32 * 32 * 4), 32, 32, THREE.RGBAFormat); stormTex.magFilter = THREE.LinearFilter;
const waterMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, uniforms: { t: { value: 0 }, night: { value: 0 } },
  vertexShader: `varying vec3 vp; void main(){ vp = (modelMatrix*vec4(position,1.0)).xyz; gl_Position = projectionMatrix*viewMatrix*vec4(vp,1.0);} `,
  fragmentShader: `uniform float t; uniform float night; varying vec3 vp;
    void main(){ float w = sin(vp.x*0.35+t*1.3)*0.5+sin(vp.z*0.42-t*1.1)*0.5;
      vec3 c = mix(vec3(0.05,0.28,0.48), vec3(0.18,0.52,0.72), 0.5+0.25*w);
      c = mix(c, vec3(0.02,0.05,0.12), night*0.8); gl_FragColor = vec4(c, 0.78); }`
});

function buildTerrain(m) {
  [terrain, water, heatMesh, clouds, grid].forEach((o) => { if (o) { world.remove(o); o.geometry.dispose(); } });
  S.W = m.W; S.H = m.H; S.step = m.step; S.gw = m.gw; S.gh = m.gh; S.hs = m.height_scale; S.sea = (m.sea_level || 0);
  const hq = asI16(m.height); const n = S.gw * S.gh;
  S.heights = new Float32Array(n);
  for (let i = 0; i < n; i++) S.heights[i] = ((hq[i] & 0xffff) / 65535) * S.hs;
  const rgb = m.rgb, wat = m.water;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), uv = new Float32Array(n * 2);
  for (let iy = 0; iy < S.gh; iy++) for (let ix = 0; ix < S.gw; ix++) {
    const i = iy * S.gw + ix;
    pos[i * 3] = toX(ix * S.step); pos[i * 3 + 1] = S.heights[i] * S.VS; pos[i * 3 + 2] = toZ(iy * S.step);
    col[i * 3] = rgb[i * 3] / 255; col[i * 3 + 1] = rgb[i * 3 + 1] / 255; col[i * 3 + 2] = rgb[i * 3 + 2] / 255;
    if (wat[i]) { col[i * 3] *= 0.55; col[i * 3 + 1] *= 0.7; col[i * 3 + 2] = Math.min(1, col[i * 3 + 2] + 0.25); }
    uv[i * 2] = ix / (S.gw - 1); uv[i * 2 + 1] = 1 - iy / (S.gh - 1);
  }
  const idx = [];
  for (let iy = 0; iy < S.gh - 1; iy++) for (let ix = 0; ix < S.gw - 1; ix++) {
    const a = iy * S.gw + ix, b = a + 1, c = a + S.gw, d = c + 1; idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(new THREE.BufferAttribute(n > 65535 ? new Uint32Array(idx) : new Uint16Array(idx), 1)); g.computeVertexNormals();
  terrain = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.93, metalness: 0.0 }));
  world.add(terrain);
  heatMesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ map: heatTex, transparent: true, opacity: 0.72, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: -2 }));
  heatMesh.visible = false; world.add(heatMesh);
  // água: quads por célula de água na altura local
  const wq = [];
  for (let iy = 0; iy < S.gh - 1; iy++) for (let ix = 0; ix < S.gw - 1; ix++) {
    const i = iy * S.gw + ix; if (!wat[i]) continue;
    const y = Math.max(S.heights[i], S.heights[i + 1], S.heights[i + S.gw], S.heights[i + S.gw + 1]) * S.VS + 0.12;
    const x0 = toX(ix * S.step), z0 = toZ(iy * S.step), x1 = x0 + S.step, z1 = z0 + S.step;
    wq.push(x0, y, z0, x0, y, z1, x1, y, z0, x1, y, z0, x0, y, z1, x1, y, z1);
  }
  const wg = new THREE.BufferGeometry(); wg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(wq), 3));
  water = new THREE.Mesh(wg, waterMat); world.add(water);
  const cg = new THREE.PlaneGeometry(S.W, S.H); cg.rotateX(-Math.PI / 2);
  clouds = new THREE.Mesh(cg, new THREE.MeshBasicMaterial({ map: stormTex, transparent: true, depthWrite: false, opacity: 0.85 }));
  clouds.position.y = S.hs * S.VS + 6; world.add(clouds);
  grid = new THREE.GridHelper(Math.max(S.W, S.H), Math.max(S.W, S.H) / 16, 0x2f4a78, 0x1a2a44); grid.position.y = 0.02;
  grid.visible = S.science; world.add(grid);
  const d = Math.max(S.W, S.H);
  controls.maxDistance = d * 4; scene.fog.far = d * 6; scene.fog.near = d * 1.2;
  if (!S.base) { camera.position.set(0, d * 0.75, d * 0.95); controls.target.set(0, 0, 0); }
  S.base = true;
  $('scale').textContent = `▭ 16 tiles · mundo ${S.W}×${S.H}`;
}

// ------------------------------------------------------------------ construções (estado real: pilhas e aterros)
const blockGeo = new THREE.BoxGeometry(0.92, 1, 0.92);
const blockMat = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0.05 });
let blocks = null; const blockIdx = new Map(); let blockCount = 0;
const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion(), tmpS = new THREE.Vector3(), tmpP = new THREE.Vector3(), tmpC = new THREE.Color();
function ensureBlocks(cap) {
  if (blocks && blocks.instanceMatrix.count >= cap) return;
  const old = blocks; const nc = Math.max(1024, cap * 2);
  blocks = new THREE.InstancedMesh(blockGeo, blockMat, nc); blocks.count = 0; blocks.frustumCulled = false;
  blocks.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  if (old) { for (let i = 0; i < blockCount; i++) { old.getMatrixAt(i, tmpM); blocks.setMatrixAt(i, tmpM); old.getColorAt(i, tmpC); blocks.setColorAt(i, tmpC); }
    world.remove(old); old.dispose(); }
  blocks.count = blockCount; world.add(blocks);
}
function setBlock(x, y, h, kind) {
  const key = x * 100000 + y;
  let i = blockIdx.get(key);
  if (h <= 0) {                               // remoção: troca com o último
    if (i === undefined) return;
    const last = blockCount - 1;
    if (i !== last) { blocks.getMatrixAt(last, tmpM); blocks.setMatrixAt(i, tmpM); blocks.getColorAt(last, tmpC); blocks.setColorAt(i, tmpC);
      for (const [k, v] of blockIdx) if (v === last) { blockIdx.set(k, i); break; } }
    blockIdx.delete(key); blockCount--; blocks.count = blockCount; return;
  }
  if (i === undefined) { ensureBlocks(blockCount + 1); i = blockCount++; blockIdx.set(key, i); blocks.count = blockCount; }
  const gy = hAt(x + 0.5, y + 0.5);
  if (kind === 1) { tmpS.set(1, 0.25, 1); tmpP.set(toX(x + 0.5), Math.max(gy, S.sea * S.VS) + 0.2, toZ(y + 0.5)); tmpC.setRGB(0.25, 0.62, 0.6); }
  else { const hh = h * S.BH; tmpS.set(0.92, hh, 0.92); tmpP.set(toX(x + 0.5), gy + hh / 2, toZ(y + 0.5));
    const k = Math.min(1, h / 6); tmpC.setRGB(0.55 + 0.25 * k, 0.5 + 0.12 * k, 0.42 - 0.1 * k); }
  tmpM.compose(tmpP, tmpQ.identity(), tmpS); blocks.setMatrixAt(i, tmpM); blocks.setColorAt(i, tmpC);
}
function loadBlocks(b, n) {
  blockIdx.clear(); blockCount = 0; ensureBlocks(n + 64);
  const a = asI16(b);
  for (let i = 0; i < n; i++) setBlock(a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]);
  blocks.instanceMatrix.needsUpdate = true; if (blocks.instanceColor) blocks.instanceColor.needsUpdate = true;
}
function applyDelta(b, n) {
  const a = asI16(b);
  for (let i = 0; i < n; i++) { const fill = a[i * 4 + 3]; setBlock(a[i * 4], a[i * 4 + 1], fill ? 1 : a[i * 4 + 2], fill ? 1 : 0);
    if (!fill && a[i * 4 + 2] <= 0) setBlock(a[i * 4], a[i * 4 + 1], 0, 0); }
  blocks.instanceMatrix.needsUpdate = true; if (blocks.instanceColor) blocks.instanceColor.needsUpdate = true;
}

// ------------------------------------------------------------------ assentamentos (detectados, nunca spawnados)
const setGroup = new THREE.Group(); world.add(setGroup);
const SET_COL = { candidate: 0x8d99ae, detected: 0x5fe0d0, city: 0xb394ff, collapsed: 0xff6b6b };
function drawSettlements(list) {
  S.settlements = list || [];
  while (setGroup.children.length) { const c = setGroup.children.pop(); c.geometry.dispose(); }
  for (const s of S.settlements) {
    const r = Math.max(3, Math.sqrt(s.cells) * s.cell * 0.6);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(r, 0.18, 6, 48), new THREE.MeshBasicMaterial({ color: SET_COL[s.state] || 0xffffff, transparent: true, opacity: 0.8 }));
    ring.rotation.x = Math.PI / 2; ring.position.set(toX(s.centre[0]), hAt(s.centre[0], s.centre[1]) + 0.6, toZ(s.centre[1]));
    ring.userData = { settlement: s.id }; setGroup.add(ring);
    const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 6, 6), new THREE.MeshBasicMaterial({ color: SET_COL[s.state] || 0xffffff, transparent: true, opacity: 0.55 }));
    beacon.position.set(ring.position.x, ring.position.y + 3, ring.position.z); beacon.userData = { settlement: s.id }; setGroup.add(beacon);
  }
}

// ------------------------------------------------------------------ agentes (instancing + interpolação entre snapshots)
const agentGeo = new THREE.OctahedronGeometry(0.34, 0); agentGeo.scale(1.5, 0.85, 0.95);
const agentMat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.1, emissive: 0x000000 });
let agents = new THREE.InstancedMesh(agentGeo, agentMat, 32); agents.count = 0; agents.frustumCulled = false; world.add(agents);
const A = { n: 0, ids: new Float64Array(0), px: new Float32Array(0), py: new Float32Array(0), nx: new Float32Array(0), ny: new Float32Array(0),
            ph: new Float32Array(0), nh: new Float32Array(0), col: new Uint8Array(0), fid: new Uint8Array(0), en: new Uint8Array(0),
            cul: new Uint8Array(0), emit: new Uint8Array(0), slot: new Map() };
function curPos(i, alpha) { return [A.px[i] + (A.nx[i] - A.px[i]) * alpha, A.py[i] + (A.ny[i] - A.py[i]) * alpha]; }
function frameAlpha() { return Math.min(1, Math.max(0, (performance.now() - S.lastFrameWall) / Math.max(60, S.frameDt))); }
function ingestAgents(m) {
  const n = m.n, ids = asI64(m.ids), q = asU16(m.pos), hd = m.head, al = frameAlpha();
  const N = { ids: new Float64Array(n), px: new Float32Array(n), py: new Float32Array(n), nx: new Float32Array(n), ny: new Float32Array(n),
              ph: new Float32Array(n), nh: new Float32Array(n), slot: new Map() };
  for (let i = 0; i < n; i++) {
    const id = Number(ids[i]); N.ids[i] = id; N.slot.set(id, i);
    const x = q[i * 2] / 65535 * S.W, y = q[i * 2 + 1] / 65535 * S.H, h = hd[i] / 255 * Math.PI * 2;
    N.nx[i] = x; N.ny[i] = y; N.nh[i] = h;
    const j = A.slot.get(id);
    if (j !== undefined && Math.hypot(A.nx[j] - x, A.ny[j] - y) < 12) { const p = curPos(j, al); N.px[i] = p[0]; N.py[i] = p[1]; N.ph[i] = A.ph[j] + (A.nh[j] - A.ph[j]) * al; }
    else { N.px[i] = x; N.py[i] = y; N.ph[i] = h; }
  }
  Object.assign(A, N); A.n = n; A.col = m.color; A.fid = m.fid; A.en = m.energy; A.cul = m.culture; A.emit = m.emit;
  if (agents.instanceMatrix.count < n) { world.remove(agents); agents.dispose(); agents = new THREE.InstancedMesh(agentGeo, agentMat, Math.max(64, n * 1.5 | 0));
    agents.frustumCulled = false; world.add(agents); }
  agents.count = n; colorAgents();
}
const CIVIDIS = [[0, 32, 77], [64, 77, 107], [124, 123, 120], [188, 175, 111], [253, 231, 55]];
function ramp(t, P) { t = Math.min(0.999, Math.max(0, t)) * (P.length - 1); const i = Math.floor(t), f = t - i;
  return [P[i][0] + (P[i + 1][0] - P[i][0]) * f, P[i][1] + (P[i + 1][1] - P[i][1]) * f, P[i][2] + (P[i + 1][2] - P[i][2]) * f]; }
function colorAgents() {
  for (let i = 0; i < A.n; i++) {
    let r, g, b;
    if (S.mode === 'affect') { const e = A.en[i] / 255; [r, g, b] = ramp(e, CIVIDIS).map((v) => v / 255); }
    else if (S.mode === 'culture') { tmpC.setHSL(((A.cul[i] * 0.137) % 1), 0.75, A.cul[i] ? 0.55 : 0.3); r = tmpC.r; g = tmpC.g; b = tmpC.b; }
    else if (S.palette === 1) { [r, g, b] = ramp(A.col[i * 2] / 255, CIVIDIS).map((v) => v / 255); }
    else { tmpC.setHSL((A.col[i * 2] / 255) * 0.85, 0.55 + 0.4 * (A.col[i * 2 + 1] / 255), 0.55); r = tmpC.r; g = tmpC.g; b = tmpC.b; }
    tmpC.setRGB(r, g, b); agents.setColorAt(i, tmpC);
  }
  if (agents.instanceColor) agents.instanceColor.needsUpdate = true;
}
const yAxis = new THREE.Vector3(0, 1, 0);
function placeAgents() {
  const al = frameAlpha(); const night = nightFactor();
  for (let i = 0; i < A.n; i++) {
    const [x, y] = curPos(i, al); const h = A.ph[i] + (A.nh[i] - A.ph[i]) * al;
    const big = A.fid[i] >= 2 ? 1.55 : 1.0;
    tmpP.set(toX(x), hAt(x, y) + 0.32 * big, toZ(y)); tmpQ.setFromAxisAngle(yAxis, -h); tmpS.set(big, big, big);
    tmpM.compose(tmpP, tmpQ, tmpS); agents.setMatrixAt(i, tmpM);
  }
  agents.instanceMatrix.needsUpdate = true;
  agentMat.emissive.setRGB(0.15 * night, 0.25 * night, 0.4 * night);   // brilho noturno = emissão real de sinais (média)
}

// ------------------------------------------------------------------ seleção e linhas sociais
const selRing = new THREE.Mesh(new THREE.RingGeometry(0.7, 0.95, 32), new THREE.MeshBasicMaterial({ color: 0x7cf0c8, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }));
selRing.rotation.x = -Math.PI / 2; selRing.visible = false; world.add(selRing);
const socialLines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x9fd3ff, transparent: true, opacity: 0.7 }));
world.add(socialLines);
function selPos() { if (S.sel === null) return null; const i = A.slot.get(S.sel); if (i === undefined) return null; return curPos(i, frameAlpha()); }
const ray = new THREE.Raycaster(); const ndc = new THREE.Vector2();
function pick(ev) {
  const r = canvas.getBoundingClientRect(); ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hits = ray.intersectObjects(setGroup.children, false);
  let best = -1, bd = 1e9; const al = frameAlpha(); const o = ray.ray.origin, dvec = ray.ray.direction; const v = new THREE.Vector3();
  for (let i = 0; i < A.n; i++) {
    const [x, y] = curPos(i, al); v.set(toX(x), hAt(x, y) + 0.3, toZ(y)).sub(o);
    const t = v.dot(dvec); if (t < 0) continue; const d2 = v.lengthSq() - t * t; const tol = Math.max(0.6, t * 0.012);
    if (d2 < tol * tol && t < bd) { bd = t; best = i; }
  }
  if (best >= 0) { select(A.ids[best]); return; }
  if (hits.length) { openCity(hits[0].object.userData.settlement); }
}
let downAt = null;
canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', (e) => { if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 5) pick(e); downAt = null; });

async function select(aid) {
  S.sel = aid; S.follow = S.mode === 'agent'; $('inspector').classList.remove('hidden');
  send({ t: 'select', aid }); await refreshCard(); $('ins-title').textContent = 'agente #' + aid;
}
function deselect() { S.sel = null; S.card = null; send({ t: 'select', aid: null }); $('inspector').classList.add('hidden'); selRing.visible = false;
  socialLines.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3)); }
$('ins-close').onclick = deselect;
$('ins-follow').onclick = () => { S.follow = !S.follow; $('ins-follow').classList.toggle('on', S.follow); };
$('ins-fav').onclick = async () => { if (S.sel === null) return; await addFavorite({ type: 'agent', id: String(S.sel), label: 'agente #' + S.sel, tick: S.tick }); toast('★', 'agente marcado (não altera a pesquisa)'); };
$('ins-promote').onclick = async () => { if (!ADMIN) { toast('⚿', 'requer token admin (local)'); return; }
  try { await admin('focus', { aid: S.sel }); toast('F4', 'promoção registrada como intervenção do usuário'); } catch (e) { toast('⚠', String(e)); } };

function bar(label, v, color, right) { return `<div class="bl"><span>${label}</span><span>${right ?? (v * 100).toFixed(0) + '%'}</span></div><div class="bar"><div style="width:${Math.max(0, Math.min(1, v)) * 100}%;background:${color}"></div></div>`; }
function kv(el, obj) { el.innerHTML = Object.entries(obj).map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join(''); }
async function refreshCard() {
  if (S.sel === null) return;
  let c; try { c = await api('/api/agent/' + S.sel); } catch (e) { return; }
  S.card = c;
  if (!c.alive) { kv($('ins-body'), { estado: 'não vivo / agregado (F0)', ancestrais: (c.ancestors || []).length }); return; }
  kv($('ins-body'), { idade: c.age.toFixed(0) + ' ticks', geração: c.generation, fidelidade: 'F' + c.fidelity, genoma: c.genome_id.slice(0, 10),
    corpo: `${c.body.nodes} nós · v=${c.body.speed.toFixed(4)}`, filhos: `${c.living_children} vivos / ${c.children_total}`, cultura: c.culture_type >= 0 ? 'tipo ' + c.culture_type : '—' });
  $('ins-bars').innerHTML = bar('energia', c.energy, '#61e39a') + bar('integridade', c.integrity, '#6fb7ff') + bar('fadiga', c.fatigue, '#ffb65c') +
    bar('aprendizado |W−W₀|/|W₀|', Math.min(1, c.learning.weight_change_relative), '#b394ff', c.learning.weight_change_relative.toFixed(3));
  const af = c.affect || {};
  $('ins-affect').innerHTML = af.projection !== undefined
    ? `<span class="badge hyp" title="eixo de valência validado (A1) na linhagem; afeto FUNCIONAL, não experiência subjetiva">valência funcional ${af.projection.toFixed(2)}</span>`
    : `<span class="badge" title="nenhum eixo afetivo validado para esta linhagem">sem eixo afetivo validado</span>`;
  $('ins-tech').innerHTML = (c.techniques || []).map((t) => `<span class="badge">${t}</span>`).join('') +
    (c.evidence || []).map((e) => `<span class="badge ok" title="níveis validados por probes">${e.levels.join(' ')}</span>`).join('');
  drawLineage(c);
  const pts = [], p0 = selPos();
  if (p0) for (const s of c.social || []) if (s.pos) pts.push(toX(p0[0]), hAt(p0[0], p0[1]) + 0.5, toZ(p0[1]), toX(s.pos[0]), hAt(s.pos[0], s.pos[1]) + 0.5, toZ(s.pos[1]));
  socialLines.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts), 3));
  socialLines.visible = S.mode === 'social' || S.mode === 'agent';
}
function drawLineage(c) {
  const svg = $('lineage'), anc = (c.ancestors || []).slice(0, 8);
  let s = ''; const n = anc.length + 1, w = 300, y = 45;
  for (let i = 0; i < n; i++) {
    const x = w - 20 - i * ((w - 40) / Math.max(1, n - 1 || 1));
    const id = i === 0 ? c.aid : anc[i - 1];
    if (i > 0) s += `<line x1="${x}" y1="${y}" x2="${x + (w - 40) / Math.max(1, n - 1)}" y2="${y}" stroke="#2d4a77"/>`;
    s += `<circle cx="${x}" cy="${y}" r="${i === 0 ? 7 : 5}" fill="${i === 0 ? '#7cf0c8' : '#33507f'}"><title>#${id}</title></circle>`;
  }
  const kids = Math.min(10, c.living_children || 0);
  for (let k = 0; k < kids; k++) s += `<line x1="${w - 20}" y1="${y}" x2="${w - 40 + k * 4}" y2="${y + 28}" stroke="#2d4a77"/><circle cx="${w - 40 + k * 4}" cy="${y + 28}" r="3" fill="#61e39a"/>`;
  svg.innerHTML = s + `<text x="6" y="14" fill="#7f8ba3" font-size="10">ancestrais ← · → filhos vivos</text>`;
}

// ------------------------------------------------------------------ corpo (Matter Nodes reais ou gêmeo visual)
const bodyR = new THREE.WebGLRenderer({ canvas: $('bodycv'), antialias: true, alpha: true }); bodyR.setPixelRatio(Math.min(2, devicePixelRatio || 1));
const bodyScene = new THREE.Scene(); const bodyCam = new THREE.PerspectiveCamera(40, 300 / 220, 0.1, 100); bodyCam.position.set(6, 5, 7);
bodyScene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.1));
let bodyNodes = null, bodyEdges = null, bodyView = 'xray', lastBody = null;
const nodeGeo = new THREE.SphereGeometry(0.28, 10, 8);
function drawBody(m) {
  lastBody = m;
  const N = m.N, P = asF32(m.nodes), act = m.active, sig = m.signal, pr = m.props;
  if (!bodyNodes || bodyNodes.instanceMatrix.count < N) { if (bodyNodes) bodyScene.remove(bodyNodes);
    bodyNodes = new THREE.InstancedMesh(nodeGeo, new THREE.MeshStandardMaterial({ transparent: true, opacity: 0.85 }), N); bodyScene.add(bodyNodes); }
  let k = 0; const cx = [0, 0, 0]; let na = 0;
  for (let i = 0; i < N; i++) if (act[i]) { cx[0] += P[i * 3]; cx[1] += P[i * 3 + 2]; cx[2] += P[i * 3 + 1]; na++; }
  for (let j = 0; j < 3; j++) cx[j] /= Math.max(1, na);
  const devStep = bodyView === 'dev' ? +$('devslider').value : null;
  for (let i = 0; i < N; i++) {
    if (!act[i]) continue;
    if (devStep !== null && S.devFrames && !S.devFrames[Math.min(devStep, S.devFrames.length - 1)][i]) continue;
    tmpP.set(P[i * 3] - cx[0], P[i * 3 + 2] - cx[1], P[i * 3 + 1] - cx[2]); tmpS.set(1, 1, 1); tmpM.compose(tmpP, tmpQ.identity(), tmpS);
    bodyNodes.setMatrixAt(k, tmpM);
    if (bodyView === 'signal') { const [r, g, b] = ramp(sig[i] / 255, [[10, 10, 40], [40, 80, 200], [80, 220, 230], [250, 250, 120]]); tmpC.setRGB(r / 255, g / 255, b / 255); }
    else { const c = pr[i * 10 + 3] / 255, s2 = pr[i * 10 + 0] / 255, cnd = pr[i * 10 + 4] / 255; tmpC.setRGB(0.25 + 0.7 * c, 0.3 + 0.5 * cnd, 0.4 + 0.5 * s2); }
    bodyNodes.setColorAt(k, tmpC); k++;
  }
  bodyNodes.count = k; bodyNodes.instanceMatrix.needsUpdate = true; if (bodyNodes.instanceColor) bodyNodes.instanceColor.needsUpdate = true;
  bodyNodes.material.opacity = bodyView === 'xray' ? 0.55 : 0.95;
  const E = asI16(m.edges), pts = [];
  for (let e = 0; e < m.n_edges; e++) { const a = E[e * 2], b = E[e * 2 + 1];
    pts.push(P[a * 3] - cx[0], P[a * 3 + 2] - cx[1], P[a * 3 + 1] - cx[2], P[b * 3] - cx[0], P[b * 3 + 2] - cx[1], P[b * 3 + 1] - cx[2]); }
  if (bodyEdges) bodyScene.remove(bodyEdges);
  bodyEdges = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts), 3)),
    new THREE.LineBasicMaterial({ color: 0x6fb7ff, transparent: true, opacity: bodyView === 'xray' ? 0.6 : 0.25 }));
  if (bodyView !== 'dev') bodyScene.add(bodyEdges);
  // heatmap temporal do estado interno
  const T = m.state_T, NU = m.state_N, st = f16(m.state), cv = $('statecv'), ctx = cv.getContext('2d');
  const img = ctx.createImageData(T, NU);
  for (let t = 0; t < T; t++) for (let u = 0; u < NU; u++) { const v = Math.tanh(st[t * NU + u]) * 0.5 + 0.5; const [r, g, b] = ramp(v, [[30, 60, 160], [12, 14, 20], [230, 120, 60]]);
    const o = (u * T + t) * 4; img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255; }
  const off = new OffscreenCanvas ? new OffscreenCanvas(T, NU) : null;
  if (off) { off.getContext('2d').putImageData(img, 0, 0); ctx.imageSmoothingEnabled = false; ctx.clearRect(0, 0, cv.width, cv.height); ctx.drawImage(off, 0, 0, cv.width, cv.height); }
  $('ins-title').textContent = `agente #${m.aid} · F${m.fidelity}${m.visual_twin ? ' · gêmeo visual' : ''}`;
  $('ins-title').title = m.note;
}
document.querySelectorAll('#bodytabs button').forEach((b) => b.onclick = async () => {
  document.querySelectorAll('#bodytabs button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); bodyView = b.dataset.v;
  $('devslider').classList.toggle('hidden', bodyView !== 'dev');
  if (bodyView === 'dev' && S.sel !== null) await loadDev(S.sel);
  if (lastBody) drawBody(lastBody);
});
async function loadDev(aid) {
  try { const d = await api('/api/develop/' + aid);
    S.devFrames = d.frames.map((fb) => { const bits = new Uint8Array(d.N); for (let i = 0; i < d.N; i++) bits[i] = (fb[i >> 3] >> (7 - (i & 7))) & 1; return bits; });
    $('devslider').max = S.devFrames.length - 1; $('devslider').value = S.devFrames.length - 1; } catch (e) { S.devFrames = null; }
}
$('devslider').oninput = () => { if (lastBody) drawBody(lastBody); };

// ------------------------------------------------------------------ cidade
async function openCity(id) {
  let c; try { c = await api('/api/city/' + id); } catch (e) { return; }
  $('cityp').classList.remove('hidden'); $('city-title').textContent = `assentamento #${id} · ${c.state}`;
  kv($('city-kv'), { população: (c.pop ?? 0).toFixed ? c.pop.toFixed(0) : c.pop, blocos: (c.blocks ?? 0).toFixed(0), abrigos: (c.enclosures ?? 0).toFixed(0),
    persistência: c.persistence + ' janelas', fundação: c.founded ?? '—', ligações: (c.links || []).length });
  const tl = c.timeline || [], svg = $('citychart');
  if (tl.length > 1) {
    const mp = Math.max(...tl.map((t) => t.pop), 1), mb = Math.max(...tl.map((t) => t.blocks), 1);
    const line = (f, m, col) => '<polyline fill="none" stroke="' + col + '" stroke-width="1.6" points="' + tl.map((t, i) => `${10 + i * 280 / (tl.length - 1)},${100 - f(t) / m * 85}`).join(' ') + '"/>';
    svg.innerHTML = line((t) => t.pop, mp, '#61e39a') + line((t) => t.blocks, mb, '#ffb65c') +
      `<text x="10" y="12" fill="#61e39a" font-size="10">população</text><text x="90" y="12" fill="#ffb65c" font-size="10">blocos</text>` +
      (c.events || []).map((e) => `<circle cx="${10 + 280 * (tl.findIndex((t) => t.tick >= e.tick) / Math.max(1, tl.length - 1))}" cy="104" r="3" fill="#b394ff"><title>${e.type} @${e.tick}</title></circle>`).join('');
  } else svg.innerHTML = '';
  const s = S.settlements.find((x) => x.id === id);
  if (s) flyTo(s.centre[0], s.centre[1], 40);
}
$('city-close').onclick = () => $('cityp').classList.add('hidden');

// ------------------------------------------------------------------ câmeras
let flight = null;
function flyTo(x, y, dist) {
  const tgt = new THREE.Vector3(toX(x), hAt(x, y), toZ(y));
  const dir = camera.position.clone().sub(controls.target).normalize();
  const pos = tgt.clone().add(dir.multiplyScalar(dist));
  pos.y = Math.max(pos.y, tgt.y + dist * 0.45);
  flight = { t0: performance.now(), from: camera.position.clone(), to: pos, tf: controls.target.clone(), tt: tgt, dur: 1100 };
}
function setMode(m) {
  S.mode = m; document.querySelectorAll('#modes button[data-mode]').forEach((b) => b.classList.toggle('on', b.dataset.mode === m));
  const d = Math.max(S.W, S.H);
  if (m === 'planet') flyTo(S.W / 2, S.H / 2, d * 1.1);
  if (m === 'city') { const s = [...S.settlements].sort((a, b) => (b.pop || 0) - (a.pop || 0))[0]; if (s) { flyTo(s.centre[0], s.centre[1], 45); openCity(s.id); } else toast('🏙', 'nenhum assentamento detectado ainda (não inventamos cidades)'); }
  if (m === 'street') { if (blockCount) { let bx = 0, by = 0; const any = blockIdx.keys().next().value; bx = Math.floor(any / 100000); by = any % 100000; flyTo(bx, by, 14); } else toast('🧱', 'nenhuma construção real ainda'); }
  if (m === 'agent') { S.follow = true; if (S.sel === null && A.n) select(A.ids[0]); }
  if (['body', 'signal', 'development'].includes(m)) { if (S.sel === null && A.n) select(A.ids[0]);
    const v = m === 'body' ? 'xray' : m === 'signal' ? 'signal' : 'dev'; document.querySelector(`#bodytabs button[data-v="${v}"]`)?.click(); }
  if (m === 'social') { socialLines.visible = true; if (S.sel === null && A.n) select(A.ids[0]); }
  if (m === 'culture') setLayer('culture'); if (m === 'affect') toast('◐', 'cor = energia (variável fisiológica real); afeto funcional só aparece com eixo validado');
  if (m === 'lab') { $('labp').classList.remove('hidden'); send({ t: 'lab', on: true }); } else { send({ t: 'lab', on: false }); }
  colorAgents();
}
document.querySelectorAll('#modes button[data-mode]').forEach((b) => b.onclick = () => setMode(b.dataset.mode));
$('lab-close').onclick = () => { $('labp').classList.add('hidden'); send({ t: 'lab', on: false }); };

// ------------------------------------------------------------------ camadas
const LAYERS = ['population', 'resources', 'activity', 'species', 'culture', 'communication', 'settlements', 'novelty', 'temperature', 'storm'];
$('layerbtns').innerHTML = '<button data-l="">nenhuma</button>' + LAYERS.map((l) => `<button data-l="${l}">${l}</button>`).join('');
function setLayer(l) { S.layer = l || null; heatMesh && (heatMesh.visible = !!l); send({ t: 'layer', name: l || '' });
  document.querySelectorAll('#layerbtns button').forEach((b) => b.classList.toggle('on', b.dataset.l === (l || ''))); }
document.querySelectorAll('#layerbtns button').forEach((b) => b.onclick = () => setLayer(b.dataset.l));
function applyHeat(m) {
  const R = m.R, d = m.data; if (heatTex.image.width !== R) { heatTex.image = { data: new Uint8Array(R * R * 4), width: R, height: R }; }
  const P = m.layer === 'temperature' ? [[40, 70, 200], [120, 200, 220], [250, 220, 90], [230, 70, 40]] : [[8, 12, 40], [40, 90, 200], [90, 220, 200], [250, 240, 120]];
  for (let i = 0; i < R * R; i++) { const v = d[i] / 255; const [r, g, b] = ramp(v, P); const o = i * 4;
    const yy = Math.floor(i / R), xx = i % R, oi = ((R - 1 - yy) * R + xx) * 4;
    heatTex.image.data[oi] = r; heatTex.image.data[oi + 1] = g; heatTex.image.data[oi + 2] = b; heatTex.image.data[oi + 3] = v > 0.02 ? 120 + 135 * v : 0; }
  heatTex.needsUpdate = true;
}
function applyStorm(b) {
  const d = stormTex.image.data;
  for (let i = 0; i < 32 * 32; i++) { const v = b[i] / 255, yy = Math.floor(i / 32), xx = i % 32, o = ((31 - yy) * 32 + xx) * 4;
    d[o] = 60; d[o + 1] = 66; d[o + 2] = 80; d[o + 3] = v > 0.05 ? Math.min(200, v * 260) : 0; }
  stormTex.needsUpdate = true;
}

// ------------------------------------------------------------------ dia/noite (fase real do ciclo artificial)
function nightFactor() { const s = Math.cos((S.dayPhase) * Math.PI * 2); return Math.max(0, Math.min(1, 0.5 - s)); }
function updateSky() {
  const ph = S.dayPhase * Math.PI * 2, el = Math.sin(ph) * 0.9 + 0.1;
  sun.position.set(Math.cos(ph) * 400, Math.max(-50, el * 400), 120); sun.intensity = Math.max(0.05, el) * 1.4;
  hemi.intensity = 0.25 + 0.55 * Math.max(0, el);
  const n = nightFactor(); const c = DAY.clone().lerp(DUSK, Math.min(1, n * 1.6)).lerp(NIGHT, Math.max(0, n * 1.4 - 0.4));
  if (!S.science) { scene.background.copy(c); scene.fog.color.copy(c); } else { scene.background.set(0x0a0f1a); scene.fog.color.set(0x0a0f1a); }
  waterMat.uniforms.night.value = n;
}

// ------------------------------------------------------------------ rede (WebSocket com heartbeat, reconexão e backpressure)
let ws = null, retry = 0, lastMsg = 0, connectedOnce = false;
function send(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/live?t=${encodeURIComponent(TOKEN)}`); ws.binaryType = 'arraybuffer';
  ws.onopen = () => { retry = 0; connectedOnce = true; $('offline').classList.add('hidden'); if (S.sel !== null) send({ t: 'select', aid: S.sel });
    if (S.layer) send({ t: 'layer', name: S.layer }); sendView(); };
  ws.onmessage = (ev) => { lastMsg = performance.now();
    if (typeof ev.data === 'string') { const m = JSON.parse(ev.data); if (m.t === 'status') onStatus(m); else if (m.t === 'pong') S.lat = performance.now() - m.ts; else if (m.t === 'hello') onHello(m); return; }
    const m = decode(new Uint8Array(ev.data));
    if (m.t === 'base') onBase(m); else if (m.t === 'frame') onFrame(m); else if (m.t === 'body') drawBody(m); else if (m.t === 'heat') applyHeat(m); else if (m.t === 'lab') drawLab(m); };
  ws.onclose = (e) => { if (e.code === 4401) { setLive('off', 'TOKEN?'); $('loadmsg').textContent = 'token inválido ou ausente — abra o link completo gerado pelo notebook'; return; }
    setLive('off', 'OFFLINE'); const wait = Math.min(15000, 600 * Math.pow(1.7, retry++)); setTimeout(connect, wait);
    if (retry > 3) showOffline(); };
  ws.onerror = () => { };
}
function onHello(m) { $('loadmsg').textContent = `protocolo ${m.proto.version} · servidor ${m.version}`; }
function onBase(m) { buildTerrain(m); loadBlocks(m.struct, m.struct_n); drawSettlements(m.settlements); S.roads = m.roads || []; S.legend = m.legend;
  $('loading').classList.add('fade'); setTimeout(() => $('loading').classList.add('hidden'), 900); }
function onFrame(m) {
  if (!S.base) return;
  const now = performance.now(); if (S.lastFrameWall) S.frameDt = 0.8 * S.frameDt + 0.2 * (now - S.lastFrameWall); S.lastFrameWall = now;
  S.tick = m.tick; S.dayPhase = m.light_phase;
  if (!S.replay) ingestAgents(m);
  if (m.sdelta) applyDelta(m.sdelta, m.sdelta_n);
  if (m.storm) applyStorm(m.storm);
  for (const e of m.events || []) onEvent(e);
}
function setLive(cls, txt) { const el = $('live'); el.classList.remove('on', 'stale', 'off'); el.classList.add(cls); $('livetxt').textContent = txt; }
function showOffline() {
  const st = store.get('laststatus', null); const ck = st && st.checkpoint && st.checkpoint.ts ? st.checkpoint.ts : 'desconhecido';
  $('offline').textContent = `SIMULATION OFFLINE — LAST PERSISTED STATE AT ${ck}`; $('offline').classList.remove('hidden');
}
function fmtB(b) { return b > 1e9 ? (b / 1e9).toFixed(1) + 'G' : (b / 1e6).toFixed(0) + 'M'; }
function onStatus(m) {
  S.status = m; store.set('laststatus', { checkpoint: m.checkpoint, tick: m.tick, ts: Date.now() });
  $('state').textContent = m.paused ? 'PAUSED' : m.state; $('tick').textContent = m.tick.toLocaleString(); $('tps').textContent = ` ${m.tps}/s`;
  $('pop').textContent = m.population.toLocaleString(); $('f0').textContent = m.f0_population > 0 ? ` +${Math.round(m.f0_population)} F0` : '';
  $('gen').textContent = `${m.generation} · QD ${m.qd_generations}`;
  $('levels').textContent = `H${m.h_level >= 0 ? m.h_level : '–'} A${m.a_level >= 0 ? m.a_level : '–'} C${m.c_level >= 0 ? m.c_level : '–'}`;
  $('ladderchip').title = `${m.h_name} · ${m.a_name} · ${m.c_name}\nnível = maior nível consecutivo validado com controles`;
  drawRings(m);
  $('gpu').textContent = m.gpu && m.gpu.total ? `${fmtB(m.gpu.allocated)}/${fmtB(m.gpu.total)}${m.gpu.util !== undefined ? ' · ' + m.gpu.util + '%' : ''}` : 'CPU';
  const ck = m.checkpoint || {}; $('ckpt').textContent = ck.ts ? `${Math.round((Date.now() - Date.parse(ck.ts)) / 60000)} min` : '—';
}
function drawRings(m) {
  const lv = [[m.h_level, 13, '#6fb7ff'], [m.a_level, 10, '#b394ff'], [m.c_level, 10, '#7cf0c8']];
  let s = ''; lv.forEach(([l, max, col], i) => { const cx = 10 + i * 22, r = 8, frac = Math.max(0, (l + 1) / max), a = frac * Math.PI * 2;
    s += `<circle cx="${cx}" cy="14" r="${r}" fill="none" stroke="#1d2a40" stroke-width="3"/>`;
    if (frac > 0) { const x = cx + r * Math.sin(a), y = 14 - r * Math.cos(a); s += `<path d="M ${cx} ${14 - r} A ${r} ${r} 0 ${a > Math.PI ? 1 : 0} 1 ${x} ${y}" fill="none" stroke="${col}" stroke-width="3"/>`; } });
  $('rings').innerHTML = s;
}
setInterval(() => { send({ t: 'ping', ts: performance.now() }); const age = (performance.now() - S.lastFrameWall) / 1000;
  $('lat').textContent = `${S.lat ? S.lat.toFixed(0) + ' ms' : '—'} · frame ${S.lastFrameWall ? age.toFixed(1) + ' s' : '—'}`;
  if (ws && ws.readyState === 1) { if (S.replay) setLive('stale', 'REPLAY'); else if (age > 5) setLive('stale', 'STALE'); else setLive('on', 'LIVE'); } }, 2000);
function sendView() {
  const t = controls.target, d = camera.position.distanceTo(t) * 1.2;
  send({ t: 'view', x0: t.x + S.W / 2 - d, y0: t.z + S.H / 2 - d, x1: t.x + S.W / 2 + d, y1: t.z + S.H / 2 + d, max_agents: S.maxAgents, hz: MOBILE ? 4 : 8 });
}
setInterval(sendView, 1000);
setInterval(() => { if (S.sel !== null) refreshCard(); }, 2500);

// ------------------------------------------------------------------ feed de eventos e descobertas
const ICON = { science_discovery: '✦', civilization_discovery: '✶', technology_discovery: '⚒', settlement_founded: '⌂', city_detected: '🏙',
  settlement_collapse: '✕', settlement_recovery: '↺', house_like_structure: '⌂', cumulative_chain: '⛓', extinction: '☠', recovery: '↺',
  recovery_injection: '⇣', immigration: '⇢', population_shift: '⇅', migration: '➚', species_split: '⑂', ETHICAL_REVIEW_TRIGGER: '⚠',
  counterfactual_fork: '⑃', civilization_level: '✶', world_created: '◎' };
function toast(ico, text, cls, onclick) {
  const el = document.createElement('div'); el.className = 'fe ' + (cls || ''); el.innerHTML = `<span class="ico">${ico}</span><span>${text}</span>`;
  if (onclick) el.onclick = onclick; $('feed').prepend(el); setTimeout(() => el.remove(), cls === 'disc' ? 20000 : 9000);
  while ($('feed').children.length > 6) $('feed').lastChild.remove();
}
function onEvent(e) {
  S.events.push(e); if (S.events.length > 800) S.events.shift();
  const d = e.data || {}; const ico = ICON[e.type] || '•';
  if (e.type === 'science_discovery') toast(ico, `descoberta ${d.level_name || d.level} · ${d.claim || ''}`, 'disc', () => openReplay(d.id));
  else if (e.type === 'civilization_discovery') toast(ico, `nível civilizacional: ${d.level} (hipótese de simulação)`, 'disc');
  else if (e.type === 'ETHICAL_REVIEW_TRIGGER') toast(ico, 'ETHICAL_REVIEW_TRIGGER (sinal p/ revisão; não declara sofrimento)', 'eth');
  else if (['settlement_founded', 'city_detected', 'technology_discovery', 'house_like_structure', 'settlement_collapse', 'cumulative_chain'].includes(e.type))
    toast(ico, e.type.replace(/_/g, ' ') + (d.tech ? ' · ' + d.tech : ''), '', () => e.pos && flyTo(e.pos[0], e.pos[1], 30));
  if (e.type.startsWith('settlement') || e.type === 'city_detected') { /* base trará os anéis atualizados */ send({ t: 'resync' }); }
  drawTimelineEvents();
}

// ------------------------------------------------------------------ timeline (anel de snapshots reais) e replay
let ring = [];
async function refreshRing() { try { ring = await api('/api/ring'); $('tl').max = Math.max(0, ring.length - 1); if (!S.replay) $('tl').value = $('tl').max; drawTimelineEvents(); } catch (e) { /* offline */ } }
setInterval(refreshRing, 5000);
function drawTimelineEvents() {
  if (!ring.length) return; const t0 = ring[0].tick, t1 = Math.max(t0 + 1, S.tick);
  $('tl-events').innerHTML = S.events.filter((e) => e.tick >= t0).slice(-120).map((e) => `<span class="${e.type.includes('discovery') ? 'd' : ''}" style="left:${(e.tick - t0) / (t1 - t0) * 100}%" title="${e.type} @${e.tick}"></span>`).join('');
}
$('tl').oninput = async () => {
  const i = +$('tl').value; if (i >= ring.length - 1) { goLive(); return; }
  S.replay = true; $('tl-live').classList.remove('on');
  try { const r = await api('/api/ring/' + i); ingestAgents({ n: r.n, ids: r.ids, pos: r.pos, head: new Uint8Array(r.n), color: r.color, fid: new Uint8Array(r.n),
    energy: new Uint8Array(r.n).fill(200), culture: new Uint8Array(r.n), emit: new Uint8Array(r.n) }); S.lastFrameWall = 0; toast('⏪', `replay tick ${r.tick}`); } catch (e) { /* sem frame */ }
};
function goLive() { S.replay = false; $('tl-live').classList.add('on'); $('tl').value = $('tl').max; }
$('tl-live').onclick = goLive; $('tl-live').classList.add('on');

const rpR = new THREE.WebGLRenderer({ canvas: $('rpcv'), antialias: true }); rpR.setPixelRatio(Math.min(2, devicePixelRatio || 1));
const rpScene = new THREE.Scene(); rpScene.background = new THREE.Color(0x02040a);
const rpCam = new THREE.PerspectiveCamera(45, 640 / 420, 0.1, 500); rpScene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.0));
let RP = null;
async function openReplay(id) {
  if (!id) return; let r; try { r = await api('/api/replay/' + id); } catch (e) { toast('⚠', 'sem replay para ' + id); return; }
  RP = { r, pos: asF32(r.pos), t: 0, playing: true, k: r.k, T: r.ticks };
  $('replay').classList.remove('hidden'); $('rp-title').textContent = `${r.name || r.level} · ${id}`; $('rpslider').max = r.ticks - 1;
  kv($('rp-kv'), { arenas: (r.layout || []).length, agentes: r.k, ticks: r.ticks, origem: 'execução real do probe (sementes de holdout)' });
  while (rpScene.children.length > 1) rpScene.remove(rpScene.children[1]);
  const L = r.layout || [], cols = Math.ceil(Math.sqrt(Math.max(1, L.length))), sz = r.size, gap = sz + 3;
  RP.off = {};
  L.forEach((ar, i) => { const ox = (i % cols) * gap, oz = Math.floor(i / cols) * gap; RP.off[ar.arena] = [ox, oz];
    const floor = new THREE.Mesh(new THREE.BoxGeometry(sz, 0.2, sz), new THREE.MeshStandardMaterial({ color: 0x1c2a22 })); floor.position.set(ox + sz / 2, -0.1, oz + sz / 2); rpScene.add(floor);
    const addT = (list, col, h, y) => { for (const [ty, tx] of list || []) { const b = new THREE.Mesh(new THREE.BoxGeometry(0.95, h, 0.95), new THREE.MeshStandardMaterial({ color: col, transparent: col === 0x2a6fd6, opacity: 0.8 }));
      b.position.set(ox + tx + 0.5, y + h / 2, oz + ty + 0.5); rpScene.add(b); } };
    addT(ar.water, 0x2a6fd6, 0.1, 0); addT(ar.filled, 0x3aa59d, 0.25, 0); addT(ar.struct, 0x9c8a6c, 0.8, 0); addT(ar.rich, 0xd6b02a, 0.06, 0);
    for (const [py, px] of ar.patches || []) if (py >= 0) { const d = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.2, 0.08, 16), new THREE.MeshBasicMaterial({ color: 0x51d17a, transparent: true, opacity: 0.6 })); d.position.set(ox + px + 0.5, 0.05, oz + py + 0.5); rpScene.add(d); }
  });
  RP.dots = new THREE.InstancedMesh(new THREE.SphereGeometry(0.35, 10, 8), new THREE.MeshStandardMaterial({ color: 0xffffff }), RP.k); rpScene.add(RP.dots);
  for (let j = 0; j < RP.k; j++) { tmpC.setHSL(j / RP.k, 0.7, 0.6); RP.dots.setColorAt(j, tmpC); }
  const W2 = cols * gap; rpCam.position.set(W2 / 2, W2 * 0.9, W2 * 1.1); rpCam.lookAt(W2 / 2, 0, W2 / 2.4);
  RP.arenaOf = asI16(r.arena_of);
}
$('rp-close').onclick = () => { $('replay').classList.add('hidden'); RP = null; };
$('rpslider').oninput = () => { if (RP) { RP.t = +$('rpslider').value; RP.playing = false; } };
function renderReplay(dt) {
  if (!RP) return; if (RP.playing) { RP.t = (RP.t + dt * 0.03) % RP.T; $('rpslider').value = RP.t | 0; }
  const t = RP.t | 0;
  for (let j = 0; j < RP.k; j++) { const o = RP.off[RP.arenaOf[j]] || [0, 0]; const x = RP.pos[(t * RP.k + j) * 2], y = RP.pos[(t * RP.k + j) * 2 + 1];
    tmpP.set(o[0] + x, 0.4, o[1] + y); tmpS.set(1, 1, 1); tmpM.compose(tmpP, tmpQ.identity(), tmpS); RP.dots.setMatrixAt(j, tmpM); }
  RP.dots.instanceMatrix.needsUpdate = true; if (RP.dots.instanceColor) RP.dots.instanceColor.needsUpdate = true; rpR.render(rpScene, rpCam);
}

// ------------------------------------------------------------------ câmera do cientista (experimento em curso)
function drawLab(m) {
  const cv = $('labcv'), ctx = cv.getContext('2d'); ctx.fillStyle = '#02040a'; ctx.fillRect(0, 0, cv.width, cv.height);
  const n = m.pos_n, P = asF32(m.pos), alive = m.alive, en = m.energy ? asF32(m.energy) : null; const sc = cv.width / Math.max(m.W, m.H);
  if (m.struct) { const st = asI16 ? null : null; }
  ctx.strokeStyle = '#1d2a40';
  const cell = m.size + 2; for (let i = 0; i <= Math.ceil(Math.sqrt(m.n_arenas)); i++) { ctx.beginPath(); ctx.moveTo(i * cell * sc, 0); ctx.lineTo(i * cell * sc, cv.height); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, i * cell * sc); ctx.lineTo(cv.width, i * cell * sc); ctx.stroke(); }
  for (let i = 0; i < n; i++) { if (!alive[i]) continue; const e = en ? en[i] : 0.5; ctx.fillStyle = `hsl(${120 * e},70%,60%)`; ctx.beginPath(); ctx.arc(P[i * 2] * sc, P[i * 2 + 1] * sc, 2.5, 0, 7); ctx.fill(); }
  $('lab-title').textContent = 'laboratório · ' + (m.name || '');
  kv($('lab-kv'), { tick: m.tick, arenas: m.n_arenas, agentes: n, nota: 'ensaio real em curso (mesma física do planeta)' });
}

// ------------------------------------------------------------------ busca, favoritos, ajuda, captura, modos visuais
$('search').addEventListener('keydown', async (e) => {
  if (e.key !== 'Enter') return; const q = $('search').value.trim(); if (!/^[A-Za-z0-9_\-]{1,64}$/.test(q)) return;
  try { const r = await api('/api/search?q=' + encodeURIComponent(q)); const h = (r.results || [])[0];
    if (!h) { toast('⌕', 'nada encontrado'); return; }
    if (h.type === 'agent' || h.type === 'genome') { if (h.pos) flyTo(h.pos[0], h.pos[1], 18); select(h.agent || h.id); }
    else if (h.type === 'settlement') openCity(h.id); else if (h.type === 'discovery') openReplay(h.id);
    else toast('⌕', `${h.type} ${h.id} ${(h.levels || []).join(' ')}`); } catch (err) { toast('⚠', 'busca falhou'); }
});
async function addFavorite(item) { let fav = []; try { fav = await api('/api/favorites'); } catch (e) { fav = []; } fav.push(Object.assign({ ts: Date.now() }, item));
  await api('/api/favorites', { method: 'POST', body: JSON.stringify(fav.slice(-500)), headers: { 'Content-Type': 'application/json' } }); }
$('btn-help').onclick = () => { $('help').classList.remove('hidden'); $('helpbody').innerHTML = HELP; };
$('help-close').onclick = () => $('help').classList.add('hidden');
$('btn-shot').onclick = () => { renderer.render(scene, camera); const a = document.createElement('a'); a.href = canvas.toDataURL('image/png'); a.download = `genesis_tick${S.tick}.png`; a.click(); };
$('btn-sci').onclick = () => { S.science = !S.science; store.set('science', S.science); if (grid) grid.visible = S.science; scene.fog.near = S.science ? 1e6 : Math.max(S.W, S.H) * 1.2;
  controls.autoRotate = !S.science && store.get('cinema', false); toast('◈', S.science ? 'modo ciência: sem efeitos, grade e escala' : 'modo cinema: visual sem alterar a simulação'); };
$('btn-pal').onclick = () => { S.palette = 1 - S.palette; store.set('palette', S.palette); colorAgents(); toast('◑', S.palette ? 'paleta acessível (cividis)' : 'paleta padrão'); };
$('btn-tour').onclick = async () => { let d; try { d = await api('/api/discoveries'); } catch (e) { return; }
  const list = (d.discoveries || []).filter((x) => x.kind !== 'civilization').slice(-6); if (!list.length) { toast('✦', 'nenhuma descoberta validada ainda'); return; }
  for (let i = 0; i < list.length; i++) setTimeout(() => openReplay(list[i].id), i * 9000); };
const HELP = `<p><b>GENESIS CIVILIZATION Ω∞ é um universo artificial</b>, não uma simulação atomística do Universo real. Nada aqui afirma vida real,
consciência, sentimentos subjetivos, AGI ou humanidade real. Tudo que aparece vem do estado da simulação.</p>
<h3>controles</h3><ul><li>mouse/toque: arrastar = orbitar · roda/pinça = zoom · botão direito/dois dedos = mover</li>
<li>teclado: WASD/setas mover · Q/E girar · +/− zoom · F seguir · Esc desmarcar · L ao vivo · H ajuda</li><li>gamepad: alavanca esquerda move, direita gira</li></ul>
<h3>cores</h3><ul><li>agentes: assinatura visual genética (identidade), não semântica · tamanho maior = corpo físico F2+</li>
<li>terreno: f(altura, temperatura, umidade, fertilidade) · neve = frio · água animada</li><li>blocos: matéria construída real · verde-azulado = aterro sobre água</li>
<li>anéis no chão: assentamentos detectados (cinza candidato · ciano detectado · violeta cidade · vermelho colapso)</li></ul>
<h3>níveis</h3><ul><li><b>H</b> desenvolvimento → mente → cultura (H0–H12) · <b>A</b> afeto funcional (A0–A9) · <b>C</b> civilização (C0–C9)</li>
<li>um nível só conta com probe com controles negativos, replicação em sementes de holdout e ablação quando aplicável</li>
<li>status: IMPLEMENTED · TESTED · BENCHMARKED · SIMULATION HYPOTHESIS · NOT YET VALIDATED</li></ul>
<h3>afeto</h3><p>“functional affect” = estado interno com papel funcional demonstrado causalmente; <b>não</b> é “o agente sente”.</p>
<h3>tempo</h3><p>tick = unidade artificial; “dia” e “estação” são ciclos artificiais, não calendário humano.</p>`;

// teclado e gamepad
const keys = new Set(); addEventListener('keydown', (e) => { if (e.target.tagName === 'INPUT') return; keys.add(e.key.toLowerCase());
  if (e.key === 'Escape') deselect(); if (e.key.toLowerCase() === 'f') S.follow = !S.follow; if (e.key.toLowerCase() === 'l') goLive(); if (e.key.toLowerCase() === 'h') $('btn-help').click(); });
addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
function keyNav(dt) {
  const sp = camera.position.distanceTo(controls.target) * 0.9 * dt; const f = new THREE.Vector3(); camera.getWorldDirection(f); f.y = 0; f.normalize();
  const r = new THREE.Vector3().crossVectors(f, yAxis); const mv = new THREE.Vector3();
  if (keys.has('w') || keys.has('arrowup')) mv.add(f); if (keys.has('s') || keys.has('arrowdown')) mv.sub(f);
  if (keys.has('d') || keys.has('arrowright')) mv.add(r); if (keys.has('a') || keys.has('arrowleft')) mv.sub(r);
  const gp = navigator.getGamepads ? [...navigator.getGamepads()].find((g) => g) : null;
  if (gp) { mv.addScaledVector(r, gp.axes[0] || 0).addScaledVector(f, -(gp.axes[1] || 0)); if (Math.abs(gp.axes[2] || 0) > 0.15) rot(gp.axes[2] * dt * 1.5); }
  if (mv.lengthSq()) { mv.normalize().multiplyScalar(sp); camera.position.add(mv); controls.target.add(mv); }
  if (keys.has('q')) rot(dt); if (keys.has('e')) rot(-dt);
  if (keys.has('+') || keys.has('=')) zoom(0.97); if (keys.has('-')) zoom(1.03);
}
function rot(a) { const o = camera.position.clone().sub(controls.target); o.applyAxisAngle(yAxis, a); camera.position.copy(controls.target).add(o); }
function zoom(k) { const o = camera.position.clone().sub(controls.target).multiplyScalar(k); camera.position.copy(controls.target).add(o); }

// WebXR opcional (não é dependência)
if (navigator.xr && navigator.xr.isSessionSupported) {
  navigator.xr.isSessionSupported('immersive-vr').then((ok) => { if (!ok) return; $('btn-xr').style.display = '';
    $('btn-xr').onclick = async () => { renderer.xr.enabled = true; const sess = await navigator.xr.requestSession('immersive-vr', { optionalFeatures: ['local-floor'] });
      world.scale.setScalar(1.2 / Math.max(S.W, S.H)); world.position.set(0, 0.9, -1.2); await renderer.xr.setSession(sess);
      sess.addEventListener('end', () => { world.scale.setScalar(1); world.position.set(0, 0, 0); renderer.xr.enabled = false; }); }; }).catch(() => { });
}

// ------------------------------------------------------------------ laço de renderização (cliente faz o trabalho visual)
let last = performance.now(), fpsAcc = 0, fpsN = 0, fpsT = performance.now();
function frame() {
  const now = performance.now(), dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (flight) { const k = Math.min(1, (now - flight.t0) / flight.dur), e = k * k * (3 - 2 * k); camera.position.lerpVectors(flight.from, flight.to, e); controls.target.lerpVectors(flight.tf, flight.tt, e); if (k >= 1) flight = null; }
  const sp = selPos();
  if (sp) { selRing.visible = true; selRing.position.set(toX(sp[0]), hAt(sp[0], sp[1]) + 0.08, toZ(sp[1]));
    if (S.follow && !flight) { const tgt = new THREE.Vector3(toX(sp[0]), hAt(sp[0], sp[1]), toZ(sp[1])); const d = tgt.clone().sub(controls.target); controls.target.add(d); camera.position.add(d); } }
  else selRing.visible = false;
  keyNav(dt); controls.update(); placeAgents(); updateSky(); waterMat.uniforms.t.value = now / 1000;
  if (clouds) clouds.visible = !S.science;
  renderer.render(scene, camera);
  if (S.sel !== null && lastBody) { if (bodyNodes) bodyScene.rotation.y += dt * 0.4; bodyCam.lookAt(0, 0, 0); bodyR.render(bodyScene, bodyCam); }
  if (RP) renderReplay(dt * 1000);
  fpsAcc += dt; fpsN++;
  if (now - fpsT > 2000) { const fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; fpsT = now;
    if (fps < 28 && pixelRatio > 0.75) { pixelRatio = Math.max(0.75, pixelRatio - 0.25); renderer.setPixelRatio(pixelRatio); S.maxAgents = Math.max(300, S.maxAgents * 0.75 | 0); }
    else if (fps > 55 && pixelRatio < Math.min(2, devicePixelRatio || 1)) { pixelRatio = Math.min(2, pixelRatio + 0.25); renderer.setPixelRatio(pixelRatio); S.maxAgents = Math.min(MOBILE ? 3000 : 20000, S.maxAgents * 1.2 | 0); } }
}
renderer.setAnimationLoop(frame);
addEventListener('resize', () => { renderer.setSize(innerWidth, innerHeight); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); });
if (!TOKEN) { $('loadmsg').textContent = 'token ausente: abra o link "OPEN GENESIS WORLD" gerado pelo notebook'; }
connect(); refreshRing();
window.__genesis = { S, A, get blocks() { return blockCount; }, select, openReplay, setMode };   // depuração/testes do frontend
