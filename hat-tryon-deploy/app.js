import * as THREE from "./vendor/three.module.min.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";
import { RoomEnvironment } from "./vendor/RoomEnvironment.js";
import { HATS, A, B } from "./hats.js?v=f4940b4";
import { loadHeadModel, HeadFitter } from "./headshape.js?v=f4940b4";

/* MediaPipe Tasks Vision: bundled copy first (works offline), CDNs as fallback */
let FaceLandmarker, ImageSegmenter, FilesetResolver;
const LOCAL = location.protocol !== "file:";
const MP_VER = "0.10.14";
for (const src of [
  ...(LOCAL ? ["./vendor/vision_bundle.mjs"] : []),
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/vision_bundle.mjs`,
  `https://unpkg.com/@mediapipe/tasks-vision@${MP_VER}/vision_bundle.mjs`,
]) {
  try { ({ FaceLandmarker, ImageSegmenter, FilesetResolver } = await import(src)); break; }
  catch (e) { console.warn("MediaPipe import failed from", src, e); }
}
const WASM_URLS = [
  ...(LOCAL ? [new URL("./vendor/wasm", location.href).href] : []),
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/wasm`,
  `https://unpkg.com/@mediapipe/tasks-vision@${MP_VER}/wasm`,
];
const FACE_MODEL_URLS = [
  "./models/face_landmarker.task",
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
];
const SEG_MODEL_URLS = [
  "./models/selfie_multiclass_256x256.tflite",
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
];

/* ---------------- DOM ---------------- */
const $ = id => document.getElementById(id);
const out = $("out"), ovl = $("ovl"), octx = ovl.getContext("2d"), vp = $("viewport");
const video = $("video");
const statusEl = $("status"), fpsEl = $("fps");
const setStatus = (t, cls = "") => { statusEl.textContent = t; statusEl.className = "pill " + cls; };

/* ---------------- Settings ---------------- */
const S = { bright: 1, shadow: 0.6, occlude: true, headShape: true, debug: false, keyBg: true, tol: 38, tuck: 0, showOrig: false };
const FIT_DEFAULT = { size: 1, y: 0, x: 0, tilt: 0, stretch: 1, curve: 0 };
const FIT_KEYS = ["size", "y", "x", "tilt", "stretch", "curve"];
const SIT = -0.08;          // default hat line at the forehead: head widths above (+) the top-of-forehead landmark
const HAT_SCALE = 1.13;      // hats are modelled on the bare skull; hair makes the head bigger
const CROWN_SCALE = 1.3;    // hat models are built low; real crowns are about 0.6 head widths tall
let hairScale = HAT_SCALE, hairTop = null;
   // hairTop: top of the hair above the ear line, in head widths   // the same, measured from the hair in the picture when the segmenter is available
const FOV_Y = 63 * Math.PI / 180;   // camera the face tracker assumes

const hats = HATS.map(h => ({ ...h, base: { ...FIT_DEFAULT, ...h.fit }, fit: { ...FIT_DEFAULT, ...h.fit }, group: null }));
let hat = hats[0];

/* ---------------- 3D scene ----------------
   World: frame pixels, x right, y up, z towards the camera, seen through an orthographic camera. The head is a group at
   the midpoint between the ears, scaled to the head width and turned with the tracked rotation, so everything inside
   it is in head widths. It holds an invisible head shape (hides what is behind the head, catches the hat's shadow)
   and the hat. */
const renderer = new THREE.WebGLRenderer({ canvas: out, antialias: true, alpha: false, preserveDrawingBuffer: false, powerPreference: "high-performance" });
renderer.setPixelRatio(1);
const scene = new THREE.Scene(), cam = new THREE.OrthographicCamera(0, 1, 1, 0, 1, 12000); cam.position.z = 6000;
let bgTex = null, bgKey = "";
const maskTex = new THREE.DataTexture(new Uint8Array([0]), 1, 1, THREE.RedFormat);
maskTex.minFilter = maskTex.magFilter = THREE.LinearFilter; maskTex.needsUpdate = true;
const MU = { uMask: { value: maskTex }, uMaskOn: { value: 0 }, uRes: { value: new THREE.Vector2(1, 1) }, uHeadVz: { value: 0 }, uSoft: { value: 1 }, uRim: { value: 0 }, uSlope: { value: 0 }, uLen: { value: 0.3 }, uSide: { value: 0 }, uCrown: { value: 1 }, uSideDrop: { value: 0 }, uSideDropS: { value: 0 }, uMX: { value: new THREE.Vector3(1, 0, 0) }, uGrain: { value: 0.02 }, uSeed: { value: 0 } };
const MASK_GLSL = "uniform vec3 uMX; uniform float uSideDropS; uniform sampler2D uMask; uniform float uMaskOn; uniform vec2 uRes; uniform float uHeadVz; uniform float uSoft; uniform float uRim; uniform float uSlope; uniform float uLen; uniform float uSide; uniform float uGrain; uniform float uSeed; varying float vVz; varying float vHy; varying vec3 vP;\n" +
  "float personMask(){ vec2 q = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y) * uMX.x + uMX.yz;   // where this pixel was when the mask was made\n" +
  " return smoothstep(.3, .7, texture2D(uMask, q / uRes).r); }\n";
/* Hat materials: the low parts of the hat (brim) that lie behind the head are hidden where the picture shows hair or
   face. Shadow material: the shadow is only drawn on the person. */
function maskAware(mat, shadow, opaque) {
  mat.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, MU);
    sh.vertexShader = "uniform float uCrown; uniform float uSideDrop; varying float vVz; varying float vHy; varying vec3 vP;\n" + (shadow ? sh.vertexShader : sh.vertexShader.replace("#include <begin_vertex>", "#include <begin_vertex>\n { float ax = transformed.x / 0.57, bz = transformed.z / 0.68; float sd = ax * ax / max(1e-4, ax * ax + bz * bz); transformed.y -= uSideDrop * sd * (1. - smoothstep(0., .45, transformed.y)); }\n if (transformed.y > 0.) transformed.y *= uCrown;")).replace("#include <project_vertex>", "#include <project_vertex>\n vVz = -mvPosition.z; vHy = position.y; vP = position;");
    sh.fragmentShader = MASK_GLSL + (shadow
      ? sh.fragmentShader.replace("#include <dithering_fragment>", "#include <dithering_fragment>\n float sd = 1. - smoothstep(0., uLen, uRim + uSlope * vP.z + uSide * vP.x - uSideDropS * vP.x * vP.x / max(1e-4, vP.x * vP.x + vP.z * vP.z) - vP.y); gl_FragColor.a *= mix(1., personMask(), uMaskOn) * pow(sd, 1.15) * smoothstep(-.8, -.35, vP.z);   // front and sides (ears, side hair), not the back")
      : (opaque ? sh.fragmentShader.replace("#include <alphatest_fragment>", "diffuseColor.a = 1.;\n#include <alphatest_fragment>") : sh.fragmentShader).replace("#include <dithering_fragment>", "#include <dithering_fragment>\n gl_FragColor.rgb *= mix(.74, 1.06, smoothstep(-.02, .5, vP.y));\n gl_FragColor.rgb += (fract(sin(dot(gl_FragCoord.xy + uSeed, vec2(12.9898, 78.233))) * 43758.5453) - .5) * uGrain;\n gl_FragColor.a *= 1. - uMaskOn * personMask() * smoothstep(0., uSoft, vVz - uHeadVz) * (1. - smoothstep(.1, .2, vHy));\n if (gl_FragColor.a < .02) discard;"));
  };
  mat.customProgramCacheKey = () => shadow ? "shade" : opaque ? "hat-opaque" : "hat";
  if (!shadow) mat.transparent = true;
  return mat;
}
const headG = new THREE.Group(); scene.add(headG);
const HEAD_R = [0.535, 0.8, 0.66];   // head shape half-axes, in head widths (an average head)
/* The person's own head, estimated once from their face (headshape.js): ratios to the average head, so an average
   person gets exactly the tuned fit. kx/ky/kz: skull half-axes, kd: forehead to skull centre (front to back),
   dTop: crown height above the forehead, in head widths. headK eases towards headTarget, so the hat never jumps. */
const HEAD_AVG = { kx: 1, ky: 1, kz: 1, kd: 1, dTop: 0 };
let headModel = null, headFitter = null, headTarget = HEAD_AVG;
const headK = { ...HEAD_AVG };
function headReset() { headFitter?.reset(); headTarget = HEAD_AVG; }
function headFeed(lm, W, H, pose, frames) {   // one tracked face; returns true when a new estimate was made
  if (!headFitter || !S.headShape || headFitter.result) return false;
  const r = headFitter.add(lm, W, H, pose, frames); if (!r) return false;
  headTarget = r; console.info("Head shape:", Object.fromEntries(["kx", "ky", "kz", "kd", "dTop"].map(k => [k, +r[k].toFixed(3)]))); return true;
}
const headGeo = new THREE.SphereGeometry(1, 48, 32);
const occluder = new THREE.Mesh(headGeo, new THREE.MeshBasicMaterial({ colorWrite: false })); occluder.renderOrder = -1;
/* shade under the hat: a soft band on the head below the hat's edge, longer under a brim */
const shadowMat = maskAware(new THREE.MeshBasicMaterial({ color: 0x0a0604, transparent: true, opacity: 0.5, depthWrite: false }), true);
const catcher = new THREE.Mesh(headGeo, shadowMat); catcher.renderOrder = 5;
const wire = new THREE.Mesh(headGeo, new THREE.MeshBasicMaterial({ color: 0x7c9cff, wireframe: true, transparent: true, opacity: 0.35, depthTest: false })); wire.renderOrder = 9;
const hatPivot = new THREE.Group();
headG.add(occluder, catcher, wire, hatPivot);
const hemi = new THREE.HemisphereLight(0xffffff, 0x1f1c1a, 1), sun = new THREE.DirectionalLight(0xffffff, 1);
scene.add(hemi, sun, sun.target);
{ const pm = new THREE.PMREMGenerator(renderer); scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture; pm.dispose(); }   // soft surround light for reflections and fill

/* Light of the picture, read from the face: how bright, what tint, and which side it comes from. */
const lightCv = document.createElement("canvas"); lightCv.width = 24; lightCv.height = 32;
const lightCtx = lightCv.getContext("2d", { willReadFrequently: true });
const grainCv = document.createElement("canvas"); grainCv.width = grainCv.height = 32;
const grainCtx = grainCv.getContext("2d", { willReadFrequently: true });
const lin = v => Math.pow(v / 255, 2.2);
const light = { exp: 1, side: 0, contrast: 0.2, grain: 0.02, tint: [1, 1, 1], ready: false };
function estimateLight(source, pose, smooth) {
  const w = pose.w;
  lightCtx.drawImage(source, pose.cx - w * 0.36, pose.cy - w * 0.3, w * 0.72, w * 0.75, 0, 0, 24, 16);   // cheeks, nose, eyes
  lightCtx.drawImage(source, 0, 0, source.width, source.height, 0, 16, 24, 16);                            // whole frame
  const d = lightCtx.getImageData(0, 0, 24, 32).data;
  let L = 0, R = 0, n = 0; const fr = [0, 0, 0];
  for (let y = 0; y < 16; y++) for (let x = 0; x < 24; x++) {
    const i = (y * 24 + x) * 4, l = 0.2126 * lin(d[i]) + 0.7152 * lin(d[i + 1]) + 0.0722 * lin(d[i + 2]);
    if (x < 9) L += l; else if (x >= 15) R += l;
    const j = i + 16 * 24 * 4; fr[0] += lin(d[j]); fr[1] += lin(d[j + 1]); fr[2] += lin(d[j + 2]); n++;
  }
  L /= 16 * 9; R /= 16 * 9;
  const face = (L + R) / 2, mx = Math.max(fr[0], fr[1], fr[2], 1e-4);
  const t = { exp: Math.min(1.5, Math.max(0.3, face / 0.3)), side: Math.max(-1, Math.min(1, 3 * (R - L) / (R + L + 1e-3))),
    contrast: Math.min(1, Math.abs(R - L) / (R + L + 1e-3) * 2.5), tint: fr.map(v => 0.6 + 0.4 * v / mx) };
  // picture noise, from a patch of cheek at full resolution
  grainCtx.drawImage(source, pose.cx - w * 0.3, pose.cy + w * 0.05, 32, 32, 0, 0, 32, 32);
  const gd = grainCtx.getImageData(0, 0, 32, 32).data; let gs = 0;
  for (let i = 0; i < gd.length - 4; i += 4) gs += Math.abs(gd[i + 1] - gd[i + 5]);
  t.grain = Math.min(0.08, Math.max(0.006, gs / (gd.length / 4) / 255 * 1.6));
  const k = smooth && light.ready ? 0.08 : 1;
  light.exp += (t.exp - light.exp) * k; light.side += (t.side - light.side) * k; light.contrast += (t.contrast - light.contrast) * k; light.grain += (t.grain - light.grain) * k;
  light.tint = light.tint.map((v, i) => v + (t.tint[i] - v) * k); light.ready = true;
}

/* ---------------- Hats ---------------- */
const TILT_BACK = 9 * Math.PI / 180;   // a hat sits lower at the back of the head than on the forehead
/* How far the hat's top may face away from (under) or towards (over) the camera. Webcams usually sit below the eyes,
   and head pitch from one camera is the least reliable angle: at full strength the brim's far side swings down past
   the face like a lampshade. Limiting the view of the hat keeps it on the head; the band stays on the forehead. */
/* Soft limits: past the knee the hat still follows the head, more and more stiffly, up to the limit. */
const VIEW_UNDER = Math.sin(22 * Math.PI / 180), VIEW_OVER = Math.sin(32 * Math.PI / 180), KNEE_UNDER = Math.sin(11 * Math.PI / 180), KNEE_OVER = Math.sin(20 * Math.PI / 180);
const soft = (v, knee, lim) => v <= knee ? v : knee + (lim - knee) * Math.tanh((v - knee) / (lim - knee));
const _hy = new THREE.Vector3(), _hz = new THREE.Vector3();
/* tilt back (radians, about the head's x axis) that keeps the hat's up axis within the limits, as close to t0 as possible */
function viewTilt(t0) {
  _hy.set(0, 1, 0).applyQuaternion(headG.quaternion); _hz.set(0, 0, 1).applyQuaternion(headG.quaternion);
  const g = t => Math.cos(t) * _hy.z - Math.sin(t) * _hz.z;   // z of the hat's up axis: + top seen, - underside seen
  const v = g(t0), goal = v >= 0 ? soft(v, KNEE_OVER, VIEW_OVER) : -soft(-v, KNEE_UNDER, VIEW_UNDER);
  if (Math.abs(goal - v) < 1e-5) return t0;
  const R = Math.hypot(_hy.z, _hz.z), phi = Math.atan2(_hz.z, _hy.z);
  if (R < 1e-6 || Math.abs(goal) > R) return t0;
  const wrap = a => Math.atan2(Math.sin(a), Math.cos(a)), c = Math.acos(goal / R);
  const ts = [c - phi, -c - phi].map(t => t0 + wrap(t - t0));
  const t = Math.abs(ts[0] - t0) < Math.abs(ts[1] - t0) ? ts[0] : ts[1];
  return t0 + Math.max(-0.6, Math.min(0.6, t - t0));
}
function disposeGroup(g) { g.traverse(o => { if (o.isMesh) { o.geometry.dispose(); o.material.map?.dispose(); o.material.bumpMap?.dispose(); o.material.dispose(); } }); }
/* A 3D hat file (glTF/GLB) is brought into head widths: its crown, measured above any brim, is scaled to the head band,
   centred, and set down so the band is at y = 0. Assumes y up and the front towards +z, as glTF asks.
   o.turn (radians about y) turns a model that does not face +z. o.ref / o.base / o.cref (fractions of the model's height, from the bottom) override where the head band is measured,
   where it sits and where the crown is centred: for crowns that bulge out above the band (peaked caps) or visors that stick forward instead of out. */
function fitModel(obj, o = {}) {
  if (o.turn) obj.rotateY(o.turn);   // model built facing -z (or sideways): turn it to face +z first
  obj.updateMatrixWorld(true);
  const pts = [], v = new THREE.Vector3();
  obj.traverse(o => { if (!o.isMesh || o.isSkinnedMesh) return; const p = o.geometry.attributes.position, st = Math.max(1, Math.floor(p.count / 20000)); for (let i = 0; i < p.count; i += st) { v.fromBufferAttribute(p, i).applyMatrix4(o.matrixWorld); pts.push(v.x, v.y, v.z); } });
  if (pts.length < 9) return null;
  let y0 = Infinity, y1 = -Infinity; for (let i = 1; i < pts.length; i += 3) { y0 = Math.min(y0, pts[i]); y1 = Math.max(y1, pts[i]); }
  const N = 20, Hh = (y1 - y0) || 1, sl = Array.from({ length: N }, () => [Infinity, -Infinity, Infinity, -Infinity]);
  for (let i = 0; i < pts.length; i += 3) { const b = sl[Math.min(N - 1, Math.floor((pts[i + 1] - y0) / Hh * N))]; b[0] = Math.min(b[0], pts[i]); b[1] = Math.max(b[1], pts[i]); b[2] = Math.min(b[2], pts[i + 2]); b[3] = Math.max(b[3], pts[i + 2]); }
  const wd = sl.map(b => b[1] - b[0]), mid = wd.slice(Math.floor(N * 0.35), Math.floor(N * 0.7)).filter(isFinite).sort((a, b) => a - b), cw = mid[mid.length >> 1] || Math.max(...wd.filter(isFinite));
  // brim: the lowest slices much wider than the crown; the band is where they end
  let band = 0; for (let i = 0; i < N * 0.5; i++) if (wd[i] > cw * 1.3) band = i + 1;
  let ri = Math.min(N - 1, Math.max(band + 1, Math.floor(N * 0.4))); for (let d = 0; d < N && !isFinite(wd[ri]); d++) for (const j of [ri + d, ri - d]) if (isFinite(wd[j])) { ri = j; break; }   // low-poly: nearest slice that has vertices
  if (o.ref !== undefined) { ri = Math.min(N - 1, Math.max(0, Math.floor(o.ref * N))); for (let d = 0; d < N && !isFinite(wd[ri]); d++) if (isFinite(wd[ri + 1])) ri++; }
  const ref = sl[o.cref !== undefined ? Math.min(N - 1, Math.floor(o.cref * N)) : ri], k = 2 * A * 1.03 / (o.ref !== undefined ? wd[ri] : cw);   // o.cref: centre front-to-back from a slice above a visor
  // bake the placement into the vertices, so the hat shaders see head widths like the built-in hats
  const g = new THREE.Group(), M = new THREE.Matrix4(), P = new THREE.Matrix4().makeScale(k, k, k).setPosition(-(ref[0] + ref[1]) / 2 * k, -(y0 + (o.base !== undefined ? o.base : band ? (band - 0.5) / N : 0.1) * Hh) * k, -(ref[2] + ref[3]) / 2 * k);
  obj.traverse(o => {
    if (!o.isMesh || o.isSkinnedMesh) return;
    const geo = o.geometry.clone(), p = geo.attributes.position;
    if (!(p.array instanceof Float32Array) || p.isInterleavedBufferAttribute) { const a = new Float32Array(p.count * 3); for (let i = 0; i < p.count; i++) v.fromBufferAttribute(p, i).toArray(a, i * 3); geo.setAttribute("position", new THREE.BufferAttribute(a, 3)); }   // quantized files store integers
    g.add(new THREE.Mesh(geo.applyMatrix4(M.multiplyMatrices(P, o.matrixWorld)), o.material));
  });
  return { group: g, brim: band > 0, top: (y1 - y0) * k };
}
const gltf = new GLTFLoader();
// uploaded files may only use what is inside them: no requests to addresses named in the file
const sealed = new THREE.LoadingManager(); sealed.setURLModifier(u => /^(data|blob):/i.test(u) ? u : "data:,");
const gltfUpload = new GLTFLoader(sealed);
function adoptModel(h, scene3) {
  const f = fitModel(scene3, h.fitModel); if (!f) throw new Error("no meshes in the file");
  h.group = f.group; h.top = f.top; if (h.brim === undefined) h.brim = f.brim;
  const seen = new Set();   // meshes may share a material
  h.group.traverse(o => { if (o.isMesh) { for (const m of [].concat(o.material)) { if (seen.has(m)) continue; seen.add(m); m.side = THREE.DoubleSide; maskAware(m, false, !m.transparent && !m.alphaTest); } } });
}
function buildHat(h) {
  if (h.model) return;
  if (h.group) { hatPivot.remove(h.group); disposeGroup(h.group); }
  if (h.custom) h.group = customHat(h);
  else { const b = h.make(h.color); h.group = b.group; h.top = h.crownTop || new THREE.Box3().setFromObject(h.group).max.y; }
  h.group.traverse(o => { if (o.isMesh) { maskAware(o.material, false); } });
}
/* A product photo becomes a sheet wrapped over the front of the head. Rows below the tuck line go round the back. */
function customHat(h) {
  const c = h.cut, k = 2 * A / c.hw, W = c.canvas.width, H = c.canvas.height, cutY = H * (1 - S.tuck), NX = 40, NY = 24;
  const pos = [], uv = [], idx = [];
  for (let j = 0; j <= NY; j++) for (let i = 0; i <= NX; i++) {
    const px = i / NX * W, py = j / NY * H, x = (px - c.cx) * k, y = (c.ry - py) * k;
    let z = B * Math.sqrt(Math.max(0, 1 - (x / A) ** 2 - (Math.max(0, y) / 0.75) ** 2));
    if (py > cutY) z = -z - 0.04;
    pos.push(x, y, z); uv.push(i / NX, 1 - j / NY);
  }
  for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) { const a = j * (NX + 1) + i, b = a + 1, d = a + NX + 1; idx.push(a, d, b, b, d, d + 1); }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2)); g.setIndex(idx);
  const map = new THREE.CanvasTexture(c.canvas); map.colorSpace = THREE.SRGBColorSpace; map.anisotropy = 4;
  const grp = new THREE.Group(); grp.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ map, side: THREE.DoubleSide, alphaTest: 0.05 })));
  return grp;
}
/* swatch pictures: the hat alone, rendered off screen */
const TH = 160, thumbRT = new THREE.WebGLRenderTarget(TH, TH, { samples: 4 }), thumbScene = new THREE.Scene();
const thumbCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 20); thumbCam.position.set(0, 0.9, 5); thumbCam.lookAt(0, 0.15, 0);
{ const d = new THREE.DirectionalLight(0xffffff, 2.4); d.position.set(-1.5, 3, 3); thumbScene.add(new THREE.HemisphereLight(0xffffff, 0x777777, 1.6), d); }
const thumbPx = new Uint8Array(TH * TH * 4);
function thumb(h, cv) {
  const g = cv.getContext("2d"); g.clearRect(0, 0, cv.width, cv.height);
  if (h.custom) { const c = h.cut.canvas, k = Math.min(cv.width / c.width, cv.height / c.height) * 0.92; g.drawImage(c, (cv.width - c.width * k) / 2, (cv.height - c.height * k) / 2, c.width * k, c.height * k); return; }
  const box = new THREE.Box3().setFromObject(h.group), r = Math.max(box.max.x - box.min.x, (box.max.y - box.min.y) * 1.25, 1.3) * 0.56;
  thumbCam.left = -r; thumbCam.right = r; thumbCam.top = r; thumbCam.bottom = -r; thumbCam.updateProjectionMatrix();
  thumbCam.position.set(0, 0.9 + (box.max.y + box.min.y) / 2, 5); thumbCam.lookAt(0, (box.max.y + box.min.y) / 2, 0);
  const parent = h.group.parent; thumbScene.add(h.group);
  const on = MU.uMaskOn.value; MU.uMaskOn.value = 0;
  renderer.setRenderTarget(thumbRT); renderer.setClearColor(0x000000, 0); renderer.clear(); renderer.render(thumbScene, thumbCam);
  renderer.readRenderTargetPixels(thumbRT, 0, 0, TH, TH, thumbPx); renderer.setRenderTarget(null);
  MU.uMaskOn.value = on; thumbScene.remove(h.group); if (parent) parent.add(h.group);
  const img = new ImageData(TH, TH);
  for (let y = 0; y < TH; y++) img.data.set(thumbPx.subarray((TH - 1 - y) * TH * 4, (TH - y) * TH * 4), y * TH * 4);
  const t = document.createElement("canvas"); t.width = t.height = TH; t.getContext("2d").putImageData(img, 0, 0);
  g.drawImage(t, 0, 0, cv.width, cv.height);
}
const sw = $("swatches");
function renderSwatches() {
  sw.innerHTML = "";
  for (const h of hats) {
    if (!h.group) { if (h.model) continue; buildHat(h); }
    const el = document.createElement("div"); el.className = "sw" + (h === hat ? " active" : "");
    const cv = document.createElement("canvas"); cv.width = cv.height = 120; thumb(h, cv);
    const sp = document.createElement("span"); sp.textContent = h.name;
    el.append(cv, sp); el.onclick = () => selectHat(h); h.el = el; sw.append(el);
  }
}
function selectHat(h) {
  hat = h; eyeLift = 0;
  sw.querySelectorAll(".sw").forEach(e => e.classList.remove("active")); h.el?.classList.add("active");
  $("customBox").style.display = h.custom ? "" : "none";
  $("colorRow").style.display = h.custom || h.model ? "none" : "";
  if (!h.custom && !h.model) $("color").value = h.color;
  syncFitUI(); requestStill();
}
$("color").addEventListener("input", e => {
  if (hat.custom || hat.model) return;
  hat.color = e.target.value; buildHat(hat); thumb(hat, hat.el.querySelector("canvas")); requestStill();
});

/* ---------------- Custom hat: background removal + auto placement ---------------- */
/* Removes the backdrop of a product photo: flood fill from the image border over pixels close to the border colour,
   so light areas inside the hat are kept. Returns a canvas cropped to the hat. */
function cutOut(img, removeBg, tol) {
  const k = Math.min(1, 1024 / Math.max(img.naturalWidth || img.width, img.naturalHeight || img.height));
  const w = Math.max(2, Math.round((img.naturalWidth || img.width) * k)), h = Math.max(2, Math.round((img.naturalHeight || img.height) * k));
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d", { willReadFrequently: true }); g.drawImage(img, 0, 0, w, h);
  const d = g.getImageData(0, 0, w, h), p = d.data, n = w * h;
  let transparent = 0; for (let i = 3; i < p.length; i += 4) if (p[i] < 250) transparent++;
  if (removeBg && transparent < n * 0.02) {
    const br = [], bg_ = [], bb = [];
    const edge = i => { br.push(p[i * 4]); bg_.push(p[i * 4 + 1]); bb.push(p[i * 4 + 2]); };
    for (let x = 0; x < w; x++) { edge(x); edge((h - 1) * w + x); }
    for (let y = 0; y < h; y++) { edge(y * w); edge(y * w + w - 1); }
    const med = a => a.sort((x, y) => x - y)[a.length >> 1];
    const R = med(br), G = med(bg_), B = med(bb), t2 = tol * tol;
    const near = i => { const dr = p[i * 4] - R, dg = p[i * 4 + 1] - G, db = p[i * 4 + 2] - B; return dr * dr + dg * dg + db * db < t2; };
    const gone = new Uint8Array(n), q = new Int32Array(n); let qh = 0, qt = 0;
    const push = i => { if (!gone[i] && near(i)) { gone[i] = 1; q[qt++] = i; } };
    for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
    for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
    while (qh < qt) {
      const i = q[qh++], x = i % w;
      if (x > 0) push(i - 1); if (x < w - 1) push(i + 1); if (i >= w) push(i - w); if (i < n - w) push(i + w);
    }
    // drop the one-pixel halo, then feather inwards
    const a = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const x = i % w;
      a[i] = (gone[i] || (x > 0 && gone[i - 1]) || (x < w - 1 && gone[i + 1]) || (i >= w && gone[i - w]) || (i < n - w && gone[i + w])) ? 0 : 255;
    }
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x; if (!a[i]) { p[i * 4 + 3] = 0; continue; }
      let s = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += a[i + dy * w + dx];
      p[i * 4 + 3] = Math.min(p[i * 4 + 3], s / 9);
    }
    for (let x = 0; x < w; x++) { p[x * 4 + 3] = 0; p[((h - 1) * w + x) * 4 + 3] = 0; }
    for (let y = 0; y < h; y++) { p[(y * w) * 4 + 3] = 0; p[(y * w + w - 1) * 4 + 3] = 0; }
    g.putImageData(d, 0, 0);
  }
  // crop to content, measure the outline row by row
  let x0 = w, x1 = -1, y0 = h, y1 = -1; const L = new Int32Array(h).fill(-1), Rr = new Int32Array(h).fill(-1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (p[(y * w + x) * 4 + 3] > 40) {
    if (L[y] < 0) L[y] = x; Rr[y] = x;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (x1 < x0 || y1 < y0) return null;
  const cw = x1 - x0 + 1, ch = y1 - y0 + 1, pad = 4;
  const o = document.createElement("canvas"); o.width = cw + pad * 2; o.height = ch + pad * 2;
  o.getContext("2d").drawImage(c, x0, y0, cw, ch, pad, pad, cw, ch);
  // guess where the head goes: a brim is much wider than the crown above it
  const wid = y => L[y] < 0 ? 0 : Rr[y] - L[y] + 1;
  const mids = [], ws = [];
  for (let y = y0 + Math.round(ch * 0.2); y <= y0 + Math.round(ch * 0.55); y++) if (L[y] >= 0) { ws.push(wid(y)); mids.push((L[y] + Rr[y]) / 2); }
  const med = a => a.length ? a.slice().sort((x, y) => x - y)[a.length >> 1] : 0;
  const crownW = med(ws) || cw;
  const meta = { cx: o.width / 2, ry: pad + ch * 0.9, hw: cw / 1.12 };
  if (cw > crownW * 1.3) {
    meta.hw = crownW / 1.04; meta.cx = med(mids) - x0 + pad; meta.ry = pad + ch * 0.8;
    // the hat line is a little below the widest row of the brim (a brim seen from above is an ellipse)
    let wy = y1, wmax = 0;
    for (let y = y0 + Math.round(ch * 0.3); y <= y1; y++) if (wid(y) > wmax) { wmax = wid(y); wy = y; }
    meta.ry = pad + Math.min(ch * 0.9, wy - y0 + ch * 0.1);
  }
  return { canvas: o, ...meta };
}
let customImg = null;
function setCustom(img, keepFit) {
  const cut = cutOut(img, S.keyBg, S.tol);
  if (!cut) { $("customBox").style.display = ""; $("hint").textContent = "Nothing left after background removal. Lower the tolerance or turn removal off."; return; }
  let h = hats.find(x => x.custom);
  if (!h) { h = { name: "Custom", custom: true, base: { ...FIT_DEFAULT }, fit: { ...FIT_DEFAULT }, group: null }; hats.push(h); }
  if (!keepFit) h.fit = { ...h.base };
  h.cut = cut; buildHat(h); hat = h; renderSwatches(); selectHat(h);
}
/* 3D hat files: the built-in scans, and the user's own .glb */
for (const h of hats) if (h.model) gltf.load(h.url, r => { adoptModel(h, r.scene); renderSwatches(); requestStill(); }, undefined, e => console.warn("Could not load " + h.url, e));
$("glbIn").addEventListener("change", async e => {
  const f = e.target.files[0]; e.target.value = ""; if (!f) return;
  try {
    const r = await gltfUpload.parseAsync(await f.arrayBuffer(), "");
    const h = { name: f.name.replace(/\.[^.]+$/, "").slice(0, 14) || "3D hat", model: true, base: { ...FIT_DEFAULT }, fit: { ...FIT_DEFAULT }, group: null };
    adoptModel(h, r.scene); hats.push(h); hat = h; renderSwatches(); selectHat(h);
  } catch (err) { console.warn(err); $("hint").textContent = "Could not read that 3D file. Use a single self-contained .glb without Draco or Meshopt compression."; }
});
$("hatIn").addEventListener("change", e => {
  const f = e.target.files[0]; e.target.value = ""; if (!f) return;
  const img = new Image();
  img.onload = () => { customImg = img; S.tuck = 0; $("tuck").value = 0; $("v_tuck").textContent = "0%"; setCustom(img, false); URL.revokeObjectURL(img.src); };
  img.onerror = () => { $("hint").textContent = "Could not read that image."; };
  img.src = URL.createObjectURL(f);
});

/* ---------------- Controls ---------------- */
const FMT = {
  size: v => v.toFixed(2) + "×", y: v => (v > 0 ? "+" : "") + Math.round(v * 100), x: v => (v > 0 ? "+" : "") + Math.round(v * 100),
  tilt: v => v.toFixed(1) + "°", stretch: v => v.toFixed(2) + "×", curve: v => v.toFixed(1) + "°",
  bright: v => Math.round(v * 100) + "%", shadow: v => Math.round(v * 100) + "%", tol: v => String(v), tuck: v => Math.round(v * 100) + "%",
};
function syncFitUI() { for (const k of FIT_KEYS) { $(k).value = hat.fit[k]; $("v_" + k).textContent = FMT[k](+hat.fit[k]); } }
for (const k of FIT_KEYS) $(k).addEventListener("input", e => { hat.fit[k] = +e.target.value; $("v_" + k).textContent = FMT[k](hat.fit[k]); requestStill(); });
for (const k of ["bright", "shadow", "tol", "tuck"]) {
  const el = $(k); $("v_" + k).textContent = FMT[k](+el.value);
  el.addEventListener("input", () => {
    S[k] = +el.value; $("v_" + k).textContent = FMT[k](S[k]);
    if (k === "tuck" && hat.custom) buildHat(hat);
    if (k === "tol" && customImg) setCustom(customImg, true);
    requestStill();
  });
}
for (const k of ["occlude", "debug", "keyBg", "headShape"]) $(k).addEventListener("change", e => {
  S[k] = e.target.checked;
  if (k === "keyBg" && customImg) setCustom(customImg, true);
  if (k === "occlude" && still) refreshStill(); else requestStill();
});
$("resetBtn").onclick = () => { hat.fit = { ...hat.base }; syncFitUI(); requestStill(); };
const cmp = $("cmpBtn");
const setOrig = v => { S.showOrig = v; requestStill(); };
cmp.addEventListener("pointerdown", () => setOrig(true));
for (const ev of ["pointerup", "pointerleave", "pointercancel"]) cmp.addEventListener(ev, () => setOrig(false));

/* ---------------- Smoothing ---------------- */
class OneEuro {
  constructor(minCutoff = 1.2, beta = 0.04, dCutoff = 1) { Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: 0, t: null }); }
  static a(cut, dt) { const tau = 1 / (2 * Math.PI * cut); return 1 / (1 + tau / dt); }
  f(v, t) {
    if (this.x === null) { this.x = v; this.t = t; return v; }
    const dt = Math.max(1e-3, (t - this.t) / 1000); this.t = t;
    const dx = (v - this.x) / dt; this.dx += OneEuro.a(this.dCutoff, dt) * (dx - this.dx);
    this.x += OneEuro.a(this.minCutoff + this.beta * Math.abs(this.dx), dt) * (v - this.x);
    return this.x;
  }
  reset() { this.x = null; this.dx = 0; }
}
/* position in pixels, angles in radians, sizes in pixels / head widths */
// phones run at a lower frame rate: each filter step is longer, so the time-based
// smoothing trails the head. Faster cutoffs there keep the hat on the head as it moves.
const FK = matchMedia("(pointer: coarse)").matches && Math.min(screen.width, screen.height) < 820 ? 2.2 : 1;
const OE = (c, b, d) => new OneEuro(c * FK, b, d);
const filt = { cx: OE(1.2, 0.25, 1), cy: OE(1.2, 0.25, 1), w: OE(0.6, 0.05, 1),
  yaw: OE(1, 6, 1), pitch: OE(1, 6, 1), roll: OE(1.2, 8, 1), h: OE(0.25, 0.5, 1), fz: OE(0.25, 0.5, 1), brow: OE(0.25, 0.5, 1),
  q0: OE(1, 6, 1), q1: OE(1, 6, 1), q2: OE(1, 6, 1), q3: OE(1, 6, 1) };
let lastQ = null;
const resetFilters = () => { Object.values(filt).forEach(f => f.reset()); lastQ = null; };

/* ---------------- Head pose ---------------- */
const OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/* Image frame: x right, y down, z towards the camera. Returns the head centre (between the ears), the head width in
   pixels, yaw / pitch / roll and the height of the top of the forehead above the ear line. */
function headPose(lm, mat, W, H, t, useFilter) {
  const P = i => [lm[i].x * W, lm[i].y * H, -lm[i].z * W];
  const A = P(234), B = P(454), T = P(10);
  const C = [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2, (A[2] + B[2]) / 2];
  let hy, hz;   // head "up" and "forward" axes
  if (mat) { const m = mat.data; hy = norm([m[4], -m[5], m[6]]); hz = norm([m[8], -m[9], m[10]]); }
  else { const hx = norm(sub(B, A)); hy = sub(T, P(152)); const d = dot(hy, hx); hy = norm(sub(hy, hx.map(v => v * d))); hz = cross(hy, hx); }
  let pitch = Math.atan2(hy[2], Math.hypot(hy[0], hy[1]));                 // + looking down
  let roll = Math.atan2(hy[0], -hy[1]);                                    // + clockwise in the image
  let yaw = Math.atan2(hz[0] * Math.cos(roll) + hz[1] * Math.sin(roll), hz[2] / Math.max(0.2, Math.cos(pitch)));  // + turned to image right
  let w = Math.hypot(B[0] - A[0], B[1] - A[1], B[2] - A[2]);
  let h = dot(sub(T, C), hy) / w, cx = C[0], cy = C[1];
  let fz = Math.min(0.62, Math.max(0.4, dot(sub(T, C), hz) / w));        // forehead in front of the ear line
  const Br = P(105), Bl = P(334); let brow = dot(sub([(Br[0] + Bl[0]) / 2, (Br[1] + Bl[1]) / 2, (Br[2] + Bl[2]) / 2], C), hy) / w;   // eyebrows: about the height of the tops of the ears
  // rotation in world axes (y up). The tracker's matrix is relative to the camera axis; turn it to how the head looks
  // from where it is in the frame, since the scene is drawn without perspective.
  const X = norm(cross([hy[0], -hy[1], hy[2]], [hz[0], -hz[1], hz[2]])), Y = [hy[0], -hy[1], hy[2]], Z = cross(X, Y);
  const Q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(new THREE.Vector3(...X), new THREE.Vector3(...Y), new THREE.Vector3(...Z)));
  if (mat) {
    const f = H / 2 / Math.tan(FOV_Y / 2), ray = new THREE.Vector3(cx - W / 2, H / 2 - cy, -f).normalize();
    Q.premultiply(new THREE.Quaternion().setFromUnitVectors(ray, new THREE.Vector3(0, 0, -1)));
  }
  let q = Q.toArray();
  if (useFilter) {
    cx = filt.cx.f(cx, t); cy = filt.cy.f(cy, t); w = filt.w.f(w, t);
    yaw = filt.yaw.f(yaw, t); pitch = filt.pitch.f(pitch, t); roll = filt.roll.f(roll, t);
    // forehead height and depth belong to the person, not the pose; the tracker's depth makes them drift when the head
    // nods or turns, which bobbed the hat up and down. Learn them while the face is near frontal, hold them otherwise.
    const frontal = Math.abs(pitch) < 0.22 && Math.abs(yaw) < 0.35;
    if (frontal || filt.h.x === null) { h = filt.h.f(h, t); fz = filt.fz.f(fz, t); brow = filt.brow.f(brow, t); } else { h = filt.h.x; fz = filt.fz.x; brow = filt.brow.x; filt.h.t = filt.fz.t = filt.brow.t = t; }
    if (lastQ && q[0] * lastQ[0] + q[1] * lastQ[1] + q[2] * lastQ[2] + q[3] * lastQ[3] < 0) q = q.map(v => -v);
    q = q.map((v, i) => filt["q" + i].f(v, t)); const l = Math.hypot(...q) || 1; q = q.map(v => v / l); lastQ = q;
  }
  return { cx, cy, w, yaw, pitch, roll, h, fz, brow, q, lm };
}

/* ---------------- Person mask (face + hair) ---------------- */
let maskBuf = null;
/* seg: { hair, face, w, h } soft masks from the segmenter, or null */
/* A mask made a few frames ago is moved and scaled with the head since then, so it stays on the person. */
let maskPose = null, lastMaskSeg = null;
function maskFollow(pose) {
  if (!maskPose || !pose) { MU.uMX.value.set(1, 0, 0); return; }
  const k = maskPose.w / pose.w; MU.uMX.value.set(k, maskPose.cx - k * pose.cx, maskPose.cy - k * pose.cy);
}
function setMask(seg, pose) {
  MU.uMaskOn.value = seg ? 1 : 0; if (!seg) return;
  if (seg === lastMaskSeg) return;   // same mask as before: only its placement changes (maskFollow)
  lastMaskSeg = seg; pose = seg.pose || pose; maskPose = pose ? { cx: pose.cx, cy: pose.cy, w: pose.w } : null;   // a worker mask carries the pose of its own frame
  const n = seg.w * seg.h;
  if (!maskBuf || maskBuf.length !== n) { maskBuf = new Uint8Array(n); maskTex.dispose(); maskTex.image = { data: maskBuf, width: seg.w, height: seg.h }; }
  for (let i = 0; i < n; i++) maskBuf[i] = Math.min(255, (seg.hair[i] + seg.face[i] + (seg.skin ? seg.skin[i] : 0)) * 255);   // the person: hair, face, ears (and neck)
  maskTex.needsUpdate = true;
}

/* How wide the head is with its hair, just above the forehead: the hat is sized to go round it. */
function hairFit(seg, pose, W, H, smooth) {
  const sr = Math.sin(pose.roll), cr = Math.cos(pose.roll), w = pose.w;
  const on = (a, b) => {   // a: head widths to the right of centre, b: above the ear line
    const x = Math.round((pose.cx + w * (a * cr + b * sr)) / W * seg.w), y = Math.round((pose.cy + w * (a * sr - b * cr)) / H * seg.h);
    if (x < 0 || y < 0 || x >= seg.w || y >= seg.h) return false;
    const i = y * seg.w + x; return seg.hair[i] + seg.face[i] > 0.5;
  };
  let sum = 0, n = 0;
  for (const b of [pose.h - 0.12, pose.h - 0.04, pose.h + 0.04]) {
    let l = 1.1, r = 1.1;
    while (l > 0 && !on(-l, b)) l -= 0.02;
    while (r > 0 && !on(r, b)) r -= 0.02;
    if (l + r > 0.6) { sum += (l + r) / 2; n++; }
  }
  // band = the head (hair included) where the hat sits, plus a little room; no floor above that, or slim heads show a gap
  const t = n ? Math.min(1.45, Math.max(0.98, sum / n * 1.04 / A)) : HAT_SCALE;
  hairScale = smooth ? hairScale + (t - hairScale) * 0.15 : t;
  let top = pose.h; while (top < pose.h + 0.9 && (on(0, top + 0.02) || on(-0.12, top + 0.02) || on(0.12, top + 0.02))) top += 0.02;
  const y = (pose.cy - w * top * cr) / H;   // hair cut off by the picture edge tells nothing
  top = y < 0.02 ? null : top;
  hairTop = smooth && hairTop !== null && top !== null ? hairTop + (top - hairTop) * 0.15 : top;
}

/* ---------------- Render ---------------- */
const TOUCH = matchMedia("(pointer: coarse)").matches, PHONE = TOUCH && Math.min(screen.width, screen.height) < 820;
const PROC_MAX = PHONE ? 480 : 1280;   // longest side of the frame the trackers work on
const MOVE_TIP = TOUCH ? "drag the hat to move it, pinch to resize" : "drag the hat to move it, scroll to resize";
const _v = new THREE.Vector3();
/* The hat's lowest points across its front (brim or visor edge), in the hat's own coordinates; found once per hat. */
function frontEdge(h) {
  if (h.frontPts) return h.frontPts;
  h.group.updateMatrixWorld(true); const inv = h.group.matrixWorld.clone().invert(), v = new THREE.Vector3(), pts = [];
  let zmax = 0, xmax = 0; h.group.traverse(o => { if (!o.isMesh) return; const P = o.geometry.attributes.position; for (let i = 0; i < P.count; i++) { v.fromBufferAttribute(P, i).applyMatrix4(o.matrixWorld).applyMatrix4(inv); zmax = Math.max(zmax, v.z); xmax = Math.max(xmax, Math.abs(v.x)); pts.push(v.x, v.y, v.z); } });
  const NB = 9, low = Array(NB).fill(null);
  for (let i = 0; i < pts.length; i += 3) {
    const x = pts[i], y = pts[i + 1], z = pts[i + 2]; if (z < 0.25 * zmax || Math.abs(x) > 0.5 * xmax) continue;
    const b = Math.min(NB - 1, Math.floor((x / xmax + 0.5) * NB)); if (!low[b] || y < low[b].y) low[b] = new THREE.Vector3(x, y, z);
  }
  return (h.frontPts = low.filter(Boolean));
}
let eyeLift = 0;   // head widths the hat is raised to keep its front edge off the eyes (smoothed in live mode)
/* Keep the eyes clear: wherever the head and camera are, the lowest point of the hat's front edge, as drawn, must stay a
   gap above the upper eyelids. Measured in the image, so it covers a nod, a webcam above the eyes, a deep hat or a low
   forehead estimate alike. Returns how far to raise the hat along the head (head widths). */
function eyeClearance(pose, H, live) {
  const L = pose.lm, w = pose.w, eyeTop = H - Math.min(L[159].y, L[386].y, L[160].y, L[385].y) * H;   // world y is up
  const guard = eyeTop + (hat.eyeGap ?? 0.085) * w;   // per hat: wide brims keep a bigger gap
  hatPivot.updateMatrixWorld(true);
  let low = Infinity; for (const p of frontEdge(hat)) low = Math.min(low, _v.copy(p).applyMatrix4(hat.group.matrixWorld).y);
  const up = Math.max(0.35, _v.set(0, 1, 0).applyQuaternion(headG.quaternion).y);
  const need = Math.max(0, guard - low) / (w * up);
  // live: lift at once, settle back slowly, so the hat never bobs over the eyes
  eyeLift = !live ? need : need > eyeLift ? need : eyeLift + (need - eyeLift) * 0.12;
  return eyeLift;
}
function clearView() { scene.background = null; headG.visible = false; renderer.setClearColor(0x08080a, 1); renderer.clear(); octx.clearRect(0, 0, ovl.width, ovl.height); }
function render(source, W, H, pose) {
  if (cam.right !== W || cam.top !== H) { renderer.setSize(W, H, false); cam.right = W; cam.top = H; cam.updateProjectionMatrix(); MU.uRes.value.set(W, H); }
  if (ovl.width !== W || ovl.height !== H) { ovl.width = W; ovl.height = H; }
  const key = W + "x" + H;
  if (!bgTex || bgKey !== key) { bgTex?.dispose(); bgTex = new THREE.Texture(source); bgTex.colorSpace = THREE.SRGBColorSpace; bgTex.generateMipmaps = false; bgTex.minFilter = bgTex.magFilter = THREE.LinearFilter; bgKey = key; }
  bgTex.image = source; bgTex.needsUpdate = true; scene.background = bgTex;
  octx.clearRect(0, 0, W, H);
  headG.visible = !!pose && !S.showOrig && !!hat.group;
  if (pose && hat.group) {
    const F = hat.fit, w = pose.w;
    headG.position.set(pose.cx, H - pose.cy, 0); headG.quaternion.fromArray(pose.q); headG.scale.setScalar(w);
    // head shape: front at the forehead, top a little above the hairline
    const tgt = S.headShape ? headTarget : HEAD_AVG, ease = stream ? 0.08 : 1;
    for (const k in headK) headK[k] += (tgt[k] - headK[k]) * ease;
    const HR = [HEAD_R[0] * headK.kx, HEAD_R[1] * headK.ky, HEAD_R[2] * headK.kz];
    const zc = pose.fz - HEAD_R[2] * 0.97 * headK.kd, top = pose.h + 0.3 + headK.dTop;
    for (const m of [occluder, catcher, wire]) { m.scale.set(HR[0], HR[1], HR[2]); m.position.set(0, top - HR[1], zc); }
    // the shadow shell is as wide as the hat band, so it reaches the ears and the hair at the sides (the person mask trims it)
    const bandX = A * hairScale * F.size * (hat.snug || 1);
    catcher.scale.x = Math.max(HR[0], bandX * 1.02);
    occluder.visible = S.occlude; wire.visible = S.debug;
    if (hat.group.parent !== hatPivot) { hatPivot.clear(); hatPivot.add(hat.group); }
    const tb0 = TILT_BACK + F.curve * Math.PI / 180, tb = viewTilt(tb0);
    hatPivot.position.set(F.x, pose.h + SIT + (hat.custom ? 0.04 : 0) + F.y - B * hairScale * Math.sin(tb0), zc - 0.03 - B * (hairScale - HAT_SCALE) * 0.6);
    // tipped by viewTilt: turn about the front of the band, so the hat stays on the forehead and only the back moves
    const zf = B * hairScale * F.size;
    hatPivot.position.y += zf * (Math.sin(tb0) - Math.sin(tb)); hatPivot.position.z += zf * (Math.cos(tb0) - Math.cos(tb));
    hatPivot.rotation.set(-tb, 0, -F.tilt * Math.PI / 180); // crown tall enough to hold the hair
    let cs = 1;
    if (!hat.custom && (!hat.model || hat.stretchCrown)) { const mt = hat.top * hairScale * F.size; cs = hairTop === null ? CROWN_SCALE : Math.min(CROWN_SCALE * 1.35, Math.max(CROWN_SCALE * 0.92, (top * 0.3 + hairTop * 0.7 + 0.03 - hatPivot.position.y) / (mt * Math.cos(tb)))); }
    const sn = hat.snug || 1;   // knits and caps stretch over the head; felt and straw hats sit a little proud of it
    hatPivot.scale.set(F.size * hairScale * sn, F.size * hairScale * F.stretch, F.size * hairScale * sn * Math.min(1.1, Math.max(0.92, headK.kz)));   // a longer head front to back needs a deeper hat MU.uCrown.value = cs;
    headG.updateMatrixWorld(true); hatPivot.position.y += eyeClearance(pose, H, !!stream);
    /* A hat band is not level on a head: it crosses the mid-forehead in front but comes down to the tops of the ears at
       the sides, where the head curves away. Bend the lower edge down at the sides to the tops of the ears, fully for
       soft hats (hat.sides = 1), less for stiff ones; without it there is open space between the hat and the ears. */
    const earTop = pose.brow - 0.1;   // tops of the ears, a little below the eyebrows: lower and the hat cuts into the ears, higher and it leaves a gap
    const sideGap = Math.max(0, hatPivot.position.y - earTop);
    const sideDrop = Math.min(0.3, sideGap * (hat.sides ?? 0.5));
    MU.uSideDrop.value = sideDrop / (F.size * hairScale * F.stretch); MU.uSideDropS.value = sideDrop / HR[1];   // the shadow line bends with it
    // light
    if (!stream || !PHONE || lightTick++ % 10 === 0) estimateLight(source, pose, !!stream);   // light changes slowly: phones sample it every 10th frame (each sample reads pixels back from the GPU)
    MU.uGrain.value = light.grain; MU.uSeed.value = stream ? (MU.uSeed.value + 17.3) % 1000 : 0;
    const e = light.exp * S.bright;
    hemi.intensity = e * (1.8 - 0.9 * light.contrast); sun.intensity = e * (1.2 + 1.8 * light.contrast);
    hemi.color.setRGB(...light.tint); sun.color.setRGB(...light.tint);
    _v.set(light.side * 0.3, 1, 0.9).normalize();
    sun.target.position.copy(headG.position); sun.position.copy(headG.position).addScaledVector(_v, w * 5);
    const tn = Math.tan(tb), cy0 = top - HR[1], yF = pose.h + SIT + F.y;
    MU.uRim.value = (yF - (tn * B + (hat.drop || 0)) * hairScale * F.size - cy0) / HR[1]; MU.uSlope.value = tn * HR[2] / HR[1];
    MU.uLen.value = (hat.brim ? 0.55 : hat.custom ? 0.25 : 0.22) * F.size / HR[1]; MU.uSide.value = -light.side * 0.12;
    shadowMat.opacity = Math.min(0.9, S.shadow * (hat.brim ? 0.95 : 0.8));
    catcher.visible = S.shadow > 0;
    hat.group.traverse(o => { if (o.isMesh && o.material.isMeshStandardMaterial) o.material.envMapIntensity = e * 0.45; });
    if (hat.custom) hat.group.children[0].material.color.setRGB(...light.tint.map(v => Math.min(1, v * (0.55 + 0.45 * e))));
    headG.updateMatrixWorld(true);
    MU.uHeadVz.value = cam.position.z - _v.set(0, top - HR[1], zc).applyMatrix4(headG.matrixWorld).z; MU.uSoft.value = w * 0.12;
  }
  renderer.render(scene, cam);
  if (pose && S.debug) drawDebug(pose, W, H);
}
const deg = r => (r * 180 / Math.PI).toFixed(0) + "°";
function drawDebug(pose, W, H) {
  const g = octx, lw = Math.max(1.5, W / 500);
  g.fillStyle = "#5ee0c6"; for (const i of OVAL) { g.beginPath(); g.arc(pose.lm[i].x * W, pose.lm[i].y * H, lw * 1.3, 0, 7); g.fill(); }
  // head axes: red = right, green = up, blue = forward
  const L = pose.w * 0.45, cols = ["#ff5a5a", "#5aff7a", "#5a8cff"];
  for (let a = 0; a < 3; a++) {
    _v.set(a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0).applyQuaternion(headG.quaternion);
    g.strokeStyle = cols[a]; g.lineWidth = lw * 2; g.beginPath(); g.moveTo(pose.cx, pose.cy); g.lineTo(pose.cx + _v.x * L, pose.cy - _v.y * L); g.stroke();
  }
}
function poseText(p) { return `yaw ${deg(p.yaw)} · pitch ${deg(p.pitch)} · roll ${deg(p.roll)}`; }

/* ---------------- Models ---------------- */
let landmarker = null, segmenter = null, mode = "VIDEO";
async function fetchModel(urls, minSize) {
  const errs = [];
  for (const u of urls) {
    if (u.startsWith("./") && !LOCAL) continue;
    try {
      const r = await fetch(u); if (!r.ok) throw new Error("HTTP " + r.status);
      const buf = new Uint8Array(await r.arrayBuffer());
      if (buf.length < minSize || buf[0] === 0x3c) throw new Error("not a model file");   // catches HTML error pages
      return buf;
    } catch (e) { errs.push(`${u.startsWith("./") ? u : u.split("/")[2]}: ${e.message}`); }
  }
  throw new Error("Model download failed → " + errs.join(" | "));
}
const delegates = [];   // GPU or CPU, per task: shown by ?debug
async function createTask(make) {
  let lastErr;
  for (const w of WASM_URLS) {
    let fs;
    try { fs = await FilesetResolver.forVisionTasks(w); } catch (e) { lastErr = new Error("WASM load failed: " + e.message); continue; }
    for (const d of ["GPU", "CPU"]) {
      try { const t = await make(fs, d); delegates.push(d); return t; }
      catch (e) { console.warn(d + " delegate failed:", e); lastErr = new Error(d + " init failed: " + (e.message || e)); }
    }
  }
  throw lastErr;
}
async function init(localModel) {
  // the head-shape model is small (~100 KB) and optional: load it alongside, carry on without it
  loadHeadModel(new URL("./models/headshape", import.meta.url).href).then(m => {
    headModel = m; headFitter = new HeadFitter(m);
    if (still && stillPose) { headFeed(stillPose.lm, still.width, still.height, stillPose, 1); requestStill(); }
  }).catch(e => console.warn("Head shape model unavailable, using the average head:", e));
  try {
    if (!FaceLandmarker) throw new Error(LOCAL ? "Could not load MediaPipe from ./vendor or any CDN"
      : "Opened as a file and the CDNs are unreachable. Serve this folder over http://localhost instead");
    setStatus("Loading face model…");
    const model = localModel || await fetchModel(FACE_MODEL_URLS, 1e6);
    const lmk = await createTask((fs, d) => FaceLandmarker.createFromOptions(fs, {
      baseOptions: { modelAssetBuffer: model, delegate: d }, runningMode: "VIDEO", numFaces: PHONE ? 2 : 3,
      outputFacialTransformationMatrixes: true,
      minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5 }));
    setStatus("Loading hair model…");
    try {
      const segModel = await fetchModel(SEG_MODEL_URLS, 1e6);
      segmenter = await createTask((fs, d) => ImageSegmenter.createFromOptions(fs, {
        baseOptions: { modelAssetBuffer: segModel, delegate: d }, runningMode: "VIDEO", outputCategoryMask: false, outputConfidenceMasks: true }));
      const warm = document.createElement("canvas"); warm.width = warm.height = 64; warm.getContext("2d").fillRect(0, 0, 64, 64);
      segmenter.segmentForVideo(warm, performance.now(), () => {});   // compile shaders now, not on the first hat with a brim
      if (PHONE) startSegWorker(segModel.slice());
    } catch (e) { console.warn("Hair segmentation unavailable, using a head-shaped mask instead:", e); segmenter = null; }
    landmarker = lmk; mode = "VIDEO";
    if (PHONE) startTrackWorker(model.slice());   // only now: both tasks are in VIDEO mode, so a photo loaded meanwhile waits in pendingPhoto
    setStatus(segmenter ? "Model ready" : "Model ready (no hair model)", "ok"); statusEl.title = "";
    $("camBtn").disabled = false; $("camBtn2").disabled = false; $("modelPick").style.display = "none";
    if (pendingPhoto) { const p = pendingPhoto; pendingPhoto = null; loadPhoto(p); }
  } catch (e) {
    console.error(e);
    const msg = String(e && e.message || e);
    setStatus("Model failed to load", "err"); statusEl.title = msg;
    $("hint").textContent = msg; $("modelPick").style.display = "";
  }
}
$("modelIn").addEventListener("change", async e => {
  const f = e.target.files[0]; e.target.value = ""; if (f) init(new Uint8Array(await f.arrayBuffer()));
});
async function setMode(m) {
  if (mode === m) return;
  await landmarker.setOptions({ runningMode: m }); if (segmenter) await segmenter.setOptions({ runningMode: m });
  mode = m;
}
/* face + hair masks for one frame; only needed when the hat has a part that goes behind the head */
const needMask = () => !!segmenter && ((S.occlude && !!(hat.brim || hat.custom)) || S.shadow > 0);   // the shadow also needs it: only the person takes it
let segTick = 0;
/* Phones: the hair mask runs in a worker, so the camera and the hat never wait for it.
   Each mask is placed with the head pose of the frame it was made from; maskFollow moves it with the head until the next one. */
let segWorker = null, segWorkerOn = false, segBusy = false, segId = 0, segSent = 0;
const segPoses = new Map();
function startSegWorker(model) {
  if (!window.Worker || !window.createImageBitmap) return;
  try { segWorker = new Worker(new URL("./seg-worker.js" + new URL(import.meta.url).search, import.meta.url)); }
  catch (e) { console.warn("Mask worker unavailable:", e); return; }
  segWorker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") { segWorkerOn = true; delegates.push("worker-" + m.delegate); }
    else if (m.type === "fail") { console.warn("Mask worker failed, masks stay on the main thread:", m.msg); segWorker.terminate(); segWorker = null; }
    else if (m.type === "error") { console.warn("Mask worker:", m.msg); }
    else if (m.type === "mask") {
      segBusy = false; prof.seg = prof.seg * 0.8 + (performance.now() - segSent) * 0.2;
      const pose = segPoses.get(m.id); segPoses.clear();
      if (!m.seg || !pose || !stream || !livePose) return;
      const W = frameCv.width, H = frameCv.height;
      m.seg.pose = pose; hairFit(m.seg, pose, W, H, true); lastLiveSeg = m.seg;
      if (needMask()) { setMask(m.seg, pose); maskFollow(livePose); }
    }
  };
  segWorker.postMessage({ type: "init", model, wasm: WASM_URLS, bundles: [
    ...(LOCAL ? [new URL("./vendor/vision_bundle.mjs", location.href).href] : []),
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/vision_bundle.mjs` ] }, [model.buffer]);
}
function segAsync(src, pose, t) {   // send one small frame; at most one in flight, and not more than ~8 a second
  if (segBusy || t - segSent < 120) return;
  segBusy = true; segSent = t; const id = ++segId;
  segPoses.set(id, { ...pose });
  createImageBitmap(segSmall(src)).then(bitmap => segWorker.postMessage({ type: "frame", bitmap, t, id }, [bitmap]))
    .catch(e => { segBusy = false; console.warn(e); });
}
// the masks come back at the size of the frame they were made from, as floats read back from the GPU:
// the model itself works at 256x256, so give it a small frame (much less to copy back and loop over)
const segCv = document.createElement("canvas"), segCtx = segCv.getContext("2d");
function segSmall(src) {
  const k = Math.min(1, (PHONE ? 384 : 512) / Math.max(src.width, src.height)), w = Math.round(src.width * k), h = Math.round(src.height * k);
  if (k === 1) return src;
  if (segCv.width !== w || segCv.height !== h) { segCv.width = w; segCv.height = h; }
  segCtx.drawImage(src, 0, 0, w, h); return segCv;
}
function segment(source, t) {
  let seg = null;
  const cb = r => {
    const m = r.confidenceMasks; if (!m || m.length < 4) return;
    // ears are labelled body skin (class 2), not face skin: keep them, or the hat behind the head is drawn over them
    seg = { hair: m[1].getAsFloat32Array().slice(), face: m[3].getAsFloat32Array().slice(), skin: m[2].getAsFloat32Array().slice(), w: m[1].width, h: m[1].height };
  };
  try { if (mode === "VIDEO") segmenter.segmentForVideo(source, t, cb); else segmenter.segment(source, cb); }
  catch (e) { console.warn("Segmentation failed, switching it off:", e); segmenter = null; }
  return seg;
}

/* ---------------- Camera ---------------- */
let lastLiveSeg = null, lastHint = 0, lightTick = 0;
let stream = null, lastVT = -1, fpsAvg = 0, lastT = 0, rafId = 0, mirrored = false, livePose = null, lastSeen = 0;
const HOLD_MS = 250;   // how long the hat stays put when tracking drops out for a moment (blink, motion blur, hand)
/* With other people in the picture, stay with the person already wearing the hat; otherwise take the biggest face
   (the one nearest the camera). Returns an index into the tracker's results, or -1. */
/* Phones: face tracking in a worker. The camera picture is drawn every frame; the hat uses the latest
   tracking result, carried forward by the head's speed to the frame on screen. */
let trackWorker = null, trackOn = false, trackBusy = false, trackId = 0, trackSent = 0, trackedPose = null, trackKey = "";
const vel = { x: 0, y: 0 };
function startTrackWorker(model) {
  if (!window.Worker || !window.createImageBitmap) return;
  try { trackWorker = new Worker(new URL("./track-worker.js" + new URL(import.meta.url).search, import.meta.url)); }
  catch (e) { console.warn("Tracking worker unavailable:", e); return; }
  trackWorker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") { trackOn = true; delegates.push("track-" + m.delegate); }
    else if (m.type === "fail") { console.warn("Tracking worker failed, tracking stays on the main thread:", m.msg); trackWorker.terminate(); trackWorker = null; }
    else if (m.type === "error") console.warn("Tracking worker:", m.msg);
    else if (m.type === "res") {
      trackBusy = false; prof.track = prof.track * 0.8 + (performance.now() - trackSent) * 0.2;
      const W = frameCv.width, H = frameCv.height;
      if (!stream || !m.res || m.key !== trackKey) return;   // camera stopped, or the frame size changed meanwhile
      const fi = pickFace(m.res, W, H, trackedPose), lm = fi < 0 ? null : m.res.faceLandmarks[fi];
      if (lm) {
        if (pickFace.switched) { resetFilters(); trackedPose = null; }
        if (pickFace.switched) headReset();
        const p = headPose(lm, m.res.facialTransformationMatrixes[fi], W, H, m.t, true); p._t = m.t; headFeed(lm, W, H, p);
        if (trackedPose) {
          const dt = m.t - trackedPose._t;
          if (dt > 0 && dt < 400) { vel.x = vel.x * 0.5 + (p.cx - trackedPose.cx) / dt * 0.5; vel.y = vel.y * 0.5 + (p.cy - trackedPose.cy) / dt * 0.5; }
        } else vel.x = vel.y = 0;
        trackedPose = p; lastSeen = m.t;
      } else if (!trackedPose || m.t - lastSeen > HOLD_MS) { trackedPose = null; vel.x = vel.y = 0; resetFilters(); }
    }
  };
  trackWorker.postMessage({ type: "init", model, numFaces: 2, wasm: WASM_URLS, bundles: [
    ...(LOCAL ? [new URL("./vendor/vision_bundle.mjs", location.href).href] : []),
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/vision_bundle.mjs` ] }, [model.buffer]);
}
function sendTrack(t) {
  trackBusy = true; trackSent = t; const id = ++trackId, key = trackKey;
  createImageBitmap(frameCv).then(bitmap => trackWorker.postMessage({ type: "frame", bitmap, t, id, key }, [bitmap]))
    .catch(e => { trackBusy = false; console.warn(e); });
}
function predictPose(t, W, H) {   // the last tracked pose, moved on by the head's speed (damped, at most 150 ms ahead)
  const p = trackedPose; if (!p) return null;
  const dt = Math.min(150, Math.max(0, t - p._t)), dx = vel.x * dt * 0.8, dy = vel.y * dt * 0.8;
  if (!dx && !dy) return p;
  return { ...p, cx: p.cx + dx, cy: p.cy + dy, lm: p.lm.map(a => ({ x: a.x + dx / W, y: a.y + dy / H, z: a.z })) };
}
function pickFace(res, W, H, prev) {
  const faces = res.faceLandmarks || []; let best = -1, score = Infinity;
  for (let i = 0; i < faces.length; i++) {
    const a = faces[i][234], b = faces[i][454], cx = (a.x + b.x) / 2 * W, cy = (a.y + b.y) / 2 * H, w = Math.hypot((b.x - a.x) * W, (b.y - a.y) * H);
    const s = prev ? Math.hypot(cx - prev.cx, cy - prev.cy) / prev.w + Math.abs(Math.log(w / prev.w))
      : -w * (1 - 0.6 * Math.hypot(cx / W - 0.5, cy / H - 0.5));   // biggest, nearest the middle
    if (s < score) { score = s; best = i; }
  }
  pickFace.switched = !!prev && best >= 0 && score > 1.2;
  if (pickFace.switched) { const i = pickFace(res, W, H, null); pickFace.switched = true; return i; }   // the tracked person left: start over with the biggest face, without sliding the hat across
  return best;
}
const frameCv = document.createElement("canvas"), frameCtx = frameCv.getContext("2d");
async function startCamera() {
  if (stream) { stopCamera(); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: PHONE ? { facingMode: "user" } : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
  } catch (e) { setStatus("Camera blocked — allow access or open via http://localhost", "err"); return; }
  video.srcObject = stream; await video.play();
  await setMode("VIDEO"); resetFilters(); headReset(); trackedPose = null; vel.x = vel.y = 0;
  still = null; stillPose = null; mirrored = true; out.classList.add("mirror"); ovl.classList.add("mirror");
  { const ps = Math.min(1, PROC_MAX / Math.max(video.videoWidth, video.videoHeight)); frameCv.width = Math.round(video.videoWidth * ps); frameCv.height = Math.round(video.videoHeight * ps); }
  $("empty").style.display = "none"; $("snapBtn").disabled = false; cmp.disabled = false; fpsEl.style.display = ""; vp.classList.add("live");
  $("camBtn").textContent = "Stop camera"; setStatus("Model ready", "ok");
  loop();
}
function stopCamera() {
  cancelAnimationFrame(rafId); stream.getTracks().forEach(t => t.stop()); stream = null; livePose = null;
  $("camBtn").textContent = "Start camera"; fpsEl.style.display = "none"; $("hint").textContent = "Camera stopped.";
  if (!still) {
    $("empty").style.display = ""; $("snapBtn").disabled = true; cmp.disabled = true; vp.classList.remove("live");
    clearView();
  }
}
const DEBUG = /[?&]debug/.test(location.search), prof = { track: 0, seg: 0, render: 0 };
const cutBox = new THREE.Box3();
function hatCut(H) {   // is the top of the hat above the camera picture (head too close to the top edge)?
  if (!headG.visible || !hat.group) return false;
  cutBox.setFromObject(hat.group); return cutBox.max.y > H + 2;
}
function loop() {
  rafId = requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVT) return;
  lastVT = video.currentTime;
  // the camera's frame size can change after it starts (iOS reports it late, and turning a phone swaps width and height)
  const vw = video.videoWidth, vh = video.videoHeight, ps = Math.min(1, PROC_MAX / Math.max(vw, vh));   // phones: work on a smaller frame
  if (vw && (Math.round(vw * ps) !== frameCv.width || Math.round(vh * ps) !== frameCv.height)) { frameCv.width = Math.round(vw * ps); frameCv.height = Math.round(vh * ps); resetFilters(); livePose = null; lastLiveSeg = null; trackedPose = null; }
  const t = performance.now(), W = frameCv.width, H = frameCv.height;
  // freeze ONE frame: track and render on the same pixels, so the hat can't trail the head
  frameCtx.drawImage(video, 0, 0, W, H);
  if (trackOn) {   // phone: never wait for the tracker
    trackKey = W + "x" + H;
    if (!trackBusy) sendTrack(t);
    livePose = predictPose(t, W, H);
    if (livePose && segWorkerOn) {
      if (segmenter && (needMask() || segTick++ % 12 === 0)) segAsync(frameCv, livePose, t);
      setMask(needMask() ? lastLiveSeg : null, livePose); maskFollow(livePose);
    } else if (!livePose) MU.uMaskOn.value = 0;
  } else {
  const res = landmarker.detectForVideo(frameCv, t); prof.track = prof.track * 0.9 + (performance.now() - t) * 0.1;
  const fi = pickFace(res, W, H, livePose), lm = fi < 0 ? null : res.faceLandmarks[fi];
  if (lm) {
    if (pickFace.switched) { resetFilters(); headReset(); }
    livePose = headPose(lm, res.facialTransformationMatrixes && res.facialTransformationMatrixes[fi], W, H, t, true); lastSeen = t;
    headFeed(lm, W, H, livePose);
    // the hair/face mask is the heaviest step: on a slow device (a phone) refresh it every 2nd or 3rd frame and reuse it between
    const every = PHONE ? (!fpsAvg || fpsAvg > 20 ? 3 : fpsAvg > 13 ? 5 : 8) : !fpsAvg || fpsAvg > 24 ? 1 : fpsAvg > 15 ? 2 : 3;
    const want = needMask() ? segTick++ % every === 0 : segTick++ % 12 === 0;
    if (segWorkerOn) {   // phone: masks arrive from the worker, nothing here waits for them
      if (segmenter && (needMask() || segTick++ % 12 === 0)) segAsync(frameCv, livePose, t);
      setMask(needMask() ? lastLiveSeg : null, livePose); maskFollow(livePose);
    } else {
    const t1 = performance.now(), seg = segmenter && want ? segment(segSmall(frameCv), t) : null; if (seg) prof.seg = prof.seg * 0.8 + (performance.now() - t1) * 0.2;
    if (seg) { hairFit(seg, livePose, W, H, true); lastLiveSeg = seg; }
    setMask(needMask() ? seg || lastLiveSeg : null, livePose); maskFollow(livePose);
    }
  } else if (!livePose || t - lastSeen > HOLD_MS) { livePose = null; resetFilters(); }   // a missed frame or two: keep the hat where it was
  }
  const t2 = performance.now(); render(frameCv, W, H, livePose); prof.render = prof.render * 0.9 + (performance.now() - t2) * 0.1;
  if (t - lastHint > 250) { lastHint = t; $("hint").textContent = livePose && hatCut(H) ? "The hat goes past the top of the picture: move the phone back a little or tilt it up." : livePose ? "Tracking · " + poseText(livePose) + " · " + MOVE_TIP : "No face found. Face the camera and keep your whole head in frame."; }   // text 4x a second: re-layout every frame costs on phones
  if (lastT) { const f = 1000 / (t - lastT); fpsAvg = fpsAvg ? fpsAvg * 0.9 + f * 0.1 : f; fpsEl.textContent = fpsAvg.toFixed(0) + " fps" + (DEBUG ? ` · track ${prof.track | 0} · mask ${prof.seg | 0} · draw ${prof.render | 0} ms · ${W}x${H} · ${delegates.join("/")} · head ${headFitter?.result ? `depth ${((headTarget.kz - 1) * 100).toFixed(0)}% crown ${(headTarget.dTop * 100).toFixed(1)}` : headFitter ? Math.round(headFitter.progress * 100) + "%" : "-"}` : ""); }
  lastT = t;
}
$("camBtn").onclick = startCamera; $("camBtn2").onclick = startCamera;

/* ---------------- Photo ---------------- */
let still = null, stillPose = null, stillSeg = null, pendingPhoto = null;
async function loadPhoto(img) {
  if (!landmarker) { pendingPhoto = img; return; }
  if (stream) stopCamera();
  const k = Math.min(1, 1600 / Math.max(img.naturalWidth, img.naturalHeight));
  still = document.createElement("canvas"); still.width = Math.round(img.naturalWidth * k); still.height = Math.round(img.naturalHeight * k);
  still.getContext("2d").drawImage(img, 0, 0, still.width, still.height);
  await setMode("IMAGE");
  mirrored = false; out.classList.remove("mirror"); ovl.classList.remove("mirror");
  const res = landmarker.detect(still), fi = pickFace(res, still.width, still.height, null), lm = fi < 0 ? null : res.faceLandmarks[fi];
  stillPose = lm ? headPose(lm, res.facialTransformationMatrixes && res.facialTransformationMatrixes[fi], still.width, still.height, 0, false) : null;
  headReset(); if (lm) headFeed(lm, still.width, still.height, stillPose, 1);   // a photo: one frame is all there is
  stillSeg = null;
  $("empty").style.display = "none"; $("snapBtn").disabled = false; cmp.disabled = false; vp.classList.toggle("live", !!stillPose);
  refreshStill();
}
function refreshStill() {
  if (!still || stream) return;
  if (segmenter && stillSeg === null) { stillSeg = (stillPose && segment(still, 0)) || false; if (stillSeg) hairFit(stillSeg, stillPose, still.width, still.height, false); }   // false: tried, nothing to use
  if (!stillSeg) { hairScale = HAT_SCALE; hairTop = null; }
  setMask(needMask() && stillSeg ? stillSeg : null, stillPose); maskPose = null; maskFollow(null);
  $("hint").textContent = stillPose ? "Face found · " + poseText(stillPose) + " · " + MOVE_TIP : "No face found. Try a clearer photo with the whole head visible.";
  requestStill();
}
$("photoIn").addEventListener("change", e => {
  const f = e.target.files[0]; e.target.value = ""; if (!f) return;
  const img = new Image();
  img.onload = () => { loadPhoto(img); URL.revokeObjectURL(img.src); };
  img.onerror = () => { $("hint").textContent = "Could not read that image."; };
  img.src = URL.createObjectURL(f);
});
let stillPending = false, lastSegWanted = false;
function requestStill() {
  if (!still || stream || stillPending) return; stillPending = true;
  requestAnimationFrame(() => {
    stillPending = false;
    if (needMask() !== lastSegWanted || (segmenter && stillSeg === null)) { lastSegWanted = needMask(); refreshStill(); return; }
    render(still, still.width, still.height, stillPose);
  });
}

/* ---------------- Drag to move; scroll or pinch to resize ---------------- */
let drag = null;
const touches = new Map();   // pointers down on the picture: one drags the hat, two pinch to resize it
let pinch = null;
const curPose = () => stream ? livePose : stillPose;
const pinchDist = () => { const [a, b] = [...touches.values()]; return Math.hypot(a.x - b.x, a.y - b.y); };
vp.addEventListener("pointerdown", e => {
  if (!curPose() || e.target !== out) return;
  touches.set(e.pointerId, { x: e.clientX, y: e.clientY }); vp.setPointerCapture(e.pointerId);
  if (touches.size === 2) { drag = null; pinch = { d: pinchDist(), size: hat.fit.size }; }
  else if (touches.size === 1) drag = { x: e.clientX, y: e.clientY };
});
vp.addEventListener("pointermove", e => {
  if (!touches.has(e.pointerId)) return;
  touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
  const p = curPose(); if (!p) return;
  if (pinch && touches.size === 2) {
    const el = $("size"); hat.fit.size = Math.min(+el.max, Math.max(+el.min, pinch.size * pinchDist() / Math.max(10, pinch.d)));
    syncFitUI(); requestStill(); return;
  }
  if (!drag) return;
  const b = out.getBoundingClientRect(), s = 1 / Math.min(b.width / out.width, b.height / out.height);
  const dx = (e.clientX - drag.x) * s * (mirrored ? -1 : 1), dy = (e.clientY - drag.y) * s;
  drag = { x: e.clientX, y: e.clientY };
  const cr = Math.cos(p.roll), sr = Math.sin(p.roll), clamp = (v, el) => Math.min(+el.max, Math.max(+el.min, v));
  hat.fit.x = clamp(hat.fit.x + (dx * cr + dy * sr) / p.w, $("x"));
  hat.fit.y = clamp(hat.fit.y + (dx * sr - dy * cr) / p.w, $("y"));
  syncFitUI(); requestStill();
});
for (const ev of ["pointerup", "pointercancel"]) vp.addEventListener(ev, e => {
  touches.delete(e.pointerId); if (touches.size < 2) pinch = null;
  if (touches.size === 1) { const [q] = touches.values(); drag = { x: q.x, y: q.y }; } else drag = null;   // lifting one finger of a pinch: carry on dragging
});
vp.addEventListener("wheel", e => {
  if (!curPose()) return; e.preventDefault();
  const el = $("size"); hat.fit.size = Math.min(+el.max, Math.max(+el.min, hat.fit.size * Math.exp(-e.deltaY * 0.001)));
  syncFitUI(); requestStill();
}, { passive: false });

/* ---------------- Snapshot ---------------- */
$("snapBtn").onclick = () => {
  renderer.render(scene, cam);   // the drawing buffer is not kept between frames: draw it again right before copying it
  const c = document.createElement("canvas"); c.width = out.width; c.height = out.height; const x = c.getContext("2d");
  if (mirrored) { x.translate(c.width, 0); x.scale(-1, 1); }
  x.drawImage(out, 0, 0);
  c.toBlob(async blob => {
    const file = new File([blob], "hat-tryon.png", { type: "image/png" });
    if (PHONE && navigator.canShare && navigator.canShare({ files: [file] })) { try { await navigator.share({ files: [file], title: "Hat try-on" }); return; } catch (e) { if (e.name === "AbortError") return; } }
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "hat-tryon.png"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }, "image/png");
};

if (matchMedia("(max-width: 760px)").matches) $("settings").open = false;   // phones: the picture and hats first, settings folded
if (TOUCH) $("hint").textContent = "Tip: drag the hat to move it, pinch to resize it.";
renderSwatches(); selectHat(hat);
window.hatTryOn = { get eyeLift() { return eyeLift; }, get pose() { return curPose(); }, get hat() { return hat; }, get hairScale() { return hairScale; }, get hairTop() { return hairTop; }, get head() { return { ...headTarget, weights: undefined, raw: undefined }; }, light, hats, S };   // for debugging from the console
init();
