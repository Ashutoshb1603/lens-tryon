/* Built-in hats as 3D models. Units are head widths (ear to ear = 1). A hat's origin is the centre of its band:
   an ellipse with half-axes A (side to side) and B (front to back), y up, +z to the front. */
import * as THREE from "./vendor/three.module.min.js";

export const A = 0.57, B = 0.68;
const PI = Math.PI, TAU = PI * 2;
const sm = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

/* ---------- geometry ---------- */
function surf(fn, nu, nv, closed) {
  const pos = [], uv = [], idx = [];
  for (let j = 0; j <= nv; j++) for (let i = 0; i <= nu; i++) { pos.push(...fn(i / nu, j / nv)); uv.push(i / nu, j / nv); }
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    const a = j * (nu + 1) + i, b = a + 1, c = a + nu + 1, d = c + 1; idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx); g.computeVertexNormals();
  if (closed) {   // weld the normals along the seam
    const n = g.attributes.normal;
    for (let j = 0; j <= nv; j++) {
      const a = j * (nu + 1), b = a + nu;
      const v = new THREE.Vector3(n.getX(a) + n.getX(b), n.getY(a) + n.getY(b), n.getZ(a) + n.getZ(b)).normalize();
      n.setXYZ(a, v.x, v.y, v.z); n.setXYZ(b, v.x, v.y, v.z);
    }
  }
  return g;
}
/* surface of revolution around the band ellipse: profile(v, th) → [r, y]; warp(p, th, v) may bend it */
function ring(profile, warp, nv = 28, nu = 96) {
  return surf((u, v) => {
    const th = u * TAU, [r, y] = profile(v, th), p = [A * r * Math.sin(th), y, B * r * Math.cos(th)];
    if (warp) warp(p, th, v);
    return p;
  }, nu, nv, true);
}

/* ---------- fabric textures ---------- */
function canvasTex(draw, size = 256) {
  const c = document.createElement("canvas"); c.width = c.height = size; const g = c.getContext("2d");
  draw(g, size);
  const mk = srgb => { const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 4; if (srgb) t.colorSpace = THREE.SRGBColorSpace; return t; };
  return { map: mk(true), bump: mk(false) };
}
function noise(g, s, lo, hi, cell = 1) {
  for (let y = 0; y < s; y += cell) for (let x = 0; x < s; x += cell) {
    const v = Math.round(255 * (lo + Math.random() * (hi - lo))); g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(x, y, cell, cell);
  }
}
const TEX = {};
const tex = name => TEX[name] || (TEX[name] = canvasTex(MAKE[name]));
const MAKE = {
  felt: (g, s) => { noise(g, s, 0.86, 1); g.globalAlpha = 0.35; noise(g, s, 0.8, 1, 3); },
  knit: (g, s) => {   // one rib per 32 px: a raised column of V stitches
    g.fillStyle = "#777"; g.fillRect(0, 0, s, s);
    for (let x = 0; x < s; x += 32) {
      const gr = g.createLinearGradient(x, 0, x + 32, 0);
      gr.addColorStop(0, "#5c5c5c"); gr.addColorStop(0.5, "#fff"); gr.addColorStop(1, "#5c5c5c");
      g.fillStyle = gr; g.fillRect(x, 0, 32, s);
      g.strokeStyle = "rgba(0,0,0,.35)"; g.lineWidth = 2;
      for (let y = 0; y < s; y += 16) { g.beginPath(); g.moveTo(x + 6, y); g.lineTo(x + 16, y + 10); g.lineTo(x + 26, y); g.stroke(); }
    }
  },
  cotton: (g, s) => {
    noise(g, s, 0.88, 1);
    g.globalAlpha = 0.14; g.strokeStyle = "#000"; g.lineWidth = 1;
    for (let i = -s; i < s * 2; i += 4) { g.beginPath(); g.moveTo(i, 0); g.lineTo(i + s, s); g.stroke(); }
  },
  panel: (g, s) => {   // cotton twill with one seam and its two rows of stitching
    MAKE.cotton(g, s); g.globalAlpha = 1;
    g.fillStyle = "rgba(0,0,0,.45)"; g.fillRect(0, 0, 3, s); g.fillRect(s - 3, 0, 3, s);
    g.strokeStyle = "rgba(0,0,0,.3)"; g.setLineDash([7, 5]); g.lineWidth = 2;
    for (const x of [12, s - 12]) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, s); g.stroke(); }
  },
  stitched: (g, s) => {   // cotton with rows of stitching across v
    MAKE.cotton(g, s); g.globalAlpha = 1; g.strokeStyle = "rgba(0,0,0,.32)"; g.setLineDash([6, 4]); g.lineWidth = 2;
    for (let y = s / 12; y < s; y += s / 6) { g.beginPath(); g.moveTo(0, y); g.lineTo(s, y); g.stroke(); }
  },
  straw: (g, s) => {
    g.fillStyle = "#e9e9e9"; g.fillRect(0, 0, s, s);
    for (let y = 0; y < s; y += 16) {
      const gr = g.createLinearGradient(0, y, 0, y + 16);
      gr.addColorStop(0, "#8a8a8a"); gr.addColorStop(0.45, "#fff"); gr.addColorStop(1, "#9a9a9a");
      g.fillStyle = gr; g.fillRect(0, y, s, 16);
      g.strokeStyle = "rgba(0,0,0,.28)"; g.lineWidth = 1.5;
      for (let x = (y / 16 % 2) * 8; x < s; x += 16) { g.beginPath(); g.moveTo(x, y + 2); g.lineTo(x + 7, y + 14); g.stroke(); }
    }
  },
  ribbon: (g, s) => {
    for (let y = 0; y < s; y += 4) { g.fillStyle = y % 8 ? "#fff" : "#b9b9b9"; g.fillRect(0, y, s, 4); }
  },
};
function fabric(color, kind, ru, rv, o = {}) {
  const t = tex(kind), map = t.map.clone(), bump = t.bump.clone();
  map.repeat.set(ru, rv); bump.repeat.set(ru, rv); map.needsUpdate = bump.needsUpdate = true;
  return new THREE.MeshStandardMaterial({ color, map, bumpMap: bump, bumpScale: (o.bump ?? 1.2) * 2, roughness: o.rough ?? 0.92, metalness: 0, side: THREE.DoubleSide });
}
const darker = (hex, k) => new THREE.Color(hex).multiplyScalar(k);

/* ---------- shared parts ---------- */
/* crown with straight-ish sides and a rounded top. taper: how much narrower the top is; H: height; dome: extra rise of the top */
function crown(H, taper, dome, warp, side = 0.62, round = 0.7) {
  return ring(v => {
    if (v < side) { const k = v / side; return [1 - taper * k, H * k]; }
    const k = (v - side) / (1 - side), a = k * PI / 2;
    return [(1 - taper) * Math.pow(Math.cos(a), round), H + dome * Math.sin(a)];
  }, warp, 40);
}
/* centre crease and two front pinches, as on a fedora */
const creased = (H, dent, pinch) => p => {
  const top = sm(H * 0.7, H, p[1]);
  p[1] -= dent * top * Math.exp(-((p[0] / 0.14) ** 2)) * (0.55 + 0.45 * sm(-B, B * 0.6, p[2]));
  if (p[2] > 0) p[2] -= pinch * Math.exp(-(((Math.abs(p[0]) - 0.2) / 0.12) ** 2) - ((p[1] - H * 0.82) / (H * 0.3)) ** 2);
};
const band = (r0, r1, y0, y1, color) => {
  const m = fabric(color, "ribbon", 1, 6, { rough: 0.6, bump: 0.5 });
  return new THREE.Mesh(ring(v => [r0 + (r1 - r0) * v + 0.012, y0 + (y1 - y0) * v], null, 2), m);
};
function build(parts) {
  const g = new THREE.Group(), tint = [];
  for (const [geo, mat, isTint = true] of parts) {
    const m = geo.isMesh ? geo : new THREE.Mesh(geo, mat);
    if (!geo.isMesh && isTint) tint.push(mat);
    g.add(m);
  }
  return { group: g, tint };
}

/* ---------- hats ---------- */

/* fit: defaults for the fit sliders. brim: the hat has parts that can end up behind the head */
export const HATS = [
  { name: "Beanie",  model: true, url: "models/hats/knit_beanie.glb", snug: 0.97, sides: 1, fitModel: { ref: 0.2, base: 0.15, cref: 0.35 }, fit: { y: -0.13, curve: -12, stretch: 0.72 } },   // "Black Knit Beanie" by Arkify, CC BY 4.0, sketchfab.com. The scan already slouches back ~19°.
  { name: "Bucket",  model: true, url: "models/hats/bucket_hat.glb", brim: true, drop: 0.15, sides: 0.8, fitModel: { ref: 0.45, base: 0.32, cref: 0.55 }, fit: { size: 0.88, y: 0.07, stretch: 0.8 } },   // deep straight crown: smaller, shorter, higher. "Bucket Hat" by oluwaphemi, CC BY 4.0, sketchfab.com
  { name: "Flat cap", model: true, url: "models/hats/flat_cap.glb", brim: true, drop: 0.1, sides: 0.6, stretchCrown: true, fitModel: { ref: 0.1, base: 0.06, cref: 0.4 }, fit: { size: 1.08, y: -0.1 } },   // worn low: visor just above the brows. "Flat Cap" by Fridge (youssefdarwish01), CC BY 4.0, sketchfab.com
  { name: "Dipper cap", model: true, url: "models/hats/dipper_cap.glb", brim: true, drop: 0.13, snug: 0.98, sides: 0.6, fitModel: { turn: Math.PI, ref: 0.08, base: 0.02, cref: 0.5 }, fit: { y: -0.12, stretch: 0.85 } },   // visor at mid-forehead, crown less tall   // "Dipper's cap" by Nick Pekarsky, CC BY 4.0, sketchfab.com; character design (c) Disney
  { name: "Fedora",  model: true, url: "models/hats/classic_fedora.glb", brim: true, drop: 0.06, fitModel: { ref: 0.35, base: 0.16, cref: 0.45 }, sides: 0.55, fit: { y: 0.02, curve: 4 } },   // steep snap brim: sit higher, tipped back a touch   // "Classic Fedora Hat" by ghostmachinesing, CC BY 4.0, sketchfab.com
  { name: "Bowler",  model: true, url: "models/hats/bowler_hat.glb", brim: true, drop: 0.06, sides: 0.35, fitModel: { ref: 0.2, base: 0.12, cref: 0.4 }, fit: { y: 0.04, stretch: 0.85 } },   // "Bowler Hat" by Joycestick (piperno.b), CC BY 4.0, sketchfab.com
  { name: "Straw hat", model: true, url: "models/hats/straw_hat.glb", brim: true, drop: 0.12, sides: 0.6, fitModel: { ref: 0.58, base: 0.48, cref: 0.65 }, fit: { y: 0.24, size: 0.9 } },   // its front and back brim dip ~7 cm below the band: the band sits high so the front clears the eyes   // "Unisex Summer Straw Hat" by arloopa, Sketchfab Standard licence
  { name: "Officer", model: true, url: "models/hats/military_cap.glb", brim: true, drop: 0.1, fitModel: { ref: 0.15, cref: 0.3, base: 0.04 }, sides: 0.45, fit: { y: -0.04 } },   // "Military Hat" by aSocialCrab, CC BY 4.0, sketchfab.com
];
