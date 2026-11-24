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
function beanie(color, bobble) {
  const body = fabric(color, "knit", 34, 5), cuff = fabric(color, "knit", 30, 2);
  /* A knit hugs the skull. Its edge is not a flat ring: it dips over the tops of the ears at the sides (SIDE),
     the cuff is thin and turned in at the bottom, and a little slack gathers at the back of the crown. */
  const SIDE = 0.13, dip = (th, k = 1) => SIDE * Math.pow(Math.sin(th), 2) * k;
  const parts = [
    [ring(v => { const a = v * PI / 2; return [Math.pow(Math.cos(a), 0.75) * (1 - 0.03 * v), 0.1 + 0.5 * Math.sin(a)]; },
      (p, th, v) => { p[1] -= dip(th, (1 - v) * (1 - v)); const back = Math.max(0, -Math.cos(th)); p[2] -= 0.07 * back * Math.sin(PI * v); p[1] += 0.025 * back * v; }, 32), body],
    [ring(v => [0.985 + 0.03 * Math.pow(Math.sin(PI * (0.25 + 0.75 * v)), 0.5) - 0.025 * (1 - v) * (1 - v), -0.04 + 0.17 * v],
      (p, th) => { p[1] -= dip(th); }, 12), cuff],
  ];
  if (bobble) {
    const g = new THREE.SphereGeometry(0.13, 32, 24), p = g.attributes.position;
    for (let i = 0; i < p.count; i++) { const k = 1 + (Math.random() - 0.5) * 0.14; p.setXYZ(i, p.getX(i) * k, p.getY(i) * k, p.getZ(i) * k); }
    g.computeVertexNormals(); g.translate(0, 0.68, -0.06);
    parts.push([g, new THREE.MeshStandardMaterial({ color: 0xf2efe8, roughness: 1 }), false]);
  }
  return build(parts);
}
function cap(color) {
  const m = fabric(color, "panel", 6, 1, { bump: 0.8 }), vm = fabric(color, "stitched", 1, 1, { bump: 0.8 });
  const dome = ring(v => { const a = v * PI / 2; return [Math.pow(Math.cos(a), 0.5), 0.44 * Math.sin(a)]; },
    (p, th, v) => { p[2] += 0.05 * Math.sin(PI * v) * Math.max(0, Math.cos(th)); }, 32);
  const visor = surf((u, v) => {
    const s = u * 2 - 1, th = s * 0.8, L = 0.55 * Math.pow(Math.cos(s * PI / 2), 0.6);
    const d = Math.hypot(0.5 * Math.sin(th), 1);
    return [A * Math.sin(th) + 0.5 * Math.sin(th) / d * L * v, 0.012 - 0.5 * v * L - 0.03 * s * s * v, B * Math.cos(th) + L * v / d];
  }, 48, 10, false);
  const button = new THREE.SphereGeometry(0.035, 16, 12); button.scale(1, 0.5, 1); button.translate(0, 0.44, 0);
  return build([[dome, m], [visor, vm], [button, vm]]);
}
function bucket(color) {
  const m = fabric(color, "cotton", 10, 3), bm = fabric(color, "stitched", 10, 1);
  return build([
    [ring(v => {
      if (v < 0.6) { const k = v / 0.6; return [1 - 0.1 * k, 0.34 * k]; }
      const k = (v - 0.6) / 0.4; return [0.9 * (1 - k * k), 0.34 + 0.035 * Math.sin(Math.min(1, k * 3) * PI / 2)];
    }, null, 30), m],
    [ring((v, th) => [1 + 0.42 * v, -0.2 * Math.pow(v, 1.15) + 0.012 * Math.sin(3 * th + 1) * v], null, 12), bm],
  ]);
}
function fedora(color) {
  const m = fabric(color, "felt", 8, 4, { bump: 0.6 });
  const H = 0.44;
  return build([
    [crown(H, 0.22, 0.07, creased(H + 0.04, 0.085, 0.08), 0.6, 0.9), m],
    // brim dips a little in front and its edge rolls up, most at the sides and back
    [ring((v, th) => { const c = Math.cos(th), e = sm(0.3, 1, v); return [1 + 0.44 * v - 0.03 * e * e, -0.02 * v - 0.05 * v * Math.max(0, c) + e * e * (0.055 + 0.035 * Math.sin(th) ** 2 + 0.03 * Math.max(0, -c))]; }, null, 20), m],
    [band(1, 1 - 0.22 * 0.3, 0.004, 0.125, darker(color, 0.28)), null],
  ]);
}
function sunhat(color) {
  const m = fabric(color, "straw", 26, 6, { bump: 1.6, rough: 0.8 });
  return build([
    [ring(v => {
      if (v < 0.5) { const k = v / 0.5; return [1 - 0.06 * k, 0.3 * k]; }
      const a = (v - 0.5) / 0.5 * PI / 2; return [0.94 * Math.pow(Math.cos(a), 0.8), 0.3 + 0.1 * Math.sin(a)];
    }, null, 30), m],
    [ring((v, th) => [1 + 1.05 * v, -0.03 * v - 0.16 * Math.pow(v, 2.2) + 0.02 * Math.sin(3 * th) * v * v], null, 20), m],
    [band(1, 0.985, 0.004, 0.1, 0x2b3a55), null],
  ]);
}
function cowboy(color) {
  const m = fabric(color, "felt", 8, 4, { bump: 0.6 });
  const H = 0.52;
  return build([
    [crown(H, 0.15, 0.05, creased(H, 0.13, 0.1)), m],
    [ring((v, th) => { const s2 = Math.sin(th) ** 2; return [1 + v * (0.62 + 0.22 * s2), -0.03 * v - 0.06 * v * (1 - s2) + 0.32 * v * v * s2]; }, null, 16), m],
    [band(1, 1 - 0.15 * 0.2, 0.004, 0.07, darker(color, 0.3)), null],
  ]);
}
function tophat(color) {
  const m = fabric(color, "felt", 8, 6, { bump: 0.3, rough: 0.55 });
  const H = 0.62;
  return build([
    [ring(v => {
      if (v < 0.72) { const k = v / 0.72; return [1 - 0.035 * Math.sin(PI * k) + 0.05 * k, H * k]; }
      const k = (v - 0.72) / 0.28; return [1.05 * (1 - Math.pow(k, 1.5)), H + 0.012 * Math.sin(Math.min(1, k * 5) * PI / 2)];
    }, null, 36), m],
    [ring((v, th) => { const s2 = Math.sin(th) ** 2; return [1 + 0.3 * v, -0.01 * v - 0.02 * v * (1 - s2) + 0.09 * v * v * s2]; }, null, 10), m],
    [band(1, 1.008, 0.004, 0.1, 0x6e1620), null],
  ]);
}

/* fit: defaults for the fit sliders. brim: the hat has parts that can end up behind the head */
export const HATS = [
  { name: "Boonie",  model: true, url: "models/hats/fishermans_hat/fishermans_hat_1k.gltf", brim: true, drop: 0.12, fit: {} },   // photo-scanned hat, CC0, polyhaven.com
  { name: "Beanie",  color: "#a8342b", make: c => beanie(c, false), snug: 0.96, fit: { y: -0.06 } },   // pulled down to mid-forehead
  { name: "Cap",     color: "#1f3557", make: cap,     brim: true, drop: 0.13, snug: 0.98, fit: {} },
  { name: "Fedora",  color: "#6a4a34", make: fedora,  brim: true, drop: 0.06, crownTop: 0.47, fit: {} },
  { name: "Sun hat", color: "#d9bd86", make: sunhat,  brim: true, drop: 0.12, fit: {} },
  { name: "Cowboy",  color: "#a5733d", make: cowboy,  brim: true, drop: 0.06, crownTop: 0.45, fit: {} },
  { name: "Top hat", color: "#1d1d22", make: tophat,  brim: true, fit: {} },
];
