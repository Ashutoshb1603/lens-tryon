/* Personal head shape, estimated once per person.

   The camera only sees the face; the skull behind and above it is hidden under hair. A 3D head model with
   identity modes (ICT Face Model, USC ICT, MIT licence; see CREDITS.md) links the two: its modes are fitted to
   the person's face landmarks, collected over about a second of near-frontal frames, and the fitted head's
   scalp is read off as an ellipsoid. Everything is in head widths about the ear line, like the app's own pose,
   so the result is used as a ratio to the average head: an average person gets exactly the hand-tuned fit.

   Data: models/headshape.json + .bin, built by tools/build_headshape.py, calibrated by tools/calibrate_headshape.py. */

const FRAMES = 24;          // near-frontal frames averaged before fitting
const LAMBDA = 1.0;         // prior strength: higher keeps the estimate closer to the average head
const ZW = 0.5;             // the tracker's depth is less reliable than x/y

export async function loadHeadModel(base) {
  const meta = await (await fetch(base + ".json")).json();
  const buf = await (await fetch(base + ".bin")).arrayBuffer();
  const B = {};
  for (const b of meta.blocks) {
    const n = b.shape.reduce((a, c) => a * c, 1), q = new Int16Array(buf, b.offset, n), f = new Float32Array(n);
    for (let i = 0; i < n; i++) f[i] = q[i] * b.scale;
    B[b.name] = f;
  }
  const m = { K: meta.K, lm: meta.lm, nL: meta.nL, nS: meta.nS, ...B };
  // real tracker output differs a little from the model's average face: this offset (the average of real faces) is removed
  m.bias = meta.bias ? Float64Array.from(meta.bias) : new Float64Array(meta.nL * 3);
  // only the face above the nose: the jaw and chin move with the mouth (smiling, talking), the rest belongs to the head
  const at0 = Object.fromEntries(meta.lm.map((v, i) => [v, i])), yN = B.Lmean[at0[1] * 3 + 1], yCut = yN - 0.15 * (yN - B.Lmean[at0[152] * 3 + 1]);   // just under the nose
  const keep = meta.lm.map((v, i) => i).filter(i => B.Lmean[i * 3 + 1] >= yCut || [234, 454, 10].includes(meta.lm[i]));
  const pick = (arr, per) => { const out = new Float32Array(arr.length / meta.nL * keep.length), k = arr.length / per / meta.nL;
    for (let c = 0; c < k; c++) keep.forEach((i, j) => { for (let d = 0; d < 3; d++) out[(c * keep.length + j) * 3 + d] = arr[(c * meta.nL + i) * 3 + d]; }); return out; };
  m.Lmean = pick(m.Lmean, 3); m.Lmodes = pick(m.Lmodes, 3); m.bias = pick(m.bias, 3);
  m.lm = keep.map(i => meta.lm[i]); m.nL = keep.length;
  m.at = Object.fromEntries(m.lm.map((v, i) => [v, i]));
  // normal equations for the identity weights do not depend on the person: build them once
  const K = m.K, nL = m.nL, G = new Float64Array(K * K);
  for (let a = 0; a < K; a++) for (let b = a; b < K; b++) {
    let s = 0;
    for (let i = 0; i < nL * 3; i++) { const w = i % 3 === 2 ? ZW * ZW : 1; s += m.Lmodes[a * nL * 3 + i] * m.Lmodes[b * nL * 3 + i] * w; }
    G[a * K + b] = G[b * K + a] = s + (a === b ? LAMBDA : 0);
  }
  m.chol = cholesky(G, K);
  m.mean = measure(m, new Float64Array(K));
  return m;
}

/* landmarks (normalised, as the tracker gives them) -> points in a right-handed frame, y up, z towards the camera */
function obsPoints(m, lm, W, H) {
  const o = new Float64Array(m.nL * 3);
  m.lm.forEach((k, i) => { const p = lm[k]; o[i * 3] = p.x * W; o[i * 3 + 1] = -p.y * H; o[i * 3 + 2] = -p.z * W; });
  return o;
}
function modelPoints(m, wts, which) {   // which: "L" landmarks or "S" scalp
  const mean = m[which + "mean"], modes = m[which + "modes"], n = mean.length, out = Float64Array.from(mean);
  for (let k = 0; k < m.K; k++) { const w = wts[k]; if (!w) continue; const off = k * n; for (let i = 0; i < n; i++) out[i] += w * modes[off + i]; }
  return out;
}

/* for calibration (tools/): a face's landmarks with pose and size removed, in model coordinates */
export function alignedObs(m, lm, W, H) { return alignTo(obsPoints(m, lm, W, H), m.Lmean); }

export class HeadFitter {
  constructor(model) { this.m = model; this.reset(); }
  reset() { this.sum = new Float64Array(this.m.nL * 3); this.n = 0; this.result = null; }
  get progress() { return this.result ? 1 : this.n / FRAMES; }
  /* one tracked frame; pose from the app (yaw/pitch in radians). Returns the result once it is ready. */
  add(lm, W, H, pose, frames = FRAMES) {
    if (this.result) return this.result;
    if (Math.abs(pose.yaw) > 0.3 || Math.abs(pose.pitch) > 0.3) return null;               // near-frontal only
    const m = this.m, mouth = Math.hypot(lm[13].x * W - lm[14].x * W, lm[13].y * H - lm[14].y * H) / pose.w;
    if (mouth > 0.3) return null;   // only a wide-open mouth (yawning, laughing) is skipped: the jaw and lips are not used
    const o = alignTo(obsPoints(m, lm, W, H), m.Lmean);                                     // pose and size removed
    for (let i = 0; i < o.length; i++) this.sum[i] += o[i] - m.bias[i];
    if (++this.n < frames) return null;
    const avg = this.sum.map(v => v / this.n);
    let wts = new Float64Array(m.K);
    for (let it = 0; it < 4; it++) {           // fit the identity, re-align the face to it, fit again
      const L = modelPoints(m, wts, "L"), o2 = alignTo(avg, L), rhs = new Float64Array(m.K), n3 = m.nL * 3;
      for (let k = 0; k < m.K; k++) { let s = 0; for (let i = 0; i < n3; i++) { const w = i % 3 === 2 ? ZW * ZW : 1; s += m.Lmodes[k * n3 + i] * (o2[i] - m.Lmean[i]) * w; } rhs[k] = s; }
      wts = cholSolve(m.chol, rhs, m.K);
    }
    const r = measure(m, wts), M = m.mean;
    const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
    this.result = {
      weights: wts, raw: r,
      kx: clamp(r.rx / M.rx, 0.9, 1.1), ky: clamp(r.ry / M.ry, 0.9, 1.1), kz: clamp(r.rz / M.rz, 0.88, 1.12),
      kd: clamp(r.d / M.d, 0.88, 1.12),            // forehead to the middle of the skull, front to back
      dTop: clamp(r.top - M.top, -0.06, 0.06),     // crown above the forehead point, in head widths
    };
    return this.result;
  }
}

/* the fitted head, measured like the app measures a pose: head widths about the middle of the ear line */
export function measure(m, wts) {
  const L = modelPoints(m, wts, "L"), S = modelPoints(m, wts, "S");
  const P = k => [L[m.at[k] * 3], L[m.at[k] * 3 + 1], L[m.at[k] * 3 + 2]];
  const a = P(234), b = P(454), f = P(10), C = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  const w = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  // axis-aligned ellipsoid, centred left-right: A x^2 + B y^2 + C z^2 + D y + E z = 1, least squares
  const N = new Float64Array(25), r = new Float64Array(5); let ymax = -1e9;
  for (let i = 0; i < m.nS; i++) {
    const x = (S[i * 3] - C[0]) / w, y = (S[i * 3 + 1] - C[1]) / w, z = (S[i * 3 + 2] - C[2]) / w; ymax = Math.max(ymax, y);
    const row = [x * x, y * y, z * z, y, z];
    for (let p = 0; p < 5; p++) { r[p] += row[p]; for (let q = 0; q < 5; q++) N[p * 5 + q] += row[p] * row[q]; }
  }
  const [A, B, Cq, D, E] = cholSolve(cholesky(N, 5), r, 5);
  const cy = -D / (2 * B), cz = -E / (2 * Cq), g = 1 + B * cy * cy + Cq * cz * cz;
  const fy = (f[1] - C[1]) / w, fz = (f[2] - C[2]) / w;
  return { rx: Math.sqrt(g / A), ry: Math.sqrt(g / B), rz: Math.sqrt(g / Cq), cy, cz, top: ymax - fy, d: fz - cz };
}

/* similarity transform taking src onto dst (Horn's quaternion method); returns src moved */
function alignTo(src, dst) {
  const n = src.length / 3, ms = [0, 0, 0], md = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let j = 0; j < 3; j++) { ms[j] += src[i * 3 + j] / n; md[j] += dst[i * 3 + j] / n; }
  const Sx = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; let ss = 0;
  for (let i = 0; i < n; i++) {
    const a = [0, 1, 2].map(j => src[i * 3 + j] - ms[j]), b = [0, 1, 2].map(j => dst[i * 3 + j] - md[j]);
    for (let p = 0; p < 3; p++) { ss += a[p] * a[p]; for (let q = 0; q < 3; q++) Sx[p][q] += a[p] * b[q]; }
  }
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = Sx;
  const Nm = [[xx + yy + zz, yz - zy, zx - xz, xy - yx], [yz - zy, xx - yy - zz, xy + yx, zx + xz],
              [zx - xz, xy + yx, -xx + yy - zz, yz + zy], [xy - yx, zx + xz, yz + zy, -xx - yy + zz]];
  const q = topEigen4(Nm), [w, x, y, z] = q;
  const R = [[w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
             [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
             [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]];
  let num = 0;
  for (let i = 0; i < n; i++) {
    const a = [0, 1, 2].map(j => src[i * 3 + j] - ms[j]), b = [0, 1, 2].map(j => dst[i * 3 + j] - md[j]);
    for (let p = 0; p < 3; p++) num += b[p] * (R[p][0] * a[0] + R[p][1] * a[1] + R[p][2] * a[2]);
  }
  const s = num / ss, out = new Float64Array(src.length);
  for (let i = 0; i < n; i++) {
    const a = [0, 1, 2].map(j => src[i * 3 + j] - ms[j]);
    for (let p = 0; p < 3; p++) out[i * 3 + p] = s * (R[p][0] * a[0] + R[p][1] * a[1] + R[p][2] * a[2]) + md[p];
  }
  return out;
}
function topEigen4(A) {   // Jacobi rotations on a symmetric 4x4; eigenvector of the largest eigenvalue
  const a = A.map(r => r.slice()), v = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0; for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) off += a[p][q] * a[p][q];
    if (off < 1e-18) break;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) {
      if (Math.abs(a[p][q]) < 1e-30) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 4; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < 4; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < 4; k++) { const vkp = v[k][p], vkq = v[k][q]; v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq; }
    }
  }
  let best = 0; for (let i = 1; i < 4; i++) if (a[i][i] > a[best][best]) best = i;
  return [v[0][best], v[1][best], v[2][best], v[3][best]];
}
function cholesky(A, n) {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i * n + j]; for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
    L[i * n + j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j * n + j];
  }
  return L;
}
function cholSolve(L, b, n) {
  const y = new Float64Array(n), x = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}
