/* Head-turn scan: measures how deep the head (hair included) is front to back, which a frontal camera can't see.

   While the head turns, the outline of hair + face at hat-band height is read from the segmentation mask. For a
   head whose band cross-section is an ellipse with half-width a and half-depth c, centred cz in front of the ear
   line, at a turn (yaw) θ the outline spans
       half-extent² = a² cos²θ + c² sin²θ        centre = cz · sin θ
   (in head widths, along the image horizontal of the tilted head). Frontal frames pin a, turned frames give c and
   cz: one small least-squares solve. The result is compared with what the average head (ICT Face Model, band
   cross-section in headshape.json) would show with this person's own hair thickness, and returned as ratios. */

const BANDS = [-0.12, -0.04, 0.04];     // band heights relative to the forehead top, as hairFit uses
const NEED = { front: 4, side: 5 };     // frames: frontal, and each side

export function bandOutline(seg, pose, W, H) {   // -> { h, m } in head widths, or null when unmeasurable
  const sr = Math.sin(pose.roll), cr = Math.cos(pose.roll), w = pose.w;
  let out = false;
  const on = (a, b) => {
    const x = Math.round((pose.cx + w * (a * cr + b * sr)) / W * seg.w), y = Math.round((pose.cy + w * (a * sr - b * cr)) / H * seg.h);
    if (x < 0 || y < 0 || x >= seg.w || y >= seg.h) { out = true; return false; }
    const i = y * seg.w + x; return seg.hair[i] + seg.face[i] > 0.5;
  };
  let hs = 0, ms = 0, n = 0;
  for (const db of BANDS) {
    const b = pose.h + db;
    if (on(-1.5, b) || on(1.5, b)) return null;          // outline runs past the search: something else is in the way
    let l = -1.5, r = 1.5;
    while (l < 0 && !on(l, b)) l += 0.01;
    while (r > 0 && !on(r, b)) r -= 0.01;
    if (out) return null;                                  // the head runs off the picture
    if (r - l < 0.6) continue;
    hs += (r - l) / 2; ms += (r + l) / 2; n++;
  }
  return n >= 2 ? { h: hs / n, m: ms / n } : null;
}

export class HeadScan {
  constructor(ref) { this.ref = ref; this.reset(); }
  reset() { this.rows = []; this.result = null; this.n = { front: 0, left: 0, right: 0 }; }
  get progress() {
    const p = Math.min(1, this.n.front / NEED.front) * 0.2 + Math.min(1, this.n.left / NEED.side) * 0.4 + Math.min(1, this.n.right / NEED.side) * 0.4;
    return this.result ? 1 : p;
  }
  /* which way the person still needs to turn: "left" / "right" (their own left/right), or null */
  get next() {
    if (this.result) return null;
    if (this.n.left < NEED.side) return "left";
    if (this.n.right < NEED.side) return "right";
    return null;
  }
  add(seg, pose, W, H) {
    if (this.result || !seg || !pose || Math.abs(pose.pitch) > 0.3) return null;
    const ay = Math.abs(pose.yaw), side = ay < 0.2 ? "front" : ay > 0.4 && ay < 1.15 ? (pose.yaw > 0 ? "left" : "right") : null;
    if (!side || this.n[side] >= 14) return null;          // in between, or plenty of this side already
    const o = bandOutline(seg, pose, W, H); if (!o) return null;
    this.rows.push({ s: Math.sin(pose.yaw), c: Math.cos(pose.yaw), ...o }); this.n[side]++;
    if (this.n.front < NEED.front || this.n.left < NEED.side || this.n.right < NEED.side) return null;
    return (this.result = this.solve());
  }
  solve() {
    // h² = A cos² + C sin²   (least squares in A = a², C = c²);   m = cz sin
    let s11 = 0, s12 = 0, s22 = 0, b1 = 0, b2 = 0, ms = 0, ss = 0;
    for (const r of this.rows) {
      const x1 = r.c * r.c, x2 = r.s * r.s, y = r.h * r.h;
      s11 += x1 * x1; s12 += x1 * x2; s22 += x2 * x2; b1 += x1 * y; b2 += x2 * y; ms += r.m * r.s; ss += r.s * r.s;
    }
    const det = s11 * s22 - s12 * s12; if (Math.abs(det) < 1e-9) return { failed: true };
    const A = (b1 * s22 - b2 * s12) / det, C = (b2 * s11 - b1 * s12) / det, cz = ms / ss;
    if (!(A > 0.12 && C > 0.16)) return { failed: true };
    const a = Math.sqrt(A), c = Math.sqrt(C);
    if (a > 0.95 || c < 0.4 || c > 1.1) return { failed: true };       // not a head: a bad mask, or a hand in the way
    const { a0, c0, z0 } = this.ref, t = Math.max(0, a - a0);          // hair thickness at the sides, taken to be the same at the back
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
    return {
      a, c, cz, frames: this.rows.length,
      kz: clamp((c / a) / ((c0 + t / 2) / (a0 + t)), 0.85, 1.18),     // depth for its width, against the average head with this hair
      dz: clamp(cz - (z0 - t / 2), -0.08, 0.08),                     // where the middle of the head is, front to back
    };
  }
}
