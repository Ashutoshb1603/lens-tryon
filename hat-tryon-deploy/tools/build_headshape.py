"""Build the head-shape data for the hat try-on from the ICT Face Model (MIT licence).

Output (hat-tryon/models/headshape.bin + headshape.json):
  for a set of MediaPipe landmark indices and a set of scalp sample points, the mean position and the
  first K identity modes, all in MediaPipe's canonical face frame (cm, y up, z out of the face).
The browser fits the identity weights to a person's tracked landmarks once and reads off their skull.
"""
import json, sys, numpy as np
from scipy.spatial import cKDTree

ICT = sys.argv[1]            # ICT-FaceKit/FaceXModel
MP = sys.argv[2]             # canonical_face_model.obj
OUT = sys.argv[3]            # output prefix
K = 40                       # identity modes kept

def obj_v(path):
    return np.array([[float(t) for t in l.split()[1:4]] for l in open(path) if l.startswith("v ")])

V0 = obj_v(f"{ICT}/generic_neutral_mesh.obj")
modes = np.stack([obj_v(f"{ICT}/identity{i:03d}.obj") - V0 for i in range(100)])   # 100 x N x 3, unit-variance weights
mp = obj_v(MP)
print("ICT", V0.shape, "MP", mp.shape, "mode rms", np.sqrt((modes ** 2).mean()))

# Multi-PIE 68 points in ICT, and the usual MediaPipe equivalents. Inner points only (brows, nose, eyes, mouth).
ict68 = [1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443, 268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528, 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841, 5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966]
mp68 = [162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71, 63, 105, 66, 107, 336, 296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87]
inner = list(range(17, 68))

def umeyama(src, dst, scale=True):
    ms, md = src.mean(0), dst.mean(0); a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (a ** 2).sum() * len(src) if scale else 1.0
    return s, R, md - s * R @ ms

# 1. ICT -> MediaPipe canonical frame
s, R, t = umeyama(V0[[ict68[i] for i in inner]], mp[[mp68[i] for i in inner]])
print("similarity scale", s)
X = lambda P: s * P @ R.T + t
V0c = X(V0); modesc = s * modes @ R.T

# 2. dense correspondence: fit ICT identity to the MediaPipe mean face (ICP + ridge), then closest face vertex
face = np.arange(0, 9409)
w = np.zeros(K)
for it in range(8):
    Vf = V0c + np.tensordot(w, modesc[:K], 1)
    tree = cKDTree(Vf[face]); d, j = tree.query(mp); idx = face[j]
    if it < 7:
        # refine rigid alignment of the MediaPipe face to the fitted head, then the identity weights
        s2, R2, t2 = umeyama(mp, Vf[idx]); mpa = s2 * mp @ R2.T + t2
        A = modesc[:K, idx].reshape(K, -1).T; b = (mpa - V0c[idx]).ravel()
        w = np.linalg.solve(A.T @ A + 0.5 * np.eye(K), A.T @ b)
print("correspondence rms (cm)", np.sqrt((d ** 2).mean()), "max", d.max())
# Put the MediaPipe frame exactly back: express everything in the frame of the MediaPipe canonical model
s3, R3, t3 = umeyama(Vf[idx], mp)
Y = lambda P: s3 * P @ R3.T + t3
V0m = Y(V0c); modesm = s3 * modesc[:K] @ R3.T
# centre the model on the ICT head that matches MediaPipe's own average face (weights w), so that an average
# MediaPipe face maps to weight 0: otherwise every tracked face reads as slightly different from the average head
V0m = V0m + np.tensordot(w, modesm, 1)
print("centred on the MediaPipe average face; |w| =", np.linalg.norm(w))

# landmarks used for fitting: not lips or eyelids (expression), not the eye/brow interior lines that move
mouth = set([0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 146, 178, 179, 180, 181, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 291, 292, 302, 303, 304, 306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 324, 325, 375, 402, 403, 404, 405, 407, 408, 409, 415])
eyes = set([7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477])
lmk = [i for i in range(468) if i not in mouth and i not in eyes][::2]
for must in (10, 234, 454, 105, 334, 152):
    if must not in lmk: lmk.append(must)
L_idx = idx[lmk]

# 3. scalp samples: head & neck part plus the upper forehead, above the ear line
C = (mp[234] + mp[454]) / 2; wdt = np.linalg.norm(mp[454] - mp[234])
cand = np.r_[np.arange(9409, 11248), face]
yy = (V0m[cand, 1] - C[1]) / wdt
scalp = cand[(yy > 0.05) & ((cand >= 9409) | ((V0m[cand, 1] - mp[10][1]) / wdt > -0.08))]
rng = np.random.default_rng(0); S_idx = np.sort(rng.choice(scalp, min(240, len(scalp)), replace=False))
print("landmarks", len(lmk), "scalp samples", len(S_idx), "head width (cm)", wdt)

# 4. how well does the face predict the skull? synthetic people, noisy landmarks, ridge fit vs. the mean head
def ell(P, Cc, ww):   # axis-aligned ellipsoid (centred in x) through scalp points, in head widths about the ear line
    Q = (P - Cc) / ww; x, y, z = Q.T
    A = np.c_[x * x, y * y, z * z, y, z]; c = np.linalg.lstsq(A, np.ones(len(Q)), rcond=None)[0]
    a, b_, c_, d_, e_ = c; cy, cz = -d_ / (2 * b_), -e_ / (2 * c_); g = 1 + b_ * cy * cy + c_ * cz * cz
    return np.array([np.sqrt(g / a), np.sqrt(g / b_), np.sqrt(g / c_), cy, cz])
def measures(V, ids_lm):
    Lp = V[L_idx]; Cc = (Lp[lmk.index(234)] + Lp[lmk.index(454)]) / 2; ww = np.linalg.norm(Lp[lmk.index(454)] - Lp[lmk.index(234)])
    e = ell(V[S_idx], Cc, ww); top = (V[S_idx][:, 1].max() - Lp[lmk.index(10)][1]) / ww
    return np.r_[e, top, (Lp[lmk.index(10)][2] - Cc[2]) / ww]
mean_m = measures(V0m, None)
print("mean head: rx ry rz cy cz | top-above-forehead | forehead z", np.round(mean_m, 3))
BL = modesm[:, L_idx]
def fit(obs, lam, zw=0.5):
    wts = np.array([1, 1, zw]); wv = np.zeros(K)
    for _ in range(4):
        model = V0m[L_idx] + np.tensordot(wv, BL, 1)
        sA, RA, tA = umeyama(obs, model); o = sA * obs @ RA.T + tA
        A = (BL * wts).reshape(K, -1).T; b = ((o - V0m[L_idx]) * wts).ravel()
        wv = np.linalg.solve(A.T @ A + lam * np.eye(K), A.T @ b)
    return wv
res = {}
for lam in (0.5, 2, 6, 15):
    err_fit, err_mean = [], []
    for k in range(200):
        wt = rng.normal(size=K); Vt = V0m + np.tensordot(wt, modesm, 1); truth = measures(Vt, None)
        obs = Vt[L_idx] + rng.normal(scale=[0.12, 0.12, 0.35], size=(len(L_idx), 3))      # tracker noise (cm), worse in depth
        ang = rng.normal(scale=0.05, size=3); Rr = np.linalg.qr(np.eye(3) + np.array([[0, -ang[2], ang[1]], [ang[2], 0, -ang[0]], [-ang[1], ang[0], 0]]))[0]
        obs = 37 * obs @ Rr.T + 5          # arbitrary pose and scale: the fit must not depend on them
        wf = fit(obs, lam); Vf2 = V0m + np.tensordot(wf, modesm, 1)
        err_fit.append(measures(Vf2, None) - truth); err_mean.append(mean_m - truth)
    ef, em = np.abs(err_fit).mean(0), np.abs(err_mean).mean(0)
    res[lam] = ef.tolist()
    print(f"lam {lam:>4}: mean abs error fit / mean-head  rx {ef[0]:.3f}/{em[0]:.3f}  ry {ef[1]:.3f}/{em[1]:.3f}  rz {ef[2]:.3f}/{em[2]:.3f}  cz {ef[4]:.3f}/{em[4]:.3f}  top {ef[5]:.3f}/{em[5]:.3f}")

# 5. export: int16, one scale per block
def q(a):
    sc = np.abs(a).max() / 32000; return (np.round(a / sc).astype("<i2"), float(sc))
blocks = {"Lmean": V0m[L_idx], "Lmodes": modesm[:, L_idx], "Smean": V0m[S_idx], "Smodes": modesm[:, S_idx]}
meta = {"K": K, "lm": lmk, "nL": len(lmk), "nS": len(S_idx), "blocks": [], "mean": mean_m.tolist(),
        "source": "ICT-FaceKit (USC ICT, MIT licence) identity model, aligned to the MediaPipe canonical face"}
buf = b""
for name, arr in blocks.items():
    qa, sc = q(arr.astype(np.float64)); meta["blocks"].append({"name": name, "shape": list(arr.shape), "scale": sc, "offset": len(buf)}); buf += qa.tobytes()
open(OUT + ".bin", "wb").write(buf); json.dump(meta, open(OUT + ".json", "w"))
print("wrote", len(buf), "bytes")
