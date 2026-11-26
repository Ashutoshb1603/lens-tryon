"""Calibrate the head-shape model to real MediaPipe output: the average aligned face of real people (not
MediaPipe's canonical mesh) is what should read as the average head. Writes 'bias' into headshape.json."""
import json, sys, numpy as np
P = sys.argv[1]; obs = json.load(open(sys.argv[2]))
M = json.load(open(P + '.json')); buf = open(P + '.bin', 'rb').read()
B = {b['name']: np.frombuffer(buf, '<i2', count=int(np.prod(b['shape'])), offset=b['offset']).astype(float).reshape(b['shape']) * b['scale'] for b in M['blocks']}
K, lm = M['K'], M['lm']; at = {k: i for i, k in enumerate(lm)}
use = [o for o in obs if abs(o['yaw']) < 0.3 and abs(o['pitch']) < 0.3 and o['w'] > 60 and o['mouth'] < 0.2]
print('faces used', len(use), 'of', len(obs))
O = np.array([o['obs'] for o in use]).reshape(len(use), -1, 3)
Lm = B['Lmean']; bias = O.mean(0) - Lm
def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum() * len(src); return s, R, md - s * R @ ms
def ell(Sp, C, w):
    Q = (Sp - C) / w; x, y, z = Q.T; A = np.c_[x * x, y * y, z * z, y, z]; c = np.linalg.lstsq(A, np.ones(len(Q)), rcond=None)[0]
    a, b_, c_, d_, e_ = c; cy, cz = -d_ / (2 * b_), -e_ / (2 * c_); g = 1 + b_ * cy * cy + c_ * cz * cz
    return np.sqrt(g / a), np.sqrt(g / b_), np.sqrt(g / c_), cy, cz
def meas(wv):
    L = Lm + np.tensordot(wv, B['Lmodes'], 1); S = B['Smean'] + np.tensordot(wv, B['Smodes'], 1)
    a, b, f = L[at[234]], L[at[454]], L[at[10]]; C = (a + b) / 2; w = np.linalg.norm(b - a)
    rx, ry, rz, cy, cz = ell(S, C, w); return np.array([rx, ry, rz, (f[2] - C[2]) / w - cz, (S[:, 1].max() - f[1]) / w])
wts = np.array([1, 1, 0.5]); A = (B['Lmodes'] * wts).reshape(K, -1).T; G = A.T @ A + 1.0 * np.eye(K)
def fit(o, b):
    wv = np.zeros(K); o = o - b
    for _ in range(4):
        L = Lm + np.tensordot(wv, B['Lmodes'], 1); s, R, t = umeyama(o, L); o2 = s * o @ R.T + t
        wv = np.linalg.solve(G, A.T @ ((o2 - Lm) * wts).ravel())
    return wv
m0 = meas(np.zeros(K)); names = ['width', 'height', 'depth', 'skull offset', 'crown']
for label, b in (('before', np.zeros_like(bias)), ('after', bias)):
    R = np.array([meas(fit(o, b)) / np.r_[m0[:4], 1] - np.r_[1, 1, 1, 1, 0] - np.r_[0, 0, 0, 0, m0[4]] for o in O])
    print(label, ' '.join(f'{n} {R[:, i].mean():+.3f}±{R[:, i].std():.3f}' for i, n in enumerate(names)))
M['bias'] = np.round(bias, 4).ravel().tolist(); M['calibration'] = f'mean of {len(use)} real faces (MediaPipe landmarks)'
json.dump(M, open(P + '.json', 'w'))
