"""Shared helpers for the state-MLP (candidate C) analyses: activations, unit labels, rendering."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.common import N_TRAIN_EPS, ROOT as _R, load_states, targets  # noqa: E402,F401
from models.state_mlp import (CKPT_DIR, N_BLOCKS, D_HIDDEN, build_dataset, list_checkpoints,  # noqa: E402,F401
                              load_model)
from sim.arm import DT, FLOOR_Z, L1, L2, RECORD_EVERY, WALL_X  # noqa: E402,F401

RESULTS_JSON = os.path.join(ROOT, "reports", "results", "state.json")
CACHE = os.path.join(CKPT_DIR, "cache")
FRAME_DT = DT * RECORD_EVERY  # 0.02 s
LIVE_FRAC = 0.01
CLEAN, PARTIAL, CONJ_GAIN = 0.5, 0.25, 0.15
R_UPPER, R_FORE = 0.045, 0.04


# ------------------------------------------------------------------ results
def update_results(section: str, payload: dict):
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    res = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            res = json.load(f)
    res[section] = payload
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def load_results():
    with open(RESULTS_JSON) as f:
        return json.load(f)


# ------------------------------------------------------------------ data
_DATA = {}


def data():
    """Test-set frames t in [0, 148] of episodes [2000, 2400): inputs, targets, states (flattened)."""
    if "d" in _DATA:
        return _DATA["d"]
    st = load_states()
    ds = build_dataset(st)
    Y, vs = targets(st)
    E, T = st["q"].shape[:2]
    sl = slice(N_TRAIN_EPS, E)
    out = dict(
        X=ds["X"][sl].reshape(-1, 8), D=ds["D"][sl].reshape(-1, 4), mu=ds["mu"], sd=ds["sd"],
        Y=Y[sl, :-1].reshape(-1, Y.shape[-1]), vars=vs, names=[v.name for v in vs],
        ep=np.repeat(np.arange(N_TRAIN_EPS, E), T - 1), t=np.tile(np.arange(T - 1), E - N_TRAIN_EPS),
        T=T - 1, st=st,
        Xtr=ds["X"][:N_TRAIN_EPS].reshape(-1, 8), Ytr=Y[:N_TRAIN_EPS, :-1].reshape(-1, Y.shape[-1]),
        Dtr=ds["D"][:N_TRAIN_EPS].reshape(-1, 4),
    )
    out["S"] = {k: v[sl, :-1].reshape((-1,) + v.shape[2:]) for k, v in st.items()
                if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[1] == T}
    _DATA["d"] = out
    return out


@torch.no_grad()
def run(model, X, edits=None, chunk=32768, device="cpu"):
    """Returns dict y [N,4] (standardized), h [L,N,1024] post-ReLU, pre [L,N,1024], resid [L+1,N,256]."""
    ys, hs, pres, rs = [], [], [], []
    for i in range(0, len(X), chunk):
        u = torch.as_tensor(X[i:i + chunk], device=device)
        x = model.embed(u)
        h_l, p_l, r_l = [], [], [x]
        for l, b in enumerate(model.blocks):
            p = b.fc_in(b.ln(x))
            h = torch.relu(p)
            if edits is not None and l in edits:
                h = edits[l](h)
            x = x + b.fc_out(h)
            h_l.append(h); p_l.append(p); r_l.append(x)
        ys.append(model.head(model.ln_f(x)).cpu().numpy())
        hs.append(torch.stack(h_l).cpu().numpy()); pres.append(torch.stack(p_l).cpu().numpy())
        rs.append(torch.stack(r_l).cpu().numpy())
    return dict(y=np.concatenate(ys), h=np.concatenate(hs, 1), pre=np.concatenate(pres, 1),
                resid=np.concatenate(rs, 1))


def predict_accel(model, X, mu, sd, edits=None):
    """Model-implied mean acceleration over the step, rad/s^2: Δqd / 0.02."""
    with torch.no_grad():
        y = model(torch.as_tensor(X), edits=edits).numpy()
    D = y * sd + mu
    return D[:, 2:] / FRAME_DT, D


# ------------------------------------------------------------------ tuning statistics
def bin_index(x, bins=30):
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)


def binned_r2_matrix(A, Y, bins=30):
    """Vectorized analysis.common.binned_r2 for every (unit, variable): A [N,U], Y [N,V] -> [U,V].
    Identical estimator (quantile bins, piecewise-constant fit, R² = SS_between / SS_total)."""
    A = A.astype(np.float64)
    mu, var = A.mean(0), A.var(0) + 1e-12
    out = np.zeros((A.shape[1], Y.shape[1]))
    for v in range(Y.shape[1]):
        idx = bin_index(Y[:, v], bins)
        out[:, v] = _ss_between(A, idx, bins, mu) / var
    return out


def _ss_between(A, idx, nb, mu):
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)
    onehot = np.zeros((len(idx), nb))
    onehot[np.arange(len(idx)), idx] = 1
    S = onehot.T @ A
    m = S / np.maximum(cnt, 1)[:, None]
    return (cnt[:, None] * (m - mu) ** 2).sum(0) / len(A)


def pair_r2(a, y1, y2, bins=10):
    """2-D binned R² of activation a on (y1, y2) with bins x bins quantile cells."""
    idx = bin_index(y1, bins) * bins + bin_index(y2, bins)
    a = a.astype(np.float64)[:, None]
    return float(_ss_between(a, idx, bins * bins, a.mean(0))[0] / (a.var() + 1e-12))


def target_aliases(Y, names, thr=0.99):
    """Pairs of target variables that are (near-)identical up to sign/scale (|r| > thr)."""
    C = np.corrcoef(Y.T)
    al = {}
    for i in range(len(names)):
        al[names[i]] = [names[j] for j in range(len(names)) if j != i and abs(C[i, j]) > thr]
    return al


# ------------------------------------------------------------------ geometry / rendering
def obstacle_gaps(q):
    """Clearance (m) of each link capsule to floor and wall. q [...,2] -> dict of [...] arrays."""
    q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
    ex, ez = L1 * np.cos(q1), L1 * np.sin(q1)
    hx, hz = ex + L2 * np.cos(q12), ez + L2 * np.sin(q12)
    return dict(
        floor_upper=np.minimum(0, ez) - R_UPPER - FLOOR_Z,
        floor_fore=np.minimum(ez, hz) - R_FORE - FLOOR_Z,
        wall_upper=WALL_X - np.maximum(0, ex) - R_UPPER,
        wall_fore=WALL_X - np.maximum(ex, hx) - R_FORE,
    )


def fk(q):
    q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
    el = np.stack([L1 * np.cos(q1), L1 * np.sin(q1)], -1)
    ee = el + np.stack([L2 * np.cos(q12), L2 * np.sin(q12)], -1)
    return el, ee


_RENDER = {}


def render(qs, size=112):
    from sim.arm import ArmRenderer
    if size not in _RENDER:
        _RENDER[size] = ArmRenderer(size)
    return _RENDER[size](np.atleast_2d(qs))


def to_px(xz, size):
    """World (x, z) in the arm plane -> pixel (u, v) for the 'side' camera (fovy 45, distance 2.6)."""
    f = (size / 2) / math.tan(math.radians(22.5))
    xz = np.asarray(xz)
    return np.stack([size / 2 + f * xz[..., 0] / 2.6, size / 2 - f * (xz[..., 1] + 0.05) / 2.6], -1)


WORKSPACE = ((-1.0, 0.86), (-0.92, 0.94))   # world x, z bounds that contain every reachable pose + obstacles


def draw_pose(ax, q_now, q_hist=None, size=144, vel=None, crop=WORKSPACE):
    """Render arm at q_now; overlay the hand (dark) / elbow (grey) trail over q_hist [k,2] (oldest first)
    and a red hand-velocity arrow (vel = ee velocity in m/s; arrow shows 0.1 s of motion).
    crop: world ((x0, x1), (z0, z1)) region to show."""
    img = render(q_now[None], size)[0]
    ax.imshow(img, interpolation="lanczos")
    if q_hist is not None and len(q_hist):
        qq = np.concatenate([q_hist, q_now[None]])
        el, ee = fk(qq)
        for pts, c in ((ee, "#0b0b0b"), (el, "#52514e")):
            p = to_px(pts, size)
            n = len(p)
            for i in range(n - 1):
                ax.plot(p[i:i + 2, 0], p[i:i + 2, 1], color=c, lw=1.0, alpha=0.15 + 0.75 * (i + 1) / n,
                        solid_capstyle="round")
    if vel is not None:
        _, ee = fk(q_now)
        p0 = to_px(ee, size)
        p1 = to_px(ee + 0.1 * np.asarray(vel), size)
        if np.linalg.norm(p1 - p0) > 1.5:
            ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.3",
                                                               color="#e34948", lw=1.1, shrinkA=0, shrinkB=0))
    if crop is not None:
        (x0, z0), (x1, z1) = to_px(np.array([crop[0][0], crop[1][0]]), size), to_px(np.array([crop[0][1], crop[1][1]]), size)
        ax.set_xlim(x0, x1); ax.set_ylim(z0, z1)
    else:
        ax.set_xlim(0, size); ax.set_ylim(size, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def label_units(R, names, fams, aliases):
    """R [U,V] binned R². Returns per-unit best var index, best R², category, family (alias-aware)."""
    best = R.argmax(1)
    bestr = R.max(1)
    cat = np.where(bestr >= CLEAN, "clean", np.where(bestr >= PARTIAL, "partial", "unexplained"))
    fam = np.array([fams[b] for b in best], dtype=object)
    return best, bestr, cat, fam


def load_census(tag="trained"):
    return dict(np.load(os.path.join(CACHE, f"census_{tag}.npz"), allow_pickle=True))


def pick_units(cz, names, var, k=1, best_in=None, exclude=(), min_r2=0.0):
    """Top-k (layer, unit, R²) by binned R² with `var`, optionally requiring the unit's best variable
    to be in `best_in` (list of names). Searches all layers."""
    j = names.index(var)
    cand = []
    L = cz["R"].shape[0]
    for l in range(L):
        r = np.nan_to_num(cz["R"][l][:, j], nan=-1)
        ok = cz["live"][l] & (r >= min_r2)
        if best_in is not None:
            ok &= np.isin(cz["best"][l], [names.index(b) for b in best_in])
        for u in np.where(ok)[0]:
            if (l, int(u)) not in exclude:
                cand.append((l, int(u), float(r[u])))
    cand.sort(key=lambda z: -z[2])
    return cand[:k]
