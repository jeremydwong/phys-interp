"""Shared helpers for the video-model analyses: checkpoints, windows, activation extraction.

Activation cache (models/video_cache/):
  acts_<tag>_<split>.npz
    max   [N, U] float16  per-unit max over the 16x16 positions (frame-t tokens)
    arg   [N, U] uint8    argmax position (row*16+col)
    mean  [N, U] float16  per-unit spatial mean
    res   [L, N, C*16]    residual stream after each block, avg-pooled to 4x4 (only if res=True)
    eps, ts               window index (frame t of the input window)
  Units are ordered layer-major: unit u = layer * H + channel.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.common import load_frames, load_states, targets  # noqa: E402
from analysis.device import get_device  # noqa: E402
from models.video_model import load as load_model_file  # noqa: E402

CKPT = os.environ.get("VIDEO_CKPT", os.path.join(ROOT, "models", "video_ckpt"))
CACHE = os.path.join(ROOT, "models", "video_cache")
RESULTS = os.path.join(ROOT, "reports", "results")
DEV = get_device()
LAYERS = ["enc0", "enc1", "enc2", "enc3", "tmp0", "tmp1", "tmp2", "tmp3"]
TEST = np.arange(2000, 2400)
PROBE_TRAIN = np.arange(0, 400)
T0, T1 = 3, 147
if os.environ.get("VIDEO_SMOKE"):          # quick end-to-end test on few episodes
    CACHE = os.environ["VIDEO_SMOKE"]
    RESULTS = os.path.join(CACHE, "results")
    TEST, PROBE_TRAIN = np.arange(2000, 2040), np.arange(0, 40)
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def ckpt_paths():
    return sorted(glob.glob(os.path.join(CKPT, "video_step*.pt")))


def ckpt_step(p):
    return int(os.path.basename(p)[len("video_step"):-3])


def load(which="final", device=DEV):
    ps = ckpt_paths()
    p = ps[-1] if which == "final" else ps[0] if which == "init" else which
    m, ck = load_model_file(p, device)
    return m, ck


_FR = None


def frames():
    global _FR
    if _FR is None:
        _FR = load_frames("main")
    return _FR


def windows(eps, ts):
    """-> float tensor [N, 4, 3, 64, 64] for input frames t-3..t (on DEV)."""
    fr = frames()
    w = np.stack([fr[e, t - 3:t + 1] for e, t in zip(eps, ts)])
    return torch.from_numpy(w).to(DEV).permute(0, 1, 4, 2, 3).float() / 255.0


def future(eps, ts, k=1):
    fr = frames()
    w = np.stack([fr[e, t + k] for e, t in zip(eps, ts)])
    return torch.from_numpy(w).to(DEV).permute(0, 3, 1, 2).float() / 255.0


def grid(eps_set, stride=3, t0=T0, t1=T1):
    ts = np.arange(t0, t1 + 1, stride)
    E = np.repeat(eps_set, len(ts))
    T = np.tile(ts, len(eps_set))
    return E, T


@torch.no_grad()
def extract(model, eps, ts, res=False, bs=128, layers=LAYERS):
    model.record = True
    mx, ag, mn, rs = [], [], [], []
    for i in range(0, len(eps), bs):
        x = windows(eps[i:i + bs], ts[i:i + bs])
        model(x)
        A = torch.stack([model.acts[l] for l in layers], 1)      # [B, L, H, 16, 16]
        B, L, H = A.shape[:3]
        A = A.reshape(B, L * H, 256)
        v, a = A.max(-1)
        mx.append(v.half().cpu().numpy()); ag.append(a.to(torch.uint8).cpu().numpy())
        mn.append(A.mean(-1).half().cpu().numpy())
        if res:
            R = torch.stack([F.adaptive_avg_pool2d(model.residual[l], 4).flatten(1) for l in layers], 0)
            rs.append(R.half().cpu().numpy())
    model.record = False
    out = dict(max=np.concatenate(mx), arg=np.concatenate(ag), mean=np.concatenate(mn), eps=eps, ts=ts)
    if res:
        out["res"] = np.concatenate(rs, 1)
    return out


def cached_acts(tag, split, model_fn, res=False, stride=3, force=False):
    """tag: 'final' | 'init' | 'step000123'; split: 'test' | 'test_dense' | 'probe'."""
    path = os.path.join(CACHE, f"acts_{tag}_{split}.npz")
    if os.path.exists(path) and not force:
        z = np.load(path)
        return {k: z[k] for k in z.files}
    eps_set = PROBE_TRAIN if split == "probe" else TEST
    if split == "probe":
        stride = max(stride, 6)          # lighter: ~9.8k probe-training windows
    E, T = grid(eps_set, 1 if split == "test_dense" else stride)
    m = model_fn()
    out = extract(m, E, T, res=res)
    np.savez(path, **out)
    return out


_TG = None


def target_matrix():
    """[E, T, 32] targets + Var list (cached)."""
    global _TG
    if _TG is None:
        st = load_states("main")
        Y, vs = targets(st)
        _TG = (Y, vs, st)
    return _TG


def binned_r2_matrix(Y, A, bins=30):
    """Vectorized analysis.common.binned_r2(x=Y[:,v], y=A[:,u]) for all (u, v) -> [U, V].
    Bins are quantile bins of the target variable; R2 = 1 - Var(resid)/Var(act)."""
    A = torch.as_tensor(np.asarray(A, np.float32), device=DEV)
    A = A - A.mean(0)
    var = (A ** 2).mean(0) + 1e-12
    N = A.shape[0]
    out = np.zeros((A.shape[1], Y.shape[1]), np.float32)
    for v in range(Y.shape[1]):
        x = Y[:, v]
        edges = np.quantile(x, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
        oh = torch.zeros(N, bins, device=DEV)
        oh[torch.arange(N), torch.as_tensor(idx, device=DEV)] = 1
        cnt = oh.sum(0).clamp(min=1)
        means = (oh.T @ A) / cnt[:, None]            # [bins, U]
        resid = A - oh @ means
        out[:, v] = (1 - (resid ** 2).mean(0) / var).cpu().numpy()
    return out


def bin_index(x, bins):
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)


def bin_indices(Y, bins=8):
    return np.stack([bin_index(Y[:, v], bins) for v in range(Y.shape[1])], 1)


def binned_r2_pairs(IDX, a, pairs, bins=8):
    """2-D binned R2 for activation a [N] against target pairs (v1, v2); IDX = bin_indices(Y, bins)."""
    out = []
    va = a.var() + 1e-12
    for v1, v2 in pairs:
        idx = IDX[:, v1] * bins + IDX[:, v2]
        means = np.bincount(idx, a, bins * bins) / np.maximum(np.bincount(idx, minlength=bins * bins), 1)
        out.append(1 - (a - means[idx]).var() / va)
    return np.array(out)


def unit_name(u, H):
    return f"{LAYERS[u // H]}.{u % H}"


def posture_velocity_basis(Ytv, vs):
    """Degree-2 polynomial basis of posture + velocity variables (state-only control)."""
    names = ["sin_q1", "cos_q1", "sin_q2", "cos_q2", "sin_q12", "cos_q12", "ee_x", "ee_z", "elbow_z",
             "qd1", "qd2", "ee_vx", "ee_vz"]
    idx = [i for i, v in enumerate(vs) if v.name in names]
    X = Ytv[:, idx]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    feats = [X] + [X[:, i:i + 1] * X[:, i:] for i in range(X.shape[1])]
    B = np.concatenate(feats, 1)
    return np.concatenate([B, np.ones((len(B), 1))], 1).astype(np.float32)


def residualize(B, M, lam=1e-3):
    """Residual of columns of M after ridge regression on basis B (fit on the same data)."""
    Bt = torch.as_tensor(B, device="cpu", dtype=torch.float64)
    Mt = torch.as_tensor(np.asarray(M, np.float64))
    G = Bt.T @ Bt + lam * len(B) * torch.eye(B.shape[1], dtype=torch.float64)
    W = torch.linalg.solve(G, Bt.T @ Mt)
    return (Mt - Bt @ W).numpy().astype(np.float32)
