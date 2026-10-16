"""Causal state-history dynamics transformer for the gravity-vector study.

Input per frame t:  [sin q1, sin q2, cos q1, cos q2, qd1/6, qd2/14, tau1/11, tau2/4.5]
                    (+ [cos θg, sin θg] for the 'given' control)
Target per frame t: [Δq1, Δq2, Δqd1, Δqd2] from frame t to t+1, z-scored.

Gravity direction is never an input of the main model: it has to be inferred in context.

Variants
  incontext : causal attention over the window (up to CTX=64 frames)
  given     : same, plus the true gravity unit vector as an input on every frame
  h1        : attention restricted to the current frame (history length 1), so gravity
              can only be guessed from the instantaneous (q, qd, tau)
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "models", "grav_ckpt")
CTX = 64
QD_SCALE = np.array([6.0, 14.0], np.float32)
TAU_SCALE = np.array([11.0, 4.5], np.float32)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ------------------------------------------------------------------ features
def features(q, qd, tau, gravity_angle=None):
    """q, qd, tau: [..., T, 2] -> [..., T, 8 or 10] float32."""
    f = [np.sin(q), np.cos(q), qd / QD_SCALE, tau / TAU_SCALE]
    if gravity_angle is not None:
        ga = np.broadcast_to(np.asarray(gravity_angle, np.float32)[..., None], q.shape[:-1])
        f.append(np.stack([np.cos(ga), np.sin(ga)], -1))
    return np.concatenate(f, -1).astype(np.float32)


def raw_targets(q, qd):
    """[..., T, 2] -> [..., T-1, 4]: wrapped Δq and Δqd."""
    dq = np.angle(np.exp(1j * (q[..., 1:, :] - q[..., :-1, :]))).astype(np.float32)
    dqd = qd[..., 1:, :] - qd[..., :-1, :]
    return np.concatenate([dq, dqd], -1).astype(np.float32)


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, d, n_heads, d_mlp):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, d_mlp), nn.Linear(d_mlp, d)
        self.n_heads = n_heads

    def forward(self, x, mask, cache=None, li=0):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, -1)
        sh = lambda t: t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)  # noqa: E731
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        h = F.gelu(self.fc1(self.ln2(x)))
        if cache is not None:
            cache[f"mlp{li}"] = h
        return x + self.fc2(h)


class GravTransformer(nn.Module):
    def __init__(self, d_in=8, d=128, n_layers=4, n_heads=4, d_mlp=512, ctx=CTX, attn="causal"):
        super().__init__()
        self.cfg = dict(d_in=d_in, d=d, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp, ctx=ctx, attn=attn)
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.randn(ctx, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, n_heads, d_mlp) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 4)
        self.attn = attn
        self.register_buffer("y_mean", torch.zeros(4))
        self.register_buffer("y_std", torch.ones(4))

    def mask(self, T, dev):
        if self.attn == "causal":
            return torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        return torch.eye(T, dtype=torch.bool, device=dev)

    def forward(self, x, cache=None, hooks=None):
        """x [B,T,d_in] -> normalised predictions [B,T,4].
        cache: dict filled with resid{L} (L=0..n_layers, L=0 is the embedding) and mlp{l}.
        hooks: {L: fn(resid)->resid} applied to resid{L} before it is passed on."""
        T = x.shape[1]
        h = self.inp(x) + self.pos[:T]
        m = self.mask(T, x.device)
        for li in range(len(self.blocks) + 1):
            if hooks and li in hooks:
                h = hooks[li](h)
            if cache is not None:
                cache[f"resid{li}"] = h
            if li < len(self.blocks):
                h = self.blocks[li](h, m, cache, li)
        return self.head(self.ln_f(h))


def load(variant: str, dev=None) -> GravTransformer:
    dev = dev or device()
    sd = torch.load(os.path.join(CKPT, f"grav_{variant}.pt"), map_location="cpu")
    m = GravTransformer(**sd["cfg"])
    m.load_state_dict(sd["state"])
    return m.to(dev).eval()


@torch.no_grad()
def run(model, X: np.ndarray, keys=(), hooks=None, bs=512):
    """X [N,T,d_in] numpy -> (pred [N,T,4] in raw units, {key: [N,T,U] numpy})."""
    dev = next(model.parameters()).device
    preds, acts = [], {k: [] for k in keys}
    for i in range(0, len(X), bs):
        cache = {} if keys else None
        p = model(torch.from_numpy(X[i:i + bs]).to(dev), cache, hooks)
        preds.append((p * model.y_std + model.y_mean).cpu().numpy())
        for k in keys:
            acts[k].append(cache[k].cpu().numpy())
    return np.concatenate(preds), {k: np.concatenate(v) for k, v in acts.items()}


@torch.no_grad()
def rollout(model, X_ctx: np.ndarray, q0, qd0, tau_seq, n_steps: int, hooks=None, ga=None):
    """Open-loop autoregressive rollout.

    X_ctx [B,Tc,d_in] are observed context frames *before* the start state; the start state
    (q0, qd0 [B,2]) is appended and the model predicts forward n_steps with torques
    tau_seq [B,n_steps,2] (tau applied at each predicted-from frame). Returns q, qd [B,n_steps+1,2].
    Keeps at most CTX frames (sliding window)."""
    dev = next(model.parameters()).device
    q, qd = [np.asarray(q0, np.float32)], [np.asarray(qd0, np.float32)]
    hist = X_ctx
    for s in range(n_steps):
        f = features(q[-1][:, None], qd[-1][:, None], tau_seq[:, s:s + 1],
                     None if ga is None else ga[:, None])
        hist = np.concatenate([hist, f], 1)[:, -model.cfg["ctx"]:]
        p = model(torch.from_numpy(hist).to(dev), None, hooks)[:, -1]
        p = (p * model.y_std + model.y_mean).cpu().numpy()
        q.append(q[-1] + p[:, :2])
        qd.append(qd[-1] + p[:, 2:])
    return np.stack(q, 1), np.stack(qd, 1)
