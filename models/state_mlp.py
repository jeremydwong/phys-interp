"""Candidate C: a readable forward-dynamics MLP for the two-link arm.

Input at frame t  : sin q, cos q, qd / QD_SCALE, tau_act / TAU_MAX            (8 dims)
Target            : standardized [dq1, dq2, dqd1, dqd2] = state(t+1) - state(t)  (20 ms)

No acceleration, contact, or gravity information is given, so the network must compute
them internally. Architecture (residual MLP, pre-LayerNorm):

    x0 = W_embed u                       (256)
    x_{l+1} = x_l + W_out,l ReLU(W_in,l LN(x_l))   l = 0..3, hidden width 1024
    y = W_head LN(x_4)

Usage:  uv run python models/state_mlp.py          # trains, saves checkpoints to models/state_ckpt/
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.common import N_TRAIN_EPS, load_states  # noqa: E402
from sim.arm import TAU_MAX  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "models", "state_ckpt")
QD_SCALE = 10.0
D_MODEL, D_HIDDEN, N_BLOCKS = 256, 1024, 4
OUT_NAMES = ["dq1", "dq2", "dqd1", "dqd2"]


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, d=D_MODEL, h=D_HIDDEN):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc_in = nn.Linear(d, h)
        self.fc_out = nn.Linear(h, d)

    def forward(self, x, edit=None):
        h = torch.relu(self.fc_in(self.ln(x)))
        if edit is not None:
            h = edit(h)
        return x + self.fc_out(h), h


class StateMLP(nn.Module):
    def __init__(self, d_in=8, d_out=4):
        super().__init__()
        self.embed = nn.Linear(d_in, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_BLOCKS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, d_out)

    def forward(self, u, edits: dict | None = None, return_acts: bool = False):
        """edits: {layer: fn(h)->h} applied to post-ReLU hidden units of that block.
        return_acts: also return (hidden list [B,1024]x4, residual list [B,256]x5)."""
        x = self.embed(u)
        hs, resid = [], [x]
        for l, b in enumerate(self.blocks):
            x, h = b(x, None if edits is None else edits.get(l))
            hs.append(h)
            resid.append(x)
        y = self.head(self.ln_f(x))
        return (y, hs, resid) if return_acts else y


# ------------------------------------------------------------------ data
def make_inputs(q, qd, tau):
    """q, qd, tau: [..., 2] arrays -> [..., 8] float32 model input."""
    return np.concatenate([np.sin(q), np.cos(q), qd / QD_SCALE, tau / TAU_MAX], -1).astype(np.float32)


def build_dataset(st=None):
    """Returns dict with inputs X [E,T-1,8], raw targets D [E,T-1,4] and normalization stats
    (computed on train episodes). Frame index t of X corresponds to frame t of states."""
    st = load_states() if st is None else st
    q, qd, tau = st["q"].astype(np.float64), st["qd"].astype(np.float64), st["tau_act"].astype(np.float64)
    X = make_inputs(q[:, :-1], qd[:, :-1], tau[:, :-1])
    D = np.concatenate([q[:, 1:] - q[:, :-1], qd[:, 1:] - qd[:, :-1]], -1).astype(np.float32)
    mu = D[:N_TRAIN_EPS].reshape(-1, 4).mean(0)
    sd = D[:N_TRAIN_EPS].reshape(-1, 4).std(0)
    return dict(X=X, D=D, mu=mu.astype(np.float32), sd=sd.astype(np.float32))


def load_model(tag: str = "final", device="cpu") -> tuple[StateMLP, dict]:
    ck = torch.load(os.path.join(CKPT_DIR, f"{tag}.pt"), map_location=device, weights_only=False)
    m = StateMLP().to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def list_checkpoints():
    with open(os.path.join(CKPT_DIR, "index.json")) as f:
        return json.load(f)


def r2(pred, true):
    return 1 - ((pred - true) ** 2).mean(0) / true.var(0)


# ------------------------------------------------------------------ training
def train(steps=60000, batch=1024, lr=3e-4, wd=0.05, seed=0, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = build_dataset()
    Xtr = torch.tensor(ds["X"][:N_TRAIN_EPS].reshape(-1, 8), device=device)
    Ytr = torch.tensor(((ds["D"][:N_TRAIN_EPS] - ds["mu"]) / ds["sd"]).reshape(-1, 4), device=device)
    Xte = torch.tensor(ds["X"][N_TRAIN_EPS:].reshape(-1, 8), device=device)
    Yte = torch.tensor(((ds["D"][N_TRAIN_EPS:] - ds["mu"]) / ds["sd"]).reshape(-1, 4), device=device)
    model = StateMLP().to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() == 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.98))
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    ck_steps = sorted(set([0] + [int(round(x)) for x in np.logspace(math.log10(30), math.log10(steps), 11)]))
    os.makedirs(CKPT_DIR, exist_ok=True)
    meta = dict(mu=ds["mu"], sd=ds["sd"], qd_scale=QD_SCALE, tau_max=TAU_MAX, out_names=OUT_NAMES,
                hparams=dict(steps=steps, batch=batch, lr=lr, wd=wd, seed=seed, d_model=D_MODEL,
                             d_hidden=D_HIDDEN, n_blocks=N_BLOCKS))
    index, log = [], []

    @torch.no_grad()
    def evaluate():
        model.eval()
        P = torch.cat([model(Xte[i:i + 65536]) for i in range(0, len(Xte), 65536)])
        model.train()
        loss = ((P - Yte) ** 2).mean().item()
        return loss, r2(P, Yte).cpu().numpy()

    def save(tag, step):
        te_loss, te_r2 = evaluate()
        torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, step=step,
                        test_loss=te_loss, test_r2=te_r2, **meta), os.path.join(CKPT_DIR, f"{tag}.pt"))
        index.append(dict(tag=tag, step=step, test_loss=te_loss, test_r2=te_r2.tolist()))
        print(f"  saved {tag} step={step} test_loss={te_loss:.4f} r2={np.round(te_r2, 4)}", flush=True)

    n = len(Xtr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm, pos = torch.randperm(n, generator=g), 0
    t0 = time.time()
    model.train()
    for step in range(steps + 1):
        if step in ck_steps:
            save(f"step{step:06d}", step)
        if step == steps:
            break
        if pos + batch > n:
            perm, pos = torch.randperm(n, generator=g), 0
        idx = perm[pos:pos + batch].to(device)
        pos += batch
        loss = ((model(Xtr[idx]) - Ytr[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            te_loss, te_r2 = evaluate()
            log.append(dict(step=step, train_loss=loss.item(), test_loss=te_loss, test_r2=te_r2.tolist()))
            print(f"step {step} train {loss.item():.4f} test {te_loss:.4f} r2 {np.round(te_r2, 4)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    save("final", steps)
    with open(os.path.join(CKPT_DIR, "index.json"), "w") as f:
        json.dump(dict(checkpoints=index, log=log), f, indent=1)
    return model


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    train(steps=steps)
