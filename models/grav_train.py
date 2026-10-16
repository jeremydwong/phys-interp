"""Train the gravity-study dynamics transformers.

    uv run python models/grav_train.py incontext   # main model, gravity must be inferred
    uv run python models/grav_train.py given       # control: gravity unit vector is an input
    uv run python models/grav_train.py h1          # control: history length 1

Train: data/grav episodes [0,1000) + data/grav_extra/train_extra (9000 eps, same distribution).
Test : data/grav episodes [1000,1200), windows starting at frames 0, 43, 85.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.common import N_TRAIN_EPS_GRAV, load_states  # noqa: E402
from analysis.grav_sim import load_extra  # noqa: E402
from models.grav_model import CKPT, CTX, GravTransformer, device, features, raw_targets  # noqa: E402

STEPS = int(os.environ.get("GRAV_STEPS", 40000))
BS, LR, WD, WARM = 256, 1e-3, 0.05, 1000
TEST_STARTS = (0, 43, 85)


def make_xy(st, given):
    X = features(st["q"], st["qd"], st["tau_act"], st["gravity_angle"] if given else None)
    Y = raw_targets(st["q"], st["qd"])
    return X[:, :-1], Y          # X[:, t] predicts Y[:, t]


def test_windows(X, Y):
    xs = np.concatenate([X[:, s:s + CTX] for s in TEST_STARTS])
    ys = np.concatenate([Y[:, s:s + CTX] for s in TEST_STARTS])
    return xs, ys


def main(variant):
    torch.manual_seed(0)
    dev = device()
    given = variant == "given"
    st = load_states("grav")
    ex = load_extra("train_extra")
    Xa, Ya = make_xy(st, given)
    Xe, Ye = make_xy(ex, given)
    Xtr = np.concatenate([Xa[:N_TRAIN_EPS_GRAV], Xe])
    Ytr = np.concatenate([Ya[:N_TRAIN_EPS_GRAV], Ye])
    Xte, Yte = test_windows(Xa[N_TRAIN_EPS_GRAV:], Ya[N_TRAIN_EPS_GRAV:])
    ym, ys = Ytr.reshape(-1, 4).mean(0), Ytr.reshape(-1, 4).std(0)
    model = GravTransformer(d_in=Xtr.shape[-1], attn="self" if variant == "h1" else "causal").to(dev)
    model.y_mean.copy_(torch.tensor(ym)); model.y_std.copy_(torch.tensor(ys))
    print(variant, "params", sum(p.numel() for p in model.parameters()), "train eps", len(Xtr), flush=True)

    Xtr_t, Ytr_t = torch.tensor(Xtr, device=dev), torch.tensor((Ytr - ym) / ys, device=dev)
    Xte_t, Yte_t = torch.tensor(Xte, device=dev), torch.tensor((Yte - ym) / ys, device=dev)
    E, Tm = Xtr.shape[:2]
    ar = torch.arange(CTX, device=dev)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and n != "pos"]
    nodecay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and n != "pos")]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0}],
                            lr=LR, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / WARM) * (0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)) * 0.97 + 0.03))

    def evaluate():
        model.eval()
        with torch.no_grad():
            per = torch.zeros(CTX, 4, device=dev)
            for i in range(0, len(Xte_t), 512):
                p = model(Xte_t[i:i + 512])
                per += ((p - Yte_t[i:i + 512]) ** 2).sum(0)
            per /= len(Xte_t)
        model.train()
        return per.cpu().numpy()

    log = []
    t0 = time.time()
    for step in range(STEPS + 1):
        if step % 2000 == 0:
            per = evaluate()
            log.append(dict(step=step, test=float(per.mean()), test_late=float(per[32:].mean()),
                            time=time.time() - t0))
            print(f"{variant} step {step:6d} test {per.mean():.5f} late {per[32:].mean():.5f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        if step == STEPS:
            break
        ei = torch.randint(0, E, (BS,), device=dev)
        si = torch.randint(0, Tm - CTX + 1, (BS,), device=dev)
        idx = si[:, None] + ar
        xb = Xtr_t[ei[:, None], idx]
        yb = Ytr_t[ei[:, None], idx]
        loss = ((model(xb) - yb) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0:
            log.append(dict(step=step, train=float(loss.detach())))
    per = evaluate()
    os.makedirs(CKPT, exist_ok=True)
    torch.save({"cfg": model.cfg, "state": model.state_dict()}, os.path.join(CKPT, f"grav_{variant}.pt"))
    with open(os.path.join(CKPT, f"grav_{variant}_log.json"), "w") as f:
        json.dump(dict(log=log, per_pos_dim=per.tolist()), f)
    print("saved", variant, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
