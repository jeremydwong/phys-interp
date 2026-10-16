"""Train the candidate-B video predictor on data/main (episodes [0,2000) train).

    uv run python models/video_train.py --steps 4000            # the checkpoint analysed (bs 32)
    uv run python models/video_train.py --steps 20000 --bs 64 --lr 8e-4   # suggested longer CUDA run
Device: cuda if available, else mps, else cpu. On CUDA the train split (3.7 GB uint8) is kept on the GPU.

Loss: weighted MSE on frames t+1 (weight 1) and t+2 (weight 0.5). Pixel weight is
1 + 25 * mask, where mask marks pixels that change between t-1..t+2 (dilated 5x5), i.e. the
arm; the background is static so unweighted MSE would be dominated by "copy the frame".
No augmentation except a small global brightness jitter (flips/rotations would break the
physics: the wall is on one side and gravity points down).
Checkpoints: step 0 (random-init control) + 10 log-spaced steps -> models/video_ckpt/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.video_model import VideoPredictor  # noqa: E402

CKPT = os.path.join(ROOT, "models", "video_ckpt")
CFG = dict(c=128, n_enc=4, n_tmp=4, heads=4, mult=4)
T0, T1 = 3, 147          # valid "frame t" indices (need t-3 .. t+2)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_input(win_u8):
    """uint8 [B, K, 64, 64, 3] -> float [B, K, 3, 64, 64]."""
    return win_u8.permute(0, 1, 4, 2, 3).float() / 255.0


def gather(frames, e, t, offs):
    return frames[e[:, None], t[:, None] + offs[None]]


def change_mask(seq):
    """seq float [B, 6, 3, 64, 64] (t-3..t+2) -> mask [B, 1, 64, 64] of moving-arm pixels."""
    d = (seq[:, 2:] - seq[:, 2:3]).abs().amax(2).amax(1, keepdim=True)   # vs frame t
    m = (d > 0.03).float()
    return F.max_pool2d(m, 5, 1, 2)


def weighted_loss(pred, tgt, mask, wts=(1.0, 0.5)):
    w = 1.0 + 25.0 * mask                     # [B,1,64,64]
    tot = 0
    for k, a in enumerate(wts):
        se = ((pred[:, k] - tgt[:, k]) ** 2).mean(1, keepdim=True)
        tot = tot + a * (w * se).sum() / w.sum()
    return tot / sum(wts)


@torch.no_grad()
def evaluate(model, frames, eps, ts, dev, bs=256):
    """Returns dict of MSEs (model, copy-last, linear extrapolation), all pixels and arm pixels, t+1."""
    offs = torch.arange(-3, 3, device=dev)
    acc = {k: 0.0 for k in ["model", "copy", "linear", "model_arm", "copy_arm", "linear_arm", "model_t2", "copy_t2"]}
    n, n_arm = 0, 0
    for i in range(0, len(eps), bs):
        e = torch.as_tensor(eps[i:i + bs], device=dev)
        t = torch.as_tensor(ts[i:i + bs], device=dev)
        seq = to_input(gather(frames, e, t, offs))
        x, y = seq[:, :4], seq[:, 4:]
        p = model(x)
        mask = change_mask(seq)[:, 0]
        lin = (2 * x[:, -1] - x[:, -2]).clamp(0, 1)
        for name, pr in [("model", p[:, 0]), ("copy", x[:, -1]), ("linear", lin)]:
            se = ((pr - y[:, 0]) ** 2).mean(1)
            acc[name] += se.sum().item()
            acc[name + "_arm"] += (se * mask).sum().item()
        acc["model_t2"] += ((p[:, 1] - y[:, 1]) ** 2).mean(1).sum().item()
        acc["copy_t2"] += ((x[:, -1] - y[:, 1]) ** 2).mean(1).sum().item()
        n += se.numel()
        n_arm += mask.sum().item()
    out = {}
    for k, v in acc.items():
        out[k] = v / (n_arm if k.endswith("_arm") else n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    dev = pick_device()
    print(f"device: {dev}", flush=True)
    os.makedirs(CKPT, exist_ok=True)

    fr = np.load(os.path.join(ROOT, "data", "main", "frames64.npy"), mmap_mode="r")
    t_load = time.time()
    train = torch.from_numpy(np.ascontiguousarray(fr[:2000])).to(dev)
    val = torch.from_numpy(np.ascontiguousarray(fr[2300:2400])).to(dev)
    print(f"loaded frames in {time.time() - t_load:.0f}s", flush=True)
    rng = np.random.default_rng(1)
    v_eps = np.repeat(np.arange(100), 8)
    v_ts = rng.integers(T0, T1 + 1, len(v_eps))

    model = VideoPredictor(**CFG).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{n_par / 1e6:.2f}M params", flush=True)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "pos" not in n]
    nodecay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "pos" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                             {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98))
    warm = min(1000, a.steps // 10)

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        return a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm)))

    save_at = sorted(set([0] + [int(round(x)) for x in np.geomspace(100, a.steps, 10)]))

    def save(step, log):
        torch.save({"model": model.state_dict(), "cfg": CFG, "step": step, "log": log},
                   os.path.join(CKPT, f"video_step{step:06d}.pt"))

    offs = torch.arange(-3, 3, device=dev)
    log = {"cfg": CFG, "params": n_par, "save_at": save_at, "train": [], "val": []}
    t_start = time.time()
    run = 0.0
    for step in range(a.steps + 1):
        if step in save_at:
            model.eval()
            ev = evaluate(model, val, v_eps, v_ts, dev)
            ev["step"], ev["time"] = step, time.time() - t_start
            log["val"].append(ev)
            print(f"[val] step {step} " + " ".join(f"{k}={v:.5f}" for k, v in ev.items() if k not in ("step",)), flush=True)
            save(step, log)
            model.train()
        if step == a.steps:
            break
        e = torch.randint(0, train.shape[0], (a.bs,), device=dev)
        t = torch.randint(T0, T1 + 1, (a.bs,), device=dev)
        seq = to_input(gather(train, e, t, offs))
        seq = (seq * (1 + 0.05 * (torch.rand(a.bs, 1, 1, 1, 1, device=dev) - 0.5))).clamp(0, 1)
        mask = change_mask(seq)
        pred = model(seq[:, :4])
        loss = weighted_loss(pred, seq[:, 4:], mask)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 200 == 0:
            el = time.time() - t_start
            print(f"step {step} loss {run:.5f} lr {lr_at(step):.2e} {el:.0f}s", flush=True)
            log["train"].append((step, run))
            if a.bench and step == 400:
                print(f"{el / 400 * 1000:.0f} ms/step"); return
        if step % 500 == 0 and step and not a.bench:
            # periodic extra validation so the plateau is visible
            model.eval()
            ev = evaluate(model, val, v_eps, v_ts, dev)
            ev["step"], ev["time"] = step, time.time() - t_start
            log["val"].append(ev)
            print(f"[val] step {step} model_arm={ev['model_arm']:.5f} copy_arm={ev['copy_arm']:.5f}", flush=True)
            model.train()
    with open(os.path.join(CKPT, "video_trainlog.json"), "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
