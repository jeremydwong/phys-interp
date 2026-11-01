"""Train the vis_ ladder models.

  uv run python models/vis_train.py --data shapes --step old --seed 0
  --step old     PlainCNN, "old" recipe: SGD + momentum 0.9, lr 0.05, coupled L2 5e-4, step decay x0.1 at
                 50% / 75% of training, random crop (pad 4) + horizontal flip only.
  --step recipe  PlainCNN, "modern" recipe (below).
  --step arch    ConvNeXtS (parameter-matched, size m; --size s|l for the sweep), modern recipe.
  --step readout independent factor/class readout used only to judge feature visualizations.
Modern recipe (our choice; the essay does not list its regularizers): AdamW lr 2e-3, wd 0.05 (not on
norms/biases/layer scale), 3-epoch linear warmup + cosine, label smoothing 0.1, mixup(0.8)/cutmix(1.0)
(one of them on every batch, 50/50), "RandAugment-lite" (random affine: rotate +-15 deg, scale 0.9-1.1,
shear +-0.1, translate +-10%; brightness/contrast x U(0.8, 1.2); no hue changes because hue is a class
factor in the shapes set), random erasing p=0.25, EMA of weights 0.999 (evaluated and saved).
No stochastic depth. Both recipes: batch 256, bf16 autocast.

Saves models/vis_ckpt/{data}_{tag}_s{seed}.pt (final; EMA weights for the modern recipe) and
{data}_{tag}_s{seed}_init.pt (the random init of the same seed, the control), plus a log json.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vis_data import load  # noqa: E402
from models.vis_models import Readout, build, n_params  # noqa: E402

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis_ckpt")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tag_of(step, size="m"):
    return step if (step != "arch" or size == "m") else f"arch{size}"


def to_float(xb):
    return xb.permute(0, 3, 1, 2).float().div_(255)


def crop_flip(x, pad):
    B, _, H, W = x.shape
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = (i[:, None] + torch.arange(H, device=x.device)[None])            # B H
    cols = (j[:, None] + torch.arange(W, device=x.device)[None])            # B W
    out = xp[torch.arange(B, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None],
             rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(B, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


def randaug_lite(x):
    B = x.shape[0]
    d = x.device
    a = (torch.rand(B, device=d) * 2 - 1) * math.radians(15)
    s = 0.9 + 0.2 * torch.rand(B, device=d)
    sh = (torch.rand(B, device=d) * 2 - 1) * 0.1
    t = (torch.rand(B, 2, device=d) * 2 - 1) * 0.2
    cos, sin = torch.cos(a) / s, torch.sin(a) / s
    theta = torch.stack([torch.stack([cos, -sin + sh, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1)
    g = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, g, padding_mode="reflection", align_corners=False)
    br = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=d)
    ct = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=d)
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * ct + m) * br
    return random_erase(x.clamp(0, 1), 0.25)


def random_erase(x, p):
    B, _, H, W = x.shape
    for b in torch.nonzero(torch.rand(B) < p).flatten().tolist():
        area = np.random.uniform(0.02, 0.25) * H * W
        ar = math.exp(np.random.uniform(math.log(0.3), math.log(3.3)))
        h, w = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
        if h < H and w < W:
            i, j = np.random.randint(0, H - h), np.random.randint(0, W - w)
            x[b, :, i:i + h, j:j + w] = torch.rand(3, h, w, device=x.device)
    return x


def mix(x, y, n_classes, smoothing):
    yo = F.one_hot(y, n_classes).float() * (1 - smoothing) + smoothing / n_classes
    perm = torch.randperm(x.shape[0], device=x.device)
    if np.random.rand() < 0.5:
        lam = float(np.random.beta(0.8, 0.8))
        x = lam * x + (1 - lam) * x[perm]
    else:
        lam0 = float(np.random.beta(1.0, 1.0))
        H, W = x.shape[2:]
        h, w = int(H * math.sqrt(1 - lam0)), int(W * math.sqrt(1 - lam0))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y0, y1, x0, x1 = max(cy - h // 2, 0), min(cy + h // 2, H), max(cx - w // 2, 0), min(cx + w // 2, W)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    return x, lam * yo + (1 - lam) * yo[perm]


@torch.no_grad()
def evaluate(model, xte, yte, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(xte), bs):
        with torch.autocast("cuda", torch.bfloat16):
            out = model(to_float(xte[i:i + bs]))
        correct += (out.argmax(1) == yte[i:i + bs]).sum().item()
    return correct / len(xte)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    D = load(args.data)
    xtr = torch.as_tensor(D["xtr"], device=DEV)
    ytr = torch.as_tensor(D["ytr"], device=DEV)
    xte = torch.as_tensor(D["xte"], device=DEV)
    yte = torch.as_tensor(D["yte"], device=DEV)
    kind = "plain" if args.step in ("old", "recipe") else "convnext"
    model = build(kind, D["n_classes"], D["stem_stride"], D["mean"], D["std"], args.size).to(DEV)
    model = model.to(memory_format=torch.channels_last)
    tag = tag_of(args.step, args.size) + args.suffix
    name = f"{args.data}_{tag}_s{args.seed}"
    os.makedirs(CKPT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CKPT, name + "_init.pt"))
    modern = args.step != "old"
    pad = 4 if D["res"] == 32 else 6
    N, bs = len(xtr), 256
    steps_per_epoch = N // bs
    total = steps_per_epoch * args.epochs
    if modern:
        decay, no_decay = [], []
        for n_, p in model.named_parameters():
            (no_decay if (p.ndim <= 1 or "gamma" in n_) else decay).append(p)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
                                lr=2e-3, betas=(0.9, 0.999))
        warm = 3 * steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.5 * total), int(0.75 * total)], 0.1)
        ema = None
    log = dict(name=name, data=args.data, step=args.step, size=args.size, seed=args.seed, epochs=args.epochs,
               params=n_params(model), curve=[])
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(N, device=DEV)
        tl = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            x = crop_flip(to_float(xtr[idx]), pad)
            y = ytr[idx]
            if modern:
                x = randaug_lite(x)
                x, yt = mix(x, y, D["n_classes"], 0.1)
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", torch.bfloat16):
                out = model(x)
                loss = (-(yt * F.log_softmax(out.float(), 1)).sum(1).mean()) if modern else F.cross_entropy(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if ema is not None:
                with torch.no_grad():
                    dcy = min(0.999, (1 + step) / (10 + step))
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - dcy)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            tl += loss.item() if i % 20 == 0 else 0
        if ep % 5 == 4 or ep == args.epochs - 1:
            acc = evaluate(ema if ema is not None else model, xte, yte)
            log["curve"].append(dict(epoch=ep + 1, test_acc=acc, train_loss=tl / (steps_per_epoch / 20), time=time.time() - t0))
            print(name, log["curve"][-1], flush=True)
    final = ema if ema is not None else model
    log["test_acc"] = evaluate(final, xte, yte)
    log["train_acc"] = evaluate(final, xtr[:10000], ytr[:10000])
    log["time_s"] = time.time() - t0
    torch.save(final.state_dict(), os.path.join(CKPT, name + ".pt"))
    json.dump(log, open(os.path.join(CKPT, name + ".json"), "w"), indent=1)
    print("DONE", name, log["test_acc"], f"{log['time_s']:.0f}s", flush=True)


# ------------------------------------------------------------------ readout (FV judge)

def readout_heads(data):
    if data == "shapes":
        return {"shape": 6, "color": 6, "texture": 6, "bg_type": 3, "size": 1, "x": 1, "y": 1, "bg_lum": 1}
    return {"label": 10}


def train_readout(args):
    torch.manual_seed(1000 + args.seed)
    D = load(args.data)
    heads = readout_heads(args.data)
    model = Readout(heads, stem_stride=D["stem_stride"], mean=D["mean"], std=D["std"]).to(DEV)
    xtr = torch.as_tensor(D["xtr"], device=DEV)
    ftr = {k: torch.as_tensor(np.asarray(D["ftr"][k]), device=DEV) for k in heads}
    cont = {k: (float(np.mean(D["ftr"][k])), float(np.std(D["ftr"][k]))) for k, n in heads.items() if n == 1}
    N, bs = len(xtr), 256
    total = (N // bs) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=total, pct_start=0.1)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(N, device=DEV)
        for i in range(N // bs):
            idx = perm[i * bs:(i + 1) * bs]
            x = to_float(xtr[idx])
            if args.data == "cifar":
                x = randaug_lite(crop_flip(x, 4))
            else:          # factors include position / size / luminance, so only noise + erasing
                x = random_erase((x + 0.06 * torch.randn_like(x) * torch.rand(len(x), 1, 1, 1, device=DEV)).clamp(0, 1), 0.2)
            with torch.autocast("cuda", torch.bfloat16):
                out = model(x)
            loss = 0
            for k, n in heads.items():
                o = out[k].float()
                if n == 1:
                    m, s = cont[k]
                    loss = loss + F.mse_loss(o[:, 0], (ftr[k][idx].float() - m) / s)
                else:
                    loss = loss + F.cross_entropy(o, ftr[k][idx].long(), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    xte = torch.as_tensor(D["xte"], device=DEV)
    res = {}
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        outs = [model(to_float(xte[i:i + 1000])) for i in range(0, len(xte), 1000)]
    for k, n in heads.items():
        o = torch.cat([u[k].float() for u in outs])
        t = np.asarray(D["fte"][k])
        if n == 1:
            m, s = cont[k]
            res[k] = float(np.corrcoef(o[:, 0].cpu().numpy() * s + m, t)[0, 1] ** 2)
        else:
            res[k] = float((o.argmax(1).cpu().numpy() == t).mean())
    name = f"{args.data}_readout"
    torch.save({"state": model.state_dict(), "heads": heads, "cont": cont}, os.path.join(CKPT, name + ".pt"))
    json.dump({"test": res, "time_s": time.time() - t0}, open(os.path.join(CKPT, name + ".json"), "w"), indent=1)
    print("DONE readout", res, flush=True)


def load_readout(data):
    ck = torch.load(os.path.join(CKPT, f"{data}_readout.pt"), map_location=DEV)
    D = load(data)
    m = Readout(ck["heads"], stem_stride=D["stem_stride"], mean=D["mean"], std=D["std"]).to(DEV).eval()
    m.load_state_dict(ck["state"])
    return m, ck["cont"]


def load_model(data, tag, seed, init=False):
    D = load(data)
    kind = "plain" if tag.startswith(("old", "recipe")) else "convnext"
    size = tag[4] if tag.startswith("arch") and len(tag) > 4 and tag[4] in "sl" else "m"
    m = build(kind, D["n_classes"], D["stem_stride"], D["mean"], D["std"], size).to(DEV)
    m.load_state_dict(torch.load(os.path.join(CKPT, f"{data}_{tag}_s{seed}{'_init' if init else ''}.pt"), map_location=DEV))
    return m.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="shapes")
    ap.add_argument("--step", default="old", choices=["old", "recipe", "arch", "readout"])
    ap.add_argument("--size", default="m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--suffix", default="", help="appended to the checkpoint tag, e.g. _long")
    a = ap.parse_args()
    train_readout(a) if a.step == "readout" else train(a)
