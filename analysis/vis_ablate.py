"""vis_ weak- vs strong-activation removal (essay, "Do Weak Activations Matter?").

Essay (medium ConvNeXt, all 18 blocks): removing the weakest 75% of activations takes accuracy 72% -> 71%;
removing the strongest 7% ("the same total activation") takes it 72% -> 4%.
Ours: per neuron (post-activation, every spatial position), tau_weak = 75th percentile of |a| over its nonzero
activations on 1,000 training images; tau_strong is set so the activation mass above it equals the mass below
tau_weak (clean_tools.weak_strong_thresholds). Masks are applied to every neuron layer at once ("all") and to
the last layer only ("late"). Reported: test accuracy, fraction of answers changed, and the mean share of
nonzero activations the strong mask removes.
Writes reports/results/vis_ablate.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import clean_tools as ct  # noqa: E402
from analysis.vis_common import DATASETS, DEV, RESULTS, available, data, load_model, to_float  # noqa: E402


@torch.no_grad()
def predictions(model, x, bs=1000):
    out = []
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", torch.bfloat16):
            out.append(model(to_float(torch.as_tensor(x[i:i + bs], device=DEV))).argmax(1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def thresholds(model, x, weak_frac=0.75, max_rows=200_000):
    act = F.relu if model.__class__.__name__ == "PlainCNN" else F.gelu
    rows = {n: [] for n in model.neuron_layers}
    for i in range(0, len(x), 250):
        with torch.autocast("cuda", torch.bfloat16):
            model(to_float(torch.as_tensor(x[i:i + 250], device=DEV)), record=model.neuron_layers)
        for n in model.neuron_layers:
            a = act(model.acts[n].float())
            rows[n].append(a.permute(0, 2, 3, 1).reshape(-1, a.shape[1]))
    model.record = set()
    out = {}
    g = torch.Generator(device="cpu").manual_seed(0)
    for n, r in rows.items():
        A = torch.cat(r)
        if len(A) > max_rows:
            A = A[torch.randperm(len(A), generator=g)[:max_rows].to(A.device)]
        tw, ts, fs = ct.weak_strong_thresholds(A.to(DEV), weak_frac)
        out[n] = (tw.to(DEV), ts.to(DEV), fs)
    return out


def run(dname, tag, seed):
    D = data(dname)
    m = load_model(dname, tag, seed)
    xc = D["xtr"][:1000]
    th = thresholds(m, xc)
    base = predictions(m, D["xte"])
    y = D["yte"]
    res = dict(base_acc=float((base == y).mean()))
    for scope, layers in (("all", m.neuron_layers), ("late", m.neuron_layers[-1:])):
        for mode in ("weak", "strong"):
            m.edit = {n: ct.mask_fn(th[n][0] if mode == "weak" else th[n][1], mode) for n in layers}
            p = predictions(m, D["xte"])
            m.edit = {}
            res[f"{scope}_{mode}_acc"] = float((p == y).mean())
            res[f"{scope}_{mode}_changed"] = float((p != base).mean())
        res[f"{scope}_strong_frac_removed"] = float(np.mean([th[n][2].mean().item() for n in layers]))
    return res


def main(tags=("old", "recipe", "arch")):
    fn = os.path.join(RESULTS, "vis_ablate.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for dname in DATASETS:
        for tag in tags:
            for seed in available(dname, tag):
                name = f"{dname}_{tag}_s{seed}"
                out[name] = run(dname, tag, seed)
                print(name, out[name], flush=True)
                json.dump(out, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("old", "recipe", "arch"))
