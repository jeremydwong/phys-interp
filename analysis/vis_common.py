"""Shared plumbing for the vis_ ladder analyses: model registry, cached pole scores, layer depth groups."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vis_data import load  # noqa: E402
from models.vis_train import CKPT, DEV, load_model, to_float  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "vis_cache")
RESULTS = os.path.join(ROOT, "reports", "results")
DATASETS = os.environ.get("VIS_DATASETS", "shapes,cifar").split(",")
STEPS = ["old", "recipe", "arch"]
SEEDS = [0, 1, 2]
STEP_LABEL = {"old": "M_old (plain CNN, old recipe)", "recipe": "M_recipe (plain CNN, modern recipe)",
              "arch": "M_arch (ConvNeXt, modern recipe)", "archs": "ConvNeXt 0.2M", "archl": "ConvNeXt 4.6M"}
# base factors used for the conjunction (pair) test on shapes; derived ones are coarsenings of these
PAIR_FACTORS = {"shapes": ["shape", "color", "texture", "bg_type", "size", "x", "y", "rot_phase", "bg_lum"], "cifar": []}

_DATA = {}


def data(name):
    if name not in _DATA:
        _DATA[name] = load(name)
    return _DATA[name]


def depth_groups(layers):
    """early = last layer of stage 1, mid = last layer of stage 3, late = last layer."""
    last_of = {}
    for n in layers:
        last_of[n[:2]] = n
    return {"early": last_of["s1"], "mid": last_of["s3"], "late": layers[-1]}


def prev_layer(layers, name):
    i = layers.index(name)
    return layers[i - 1]


@torch.no_grad()
def pole_scores(model, x_uint8, layers=None, bs=500):
    """-> {layer: (pos [N, C], neg [N, C]) float32} for the given uint8 images [N, H, W, 3]."""
    layers = layers or model.neuron_layers
    out = {n: ([], []) for n in layers}
    for i in range(0, len(x_uint8), bs):
        xb = to_float(torch.as_tensor(x_uint8[i:i + bs], device=DEV))
        with torch.autocast("cuda", torch.bfloat16):
            model(xb, record=layers)
        for n in layers:
            z = model.acts[n].float()
            p, m = z.amax((2, 3)), -z.amin((2, 3))
            out[n][0].append(p.cpu().numpy())
            out[n][1].append(m.cpu().numpy())
    model.record = set()
    return {n: (np.concatenate(a), np.concatenate(b)) for n, (a, b) in out.items()}


def cached_scores(dname, tag, seed, init=False):
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"{dname}_{tag}_s{seed}{'_init' if init else ''}_scores.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        layers = [k[:-4] for k in z.files if k.endswith("_pos")]
        return {n: (z[n + "_pos"], z[n + "_neg"]) for n in layers}
    m = load_model(dname, tag, seed, init)
    sc = pole_scores(m, data(dname)["xte"])
    np.savez(fn, **{f"{n}_pos": p for n, (p, q) in sc.items()}, **{f"{n}_neg": q for n, (p, q) in sc.items()})
    return sc


def cached_resid(dname, tag, seed, init=False):
    """ConvNeXt only: pole scores of the final residual-stream channels (input to the head), [N, C] each."""
    fn = os.path.join(CACHE, f"{dname}_{tag}_s{seed}{'_init' if init else ''}_resid.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return z["pos"], z["neg"]
    m = load_model(dname, tag, seed, init)
    sc = pole_scores(m, data(dname)["xte"], layers=["resid"])["resid"]
    np.savez(fn, pos=sc[0], neg=sc[1])
    return sc


def available(dname, tag):
    return [s for s in SEEDS + [3, 4] if os.path.exists(os.path.join(CKPT, f"{dname}_{tag}_s{s}.pt"))]


def train_log(dname, tag, seed):
    return json.load(open(os.path.join(CKPT, f"{dname}_{tag}_s{seed}.json")))


def mean_ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    from scipy import stats
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
