"""vis_ dictionaries vs neurons (essay, "What Dictionaries Learn").

Essay: "On the last layer, 88% of its activity is near-copies of single neurons, and the rest is faint:
removing every dictionary direction that matches no neuron leaves accuracy unchanged", and a rotation search
started from a random rotation lands "back on the neurons, in every block".

Ours (formalizations in clean_tools): last neuron layer, post-activation vectors at every spatial position of
6,250 training images (100k vectors). TopK SAE with 4x as many features as neurons and k = d/32, 4,000 Adam steps.
  near-copy share: activity share of features whose decoder has max |cos| >= 0.9 with one neuron axis.
  controls: (i) the same activations under a random rotation (near copies measured against the rotated
  neuron axes, i.e. the basis has no privileged meaning); (ii) the random-init network.
  splice: replace the layer by the SAE reconstruction using all features / only near-copy / only other features.
The rotation search lives in analysis/vis_rotation.py.
Writes reports/results/vis_dict.json.
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
from analysis.vis_ablate import predictions  # noqa: E402
from analysis.vis_common import DATASETS, DEV, RESULTS, available, data, load_model, to_float  # noqa: E402


@torch.no_grad()
def late_vectors(model, x, layer, bs=500):
    act = F.relu if model.__class__.__name__ == "PlainCNN" else F.gelu
    out = []
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", torch.bfloat16):
            model(to_float(torch.as_tensor(x[i:i + bs], device=DEV)), record=[layer])
        a = act(model.acts[layer].float())
        out.append(a.permute(0, 2, 3, 1).reshape(-1, a.shape[1]))
    model.record = set()
    return torch.cat(out)


def splice_fn(sae, subset_mask):
    """model.edit hook: replace post-activations by the SAE reconstruction restricted to a feature subset."""
    Wd = sae["W_dec"] * subset_mask[:, None].float()

    def f(a, name):
        B, C, H, W = a.shape
        x = a.permute(0, 2, 3, 1).reshape(-1, C).float()
        z = sae["encode"](x)
        rec = z @ Wd + sae["b_pre"]
        return rec.view(B, H, W, C).permute(0, 3, 1, 2).to(a.dtype)
    return f


def run(dname, tag, seed, init=False):
    D = data(dname)
    m = load_model(dname, tag, seed, init)
    layer = m.neuron_layers[-1]
    X = late_vectors(m, D["xtr"][:6250], layer)        # 100k position vectors
    d = X.shape[1]
    res = dict(layer=layer, d=d, n_vectors=len(X))
    # SAE in the neuron basis
    sae = ct.train_topk_sae(X, 4 * d, max(4, d // 32), steps=4000, seed=seed)
    share, frac, c = ct.near_copy_share(sae["W_dec"], sae["mean_z"])
    res.update(sae_fve=sae["fve"], near_copy_share=share, near_copy_frac_live=frac,
               cos_hist=np.histogram(c[sae["mean_z"] > 0], bins=10, range=(0, 1))[0].tolist())
    if not init:
        base = predictions(m, D["xte"])
        y = D["yte"]
        res["base_acc"] = float((base == y).mean())
        nc = torch.as_tensor(c >= 0.9, device=DEV)
        for nm, mask in (("all", torch.ones_like(nc)), ("near_copy_only", nc), ("other_only", ~nc)):
            m.edit = {layer: splice_fn(sae, mask)}
            p = predictions(m, D["xte"])
            m.edit = {}
            res[f"splice_{nm}_acc"] = float((p == y).mean())
    # control: random rotation of the same activations
    Q = torch.as_tensor(ct.random_rotation(d, 1234 + seed), dtype=torch.float32, device=DEV)
    sae_r = ct.train_topk_sae(X @ Q.T, 4 * d, max(4, d // 32), steps=4000, seed=seed)
    share_r, frac_r, _ = ct.near_copy_share(sae_r["W_dec"], sae_r["mean_z"])
    res.update(rot_sae_fve=sae_r["fve"], rot_near_copy_share=share_r, rot_near_copy_frac_live=frac_r)
    del X, sae, sae_r
    torch.cuda.empty_cache()
    return res


def main(tags=("old", "recipe", "arch")):
    fn = os.path.join(RESULTS, "vis_dict.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for dname in DATASETS:
        for tag in tags:
            for seed in available(dname, tag):
                for init in (False, True):
                    name = f"{dname}_{tag}_s{seed}{'_init' if init else ''}"
                    if name in out and os.environ.get("VIS_RECOMPUTE") != "1":
                        continue
                    out[name] = run(dname, tag, seed, init)
                    print(name, {k: v for k, v in out[name].items() if not isinstance(v, list)}, flush=True)
                    json.dump(out, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("old", "recipe", "arch"))
