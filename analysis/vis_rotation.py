"""vis_ rotation search, two formalizations (both OURS; the essay gives no objective).

Essay ("What Dictionaries Learn"): "if you search the rotations of a layer for the one where a handful of units
carry the most activity, starting from a random one, you land back on the neurons, in every block."

  raw : maximise E sum_i (R x)_i^4 over orthogonal R on centred, globally scaled activations (clean_tools.rotation_search).
        On unwhitened data this objective is dominated by the few highest-variance, heavy-tailed directions.
  ica : the same objective after ZCA whitening (basis-free: C^-1/2 is rotation-equivariant), i.e. kurtosis ICA.
        Each found component's pattern (column of C^1/2 R^T, the direction it writes in activation space) is
        compared to the neuron axes.
For each: 3 random 256-unit subsets of the last neuron layer, post-activation vectors at every position of
6,250 training images (100k vectors). Reports share of found axes with |cos| >= 0.9 to a neuron axis and the
mean max |cos| (random 256-d directions: mean max |cos| ~ 0.2, share 0).
Writes reports/results/vis_rotation.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import clean_tools as ct  # noqa: E402
from analysis.vis_common import DATASETS, DEV, RESULTS, available, data, load_model  # noqa: E402
from analysis.vis_dict import late_vectors  # noqa: E402


def ica_alignment(X, seed):
    X = X - X.mean(0)
    C = X.T @ X / len(X)
    ev, V = torch.linalg.eigh(C)
    ev = ev.clamp(min=ev.max() * 1e-6)
    W = V @ torch.diag(ev ** -0.5) @ V.T          # ZCA whitening
    Wi = V @ torch.diag(ev ** 0.5) @ V.T
    rs = ct.rotation_search(X @ W, steps=1000, seed=seed)
    P = Wi @ torch.as_tensor(rs["R"], device=X.device).T      # columns = patterns
    P = P / P.norm(dim=0, keepdim=True)
    al = P.abs().max(0).values.cpu().numpy()
    return al


def run(dname, tag, seed, init=False):
    D = data(dname)
    m = load_model(dname, tag, seed, init)
    X = late_vectors(m, D["xtr"][:6250], m.neuron_layers[-1])
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    raw, ica, rawm, icam = [], [], [], []
    for r in range(3):
        cols = torch.as_tensor(rng.choice(d, min(256, d), replace=False), device=X.device)
        Xs = X[:, cols]
        live = Xs.std(0) > 1e-6
        Xs = Xs[:, live]
        a = ct.rotation_search(Xs, steps=1000, seed=100 * seed + r)["align"]
        b = ica_alignment(Xs, 100 * seed + r)
        raw.append(float((a >= 0.9).mean()))
        rawm.append(float(a.mean()))
        ica.append(float((b >= 0.9).mean()))
        icam.append(float(b.mean()))
    return dict(raw_aligned=float(np.mean(raw)), raw_mean_cos=float(np.mean(rawm)),
                ica_aligned=float(np.mean(ica)), ica_mean_cos=float(np.mean(icam)), dead_frac=float(1 - live.float().mean()))


def main(tags=("old", "recipe", "arch")):
    fn = os.path.join(RESULTS, "vis_rotation.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for dname in DATASETS:
        for tag in tags:
            for seed in available(dname, tag):
                for init in (False, True):
                    name = f"{dname}_{tag}_s{seed}{'_init' if init else ''}"
                    if name in out and os.environ.get("VIS_RECOMPUTE") != "1":
                        continue
                    out[name] = run(dname, tag, seed, init)
                    print(name, out[name], flush=True)
                    json.dump(out, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main()
