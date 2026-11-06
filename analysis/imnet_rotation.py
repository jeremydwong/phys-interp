"""imnet_ rotation search on the last layer (essay, "What Dictionaries Learn").

Essay: "if you search the rotations of a layer for the one where a handful of units carry the most activity,
starting from a random one, you land back on the neurons, in every block."

Ours (OURS; the essay gives no objective): clean_tools.rotation_search (fixed version: decaying step x1, 1/4, 1/16,
1/64; 1,000 steps), maximising E sum_i (R x)_i^4 over orthogonal R, on n = 40,000 last-layer POST-activation
vectors (2 random positions per image of the eval set), all live units (std > 1e-6).
  raw : centred, globally scaled activations, from a random rotation and (sanity check) from the identity.
  ica : the same objective after ZCA whitening (= kurtosis ICA); each found component's pattern (column of
        C^1/2 R^T) is compared to the neuron axes.
Reported: share of found axes with max |cos| >= 0.9 to a neuron axis, mean max |cos|, and the objective at the
end relative to the neuron basis. Random directions in d = 1024 / 3072 have mean max |cos| ~ 0.12 / 0.08.
Writes reports/results/imnet_rotation.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import clean_tools as ct  # noqa: E402
from analysis import imnet_common as C  # noqa: E402
from analysis.imnet_sae import vectors  # noqa: E402


def ica(X, seed, steps):
    X = X - X.mean(0)
    Cv = X.T @ X / len(X)
    ev, V = torch.linalg.eigh(Cv.double())
    ev = ev.clamp(min=ev.max() * 1e-6)
    W = (V @ torch.diag(ev ** -0.5) @ V.T).float()
    Wi = (V @ torch.diag(ev ** 0.5) @ V.T).float()
    rs = ct.rotation_search(X @ W, steps=steps, seed=seed, check_identity=False)
    P = Wi @ torch.as_tensor(rs["R"], device=X.device).T
    P = P / P.norm(dim=0, keepdim=True)
    al = P.abs().max(0).values.cpu().numpy()
    return dict(aligned_0p9=float((al >= 0.9).mean()), mean_max_cos=float(al.mean()),
                obj_end_over_identity=rs["obj_end"] / rs["obj_neuron"])


def run(dataset, arch, trained, n=40_000, steps=1000):
    X = vectors(dataset, arch, trained, n=n, seed=1)
    live = X.std(0) > 1e-6
    X = X[:, live]
    d = X.shape[1]
    t0 = time.time()
    rs = ct.rotation_search(X, steps=steps, seed=0)
    res = dict(d=d, dead_frac=float(1 - live.float().mean()), n_vectors=n,
               raw=dict(aligned_0p9=float((rs["align"] >= 0.9).mean()), mean_max_cos=float(rs["align"].mean()),
                        obj_start_over_neuron=rs["obj_start"] / rs["obj_neuron"],
                        obj_end_over_neuron=rs["obj_end"] / rs["obj_neuron"], history=rs["history"],
                        identity_start=dict(aligned_0p9=rs["identity_aligned_0p9"], mean_max_cos=rs["align_identity_mean"],
                                            obj_end_over_neuron=rs["obj_identity_end"] / rs["obj_neuron"])))
    del rs
    torch.cuda.empty_cache()
    res["ica"] = ica(X, 0, steps)
    res["seconds"] = time.time() - t0
    print(f"[rotation] {dataset} {C.tag(arch, trained)} d={d}: raw aligned {res['raw']['aligned_0p9']:.3f} "
          f"(mean cos {res['raw']['mean_max_cos']:.3f}, obj/neuron {res['raw']['obj_end_over_neuron']:.2f}; identity "
          f"start aligned {res['raw']['identity_start']['aligned_0p9']:.3f}) ica aligned {res['ica']['aligned_0p9']:.3f} "
          f"({res['seconds']:.0f} s)", flush=True)
    return res


def run_subsets(dataset, arch, trained, n=40_000, steps=1000, size=256, reps=3):
    """The vis_ protocol: 3 random 256-unit subsets (at d = 256 the 1,000-step search converges; at d = 1024 / 3072
    the full-layer search from a random start ends far below the identity-start optimum, i.e. is under-converged)."""
    X = vectors(dataset, arch, trained, n=n, seed=1)
    X = X[:, X.std(0) > 1e-6]
    rng = np.random.default_rng(0)
    rows = []
    for r in range(reps):
        cols = torch.as_tensor(np.sort(rng.choice(X.shape[1], size, replace=False)), device=X.device)
        Xs = X[:, cols]
        rs = ct.rotation_search(Xs, steps=steps, seed=r)
        ic = ica(Xs, r, steps)
        rows.append(dict(raw_aligned=float((rs["align"] >= 0.9).mean()), raw_mean_cos=float(rs["align"].mean()),
                         raw_obj_end_over_neuron=rs["obj_end"] / rs["obj_neuron"],
                         identity_aligned=rs["identity_aligned_0p9"], identity_mean_cos=rs["align_identity_mean"],
                         identity_obj_end_over_neuron=rs["obj_identity_end"] / rs["obj_neuron"],
                         ica_aligned=ic["aligned_0p9"], ica_mean_cos=ic["mean_max_cos"]))
    res = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    res["reps"] = rows
    print(f"[rotation] {dataset} {C.tag(arch, trained)} 256-subsets: {({k: round(v, 3) for k, v in res.items() if k != 'reps'})}",
          flush=True)
    return res


def main(dataset="imagenet_val", archs=("googlenet", "convnext_tiny")):
    fn = os.path.join(C.RESULTS, "imnet_rotation.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    out.setdefault(dataset, {})
    for arch in archs:
        for trained in (True, False):
            tag = C.tag(arch, trained)
            if tag in out[dataset] and os.environ.get("IMNET_RECOMPUTE") != "1":
                continue
            out[dataset][tag] = run(dataset, arch, trained)
            json.dump(out, open(fn, "w"), indent=1)
    for arch in archs:
        for trained in (True, False):
            tag = C.tag(arch, trained) + "_sub256"
            if tag in out[dataset] and os.environ.get("IMNET_RECOMPUTE") != "1":
                continue
            out[dataset][tag] = run_subsets(dataset, arch, trained)
            json.dump(out, open(fn, "w"), indent=1)
    return out


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["imagenet_val"]))
