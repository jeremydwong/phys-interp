"""imnet_ weak-activation grid (essay, "Weak Activations: Further Measurements", figure "weak-ablation-grid").

Essay: "Keeping roughly the strongest quarter of positive activations in Inception's final layer preserves 69.47%
accuracy, versus 70.08% originally. ConvNeXt's late MLP branches are similarly tolerant of weak-response removal,
with their residual bypass intact." "The bottom row of each ConvNeXt panel applies the cutoff at all 18 blocks at
once. At p99, medium falls to 3.80% accuracy and large to 27.78%". The ConvNeXt panel is labelled "Magnitude of
SwiGLU product. Residual bypass stays intact." Also (medium ConvNeXt, all 18 blocks): "Removing the weakest 75%:
accuracy 72% -> to 71% · 1 in 10 answers change. Removing the strongest 7%: accuracy 72% -> to 4% · 96 in 100
answers change. The same total activation as the weakest 75%".

Ours. Cutoff definition (OURS, chosen to match the essay's wording; the p75 cell is "keeping roughly the strongest
quarter"): per neuron (channel), tau_p = the p-th percentile of its activation magnitude, estimated on the OTHER
dataset (fit ImageNetV2 -> evaluate ImageNet val), from 4 random positions x 5,000 images.
  googlenet: the layer's post-ReLU output (conv1..3, inception3a..5b block outputs = the essay's conv2d0..2,
             mixed3a..5b); percentile over the neuron's POSITIVE activations; zero every activation < tau_p.
  convnext:  each block's MLP hidden after GELU (the GELU product, our analogue of the essay's SwiGLU product);
             percentile over |GELU(h)| at all positions; zero |a| < tau_p. The residual bypass is untouched.
One layer at a time, plus (convnext) every block at once ("all"). Top-1 on the eval set (fp16 autocast; cuDNN
nondeterminism moves top-1 by ~0.01 pp between identical runs). Random-init controls are at chance (0.1%), so the
grid is reported for trained models only.
Mass-matched pair (clean_tools.weak_strong_thresholds, OURS): remove the weakest 75% of nonzero |a| per neuron vs
the strongest set carrying the same total activation, at the last layer and (convnext) all blocks.
Writes reports/results/imnet_weak.json.
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
from analysis import imnet_extract as E  # noqa: E402
from analysis.imnet_sae import FIT  # noqa: E402

PCTS = (25, 50, 75, 95, 99)


@torch.no_grad()
def fit_samples(fit, arch, n_img=5000, n_pos=4, bs=100):
    """-> {layer: post-activation samples [n_img * n_pos, C] float16 (cpu)}."""
    fn = os.path.join(C.cache_dir(fit), f"{arch}_weakfit.pt")
    if os.path.exists(fn):
        return torch.load(fn)
    m = C.load_model(arch, True)
    names = C.block_names(arch)
    rec = C.Recorder(m, arch).attach(names)
    g = torch.Generator(device="cpu").manual_seed(0)
    out = {n: [] for n in names}

    def on_batch(j, acts):
        for n in names:
            a = C.post_act(arch, acts[n].float()).flatten(2)
            pi = torch.randint(0, a.shape[2], (a.shape[0], n_pos), generator=g).to(a.device)
            v = a.gather(2, pi[:, None, :].expand(-1, a.shape[1], -1)).permute(0, 2, 1).reshape(-1, a.shape[1])
            out[n].append(v.half().cpu())

    C.predict(m, fit, idx=np.arange(n_img), bs=bs, rec=rec, record=names, on_batch=on_batch)
    rec.detach()
    out = {n: torch.cat(v) for n, v in out.items()}
    torch.save(out, fn)
    return out


def cutoffs(arch, A):
    """A [n, C] post-activation samples -> {p: tau [C]} (see module docstring)."""
    A = A.to(C.DEV).float()
    if arch == "googlenet":
        A = torch.where(A > 0, A, torch.full_like(A, float("nan")))
    else:
        A = A.abs()
    q = torch.tensor([p / 100 for p in PCTS], device=C.DEV)
    step = max(1, (1 << 24) // len(A) - 1)                   # torch.quantile's input-size limit
    T = torch.cat([torch.nanquantile(A[:, i:i + step], q, dim=0) for i in range(0, A.shape[1], step)], 1)
    T = torch.nan_to_num(T, nan=float("inf"))                # a never-positive neuron: everything is "weak"
    return {p: T[i] for i, p in enumerate(PCTS)}


def cut(tau):
    def f(a):
        return a * (a.abs() >= tau.view(1, -1, 1, 1).to(a.dtype))
    return f


def cut_strong(tau):
    def f(a):
        return a * (a.abs() < tau.view(1, -1, 1, 1).to(a.dtype))
    return f


def run(dataset, arch, out, save, bs=200):
    fit = FIT[dataset]
    S = fit_samples(fit, arch)
    names = C.block_names(arch)
    taus = {n: cutoffs(arch, S[n]) for n in names}
    m = C.load_model(arch, True)
    rec = C.Recorder(m, arch).attach(names)
    _, y = C.images(dataset)
    base = E.load(dataset, arch, True)
    p0 = base["pred"]
    r = out.setdefault(C.tag(arch, True), dict(fit_on=fit, eval_on=dataset, base_top1=float((p0 == y).mean()),
                                               cells={}, layers=names))

    def cell(key, edit):
        if key in r["cells"]:
            return
        t0 = time.time()
        rec.edit = edit
        p, _ = C.predict(m, dataset, bs=bs, rec=rec)
        rec.edit = {}
        acc = float((p == y).mean())
        r["cells"][key] = dict(top1=acc, delta_pp=100 * (acc - r["base_top1"]), changed=float((p != p0).mean()))
        print(f"[weak] {dataset} {arch} {key}: top-1 {100 * acc:.3f}% ({r['cells'][key]['delta_pp']:+.3f} pp, "
              f"{100 * r['cells'][key]['changed']:.1f}% answers changed) {time.time() - t0:.0f} s", flush=True)
        save()

    # headline cells first: the last layer, then (convnext) all blocks, then mass-matched pairs, then the rest
    order = [names[-1]] + (["all"] if arch != "googlenet" else []) + ["pairs"] + names[-2::-1]
    for n in order:
        if n == "pairs":
            for nm in ([names[-1]] + (["all"] if arch != "googlenet" else [])):
                ls = names if nm == "all" else [nm]
                tw, ts = {}, {}
                for l in ls:
                    a, b, fs = ct.weak_strong_thresholds(S[l].float(), 0.75)
                    tw[l], ts[l] = a.to(C.DEV), b.to(C.DEV)
                    if nm != "all":
                        r.setdefault("strong_frac", {})[l] = float(fs.mean())
                cell(f"{nm}|weak75", {l: cut(tw[l]) for l in ls})
                cell(f"{nm}|strong_mass_matched", {l: cut_strong(ts[l]) for l in ls})
            continue
        for p in PCTS:
            ls = names if n == "all" else [n]
            cell(f"{n}|p{p}", {l: cut(taus[l][p]) for l in ls})
    rec.detach()
    del m
    torch.cuda.empty_cache()


def main(dataset="imagenet_val", archs=("googlenet", "convnext_tiny")):
    fn = os.path.join(C.RESULTS, "imnet_weak.json")
    allout = json.load(open(fn)) if os.path.exists(fn) else {}
    out = allout.setdefault(dataset, {})

    def save():
        json.dump(allout, open(fn, "w"), indent=1)
    for arch in archs:
        run(dataset, arch, out, save)
    save()
    return allout


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["imagenet_val"]))
