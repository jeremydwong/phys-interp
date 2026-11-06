"""imnet_ dictionaries vs neurons on the last layer (essay, "What Dictionaries Learn").

Essay: "On the last layer, 88% of its activity is near-copies of single neurons, and the rest is faint: removing
every dictionary direction that matches no neuron leaves accuracy unchanged, and the few answers it flips were
already near-ties."

Ours (formalizations in clean_tools, OURS):
  vectors   last-layer POST-activation vectors (googlenet: inception5b output, ReLU of the branch BNs; convnext: the
            last block's MLP hidden after GELU) at 2 random positions per image. The SAE is FIT on the other
            dataset's vectors (fit ImageNetV2 -> evaluate ImageNet val, and vice versa), so the removal test is
            held out.
  SAE       clean_tools.train_topk_sae, 4x as many features as neurons, k in {8, 32}, 4,000 Adam steps.
  near copy decoder row d_f with max_i |cos(d_f, e_i)| >= 0.9; share = sum of mean activation of near-copy
            features / sum over all features.
  controls  (i) the same vectors under a random rotation (near copies measured against the rotated neuron axes);
            (ii) the random-init network; (iii) planted: synthetic sparse non-negative data whose true features
            ARE the neuron axes (share should be ~1) and the same data rotated (share should be ~0).
  removal   at the layer, in the forward pass on the eval set, x -> x - sum_{f not near copy} z_f(x) d_f (the
            reconstruction error is kept, so only the non-matching directions' contribution is removed). Report
            top-1 change, number of flipped answers, and the flipped images' baseline top-1/top-2 margin vs all.
            Also splice controls: x -> reconstruction from all features / near copies only.
  secondary ConvNeXt's final residual stream (stage-3 output, 768 ch; the essay's SAE figure is anchored
            "convnext-sae-residual", so its SAE may have been on the residual stream; whether that basis counts as
            "neurons" is the essay's call, not ours).
Writes reports/results/imnet_sae.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import clean_tools as ct  # noqa: E402
from analysis import imnet_common as C  # noqa: E402
from analysis import imnet_extract as E  # noqa: E402

FIT = {"imagenet_val": "imagenetv2", "imagenetv2": "imagenet_val"}


def vectors(dataset, arch, trained, n=100_000, seed=0):
    s = E.load_samples(dataset, arch, trained)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(s), min(n, len(s)), replace=False))
    X = torch.as_tensor(np.asarray(s[idx]), device=C.DEV).float()
    return C.post_act(arch, X)


def planted(d, n=100_000, k=8, seed=0):
    """Sparse non-negative codes on the neuron axes (k active of d, exponential magnitudes, Zipf-ish frequencies)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = 1.0 / torch.arange(1, d + 1).float() ** 0.5
    idx = torch.multinomial(p.expand(n, -1), k, replacement=False, generator=g)
    val = torch.empty(n, k).exponential_(generator=g)
    X = torch.zeros(n, d).scatter_(1, idx, val)
    return X.to(C.DEV)


def removal_fn(sae, drop_mask, mode="remove"):
    """Edit hook on the post-activation [B, C, H, W]. mode 'remove': x - z_drop @ W_drop; 'splice': z_keep @ W + b."""
    W = sae["W_dec"]

    def f(a):
        B, Ch, H, Wd = a.shape
        x = a.permute(0, 2, 3, 1).reshape(-1, Ch).float()
        z = sae["encode"](x)
        if mode == "remove":
            out = x - (z * drop_mask[None].float()) @ W
        else:
            out = (z * (~drop_mask)[None].float()) @ W + sae["b_pre"]
        return out.view(B, H, Wd, Ch).permute(0, 3, 1, 2).to(a.dtype)
    return f


def removal_test(dataset, arch, sae, near, bs=100, layer=None):
    m = C.load_model(arch, True)
    L = layer or C.last_layer(arch)
    rec = C.Recorder(m, arch).attach([L])
    _, y = C.images(dataset)
    base = E.load(dataset, arch, True)
    p0, m0 = base["pred"], base["margin"]
    out = dict(base_top1=float((p0 == y).mean()))
    for nm, drop, mode in (("remove_unmatched", ~near, "remove"), ("splice_all", torch.zeros_like(near), "splice"),
                           ("splice_near_copies_only", ~near, "splice")):
        rec.edit = {L: removal_fn(sae, drop, mode)}
        p, _ = C.predict(m, dataset, bs=bs, rec=rec)
        rec.edit = {}
        fl = p != p0
        out[nm] = dict(top1=float((p == y).mean()), delta_pp=float(100 * ((p == y).mean() - (p0 == y).mean())),
                       flipped=int(fl.sum()), flipped_frac=float(fl.mean()),
                       flipped_median_margin=float(np.median(m0[fl])) if fl.any() else None,
                       all_median_margin=float(np.median(m0)))
    rec.detach()
    del m
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def resid_samples(dataset, arch, trained, n_img=None, n_pos=None, layer="features.7"):
    """ConvNeXt final residual stream (stage-3 output, 768 ch, no nonlinearity): vectors at random positions.
    10,000 images x 10 positions (ImageNetV2) or 50,000 x 2 (val) = 100k vectors, cached as float16."""
    fn = os.path.join(C.cache_dir(dataset), f"{C.tag(arch, trained)}_resid_samples.npy")
    if os.path.exists(fn):
        return np.load(fn)
    _, y = C.images(dataset)
    n_pos = n_pos or max(1, 100_000 // len(y))
    m = C.load_model(arch, trained)
    rec = C.Recorder(m, arch).attach([layer])
    g = torch.Generator(device="cpu").manual_seed(0)
    out = []

    def on_batch(j, acts):
        a = acts[layer].float().flatten(2)
        pi = torch.randint(0, a.shape[2], (a.shape[0], n_pos), generator=g).to(a.device)
        out.append(a.gather(2, pi[:, None, :].expand(-1, a.shape[1], -1)).permute(0, 2, 1)
                   .reshape(-1, a.shape[1]).half().cpu().numpy())

    C.predict(m, dataset, bs=200, rec=rec, record=[layer], on_batch=on_batch)
    rec.detach()
    X = np.concatenate(out)
    np.save(fn, X)
    return X


def run(dataset, arch, trained, ks=(8, 32), steps=4000, resid=False):
    fit = FIT[dataset]
    if resid:
        X = torch.as_tensor(resid_samples(fit, arch, trained), device=C.DEV).float()
    else:
        X = vectors(fit, arch, trained)
    d = X.shape[1]
    res = dict(fit_on=fit, eval_on=dataset, d=d, n_vectors=len(X), n_feat=4 * d)
    Q = torch.as_tensor(ct.random_rotation(d, 1234), dtype=torch.float32, device=C.DEV)
    for k in ks:
        sae = ct.train_topk_sae(X, 4 * d, k, steps=steps, seed=0)
        share, frac, c = ct.near_copy_share(sae["W_dec"], sae["mean_z"])
        r = dict(fve=sae["fve"], near_copy_share=share, near_copy_frac_live=frac,
                 cos_hist=np.histogram(c[sae["mean_z"] > 0], bins=10, range=(0, 1))[0].tolist(),
                 live=int((sae["mean_z"] > 0).sum()))
        sae_r = ct.train_topk_sae(X @ Q.T, 4 * d, k, steps=steps, seed=0)
        r["rotated_control"] = dict(zip(("near_copy_share", "near_copy_frac_live"),
                                        ct.near_copy_share(sae_r["W_dec"], sae_r["mean_z"])[:2]), fve=sae_r["fve"])
        del sae_r
        if trained:
            near = torch.as_tensor(c >= 0.9, device=C.DEV)
            r["removal"] = removal_test(dataset, arch, sae, near, layer="features.7" if resid else None)
        res[f"k{k}"] = r
        print(f"[sae] {dataset} {C.tag(arch, trained)}{'_resid' if resid else ''} k={k}: fve {r['fve']:.3f} near-copy share {share:.3f} "
              f"(rotated {r['rotated_control']['near_copy_share']:.3f})"
              + (f" removal d {r['removal']['remove_unmatched']['delta_pp']:+.3f} pp" if trained else ""), flush=True)
        del sae
        torch.cuda.empty_cache()
    return res


def planted_check(d=1024, k=8, steps=4000):
    X = planted(d)
    Q = torch.as_tensor(ct.random_rotation(d, 99), dtype=torch.float32, device=C.DEV)
    out = {}
    for nm, Z in (("axes", X), ("rotated", X @ Q.T)):
        sae = ct.train_topk_sae(Z, 4 * d, k, steps=steps, seed=0)
        share, frac, _ = ct.near_copy_share(sae["W_dec"], sae["mean_z"])
        # recovery of the planted directions themselves (columns of Q for the rotated version)
        B = torch.eye(d, device=C.DEV) if nm == "axes" else Q.T     # data = X Q^T: planted directions = columns of Q
        _, _, cr = ct.near_copy_share(sae["W_dec"], sae["mean_z"], basis=B)
        out[nm] = dict(fve=sae["fve"], near_copy_share=share, near_copy_frac_live=frac,
                       share_matching_planted=float(sae["mean_z"][cr >= 0.9].sum() / sae["mean_z"].sum()))
        print(f"[sae] planted {nm}: {out[nm]}", flush=True)
    return out


def main(dataset="imagenet_val", archs=("googlenet", "convnext_tiny")):
    fn = os.path.join(C.RESULTS, "imnet_sae.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    out.setdefault(dataset, {})
    if "planted" not in out:
        out["planted"] = planted_check()
    for arch in archs:
        for trained in (True, False):
            tag = C.tag(arch, trained)
            if tag in out[dataset] and os.environ.get("IMNET_RECOMPUTE") != "1":
                continue
            out[dataset][tag] = run(dataset, arch, trained)
            json.dump(out, open(fn, "w"), indent=1)
    for arch in archs:                       # secondary: ConvNeXt final residual stream (stage-3 output)
        if arch == "googlenet":
            continue
        for trained in (True, False):
            tag = C.tag(arch, trained) + "_resid"
            if tag in out[dataset] and os.environ.get("IMNET_RECOMPUTE") != "1":
                continue
            out[dataset][tag] = run(dataset, arch, trained, resid=True)
            json.dump(out, open(fn, "w"), indent=1)
    json.dump(out, open(fn, "w"), indent=1)
    return out


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["imagenet_val"]))


def zero_layer_control(dataset="imagenet_val", archs=("googlenet", "convnext_tiny")):
    """Control for the removal test: zero the whole last-layer post-activation (convnext: the last block's MLP
    hidden, residual bypass intact; googlenet: the inception5b output). Shows how much the layer matters at all."""
    fn = os.path.join(C.RESULTS, "imnet_sae.json")
    _, y = C.images(dataset)
    res = {}
    for arch in archs:
        m = C.load_model(arch, True)
        L = C.last_layer(arch)
        rec = C.Recorder(m, arch).attach([L])
        rec.edit = {L: lambda a: a * 0}
        p, _ = C.predict(m, dataset, rec=rec)
        rec.detach()
        base = E.load(dataset, arch, True)["pred"]
        res[arch] = dict(top1=float((p == y).mean()), delta_pp=float(100 * ((p == y).mean() - (base == y).mean())),
                         changed=float((p != base).mean()))
        print(f"[sae] zero-layer control {arch}: {res[arch]}", flush=True)
    out = json.load(open(fn))
    out.setdefault(dataset, {})["zero_layer_control"] = res
    json.dump(out, open(fn, "w"), indent=1)
