"""imnet_ feature visualization of pretrained-net last-layer neurons under the essay's three microscope settings.

Matches the essay's "microscope-timeline" figure (Cammarata 2026, "A History of Microscope Errors"): the same
neurons rendered by 2009-style pixel feature vis, 2017 Fourier + transformation-robust feature vis, and the
essay's fixed settings, to be shown beside their strongest photos (analysis/imnet_gallery.py).

Neurons (per arch, "the last layer" of imnet_common.last_layer): googlenet inception5b, pre-ReLU concat of the
branch BatchNorm outputs (1024 ch); convnext_tiny features.7.2.mlp, pwconv1 output pre-GELU (3072 units).
Objective = the positive pole score: spatial max of the unit's pre-activation on a 224 x 224 image, fp32.

Settings (reusing analysis/vis_featvis.py):
  pixel   "2009-style": raw-pixel sigmoid parameterization, plain gradient ascent, no transforms.
  robust  "2017": Fourier parameterization with 1/f scaling + colour decorrelation, random translate / scale /
          rotate, 4 samples per step.
  path    "path-anchored (OURS)": robust + for the first 64 steps loss = -s(x) - rho * cos(A(x), r_bar), where
          A(x) = sum over positions of a_prev * ds/da_prev on the previous layer (googlenet: inception5a output;
          convnext_tiny: the residual stream entering block features.7.2), r_bar = its mean over the pole's 60
          strongest val images, rho = the pole's activation range (max - median over the 50,000 images).
Adam lr 0.05, 256 steps, FV seed 0; two extra seeds for two of the neurons (seed stability).

Neuron sample per model: the 8 units with the highest top-20 class purity (ties broken by the mean top-20 score),
plus 12 drawn uniformly at random (numpy seed 0) from the remaining units. Pole scores come from
data/imnet_cache/imagenet_val/<arch>_extract.npz (imnet_extract), no recomputation.

Writes reports/figs/imnet_fv/<arch>/<unit>_<setting>[_s<seed>].png (224 px) and reports/results/imnet_featvis.json:
per unit the final objective per setting and the dataset max of the pole score (FV activation / dataset max).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import imnet_common as C  # noqa: E402
from analysis.imnet_purity import modal, top_k  # noqa: E402
from analysis.vis_featvis import fourier_param, pixel_param, robust_transform  # noqa: E402

DEV = C.DEV
SETTINGS = ["pixel", "robust", "path", "tuned"]
LABELS = {"pixel": "2009-style", "robust": "2017", "path": "path-anchored (ours)", "tuned": "tuned (ours)"}
RUN_SETTINGS = SETTINGS[:3]        # rendered here; "tuned" is rendered by imnet_featvis2.py --render
RES, STEPS, PATH_STEPS, N_AUG, TOPK, LR = 224, 256, 64, 4, 60, 0.05
N_PICK, N_RAND, N_SEED_UNITS, EXTRA_SEEDS = 8, 12, 2, (1,)
ARCHS = ("googlenet", "convnext_tiny")
FV_DIR = os.path.join(C.FIGS, "imnet_fv")
PREV = {"googlenet": "inception5a", "convnext_tiny": "features.7.1 -> features.7.2 input (residual stream)"}


class Hooks:
    """Captures the last-layer pre-activation `z` [B, C, H, W] and the previous-layer tensor `prev` (in the graph)."""

    def __init__(self, model, arch):
        self.arch, self.z, self.prev, self.store, self.h = arch, None, None, {}, []
        if arch == "googlenet":
            blk = model.inception5b
            for k, bn in enumerate([blk.branch1.bn, blk.branch2[1].bn, blk.branch3[1].bn, blk.branch4[1].bn]):
                self.h.append(bn.register_forward_hook(self._store(k)))
            self.h.append(blk.register_forward_hook(self._cat))
            self.h.append(model.inception5a.register_forward_hook(self._prev_out))
        else:
            blk = model.features[7][2]
            self.h.append(blk.block[3].register_forward_hook(self._nhwc))
            self.h.append(blk.register_forward_pre_hook(self._prev_in))

    def _store(self, k):
        def f(mod, inp, out):
            self.store[k] = out.clone()        # BasicConv2d applies ReLU in place right after the BN
        return f

    def _cat(self, mod, inp, out):
        self.z = torch.cat([self.store[k] for k in range(4)], 1)

    def _prev_out(self, mod, inp, out):
        self.prev = out

    def _nhwc(self, mod, inp, out):
        self.z = out.permute(0, 3, 1, 2)

    def _prev_in(self, mod, inp):
        self.prev = inp[0]

    def detach(self):
        for h in self.h:
            h.remove()


def norm(x):
    """[B, 3, H, W] in [0, 1] -> the models' normalised input (channels_last)."""
    x = (x - MEAN) / STD
    return x.contiguous(memory_format=torch.channels_last)


MEAN = (C.MEAN / 255).to(DEV)
STD = (C.STD / 255).to(DEV)


def score(z, units):
    """Positive pole score: per row b, max over positions of z[b, units[b]]."""
    return z[torch.arange(len(units), device=DEV), units].flatten(1).amax(1)


def attribution(model, hk, x, units, create_graph):
    """A [B, C_prev] = sum_hw a_prev * ds/da_prev, plus s [B]. x: normalised input requiring grad somewhere."""
    model(x)
    s = score(hk.z, units)
    g, = torch.autograd.grad(s.sum(), hk.prev, create_graph=create_graph)
    return (hk.prev * g).sum((2, 3)), s


def path_targets(model, hk, X, S, units):
    """Mean attribution profile r_bar over each unit's 60 strongest val images (by pole score)."""
    out = []
    for u in units.tolist():
        top = np.sort(np.argsort(-S[:, u].astype(np.float32), kind="stable")[:TOPK])
        x = C.to_input(X[top]).requires_grad_(True)
        A, _ = attribution(model, hk, x, torch.full((len(top),), u, device=DEV), False)
        out.append(A.mean(0).detach())
    return torch.stack(out)


def visualize(model, hk, units, rho, rbar, setting, seed):
    n = len(units)
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    p, img = (pixel_param if setting == "pixel" else fourier_param)(n, RES, gen)
    opt = torch.optim.Adam([p], lr=LR)
    for step in range(STEPS):
        x = img()
        if setting == "pixel":
            xs, uu = x, units
        else:
            xs, uu = robust_transform(x, N_AUG, RES), units.repeat(N_AUG)
        if setting == "path" and step < PATH_STEPS:
            A, s = attribution(model, hk, norm(xs), uu, True)
            A = A.view(-1, n, A.shape[1]).mean(0)
            cos = F.cosine_similarity(A, rbar, dim=1)
            loss = -(s.view(-1, n).mean(0)).sum() - (rho * cos).sum()
        else:
            model(norm(xs))
            s = score(hk.z, uu)
            loss = -s.view(-1, n).mean(0).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = img().detach()
        model(norm(x))
        s = score(hk.z, units)
    return x, s


def select_units(S, y, n_pick=N_PICK, n_rand=N_RAND, seed=0):
    """-> (picked [n_pick], random [n_rand], modal count [U], modal class [U], top-20 idx [U, 20])."""
    T = top_k(S, 20)
    cnt, val = modal(y[T])
    mean_top = np.take_along_axis(S.astype(np.float32), T.T, 0).mean(0)
    order = np.lexsort((-mean_top, -cnt))
    picked = order[:n_pick]
    rng = np.random.default_rng(seed)
    rest = np.setdiff1d(np.arange(S.shape[1]), picked)
    rand = np.sort(rng.choice(rest, n_rand, replace=False))
    return picked, rand, cnt, val, T


def save_png(x, fn):
    from PIL import Image
    a = (x.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(a).save(fn)


def run(arch, nb=None, settings=RUN_SETTINGS):
    torch.backends.cudnn.benchmark = True
    X, y = C.images("imagenet_val")
    _, names = C.class_meta()
    L = C.last_layer(arch)
    S = np.load(os.path.join(C.cache_dir("imagenet_val"), f"{arch}_extract.npz"))[f"{L}_pos"]
    picked, rand, cnt, val, T = select_units(S, y)
    units_all = np.concatenate([picked, rand])
    seed_units = [int(picked[0]), int(rand[0])]
    model = C.load_model(arch)
    for q in model.parameters():
        q.requires_grad_(False)
    hk = Hooks(model, arch)
    out_dir = os.path.join(FV_DIR, arch)
    os.makedirs(out_dir, exist_ok=True)
    Sf = S.astype(np.float32)
    smax = Sf[:, units_all].max(0)
    smed = np.median(Sf[:, units_all], 0)
    res = dict(layer=L, prev_layer=PREV[arch], picked=[int(u) for u in picked], random=[int(u) for u in rand],
               seed_units=seed_units, n_images=int(len(y)), steps=STEPS, path_steps=PATH_STEPS, lr=LR, n_aug=N_AUG,
               topk_path=TOPK, units={}, time_s={})
    for i, u in enumerate(units_all.tolist()):
        res["units"][str(u)] = dict(role="picked" if i < len(picked) else "random", majority_class=names[val[u]],
                                    majority_count=int(cnt[u]), dataset_max=float(smax[i]), dataset_median=float(smed[i]),
                                    rho=float(smax[i] - smed[i]), fv={}, ratio={})
    t0 = time.time()
    units_t = torch.as_tensor(units_all, device=DEV)
    rbar = path_targets(model, hk, X, S, units_t)
    res["time_s"]["path_targets"] = time.time() - t0
    print(f"[featvis] {arch}: {len(units_all)} units, r_bar in {time.time() - t0:.0f} s", flush=True)
    jobs = [(u, 0) for u in units_all.tolist()] + [(u, s) for u in seed_units for s in EXTRA_SEEDS]
    nb = nb or len(jobs)
    for st in settings:
        t1 = time.time()
        for b in range(0, len(jobs), nb):
            chunk = jobs[b:b + nb]
            uu = torch.as_tensor([u for u, _ in chunk], device=DEV)
            pos = [int(np.where(units_all == u)[0][0]) for u, _ in chunk]
            rho = torch.as_tensor(smax[pos] - smed[pos], dtype=torch.float32, device=DEV)
            # one FV seed per batch: group the chunk by seed
            for sd in sorted({s for _, s in chunk}):
                m = [k for k, (_, s) in enumerate(chunk) if s == sd]
                x, s = visualize(model, hk, uu[m], rho[m], rbar[[pos[k] for k in m]], st, 1000 * sd)
                for k, j in enumerate(m):
                    u = chunk[j][0]
                    tag = f"{u}_{st}" + (f"_s{sd}" if sd else "")
                    save_png(x[k], os.path.join(out_dir, tag + ".png"))
                    r = res["units"][str(u)]
                    r["fv"].setdefault(st, {})[f"s{sd}"] = float(s[k])
                    r["ratio"].setdefault(st, {})[f"s{sd}"] = float(s[k] / max(r["dataset_max"], 1e-6))
            torch.cuda.synchronize()
        res["time_s"][st] = time.time() - t1
        ratios = [res["units"][str(u)]["ratio"][st]["s0"] for u in units_all.tolist()]
        print(f"[featvis] {arch} {st}: {time.time() - t1:.0f} s, FV/dataset-max median {np.median(ratios):.2f} "
              f"(min {min(ratios):.2f}, max {max(ratios):.2f}), peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
              flush=True)
    hk.detach()
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=list(ARCHS))
    ap.add_argument("--nb", type=int, default=None, help="neurons per optimisation batch (default: all)")
    a = ap.parse_args()
    fn = os.path.join(C.RESULTS, "imnet_featvis.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for arch in a.archs:
        out[arch] = run(arch, a.nb)
        os.makedirs(C.RESULTS, exist_ok=True)
        json.dump(out, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main()
