"""vis_ feature visualization under three settings, with an objective legibility judge.

Settings (per unit pole of the LAST neuron layer; objective = the pole score, i.e. spatial max of +z or of -z):
  pixel   ("2009"): raw-pixel parameterization (sigmoid), plain gradient ascent, no transforms.
  robust  ("2017"): Fourier parameterization with 1/f scaling + colour decorrelation (Lucid's fft_image /
                    color_correlation_svd_sqrt, as in analysis/video_featvis.py), transformation robustness:
                    per sample random translation (+-res/16 px), scale 0.9-1.1, rotation +-10 deg, 4 samples/step.
  path    ("fixed"): robust + a path-anchoring term for the first 64 steps (essay, appendix "Feature
                    Visualization in Later Layers": "traced the paths they take, and gave the optimization an
                    incentive to follow those paths, often only for the first 64 steps"; "Path-guided
                    visualizations use the strongest 60 examples"). OUR formalization (cammarata_math.md 3.5):
                    loss = -s(x) - rho * cos(A(x), r_bar) for steps < 64, where A(x) = sum over positions of
                    a_prev * d s / d a_prev (gradient x activation on the previous neuron layer), r_bar its mean
                    over the pole's 60 strongest test images, rho = the pole's activation range (max - median).
All: Adam lr 0.05, 256 steps, 3 FV seeds.

Legibility judge (OURS): an independently trained readout (models/vis_train.py --step readout; a ResNet,
different from the studied models) reads the final FV image. A pole's FV is legible if the readout's value of
the pole's job factor equals the job's modal value (job = strong-set purity winner: raw-shape strong set for
the fixed tools, top-1% percentile set for the old tools). Continuous factors are compared in quintiles;
rot_phase has no readout head and is never legible. CIFAR: readout class == the pole's modal class.
Also: FV activation / dataset max, and seed agreement (share of the 3 seeds whose judged value equals the mode).
Writes data/vis_cache/<model>_fv.npz and reports/results/vis_featvis.json.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.vis_common import DATASETS, CACHE, DEV, RESULTS, available, cached_scores, data, load_model, prev_layer, to_float  # noqa: E402
from models.vis_train import load_readout  # noqa: E402

COLOR_CORR = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])
COLOR_CORR = COLOR_CORR / COLOR_CORR.norm(dim=0).max()
SETTINGS = ["pixel", "robust", "path"]
N_UNITS, STEPS, PATH_STEPS, N_AUG, TOPK = 48, 256, 64, 4, 60


def fourier_param(n, size, gen):
    fy = torch.fft.fftfreq(size)[:, None]
    fx = torch.fft.rfftfreq(size)[None]
    scale = 1.0 / torch.clamp(torch.sqrt(fx ** 2 + fy ** 2), min=1.0 / size)
    scale = (scale / scale.max() * size).to(DEV)
    spec = (torch.randn(n, 3, size, size // 2 + 1, 2, generator=gen) * 0.01).to(DEV).requires_grad_(True)
    cc = COLOR_CORR.to(DEV)

    def img():
        x = torch.fft.irfft2(torch.view_as_complex(spec) * scale, s=(size, size), norm="ortho")
        return torch.sigmoid(torch.einsum("nchw,dc->ndhw", x, cc))
    return spec, img


def pixel_param(n, size, gen):
    p = (torch.randn(n, 3, size, size, generator=gen) * 0.01).to(DEV).requires_grad_(True)
    return p, lambda: torch.sigmoid(p)


def robust_transform(x, k, res):
    """k random (translate, scale, rotate) copies of each image, grey padding -> [k*n, 3, H, W]."""
    n = x.shape[0]
    xr = x.repeat(k, 1, 1, 1)
    B = xr.shape[0]
    a = (torch.rand(B, device=DEV) * 2 - 1) * math.radians(10)
    s = 0.9 + 0.2 * torch.rand(B, device=DEV)
    t = (torch.rand(B, 2, device=DEV) * 2 - 1) * (res / 16) * 2 / res
    cos, sin = torch.cos(a) / s, torch.sin(a) / s
    theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1)
    g = F.affine_grid(theta, xr.shape, align_corners=False)
    return F.grid_sample(xr - 0.5, g, padding_mode="zeros", align_corners=False) + 0.5


def pole_score(z, units, signs):
    """z [B, C, H, W]; per row b: max over positions of signs[b] * z[b, units[b]]."""
    zz = z[torch.arange(len(units), device=DEV), units] * signs[:, None, None]
    return zz.flatten(1).amax(1)


def attribution(model, x, layer, prev, units, signs, create_graph):
    """A [B, C_prev] = sum_hw a_prev * d s / d a_prev, plus s [B]."""
    model(x, record=[layer, prev])          # model.keep_graph is True during feature vis
    s = pole_score(model.acts[layer], units, signs)
    a = model.post[prev]
    g, = torch.autograd.grad(s.sum(), a, create_graph=create_graph)
    return (a * g).sum((2, 3)), s


def path_targets(model, D, layer, prev, units, signs, scores):
    """Mean attribution profile over each pole's 60 strongest test images."""
    out = []
    for u, sg in zip(units.tolist(), signs.tolist()):
        S = scores[0][:, u] if sg > 0 else scores[1][:, u]
        top = np.argsort(-S)[:TOPK]
        x = to_float(torch.as_tensor(D["xte"][np.sort(top)], device=DEV)).requires_grad_(True)
        A, _ = attribution(model, x, layer, prev, torch.full((len(top),), u, device=DEV),
                           torch.full((len(top),), float(sg), device=DEV), False)
        out.append(A.mean(0).detach())
    return torch.stack(out)


def visualize(model, D, layer, prev, units, signs, rho, rbar, setting, seed):
    n, res = len(units), D["res"]
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    p, img = (pixel_param if setting == "pixel" else fourier_param)(n, res, gen)
    opt = torch.optim.Adam([p], lr=0.05)
    for step in range(STEPS):
        x = img()
        if setting == "pixel":
            xs, uu, ss = x, units, signs
        else:
            xs, uu, ss = robust_transform(x, N_AUG, res), units.repeat(N_AUG), signs.repeat(N_AUG)
        if setting == "path" and step < PATH_STEPS:
            A, s = attribution(model, xs, layer, prev, uu, ss, True)
            A = A.view(-1, n, A.shape[1]).mean(0)
            cos = F.cosine_similarity(A, rbar, dim=1)
            loss = -(s.view(-1, n).mean(0)).sum() - (rho * cos).sum()
        else:
            model(xs, record=[layer])
            s = pole_score(model.acts[layer], uu, ss)
            loss = -s.view(-1, n).mean(0).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = img().detach()
        model(x, record=[layer])
        s = pole_score(model.acts[layer], units, signs)
    return x, s


@torch.no_grad()
def judge(readout, cont, x, dname):
    """Readout predictions for FV images -> {factor: predicted value} (continuous: raw value)."""
    out = readout(x)
    if dname == "cifar":
        return {"label": out["label"].argmax(1).cpu().numpy()}
    p = {k: out[k].argmax(1).cpu().numpy() for k in ("shape", "color", "texture", "bg_type")}
    for k in ("size", "x", "y", "bg_lum"):
        m, s = cont[k]
        p[k] = out[k][:, 0].float().cpu().numpy() * s + m
    p["curved"] = np.isin(p["shape"], [0, 3]).astype(int)
    p["warm"] = np.isin(p["color"], [0, 1, 5]).astype(int)
    p["textured"] = (p["texture"] > 0).astype(int)
    p["label"] = p["shape"] * 4 + p["warm"] * 2 + p["textured"]
    return p


def judged_value(pred, i, factor, D):
    """Readout's value of `factor` for image i, in the purity code space (categorical index / quintile)."""
    if factor not in pred:
        return -1
    v = pred[factor][i]
    if D["kinds"][factor] == "cat":
        vals = np.unique(np.asarray(D["fte"][factor]))
        return int(np.searchsorted(vals, v))
    edges = np.quantile(np.asarray(D["fte"][factor], np.float64), np.linspace(0, 1, 6))
    return int(np.clip(np.searchsorted(edges, v, side="right") - 1, 0, 4))


def run(dname, tag, seed, init=False, fv_seeds=(0, 1, 2)):
    D = data(dname)
    model = load_model(dname, tag, seed, init)
    for q in model.parameters():
        q.requires_grad_(False)
    model.keep_graph = True
    readout, cont = load_readout(dname)
    layer = model.neuron_layers[-1]
    prev = prev_layer(model.neuron_layers, layer)
    name = f"{dname}_{tag}_s{seed}{'_init' if init else ''}"
    cen = pickle.load(open(os.path.join(CACHE, name + "_census.pkl"), "rb"))["late"]
    scores = cached_scores(dname, tag, seed, init)[layer]
    C = scores[0].shape[1]
    rng = np.random.default_rng(7 + seed)
    sel = np.sort(rng.choice(C, min(N_UNITS, C), replace=False))
    units = torch.as_tensor(np.concatenate([sel, sel]), device=DEV)
    signs = torch.as_tensor(np.r_[np.ones(len(sel)), -np.ones(len(sel))], dtype=torch.float32, device=DEV)
    poles = ["pos"] * len(sel) + ["neg"] * len(sel)
    smax = np.array([scores[0 if p == "pos" else 1][:, u].max() for u, p in zip(units.tolist(), poles)])
    smed = np.array([np.median(scores[0 if p == "pos" else 1][:, u]) for u, p in zip(units.tolist(), poles)])
    rho = torch.as_tensor(smax - smed, dtype=torch.float32, device=DEV)
    rbar = path_targets(model, D, layer, prev, units, signs, scores)
    imgs, acts = {}, {}
    for st in SETTINGS:
        for fs in fv_seeds:
            x, s = visualize(model, D, layer, prev, units, signs, rho, rbar, st, 1000 * fs + seed)
            imgs[(st, fs)] = x.cpu()
            acts[(st, fs)] = s.cpu().numpy()
    # judge
    res = dict(units=sel.tolist(), layer=layer)
    jobs = {}
    for mode in ("raw", "pct"):
        f = [cen[p][f"pur_{mode}"]["factor"][u] for u, p in zip(units.tolist(), poles)]
        v = [int(cen[p][f"pur_{mode}"]["value"][u]) for u, p in zip(units.tolist(), poles)]
        oj = [bool(cen[p][f"pur_{mode}"]["one_job"][u]) for u, p in zip(units.tolist(), poles)]
        jobs[mode] = (f, v, np.array(oj))
    per = {}
    for st in SETTINGS:
        for mode in ("raw", "pct"):
            f, v, oj = jobs[mode]
            match = np.zeros((len(fv_seeds), len(poles)), bool)
            for si, fs in enumerate(fv_seeds):
                pred = judge(readout, cont, imgs[(st, fs)].to(DEV), dname)
                match[si] = [judged_value(pred, i, f[i], D) == v[i] for i in range(len(poles))]
            per[(st, mode)] = match
        ratio = np.stack([acts[(st, fs)] for fs in fv_seeds]) / np.maximum(smax, 1e-6)
        # pixel-space seed spread: mean pairwise correlation of the 3 FV images
        X = torch.stack([imgs[(st, fs)] for fs in fv_seeds]).flatten(2).numpy()       # [S, P, D]
        X = X - X.mean(2, keepdims=True)
        X /= np.linalg.norm(X, axis=2, keepdims=True) + 1e-9
        corr = np.mean([(X[a] * X[b]).sum(1) for a in range(len(fv_seeds)) for b in range(a + 1, len(fv_seeds))], 0)
        res[st] = dict(
            legible_raw=float(per[(st, "raw")].mean()), legible_pct=float(per[(st, "pct")].mean()),
            legible_raw_pos=float(per[(st, "raw")][:, :len(sel)].mean()), legible_pct_pos=float(per[(st, "pct")][:, :len(sel)].mean()),
            seed_agree=float(np.mean(np.abs(per[(st, "raw")].mean(0) - 0.5) * 2)),
            act_ratio_median=float(np.median(ratio)), act_ratio_iqr=[float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
            seed_pixel_corr=float(np.mean(corr)))
    # tool verdicts per pole (FV seed 0), OURS
    oj_raw, oj_pct = jobs["raw"][2], jobs["pct"][2]
    leg = {k: per[k][0] for k in per}
    n = len(sel)
    res["verdict"] = dict(
        # old tools: positive pole only, percentile top-1% purity, raw-pixel FV; negative poles never inspected
        old_tools_poles=float(np.r_[(oj_pct & leg[("pixel", "pct")])[:n], np.zeros(n, bool)].mean()),
        old_tools_pos=float((oj_pct & leg[("pixel", "pct")])[:n].mean()),
        # 2017 FV but otherwise old (positive pole, percentiles)
        robustfv_pos=float((oj_pct & leg[("robust", "pct")])[:n].mean()),
        # fixed tools: both poles, raw-shape purity, path-anchored FV
        fixed_tools_poles=float((oj_raw & leg[("path", "raw")]).mean()),
        fixed_tools_pos=float((oj_raw & leg[("path", "raw")])[:n].mean()),
        fixed_robust_poles=float((oj_raw & leg[("robust", "raw")]).mean()),
        one_job_raw_poles=float(oj_raw.mean()), one_job_pct_pos=float(oj_pct[:n].mean()))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, name + "_fv.npz"), units=units.cpu().numpy(), signs=signs.cpu().numpy(),
                        **{f"{st}_{fs}": imgs[(st, fs)].numpy().astype(np.float16) for st, fs in imgs},
                        **{f"act_{st}_{fs}": acts[(st, fs)] for st, fs in acts}, smax=smax)
    return name, res


def main(tags=("old", "recipe", "arch"), datasets=DATASETS):
    fn = os.path.join(RESULTS, "vis_featvis.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for dname in datasets:
        for tag in tags:
            for seed in available(dname, tag):
                for init in (False, True):
                    name = f"{dname}_{tag}_s{seed}{'_init' if init else ''}"
                    if name in out and os.environ.get("VIS_RECOMPUTE") != "1":
                        continue
                    name, r = run(dname, tag, seed, init, (0,) if init else (0, 1, 2))
                    out[name] = r
                    print(name, {st: (round(r[st]["legible_raw"], 2), round(r[st]["act_ratio_median"], 2)) for st in SETTINGS},
                          r["verdict"], flush=True)
                    json.dump(out, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main()
