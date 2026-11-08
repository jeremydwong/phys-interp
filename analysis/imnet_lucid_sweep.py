"""imnet_ microscope sweep on the essay's own net (Lucid's 2014 InceptionV1 via lucent), mixed5b.

Question: starting from the 2017 look (lucent defaults, imnet_lucid.py), do any of the essay's 2026 "tweaks" (longer
optimisation, higher resolution, path anchoring, total variation) make the renders look like the essay's 2026 column
(one clean object, less clutter)? 12 units: the 8 purity-picked units of imnet_lucid.json plus 4 of its random ones.

Variants (all: Fourier 1/f image + colour decorrelation, lucent's standard transforms = pad 12 grey / jitter 8 /
scale 0.9..1.1 / rotate +-10 deg / jitter 4, then bilinear upsample to 224 when the image is smaller, Adam lr 0.05;
objective = spatial mean of the unit's pre-ReLU branch channel):
  default    128 px, 512 steps: the existing imnet_lucid renders (reports/figs/imnet_fv/inceptionv1_lucid), not re-rendered.
  long       128 px, 2560 steps (lucent render_vis).
  hires      224 px image, 1024 steps (lucent render_vis; no upsample, the model sees the 248 px padded image).
  path       default + the essay's path anchoring for the first 64 steps (own loop that copies lucent's transform
             composition): loss = -s(x) - rho * cos(A(x), r_bar), A(x) = sum_hw a_prev * ds/da_prev on the mixed5a
             output (post-ReLU concat, 832 ch), r_bar = mean A over the unit's 60 strongest val images (by mixed5b_pos),
             rho = dataset max - dataset median of the unit's pole score.
  path_long  path, 2560 steps in total (anchor still only the first 64).
  tv         default + a total-variation penalty on the 224 px (upsampled) image; weight per unit = TV_FRAC x
             s(default render) / TV(default render), i.e. the penalty is 10% of the objective at the default solution
             (at step 0 both s and TV are ~0, so "10% at the start" is calibrated on the default render instead; the
             step-0 ratio is recorded in the json).
Judges of each render (rank of the unit's top-20 majority class): the net itself through the recovered class map,
ConvNeXt-T, torchvision GoogLeNet (both the ImageNet-normalised 224 px render).
Writes reports/figs/imnet_fv2/inceptionv1_lucid/<unit>_<variant>.png (224 px, bicubic upsample of the 128 px renders),
reports/results/imnet_lucid_sweep.json (per unit and variant: objective on the saved 224 px render, objective /
dataset max, the three ranks; per variant: wall time, peak GPU memory) and reports/figs/imnet_lucid_sweep.jpg.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import imnet_common as C  # noqa: E402
from analysis.imnet_featvis2 import total_variation  # noqa: E402
from analysis.imnet_lucid import (FV_DIR as DEFAULT_DIR, LAYER, LR, TAG, _font, branch_of, judge_ranks,  # noqa: E402
                                  load_lucid, own_ranks, render_units)

warnings.filterwarnings("ignore")
DEV = C.DEV
UNITS = [852, 860, 670, 105, 59, 894, 813, 1013, 41, 273, 642, 934]
N_PICKED = 8
VARIANTS = ["default", "long", "hires", "path", "path_long", "tv"]
CFG = {
    "default":   dict(size=128, steps=512),
    "long":      dict(size=128, steps=2560),
    "hires":     dict(size=224, steps=1024),
    "path":      dict(size=128, steps=512, path=True),
    "path_long": dict(size=128, steps=2560, path=True),
    "tv":        dict(size=128, steps=512, tv=True),
}
PATH_STEPS, TOPK, TV_FRAC, NB = 64, 60, 0.10, 8
PREV = "mixed5a"
OUT_DIR = os.path.join(C.FIGS, "imnet_fv2", TAG)
JSON = os.path.join(C.RESULTS, "imnet_lucid_sweep.json")
SHEET = os.path.join(C.FIGS, "imnet_lucid_sweep.jpg")


# ------------------------------------------------------------------ hooks and objective

class Hooks:
    """Captures the four mixed5b pre-ReLU branch convs and the mixed5a output (in the graph)."""

    def __init__(self, model):
        from analysis.imnet_lucid import BRANCHES
        self.store, self.prev, self.h = {}, None, []
        for n, _ in BRANCHES:
            self.h.append(getattr(model, n).register_forward_hook(lambda m, i, o, n=n: self.store.__setitem__(n, o)))
        self.h.append(getattr(model, PREV).register_forward_hook(lambda m, i, o: setattr(self, "prev", o)))

    def score(self, pairs):
        """pairs[i] = (branch module name, channel) for image i -> spatial mean per image [B]."""
        return torch.stack([self.store[n][i, c].mean() for i, (n, c) in enumerate(pairs)])

    def detach(self):
        for h in self.h:
            h.remove()


def lucent_transform(size):
    """lucent render_vis's transform chain for this model: standard transforms, x*255-117, upsample to 224 if < 224."""
    from lucent.optvis import transform
    ts = transform.standard_transforms.copy() + [transform.preprocess_inceptionv1()]
    if size < 224:
        ts.append(torch.nn.Upsample(size=224, mode="bilinear", align_corners=True))
    return transform.compose(ts)


def to224(x):
    return x if x.shape[-1] == 224 else F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=True)


def attribution(model, hk, x, pairs, create_graph):
    """A [B, 832] = sum_hw a_prev * ds/da_prev at mixed5a, and s [B]; x is the model input (already preprocessed)."""
    model(x)
    s = hk.score(pairs)
    g, = torch.autograd.grad(s.sum(), hk.prev, create_graph=create_graph)
    return (hk.prev * g).sum((2, 3)), s


def path_targets(model, hk, X, S, units):
    """r_bar per unit: mean attribution over its 60 strongest val images (pole score mixed5b_pos)."""
    out = []
    for u in units:
        top = np.sort(np.argsort(-S[:, u], kind="stable")[:TOPK])
        A = []
        for b in range(0, TOPK, 20):
            x = torch.from_numpy(np.ascontiguousarray(X[top[b:b + 20]])).to(DEV).permute(0, 3, 1, 2).float() - 117.0
            x.requires_grad_(True)
            a, _ = attribution(model, hk, x, [branch_of(u)] * len(x), False)
            A.append(a.detach())
        out.append(torch.cat(A).mean(0))
    return torch.stack(out)


# ------------------------------------------------------------------ own optimisation loop (path / tv)

def render_own(model, hk, units, cfg, rho=None, rbar=None, tv_w=None, seed=0):
    """lucent's recipe re-implemented step by step so the loss can carry the path anchor or a TV term.
    Returns (PIL images at the param size, step-0 diagnostics)."""
    from lucent.optvis import param
    n, size, steps = len(units), cfg["size"], cfg["steps"]
    pairs = [branch_of(u) for u in units]
    torch.manual_seed(seed)
    np.random.seed(seed)
    params, image_f = param.image(size, batch=n)
    opt = torch.optim.Adam(params, lr=LR)
    tf = lucent_transform(size)
    diag = {}
    for step in range(steps):
        opt.zero_grad()
        x = image_f()
        xs = tf(x)
        if cfg.get("path") and step < PATH_STEPS:
            A, s = attribution(model, hk, xs, pairs, True)
            cos = F.cosine_similarity(A, rbar, dim=1)
            loss = -s.sum() - (rho * cos).sum()
            if step == 0:
                diag["cos0"] = cos.detach().cpu().tolist()
        else:
            model(xs)
            s = hk.score(pairs)
            loss = -s.sum()
        if cfg.get("tv"):
            tv = total_variation(to224(x))
            loss = loss + (tv_w * tv).sum()
            if step == 0:
                diag["tv0"] = tv.detach().cpu().tolist()
        if step == 0:
            diag["s0"] = s.detach().cpu().tolist()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ims = image_f().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray((im * 255).astype(np.uint8)) for im in ims], diag


# ------------------------------------------------------------------ measurement

@torch.no_grad()
def objective_on(model, hk, ims, units):
    """Spatial mean of the unit's pre-ReLU channel on the saved 224 px render, no transforms."""
    x = torch.stack([torch.from_numpy(np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC))).permute(2, 0, 1).float()
                     for im in ims]).to(DEV) - 117.0
    model(x)
    return hk.score([branch_of(u) for u in units]).cpu().numpy()


@torch.no_grad()
def tv_on(ims):
    x = torch.stack([torch.from_numpy(np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC))).permute(2, 0, 1).float() / 255
                     for im in ims]).to(DEV)
    return total_variation(x).cpu().numpy()


def gmean(v):
    v = [r for r in v if r is not None]
    return float(np.exp(np.mean(np.log(v)))) if v else None


# ------------------------------------------------------------------ contact sheet

def contact_sheet(res, variants, short, out, T=224, gap=6):
    units = res["units_order"]
    f, fb, fs = _font(18), _font(22, True), _font(17)
    W = max(len(variants) * (T + gap) + 10, 1400)
    H = 50 + len(units) * (T + 56)
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for j, v in enumerate(variants):
        d.text((j * (T + gap), 8), v, font=fb, fill=(40, 40, 40))
    for i, u in enumerate(units):
        r = res["units"][str(u)]
        y0 = 50 + i * (T + 56)
        d.text((0, y0), f"unit {u} ({r['role']}) · {short[r['majority']]} {r['count']}/20 · cell label = rank of this class: own net / ConvNeXt-T / GoogLeNet",
               font=f, fill=(60, 60, 60))
        y0 += 26
        for j, v in enumerate(variants):
            if v not in r["variants"]:
                continue
            rv = r["variants"][v]
            p = Image.open(os.path.join(OUT_DIR, f"{u}_{v}.png")).convert("RGB").resize((T, T))
            dd = ImageDraw.Draw(p)
            txt = f"{rv['rank_own']} / {rv['rank_convnext']} / {rv['rank_googlenet']}"
            dd.rectangle([4, T - 28, 12 + dd.textlength(txt, font=fs), T - 4], fill=(20, 24, 30))
            dd.text((7, T - 26), txt, font=fs, fill="white")
            im.paste(p, (j * (T + gap), y0))
    im.save(out, quality=90)
    print("wrote", out, im.size, flush=True)


# ------------------------------------------------------------------ main

def main(variants, units=UNITS, force=False):
    torch.backends.cudnn.benchmark = True
    os.makedirs(OUT_DIR, exist_ok=True)
    J = json.load(open(os.path.join(C.RESULTS, "imnet_lucid.json")))
    z = np.load(os.path.join(C.cache_dir("imagenet_val"), f"{TAG}_extract.npz"))
    S, m1008 = z[f"{LAYER}_pos"].astype(np.float32), z["map1008"]
    X, _ = C.images("imagenet_val")
    _, names = C.class_meta()
    short = [str(n).split(",")[0] for n in names]
    model = load_lucid()
    for q in model.parameters():
        q.requires_grad_(False)
    hk = Hooks(model)
    res = json.load(open(JSON)) if os.path.exists(JSON) and not force else {}
    res.update(tag=TAG, layer=LAYER, prev_layer=f"{PREV} (post-ReLU concat, 832 ch)", units_order=[int(u) for u in UNITS],
               picked=[int(u) for u in UNITS[:N_PICKED]], random=[int(u) for u in UNITS[N_PICKED:]],
               cfg={k: v for k, v in CFG.items()}, lr=LR, path_steps=PATH_STEPS, topk_path=TOPK, tv_frac=TV_FRAC, batch=NB,
               objective="spatial mean of the pre-ReLU branch channel, measured on the saved 224 px render without transforms")
    res.setdefault("units", {})
    res.setdefault("stages", {})
    for u in units:
        r = res["units"].setdefault(str(u), {})
        r.update(role="picked" if u in UNITS[:N_PICKED] else "random", majority=int(J["units"][str(u)]["majority"]),
                 count=int(J["units"][str(u)]["count"]), branch=list(branch_of(u)), dataset_max=float(S[:, u].max()),
                 dataset_median=float(np.median(S[:, u])), rho=float(S[:, u].max() - np.median(S[:, u])))
        r.setdefault("variants", {})
    classes = [res["units"][str(u)]["majority"] for u in units]
    smax = np.array([res["units"][str(u)]["dataset_max"] for u in units])
    rho = torch.tensor([res["units"][str(u)]["rho"] for u in units], device=DEV)
    # judges
    convnext = C.load_model("convnext_tiny").to(memory_format=torch.contiguous_format)
    googlenet = C.load_model("googlenet").to(memory_format=torch.contiguous_format)
    judges = {"own": lambda ims: own_ranks(model, ims, classes, m1008),
              "convnext": lambda ims: judge_ranks(convnext, ims, classes),
              "googlenet": lambda ims: judge_ranks(googlenet, ims, classes)}
    # path targets
    rbar = None
    if any(CFG[v].get("path") for v in variants):
        t0 = time.time()
        rbar = path_targets(model, hk, X, S, units)
        res["stages"]["path_targets"] = dict(time_s=time.time() - t0)
        print(f"r_bar for {len(units)} units in {time.time() - t0:.0f} s", flush=True)
        torch.cuda.empty_cache()
    # default renders: the existing ones, for the TV weight too
    default_ims = [Image.open(os.path.join(DEFAULT_DIR, f"{u}_lucid.png")).convert("RGB") for u in units]
    s_def = objective_on(model, hk, default_ims, units)
    tv_def = tv_on(default_ims)
    tv_w = torch.tensor(TV_FRAC * s_def / tv_def, dtype=torch.float32, device=DEV)
    res["tv_weight"] = dict(per_unit=tv_w.cpu().tolist(), rule=f"{TV_FRAC} * s(default render) / TV(default render)",
                            s_default=s_def.tolist(), tv_default=tv_def.tolist())
    for v in variants:
        cfg = CFG[v]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        t0 = time.time()
        ims, diag = [], {}
        if v == "default":
            ims = default_ims
        else:
            for b in range(0, len(units), NB):
                chunk = units[b:b + NB]
                sl = slice(b, b + len(chunk))
                if cfg.get("path") or cfg.get("tv"):
                    out, dg = render_own(model, hk, chunk, cfg, rho=rho[sl], rbar=None if rbar is None else rbar[sl], tv_w=tv_w[sl])
                    for k, val in dg.items():
                        diag.setdefault(k, []).extend(val)
                else:
                    out, _ = render_units(model, [branch_of(u) for u in chunk], seed=0, size=cfg["size"], steps=cfg["steps"])
                ims.extend(out)
                print(f"  {v}: rendered {chunk} ({time.time() - t0:.0f} s so far)", flush=True)
        ims = [im.resize((224, 224), Image.BICUBIC) if im.size != (224, 224) else im for im in ims]
        for u, im in zip(units, ims):
            im.save(os.path.join(OUT_DIR, f"{u}_{v}.png"))
        torch.cuda.synchronize()
        t_render = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        s = objective_on(model, hk, ims, units)
        ranks = {k: f(ims) for k, f in judges.items()}
        for i, u in enumerate(units):
            res["units"][str(u)]["variants"][v] = dict(
                objective=float(s[i]), ratio=float(s[i] / smax[i]), rank_own=ranks["own"][i],
                rank_convnext=ranks["convnext"][i], rank_googlenet=ranks["googlenet"][i],
                **{k: val[i] for k, val in diag.items()})
        old = res["stages"].get(v, {})
        stage = dict(time_s=t_render + (old.get("time_s", 0.0) if set(units) != set(UNITS) else 0.0),
                     peak_gpu_gb=max(peak, old.get("peak_gpu_gb", 0.0)), steps=cfg["steps"], size=cfg["size"])
        # summaries over every unit of UNITS that has this variant (so subsets rendered in separate calls add up)
        for k in judges:
            for role, grp in (("picked", UNITS[:N_PICKED]), ("random", UNITS[N_PICKED:])):
                stage[f"gmean_rank_{k}_{role}"] = gmean([res["units"][str(u)]["variants"][v][f"rank_{k}"] for u in grp
                                                        if str(u) in res["units"] and v in res["units"][str(u)]["variants"]])
        if "tv0" in diag:
            stage["tv_penalty_over_objective_step0"] = float(np.mean(np.array(tv_w.cpu()) * np.array(diag["tv0"]) / np.maximum(np.abs(diag["s0"]), 1e-6)))
        res["stages"][v] = stage
        fmt = lambda x: "-" if x is None else f"{x:.1f}"  # noqa: E731
        print(f"[{v}] {t_render:.0f} s, peak {peak:.2f} GB, ratio median {np.median(s / smax):.2f}; gmean rank picked/random: " +
              ", ".join(f"{k} {fmt(stage[f'gmean_rank_{k}_picked'])}/{fmt(stage[f'gmean_rank_{k}_random'])}" for k in judges), flush=True)
        json.dump(res, open(JSON, "w"), indent=1)
        torch.cuda.empty_cache()
    done = [v for v in VARIANTS if all(str(u) in res["units"] and v in res["units"][str(u)]["variants"] for u in UNITS)]
    if done:
        contact_sheet(res, done, short, SHEET)
    hk.detach()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--units", nargs="*", type=int, default=UNITS)
    ap.add_argument("--force", action="store_true", help="start the json afresh")
    a = ap.parse_args()
    main(a.variants, a.units, a.force)
