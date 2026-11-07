"""imnet_ feature-visualization microscope tuning on the pretrained nets (follow-up to imnet_featvis.py).

Our first renders (imnet_vis.md, observation 3) came out as textured fragments where the essay's come out as
objects. This sweep tries, on the same units, the settings that differ between our implementation and the usual
good ones, and judges every render the same way: an independent ImageNet classifier (the OTHER pretrained net)
reads the render, and we record the rank of the unit's majority class (top-20 photos) in its prediction, plus
the unit's own net as a second, non-independent judge. A render "reads as the unit" if the independent judge
ranks the majority class in its top 5.

Variants (all positive pole of the last layer, same hooks as imnet_featvis.py, 224 px model input):
  robust      the imnet_featvis.py "2017" setting (objective = spatial MAX of the pre-activation, 4 augmented
              samples, 256 steps): the baseline whose renders we are unhappy with.
  mean        same, but objective = spatial MEAN of the pre-activation (Lucid's "channel" objective) and the
              Lucid-sized transforms (jitter 16 px, scale 0.9-1.1, rotate +-10 deg), 8 samples, 512 steps.
  center      objective = the centre cell of the 7 x 7 map (Lucid's "neuron" objective), same transforms.
  center_tv   center + total-variation penalty (Lucid's total_variation regularizer), weight per unit
              0.3 x its dataset max per unit of mean |grad x| (about 3% of the objective at TV 0.1).
  maco        magnitude-constrained optimisation (Fel et al. 2023): the Fourier MAGNITUDE is fixed to the mean
              magnitude spectrum of 2,000 ImageNet val images and only the PHASE is optimised; 6 random crops
              (40-60% of a 448 px canvas, resized to 224) plus Gaussian noise per step, objective = spatial
              mean, NAdam lr 1.0, 512 steps. Published as the fix for feature vis on modern nets.
  maco_path   maco + our path anchor for the first 64 steps (as imnet_featvis.py "path").
  mean_lowf   mean with a stronger low-frequency prior (Fourier scale 1/f^1.5 instead of 1/f) and larger
              transformations (jitter 24 px, scale 0.75-1.25): our guess at the essay's "regularization tweaks".
  maco_long   maco for 1,000 steps (the paper's default length).
  maco_lowf   maco_long with an extra 1/sqrt(f) on the magnitude and a grey start (the three colour channels
              share one random phase, plus 0.1 rad of per-channel noise), to cut the rainbow noise of random
              phases.
  maco_dc     maco_lowf with the phases optimised in Lucid's decorrelated colour space (the magnitude spectrum is
              measured there too), so colour noise is not independent per RGB channel.
  maco_dc1    maco_dc with 1/f on the magnitude instead of 1/sqrt(f).
Writes reports/figs/imnet_fv2/<arch>/<unit>_<variant>.png, reports/figs/imnet_fv2_<arch>.png (contact sheet:
units x variants, then the 3 strongest photos) and reports/results/imnet_featvis2.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import imnet_common as C  # noqa: E402
from analysis.imnet_featvis import Hooks, attribution, norm, path_targets, save_png  # noqa: E402
from analysis.vis_featvis import COLOR_CORR  # noqa: E402

DEV = C.DEV
RES = 224
VARIANTS = ["robust", "mean", "center", "center_tv", "mean_lowf", "maco", "maco_path", "maco_long", "maco_lowf", "maco_dc", "maco_dc1"]
# per-variant knobs: objective kind, samples per step, steps, Fourier decay power (Lucid: 1), jitter px, scale range,
# extra 1/f^p on the MACO magnitude, MACO phase init ("rand" = independent per channel, "grey" = shared + 0.1 rad)
CFG = {
    "robust":    dict(kind="max",    k=4, steps=256,  decay=1.0, jitter=14, scale=(0.9, 1.1)),
    "mean":      dict(kind="mean",   k=8, steps=512,  decay=1.0, jitter=16, scale=(0.9, 1.1)),
    "center":    dict(kind="center", k=8, steps=512,  decay=1.0, jitter=16, scale=(0.9, 1.1)),
    "center_tv": dict(kind="center", k=8, steps=512,  decay=1.0, jitter=16, scale=(0.9, 1.1), tv=True),
    "mean_lowf": dict(kind="mean",   k=8, steps=512,  decay=1.5, jitter=24, scale=(0.75, 1.25)),
    "maco":      dict(kind="mean",   k=6, steps=512,  lowf=0.0, init="rand"),
    "maco_path": dict(kind="mean",   k=6, steps=512,  lowf=0.0, init="rand", path=True),
    "maco_long": dict(kind="mean",   k=6, steps=1000, lowf=0.0, init="rand"),
    "maco_lowf": dict(kind="mean",   k=6, steps=1000, lowf=0.5, init="grey"),
    "maco_dc":   dict(kind="mean",   k=6, steps=1000, lowf=0.5, init="grey", dc=True),
    "maco_dc1":  dict(kind="mean",   k=6, steps=1000, lowf=1.0, init="grey", dc=True),
}
UNITS = {"googlenet": [748, 502, 476, 41, 177, 16], "convnext_tiny": [312, 702, 552, 945, 50, 2803]}
OTHER = {"googlenet": "convnext_tiny", "convnext_tiny": "googlenet"}
FV_DIR = os.path.join(C.FIGS, "imnet_fv2")
TV_W = 0.3
MACO_CANVAS, MACO_BOX, MACO_CROPS, MACO_NOISE, MACO_LR = 448, (0.40, 0.60), 6, 0.05, 1.0
MEAN = (C.MEAN / 255).to(DEV)
STD = (C.STD / 255).to(DEV)


# ------------------------------------------------------------------ objectives and transforms

def objective(z, units, kind):
    zz = z[torch.arange(len(units), device=DEV), units]           # [B, H, W]
    if kind == "max":
        return zz.flatten(1).amax(1)
    if kind == "mean":
        return zz.flatten(1).mean(1)
    h, w = zz.shape[1] // 2, zz.shape[2] // 2
    return zz[:, h, w]


def lucid_transform(x, k, res, jitter=16, scale=(0.9, 1.1), rot=10):
    """k random (translate, scale, rotate) copies, grey padding: Lucid's standard_transforms as one affine."""
    n = x.shape[0]
    xr = x.repeat(k, 1, 1, 1)
    B = xr.shape[0]
    a = (torch.rand(B, device=DEV) * 2 - 1) * math.radians(rot)
    s = scale[0] + (scale[1] - scale[0]) * torch.rand(B, device=DEV)
    t = (torch.rand(B, 2, device=DEV) * 2 - 1) * jitter * 2 / res
    cos, sin = torch.cos(a) / s, torch.sin(a) / s
    theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1)
    g = F.affine_grid(theta, xr.shape, align_corners=False)
    return F.grid_sample(xr - 0.5, g, padding_mode="zeros", align_corners=False) + 0.5


def total_variation(x):
    """Mean absolute finite difference per image [B]."""
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().flatten(1).mean(1) + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().flatten(1).mean(1)


def fourier_param(n, size, gen, decay=1.0):
    """Lucid fft_image + colour decorrelation (vis_featvis.fourier_param) with a settable 1/f^decay."""
    fy = torch.fft.fftfreq(size)[:, None]
    fx = torch.fft.rfftfreq(size)[None]
    scale = 1.0 / torch.clamp(torch.sqrt(fx ** 2 + fy ** 2), min=1.0 / size) ** decay
    scale = (scale / scale.max() * size).to(DEV)
    spec = (torch.randn(n, 3, size, size // 2 + 1, 2, generator=gen) * 0.01).to(DEV).requires_grad_(True)
    cc = COLOR_CORR.to(DEV)

    def img():
        x = torch.fft.irfft2(torch.view_as_complex(spec) * scale, s=(size, size), norm="ortho")
        return torch.sigmoid(torch.einsum("nchw,dc->ndhw", x, cc))
    return spec, img


# ------------------------------------------------------------------ MACO

def natural_magnitude(canvas, n=2000, seed=0, dc=False):
    """Mean |rfft2| of n val images (model-normalised, 224 px; dc: in Lucid's decorrelated colour space),
    resampled onto the canvas's frequency grid."""
    X, _ = C.images("imagenet_val")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), n, replace=False))
    acc = None
    cc_inv = torch.linalg.inv(COLOR_CORR).to(DEV)
    for b in range(0, n, 200):
        x = C.to_input(X[idx[b:b + 200]]).float()
        if dc:
            x = torch.einsum("ndhw,cd->nchw", x, cc_inv)
        m = torch.fft.rfft2(x, norm="ortho").abs().mean(0)                  # [3, 224, 113]
        acc = m if acc is None else acc + m
    m224 = acc / math.ceil(n / 200)
    # resample onto the canvas grid in normalised frequency (fftshift along the y axis first)
    m224 = torch.fft.fftshift(m224, dim=1)                                  # fy from -0.5 to 0.5
    m = F.interpolate(m224[None], size=(canvas, canvas // 2 + 1), mode="bilinear", align_corners=True)[0]
    return torch.fft.ifftshift(m, dim=1).contiguous()


def maco_param(n, canvas, mag, gen, lowf=0.0, init="rand", dc=False):
    """Phase-only Fourier parameterisation with a fixed natural magnitude; image in model-normalised units."""
    if lowf:
        fy = torch.fft.fftfreq(canvas)[:, None]
        fx = torch.fft.rfftfreq(canvas)[None]
        mag = mag * (1.0 / torch.clamp(torch.sqrt(fx ** 2 + fy ** 2), min=1.0 / canvas) ** lowf).to(mag.device)
    if init == "grey":
        ph = (torch.rand(n, 1, canvas, canvas // 2 + 1, generator=gen) * 2 - 1) * math.pi
        ph = ph.repeat(1, 3, 1, 1) + 0.1 * torch.randn(n, 3, canvas, canvas // 2 + 1, generator=gen)
    else:
        ph = (torch.rand(n, 3, canvas, canvas // 2 + 1, generator=gen) * 2 - 1) * math.pi
    phase = ph.to(DEV).requires_grad_(True)

    cc = COLOR_CORR.to(DEV)

    def img():
        spec = mag[None] * torch.exp(1j * phase)
        x = torch.fft.irfft2(spec, s=(canvas, canvas), norm="ortho")
        if dc:
            x = torch.einsum("nchw,dc->ndhw", x, cc)
        x = x - x.mean((1, 2, 3), keepdim=True)
        x = x / (x.flatten(1).std(1)[:, None, None, None] + 1e-5)
        return x.clamp(-2.5, 2.5)
    return phase, img


def maco_crops(x, k, res, box, noise):
    """k random crops (side = box fraction of the canvas) per image, resized to res, plus Gaussian noise."""
    n, canvas = x.shape[0], x.shape[-1]
    xr = x.repeat(k, 1, 1, 1)
    B = xr.shape[0]
    s = box[0] + (box[1] - box[0]) * torch.rand(B, device=DEV)                 # crop side as a fraction
    c = (torch.rand(B, 2, device=DEV) * 2 - 1) * (1 - s)[:, None]              # centre in [-1+s, 1-s]
    theta = torch.zeros(B, 2, 3, device=DEV)
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, :, 2] = c
    g = F.affine_grid(theta, (B, 3, res, res), align_corners=False)
    out = F.grid_sample(xr, g, padding_mode="reflection", align_corners=False)
    return out + noise * torch.randn_like(out)


def maco_to_pixels(x):
    """Model-normalised canvas -> [0, 1] RGB."""
    return (x * STD + MEAN).clamp(0, 1)


# ------------------------------------------------------------------ one optimisation

def visualize(model, hk, units, variant, rho=None, rbar=None, smax=None, seed=0):
    n, cfg = len(units), CFG[variant]
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    k, steps, kind = cfg["k"], cfg["steps"], cfg["kind"]
    if variant.startswith("maco"):
        dc = cfg.get("dc", False)
        p, img = maco_param(n, MACO_CANVAS, visualize.mag_dc if dc else visualize.mag, gen, cfg["lowf"], cfg["init"], dc)
        opt = torch.optim.NAdam([p], lr=MACO_LR)
    else:
        p, img = fourier_param(n, RES, gen, cfg["decay"])
        opt = torch.optim.Adam([p], lr=0.05)
    uu = units.repeat(k)
    for step in range(steps):
        x = img()
        if variant.startswith("maco"):
            xs = maco_crops(x, k, RES, MACO_BOX, MACO_NOISE).contiguous(memory_format=torch.channels_last)
        elif variant == "robust":
            from analysis.vis_featvis import robust_transform
            xs = norm(robust_transform(x, k, RES))
        else:
            xs = norm(lucid_transform(x, k, RES, cfg["jitter"], cfg["scale"]))
        if cfg.get("path") and step < 64:
            model(xs)
            s = objective(hk.z, uu, kind)
            g, = torch.autograd.grad(s.sum(), hk.prev, create_graph=True)
            A = (hk.prev * g).sum((2, 3)).view(k, n, -1).mean(0)
            cos = F.cosine_similarity(A, rbar, dim=1)
            loss = -s.view(k, n).mean(0).sum() - (rho * cos).sum()
        else:
            model(xs)
            s = objective(hk.z, uu, kind)
            loss = -s.view(k, n).mean(0).sum()
        if cfg.get("tv"):
            loss = loss + (TV_W * smax * total_variation(x)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = img().detach()
        if variant.startswith("maco"):
            x = maco_to_pixels(x)
            x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False, antialias=True)
        model(norm(x))
        out = {kd: objective(hk.z, units, kd).cpu().numpy() for kd in ("max", "mean", "center")}
    return x, out


@torch.no_grad()
def judge(model, x, classes):
    """Rank (1 = top) of each unit's majority class in the model's prediction for its render."""
    logits = model(norm(x))
    if hasattr(logits, "logits"):
        logits = logits.logits
    order = logits.argsort(1, descending=True)
    return [int((order[i] == c).nonzero()[0, 0]) + 1 for i, c in enumerate(classes)]


def run(arch, variants):
    torch.backends.cudnn.benchmark = True
    J = json.load(open(os.path.join(C.RESULTS, "imnet_featvis.json")))[arch]
    _, names = C.class_meta()
    X, y = C.images("imagenet_val")
    L = C.last_layer(arch)
    S = np.load(os.path.join(C.cache_dir("imagenet_val"), f"{arch}_extract.npz"))[f"{L}_pos"]
    units = UNITS[arch]
    units_t = torch.as_tensor(units, device=DEV)
    classes = []
    for u in units:
        top = np.argsort(-S[:, u].astype(np.float32), kind="stable")[:20]
        v, c = np.unique(y[top], return_counts=True)
        classes.append(int(v[np.argmax(c)]))
    model = C.load_model(arch)
    other = C.load_model(OTHER[arch])
    for q in list(model.parameters()) + list(other.parameters()):
        q.requires_grad_(False)
    hk = Hooks(model, arch)
    Sf = S[:, units].astype(np.float32)
    smax = torch.as_tensor(Sf.max(0), device=DEV)
    rho = torch.as_tensor(Sf.max(0) - np.median(Sf, 0), device=DEV)
    rbar = path_targets(model, hk, X, S, units_t)
    visualize.mag = natural_magnitude(MACO_CANVAS).to(DEV)
    visualize.mag_dc = natural_magnitude(MACO_CANVAS, dc=True).to(DEV)
    out_dir = os.path.join(FV_DIR, arch)
    os.makedirs(out_dir, exist_ok=True)
    res = dict(layer=L, units=units, majority_class=[names[c] for c in classes], judge_other=OTHER[arch],
               dataset_max=Sf.max(0).tolist(), variants={})
    for v in variants:
        t0 = time.time()
        x, obj = visualize(model, hk, units_t, v, rho=rho, rbar=rbar, smax=smax)
        r_self = judge(model, x, classes)
        r_other = judge(other, x, classes)
        for i, u in enumerate(units):
            save_png(x[i], os.path.join(out_dir, f"{u}_{v}.png"))
        res["variants"][v] = dict(time_s=time.time() - t0, rank_self=r_self, rank_other=r_other,
                                  obj_max=obj["max"].tolist(), obj_mean=obj["mean"].tolist(), obj_center=obj["center"].tolist(),
                                  ratio_max=(obj["max"] / Sf.max(0)).tolist())
        print(f"[fv2] {arch} {v:10s} {time.time() - t0:5.0f} s  rank(other) {r_other}  rank(self) {r_self}  "
              f"max/dsmax {np.round(obj['max'] / Sf.max(0), 2).tolist()}  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    hk.detach()
    return res


def contact_sheet(arch, res, variants, out, T=224, gap=6, nphoto=3):
    from PIL import Image, ImageDraw, ImageFont
    X, y = C.images("imagenet_val")
    _, names = C.class_meta()
    L = C.last_layer(arch)
    S = np.load(os.path.join(C.cache_dir("imagenet_val"), f"{arch}_extract.npz"))[f"{L}_pos"]
    try:
        f, fb = ImageFont.truetype("arial.ttf", 18), ImageFont.truetype("arialbd.ttf", 22)
    except Exception:
        f = fb = ImageFont.load_default()
    units = res["units"]
    W = len(variants) * (T + gap) + 40 + nphoto * (T + gap)
    H = 50 + len(units) * (T + 52)
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for j, v in enumerate(variants):
        d.text((j * (T + gap), 8), v, font=fb, fill=(40, 40, 40))
    d.text((len(variants) * (T + gap) + 40, 8), "strongest photos", font=fb, fill=(40, 40, 40))
    for i, u in enumerate(units):
        y0 = 50 + i * (T + 52)
        d.text((0, y0), f"{arch} unit {u} · {res['majority_class'][i]} · cell label = rank of this class for the independent judge ({res['judge_other']}) / own net",
               font=f, fill=(60, 60, 60))
        y0 += 24
        for j, v in enumerate(variants):
            r = res["variants"][v]
            p = Image.open(os.path.join(FV_DIR, arch, f"{u}_{v}.png")).convert("RGB").resize((T, T))
            dd = ImageDraw.Draw(p)
            txt = f"{r['rank_other'][i]} / {r['rank_self'][i]}"
            dd.rectangle([4, T - 28, 12 + dd.textlength(txt, font=f), T - 4], fill=(20, 24, 30))
            dd.text((7, T - 26), txt, font=f, fill="white")
            im.paste(p, (j * (T + gap), y0))
        top = np.argsort(-S[:, u].astype(np.float32), kind="stable")[:nphoto]
        for k, t in enumerate(top):
            im.paste(Image.fromarray(np.asarray(X[t])).resize((T, T)), (len(variants) * (T + gap) + 40 + k * (T + gap), y0))
    im.save(out, quality=90)
    print("wrote", out, im.size)


def render_all(arch, variant, seeds_extra=(1,)):
    """Render the gallery sample of imnet_featvis.json (8 picked + 12 random units, plus seed 1 for the two seed
    units) with `variant` as the "tuned" setting: writes reports/figs/imnet_fv/<arch>/<unit>_tuned[_s1].png, adds
    fv/ratio["tuned"] per unit, and judges every setting's render of every unit with the independent classifier
    (rank of the unit's majority class; summary = class-dominated units whose render it ranks in the top 5)."""
    import analysis.imnet_featvis as FV1
    torch.backends.cudnn.benchmark = True
    fn = os.path.join(C.RESULTS, "imnet_featvis.json")
    J = json.load(open(fn))
    Ja = J[arch]
    units = Ja["picked"] + Ja["random"]
    X, y = C.images("imagenet_val")
    L = C.last_layer(arch)
    S = np.load(os.path.join(C.cache_dir("imagenet_val"), f"{arch}_extract.npz"))[f"{L}_pos"]
    classes = []
    for u in units:
        top = np.argsort(-S[:, u].astype(np.float32), kind="stable")[:20]
        v, c = np.unique(y[top], return_counts=True)
        classes.append(int(v[np.argmax(c)]))
    model = C.load_model(arch)
    other = C.load_model(OTHER[arch])
    for q in list(model.parameters()) + list(other.parameters()):
        q.requires_grad_(False)
    hk = Hooks(model, arch)
    units_t = torch.as_tensor(units, device=DEV)
    Sf = S[:, units].astype(np.float32)
    smax = torch.as_tensor(Sf.max(0), device=DEV)
    rho = torch.as_tensor(Sf.max(0) - np.median(Sf, 0), device=DEV)
    rbar = path_targets(model, hk, X, S, units_t) if CFG[variant].get("path") else None
    visualize.mag = natural_magnitude(MACO_CANVAS).to(DEV)
    visualize.mag_dc = natural_magnitude(MACO_CANVAS, dc=True).to(DEV)
    out_dir = os.path.join(FV1.FV_DIR, arch)
    t0 = time.time()
    jobs = [(units, 0)] + [([u], sd) for u in Ja["seed_units"] for sd in seeds_extra]
    for uu, sd in jobs:
        pos = [units.index(u) for u in uu]
        x, obj = visualize(model, hk, units_t[pos], variant, rho=rho[pos], rbar=None if rbar is None else rbar[pos],
                           smax=smax[pos], seed=1000 * sd)
        for k, u in enumerate(uu):
            save_png(x[k], os.path.join(out_dir, f"{u}_tuned" + (f"_s{sd}" if sd else "") + ".png"))
            r = Ja["units"][str(u)]
            r["fv"].setdefault("tuned", {})[f"s{sd}"] = float(obj["max"][k])
            r["ratio"].setdefault("tuned", {})[f"s{sd}"] = float(obj["max"][k] / max(r["dataset_max"], 1e-6))
    Ja["time_s"]["tuned"] = time.time() - t0
    Ja["tuned"] = dict(variant=variant, cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG[variant].items()},
                       canvas=MACO_CANVAS, crop_box=list(MACO_BOX), crops=MACO_CROPS, noise=MACO_NOISE, lr=MACO_LR)
    # judge every setting's seed-0 render of every unit
    from PIL import Image
    summary = {}
    dominated = [int(Ja["units"][str(u)]["majority_count"]) >= 10 for u in units]
    for st in FV1.SETTINGS:
        ims = [np.asarray(Image.open(os.path.join(out_dir, f"{u}_{st}.png")).convert("RGB")) for u in units]
        x = torch.as_tensor(np.stack(ims)).to(DEV).permute(0, 3, 1, 2).float() / 255
        r_other, r_self = judge(other, x, classes), judge(model, x, classes)
        for k, u in enumerate(units):
            Ja["units"][str(u)].setdefault("judge", {})[st] = dict(other=r_other[k], own=r_self[k])
        summary[st] = dict(top5_other_dominated=int(sum(r <= 5 for r, d in zip(r_other, dominated) if d)),
                           top5_own_dominated=int(sum(r <= 5 for r, d in zip(r_self, dominated) if d)),
                           n_dominated=int(sum(dominated)), median_rank_other=float(np.median(r_other)))
        print(f"[tuned] {arch} {st:8s} independent judge top-5: {summary[st]['top5_other_dominated']}/{summary[st]['n_dominated']} "
              f"class-dominated units (own net {summary[st]['top5_own_dominated']}), median rank {summary[st]['median_rank_other']:.0f}", flush=True)
    Ja["judge_summary"] = dict(judge=OTHER[arch], **summary)
    json.dump(J, open(fn, "w"), indent=1)
    hk.detach()
    print(f"[tuned] {arch}: {variant} rendered for {len(units)} units in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=list(UNITS))
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--render", default=None, help="variant to render as the 'tuned' setting for the full gallery sample")
    a = ap.parse_args()
    if a.render:
        for arch in a.archs:
            render_all(arch, a.render)
            torch.cuda.empty_cache()
        return
    fn = os.path.join(C.RESULTS, "imnet_featvis2.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for arch in a.archs:
        r = run(arch, a.variants)
        if arch in out:
            out[arch]["variants"].update(r["variants"])
            r = out[arch]
        out[arch] = r
        json.dump(out, open(fn, "w"), indent=1)
        contact_sheet(arch, r, [v for v in VARIANTS if v in r["variants"]], os.path.join(C.FIGS, f"imnet_fv2_{arch}.jpg"))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
