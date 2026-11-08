"""imnet_ the essay's own InceptionV1 (Lucid's 2014 inception5h weights, via the lucent port) as a third
pretrained model, with the feature-vis microscope at Lucid's defaults.

Why this exists. Our torchvision GoogLeNet renders (imnet_featvis*.py) stayed far below the essay's 2017
InceptionV1 column however we set the microscope. The smell test that settled it: the same library with the
same defaults (lucent = Lucid's PyTorch port; 128 px Fourier image, colour decorrelation, pad / jitter / scale /
rotate, bilinear upsample to 224, Adam 0.05, 512 steps) gives the essay's look on Lucid's 2014 weights at once and
mush on torchvision's BatchNorm-retrained GoogLeNet. The model was the difference, not the recipe.

What this script does for the 2014 net (tag `inceptionv1_lucid`, layer mixed5b):
  1. one pass over the 50,000 val images: pre-ReLU mixed5b pole scores (spatial max, -(spatial min); concat of the
     four branch convs in lucent's order 1x1 | 3x3 | 5x5 | pool_reduce = 1024 ch) and the raw 1008-way argmax, saved
     like imnet_extract.py to data/imnet_cache/imagenet_val/inceptionv1_lucid_extract.npz.
     Input convention (lucent's): RGB in [0,1] * 255 - 117, 224 px centre crops from the shared image cache.
  2. the net's 1008 outputs use the 2012 devkit class order, not the sorted-wnid order; the map is recovered as
     the modal true label per output index and verified by the top-1 accuracy it yields (and by the RGB vs BGR
     accuracy, which checks the channel order).
  3. the unit sample of imnet_featvis.select_units (8 highest top-20 class purity + 12 random, seed 0).
  4. renders with lucent defaults, objective = spatial mean of the unit's pre-ReLU branch channel (Lucid's
     "channel" objective on the *_pre_relu layer), FV seed 0, plus seed 1 for the first two picked units.
     Saved to reports/figs/imnet_fv/inceptionv1_lucid/<unit>_lucid[_s1].png (128 px render upsampled to 224).
  5. the independent judge of imnet_featvis2 (ConvNeXt-T reads each render; rank of the unit's majority class),
     and the own-net rank through the recovered map.
  6. the same lucent defaults on torchvision GoogLeNet (748, 502) and ConvNeXt-T (312, 702) for the
     "one microscope, three models" figure.
Figures: reports/figs/imnet_gallery_inceptionv1_lucid.png (20 units: render + 20 strongest photos),
reports/figs/imnet_lucid_models.png (same microscope on the three nets), reports/figs/imnet_lucid_seeds.png.
Results: reports/results/imnet_lucid.json.
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
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import imnet_common as C  # noqa: E402
from analysis.imnet_featvis import select_units  # noqa: E402
from analysis.imnet_purity import top_k  # noqa: E402

warnings.filterwarnings("ignore")
DEV = C.DEV
TAG, LAYER = "inceptionv1_lucid", "mixed5b"
BRANCHES = [("mixed5b_1x1_pre_relu_conv", 384), ("mixed5b_3x3_pre_relu_conv", 384),
            ("mixed5b_5x5_pre_relu_conv", 128), ("mixed5b_pool_reduce_pre_relu_conv", 128)]
FV_DIR = os.path.join(C.FIGS, "imnet_fv", TAG)
SIZE, STEPS, LR = 128, 512, 0.05
SEED_UNITS, EXTRA_SEEDS = 2, (1,)
MEAN = (C.MEAN / 255).reshape(1, 3, 1, 1).to(DEV)
STD = (C.STD / 255).reshape(1, 3, 1, 1).to(DEV)


def load_lucid():
    from lucent.modelzoo import inceptionv1
    return inceptionv1(pretrained=True).to(DEV).eval()


def branch_of(u):
    for name, n in BRANCHES:
        if u < n:
            return name, u
        u -= n
    raise ValueError(u)


# ------------------------------------------------------------------ 1-2. extraction + class map

@torch.no_grad()
def extract(model, bs=250, force=False):
    d = C.cache_dir("imagenet_val")
    fn = os.path.join(d, f"{TAG}_extract.npz")
    if os.path.exists(fn) and not force:
        return fn
    X, y = C.images("imagenet_val")
    N = len(X)
    store = {}
    hooks = [getattr(model, n).register_forward_hook(lambda m, i, o, n=n: store.__setitem__(n, o)) for n, _ in BRANCHES]
    pos = np.zeros((N, 1024), np.float16)
    neg = np.zeros((N, 1024), np.float16)
    pred = np.zeros(N, np.int64)
    pred_bgr = np.full(N, -1, np.int64)
    t = time.time()
    for b in range(0, N, bs):
        x = torch.from_numpy(np.ascontiguousarray(X[b:b + bs])).to(DEV).permute(0, 3, 1, 2).float() - 117.0
        out = model(x)
        a = torch.cat([store[n] for n, _ in BRANCHES], 1).flatten(2)          # [B, 1024, HW]
        pos[b:b + bs] = a.amax(-1).cpu().numpy().astype(np.float16)
        neg[b:b + bs] = (-a.amin(-1)).cpu().numpy().astype(np.float16)
        pred[b:b + bs] = out.argmax(1).cpu().numpy()
        if b % 5000 == 0:                                                       # channel-order check on 1 in 20 batches
            pred_bgr[b:b + bs] = model(x.flip(1)).argmax(1).cpu().numpy()
    for h in hooks:
        h.remove()
    # class map: modal true label per output index
    m1008 = np.full(1008, -1, np.int64)
    for o in np.unique(pred):
        m1008[o] = np.bincount(y[pred == o], minlength=1000).argmax()
    acc = float((m1008[pred] == y).mean())
    sel = pred_bgr >= 0
    acc_bgr = float((m1008[pred_bgr[sel]] == y[sel]).mean())
    np.savez(fn, pred=pred, layers=np.array([LAYER]), **{f"{LAYER}_pos": pos, f"{LAYER}_neg": neg},
             map1008=m1008, acc=acc, acc_bgr=acc_bgr, time_s=time.time() - t)
    print(f"extract {N} images {time.time()-t:.0f}s; top-1 through recovered map: RGB {acc:.3f}, BGR {acc_bgr:.3f}; "
          f"classes covered {len(np.unique(m1008[m1008 >= 0]))}", flush=True)
    return fn


# ------------------------------------------------------------------ 4. renders

def render_units(model, pairs, seed=0, size=SIZE, steps=STEPS):
    """lucent defaults, one batched optimisation: image i maximises the spatial mean of channel c_i of module
    layer_i (pre-ReLU for the Lucid net). pairs = [(layer, c), ...]. Returns PIL images and the wall time."""
    from lucent.optvis import objectives, param, render

    @objectives.wrap_objective()
    def chans(pairs):
        def inner(model):
            return sum(-model(layer)[i, c].mean() for i, (layer, c) in enumerate(pairs))
        return inner

    torch.manual_seed(seed)
    t = time.time()
    ims = render.render_vis(model, chans(pairs), param_f=lambda: param.image(size, batch=len(pairs)),
                            thresholds=(steps,), show_image=False, progress=False)[0]
    return [Image.fromarray((im * 255).astype(np.uint8)) for im in ims], time.time() - t


def render_unit(model, layer, c, seed=0, size=SIZE, steps=STEPS):
    ims, dt = render_units(model, [(layer, c)], seed, size, steps)
    return ims[0], dt


def tv_models():
    """torchvision GoogLeNet / ConvNeXt-T with a pre-activation hook that lucent can address by module name."""
    out = {}
    g = C.load_model("googlenet").to(memory_format=torch.contiguous_format)
    out["googlenet"] = (g, None)
    c = C.load_model("convnext_tiny").to(memory_format=torch.contiguous_format)
    out["convnext_tiny"] = (c, None)
    return out


class PreAct(torch.nn.Module):
    """Wraps torchvision GoogLeNet so that module `pre5b` outputs the inception5b pre-ReLU concat [B, 1024, 7, 7]
    (branch BatchNorm outputs), and ConvNeXt-T so that `pre_mlp` outputs the last MLP hidden pre-GELU as [B, 3072, 7, 7]."""

    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.net = C.load_model(arch).to(memory_format=torch.contiguous_format)
        self.pre5b = torch.nn.Identity()
        self.pre_mlp = torch.nn.Identity()
        self.store = {}
        if arch == "googlenet":
            blk = self.net.inception5b
            for k, bn in enumerate([blk.branch1.bn, blk.branch2[1].bn, blk.branch3[1].bn, blk.branch4[1].bn]):
                bn.register_forward_hook(lambda m, i, o, k=k: self.store.__setitem__(k, o))
        else:
            self.net.features[7][2].block[3].register_forward_hook(lambda m, i, o: self.store.__setitem__(0, o))

    def forward(self, x):
        self.store.clear()
        out = self.net(x)
        if self.arch == "googlenet":
            self.pre5b(torch.cat([self.store[k] for k in range(4)], 1))
        else:
            self.pre_mlp(self.store[0].permute(0, 3, 1, 2))
        return out


# ------------------------------------------------------------------ 5. judge

@torch.no_grad()
def judge_ranks(judge_model, ims, classes):
    x = torch.stack([torch.from_numpy(np.asarray(im.resize((224, 224), Image.BICUBIC))).permute(2, 0, 1).float() / 255
                     for im in ims]).to(DEV)
    logits = judge_model((x - MEAN) / STD)
    order = logits.argsort(1, descending=True)
    return [int((order[i] == c).nonzero()[0, 0]) + 1 for i, c in enumerate(classes)]


@torch.no_grad()
def own_ranks(model, ims, classes, m1008):
    inv = {int(c): o for o, c in enumerate(m1008) if c >= 0}
    x = torch.stack([torch.from_numpy(np.asarray(im.resize((224, 224), Image.BICUBIC))).permute(2, 0, 1).float()
                     for im in ims]).to(DEV) - 117.0
    order = model(x).argsort(1, descending=True)
    return [int((order[i] == inv[int(c)]).nonzero()[0, 0]) + 1 if int(c) in inv else None for i, c in enumerate(classes)]


# ------------------------------------------------------------------ figures (PIL, high-res)

def _font(size, bold=False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _label(im, text, font):
    im = im.copy()
    d = ImageDraw.Draw(im)
    w = d.textlength(text, font=font)
    d.rectangle([3, im.height - 24, 9 + w, im.height - 3], fill=(20, 24, 30))
    d.text((6, im.height - 23), text, font=font, fill="white")
    return im


def gallery(units, info, X, y, short, Spos, out, T=224, P=112, gap=4, K=20):
    rows = len(units)
    W = T + 16 + 10 * (P + gap)
    H = 40 + rows * (T + 34)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f, fb, fs = _font(20), _font(24, True), _font(15)
    d.text((0, 6), "InceptionV1, Lucid's 2014 weights, mixed5b: feature vis (lucent defaults) and the 20 strongest of 50,000 val images",
           font=fb, fill=(30, 30, 30))
    top = top_k(Spos[:, units], K)
    yy = 40
    for r, u in enumerate(units):
        m, k, role = info[u]["majority"], info[u]["count"], info[u]["role"]
        d.text((0, yy), f"unit {u} ({role}) · top-20 majority: {short[m]} {k}/20 · judge rank {info[u]['judge_other']} (ConvNeXt-T), own {info[u]['judge_own']}",
               font=f, fill=(60, 60, 60))
        yy += 28
        canvas.paste(Image.open(os.path.join(FV_DIR, f"{u}_lucid.png")).convert("RGB").resize((T, T)), (0, yy))
        for i, idx in enumerate(top[r]):
            im = Image.fromarray(np.asarray(X[idx])).resize((P, P))
            if int(y[idx]) != m:
                im = _label(im, short[int(y[idx])], fs)
            canvas.paste(im, (T + 16 + (i % 10) * (P + gap), yy + (i // 10) * (P + gap)))
        yy += T + 6
    canvas.save(out)
    print("wrote", out, canvas.size, flush=True)


def models_fig(cells, out, T=224, gap=8):
    """cells: list of rows (title, [(label, PIL image), ...]); first image per row is the render, the rest photos."""
    f, fb = _font(20), _font(24, True)
    n = max(len(r[1]) for r in cells)
    W = n * (T + gap) + 40
    H = len(cells) * (T + 40) + 50
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((0, 6), "One microscope (Lucid's defaults, via lucent) on three pretrained nets", font=fb, fill=(30, 30, 30))
    yy = 44
    for title, ims in cells:
        d.text((0, yy), title, font=f, fill=(60, 60, 60))
        yy += 28
        for i, (lab, im) in enumerate(ims):
            x = i * (T + gap) + (40 if i else 0)
            canvas.paste(im.resize((T, T), Image.BICUBIC), (x, yy))
            if lab:
                d.text((x + 4, yy + T - 24), lab, font=_font(16), fill=(240, 240, 240),
                       stroke_width=2, stroke_fill=(0, 0, 0))
        yy += T + 12
    canvas.save(out)
    print("wrote", out, canvas.size, flush=True)


# ------------------------------------------------------------------ main

def main(force=False, skip_models_fig=False):
    os.makedirs(FV_DIR, exist_ok=True)
    model = load_lucid()
    fn = extract(model, force=force)
    z = np.load(fn)
    S, m1008 = z[f"{LAYER}_pos"].astype(np.float32), z["map1008"]
    X, y = C.images("imagenet_val")
    _, names = C.class_meta()
    short = [str(n).split(",")[0] for n in names]
    picked, rand, cnt, val, T20 = select_units(S, y)
    units = [int(u) for u in picked] + [int(u) for u in rand]
    info = {u: {"role": "picked" if i < len(picked) else "random", "majority": int(val[u]), "count": int(cnt[u]),
                "dataset_max": float(S[:, u].max())} for i, u in enumerate(units)}
    torch.cuda.empty_cache()
    # renders, batched 8 units per optimisation
    ims, times = {}, {}
    todo = [u for u in units if force or not os.path.exists(os.path.join(FV_DIR, f"{u}_lucid.png"))]
    for b in range(0, len(todo), 8):
        chunk = todo[b:b + 8]
        out, dt = render_units(model, [branch_of(u) for u in chunk])
        for u, im in zip(chunk, out):
            im = im.resize((224, 224), Image.BICUBIC)
            im.save(os.path.join(FV_DIR, f"{u}_lucid.png"))
            times[u] = dt / len(chunk)
        print(f"rendered {chunk} in {dt:.0f}s", flush=True)
    for u in units:
        ims[u] = Image.open(os.path.join(FV_DIR, f"{u}_lucid.png")).convert("RGB")
        print(f"unit {u} ({branch_of(u)}) {short[info[u]['majority']]} {info[u]['count']}/20", flush=True)
    for s in EXTRA_SEEDS:
        need = [u for u in units[:SEED_UNITS] if force or not os.path.exists(os.path.join(FV_DIR, f"{u}_lucid_s{s}.png"))]
        if need:
            out, _ = render_units(model, [branch_of(u) for u in need], seed=s)
            for u, im in zip(need, out):
                im.resize((224, 224), Image.BICUBIC).save(os.path.join(FV_DIR, f"{u}_lucid_s{s}.png"))
    # judges
    judge_model = C.load_model("convnext_tiny").to(memory_format=torch.contiguous_format)
    classes = [info[u]["majority"] for u in units]
    r_other = judge_ranks(judge_model, [ims[u] for u in units], classes)
    r_own = own_ranks(model, [ims[u] for u in units], classes, m1008)
    for u, a, b in zip(units, r_other, r_own):
        info[u]["judge_other"], info[u]["judge_own"] = a, b
    dom = [u for u in units if info[u]["count"] >= 10]
    summary = {"n_units": len(units), "class_dominated": len(dom),
               "dominated_top5_other": int(sum(info[u]["judge_other"] <= 5 for u in dom)),
               "dominated_top5_own": int(sum((info[u]["judge_own"] or 9999) <= 5 for u in dom)),
               "median_rank_other_all": float(np.median(r_other)),
               "median_rank_own_all": float(np.median([r for r in r_own if r is not None]))}
    print("judge summary", summary, flush=True)
    # figures
    gallery(units, info, X, y, short, S, os.path.join(C.FIGS, f"imnet_gallery_{TAG}.png"))
    # seeds
    f = _font(18)
    canvas = Image.new("RGB", (SEED_UNITS * 2 * 230 + 20, 224 + 30), "white")
    d = ImageDraw.Draw(canvas)
    for i, u in enumerate(units[:SEED_UNITS]):
        for j, s in enumerate((0,) + EXTRA_SEEDS):
            fnp = os.path.join(FV_DIR, f"{u}_lucid" + (f"_s{s}" if s else "") + ".png")
            x = (i * 2 + j) * 230 + (20 if i else 0)
            canvas.paste(Image.open(fnp).convert("RGB"), (x, 28))
            d.text((x, 4), f"unit {u} seed {s}", font=f, fill=(60, 60, 60))
    canvas.save(os.path.join(C.FIGS, f"imnet_lucid_seeds.png"))
    # one microscope, three models
    cmp = {}
    if not skip_models_fig:
        ext = {a: np.load(os.path.join(C.CACHE, "imagenet_val", f"{a}_extract.npz"))[f"{C.last_layer(a)}_pos"]
               for a in ("googlenet", "convnext_tiny")}
        rows = []
        for u in units[:2]:
            idx = np.argsort(-S[:, u])[:4]
            rows.append((f"InceptionV1 (Lucid 2014) mixed5b unit {u}: {short[info[u]['majority']]} {info[u]['count']}/20",
                         [("render", ims[u])] + [(short[int(y[i])] if int(y[i]) != info[u]["majority"] else "", Image.fromarray(np.asarray(X[i]))) for i in idx]))
        for arch, layer, uu, title in (("googlenet", "pre5b", (748, 502), "GoogLeNet (torchvision BN retrain) inception5b"),
                                       ("convnext_tiny", "pre_mlp", (312, 702), "ConvNeXt-T last MLP")):
            wrapped = PreAct(arch)
            for u in uu:
                im, dt = render_unit(wrapped, layer, u)
                im = im.resize((224, 224), Image.BICUBIC)
                im.save(os.path.join(FV_DIR, f"cmp_{arch}_{u}_lucentdefault.png"))
                sc = ext[arch][:, u].astype(np.float32)
                order = np.argsort(-sc)
                idx = order[:4]
                bc = np.bincount(y[order[:20]], minlength=1000)
                m, k = int(bc.argmax()), int(bc.max())
                other = C.load_model({"googlenet": "convnext_tiny", "convnext_tiny": "googlenet"}[arch]).to(memory_format=torch.contiguous_format)
                cmp[f"{arch}_{u}"] = {"majority": m, "count": k, "judge_other": judge_ranks(other, [im], [m])[0], "time_s": dt}
                rows.append((f"{title} unit {u}: {short[m]} {k}/20  (same microscope)",
                             [("render", im)] + [(short[int(y[i])] if int(y[i]) != m else "", Image.fromarray(np.asarray(X[i]))) for i in idx]))
        models_fig(rows, os.path.join(C.FIGS, "imnet_lucid_models.png"))
    out = {"tag": TAG, "layer": LAYER, "model": "Lucid's InceptionV1 (inception5h, 2014) via lucent 0.1.8; input RGB*255-117",
           "recipe": {"library": "lucent render_vis defaults", "size": SIZE, "steps": STEPS, "lr": LR,
                      "transforms": "pad 12 (grey), jitter 8, scale 0.9..1.1 (11 values), rotate -10..10 deg, jitter 4, bilinear upsample 128->224",
                      "param": "Fourier (1/f) + colour decorrelation, sigmoid", "objective": "spatial mean of the pre-ReLU branch channel"},
           "class_map": {"acc_rgb": float(z["acc"]), "acc_bgr": float(z["acc_bgr"]), "covered": int((m1008 >= 0).sum())},
           "picked": [int(u) for u in picked], "random": [int(u) for u in rand], "seed_units": units[:SEED_UNITS],
           "units": {str(u): info[u] for u in units}, "render_time_s": times, "judge_summary": summary, "same_microscope_other_nets": cmp}
    json.dump(out, open(os.path.join(C.RESULTS, "imnet_lucid.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-models-fig", action="store_true")
    a = ap.parse_args()
    main(a.force, a.skip_models_fig)
