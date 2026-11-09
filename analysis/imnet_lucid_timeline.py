"""imnet_ the essay's own microscope-timeline neurons, rendered on the essay's own weights.

The essay's figure "A History of Microscope Errors" names its two neurons (image alt text on the page): InceptionV1
mixed4d:161 (jars, jugs, lamp bases) and mixed5a:400 (boxy devices with a round lens), each shown as 2009-style gradient
ascent, 2017 Lucid defaults and "2026 settings", beside four of its strongest photos. The essay never specifies the
2026 settings ("a few subtle tweaks to reasonable settings"; "small regularization tweaks"). With the same 2014
weights (lucent) we can put our renders of the SAME neurons beside theirs and judge by eye which knobs move ours
toward their 2026 look (pale, soft, low saturation, one whole object, plain background).

Variants (all 128 px, 512 steps, Adam 0.05, upsampled to 224 unless noted):
  2009     pixel parameterization, no transforms, no decorrelation (plain gradient ascent).
  2017     lucent defaults (Fourier 1/f, colour decorrelation, standard transforms).
  path     2017 + our path anchor (first 64 steps; cosine to the mean attribution of the 60 strongest val images on
           the module's input tensor; weight rho = dataset max - median of the pole score).
  lowlr    2017 with lr 0.02 (less saturation).
  decay    2017 with a steeper spectrum, 1/|f|^1.5.
  jit      2017 with wider transforms: pad 16, jitter 16, scale 0.8-1.2, rotate +-15, jitter 8.
  pale     2017 + an L2 penalty on the image's departure from mid grey (towards the essay's washed-out look).
  combo    path + decay + jit + pale.
  combo_lr combo at lr 0.02.
Judgement is by eye against the essay's images (downloaded by the caller into ESSAY_DIR); the sheet puts them side by
side. Writes reports/figs/imnet_fv2/inceptionv1_lucid/timeline_<unit>_<variant>.png, reports/figs/imnet_lucid_timeline.png
and reports/results/imnet_lucid_timeline.json (objective, objective / dataset max, step-0 cosine for path variants).
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
from analysis.imnet_lucid import TAG, _font, load_lucid  # noqa: E402

warnings.filterwarnings("ignore")
DEV = C.DEV
# (layer, unit on the post-ReLU concat, what the essay's alt text says, module whose OUTPUT feeds the layer)
NEURONS = [("mixed4d", 161, "jars, jugs, lamp bases", "mixed4c"),
           ("mixed5a", 400, "boxy device with a round lens", "maxpool4")]
ORDER = ["1x1", "3x3", "5x5", "pool_reduce"]
VARIANTS = ["2009", "2017", "path", "lowlr", "decay", "jit", "pale", "combo", "combo_lr"]
CFG = {
    "2009":     dict(pixel=True),
    "2017":     dict(),
    "path":     dict(path=True),
    "lowlr":    dict(lr=0.02),
    "decay":    dict(decay=1.5),
    "jit":      dict(wide=True),
    "pale":     dict(pale=0.5),
    "combo":    dict(path=True, decay=1.5, wide=True, pale=0.5),
    "combo_lr": dict(path=True, decay=1.5, wide=True, pale=0.5, lr=0.02),
}
SIZE, STEPS, LR, PATH_STEPS, TOPK = 128, 512, 0.05, 64, 60
OUT_DIR = os.path.join(C.FIGS, "imnet_fv2", TAG)
JSON = os.path.join(C.RESULTS, "imnet_lucid_timeline.json")
SHEET = os.path.join(C.FIGS, "imnet_lucid_timeline.png")
ESSAY_DIR = os.environ.get("ESSAY_TIMELINE_DIR", "")


def branch_of(model, layer, u):
    """unit on the concat -> (pre-ReLU branch module name, local channel), using the convs' out_channels."""
    for b in ORDER:
        n = f"{layer}_{b}_pre_relu_conv"
        k = getattr(model, n).out_channels
        if u < k:
            return n, u
        u -= k
    raise ValueError(u)


class Hooks:
    def __init__(self, model, branch, prev):
        self.out, self.prev, self.h = None, None, []
        self.h.append(getattr(model, branch).register_forward_hook(lambda m, i, o: setattr(self, "out", o)))
        self.h.append(getattr(model, prev).register_forward_hook(lambda m, i, o: setattr(self, "prev", o)))

    def score(self, c):
        return self.out[:, c].mean((1, 2))          # spatial mean per image

    def detach(self):
        for h in self.h:
            h.remove()


# ------------------------------------------------------------------ pole scores for the two channels (top-60 photos)

@torch.no_grad()
def pole_scores(model, X, bs=250):
    fn = os.path.join(C.cache_dir("imagenet_val"), f"{TAG}_timeline_scores.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return {f"{l}:{u}": z[f"{l}:{u}"] for l, u, _, _ in NEURONS}
    store = {}
    hs = []
    for l, u, _, _ in NEURONS:
        n, c = branch_of(model, l, u)
        hs.append(getattr(model, n).register_forward_hook(
            lambda m, i, o, key=f"{l}:{u}", c=c: store.__setitem__(key, o[:, c].amax((1, 2)).float().cpu().numpy())))
    S = {f"{l}:{u}": [] for l, u, _, _ in NEURONS}
    t = time.time()
    for b in range(0, len(X), bs):
        x = torch.from_numpy(np.ascontiguousarray(X[b:b + bs])).to(DEV).permute(0, 3, 1, 2).float() - 117.0
        model(x)
        for k in S:
            S[k].append(store[k])
        if b % 10000 == 0:
            print(f"  scores {b}/{len(X)} {time.time() - t:.0f}s", flush=True)
    for h in hs:
        h.remove()
    S = {k: np.concatenate(v) for k, v in S.items()}
    np.savez(fn, **S)
    return S


def attribution(model, hk, x, c, create_graph):
    model(x)
    s = hk.score(c)
    g, = torch.autograd.grad(s.sum(), hk.prev, create_graph=create_graph)
    return (hk.prev * g).sum((2, 3)), s


def path_target(model, hk, X, scores, c):
    top = np.sort(np.argsort(-scores, kind="stable")[:TOPK])
    A = []
    for b in range(0, TOPK, 20):
        x = torch.from_numpy(np.ascontiguousarray(X[top[b:b + 20]])).to(DEV).permute(0, 3, 1, 2).float() - 117.0
        x.requires_grad_(True)
        a, _ = attribution(model, hk, x, c, False)
        A.append(a.detach())
    return torch.cat(A).mean(0), top


# ------------------------------------------------------------------ rendering

def transforms(cfg, size):
    from lucent.optvis import transform
    if cfg.get("pixel"):
        ts = []
    elif cfg.get("wide"):
        ts = [transform.pad(16, mode="constant", constant_value=0.5), transform.jitter(16),
              transform.random_scale([0.8 + i * 0.04 for i in range(11)]),
              transform.random_rotate(list(range(-15, 16)) + 5 * [0]), transform.jitter(8)]
    else:
        ts = transform.standard_transforms.copy()
    ts.append(transform.preprocess_inceptionv1())
    if size < 224:
        ts.append(torch.nn.Upsample(size=224, mode="bilinear", align_corners=True))
    return transform.compose(ts)


def render(model, hk, c, cfg, rho=None, rbar=None, seed=0, size=SIZE, steps=STEPS):
    from lucent.optvis.param import color, spatial
    torch.manual_seed(seed)
    np.random.seed(seed)
    shape = [1, 3, size, size]
    if cfg.get("pixel"):
        params, image_f = spatial.pixel_image(shape)
        image_f = color.to_valid_rgb(image_f, decorrelate=False)
    else:
        params, image_f = spatial.fft_image(shape, decay_power=cfg.get("decay", 1))
        image_f = color.to_valid_rgb(image_f, decorrelate=True)
    opt = torch.optim.Adam(params, lr=cfg.get("lr", LR))
    tf = transforms(cfg, size)
    diag = {}
    for step in range(steps):
        opt.zero_grad()
        x = image_f()
        xs = tf(x)
        if cfg.get("path") and step < PATH_STEPS:
            A, s = attribution(model, hk, xs, c, True)
            cos = F.cosine_similarity(A, rbar[None], dim=1)
            loss = -s.sum() - rho * cos.sum()
            if step == 0:
                diag["cos0"] = float(cos.mean())
        else:
            model(xs)
            s = hk.score(c)
            loss = -s.sum()
        if cfg.get("pale"):
            loss = loss + cfg["pale"] * ((x - 0.5) ** 2).mean() * s.detach().abs().clamp(min=1.0).sum()
        if step == 0:
            diag["s0"] = float(s.mean())
        loss.backward()
        opt.step()
    with torch.no_grad():
        im = image_f()[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        xs = tf(image_f())
        model(xs)
        diag["s_final"] = float(hk.score(c).mean())
    return Image.fromarray((im * 255).astype(np.uint8)), diag


# ------------------------------------------------------------------ sheet

def sheet(res, X, tops, out, T=160, gap=5):
    f, fb, fs = _font(15), _font(19, True), _font(12)
    cols_e = ["2009", "2017", "2026"]
    ncol = len(cols_e) + len(VARIANTS) + 4
    W = gap + ncol * (T + gap) + 2 * 14
    H = 46 + len(NEURONS) * (T + 44)
    cv = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(cv)
    d.text((gap, 6), "The essay's two microscope-timeline neurons, on the essay's own weights: their renders (left), ours (middle), the strongest val photos (right)",
           font=fb, fill=(30, 30, 30))
    y = 46
    for (l, u, desc, _), row in zip(NEURONS, res):
        key = f"{l}:{u}"
        x = gap
        d.text((x, y - 16), f"{key}  ({desc})", font=f, fill=(30, 30, 30))
        for v in cols_e:
            p = os.path.join(ESSAY_DIR, f"{v}-{l}-{u}.webp")
            im = Image.open(p).convert("RGB").resize((T, T)) if os.path.exists(p) else Image.new("RGB", (T, T), (235, 235, 235))
            cv.paste(im, (x, y))
            d.text((x + 3, y + T + 2), f"essay {v}", font=fs, fill=(120, 40, 40))
            x += T + gap
        x += 14
        for v in VARIANTS:
            cv.paste(row[v]["im"].resize((T, T), Image.BICUBIC), (x, y))
            d.text((x + 3, y + T + 2), f"ours {v}", font=fs, fill=(30, 30, 30))
            d.text((x + 3, y + T + 15), f"{row[v]['ratio']:.2f}x max", font=fs, fill=(90, 90, 90))
            x += T + gap
        x += 14
        for i in tops[key][:4]:
            cv.paste(Image.fromarray(np.ascontiguousarray(X[i])).resize((T, T)), (x, y))
            x += T + gap
        d.text((x - 4 * (T + gap) + 3, y + T + 2), "strongest val photos (ours)", font=fs, fill=(30, 30, 30))
        y += T + 44
    cv.save(out)
    return cv


def main(variants=VARIANTS, force=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_lucid()
    X, _ = C.images("imagenet_val")
    S = pole_scores(model, X)
    torch.cuda.empty_cache()
    res, tops, J = [], {}, {}
    for l, u, desc, prev in NEURONS:
        key = f"{l}:{u}"
        n, c = branch_of(model, l, u)
        hk = Hooks(model, n, prev)
        s = S[key]
        rbar, top = path_target(model, hk, X, s, c)
        tops[key] = np.argsort(-s, kind="stable")[:TOPK]
        rho = float(np.max(s) - np.median(s))
        row, J[key] = {}, dict(branch=n, channel=int(c), prev=prev, dataset_max=float(np.max(s)), rho=rho,
                             top_val=[int(i) for i in tops[key][:10]], variants={})
        for v in variants:
            fn = os.path.join(OUT_DIR, f"timeline_{l}_{u}_{v}.png")
            t = time.time()
            im, diag = render(model, hk, c, CFG[v], rho=rho, rbar=rbar)
            im.resize((224, 224), Image.BICUBIC).save(fn)
            ratio = diag["s_final"] / float(np.max(s))
            row[v] = dict(im=im, ratio=ratio)
            J[key]["variants"][v] = dict(**diag, ratio=ratio, time_s=time.time() - t)
            print(f"{key} {v:9s} s={diag['s_final']:.2f} ({ratio:.2f}x max) {time.time() - t:.0f}s", flush=True)
        hk.detach()
        res.append(row)
    sheet(res, X, tops, SHEET)
    with open(JSON, "w") as fh:
        json.dump(dict(neurons=J, cfg=CFG, size=SIZE, steps=STEPS, lr=LR, path_steps=PATH_STEPS, topk=TOPK), fh, indent=1)
    print("wrote", SHEET, JSON)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    main(a.variants.split(","))
