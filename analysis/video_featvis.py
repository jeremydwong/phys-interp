"""Analysis 4: feature visualization of selected units by optimizing the 4-frame input clip.

Three settings, after the essay's microscope comparison:
  naive      ("2009"): pixel parameterization, plain gradient ascent, no transforms.
  fourier    ("2017"): decorrelated Fourier parameterization (1/f scaling + colour decorrelation), +-1 px jitter.
  robust     ("2026"): strong transformation robustness (jitter +-3 px, scale 0.92-1.08, same transform
                       for all 4 frames) + a scene-palette parameterization (static background + two
                       Fourier-parameterized alpha masks painted in the blue upper-arm / orange forearm
                       colours) + mild sparsity and total-variation penalties on the masks.
                       The palette is a domain prior we add; it is not in the essay.
Objective: the unit's frame-t activation at its modal argmax position (from the top-1% dataset clips).
We report the activation reached and the activation under random 2-px shifts (robustness check).
Figure: video_featvis.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, INK, INK2, MUTED, savefig, setup_style  # noqa: E402
from analysis.video_census import load_census  # noqa: E402
from analysis.video_common import CACHE, DEV, LAYERS, RESULTS, cached_acts, load, target_matrix, windows  # noqa: E402
from analysis.video_viz import clean, trail  # noqa: E402

COLOR_CORR = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])
COLOR_CORR = COLOR_CORR / COLOR_CORR.norm(dim=0).max()


def fourier_image(n=4, size=64, decay=1.0, sd=0.01, gen=None):
    fy = torch.fft.fftfreq(size)[:, None]
    fx = torch.fft.rfftfreq(size)[None]
    freq = torch.sqrt(fx ** 2 + fy ** 2)
    scale = (1.0 / torch.clamp(freq, min=1.0 / size) ** decay)
    scale = scale / scale.max() * size
    spec = (torch.randn(n, 3, size, size // 2 + 1, 2, generator=gen) * sd).to(DEV).requires_grad_(True)
    scale = scale.to(DEV)
    cc = COLOR_CORR.to(DEV)

    def img():
        s = torch.view_as_complex(spec) * scale
        x = torch.fft.irfft2(s, s=(size, size), norm="ortho")            # [n,3,H,W]
        x = torch.einsum("nchw,dc->ndhw", x, cc)
        return torch.sigmoid(x + 1.5)                                       # bias toward the light background
    return spec, img


_BG = None


def background():
    """Static scene without the arm: per-pixel median over frames of many episodes."""
    global _BG
    if _BG is None:
        from analysis.video_common import frames
        fr = frames()
        x = np.stack([fr[e, t] for e in range(2000, 2400, 8) for t in (0, 75, 149)]).astype(np.float32) / 255
        _BG = torch.as_tensor(np.median(x, 0).transpose(2, 0, 1), device=DEV)
    return _BG


ARM_BLUE = torch.tensor([0.16, 0.33, 0.78])
ARM_ORANGE = torch.tensor([0.85, 0.45, 0.12])


def palette_image(n=4, size=64, gen=None):
    """Scene palette parameterization: background + two Fourier-parameterized alpha masks painted
    in the upper-arm (blue) and forearm (orange) colours."""
    spec, raw = fourier_image(n=n, size=size, gen=gen)
    bg = background()
    cols = torch.stack([ARM_BLUE, ARM_ORANGE]).to(DEV)          # [2,3]

    def masks():
        s = torch.view_as_complex(spec[:, :2])
        return s

    fy = torch.fft.fftfreq(size)[:, None]
    fx = torch.fft.rfftfreq(size)[None]
    scale = 1.0 / torch.clamp(torch.sqrt(fx ** 2 + fy ** 2), min=1.0 / size)
    scale = (scale / scale.max() * size).to(DEV)

    def alphas():
        x = torch.fft.irfft2(torch.view_as_complex(spec[:, :2]) * scale, s=(size, size), norm="ortho")
        return torch.sigmoid(4 * x - 3)                           # [n,2,H,W], biased toward 0 (empty scene)

    def img():
        a = alphas()
        a_tot = a.sum(1, keepdim=True).clamp(max=1)
        paint = torch.einsum("nkhw,kc->nchw", a, cols) / a.sum(1, keepdim=True).clamp(min=1e-6)
        return bg[None] * (1 - a_tot) + paint * a_tot
    img.alphas = alphas
    return spec, img


def pixel_image(n=4, size=64, gen=None):
    p = (torch.randn(n, 3, size, size, generator=gen) * 0.01 + 1.5).to(DEV).requires_grad_(True)
    return p, lambda: torch.sigmoid(p)


def transform(x, jitter, scale_rng):
    """Same random affine (translation + isotropic scale) for all frames. x [4,3,64,64]."""
    s = float(np.random.uniform(*scale_rng)) if scale_rng else 1.0
    tx, ty = np.random.uniform(-jitter, jitter, 2) * 2 / 64 if jitter else (0.0, 0.0)
    theta = torch.tensor([[1 / s, 0, tx], [0, 1 / s, ty]], dtype=torch.float32, device=DEV)[None].expand(x.shape[0], 2, 3)
    g = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, g, padding_mode="border", align_corners=False)


def unit_act(model, x, layer, ch, pos):
    model.record = True
    model(x)
    model.record = False
    a = model.acts[layer][:, ch]
    return a[:, pos[0], pos[1]]


def visualize(model, layer, ch, pos, mode, steps=320, seed=0):
    g = torch.Generator().manual_seed(seed)
    np.random.seed(seed)
    if mode == "naive":
        p, img = pixel_image(gen=g); lr = 0.05
    elif mode == "fourier":
        p, img = fourier_image(gen=g); lr = 0.03
    else:
        p, img = palette_image(gen=g); lr = 0.03
    opt = torch.optim.Adam([p], lr=lr)
    for s in range(steps):
        x = img()
        if mode == "fourier":
            xs = torch.stack([transform(x, 1, None) for _ in range(4)])
        elif mode == "robust":
            xs = torch.stack([transform(x, 3, (0.92, 1.08)) for _ in range(4)])
        else:
            xs = x[None]
        act = unit_act(model, xs, layer, ch, pos).mean()
        loss = -act
        if mode == "robust":
            a = img.alphas()
            tv = (a[..., 1:, :] - a[..., :-1, :]).abs().mean() + (a[..., 1:] - a[..., :-1]).abs().mean()
            loss = loss + 2.0 * a.mean() + 1.0 * tv          # sparse, smooth strokes
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x = img()
        a0 = unit_act(model, x[None], layer, ch, pos).item()
        np.random.seed(123)
        shifted = torch.stack([transform(x, 2, None) for _ in range(16)])
        a_sh = unit_act(model, shifted, layer, ch, pos).mean().item()
    return x.cpu().numpy().transpose(0, 2, 3, 1), a0, a_sh


def modal_pos(acts, u, q=0.99):
    a = acts["max"][:, u].astype(np.float32)
    top = a >= np.quantile(a, q)
    arg = acts["arg"][top, u].astype(int)
    m = np.bincount(arg, minlength=256).argmax()
    return (m // 16, m % 16)


def main():
    setup_style()
    Y, vs, st = target_matrix()
    c = load_census("final")
    H = int(c["H"])
    acts = cached_acts("final", "test", lambda: load("final")[0], res=True)
    model, _ = load("final")
    for p in model.parameters():
        p.requires_grad_(False)
    units = json.load(open(os.path.join(CACHE, "units.json")))
    want = ["posture", "velocity", "contact", "accel", "interaction", "gravity"]
    sel = []
    for w in want:
        cand = [d for d in units if d["slot"] == w]
        if cand:
            sel.append(cand[-1] if w in ("posture", "contact") else cand[0])
    rows, info = [], []
    for d in sel:
        u = d["unit"]
        layer, ch = LAYERS[u // H], u % H
        pos = modal_pos(acts, u)
        res = {m: visualize(model, layer, ch, pos, m) for m in ["naive", "fourier", "robust"]}
        a = acts["max"][:, u].astype(np.float32)
        top = np.argsort(-a)[:64]
        with torch.no_grad():
            at = unit_act(model, windows(acts["eps"][top], acts["ts"][top]), layer, ch, pos).cpu().numpy()
        i = int(top[np.argmax(at)])                 # top dataset clip *at the optimized position*
        a_ds = float(at.max())
        clip = windows([acts["eps"][i]], [acts["ts"][i]])[0].cpu().numpy().transpose(0, 2, 3, 1)
        rows.append((d, layer, ch, pos, res, clip))
        info.append({"unit": d["name"], "best_var": d["best_var"], "best_r2": d["best_r2"], "pos": list(map(int, pos)),
                     "dataset_max_at_pos": a_ds,
                     **{f"{m}_act": res[m][1] for m in res}, **{f"{m}_act_shift2px": res[m][2] for m in res}})
        print(info[-1], flush=True)
    json.dump(info, open(os.path.join(RESULTS, "video_featvis.json"), "w"), indent=1)

    import matplotlib.pyplot as plt
    n = len(rows)
    fig = plt.figure(figsize=(7.2, 0.92 * n + 0.45))
    gs = fig.add_gridspec(n, 9, width_ratios=[1.3, 1, 1, 0.15, 1, 1, 1, 1, 1.15], wspace=0.06, hspace=0.18,
                          left=0.01, right=0.99, bottom=0.02, top=1 - 0.4 / (0.92 * n + 0.45))
    heads = {1: "naive\n(“2009”)", 2: "Fourier\n(“2017”)", 4: "robust t−3", 5: "t−2", 6: "t−1",
             7: "robust t\n(“2026”)", 8: "top dataset\nclip"}
    for r, (d, layer, ch, pos, res, clip) in enumerate(rows):
        fam = vs[[v.name for v in vs].index(d["best_var"])].family
        ax = fig.add_subplot(gs[r, 0]); clean(ax, False)
        ax.text(0, 0.85, f"{layer}.{ch}", fontsize=8, fontweight="bold", color=INK, transform=ax.transAxes, va="top")
        ax.text(0, 0.55, vs[[v.name for v in vs].index(d["best_var"])].label, fontsize=5.8,
                color=FAMILIES[fam][0], transform=ax.transAxes, va="top")
        ax.text(0, 0.28, f"R² = {d['best_r2']:.2f}", fontsize=6, color=INK2, transform=ax.transAxes, va="top")
        imgs = {1: res["naive"][0][-1], 2: res["fourier"][0][-1], 4: res["robust"][0][0], 5: res["robust"][0][1],
                6: res["robust"][0][2], 7: res["robust"][0][3], 8: trail(clip)}
        acts_txt = {1: res["naive"][1], 2: res["fourier"][1], 7: res["robust"][1]}
        for j, im in imgs.items():
            ax = fig.add_subplot(gs[r, j])
            ax.imshow(np.clip(im, 0, 1), interpolation="bilinear")
            ax.plot(pos[1] * 4 + 1.5, pos[0] * 4 + 1.5, "+", ms=4, mew=0.8, color=(0.85, 0.1, 0.55))
            clean(ax)
            if j in acts_txt:
                ax.text(0.03, 0.03, f"{acts_txt[j]:.1f}", fontsize=5, color=INK2, transform=ax.transAxes,
                        bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))
            if r == 0:
                ax.set_title(heads[j], fontsize=6.5, loc="center", fontweight="normal", color=INK2)
    savefig(fig, "video_featvis.png")


if __name__ == "__main__":
    main()
