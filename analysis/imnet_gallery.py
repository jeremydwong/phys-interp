"""imnet_ galleries of pretrained-net last-layer neurons, in the layout of the essay's figures (Cammarata 2026).

Inputs: the feature visualizations of analysis/imnet_featvis.py (reports/figs/imnet_fv/<arch>/, imnet_featvis.json)
and the cached pole scores of imnet_extract (data/imnet_cache/imagenet_val/<arch>_extract.npz; pos = spatial max
of the pre-activation, neg = minus the spatial min, over the 50,000 val images).

Per arch (googlenet, convnext_tiny):
  imnet_gallery_<arch>.png     "ladybug-figure" + "microscope-timeline": one row per sampled neuron: the three FVs
                               (2009-style | 2017 | path-anchored | tuned) then the 20 strongest val images in order of
                               pole score; a white-on-dark label names the class of any image outside the row's
                               majority class; row title "unit N · <majority class> · k of 20" plus the sample
                               role (picked by purity / random).
  imnet_gallery/<arch>/<unit>.jpg  the same row on its own (~2400 px wide, JPEG quality 92).
  imnet_fv_seeds_<arch>.png    seed stability: two neurons x three settings x FV seeds 0 and 1.
  imnet_negpoles_<arch>.png    "negative-poles": the 6 neurons whose two poles are both most class-dominated
                               (max of min(purity_pos, purity_neg) on the top 20 of each pole), two rows each:
                               "+ <class>" top 8 of the positive pole, "- <class>" top 8 of the negative pole.
  imnet_shapes_<arch>.png      raw activation shapes (claim D): for the 12 random-sample neurons the sorted
                               positive pole score over all 50,000 images (score / max vs image rank) with the
                               percentile-rank version as a dashed diagonal.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as CM  # noqa: E402
from analysis import imnet_common as C  # noqa: E402
from analysis.imnet_featvis import ARCHS, FV_DIR, LABELS, SETTINGS  # noqa: E402
from analysis.imnet_purity import modal, top_k  # noqa: E402

DPI = 200
TILE, GAP, GROUP_GAP, TITLE_H = 100, 4, 16, 30     # pixels at 200 dpi
K = 20
PICK_COL, RAND_COL = "#1c5cab", CM.INK
ARCH_NAME = {"googlenet": "GoogLeNet (torchvision, BN retrain) inception5b",
             "convnext_tiny": "ConvNeXt-T features.7.2.mlp (pwconv1, pre-GELU)"}


def _px_fig(w, h):
    fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    return fig


def _tile(fig, x, y, w, h, W, H, img):
    """Axes at pixel box (x, y from the top) showing img, no axes decoration."""
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ax.imshow(img, interpolation="antialiased")
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    ax.set_axis_off()
    return ax


def _label(ax, text, color="white", bg="#111111", fontsize=3.6):
    ax.text(0.5, 0.03, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=fontsize, color=color,
            bbox=dict(facecolor=bg, alpha=0.72, pad=1.0, lw=0))


def load_fv(arch, u, st, seed=0):
    fn = os.path.join(FV_DIR, arch, f"{u}_{st}" + (f"_s{seed}" if seed else "") + ".png")
    return np.asarray(Image.open(fn).convert("RGB"))


def row_width():
    return len(SETTINGS) * (TILE + GAP) + GROUP_GAP + K * (TILE + GAP) - GAP


def draw_row(fig, y0, W, H, arch, u, info, X, y, names, T):
    """One gallery row at vertical pixel offset y0 (title above, tiles below). T = top-20 image indices."""
    maj = np.bincount(y[T], minlength=1000).argmax()
    k = int((y[T] == maj).sum())
    role = "picked by purity" if info["role"] == "picked" else "random"
    col = PICK_COL if info["role"] == "picked" else RAND_COL
    t = fig.text(0 / W, 1 - (y0 + 6) / H, f"unit {u} · {names[maj]} · {k} of 20", fontsize=6.4, weight="bold", color=col,
                 ha="left", va="top")
    fig.canvas.draw_idle()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    fig.text((bb.x1 + 12) / W, 1 - (y0 + 6) / H, f"{role}", fontsize=5.6, color=CM.MUTED, ha="left", va="top", style="italic")
    fig.text(1.0, 1 - (y0 + 6) / H, "  ".join(f"{LABELS[st].split(' ')[0]} {info['ratio'][st]['s0']:.2f}" for st in SETTINGS)
             + " × dataset max", fontsize=5.0, color=CM.MUTED, ha="right", va="top")
    yt = y0 + TITLE_H
    for i, st in enumerate(SETTINGS):
        ax = _tile(fig, i * (TILE + GAP), yt, TILE, TILE, W, H, load_fv(arch, u, st))
        _label(ax, LABELS[st], color="#f2f2f2", bg="#333333", fontsize=3.6)
    x0 = len(SETTINGS) * (TILE + GAP) + GROUP_GAP
    for c in range(K):
        ax = _tile(fig, x0 + c * (TILE + GAP), yt, TILE, TILE, W, H, np.asarray(X[T[c]]))
        if y[T[c]] != maj:
            _label(ax, names[y[T[c]]][:16])
    return maj, k


def gallery(arch, J, X, y, names, Spos):
    units = J["picked"] + J["random"]
    T = top_k(Spos[:, units], K)                   # [n, 20]
    W = row_width()
    head = 36
    H = head + len(units) * (TITLE_H + TILE + GAP)
    fig = _px_fig(W, H)
    fig.text(0, 1 - 4 / H, f"{ARCH_NAME[arch]}: feature visualizations and the 20 strongest of 50,000 val images, in order",
             fontsize=7.2, weight="bold", ha="left", va="top")
    fig.text(0, 1 - 20 / H, "left: " + " | ".join(LABELS[s] for s in SETTINGS) + " (positive pole, 224 px, FV seed 0)   "
             "right: labels mark images outside the row's majority class   blue titles: 8 units picked by top-20 purity; "
             "black: 12 random units (numpy seed 0)", fontsize=5.2, color=CM.INK2, ha="left", va="top")
    for r, u in enumerate(units):
        draw_row(fig, head + r * (TITLE_H + TILE + GAP), W, H, arch, u, J["units"][str(u)], X, y, names, T[r])
    fn = os.path.join(C.FIGS, f"imnet_gallery_{arch}.png")
    fig.savefig(fn, dpi=DPI, bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    if os.path.getsize(fn) > 8 * 2**20:           # photo grids: 256-colour palette keeps the PNG ~4x smaller
        Image.open(fn).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(fn, optimize=True)
    # individual strips
    d = os.path.join(C.FIGS, "imnet_gallery", arch)
    os.makedirs(d, exist_ok=True)
    for r, u in enumerate(units):
        Hs = TITLE_H + TILE
        fig = _px_fig(W, Hs)
        draw_row(fig, 0, W, Hs, arch, u, J["units"][str(u)], X, y, names, T[r])
        fig.savefig(os.path.join(d, f"{u}.jpg"), dpi=DPI, bbox_inches=None, pad_inches=0, facecolor="white",
                    pil_kwargs={"quality": 92})
        plt.close(fig)
    return fn, (W, H)


def seeds_fig(arch, J):
    units = J["seed_units"]
    seeds = (0, 1)
    W = len(SETTINGS) * len(seeds) * (TILE + GAP) + (len(SETTINGS) - 1) * GROUP_GAP
    H = 30 + len(units) * (TILE + GAP + 18)
    fig = _px_fig(W, H)
    fig.text(0, 1 - 4 / H, f"{arch}: FV seed stability (seeds 0 and 1 per setting)", fontsize=6.4, weight="bold", ha="left", va="top")
    for r, u in enumerate(units):
        info = J["units"][str(u)]
        y0 = 30 + r * (TILE + GAP + 18)
        fig.text(0, 1 - (y0 + 2) / H, f"unit {u} · {info['majority_class']} · {info['majority_count']} of 20 ({info['role']})",
                 fontsize=5.4, ha="left", va="top", color=CM.INK2)
        for i, st in enumerate(SETTINGS):
            for j, sd in enumerate(seeds):
                x = (i * len(seeds) + j) * (TILE + GAP) + i * GROUP_GAP
                ax = _tile(fig, x, y0 + 16, TILE, TILE, W, H, load_fv(arch, u, st, sd))
                _label(ax, f"{LABELS[st].split('-')[0]} s{sd} · {info['ratio'][st][f's{sd}']:.2f}×", color="#f2f2f2", bg="#333333")
    fn = os.path.join(C.FIGS, f"imnet_fv_seeds_{arch}.png")
    fig.savefig(fn, dpi=DPI, bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    return fn, (W, H)


def negpoles(arch, X, y, names, Spos, Sneg, n=6, k=8):
    Tp, Tn = top_k(Spos, K), top_k(Sneg, K)
    cp, vp = modal(y[Tp])
    cn, vn = modal(y[Tn])
    order = np.lexsort((-(cp + cn), -np.minimum(cp, cn)))
    units = order[:n]
    tile, lab_w = 150, 300
    W = lab_w + k * (tile + GAP) - GAP
    block = 2 * (tile + GAP) + 22
    head = 40
    H = head + n * block
    fig = _px_fig(W, H)
    fig.text(0, 1 - 4 / H, f"{ARCH_NAME[arch]}: the two poles of one neuron", fontsize=7.2, weight="bold", ha="left", va="top")
    fig.text(0, 1 - 22 / H, "the 6 units where both poles are most class-dominated (max of min(purity+, purity−) on the top 20 of "
             "each pole); 8 strongest val images per pole, in order; count = majority class in the top 20",
             fontsize=5.2, color=CM.INK2, ha="left", va="top")
    rows = []
    for r, u in enumerate(units):
        y0 = head + r * block
        for p, (T, cnt, val, sign) in enumerate(((Tp, cp, vp, "+"), (Tn, cn, vn, "−"))):
            yy = y0 + p * (tile + GAP)
            fig.text(0, 1 - (yy + tile * 0.35) / H, f"{sign} {names[val[u]]}", fontsize=8, weight="bold",
                     color=PICK_COL if sign == "+" else "#c0392b", ha="left", va="center")
            fig.text(0, 1 - (yy + tile * 0.62) / H, f"unit {u} · {cnt[u]} of 20", fontsize=5.6, color=CM.MUTED, ha="left", va="center")
            for c in range(k):
                ax = _tile(fig, lab_w + c * (tile + GAP), yy, tile, tile, W, H, np.asarray(X[T[u][c]]))
                if y[T[u][c]] != val[u]:
                    _label(ax, names[y[T[u][c]]][:18], fontsize=4.2)
        rows.append(dict(unit=int(u), pos_class=names[vp[u]], pos_count=int(cp[u]), neg_class=names[vn[u]], neg_count=int(cn[u])))
    fn = os.path.join(C.FIGS, f"imnet_negpoles_{arch}.png")
    fig.savefig(fn, dpi=DPI, bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)
    if os.path.getsize(fn) > 8 * 2**20:
        Image.open(fn).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(fn, optimize=True)
    return fn, (W, H), rows


def shapes(arch, J, Spos, y, names):
    CM.setup_style()
    units = J["random"]
    N = Spos.shape[0]
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.4), sharex=True)
    rank = np.arange(1, N + 1)
    for ax, u in zip(axes.ravel(), units):
        s = np.sort(Spos[:, u].astype(np.float32))[::-1]
        ax.plot(rank, s / s.max(), color=CM.SERIES[0], lw=1.4, label="raw score / max")
        ax.plot(rank, 1 - (rank - 1) / (N - 1), color=CM.MUTED, lw=1.0, ls="--", label="percentile rank")
        info = J["units"][str(u)]
        ax.set_title(f"unit {u}", fontsize=8)
        ax.text(0.97, 0.93, f"{info['majority_class'][:16]} {info['majority_count']}/20", transform=ax.transAxes, ha="right",
                va="top", fontsize=6.5, color=CM.INK2)
        ax.set_xlim(0, N)
        ax.set_xticks([0, 25000, 50000])
        ax.set_xticklabels(["1", "25k", "50k"])
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("image rank", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("score / max", fontsize=8)
    axes[0, 0].legend(loc="lower left", fontsize=6.5)
    fig.suptitle(f"{ARCH_NAME[arch]}: raw shape of the positive pole over 50,000 val images (12 random units)",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fn = os.path.join(C.FIGS, f"imnet_shapes_{arch}.png")
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)
    return fn


def main(archs=ARCHS):
    J = json.load(open(os.path.join(C.RESULTS, "imnet_featvis.json")))
    X, y = C.images("imagenet_val")
    _, names = C.class_meta()
    summary = {}
    for arch in archs:
        L = C.last_layer(arch)
        z = np.load(os.path.join(C.cache_dir("imagenet_val"), f"{arch}_extract.npz"))
        Spos, Sneg = z[f"{L}_pos"], z[f"{L}_neg"]
        fn, wh = gallery(arch, J[arch], X, y, names, Spos)
        print(f"[gallery] {fn} {wh}", flush=True)
        fn, wh = seeds_fig(arch, J[arch])
        print(f"[gallery] {fn} {wh}", flush=True)
        fn, wh, rows = negpoles(arch, X, y, names, Spos, Sneg)
        print(f"[gallery] {fn} {wh}", flush=True)
        for r in rows:
            print(f"   unit {r['unit']}: + {r['pos_class']} {r['pos_count']}/20   - {r['neg_class']} {r['neg_count']}/20")
        fn = shapes(arch, J[arch], Spos, y, names)
        print(f"[gallery] {fn}", flush=True)
        summary[arch] = dict(negpoles=rows)
    J_fn = os.path.join(C.RESULTS, "imnet_featvis.json")
    for arch in archs:
        J[arch]["negpoles"] = summary[arch]["negpoles"]
    json.dump(J, open(J_fn, "w"), indent=1)


if __name__ == "__main__":
    main(*( [sys.argv[1:]] if len(sys.argv) > 1 else []))
