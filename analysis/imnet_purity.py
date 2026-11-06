"""imnet_ top-20 class purity, the closest automatic proxy we have for the essay's last-layer yardstick.

Essay ("A History of Microscope Errors"): InceptionV1's last layer "around 5% with tools of that era ... more like
50% on the same network [better tools] ... Retraining the same architecture with more reasonable modern
regularization it's more like 70%. In the ConvNeXt of the same size, it's about 99%." The neurons are shown as
"Each neuron's 20 strongest responses out of 50,000 images, in order."

Ours (OURS throughout): per unit pole (positive = spatial max of the pre-activation, negative = -spatial min), the
20 strongest of the 50,000 val images; the pole is class-dominated if >= 10 of the 20 share one ground-truth class,
superclass-dominated if >= 10 share one WordNet superclass (imnet_common.superclasses, K = 60: 125 groups; K = 130:
47 groups). This is stricter than human legibility: a neuron for "fur texture" or "water surface" is legible and
spans classes. Also: clean_tools.purity (top 1% = 500 images, chance-corrected, categorical factors only) and,
optionally, clean_tools.census against class and superclass indicators (both poles; raw vs percentile-rank).
Contact sheets of the positive poles of 24 random units per model: reports/figs/imnet_top20_<tag>.png.
Writes reports/results/imnet_purity.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import clean_tools as ct  # noqa: E402
from analysis import imnet_common as C  # noqa: E402
from analysis import imnet_extract as E  # noqa: E402


def top_k(S, k=20):
    """S [N, U] -> [U, k] indices of the k largest per column, in descending order."""
    S = np.asarray(S, np.float32)
    part = np.argpartition(-S, k, axis=0)[:k]                      # [k, U]
    vals = np.take_along_axis(S, part, 0)
    o = np.argsort(-vals, 0)
    return np.take_along_axis(part, o, 0).T


def modal(groups_of_top):
    """[U, k] group ids -> (modal count [U], modal group [U])."""
    U, k = groups_of_top.shape
    cnt = np.zeros(U, int)
    val = np.zeros(U, int)
    for u in range(U):
        b = np.bincount(groups_of_top[u])
        cnt[u], val[u] = b.max(), b.argmax()
    return cnt, val


def chance_dominated(groups, k=20, n=20000, seed=0):
    """Share of random k-sets of images that are >= 50% one group (null for the dominance test)."""
    rng = np.random.default_rng(seed)
    g = groups[rng.integers(0, len(groups), (n, k))]
    return float(np.mean([np.bincount(r).max() >= k / 2 for r in g]))


def pole_purity(S, y, sup60, sup130, k=20):
    T = top_k(S, k)
    out = {}
    for nm, g in (("class", y), ("super60", sup60[y]), ("super130", sup130[y])):
        cnt, val = modal(g[T])
        out[nm] = dict(dominated=float((cnt >= k / 2).mean()), mean_modal_share=float(cnt.mean() / k),
                       modal_count=cnt, modal_value=val)
    out["top"] = T
    return out


def contact_sheet(dataset, tag, poles, names, fn, n=24, seed=0):
    import matplotlib.pyplot as plt
    from analysis.common import setup_style
    setup_style()
    X, y = C.images(dataset)
    rng = np.random.default_rng(seed)
    U = poles["pos"]["top"].shape[0]
    pick = [(int(u), "pos") for u in rng.choice(U, n, replace=False)]   # the essay's view: strongest responses
    k = 20
    fig = plt.figure(figsize=(7.0, n * 0.62))
    for r, (u, p) in enumerate(pick):
        T = poles[p]["top"][u]
        cnt = poles[p]["class"]["modal_count"][u]
        val = poles[p]["class"]["modal_value"][u]
        s60 = poles[p]["super60"]["modal_count"][u]
        for c in range(k):
            ax = fig.add_axes([c / k, 1 - (r + 1) / n + 0.2 / n, 1 / k * 0.96, 1 / n * 0.56])
            ax.imshow(np.asarray(X[T[c]])[::4, ::4])
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.text(0.5, -0.04, names[y[T[c]]][:11], transform=ax.transAxes, ha="center", va="top", fontsize=3.2,
                    color="#555555")
        fig.text(0.0, 1 - r / n - 0.02 / n, f"unit {u}  {'+' if p == 'pos' else '−'} pole   modal class "
                 f"'{names[val]}' {cnt}/20   modal superclass {s60}/20", fontsize=5.2, va="top", ha="left",
                 color="#1c5cab" if cnt >= 10 else "#333333", weight="bold" if cnt >= 10 else "normal")
    fig.suptitle(f"{tag}: top-20 of {len(y):,} images, positive poles of 24 random units (blue = class-dominated)", y=1.0 + 0.6 / n,
                 fontsize=8, x=0.0, ha="left")
    fig.savefig(fn, dpi=200)
    plt.close(fig)
    from PIL import Image                     # photo grids: a 256-colour palette keeps the PNG ~4x smaller
    Image.open(fn).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(fn, optimize=True)


def census_block(S, y, sup60, sup130):
    fac = {"class": y, "super60": sup60[y], "super130": sup130[y]}
    kinds = {k: "cat" for k in fac}
    c = ct.census(S, fac, kinds, pair_factors=[])
    s = ct.census_summary(c)
    s["best_factor_share"] = {n: float((c["best_name"] == n).mean()) for n in fac}
    p = ct.purity(S, fac, kinds, mode="pct")
    s["purity_top1pct_one_job"] = float(p["one_job"].mean())
    s["purity_top1pct_mean"] = float(p["purity"].mean())
    R = np.empty(S.shape, np.float32)
    for i in range(0, S.shape[1], 256):
        R[:, i:i + 256] = ct.rank_transform(S[:, i:i + 256])
    s["rank_census"] = ct.census_summary(ct.census(R, fac, kinds, pair_factors=[]))
    return s


def run(dataset, arch, trained, do_census=True, sheets=True):
    z = E.load(dataset, arch, trained)
    _, y = C.images(dataset)
    _, names = C.class_meta()
    sup60, n60 = C.superclasses(60)
    sup130, _ = C.superclasses(130)
    tag = C.tag(arch, trained)
    res = dict(top1=float((z["pred"] == y).mean()), n_images=int(len(y)))
    for L in z["layers"]:
        L = str(L)
        poles = {p: pole_purity(z[f"{L}_{p}"], y, sup60, sup130) for p in ("pos", "neg")}
        r = {p: {g: {k: v for k, v in poles[p][g].items() if not isinstance(v, np.ndarray)}
                 for g in ("class", "super60", "super130")} for p in poles}
        r["both"] = {g: dict(dominated=(r["pos"][g]["dominated"] + r["neg"][g]["dominated"]) / 2)
                     for g in ("class", "super60", "super130")}
        r["n_units"] = int(z[f"{L}_pos"].shape[1])
        if do_census and L == C.last_layer(arch):
            r["census"] = {p: census_block(z[f"{L}_{p}"].astype(np.float32), y, sup60, sup130) for p in ("pos", "neg")}
        res[L] = r
        if sheets and L == C.last_layer(arch) and dataset == "imagenet_val":
            contact_sheet(dataset, tag, poles, names, os.path.join(C.FIGS, f"imnet_top20_{tag}.png"))
        print(f"[purity] {dataset} {tag} {L}: class-dominated +{r['pos']['class']['dominated']:.3f} "
              f"-{r['neg']['class']['dominated']:.3f}  super60 +{r['pos']['super60']['dominated']:.3f} "
              f"-{r['neg']['super60']['dominated']:.3f}", flush=True)
    return tag, res


def main(dataset="imagenet_val", archs=C.MODELS):
    fn = os.path.join(C.RESULTS, "imnet_purity.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    out.setdefault(dataset, {})
    _, y = C.images(dataset)
    sup60, _ = C.superclasses(60)
    sup130, _ = C.superclasses(130)
    out[dataset]["chance_dominated"] = dict(cls=chance_dominated(y), super60=chance_dominated(sup60[y]),
                                            super130=chance_dominated(sup130[y]))
    for arch in archs:
        for trained in (True, False):
            if not os.path.exists(os.path.join(C.cache_dir(dataset), f"{C.tag(arch, trained)}_extract.npz")):
                continue
            tag, r = run(dataset, arch, trained)
            out[dataset][tag] = r
            json.dump(out, open(fn, "w"), indent=1)
    return out


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["imagenet_val"]))
