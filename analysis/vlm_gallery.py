"""Distill-style neuron gallery for Qwen3-VL-4B units on the arm videos.

For each selected unit: the k temporal groups with the highest pooled activation (distinct
episodes) and the k with the lowest (for SwiGLU "−" units, the most negative raw values =
the unit's other pole). Each example shows the later frame of the group with a fading motion
trail of the clip's earlier frames and the unit's spatial activation map over the visual
tokens (20x20 vision patches or 10x10 LM tokens) as an overlay. Right: tuning curve of the
pooled activation against the unit's best-explaining variable over all groups.

Usage: uv run python analysis/vlm_gallery.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, load_states, savefig, setup_style, targets, INK, INK2, MUTED  # noqa: E402
from analysis.vlm_common import VDIR, n_used, renderer, group_index, clip_frames  # noqa: E402

K_TOP, K_BOT = 4, 2


def load_census():
    z = np.load(os.path.join(VDIR, "census_units.npz"))
    with open(os.path.join(VDIR, "census.json")) as f:
        summ = json.load(f)
    return z, summ


def unit_values(key, u, NG):
    mode, part, L, pole = key.split("_")
    l = int(L[1:])
    fn = f"{mode}_vis_mlp_L{l}.npy" if part == "vision" else f"{mode}_llm_mlp_L{l}.npy"
    Z = np.load(os.path.join(VDIR, fn), mmap_mode="r")
    pos, neg = np.asarray(Z[:NG, 0, u], np.float32), np.asarray(Z[:NG, 1, u], np.float32)
    return (part, l, "+" if pole == "pos" else "−"), pos, neg


def select_units(z, summ, names, fams, n=12):
    """Pick a diverse set: best clean unit per (stream, family) prioritising physics families."""
    cands = []
    for s in summ:
        if s["mode"] != "video":
            continue
        k = s["key"]
        idx, best, bestr, conj = z[f"{k}__idx"], z[f"{k}__best"], z[f"{k}__bestr"], z[f"{k}__conj"]
        for f in ["posture", "velocity", "accel", "gravity", "interaction", "contact", "applied", "energy"]:
            m = np.where(np.array(fams)[best] == f)[0]
            if len(m):
                j = m[np.argmax(bestr[m])]
                cands.append(dict(key=k, unit=int(idx[j]), var=names[best[j]], fam=f, r2=float(bestr[j]),
                                  conj=bool(conj[j]), part=s["part"], layer=s["layer"], pole=s["pole"]))
    return cands


BAND_FAMS = ["posture", "velocity", "gravity", "contact"]


def load_spec(names, fams):
    """data/vlm/gallery_spec.json; if missing, build it: best vision and best LM unit per family
    (binned R² ≥ 0.1), with the quantile-band flag on the best unit of each BAND_FAMS family."""
    path = os.path.join(VDIR, "gallery_spec.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    z, summ = load_census()
    cands = select_units(z, summ, names, fams)
    spec = []
    for part in ["vision", "LM"]:
        for f in ["posture", "velocity", "accel", "gravity", "interaction", "contact", "applied", "energy"]:
            c = [d for d in cands if d["part"] == part and d["fam"] == f and d["r2"] >= 0.1]
            if c:
                spec.append(max(c, key=lambda d: d["r2"]))
    for f in BAND_FAMS:
        c = [d for d in spec if d["fam"] == f]
        if c:
            max(c, key=lambda d: d["r2"])["bands"] = True
    with open(path, "w") as fh:
        json.dump(spec, fh, indent=1)
    print(f"wrote {path}: {len(spec)} units", flush=True)
    return spec


def trail_image(frames, upto):
    """frames [8,H,W,3]; composite of frames[:upto+1] with fading trail."""
    base = frames[upto].astype(np.float32)
    def armmask(f):
        f = f.astype(np.float32)
        return (f.max(-1) - f.min(-1)) > 60
    m_now = armmask(frames[upto])
    out = base.copy()
    for k in range(upto - 1, -1, -1):
        a = 0.55 * (k + 1) / (upto + 1)
        m = armmask(frames[k]) & ~m_now
        out[m] = (1 - a) * out[m] + a * frames[k][m].astype(np.float32)
    return out.astype(np.uint8)


def draw_example(ax, img, amap, vmax, cmap):
    from scipy.ndimage import zoom
    ax.imshow(img)
    H = img.shape[0]
    up = np.kron(amap, np.ones((H // amap.shape[0], H // amap.shape[1])))
    a = np.clip(up / (vmax + 1e-8), 0, 1)
    rgba = cmap(np.ones_like(a))
    rgba[..., 3] = 0.65 * a
    ax.imshow(rgba, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)


def pick_examples(a, pos, neg, pole, part, gi):
    """Top: strongest values of the unit's pole. Bottom: for SwiGLU (LM) units the opposite pole
    (most negative min for '+', largest max for '−'); for GELU vision units the weakest groups.
    Returns example indices and, per example, (which stored stat: 0=max/1=min)."""
    def distinct(order, k):
        out, seen = [], set()
        for i in order:
            if gi[i, 0] not in seen:
                out.append(i); seen.add(gi[i, 0])
            if len(out) == k:
                break
        return out
    top = distinct(np.argsort(-a), K_TOP)
    if part == "LM":
        bot = distinct(np.argsort(neg) if pole == "+" else np.argsort(-pos), K_BOT)
        stat = [0 if pole == "+" else 1] * K_TOP + [1 if pole == "+" else 0] * K_BOT
    else:
        bot = distinct(np.argsort(a), K_BOT)
        stat = [0] * (K_TOP + K_BOT)
    return top + bot, stat


def main_cache(spec=None):
    """Gallery built only from cached activations (no model forward passes): the location of
    the unit's strongest token (stored argmax / argmin index) is outlined on the frame."""
    setup_style()
    st = load_states()
    Yall, vs = targets(st)
    names, fams, labels = [v.name for v in vs], [v.family for v in vs], [v.label for v in vs]
    NG, _ = n_used("video")
    gi = group_index()[:NG]
    Y = Yall[gi[:, 0], gi[:, 3]]
    if spec is None:
        spec = load_spec(names, fams)
    rows = []
    for sp in spec:
        (part, l, pole), pos, neg = unit_values(sp["key"], sp["unit"], NG)
        a = pos if pole == "+" else -neg
        ex, stat = pick_examples(a, pos, neg, pole, part, gi)
        fn = f"video_vis_mlp_L{l}_arg.npy" if part == "vision" else f"video_llm_mlp_L{l}_arg.npy"
        Zarg = np.load(os.path.join(VDIR, fn), mmap_mode="r")
        args = [int(Zarg[i, k, sp["unit"]]) for i, k in zip(ex, stat)]
        clips = [renderer()(st["q"][gi[i, 0]][clip_frames(gi[i, 4] - 4 * gi[i, 2])]) for i in ex]
        rows.append(dict(sp=sp, a=a, pos=pos, neg=neg, ex=ex, clips=clips, maps=None, args=args,
                         grid=20 if part == "vision" else 10, raw=(pos, neg), stat=stat))
    draw(rows, gi, Y, names, fams, labels)


def draw_box(ax, img, arg, grid, color):
    import matplotlib.patches as mp
    ax.imshow(img)
    H = img.shape[0]
    c = H / grid
    r_, q_ = divmod(arg, grid)
    ax.add_patch(mp.Rectangle((q_ * c, r_ * c), c, c, fill=False, ec=color, lw=1.4))
    ax.add_patch(mp.Circle(((q_ + 0.5) * c, (r_ + 0.5) * c), 2.2 * c, fill=False, ec=color, lw=0.8, alpha=0.6))
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.grid(False)


def main(spec=None):
    setup_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import torch
    from analysis.vlm_common import load_model
    from analysis.vlm_capture import capture
    st = load_states()
    Yall, vs = targets(st)
    names, fams, labels = [v.name for v in vs], [v.family for v in vs], [v.label for v in vs]
    NG, _ = n_used("video")
    gi = group_index()[:NG]
    Y = Yall[gi[:, 0], gi[:, 3]]
    z, summ = load_census()
    if spec is None:
        spec = load_spec(names, fams)
    model, proc = load_model()
    rows = []
    for sp in spec:
        (part, l, pole), pos, neg = unit_values(sp["key"], sp["unit"], NG)
        a = pos if pole == "+" else -neg          # the pooled pole used in the census
        ex, stat = pick_examples(a, pos, neg, pole, part, gi)
        clips = [renderer()(st["q"][gi[i, 0]][clip_frames(gi[i, 4] - 4 * gi[i, 2])]) for i in ex]
        maps, _ = capture(model, proc, clips, [(part, l, sp["unit"])])
        m = maps[0][np.arange(len(ex)), gi[ex, 2]]           # [n_ex, h, w]
        rows.append(dict(sp=sp, a=a, pos=pos, neg=neg, ex=ex, clips=clips, maps=m, raw=(pos, neg), stat=stat))
        print("captured", sp["key"], sp["unit"], flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    np.savez_compressed(os.path.join(VDIR, "gallery_cache.npz"), **{f"r{i}_{k}": r[k] for i, r in enumerate(rows) for k in ("ex", "maps")})
    draw(rows, gi, Y, names, fams, labels)


def draw(rows, gi, Y, names, fams, labels):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    halves = [("vlm_gallery_vision.png", [r for r in rows if r["sp"]["key"].split("_")[1] == "vision"],
               "A  Vision-encoder MLP units (GELU; spatial map = 20×20 patches)"),
              ("vlm_gallery_lm.png", [r for r in rows if r["sp"]["key"].split("_")[1] == "LM"],
               "B  Language-model MLP units at visual tokens (SwiGLU; map = 10×10 tokens)")]
    for fname, rr, title in halves:
        if not rr:
            continue
        n = len(rr)
        fig = plt.figure(figsize=(7.0, 1.08 * n + 0.45))
        gs = fig.add_gridspec(n, 1 + K_TOP + K_BOT + 1, width_ratios=[1.25] + [1] * (K_TOP + K_BOT) + [1.3],
                              wspace=0.06, hspace=0.12, left=0.01, right=0.99, top=1 - 0.42 / (1.08 * n + 0.45), bottom=0.02)
        for ri, r in enumerate(rr):
            sp = r["sp"]
            fam = sp["fam"]
            col = FAMILIES[fam][0]
            cmap_hot = LinearSegmentedColormap.from_list("h", ["#ffffff", "#d6186e"])
            cmap_cold = LinearSegmentedColormap.from_list("c", ["#ffffff", "#0aa5a5"])
            ax = fig.add_subplot(gs[ri, 0]); ax.axis("off")
            part, L, pole = sp["key"].split("_")[1:4]
            lname = ("V" if part == "vision" else "L") + L[1:]
            ax.text(0, 0.95, f"{lname} #{sp['unit']}" + (" (−)" if pole == "neg" else ""), fontsize=8, fontweight="bold",
                    va="top", transform=ax.transAxes, color=INK)
            import textwrap
            ax.text(0, 0.66, textwrap.fill(sp.get("title", labels[names.index(sp["var"])]), 20), fontsize=6.6, va="top",
                    color=col, transform=ax.transAxes, linespacing=1.0)
            ax.text(0, 0.22, f"binned R² {sp['r2']:.2f}" + ("\nconjunction" if sp.get("conj") else ""), fontsize=6,
                    va="top", color=INK2, transform=ax.transAxes)
            sign = 1 if pole == "pos" else -1
            vmax = np.percentile(np.abs(r["a"]), 99.5)
            for j in range(K_TOP + K_BOT):
                ax = fig.add_subplot(gs[ri, 1 + j])
                i = r["ex"][j]
                g = gi[i, 2]
                img = trail_image(r["clips"][j], 2 * g + 1)
                if r["maps"] is not None:
                    amap = sign * r["maps"][j] if (j < K_TOP or part == "vision") else -sign * r["maps"][j]
                    draw_example(ax, img, amap, vmax, cmap_hot if j < K_TOP else cmap_cold)
                else:
                    draw_box(ax, img, r["args"][j], r["grid"], "#d6186e" if j < K_TOP else "#0aa5a5")
                    v = r["raw"][r["stat"][j]][i]
                    ax.text(0.97, 0.04, f"a={v:+.2f}", transform=ax.transAxes, fontsize=5.5, ha="right",
                            color="#d6186e" if j < K_TOP else "#0aa5a5",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
                val = Y[i, names.index(sp["var"])]
                ax.text(0.03, 0.04, f"{val:+.2f}", transform=ax.transAxes, fontsize=5.5, color=INK2,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
                if ri == 0:
                    ax.set_title("top activations" if j == 0 else ("opposite pole / weakest" if j == K_TOP else ""),
                                 fontsize=7, loc="left", fontweight="normal", color=INK2)
            ax = fig.add_subplot(gs[ri, -1])
            x = Y[:, names.index(sp["var"])]
            a = r["a"]
            qs = np.quantile(x, np.linspace(0, 1, 21))
            bi = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, 19)
            sub = np.random.default_rng(0).choice(len(x), min(1500, len(x)), replace=False)
            ax.scatter(x[sub], a[sub], s=1.2, color=col, alpha=0.25, lw=0)
            mx = [x[bi == b].mean() for b in range(20) if (bi == b).any()]
            my = [a[bi == b].mean() for b in range(20) if (bi == b).any()]
            ax.plot(mx, my, color=INK, lw=1.2)
            ax.tick_params(labelsize=5, length=2, pad=1)
            ax.text(0.03, 0.97, sp["var"], transform=ax.transAxes, fontsize=6, va="top", color=INK2,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
            ax.yaxis.tick_right()
            if ri == 0:
                ax.set_title("tuning (all groups)", fontsize=7, loc="left", fontweight="normal", color=INK2)
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", y=0.999)
        savefig(fig, fname)


def main_redraw():
    """Redraw the gallery from data/vlm/gallery_cache.npz (spatial maps captured by main()); no GPU."""
    setup_style()
    st = load_states()
    Yall, vs = targets(st)
    names, fams, labels = [v.name for v in vs], [v.family for v in vs], [v.label for v in vs]
    NG, _ = n_used("video")
    gi = group_index()[:NG]
    Y = Yall[gi[:, 0], gi[:, 3]]
    spec = load_spec(names, fams)
    z = np.load(os.path.join(VDIR, "gallery_cache.npz"))
    rows = []
    for i, sp in enumerate(spec):
        (part, l, pole), pos, neg = unit_values(sp["key"], sp["unit"], NG)
        a = pos if pole == "+" else -neg
        ex = z[f"r{i}_ex"]
        _, stat = pick_examples(a, pos, neg, pole, part, gi)
        clips = [renderer()(st["q"][gi[j, 0]][clip_frames(gi[j, 4] - 4 * gi[j, 2])]) for j in ex]
        rows.append(dict(sp=sp, a=a, pos=pos, neg=neg, ex=ex, clips=clips, maps=z[f"r{i}_maps"], raw=(pos, neg), stat=stat))
    draw(rows, gi, Y, names, fams, labels)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "redraw":
        main_redraw()
        sys.exit()
    # default: cache-only gallery (argmax-token outline). "maps" re-runs the model for full spatial maps.
    if len(sys.argv) > 1 and sys.argv[1] == "maps":
        main()
    else:
        main_cache()
