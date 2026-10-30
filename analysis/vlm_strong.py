"""C1: strong-activation census.

The first-pass census scores binned R² over all 10,800 groups, so it is dominated by weak
activations. The essay's numbers ("~90% of neuron poles have one main job", "closer to 97%
non-arbitrary" for strong activations) are about strong activations. Here, per census key (layer x
pole x stream, including the random-init vision control), for a seeded sample of up to 1,000 live
units, we recompute binned R² against the 32 targets on two subsets of groups:
  strong half: a >= median + 0.5 (max - median)   (ours: "half the unit's max" above its baseline,
               since many units have a large offset)
  top 1%:      the 108 groups with the largest a
with 10 quantile bins (fewer if n < 100: n // 10). Units with < 30 strong groups are counted as
"too sparse". Classes as in the census: clean >= 0.5, partial 0.25-0.5, else unexplained.
Because a 10-bin fit on ~100 points has a positive bias, every unit also gets a null: the same fit
with the targets shuffled across the subset's groups; we report the null rates next to the real ones.
We also record whether the strong-subset best variable equals the all-data label, or its family.
Finally the B3 blind descriptions add a "described, not one of our 32 variables" bucket for the
units they cover (see vlm.md §4).

Usage: uv run python analysis/vlm_strong.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import savefig, setup_style, FAMILIES, INK2, MUTED  # noqa: E402
from analysis.vlm_common import VDIR, n_used  # noqa: E402
from analysis.vlm_probe import aligned_targets  # noqa: E402

N_SAMPLE = 1000


def r2_subset(a, Ys, b):
    """binned R² of a against every column of Ys (quantile bins, ties kept together)."""
    out = np.zeros(Ys.shape[1])
    va = a.var() + 1e-12
    for j in range(Ys.shape[1]):
        x = Ys[:, j]
        edges = np.quantile(x, np.linspace(0, 1, b + 1))
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, b - 1)
        cnt = np.bincount(idx, minlength=b)
        m = np.bincount(idx, a, b) / np.maximum(cnt, 1)
        out[j] = 1 - (a - m[idx]).var() / va
    return out


def cls(r):
    return 2 if r >= 0.5 else 1 if r >= 0.25 else 0


def main():
    Y, vs, gi, st = aligned_targets()
    NG, _ = n_used("video")
    Y = Y[:NG]
    names, fams = [v.name for v in vs], [v.family for v in vs]
    z = np.load(os.path.join(VDIR, "census_units.npz"))
    with open(os.path.join(VDIR, "census.json")) as f:
        summ = json.load(f)
    rng = np.random.default_rng(0)
    res = {"keys": {}}
    for s in summ:
        key = s["key"]
        if s["mode"] == "image":
            continue
        fn = f"{s['mode']}_vis_mlp_L{s['layer']}.npy" if s["part"] == "vision" else f"video_llm_mlp_L{s['layer']}.npy"
        Z = np.load(os.path.join(VDIR, fn), mmap_mode="r")
        idx, best, bestr = z[f"{key}__idx"], z[f"{key}__best"], z[f"{key}__bestr"]
        pick = np.sort(rng.choice(len(idx), min(N_SAMPLE, len(idx)), replace=False))
        A = np.asarray(Z[:NG, 0 if s["pole"] == "+" else 1][:, idx[pick]], np.float32)
        if s["pole"] != "+":
            A = -A
        rec = {"all": [], "strong": [], "top1": [], "strong_null": [], "top1_null": [], "same_var": [], "same_fam": [],
               "n_strong": []}
        for k in range(len(pick)):
            a = A[:, k]
            rec["all"].append(cls(bestr[pick[k]]))
            med, mx = np.median(a), a.max()
            subsets = {"strong": np.where(a >= med + 0.5 * (mx - med))[0], "top1": np.argsort(-a)[:NG // 100]}
            rec["n_strong"].append(len(subsets["strong"]))
            for nm, sub in subsets.items():
                if len(sub) < 30 or a[sub].std() < 1e-6:
                    rec[nm].append(-1); rec[nm + "_null"].append(-1)
                    if nm == "strong":
                        rec["same_var"].append(False); rec["same_fam"].append(False)
                    continue
                b = 10 if len(sub) >= 100 else len(sub) // 10
                r = r2_subset(a[sub], Y[sub], b)
                rn = r2_subset(a[sub], Y[rng.permutation(sub)], b)
                rec[nm].append(cls(r.max())); rec[nm + "_null"].append(cls(rn.max()))
                if nm == "strong":
                    rec["same_var"].append(bool(r.argmax() == best[pick[k]]))
                    rec["same_fam"].append(bool(fams[r.argmax()] == fams[best[pick[k]]]))
        def frac(v):
            v = np.array(v)
            n = len(v)
            return dict(clean=float((v == 2).mean()), partial=float((v == 1).mean()),
                        unexplained=float((v == 0).mean()), too_sparse=float((v == -1).mean()), n=n)
        res["keys"][key] = dict(part=s["part"], mode=s["mode"], layer=s["layer"], pole=s["pole"], n_live=s["n_live"],
                                n_sampled=len(pick), all=frac(rec["all"]), strong=frac(rec["strong"]), top1=frac(rec["top1"]),
                                strong_null=frac(rec["strong_null"]), top1_null=frac(rec["top1_null"]),
                                strong_best_is_census_var=float(np.mean(rec["same_var"])),
                                strong_best_same_family=float(np.mean(rec["same_fam"])),
                                median_n_strong=float(np.median(rec["n_strong"])))
        print(key, {k: v for k, v in res["keys"][key].items() if k in ("all", "strong", "top1", "strong_null")}, flush=True)
    with open(os.path.join(VDIR, "strong.json"), "w") as f:
        json.dump(res, f, indent=1)
    figure(res)


def figure(res=None):
    setup_style()
    import matplotlib.pyplot as plt
    if res is None:
        with open(os.path.join(VDIR, "strong.json")) as f:
            res = json.load(f)
    groups = [("Vision, video", lambda d: d["part"] == "vision" and d["mode"] == "video"),
              ("Vision, random init", lambda d: d["mode"] == "randvis"),
              ("LM + pole", lambda d: d["part"] == "LM" and d["pole"] == "+"),
              ("LM − pole", lambda d: d["part"] == "LM" and d["pole"] == "−")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 3.1), sharey=True)
    cols = {"clean": FAMILIES["posture"][0], "partial": "#9ec5f4", "unexplained": FAMILIES["none"][0], "too_sparse": "#f0efec"}
    for ax, (ttl, sel) in zip(axes, groups):
        ks = [k for k, d in res["keys"].items() if sel(d)]
        x = 0
        ticks, tl = [], []
        for k in ks:
            d = res["keys"][k]
            for j, sub in enumerate(("all", "strong", "top1")):
                bottom = 0
                for c in ("clean", "partial", "unexplained", "too_sparse"):
                    v = d[sub][c]
                    ax.bar(x + j * 0.27, v, 0.25, bottom=bottom, color=cols[c], label=c.replace("_", " ") if (k == ks[0] and j == 0 and ax is axes[0]) else None)
                    bottom += v
                if sub != "all":
                    nl = d[sub + "_null"]["clean"] + d[sub + "_null"]["partial"]
                    ax.plot([x + j * 0.27 - 0.13, x + j * 0.27 + 0.13], [nl, nl], color="#e34948", lw=1.6)
            ticks.append(x + 0.27); tl.append(f"{'V' if d['part'] == 'vision' else 'L'}{d['layer']}")
            x += 1
        ax.set_xticks(ticks, tl, fontsize=7)
        ax.set_title(ttl, fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("fraction of sampled units", fontsize=7)
    h, l_ = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([], [], color="#e34948", lw=1)); l_.append("clean+partial rate, shuffled targets")
    fig.legend(h, l_, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("C1  Census on all groups vs strong half vs top 1% (three bars per layer, left to right)", x=0.01,
                 ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, "vlm_strong.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fig":
        figure()
    else:
        main()
