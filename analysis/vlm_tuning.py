"""C4: dial vs detector, from the raw tuning curve of each gallery unit.

The essay: "Some neurons hold their meaning far down the range; others drop off a cliff ...
Percentiles flatten that, and it's often what tells you whether you're looking at a dial, where
the middle matters, or a detector, where it mostly doesn't." It gives no rule, so the rule below is
ours.

For each of the 14 gallery units (census pole, pooled activation over all 10,800 groups) we take
the mean raw activation in 30 equal-width bins of the unit's best variable between its 1st and 99th
percentile (bins with >= 20 groups), rescale the curve to [0, 1] and classify:
(after 3-bin smoothing)
  tuned        if both ends of the curve are below 0.5 (the response peaks inside the range), or both
               are above 0.5 and the minimum lies inside (a dip);
  otherwise, walking inward from the higher end, w = width over which the curve falls from 0.75 to
  0.25 (as a fraction of the variable's range) and off = the range left beyond that fall:
  detector     if w < 0.25 and off >= 0.4 (flat, then steep: the middle does not matter);
  dial         if w >= 0.3 (the response changes across much of the range);
  intermediate otherwise.
Also reported: the fraction of the unit's raw activation range used by the bottom 90% of groups
(small = the unit idles near zero and only fires on rare groups).

Usage: uv run python analysis/vlm_tuning.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, load_states, savefig, setup_style, targets, INK, INK2, MUTED  # noqa: E402
from analysis.vlm_common import VDIR, group_index, n_used  # noqa: E402
from analysis.vlm_gallery import unit_values  # noqa: E402

NB = 30


def curve(x, a):
    lo, hi = np.percentile(x, [1, 99])
    edges = np.linspace(lo, hi, NB + 1)
    bi = np.clip(np.digitize(x, edges) - 1, 0, NB - 1)
    ok = (x >= lo) & (x <= hi)
    cnt = np.bincount(bi[ok], minlength=NB)
    m = np.bincount(bi[ok], a[ok], NB) / np.maximum(cnt, 1)
    c = 0.5 * (edges[1:] + edges[:-1])
    keep = cnt >= 20
    return c[keep], m[keep], (lo, hi)


def classify(c, m, rng_):
    span = rng_[1] - rng_[0]
    ms = np.convolve(np.pad(m, 1, mode="edge"), np.ones(3) / 3, mode="valid")    # 3-bin smoothing
    mn = (ms - ms.min()) / (np.ptp(ms) + 1e-12)
    out = dict(spearman=float(spearmanr(c, m).statistic))
    if mn[0] < 0.5 and mn[-1] < 0.5:
        k = int(np.argmax(mn))
        out.update(kind="tuned", peak_at=float(c[k]))
        return out
    kmin = int(np.argmin(mn))
    if mn[0] >= 0.5 and mn[-1] >= 0.5 and 2 < kmin < len(mn) - 3:
        out.update(kind="tuned", dip_at=float(c[kmin]))
        return out
    hi_left = mn[0] >= mn[-1]
    y, xs = (mn, c) if hi_left else (mn[::-1], c[::-1])
    below75 = np.where(y < 0.75)[0]
    below25 = np.where(y < 0.25)[0]
    i75 = below75[0] if len(below75) else len(y) - 1
    i25 = below25[0] if len(below25) else len(y) - 1
    w = abs(xs[i25] - xs[i75]) / span                 # width of the 75% -> 25% fall
    off = abs(xs[-1] - xs[i25]) / span                # range left after the fall (the flat "off" part)
    kind = "detector" if (w < 0.25 and off >= 0.4) else "dial" if w >= 0.3 else "intermediate"
    out.update(kind=kind, fall_span=float(w), off_span=float(off), high_end="low values" if hi_left else "high values")
    return out


def main():
    setup_style()
    import matplotlib.pyplot as plt
    st = load_states()
    Yall, vs = targets(st)
    names, labels = [v.name for v in vs], [v.label for v in vs]
    NG, _ = n_used("video")
    gi = group_index()[:NG]
    Y = Yall[gi[:, 0], gi[:, 3]]
    with open(os.path.join(VDIR, "gallery_spec.json")) as f:
        spec = json.load(f)
    res = {}
    fig, axes = plt.subplots(2, 7, figsize=(7.0, 3.3))
    for ax, sp in zip(axes.ravel(), spec):
        (part, l, pole), pos, neg = unit_values(sp["key"], sp["unit"], NG)
        a = pos if pole == "+" else -neg
        x = Y[:, names.index(sp["var"])]
        c, m, rg = curve(x, a)
        cl = classify(c, m, rg)
        p90 = np.percentile(a, 90)
        cl["bottom90_share_of_range"] = float((p90 - a.min()) / (np.ptp(a) + 1e-12))
        lab = f"{'V' if part == 'vision' else 'L'}{l}{'−' if pole == '−' else ''}#{sp['unit']}"
        res[lab] = dict(var=sp["var"], fam=sp["fam"], census_r2=sp["r2"], **cl)
        col = FAMILIES[sp["fam"]][0]
        sub = np.random.default_rng(0).choice(len(x), 1500, replace=False)
        ax.scatter(x[sub], a[sub], s=1, color=col, alpha=0.25, lw=0)
        ax.plot(c, m, color=INK, lw=1.2)
        ax.set_title(f"{lab}\n{sp['var']}", fontsize=6.3, fontweight="normal", loc="center")
        ax.text(0.5, 0.97, cl["kind"], transform=ax.transAxes, ha="center", va="top", fontsize=7, fontweight="bold",
                color={"dial": "#2a78d6", "detector": "#e34948", "tuned": "#6250d6", "intermediate": MUTED}[cl["kind"]],
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        ax.tick_params(labelsize=5, length=2, pad=1)
        ax.set_xlim(rg[0] - 0.05 * (rg[1] - rg[0]), rg[1] + 0.05 * (rg[1] - rg[0]))
        print(lab, res[lab], flush=True)
    axes[0, 0].set_ylabel("raw activation", fontsize=7)
    axes[1, 0].set_ylabel("raw activation", fontsize=7)
    fig.suptitle("C4  Raw tuning of the 14 gallery units: dial, detector or tuned (our rule)", x=0.01, ha="left",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig(fig, "vlm_tuning.png")
    with open(os.path.join(VDIR, "tuning.json"), "w") as f:
        json.dump(dict(rule=__doc__.split("ours.")[1].split("Usage")[0].strip(), units=res), f, indent=1)


if __name__ == "__main__":
    main()
