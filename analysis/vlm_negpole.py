"""B2: the negative pole of the vision-tower GELU units, read from fc1 pre-activations.

The first-pass census read vision units after GELU, whose negative side is capped at -0.17, so the
- pole was invisible. vlm_recapture.py stored the pre-GELU fc1 output pooled per temporal group
(max, min, mean over 400 patches). Here:
  * census of the - pole (-min of the pre-activation) for all live units of each hooked block, next
    to the census of the + pole computed on the same pre-activations (max);
  * for the 7 gallery vision units: best variable of each pole, and whether the - pole carries a
    distinction the + pole does not (contact vs free: AUC of each pole for the contact flag);
  * figure: top + and top - examples (strongest patch outlined) with the - pole's tuning curve.

Usage: uv run python analysis/vlm_negpole.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, savefig, setup_style, INK, INK2  # noqa: E402
from analysis.vlm_common import VDIR, VIS_LAYERS, n_used, renderer, clip_frames  # noqa: E402
from analysis.vlm_probe import aligned_targets, residualise  # noqa: E402
from analysis.vlm_census import census_layer  # noqa: E402
from analysis.vlm_gallery import trail_image, draw_box  # noqa: E402


def auc(x, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, x))


def main():
    Y, vs, gi, st = aligned_targets()
    R, _ = residualise(Y, vs, st, gi)
    NG, _ = n_used("video")
    Y, gi, R = Y[:NG], gi[:NG], R[:NG]
    names, fams, labels = [v.name for v in vs], [v.family for v in vs], [v.label for v in vs]
    cflag = Y[:, names.index("contact")] > 0.5
    with open(os.path.join(VDIR, "gallery_spec.json")) as f:
        spec = [s for s in json.load(f) if s["part"] == "vision"]
    res = {"census": [], "gallery": {}}
    for l in VIS_LAYERS:
        Z = np.load(os.path.join(VDIR, f"video_vis_pre_L{l}.npy"), mmap_mode="r")
        for pole in "+−":
            A = np.asarray(Z[:NG, 0] if pole == "+" else -Z[:NG, 1], np.float32)
            c = census_layer(A, Y, R, fams, names)
            nl = int(c["live"].sum())
            fam_best = np.array(fams)[c["best"]]
            res["census"].append(dict(layer=l, pole=pole, n_live=nl, n_clean=int((c["cls"] == 2).sum()),
                                      n_partial=int((c["cls"] == 1).sum()),
                                      by_family_clean={f: int(((c["cls"] == 2) & (fam_best == f)).sum()) for f in FAMILIES if f != "none"},
                                      by_family_clean_partial={f: int(((c["cls"] >= 1) & (fam_best == f)).sum()) for f in FAMILIES if f != "none"}))
            if pole == "+":
                cp = c
            else:
                cn = c
            print(res["census"][-1], flush=True)
        # units whose two poles both reach partial: same or different family?
        live_both = np.where(cp["live"] & cn["live"])[0]
        ip = np.cumsum(cp["live"]) - 1
        inn = np.cumsum(cn["live"]) - 1
        bp, bn = cp["bestr"][ip[live_both]], cn["bestr"][inn[live_both]]
        fp, fn_ = np.array(fams)[cp["best"][ip[live_both]]], np.array(fams)[cn["best"][inn[live_both]]]
        both = (bp >= 0.25) & (bn >= 0.25)
        res["census"][-1]["n_both_poles_partial"] = int(both.sum())
        res["census"][-1]["n_both_poles_partial_diff_family"] = int((both & (fp != fn_)).sum())
        res["census"][-1]["n_neg_only_partial"] = int(((bn >= 0.25) & (bp < 0.25)).sum())
        # gallery units
        for sp in spec:
            if sp["layer"] != l:
                continue
            u = sp["unit"]
            pos = np.asarray(Z[:NG, 0, u], np.float32)
            neg = -np.asarray(Z[:NG, 1, u], np.float32)
            from analysis.common import binned_r2
            rp = np.array([binned_r2(Y[:, j], pos) for j in range(len(names))])
            rn = np.array([binned_r2(Y[:, j], neg) for j in range(len(names))])
            res["gallery"][f"V{l}#{u}"] = dict(
                census_var=sp["var"], census_r2_postgelu=sp["r2"],
                pos_best=names[int(rp.argmax())], pos_best_r2=float(rp.max()),
                neg_best=names[int(rn.argmax())], neg_best_r2=float(rn.max()),
                neg_r2_census_var=float(rn[names.index(sp["var"])]),
                neg_top3=[(names[j], float(rn[j])) for j in np.argsort(-rn)[:3]],
                contact_auc_pos=auc(pos, cflag), contact_auc_neg=auc(neg, cflag),
                corr_pos_neg=float(np.corrcoef(pos, neg)[0, 1]),
                neg_range=[float(neg.min()), float(neg.max())])
            print(f"V{l}#{u}", res["gallery"][f"V{l}#{u}"], flush=True)
    with open(os.path.join(VDIR, "negpole.json"), "w") as f:
        json.dump(res, f, indent=1)
    figure(res, Y, gi, st, names, labels, spec, NG)


def figure(res, Y, gi, st, names, labels, spec, NG):
    setup_style()
    import matplotlib.pyplot as plt
    K = 3
    n = len(spec)
    fig = plt.figure(figsize=(7.0, 1.0 * n + 0.5))
    gs = fig.add_gridspec(n, 2 + 2 * K, width_ratios=[1.3] + [1] * (2 * K) + [1.3], wspace=0.06, hspace=0.15,
                          left=0.01, right=0.95, top=1 - 0.45 / (1.0 * n + 0.5), bottom=0.02)
    for ri, sp in enumerate(spec):
        l, u = sp["layer"], sp["unit"]
        g = res["gallery"][f"V{l}#{u}"]
        Z = np.load(os.path.join(VDIR, f"video_vis_pre_L{l}.npy"), mmap_mode="r")
        Za = np.load(os.path.join(VDIR, f"video_vis_pre_L{l}_arg.npy"), mmap_mode="r")
        pos, neg = np.asarray(Z[:NG, 0, u], np.float32), -np.asarray(Z[:NG, 1, u], np.float32)
        ax = fig.add_subplot(gs[ri, 0]); ax.axis("off")
        ax.text(0, 0.95, f"V{l} #{u}", fontsize=8, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0, 0.68, f"+ {g['pos_best']} ({g['pos_best_r2']:.2f})", fontsize=6.3, va="top", transform=ax.transAxes,
                color=FAMILIES[sp["fam"]][0] if g["pos_best"] == sp["var"] else INK2)
        ax.text(0, 0.46, f"− {g['neg_best']} ({g['neg_best_r2']:.2f})", fontsize=6.3, va="top", transform=ax.transAxes, color=INK)
        ax.text(0, 0.24, f"contact AUC +{g['contact_auc_pos']:.2f} / −{g['contact_auc_neg']:.2f}", fontsize=5.6, va="top",
                transform=ax.transAxes, color=INK2)
        for side, (a, stat, col) in enumerate([(pos, 0, "#d6186e"), (neg, 1, "#0aa5a5")]):
            order, seen, ex = np.argsort(-a), set(), []
            for i in order:
                if gi[i, 0] not in seen:
                    ex.append(i); seen.add(gi[i, 0])
                if len(ex) == K:
                    break
            for j, i in enumerate(ex):
                ax = fig.add_subplot(gs[ri, 1 + side * K + j])
                clip = renderer()(st["q"][gi[i, 0]][clip_frames(gi[i, 4] - 4 * gi[i, 2])])
                draw_box(ax, trail_image(clip, 2 * gi[i, 2] + 1), int(Za[i, stat, u]), 20, col)
                ax.text(0.97, 0.04, f"{'+' if side == 0 else '−'}{a[i]:.1f}", transform=ax.transAxes, fontsize=5.5,
                        ha="right", color=col, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
                if ri == 0 and j == 0:
                    ax.set_title("+ pole (max pre-act.)" if side == 0 else "− pole (−min pre-act.)", fontsize=7,
                                 loc="left", fontweight="normal", color=INK2)
        ax = fig.add_subplot(gs[ri, -1])
        x = Y[:, names.index(g["neg_best"])]
        sub = np.random.default_rng(0).choice(len(x), 1500, replace=False)
        ax.scatter(x[sub], neg[sub], s=1.2, color="#0aa5a5", alpha=0.3, lw=0)
        qs = np.quantile(x, np.linspace(0, 1, 21))
        bi = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, 19)
        mx = [x[bi == b].mean() for b in range(20) if (bi == b).any()]
        my = [neg[bi == b].mean() for b in range(20) if (bi == b).any()]
        ax.plot(mx, my, color=INK, lw=1.1)
        ax.tick_params(labelsize=5, length=2, pad=1)
        ax.yaxis.tick_right()
        ax.text(0.03, 0.97, g["neg_best"], transform=ax.transAxes, fontsize=6, va="top", color=INK2,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        if ri == 0:
            ax.set_title("− pole tuning", fontsize=7, loc="left", fontweight="normal", color=INK2)
    fig.suptitle("Vision units read before GELU: the − pole (B2)", x=0.01, ha="left", fontsize=9.5, fontweight="bold", y=0.999)
    savefig(fig, "vlm_negpole.png")


if __name__ == "__main__":
    main()
