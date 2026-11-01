"""vis_ figures and the merged reports/results/vis.json (ladder table with seed means and 95% CIs).

Reads vis_census.json, vis_featvis.json, vis_ablate.json, vis_dict.json and the per-model caches.
Figures: vis_ladder.png, vis_census_depth.png, vis_jobs.png, vis_weak.png, vis_dict.png, vis_dial.png,
vis_fv_legibility.png, vis_gallery_shapes.png, vis_gallery_cifar.png, vis_shapes_examples.png.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import INK, INK2, MUTED, SERIES, savefig, setup_style  # noqa: E402
from analysis.vis_common import CACHE, RESULTS, available, cached_scores, data, mean_ci, train_log  # noqa: E402
from models.vis_data import CIFAR_CLASSES  # noqa: E402

TASK = {"shape", "color", "texture", "curved", "warm", "textured", "label"}
STEP_COL = {"old_old": "#b5b3ab", "old": SERIES[1], "recipe": SERIES[3], "arch": SERIES[0]}
LADDER = [("old_old", "old", "M_old\n2017-era tools"), ("old", "old", "M_old\nfixed tools"),
          ("recipe", "recipe", "M_recipe\nfixed tools"), ("arch", "arch", "M_arch\nfixed tools")]


def J(name):
    fn = os.path.join(RESULTS, name)
    return json.load(open(fn)) if os.path.exists(fn) else {}


def seeds(dname, tag):
    return available(dname, tag)


def collect(dname, tag, fn, init=False):
    vals = []
    for s in seeds(dname, tag):
        try:
            v = fn(f"{dname}_{tag}_s{s}{'_init' if init else ''}")
        except KeyError:
            continue
        if v is not None:
            vals.append(v)
    return vals


def clean_task(name, view, g="late"):
    """Share of the layer's poles that are clean on a TASK factor (shapes: shape/colour/texture and their
    groupings/class; CIFAR: class). view "both": both pre-activation poles; "old": post-activation + pole only,
    counted over all poles (negative poles uninspected); "old_pos"/"pos": per positive pole."""
    fn = os.path.join(CACHE, name + "_census.pkl")
    if not os.path.exists(fn):
        raise KeyError(name)
    c = pickle.load(open(fn, "rb"))[g]

    def frac(cc):
        return ((cc["label"] == "clean") & np.isin(cc["best_name"], list(TASK))).mean()
    if view == "both":
        return (frac(c["pos"]["census"]) + frac(c["neg"]["census"])) / 2
    if view == "pos":
        return frac(c["pos"]["census"])
    if view == "old":
        return frac(c["old"]["census"]) / 2
    return frac(c["old"]["census"])


def ladder_metrics(dname):
    cen, fv = J("vis_census.json"), J("vis_featvis.json")
    out = {}
    for key, tag, label in LADDER:
        if not seeds(dname, tag):
            continue
        old = key == "old_old"
        row = {"label": label.replace("\n", ", "), "model": tag}
        for init in (False, True):
            sfx = "_init" if init else ""
            if old:
                interp = collect(dname, tag, lambda n: fv[n]["verdict"]["old_tools_poles"], init)
                interp_pos = collect(dname, tag, lambda n: fv[n]["verdict"]["old_tools_pos"], init)
                clean = collect(dname, tag, lambda n: cen[n]["late"]["old_tools"]["clean"] / 2, init)
                clean_pos = collect(dname, tag, lambda n: cen[n]["late"]["old_tools"]["clean"], init)
                task = collect(dname, tag, lambda n: clean_task(n, "old"), init)
            else:
                interp = collect(dname, tag, lambda n: fv[n]["verdict"]["fixed_tools_poles"], init)
                interp_pos = collect(dname, tag, lambda n: fv[n]["verdict"]["fixed_tools_pos"], init)
                clean = collect(dname, tag, lambda n: cen[n]["late"]["both"]["clean"], init)
                clean_pos = collect(dname, tag, lambda n: cen[n]["late"]["pos"]["clean"], init)
                task = collect(dname, tag, lambda n: clean_task(n, "both"), init)
            row["interp" + sfx] = mean_ci(interp) + (len(interp),) if interp else None
            row["interp_pos" + sfx] = mean_ci(interp_pos) + (len(interp_pos),) if interp_pos else None
            row["clean" + sfx] = mean_ci(clean) + (len(clean),) if clean else None
            row["clean_pos" + sfx] = mean_ci(clean_pos) + (len(clean_pos),) if clean_pos else None
            row["clean_task" + sfx] = mean_ci(task) + (len(task),) if task else None
        accs = [train_log(dname, tag, s)["test_acc"] for s in seeds(dname, tag)]
        row["test_acc"] = mean_ci(accs)
        row["params"] = train_log(dname, tag, seeds(dname, tag)[0])["params"]
        out[key] = row
    return out


def bar_panel(ax, rows, metric, title, ylabel):
    xs = np.arange(len(LADDER))
    for i, (key, tag, label) in enumerate(LADDER):
        r = rows.get(key, {})
        if r.get(metric):
            m, ci, n = r[metric]
            ax.bar(i - 0.18, m, 0.36, color=STEP_COL[key], yerr=ci, capsize=2, error_kw=dict(lw=0.8, ecolor=INK2))
            ax.text(i - 0.18, m + ci + 0.02, f"{100 * m:.0f}", ha="center", fontsize=7, color=INK)
        if r.get(metric + "_init"):
            m, ci, n = r[metric + "_init"]
            ax.bar(i + 0.18, m, 0.36, color="white", edgecolor=STEP_COL[key], hatch="////", lw=0.8,
                   yerr=ci, capsize=2, error_kw=dict(lw=0.8, ecolor=INK2))
            ax.text(i + 0.18, m + ci + 0.02, f"{100 * m:.0f}", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, _, l in LADDER], fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def fig_ladder(L):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for r, dname in enumerate(("shapes", "cifar")):
        bar_panel(axs[r, 0], L[dname], "interp", f"{dname}: poles read correctly (tool verdict)",
                  "share of late-layer poles")
        bar_panel(axs[r, 1], L[dname], "clean_task", f"{dname}: poles clean on a task factor (census)",
                  "share of late-layer poles")
    axs[0, 0].legend(handles=[Patch(color=SERIES[0], label="trained (mean of seeds, 95% CI)"),
                              Patch(facecolor="white", edgecolor=INK2, hatch="////", label="random-init control")],
                     loc="upper left", fontsize=7)
    fig.tight_layout()
    return savefig(fig, "vis_ladder.png")


def fig_census_depth():
    import matplotlib.pyplot as plt
    cen = J("vis_census.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, dname in zip(axs, ("shapes", "cifar")):
        for j, tag in enumerate(("old", "recipe", "arch")):
            for init in (False, True):
                ms, cis = [], []
                for g in ("early", "mid", "late"):
                    m, ci = mean_ci(collect(dname, tag, lambda n: cen[n][g]["both"]["clean"], init) or [np.nan])
                    ms.append(m)
                    cis.append(ci)
                ax.errorbar(np.arange(3) + (j - 1) * 0.08, ms, yerr=cis, color=STEP_COL[tag], marker="o", ms=4,
                            ls="--" if init else "-", lw=1.4 if not init else 1.0, alpha=1 if not init else 0.6,
                            label=f"{tag}{' random-init' if init else ''}", capsize=2)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["early", "mid", "late"])
        ax.set_ylim(0, 1)
        ax.set_title(f"{dname}: clean poles by depth")
        ax.set_ylabel("share of poles (both signs)")
    axs[1].legend(fontsize=6.5, ncol=2, loc="upper left")
    fig.tight_layout()
    return savefig(fig, "vis_census_depth.png")


def job_split(dname, tag, init, g="late"):
    """Share of poles clean on a task factor / clean on a nuisance factor, averaged over seeds."""
    tk, nu = [], []
    for s in seeds(dname, tag):
        fn = os.path.join(CACHE, f"{dname}_{tag}_s{s}{'_init' if init else ''}_census.pkl")
        if not os.path.exists(fn):
            continue
        c = pickle.load(open(fn, "rb"))[g]
        t = n = 0
        for pole in ("pos", "neg"):
            cc = c[pole]["census"]
            cl = cc["label"] == "clean"
            task = np.isin(cc["best_name"], list(TASK))
            t += (cl & task).sum()
            n += (cl & ~task).sum()
        tot = 2 * len(c["pos"]["census"]["label"])
        tk.append(t / tot)
        nu.append(n / tot)
    return mean_ci(tk) if tk else (np.nan, 0), mean_ci(nu) if nu else (np.nan, 0)


def fig_jobs():
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, g in zip(axs, ("early", "mid", "late")):
        labels, xs = [], 0
        for tag in ("old", "recipe", "arch"):
            for init in (False, True):
                (tm, tc), (nm, nc) = job_split("shapes", tag, init, g)
                ax.bar(xs, tm, 0.7, color=STEP_COL[tag], alpha=1 if not init else 0.45)
                ax.bar(xs, nm, 0.7, bottom=tm, color="#d8d6cf", hatch="..", edgecolor="white", lw=0)
                labels.append(f"{tag}\n{'init' if init else 'trained'}")
                xs += 1
        ax.set_xticks(range(xs))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_title(f"shapes, {g} layer", fontsize=8.5)
    axs[0].set_ylabel("share of poles clean on a\ntask factor (colour) / nuisance (grey)")
    fig.tight_layout()
    return savefig(fig, "vis_jobs.png")


def fig_weak():
    import matplotlib.pyplot as plt
    ab = J("vis_ablate.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, dname in zip(axs, ("shapes", "cifar")):
        for j, tag in enumerate(("old", "recipe", "arch")):
            for k, (key, lab) in enumerate((("base_acc", "intact"), ("all_weak_acc", "weakest 75% removed"),
                                            ("all_strong_acc", "strongest (equal mass) removed"))):
                m, ci = mean_ci(collect(dname, tag, lambda n: ab[n][key]) or [np.nan])
                ax.bar(j + (k - 1) * 0.27, m, 0.27, color=STEP_COL[tag], alpha=[1, 0.6, 0.25][k], yerr=ci, capsize=2,
                       error_kw=dict(lw=0.8), label=lab if j == 0 else None)
                ax.text(j + (k - 1) * 0.27, m + 0.02, f"{100 * m:.0f}", ha="center", fontsize=6.5)
        sf = [mean_ci(collect(dname, t, lambda n: ab[n]["all_strong_frac_removed"]) or [np.nan])[0] for t in ("old", "recipe", "arch")]
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{t}\n(strong mask: {100 * f:.0f}% of acts)" for t, f in zip(("old", "recipe", "arch"), sf)], fontsize=7)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("test accuracy")
        ax.set_title(f"{dname}: masks in every neuron layer")
    axs[0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    return savefig(fig, "vis_weak.png")


def fig_dict():
    import matplotlib.pyplot as plt
    dc, rt = J("vis_dict.json"), J("vis_rotation.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    series = [(dc, "near_copy_share", False, "SAE activity in neuron near-copies"),
              (dc, "near_copy_share", True, "... random-init"),
              (dc, "rot_near_copy_share", False, "... randomly rotated basis"),
              (rt, "raw_aligned", False, "rotation search (raw L4): axes on a neuron"),
              (rt, "ica_aligned", False, "rotation search (whitened / ICA)"),
              (rt, "ica_aligned", True, "... random-init")]
    cols = [SERIES[0], SERIES[0], SERIES[7], SERIES[2], SERIES[4], SERIES[4]]
    alph = [1, 0.45, 0.8, 1, 1, 0.45]
    w = 0.13
    for ax, dname in zip(axs, ("shapes", "cifar")):
        for j, tag in enumerate(("old", "recipe", "arch")):
            for k, (src, key, init, lab) in enumerate(series):
                v = collect(dname, tag, lambda n: src[n][key], init)
                m, ci = mean_ci(v or [np.nan])
                ax.bar(j + (k - 2.5) * w, m, w, color=cols[k], alpha=alph[k], yerr=ci, capsize=1.5,
                       error_kw=dict(lw=0.7), label=lab if j == 0 else None)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["M_old", "M_recipe", "M_arch"])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{dname}: last layer, dictionaries vs neurons")
        ax.set_ylabel("share")
    axs[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    return savefig(fig, "vis_dict.png")


def fig_dial():
    import matplotlib.pyplot as plt
    cen = J("vis_census.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, dname in zip(axs, ("shapes", "cifar")):
        for j, tag in enumerate(("old", "recipe", "arch")):
            ms, cis = [], []
            for g in ("early", "mid", "late"):
                v = collect(dname, tag, lambda n: np.nanmean([x for x in (cen[n][g]["pos"]["dial_frac_of_clean"],
                                                                          cen[n][g]["neg"]["dial_frac_of_clean"]) if x is not None] or [np.nan]))
                m, ci = mean_ci([x for x in v if np.isfinite(x)] or [np.nan])
                ms.append(m)
                cis.append(ci)
            ax.errorbar(np.arange(3) + (j - 1) * 0.08, ms, yerr=cis, color=STEP_COL[tag], marker="o", ms=4, capsize=2, label=tag)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["early", "mid", "late"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("dials / clean poles")
        ax.set_title(f"{dname}: dials (meaning in weak range)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    return savefig(fig, "vis_dial.png")


def fig_fv():
    import matplotlib.pyplot as plt
    fv = J("vis_featvis.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, dname in zip(axs, ("shapes", "cifar")):
        for j, tag in enumerate(("old", "recipe", "arch")):
            for k, st in enumerate(("pixel", "robust", "path")):
                m, ci = mean_ci(collect(dname, tag, lambda n: fv[n][st]["legible_raw"]) or [np.nan])
                mi, _ = mean_ci(collect(dname, tag, lambda n: fv[n][st]["legible_raw"], True) or [np.nan])
                x = j + (k - 1) * 0.27
                ax.bar(x, m, 0.27, color=STEP_COL[tag], alpha=[0.35, 0.65, 1][k], yerr=ci, capsize=2, error_kw=dict(lw=0.8),
                       label=["pixel (2009)", "Fourier+robust (2017)", "+ path anchoring"][k] if j == 0 else None)
                ax.plot([x - 0.12, x + 0.12], [mi, mi], color=INK, lw=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["M_old", "M_recipe", "M_arch"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("FV legible (readout = job)")
        ax.set_title(f"{dname}: feature-vis legibility, late layer")
    axs[0].legend(fontsize=6.5, loc="lower left")
    axs[1].text(0.99, 0.98, "black tick = random-init control", transform=axs[1].transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
    fig.tight_layout()
    return savefig(fig, "vis_fv_legibility.png")


def describe_job(dname, factor, value):
    D = data(dname)
    if dname == "cifar":
        return CIFAR_CLASSES[value]
    from sim.vis_shapes import COLORS, SHAPES, TEXTURES
    names = {"shape": SHAPES, "color": COLORS, "texture": TEXTURES, "bg_type": ["plain bg", "gradient bg", "noise bg"],
             "curved": ["straight", "curved"], "warm": ["cool", "warm"], "textured": ["solid", "textured"]}
    if factor in names:
        return f"{factor}={names[factor][value]}"
    if factor == "label":
        from sim.vis_shapes import SHAPES as S
        return f"{S[value // 4]},{'warm' if value // 2 % 2 else 'cool'},{'tex' if value % 2 else 'solid'}"
    return f"{factor} q{value + 1}/5"


def fig_gallery(dname, n_units=2):
    import matplotlib.pyplot as plt
    D = data(dname)
    rows = []
    for tag in ("old", "recipe", "arch"):
        name = f"{dname}_{tag}_s0"
        fz = os.path.join(CACHE, name + "_fv.npz")
        if not os.path.exists(fz):
            continue
        z = np.load(fz)
        cen = pickle.load(open(os.path.join(CACHE, name + "_census.pkl"), "rb"))["late"]
        sc = cached_scores(dname, tag, 0)[cen["layer"]]
        units = z["units"][: len(z["units"]) // 2]
        pr = cen["pos"]["pur_raw"]
        order = sorted(range(len(units)), key=lambda i: -pr["purity"][units[i]])
        chosen, seen = [], set()
        for i in order:
            f = pr["factor"][units[i]]
            if f not in seen:
                chosen.append(i)
                seen.add(f)
            if len(chosen) == n_units:
                break
        for i in chosen:
            u = int(units[i])
            for pole, off in (("pos", 0), ("neg", len(units))):
                S = sc[0 if pole == "pos" else 1][:, u]
                top = np.argsort(-S)[:5]
                p = cen[pole]["pur_raw"]
                lab = f"{tag} {cen['layer']}:{u} {'+' if pole == 'pos' else '−'}\n{describe_job(dname, p['factor'][u], int(p['value'][u]))}"
                lab += f"\npurity {p['purity'][u]:.2f}, R² {cen[pole]['census']['best_r2'][u]:.2f}"
                fvs = [z[f"{st}_0"][off + i].astype(np.float32).transpose(1, 2, 0) for st in ("pixel", "robust", "path")]
                rows.append((lab, [D["xte"][t] for t in top], fvs))
    n = len(rows)
    fig = plt.figure(figsize=(7.2, 0.82 * n + 0.4))
    gs = fig.add_gridspec(n, 10, width_ratios=[2.1, 1, 1, 1, 1, 1, 0.15, 1, 1, 1], wspace=0.05, hspace=0.12,
                          left=0.01, right=0.99, bottom=0.01, top=1 - 0.35 / (0.82 * n + 0.4))
    heads = ["top 5 test images (raw max)"] + [""] * 4
    for r, (lab, tops, fvs) in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0])
        ax.axis("off")
        ax.text(0, 0.5, lab, fontsize=6.2, va="center", color=INK)
        for j, im in enumerate(tops):
            ax = fig.add_subplot(gs[r, 1 + j])
            ax.imshow(im, interpolation="nearest")
            ax.axis("off")
            if r == 0 and j == 0:
                ax.set_title(heads[0], fontsize=6.5, loc="left", fontweight="normal", color=INK2)
        for j, im in enumerate(fvs):
            ax = fig.add_subplot(gs[r, 7 + j])
            ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
            ax.axis("off")
            if r == 0:
                ax.set_title(["pixel FV", "Fourier+robust", "path-anchored"][j], fontsize=6.5, loc="center",
                             fontweight="normal", color=INK2)
    return savefig(fig, f"vis_gallery_{dname}.png")


def fig_examples():
    import matplotlib.pyplot as plt
    D = data("shapes")
    fig, axs = plt.subplots(2, 12, figsize=(7.2, 1.35))
    for i, ax in enumerate(axs.flat):
        ax.imshow(D["xte"][i], interpolation="nearest")
        ax.axis("off")
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0, right=1, top=1, bottom=0)
    return savefig(fig, "vis_shapes_examples.png")


def main():
    setup_style()
    L = {d: ladder_metrics(d) for d in ("shapes", "cifar")}
    paths = [fig_ladder(L), fig_census_depth(), fig_jobs(), fig_weak(), fig_dict(), fig_dial(), fig_fv(),
             fig_gallery("shapes"), fig_gallery("cifar"), fig_examples()]
    out = dict(ladder=L, census=J("vis_census.json"), featvis=J("vis_featvis.json"), ablate=J("vis_ablate.json"),
               dict=J("vis_dict.json"), rotation=J("vis_rotation.json"), figures=[os.path.relpath(p, os.path.dirname(RESULTS)) for p in paths])
    json.dump(out, open(os.path.join(RESULTS, "vis.json"), "w"), indent=1, default=float)
    for d in L:
        for k, r in L[d].items():
            print(d, k, {m: (None if r[m] is None else tuple(round(x, 3) for x in r[m][:2])) for m in r if m.startswith(("interp", "clean")) and r[m] is not None},
                  "acc", round(r["test_acc"][0], 4))
    print("\n".join(paths))


if __name__ == "__main__":
    main()
