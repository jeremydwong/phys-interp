"""imnet_ figures: purity bars, weak-activation grids (ours vs the essay's), SAE near-copy and rotation bars.

The essay's weak-ablation grid values are copied from the figure's cell labels (Cammarata (2026), Neural Networks
Are Already Clean, Goodfire; figure "weak-ablation-grid"), stored in reports/results/imnet_essay_weak_grid.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import imnet_common as C  # noqa: E402
from analysis.common import SERIES, setup_style  # noqa: E402

PCTS = (25, 50, 75, 95, 99)
NICE = {"googlenet": "GoogLeNet (InceptionV1)", "convnext_tiny": "ConvNeXt-T", "convnext_small": "ConvNeXt-S"}


def _load(name):
    fn = os.path.join(C.RESULTS, name)
    return json.load(open(fn)) if os.path.exists(fn) else {}


def fig_purity(dataset):
    import matplotlib.pyplot as plt
    P = _load("imnet_purity.json").get(dataset, {})
    archs = [a for a in C.MODELS if a in P]
    if not archs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, g, title in zip(axes, ("class", "super60"), ("top-20 >= 50% one class", "top-20 >= 50% one superclass (K=60)")):
        x = np.arange(len(archs))
        for i, (tr, col, lab) in enumerate(((True, SERIES[0], "trained, + pole"), (True, SERIES[1], "trained, − pole"),
                                            (False, "#999999", "random init, + pole"))):
            vals = []
            for a in archs:
                t = C.tag(a, tr)
                L = C.last_layer(a)
                pole = "neg" if "−" in lab else "pos"
                vals.append(100 * P[t][L][pole][g]["dominated"] if t in P else np.nan)
            ax.bar(x + (i - 1) * 0.26, vals, 0.25, color=col, label=lab)
            for xi, v in zip(x + (i - 1) * 0.26, vals):
                if np.isfinite(v):
                    ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=6.5)
        ax.set_xticks(x, [NICE[a] for a in archs], fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 105)
    axes[0].axhline(70, color="#b52f2f", lw=1, ls="--")
    axes[0].axhline(99, color="#b52f2f", lw=1, ls=":")
    axes[0].text(-0.45, 72, "essay, InceptionV1 retrained: ~70% (human)", fontsize=6, color="#b52f2f", ha="left")
    axes[0].text(-0.45, 101, "essay, ConvNeXt: ~99% (human)", fontsize=6, color="#b52f2f", ha="left")
    leg = _load("imnet_legibility_ours.json")
    if leg and dataset == "imagenet_val":
        xs = [i for i, a in enumerate(archs) if a in leg]
        axes[0].scatter(xs, [100 * leg[archs[i]]["share"] for i in xs], marker="D", s=22, color="#222222", zorder=5,
                        label="ours, one reader: legible (n = 24)")
        for i in xs:
            axes[0].text(i + 0.12, 100 * leg[archs[i]]["share"], f"{100 * leg[archs[i]]['share']:.0f}", fontsize=6.5,
                         va="center")
    axes[0].set_ylabel("% of last-layer units")
    h0, l0 = axes[0].get_legend_handles_labels()
    axes[1].legend(h0, l0, fontsize=6.5, loc="upper right")
    fig.suptitle("Last-layer top-20 purity on ImageNet val (ours: class dominance is stricter than legibility)",
                 fontsize=9, x=0.0, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGS, "imnet_purity.png"), dpi=200)
    plt.close(fig)


def _grid(ax, rows, M, title, vmax=70, cmap=None):
    im = ax.imshow(-M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(PCTS)), [f"p{p}" for p in PCTS], fontsize=7)
    ax.set_yticks(range(len(rows)), rows, fontsize=6)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.1f}" if abs(M[i, j]) >= 0.95 else f"{M[i, j]:+.2f}", ha="center",
                        va="center", fontsize=5.6, color="white" if -M[i, j] > vmax * 0.5 else "#222222")
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=8)
    return im


def fig_weak(dataset):
    import matplotlib.pyplot as plt
    _, seq = setup_style()
    W = _load("imnet_weak.json").get(dataset, {})
    Ess = _load("imnet_essay_weak_grid.json")
    panels = []
    if "googlenet" in W:
        g = W["googlenet"]
        rows = g["layers"]
        M = np.array([[g["cells"].get(f"{r}|p{p}", {}).get("delta_pp", np.nan) for p in PCTS] for r in rows])
        panels.append((f"ours: GoogLeNet\nbase {100 * g['base_top1']:.2f}%", rows, M))
        e = Ess.get("Inception v1", {})
        if e:
            er = list(e["rows"])
            panels.append((f"essay: Inception v1\nbase {e['base']:.2f}%", er,
                           np.array([[e["rows"][r].get(f"p{p}", np.nan) for p in PCTS] for r in er])))
    for a in ("convnext_tiny", "convnext_small"):
        if a in W:
            g = W[a]
            rows = g["layers"] + ["all"]
            M = np.array([[g["cells"].get(f"{r}|p{p}", {}).get("delta_pp", np.nan) for p in PCTS] for r in rows])
            lab = [r.replace("features.", "f").replace(".mlp", "") for r in rows]
            panels.append((f"ours: {NICE[a]} (GELU MLP)\nbase {100 * g['base_top1']:.2f}%", lab, M))
    e = Ess.get("ConvNeXt medium", {})
    if e:
        er = list(e["rows"])
        panels.append((f"essay: ConvNeXt medium (SwiGLU)\nbase {e['base']:.2f}%", er,
                       np.array([[e["rows"][r].get(f"p{p}", np.nan) for p in PCTS] for r in er])))
    if not panels:
        return
    panels = panels[:4]
    fig, axes = plt.subplots(2, 2, figsize=(7, 10), gridspec_kw=dict(height_ratios=[12, 19], wspace=0.55, hspace=0.18))
    axes = axes.ravel()[:len(panels)]
    for ax, (t, r, M) in zip(axes, panels):
        im = _grid(ax, r, M, t, cmap=seq)
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02)
    cb.set_label("top-1 loss (pp)", fontsize=7)
    fig.suptitle("Zeroing activations below each neuron's p-th percentile, one layer at a time\n(change in top-1, pp; "
                 "'all' = every ConvNeXt block at once)", fontsize=8.5, x=0.02, ha="left", y=0.95)
    fig.savefig(os.path.join(C.FIGS, "imnet_weak_grid.png"), dpi=200)
    plt.close(fig)


def fig_sae_rot(dataset):
    import matplotlib.pyplot as plt
    S = _load("imnet_sae.json")
    R = _load("imnet_rotation.json").get(dataset, {})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    ax = axes[0]
    labels, vals, cols = [], [], []
    for a in ("googlenet", "convnext_tiny"):
        for k in ("k8", "k32"):
            for t, col, suf in ((C.tag(a, True), SERIES[0], ""), (C.tag(a, False), "#999999", " init")):
                r = S.get(dataset, {}).get(t, {}).get(k)
                if r:
                    labels.append(f"{NICE[a].split(' ')[0]} {k}{suf}")
                    vals.append(100 * r["near_copy_share"])
                    cols.append(col)
            r = S.get(dataset, {}).get(C.tag(a, True) + "_resid", {}).get(k)
            if r:
                labels.append(f"{NICE[a].split(' ')[0]} {k} residual")
                vals.append(100 * r["near_copy_share"])
                cols.append(SERIES[6])
            r = S.get(dataset, {}).get(C.tag(a, True), {}).get(k)
            if r:
                labels.append(f"{NICE[a].split(' ')[0]} {k} rot.")
                vals.append(100 * r["rotated_control"]["near_copy_share"])
                cols.append(SERIES[3])
    for nm in ("axes", "rotated"):
        r = S.get("planted", {}).get(nm)
        if r:
            labels.append(f"planted {nm}")
            vals.append(100 * r["near_copy_share"])
            cols.append(SERIES[2])
    y = np.arange(len(vals))
    ax.barh(y, vals, color=cols)
    ax.set_yticks(y, labels, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(88, color="#b52f2f", ls="--", lw=1)
    ax.text(87, len(vals) - 0.5, "essay 88%", color="#b52f2f", fontsize=6.5, ha="right")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of SAE activity in near-copies of one neuron (|cos| >= 0.9)", fontsize=7)
    ax.set_title("SAE near-copy share (last layer)", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    labels, vals, cols = [], [], []
    for a in ("googlenet", "convnext_tiny"):
        for tr in (True, False):
            t = C.tag(a, tr)
            if t not in R:
                continue
            sub = R.get(t + "_sub256", {})
            items = [("full layer, random start", R[t]["raw"]["aligned_0p9"], SERIES[4])]
            if sub:
                items += [("256-unit sets, random start", sub["raw_aligned"], SERIES[0]),
                          ("256-unit sets, identity start", sub["identity_aligned"], SERIES[1]),
                          ("256-unit sets, whitened (ICA)", sub["ica_aligned"], SERIES[2])]
            for nm, v, col in items:
                labels.append(f"{NICE[a].split(' ')[0]}{'' if tr else ' init'}: {nm}")
                vals.append(100 * v)
                cols.append(col if tr else "#999999")
    y = np.arange(len(vals))
    ax.barh(y, vals, color=cols)
    ax.set_yticks(y, labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of found axes within |cos| >= 0.9 of a neuron", fontsize=7)
    ax.set_title("Rotation search (last layer)", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGS, "imnet_sae_rotation.png"), dpi=200)
    plt.close(fig)


def main(dataset="imagenet_val"):
    setup_style()
    fig_purity(dataset)
    fig_weak(dataset)
    fig_sae_rot(dataset)
    summary(dataset)



def summary(dataset="imagenet_val"):
    """Collect the headline numbers (essay vs ours) into reports/results/imnet.json."""
    P = _load("imnet_purity.json").get(dataset, {})
    S = _load("imnet_sae.json")
    R = _load("imnet_rotation.json").get(dataset, {})
    W = _load("imnet_weak.json").get(dataset, {})
    Ess = _load("imnet_essay_weak_grid.json")
    leg = _load("imnet_legibility_ours.json")
    out = dict(source="Cammarata (2026), Neural Networks Are Already Clean, Goodfire", dataset=dataset)
    out["top1"] = {t: P[t]["top1"] for t in P if isinstance(P[t], dict) and "top1" in P[t]}
    out["interpretability_last_layer"] = dict(
        essay=dict(inceptionv1_2014_old_tools=0.05, inceptionv1_2014_better_tools=0.50, inceptionv1_retrained=0.70,
                   convnext_same_size=0.99, yardstick="human/agent reading of top-20 of 50,000"),
        ours_class_dominated_pos={t: P[t][C.last_layer(t.replace("_init", ""))]["pos"]["class"]["dominated"]
                                  for t in P if isinstance(P[t], dict) and "top1" in P[t]},
        ours_super60_dominated_pos={t: P[t][C.last_layer(t.replace("_init", ""))]["pos"]["super60"]["dominated"]
                                    for t in P if isinstance(P[t], dict) and "top1" in P[t]},
        ours_one_reader_legible={k: v["share"] for k, v in leg.items() if isinstance(v, dict) and "share" in v},
        verdict="ordering reproduces (ConvNeXt >> InceptionV1); absolute levels do not")
    d = S.get(dataset, {})
    out["sae_near_copy"] = dict(
        essay=0.88, ours={t: {k: d[t][k]["near_copy_share"] for k in ("k8", "k32") if k in d[t]} for t in d
                          if isinstance(d[t], dict) and "k8" in d[t]},
        planted=S.get("planted"),
        removal={t: {k: d[t][k]["removal"]["remove_unmatched"] for k in ("k8", "k32") if "removal" in d[t].get(k, {})}
                 for t in d if isinstance(d[t], dict) and "k8" in d[t]},
        zero_layer_control=d.get("zero_layer_control"), verdict="does not reproduce")
    out["rotation"] = dict(essay="from a random rotation you land back on the neurons, in every block",
                           ours={t: ({k: v for k, v in R[t].items() if k != "reps"} if "sub" in t else
                                     dict(raw=R[t]["raw"]["aligned_0p9"], ica=R[t]["ica"]["aligned_0p9"],
                                          identity=R[t]["raw"]["identity_start"]["aligned_0p9"])) for t in R},
                           verdict="reproduces for ConvNeXt-T (256-unit sets), not for GoogLeNet")
    wk = {}
    for a, g in W.items():
        wk[a] = dict(base_top1=g["base_top1"], cells=g["cells"])
    out["weak_grid"] = dict(essay={m: dict(base=v["base"], rows=v["rows"]) for m, v in Ess.items()
                                   if isinstance(v, dict) and "rows" in v and m in ("Inception v1", "ConvNeXt medium")},
                            ours=wk, verdict="last layers reproduce closely; all-blocks row reproduces in shape; "
                                             "early GoogLeNet layers far more fragile in ours")
    json.dump(out, open(os.path.join(C.RESULTS, "imnet.json"), "w"), indent=1)


def convnext_table(dataset="imagenet_val", arch="convnext_tiny"):
    """Markdown rows: ours (essay ConvNeXt medium) per block, plus 'all'."""
    W = _load("imnet_weak.json").get(dataset, {}).get(arch)
    E = _load("imnet_essay_weak_grid.json").get("ConvNeXt medium", {}).get("rows", {})
    if not W:
        return ""
    ess = list(E)
    rows = ["| block (essay row) | p25 | p50 | p75 | p95 | p99 |", "|---|---|---|---|---|---|"]
    for i, n in enumerate(W["layers"] + ["all"]):
        en = "all" if n == "all" else (ess[i] if i < len(ess) else "")
        cells = []
        for p in PCTS:
            o = W["cells"].get(f"{n}|p{p}", {}).get("delta_pp")
            e = E.get(en, {}).get(f"p{p}")
            cells.append((f"{o:+.2f}" if o is not None else "–") + (f" ({e:+.2f})" if e is not None else ""))
        lab = n.replace("features.", "f").replace(".mlp", "")
        rows.append(f"| {'**' + lab + '**' if n in ('all', W['layers'][-1]) else lab} ({en}) | " + " | ".join(cells) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["imagenet_val"]))
