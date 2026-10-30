"""Recursive-generalisation lens.

(1) Activation-quantile bands (CPU): for selected units, split the groups by the unit's pooled
    activation into bands (<50th, 50-90th, 90-99th, top 1%) and ask which physical variable
    explains the variation *within* each band (binned R², 10 bins) and how the best variable's
    value shifts across bands. Example frames from each band are shown.
(2) Rendering variants (GPU): the same trajectories re-rendered with swapped colours, grey
    links, dark backdrop, no visible floor/wall, and a flat 2-D cartoon (vlm_variants). For each
    gallery unit we measure the tuning to its best variable in each variant, and the
    correlation of the unit's activation with its activation on the original render of the
    same physical state. Linear probes trained on original renders are applied to variants.

Usage: uv run python analysis/vlm_general.py [bands|collect|analyze]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, binned_r2, load_states, savefig, setup_style, targets, INK, INK2, MUTED, SERIES  # noqa: E402
from analysis.vlm_common import VDIR, n_used, group_index, clip_index, clip_frames, GROUPS  # noqa: E402
from analysis.vlm_gallery import unit_values, trail_image  # noqa: E402
from analysis.vlm_variants import VARIANTS, variant_renderer  # noqa: E402

BANDS = [(0, 50), (50, 90), (90, 99), (99, 100)]


def _f(x):
    return None if np.isnan(x) else float(x)
N_VAR_EPS, VAR_CLIPS = 30, (1, 4, 7)


def bands(units=None):
    setup_style()
    import matplotlib.pyplot as plt
    st = load_states()
    Yall, vs = targets(st)
    names, fams, labels = [v.name for v in vs], [v.family for v in vs], [v.label for v in vs]
    NG, _ = n_used("video")
    gi = group_index()[:NG]
    Y = Yall[gi[:, 0], gi[:, 3]]
    with open(os.path.join(VDIR, "gallery_spec.json")) as f:
        spec = json.load(f)
    units = [sp for sp in spec if sp.get("bands")] or spec[:4]
    from analysis.vlm_common import renderer
    res = {}
    fig = plt.figure(figsize=(7.0, 1.55 * len(units) + 0.3))
    gs = fig.add_gridspec(len(units), 6, width_ratios=[2.2, 1.6, 1, 1, 1, 1], wspace=0.12, hspace=0.45,
                          left=0.07, right=0.99, top=0.93, bottom=0.05)
    for ui, sp in enumerate(units):
        (part, l, pole), pos, neg = unit_values(sp["key"], sp["unit"], NG)
        a = pos if pole == "+" else -neg
        pct = np.argsort(np.argsort(a)) / len(a) * 100
        out = []
        for lo, hi in BANDS:
            m = (pct >= lo) & (pct < hi) if hi < 100 else pct >= lo
            # a band where the unit is constant (e.g. all zeros below its threshold) has no within-band R²
            r2 = np.full(len(names), np.nan) if a[m].std() < 1e-6 else \
                np.array([binned_r2(Y[m, j], a[m], bins=10) for j in range(len(names))])
            top3 = np.argsort(-np.nan_to_num(r2, nan=-1.0))[:3]
            out.append(dict(band=f"{lo}-{hi}", n=int(m.sum()), top=[(names[j], _f(r2[j])) for j in top3],
                            r2_best_var=_f(r2[names.index(sp["var"])]),
                            mean_best_var=float(Y[m, names.index(sp["var"])].mean()),
                            mean_act=float(a[m].mean()), idx=np.where(m)[0]))
        res[f"{sp['key']}#{sp['unit']}"] = [{k: v for k, v in o.items() if k != "idx"} for o in out]
        # panel 1: within-band R2 for the best variable and best other variable
        ax = fig.add_subplot(gs[ui, 0])
        x = np.arange(len(BANDS))
        ax.bar(x - 0.2, [np.nan if o["r2_best_var"] is None else o["r2_best_var"] for o in out], 0.4,
               color=FAMILIES[sp["fam"]][0], label=f"{sp['var']} (overall)")
        ax.bar(x + 0.2, [np.nan if o["top"][0][1] is None else o["top"][0][1] for o in out], 0.4, color=MUTED,
               label="best in band")
        for xi, o in zip(x, out):
            t1 = o["top"][0][1]
            ax.text(xi + 0.2, (t1 if t1 is not None else 0) + 0.02, o["top"][0][0] if t1 is not None else "constant",
                    rotation=90, fontsize=5.5, ha="center", va="bottom", color=INK2)
        ax.set_xticks(x, ["<50", "50–90", "90–99", "top 1%"], fontsize=6.5)
        ax.set_ylim(0, 1.5)
        ax.set_xlim(-0.6, len(BANDS) - 0.4)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel("within-band R²", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{('V' if part == 'vision' else 'L')}{l} #{sp['unit']}{' (−)' if pole == '−' else ''}: {sp['var']}",
                     fontsize=8)
        if ui == 0:
            ax.legend(fontsize=5.5, loc="upper left", ncol=1)
        # panel 2: distribution of best variable per band
        ax = fig.add_subplot(gs[ui, 1])
        xv = Y[:, names.index(sp["var"])]
        for bi_, o in enumerate(out):
            ax.hist(xv[o["idx"]], bins=30, density=True, histtype="step", lw=1.2, color=SERIES[bi_],
                    label=["<50", "50–90", "90–99", "top 1%"][bi_])
        ax.set_xlabel(sp["var"], fontsize=6.5, labelpad=1)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        if ui == 0:
            ax.legend(fontsize=5.5, loc="upper right")
        # example frames: band medians
        for bi_, o in enumerate(out):
            ax = fig.add_subplot(gs[ui, 2 + bi_])
            idx = o["idx"]
            i = idx[np.argsort(a[idx])[len(idx) // 2]]
            e, _, g, f, fa, fb = gi[i]
            fr = renderer()(st["q"][e][clip_frames(fa - 4 * g)])
            ax.imshow(trail_image(fr, 2 * g + 1))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_color(SERIES[bi_]); s.set_visible(True); s.set_linewidth(1.5)
            ax.set_title(f"{['<50', '50–90', '90–99', 'top 1%'][bi_]}: a={a[i]:.2f}", fontsize=6, color=INK2)
    fig.suptitle("Activation-quantile bands: what explains each range of a unit's activity", x=0.01, ha="left",
                 fontsize=9.5, fontweight="bold")
    savefig(fig, "vlm_bands.png")
    with open(os.path.join(VDIR, "bands_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


def collect():
    import torch
    from analysis.vlm_common import load_model
    from analysis.vlm_capture import capture
    st = load_states()
    with open(os.path.join(VDIR, "gallery_spec.json")) as f:
        spec = json.load(f)
    with open(os.path.join(VDIR, "probe_results.json")) as f:
        L = json.load(f)["oof_llm_layer"]
    ci = clip_index()
    eps = np.unique(ci[:, 0])[:N_VAR_EPS]
    rows = np.array([r for r in range(len(ci)) if ci[r, 0] in set(eps) and ci[r, 1] in VAR_CLIPS])
    units = []
    for sp in spec:
        part, Ls = sp["key"].split("_")[1:3]
        units.append((part, int(Ls[1:]), sp["unit"]))
    model, proc = load_model()
    for kind in VARIANTS:
        path = os.path.join(VDIR, f"variant_{kind}.npz")
        if os.path.exists(path):
            continue
        r = variant_renderer(kind)
        clips = [r(st["q"][ci[i, 0]][clip_frames(ci[i, 2])]) for i in rows]
        maps, res = capture(model, proc, clips, units, res_layers=("V23", f"L{L}"))
        out = dict(rows=rows)
        for k, (sp, m) in enumerate(zip(spec, maps)):
            out[f"max{k}"] = m.reshape(m.shape[0], GROUPS, -1).max(-1)
            out[f"min{k}"] = m.reshape(m.shape[0], GROUPS, -1).min(-1)
        out["V23"], out["Lres"] = res["V23"], res[f"L{L}"]
        np.savez(path, **out)
        print("variant", kind, flush=True)
    del model


def analyze():
    setup_style()
    import matplotlib.pyplot as plt
    from analysis.vlm_probe import ridge_cv
    import torch
    st = load_states()
    Yall, vs = targets(st)
    names, fams, labels = [v.name for v in vs], [v.family for v in vs], [v.label for v in vs]
    NG, _ = n_used("video")
    gi = group_index()
    with open(os.path.join(VDIR, "gallery_spec.json")) as f:
        spec = json.load(f)
    with open(os.path.join(VDIR, "probe_results.json")) as f:
        L = json.load(f)["oof_llm_layer"]
    Z = {k: np.load(os.path.join(VDIR, f"variant_{k}.npz")) for k in VARIANTS}
    rows = Z["original"]["rows"]
    grows = (rows[:, None] * GROUPS + np.arange(GROUPS)).ravel()
    Y = Yall[gi[grows, 0], gi[grows, 3]]
    res = {"units": {}, "probes": {}}
    T = np.zeros((len(spec), len(VARIANTS)))
    C = np.zeros((len(spec), len(VARIANTS)))
    for k, sp in enumerate(spec):
        pole = sp["key"].split("_")[3]
        def act(kind):
            return (Z[kind][f"max{k}"] if pole == "pos" else -Z[kind][f"min{k}"]).ravel()
        a0 = act("original")
        j = names.index(sp["var"])
        d = {}
        for vi, kind in enumerate(VARIANTS):
            a = act(kind)
            T[k, vi] = np.nan if a.std() < 1e-6 else binned_r2(Y[:, j], a, bins=10)
            # a unit that is constant on a variant (or on the original) has no tuning: n/a, not R² = 1
            dead = a.std() < 1e-6 or a0.std() < 1e-6
            C[k, vi] = np.nan if dead else np.corrcoef(a, a0)[0, 1]
            d[kind] = dict(tuning_r2=float(T[k, vi]), corr_with_original=float(C[k, vi]),
                           mean_act_ratio=float(a.mean() / (a0.mean() + 1e-8)))
        res["units"][f"{sp['key']}#{sp['unit']}"] = dict(var=sp["var"], **d)
    # probes trained on main data (original renders, excluding the variant episodes) -> variants
    var_eps = set(gi[grows, 0])
    train = np.array([i for i in range(NG) if gi[i, 0] not in var_eps])
    Ytr = Yall[gi[train, 0], gi[train, 3]]
    pv = ["sin_q1", "cos_q1", "sin_q12", "ee_x", "ee_z", "qd1", "qd2", "ee_vx", "ee_vz", "qdd1", "contact", "c_floor", "c_wall", "ke"]
    pj = [names.index(n) for n in pv]
    P = np.zeros((2, len(pv), len(VARIANTS)))
    for si, (src, fn, lay) in enumerate([("V23", "video_vis_res.npy", 23), ("Lres", "video_llm_res.npy", L)]):
        A = np.asarray(np.load(os.path.join(VDIR, fn), mmap_mode="r")[train, lay], np.float32)
        mu, sd = A.mean(0), A.std(0) + 1e-6
        X = (A - mu) / sd
        Xt = torch.as_tensor(X, dtype=torch.float64)
        ym = Ytr[:, pj].mean(0)
        G = Xt.T @ Xt
        W = torch.linalg.solve(G + 1000.0 * torch.eye(G.shape[0], dtype=torch.float64), Xt.T @ torch.as_tensor(Ytr[:, pj] - ym, dtype=torch.float64)).numpy()
        for vi, kind in enumerate(VARIANTS):
            F = (Z[kind][src].reshape(-1, A.shape[1]) - mu) / sd
            pred = F @ W + ym
            yy = Y[:, pj]
            P[si, :, vi] = 1 - ((pred - yy) ** 2).sum(0) / ((yy - yy.mean(0)) ** 2).sum(0)
        res["probes"][src] = {pv[i]: {kind: float(P[si, i, vi]) for vi, kind in enumerate(VARIANTS)} for i in range(len(pv))}
    with open(os.path.join(VDIR, "general_results.json"), "w") as f:
        json.dump(res, f, indent=1)

    # figure
    fig = plt.figure(figsize=(7.0, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.3, 1], width_ratios=[1, 1, 1.05], hspace=0.28, wspace=0.35,
                          left=0.17, right=0.98, top=0.9, bottom=0.08)
    # row of variant thumbnails
    gsr = gs[0, :].subgridspec(1, len(VARIANTS), wspace=0.05)
    q = st["q"][gi[grows[5], 0]][clip_frames(gi[grows[5], 4] - 4 * gi[grows[5], 2])]
    for vi, kind in enumerate(VARIANTS):
        ax = fig.add_subplot(gsr[0, vi])
        fr = variant_renderer(kind)(q)
        ax.imshow(trail_image(fr, 7) if kind not in ("cartoon",) else fr[7])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(kind, fontsize=7.5)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("r2", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    ulabels = [f"{('V' if sp['key'].split('_')[1] == 'vision' else 'L')}{sp['key'].split('_')[2][1:]}#{sp['unit']}"
               f"{'−' if sp['key'].split('_')[3] == 'neg' else ''} {sp['var']}" for sp in spec]
    for pi_, (M, ttl) in enumerate([(T, "A  unit tuning R² to its variable"), (C, "B  corr. with activation on original")]):
        ax = fig.add_subplot(gs[1, pi_])
        im = ax.imshow(np.clip(M, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if v > 0.6 else INK)
        ax.set_xticks(range(len(VARIANTS)), VARIANTS, rotation=60, fontsize=6.5)
        ax.set_yticks(range(len(spec)), ulabels if pi_ == 0 else [], fontsize=6)
        ax.grid(False)
        ax.set_title(ttl, fontsize=8)
    ax = fig.add_subplot(gs[1, 2])
    for si, (src, ls) in enumerate([("vision V23", "--"), (f"LM L{L}", "-")]):
        for gname, members, c in [("posture", ["sin_q1", "cos_q1", "sin_q12", "ee_x", "ee_z"], FAMILIES["posture"][0]),
                                  ("velocity", ["qd1", "qd2", "ee_vx", "ee_vz"], FAMILIES["velocity"][0]),
                                  ("contact", ["contact", "c_floor", "c_wall"], FAMILIES["contact"][0])]:
            ii = [pv.index(m) for m in members]
            ax.plot(range(len(VARIANTS)), np.clip(P[si, ii].mean(0), -0.2, 1), ls, color=c, marker="o", ms=3, lw=1.3,
                    label=None)
    ax.set_xticks(range(len(VARIANTS)), VARIANTS, rotation=60, fontsize=6.5)
    ax.set_ylim(-0.2, 1)
    ax.set_ylabel("R² (probe trained on original)", fontsize=7)
    ax.set_title("C  probe transfer", fontsize=8)
    from matplotlib.lines import Line2D
    hh = [Line2D([], [], color=FAMILIES[f][0], lw=1.3, label=f) for f in ("posture", "velocity", "contact")] + \
         [Line2D([], [], color=INK2, lw=1.1, ls="--", label="vision V23"), Line2D([], [], color=INK2, lw=1.1, label=f"LM L{L}")]
    ax.legend(handles=hh, fontsize=5.5, loc="upper right", ncol=1)
    fig.suptitle("Do arm-physics units and probes survive a change of rendering?", x=0.01, ha="left", fontsize=9.5,
                 fontweight="bold")
    savefig(fig, "vlm_generalisation.png")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"bands": bands, "collect": collect, "analyze": analyze}[sys.argv[1]]()
