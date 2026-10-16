"""Analysis 9: training dynamics of tuning, and what explains different activation bands.

(a) For every saved checkpoint (log-spaced steps) we recompute the binned R² of all live units against all
    32 targets on a fixed random subset of 30k held-out frames, giving the fraction of clean units per layer
    over training, and track 4 exemplar units (chosen in the final model) by index: their R² against the
    variables they are most tuned to at any checkpoint, plus their top dataset examples.
(b) Activation-quantile bands (final model): within the top 1% of a unit's active frames and within its
    50-90th percentile band, which variables explain the activation (binned R², 10 bins, within band)?
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, binned_r2, savefig, setup_style  # noqa: E402
from analysis.state_common import (CLEAN, LIVE_FRAC, PARTIAL, binned_r2_matrix, data, draw_pose, list_checkpoints,  # noqa
                                   load_census, load_model, pick_units, run, update_results)

TRACK = [("grav1", ["grav1"], "gravity · shoulder"), ("c_floor", None, "floor contact"),
         ("int1", None, "interaction · shoulder"), ("qd1", ["qd1"], "velocity · shoulder")]


def main():
    d = data()
    names, vs = d["names"], d["vars"]
    fams = [v.family for v in vs]
    cz = load_census()
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(d["X"]), 30000, replace=False))
    Xs, Ys, eps, ts = d["X"][sub], d["Y"][sub], d["ep"][sub], d["t"][sub]
    tracked, excl = [], set()
    for var, best_in, title in TRACK:
        u = pick_units(cz, names, var, k=1, best_in=best_in, exclude=excl)[0]
        excl.add(u[:2]); tracked.append((var, title) + u)
    cks = [c for c in list_checkpoints()["checkpoints"] if c["tag"].startswith("step")]
    steps = [c["step"] for c in cks]
    census_t, unitR, unitTop = [], {i: [] for i in range(len(tracked))}, {i: [] for i in range(len(tracked))}
    for c in cks:
        m, _ = load_model(c["tag"])
        H = run(m, Xs)["h"]
        row = []
        for l in range(4):
            live = (H[l] > 0).mean(0) > LIVE_FRAC
            R = binned_r2_matrix(H[l][:, live], Ys)
            br = R.max(1)
            der = np.isin(np.array(fams)[R.argmax(1)], ["accel", "gravity", "interaction", "contact", "energy"])
            row.append(dict(n_live=int(live.sum()), clean=float((br >= CLEAN).mean()),
                            clean_derived=float(((br >= CLEAN) & der).mean()),
                            partial=float(((br >= PARTIAL) & (br < CLEAN)).mean()), median_best=float(np.median(br))))
        census_t.append(row)
        for i, (var, title, l, k, _) in enumerate(tracked):
            a = H[l][:, k]
            if (a > 0).mean() > LIVE_FRAC:
                unitR[i].append(binned_r2_matrix(a[:, None], Ys)[0])
            else:
                unitR[i].append(np.zeros(len(names)))
            top, seen = [], set()
            for j in np.argsort(-a):
                if eps[j] not in seen:
                    seen.add(eps[j]); top.append(j)
                if len(top) == 2:
                    break
            unitTop[i].append(top if a.max() > 0 else [])
        print(c["tag"], [f"{r['clean']:.2f}/{r['n_live']}" for r in row],
              [names[int(np.argmax(unitR[i][-1]))] + f"={np.max(unitR[i][-1]):.2f}" for i in range(len(tracked))])

    setup_style()
    import matplotlib.pyplot as plt
    st = d["st"]
    # ---------------- figure A: census over training + test loss
    idx = list_checkpoints()
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.2))
    ax = axes[0]
    xs = np.maximum(steps, 10)
    for l in range(4):
        c = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"][l]
        ax.plot(xs, [census_t[i][l]["clean"] for i in range(len(steps))], "-o", ms=2.5, color=c, label=f"block {l + 1}")
        ax.plot(xs, [census_t[i][l]["clean_derived"] for i in range(len(steps))], "--", lw=1.0, color=c)
    ax.plot([], [], "--", color="#898781", lw=1.0, label="clean on a derived\nquantity (dashed)")
    ax.set_xscale("log")
    ax.set_xlabel("training step (step 0 plotted at 10)")
    ax.set_ylabel("fraction of live units clean\n(best R² ≥ 0.5)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.0, -0.32), ncol=3)
    ax = axes[1]
    ax.plot(xs, [c["test_loss"] for c in cks], "-o", ms=2.5, color="#0b0b0b")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("test MSE (standardized)")
    fig.subplots_adjust(wspace=0.35)
    savefig(fig, "state_dynamics_census.png")

    # ---------------- figure B: tracked units
    show_ck = [i for i, s in enumerate(steps) if s in sorted(set([steps[1], steps[len(steps) // 3], steps[2 * len(steps) // 3], steps[-1]]))]
    nT = len(tracked)
    fig = plt.figure(figsize=(7.2, 1.25 * nT + 0.3))
    gs = fig.add_gridspec(nT, 1 + 2 * len(show_ck), width_ratios=[3.2] + [0.62] * (2 * len(show_ck)),
                          wspace=0.05, hspace=0.45)
    dyn = {}
    for i, (var, title, l, k, r2v) in enumerate(tracked):
        Rm = np.array(unitR[i])                        # [C, V]
        top_vars = list(np.argsort(-Rm.max(0))[:4])
        ax = fig.add_subplot(gs[i, 0])
        lcol = ["#0b0b0b", "#e87ba4", "#2a78d6", "#eda100"]
        for ci, j in enumerate(top_vars):
            ax.plot(xs, Rm[:, j], "-o", ms=2.2, lw=1.8 if j == names.index(var) else 1.1, color=lcol[ci],
                    label=f"{names[j]} ({fams[j]})")
        ax.set_xscale("log"); ax.set_ylim(0, 1.38); ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("binned R²", fontsize=7)
        ax.set_title(f"L{l + 1}/{k} · {title}", fontsize=7.5)
        ax.legend(fontsize=5.3, loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8)
        if i == nT - 1:
            ax.set_xlabel("training step (step 0 plotted at 10)")
        else:
            ax.set_xticklabels([])
        for s in show_ck:
            ax.axvline(xs[s], color="#c3c2b7", lw=0.6)
        for c, s in enumerate(show_ck):
            for e_ in range(2):
                a_ = fig.add_subplot(gs[i, 1 + 2 * c + e_])
                if e_ < len(unitTop[i][s]):
                    j = unitTop[i][s][e_]
                    e, t = eps[j], ts[j]
                    draw_pose(a_, st["q"][e, t], st["q"][e, max(0, t - 6):t], vel=st["ee_vel"][e, t])
                else:
                    a_.axis("off")
                if i == 0 and e_ == 0:
                    a_.set_title(f"step {steps[s]}", fontsize=6.5, loc="left")
        dyn[f"L{l + 1}/{k}"] = dict(var=var, title=title, steps=steps,
                                    r2={names[j]: Rm[:, j].tolist() for j in top_vars},
                                    best_per_step=[names[int(np.argmax(r))] for r in Rm])
    savefig(fig, "state_dynamics_units.png")

    # ---------------- figure C: activation quantile bands (final model, all test frames)
    model, _ = load_model("final")
    H = run(model, d["X"])["h"]
    bands = {}
    fig, axes = plt.subplots(nT, 2, figsize=(7, 1.35 * nT), gridspec_kw=dict(width_ratios=[1.3, 1.7]))
    for i, (var, title, l, k, r2v) in enumerate(tracked):
        a = H[l][:, k]
        act = a > 0
        p50, p90, p99 = np.percentile(a[act], [50, 90, 99])
        top = a >= p99
        mid = act & (a >= p50) & (a < p90)
        res_b = {}
        for bn, m in (("top1pct", top), ("p50_90", mid), ("all", np.ones_like(act))):
            Rb = np.array([binned_r2(d["Y"][m, v], a[m], bins=10) if np.ptp(d["Y"][m, v]) > 0 else 0
                           for v in range(len(names))])
            order = np.argsort(-Rb)[:3]
            res_b[bn] = dict(n=int(m.sum()), top=[(names[o], float(Rb[o])) for o in order],
                             r2_tracked_var=float(Rb[names.index(var)]))
        bands[f"L{l + 1}/{k}"] = res_b
        ax = axes[i, 0]
        x = d["Y"][:, names.index(var)]
        s = rng.choice(len(a), 8000, replace=False)
        ax.scatter(x[s], a[s], s=1, color="#c3c2b7", lw=0, rasterized=True)
        for m, c in ((mid, "#5598e7"), (top, "#e34948")):
            ss = np.where(m)[0]
            ss = ss[rng.choice(len(ss), min(len(ss), 1500), replace=False)]
            ax.scatter(x[ss], a[ss], s=1.5, color=c, lw=0, rasterized=True)
        ax.set_xlabel(var, fontsize=6.5, labelpad=1)
        ax.set_ylabel(f"L{l + 1}/{k}\nactivation", fontsize=6.5)
        if i == 0:
            ax.scatter([], [], s=6, color="#e34948", label="top 1% of active frames")
            ax.scatter([], [], s=6, color="#5598e7", label="50–90th percentile")
            ax.legend(fontsize=5.5, loc="upper right", markerscale=1.5, handletextpad=0.2)
        ax.tick_params(labelsize=6)
        ax = axes[i, 1]
        labs, vals, cols = [], [], []
        for bn, lab in (("top1pct", "top 1%"), ("p50_90", "50–90%")):
            for nmv, rv in res_b[bn]["top"]:
                labs.append(f"{lab}: {nmv}"); vals.append(rv); cols.append(FAMILIES[fams[names.index(nmv)]][0])
        y = np.arange(len(labs))[::-1]
        ax.barh(y, vals, color=cols, height=0.7)
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6)
        for yy, lab in zip(y, labs):
            ax.get_yticklabels()[list(y).index(yy)].set_color("#e34948" if lab.startswith("top") else "#256abf")
        ax.set_xlim(0, 1)
        ax.tick_params(axis="x", labelsize=6)
        if i == nT - 1:
            ax.set_xlabel("within-band binned R²", fontsize=6.5)
    fig.subplots_adjust(hspace=0.45, wspace=0.55)
    savefig(fig, "state_bands.png")
    for k_, v in bands.items():
        print(k_, {b: (x["n"], [(n_, round(r_, 2)) for n_, r_ in x["top"]]) for b, x in v.items()})
    update_results("dynamics", dict(steps=steps, census_over_training=census_t, units=dyn, bands=bands))


if __name__ == "__main__":
    main()
