"""Analysis 8: training dynamics and activation-quantile bands.

(a) For every saved checkpoint: census of all units on test windows (stride 12, 4.8k windows) -> fraction of
    live units labelled clean/partial per family over training.
(b) 4 tracked units (chosen at the final checkpoint): R2 with their top-3 final variables at each
    checkpoint, their best variable at each checkpoint, and the top-3 clips at selected checkpoints.
(c) Quantile bands at the final checkpoint: for each tracked unit, split samples by activation
    percentile (0-50, 50-90, 90-99, 99-100); per band report the variable with the largest
    standardized mean shift vs. all data, and the within-band binned R2 of the best variable.
Figures: video_dynamics.png, video_dynamics_clips.png, video_bands.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, INK, INK2, MUTED, SERIES, savefig, setup_style  # noqa: E402
from analysis.video_census import FAM_ORDER, LIVE_STD, load_census  # noqa: E402
from analysis.video_common import (CACHE, LAYERS, RESULTS, binned_r2_matrix, cached_acts,  # noqa: E402
                                   ckpt_paths, ckpt_step, load, target_matrix, windows)
from analysis.video_gallery import top_distinct, unit_maps  # noqa: E402
from analysis.video_viz import clean, overlay  # noqa: E402

BANDS = [(0, 50), (50, 90), (90, 99), (99, 100)]


def main():
    setup_style()
    import matplotlib.pyplot as plt
    Y, vs, st = target_matrix()
    names = [v.name for v in vs]
    cfin = load_census("final")
    H = int(cfin["H"])
    units = json.load(open(os.path.join(CACHE, "units.json")))
    pick = []
    for slot in ["posture", "velocity", "contact", "accel"]:
        cand = [d for d in units if d["slot"] == slot]
        if cand:
            pick.append(cand[-1] if slot == "contact" else cand[0])
    tracked = [d["unit"] for d in pick]
    paths = ckpt_paths()
    steps = [ckpt_step(p) for p in paths]
    hist = {"steps": steps, "frac": [], "units": {d["name"]: {"best": [], "r2_top3": []} for d in pick}}
    per_ckpt_acts = {}
    for p, s in zip(paths, steps):
        tag = f"step{s:06d}"
        path = os.path.join(CACHE, f"acts_{tag}_dyn.npz")
        if os.path.exists(path):
            z = np.load(path); a = {k: z[k] for k in z.files}
        else:
            from analysis.video_common import TEST, extract, grid
            E, T = grid(TEST, 12)
            a = extract(load(p)[0], E, T)
            np.savez(path, **{k: a[k] for k in ("max", "arg", "eps", "ts")})
        A = a["max"].astype(np.float32)
        Yt = Y[a["eps"], a["ts"]]
        R = binned_r2_matrix(Yt, A)
        live = A.std(0) > LIVE_STD
        best, br = R.argmax(1), R.max(1)
        fam = np.array([vs[b].family for b in best])
        row = {"live": float(live.mean())}
        for f in FAM_ORDER:
            row[f] = float(((br >= 0.5) & (fam == f) & live).sum() / max(live.sum(), 1))
        row["clean"] = float(((br >= 0.5) & live).sum() / max(live.sum(), 1))
        row["partial"] = float(((br >= 0.25) & (br < 0.5) & live).sum() / max(live.sum(), 1))
        hist["frac"].append(row)
        for d in pick:
            u = d["unit"]
            top3 = [names.index(v) for v, _ in d["top3"]]
            hist["units"][d["name"]]["best"].append((names[best[u]], float(br[u])))
            hist["units"][d["name"]]["r2_top3"].append([float(R[u, j]) for j in top3])
        per_ckpt_acts[s] = a
        print(s, row["clean"], flush=True)

    # (a)+(b) figure
    fig, axes = plt.subplots(1, 1 + len(pick), figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1.4] + [1] * len(pick)))
    xs = np.array([max(s, 1) for s in steps])
    ax = axes[0]
    bottom = np.zeros(len(steps))
    for f in FAM_ORDER:
        v = np.array([r[f] for r in hist["frac"]])
        if v.max() == 0:
            continue
        ax.fill_between(xs, bottom, bottom + v, color=FAMILIES[f][0], lw=0, label=f, step=None)
        bottom += v
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    ax.set_xlabel("training step (step 0 plotted at 1)"); ax.set_ylabel("fraction of live units, R² ≥ 0.5")
    ax.set_title("Clean units over training", fontsize=8)
    ax.legend(fontsize=5.5, loc="upper left")
    for ax, d in zip(axes[1:], pick):
        h = hist["units"][d["name"]]
        R3 = np.array(h["r2_top3"])
        for j, (v, _) in enumerate(d["top3"]):
            fam = vs[names.index(v)].family
            ax.plot(xs, R3[:, j], "-o", ms=2.5, lw=1.3, color=SERIES[j], label=vs[names.index(v)].label)
        ax.set_xscale("log"); ax.set_ylim(0, 1)
        ax.set_title(d["name"], fontsize=8)
        ax.legend(fontsize=4.8, loc="upper left")
        ax.set_xlabel("step")
    axes[1].set_ylabel("binned R²")
    fig.tight_layout()
    savefig(fig, "video_dynamics.png")

    # clips per checkpoint for tracked units
    show = [s for s in steps if s > 0]
    show = [show[i] for i in np.linspace(0, len(show) - 1, 5).round().astype(int)]
    fig = plt.figure(figsize=(7.2, 1.0 * len(pick) + 0.3))
    gs = fig.add_gridspec(len(pick), len(show) * 3 + (len(show) - 1), wspace=0.04, hspace=0.25,
                          width_ratios=([1, 1, 1, 0.3] * len(show))[:-1], left=0.01, right=0.99, top=0.92, bottom=0.02)
    for r, d in enumerate(pick):
        u = d["unit"]
        for k, s in enumerate(show):
            a = per_ckpt_acts[s]
            m, _ = load(paths[steps.index(s)])
            idx = top_distinct(a["max"][:, u].astype(np.float32), a["eps"], 3)
            maps, clips = unit_maps(m, LAYERS[u // H], u % H, a["eps"][idx], a["ts"][idx])
            axs = []
            for j in range(3):
                ax = fig.add_subplot(gs[r, k * 4 + j])
                overlay(ax, clips[j], maps[j], max(maps.max(), 1e-6), vmin=float(np.median(maps)))
                axs.append(ax)
                if j == 1 and r == 0:
                    ax.set_title(f"step {s}", fontsize=7, loc="center", fontweight="normal", color=INK2)
            b = hist["units"][d["name"]]["best"][steps.index(s)]
            axs[1].text(0.5, -0.16, f"{b[0]} {b[1]:.2f}", fontsize=5, color=MUTED, ha="center", va="top",
                        transform=axs[1].transAxes)
            if k == 0:
                axs[0].set_ylabel(d["name"], fontsize=6.5, color=INK)
    savefig(fig, "video_dynamics_clips.png")

    # (c) quantile bands at final
    afin = cached_acts("final", "test", lambda: load("final")[0], res=True)
    Yt = Y[afin["eps"], afin["ts"]]
    Z = (Yt - Yt.mean(0)) / (Yt.std(0) + 1e-8)
    bands = {}
    fig, axes = plt.subplots(1, len(pick), figsize=(7.2, 2.5), sharey=False)
    for ax, d in zip(axes, pick):
        u = d["unit"]
        a = afin["max"][:, u].astype(np.float32)
        q = np.percentile(a, [0, 50, 90, 99, 100])
        bv = names.index(d["best_var"])
        rows = []
        for (lo, hi), (ql, qh) in zip(BANDS, zip(q[:-1], q[1:])):
            sel = (a >= ql) & (a <= qh)
            shift = Z[sel].mean(0)
            top = np.argsort(-np.abs(shift))[:3]
            rin = binned_r2_matrix(Yt[sel], a[sel, None], bins=10)[0] if sel.sum() > 50 else np.full(len(names), np.nan)
            rows.append({"band": f"{lo}-{hi}", "n": int(sel.sum()),
                         "top_shift": [(names[j], float(shift[j])) for j in top],
                         "within_best_var": names[int(np.nanargmax(rin))], "within_best_r2": float(np.nanmax(rin)),
                         "within_r2_of_overall_best": float(rin[bv])})
        bands[d["name"]] = rows
        # plot: standardized mean of the unit's top-3 overall variables per band
        for j, (v, _) in enumerate(d["top3"]):
            jj = names.index(v)
            vals = []
            for (ql, qh) in zip(q[:-1], q[1:]):
                sel = (a >= ql) & (a <= qh)
                vals.append(Z[sel, jj].mean())
            ax.plot(range(len(BANDS)), vals, "-o", ms=3, color=SERIES[j], label=vs[jj].label)
        ax.set_xticks(range(len(BANDS)), [f"{lo}–{hi}" for lo, hi in BANDS], fontsize=6)
        ax.set_xlabel("activation percentile band", fontsize=7)
        ax.set_title(d["name"], fontsize=8)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.legend(fontsize=4.8, loc="best")
    axes[0].set_ylabel("mean of variable in band (z)")
    fig.tight_layout()
    savefig(fig, "video_bands.png")
    hist["bands"] = bands
    json.dump(hist, open(os.path.join(RESULTS, "video_dynamics.json"), "w"), indent=1)
    for k, v in bands.items():
        print(k, v)


if __name__ == "__main__":
    main()
