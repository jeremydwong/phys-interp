"""Analysis 3: Distill-style neuron galleries.

For ~12 exemplary units: top-5 input clips (distinct episodes; last frame + faint motion trail of
t-3..t-1, unit's 16x16 activation map overlaid, '+' = argmax), the tuning curve vs the best
variable, and the 3 most negative clips. Units chosen automatically from the census (best unit
per family/stage), plus one high-variance "unexplained" unit for honesty.
Figures: video_gallery_a.png, video_gallery_b.png. Writes models/video_cache/units.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, INK, INK2, MUTED, savefig, setup_style  # noqa: E402
from analysis.video_census import load_census  # noqa: E402
from analysis.video_common import (CACHE, LAYERS, cached_acts, load, target_matrix,  # noqa: E402
                                   unit_name, windows)
from analysis.video_viz import clean, overlay  # noqa: E402

SLOTS = [("posture", "enc"), ("posture", "tmp"), ("velocity", "tmp"), ("velocity", "tmp"),
         ("accel", "tmp"), ("contact", "enc"), ("contact", "tmp"), ("gravity", "any"),
         ("interaction", "any"), ("energy", "any"), ("applied", "any"), ("unexplained", "tmp")]


def select_units(c, vs):
    H = int(c["H"])
    chosen, used_vars = [], set()
    for fam, stage in SLOTS:
        cand = np.where(c["live"])[0]
        if stage != "any":
            cand = cand[[LAYERS[u // H].startswith(stage) for u in cand]]
        if fam == "unexplained":
            cand = cand[c["status"][cand] == "unexplained"]
            # pick the unexplained unit with the largest activation variance proxy: lowest best R2 among top-variance
            order = cand[np.argsort(c["best_r2"][cand])]
            pick = [u for u in order if u not in [x[0] for x in chosen]][:1]
        else:
            cand = cand[(c["fam"][cand] == fam) & (c["best_r2"][cand] >= 0.25)]
            order = cand[np.argsort(-c["best_r2"][cand])]
            fresh = [u for u in order if vs[c["best"][u]].name not in used_vars and u not in [x[0] for x in chosen]]
            pick = fresh[:1] or [u for u in order if u not in [x[0] for x in chosen]][:1]
        if pick:
            u = int(pick[0])
            chosen.append((u, fam, stage))
            used_vars.add(vs[c["best"][u]].name)
    return chosen


@torch.no_grad()
def unit_maps(model, layer, ch, E, T):
    model.record = True
    x = windows(E, T)
    model(x)
    model.record = False
    return model.acts[layer][:, ch].cpu().numpy(), x.cpu().numpy().transpose(0, 1, 3, 4, 2)


def top_distinct(vals, E, k, largest=True):
    order = np.argsort(-vals if largest else vals)
    out, seen = [], set()
    for i in order:
        if E[i] not in seen:
            out.append(i); seen.add(E[i])
        if len(out) == k:
            break
    return np.array(out)


def tuning_curve(ax, x, a, color, label):
    edges = np.quantile(x, np.linspace(0, 1, 21))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 19)
    mids = np.array([x[idx == b].mean() if (idx == b).any() else np.nan for b in range(20)])
    mean = np.array([a[idx == b].mean() if (idx == b).any() else np.nan for b in range(20)])
    lo = np.array([np.percentile(a[idx == b], 10) if (idx == b).any() else np.nan for b in range(20)])
    hi = np.array([np.percentile(a[idx == b], 90) if (idx == b).any() else np.nan for b in range(20)])
    ax.fill_between(mids, lo, hi, color=color, alpha=0.2, lw=0)
    ax.plot(mids, mean, color=color, lw=1.5)
    ax.set_xlabel(label, fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5, pad=1)
    ax.grid(True, lw=0.4)


def draw(units, c, acts, model, vs, fname, title):
    import matplotlib.pyplot as plt
    setup_style()
    Y, _, st = target_matrix()
    E, T = acts["eps"], acts["ts"]
    H = int(c["H"])
    n = len(units)
    ncol = 10
    fig = plt.figure(figsize=(7.2, 0.86 * n + 0.35))
    gs = fig.add_gridspec(n, ncol, width_ratios=[1.35, 1, 1, 1, 1, 1, 1.45, 1, 1, 1],
                          wspace=0.08, hspace=0.28, left=0.01, right=0.99, top=1 - 0.3 / (0.86 * n + 0.35), bottom=0.02)
    for r, (u, fam, stage) in enumerate(units):
        layer, ch = LAYERS[u // H], u % H
        a = acts["max"][:, u].astype(np.float32)
        b = int(c["best"][u]); v = vs[b]
        col = FAMILIES[v.family][0] if c["status"][u] != "unexplained" else FAMILIES["none"][0]
        top = top_distinct(a, E, 5, True)
        bot = top_distinct(a, E, 3, False)
        sel = np.concatenate([top, bot])
        maps, clips = unit_maps(model, layer, ch, E[sel], T[sel])
        vmax = max(maps.max(), 1e-6)
        vmin = float(np.median(maps))          # unit's typical level -> overlay shows where it is *elevated*
        ax = fig.add_subplot(gs[r, 0]); clean(ax, False)
        ax.text(0.0, 0.92, f"{layer}.{ch}", fontsize=8, fontweight="bold", color=INK, transform=ax.transAxes, va="top")
        lab = v.label if c["status"][u] != "unexplained" else f"(best: {v.label})"
        import textwrap
        ax.text(0.0, 0.66, textwrap.fill(lab, 22), fontsize=5.8, color=col, transform=ax.transAxes, va="top",
                linespacing=1.1)
        tag = c["status"][u] + (" · conj." if c["conj"][u] else "")
        ax.text(0.0, 0.30, f"R² = {c['best_r2'][u]:.2f}", fontsize=6, color=INK2, transform=ax.transAxes, va="top")
        ax.text(0.0, 0.12, tag, fontsize=5.5, color=MUTED, transform=ax.transAxes, va="top")
        for j in range(5):
            ax = fig.add_subplot(gs[r, 1 + j])
            overlay(ax, clips[j], maps[j], vmax, vmin=vmin)
            ax.text(0.03, 0.97, f"{a[sel[j]]:.1f}", fontsize=4.5, color=INK2, transform=ax.transAxes, va="top")
            if r == 0 and j == 2:
                ax.set_title("strongest activations (distinct episodes)", fontsize=7, loc="center", fontweight="normal", color=INK2)
        ax = fig.add_subplot(gs[r, 6])
        x = Y[E, T, b]
        tuning_curve(ax, x, a, col, v.label)
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.022, pos.y0 + 0.012, pos.width - 0.03, pos.height - 0.012])
        if r == 0:
            ax.set_title("tuning (mean, 10–90%)", fontsize=7, loc="center", fontweight="normal", color=INK2)
        for j in range(3):
            ax = fig.add_subplot(gs[r, 7 + j])
            overlay(ax, clips[5 + j], maps[5 + j], vmax, color=(0.1, 0.45, 0.85), vmin=vmin)
            ax.text(0.03, 0.97, f"{a[sel[5 + j]]:.2f}", fontsize=4.5, color=INK2, transform=ax.transAxes, va="top")
            if r == 0 and j == 1:
                ax.set_title("weakest", fontsize=7, loc="center", fontweight="normal", color=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold", y=1.0)
    savefig(fig, fname)


def main():
    Y, vs, st = target_matrix()
    c = load_census("final")
    acts = cached_acts("final", "test", lambda: load("final")[0], res=True)
    model, _ = load("final")
    units = select_units(c, vs)
    info = [{"unit": int(u), "name": unit_name(u, int(c["H"])), "slot": f, "stage": s,
             "best_var": vs[int(c["best"][u])].name, "best_r2": float(c["best_r2"][u]), "status": str(c["status"][u]),
             "conj": bool(c["conj"][u]),
             "partial_r2": float(c["R_partial"][u].max()) if "R_partial" in c else None,
             "r2_state": float(c["r2_state"][u]) if "r2_state" in c else None,
             "top3": [(vs[j].name, float(c["R"][u, j])) for j in np.argsort(-c["R"][u])[:3]]}
            for u, f, s in units]
    json.dump(info, open(os.path.join(CACHE, "units.json"), "w"), indent=1)
    for d in info:
        print(d)
    half = (len(units) + 1) // 2
    draw(units[:half], c, acts, model, vs, "video_gallery_a.png", "Neuron gallery I — what the units see")
    draw(units[half:], c, acts, model, vs, "video_gallery_b.png", "Neuron gallery II")


if __name__ == "__main__":
    main()
