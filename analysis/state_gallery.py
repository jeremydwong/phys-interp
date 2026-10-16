"""Analysis 3: Distill-style neuron gallery.

For ~12 exemplar units (chosen automatically from the census: the best unit for each of a list of
physical variables), show the top-5 held-out dataset examples (highest post-ReLU activation, at most one
frame per episode) and the bottom-3 examples (lowest pre-activation = the unit's negative pole), each
rendered as the arm pose with a 0.12 s motion trail of hand (dark) and elbow (grey) and a red arrow
showing 0.1 s of hand velocity. Left: unit id, best variable, binned R², and the unit's tuning curve
(mean activation in 30 quantile bins of the best variable, shaded 10-90% range).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, savefig, setup_style  # noqa: E402
from analysis.state_common import bin_index, data, draw_pose, load_census, load_model, pick_units, run, update_results  # noqa

CONTACT = ["contact", "c_floor", "c_wall", "c_fore", "ctq1", "ctq2"]
EXEMPLARS = [  # (variable, preferred best variables (fallback: any), short title)
    ("grav1", ["grav1"], "gravity torque · shoulder"),
    ("grav1", ["grav1"], "gravity torque · shoulder"),
    ("grav2", ["grav2", "cos_q12"], "gravity torque · elbow"),
    ("int1", ["int1", "cor1"], "interaction torque · shoulder"),
    ("cor2", ["int2", "cor2"], "Coriolis/centrip. · elbow"),
    ("qdd1", ["qdd1"], "acceleration · shoulder"),
    ("qdd2", ["qdd2"], "acceleration · elbow"),
    ("c_floor", CONTACT, "floor contact"),
    ("c_wall", CONTACT, "wall contact"),
    ("ctq2", CONTACT, "contact torque · elbow"),
    ("qd1", ["qd1"], "velocity · shoulder"),
    ("ee_z", ["ee_z", "elbow_z", "pe"], "hand height"),
]
UNITS = {"grav1": "N·m", "grav2": "N·m", "int1": "N·m", "int2": "N·m", "cor1": "N·m", "cor2": "N·m",
         "qdd1": "rad/s²", "qdd2": "rad/s²", "c_floor": "log(1+N)", "c_wall": "log(1+N)", "ctq1": "N·m",
         "ctq2": "N·m", "qd1": "rad/s", "qd2": "rad/s", "ee_z": "m", "pe": "J", "contact": "0/1", "c_fore": "log(1+N)"}
N_TOP, N_BOT, TRAIL = 5, 3, 6


def diverse_extreme(a, ep, k, largest=True):
    order = np.argsort(-a if largest else a)
    seen, out = set(), []
    for i in order:
        if ep[i] in seen:
            continue
        seen.add(ep[i]); out.append(i)
        if len(out) == k:
            break
    return out


def main():
    d = data()
    names, vs = d["names"], d["vars"]
    cz = load_census()
    model, _ = load_model("final")
    o = run(model, d["X"])
    H, P = o["h"], o["pre"]
    st, S = d["st"], d["S"]
    picks, excl = [], set()
    for var, best_in, title in EXEMPLARS:
        u = pick_units(cz, names, var, k=1, best_in=best_in, exclude=excl)
        if not u:
            u = pick_units(cz, names, var, k=1, exclude=excl)
        l, k, r2v = u[0]
        excl.add((l, k))
        picks.append((var, title, l, k, r2v))

    setup_style()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    n = len(picks)
    img_w = 0.64
    tw, cw, gap = 1.2, 0.62, 0.14
    W = tw + cw + 0.18 + (N_TOP + N_BOT) * img_w + gap
    Hh = n * img_w * 1.12 + 0.3
    fig = plt.figure(figsize=(W, Hh))
    gs = GridSpec(n, 3 + N_TOP + 1 + N_BOT, figure=fig,
                  width_ratios=[tw, cw, 0.18] + [img_w] * N_TOP + [gap] + [img_w] * N_BOT,
                  wspace=0.03, hspace=0.2, left=0.005, right=0.997, top=1 - 0.25 / Hh, bottom=0.25 / Hh)
    info = []
    for r, (var, title, l, k, r2v) in enumerate(picks):
        a, p = H[l][:, k], P[l][:, k]
        j = names.index(var)
        bj = int(cz["best"][l][k])
        fam = vs[j].family
        col = FAMILIES[fam][0]
        # text panel
        ax = fig.add_subplot(gs[r, 0]); ax.axis("off")
        ax.text(0, 0.88, f"L{l + 1} / {k}", fontsize=7.5, weight="bold", transform=ax.transAxes, va="top")
        ax.text(0, 0.64, title, fontsize=6.2, color=col, transform=ax.transAxes, va="top")
        ax.text(0, 0.42, f"R²({var}) = {r2v:.2f}  {cz['cat'][l][k] if bj == j else ''}", fontsize=5.8,
                color="#52514e", transform=ax.transAxes, va="top")
        lab = f"best: {names[bj]} ({cz['bestr'][l][k]:.2f})" if bj != j else ""
        if cz["conj"][l][k]:
            pa, pb = cz["pair"][l][k]
            lab += ("\n" if lab else "") + f"2-D {names[pa]}×{names[pb]}: {cz['pair_r2'][l][k]:.2f}"
        ax.text(0, 0.22, lab, fontsize=5.3, color="#898781", transform=ax.transAxes, va="top")
        # tuning curve
        ax = fig.add_subplot(gs[r, 1])
        x = d["Y"][:, j]
        idx = bin_index(x, 30)
        xm = np.array([np.median(x[idx == b]) for b in range(30) if (idx == b).any()])
        am = np.array([a[idx == b].mean() for b in range(30) if (idx == b).any()])
        lo = np.array([np.percentile(a[idx == b], 10) for b in range(30) if (idx == b).any()])
        hi = np.array([np.percentile(a[idx == b], 90) for b in range(30) if (idx == b).any()])
        ax.fill_between(xm, lo, hi, color=col, alpha=0.2, lw=0)
        ax.plot(xm, am, color=col, lw=1.2)
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        ax.locator_params(nbins=3)
        ax.grid(False)
        # examples
        top = diverse_extreme(a, d["ep"], N_TOP, True)
        bot = diverse_extreme(p, d["ep"], N_BOT, False)
        for c, i in enumerate(top + bot):
            gc = 3 + c if c < N_TOP else 4 + c
            ax = fig.add_subplot(gs[r, gc])
            e, t = d["ep"][i], d["t"][i]
            hist = st["q"][e, max(0, t - TRAIL):t]
            draw_pose(ax, st["q"][e, t], hist, vel=st["ee_vel"][e, t])
            val = a[i] if c < N_TOP else p[i]
            ax.text(0.03, 0.97, f"{val:.1f}\n", transform=ax.transAxes, fontsize=4.8, va="top",
                    color="#0b0b0b" if c < N_TOP else "#1c5cab",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.75))
            ax.text(0.03, 0.97, f"\n{x[i]:.1f}", transform=ax.transAxes, fontsize=4.8, va="top", color=col)
            if r == 0 and c == 0:
                ax.set_title("strongest activation", fontsize=7, loc="left")
            if r == 0 and c == N_TOP:
                ax.set_title("negative pole (min pre-act.)", fontsize=7, loc="left", color="#1c5cab")
        info.append(dict(unit=f"L{l + 1}/{k}", var=var, title=title, r2=r2v, best=names[bj],
                         best_r2=float(cz["bestr"][l][k]), category=str(cz["cat"][l][k]),
                         conjunction=bool(cz["conj"][l][k])))
    savefig(fig, "state_gallery.png")
    for i in info:
        print(i)
    update_results("gallery", info)


if __name__ == "__main__":
    main()
