"""E1: SwiGLU sign regimes ("echo") for the LM units.

The LM MLP unit is a = silu(g) * v. Its pooled extreme on a temporal group can come from four sign
regimes of (g, v): + x + and - x - give a > 0; + x - and - x + give a < 0 (silu(g) < 0 only down to
-0.28, so the gate-negative regimes are small in magnitude). The essay ("Separating the Sign
Regimes") found that for ~4% of neurons the top of the list is - x -, an "echo" of the other pole.

Input: video_llm_reg_L{l}.npy from vlm_recapture.py (regime of the max and of the min token per
group) and the cached pooled values video_llm_mlp_L{l}.npy.

Definitions (ours): for the + pole the regular regime is + x + and the echo is - x -; for the - pole
the regular regime is + x - and the echo is - x + (gate negative in both echoes). A unit's pole is
re-labelled "echo" if > 50% of its top-100 groups for that pole are in the echo regime.
Rank bands: top 1-10, 11-50, 51-100, the essay's p75-p99 band of the pooled value, and the
opposite pole's top 100.

Usage: uv run python analysis/vlm_regimes.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import savefig, setup_style, INK, INK2, MUTED  # noqa: E402
from analysis.vlm_common import LLM_LAYERS, VDIR  # noqa: E402

REG = ["+×+", "+×−", "−×+", "−×−"]
REG_COL = ["#2a78d6", "#eb6834", "#9ec5f4", "#b52f2f"]
ECHO = {"+": 3, "−": 2}
BANDS = [("top 1–10", 0, 10), ("11–50", 10, 50), ("51–100", 50, 100)]


def pole_values(Z, pole):
    return np.asarray(Z[:, 0] if pole == "+" else -Z[:, 1], np.float32)


def band_regimes(a, reg_stat, lo_hi=None, idx=None):
    """Fractions of the 4 regimes among the given rows."""
    c = np.bincount(reg_stat[idx].astype(np.int64), minlength=4)[:4]
    return (c / max(c.sum(), 1)).tolist()


def unit_profile(Z, R, u, pole):
    a = np.asarray(Z[:, 0, u] if pole == "+" else -Z[:, 1, u], np.float32)
    b = np.asarray(-Z[:, 1, u] if pole == "+" else Z[:, 0, u], np.float32)      # opposite pole
    rs = np.asarray(R[:, 0 if pole == "+" else 1, u])
    ro = np.asarray(R[:, 1 if pole == "+" else 0, u])
    order = np.argsort(-a)
    out = {}
    for name, lo, hi in BANDS:
        out[name] = band_regimes(a, rs, idx=order[lo:hi])
    p75, p99 = np.percentile(a, [75, 99])
    out["p75–p99"] = band_regimes(a, rs, idx=np.where((a >= p75) & (a < p99))[0])
    out["opposite pole top 100"] = band_regimes(b, ro, idx=np.argsort(-b)[:100])
    top100 = order[:100]
    echo_frac = float((rs[top100] == ECHO[pole]).mean())
    return out, echo_frac, a[order[:100]].mean()


def main():
    setup_style()
    import matplotlib.pyplot as plt
    with open(os.path.join(VDIR, "gallery_spec.json")) as f:
        spec = [s for s in json.load(f) if s["part"] == "LM"]
    res = {"definition": __doc__.split("Definitions (ours):")[1].split("Usage")[0].strip(), "gallery": {}, "population": {}}
    # population: per layer and pole, fraction of each unit's top-100 groups in the echo regime
    pop = {}
    for l in LLM_LAYERS:
        Z = np.load(os.path.join(VDIR, f"video_llm_mlp_L{l}.npy"), mmap_mode="r")
        R = np.load(os.path.join(VDIR, f"video_llm_reg_L{l}.npy"), mmap_mode="r")
        Zl = np.asarray(Z[:, :2], np.float32)
        Rl = np.asarray(R)
        for pole in "+−":
            a = Zl[:, 0] if pole == "+" else -Zl[:, 1]
            rs = Rl[:, 0] if pole == "+" else Rl[:, 1]
            live = (a.std(0) > 1e-3) & (np.ptp(a, 0) > 1e-2)
            top = np.argpartition(-a, 100, axis=0)[:100]                # [100, U] unordered
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(a, top, 0), axis=0), 0)
            tr = np.take_along_axis(rs, top, 0)                        # [100, U]
            ef = (tr == ECHO[pole]).mean(0)[live]
            regfrac = np.stack([(tr == k).mean(0)[live] for k in range(4)], 1).mean(0)
            pop[(l, pole)] = ef
            ef_all = (tr == ECHO[pole]).mean(0)
            if pole == "+":
                ef_plus, live_plus = ef_all, live
            else:
                lb = live & live_plus
                res["population"][f"L{l}"] = dict(
                    frac_units_gate_closed_both_poles=float(((ef_plus > 0.5) & (ef_all > 0.5))[lb].mean()),
                    frac_units_regular_both_poles=float(((ef_plus <= 0.5) & (ef_all <= 0.5))[lb].mean()))
            res["population"][f"L{l}{pole}"] = dict(
                n_live=int(live.sum()), frac_units_echo_majority=float((ef > 0.5).mean()),
                frac_units_echo_whole_top10=float(((tr[:10] == ECHO[pole]).mean(0)[live] == 1).mean()),
                mean_echo_frac=float(ef.mean()), mean_regime_frac_top100={REG[k]: float(regfrac[k]) for k in range(4)})
            print(l, pole, res["population"][f"L{l}{pole}"], flush=True)
        # gallery units in this layer
        for sp in spec:
            if sp["layer"] != l:
                continue
            u, pole = sp["unit"], sp["pole"]
            prof, ef, mean_top = unit_profile(Z, R, u, pole)
            label = f"L{l}{'−' if pole == '−' else '+'}#{u}"
            res["gallery"][label] = dict(var=sp["var"], census_r2=sp["r2"], pole=pole, regimes=prof,
                                         echo_frac_top100=ef, relabel_echo=bool(ef > 0.5))
            print(label, sp["var"], "echo", ef, flush=True)
    with open(os.path.join(VDIR, "regimes.json"), "w") as f:
        json.dump(res, f, indent=1)

    # ---- figure
    g = res["gallery"]
    units = list(g)
    bands = [b[0] for b in BANDS] + ["p75–p99", "opposite pole top 100"]
    fig = plt.figure(figsize=(7.0, 5.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.62, left=0.08, right=0.98, top=0.84, bottom=0.08)
    gsa = gs[0].subgridspec(1, len(units), wspace=0.12)
    for ui, un in enumerate(units):
        ax = fig.add_subplot(gsa[0, ui])
        bottom = np.zeros(len(bands))
        for k in range(4):
            v = np.array([g[un]["regimes"][b][k] for b in bands])
            ax.bar(range(len(bands)), v, bottom=bottom, color=REG_COL[k], width=0.8, label=REG[k])
            bottom += v
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(bands)), ["1–10", "11–50", "51–100", "p75–99", "opp."], rotation=90, fontsize=6)
        ax.set_title(f"{un}\n{g[un]['var']}", fontsize=6.5, fontweight="normal", loc="center")
        ax.grid(axis="x", visible=False)
        if ui:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("fraction of groups", fontsize=7)
        if g[un]["relabel_echo"]:
            ax.text(0.5, 0.5, "gate closed\n(echo)", transform=ax.transAxes, ha="center", va="center", fontsize=6,
                    color="#b52f2f", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    h, l_ = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper left", ncol=4, fontsize=7, bbox_to_anchor=(0.06, 0.955), title=None)
    fig.text(0.62, 0.935, "sign of (gate, up) at the pooled-extreme token", fontsize=7, color=INK2, va="center")
    fig.text(0.01, 0.975, "A  Sign regime of the pooled-extreme token, gallery LM units (rank bands of the census pole)",
             fontsize=9, fontweight="bold", va="top")
    ax = fig.add_subplot(gs[1])
    pos = []
    labels = []
    for i, l in enumerate(LLM_LAYERS):
        for j, pole in enumerate("+−"):
            x = i * 2.6 + j
            ef = pop[(l, pole)]
            parts = ax.violinplot([ef], positions=[x], widths=0.85, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(REG_COL[3] if pole == "+" else REG_COL[2]); pc.set_alpha(0.6)
            frac = (ef > 0.5).mean()
            ax.text(x, 1.04, f"{100 * frac:.1f}%", ha="center", fontsize=6, color=INK2)
            pos.append(x); labels.append(f"L{l} {pole}")
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(pos, labels, fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("echo share of top 100", fontsize=7)
    ax.set_title("B  All live LM units: share of each unit's top-100 groups in the echo regime "
                 "(% above = units re-labelled echo, > 50%)", fontsize=8.5)
    savefig(fig, "vlm_regimes.png")


if __name__ == "__main__":
    main()
