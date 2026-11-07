"""Figure reports/figs/imnet_sae2.png: near-copy share vs held-out FVE across the essay-way SAE sweep (left), and the
activity-weighted distribution of decoder-to-nearest-neuron |cos| for one config per layer vs its controls (right)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import imnet_common as C  # noqa: E402
from analysis.common import INK2, MUTED, SERIES, setup_style  # noqa: E402

LAYER_NAME = {"g5b": "GoogLeNet inception5b (1024)", "cnt_mlp": "ConvNeXt-T last MLP (3072)",
              "cnt_res": "ConvNeXt-T residual (768)", "cns_mlp": "ConvNeXt-S last MLP (3072)"}
LAYER_COL = {"g5b": SERIES[0], "cnt_mlp": SERIES[1], "cnt_res": SERIES[2], "cns_mlp": SERIES[6]}
K_MARK = {2: "o", 8: "s", 32: "^", 64: "D", 128: "v"}
ESSAY_RED = "#b52f2f"


def main():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    setup_style()
    d = json.load(open(os.path.join(C.RESULTS, "imnet_sae2.json")))
    runs = d["runs"]
    old = json.load(open(os.path.join(C.RESULTS, "imnet_sae.json")))["imagenet_val"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw=dict(width_ratios=[1.25, 1]))

    # step-1 SAEs (ImageNetV2, no aux, 4k steps) as small grey points
    for t in ("googlenet", "convnext_tiny", "convnext_tiny_resid"):
        for kk in ("k8", "k32"):
            r = old.get(t, {}).get(kk)
            if r:
                a1.scatter(r["fve"], 100 * r["near_copy_share"], s=10, color="#bbbbbb", marker="o", zorder=2, lw=0)
    for name, r in runs.items():
        col = LAYER_COL[r["layer"]]
        m = K_MARK.get(r["k"], "o")
        s = 14 + 6 * np.log2(r["width"])
        if not r["trained"]:
            a1.scatter(r["fve"], 100 * r["near_copy_share"], s=s, marker=m, color="#999999", zorder=3, lw=0.6,
                       edgecolor="white")
        elif r["rotated"]:
            a1.scatter(r["fve"], 100 * r["near_copy_share"], s=s, marker=m, facecolor="none", edgecolor=col, lw=1,
                       zorder=4)
        else:
            a1.scatter(r["fve"], 100 * r["near_copy_share"], s=s, marker=m, color=col, zorder=5, lw=0.6,
                       edgecolor="white")
    a1.axhline(88, color=ESSAY_RED, lw=1, ls="--")
    a1.scatter([0.82], [88], marker="*", s=90, color=ESSAY_RED, zorder=6, lw=0)
    a1.text(0.81, 90.5, "essay: 88% of activity, FVE 0.82", color=ESSAY_RED, fontsize=6.3, ha="right", va="bottom")
    a1.set_xlim(0, 1)
    a1.set_ylim(-3, 100)
    a1.set_xlabel("held-out FVE (ImageNet val positions)")
    a1.set_ylabel("near-copy share of SAE activity (%)")
    a1.set_title("Near-copies of single neurons vs reconstruction", fontsize=8, loc="left")
    h = [Line2D([], [], ls="", marker="o", color=LAYER_COL[L], label=LAYER_NAME[L])
         for L in LAYER_COL if any(r["layer"] == L for r in runs.values())]
    h += [Line2D([], [], ls="", marker=K_MARK[k], color=INK2, label=f"k = {k}")
          for k in K_MARK if any(r["k"] == k for r in runs.values())]
    h += [Line2D([], [], ls="", marker="o", mfc="none", mec=INK2, label="rotated-basis control"),
          Line2D([], [], ls="", marker="o", color="#999999", label="random-init network"),
          Line2D([], [], ls="", marker="o", color="#bbbbbb", ms=4, label="step-1 SAEs (ImageNetV2)")]
    a1.legend(handles=h, fontsize=5.4, loc="upper left", bbox_to_anchor=(0.0, 0.86), frameon=False, ncol=1)
    a1.text(0.99, 0.01, "marker size = width (4x, 16x, 64x)", transform=a1.transAxes, fontsize=5.4, color=MUTED,
            ha="right", va="bottom")

    # right: activity-weighted |cos| histograms, 16x k=2 per layer, trained vs rotated vs init
    edges = np.linspace(0, 1, 21)
    ctr = (edges[:-1] + edges[1:]) / 2
    for L in ("g5b", "cnt_res", "cnt_mlp", "cns_mlp"):
        r = runs.get(f"{L}_w16_k2")
        if not r:
            continue
        a2.plot(ctr, 100 * np.asarray(r["cos_hist_weighted"]), color=LAYER_COL[L], lw=1.6, label=LAYER_NAME[L],
                drawstyle="steps-mid")
        rr = runs.get(f"{L}_w16_k2_rot")
        if rr:
            a2.plot(ctr, 100 * np.asarray(rr["cos_hist_weighted"]), color=LAYER_COL[L], lw=0.9, ls=":",
                    drawstyle="steps-mid")
    a2.axvline(0.9, color=MUTED, lw=0.8, ls="--")
    a2.text(0.89, a2.get_ylim()[1] * 0.97, "near-copy\n|cos| ≥ 0.9", fontsize=5.6, color=MUTED, ha="right", va="top")
    a2.set_xlabel("max |cos| of decoder row to a neuron axis")
    a2.set_ylabel("share of SAE activity (%)")
    a2.set_title("Where the activity sits (16x, k = 2; dotted = rotated)", fontsize=8, loc="left")
    a2.legend(fontsize=5.4, frameon=False, loc="upper left")
    a2.set_xlim(0, 1)
    fig.tight_layout()
    fn = os.path.join(C.FIGS, "imnet_sae2.png")
    fig.savefig(fn, dpi=200)
    print("wrote", fn)


if __name__ == "__main__":
    main()
