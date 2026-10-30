"""B3: blind, free-form descriptions of randomly chosen units ("verbs, not nouns").

Stage "strips": draw 40 units at random (seeded; 20 vision, 20 LM; not chosen by the census), and
for each write one image reports/figs/vlm_blind/U##.png with its top-12 and bottom-12 temporal
groups (distinct episodes; each tile = the group's later frame with a fading trail of the clip's
earlier frames, the strongest patch/token outlined, and the raw activation value) plus a histogram
of the raw pooled activation. The images carry no unit id, layer, label or ground-truth value.
The key (U## -> unit) is written to data/vlm/blind_key.json, which the describer must not read.
  vision units: top = max of the pre-GELU fc1 output over the 400 patches, bottom = min (the - pole)
  LM units: top = max of silu(g)*v over the 100 visual tokens, bottom = min (the other SwiGLU pole)

Stage "join": after the describer has saved reports/results/vlm_blind_descriptions.json, attach the
census labels (both poles, from census_units.npz / negpole census) and write data/vlm/blind.json.

Usage: uv run python analysis/vlm_blind.py [strips|join]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import ROOT, FIGS, binned_r2, load_states, savefig, setup_style, targets  # noqa: E402
from analysis.vlm_common import LLM_LAYERS, VDIR, VIS_LAYERS, clip_frames, group_index, n_used, renderer  # noqa: E402
from analysis.vlm_gallery import trail_image, draw_box  # noqa: E402

N_VIS, N_LM, K = 20, 20, 12
BDIR = os.path.join(FIGS, "vlm_blind")
DESC = os.path.join(ROOT, "reports", "results", "vlm_blind_descriptions.json")


def unit_arrays(part, l, u, NG):
    if part == "vision":
        Z = np.load(os.path.join(VDIR, f"video_vis_pre_L{l}.npy"), mmap_mode="r")
        A = np.load(os.path.join(VDIR, f"video_vis_pre_L{l}_arg.npy"), mmap_mode="r")
    else:
        Z = np.load(os.path.join(VDIR, f"video_llm_mlp_L{l}.npy"), mmap_mode="r")
        A = np.load(os.path.join(VDIR, f"video_llm_mlp_L{l}_arg.npy"), mmap_mode="r")
    return (np.asarray(Z[:NG, 0, u], np.float32), np.asarray(Z[:NG, 1, u], np.float32),
            np.asarray(A[:NG, 0, u]), np.asarray(A[:NG, 1, u]))


def choose(NG):
    rng = np.random.default_rng(2026)
    units = []
    for part, layers, n, width, tmpl in [("vision", VIS_LAYERS, N_VIS, 4096, "video_vis_mlp_L{}.npy"),
                                         ("LM", LLM_LAYERS, N_LM, 9728, "video_llm_mlp_L{}.npy")]:
        per = n // len(layers)
        for l in layers:
            Z = np.load(os.path.join(VDIR, tmpl.format(l)), mmap_mode="r")
            mx = np.asarray(Z[:NG, 0], np.float32)
            live = np.where((mx.std(0) > 1e-3) & (np.ptp(mx, 0) > 1e-2))[0]
            for u in rng.choice(live, per, replace=False):
                units.append((part, int(l), int(u)))
    order = rng.permutation(len(units))
    return [units[i] for i in order]


def distinct(order, gi, k):
    out, seen = [], set()
    for i in order:
        if gi[i, 0] not in seen:
            out.append(i); seen.add(gi[i, 0])
        if len(out) == k:
            break
    return out


def strips():
    setup_style()
    import matplotlib.pyplot as plt
    st = load_states()
    NG, _ = n_used("video")
    gi = group_index()[:NG]
    units = choose(NG)
    os.makedirs(BDIR, exist_ok=True)
    key = {}
    for bi, (part, l, u) in enumerate(units):
        bid = f"U{bi + 1:02d}"
        key[bid] = dict(part=part, layer=l, unit=u)
        mx, mn, amx, amn = unit_arrays(part, l, u, NG)
        top = distinct(np.argsort(-mx), gi, K)
        bot = distinct(np.argsort(mn), gi, K)
        grid = 20 if part == "vision" else 10
        fig = plt.figure(figsize=(7.0, 5.6))
        gs = fig.add_gridspec(5, 6, height_ratios=[1, 1, 1, 1, 0.9], hspace=0.25, wspace=0.05,
                              left=0.02, right=0.98, top=0.93, bottom=0.06)
        for side, (ex, vals, args, col, ttl) in enumerate([(top, mx, amx, "#d6186e", "12 strongest positive activations"),
                                                           (bot, mn, amn, "#0aa5a5", "12 strongest negative activations")]):
            for j, i in enumerate(ex):
                ax = fig.add_subplot(gs[2 * side + j // 6, j % 6])
                clip = renderer()(st["q"][gi[i, 0]][clip_frames(gi[i, 4] - 4 * gi[i, 2])])
                draw_box(ax, trail_image(clip, 2 * gi[i, 2] + 1)[::2, ::2], int(args[i]), grid, col)
                ax.text(0.97, 0.04, f"{vals[i]:+.2f}", transform=ax.transAxes, fontsize=6, ha="right", color=col,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
                if j == 0:
                    ax.set_title(ttl, fontsize=7.5, loc="left", fontweight="normal")
        ax = fig.add_subplot(gs[4, :])
        ax.hist(mx, bins=120, color="#d6186e", alpha=0.6, label="pooled max (per frame group)")
        ax.hist(mn, bins=120, color="#0aa5a5", alpha=0.6, label="pooled min")
        ax.set_yscale("log")
        ax.set_xlabel("raw activation", fontsize=7)
        ax.legend(fontsize=6.5, loc="upper right")
        fig.suptitle(f"{bid}: what do these clips share?", x=0.01, ha="left", fontsize=9.5, fontweight="bold")
        fig.savefig(os.path.join(BDIR, f"{bid}.png"), dpi=130)
        plt.close(fig)
        print(bid, flush=True)
    with open(os.path.join(VDIR, "blind_key.json"), "w") as f:
        json.dump(key, f, indent=1)


def join():
    """Attach census labels to the blind descriptions (run only after the descriptions exist)."""
    with open(DESC) as f:
        desc = json.load(f)
    with open(os.path.join(VDIR, "blind_key.json")) as f:
        key = json.load(f)
    st = load_states()
    Yall, vs = targets(st)
    names = [v.name for v in vs]
    NG, _ = n_used("video")
    gi = group_index()[:NG]
    Y = Yall[gi[:, 0], gi[:, 3]]
    out = {}
    for bid, k in key.items():
        mx, mn, _, _ = unit_arrays(k["part"], k["layer"], k["unit"], NG)
        if k["part"] == "vision":   # + pole label as in the census: post-GELU max
            mx = np.asarray(np.load(os.path.join(VDIR, f"video_vis_mlp_L{k['layer']}.npy"),
                                    mmap_mode="r")[:NG, 0, k["unit"]], np.float32)
        lab = {}
        for pole, a in (("+", mx), ("−", -mn)):
            r = np.array([binned_r2(Y[:, j], a) for j in range(len(names))])
            # strong-activation version (top 1% of groups): which variable separates them?
            lab[pole] = dict(best=names[int(r.argmax())], r2=float(r.max()),
                             cls="clean" if r.max() >= 0.5 else "partial" if r.max() >= 0.25 else "unexplained",
                             top3=[(names[j], float(r[j])) for j in np.argsort(-r)[:3]])
        out[bid] = dict(**k, description=desc.get(bid), census=lab)
    with open(os.path.join(VDIR, "blind.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"strips": strips, "join": join}[sys.argv[1]]()
