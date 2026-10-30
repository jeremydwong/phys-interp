"""Unit census for Qwen3-VL-4B MLP hidden units on arm videos.

Unit activation = per temporal group, the max over the group's spatial tokens (400 vision
patches or 100 LM visual tokens) of the MLP hidden unit:
  vision: GELU(fc1 x)  (ungated; negative pole bounded at -0.17, so only the + pole is read)
  LM:     silu(gate x) * (up x)  (SwiGLU; both poles read: "+" = max-pool, "−" = −min-pool)

For each live unit we compute common.binned_r2 against all 32 targets (aligned to the group's
midpoint sim frame) and classify: clean (best >= 0.5), partial (0.25-0.5), unexplained.
Conjunction flag: the best variables from two different families both reach >= 0.25 and a
joint 6x6 quantile-bin model beats the best single variable by >= 0.10.
Partial r² for dynamics: squared Pearson r between the unit and y − E[y|posture, velocity]
(see vlm_probe.residualise); null = same statistic with episodes of the residual permuted.

Usage: uv run python analysis/vlm_census.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, r2_per_unit, savefig, setup_style, INK2, MUTED  # noqa: E402
from analysis.vlm_common import VDIR, VIS_LAYERS, LLM_LAYERS  # noqa: E402
from analysis.vlm_probe import aligned_targets, residualise, DYN  # noqa: E402

BINS = 30


def bin_index(x, bins=BINS):
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)


def binned_r2_matrix(A, Y, bins=BINS):
    """Vectorised common.binned_r2(Y[:, v], A[:, u]) for all units/targets -> [U, V]."""
    A = A.astype(np.float64)
    mu, var = A.mean(0), A.var(0) + 1e-12
    out = np.zeros((A.shape[1], Y.shape[1]))
    for v in range(Y.shape[1]):
        idx = bin_index(Y[:, v], bins)
        cnt = np.bincount(idx, minlength=bins).astype(np.float64)
        oh = np.zeros((bins, len(idx)))
        oh[idx, np.arange(len(idx))] = 1
        S = oh @ A
        means = S / np.maximum(cnt, 1)[:, None]
        expl = (cnt[:, None] * (means - mu) ** 2).sum(0) / len(A)
        out[:, v] = expl / var
    return out


def joint_r2(a, x1, x2, b=6):
    i = bin_index(x1, b) * b + bin_index(x2, b)
    cnt = np.bincount(i, minlength=b * b)
    m = np.bincount(i, a, b * b) / np.maximum(cnt, 1)
    return 1 - (a - m[i]).var() / (a.var() + 1e-12)


def census_layer(A, Y, R, fams, names):
    live = (A.std(0) > 1e-3) & (np.ptp(A, 0) > 1e-2)
    Al = A[:, live].astype(np.float32)
    B = binned_r2_matrix(Al, Y)
    best = B.argmax(1)
    bestr = B.max(1)
    cls = np.where(bestr >= 0.5, 2, np.where(bestr >= 0.25, 1, 0))
    fam_arr = np.array(fams)
    conj = np.zeros(len(best), bool)
    second = np.full(len(best), -1)
    for u in np.where(bestr >= 0.25)[0]:
        other = np.where(fam_arr != fams[best[u]])[0]
        j = other[B[u, other].argmax()]
        if B[u, j] >= 0.25:
            second[u] = j
            jr = joint_r2(Al[:, u], Y[:, best[u]], Y[:, j])
            conj[u] = jr >= bestr[u] + 0.10
    di = [names.index(n) for n in DYN if n in names]
    P = r2_per_unit(Al, R[:, di])
    return dict(live=live, B=B, best=best, bestr=bestr, cls=cls, conj=conj, second=second, partial=P)


def main(only_randvis=False):
    Y, vs, gi, st = aligned_targets()
    names, fams = [v.name for v in vs], [v.family for v in vs]
    R, _ = residualise(Y, vs, st, gi)
    di = [names.index(n) for n in DYN]
    # null for partial r²: permute episodes of residual targets
    rng = np.random.default_rng(1)
    eps = np.unique(gi[:, 0])
    perm = dict(zip(eps, rng.permutation(eps)))
    order = np.concatenate([np.where(gi[:, 0] == perm[e])[0] for e in eps])
    Rnull = R[order]

    layers = [("vision", l, "video", "+", f"video_vis_mlp_L{l}.npy") for l in VIS_LAYERS] + \
             [("vision", l, "image", "+", f"image_vis_mlp_L{l}.npy") for l in VIS_LAYERS] + \
             [("LM", l, "video", s, f"video_llm_mlp_L{l}.npy") for l in LLM_LAYERS for s in "+−"] + \
             [("vision", l, "randvis", "+", f"randvis_vis_mlp_L{l}.npy") for l in VIS_LAYERS]
    summary, units = [], {}
    if only_randvis:
        # random-init control only: append to the existing census (keeps the trained-model entries)
        layers = [x for x in layers if x[2] == "randvis"]
        with open(os.path.join(VDIR, "census.json")) as f:
            summary = [s for s in json.load(f) if s["mode"] != "randvis"]
        z = np.load(os.path.join(VDIR, "census_units.npz"))
        for k in z.files:
            key, kk = k.split("__")
            if not key.startswith("randvis"):
                units.setdefault(key, {})[kk] = z[k]
    for part, l, mode, pole, fn in layers:
        p = os.path.join(VDIR, fn)
        if not os.path.exists(p):
            continue
        Z = np.load(p, mmap_mode="r")
        A = np.asarray(Z[:, 0] if pole == "+" else -Z[:, 1], np.float32)
        c = census_layer(A, Y, R, fams, names)
        Pnull = r2_per_unit(A[:, c["live"]], Rnull[:, di])
        thr = np.quantile(Pnull.max(1), 0.999)
        key = f"{mode}_{part}_L{l}_{'pos' if pole == '+' else 'neg'}"
        nl = int(c["live"].sum())
        famcount = {f: [int(((c["cls"] == k) & (np.array(fams)[c["best"]] == f)).sum()) for k in (2, 1)] for f in FAMILIES if f != "none"}
        pmax = c["partial"].max(1)
        s = dict(key=key, part=part, layer=l, mode=mode, pole=pole, n_units=int(A.shape[1]), n_live=nl,
                 n_clean=int((c["cls"] == 2).sum()), n_partial=int((c["cls"] == 1).sum()),
                 n_unexplained=int((c["cls"] == 0).sum()), n_conj=int(c["conj"].sum()),
                 by_family_clean_partial=famcount,
                 partial_dyn_max=float(pmax.max()), partial_dyn_null_999=float(thr),
                 n_partial_dyn_above_0p1=int((pmax >= 0.1).sum()),
                 n_partial_dyn_above_null=int((pmax > thr).sum()),
                 partial_dyn_best_target=DYN[int(c["partial"].max(0).argmax())],
                 partial_dyn_max_by_target={DYN[j]: float(c["partial"][:, j].max()) for j in range(len(DYN))})
        print(json.dumps({k: v for k, v in s.items() if k not in ("by_family_clean_partial", "partial_dyn_max_by_target")}), flush=True)
        summary.append(s)
        units[key] = dict(idx=np.where(c["live"])[0], best=c["best"], bestr=c["bestr"], cls=c["cls"], conj=c["conj"],
                          second=c["second"], B=c["B"].astype(np.float16), partial=c["partial"].astype(np.float16))
    with open(os.path.join(VDIR, "census.json"), "w") as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(os.path.join(VDIR, "census_units.npz"),
                        **{f"{k}__{kk}": vv for k, d in units.items() for kk, vv in d.items()})
    figure(summary)


def figure(summary=None):
    setup_style()
    import matplotlib.pyplot as plt
    if summary is None:
        with open(os.path.join(VDIR, "census.json")) as f:
            summary = json.load(f)
    fam_order = ["posture", "velocity", "accel", "gravity", "interaction", "contact", "applied", "energy"]
    groups = [("Vision MLP, video", [s for s in summary if s["part"] == "vision" and s["mode"] == "video"]),
              ("Vision MLP, random init", [s for s in summary if s["part"] == "vision" and s["mode"] == "randvis"]),
              ("Vision MLP, single image", [s for s in summary if s["part"] == "vision" and s["mode"] == "image"]),
              ("LM MLP (+ pole), video", [s for s in summary if s["part"] == "LM" and s["pole"] == "+"]),
              ("LM MLP (− pole), video", [s for s in summary if s["part"] == "LM" and s["pole"] == "−"])]
    groups = [g for g in groups if g[1]]
    fig, axes = plt.subplots(1, len(groups), figsize=(7.0, 3.0), sharey=True)
    for ax, (title, ss) in zip(axes, groups):
        for i, s in enumerate(ss):
            bottom = 0.0
            for f in fam_order:
                cl, pa = s["by_family_clean_partial"][f]
                c = FAMILIES[f][0]
                ax.bar(i, cl / s["n_live"], bottom=bottom, color=c, width=0.75)
                bottom += cl / s["n_live"]
            for f in fam_order:
                cl, pa = s["by_family_clean_partial"][f]
                ax.bar(i, pa / s["n_live"], bottom=bottom, color=FAMILIES[f][0], alpha=0.38, width=0.75)
                bottom += pa / s["n_live"]
            ax.bar(i, 1 - bottom, bottom=bottom, color=FAMILIES["none"][0], alpha=0.5, width=0.75)
        ax.set_xticks(range(len(ss)), [f"{'V' if s['part'] == 'vision' else 'L'}{s['layer']}" for s in ss], fontsize=7)
        ax.set_title(title.replace(", ", ",\n", 1), fontsize=7.5)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("fraction of live units")
    from matplotlib.patches import Patch
    h = [Patch(color=FAMILIES[f][0], label=f) for f in fam_order] + \
        [Patch(color=MUTED, alpha=0.38, label="light = partial (0.25–0.5)"), Patch(color=FAMILIES["none"][0], alpha=0.5, label="unexplained")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Unit census: best-explaining physical variable (binned R², max-pooled per frame group)",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    savefig(fig, "vlm_census.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fig":
        figure()
    elif len(sys.argv) > 1 and sys.argv[1] == "randvis":
        main(only_randvis=True)
    else:
        main()
