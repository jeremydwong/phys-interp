"""E2: residual-stream "sink" dimensions in Qwen3-VL-4B (vision tower and LM).

The essay found a ConvNeXt putting 94.5% of the residual's raw energy into one unread coordinate.
Token-level statistics come from vlm_recapture.py (sink_stats.npz: per-coordinate sum of x^2 over
all vision tokens / all LM visual tokens / all LM tokens of the 2,700 clips, and per-token largest
coordinate share). For each layer we list the coordinates that hold > 50% and > 10% of the energy.

Then we re-run the probes with those coordinates removed (cached residuals, mean-pooled per
temporal group, 5-fold ridge grouped by episode, all 32 targets):
  (a) standardised ridge on all coordinates (as in vlm_probe.py) vs sinks removed;
  (b) a PCA probe (ours): PCA-64 of the *raw* (unstandardised) residual, then ridge on the 64 PCs,
      with vs without the sink coordinates. PCA is the analysis a sink would distort.
We also report the share of raw variance on PC1 and the PC1 loading on the sink coordinates.

Usage: uv run python analysis/vlm_sink.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, savefig, setup_style, INK, INK2, MUTED, SERIES  # noqa: E402
from analysis.vlm_common import VDIR, n_used  # noqa: E402
from analysis.vlm_probe import aligned_targets, folds, ridge_cv  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_LAYERS = {"vis": [0, 4, 8, 12, 16, 20, 23], "lm_vis": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]}
CACHE = {"vis": "video_vis_res.npy", "lm_vis": "video_llm_res.npy"}


def sinks(E, thr):
    share = E / E.sum(-1, keepdims=True)
    return [np.where(s > thr)[0].tolist() for s in share], share


def pca_probe(X, Y, fold, k=64):
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    Xt = Xt - Xt.mean(0)
    U, S, V = torch.pca_lowrank(Xt, q=k, center=False, niter=4)
    P = (Xt @ V).cpu().numpy()
    tot = (Xt ** 2).sum()
    pc1 = float(S[0] ** 2 / tot)
    r2 = ridge_cv(P, Y, fold, device=DEV, dtype=torch.float32).max(0)
    return r2, pc1, V[:, 0].cpu().numpy()


def main():
    z = np.load(os.path.join(VDIR, "sink_stats.npz"))
    res = {"token_level": {}, "probes": {}}
    S = {}
    for key in ("vis", "lm_vis", "lm_all"):
        E = z[f"E_{key}"]
        s50, share = sinks(E, 0.5)
        s10, _ = sinks(E, 0.1)
        n = float(z[f"NTOK_{key}"])
        top = share.argmax(1)
        topc = z[f"TOPC_{key}"]
        res["token_level"][key] = [dict(layer=l, top_coord=int(top[l]), top_share=float(share[l, top[l]]),
                                        coords_gt50=s50[l], coords_gt10=s10[l],
                                        tokens_with_coord_gt50=float(z[f"S50_{key}"][l] / n),
                                        tokens_with_coord_gt10=float(z[f"S10_{key}"][l] / n),
                                        most_frequent_top_coord=int(topc[l].argmax()),
                                        its_token_share=float(topc[l].max() / n)) for l in range(len(E))]
        S[key] = share
        print(key, [(l, int(top[l]), round(float(share[l, top[l]]), 3)) for l in range(len(E))], flush=True)
    res["argmax_match_recapture_vs_cache"] = z["argmax_match"].tolist()

    Y, vs, gi, st = aligned_targets()
    NG, _ = n_used("video")
    Y, gi = Y[:NG], gi[:NG]
    names, fams = [v.name for v in vs], [v.family for v in vs]
    fold = folds(gi[:, 0])
    for key, layers in PROBE_LAYERS.items():
        A = np.load(os.path.join(VDIR, CACHE[key]), mmap_mode="r")
        for l in layers:
            X = np.asarray(A[:NG, l], np.float32)
            sk = sorted(set(res["token_level"][key][l]["coords_gt10"]))
            # pooled-cache energy share of the sink coordinates (mean-pooling can hide or keep them)
            e_pool = (X ** 2).sum(0)
            pool_share = float(e_pool[sk].sum() / e_pool.sum()) if sk else 0.0
            keep = np.setdiff1d(np.arange(X.shape[1]), sk)
            r_std = ridge_cv(X, Y, fold, device=DEV, dtype=torch.float32).max(0)
            r_std_ns = ridge_cv(X[:, keep], Y, fold, device=DEV, dtype=torch.float32).max(0) if sk else r_std
            r_pca, pc1, v1 = pca_probe(X, Y, fold)
            if sk:
                r_pca_ns, pc1_ns, _ = pca_probe(X[:, keep], Y, fold)
            else:
                r_pca_ns, pc1_ns = r_pca, pc1
            out = dict(layer=l, sink_coords=sk, sink_energy_share_token=float(S[key][l, sk].sum()) if sk else 0.0,
                       sink_energy_share_pooled=pool_share, pc1_var_share=pc1, pc1_var_share_no_sink=pc1_ns,
                       pc1_loading_on_sinks=float((v1[sk] ** 2).sum()) if sk else 0.0,
                       r2_std=r_std.tolist(), r2_std_no_sink=r_std_ns.tolist(),
                       r2_pca64=r_pca.tolist(), r2_pca64_no_sink=r_pca_ns.tolist())
            res["probes"].setdefault(key, []).append(out)
            print(key, l, sk, "pc1 %.3f -> %.3f" % (pc1, pc1_ns),
                  "mean R2 std %.3f->%.3f pca %.3f->%.3f" % (np.clip(r_std, 0, 1).mean(), np.clip(r_std_ns, 0, 1).mean(),
                                                            np.clip(r_pca, 0, 1).mean(), np.clip(r_pca_ns, 0, 1).mean()), flush=True)
    res["targets"] = names
    with open(os.path.join(VDIR, "sink.json"), "w") as f:
        json.dump(res, f, indent=1)
    figure(res, fams)


def figure(res=None, fams=None):
    setup_style()
    import matplotlib.pyplot as plt
    if res is None:
        with open(os.path.join(VDIR, "sink.json")) as f:
            res = json.load(f)
        from analysis.common import targets, load_states
        fams = [v.family for v in targets(load_states())[1]]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    for ci, (key, title) in enumerate([("vis", "vision blocks"), ("lm_vis", "LM layers (visual tokens)")]):
        tl = res["token_level"][key]
        ax = axes[0, ci]
        ax.bar(range(len(tl)), [t["top_share"] for t in tl], color=SERIES[0], width=0.8, label="top coordinate")
        if key == "lm_vis":
            ta = res["token_level"]["lm_all"]
            ax.plot(range(len(ta)), [t["top_share"] for t in ta], color=SERIES[1], lw=1.2, marker="o", ms=2,
                    label="top coordinate, all tokens")
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
        ax.axhline(0.1, color=MUTED, lw=0.8, ls=":")
        ax.set_ylim(0, 1)
        ax.set_xlabel("layer")
        ax.set_ylabel("share of residual energy (Σx²)")
        ax.set_title(f"{'AB'[ci]}  Largest coordinate, {title}", fontsize=8.5)
        for t in tl:
            if t["top_share"] > 0.1:
                ax.text(t["layer"], t["top_share"] + 0.02, str(t["top_coord"]), rotation=90, ha="center", fontsize=5,
                        color=INK2)
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.8) if key == "lm_vis" else None)
        ax = axes[1, ci]
        pr = res["probes"][key]
        L = [p["layer"] for p in pr]
        fm = lambda k: [np.clip(np.array(p[k]), 0, 1).mean() for p in pr]
        ax.plot(L, fm("r2_std"), color=INK, lw=1.4, label="standardised ridge, all coords")
        ax.plot(L, fm("r2_std_no_sink"), color=INK, lw=1.0, ls="--", label="standardised ridge, sinks removed")
        ax.plot(L, fm("r2_pca64"), color=SERIES[1], lw=1.4, label="PCA-64 of raw residual")
        ax.plot(L, fm("r2_pca64_no_sink"), color=SERIES[1], lw=1.0, ls="--", label="PCA-64, sinks removed")
        ax.set_ylim(0, 0.8)
        ax.set_xlabel("layer")
        ax.set_ylabel("mean probe R² (32 targets)")
        ax.set_title(f"{'CD'[ci]}  Probes with and without sink coordinates", fontsize=8.5)
        ax.legend(fontsize=5.5, loc="lower center")
    fig.tight_layout()
    savefig(fig, "vlm_sink.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fig":
        figure()
    else:
        main()
