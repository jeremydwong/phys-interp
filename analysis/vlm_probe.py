"""Linear probes (ridge, 5-fold CV grouped by episode) for all 32 targets at every layer.

Representations (one row per temporal group, aligned to its midpoint sim frame):
  video vision residual   (24 layers, mean over the 400 patches of the group)
  video LM residual       (37 hidden states, mean over the 100 visual tokens of the group)
  image vision / LM residual (same, single midpoint frame given as an image)
  random-init vision encoder residual (24 layers, same clips)
  raw pixels: both frames of the group, 10x10 block-averaged to 32x32x3, PCA to 512
Also: clip-level LM residual at the last prompt token vs targets at the last group's midpoint.

"Beyond posture+velocity" (partial) probes: each dynamics target y is residualised against a
gradient-boosted regressor f(sin/cos q, qdot) fitted on ground truth of the 2000 training
episodes; we then probe y - f(state). Out-of-fold predictions are saved for later analyses.

Usage: uv run python analysis/vlm_probe.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, load_states, savefig, setup_style, targets, INK2, MUTED, GRID  # noqa: E402
from analysis.vlm_common import VDIR, group_index, clip_index, renderer, GROUPS, n_used  # noqa: E402

ALPHAS = np.logspace(-1, 5, 7)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DYN = ["qdd1", "qdd2", "int1", "int2", "tau1", "tau2", "contact", "c_floor", "c_wall", "c_fore", "ctq1", "ctq2",
       "grav1", "grav2", "cor1", "cor2", "ke", "pe"]


def aligned_targets():
    st = load_states()
    Y, vs = targets(st)
    gi = group_index()
    return Y[gi[:, 0], gi[:, 3]], vs, gi, st


def state_feats(st, e, f):
    q, qd = st["q"][e, f], st["qd"][e, f]
    return np.concatenate([np.sin(q), np.cos(q), np.sin(q.sum(-1, keepdims=True)), np.cos(q.sum(-1, keepdims=True)), qd], -1)


def residualise(Ytest, vs, st, gi):
    """y - E[y | posture, velocity] with E fitted by HistGradientBoosting on training episodes."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    path = os.path.join(VDIR, "resid_targets.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["R"], z["r2_state"]
    Yall, _ = targets(st)
    rng = np.random.default_rng(0)
    e_tr = rng.integers(0, 2000, 60000)
    f_tr = rng.integers(0, 150, 60000)
    Xtr = state_feats(st, e_tr, f_tr)
    Xte = state_feats(st, gi[:, 0], gi[:, 3])
    R = np.zeros_like(Ytest)
    r2 = np.zeros(len(vs))
    for j, v in enumerate(vs):
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.1, max_leaf_nodes=63)
        m.fit(Xtr, Yall[e_tr, f_tr, j])
        p = m.predict(Xte)
        R[:, j] = Ytest[:, j] - p
        r2[j] = 1 - R[:, j].var() / Ytest[:, j].var()
        print("state->", v.name, round(r2[j], 3), flush=True)
    np.savez(path, R=R, r2_state=r2)
    return R, r2


def folds(groups, k=5):
    ue = np.unique(groups)
    rng = np.random.default_rng(0)
    perm = rng.permutation(ue)
    fid = {e: i % k for i, e in enumerate(perm)}
    return np.array([fid[g] for g in groups])


def ridge_cv(X, Y, fold, alphas=ALPHAS, return_pred=False, device="cpu", dtype=torch.float64, std_floor=False):
    """Returns R2 [n_alpha, V] (pooled out-of-fold) and optionally OOF predictions at best alpha per target.

    device/dtype: the wide max-pooled unit probes run on the GPU in float32 (float64 copies of a
    10800 x 19456 matrix on the CPU exhausted the msi's RAM). std_floor: floor each feature's
    train-fold std at 10% of the median std, so a unit that is almost never active in the training
    fold cannot blow up when standardised on the test fold."""
    X = torch.as_tensor(np.asarray(X, np.float32), dtype=dtype, device=device)
    Y = torch.as_tensor(np.asarray(Y, np.float32), dtype=dtype, device=device)
    preds = torch.zeros((len(alphas),) + Y.shape, dtype=dtype, device=device)
    for k in np.unique(fold):
        tr, te = torch.as_tensor(fold != k, device=device), torch.as_tensor(fold == k, device=device)
        mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-6
        if std_floor:
            sx = torch.clamp(sx, min=0.1 * sx.median())
        my = Y[tr].mean(0)
        Xtr, Xte = (X[tr] - mx) / sx, (X[te] - mx) / sx
        Ytr = Y[tr] - my
        n, d = Xtr.shape
        if d <= n:
            evals, evecs = torch.linalg.eigh(Xtr.T @ Xtr)
            XtY = evecs.T @ (Xtr.T @ Ytr)
            for ai, a in enumerate(alphas):
                W = evecs @ (XtY / (evals[:, None]+ a))
                preds[ai][te] = Xte @ W + my
        else:
            evals, evecs = torch.linalg.eigh(Xtr @ Xtr.T)
            UtY = evecs.T @ Ytr
            K = Xte @ Xtr.T @ evecs
            for ai, a in enumerate(alphas):
                preds[ai][te] = K @ (UtY / (evals[:, None]+ a)) + my
    ss = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = 1 - ((preds - Y[None]) ** 2).sum(1) / ss
    r2 = r2.cpu().numpy()
    if return_pred:
        best = r2.argmax(0)
        P = np.stack([preds[best[j], :, j].cpu().numpy() for j in range(Y.shape[1])], 1)
        return r2, P
    return r2


def pixel_feats(gi, st):
    path = os.path.join(VDIR, "pixels32.npy")
    if os.path.exists(path):
        return np.load(path)
    r = renderer()
    out = np.zeros((len(gi), 2, 32, 32, 3), np.float16)
    for i, (e, _, _, _, fa, fb) in enumerate(gi):
        fr = r(st["q"][e][[fa, fb]]).astype(np.float32) / 255
        out[i] = fr.reshape(2, 32, 10, 32, 10, 3).mean((2, 4))
    out = out.reshape(len(gi), -1)
    np.save(path, out)
    return out


def pca(X, k=512):
    X = torch.as_tensor(np.asarray(X, np.float32))
    X = X - X.mean(0)
    U, S, V = torch.pca_lowrank(X, q=k, center=False, niter=4)
    return (X @ V).numpy()


def main():
    Y, vs, gi, st = aligned_targets()
    names = [v.name for v in vs]
    fold = folds(gi[:, 0])
    R, r2_state = residualise(Y, vs, st, gi)
    NG, NC = n_used("video")
    Y, gi, R, fold = Y[:NG], gi[:NG], R[:NG], fold[:NG]
    print("using", NG, "groups /", NC, "clips", flush=True)
    dyn_idx = [names.index(n) for n in DYN]
    res = {"targets": names, "families": [v.family for v in vs], "r2_state_gbm": r2_state.tolist(), "dyn_targets": DYN}

    def run(tag, X):
        r2 = ridge_cv(X, Y, fold)
        r2p = ridge_cv(X, R[:, dyn_idx], fold)
        return r2.max(0), r2p.max(0)

    out = {}
    # pixels
    px = pca(pixel_feats(group_index(), st)[:NG].astype(np.float32), 512)
    out["pixels"] = [run("pixels", px)]
    print("pixels", np.round(out["pixels"][0][0], 2), flush=True)
    sources = [("video_vis", "video_vis_res.npy"), ("video_llm", "video_llm_res.npy"),
               ("image_vis", "image_vis_res.npy"), ("image_llm", "image_llm_res.npy"),
               ("randvis_vis", "randvis_vis_res.npy")]
    for tag, fn in sources:
        p = os.path.join(VDIR, fn)
        if not os.path.exists(p):
            print("missing", fn)
            continue
        A = np.load(p, mmap_mode="r")
        mode = tag.split("_")[0]
        if n_used(mode)[0] < NG:
            print("incomplete", fn, n_used(mode)[0])
            continue
        A = A[:NG]
        out[tag] = []
        for l in range(A.shape[1]):
            out[tag].append(run(tag, np.asarray(A[:, l], np.float32)))
            print(tag, l, "R2 sin_q1 %.2f qd1 %.2f qdd1 %.2f contact %.2f" % tuple(out[tag][-1][0][[0, 9, 14, 24]]), flush=True)
    for k, v in out.items():
        res[k] = dict(r2=[x[0].tolist() for x in v], r2_partial=[x[1].tolist() for x in v])

    # OOF predictions from the best video layer per target for event-aligned analyses
    A = np.load(os.path.join(VDIR, "video_llm_res.npy"), mmap_mode="r")[:NG]
    best_l = int(np.argmax([np.mean(x[0][:24]) for x in out["video_llm"]]))
    _, P = ridge_cv(np.asarray(A[:, best_l], np.float32), Y, fold, return_pred=True)
    Av = np.load(os.path.join(VDIR, "video_vis_res.npy"), mmap_mode="r")[:NG]
    _, Pv = ridge_cv(np.asarray(Av[:, 23], np.float32), Y, fold, return_pred=True)
    np.savez(os.path.join(VDIR, "probe_oof.npz"), P_llm=P, P_vis=Pv, layer_llm=best_l)
    res["oof_llm_layer"] = best_l

    # last-token (clip level) residual vs. targets at last group midpoint / last shown frame
    ci = clip_index()[:NC]
    L = np.load(os.path.join(VDIR, "video_llm_last.npy"), mmap_mode="r")[:NC]
    Yall, _ = targets(st)
    Ylast = Yall[ci[:, 0], ci[:, 2] + 13]
    fold_c = folds(ci[:, 0])
    res["last_token"] = [ridge_cv(np.asarray(L[:, l], np.float32), Ylast, fold_c).max(0).tolist() for l in range(37)]
    with open(os.path.join(VDIR, "probe_results.json"), "w") as f:
        json.dump(res, f)
    print("saved")


def mlp_probes():
    """Probes on the max-pooled MLP hidden units of the census layers (keeps the spatial
    'where is the strongest response' information that mean-pooling the residual discards)."""
    from analysis.vlm_common import VIS_LAYERS, LLM_LAYERS
    Y, vs, gi, st = aligned_targets()
    NG, _ = n_used("video")
    Y, gi = Y[:NG], gi[:NG]
    fold = folds(gi[:, 0])
    path = os.path.join(VDIR, "probe_results.json")
    with open(path) as f:
        res = json.load(f)
    R, _ = residualise(*aligned_targets()[:2], st, group_index())
    dyn_idx = [res["targets"].index(n) for n in DYN]
    R = R[:NG]
    res["mlp_max"] = {}
    for part, layers, tmpl in [("V", VIS_LAYERS, "video_vis_mlp_L{}.npy"), ("L", LLM_LAYERS, "video_llm_mlp_L{}.npy")]:
        for l in layers:
            Z = np.load(os.path.join(VDIR, tmpl.format(l)), mmap_mode="r")[:NG]
            X = np.concatenate([np.asarray(Z[:, 0], np.float32), np.asarray(Z[:, 1], np.float32)], 1) if part == "L" \
                else np.asarray(Z[:, 0], np.float32)
            # drop near-constant and very sparse units (active away from their median in <1% of groups)
            keep = (X.std(0) > 1e-4) & (np.mean(np.abs(X - np.median(X, 0)) > 1e-3, 0) > 0.01)
            X = X[:, keep]
            kw = dict(alphas=np.logspace(1, 5, 5), device=DEVICE, dtype=torch.float32, std_floor=True)
            r2 = ridge_cv(X, Y, fold, **kw).max(0)
            r2p = ridge_cv(X, R[:, dyn_idx], fold, **kw).max(0)
            del X, Z
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            res["mlp_max"][f"{part}{l}"] = dict(r2=r2.tolist(), r2_partial=r2p.tolist())
            print("mlp", part, l, "sin_q1 %.2f qd1 %.2f qd2 %.2f ee_vx %.2f qdd1 %.2f contact %.2f ke %.2f" %
                  tuple(r2[[0, 9, 10, 11, 14, 24, 30]]), flush=True)
    with open(path, "w") as f:
        json.dump(res, f)


# ------------------------------------------------------------------ figures

def figures():
    div, seq = setup_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    with open(os.path.join(VDIR, "probe_results.json")) as f:
        res = json.load(f)
    _, vs, _, _ = aligned_targets() if False else (None, targets(load_states())[1], None, None)
    names, fams = res["targets"], res["families"]
    labels = [v.label for v in vs]
    V = len(names)

    # --- Fig: heatmap layer x variable (video), with family colour strip
    M = np.array([r for r in res["video_vis"]["r2"]] + [r for r in res["video_llm"]["r2"]])
    rows = [f"V{l}" for l in range(24)] + [f"L{l}" for l in range(37)]
    fig = plt.figure(figsize=(7.0, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.035, 1], width_ratios=[1, 0.025], hspace=0.02, wspace=0.03)
    ax = fig.add_subplot(gs[1, 0])
    cmap = LinearSegmentedColormap.from_list("r2", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    im = ax.imshow(np.clip(M, 0, 1), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(np.arange(0, 61, 4), [rows[i] for i in range(0, 61, 4)], fontsize=6.5)
    ax.set_xticks(range(V), labels, rotation=90, fontsize=6.5)
    ax.axhline(23.5, color=INK2, lw=1.0)
    ax.text(V - 0.4, 23.0, "vision encoder ↑   language model ↓", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.grid(False)
    ax.set_ylabel("layer (V = vision block output, L = LM hidden state at visual tokens)")
    axf = fig.add_subplot(gs[0, 0], sharex=ax)
    for j, f in enumerate(fams):
        axf.add_patch(plt.Rectangle((j - 0.5, 0), 1, 1, color=FAMILIES[f][0]))
    axf.set_xlim(-0.5, V - 0.5); axf.set_ylim(0, 1); axf.axis("off")
    axf.set_title("Linear decodability (5-fold ridge R²) of each physical variable from Qwen3-VL-4B, video input", fontsize=9)
    cax = fig.add_subplot(gs[1, 1])
    fig.colorbar(im, cax=cax, label="cross-validated R²")
    savefig(fig, "vlm_probe_heatmap.png")

    # --- Fig: family summary across depth, video vs image vs random vs pixels
    fam_order = ["posture", "velocity", "accel", "gravity", "interaction", "contact", "applied", "energy"]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9), sharey=True, sharex=True)
    xs_v, xs_l = np.arange(24), 24 + np.arange(37)
    for ax, fam in zip(axes.ravel(), fam_order):
        idx = [i for i, f in enumerate(fams) if f == fam]
        def fm(key):
            return np.clip(np.array(res[key]["r2"])[:, idx], 0, 1).mean(1)
        c = FAMILIES[fam][0]
        ax.plot(xs_v, fm("video_vis"), color=c, lw=1.6, label="video")
        ax.plot(xs_l, fm("video_llm"), color=c, lw=1.6)
        if "image_vis" in res:
            ax.plot(xs_v, fm("image_vis"), color=c, lw=1.1, ls="--", label="single image")
            ax.plot(xs_l, fm("image_llm"), color=c, lw=1.1, ls="--")
        if "randvis_vis" in res:
            ax.plot(xs_v, fm("randvis_vis"), color=MUTED, lw=1.0, ls=":", label="random-init encoder")
        ax.axhline(fm("pixels")[0], color=INK2, lw=0.8, ls="-.", label="raw pixels (PCA-512)")
        ax.axvline(23.5, color=GRID, lw=1.5)
        ax.set_title(fam, fontsize=8.5, color=c)
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xticks([0, 12, 24, 36, 48, 60], ["V0", "V12", "L0", "L12", "L24", "L36"])
        ax.set_xlabel("depth")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean probe R²")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=7.5)
    fig.suptitle("Probe R² by variable family through the vision encoder (V) and language model (L)", x=0.01, ha="left",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, "vlm_probe_families.png")

    # --- Fig: dynamics beyond posture+velocity
    dyn = res["dyn_targets"]
    di = [names.index(n) for n in dyn]
    Pp = np.array(res["video_llm"]["r2_partial"])
    Pv = np.array(res["video_vis"]["r2_partial"])
    Pi = np.array(res["image_llm"]["r2_partial"]) if "image_llm" in res else None
    best_vid = np.maximum(Pp.max(0), Pv.max(0))
    best_img = np.maximum(Pi.max(0), np.array(res["image_vis"]["r2_partial"]).max(0)) if Pi is not None else None
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    x = np.arange(len(dyn))
    ax.bar(x - 0.27, [res["r2_state_gbm"][i] for i in di], 0.27, color=MUTED, label="explained by true posture+velocity (GBM)")
    ax.bar(x, np.clip(best_vid, 0, 1), 0.27, color=[FAMILIES[fams[i]][0] for i in di], label="probe on residual, video (best layer)")
    if best_img is not None:
        ax.bar(x + 0.27, np.clip(best_img, 0, 1), 0.27, color=[FAMILIES[fams[i]][0] for i in di], alpha=0.4,
               label="probe on residual, single image (best layer)")
    ax.set_xticks(x, [labels[i] for i in di], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1)
    ax.set_title("Dynamics beyond kinematics: probing y − E[y | posture, velocity]")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="x", visible=False)
    savefig(fig, "vlm_probe_partial.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fig":
        figures()
    elif len(sys.argv) > 1 and sys.argv[1] == "mlp":
        mlp_probes()
        figures()
    else:
        main()
        mlp_probes()
        figures()
