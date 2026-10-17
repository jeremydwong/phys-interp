"""Analysis 2 + 6: ridge probes on the residual stream after every block (avg-pooled to 4x4, 2048-d),
trained vs random-init vs raw-pixel baselines. Fit on train episodes 0-399, test on 2000-2399.

Also: probes on the max-pooled hidden units of each layer, alignment (targets at t-1/t/t+1)
for acceleration, and a nonlinear (MLP) probe on the best layer as a check on linear decodability.
Figures: video_probes.png (heatmap), video_accel_gravity.png (layer curves)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, INK2, MUTED, SERIES, savefig, setup_style  # noqa: E402
from analysis.video_common import (CACHE, DEV, LAYERS, RESULTS, cached_acts, load,  # noqa: E402
                                   target_matrix, windows)

ALPHAS = np.logspace(-1, 5, 13)


def ridge_fit_eval(Xtr, Ytr, Xte, Yte, alphas=ALPHAS):
    """Ridge with per-target alpha chosen on a 20% validation split of the train set; R2 on test."""
    Xtr = torch.as_tensor(Xtr, dtype=torch.float64)
    Xte = torch.as_tensor(Xte, dtype=torch.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-8
    Ytr_ = torch.as_tensor((Ytr - ym) / ys, dtype=torch.float64)
    n = len(Xtr)
    nv = n // 5
    # contiguous split by sample order (episodes are contiguous) to avoid temporal leakage
    Xa, Xv, Ya, Yv = Xtr[:-nv], Xtr[-nv:], Ytr_[:-nv], Ytr_[-nv:]

    def solve(X, Yy, lam):
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        return U, S, Vt, Yy

    U, S, Vh = torch.linalg.svd(Xa, full_matrices=False)
    UtY = U.T @ Ya
    best_err, best_a = None, None
    for a in alphas:
        W = Vh.T @ ((S / (S ** 2 + a))[:, None] * UtY)
        err = ((Xv @ W - Yv) ** 2).mean(0)
        if best_err is None:
            best_err, best_a = err.clone(), torch.full_like(err, a)
        else:
            m = err < best_err
            best_err[m], best_a[m] = err[m], a
    U, S, Vh = torch.linalg.svd(Xtr, full_matrices=False)
    UtY = U.T @ Ytr_
    pred = torch.zeros(len(Xte), Ytr.shape[1], dtype=torch.float64)
    for a in torch.unique(best_a):
        cols = best_a == a
        W = Vh.T @ ((S / (S ** 2 + a))[:, None] * UtY[:, cols])
        pred[:, cols] = Xte @ W
    pred = pred.numpy() * ys + ym
    r2 = 1 - ((pred - Yte) ** 2).mean(0) / Yte.var(0)
    return r2, pred


def pixel_features(E, T, last_only=False):
    out = []
    for i in range(0, len(E), 512):
        x = windows(E[i:i + 512], T[i:i + 512])
        if last_only:
            x = x[:, -1:]
        B = x.shape[0]
        out.append(F.avg_pool2d(x.reshape(B, -1, 64, 64), 4).reshape(B, -1).cpu().numpy())
    return np.concatenate(out)


def mlp_probe(Xtr, Ytr, Xte, Yte, epochs=40):
    torch.manual_seed(0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-8
    Xt = torch.as_tensor((Xtr - mu) / sd, device=DEV, dtype=torch.float32)
    Yt = torch.as_tensor((Ytr - ym) / ys, device=DEV, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(Xt.shape[1], 512), torch.nn.GELU(),
                              torch.nn.Linear(512, Yt.shape[1])).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    steps = epochs * len(Xt) // 256
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(0, len(Xt), (256,), device=DEV)
        loss = ((net(Xt[i]) - Yt[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        p = net(torch.as_tensor((Xte - mu) / sd, device=DEV, dtype=torch.float32)).cpu().numpy() * ys + ym
    return 1 - ((p - Yte) ** 2).mean(0) / Yte.var(0)


def main():
    Y, vs, st = target_matrix()
    names = [v.name for v in vs]
    res = {}
    acts = {}
    for tag in ["final", "init"]:
        te = cached_acts(tag, "test", lambda: load(tag)[0], res=True)
        tr = cached_acts(tag, "probe", lambda: load(tag)[0], res=True)
        acts[tag] = (tr, te)
    tr, te = acts["final"]
    Etr, Ttr, Ete, Tte = tr["eps"], tr["ts"], te["eps"], te["ts"]
    Ytr, Yte = Y[Etr, Ttr], Y[Ete, Tte]
    R = {}
    for tag in ["final", "init"]:
        tr, te = acts[tag]
        R[tag] = np.stack([ridge_fit_eval(tr["res"][l].astype(np.float32), Ytr, te["res"][l].astype(np.float32), Yte)[0]
                           for l in range(len(LAYERS))])
        H = tr["max"].shape[1] // len(LAYERS)
        R[tag + "_units"] = np.stack([ridge_fit_eval(tr["max"][:, l * H:(l + 1) * H].astype(np.float32), Ytr,
                                                     te["max"][:, l * H:(l + 1) * H].astype(np.float32), Yte)[0]
                                      for l in range(len(LAYERS))])
        print(tag, "done", flush=True)
    pc = os.path.join(CACHE, "pixels.npz")
    if not os.path.exists(pc):
        np.savez(pc, tr=pixel_features(Etr, Ttr), te=pixel_features(Ete, Tte))
    z = np.load(pc)
    R["pixels4"] = ridge_fit_eval(z["tr"], Ytr, z["te"], Yte)[0]
    R["pixels1"] = ridge_fit_eval(z["tr"][:, -z["tr"].shape[1] // 4:], Ytr, z["te"][:, -z["te"].shape[1] // 4:], Yte)[0]
    # alignment for the final layer: targets at t-1, t, t+1
    tr, te = acts["final"]
    align = {}
    for d in [-1, 0, 1]:
        align[d] = np.stack([ridge_fit_eval(tr["res"][l].astype(np.float32), Y[Etr, Ttr + d],
                                            te["res"][l].astype(np.float32), Y[Ete, Tte + d])[0]
                             for l in range(len(LAYERS))])
    # nonlinear probe check at enc3 and tmp3
    mlp = {l: mlp_probe(tr["res"][LAYERS.index(l)].astype(np.float32), Ytr, te["res"][LAYERS.index(l)].astype(np.float32), Yte)
           for l in ["enc3", "tmp3"]}
    mlp["pixels4"] = mlp_probe(z["tr"], Ytr, z["te"], Yte)
    np.savez(os.path.join(CACHE, "probes.npz"), **{k: v for k, v in R.items()},
             **{f"align{d}": v for d, v in align.items()}, **{f"mlp_{k}": v for k, v in mlp.items()})

    out = {"variables": names}
    for k, v in R.items():
        out[k] = {LAYERS[l] if v.ndim == 2 else "": None for l in range(1)}
    out = {"variables": names,
           "ridge_res_final": {LAYERS[l]: dict(zip(names, map(float, R["final"][l]))) for l in range(len(LAYERS))},
           "ridge_res_init": {LAYERS[l]: dict(zip(names, map(float, R["init"][l]))) for l in range(len(LAYERS))},
           "ridge_units_final": {LAYERS[l]: dict(zip(names, map(float, R["final_units"][l]))) for l in range(len(LAYERS))},
           "ridge_pixels4": dict(zip(names, map(float, R["pixels4"]))),
           "ridge_pixels1": dict(zip(names, map(float, R["pixels1"]))),
           "align_final": {str(d): {LAYERS[l]: dict(zip(names, map(float, a[l]))) for l in range(len(LAYERS))} for d, a in align.items()},
           "mlp": {k: dict(zip(names, map(float, v))) for k, v in mlp.items()}}
    json.dump(out, open(os.path.join(RESULTS, "video_probes.json"), "w"), indent=1)
    plot(R, align, mlp, vs)


def plot(R, align, mlp, vs):
    import matplotlib.pyplot as plt
    div, seq = setup_style()
    names = [v.label for v in vs]
    rows = [("pixels, last frame", R["pixels1"]), ("pixels, 4 frames", R["pixels4"])]
    rows += [(f"init {l}", R["init"][i]) for i, l in enumerate(LAYERS) if l in ("enc3", "tmp3")]
    rows += [(f"trained {l}", R["final"][i]) for i, l in enumerate(LAYERS)]
    M = np.clip(np.stack([r[1] for r in rows]), 0, 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    im = ax.imshow(M, cmap=seq, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.set_xticks(range(len(vs)), names, rotation=70, ha="right", fontsize=6)
    for i, v in enumerate(vs):
        ax.get_xticklabels()[i].set_color(FAMILIES[v.family][0])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=4.5,
                    color="white" if M[i, j] > 0.55 else INK2)
    ax.axhline(1.5, color="white", lw=2); ax.axhline(3.5, color="white", lw=2); ax.axhline(7.5, color="white", lw=1)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("held-out R² (ridge)")
    ax.set_title("Linear probes on the residual stream (4×4 avg-pooled), test episodes")
    savefig(fig, "video_probes.png")

    # layer curves for physics quantities
    ix = {v.name: i for i, v in enumerate(vs)}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True)
    groups = [("State", ["sin_q1", "sin_q2", "qd1", "qd2"]),
              ("Acceleration & gravity", ["qdd1", "qdd2", "grav1", "grav2"]),
              ("Interaction & contact", ["int1", "int2", "c_floor", "ctq2"])]
    x = np.arange(len(LAYERS))
    for ax, (title, vv) in zip(axes, groups):
        for k, v in enumerate(vv):
            col = SERIES[k]
            ax.plot(x, np.clip(R["final"][:, ix[v]], 0, 1), "-o", ms=3, color=col, label=vs[ix[v]].label)
            ax.plot(x, np.clip(R["init"][:, ix[v]], 0, 1), ":", lw=1, color=col)
            ax.axhline(np.clip(R["pixels4"][ix[v]], 0, 1), color=col, lw=0.8, ls="--", alpha=0.6)
        ax.axvline(3.5, color=INK2, lw=0.8, ls=":")
        ax.set_xticks(x, LAYERS, rotation=60, fontsize=6.5)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("held-out R²")
    fig.text(0.5, -0.1, "solid: trained · dotted: random init · dashed: ridge on raw pixels (4 frames)",
             ha="center", fontsize=7, color=MUTED)
    fig.tight_layout()
    savefig(fig, "video_probe_layers.png")


if __name__ == "__main__":
    main()
