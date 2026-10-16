"""Analysis 1 -- how fast is gravity inferred? Linear probes for (cos θg, sin θg) from the
residual stream at each layer and context position vs. an ideal inverse-dynamics observer.
Also: next-step prediction error vs context length for the three model variants.

    uv run python analysis/grav_probe.py
Figures: grav_context_loss.png, grav_probe_time.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import INK2, MUTED, SERIES, savefig, setup_style  # noqa: E402
from analysis.grav_common import (N_PROBE_TRAIN, Dyn, analysis_windows, angerr_deg, frame_angle,  # noqa: E402
                                  ideal_observer, ridge_apply, ridge_fit, unitvec, update_json, vec_angle)
from models.grav_model import CKPT, CTX, features, load, run  # noqa: E402

VARIANTS = {"incontext": "in-context (θg inferred)", "given": "θg given as input", "h1": "history length 1"}
FR = {"world": "θg, world frame", "upper": "θg − q₁, upper-arm frame"}
VCOL = {"incontext": SERIES[0], "given": SERIES[2], "h1": SERIES[1]}


def context_loss_fig(res):
    """Per-position next-frame error on the data/grav test windows (episodes 1000-1199, starts 0/43/85).
    Shows the mean and the median over windows: the mean is dominated by rare fast elbow spins."""
    from analysis.common import N_TRAIN_EPS_GRAV, load_states
    from models.grav_train import make_xy, test_windows
    st = load_states("grav")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    x = np.arange(1, CTX + 1)
    for v, lab in VARIANTS.items():
        X, Y = make_xy(st, v == "given")
        Xte, Yte = test_windows(X[N_TRAIN_EPS_GRAV:], Y[N_TRAIN_EPS_GRAV:])
        m = load(v)
        pred, _ = run(m, Xte)
        se = (((pred - Yte) / m.y_std.cpu().numpy()) ** 2).mean(-1)        # [N, CTX]
        axes[0].plot(x, se.mean(0), color=VCOL[v], lw=1.4, label=lab)
        axes[1].plot(x, np.median(se, 0), color=VCOL[v], lw=1.8, label=lab)
        res[v] = dict(mse_all=float(se.mean()), mse_pos1=float(se[:, 0].mean()), mse_pos2=float(se[:, 1].mean()),
                      mse_pos3=float(se[:, 2].mean()), mse_late=float(se[:, 32:].mean()),
                      median_se_pos1=float(np.median(se[:, 0])), median_se_pos2=float(np.median(se[:, 1])),
                      median_se_pos5=float(np.median(se[:, 4])), median_se_late=float(np.median(se[:, 32:])))
    for ax in axes:
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_xlabel("frames observed (context length)")
    axes[0].set_ylabel("squared error (z-scored targets)")
    axes[0].set_title("Mean over windows (tail-dominated)", fontsize=9)
    axes[1].set_title("Median over windows", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    savefig(fig, "grav_context_loss.png")


def main():
    setup_style()
    res = {}
    context_loss_fig(res)
    w = analysis_windows()
    tr = w["ep"] < N_PROBE_TRAIN
    te = ~tr
    model = load("incontext")
    nL = model.cfg["n_layers"] + 1
    _, acts = run(model, w["X"], keys=[f"resid{L}" for L in range(nL)])
    have_h1 = os.path.exists(os.path.join(CKPT, "grav_h1.pt"))
    if have_h1:
        _, a1 = run(load("h1"), w["X"], keys=[f"resid{L}" for L in range(nL)])
    err, err_h1 = {}, {}
    for fr in FR:
        ang = frame_angle(w["theta"], w["q"], fr)                      # [N, CTX]
        Y = unitvec(ang)
        err[fr] = np.zeros((nL, CTX, te.sum()))
        err_h1[fr] = np.full((nL, CTX, te.sum()), np.nan)
        for L in range(nL):
            for t in range(CTX):
                H = acts[f"resid{L}"][:, t]
                p = ridge_fit(H[tr], Y[tr, t])
                err[fr][L, t] = angerr_deg(vec_angle(ridge_apply(p, H[te])), ang[te, t])
                if have_h1 and t % 3 == 0:
                    H = a1[f"resid{L}"][:, t]
                    p = ridge_fit(H[tr], Y[tr, t])
                    err_h1[fr][L, t] = angerr_deg(vec_angle(ridge_apply(p, H[te])), ang[te, t])
    # nonlinear (MLP) probes, pooled over positions >= 32, to check that a weak *linear* readout
    # is not hiding a nonlinear world-frame code
    from sklearn.neural_network import MLPRegressor
    nonlin = {}
    for fr in FR:
        ang = frame_angle(w["theta"], w["q"], fr)
        for L in range(1, nL):
            H = acts[f"resid{L}"][:, 32::4]
            Yp = unitvec(ang[:, 32::4])
            mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=40, early_stopping=True, random_state=0)
            mlp.fit(H[tr].reshape(-1, H.shape[-1]), Yp[tr].reshape(-1, 2))
            pr = mlp.predict(H[te].reshape(-1, H.shape[-1]))
            nonlin[f"{fr}_L{L}"] = float(np.median(angerr_deg(vec_angle(pr), ang[te, 32::4].ravel())))
    print("nonlinear probes", nonlin, flush=True)
    # ideal observer (finite-difference qdd, causal); error is the same in any frame since q1 is observed
    u = ideal_observer(w["q"][te], w["qd"][te], w["tau_act"][te], dyn=Dyn())
    err_obs = angerr_deg(vec_angle(u), w["theta"][te, None]).T   # [CTX, Nte]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    x = np.arange(1, CTX + 1)
    lcols = ["#c3c2b7", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
    eo = np.nanmedian(err_obs, 1)
    for ax, fr in zip(axes[:2], FR):
        for L in range(nL):
            ax.plot(x, np.median(err[fr][L], 1), color=lcols[L], lw=1.5,
                    label="embedding" if L == 0 else f"after block {L}")
        hb = np.nanmin(np.nanmedian(err_h1[fr], 2), 0)
        ok = np.isfinite(hb)
        if ok.any():
            ax.plot(x[ok], hb[ok], color=SERIES[1], lw=1.3, ls="--", label="history-1 model")
        ok = np.isfinite(eo)
        ax.plot(x[ok], eo[ok], color=INK2, lw=1.4, ls=":", label="ideal observer")
        ax.axhline(90, color=MUTED, lw=0.8, zorder=0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.1, 130)
        ax.set_xlabel("frames observed")
        ax.set_title(FR[fr], fontsize=9)
    axes[0].set_ylabel("median |decoding error| (°)")
    axes[0].text(1.1, 100, "chance = 90°", color=MUTED, fontsize=6.5)
    axes[1].legend(fontsize=6, loc="lower left")
    fbest = min(FR, key=lambda f: np.median(err[f][:, 32:], 2).mean(1).min())
    bestL = int(np.argmin(np.median(err[fbest][:, 32:], 2).mean(1)))
    bestLw = int(np.argmin(np.median(err["world"][:, 32:], 2).mean(1)))
    ax = axes[2]
    modes = w["mode"][te]
    for i, md in enumerate(["passive", "sines", "reach"]):
        sel = modes == md
        ax.plot(x, np.median(err[fbest][bestL][:, sel], 1), color=SERIES[i + 3], label=md)
        ax.plot(x[2:], np.nanmedian(err_obs[2:, sel], 1), color=SERIES[i + 3], lw=1, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.1, 130)
    ax.set_xlabel("frames observed")
    ax.set_title(f"By controller ({'upper-arm' if fbest == 'upper' else fbest} frame, block {bestL})", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    savefig(fig, "grav_probe_time.png")

    def summ(e):
        return dict(median=float(np.median(e)), mean=float(np.mean(e)), p90=float(np.percentile(e, 90)))
    for fr in FR:
        res[f"probe_{fr}"] = {f"L{L}": {f"t{t + 1}": summ(err[fr][L, t]) for t in (0, 1, 2, 4, 9, 19, 63)}
                              for L in range(nL)}
        res[f"probe_{fr}_late_median_by_layer"] = [float(np.median(err[fr][L, 32:])) for L in range(nL)]
        hb = np.nanmin(np.nanmedian(err_h1[fr], 2), 0)
        res[f"h1_best_{fr}"] = {f"t{t + 1}": float(hb[t]) for t in (0, 30, 63) if np.isfinite(hb[t])}
    res["nonlinear_probe_late_median"] = nonlin
    res["best_frame"] = fbest
    res["probe_best_layer"] = bestL
    res["probe_best_layer_world"] = bestLw
    res["observer"] = {f"t{t + 1}": summ(err_obs[t]) for t in (2, 4, 9, 19, 63)}
    res["by_mode_late"] = {md: dict(model=float(np.median(err[fbest][bestL][32:, w["mode"][te] == md])),
                                    observer=float(np.nanmedian(err_obs[32:, w["mode"][te] == md])))
                           for md in ["passive", "sines", "reach"]}
    # frames to reach 10 deg / 5 deg (median), best layer per frame
    for fr in FR:
        med = np.median(err[fr], 2).min(0)
        res[f"frames_to_10deg_{fr}"] = int(np.argmax(med < 10) + 1) if (med < 10).any() else None
        res[f"frames_to_5deg_{fr}"] = int(np.argmax(med < 5) + 1) if (med < 5).any() else None
    update_json("probe", res)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("probe_")}, indent=1, default=float))
    print({k: v for k, v in res.items() if k.endswith("by_layer")})


if __name__ == "__main__":
    if sys.argv[1:] == ["context"]:      # just the prediction-error figure (cheap)
        setup_style()
        r = {}
        context_loss_fig(r)
        d = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "reports", "results", "grav.json")))
        d["probe"].update(r)
        update_json("probe", d["probe"])
        print(json.dumps(r, indent=1))
    else:
        main()
