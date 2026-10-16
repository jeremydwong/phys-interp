"""Supplementary -- gravity *magnitude* (out of distribution).

The model was trained only with |g| = 9.81 m/s^2. On 600 held-out episodes with |g| ~ U(4, 16) we ask:
  * does the gravity-direction probe (fit at |g| = 9.81) still decode the direction?
  * is |g| linearly decodable from the residual stream (ridge probe, cross-validated by episode)?
  * how does next-frame error grow with |g - 9.81| (does the model adapt in context)?
Compared with the ideal observer, which recovers |g| exactly as the norm of its LS estimate.

    uv run python analysis/grav_magnitude.py
Figure: grav_magnitude.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import INK2, MUTED, SERIES, savefig, setup_style  # noqa: E402
from analysis.grav_common import (Dyn, angerr_deg, decode_world, ideal_observer, pooled_probe,  # noqa: E402
                                  probe_frame, probe_layer, ridge_apply, ridge_fit, update_json, vec_angle)
from analysis.grav_sim import load_extra  # noqa: E402
from models.grav_model import CTX, features, load, raw_targets, run  # noqa: E402

T_MIN = 32


def main():
    setup_style()
    st = load_extra("vary_g")
    q, qd, tau = st["q"][:, :CTX + 1], st["qd"][:, :CTX + 1], st["tau_act"][:, :CTX + 1]
    g, th = st["g"].astype(float), st["gravity_angle"].astype(float)
    X = features(q[:, :-1], qd[:, :-1], tau[:, :-1])
    Y = raw_targets(q, qd)
    model = load("incontext")
    L = probe_layer()
    pp = pooled_probe(model, L, frame=probe_frame())
    pred, acts = run(model, X, keys=[f"resid{L}"])
    H = acts[f"resid{L}"]
    dec = decode_world(pp, H[:, T_MIN:], q[:, T_MIN:CTX])
    dir_err = np.median(angerr_deg(dec, th[:, None]), 1)
    nmse = ((((pred - Y) / model.y_std.cpu().numpy()) ** 2)[:, T_MIN:]).mean((1, 2))
    # magnitude probe, pooled over late positions, split by episode
    tr = np.arange(len(g)) < 400
    Hp = H[:, T_MIN:]
    pr = ridge_fit(Hp[tr].reshape(-1, Hp.shape[-1]), np.repeat(g[tr], CTX - T_MIN)[:, None])
    gh = ridge_apply(pr, Hp[~tr].reshape(-1, Hp.shape[-1])).reshape((~tr).sum(), -1).mean(1)
    r2 = 1 - ((gh - g[~tr]) ** 2).mean() / g[~tr].var()
    # nonlinear probe for |g| and norm of the direction-probe output
    from sklearn.neural_network import MLPRegressor
    Hs = Hp[:, ::4]
    mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=200, early_stopping=True, random_state=0)
    mlp.fit(Hs[tr].reshape(-1, Hs.shape[-1]), np.repeat(g[tr], Hs.shape[1]))
    gh_nl = mlp.predict(Hs[~tr].reshape(-1, Hs.shape[-1])).reshape((~tr).sum(), -1).mean(1)
    r2_nl = 1 - ((gh_nl - g[~tr]) ** 2).mean() / g[~tr].var()
    pn = np.linalg.norm(ridge_apply(pp["probe"], Hp), axis=-1).mean(1)
    r_norm = float(np.corrcoef(pn, g)[0, 1])
    u = ideal_observer(q[:, :-1], qd[:, :-1], tau[:, :-1], dyn=Dyn())
    g_obs = np.linalg.norm(u[:, -1], axis=-1)
    r2_obs = 1 - ((g_obs[~tr] - g[~tr]) ** 2).mean() / g[~tr].var()
    bins = np.linspace(4, 16, 7)
    idx = np.digitize(g, bins) - 1
    res = dict(n=int(len(g)), probe_layer=L,
               dir_err_median_by_gbin={f"{bins[i]:.0f}-{bins[i + 1]:.0f}": float(np.median(dir_err[idx == i]))
                                       for i in range(6)},
               nmse_by_gbin={f"{bins[i]:.0f}-{bins[i + 1]:.0f}": float(np.median(nmse[idx == i])) for i in range(6)},
               g_probe_r2=float(r2), g_probe_rmse=float(np.sqrt(((gh - g[~tr]) ** 2).mean())),
               g_observer_r2=float(r2_obs), g_mlp_probe_r2=float(r2_nl), corr_probe_norm_vs_g=r_norm)
    update_json("magnitude", res)
    print(res)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4))
    ax = axes[0]
    ax.scatter(g, dir_err, s=4, color=SERIES[0], lw=0, alpha=0.6)
    ax.axvline(9.81, color=MUTED, lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("|g| (m/s²)"); ax.set_ylabel("direction error (°)")
    ax.set_title("Direction probe (fit at 9.81)", fontsize=9)
    ax = axes[1]
    ax.scatter(g[~tr], gh, s=5, color=SERIES[2], lw=0, alpha=0.7, label="linear")
    ax.scatter(g[~tr], gh_nl, s=5, color=SERIES[6], lw=0, alpha=0.7, label="MLP")
    ax.legend(fontsize=6, markerscale=2)
    ax.plot([4, 16], [4, 16], color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("true |g| (m/s²)"); ax.set_ylabel("decoded |g| (m/s²)")
    ax.set_title(f"|g| probes: R² lin {r2:.2f}, MLP {r2_nl:.2f}", fontsize=8.5)
    ax = axes[2]
    ax.scatter(g, nmse, s=4, color=SERIES[1], lw=0, alpha=0.6)
    ax.axvline(9.81, color=MUTED, lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("|g| (m/s²)"); ax.set_ylabel("next-frame MSE (z-scored)")
    ax.set_title("Prediction error", fontsize=9)
    fig.tight_layout()
    savefig(fig, "grav_magnitude.png")


if __name__ == "__main__":
    main()
