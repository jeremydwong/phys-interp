"""Analysis 4 -- rotation equivariance.

Rotating the whole scenario (θg -> θg + α, q1 -> q1 + α at every frame, q2 / velocities / torques
unchanged) is an exact symmetry of the physics (free shoulder hinge, no obstacles; verified
numerically to 5e-7 rad in grav_sim). We feed rotated copies of held-out windows to the model and ask:
  (a) does the decoded gravity angle rotate by α?
  (b) which parts of the representation rotate (world-frame) vs stay fixed (body-frame)?
      Per MLP unit: rotation-variance index = Var_α[a] / Var_total[a] (mean over windows/positions).

    uv run python analysis/grav_equivariance.py
Figure: grav_equivariance.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import INK2, MUTED, SERIES, savefig, setup_style  # noqa: E402
from analysis.grav_common import (N_PROBE_TRAIN, analysis_windows, decode, pooled_probe, probe_layer,  # noqa: E402
                                  update_json, wrap)
from analysis.grav_common import decode_world, probe_frame  # noqa: E402
from models.grav_model import CKPT, features, load, run  # noqa: E402

NA = 24
T_MIN = 16


def main():
    setup_style()
    w = analysis_windows(starts=(0,))
    model = load("incontext")
    nl = model.cfg["n_layers"]
    tab = np.load(os.path.join(CKPT, "grav_unit_table.npz"))
    bestL = probe_layer()
    pps = {fr: pooled_probe(model, bestL, frame=fr) for fr in ("world", "upper")}
    te = np.where(w["ep"] >= N_PROBE_TRAIN)[0][:200]
    alphas = np.linspace(0, 2 * np.pi, NA, endpoint=False)
    dec, mlp = {fr: [] for fr in pps}, []
    dec_raw_upper = []
    resid = {L: [] for L in range(nl + 1)}
    for a in alphas:
        q = w["q"][te].copy(); q[..., 0] += a
        X = features(q, w["qd"][te], w["tau_act"][te])
        _, acts = run(model, X, keys=[f"resid{L}" for L in range(nl + 1)] + [f"mlp{l}" for l in range(nl)])
        for fr, pp in pps.items():
            dec[fr].append(decode_world(pp, acts[f"resid{bestL}"][:, T_MIN:], q[:, T_MIN:]))
        dec_raw_upper.append(decode(pps["upper"], acts[f"resid{bestL}"][:, T_MIN:]))
        mlp.append(np.concatenate([acts[f"mlp{l}"][:, T_MIN::4] for l in range(nl)], -1))
        for L in range(nl + 1):
            resid[L].append(acts[f"resid{L}"][:, T_MIN::4])
    rots, err_rot, abs_err = {}, {}, {}
    for fr in pps:
        dd = np.stack(dec[fr])                             # [NA, N, T] world-frame decoded angle
        rots[fr] = wrap(dd - dd[0])                        # decoded rotation
        err_rot[fr] = np.degrees(np.abs(wrap(rots[fr] - alphas[:, None, None])))
        abs_err[fr] = np.degrees(np.abs(wrap(dd - (w["theta"][te][None, :, None] + alphas[:, None, None]))))
    du = np.stack(dec_raw_upper)
    rots["upper_raw"] = wrap(du - du[0])                   # body-frame readout: should not move
    # rotation-variance index per MLP unit
    M = np.stack(mlp)                                      # [NA, N, T', U]
    var_alpha = M.var(0).mean((0, 1))
    var_tot = M.reshape(-1, M.shape[-1]).var(0) + 1e-12
    rvi = var_alpha / var_tot
    # Fourier content of the residual stream across α (fraction of α-variance in harmonic k)
    harm = {}
    for L in range(nl + 1):
        R = np.stack(resid[L])                             # [NA, N, T', D]
        Fz = np.fft.rfft(R, axis=0)
        p = (np.abs(Fz[1:]) ** 2).sum((1, 2, 3))
        tot_var_alpha = R.var(0).mean((0, 1)).sum()
        all_var = R.reshape(-1, R.shape[-1]).var(0).sum()
        harm[L] = dict(h1=float(p[0] / p.sum()), h2=float(p[1] / p.sum()), h3plus=float(p[2:].sum() / p.sum()),
                       invariant_frac=float(1 - tot_var_alpha / all_var))
    alive = var_tot > 1e-6
    r2w, r2u = tab["r2_world"], tab["r2_upper"]
    tuned = tab["best_r2"] >= 0.3
    res = dict(probe_layer=bestL, n_windows=int(len(te)), n_alpha=NA,
               rot_err_deg_median={fr: float(np.median(e)) for fr, e in err_rot.items()},
               rot_err_deg_p90={fr: float(np.percentile(e, 90)) for fr, e in err_rot.items()},
               abs_err_deg_median={fr: float(np.median(e)) for fr, e in abs_err.items()},
               upper_raw_change_deg_median=float(np.degrees(np.median(np.abs(rots["upper_raw"])))),
               residual_harmonics=harm,
               rvi_median_by_layer=[float(np.median(rvi[(tab["layer"] == l) & alive])) for l in range(nl)],
               rvi_median_world_tuned=float(np.median(rvi[tuned & (tab["best"] == "world")])) if (tuned & (tab["best"] == "world")).any() else None,
               rvi_median_upper_tuned=float(np.median(rvi[tuned & (tab["best"] == "upper")])) if (tuned & (tab["best"] == "upper")).any() else None,
               rvi_median_fore_tuned=float(np.median(rvi[tuned & (tab["best"] == "fore")])) if (tuned & (tab["best"] == "fore")).any() else None)
    update_json("equivariance", res)
    print(res)
    np.savez(os.path.join(CKPT, "grav_rvi.npz"), rvi=rvi)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    ax = axes[0]
    ad = np.degrees(alphas)
    for key, col, lab in (("world", SERIES[0], "world-frame probe"),
                          ("upper", SERIES[2], "upper-arm probe + q₁"),
                          ("upper_raw", SERIES[1], "upper-arm probe, raw (θg−q₁)")):
        dev = wrap(rots[key] - (0 if key == "upper_raw" else alphas[:, None, None])).reshape(NA, -1)
        base = 0 * ad if key == "upper_raw" else ad
        lo, hi = np.degrees(np.percentile(dev, 5, 1)), np.degrees(np.percentile(dev, 95, 1))
        ax.fill_between(ad, base + lo, base + hi, color=col, alpha=0.2, lw=0)
        ax.plot(ad, base + np.degrees(np.median(dev, 1)), color=col, lw=1.6, label=lab)
    ax.plot([0, 360], [0, 360], color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("scenario rotation α (°)"); ax.set_ylabel("decoded rotation of g (°)")
    ax.set_xticks([0, 90, 180, 270, 360]); ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_title("Rotating the whole scene", fontsize=9); ax.legend(fontsize=5.8, loc="upper left")
    ax = axes[1]
    xs = np.arange(nl + 1)
    ax.bar(xs - 0.2, [harm[L]["invariant_frac"] for L in xs], width=0.4, color=SERIES[1], label="rotation-invariant")
    ax.bar(xs + 0.2, [harm[L]["h1"] for L in xs], width=0.4, color=SERIES[0], label="1st-harmonic share of α-variance")
    ax.set_xticks(xs); ax.set_xticklabels(["emb"] + [f"B{L}" for L in range(1, nl + 1)])
    ax.set_ylim(0, 1.35); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_ylabel("fraction of variance")
    ax.set_title("Residual stream", fontsize=9); ax.legend(fontsize=6, loc="upper left", ncol=1)
    ax = axes[2]
    s = alive & (tab["best_r2"] > 0.05)
    d = r2w[s] - r2u[s]
    sc = ax.scatter(d, rvi[s], c=tab["layer"][s], s=4, cmap="physseq", vmin=-1, vmax=nl - 1, lw=0, alpha=0.7)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("R²(θg world) − R²(θg − q₁)"); ax.set_ylabel("rotation-variance index")
    ax.set_title("Units: frame vs rotation", fontsize=9); ax.set_ylim(0, 1.02)
    cb = fig.colorbar(sc, ax=ax, ticks=range(nl), fraction=0.06); cb.set_label("layer", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    savefig(fig, "grav_equivariance.png")


if __name__ == "__main__":
    main()
