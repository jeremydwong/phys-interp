"""Analyses 2 + 3 -- the population code for gravity direction in MLP hidden units.

For every MLP hidden unit (post-GELU, 4 layers x 512) at context positions >= 24 frames:
  * binned R^2 (36 equal-width circular bins) of activation vs
       θg            gravity angle in the world/camera frame
       θg - q1       gravity relative to the upper arm  ("otolith on the upper arm")
       θg - q1 - q2  gravity relative to the forearm
       q1            upper-arm angle (posture control)
    and vs the gravity torques at shoulder / elbow (-g_bias), 30 quantile bins.
  * von Mises fit of the mean tuning curve a + b exp(κ(cos(x-μ)-1)) in the best frame.

    uv run python analysis/grav_tuning.py
Figures: grav_tuning_examples.png, grav_population.png, grav_torque_units.png
Writes models/grav_ckpt/grav_unit_table.npz for the other analyses.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, INK2, MUTED, SERIES, binned_r2, savefig, setup_style  # noqa: E402
from analysis.grav_common import analysis_windows, update_json, wrap  # noqa: E402
from models.grav_model import CKPT, load, run  # noqa: E402

NB = 36
T_MIN, T_STEP = 24, 3
FRAMES = {"world": "θg (world)", "upper": "θg − q₁ (upper-arm frame)", "fore": "θg − q₁ − q₂ (forearm frame)"}
FCOL = {"world": SERIES[0], "upper": SERIES[1], "fore": SERIES[2]}


def circ_bins(x):
    return np.clip(((wrap(x) + np.pi) / (2 * np.pi) * NB).astype(int), 0, NB - 1)


def circ_r2(idx, A):
    """A [N,U]; fraction of variance explained by bin means. returns r2 [U], means [NB,U]."""
    cnt = np.bincount(idx, minlength=NB).astype(float)
    S = np.zeros((NB, A.shape[1]))
    np.add.at(S, idx, A)
    means = S / np.maximum(cnt, 1)[:, None]
    resid = A - means[idx]
    return 1 - resid.var(0) / (A.var(0) + 1e-12), means, cnt


def vm(x, a, b, k, mu):
    return a + b * np.exp(k * (np.cos(x - mu) - 1))


def fit_vm(centers, y):
    i = int(np.argmax(np.abs(y - np.median(y))))
    sgn = np.sign(y[i] - np.median(y)) or 1.0
    p0 = [np.median(y), y[i] - np.median(y), 1.5, centers[i]]
    try:
        p, _ = curve_fit(vm, centers, y, p0=p0, maxfev=4000,
                         bounds=([-np.inf, -np.inf, 0.01, -10], [np.inf, np.inf, 60, 10]))
    except RuntimeError:
        return None
    r2 = 1 - ((y - vm(centers, *p)) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)
    a, b, k, mu = p
    mu = wrap(mu if b > 0 else mu + np.pi)   # direction of the peak response
    fwhm = 360.0 if k < np.log(2) / 2 else np.degrees(2 * np.arccos(1 - np.log(2) / k))
    return dict(a=a, b=b, k=k, mu=float(mu), r2=float(r2), fwhm=float(fwhm), sign=float(np.sign(b)), p=p)


def main():
    setup_style()
    w = analysis_windows()
    model = load("incontext")
    nl = model.cfg["n_layers"]
    _, acts = run(model, w["X"], keys=[f"mlp{l}" for l in range(nl)])
    ts = np.arange(T_MIN, 64, T_STEP)
    A = np.concatenate([acts[f"mlp{l}"][:, ts] for l in range(nl)], -1).reshape(-1, nl * 512)  # [N, 2048]
    del acts
    th = np.repeat(w["theta"][:, None], len(ts), 1).ravel()
    q = w["q"][:, ts].reshape(-1, 2)
    grav = -w["g_bias"][:, ts].reshape(-1, 2)
    var = {"world": th, "upper": th - q[:, 0], "fore": th - q[:, 0] - q[:, 1], "q1": q[:, 0], "q2": q[:, 1]}
    layer = np.repeat(np.arange(nl), 512)
    alive = A.std(0) > 1e-3
    R2, MEANS = {}, {}
    for k, x in var.items():
        R2[k], MEANS[k], _ = circ_r2(circ_bins(x), A)
    # gravity torques (quantile bins, common.binned_r2)
    R2["grav1"] = np.array([binned_r2(grav[:, 0], A[:, u]) for u in range(A.shape[1])])
    R2["grav2"] = np.array([binned_r2(grav[:, 1], A[:, u]) for u in range(A.shape[1])])
    # joint (upper-arm-frame gravity, elbow) -> this is everything gravity torque depends on
    j = circ_bins(var["upper"]) * 12 + np.clip(((wrap(q[:, 1]) + np.pi) / (2 * np.pi) * 12).astype(int), 0, 11)
    cnt = np.bincount(j, minlength=NB * 12).astype(float)
    S = np.zeros((NB * 12, A.shape[1])); np.add.at(S, j, A)
    R2["upper_q2"] = 1 - (A - (S / np.maximum(cnt, 1)[:, None])[j]).var(0) / (A.var(0) + 1e-12)
    # conditional (gain-field) analysis: variance explained by gravity *beyond* posture, velocity and
    # torque, cross-validated across episodes. Interaction terms make world/body frames equivalent here.
    qd = w["qd"][:, ts].reshape(-1, 2) / np.array([6.0, 14.0])
    tau = w["tau_act"][:, ts].reshape(-1, 2) / np.array([11.0, 4.5])
    post = [np.ones(len(th))]
    for x in (q[:, 0], q[:, 1], q[:, 0] + q[:, 1]):
        post += [np.cos(x), np.sin(x), np.cos(2 * x), np.sin(2 * x)]
    post = np.stack(post, 1)
    kin = np.concatenate([post, qd, qd ** 2, qd[:, :1] * qd[:, 1:], np.abs(qd), tau,
                          post[:, 1:5] * qd[:, :1], post[:, 1:5] * qd[:, 1:]], 1)
    gh = np.stack([np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th)], 1)
    gfull = (gh[:, :, None] * post[:, None, :7]).reshape(len(th), -1)
    trn = np.repeat(w["ep"] < 700, len(ts))

    def cv_r2(F):
        Wt = np.linalg.lstsq(F[trn], A[trn], rcond=None)[0]
        r = A[~trn] - F[~trn] @ Wt
        return 1 - r.var(0) / (A[~trn].var(0) + 1e-12)
    R2["kin"] = cv_r2(kin)
    R2["kin_grav"] = cv_r2(np.concatenate([kin, gfull], 1))
    R2["grav_unique"] = R2["kin_grav"] - R2["kin"]
    for k in R2:
        R2[k] = np.where(alive, R2[k], 0)

    centers = -np.pi + (np.arange(NB) + 0.5) * 2 * np.pi / NB
    best = np.array(["world", "upper", "fore"])[np.argmax(np.stack([R2[f] for f in FRAMES]), 0)]
    best_r2 = np.max(np.stack([R2[f] for f in FRAMES]), 0)
    fits = []
    for u in range(A.shape[1]):
        f = fit_vm(centers, MEANS[best[u]][:, u]) if best_r2[u] > 0.1 else None
        fits.append(f)
    TUNED = 0.3
    tuned = best_r2 >= TUNED
    well = np.array([f is not None and f["r2"] >= 0.9 for f in fits]) & tuned
    WEAK = 0.1      # population plots use weakly tuned units (strongly tuned ones are rare)
    weak = np.array([f is not None and f["r2"] >= 0.8 for f in fits]) & (best_r2 >= WEAK)
    mu = np.array([f["mu"] if f else np.nan for f in fits])
    fwhm = np.array([f["fwhm"] if f else np.nan for f in fits])
    np.savez(os.path.join(CKPT, "grav_unit_table.npz"), layer=layer, best=best, best_r2=best_r2, mu=mu,
             fwhm=fwhm, well=well, **{f"r2_{k}": v for k, v in R2.items()})

    # ---------------------------------------------------------------- summary numbers
    res = dict(n_units=int(A.shape[1]), n_alive=int(alive.sum()), tuned_threshold=TUNED)
    for l in range(nl):
        s = layer == l
        res[f"layer{l}"] = dict(
            tuned=int(tuned[s].sum()), well_fit_vonmises=int(well[s].sum()),
            weak_tuned_fit=int(weak[s].sum()),
            weak_frame_counts={f: int(((best == f) & weak & s).sum()) for f in FRAMES},
            median_fwhm_weak=float(np.nanmedian(fwhm[s & weak])) if (s & weak).any() else None,
            best_frame_counts={f: int(((best == f) & tuned & s).sum()) for f in FRAMES},
            median_r2={k: float(np.median(R2[k][s & alive])) for k in R2},
            torque_units=int(((np.maximum(R2["grav1"], R2["grav2"]) >= TUNED) & s).sum()),
            grav_unique_ge_0p1=int(((R2["grav_unique"] >= 0.1) & s).sum()),
            grav_unique_ge_0p3=int(((R2["grav_unique"] >= 0.3) & s).sum()),
            median_grav_unique=float(np.median(R2["grav_unique"][s & alive])),
            median_kin=float(np.median(R2["kin"][s & alive])),
            median_kin_grav=float(np.median(R2["kin_grav"][s & alive])),
            median_fwhm_well=float(np.nanmedian(fwhm[s & well])) if (s & well).any() else None)
    for f in FRAMES:
        sel = weak & (best == f)
        if sel.sum() > 3:
            z = np.exp(1j * mu[sel])
            Rbar = np.abs(z.mean()); n = sel.sum()
            res[f"pref_uniformity_{f}"] = dict(n=int(n), resultant=float(Rbar),
                                               rayleigh_p=float(np.exp(-n * Rbar ** 2)))
    # the frame comparison among tuned units
    t_any = tuned
    res["frame_r2_median_among_tuned"] = {f: float(np.median(R2[f][t_any])) for f in FRAMES}
    res["frac_tuned_world_gt_upper"] = float((R2["world"][t_any] > R2["upper"][t_any]).mean())
    update_json("tuning", res)
    print(res)

    # ---------------------------------------------------------------- figure: examples
    ex = []
    for f in FRAMES:
        sel = np.where(weak & (best == f))[0]
        sel = sel[np.argsort(-best_r2[sel])]
        ex += [(u, f) for u in sel[:4]]
    ex = ex[:12]
    if not ex:   # fall back to the most strongly tuned units
        ex = [(u, best[u]) for u in np.argsort(-best_r2)[:8] if fits[u] is not None]
    nc = 4; nr = int(np.ceil(len(ex) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(7, 1.75 * nr + 0.2), subplot_kw=dict(projection="polar"))
    axes = np.atleast_1d(axes).ravel()
    cc = np.append(centers, centers[0])
    for ax, (u, f) in zip(axes, ex):
        y = MEANS[f][:, u]
        lo = y.min()
        ax.plot(cc, np.append(y, y[0]) - lo, color=FCOL[f], lw=1.6)
        xx = np.linspace(-np.pi, np.pi, 200)
        ax.plot(xx, vm(xx, *fits[u]["p"]) - lo, color=INK2, lw=0.8, ls="--")
        y2 = MEANS["world" if f != "world" else "upper"][:, u]
        ax.plot(cc, np.append(y2, y2[0]) - lo, color=MUTED, lw=0.8)
        ax.set_theta_zero_location("S"); ax.set_theta_direction(1)
        ax.set_xticks(np.radians([0, 90, 180, 270])); ax.set_xticklabels(["0", "90", "180", "270"], fontsize=6)
        ax.set_yticklabels([]); ax.grid(lw=0.4)
        ax.set_title(f"L{layer[u]}·u{u % 512}  R²={best_r2[u]:.2f}\nFWHM {fits[u]['fwhm']:.0f}°", fontsize=7,
                     loc="center", pad=4, color=FCOL[f])
    for ax in axes[len(ex):]:
        ax.axis("off")
    fig.suptitle("Example gravity-tuned MLP units (angle measured from 'down'; colour = best frame: "
                 "blue world, orange upper arm, green forearm; grey = other frame)", fontsize=7.5, y=1.0)
    fig.tight_layout()
    savefig(fig, "grav_tuning_examples.png")

    # ---------------------------------------------------------------- figure: population
    fig = plt.figure(figsize=(7, 4.6))
    gs = fig.add_gridspec(2, 3)
    for i, f in enumerate(FRAMES):
        ax = fig.add_subplot(gs[0, i], projection="polar")
        sel = weak & (best == f)
        bins = np.linspace(-np.pi, np.pi, 25)
        for l in range(nl):
            pass
        h, _ = np.histogram(mu[sel], bins)
        ax.bar(bins[:-1] + np.pi / 24, h, width=2 * np.pi / 24, color=FCOL[f], alpha=0.85, edgecolor="white", lw=0.4)
        ax.set_theta_zero_location("S"); ax.set_theta_direction(1)
        ax.set_xticks(np.radians([0, 90, 180, 270])); ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=6)
        ax.tick_params(axis="y", labelsize=5)
        ax.set_title(f"{FRAMES[f]}\npref. dir. (n={sel.sum()}, R² ≥ {WEAK})", fontsize=7.5, loc="center", pad=8)
    ax = fig.add_subplot(gs[1, 0])
    for f in FRAMES:
        sel = weak & (best == f)
        if sel.sum():
            ax.hist(fwhm[sel], bins=np.linspace(0, 360, 25), color=FCOL[f], alpha=0.6, label=f)
    ax.set_xlabel("tuning width FWHM (°)"); ax.set_ylabel("units"); ax.set_title("Tuning width", fontsize=9)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax = fig.add_subplot(gs[1, 1])
    hb = np.linspace(0, 0.6, 25)
    ax.hist(best_r2[alive], bins=hb, color=SERIES[1], alpha=0.7, label="best-frame R² (marginal)")
    ax.hist(np.clip(R2["grav_unique"][alive], 0, None), bins=hb, color=SERIES[2], alpha=0.7,
            label="gravity R² beyond kinematics")
    ax.axvline(WEAK, color=MUTED, lw=0.8, ls="--"); ax.axvline(TUNED, color=INK2, lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("single-unit R²"); ax.set_ylabel("units (all layers)")
    ax.set_title("Gravity selectivity is weak", fontsize=9); ax.legend(fontsize=6)
    ax = fig.add_subplot(gs[1, 2])
    s = alive & (best_r2 > 0.05)
    sc = ax.scatter(R2["world"][s], R2["upper"][s], c=layer[s], s=4, cmap="physseq", vmin=-1, vmax=nl - 1,
                    alpha=0.7, lw=0)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("R² vs θg (world)"); ax.set_ylabel("R² vs θg − q₁ (upper arm)")
    ax.set_title("World vs body frame", fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    cb = fig.colorbar(sc, ax=ax, ticks=range(nl), fraction=0.05); cb.set_label("layer", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    savefig(fig, "grav_population.png")

    # ---------------------------------------------------------------- figure: torque units
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9))
    ax = axes[0]
    gmax = np.maximum(R2["grav1"], R2["grav2"])
    for l in range(nl):
        s = alive & (layer == l)
        ax.scatter(R2["world"][s], gmax[s], s=4, color=plt.get_cmap("physseq")((l + 1) / nl), lw=0, alpha=0.7,
                   label=f"L{l}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("R² vs θg alone (world)"); ax.set_ylabel("R² vs gravity torque (best joint)")
    ax.set_title("θg units vs gravity-torque units", fontsize=9); ax.legend(fontsize=6.5, markerscale=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax = axes[1]
    keys = [("world", "θg"), ("upper", "θg−q₁"), ("grav1", "τg shoulder"), ("grav2", "τg elbow"),
            ("grav_unique", "θg beyond kin."), ("q2", "q₂ (posture)"), ("upper_q2", "(θg−q₁, q₂)")]
    cols = [FCOL["world"], FCOL["upper"], FAMILIES["gravity"][0], "#0f6b4a", "#8a5ac2", FAMILIES["posture"][0], MUTED]
    xw = 0.115
    for i, ((k, lb), c) in enumerate(zip(keys, cols)):
        m = [np.mean(np.clip(R2[k][(layer == l) & alive], 0, 1)) for l in range(nl)]   # cv R² < 0 -> 0
        ax.bar(np.arange(nl) + (i - 3) * xw, m, width=xw, color=c, label=lb)
    ax.set_xticks(range(nl)); ax.set_xticklabels([f"MLP L{l}" for l in range(nl)])
    ax.set_ylabel("mean single-unit R²"); ax.set_title("What single units track, by layer", fontsize=9)
    ax.legend(fontsize=6.3, ncol=2)
    fig.tight_layout()
    savefig(fig, "grav_torque_units.png")


if __name__ == "__main__":
    main()
