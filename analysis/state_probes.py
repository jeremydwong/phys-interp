"""Analyses 2 and 7: linear probes of the residual stream.

Probes are ridge regressions (lambda = 1e-2 on standardized features) fit on 80k frames from training
episodes and evaluated (R²) on all held-out frames. Layers: raw 8-d input, embedding, after each block.

Analysis 7 additionally decomposes the true joint acceleration into the contributions of each torque
component, qdd = M^-1 (tau_act - g - c + tau_contact - b qd) = a_act + a_grav + a_cor + a_contact + a_damp,
and probes each contribution, the net torque M qdd, and qdd itself layer by layer.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, N_TRAIN_EPS, SHOULDER, ELBOW, load_states, savefig, setup_style, targets  # noqa
from analysis.state_common import data, load_model, run, update_results  # noqa: E402
from sim.arm import DAMPING  # noqa: E402

LAYERS = ["input", "embed", "block 1", "block 2", "block 3", "block 4"]


def fit_ridge(Xtr, Ytr, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = (Xtr - mu) / sd
    Z = np.c_[Z, np.ones(len(Z))]
    A = Z.T @ Z / len(Z) + lam * np.diag(np.r_[np.ones(Z.shape[1] - 1), 0])
    W = np.linalg.solve(A, Z.T @ Ytr / len(Z))
    return lambda X: np.c_[(X - mu) / sd, np.ones(len(X))] @ W


def r2(P, Y):
    return 1 - ((P - Y) ** 2).mean(0) / Y.var(0)


def accel_components(st):
    """Per-frame acceleration contributions [E,T,2] of each torque component (unclipped)."""
    M = np.stack([np.stack([st["M11"], st["M12"]], -1), np.stack([st["M12"], st["M22"]], -1)], -2)
    Minv = np.linalg.inv(M.astype(np.float64))
    comps = dict(act=st["tau_act"], grav=-st["g_bias"], cor=-st["c_bias"], contact=st["tau_contact"],
                 damp=-DAMPING * st["qd"])
    out = {f"tau_{k}": v.astype(np.float64) for k, v in comps.items()}
    out.update({f"a_{k}": np.einsum("etij,etj->eti", Minv, v.astype(np.float64)) for k, v in comps.items()})
    out["tau_net"] = sum(comps[k].astype(np.float64) for k in comps)
    out["qdd"] = st["qdd"].astype(np.float64)
    return out


def layer_feats(model, X):
    o = run(model, X)
    return [X] + [o["resid"][i] for i in range(o["resid"].shape[0])]


def main():
    d = data()
    st = d["st"]
    rng = np.random.default_rng(0)
    tr_idx = rng.choice(len(d["Xtr"]), 80000, replace=False)
    Xtr, Ytr = d["Xtr"][tr_idx], d["Ytr"][tr_idx]
    # acceleration decomposition targets
    ac = accel_components(st)
    E, T = st["q"].shape[:2]
    a_step = np.zeros_like(ac["qdd"])
    a_step[:, :-1] = (st["qd"][:, 1:] - st["qd"][:, :-1]) / 0.02     # mean acceleration over the 20 ms step
    ac["a_step"] = a_step
    cfl = st["contact_flag"] > 0
    free = ~cfl[:, :-1] & ~cfl[:, 1:]                                # no contact at t or t+1
    free_tr = free[:N_TRAIN_EPS].reshape(-1)[tr_idx]
    free_te = free[N_TRAIN_EPS:].reshape(-1)
    acn = list(ac)
    AC = np.concatenate([ac[k] for k in acn], -1)                 # [E,T,2K]
    ACtr = AC[:N_TRAIN_EPS, :-1].reshape(-1, AC.shape[-1])[tr_idx]
    ACte = AC[N_TRAIN_EPS:, :-1].reshape(-1, AC.shape[-1])
    # robust: clip contact-dominated outliers to the 0.5-99.5 percentile of the train set for probing
    lo, hi = np.percentile(ACtr, 0.5, 0), np.percentile(ACtr, 99.5, 0)
    ACtr, ACte = np.clip(ACtr, lo, hi), np.clip(ACte, lo, hi)
    acn2 = [f"{k}{j + 1}" for k in acn for j in range(2)]
    qf = ac["qdd"][N_TRAIN_EPS:, :-1][free[N_TRAIN_EPS:]]
    var_share = {k: (np.var(ac[k][N_TRAIN_EPS:, :-1][free[N_TRAIN_EPS:]], 0) / np.var(qf, 0)).tolist()
                 for k in acn if k.startswith("a_")}       # on contact-free frames
    var_share["a_contact"] = (np.var(ac["a_contact"][N_TRAIN_EPS:, :-1], axis=(0, 1)) /
                              np.var(ac["qdd"][N_TRAIN_EPS:, :-1], axis=(0, 1))).tolist()   # all frames

    res = {}
    for tag, ck in (("trained", "final"), ("random", "step000000")):
        model, _ = load_model(ck)
        Ftr, Fte = layer_feats(model, Xtr), layer_feats(model, d["X"])
        R = np.zeros((len(LAYERS), Ytr.shape[1]))
        RA = np.zeros((len(LAYERS), ACtr.shape[1]))
        RAF = np.zeros((len(LAYERS), ACtr.shape[1]))
        for l in range(len(LAYERS)):
            R[l] = r2(fit_ridge(Ftr[l], Ytr)(Fte[l]), d["Y"])
            RA[l] = r2(fit_ridge(Ftr[l], ACtr)(Fte[l]), ACte)
            RAF[l] = r2(fit_ridge(Ftr[l][free_tr], ACtr[free_tr])(Fte[l][free_te]), ACte[free_te])
        res[tag] = dict(probe_r2=R, accel_probe_r2=RA, accel_probe_r2_free=RAF)
        print(tag)
        for l in range(len(LAYERS)):
            print(f"  {LAYERS[l]:8s}", " ".join(f"{n}={v:.2f}" for n, v in zip(d["names"], R[l])))
            print(f"  {'all':8s}", " ".join(f"{n}={v:.2f}" for n, v in zip(acn2, RA[l])))
            print(f"  {'free':8s}", " ".join(f"{n}={v:.2f}" for n, v in zip(acn2, RAF[l])))

    # ---------------- figure: probe heatmap
    setup_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
    for ax, tag in zip(axes, ("trained", "random")):
        R = np.clip(res[tag]["probe_r2"], 0, 1)
        ax.imshow(R, cmap="physseq", vmin=0, vmax=1, aspect="auto")
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                ax.text(j, i, f"{R[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.3,
                        color="white" if R[i, j] > 0.6 else "#0b0b0b")
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels(LAYERS, fontsize=7)
        ax.grid(False)
        ax.set_title("Trained network" if tag == "trained" else "Random-init control", fontsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
    ax = axes[-1]
    ax.set_xticks(range(len(d["names"])))
    ax.set_xticklabels([v.label for v in d["vars"]], rotation=60, ha="right", fontsize=6.5, rotation_mode="anchor")
    for t, v in zip(ax.get_xticklabels(), d["vars"]):
        t.set_color(FAMILIES[v.family][0])
    fig.subplots_adjust(hspace=0.18)
    savefig(fig, "state_probes_heatmap.png")

    # ---------------- figure: acceleration decomposition probes
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), sharey=True)
    show = [("qdd", "q̈(t), instantaneous", "#0b0b0b"), ("a_step", "Δq̇/Δt, step mean", "#898781"),
            ("a_act", "motor part", FAMILIES["applied"][0]),
            ("a_grav", "gravity part", FAMILIES["gravity"][0]), ("a_cor", "Coriolis/centrip. part", FAMILIES["interaction"][0]),
            ("a_contact", "contact part", FAMILIES["contact"][0])]
    x = np.arange(len(LAYERS))
    for j, ax in enumerate(axes):
        for k, lab, c in show:
            i = acn2.index(f"{k}{j + 1}")
            key = "accel_probe_r2" if k == "a_contact" else "accel_probe_r2_free"
            lab2 = lab + (" (all frames)" if k == "a_contact" else "")
            ax.plot(x, np.clip(res["trained"][key][:, i], 0, 1), "-o", color=c, ms=3.5, lw=1.6, label=lab2)
            ax.plot(x, np.clip(res["random"][key][:, i], 0, 1), "--", color=c, lw=0.9, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(LAYERS, rotation=30, ha="right", fontsize=7)
        ax.set_title(["Shoulder (joint 1)", "Elbow (joint 2)"][j], fontsize=8.5,
                     color=[SHOULDER, ELBOW][j])
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("probe test R²")
    from matplotlib.lines import Line2D
    h, l_ = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], color="#898781", ls="--", lw=0.9)]
    l_ += ["dashed: random-init control"]
    fig.legend(h, l_, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.24), handlelength=1.5)
    savefig(fig, "state_accel_probes.png")

    sel = {}
    for tag in res:
        sel[tag] = {LAYERS[l]: {n: float(res[tag]["probe_r2"][l, i]) for i, n in enumerate(d["names"])}
                    for l in range(len(LAYERS))}
        sel[tag + "_accel_allframes"] = {LAYERS[l]: {n: float(res[tag]["accel_probe_r2"][l, i]) for i, n in enumerate(acn2)}
                                         for l in range(len(LAYERS))}
        sel[tag + "_accel_contactfree"] = {LAYERS[l]: {n: float(res[tag]["accel_probe_r2_free"][l, i]) for i, n in enumerate(acn2)}
                                           for l in range(len(LAYERS))}
    sel["accel_variance_share"] = var_share
    sel["frac_contact_free_test_frames"] = float(free_te.mean())
    update_results("probes", sel)


if __name__ == "__main__":
    main()
