"""Analysis 4: receptive fields and feature visualization.

(a) Posture receptive fields: activation of gravity units on a 90x90 grid of (q1, q2) with qd = 0, tau = 0,
    next to the analytic gravity torque -g(q) computed with MuJoCo (zero-velocity bias force). Poses that
    penetrate the floor or wall are shaded (they never occur in data).
(b) Input optimization: gradient ascent on the unit's pre-activation over a valid input domain
    (angles as free parameters fed through sin/cos, qd = 15 tanh(a) rad/s, tau = TAU_MAX tanh(b)),
    64 random restarts x 400 Adam steps (lr 0.05); we show the best distinct optima.
"""
from __future__ import annotations

import os
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, binned_r2, savefig, setup_style  # noqa: E402
from analysis.state_common import (data, draw_pose, fk, load_census, load_model, obstacle_gaps, pick_units,  # noqa
                                   run, update_results)
from models.state_mlp import QD_SCALE, make_inputs  # noqa: E402
from sim.arm import TAU_MAX, build  # noqa: E402

NG = 90


def analytic_gravity(Q):
    m = build(with_obstacles=False)
    dd = mujoco.MjData(m)
    out = np.zeros((len(Q), 2))
    for i, q in enumerate(Q):
        dd.qpos[:] = q; dd.qvel[:] = 0
        mujoco.mj_forward(m, dd)
        out[i] = -dd.qfrc_bias
    return out


def optimize(model, layer, unit, n=64, steps=400, seed=0, sign=1.0):
    g = torch.Generator().manual_seed(seed)
    th = (torch.rand(n, 2, generator=g) * 2 - 1) * np.pi
    a = torch.randn(n, 2, generator=g) * 0.5
    b = torch.randn(n, 2, generator=g) * 0.5
    params = [th.requires_grad_(), a.requires_grad_(), b.requires_grad_()]
    opt = torch.optim.Adam(params, lr=0.05)
    tm = torch.tensor(TAU_MAX, dtype=torch.float32)
    for _ in range(steps):
        qd = 15 * torch.tanh(a)
        tau = tm * torch.tanh(b)
        u = torch.cat([torch.sin(th), torch.cos(th), qd / QD_SCALE, tau / tm], 1)
        x = model.embed(u)
        for l, blk in enumerate(model.blocks):
            p = blk.fc_in(blk.ln(x))
            if l == layer:
                obj = sign * p[:, unit]
                break
            x = x + blk.fc_out(torch.relu(p))
        opt.zero_grad()
        (-obj.sum()).backward()
        opt.step()
    with torch.no_grad():
        qd = (15 * torch.tanh(a)).numpy()
        tau = (tm * torch.tanh(b)).numpy()
        q = np.angle(np.exp(1j * th.detach().numpy()))
    return q, qd, tau, obj.detach().numpy()


def distinct(q, val, k=3, tol=0.35):
    order = np.argsort(-val)
    keep = []
    for i in order:
        if all(np.abs(np.angle(np.exp(1j * (q[i] - q[j])))).max() > tol for j in keep):
            keep.append(i)
        if len(keep) == k:
            break
    return keep


def main():
    d = data()
    names = d["names"]
    cz = load_census()
    model, _ = load_model("final")
    for p in model.parameters():
        p.requires_grad_(False)
    setup_style()
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    mpl.rcParams.update({"hatch.color": "#898781", "hatch.linewidth": 0.5})

    # ---------------------------------------------------------------- (a) receptive fields
    g1 = pick_units(cz, names, "grav1", k=2, best_in=["grav1"])
    g2 = pick_units(cz, names, "grav2", k=2, best_in=["grav2", "cos_q12"])
    qs = np.linspace(-np.pi, np.pi, NG)
    Q1, Q2 = np.meshgrid(qs, qs, indexing="xy")
    Q = np.stack([Q1.ravel(), Q2.ravel()], -1)
    G = analytic_gravity(Q)
    gaps = obstacle_gaps(Q)
    infeasible = (np.stack(list(gaps.values()), -1).min(-1) < 0).reshape(NG, NG)
    X = make_inputs(Q, np.zeros_like(Q), np.zeros_like(Q))
    H = run(model, X)["h"]
    panels = [("analytic", 0, None)] + [("unit", 0, u) for u in g1] + [("analytic", 1, None)] + [("unit", 1, u) for u in g2]
    fig, axes = plt.subplots(1, 6, figsize=(7.2, 1.75))
    rf = {}
    for ax, (kind, j, u) in zip(axes, panels):
        if kind == "analytic":
            Z = G[:, j].reshape(NG, NG)
            lim = np.abs(Z).max()
            ax.imshow(Z, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi], cmap="physdiv",
                      norm=TwoSlopeNorm(0, -lim, lim))
            ax.set_title(["τ_g shoulder", "τ_g elbow"][j], fontsize=7.5, color=FAMILIES["gravity"][0])
        else:
            l, k, r2v = u
            Z = H[l][:, k].reshape(NG, NG)
            ax.imshow(Z, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi], cmap="physseq")
            feas = ~infeasible.ravel()
            r = binned_r2(G[feas, j], Z.ravel()[feas])
            rf[f"L{l + 1}/{k}"] = dict(target=["grav1", "grav2"][j], data_r2=r2v, grid_binned_r2=float(r))
            ax.set_title(f"L{l + 1}/{k}", fontsize=7.5)
            ax.text(0.03, 0.03, f"R²={r:.2f}", transform=ax.transAxes, fontsize=6, color="#0b0b0b",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        ax.contourf(Q1, Q2, infeasible.astype(float), levels=[0.5, 1.5], colors="none", hatches=["/////"])
        ax.contour(Q1, Q2, infeasible.astype(float), levels=[0.5], colors="#52514e", linewidths=0.4)
        ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"], fontsize=6)
        ax.set_yticks([-np.pi, 0, np.pi]); ax.set_yticklabels(["-π", "0", "π"] if ax is axes[0] else [], fontsize=6)
        ax.grid(False)
        ax.set_xlabel("q₁ (rad)", fontsize=6.5, labelpad=1)
    axes[0].set_ylabel("q₂ (rad)", fontsize=6.5, labelpad=1)
    fig.subplots_adjust(wspace=0.12)
    savefig(fig, "state_featvis_gravity_rf.png")
    print("receptive fields", rf)

    # ---------------------------------------------------------------- (b) input optimization
    picks = []
    excl = set()
    for var, best_in, lab in [("grav1", ["grav1"], "gravity, shoulder"), ("grav2", ["grav2", "cos_q12"], "gravity, elbow"),
                              ("c_floor", None, "floor contact"), ("c_wall", None, "wall contact"),
                              ("qdd1", None, "shoulder accel."), ("qd2", ["qd2"], "elbow velocity")]:
        u = pick_units(cz, names, var, k=1, best_in=best_in, exclude=excl)
        if u:
            excl.add(u[0][:2])
            picks.append((lab, var) + u[0])
    K = 4
    fig, axes = plt.subplots(len(picks), K, figsize=(3.6, 0.95 * len(picks)))
    opt_res = {}
    for r, (lab, var, l, k, r2v) in enumerate(picks):
        q, qd, tau, val = optimize(model, l, k)
        keep = distinct(q, val, K)
        gaps = obstacle_gaps(q[keep])
        mingap = np.stack(list(gaps.values()), -1).min(-1)
        opt_res[f"L{l + 1}/{k}"] = dict(label=lab, var=var, r2=r2v, q=q[keep], qd=qd[keep], tau=tau[keep],
                                         preact=val[keep], min_clearance_m=mingap,
                                         data_max_preact=float(run(model, d["X"])["pre"][l][:, k].max()))
        for c in range(K):
            ax = axes[r, c]
            if c >= len(keep):
                ax.axis("off"); continue
            i = keep[c]
            # hand velocity from joint velocities (Jacobian)
            q1, q12 = q[i, 0], q[i].sum()
            J = np.array([[-0.5 * np.sin(q1) - 0.45 * np.sin(q12), -0.45 * np.sin(q12)],
                          [0.5 * np.cos(q1) + 0.45 * np.cos(q12), 0.45 * np.cos(q12)]])
            draw_pose(ax, q[i], None, vel=J @ qd[i])
            ax.text(0.02, 0.98, f"q̇=({qd[i, 0]:.0f},{qd[i, 1]:.0f})\nτ=({tau[i, 0]:.0f},{tau[i, 1]:.0f})",
                    transform=ax.transAxes, fontsize=4.6, va="top", color="#52514e")
            if c == 0:
                ax.set_ylabel(f"L{l + 1}/{k}\n{lab}", fontsize=6.3, rotation=0, ha="right", va="center", labelpad=4)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    savefig(fig, "state_featvis_optimized.png")
    for kk, v in opt_res.items():
        print(kk, v["label"], "preact", np.round(v["preact"], 2), "data max", round(v["data_max_preact"], 2),
              "clearance", np.round(v["min_clearance_m"], 3))
    update_results("featvis", dict(receptive_fields=rf, optimized=opt_res))


if __name__ == "__main__":
    main()
