"""Analysis 5 -- causal steering of the gravity representation.

Scenario ("hold, then release"): the arm is driven by a PD controller to a nearby posture and
held there under true gravity θg for 32 frames (0.64 s; the model sees the motion and the holding
torques), then the motors are switched off and the model rolls the arm forward open-loop for 25
frames (0.5 s) with zero torque, feeding back its own predictions.

Intervention: at residual layer L, h <- h + B (u' - B⁺(h - μ)), where B [D,2] is the least-squares
*encoding* map of the gravity unit vector u = (cos θg, sin θg) at that layer (fit on held-out
analysis windows, positions >= 16). This replaces the component of h in the gravity-encoding plane
with the encoding of a steered direction θ' = θg + δ, at every position.
Control: the same edit in a random 2-D subspace with matched column norms.

Readouts: (1) "behavioural gravity" -- run the ideal inverse-dynamics observer on the model's own
rolled-out trajectory; (2) direction of the hand's displacement 4 frames after release vs the
displacement in a true MuJoCo rollout under θ'.

    uv run python analysis/grav_steer.py
Figures: grav_steer.png, grav_steer_strip.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import ELBOW, INK, INK2, MUTED, SERIES, SHOULDER, savefig, setup_style  # noqa: E402
from analysis.grav_common import (Dyn, angerr_deg, elbow_xz, hand_xz, ideal_observer, pooled_probe,  # noqa: E402
                                  probe_frame, probe_layer, update_json, vec_angle, wrap)
from analysis.grav_sim import simulate_custom  # noqa: E402
from models.grav_model import features, load  # noqa: E402
from sim.arm import TAU_MAX  # noqa: E402

TC, NR, KH = 32, 25, 4
DELTAS = np.radians(np.arange(0, 360, 30))
N_SCEN = 80
FRAME = "world"


def make_scenarios(n, seed=0):
    rng = np.random.default_rng(seed)
    kp, kd = np.array([40.0, 18.0]), np.array([4.0, 1.5])
    out = []
    i = 0
    while len(out) < n:
        i += 1
        th = rng.uniform(-np.pi, np.pi)
        q0 = rng.uniform(-np.pi, np.pi, 2) * np.array([1, 0.8])
        tgt = q0 + rng.uniform(-0.8, 0.8, 2)

        def ctrl(t, q, qd, tgt=tgt):
            if t >= TC * 0.02 - 1e-9:
                return np.zeros(2)
            return np.clip(kp * wrap(tgt - q) - kd * qd, -TAU_MAX, TAU_MAX)

        ep = simulate_custom(10_000 + i, "reach", TC + NR + 1, th, q0=q0, qd0=np.zeros(2), ctrl=ctrl)
        # frame index f in ep is the state after f+1 intervals; prepend the initial state
        if np.abs(ep["qd"][TC - 1]).max() > 0.3:
            continue
        out.append(dict(theta=th, q=ep["q"], qd=ep["qd"], tau=ep["tau_act"]))
    return out


def physics_release(q0, qd0, th, n=NR):
    ep = simulate_custom(0, "passive", n, th, q0=q0, qd0=qd0)
    return np.concatenate([q0[None], ep["q"]]), np.concatenate([qd0[None], ep["qd"]])


def steer_basis(pp, mode="encode", rng=None):
    B = pp["B"]
    if mode == "random":
        R, _ = np.linalg.qr(rng.standard_normal(B.shape))
        B = R * np.linalg.norm(B, axis=0, keepdims=True)
    from analysis.device import get_device
    dev = get_device()
    f = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    return dict(B=f(B), Bi=f(np.linalg.pinv(B)), mu=f(pp["mu"]), frame=pp["frame"], dev=dev)


def make_hook(sb, target_u):
    """Replace the component of h in span(B) by the encoding of target_u [B,T,2]."""
    tu = torch.tensor(target_u, dtype=torch.float32, device=sb["dev"])

    def hook(h):
        c = (h - sb["mu"]) @ sb["Bi"].T                 # current coordinates [B,T,2]
        return h + (tu[:, -h.shape[1]:] - c) @ sb["B"].T
    return hook


@torch.no_grad()
def steered_rollout(model, Xc, qc, q0, qd0, n, L=None, sb=None, theta_t=None):
    """Like models.grav_model.rollout with zero torque, but with a steering hook at layer L whose
    target is gravity θ' (theta_t [B]) expressed in the probe's frame for every position
    (upper-arm frame: θ' - q1 of that frame, using the model's own rolled-out q)."""
    dev = next(model.parameters()).device
    q, qd = [q0.astype(np.float32)], [qd0.astype(np.float32)]
    hist, qh = Xc, qc
    for s in range(n):
        f = features(q[-1][:, None], qd[-1][:, None], np.zeros((len(q0), 1, 2), np.float32))
        hist = np.concatenate([hist, f], 1)[:, -model.cfg["ctx"]:]
        qh = np.concatenate([qh, q[-1][:, None]], 1)[:, -model.cfg["ctx"]:]
        hooks = None
        if sb is not None:
            ang = theta_t[:, None] - (qh[..., 0] if sb["frame"] == "upper" else 0)
            hooks = {L: make_hook(sb, np.stack([np.cos(ang), np.sin(ang)], -1))}
        p = model(torch.from_numpy(hist).to(dev), None, hooks)[:, -1]
        p = (p * model.y_std + model.y_mean).cpu().numpy()
        q.append(q[-1] + p[:, :2]); qd.append(qd[-1] + p[:, 2:])
    return np.stack(q, 1), np.stack(qd, 1)


def main():
    setup_style()
    model = load("incontext")
    global FRAME
    FRAME = probe_frame()
    nl = model.cfg["n_layers"]
    scen = make_scenarios(N_SCEN)
    th = np.array([s["theta"] for s in scen])
    Xc = features(np.stack([s["q"][:TC] for s in scen]), np.stack([s["qd"][:TC] for s in scen]),
                  np.stack([s["tau"][:TC] for s in scen]))
    q0 = np.stack([s["q"][TC - 1] for s in scen])
    qd0 = np.stack([s["qd"][TC - 1] for s in scen])
    Xc = Xc[:, :TC - 1]           # context: frames before the release state
    dyn = Dyn()

    def behav(qr, qdr):
        u = ideal_observer(qr, qdr, np.zeros_like(qr), dyn=dyn)
        return vec_angle(u[:, -1])

    # unsteered rollouts and physics
    qc = np.stack([s["q"][:TC - 1] for s in scen])
    q_m, qd_m = steered_rollout(model, Xc, qc, q0, qd0, NR)
    phys = {}
    for d in DELTAS:
        phys[d] = [physics_release(q0[i].astype(np.float64), qd0[i].astype(np.float64), th[i] + d) for i in range(N_SCEN)]
    q_p0 = np.stack([p[0] for p in phys[0.0]])
    res = dict(n_scenarios=N_SCEN, release_frames=NR, context_frames=TC,
               unsteered_behav_err_deg=float(np.median(angerr_deg(behav(q_m, qd_m), th))),
               physics_behav_err_deg=float(np.median(angerr_deg(behav(q_p0, np.stack([p[1] for p in phys[0.0]])), th))),
               unsteered_hand_rmse_m=float(np.sqrt(((hand_xz(q_m) - hand_xz(q_p0)) ** 2).sum(-1).mean())))
    rng = np.random.default_rng(1)
    layers = list(range(1, nl + 1))
    out = {}
    other = "world" if FRAME == "upper" else "upper"
    runs = [(FRAME, L, "encode") for L in layers] + [(FRAME, probe_layer(), "random"), (other, probe_layer(), "encode")]
    for fr, L, mode in runs:
        pp = pooled_probe(model, L, frame=fr)
        if True:
            sb = steer_basis(pp, mode, rng)
            bd, hd = [], []
            for d in DELTAS:
                qs, qds = steered_rollout(model, Xc, qc, q0, qd0, NR, L, sb, th + d)
                b = wrap(behav(qs, qds) - th)
                qp = np.stack([p[0] for p in phys[d]])
                hm = hand_xz(qs[:, KH]) - hand_xz(qs[:, 0])
                hp = hand_xz(qp[:, KH]) - hand_xz(qp[:, 0])
                hd.append(np.degrees(wrap(np.arctan2(hm[:, 1], hm[:, 0]) - np.arctan2(hp[:, 1], hp[:, 0]))))
                bd.append(b)
                if mode == "encode" and L == probe_layer() and fr == FRAME:
                    out.setdefault("traj", {})[d] = qs
            bd, hd = np.array(bd), np.array(hd)
            err = np.degrees(np.abs(wrap(bd - DELTAS[:, None])))
            key = f"L{L}_{mode}" if fr == FRAME else f"L{L}_{mode}_{fr}frame"
            res[key] = dict(behav_err_vs_steer_deg_median=float(np.median(err)),
                                       behav_err_by_delta=[float(np.median(e)) for e in err],
                                       hand_dir_err_vs_physics_deg_median=float(np.median(np.abs(hd))),
                                       behav_shift_median_deg=[float(np.degrees(np.median(np.abs(wrap(b))))) for b in bd],
                                       var_frac=pp["var_frac"])
            out[(L, mode) if fr == FRAME else (L, mode, fr)] = (bd, hd)
            print(fr, L, mode, res[key]["behav_err_vs_steer_deg_median"],
                  res[key]["hand_dir_err_vs_physics_deg_median"], flush=True)
    Ls = probe_layer()
    res["main_layer"] = Ls
    res["frame"] = FRAME
    update_json("steer", res)

    # ------------------------------------------------------------- figure: quantification
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9))
    dd = np.degrees(DELTAS)
    lc = {1: "#9ec5f4", 2: "#5598e7", 3: "#256abf", 4: "#0d366b"}
    for L in layers:
        bd, hd = out[(L, "encode")]
        y = np.degrees(np.mod(bd + np.radians(15), 2 * np.pi)) - 15
        med = dd + np.degrees(np.median(wrap(bd - DELTAS[:, None]), 1))
        axes[0].plot(dd, med, "-o", ms=3, color=lc[L], lw=1.4, label=f"steer @ block {L}" if L else "embedding")
    bd, _ = out[(Ls, "random")]
    axes[0].plot(dd, np.degrees(np.median(wrap(bd), 1)), "-s", ms=3, color=SERIES[1], lw=1.2,
                 label=f"random plane @ block {Ls}")
    bd, _ = out[(Ls, "encode", other)]
    axes[0].plot(dd, dd + np.degrees(np.median(wrap(bd - DELTAS[:, None]), 1)), "-^", ms=3, color=SERIES[4], lw=1.2,
                 label=f"{other}-frame plane @ block {Ls}")
    axes[0].plot([0, 330], [0, 330], color=INK2, lw=0.8, ls="--")
    axes[0].set_xlabel("steered offset δ = θ' − θg (°)")
    axes[0].set_ylabel("behavioural gravity − θg (°)")
    axes[0].set_xticks(range(0, 361, 90)); axes[0].set_yticks(range(0, 361, 90))
    axes[0].set_title("Rolled-out physics follows the steered g", fontsize=9)
    axes[0].legend(fontsize=6.3)
    ax = axes[1]
    for L in layers:
        _, hd = out[(L, "encode")]
        ax.plot(dd, np.median(np.abs(hd), 1), "-o", ms=3, color=lc[L], lw=1.4, label=f"block {L}")
    _, hd = out[(Ls, "random")]
    ax.plot(dd, np.median(np.abs(hd), 1), "-s", ms=3, color=SERIES[1], lw=1.2, label="random plane")
    ax.set_xlabel("steered offset δ (°)"); ax.set_ylabel("|hand-direction error| vs physics(θ') (°)")
    ax.set_xticks(range(0, 361, 90))
    ax.set_title("Initial hand fall direction (first 80 ms)", fontsize=9)
    ax.legend(fontsize=6.3); ax.set_ylim(0, None)
    fig.tight_layout()
    savefig(fig, "grav_steer.png")

    # ------------------------------------------------------------- figure: arm strips
    ex = [0, 1, 2]
    show = np.radians([0, 90, 180, 270])
    fig, axes = plt.subplots(len(ex), len(show), figsize=(7, 1.85 * len(ex)))
    for r, i in enumerate(ex):
        for c, d in enumerate(show):
            ax = axes[r, c]
            qs = out["traj"][min(out["traj"], key=lambda k: abs(wrap(k - d)))][i]
            qp = phys[min(phys, key=lambda k: abs(wrap(k - d)))][i][0]
            for k in range(0, NR + 1, 5):
                al = 0.25 + 0.75 * k / NR
                e, h = elbow_xz(qs[k]), hand_xz(qs[k])
                ax.plot([0, e[0]], [0, e[1]], color=SHOULDER, lw=2.2, alpha=al, solid_capstyle="round")
                ax.plot([e[0], h[0]], [e[1], h[1]], color=ELBOW, lw=1.8, alpha=al, solid_capstyle="round")
            hm, hp = hand_xz(qs), hand_xz(qp)
            ax.plot(hm[:, 0], hm[:, 1], color=INK, lw=1.0, label="model (steered)")
            ax.plot(hp[:, 0], hp[:, 1], color=INK2, lw=1.0, ls="--", label="MuJoCo under θ'")
            for ang, col, lab in ((th[i], MUTED, "true g"), (th[i] + d, SERIES[2], "steered g")):
                gx, gz = np.sin(ang), -np.cos(ang)
                ax.annotate("", xy=(0.75 * gx, 0.75 * gz), xytext=(0.35 * gx, 0.35 * gz),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
            ax.plot(0, 0, "o", color=INK, ms=3)
            ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"steer δ = {np.degrees(d):.0f}°", fontsize=8, loc="center")
    axes[0, 0].legend(fontsize=5.5, loc="upper left")
    fig.text(0.5, -0.01, "Grey arrow: true gravity; green arrow: steered gravity. Arm drawn every 100 ms after "
             "release (faint to opaque); black: model hand path; dashed: MuJoCo under steered gravity.",
             ha="center", fontsize=7, color=INK2)
    fig.tight_layout()
    savefig(fig, "grav_steer_strip.png")


if __name__ == "__main__":
    main()
