"""Analysis 6 -- cue conflict: can an unseen constant torque masquerade as rotated gravity?

Construction. True gravity θg. Pick a "decoy" direction θ' = θg + δ and the posture q* that hangs
straight along θ' (q1* = θ' - π/2, q2* = 0). Apply a constant joint torque τh = g_bias(q*; θg)
that makes q* a static equilibrium under the *true* gravity, i.e. exactly what gravity θ' would
look like at that posture. Start the passive arm at q* + small or large perturbation (|Δq| = 0.15
or 1.2 rad, at rest) and let it swing for 64 frames.
  * hidden:  the model is told tau = 0 (training never contains unobserved torques)
  * visible: the model is given tau = τh (an ordinary in-distribution torque input)
Near q* the constant torque and the rotated gravity give nearly the same restoring force, so a
small swing is (almost) ambiguous; a large swing samples postures where they differ.
Readouts: decoded gravity (pooled probe at the main probe layer), the ideal observer under the
same input assumption, and the model's next-frame prediction error.

    uv run python analysis/grav_conflict.py
Figure: grav_conflict.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import INK2, MUTED, SERIES, savefig, setup_style  # noqa: E402
from analysis.grav_common import (Dyn, decode_world, g_bias_from, ideal_observer, pooled_probe,  # noqa: E402
                                  probe_frame, probe_layer, update_json, vec_angle, wrap)
from analysis.grav_sim import simulate_custom  # noqa: E402
from models.grav_model import CTX, features, load, raw_targets, run  # noqa: E402

DELTAS = np.radians([-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90])
AMPS = {"small": 0.15, "large": 1.2}
N = 40
T_MIN = 16


def circ_mean(a, axis=None):
    return np.angle(np.exp(1j * a).mean(axis))


def main():
    setup_style()
    model = load("incontext")
    L = probe_layer()
    pp = pooled_probe(model, L, frame=probe_frame())
    dyn = Dyn()
    rng = np.random.default_rng(3)
    th = rng.uniform(-np.pi, np.pi, N)
    signs = rng.choice([-1, 1], (N, 2))
    res = dict(probe_layer=L, probe_frame=pp["frame"], n=N)
    curves = {}
    for amp_name, amp in AMPS.items():
        for d in DELTAS:
            qs = np.stack([th + d - np.pi / 2, np.zeros(N)], -1)
            tau_h = g_bias_from(qs, th, dyn=dyn)
            eps = [simulate_custom(0, "passive", CTX + 1, th[i], q0=qs[i] + amp * signs[i] * np.array([1, 0.6]),
                                   qd0=np.zeros(2), hidden_tau=tau_h[i]) for i in range(N)]
            q = np.stack([e["q"] for e in eps]); qd = np.stack([e["qd"] for e in eps])
            Ytrue = raw_targets(q, qd)
            for vis in ("hidden", "visible"):
                tau = np.zeros_like(q) if vis == "hidden" else np.repeat(tau_h[:, None], CTX + 1, 1).astype(np.float32)
                X = features(q[:, :-1], qd[:, :-1], tau[:, :-1])
                pred, acts = run(model, X, keys=[f"resid{L}"])
                dec = decode_world(pp, acts[f"resid{L}"], q[:, :-1])        # [N, CTX] world-frame angle
                off = wrap(circ_mean(dec[:, T_MIN:], 1) - th)
                u = ideal_observer(q[:, :-1], qd[:, :-1], tau[:, :-1], dyn=dyn)
                obs = wrap(vec_angle(u[:, -1]) - th)
                nmse = float((((pred - Ytrue) / model.y_std.cpu().numpy()) ** 2)[:, T_MIN:].mean())
                curves[(amp_name, vis, d)] = (off, obs, nmse, dec)
                res[f"{amp_name}_{vis}_d{int(round(np.degrees(d)))}"] = dict(
                    model_offset_deg=float(np.degrees(circ_mean(off))),
                    observer_offset_deg=float(np.degrees(circ_mean(obs))),
                    model_offset_sd_deg=float(np.degrees(np.std(wrap(off - circ_mean(off))))),
                    pred_nmse=nmse)
            print(amp_name, np.degrees(d), res[f"{amp_name}_hidden_d{int(round(np.degrees(d)))}"], flush=True)
    # slope of decoded offset vs decoy offset (|δ| <= 45°)
    sel = np.abs(DELTAS) <= np.radians(45) + 1e-9
    for amp_name in AMPS:
        for vis in ("hidden", "visible"):
            y = np.array([circ_mean(curves[(amp_name, vis, d)][0]) for d in DELTAS])
            yo = np.array([circ_mean(curves[(amp_name, vis, d)][1]) for d in DELTAS])
            res[f"slope_{amp_name}_{vis}"] = float(np.polyfit(DELTAS[sel], y[sel], 1)[0])
            res[f"slope_observer_{amp_name}_{vis}"] = float(np.polyfit(DELTAS[sel], yo[sel], 1)[0])
    update_json("conflict", res)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    dd = np.degrees(DELTAS)
    cols = {"small": SERIES[0], "large": SERIES[1]}
    for ax, vis in zip(axes[:2], ("hidden", "visible")):
        for amp_name in AMPS:
            off = np.array([curves[(amp_name, vis, d)][0] for d in DELTAS])
            obs = np.array([curves[(amp_name, vis, d)][1] for d in DELTAS])
            m = np.degrees([circ_mean(o) for o in off])
            lo = np.degrees([np.percentile(wrap(o - circ_mean(o)), 25) for o in off]) + m
            hi = np.degrees([np.percentile(wrap(o - circ_mean(o)), 75) for o in off]) + m
            ax.fill_between(dd, lo, hi, color=cols[amp_name], alpha=0.2, lw=0)
            ax.plot(dd, m, "-o", ms=3, color=cols[amp_name], label=f"model, swing {AMPS[amp_name]} rad")
            ax.plot(dd, np.degrees([circ_mean(o) for o in obs]), ls=":", color=cols[amp_name], lw=1.3,
                    label="ideal observer")
        ax.plot(dd, dd, color=INK2, lw=0.8, ls="--")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("decoy offset δ (°)")
        ax.set_xticks([-90, -45, 0, 45, 90])
        ax.set_ylim(-185, 185); ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_title(f"Constant torque {'hidden (τ input = 0)' if vis == 'hidden' else 'visible'}", fontsize=9)
    axes[0].set_ylabel("decoded θg − true θg (°)")
    axes[0].legend(fontsize=6, loc="upper left")
    ax = axes[2]
    for amp_name in AMPS:
        for vis, ls in (("hidden", "-"), ("visible", "--")):
            ax.plot(dd, [curves[(amp_name, vis, d)][2] for d in DELTAS], ls=ls, color=cols[amp_name],
                    label=f"{amp_name} swing, {vis}")
    ax.set_yscale("log"); ax.set_xlabel("decoy offset δ (°)"); ax.set_ylabel("next-frame MSE (z-scored)")
    ax.set_xticks([-90, -45, 0, 45, 90]); ax.set_title("Prediction error", fontsize=9)
    ax.legend(fontsize=5.8)
    fig.tight_layout()
    savefig(fig, "grav_conflict.png")


if __name__ == "__main__":
    main()
