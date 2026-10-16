"""Analysis 6: acceleration, gravity and segment specificity.

(a) Layer-wise ridge R2 for qdd, gravity torque and interaction torque (from video_probes),
    contrasted with what the instantaneous state alone can explain: an MLP from ground-truth
    (sin/cos q, qd) to each target (fit on probe-train episodes, R2 on test). Targets that are
    functions of the state (gravity, Coriolis) are 'free' once posture/velocity are encoded;
    acceleration requires the unobserved motor torque, i.e. must be read from the frames.
(b) Encoder (per-frame) vs space-time stage: gain in probe R2 from enc3 -> tmp3.
(c) Shoulder-vs-elbow selectivity index SI = (R2_shoulder - R2_elbow) / (R2_shoulder + R2_elbow)
    for units whose best variable is a torque/acceleration with a joint index.
Figure: video_accel.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, INK2, MUTED, SERIES, savefig, setup_style  # noqa: E402
from analysis.video_census import load_census, state_control, state_features  # noqa: E402
from analysis.video_common import CACHE, LAYERS, RESULTS, PROBE_TRAIN, TEST, grid, target_matrix  # noqa: E402

PAIRS = [("qdd1", "qdd2"), ("grav1", "grav2"), ("int1", "int2"), ("cor1", "cor2"), ("tau1", "tau2"), ("ctq1", "ctq2")]


def main():
    setup_style()
    import matplotlib.pyplot as plt
    Y, vs, st = target_matrix()
    names = [v.name for v in vs]
    P = np.load(os.path.join(CACHE, "probes.npz"))
    Etr, Ttr = grid(PROBE_TRAIN, 3); Ete, Tte = grid(TEST, 3)
    _, r2_state = state_control(state_features(st, Etr, Ttr), Y[Etr, Ttr], state_features(st, Ete, Tte), Y[Ete, Tte])
    # previous-frames state: (q,qd) at t and t-1 (what 4 frames could at most give, roughly)
    Xtr2 = np.concatenate([state_features(st, Etr, Ttr), state_features(st, Etr, Ttr - 1), state_features(st, Etr, Ttr - 2)], 1)
    Xte2 = np.concatenate([state_features(st, Ete, Tte), state_features(st, Ete, Tte - 1), state_features(st, Ete, Tte - 2)], 1)
    _, r2_hist = state_control(Xtr2, Y[Etr, Ttr], Xte2, Y[Ete, Tte])
    c = load_census("final")
    H = int(c["H"])
    # (c) selectivity index
    si, si_fam, si_layer = [], [], []
    for u in np.where(c["live"] & (c["best_r2"] >= 0.25))[0]:
        b = names[int(c["best"][u])]
        for p in PAIRS:
            if b in p:
                rs, re = c["R"][u, names.index(p[0])], c["R"][u, names.index(p[1])]
                si.append((rs - re) / (rs + re + 1e-9)); si_fam.append(vs[int(c["best"][u])].family)
                si_layer.append(LAYERS[u // H])
    si = np.array(si)
    res = {"state_mlp_r2": dict(zip(names, map(float, r2_state))),
           "state_history_mlp_r2": dict(zip(names, map(float, r2_hist))),
           "n_torque_units": int(len(si)),
           "si_frac_shoulder_selective(>0.5)": float((si > 0.5).mean()) if len(si) else None,
           "si_frac_elbow_selective(<-0.5)": float((si < -0.5).mean()) if len(si) else None,
           "si_frac_mixed": float((np.abs(si) <= 0.5).mean()) if len(si) else None,
           "si_by_family": {f: int(sum(1 for x in si_fam if x == f)) for f in set(si_fam)},
           "enc3_vs_tmp3": {v: {"enc3": float(P["final"][3, names.index(v)]), "tmp3": float(P["final"][7, names.index(v)]),
                                "best_layer": LAYERS[int(np.argmax(P["final"][:, names.index(v)]))],
                                "best": float(P["final"][:, names.index(v)].max()),
                                "pixels4": float(P["pixels4"][names.index(v)]),
                                "init_tmp3": float(P["init"][7, names.index(v)]),
                                "mlp_tmp3": float(P["mlp_tmp3"][names.index(v)]),
                                "mlp_enc3": float(P["mlp_enc3"][names.index(v)]),
                                "state_mlp": float(r2_state[names.index(v)]),
                                "state_hist_mlp": float(r2_hist[names.index(v)])}
                            for v in ["qdd1", "qdd2", "grav1", "grav2", "int1", "int2", "cor1", "cor2", "tau1", "tau2",
                                      "c_floor", "contact", "qd1", "qd2", "ke"]}}
    json.dump(res, open(os.path.join(RESULTS, "video_accel.json"), "w"), indent=1)
    print(json.dumps(res["enc3_vs_tmp3"], indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axes[0]
    vv = ["qdd1", "qdd2", "grav1", "grav2", "int1", "int2", "tau1", "tau2"]
    x = np.arange(len(vv))
    w = 0.16
    series = [("raw pixels (ridge)", [P["pixels4"][names.index(v)] for v in vv], "#c3c2b7"),
              ("encoder enc3 (ridge)", [P["final"][3, names.index(v)] for v in vv], SERIES[0]),
              ("space-time tmp3 (ridge)", [P["final"][7, names.index(v)] for v in vv], SERIES[1]),
              ("tmp3 (MLP probe)", [P["mlp_tmp3"][names.index(v)] for v in vv], SERIES[2]),
              ("true state q, q̇ (MLP)", [r2_state[names.index(v)] for v in vv], INK2)]
    for k, (lab, vals, col) in enumerate(series):
        ax.bar(x + (k - 2) * w, np.clip(vals, 0, 1), w, color=col, label=lab)
    ax.set_xticks(x, [vs[names.index(v)].label for v in vv], rotation=35, ha="right", fontsize=6.5)
    for i, v in enumerate(vv):
        ax.get_xticklabels()[i].set_color(FAMILIES[vs[names.index(v)].family][0])
    ax.set_ylabel("held-out R²"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5.8, ncol=2, loc="upper right")
    ax.set_title("Decoding acceleration and torques", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    if len(si):
        ax.hist(si, bins=np.linspace(-1, 1, 21), color=FAMILIES["interaction"][0], alpha=0.85)
    ax.set_xlabel("selectivity  (shoulder − elbow)/(sum)")
    ax.set_ylabel("units")
    ax.set_title(f"Segment specificity (n={len(si)})", fontsize=9)
    ax.text(0.02, 0.97, "elbow", transform=ax.transAxes, fontsize=6.5, color=MUTED, va="top")
    ax.text(0.98, 0.97, "shoulder", transform=ax.transAxes, fontsize=6.5, color=MUTED, va="top", ha="right")
    fig.tight_layout()
    savefig(fig, "video_accel.png")


if __name__ == "__main__":
    main()
