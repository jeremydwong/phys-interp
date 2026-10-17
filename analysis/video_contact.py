"""Analysis 5: contact units around contact onsets (PSTH), floor vs wall, near-miss control, anticipation.

Contact units: top-3 encoder + top-3 space-time units by census R2 with any contact variable.
Activations computed densely (every frame t) on test episodes for these units only.
Events: clean onsets (no contact in the 6 frames before) at frame k; lag = t - k, where t is the
last input frame. Near misses: local minima of the arm-surface clearance < 2 cm with no contact
within +-6 frames (clearance computed from capsule geometry; arm radius 4-4.5 cm).
Anticipation test: among windows with no contact at t and clearance(t) < 6 cm, AUC of each unit's
activation for "contact at t+1" vs "no contact at t+1"; compared to a state-only baseline
(logistic regression on clearance and approach speed from ground truth).
Figure: video_contact.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, INK2, MUTED, SERIES, savefig, setup_style  # noqa: E402
from analysis.video_census import load_census, rank_units  # noqa: E402
from analysis.video_common import CACHE, LAYERS, RESULTS, load, target_matrix, windows  # noqa: E402
from analysis.video_common import TEST as TEST_ALL  # noqa: E402

TEST = TEST_ALL[:250]     # dense (every frame) on 250 test episodes to keep compute light
from sim.arm import FLOOR_Z, L1, L2, WALL_X  # noqa: E402

LAGS = np.arange(-10, 11)


def clearance(q):
    """q [..., 2] -> (floor clearance, wall clearance) in metres (min over both capsules)."""
    s = np.linspace(0, 1, 12)
    q1, q12 = q[..., 0:1], q[..., 0:1] + q[..., 1:2]
    ex, ez = L1 * np.cos(q1), L1 * np.sin(q1)
    # MuJoCo hinge axis (0,-1,0): positive q rotates x toward +z
    ux, uz = s * ex, s * ez
    fx, fz = ex + s * L2 * np.cos(q12), ez + s * L2 * np.sin(q12)
    floor = np.minimum((uz - FLOOR_Z - 0.045).min(-1), (fz - FLOOR_Z - 0.04).min(-1))
    wall = np.minimum((WALL_X - ux - 0.045).min(-1), (WALL_X - fx - 0.04).min(-1))
    return floor, wall


@torch.no_grad()
def dense_units(model, units, H, eps, bs=128):
    E = np.repeat(eps, 145); T = np.tile(np.arange(3, 148), len(eps))
    out = np.zeros((len(E), len(units)), np.float32)
    model.record = True
    for i in range(0, len(E), bs):
        model(windows(E[i:i + bs], T[i:i + bs]))
        for k, u in enumerate(units):
            out[i:i + bs, k] = model.acts[LAYERS[u // H]][:, u % H].amax((1, 2)).cpu().numpy()
    model.record = False
    return out.reshape(len(eps), 145, len(units))      # index [e, t-3, k]


def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))


def main():
    setup_style()
    Y, vs, st = target_matrix()
    names = [v.name for v in vs]
    c = load_census("final")
    H = int(c["H"])
    cv = ["contact", "c_floor", "c_wall", "c_fore", "ctq1", "ctq2"]
    enc = rank_units(c, vs, cv, 3, "contact", "enc")
    tmp = rank_units(c, vs, cv, 3, "contact", "tmp")
    units = enc + tmp
    unit_names = [f"{LAYERS[u // H]}.{u % H}" for u in units]
    model, _ = load("final")
    path = os.path.join(CACHE, "contact_dense.npz")
    if os.path.exists(path) and list(np.load(path)["units"]) == units:
        D = np.load(path)["D"]
    else:
        D = dense_units(model, units, H, TEST)
        np.savez(path, D=D, units=np.array(units))
    mu, sd = D.reshape(-1, len(units)).mean(0), D.reshape(-1, len(units)).std(0) + 1e-6
    Z = (D - mu) / sd
    flag = st["contact_flag"][TEST]; cf = st["contact_floor"][TEST]; cw = st["contact_wall"][TEST]
    fl, wl = clearance(st["q"][TEST])

    def collect(events):
        rows = []
        for e, k in events:
            ts = k + LAGS
            if ts.min() < 3 or ts.max() > 147:
                continue
            rows.append(Z[e, ts - 3])
        return np.array(rows)     # [n, lags, units]

    ev = {"floor": [], "wall": [], "near_floor": [], "near_wall": []}
    for e in range(len(TEST)):
        for k in range(8, 148):
            if flag[e, k] > 0 and flag[e, k - 6:k].sum() == 0:
                if cf[e, k] > 0 and cw[e, k] == 0:
                    ev["floor"].append((e, k))
                elif cw[e, k] > 0 and cf[e, k] == 0:
                    ev["wall"].append((e, k))
        for name, cl in [("near_floor", fl[e]), ("near_wall", wl[e])]:
            if cl.min() < -0.05:      # a few episodes start with the arm through the wall: skip
                continue
            for k in range(8, 142):
                if (0 < cl[k] < 0.02 and cl[k] <= cl[k - 1] and cl[k] <= cl[k + 1]
                        and flag[e, k - 6:k + 7].sum() == 0):
                    ev[name].append((e, k))
    P = {k: collect(v) for k, v in ev.items()}
    counts = {k: int(len(v)) for k, v in P.items()}
    print(counts)

    # anticipation: no contact at t, clearance < 6 cm; does unit at t predict contact at t+1?
    E_, T_ = np.meshgrid(np.arange(len(TEST)), np.arange(3, 147), indexing="ij")
    E_, T_ = E_.ravel(), T_.ravel()
    clr = np.minimum(fl, wl)
    sel = (flag[E_, T_] == 0) & (clr[E_, T_] < 0.06)
    y = flag[E_, T_ + 1][sel] > 0
    A = Z[E_, T_ - 3][sel]
    aucs = [float(auc(A[y, k], A[~y, k])) for k in range(len(units))]
    # state baseline: clearance and its rate of change (approach speed), ground truth
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    Xs = np.stack([clr[E_, T_], clr[E_, T_] - clr[E_, T_ - 1], fl[E_, T_], wl[E_, T_]], 1)[sel]
    half = E_[sel] < len(TEST) // 2
    lr = LogisticRegression(max_iter=1000).fit(Xs[half], y[half])
    auc_state = float(roc_auc_score(y[~half], lr.predict_proba(Xs[~half])[:, 1]))
    lr2 = LogisticRegression(max_iter=1000, C=1.0).fit(A[half], y[half])
    auc_units = float(roc_auc_score(y[~half], lr2.predict_proba(A[~half])[:, 1]))
    # population: 'is contact at t+1' using all units of tmp3 + clearance-matched? use unit combo above.
    res = {"units": unit_names, "event_counts": counts, "anticipation_auc_per_unit": dict(zip(unit_names, aucs)),
           "anticipation_auc_state_baseline": auc_state, "anticipation_auc_6units_logistic": auc_units,
           "n_anticipation_windows": int(sel.sum()), "n_contact_next": int(y.sum())}
    for k in P:
        if len(P[k]):
            m = P[k].mean(0)
            res[f"psth_{k}"] = {unit_names[j]: [float(x) for x in m[:, j]] for j in range(len(units))}
    res["lags"] = LAGS.tolist()
    res["unit_best"] = {unit_names[j]: (vs[int(c["best"][u])].name, float(c["best_r2"][u])) for j, u in enumerate(units)}
    json.dump(res, open(os.path.join(RESULTS, "video_contact.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("psth")}, indent=1))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for j in range(len(units)):
        ax = axes.flat[j]
        for k, col, ls in [("floor", FAMILIES["contact"][0], "-"), ("wall", SERIES[3], "-"),
                           ("near_floor", FAMILIES["contact"][0], ":"), ("near_wall", SERIES[3], ":")]:
            if len(P[k]) == 0:
                continue
            m, se = P[k][:, :, j].mean(0), P[k][:, :, j].std(0) / np.sqrt(len(P[k]))
            lab = {"floor": f"floor onset (n={counts['floor']})", "wall": f"wall onset (n={counts['wall']})",
                   "near_floor": f"floor near-miss (n={counts['near_floor']})",
                   "near_wall": f"wall near-miss (n={counts['near_wall']})"}[k]
            ax.plot(LAGS, m, ls, color=col, lw=1.4, label=lab)
            ax.fill_between(LAGS, m - se, m + se, color=col, alpha=0.15, lw=0)
        ax.axvline(0, color=INK2, lw=0.7)
        ax.axvline(-1, color=MUTED, lw=0.5, ls="--")
        u = units[j]
        ax.set_title(f"{unit_names[j]} · {vs[int(c['best'][u])].label} ({c['best_r2'][u]:.2f})\n"
                     f"AUC for contact at t+1: {aucs[j]:.2f}", fontsize=7)
        if j % 3 == 0:
            ax.set_ylabel("activation (z)")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
    fig.supxlabel("frames from contact onset / closest approach (t − k); dashed line: last input frame before contact",
                  fontsize=7.5, color=INK2, y=0.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, "video_contact.png")


if __name__ == "__main__":
    main()
