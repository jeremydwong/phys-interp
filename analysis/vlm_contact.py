"""Event-aligned activity around contact onsets (floor vs wall).

Timeline: each episode is covered by 9 consecutive clips x 4 temporal groups, i.e. one group
every 4 sim frames (80 ms), aligned to frames 1, 5, ..., 141. A contact onset is a sim frame f
with contact_flag[f] = 1 and no contact in the preceding 8 frames; it is a floor (wall) onset
if only the floor (wall) carries force during f..f+2. For every onset we take each group of the
same episode at lag = aligned_frame - f and average in 4-frame lag bins.

Signals: out-of-fold ridge-probe predictions of "any contact" (LM layer chosen in vlm_probe,
and final vision block), the 3 most contact-selective units of each hooked stream (max-pooled,
z-scored), ground-truth contact fraction, and a purely geometric control: clearance = smallest
distance from elbow/hand to the floor or wall surface. Anticipation is tested by asking
whether the pre-onset rise of a signal exceeds what its relation to clearance predicts
(regress signal on clearance with 20 quantile bins fitted on non-contact groups away from
onsets; report the mean residual at lags -12..-1 vs a far baseline -40..-24).

Usage: uv run python analysis/vlm_contact.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, savefig, setup_style, INK, INK2, MUTED, SERIES  # noqa: E402
from analysis.vlm_common import VDIR, VIS_LAYERS, LLM_LAYERS, n_used  # noqa: E402
from analysis.vlm_probe import aligned_targets  # noqa: E402
from sim.arm import FLOOR_Z, WALL_X  # noqa: E402

LAG_BINS = np.arange(-40, 29, 4)    # bin left edges (sim frames)


def onsets(st, eps):
    out = []
    cf = st["contact_flag"] > 0
    for e in eps:
        for f in range(9, 142):
            if cf[e, f] and not cf[e, f - 8:f].any():
                fl = (st["contact_floor"][e, f:f + 3] > 0).any()
                wl = (st["contact_wall"][e, f:f + 3] > 0).any()
                kind = "floor" if fl and not wl else "wall" if wl and not fl else "both"
                out.append((e, f, kind))
    return out


def clearance(st, e, f):
    pts = np.stack([st["elbow_xz"][e, f], st["ee_xz"][e, f]])
    return np.min(np.concatenate([pts[:, 1] - FLOOR_Z, WALL_X - pts[:, 0]]))


def contact_units(names, fams, NG):
    z = np.load(os.path.join(VDIR, "census_units.npz"))
    with open(os.path.join(VDIR, "census.json")) as f:
        summ = json.load(f)
    ci = names.index("contact")
    picks = []
    for s in summ:
        if s["mode"] != "video":
            continue
        k = s["key"]
        B = z[f"{k}__B"].astype(np.float32)
        u = np.argsort(-B[:, ci])[:3]
        fn = f"video_vis_mlp_L{s['layer']}.npy" if s["part"] == "vision" else f"video_llm_mlp_L{s['layer']}.npy"
        picks.append((k, fn, s["pole"], z[f"{k}__idx"][u], B[u, ci]))
    return picks


def main():
    setup_style()
    import matplotlib.pyplot as plt
    Y, vs, gi, st = aligned_targets()
    NG, _ = n_used("video")
    Y, gi = Y[:NG], gi[:NG]
    names, fams = [v.name for v in vs], [v.family for v in vs]
    eps = np.unique(gi[:, 0])
    ons = onsets(st, eps)
    oof = np.load(os.path.join(VDIR, "probe_oof.npz"))
    ci = names.index("contact")
    sig = {"probe: LM L%d" % int(oof["layer_llm"]): oof["P_llm"][:NG, ci], "probe: vision V23": oof["P_vis"][:NG, ci]}
    picks = contact_units(names, fams, NG)
    # best unit per stream (by contact binned R2)
    best_picks = sorted(picks, key=lambda p: -p[4][0])
    for k, fn, pole, idx, r in best_picks[:4]:
        Z = np.load(os.path.join(VDIR, fn), mmap_mode="r")
        a = np.asarray(Z[:NG, 0, idx[0]] if pole == "+" else -Z[:NG, 1, idx[0]], np.float32)
        sig[f"unit {k.replace('video_', '')}#{idx[0]} (R²={r[0]:.2f})"] = a
    clr = np.array([clearance(st, e, f) for e, f in zip(gi[:, 0], gi[:, 3])])
    gt = Y[:, ci]
    # index groups by (episode, aligned frame)
    by_ep = {e: np.where(gi[:, 0] == e)[0] for e in eps}
    near = np.zeros(NG, bool)
    lagmat = {}
    for kind in ("floor", "wall"):
        rows, lags = [], []
        for e, f, k in ons:
            if k != kind:
                continue
            r = by_ep[e]
            lag = gi[r, 3] - f
            m = (lag >= LAG_BINS[0]) & (lag < LAG_BINS[-1] + 4)
            rows.append(r[m]); lags.append(lag[m])
            near[r[(lag > -48) & (lag < 32)]] = True
        lagmat[kind] = (np.concatenate(rows), np.concatenate(lags), sum(1 for o in ons if o[2] == kind))
    # clearance model fitted on non-contact groups far from any onset
    fitm = (~near) & (gt == 0)
    from analysis.vlm_census import bin_index
    edges = np.quantile(clr[fitm], np.linspace(0, 1, 21))

    def clr_model(x):
        bi = np.clip(np.searchsorted(edges, clr, side="right") - 1, 0, 19)
        m = np.array([x[fitm & (bi == b)].mean() if (fitm & (bi == b)).any() else np.nan for b in range(20)])
        return m[bi]

    def z(x):
        return (x - x[~near].mean()) / (x[~near].std() + 1e-8)

    results = {"n_onsets": {k: v[2] for k, v in lagmat.items()}, "signals": {}}
    centers = LAG_BINS + 1.5
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    for col, kind in enumerate(("floor", "wall")):
        rows, lags, n = lagmat[kind]
        bi = np.digitize(lags, LAG_BINS) - 1

        def curve(x):
            return np.array([x[rows[bi == b]].mean() if (bi == b).any() else np.nan for b in range(len(LAG_BINS))]), \
                   np.array([x[rows[bi == b]].std() / np.sqrt(max((bi == b).sum(), 1)) for b in range(len(LAG_BINS))])

        ax = axes[0, col]
        c, _ = curve(gt)
        ax.fill_between(centers, 0, c, color=FAMILIES["contact"][0], alpha=0.12, step="mid", label="true contact fraction")
        c, _ = curve(np.clip(1 - clr / 0.3, 0, 1))
        ax.plot(centers, c, color=MUTED, lw=1.2, ls="--", label="proximity (1 − clearance/0.3 m)")
        for i, (name, x) in enumerate(list(sig.items())[:2]):
            m, se = curve(x)
            ax.plot(centers, m, color=SERIES[[0, 2][i]], lw=1.6, label=name)
            ax.fill_between(centers, m - se, m + se, color=SERIES[[0, 2][i]], alpha=0.2, lw=0)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"{'AB'[col]}  {kind} onsets (n={n}): contact probes", fontsize=9)
        ax.set_ylim(-0.15, 1.1)
        if col == 0:
            ax.set_ylabel("P(contact) / fraction")
        ax.legend(fontsize=6, loc="upper left")
        ax = axes[1, col]
        for i, (name, x) in enumerate(list(sig.items())[2:]):
            m, se = curve(z(x))
            ax.plot(centers, m, color=SERIES[(i + 3) % 8], lw=1.4, label=name)
            ax.fill_between(centers, m - se, m + se, color=SERIES[(i + 3) % 8], alpha=0.2, lw=0)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"{'CD'[col]}  {kind} onsets: most contact-selective units", fontsize=9)
        ax.set_xlabel("lag from contact onset (sim frames, 20 ms)")
        if col == 0:
            ax.set_ylabel("activation (z)")
        ax.legend(fontsize=5.5, loc="upper left")
        # anticipation stats
        for name, x in sig.items():
            resid = x - clr_model(x)
            pre = rows[(lags >= -12) & (lags < 0)]
            base = rows[(lags >= -40) & (lags < -24)]
            raw_pre, raw_base = x[pre].mean(), x[base].mean()
            sd = x[~near].std() + 1e-8
            d_resid = (resid[pre].mean() - np.nanmean(resid[base])) / sd
            # bootstrap over onsets is approximated by group-level SE
            se = np.sqrt(np.nanvar(resid[pre]) / len(pre) + np.nanvar(resid[base]) / len(base)) / sd
            results["signals"].setdefault(name, {})[kind] = dict(
                pre_minus_base_raw_z=float((raw_pre - raw_base) / sd),
                pre_minus_base_after_clearance_z=float(d_resid), se=float(se),
                post_0_8_z=float((x[rows[(lags >= 0) & (lags < 8)]].mean() - raw_base) / sd))
    fig.tight_layout()
    savefig(fig, "vlm_contact_events.png")
    with open(os.path.join(VDIR, "contact_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
