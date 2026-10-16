"""Analysis 6: contact units, event-aligned.

Contact onsets (held-out episodes): first frame with contact after >= 4 contact-free frames. Classified by
surface (floor / wall, from the logged per-geom normal force) and impact speed (hand speed normal to the
surface at the last free frame, split at the per-surface median). The upper arm cannot reach either
obstacle, so every contact is a forearm contact and no upper-arm/forearm split is possible.
Near misses: frames where the minimum capsule clearance to floor or wall has a local minimum in (0, 2 cm]
and no contact occurs within +-6 frames. Activations are aligned to the onset (first contact frame) or to
the minimum-clearance frame. The model at frame t sees the state at t and predicts t -> t+1, so a unit
that 'predicts' the bounce should fire at t = -1 (the last contact-free frame).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, SHOULDER, ELBOW, savefig, setup_style  # noqa: E402
from analysis.state_common import CACHE, data, load_census, load_model, obstacle_gaps, pick_units, run, update_results  # noqa

W = 8


def events(d):
    st = d["st"]
    E0 = 2000
    T = d["T"]
    cf = st["contact_flag"][E0:, :T] > 0
    ev = []
    for e in range(cf.shape[0]):
        for t in range(W, T - W):
            if cf[e, t] and not cf[e, t - 4:t].any():
                fl, wa = st["contact_floor"][E0 + e, t], st["contact_wall"][E0 + e, t]
                fo = st["contact_on_fore"][E0 + e, t]
                up = st["contact_force"][E0 + e, t] - fo
                surf = "floor" if fl >= wa else "wall"
                v = st["ee_vel"][E0 + e, t - 1]
                ev.append(dict(e=e, t=t, surface=surf, segment="forearm" if fo >= up else "upper arm",
                               speed=float(-v[1] if surf == "floor" else v[0])))
    for surf in ("floor", "wall"):
        sp = np.array([x["speed"] for x in ev if x["surface"] == surf])
        med = float(np.median(sp)) if len(sp) else 0.0
        for x in ev:
            if x["surface"] == surf:
                x["speed_cat"] = "fast" if x["speed"] > med else "slow"
                x["speed_median"] = med
    gaps = obstacle_gaps(st["q"][E0:, :T])
    gmin = np.stack(list(gaps.values()), -1)
    which = gmin.argmin(-1)
    gmin = gmin.min(-1)
    nm = []
    for e in range(cf.shape[0]):
        g = gmin[e]
        for t in range(W, T - W):
            if 0 < g[t] <= 0.02 and g[t] <= g[t - 1] and g[t] <= g[t + 1] and not cf[e, t - 6:t + 7].any():
                k = list(gaps)[which[e, t]]
                nm.append(dict(e=e, t=t, surface=k.split("_")[0], segment="forearm" if "fore" in k else "upper arm"))
    return ev, nm


def main():
    d = data()
    names = d["names"]
    T = d["T"]
    cz = load_census()
    model, _ = load_model("final")
    H = run(model, d["X"])["h"]
    ev, nm = events(d)
    cats = [("floor", "slow"), ("floor", "fast"), ("wall", "slow"), ("wall", "fast")]
    seg_counts = {g: sum(1 for x in ev if x["segment"] == g) for g in ("upper arm", "forearm")}
    counts = {f"{s}/{g}": sum(1 for x in ev if x["surface"] == s and x["speed_cat"] == g) for s, g in cats}
    counts_nm = {s: sum(1 for x in nm if x["surface"] == s) for s in ("floor", "wall")}
    med = {s: next(x["speed_median"] for x in ev if x["surface"] == s) for s in ("floor", "wall")}
    print("onsets", counts, "near misses", counts_nm, "segments", seg_counts)
    # units: best contact unit for several contact variables
    CONTACT = ["contact", "c_floor", "c_wall", "c_fore", "ctq1", "ctq2"]
    units, excl = [], set()
    for var in ["c_floor", "c_wall", "ctq2"]:
        u = pick_units(cz, names, var, k=1, best_in=CONTACT, exclude=excl) or pick_units(cz, names, var, k=1, exclude=excl)
        excl.add(u[0][:2]); units.append((var,) + u[0])
    # functional selection: 'impact units' separate the pre-impact frame (the step that contains the collision)
    # from near-miss frames (d', pooled sd), across all live units
    pre_idx = np.array([x["e"] * T + x["t"] - 1 for x in ev])
    nm_idx = np.array([x["e"] * T + x["t"] for x in nm])
    dprime = []
    for l in range(H.shape[0]):
        a_pre, a_nm = H[l][pre_idx], H[l][nm_idx]
        dp = (a_pre.mean(0) - a_nm.mean(0)) / np.sqrt(0.5 * (a_pre.var(0) + a_nm.var(0)) + 1e-6)
        dp[~cz["live"][l]] = 0
        dprime.append(dp)
    dprime = np.stack(dprime)
    order = np.dstack(np.unravel_index(np.argsort(-dprime.ravel()), dprime.shape))[0]
    n_imp = 0
    impact_units = []
    for l, k in order:
        if (int(l), int(k)) in excl:
            continue
        impact_units.append((int(l), int(k), float(dprime[l, k])))
        units.append(("impact d′", int(l), int(k), float(dprime[l, k])))
        n_imp += 1
        if n_imp == 3:
            break
    np.save(os.path.join(CACHE, "impact_dprime.npy"), dprime)
    lags = np.arange(-W, W + 1)

    def psth(a, evs):
        if not evs:
            return None, None
        M = np.stack([a.reshape(-1, T)[x["e"], x["t"] - W:x["t"] + W + 1] for x in evs])
        return M.mean(0), M.std(0) / np.sqrt(len(M))

    setup_style()
    import matplotlib.pyplot as plt
    st = d["st"]
    fig, axes = plt.subplots(len(units) + 1, 4, figsize=(7, 1.05 * (len(units) + 1) + 0.3), sharex=True)
    out = dict(impact_units=[dict(unit=f"L{l + 1}/{k}", dprime=v, best=names[int(cz["best"][l][k])],
                                  best_r2=float(cz["bestr"][l][k])) for l, k, v in impact_units],
               n_units_dprime_gt1=int((dprime > 1).sum()), onset_counts=counts, near_miss_counts=counts_nm, onset_segment_counts=seg_counts,
               median_impact_speed=med, units={})
    force = np.log1p(st["contact_force"][2000:, :T]).reshape(-1)
    for c, (s, g) in enumerate(cats):
        evs = [x for x in ev if x["surface"] == s and x["speed_cat"] == g]
        m, se = psth(force, evs)
        ax = axes[0, c]
        if m is not None:
            ax.plot(lags, m, color="#0b0b0b", lw=1.3)
            ax.fill_between(lags, m - se, m + se, color="#0b0b0b", alpha=0.15, lw=0)
        ax.set_title(f"{s}, {g} ({'>' if g == 'fast' else '≤'}{med[s]:.1f} m/s)\n{counts[f'{s}/{g}']} onsets · "
                     f"{counts_nm[s]} near-misses", fontsize=6.6)
        if c == 0:
            ax.set_ylabel("log(1+F)\ntrue force", fontsize=6.5)
    for r, (var, l, k, r2v) in enumerate(units):
        a = H[l][:, k]
        base = a[(st["contact_flag"][2000:, :T].reshape(-1) == 0)].mean()
        res_u = {}
        for c, (s, g) in enumerate(cats):
            ax = axes[r + 1, c]
            evs = [x for x in ev if x["surface"] == s and x["speed_cat"] == g]
            nms = [x for x in nm if x["surface"] == s]
            m, se = psth(a, evs)
            if m is not None:
                ax.plot(lags, m, color=FAMILIES["contact"][0], lw=1.3)
                ax.fill_between(lags, m - se, m + se, color=FAMILIES["contact"][0], alpha=0.2, lw=0)
            mn, sen = psth(a, nms)
            if mn is not None:
                ax.plot(lags, mn, color="#52514e", lw=1.0, ls="--")
                ax.fill_between(lags, mn - sen, mn + sen, color="#52514e", alpha=0.12, lw=0)
            ax.axhline(base, color="#898781", lw=0.5, ls=":")
            res_u[f"{s}/{g}"] = dict(onset_curve=None if m is None else m, near_miss_curve=None if mn is None else mn,
                                     at_minus1=None if m is None else float(m[W - 1]),
                                     at_0=None if m is None else float(m[W]),
                                     near_miss_at_0=None if mn is None else float(mn[W]))
            if c == 0:
                ax.set_ylabel(f"L{l + 1}/{k}\n" + (f"{var} R²={r2v:.2f}" if var != "impact d′" else f"impact d′={r2v:.1f}"),
                              fontsize=6.3)
        out["units"][f"L{l + 1}/{k}"] = dict(var=var, r2=r2v, baseline_noncontact=float(base), psth=res_u)
    for ax in axes.ravel():
        ax.axvline(0, color="#e34948", lw=0.5, alpha=0.6)
        ax.axvline(-1, color="#898781", lw=0.4, alpha=0.5, ls=":")
        ax.tick_params(labelsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("frames from onset (20 ms)", fontsize=6.5)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=FAMILIES["contact"][0], label="aligned to contact onset (mean ± s.e.m.)"),
                        Line2D([], [], color="#52514e", ls="--", label="aligned to near miss (< 2 cm clearance)"),
                        Line2D([], [], color="#898781", ls=":", lw=0.8, label="mean on contact-free frames")],
               loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(hspace=0.25, wspace=0.28, bottom=0.07)
    savefig(fig, "state_contact_psth.png")
    for k, v in out["units"].items():
        print(k, v["var"], round(v["r2"], 2), "base", round(v["baseline_noncontact"], 2),
              {c: (None if x["at_0"] is None else round(x["at_minus1"], 2), None if x["at_0"] is None else round(x["at_0"], 2),
                   None if x["near_miss_at_0"] is None else round(x["near_miss_at_0"], 2)) for c, x in v["psth"].items()})
    update_results("contact", out)


if __name__ == "__main__":
    main()
