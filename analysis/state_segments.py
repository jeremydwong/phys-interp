"""Analysis 5: shoulder- vs elbow-specific torque units.

For every live unit whose best variable (census) is a gravity, interaction or contact quantity with
R² >= 0.25, we compute the selectivity index SI = (R²_shoulder - R²_elbow) / (R²_shoulder + R²_elbow)
between the shoulder and elbow versions of the same quantity (binned R²). For contact we also compare
force on the upper arm vs force on the forearm. Reference: the binned R² between the two target versions
themselves bounds how much an ideal single-segment unit would 'leak' onto the other segment.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, binned_r2, savefig, setup_style  # noqa: E402
from analysis.state_common import CACHE, binned_r2_matrix, data, load_model, run, update_results  # noqa: E402

PAIRS = {"gravity": ("grav1", "grav2"), "interaction": ("int1", "int2"), "coriolis": ("cor1", "cor2"),
         "contact torque": ("ctq1", "ctq2")}
# note: the upper arm can never touch floor or wall (reach 0.545 m < wall 0.68 m; floor 0.86 m below the
# shoulder), so every contact is on the forearm and an upper-arm/forearm contact split does not exist.
FAMILY_OF = {"gravity": "gravity", "interaction": "interaction", "coriolis": "interaction",
             "contact torque": "contact", "contact force": "contact"}


def main():
    d = data()
    S = d["S"]
    names = list(d["names"])
    Y = d["Y"]
    c_upper = np.log1p(np.clip(S["contact_force"] - S["contact_on_fore"], 0, None))
    Yx = np.c_[Y, c_upper]
    names_x = names + ["c_upper"]
    cz = np.load(os.path.join(CACHE, "census_trained.npz"), allow_pickle=True)
    model, _ = load_model("final")
    H = run(model, d["X"])["h"]
    out, sis = {}, {}
    for q, (a, b) in PAIRS.items():
        ia, ib = names_x.index(a), names_x.index(b)
        ref = (binned_r2(Yx[:, ia], Yx[:, ib]), binned_r2(Yx[:, ib], Yx[:, ia]))
        si_all, units = [], []
        for l in range(H.shape[0]):
            fam = cz["fam"][l].copy()
            fam[cz["best"][l] == names.index("cos_q12")] = "gravity"   # cos(q1+q2) is identical to elbow gravity
            fam_ok = (fam == FAMILY_OF[q]) & (cz["bestr"][l] >= 0.25) & cz["live"][l]
            if q == "interaction":
                fam_ok &= np.isin(cz["best"][l], [names.index("int1"), names.index("int2")])
            if q == "coriolis":
                fam_ok &= np.isin(cz["best"][l], [names.index("cor1"), names.index("cor2")])
            if q == "contact torque":
                fam_ok &= np.isin(cz["best"][l], [names.index("ctq1"), names.index("ctq2")])
            us = np.where(fam_ok)[0]
            if len(us) == 0:
                continue
            R = binned_r2_matrix(H[l][:, us], Yx[:, [ia, ib]])
            si = (R[:, 0] - R[:, 1]) / (R[:, 0] + R[:, 1] + 1e-9)
            si_all.append(si)
            units += [(l, int(u), float(r0), float(r1), float(s)) for u, (r0, r1), s in zip(us, R, si)]
        si_all = np.concatenate(si_all) if si_all else np.zeros(0)
        sis[q] = si_all
        units.sort(key=lambda z: -abs(z[4]))
        out[q] = dict(n_units=int(len(si_all)), frac_shoulder_specific=float((si_all > 0.5).mean()) if len(si_all) else None,
                      frac_elbow_specific=float((si_all < -0.5).mean()) if len(si_all) else None,
                      frac_mixed=float((np.abs(si_all) <= 0.5).mean()) if len(si_all) else None,
                      target_cross_r2=[float(ref[0]), float(ref[1])],
                      top_shoulder=[u for u in sorted(units, key=lambda z: -z[4])[:5]],
                      top_elbow=[u for u in sorted(units, key=lambda z: z[4])[:5]])
        print(q, {k: v for k, v in out[q].items() if not k.startswith("top")})

    setup_style()
    import matplotlib.pyplot as plt
    qs = [q for q in PAIRS if len(sis[q]) >= 5]
    fig, axes = plt.subplots(1, len(qs), figsize=(1.75 * len(qs) + 0.3, 1.9), sharey=False)
    axes = np.atleast_1d(axes)
    bins = np.linspace(-1, 1, 21)
    for ax, q in zip(axes, qs):
        ax.hist(sis[q], bins=bins, color=FAMILIES[FAMILY_OF[q]][0], lw=0)
        ax.axvline(0, color="#898781", lw=0.6)
        ax.set_title(f"{q if q != 'coriolis' else 'Coriolis/centrip.'}  (n={len(sis[q])})", fontsize=8)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["elbow", "0", "shoulder"], fontsize=7)
        ax.set_xlabel("selectivity index", fontsize=7.5)
    axes[0].set_ylabel("units")
    fig.subplots_adjust(wspace=0.35)
    savefig(fig, "state_segment_selectivity.png")
    update_results("segments", out)


if __name__ == "__main__":
    main()
