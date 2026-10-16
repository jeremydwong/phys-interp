"""Analysis 1: how many hidden units have one job?

For every live unit (post-ReLU activation > 0 on > 1% of held-out frames) we compute the binned R²
(30 quantile bins, piecewise-constant fit = analysis.common.binned_r2) against each of the 32 physical
targets. Label: best variable if R² >= 0.5 (clean), 0.25-0.5 (partial), else unexplained.
Conjunction: the best 2-D binned fit (10x10 quantile cells) over pairs drawn from the unit's top-6
variables beats the best 1-D fit by > 0.15.
Same procedure for the randomly initialized network (checkpoint at step 0).
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, savefig, setup_style  # noqa: E402
from analysis.state_common import (CACHE, CLEAN, CONJ_GAIN, LIVE_FRAC, PARTIAL, _ss_between, bin_index,  # noqa
                                   binned_r2_matrix, data, load_model, run, target_aliases, update_results)

FAM_ORDER = ["posture", "velocity", "accel", "gravity", "interaction", "contact", "applied", "energy"]
DERIVED = ["accel", "gravity", "interaction", "contact", "energy"]   # not trivially available from the input


def census(H, Y, names, fams, aliases, n_top=6, pair_bins=10):
    """H [L,N,U] activations -> dict of per-unit arrays [L,U]."""
    L, N, U = H.shape
    out = {k: [] for k in ["live", "R", "best", "bestr", "pair", "pair_r2", "alias"]}
    for l in range(L):
        A = H[l]
        live = (A > 0).mean(0) > LIVE_FRAC
        R = np.full((U, Y.shape[1]), np.nan)
        R[live] = binned_r2_matrix(A[:, live], Y)
        best = np.where(live, np.nan_to_num(R, nan=-1).argmax(1), -1)
        bestr = np.where(live, np.nan_to_num(R, nan=-1).max(1), np.nan)
        # conjunction: 2-D binned fits on pairs of top variables, grouped by pair for speed
        pr2 = np.full(U, np.nan)
        pbest = np.full((U, 2), -1)
        need = {}
        for u in np.where(live)[0]:
            top = np.argsort(-R[u])[:n_top]
            for a, b in itertools.combinations(sorted(top), 2):
                need.setdefault((a, b), []).append(u)
        for (a, b), us in need.items():
            idx = bin_index(Y[:, a], pair_bins) * pair_bins + bin_index(Y[:, b], pair_bins)
            Au = A[:, us].astype(np.float64)
            r = _ss_between(Au, idx, pair_bins ** 2, Au.mean(0)) / (Au.var(0) + 1e-12)
            for u, rv in zip(us, r):
                if np.isnan(pr2[u]) or rv > pr2[u]:
                    pr2[u], pbest[u] = rv, (a, b)
        # alias flag: best variable has a near-identical twin in another family that fits equally well
        al = np.zeros(U, bool)
        for u in np.where(live)[0]:
            for tw in aliases[names[best[u]]]:
                j = names.index(tw)
                if fams[j] != fams[best[u]] and R[u, j] >= R[u, best[u]] - 1e-3:
                    al[u] = True
        for k, v in zip(out, [live, R, best, bestr, pbest, pr2, al]):
            out[k].append(v)
    out = {k: np.stack(v) for k, v in out.items()}
    br = out["bestr"]
    out["cat"] = np.where(~out["live"], "dead", np.where(br >= CLEAN, "clean",
                          np.where(br >= PARTIAL, "partial", "unexplained")))
    out["conj"] = out["live"] & (out["pair_r2"] - np.nan_to_num(br) > CONJ_GAIN)
    out["fam"] = np.where(out["live"], np.array(fams + ["none"])[out["best"]], "none")
    return out


def summarize(c, names):
    L = c["live"].shape[0]
    s = {"per_layer": []}
    for l in range(L):
        lv = c["live"][l]
        n = lv.sum()
        d = dict(layer=l + 1, n_units=int(lv.size), n_live=int(n),
                 clean=float((c["cat"][l] == "clean").sum() / n), partial=float((c["cat"][l] == "partial").sum() / n),
                 unexplained=float((c["cat"][l] == "unexplained").sum() / n),
                 conjunction=float(c["conj"][l][lv].mean()),
                 conjunction_among_unexplained_or_partial=float(
                     c["conj"][l][lv & (c["cat"][l] != "clean")].mean()),
                 aliased_best=float(c["alias"][l][lv].mean()),
                 median_best_r2=float(np.median(c["bestr"][l][lv])),
                 clean_derived=float(((c["cat"][l] == "clean") & np.isin(c["fam"][l], DERIVED)).sum() / n))
        d["clean_by_family"] = {f: float(((c["cat"][l] == "clean") & (c["fam"][l] == f)).sum() / n)
                                for f in FAM_ORDER}
        d["partial_by_family"] = {f: float(((c["cat"][l] == "partial") & (c["fam"][l] == f)).sum() / n)
                                  for f in FAM_ORDER}
        bv = c["best"][l][lv & (c["cat"][l] == "clean")]
        d["clean_top_vars"] = {names[i]: int((bv == i).sum()) for i in np.argsort(-np.bincount(bv, minlength=len(names)))[:8]
                               if (bv == i).sum() > 0}
        s["per_layer"].append(d)
    lv = c["live"]
    s["all"] = dict(n_live=int(lv.sum()), clean=float((c["cat"] == "clean").sum() / lv.sum()),
                    partial=float((c["cat"] == "partial").sum() / lv.sum()),
                    unexplained=float((c["cat"] == "unexplained").sum() / lv.sum()),
                    conjunction=float(c["conj"][lv].mean()),
                    clean_or_conjunction=float(((c["cat"] == "clean") | c["conj"])[lv].mean()),
                    aliased_best=float(c["alias"][lv].mean()),
                    clean_derived=float(((c["cat"] == "clean") & np.isin(c["fam"], DERIVED)).sum() / lv.sum()))
    return s


def figure(cs, sums):
    setup_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(7, 3.0))
    L = len(sums["trained"]["per_layer"])
    width = 0.38
    for k, (tag, off) in enumerate((("trained", -0.21), ("random", 0.21))):
        for l in range(L):
            d = sums[tag]["per_layer"][l]
            x = l + off
            bottom = 0
            for f in FAM_ORDER:
                v = d["clean_by_family"][f]
                ax.bar(x, v, width, bottom=bottom, color=FAMILIES[f][0], lw=0)
                bottom += v
            for f in FAM_ORDER:
                v = d["partial_by_family"][f]
                ax.bar(x, v, width, bottom=bottom, color=FAMILIES[f][0], alpha=0.35, lw=0)
                bottom += v
            ax.bar(x, d["unexplained"], width, bottom=bottom, color="white", edgecolor="#b5b3ab", hatch="////", lw=0)
            # alias hatch over aliased fraction is reported in text; mark clean boundary
            ax.plot([x - width / 2, x + width / 2], [d["clean"]] * 2, color="#0b0b0b", lw=0.8)
            ax.text(x, 1.015, f"{d['clean']*100:.0f}%", ha="center", va="bottom", fontsize=7, color="#52514e")
            ax.text(x, -0.03, "trained" if tag == "trained" else "rand.", ha="center", va="top", fontsize=7,
                    color="#52514e", rotation=0)
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"\n\nblock {l + 1}  (live {sums['trained']['per_layer'][l]['n_live']} / "
                        f"{sums['random']['per_layer'][l]['n_live']})" for l in range(L)], fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("live units (%)")
    ax.grid(axis="x", visible=False)
    handles = [Patch(color=FAMILIES[f][0], label=f) for f in FAM_ORDER]
    handles += [Patch(color="#898781", alpha=0.3, label="partial (0.25–0.5),\nlight tint of family"),
                Patch(facecolor="white", edgecolor="#b5b3ab", hatch="////", label="unexplained (< 0.25)")]
    ax.legend(handles=handles, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7, handlelength=1.2)
    ax.set_title("Best single physical variable per live unit (solid: clean, R² ≥ 0.5)", fontsize=8.5)
    return savefig(fig, "state_census.png")


def main():
    d = data()
    Y, names = d["Y"], d["names"]
    fams = [v.family for v in d["vars"]]
    aliases = target_aliases(Y, names)
    al_pairs = {k: v for k, v in aliases.items() if v}
    print("aliased targets:", al_pairs)
    os.makedirs(CACHE, exist_ok=True)
    cs, sums = {}, {}
    for tag, ck in (("trained", "final"), ("random", "step000000")):
        model, _ = load_model(ck)
        H = run(model, d["X"])["h"]
        c = census(H, Y, names, fams, aliases)
        np.savez(os.path.join(CACHE, f"census_{tag}.npz"), **{k: v for k, v in c.items()})
        cs[tag], sums[tag] = c, summarize(c, names)
        print(tag, {k: round(v, 3) if isinstance(v, float) else v for k, v in sums[tag]["all"].items()})
        for pl in sums[tag]["per_layer"]:
            print("  L", pl["layer"], "live", pl["n_live"], "clean %.3f (derived %.3f) partial %.3f unexpl %.3f conj %.3f alias %.3f"
                  % (pl["clean"], pl["clean_derived"], pl["partial"], pl["unexplained"], pl["conjunction"], pl["aliased_best"]),
                  pl["clean_top_vars"])
    figure(cs, sums)
    update_results("census", dict(aliased_targets=al_pairs, **sums,
                                  thresholds=dict(clean=CLEAN, partial=PARTIAL, conj_gain=CONJ_GAIN, live=LIVE_FRAC)))


if __name__ == "__main__":
    main()
