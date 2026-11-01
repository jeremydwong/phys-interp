"""vis_ census: ground-truth census, strong-activation purity, dial vs detector, raw vs percentile.

For every dataset x step (old / recipe / arch [/ size sweep]) x seed, trained and random-init:
  per depth group (early / mid / late) and pole (+ / -):
    census on pre-activation pole scores (clean / partial / unexplained / conjunction), and on the
    "old tools" view (post-ReLU/GELU positive pole only);
    one-main-job purity with the percentile strong set (top 1%) and the raw-shape strong set;
    dial vs detector for clean poles; census on rank-transformed (percentile) scores.
Writes data/vis_cache/<model>_census.pkl (per-unit arrays) and reports/results/vis_census.json (summaries).
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import clean_tools as ct  # noqa: E402
from analysis.vis_common import DATASETS, CACHE, PAIR_FACTORS, RESULTS, available, cached_scores, data, depth_groups  # noqa: E402


def gelu_np(x):
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def analyse(dname, tag, seed, init):
    D = data(dname)
    fac, kinds = D["fte"], D["kinds"]
    sc = cached_scores(dname, tag, seed, init)
    layers = list(sc)
    groups = depth_groups(layers)
    act = (lambda x: np.maximum(x, 0)) if tag.startswith(("old", "recipe")) else gelu_np
    out, summ = {}, {}
    for g, layer in groups.items():
        pos, neg = sc[layer]
        res = {}
        for pole, S in (("pos", pos), ("neg", neg)):
            c = ct.census(S, fac, kinds, PAIR_FACTORS[dname])
            pp = ct.purity(S, fac, kinds, mode="pct")
            pr = ct.purity(S, fac, kinds, mode="raw")
            dd = np.zeros(S.shape[1], bool)
            wr = np.full(S.shape[1], np.nan)
            above = np.zeros(S.shape[1])
            for u in np.nonzero(c["label"] == "clean")[0]:
                k = c["names"][c["best"][u]]
                dd[u], wr[u], _, above[u] = ct.dial_detector(S[:, u], fac[k], kinds[k], c["best_value"][u])
            res[pole] = dict(census=c, pur_pct=pp, pur_raw=pr, dial=dd, weak_r2=wr, above_half=above)
        # old tools: post-activation positive pole, percentile strong set
        Sold = act(pos)
        res["old"] = dict(census=ct.census(Sold, fac, kinds, PAIR_FACTORS[dname]),
                          pur_pct=ct.purity(Sold, fac, kinds, mode="pct"))
        if g == "late":
            res["pos_rank"] = dict(census=ct.census(ct.rank_transform(pos), fac, kinds, []))
        out[g] = dict(layer=layer, **res)
        s = {}
        for pole in ("pos", "neg"):
            r = res[pole]
            cl = r["census"]["label"] == "clean"
            s[pole] = dict(**ct.census_summary(r["census"]), one_job_pct=float(r["pur_pct"]["one_job"].mean()),
                           one_job_raw=float(r["pur_raw"]["one_job"].mean()),
                           purity_pct=float(r["pur_pct"]["purity"].mean()), purity_raw=float(r["pur_raw"]["purity"].mean()),
                           dial_frac_of_clean=float(r["dial"][cl].mean()) if cl.any() else None,
                           n_clean=int(cl.sum()))
        s["both"] = {k: (s["pos"][k] + s["neg"][k]) / 2 for k in ("clean", "partial", "unexplained", "conj", "one_job_pct", "one_job_raw")}
        s["old_tools"] = dict(**ct.census_summary(res["old"]["census"]), one_job_pct=float(res["old"]["pur_pct"]["one_job"].mean()))
        if g == "late":
            s["pos_rank"] = ct.census_summary(res["pos_rank"]["census"])
        summ[g] = dict(layer=layer, **s)
    name = f"{dname}_{tag}_s{seed}{'_init' if init else ''}"
    pickle.dump(out, open(os.path.join(CACHE, name + "_census.pkl"), "wb"))
    return name, summ


def main(tags=("old", "recipe", "arch", "archs", "archl")):
    fn = os.path.join(RESULTS, "vis_census.json")
    allres = json.load(open(fn)) if os.path.exists(fn) else {}
    for dname in DATASETS:
        for tag in tags:
            for seed in available(dname, tag):
                for init in (False, True):
                    name, s = analyse(dname, tag, seed, init)
                    allres[name] = s
                    L = s["late"]
                    print(name, "late clean+%.2f -%.2f  onejob(raw)+%.2f -%.2f  old-tools clean %.2f" % (
                        L["pos"]["clean"], L["neg"]["clean"], L["pos"]["one_job_raw"], L["neg"]["one_job_raw"],
                        L["old_tools"]["clean"]), flush=True)
                    json.dump(allres, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("old", "recipe", "arch", "archs", "archl"))
