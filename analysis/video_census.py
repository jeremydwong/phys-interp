"""Analysis 1: census of every hidden MLP unit (max over the 16x16 positions of frame t).

For each live unit: binned R2 (30 quantile bins) vs each of the 32 targets at frame t.
Label = best variable's family if R2 >= 0.5 ("clean"), 0.25-0.5 ("partial"), else "unexplained";
"conjunction" if best 2-D (8x8 bins) pair beats best single by > 0.15.

State control: a 2-layer MLP maps the instantaneous state (sin/cos q, qd) to every unit's activation
(fit on probe-train episodes, evaluated on test). Units whose non-posture/velocity label survives
on the residual (partial binned R2) carry information beyond the visible state.
Also checks alignment: R2 vs targets at t-1, t, t+1.

Outputs: models/video_cache/census_<tag>.npz, figures video_census*.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, INK2, MUTED, savefig, setup_style  # noqa: E402
from analysis.video_common import (CACHE, DEV, LAYERS, RESULTS, bin_indices, binned_r2_matrix,  # noqa: E402
                                   binned_r2_pairs, cached_acts, load, target_matrix)

FAM_ORDER = ["posture", "velocity", "accel", "gravity", "interaction", "contact", "applied", "energy"]
LIVE_STD = 0.02


def acts_for(tag):
    test = cached_acts(tag, "test", lambda: load(tag)[0], res=True)
    probe = cached_acts(tag, "probe", lambda: load(tag)[0], res=True)
    return test, probe


def state_features(st, E, T):
    q, qd = st["q"][E, T], st["qd"][E, T]
    return np.concatenate([np.sin(q), np.cos(q), np.sin(q.sum(-1, keepdims=True)),
                           np.cos(q.sum(-1, keepdims=True)), qd / 5.0], 1).astype(np.float32)


def state_control(Xtr, Atr, Xte, Ate, epochs=60):
    """Fit MLP state -> all unit activations; return test residuals and test R2 per unit."""
    torch.manual_seed(0)
    mu, sd = Atr.mean(0), Atr.std(0) + 1e-6
    At = torch.as_tensor((Atr - mu) / sd, device=DEV)
    Xt = torch.as_tensor(Xtr, device=DEV)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 512), torch.nn.GELU(),
                              torch.nn.Linear(512, 512), torch.nn.GELU(),
                              torch.nn.Linear(512, Atr.shape[1])).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    n = len(Xt)
    steps = epochs * (n // 512)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        idx = torch.randint(0, n, (512,), device=DEV)
        loss = ((net(Xt[idx]) - At[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        pred = net(torch.as_tensor(Xte, device=DEV)).cpu().numpy() * sd + mu
    resid = Ate - pred
    r2 = 1 - resid.var(0) / (Ate.var(0) + 1e-12)
    return resid.astype(np.float32), r2


def census(tag, do_control=True):
    Y, vs, st = target_matrix()
    test, probe = acts_for(tag)
    E, T = test["eps"], test["ts"]
    A = test["max"].astype(np.float32)
    U = A.shape[1]
    H = U // len(LAYERS)
    live = A.std(0) > LIVE_STD
    Yt = Y[E, T]
    R = binned_r2_matrix(Yt, A)                                   # [U, 32]
    R_next = binned_r2_matrix(Y[E, T + 1], A)
    R_prev = binned_r2_matrix(Y[E, T - 1], A)
    best = R.argmax(1)
    best_r2 = R.max(1)
    # conjunction: pairs among top-6 single variables
    IDX = bin_indices(Yt, 8)
    pair_r2 = np.zeros(U, np.float32)
    pair_best = np.zeros((U, 2), int)
    for u in np.where(live & (best_r2 > 0.05))[0]:
        top = np.argsort(-R[u])[:6]
        pairs = [(top[i], top[j]) for i in range(6) for j in range(i + 1, 6)]
        pr = binned_r2_pairs(IDX, A[:, u], pairs)
        k = pr.argmax()
        pair_r2[u], pair_best[u] = pr[k], pairs[k]
    conj = live & (pair_r2 - best_r2 > 0.15)
    status = np.where(~live, "dead", np.where(best_r2 >= 0.5, "clean", np.where(best_r2 >= 0.25, "partial", "unexplained")))
    fam = np.array([vs[b].family for b in best])
    out = dict(R=R, R_next=R_next, R_prev=R_prev, best=best, best_r2=best_r2, pair_r2=pair_r2,
               pair_best=pair_best, conj=conj, live=live, status=status, fam=fam, H=H)
    if do_control:
        Xtr = state_features(st, probe["eps"], probe["ts"])
        Xte = state_features(st, E, T)
        resid, r2_state = state_control(Xtr, probe["max"].astype(np.float32), Xte, A)
        Rp = binned_r2_matrix(Yt, resid)
        out.update(R_partial=Rp, r2_state=r2_state)
    np.savez(os.path.join(CACHE, f"census_{tag}.npz"), **out)
    return out


def load_census(tag):
    z = np.load(os.path.join(CACHE, f"census_{tag}.npz"))
    return {k: z[k] for k in z.files}


def layer_fractions(c):
    """-> dict layer -> {family or 'unexplained'/'partial-of-fam'}: fractions of live units."""
    H = int(c["H"])
    rows = []
    for li in range(len(LAYERS)):
        sl = slice(li * H, (li + 1) * H)
        live = c["live"][sl]
        st, fa = c["status"][sl][live], c["fam"][sl][live]
        n = max(live.sum(), 1)
        row = {"n_live": int(live.sum()), "n": H}
        for f in FAM_ORDER:
            row[f + "_clean"] = float(((st == "clean") & (fa == f)).sum() / n)
            row[f + "_partial"] = float(((st == "partial") & (fa == f)).sum() / n)
        row["unexplained"] = float((st == "unexplained").sum() / n)
        row["conj"] = float(c["conj"][sl][live].sum() / n)
        rows.append(row)
    return rows


def plot_census(cf, ci, vs):
    setup_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0), sharey=True)
    for ax, c, title in [(axes[0], cf, "Trained"), (axes[1], ci, "Random init")]:
        rows = layer_fractions(c)
        x = np.arange(len(LAYERS))
        bottom = np.zeros(len(LAYERS))
        for f in FAM_ORDER:
            col = FAMILIES[f][0]
            v = np.array([r[f + "_clean"] for r in rows])
            ax.bar(x, v, bottom=bottom, color=col, width=0.75, lw=0)
            bottom += v
            v = np.array([r[f + "_partial"] for r in rows])
            ax.bar(x, v, bottom=bottom, color=col, alpha=0.4, width=0.75, lw=0)
            bottom += v
        v = np.array([r["unexplained"] for r in rows])
        ax.bar(x, v, bottom=bottom, color=FAMILIES["none"][0], alpha=0.5, width=0.75, lw=0)
        for i, r in enumerate(rows):
            ax.text(i, 1.01, f"{r['n_live']}", ha="center", va="bottom", fontsize=6, color=MUTED)
        ax.set_xticks(x, LAYERS, rotation=0, fontsize=7)
        ax.set_title(title)
        ax.set_ylim(0, 1.08)
        ax.axvline(3.5, color=INK2, lw=0.8, ls=":")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("fraction of live units")
    axes[0].text(1.5, -0.2, "per-frame encoder", ha="center", fontsize=7, color=INK2, transform=axes[0].get_xaxis_transform())
    axes[0].text(5.5, -0.2, "space-time stage", ha="center", fontsize=7, color=INK2, transform=axes[0].get_xaxis_transform())
    handles = [Patch(color=FAMILIES[f][0], label=f) for f in FAM_ORDER if any(
        (c["fam"][c["live"] & (c["status"] != "unexplained")] == f).any() for c in (cf, ci))]
    handles += [Patch(color="#999", alpha=0.4, label="light = partial (0.25–0.5)"),
                Patch(color=FAMILIES["none"][0], alpha=0.5, label="unexplained")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12), fontsize=7)
    fig.tight_layout()
    savefig(fig, "video_census.png")


def plot_control(cf, vs):
    """Best R2 vs best partial R2 (after removing the state-explained part), per family."""
    import matplotlib.pyplot as plt
    setup_style()
    live = cf["live"] & (cf["best_r2"] >= 0.25)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9))
    ax = axes[0]
    for f in FAM_ORDER:
        sel = live & (cf["fam"] == f)
        if sel.sum() == 0:
            continue
        ax.scatter(cf["best_r2"][sel], cf["R_partial"][sel].max(1), s=6, color=FAMILIES[f][0], label=f, alpha=0.7, lw=0)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("best binned R² (raw activation)")
    ax.set_ylabel("best partial R² (state removed)")
    ax.set_title("What survives removing the state")
    ax.legend(fontsize=6, markerscale=2, loc="upper left")
    ax.set_xlim(0.2, 1); ax.set_ylim(0, 1)
    ax = axes[1]
    ax.hist(cf["r2_state"][cf["live"]], bins=40, color=FAMILIES["posture"][0], alpha=0.8)
    ax.set_xlabel("R² of unit predicted from state (q, q̇) — held out")
    ax.set_ylabel("units")
    ax.set_title("How much of each unit is 'just state'")
    fig.tight_layout()
    savefig(fig, "video_census_control.png")


def main():
    Y, vs, st = target_matrix()
    cf = census("final")
    ci = census("init", do_control=False)
    plot_census(cf, ci, vs)
    plot_control(cf, vs)
    names = [v.name for v in vs]
    summary = {}
    for tag, c in [("final", cf), ("init", ci)]:
        live = c["live"]
        s = {"n_units": int(len(live)), "n_live": int(live.sum()),
             "clean": float((c["status"][live] == "clean").mean()),
             "partial": float((c["status"][live] == "partial").mean()),
             "unexplained": float((c["status"][live] == "unexplained").mean()),
             "conjunction": float(c["conj"][live].mean()),
             "per_layer": layer_fractions(c),
             "family_counts_clean": {f: int(((c["status"] == "clean") & (c["fam"] == f)).sum()) for f in FAM_ORDER},
             "family_counts_partial": {f: int(((c["status"] == "partial") & (c["fam"] == f)).sum()) for f in FAM_ORDER}}
        # alignment: mean best R2 using targets at t-1, t, t+1 for labelled live units
        lab = live & (c["best_r2"] >= 0.25)
        s["align_mean_bestR2"] = {"t-1": float(c["R_prev"][lab].max(1).mean()), "t": float(c["R"][lab].max(1).mean()),
                                  "t+1": float(c["R_next"][lab].max(1).mean())}
        for v in ["qdd1", "qdd2", "contact", "c_floor", "qd1", "qd2"]:
            j = names.index(v)
            s.setdefault("align_maxR2_by_var", {})[v] = {
                "t-1": float(c["R_prev"][live, j].max()), "t": float(c["R"][live, j].max()), "t+1": float(c["R_next"][live, j].max())}
        if "R_partial" in c:
            nonpv = lab & ~np.isin(c["fam"], ["posture", "velocity"])
            s["state_control"] = {
                "median_r2_state_live": float(np.median(c["r2_state"][live])),
                "n_nonposture_labelled": int(nonpv.sum()),
                "n_nonposture_surviving_partial>=0.25": int((c["R_partial"][nonpv].max(1) >= 0.25).sum()),
                "max_partial_by_var": {names[j]: float(c["R_partial"][live, j].max()) for j in range(len(names))},
            }
        summary[tag] = s
        print(tag, {k: v for k, v in s.items() if k not in ("per_layer",)})
    json.dump(summary, open(os.path.join(RESULTS, "video_census.json"), "w"), indent=1)


if __name__ == "__main__":
    main()


def rank_units(c, vs, var_names, k, family=None, layer_prefix=None):
    """Top-k live units by max R2 over var_names, preferring units whose best variable is in
    `family` (if given); falls back to the remaining units by the same score."""
    from analysis.video_common import LAYERS
    names = [v.name for v in vs]
    H = int(c["H"])
    score = c["R"][:, [names.index(v) for v in var_names]].max(1) * c["live"]
    ok = np.ones(len(score), bool)
    if layer_prefix:
        ok = np.array([LAYERS[u // H].startswith(layer_prefix) for u in range(len(score))])
    order = [u for u in np.argsort(-score) if ok[u]]
    pref = [u for u in order if family is None or c["fam"][u] == family]
    rest = [u for u in order if u not in set(pref)]
    return [int(u) for u in (pref + rest)[:k]]
