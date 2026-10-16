"""Analysis 7: causal tests. Mean-ablate the top contact units (and, separately, the top
acceleration units) and measure how the predicted next frames change.

Ablation: replace the unit's hidden activation (all positions, all 4 frames) with its dataset mean.
Controls: the same number of random live units from the same layers (5 draws).
Metrics, on contact-onset windows (contact begins between t and t+1) and free-motion windows:
  * MSE on moving-arm pixels for t+1 and t+2
  * "counterfactual score" on pixels where the true and no-obstacle frames differ (|diff| > 0.1):
    s = MSE(pred, true) / (MSE(pred, true) + MSE(pred, counterfactual)); 0 = like reality, 1 = like no obstacles
  * distance to a counterfactual frame where the same state evolves for one/two frames with the
    obstacles removed (MuJoCo re-simulation from the logged q, qd with the logged motor torque),
    rendered with obstacles visible -- i.e. "the arm continues into the floor".
Figure: video_causal.png
"""
from __future__ import annotations

import json
import os
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, INK2, MUTED, SERIES, savefig, setup_style  # noqa: E402
from analysis.video_census import load_census, rank_units  # noqa: E402
from analysis.video_common import (DEV, LAYERS, RESULTS, TEST, cached_acts, future, load,  # noqa: E402
                                   target_matrix, windows)
from analysis.video_viz import arm_mask, clean  # noqa: E402
from models.video_train import change_mask  # noqa: E402
from sim.arm import RECORD_EVERY, ArmRenderer, build  # noqa: E402

K = 32


def obstacle_region():
    """Boolean [64,64] mask of floor + wall pixels (from an empty-arm render)."""
    r = ArmRenderer(64)
    img = r(np.array([[np.pi / 2, 0.0]]))[0].astype(np.float32) / 255   # arm pointing straight up
    grey = (img.max(-1) - img.min(-1) < 0.08) & (img.mean(-1) < 0.8)
    return grey


def counterfactual(st, e, t, n=2):
    """Re-simulate from state at t without obstacles for n frames, holding the logged motor torque."""
    m = build(with_obstacles=False)
    d = mujoco.MjData(m)
    d.qpos[:] = st["q"][e, t]; d.qvel[:] = st["qd"][e, t]
    qs = []
    for k in range(1, n + 1):
        tau = st["tau_act"][e, t + k]
        for _ in range(RECORD_EVERY):
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        qs.append(d.qpos.copy())
    return np.array(qs)


def make_edit(H, chans_by_layer, means):
    def fn(name, h):
        ch = chans_by_layer.get(name)
        if not ch:
            return h
        h = h.clone()
        idx = torch.as_tensor(ch, device=h.device)
        h[..., idx] = torch.as_tensor(means[name][ch], device=h.device, dtype=h.dtype)
        return h
    return fn


@torch.no_grad()
def predict(model, E, T, units, H, means, bs=64):
    by = {}
    for u in units:
        by.setdefault(LAYERS[u // H], []).append(u % H)
    model.edits = {l: make_edit(H, by, means) for l in by}
    out = []
    for i in range(0, len(E), bs):
        out.append(model(windows(E[i:i + bs], T[i:i + bs])).clamp(0, 1).cpu())
    model.edits = {}
    return torch.cat(out).numpy()          # [N, 2, 3, 64, 64]


def main():
    setup_style()
    Y, vs, st = target_matrix()
    names = [v.name for v in vs]
    c = load_census("final")
    H = int(c["H"])
    acts = cached_acts("final", "test", lambda: load("final")[0], res=True)
    means = {l: acts["mean"][:, li * H:(li + 1) * H].astype(np.float32).mean(0) for li, l in enumerate(LAYERS)}
    model, _ = load("final")
    live = c["live"]
    cvars = [names.index(v) for v in ["contact", "c_floor", "c_wall", "c_fore"]]
    avars = [names.index(v) for v in ["qdd1", "qdd2"]]
    sets = {}
    sets["contact"] = rank_units(c, vs, ["contact", "c_floor", "c_wall", "c_fore", "ctq1", "ctq2"], K, "contact")
    sets["accel"] = rank_units(c, vs, ["qdd1", "qdd2"], K, "accel")
    rng = np.random.default_rng(0)

    def random_like(us, seed):
        r = np.random.default_rng(seed)
        out = []
        for u in us:
            li = u // H
            pool = [v for v in range(li * H, (li + 1) * H) if live[v] and v not in us]
            out.append(int(r.choice(pool)))
        return out

    # windows
    flag = st["contact_flag"]
    cf, cw = st["contact_floor"], st["contact_wall"]
    speed = np.abs(st["qd"]).sum(-1)
    onset = [(e, t) for e in TEST for t in range(5, 146) if flag[e, t + 1] > 0 and flag[e, t - 2:t + 1].sum() == 0
             and cf[e, t + 1] > 0 and speed[e, t] > 3]
    free = [(e, t) for e in TEST for t in range(5, 146, 9) if flag[e, t - 3:t + 3].sum() == 0 and speed[e, t] > 3]
    rng.shuffle(onset); rng.shuffle(free)
    onset, free = onset[:300], free[:300]
    rend = ArmRenderer(64)
    res = {"K": K, "sets": {k: [f"{LAYERS[u // H]}.{u % H}" for u in v] for k, v in sets.items()},
           "set_best": {k: [(vs[int(c['best'][u])].name, float(c['best_r2'][u])) for u in v] for k, v in sets.items()}}

    def metrics(P, E, T, cfimgs=None):
        y1 = future(E, T, 1).cpu().numpy(); y2 = future(E, T, 2).cpu().numpy()
        x = windows(E, T).cpu()
        seq = torch.cat([x, torch.as_tensor(y1)[:, None], torch.as_tensor(y2)[:, None]], 1).to(DEV)
        mask = change_mask(seq)[:, 0].cpu().numpy()
        out = {}
        for k, y in [(0, y1), (1, y2)]:
            se = ((P[:, k] - y) ** 2).mean(1)
            out[f"mse_arm_t{k + 1}"] = float(((se * mask).sum((1, 2)) / np.maximum(mask.sum((1, 2)), 1)).mean())
            if cfimgs is not None:
                y = y1 if k == 0 else y2
                dm = (np.abs(cfimgs[:, k] - y).max(1) > 0.1)
                a_ = (((P[:, k] - y) ** 2).mean(1) * dm).sum((1, 2))
                b_ = (((P[:, k] - cfimgs[:, k]) ** 2).mean(1) * dm).sum((1, 2))
                ok = dm.sum((1, 2)) > 0
                out[f"cf_score_t{k + 1}"] = float((a_[ok] / (a_[ok] + b_[ok] + 1e-12)).mean())
                d_cf = ((P[:, k] - cfimgs[:, k]) ** 2).mean(1)
                out[f"mse_arm_vs_cf_t{k + 1}"] = float(((d_cf * mask).sum((1, 2)) / np.maximum(mask.sum((1, 2)), 1)).mean())
        return out

    Eo, To = np.array([e for e, _ in onset]), np.array([t for _, t in onset])
    Ef, Tf = np.array([e for e, _ in free]), np.array([t for _, t in free])
    cfimgs = np.stack([rend(counterfactual(st, e, t)) for e, t in onset]).astype(np.float32).transpose(0, 1, 4, 2, 3) / 255
    gt = np.stack([future(Eo, To, 1).cpu().numpy(), future(Eo, To, 2).cpu().numpy()], 1)
    res["onset_reference"] = {f"frac_windows_cf_differs_t{k + 1}": float((np.abs(cfimgs[:, k] - gt[:, k]).max(1) > 0.1).any((1, 2)).mean())
                              for k in range(2)}
    preds = {}
    for cond, E, T, cfi in [("onset", Eo, To, cfimgs), ("free", Ef, Tf, None)]:
        res[cond] = {"n": int(len(E))}
        base = predict(model, E, T, [], H, means)
        res[cond]["baseline"] = metrics(base, E, T, cfi)
        if cond == "onset":
            preds["baseline"] = base
        for sname, us in sets.items():
            P = predict(model, E, T, us, H, means)
            res[cond][f"ablate_{sname}"] = metrics(P, E, T, cfi)
            if cond == "onset":
                preds[sname] = P
            rnd = [metrics(predict(model, E, T, random_like(us, s), H, means), E, T, cfi) for s in range(5)]
            res[cond][f"random_{sname}"] = {k: float(np.mean([r[k] for r in rnd])) for k in rnd[0]}
            res[cond][f"random_{sname}_sd"] = {k: float(np.std([r[k] for r in rnd])) for k in rnd[0]}
        print(cond, json.dumps(res[cond], indent=1), flush=True)
    json.dump(res, open(os.path.join(RESULTS, "video_causal.json"), "w"), indent=1)

    # ---------- figure: examples with largest ablation effect + bar chart
    import matplotlib.pyplot as plt
    diff = np.abs(preds["contact"][:, 1] - preds["baseline"][:, 1]).sum((1, 2, 3))
    ex = np.argsort(-diff)[:3]
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(3, 8, width_ratios=[1, 1, 1, 1, 1, 0.25, 1.05, 1.05], wspace=0.06, hspace=0.12,
                          left=0.01, right=0.99, top=0.93, bottom=0.08)
    heads = ["input t", "true t+2", "pred t+2", "pred t+2,\ncontact units ablated",
             "counterfactual t+2\n(no obstacles)"]
    for r, i in enumerate(ex):
        e, t = Eo[i], To[i]
        xt = windows([e], [t])[0, -1].cpu().numpy().transpose(1, 2, 0)
        ims = [xt, gt[i, 1].transpose(1, 2, 0), preds["baseline"][i, 1].transpose(1, 2, 0),
               preds["contact"][i, 1].transpose(1, 2, 0), cfimgs[i, 1].transpose(1, 2, 0)]
        for j, im in enumerate(ims):
            ax = fig.add_subplot(gs[r, j])
            ax.imshow(im, interpolation="nearest")
            clean(ax)
            if r == 0:
                ax.set_title(heads[j], fontsize=6.5, loc="center", fontweight="normal", color=INK2)
        # difference image ablated - baseline
        ax = fig.add_subplot(gs[r, 6])
        dimg = (preds["contact"][i, 1] - preds["baseline"][i, 1]).mean(0)
        lim = max(np.abs(dimg).max(), 1e-3)
        ax.imshow(dimg, cmap="physdiv", vmin=-lim, vmax=lim)
        clean(ax)
        if r == 0:
            ax.set_title("ablated − intact", fontsize=6.5, loc="center", fontweight="normal", color=INK2)
        ax = fig.add_subplot(gs[r, 7])
        dimg = (cfimgs[i, 1] - gt[i, 1]).mean(0)
        lim = max(np.abs(dimg).max(), 1e-3)
        ax.imshow(dimg, cmap="physdiv", vmin=-lim, vmax=lim)
        clean(ax)
        if r == 0:
            ax.set_title("counterfactual − true", fontsize=6.5, loc="center", fontweight="normal", color=INK2)
    savefig(fig, "video_causal_examples.png")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    groups = [("onset", "mse_arm_t2", "arm-pixel MSE, t+2\n(contact-onset windows)"),
              ("free", "mse_arm_t2", "arm-pixel MSE, t+2\n(free-motion windows)"),
              ("onset", "cf_score_t2", "counterfactual score, t+2 (onset)\n0 = like reality, 1 = like no obstacles")]
    for ax, (cond, key, lab) in zip(axes, groups):
        bars = [("intact", res[cond]["baseline"][key], 0, MUTED)]
        for k, colr in [("contact", FAMILIES["contact"][0]), ("accel", FAMILIES["accel"][0])]:
            bars.append((f"ablate {k}", res[cond][f"ablate_{k}"][key], 0, colr))
            bars.append((f"random ×{K}\n({k} layers)", res[cond][f"random_{k}"][key], res[cond][f"random_{k}_sd"][key], "#c3c2b7"))
        x = np.arange(len(bars))
        ax.bar(x, [b[1] for b in bars], yerr=[b[2] for b in bars], color=[b[3] for b in bars], width=0.7,
               error_kw=dict(lw=0.8))
        ax.set_xticks(x, [b[0] for b in bars], rotation=60, ha="right", fontsize=6)
        ax.set_title(lab, fontsize=7.5)
        ax.grid(axis="x", visible=False)
        if key.startswith("cf_score"):
            ax.axhline(0.5, color=MUTED, ls=":", lw=0.8)
    fig.tight_layout()
    savefig(fig, "video_causal.png")


if __name__ == "__main__":
    main()
