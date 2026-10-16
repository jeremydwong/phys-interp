"""Analysis 8: causal interventions on gravity and contact units.

Gravity. 'Gravity units' = live units whose census best variable is a gravity torque (grav1, grav2, or
the forearm angle cos(q1+q2), which is identical to the elbow gravity torque up to scale) with R² >= 0.5.
We zero-ablate (and mean-ablate) the top-k of them, or multiply them by a steering factor s, and measure the
change in the model's predicted acceleration a = Δqd / 20 ms on contact-free held-out frames. The change
expected from scaling gravity by s is Δa* = (s - 1) M(q)^-1 τ_g(q). We report the pooled regression slope
of Δa_model on Δa* (1 = the model behaves exactly as if gravity were scaled), the fraction of variance of
Δa_model explained by Δa*, and the median per-frame cosine between the 2-D vectors. Controls: equally many
random live units drawn from the same layers, and the top velocity units.

Contact. 'Contact units' = units whose best variable is in the contact family with R² >= 0.5. On frames
whose 20 ms step contains contact, the contact-induced velocity change is Δqd_true - Δqd_rigid (rigid-body
physics without obstacles). We regress the model's contact component (Δqd_model - Δqd_rigid) on it, intact
vs ablated, and roll the model out autoregressively through a floor impact.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, SHOULDER, ELBOW, savefig, setup_style  # noqa: E402
from analysis.state_common import (CACHE, FRAME_DT, FLOOR_Z, R_FORE, data, draw_pose, fk, load_census,  # noqa
                                   load_model, run, update_results)
from analysis.state_probes import accel_components  # noqa: E402
from models.state_mlp import make_inputs  # noqa: E402


def make_edits(units, mode="zero", scale=0.0, means=None):
    """units: list of (layer, idx). mode zero | mean | scale."""
    by = {}
    for l, k in units:
        by.setdefault(l, []).append(k)
    edits = {}
    for l, ks in by.items():
        ks_t = torch.tensor(ks)
        def fn(h, ks_t=ks_t, l=l):
            h = h.clone()
            if mode == "zero":
                h[:, ks_t] = 0
            elif mode == "mean":
                h[:, ks_t] = torch.as_tensor(means[l][ks_t.numpy()], dtype=h.dtype)
            else:
                h[:, ks_t] = h[:, ks_t] * scale
            return h
        edits[l] = fn
    return edits


@torch.no_grad()
def accel(model, X, mu, sd, edits=None):
    y = model(torch.as_tensor(X), edits=edits).numpy()
    D = y * sd + mu
    return D[:, 2:] / FRAME_DT, D


@torch.no_grad()
def accel_from_resid(model, x, l, mu, sd):
    """Continue the forward pass from residual stream x after `l` blocks (l=0: embedding output)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    for b in model.blocks[l:]:
        x, _ = b(x)
    D = model.head(model.ln_f(x)).numpy() * sd + mu
    return D[:, 2:] / FRAME_DT


def direction_steering(model, d, free, ag, g_true, mu, sd, scales, n_fit=60000, seed=0):
    """Edit the residual stream after block l so that the gravity readout changes by (s-1) * tau_g, along
    (a) the decoding direction (minimal-norm change that moves a linear probe's output) or
    (b) the encoding direction (regression of the residual on tau_g), and compare the resulting change in
    predicted acceleration with the change expected if gravity were scaled, (s-1) M^-1 tau_g."""
    from analysis.state_common import run as run_
    rng = np.random.default_rng(seed)
    tr = rng.choice(len(d["Xtr"]), n_fit, replace=False)
    Rtr = run_(model, d["Xtr"][tr])["resid"]
    gi = [d["names"].index("grav1"), d["names"].index("grav2")]
    Gtr = d["Ytr"][tr][:, gi].astype(np.float64)
    Rte = run_(model, d["X"][free])["resid"]
    out = {}
    for l in range(Rtr.shape[0] - 1):
        Xc = Rtr[l] - Rtr[l].mean(0)
        Gc = Gtr - Gtr.mean(0)
        # decoding: ridge probe W [256,2]; minimal-norm edit = ΔG @ pinv(W)
        W = np.linalg.solve(Xc.T @ Xc / len(Xc) + 1e-4 * np.eye(Xc.shape[1]), Xc.T @ Gc / len(Xc))
        dec = np.linalg.pinv(W)                       # [2,256]
        enc = np.linalg.solve(Gc.T @ Gc, Gc.T @ Xc)   # [2,256] encoding pattern
        a0 = accel_from_resid(model, Rte[l], l, mu, sd)
        res_l = {}
        for name, P in (("decoding", dec), ("encoding", enc)):
            row = {}
            for sc in scales:
                x1 = Rte[l] + ((sc - 1) * g_true) @ P
                da = accel_from_resid(model, x1, l, mu, sd) - a0
                c = compare(da, (sc - 1) * ag)
                x_, y_ = ag.ravel(), da.ravel()
                c["effective_gravity_change"] = float((x_ @ y_) / (x_ @ x_))
                row[sc] = c
            res_l[name] = row
        out[["embed", "block 1", "block 2", "block 3"][l]] = res_l
    return out


def scatter_edit(model, d, free, sub, gi, mu, sd, layer, n_fit=60000, seed=0):
    """Δ predicted acceleration when the encoding-direction edit with s = 0 is applied after `layer` blocks."""
    from analysis.state_common import run as run_
    rng = np.random.default_rng(seed)
    tr = rng.choice(len(d["Xtr"]), n_fit, replace=False)
    Rtr = run_(model, d["Xtr"][tr])["resid"][layer]
    Gtr = d["Ytr"][tr][:, gi].astype(np.float64)
    Xc, Gc = Rtr - Rtr.mean(0), Gtr - Gtr.mean(0)
    enc = np.linalg.solve(Gc.T @ Gc, Gc.T @ Xc)
    Xf = d["X"][free][sub]
    R = run_(model, Xf)["resid"][layer]
    g = d["Y"][free][sub][:, gi]
    return accel_from_resid(model, R - g @ enc, layer, mu, sd) - accel_from_resid(model, R, layer, mu, sd)


def compare(da, dexp):
    """da, dexp [N,2]. slope (through origin, pooled), R², median cosine."""
    x, y = dexp.ravel(), da.ravel()
    slope = (x @ y) / (x @ x)
    r2 = 1 - ((y - slope * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    nrm = np.linalg.norm(da, axis=1) * np.linalg.norm(dexp, axis=1)
    ok = nrm > 1e-6
    cos = (da * dexp).sum(1)[ok] / nrm[ok]
    return dict(slope=float(slope), r2=float(r2), median_cos=float(np.median(cos)),
                rms_change=float(np.sqrt((da ** 2).mean())), rms_expected=float(np.sqrt((dexp ** 2).mean())))


def main():
    d = data()
    names = d["names"]
    st = d["st"]
    T = d["T"]
    cz = load_census()
    model, _ = load_model("final")
    mu, sd = d["mu"], d["sd"]
    means = run(model, d["X"])["h"].mean(1)
    ac = accel_components(st)
    a_grav = ac["a_grav"][2000:, :T].reshape(-1, 2)
    cf0 = st["contact_flag"][2000:, :T].reshape(-1) > 0
    cf1 = st["contact_flag"][2000:, 1:T + 1].reshape(-1) > 0
    free = ~cf0 & ~cf1
    X = d["X"][free]
    ag = a_grav[free]
    a0, _ = accel(model, X, mu, sd)
    rng = np.random.default_rng(0)

    gsel = [names.index(v) for v in ("grav1", "grav2", "cos_q12")]
    grav_units = [(l, int(u)) for l in range(4) for u in np.where(cz["live"][l] & np.isin(cz["best"][l], gsel)
                                                                   & (cz["bestr"][l] >= 0.5))[0]]
    grav_units.sort(key=lambda lu: -cz["bestr"][lu[0]][lu[1]])
    vsel = [names.index(v) for v in ("qd1", "qd2", "ee_vx", "ee_vz")]
    vel_units = [(l, int(u)) for l in range(4) for u in np.where(cz["live"][l] & np.isin(cz["best"][l], vsel)
                                                                  & (cz["bestr"][l] >= 0.5))[0]]
    vel_units.sort(key=lambda lu: -cz["bestr"][lu[0]][lu[1]])
    live = [(l, int(u)) for l in range(4) for u in np.where(cz["live"][l])[0]]
    print("n gravity units", len(grav_units), "per layer", np.bincount([l for l, _ in grav_units], minlength=4),
          "n velocity units", len(vel_units))

    def random_matched(units, seed):
        r = np.random.default_rng(seed)
        out = []
        gset = set(grav_units)
        for l, _ in units:
            pool = [x for x in live if x[0] == l and x not in gset and x not in out]
            out.append(pool[r.integers(len(pool))])
        return out

    res = dict(n_gravity_units=len(grav_units), n_velocity_units=len(vel_units),
               gravity_units_per_layer=np.bincount([l for l, _ in grav_units], minlength=4).tolist(),
               ablation={}, steering={})
    ks = [k for k in (1, 5, 10, 25, 50, 100, len(grav_units)) if k <= len(grav_units)]
    for k in ks:
        row = {}
        for mode in ("zero", "mean"):
            a1, _ = accel(model, X, mu, sd, make_edits(grav_units[:k], mode, means=means))
            row[f"gravity_{mode}"] = compare(a1 - a0, -ag)
            ctl = [compare(accel(model, X, mu, sd, make_edits(random_matched(grav_units[:k], s), mode, means=means))[0] - a0, -ag)
                   for s in range(3)]
            row[f"random_{mode}"] = {m: float(np.mean([c[m] for c in ctl])) for m in ctl[0]}
        if k <= len(vel_units):
            a1, _ = accel(model, X, mu, sd, make_edits(vel_units[:k], "zero"))
            row["velocity_zero"] = compare(a1 - a0, -ag)
        res["ablation"][k] = row
        print("k", k, {m: (round(v["slope"], 3), round(v["r2"], 3), round(v["median_cos"], 3)) for m, v in row.items()})
    scales = [0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0]
    for s in scales:
        a1, _ = accel(model, X, mu, sd, make_edits(grav_units, "scale", scale=s))
        r = compare(a1 - a0, (s - 1) * ag)
        x, y = ag.ravel(), (a1 - a0).ravel()
        r["effective_gravity_change"] = float((x @ y) / (x @ x))
        a2, _ = accel(model, X, mu, sd, make_edits(random_matched(grav_units, 0), "scale", scale=s))
        y2 = (a2 - a0).ravel()
        r["random_effective_gravity_change"] = float((x @ y2) / (x @ x))
        res["steering"][s] = r
        print("scale", s, round(r["effective_gravity_change"], 3), round(r["random_effective_gravity_change"], 3),
              round(r["median_cos"], 3))
    # direction-level steering in the residual stream
    gi = [names.index("grav1"), names.index("grav2")]
    g_true = d["Y"][free][:, gi].astype(np.float64)
    dsteer = direction_steering(model, d, free, ag, g_true, mu, sd, scales)
    res["direction_steering"] = dsteer
    for lname, v in dsteer.items():
        print("steer", lname, {nm: [round(v[nm][sc]["effective_gravity_change"], 2) for sc in scales] for nm in v},
              "R2(s=0)", {nm: round(v[nm][0.0]["r2"], 2) for nm in v})
    # per-segment: shoulder-type vs elbow-type gravity units
    seg = {}
    for nm_, sel in (("shoulder_units", [names.index("grav1")]), ("elbow_units", [names.index("grav2"), names.index("cos_q12")])):
        us = [u for u in grav_units if cz["best"][u[0]][u[1]] in sel]
        a1, _ = accel(model, X, mu, sd, make_edits(us, "zero"))
        dd_ = a1 - a0
        seg[nm_] = dict(n=len(us), slope_joint1=float((dd_[:, 0] @ -ag[:, 0]) / (ag[:, 0] @ ag[:, 0])),
                        slope_joint2=float((dd_[:, 1] @ -ag[:, 1]) / (ag[:, 1] @ ag[:, 1])),
                        rms_change_joint=np.sqrt((dd_ ** 2).mean(0)).tolist())
    res["per_segment"] = seg
    print(seg)

    # ------------------------------------------------ figure: gravity interventions
    setup_style()
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(7, 4.6))
    axes = axs.ravel()
    kk = list(res["ablation"])
    ax = axes[0]
    ax.plot(kk, [res["ablation"][k]["gravity_zero"]["slope"] for k in kk], "-o", ms=3, color=FAMILIES["gravity"][0], label="gravity units, zero")
    ax.plot(kk, [res["ablation"][k]["gravity_mean"]["slope"] for k in kk], "--o", ms=3, color=FAMILIES["gravity"][0], alpha=0.6, label="gravity units, mean")
    ax.plot(kk, [res["ablation"][k]["random_zero"]["slope"] for k in kk], "-o", ms=3, color="#898781", label="random units, zero")
    vk = [k for k in kk if "velocity_zero" in res["ablation"][k]]
    ax.plot(vk, [res["ablation"][k]["velocity_zero"]["slope"] for k in vk], "-o", ms=3, color=FAMILIES["velocity"][0], label="velocity units, zero")
    ax.set_xscale("log")
    ax.axhline(1, color="#0b0b0b", lw=0.6, ls=":")
    ax.set_ylim(-0.3, 1.1)
    ax.set_xlabel("number of units ablated (top-k by R²)")
    ax.set_ylabel("fraction of gravity removed\n(slope of Δq̈ on −M⁻¹τ_g)")
    ax.set_title("a  Ablating gravity-tuned units", fontsize=8.5)
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax = axes[1]
    ax.plot(scales, [res["steering"][s]["effective_gravity_change"] for s in scales], "-o", ms=3, color=FAMILIES["gravity"][0], label="all gravity units")
    ax.plot(scales, [res["steering"][s]["random_effective_gravity_change"] for s in scales], "-o", ms=3, color="#898781", label="random units")
    ax.plot([0, 2], [-1, 1], color="#0b0b0b", lw=0.6, ls=":", label="ideal: s − 1")
    ax.set_xlabel("gain s applied to unit activations")
    ax.set_ylabel("effective gravity change\n(slope of Δq̈ on M⁻¹τ_g)")
    ax.set_ylim(-1.6, 1.6)
    ax.set_title("b  Scaling gravity-tuned units", fontsize=8.5)
    ax.legend(fontsize=6, loc="upper left")
    ax = axes[2]
    lcols = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
    for li, (lname, v) in enumerate(dsteer.items()):
        ax.plot(scales, [v["encoding"][sc]["effective_gravity_change"] for sc in scales], "-o", ms=2.5, color=lcols[li],
                label=f"after {lname}" if lname != "embed" else "embedding")
    v = dsteer["block 3"]
    ax.plot(scales, [v["decoding"][sc]["effective_gravity_change"] for sc in scales], "--", lw=1.0, color="#e34948",
            label="decoding dir., any layer")
    ax.plot([0, 2], [-1, 1], color="#0b0b0b", lw=0.6, ls=":")
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("gravity scale s written into the residual stream")
    ax.set_ylabel("effective gravity change")
    ax.set_title("c  Editing the gravity direction (encoding)", fontsize=8.5)
    ax.legend(fontsize=6, loc="upper left")
    ax = axes[3]
    # scatter: encoding edit after block 3 with s = 0 (remove gravity)
    from analysis.state_common import run as run_
    gi = [names.index("grav1"), names.index("grav2")]
    sub = rng.choice(len(X), 4000, replace=False)
    dsc = scatter_edit(model, d, free, sub, gi, mu, sd, layer=3)
    for j, c in ((0, SHOULDER), (1, ELBOW)):
        ax.scatter(-ag[sub, j], dsc[:, j], s=1.5, alpha=0.35, color=c, lw=0, label=["shoulder", "elbow"][j], rasterized=True)
    lim = np.percentile(np.abs(ag[sub]), 99.5)
    ax.plot([-lim, lim], [-lim, lim], color="#0b0b0b", lw=0.6, ls=":")
    ax.set_xlim(-lim * 1.1, lim * 1.1)
    ax.set_ylim(-lim * 1.6, lim * 1.6)
    ax.set_xlabel("expected Δq̈ if gravity removed, −M⁻¹τ_g (rad/s²)")
    ax.set_ylabel("model Δq̈ (rad/s²)")
    ax.set_title("d  s = 0 written after block 3", fontsize=8.5)
    ax.legend(fontsize=6, markerscale=4, loc="upper left")
    fig.subplots_adjust(wspace=0.38, hspace=0.55)
    savefig(fig, "state_causal_gravity.png")

    # ------------------------------------------------ contact ablation
    csel = [i for i, v in enumerate(d["vars"]) if v.family == "contact"]
    con_units = [(l, int(u)) for l in range(4) for u in np.where(cz["live"][l] & np.isin(cz["best"][l], csel)
                                                                  & (cz["bestr"][l] >= 0.5))[0]]
    Prig = np.load(os.path.join(CACHE, "physics_no_contact.npy"))
    D = d["D"]
    bounce = cf1 & ~cf0     # contact begins within the step
    incon = cf1
    dprime = np.load(os.path.join(CACHE, "impact_dprime.npy"))
    imp_units = [(int(l), int(k)) for l, k in zip(*np.where(dprime > 0.5))]
    _, Dm0 = accel(model, d["X"], mu, sd)
    _, Dm1 = accel(model, d["X"], mu, sd, make_edits(con_units, "zero"))
    _, Dm2 = accel(model, d["X"], mu, sd, make_edits(random_matched(con_units, 0), "zero"))
    _, Dm3 = accel(model, d["X"], mu, sd, make_edits(imp_units, "zero"))
    _, Dm4 = accel(model, d["X"], mu, sd, make_edits(random_matched(imp_units, 1), "zero"))
    true_c = (D - Prig)[:, 2:]
    cres = dict(n_contact_units=len(con_units),
                contact_units_per_layer=np.bincount([l for l, _ in con_units], minlength=4).tolist(),
                n_impact_units=len(imp_units),
                impact_units_per_layer=np.bincount([l for l, _ in imp_units], minlength=4).tolist())
    variants = (("intact", Dm0), ("contact_ablated", Dm1), ("random_ablated", Dm2), ("impact_ablated", Dm3),
                ("random_for_impact_ablated", Dm4))
    for nm_, sel in (("onset_steps", bounce), ("all_contact_steps", incon)):
        for tag, Dm in variants:
            mc = (Dm - Prig)[sel, 2:]
            x, y = true_c[sel].ravel(), mc.ravel()
            cres[f"{nm_}_{tag}_slope"] = float((x @ y) / (x @ x))
            cres[f"{nm_}_{tag}_r2"] = float(1 - ((y - x) ** 2).sum() / ((x - x.mean()) ** 2).sum())
    for tag, Dm in variants:
        cres[f"free_steps_{tag}_r2_dqd"] = (1 - ((Dm - D)[free, 2:] ** 2).mean(0) / D[free, 2:].var(0)).tolist()
    print(cres)

    # rollout through a floor impact (passive episodes, forearm hits floor)
    modes = st["mode"][2000:]
    best, bv = None, -1
    for e in range(400):
        if modes[e] != "passive":
            continue
        c = st["contact_flag"][2000 + e]
        for t in range(20, T - 25):
            if c[t] and not c[t - 15:t].any() and st["contact_floor"][2000 + e, t] > 0:
                v = -st["ee_vel"][2000 + e, t - 1, 1]
                if v > bv:
                    best, bv = (e, t), v
                break
    e, t0 = best
    E = 2000 + e
    start, n = t0 - 12, 30

    def rollout(edits):
        q, qd = st["q"][E, start].astype(np.float64), st["qd"][E, start].astype(np.float64)
        qs = [q.copy()]
        for t in range(start, start + n):
            u = make_inputs(q[None], qd[None], st["tau_act"][E, t][None].astype(np.float64))
            _, Dp = accel(model, u, mu, sd, edits)
            q, qd = q + Dp[0, :2], qd + Dp[0, 2:]
            qs.append(q.copy())
        return np.array(qs)

    abl_units = sorted(set(con_units) | set(imp_units))
    cres["n_union_units"] = len(abl_units)
    roll = {"truth": st["q"][E, start:start + n + 1], "intact": rollout(None),
            "contact units ablated": rollout(make_edits(abl_units, "zero")),
            "random units ablated": rollout(make_edits(random_matched(abl_units, 0), "zero"))}
    fig = plt.figure(figsize=(7, 3.0))
    gs = fig.add_gridspec(3, 7, width_ratios=[2.6, 0.62, 1, 1, 1, 1, 1], wspace=0.05, hspace=0.08)
    ax = fig.add_subplot(gs[:, 0])
    tt = (np.arange(n + 1) - 12) * FRAME_DT * 1000
    cols = {"truth": "#0b0b0b", "intact": FAMILIES["contact"][0], "contact units ablated": "#2a78d6",
            "random units ablated": "#898781"}
    ymin = {}
    for k, qq in roll.items():
        el, ee = fk(qq)
        z = np.minimum(el[:, 1], ee[:, 1])
        ymin[k] = float(z.min())
        ax.plot(tt, z, color=cols[k], lw=1.6 if k == "truth" else 1.2, ls="-" if k != "random units ablated" else "--", label=k)
    ax.axhline(FLOOR_Z + R_FORE, color="#52514e", lw=0.8)
    ax.text(tt[0], FLOOR_Z + R_FORE + 0.01, "floor contact height", fontsize=6, color="#52514e", va="bottom")
    ax.set_xlabel("time from impact (ms)")
    ax.set_ylabel("lowest point of arm, z (m)")
    ax.legend(fontsize=6, loc="upper right")
    show = [0, 9, 13, 18, 26]
    for r, k in enumerate(["truth", "intact", "contact units ablated"]):
        for c, i in enumerate(show):
            a_ = fig.add_subplot(gs[r, 2 + c])
            draw_pose(a_, roll[k][i], roll[k][max(0, i - 4):i], size=144, crop=((-0.75, 0.86), (-0.95, 0.66)))
            if r == 0:
                a_.set_title(f"{tt[i]:+.0f} ms", fontsize=6.5)
            if c == 0:
                a_.text(-0.08, 0.5, k.replace(" units ", "\n"), transform=a_.transAxes, fontsize=6, ha="right",
                        va="center", color=cols[k])
    savefig(fig, "state_causal_contact.png")
    cres["rollout"] = dict(episode=int(E), impact_frame=int(t0), min_height=ymin,
                           floor_contact_height=float(FLOOR_Z + R_FORE))
    res["contact"] = cres
    update_results("causal", res)


if __name__ == "__main__":
    main()
