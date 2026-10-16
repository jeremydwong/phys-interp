"""Prediction quality of the video model vs copy-last-frame and linear-extrapolation baselines.

Figures: video_training.png (validation curve), video_prediction.png (examples incl. collisions).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import INK2, MUTED, SERIES, savefig, setup_style  # noqa: E402
from analysis.video_common import (CKPT, RESULTS, TEST, future, grid, load,  # noqa: E402
                                   target_matrix, windows)
from models.video_train import change_mask  # noqa: E402


@torch.no_grad()
def metrics(model, E, T, bs=256):
    rows = []
    for i in range(0, len(E), bs):
        e, t = E[i:i + bs], T[i:i + bs]
        x = windows(e, t)
        y1, y2 = future(e, t, 1), future(e, t, 2)
        p = model(x)
        seq = torch.cat([x, y1[:, None], y2[:, None]], 1)
        mask = change_mask(seq)[:, 0]
        lin1 = (2 * x[:, -1] - x[:, -2]).clamp(0, 1)
        lin2 = (3 * x[:, -1] - 2 * x[:, -2]).clamp(0, 1)
        out = []
        for pr, y in [(p[:, 0], y1), (x[:, -1], y1), (lin1, y1), (p[:, 1], y2), (x[:, -1], y2), (lin2, y2)]:
            se = ((pr - y) ** 2).mean(1)
            out += [se.mean((1, 2)), (se * mask).sum((1, 2)) / mask.sum((1, 2)).clamp(min=1)]
        rows.append(torch.stack(out, 1).cpu().numpy())
    return np.concatenate(rows)   # [N, 12]


COLS = ["model1", "model1_arm", "copy1", "copy1_arm", "lin1", "lin1_arm",
        "model2", "model2_arm", "copy2", "copy2_arm", "lin2", "lin2_arm"]


def main():
    setup_style()
    model, ck = load("final")
    Y, vs, st = target_matrix()
    E, T = grid(TEST, 3)
    M = metrics(model, E, T)
    c = st["contact_flag"]
    onset = (c[E, T + 1] > 0) & (c[E, T] == 0)          # contact begins between t and t+1
    in_contact = (c[E, T] > 0) | (c[E, T + 1] > 0)
    res = {"n_windows": int(len(E)), "params": int(sum(p.numel() for p in model.parameters())),
           "step": int(ck["step"])}
    for name, sel in [("all", np.ones(len(E), bool)), ("free", ~in_contact), ("contact", in_contact),
                      ("onset", onset)]:
        res[name] = {k: float(M[sel, j].mean()) for j, k in enumerate(COLS)}
        res[name]["n"] = int(sel.sum())
    for name in ["all", "free", "contact", "onset"]:
        r = res[name]
        r["skill_vs_copy_arm"] = 1 - r["model1_arm"] / r["copy1_arm"]
        r["skill_vs_lin_arm"] = 1 - r["model1_arm"] / r["lin1_arm"]
        print(name, {k: round(v, 5) for k, v in r.items()})
    json.dump(res, open(os.path.join(RESULTS, "video_predict.json"), "w"), indent=1)

    # ---------------- training curve
    log = json.load(open(os.path.join(CKPT, "video_trainlog.json")))
    val = sorted(log["val"], key=lambda d: d["step"])
    steps = np.array([d["step"] for d in val])
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
    ax[0].plot(steps[1:], [d["model_arm"] for d in val][1:], color=SERIES[0], label="model")
    ax[0].axhline(val[0]["copy_arm"], color=MUTED, ls="--", lw=1.2, label="copy last frame")
    ax[0].axhline(val[0]["linear_arm"], color=SERIES[1], ls=":", lw=1.2, label="linear extrapolation")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("training step"); ax[0].set_ylabel("MSE on moving-arm pixels (t+1)")
    ax[0].set_title("Validation error"); ax[0].legend(loc="lower left")
    for s in log["save_at"]:
        ax[0].axvline(max(s, 1), color="#e1e0d9", lw=0.6, zorder=0)
    tr = np.array(log["train"])
    ax[1].plot(tr[1:, 0], tr[1:, 1], color=SERIES[0])
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("training step"); ax[1].set_ylabel("weighted training loss")
    ax[1].set_title("Training loss (EMA)")
    fig.tight_layout()
    savefig(fig, "video_training.png")

    # ---------------- examples
    cf, cw = st["contact_floor"], st["contact_wall"]
    speed = np.abs(st["qd"]).sum(-1)
    picks = []
    rng = np.random.default_rng(0)
    # free motion: fast, no contact near
    cand = [(e, t) for e in TEST for t in range(10, 140, 7) if c[e, t - 3:t + 3].sum() == 0 and speed[e, t] > 6]
    picks.append(("free motion", cand[rng.integers(len(cand))]))
    for name, arr in [("floor impact", cf), ("wall impact", cw)]:
        cand = [(e, t) for e in TEST for t in range(10, 140) if arr[e, t + 1] > 0 and c[e, t - 3:t + 1].sum() == 0
                and speed[e, t] > 5]
        picks.append((name, cand[rng.integers(len(cand))]))
    fig, axes = plt.subplots(len(picks), 9, figsize=(7.2, 2.75), gridspec_kw=dict(wspace=0.05, hspace=0.12))
    heads = ["t−3", "t−2", "t−1", "t", "true t+1", "pred t+1", "true Δ", "pred Δ", "pred t+2"]
    for r, (name, (e, t)) in enumerate(picks):
        x = windows([e], [t])
        with torch.no_grad():
            p = model(x)[0].clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
        xs = x[0].cpu().numpy().transpose(0, 2, 3, 1)
        y1 = future([e], [t], 1)[0].cpu().numpy().transpose(1, 2, 0)
        dt = (y1 - xs[-1]).mean(-1)
        dp = (p[0] - xs[-1]).mean(-1)
        lim = max(np.abs(dt).max(), 1e-3)
        ims = list(xs) + [y1, p[0]]
        for j in range(9):
            a = axes[r, j]
            if j < 6:
                a.imshow(ims[j], interpolation="nearest")
            elif j == 6:
                a.imshow(dt, cmap="physdiv", vmin=-lim, vmax=lim, interpolation="nearest")
            elif j == 7:
                a.imshow(dp, cmap="physdiv", vmin=-lim, vmax=lim, interpolation="nearest")
            else:
                a.imshow(p[1], interpolation="nearest")
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            for s in a.spines.values():
                s.set_visible(False)
            if r == 0:
                a.set_title(heads[j], fontsize=7, loc="center", fontweight="normal", color=INK2)
        axes[r, 0].set_ylabel(name, fontsize=7, color=INK2)
    savefig(fig, "video_prediction.png")
    return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # noqa: F401
    main()
else:
    import matplotlib.pyplot as plt  # noqa: F401,E402
