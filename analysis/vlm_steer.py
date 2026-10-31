"""D1: steering LM units and asking the behaviour questions again.

For each of the 7 gallery LM units, a pre-hook on that layer's down_proj sets the unit's SwiGLU
activation at all 400 visual tokens of the clip to
  "ablate": 0, or
  "steer":  2x the unit's corpus extreme on its census pole (2 x pooled max for + units,
            2 x pooled min for - units),
and we re-ask the five vlm_behaviour questions on 64 clips each (32 per class, drawn from the
behaviour sets). Read-out: the first-token logit difference between the two answer options
(option 0 = "yes"/"floor"/"faster"/"blue"), turned into p(option 0) = sigmoid(diff), and the
fraction of clips whose two-way answer flips. Control: 3 random units from the same layer, steered
the same way (set to 2x their own extreme, or ablated).
Group condition (ours): the 16 LM units of layers 2 and 8 with the highest contact binned R²
(either pole) steered together, vs 16 random units from the same layers, on the contact,
floor/wall and anticipation questions.

Usage: uv run python analysis/vlm_steer.py [run|analyze]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import load_states, savefig, setup_style, INK, INK2, MUTED, SERIES  # noqa: E402
from analysis.vlm_common import VDIR, n_used, renderer, build_video_inputs, load_model  # noqa: E402
from analysis.vlm_behaviour import QUESTIONS, sample_sets  # noqa: E402

N_PER = 32
N_RAND = 3
BS = 4
CONTACT_QS = ["contact", "floor_wall", "anticipate"]


def unit_extreme(layer, u, pole, NG):
    Z = np.load(os.path.join(VDIR, f"video_llm_mlp_L{layer}.npy"), mmap_mode="r")
    return float(np.asarray(Z[:NG, 0, u], np.float32).max()) if pole == "+" else float(np.asarray(Z[:NG, 1, u], np.float32).min())


def contact_group(NG, k=16):
    z = np.load(os.path.join(VDIR, "census_units.npz"))
    from analysis.common import targets
    names = [v.name for v in targets(load_states())[1]]
    ci = names.index("contact")
    cands = []
    for l in (2, 8):
        for pole, tag in (("+", "pos"), ("−", "neg")):
            key = f"video_LM_L{l}_{tag}"
            B = z[f"{key}__B"].astype(np.float32)[:, ci]
            idx = z[f"{key}__idx"]
            for j in np.argsort(-B)[:k]:
                cands.append((float(B[j]), l, int(idx[j]), pole))
    cands.sort(reverse=True)
    out, seen = [], set()
    for r2, l, u, pole in cands:
        if (l, u) in seen:
            continue
        seen.add((l, u)); out.append((l, u, pole, r2))
        if len(out) == k:
            break
    return out


def run():
    import torch
    st = load_states()
    NG, _ = n_used("video")
    sets = sample_sets(st)
    rng = np.random.default_rng(7)
    data = {}
    for q, S in sets.items():
        lab = S["label"]
        pi = rng.choice(np.where(lab == 1)[0], N_PER, replace=False)
        ni = rng.choice(np.where(lab == 0)[0], N_PER, replace=False)
        idx = np.r_[pi, ni]
        data[q] = dict(items=S["items"][idx], label=lab[idx],
                       clips=[renderer()(st["q"][e][s + 2 * np.arange(8)]) for e, s in S["items"][idx]])
    with open(os.path.join(VDIR, "gallery_spec.json")) as f:
        spec = [s for s in json.load(f) if s["part"] == "LM"]
    conds = []   # (name, kind, [(layer, unit, pole)], target questions)
    for sp in spec:
        l, u, pole = sp["layer"], sp["unit"], sp["pole"]
        lab = f"L{l}{'−' if pole == '−' else ''}#{u}"
        conds.append((lab, "unit", [(l, u, pole)], list(QUESTIONS)))
        for r in range(N_RAND):
            ur = int(rng.integers(0, 9728))
            conds.append((f"{lab}/rand{r}", "random", [(l, ur, pole)], list(QUESTIONS)))
    grp = contact_group(NG)
    conds.append(("contact group (16)", "group", [(l, u, p) for l, u, p, _ in grp], CONTACT_QS))
    for r in range(N_RAND):
        rg = [(l, int(rng.integers(0, 9728)), p) for l, u, p, _ in grp]
        conds.append((f"contact group (16)/rand{r}", "random_group", rg, CONTACT_QS))
    ext = {}
    for _, _, us, _ in conds:
        for l, u, p in us:
            if (l, u, p) not in ext:
                ext[(l, u, p)] = unit_extreme(l, u, p, NG)

    model, proc = load_model()
    lm = model.model.language_model
    tok = proc.tokenizer
    vid_id = tok.convert_tokens_to_ids("<|video_pad|>")
    state = {"pos": None, "edits": []}

    def make_hook(l):
        def hook(m, args):
            if not state["edits"]:
                return None
            x = args[0]
            changed = False
            for (ll, u, val) in state["edits"]:
                if ll != l:
                    continue
                if not changed:
                    x = x.clone(); changed = True
                b, p = state["pos"]
                x[b, p, u] = val
            return (x,) if changed else None
        return hook

    hooks = [lm.layers[l].mlp.down_proj.register_forward_pre_hook(make_hook(l)) for l in sorted({l for _, _, us, _ in conds for l, _, _ in us})]
    out = {"conditions": [(c[0], c[1], [list(x) for x in c[2]], c[3]) for c in conds],
           "contact_group": grp, "extremes": {f"{l},{u},{p}": v for (l, u, p), v in ext.items()}}

    def ask(q, edits):
        Q = QUESTIONS[q]
        ids = [tok.encode(f_, add_special_tokens=False)[0] for f_ in Q["first"]]
        res = []
        clips = data[q]["clips"]
        for bi in range(0, len(clips), BS):
            inp = build_video_inputs(proc, clips[bi:bi + BS], prompt=Q["prompt"]).to(model.device)
            pos = torch.where(inp["input_ids"] == vid_id)
            state["pos"] = pos
            state["edits"] = edits
            with torch.no_grad():
                lg = model(**inp, logits_to_keep=1).logits[:, -1].float()
            state["edits"] = []
            res += (lg[:, ids[0]] - lg[:, ids[1]]).cpu().tolist()
        return res

    t0 = time.time()
    out["baseline"] = {q: ask(q, []) for q in QUESTIONS}
    out["labels"] = {q: data[q]["label"].tolist() for q in QUESTIONS}
    print("baseline", {q: np.mean(np.array(v) > 0) for q, v in out["baseline"].items()}, time.time() - t0, flush=True)
    out["results"] = {}
    for name, kind, us, qs in conds:
        for setting in ("ablate", "steer"):
            edits = [(l, u, 0.0 if setting == "ablate" else 2.0 * ext[(l, u, p)]) for l, u, p in us]
            for q in qs:
                out["results"][f"{name}|{setting}|{q}"] = ask(q, edits)
        print(name, f"{time.time() - t0:.0f}s", flush=True)
        with open(os.path.join(VDIR, "steer_raw.json"), "w") as f:
            json.dump(out, f)
    for h in hooks:
        h.remove()


def analyze():
    setup_style()
    import matplotlib.pyplot as plt
    with open(os.path.join(VDIR, "steer_raw.json")) as f:
        raw = json.load(f)
    sig = lambda x: 1 / (1 + np.exp(-np.asarray(x)))
    base = {q: np.asarray(v) for q, v in raw["baseline"].items()}
    res = {"baseline_p_option0": {q: float(sig(v).mean()) for q, v in base.items()},
           "baseline_acc_first_token": {q: float(((base[q] > 0) == np.asarray(raw["labels"][q])).mean()) for q in base},
           "conditions": {}}
    rows = []
    for name, kind, us, qs in raw["conditions"]:
        for setting in ("ablate", "steer"):
            for q in qs:
                v = np.asarray(raw["results"][f"{name}|{setting}|{q}"])
                dp = float((sig(v) - sig(base[q])).mean())
                flip = float(((v > 0) != (base[q] > 0)).mean())
                dl = float((v - base[q]).mean())
                res["conditions"][f"{name}|{setting}|{q}"] = dict(kind=kind, d_p_option0=dp, flip_rate=flip, d_logit=dl)
                rows.append((name.split("/")[0], kind, setting, q, dp, flip, dl))
    # summary: unit vs its random controls
    summ = {}
    for name, kind, setting, q, dp, flip, dl in rows:
        if kind not in ("unit", "group"):
            continue
        ctrl = [r for r in rows if r[0] == name and r[1] in ("random", "random_group") and r[2] == setting and r[3] == q]
        cd = np.array([r[6] for r in ctrl])
        summ[f"{name}|{setting}|{q}"] = dict(d_logit=dl, d_p=dp, flip=flip, ctrl_d_logit_mean=float(cd.mean()),
                                            ctrl_d_logit_absmax=float(np.abs(cd).max()),
                                            ctrl_flip_mean=float(np.mean([r[5] for r in ctrl])),
                                            exceeds_controls=bool(abs(dl) > np.abs(cd).max()))
    res["summary"] = summ
    with open(os.path.join(VDIR, "steer.json"), "w") as f:
        json.dump(res, f, indent=1)
    for k, v in summ.items():
        print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})

    # figure: d_logit per (unit, question) for steer and ablate, with random-control range
    qs = list(base)
    units = [n for n, k, _, _ in raw["conditions"] if k in ("unit", "group")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    qcol = {q: SERIES[i] for i, q in enumerate(qs)}
    for ax, setting in zip(axes, ("steer", "ablate")):
        for ui, un in enumerate(units):
            for qi, q in enumerate(qs):
                k = f"{un}|{setting}|{q}"
                if k not in summ:
                    continue
                s = summ[k]
                x = ui + (qi - 2) * 0.15
                ctrl = [res["conditions"][kk]["d_logit"] for kk in res["conditions"]
                        if kk.startswith(un + "/") and kk.endswith(f"|{setting}|{q}")]
                ax.scatter([x] * len(ctrl), ctrl, s=6, color=MUTED, lw=0, zorder=2)
                ax.scatter(x, s["d_logit"], s=16, color=qcol[q], zorder=3, label=q if ui == 0 and setting == "steer" else None,
                           edgecolor=INK if s["exceeds_controls"] else "none", linewidth=0.8)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(range(len(units)), [u.replace(" (16)", "\n(16)") for u in units], rotation=60, fontsize=6)
        ax.set_title(f"{'AB'[setting == 'ablate']}  {setting}: Δ answer logit (option 0 − option 1)", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mean Δ logit vs unsteered", fontsize=7)
    axes[0].legend(fontsize=6, loc="upper left", title="question (grey: 3 random units;
black edge: beyond all 3)", title_fontsize=6)
    fig.tight_layout()
    savefig(fig, "vlm_steer.png")


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
