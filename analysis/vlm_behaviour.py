"""Behavioural physics questions for Qwen3-VL-4B, plus probes on the same forward passes.

For each question we draw a balanced set of clips (8 frames, stride 2, 320 px; same protocol
as vlm_extract) from all 400 test episodes, ask the question, and record
  * the greedy answer (max 8 new tokens), parsed to one of the two options,
  * the first-token logit difference between the two options (threshold-free AUC),
  * the residual stream at the last prompt token for all 37 LM layers (for probes),
  * the mean-pooled final vision-block output over the last temporal group (vision probe).
Probe accuracy = 5-fold logistic regression, folds grouped by episode.

Usage: uv run python analysis/vlm_behaviour.py [collect|analyze]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import load_states, savefig, setup_style, INK2, MUTED, SERIES  # noqa: E402
from analysis.vlm_common import BS, VDIR, renderer, load_model, build_video_inputs  # noqa: E402

N_PER_CLASS = 150
SUFFIX = " Answer with one word."

QUESTIONS = {
    "contact": dict(
        prompt="Is the arm touching the floor or the wall at any point in this video? Answer yes or no.",
        options=("yes", "no"), first=("Yes", "No"),
        gt="positive: contact in >=3 of the 8 shown frames; negative: no contact in any sim frame of the clip"),
    "floor_wall": dict(
        prompt="The arm touches a surface in this video. Is it touching the floor or the wall? Answer with one word: floor or wall.",
        options=("floor", "wall"), first=("Floor", "Wall"),
        gt="clips with contact in >=3 shown frames; positive=only floor contact, negative=only wall contact"),
    "speed": dict(
        prompt="Is the arm speeding up or slowing down during this video? Answer with one word: faster or slower.",
        options=("faster", "slower"), first=("Faster", "Slower"),
        gt="kinetic energy at last shown frame > 2x first (speeding) or < 0.5x (slowing); max KE > 0.5 J"),
    "segment": dict(
        prompt="Which part of the arm is rotating faster: the blue part or the orange part? Answer with one word: blue or orange.",
        options=("blue", "orange"), first=("Blue", "Orange"),
        gt="mean |q1dot| (blue) vs mean |q1dot+q2dot| (orange, world-frame rotation) over clip, ratio > 2, faster > 1.5 rad/s"),
    "anticipate": dict(
        prompt="Will the arm hit the floor or the wall right after this video ends? Answer yes or no.",
        options=("yes", "no"), first=("Yes", "No"),
        gt="no contact in clip; positive: contact onset within 10 sim frames (0.2 s) after the last shown frame; negative: no contact in the next 30 frames"),
}


def sample_sets(st, seed=0):
    rng = np.random.default_rng(seed)
    E = np.arange(2000, 2400)
    cf = st["contact_flag"] > 0
    fl, wl = st["contact_floor"] > 0, st["contact_wall"] > 0
    ke = st["ke"]
    qd = st["qd"]
    cands = {k: ([], []) for k in QUESTIONS}
    for e in E:
        for s in range(0, 150 - 15, 3):
            shown = s + 2 * np.arange(8)
            span = np.arange(s, s + 15)
            c_shown = cf[e, shown].sum()
            any_span = cf[e, span].any()
            if c_shown >= 3:
                cands["contact"][0].append((e, s))
                f, w = fl[e, shown].any(), wl[e, shown].any()
                if f and not w:
                    cands["floor_wall"][0].append((e, s))
                elif w and not f:
                    cands["floor_wall"][1].append((e, s))
            elif not any_span:
                cands["contact"][1].append((e, s))
            k0, k1 = ke[e, shown[0]], ke[e, shown[-1]]
            if max(k0, k1) > 0.5:
                if k1 > 2 * k0:
                    cands["speed"][0].append((e, s))
                elif k1 < 0.5 * k0:
                    cands["speed"][1].append((e, s))
            wb = np.abs(qd[e, span, 0]).mean()
            wo = np.abs(qd[e, span, 0] + qd[e, span, 1]).mean()
            if max(wb, wo) > 1.5:
                if wb > 2 * wo:
                    cands["segment"][0].append((e, s))
                elif wo > 2 * wb:
                    cands["segment"][1].append((e, s))
            last = s + 14
            if not any_span and last + 30 < 150:
                nxt = cf[e, last + 1:last + 11]
                if nxt.any():
                    cands["anticipate"][0].append((e, s))
                elif not cf[e, last + 1:last + 31].any():
                    cands["anticipate"][1].append((e, s))
    sets = {}
    for k, (pos, neg) in cands.items():
        n = min(N_PER_CLASS, len(pos), len(neg))
        p = [pos[i] for i in rng.choice(len(pos), n, replace=False)]
        q = [neg[i] for i in rng.choice(len(neg), n, replace=False)]
        sets[k] = dict(items=np.array(p + q), label=np.array([1] * n + [0] * n), n_cand=(len(pos), len(neg)))
    return sets


def parse(ans, options):
    a = ans.strip().lower()
    hits = [o for o in options if o in a]
    if len(hits) == 1:
        return options.index(hits[0])
    if "speed" in a or "accelerat" in a:
        return 0 if options[0] == "faster" else -1
    if "slow" in a or "decelerat" in a:
        return 1 if options[1] == "slower" else -1
    return -1


def collect(bs=None):
    import torch
    st = load_states()
    sets = sample_sets(st)
    bs = bs or BS
    model, proc = load_model()
    tok = proc.tokenizer
    vres = {}
    hook = model.model.visual.blocks[23].register_forward_hook(lambda m, i, o: vres.__setitem__("x", o.detach()))
    for name, q in QUESTIONS.items():
        path = os.path.join(VDIR, f"beh_{name}.npz")
        if os.path.exists(path):
            continue
        S = sets[name]
        ids = [tok.encode(f, add_special_tokens=False)[0] for f in q["first"]]
        assert ids[0] != ids[1]
        answers, ldiff, last, vis = [], [], [], []
        for bi in range(0, len(S["items"]), bs):
            it = S["items"][bi:bi + bs]
            clips = [renderer()(st["q"][e][s + 2 * np.arange(8)]) for e, s in it]
            inp = build_video_inputs(proc, clips, prompt=q["prompt"]).to(model.device)
            with torch.no_grad():
                o = model(**inp, output_hidden_states=True, logits_to_keep=1)
                lg = o.logits[:, -1].float()
                ldiff += (lg[:, ids[0]] - lg[:, ids[1]]).cpu().tolist()
                last.append(torch.stack(o.hidden_states, 1)[:, :, -1].float().cpu().numpy().astype(np.float16))
                B = len(it)
                vis.append(vres["x"].reshape(B, 4, 400, -1)[:, -1].float().mean(1).cpu().numpy().astype(np.float16))
                g = model.generate(**inp, max_new_tokens=8, do_sample=False)
            answers += tok.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            print(name, bi + B, answers[-1], flush=True)
        np.savez(path, items=S["items"], label=S["label"], answers=np.array(answers), ldiff=np.array(ldiff),
                 last=np.concatenate(last), vis=np.concatenate(vis), n_cand=np.array(S["n_cand"]))
    hook.remove()


def probe_acc(X, y, groups, C=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return (pred == y).mean()


def analyze():
    from sklearn.metrics import roc_auc_score
    setup_style()
    import matplotlib.pyplot as plt
    res = {}
    for name, q in QUESTIONS.items():
        z = np.load(os.path.join(VDIR, f"beh_{name}.npz"))
        y, items = z["label"], z["items"]
        parsed = np.array([parse(a, q["options"]) for a in z["answers"]])
        # option 0 corresponds to label 1
        verbal_pred = np.where(parsed == 0, 1, np.where(parsed == 1, 0, -1))
        acc = float((verbal_pred == y).mean())
        bal = float(0.5 * ((verbal_pred[y == 1] == 1).mean() + (verbal_pred[y == 0] == 0).mean()))
        auc = float(roc_auc_score(y, z["ldiff"]))
        groups = items[:, 0]
        layer_acc = [probe_acc(z["last"][:, l].astype(np.float32), y, groups) for l in range(37)]
        vis_acc = probe_acc(z["vis"].astype(np.float32), y, groups)
        rng = np.random.default_rng(0)
        null = [probe_acc(z["last"][:, 18].astype(np.float32), rng.permutation(y), groups) for _ in range(5)]
        res[name] = dict(prompt=q["prompt"], ground_truth=q["gt"], n=int(len(y)), n_candidates=z["n_cand"].tolist(),
                         chance=0.5, verbal_acc=acc, verbal_balanced_acc=bal,
                         frac_unparsed=float((parsed < 0).mean()),
                         frac_answer_option0=float((parsed == 0).mean()),
                         logit_auc=auc, probe_acc_by_layer=[float(a) for a in layer_acc],
                         probe_best_layer=int(np.argmax(layer_acc)), probe_best_acc=float(np.max(layer_acc)),
                         probe_layer18_acc=float(layer_acc[18]), vision_probe_acc=float(vis_acc),
                         shuffled_label_probe_acc=float(np.mean(null)),
                         answer_examples=list(dict.fromkeys(z["answers"].tolist()))[:6])
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "probe_acc_by_layer"}, indent=None))
    with open(os.path.join(VDIR, "behaviour_results.json"), "w") as f:
        json.dump(res, f, indent=1)

    # figure: verbal vs logit vs probe
    names = list(QUESTIONS)
    labels = {"contact": "touching\nfloor/wall?", "floor_wall": "floor or\nwall?", "speed": "speeding up or\nslowing down?",
              "segment": "blue or orange\nrotating faster?", "anticipate": "will it hit\nnext?"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.2
    vals = [[res[n]["verbal_balanced_acc"] for n in names], [res[n]["logit_auc"] for n in names],
            [res[n]["vision_probe_acc"] for n in names], [res[n]["probe_best_acc"] for n in names]]
    labs = ["verbal answer (bal. acc.)", "answer logit (AUC)", "probe: vision enc. (acc.)", "probe: best LM layer (acc.)"]
    cols = [MUTED, "#b5b3ab", SERIES[2], SERIES[0]]
    for i, (v, l, c) in enumerate(zip(vals, labs, cols)):
        ax.bar(x + (i - 1.5) * w, v, w, label=l, color=c)
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.text(len(names) - 0.5, 0.51, "chance", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, [labels[n] for n in names], fontsize=7)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("accuracy / AUC")
    ax.set_title("A  What the VLM says vs. what it encodes")
    ax.legend(fontsize=6.5, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    for i, n in enumerate(names):
        ax.plot(res[n]["probe_acc_by_layer"], color=SERIES[i], lw=1.4, label=labels[n].replace("\n", " "))
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("LM layer (residual at last prompt token)")
    ax.set_ylabel("probe accuracy (5-fold)")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("B  Probe accuracy by depth")
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    savefig(fig, "vlm_behaviour.png")


if __name__ == "__main__":
    {"collect": collect, "analyze": analyze}[sys.argv[1]]()
