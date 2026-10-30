"""One extra GPU pass over the full 2700-clip corpus for the essay tool fixes (E1, E2, B2).

Same clips, prompt and protocol as vlm_extract (8 frames, stride 2, 320 px, 4 temporal groups).
Hooks:
  * LM layers LLM_LAYERS: gate_proj and up_proj outputs. At the visual tokens of each temporal
    group we recompute a = silu(g) * v, take the token of the pooled max and of the pooled min
    (the census poles), and store the SwiGLU sign regime of that token:
        code = 2 * (g < 0) + (v < 0)      0: +x+   1: +x-   2: -x+   3: -x-
    -> video_llm_reg_L{l}.npy int8 [N_G, 2, 9728]   (stat 0 = max token, 1 = min token)
    A consistency check (recomputed argmax token == cached argmax token) is logged.
  * Vision blocks VIS_LAYERS: linear_fc1 output (pre-GELU), pooled over the 400 patches of a group
    -> video_vis_pre_L{l}.npy fp16 [N_G, 3, 4096] (max, min, mean), _arg int16 [N_G, 2, 4096]
  * Residual energy (E2): for every vision block output (24) and LM hidden state (37), per
    coordinate sum of x^2 over tokens, plus per-token "largest coordinate share" statistics.
    LM statistics are kept for visual tokens and for all tokens. -> sink_stats.npz

Usage: uv run python analysis/vlm_recapture.py [limit]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import load_states  # noqa: E402
from analysis.vlm_common import (BS, GROUPS, LLM_LAYERS, VDIR, VIS_LAYERS, build_video_inputs, clip_index,  # noqa: E402
                                 load_model, renderer, free_memory)
from analysis.vlm_extract import mm, vis_group_view, NP  # noqa: E402


def main(limit=None, bs=None):
    bs = bs or BS
    st = load_states()
    ci = clip_index()
    NC = len(ci)
    NG = NC * GROUPS
    model, proc = load_model()
    vis, lm = model.model.visual, model.model.language_model
    store = {}
    hooks = []
    for l in LLM_LAYERS:
        hooks.append(lm.layers[l].mlp.gate_proj.register_forward_hook(lambda m, i, o, l=l: store.__setitem__(f"g{l}", o.detach())))
        hooks.append(lm.layers[l].mlp.up_proj.register_forward_hook(lambda m, i, o, l=l: store.__setitem__(f"u{l}", o.detach())))
    for l in VIS_LAYERS:
        hooks.append(vis.blocks[l].mlp.linear_fc1.register_forward_hook(lambda m, i, o, l=l: store.__setitem__(f"p{l}", o.detach())))
    for l, b in enumerate(vis.blocks):
        hooks.append(b.register_forward_hook(lambda m, i, o, l=l: store.__setitem__(f"vr{l}", (o[0] if isinstance(o, tuple) else o).detach())))
    reg = {l: mm(f"video_llm_reg_L{l}.npy", (NG, 2, 9728), np.int8) for l in LLM_LAYERS}
    pre = {l: (mm(f"video_vis_pre_L{l}.npy", (NG, 3, 4096)), mm(f"video_vis_pre_L{l}_arg.npy", (NG, 2, 4096), np.int16))
           for l in VIS_LAYERS}
    cached_arg = {l: np.load(os.path.join(VDIR, f"video_llm_mlp_L{l}_arg.npy"), mmap_mode="r") for l in LLM_LAYERS}
    match = {l: [0, 0] for l in LLM_LAYERS}
    # energy accumulators (float64 on GPU)
    dev = model.device
    E = {"vis": torch.zeros(24, 1024, dtype=torch.float64, device=dev),
         "lm_vis": torch.zeros(37, 2560, dtype=torch.float64, device=dev),
         "lm_all": torch.zeros(37, 2560, dtype=torch.float64, device=dev)}
    # per-token largest-coordinate share: count of tokens whose top coordinate holds >50% / >10% of x^2,
    # and a histogram of which coordinate is the top one
    S50 = {k: torch.zeros(v.shape[0], dtype=torch.float64, device=dev) for k, v in E.items()}
    S10 = {k: torch.zeros(v.shape[0], dtype=torch.float64, device=dev) for k, v in E.items()}
    TOPC = {k: torch.zeros(v.shape, dtype=torch.float64, device=dev) for k, v in E.items()}
    NTOK = {k: 0 for k in E}
    donep = os.path.join(VDIR, "recapture_done.npy")
    done = np.load(donep) if os.path.exists(donep) else np.zeros(NC, bool)
    if done.any():
        sp = os.path.join(VDIR, "sink_stats_partial.npz")
        if os.path.exists(sp):
            z = np.load(sp, allow_pickle=True)
            for k in E:
                E[k] += torch.as_tensor(z[f"E_{k}"], device=dev)
                S50[k] += torch.as_tensor(z[f"S50_{k}"], device=dev)
                S10[k] += torch.as_tensor(z[f"S10_{k}"], device=dev)
                TOPC[k] += torch.as_tensor(z[f"TOPC_{k}"], device=dev)
                NTOK[k] = int(z[f"NTOK_{k}"])
            match = z["match"].item()
        else:
            done[:] = False
    todo = np.where(~done)[0][:limit]
    vid_id = proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    def acc(key, X):
        """X: list over layers of [T, D] activations (one layer at a time keeps VRAM low)."""
        for li, x in enumerate(X):
            x2 = x.float() ** 2
            E[key][li] += x2.double().sum(0)
            share, arg = (x2 / (x2.sum(-1, keepdim=True) + 1e-30)).max(-1)      # [T]
            S50[key][li] += float((share > 0.5).sum())
            S10[key][li] += float((share > 0.1).sum())
            TOPC[key][li].scatter_add_(0, arg, torch.ones_like(arg, dtype=torch.float64))
            del x2
        NTOK[key] += X[0].shape[0]

    def save_partial():
        np.savez(os.path.join(VDIR, "sink_stats_partial.npz"),
                 **{f"E_{k}": E[k].cpu().numpy() for k in E}, **{f"S50_{k}": S50[k].cpu().numpy() for k in E},
                 **{f"S10_{k}": S10[k].cpu().numpy() for k in E}, **{f"TOPC_{k}": TOPC[k].cpu().numpy() for k in E},
                 **{f"NTOK_{k}": NTOK[k] for k in E}, match=np.array(match, dtype=object))

    t0 = time.time()
    for bi in range(0, len(todo), bs):
        rows = todo[bi:bi + bs]
        B = len(rows)
        clips = [renderer()(st["q"][ci[r, 0]][ci[r, 2] + 2 * np.arange(8)]) for r in rows]
        inp = build_video_inputs(proc, clips).to(dev)
        with torch.no_grad():
            o = model(**inp, output_hidden_states=True, logits_to_keep=1)
            grows = (rows[:, None] * GROUPS + np.arange(GROUPS)).ravel()
            ids = inp["input_ids"]
            pos = torch.stack([torch.where(ids[b] == vid_id)[0] for b in range(B)]).reshape(B, GROUPS, 100)
            for l in LLM_LAYERS:
                g = torch.stack([store[f"g{l}"][b, pos[b]] for b in range(B)]).float()   # [B,G,100,U]
                v = torch.stack([store[f"u{l}"][b, pos[b]] for b in range(B)]).float()
                a = torch.nn.functional.silu(g) * v
                ia, ib = a.argmax(2), a.argmin(2)                                       # [B,G,U]
                code = (2 * (g < 0) + (v < 0)).to(torch.int8)                           # [B,G,100,U]
                cmax = code.gather(2, ia[:, :, None]).squeeze(2)
                cmin = code.gather(2, ib[:, :, None]).squeeze(2)
                reg[l][grows] = torch.stack([cmax, cmin], 2).reshape(B * GROUPS, 2, -1).cpu().numpy()
                ca = np.asarray(cached_arg[l][grows, 0])
                match[l][0] += int((ia.reshape(B * GROUPS, -1).cpu().numpy() == ca).sum())
                match[l][1] += ca.size
                del g, v, a, code
            for l in VIS_LAYERS:
                x = vis_group_view(store[f"p{l}"], B, GROUPS).float()                    # [B,G,400,U]
                mx, ia = x.max(2)
                mn, ib = x.min(2)
                pre[l][0][grows] = torch.stack([mx, mn, x.mean(2)], 2).reshape(B * GROUPS, 3, -1).half().cpu().numpy()
                pre[l][1][grows] = torch.stack([ia, ib], 2).reshape(B * GROUPS, 2, -1).short().cpu().numpy()
                del x
            acc("vis", [store[f"vr{l}"] for l in range(24)])
            hs = o.hidden_states                                                         # 37 x [B,S,D]
            acc("lm_vis", [torch.cat([h[b, pos[b].reshape(-1)] for b in range(B)]) for h in hs])
            acc("lm_all", [h.reshape(-1, h.shape[-1]) for h in hs])
            del o, hs
        store.clear()
        done[rows] = True
        if (bi // bs) % 50 == 0:
            for l in LLM_LAYERS:
                reg[l].flush()
            for l in VIS_LAYERS:
                pre[l][0].flush(); pre[l][1].flush()
            np.save(donep, done)
            save_partial()
            el = time.time() - t0
            m = {l: round(match[l][0] / max(match[l][1], 1), 4) for l in LLM_LAYERS}
            print(f"{bi + B}/{len(todo)} {el / (bi + B):.2f}s/clip argmax-match {m} "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    for l in LLM_LAYERS:
        reg[l].flush()
    for l in VIS_LAYERS:
        pre[l][0].flush(); pre[l][1].flush()
    np.save(donep, done)
    save_partial()
    for h in hooks:
        h.remove()
    if done.all():
        np.savez(os.path.join(VDIR, "sink_stats.npz"),
                 **{f"E_{k}": E[k].cpu().numpy() for k in E}, **{f"S50_{k}": S50[k].cpu().numpy() for k in E},
                 **{f"S10_{k}": S10[k].cpu().numpy() for k in E}, **{f"TOPC_{k}": TOPC[k].cpu().numpy() for k in E},
                 **{f"NTOK_{k}": NTOK[k] for k in E},
                 argmax_match=np.array([match[l][0] / max(match[l][1], 1) for l in LLM_LAYERS]))
    print("done; argmax match", {l: match[l][0] / max(match[l][1], 1) for l in LLM_LAYERS}, flush=True)
    del model
    free_memory()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
