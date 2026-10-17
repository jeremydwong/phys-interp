"""Re-run Qwen3-VL on chosen clips and capture full spatial maps of chosen units
(used by the gallery, contact and generalisation analyses)."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.vlm_common import BS, GRID, GROUPS, TOK, build_video_inputs  # noqa: E402
from analysis.vlm_extract import vis_group_view  # noqa: E402


def capture(model, proc, clips, units, res_layers=(), bs=None, prompt=None):
    """clips: list of uint8 [8,H,W,3]; units: list of (part 'vision'|'LM', layer, unit_idx).
    Returns maps: list per unit of array [n_clips, 4, h, w] (vision 20x20, LM 10x10),
    and res: dict layer-> [n_clips, 4, D] mean-pooled residuals ('V{l}' / 'L{l}')."""
    bs = bs or BS
    store = {}
    hooks = []
    vis = model.model.visual
    lm = model.model.language_model
    vl = sorted({l for p, l, u in units if p == "vision"})
    ll = sorted({l for p, l, u in units if p == "LM"})
    for l in vl:
        hooks.append(vis.blocks[l].mlp.act_fn.register_forward_hook(lambda m, i, o, l=l: store.__setitem__(f"v{l}", o.detach())))
    for l in ll:
        hooks.append(lm.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: store.__setitem__(f"l{l}", i[0].detach())))
    for r in res_layers:
        if r.startswith("V"):
            l = int(r[1:])
            hooks.append(vis.blocks[l].register_forward_hook(lambda m, i, o, r=r: store.__setitem__(r, o.detach())))
    vid_id = proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    maps = [[] for _ in units]
    res = {r: [] for r in res_layers}
    need_hs = any(r.startswith("L") for r in res_layers)
    try:
        for bi in range(0, len(clips), bs):
            cb = clips[bi:bi + bs]
            B = len(cb)
            kw = {} if prompt is None else dict(prompt=prompt)
            inp = build_video_inputs(proc, cb, **kw).to(model.device)
            with torch.no_grad():
                o = model(**inp, output_hidden_states=need_hs, logits_to_keep=1)
            ids = inp["input_ids"]
            pos = torch.stack([torch.where(ids[b] == vid_id)[0] for b in range(B)]).reshape(B, GROUPS, TOK * TOK)
            for k, (p, l, u) in enumerate(units):
                if p == "vision":
                    x = vis_group_view(store[f"v{l}"][:, [u]], B, GROUPS)[..., 0]       # [B,G,400]
                    maps[k].append(x.float().reshape(B, GROUPS, GRID, GRID).cpu().numpy())
                else:
                    h = store[f"l{l}"][..., u]                                            # [B,S]
                    x = torch.stack([h[b, pos[b]] for b in range(B)])                     # [B,G,100]
                    maps[k].append(x.float().reshape(B, GROUPS, TOK, TOK).cpu().numpy())
            for r in res_layers:
                if r.startswith("V"):
                    res[r].append(store[r].reshape(B, GROUPS, GRID * GRID, -1).float().mean(2).cpu().numpy())
                else:
                    hs = o.hidden_states[int(r[1:])]
                    res[r].append(torch.stack([hs[b, pos[b]].float().mean(1) for b in range(B)]).cpu().numpy())
            store.clear()
    finally:
        for h in hooks:
            h.remove()
    return [np.concatenate(m) for m in maps], {r: np.concatenate(v) for r, v in res.items()}
