"""Run Qwen3-VL-4B on arm clips and cache pooled activations under data/vlm/.

Usage:
  uv run python analysis/vlm_extract.py video      # 8-frame clips (main dataset)
  uv run python analysis/vlm_extract.py image      # single midpoint frame of every temporal group
  uv run python analysis/vlm_extract.py randvis    # randomly re-initialised vision encoder, same clips

Stored (float16, N_G = 300 eps x 9 clips x 4 groups = 10800 temporal groups):
  {mode}_vis_mlp_L{l}.npy   [N_G, 3, 4096]  vision MLP hidden (post-GELU) pooled over the 400
                                            patches of a group: (max, min, mean)
  {mode}_vis_mlp_L{l}_arg.npy [N_G, 2, 4096] int16 patch index of max / min (0..399, raster order)
  {mode}_llm_mlp_L{l}.npy   [N_G, 3, 9728]  LM SwiGLU hidden silu(gate)*up at the 100 visual tokens
  {mode}_llm_mlp_L{l}_arg.npy [N_G, 2, 9728] token index of max / min (0..99, raster order)
  {mode}_vis_res.npy        [N_G, 24, 1024] vision residual (block outputs), mean over patches
  {mode}_llm_res.npy        [N_G, 37, 2560] LM residual stream (hidden_states), mean over group tokens
  {mode}_llm_last.npy       [N_clip, 37, 2560] LM residual at the last prompt token
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import load_states  # noqa: E402
from analysis.vlm_common import (GRID, GROUPS, LLM_LAYERS, RES, VDIR, VIS_LAYERS, clip_index,  # noqa: E402
                                 group_index, load_model, build_video_inputs, build_image_inputs,
                                 vision_patch_to_grid, renderer)

from analysis.vlm_common import get_device  # noqa: E402
DEV = get_device()
NP = GRID * GRID   # 400 patches per temporal group


def mm(name, shape, dtype=np.float16):
    path = os.path.join(VDIR, name)
    if os.path.exists(path):
        return np.load(path, mmap_mode="r+")
    return np.lib.format.open_memmap(path, "w+", dtype, shape)


class Recorder:
    def __init__(self, model, vis_layers, llm_layers, vis_res=True):
        self.store, self.hooks = {}, []
        vis = model.model.visual
        for l in vis_layers:
            self.hooks.append(vis.blocks[l].mlp.act_fn.register_forward_hook(self._save(f"vmlp{l}")))
        if vis_res:
            for l, b in enumerate(vis.blocks):
                self.hooks.append(b.register_forward_hook(self._save(f"vres{l}")))
        if llm_layers:
            lm = model.model.language_model
            for l in llm_layers:
                self.hooks.append(lm.layers[l].mlp.down_proj.register_forward_pre_hook(self._save_in(f"lmlp{l}")))

    def _save(self, k):
        def f(m, i, o):
            self.store[k] = (o[0] if isinstance(o, tuple) else o).detach()
        return f

    def _save_in(self, k):
        def f(m, i):
            self.store[k] = i[0].detach()
        return f

    def remove(self):
        for h in self.hooks:
            h.remove()


def pool3(x):
    """x [..., P, U] -> (stack(max,min,mean) [...,3,U] fp16, argmax/argmin [...,2,U] int16)."""
    x = x.float()
    mx, ia = x.max(-2)
    mn, ib = x.min(-2)
    me = x.mean(-2)
    return (torch.stack([mx, mn, me], -2).half().cpu().numpy(),
            torch.stack([ia, ib], -2).short().cpu().numpy())


def vis_group_view(x, n_items, n_groups):
    """vision tensor [n_items*n_groups*NP, U] (patch order t,hb,wb,mh,mw) -> [n_items, n_groups, NP(raster), U]."""
    x = x.reshape(n_items, n_groups, NP, -1)
    g = vision_patch_to_grid(1)
    raster = torch.as_tensor(g[:, 1] * GRID + g[:, 2], device=x.device)
    out = torch.empty_like(x)
    out[:, :, raster] = x
    return out


def run_video(mode="video", bs=4, limit=None):
    st = load_states()
    ci, gi = clip_index(), group_index()
    NC, NG = len(ci), len(gi)
    model, proc = load_model(DEV)
    if mode == "randvis":
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        torch.manual_seed(0)
        rv = Qwen3VLVisionModel._from_config(model.config.vision_config, dtype=model.dtype)
        rv.apply(model._init_weights)
        model.model.visual = rv.to(DEV).eval()
    vis_layers, llm_layers = VIS_LAYERS, ([] if mode == "randvis" else LLM_LAYERS)
    rec = Recorder(model, vis_layers, llm_layers)
    out = {}
    for l in vis_layers:
        out[f"vmlp{l}"] = (mm(f"{mode}_vis_mlp_L{l}.npy", (NG, 3, 4096)), mm(f"{mode}_vis_mlp_L{l}_arg.npy", (NG, 2, 4096), np.int16))
    for l in llm_layers:
        out[f"lmlp{l}"] = (mm(f"{mode}_llm_mlp_L{l}.npy", (NG, 3, 9728)), mm(f"{mode}_llm_mlp_L{l}_arg.npy", (NG, 2, 9728), np.int16))
    vres = mm(f"{mode}_vis_res.npy", (NG, 24, 1024))
    if mode != "randvis":
        lres = mm(f"{mode}_llm_res.npy", (NG, 37, 2560))
        llast = mm(f"{mode}_llm_last.npy", (NC, 37, 2560))
    donep = os.path.join(VDIR, f"{mode}_done.npy")
    done = np.load(donep) if os.path.exists(donep) else np.zeros(NC, bool)
    todo = np.where(~done)[0][:limit]
    vid_id = proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    t0 = time.time()
    for bi in range(0, len(todo), bs):
        rows = todo[bi:bi + bs]
        clips = [renderer()(st["q"][ci[r, 0]][ci[r, 2] + 2 * np.arange(8)]) for r in rows]
        inp = build_video_inputs(proc, clips).to(DEV)
        B = len(rows)
        with torch.no_grad():
            if mode == "randvis":
                model.model.visual(inp["pixel_values_videos"].to(model.dtype), grid_thw=inp["video_grid_thw"])
            else:
                o = model(**inp, output_hidden_states=True, logits_to_keep=1)
        grows = (rows[:, None] * GROUPS + np.arange(GROUPS)).ravel()
        for l in vis_layers:
            v, a = pool3(vis_group_view(rec.store[f"vmlp{l}"], B, GROUPS))
            out[f"vmlp{l}"][0][grows] = v.reshape(B * GROUPS, 3, -1)
            out[f"vmlp{l}"][1][grows] = a.reshape(B * GROUPS, 2, -1)
        vr = torch.stack([rec.store[f"vres{l}"].reshape(B, GROUPS, NP, -1).float().mean(2) for l in range(24)], 2)
        vres[grows] = vr.reshape(B * GROUPS, 24, -1).half().cpu().numpy()
        if mode != "randvis":
            ids = inp["input_ids"]
            pos = torch.stack([torch.where(ids[b] == vid_id)[0] for b in range(B)])   # [B, 400]
            assert pos.shape[1] == GROUPS * 100
            idx = pos.reshape(B, GROUPS, 100)
            for l in llm_layers:
                h = rec.store[f"lmlp{l}"]
                hv = torch.stack([h[b, idx[b]] for b in range(B)])       # [B, G, 100, U]
                v, a = pool3(hv)
                out[f"lmlp{l}"][0][grows] = v.reshape(B * GROUPS, 3, -1)
                out[f"lmlp{l}"][1][grows] = a.reshape(B * GROUPS, 2, -1)
            hs = torch.stack(o.hidden_states, 1)                          # [B, 37, S, D]
            lr = torch.stack([hs[b][:, idx[b]].float().mean(-2) for b in range(B)])   # [B, 37, G, D]
            lres[grows] = lr.permute(0, 2, 1, 3).reshape(B * GROUPS, 37, -1).half().cpu().numpy()
            llast[rows] = hs[:, :, -1].float().half().cpu().numpy()
        rec.store.clear()
        done[rows] = True
        if (bi // bs) % 25 == 0:
            np.save(donep, done)
            el = time.time() - t0
            print(f"{mode} {bi + B}/{len(todo)}  {el / (bi + B):.2f}s/clip", flush=True)
    np.save(donep, done)
    for v in out.values():
        v[0].flush(); v[1].flush()
    print("done", mode)


def run_image(bs=16, limit=None):
    """Single frame (the aligned midpoint frame of each temporal group) as an image input.
    Qwen duplicates a still image across the temporal patch, so each image = 1 group of 400 patches."""
    st = load_states()
    gi = group_index()
    NG = len(gi)
    model, proc = load_model(DEV)
    rec = Recorder(model, VIS_LAYERS, [])
    out = {l: (mm(f"image_vis_mlp_L{l}.npy", (NG, 3, 4096)), mm(f"image_vis_mlp_L{l}_arg.npy", (NG, 2, 4096), np.int16))
           for l in VIS_LAYERS}
    vres = mm("image_vis_res.npy", (NG, 24, 1024))
    lres = mm("image_llm_res.npy", (NG, 37, 2560))
    donep = os.path.join(VDIR, "image_done.npy")
    done = np.load(donep) if os.path.exists(donep) else np.zeros(NG, bool)
    todo = np.where(~done)[0][:limit]
    img_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    t0 = time.time()
    for bi in range(0, len(todo), bs):
        rows = todo[bi:bi + bs]
        B = len(rows)
        imgs = [renderer()(st["q"][gi[r, 0]][[gi[r, 3]]])[0] for r in rows]
        inp = build_image_inputs(proc, imgs).to(DEV)
        with torch.no_grad():
            o = model(**inp, output_hidden_states=True, logits_to_keep=1)
        for l in VIS_LAYERS:
            v, a = pool3(vis_group_view(rec.store[f"vmlp{l}"], B, 1))
            out[l][0][rows] = v[:, 0]
            out[l][1][rows] = a[:, 0]
        vres[rows] = torch.stack([rec.store[f"vres{l}"].reshape(B, NP, -1).float().mean(1) for l in range(24)], 1).half().cpu().numpy()
        ids = inp["input_ids"]
        pos = torch.stack([torch.where(ids[b] == img_id)[0] for b in range(B)])
        hs = torch.stack(o.hidden_states, 1)
        lres[rows] = torch.stack([hs[b][:, pos[b]].float().mean(-2) for b in range(B)]).half().cpu().numpy()
        rec.store.clear()
        done[rows] = True
        if (bi // bs) % 25 == 0:
            np.save(donep, done)
            print(f"image {bi + B}/{len(todo)} {(time.time() - t0) / (bi + B):.3f}s/img", flush=True)
    np.save(donep, done)
    print("done image")


if __name__ == "__main__":
    mode = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if mode == "image":
        run_image(limit=limit)
    else:
        run_video(mode, limit=limit)
