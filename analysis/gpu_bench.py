"""Throughput / VRAM benchmark for the VLM and video-model studies on the local accelerator.

Renders 8-frame 224 px clips of test episodes from data/main, runs them through
Qwen3-VL (video input) and V-JEPA 2 ViT-L with forward hooks on a few MLP layers,
and reports clips/s, peak VRAM and activation shapes.

    uv run python analysis/gpu_bench.py [--models qwen4b vjepa qwen8b] [--clips 32] [--batch 4]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import N_TRAIN_EPS, load_states  # noqa: E402
from analysis.device import default_dtype, get_device  # noqa: E402
from sim.arm import ArmRenderer  # noqa: E402

N_FRAMES, SIZE, STRIDE = 8, 224, 4    # 8 frames every 4th sim frame = 0.64 s of motion
QWEN = {"qwen4b": "Qwen/Qwen3-VL-4B-Instruct", "qwen8b": "Qwen/Qwen3-VL-8B-Instruct"}
VJEPA = "facebook/vjepa2-vitl-fpc64-256"
PROMPT = "Describe the motion of the arm."


def render_clips(n: int) -> list[np.ndarray]:
    st = load_states("main")
    r = ArmRenderer(SIZE)
    eps = np.arange(N_TRAIN_EPS, N_TRAIN_EPS + n)
    return [r(st["q"][e, 20:20 + N_FRAMES * STRIDE:STRIDE]) for e in eps]   # [T, H, W, 3] uint8


def mlp_modules(model, pattern: str, k: int) -> list[str]:
    """Pick k evenly spaced modules whose names match pattern (first, middle(s), last)."""
    names = [n for n, _ in model.named_modules() if re.fullmatch(pattern, n)]
    if not names:
        raise RuntimeError(f"no module matches {pattern}")
    idx = np.unique(np.linspace(0, len(names) - 1, k).round().astype(int))
    return [names[i] for i in idx]


class Hooks:
    def __init__(self, model, names):
        mods = dict(model.named_modules())
        self.acts, self.handles = {}, []
        for n in names:
            self.handles.append(mods[n].register_forward_hook(self._hook(n)))

    def _hook(self, name):
        def f(_m, _i, out):
            out = out[0] if isinstance(out, tuple) else out
            self.acts[name] = out.detach().float().cpu()
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


def summarize(acts: dict) -> list[str]:
    lines = []
    for n, a in acts.items():
        finite = torch.isfinite(a).all().item()
        lines.append(f"  {n}: shape {tuple(a.shape)}, mean {a.mean():+.3f}, std {a.std():.3f}, "
                     f"absmax {a.abs().max():.1f}, finite {finite}")
    return lines


def timed(fn, clips, batch, dev):
    """Run fn on batches; first batch is warm-up and excluded from timing."""
    fn(clips[:batch])
    if dev.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    acts = None
    for i in range(0, len(clips), batch):
        acts = fn(clips[i:i + batch])
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else float("nan")
    return len(clips) / dt, peak, acts


def bench_qwen(key, clips, batch, dev, out):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    kw = dict(attn_implementation="sdpa")
    if key == "qwen8b":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": 0}
    else:
        kw["dtype"] = default_dtype(dev)
    t = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(QWEN[key], **kw).eval()
    if key != "qwen8b":
        model.to(dev)
    load_s = time.perf_counter() - t
    proc = AutoProcessor.from_pretrained(QWEN[key])
    proc.tokenizer.padding_side = "left"
    names = [n for n, _ in model.named_modules()]
    out.append(f"- model load: {load_s:.1f} s, {len(names)} modules, "
               f"resident VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    vis = mlp_modules(model, r".*visual\.blocks\.\d+\.mlp", 3)
    llm = mlp_modules(model, r".*language_model\.layers\.\d+\.mlp", 3)
    print("\n".join(n for n in names if re.fullmatch(r".*(visual\.blocks|language_model\.layers)\.0\..*", n)))
    hooks = Hooks(model, vis + llm)
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def run(batch_clips):
        inp = proc(text=[text] * len(batch_clips), videos=list(batch_clips), padding=True,
                   videos_kwargs=dict(fps=50 / STRIDE, do_sample_frames=False),
                   return_tensors="pt").to(dev)
        model(**inp)
        return dict(hooks.acts), inp

    sps, peak, (acts, inp) = timed(run, clips, batch, dev)
    out += [f"- input_ids {tuple(inp['input_ids'].shape)}, pixel_values_videos "
            f"{tuple(inp['pixel_values_videos'].shape)}, video_grid_thw {inp['video_grid_thw'][0].tolist()}",
            f"- **{sps:.2f} clips/s** (batch {batch}), peak VRAM **{peak:.2f} GiB**",
            "- hooked activations (last batch):"] + summarize(acts)
    hooks.remove()
    del model
    torch.cuda.empty_cache()


def bench_vjepa(clips, batch, dev, out):
    from transformers import AutoModel, AutoVideoProcessor
    t = time.perf_counter()
    model = AutoModel.from_pretrained(VJEPA, dtype=default_dtype(dev), attn_implementation="sdpa").to(dev).eval()
    proc = AutoVideoProcessor.from_pretrained(VJEPA)
    out.append(f"- model load: {time.perf_counter() - t:.1f} s, "
               f"resident VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    names = mlp_modules(model, r".*encoder\.layer\.\d+\.mlp", 4)
    print("\n".join(n for n, _ in model.named_modules() if re.fullmatch(r".*encoder\.layer\.0\..*", n)))
    hooks = Hooks(model, names)

    @torch.inference_mode()
    def run(batch_clips):
        # [B, T, H, W, 3] -> processor resizes/crops to 256 and normalizes -> [B, T, C, H, W]
        inp = proc(list(batch_clips), return_tensors="pt").to(dev)
        pv = inp["pixel_values_videos"].to(default_dtype(dev))
        model.get_vision_features(pv)
        return dict(hooks.acts), pv

    sps, peak, (acts, pv) = timed(run, clips, batch, dev)
    out += [f"- pixel_values_videos {tuple(pv.shape)}",
            f"- **{sps:.2f} clips/s** (batch {batch}), peak VRAM **{peak:.2f} GiB**",
            "- hooked activations (last batch):"] + summarize(acts)
    hooks.remove()
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen4b", "vjepa", "qwen8b"])
    ap.add_argument("--clips", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    dev = get_device()
    t = time.perf_counter()
    clips = render_clips(args.clips)
    out = [f"### Clips", f"- {args.clips} test episodes, {N_FRAMES} frames x {SIZE} px, "
           f"rendered in {time.perf_counter() - t:.1f} s on CPU"]
    for m in args.models:
        out.append(f"\n### {QWEN.get(m, VJEPA)}")
        try:
            bench_vjepa(clips, args.batch, dev, out) if m == "vjepa" else bench_qwen(m, clips, args.batch, dev, out)
        except Exception as e:   # keep benchmarking the other models; the error goes in the report
            out.append(f"- **FAILED**: {type(e).__name__}: {e}")
        print("\n".join(out), flush=True)
    print("\n=== RESULT ===\n" + "\n".join(out))


if __name__ == "__main__":
    main()
