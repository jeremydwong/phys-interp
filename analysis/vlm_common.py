"""Shared machinery for the VLM study (Qwen3-VL-4B-Instruct on two-link-arm videos).

Stimulus protocol (see reports/results/vlm.md):
  * 300 test episodes (seeded draw from [2000, 2400)).
  * Each episode is cut into 9 consecutive, non-overlapping clips. Clip c has
    start frame s = 16c and shows the 8 sim frames s, s+2, ..., s+14 (stride 2 -> 25 fps,
    0.30 s span), rendered at 320x320 px with sim.arm.ArmRenderer.
  * Qwen3-VL's patch embedding is a Conv3d with temporal_patch_size=2, so the 8 frames
    become 4 temporal groups; group g contains frames s+4g and s+4g+2 and is aligned to
    the ground truth of the sim frame between them, s+4g+1 (exact temporal midpoint).
  * 320 px / 16 px patches = 20x20 vision patches per group (400); the 2x2 spatial merger
    turns these into 10x10 = 100 LLM tokens per group.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.common import DATA  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
VDIR = os.path.join(DATA, "vlm")
os.makedirs(VDIR, exist_ok=True)

RES = 320                    # px per side
N_FRAMES = 8                 # frames per clip
STRIDE = 2                   # sim frames between clip frames (50 Hz / 2 = 25 fps)
CLIP_FPS = 50.0 / STRIDE
CLIPS_PER_EP = 9
CLIP_SPAN = 16               # start-frame spacing of consecutive clips
N_EPS = 300
GROUPS = N_FRAMES // 2       # temporal groups per clip
PATCH, MERGE = 16, 2
GRID = RES // PATCH          # 20 vision patches per side
TOK = GRID // MERGE          # 10 LLM tokens per side
PROMPT = "Describe the motion of the arm."

VIS_LAYERS = [3, 8, 13, 18, 23]        # vision blocks whose MLP hidden units we store (0-indexed, of 24)
LLM_LAYERS = [2, 8, 16, 24, 32]        # LM decoder layers whose MLP hidden units we store (of 36)


def episodes() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(np.arange(2000, 2400), N_EPS, replace=False))


def clip_index():
    """Rows = clips: (episode, clip_no, start_frame)."""
    rows = [(e, c, CLIP_SPAN * c) for e in episodes() for c in range(CLIPS_PER_EP)]
    return np.array(rows, dtype=np.int64)


def clip_frames(start: int) -> np.ndarray:
    return start + STRIDE * np.arange(N_FRAMES)


def group_index():
    """Rows = temporal groups: (episode, clip_row, group, aligned_frame, frame_a, frame_b)."""
    ci = clip_index()
    rows = []
    for r, (e, c, s) in enumerate(ci):
        for g in range(GROUPS):
            rows.append((e, r, g, s + 4 * g + 1, s + 4 * g, s + 4 * g + 2))
    return np.array(rows, dtype=np.int64)


# ------------------------------------------------------------------ rendering

_REND = {}


def renderer(res=RES, **kw):
    from sim.arm import ArmRenderer
    key = (res, tuple(sorted(kw.items())))
    if key not in _REND:
        _REND[key] = ArmRenderer(res, **kw)
    return _REND[key]


def render_clip(q_ep: np.ndarray, start: int, res=RES) -> np.ndarray:
    return renderer(res)(q_ep[clip_frames(start)])


# ------------------------------------------------------------------ model

def get_device() -> str:
    """cuda, then mps, then cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEV = None


def load_model(device=None):
    import torch
    global DEV
    device = device or get_device()
    DEV = device
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16 if device != "cpu" else torch.float32).to(device).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    return model, proc


def build_video_inputs(proc, clips: list[np.ndarray], prompt: str = PROMPT, add_generation_prompt=True):
    """clips: list of uint8 [8, H, W, 3]. Returns processor batch (no frame resampling/resizing)."""
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    n = clips[0].shape[0]
    meta = [{"fps": CLIP_FPS, "total_num_frames": n, "frames_indices": list(range(n))} for _ in clips]
    px = clips[0].shape[1] * clips[0].shape[2]
    return proc(text=[text] * len(clips), videos=[c for c in clips], video_metadata=meta,
                do_sample_frames=False, return_tensors="pt", padding=True,
                videos_kwargs=dict(size={"shortest_edge": 64 * 64, "longest_edge": px * n}, cap_pixels_per_frame=False))


def build_image_inputs(proc, imgs: list[np.ndarray], prompt: str = PROMPT, add_generation_prompt=True):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    px = imgs[0].shape[0] * imgs[0].shape[1]
    from PIL import Image
    return proc(text=[text] * len(imgs), images=[Image.fromarray(i) for i in imgs], return_tensors="pt",
                padding=True, images_kwargs=dict(size={"shortest_edge": 64 * 64, "longest_edge": px}))


def vision_patch_to_grid(n_t: int, gh: int = GRID, gw: int = GRID) -> np.ndarray:
    """Qwen3-VL orders vision patches as (t, h/2, w/2, 2, 2). Returns for each flat patch index
    its (t, row, col)."""
    t, hb, wb, mh, mw = np.meshgrid(np.arange(n_t), np.arange(gh // MERGE), np.arange(gw // MERGE),
                                    np.arange(MERGE), np.arange(MERGE), indexing="ij")
    return np.stack([t.ravel(), (hb * MERGE + mh).ravel(), (wb * MERGE + mw).ravel()], 1)


def n_used(mode: str = "video"):
    """Analyses use the leading block of fully extracted episodes (extraction runs episode-major).
    Returns (n_groups, n_clips)."""
    p = os.path.join(VDIR, f"{mode}_done.npy")
    done = np.load(p)
    k = int(np.argmin(done)) if not done.all() else len(done)
    per = CLIPS_PER_EP if len(done) == N_EPS * CLIPS_PER_EP else CLIPS_PER_EP * GROUPS
    k_eps = k // per
    return k_eps * CLIPS_PER_EP * GROUPS, k_eps * CLIPS_PER_EP
