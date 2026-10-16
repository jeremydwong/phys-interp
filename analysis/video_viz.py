"""Drawing helpers for video-model figures: motion-trail composites and activation overlays."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from analysis.common import AXIS


def arm_mask(img):
    """img [...,64,64,3] float in [0,1] -> bool mask of saturated (arm) pixels."""
    return (img.max(-1) - img.min(-1)) > 0.25


def trail(clip):
    """clip [4, 64, 64, 3] float -> composite of frame t with faint ghosts of t-3..t-1."""
    out = clip[-1].copy()
    now = arm_mask(clip[-1])
    for k, a in zip(range(3), (0.18, 0.28, 0.42)):
        m = arm_mask(clip[k]) & ~now
        out[m] = (1 - a) * out[m] + a * clip[k][m]
    return out


def upsample_map(amap, size=64):
    t = torch.as_tensor(np.asarray(amap, np.float32))[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def overlay(ax, clip, amap, vmax, color=(0.85, 0.1, 0.55), show_arg=True, vmin=0.0):
    """Draw trail composite with the unit's spatial activation map overlaid (scaled vmin..vmax)."""
    img = trail(clip)
    up = np.clip((upsample_map(amap) - vmin) / (vmax - vmin + 1e-8), 0, 1)
    rgba = np.zeros((64, 64, 4))
    rgba[..., :3] = color
    rgba[..., 3] = 0.6 * up ** 1.5
    ax.imshow(img, interpolation="bilinear")
    ax.imshow(rgba, interpolation="bilinear")
    if show_arg:
        r, c = np.unravel_index(np.argmax(amap), amap.shape)
        ax.plot(c * 4 + 1.5, r * 4 + 1.5, marker="+", ms=4, mew=0.8, color=color)
    clean(ax)


def clean(ax, frame=True):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(frame)
        s.set_color(AXIS); s.set_linewidth(0.5)
