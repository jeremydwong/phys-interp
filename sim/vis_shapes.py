"""Procedural shapes dataset with ground-truth generative factors (for the vis_ ladder study).

Every 64x64 RGB image holds one object on a grey background. Factors (all logged):
  shape     categorical 6: circle, square, triangle, arc, cross, bar   (circle and arc are the curved ones)
  color     categorical 6: hue k*60 deg (red, yellow, green, cyan, blue, magenta), jittered sat/value
  texture   categorical 6: solid, stripes 0/45/90/135 deg, dots       (solid w.p. 0.5)
  bg_type   categorical 3: plain, linear gradient, smooth noise (background is grey, no hue)
  size      continuous: object radius as a fraction of the image width, U[0.16, 0.34]
  x, y      continuous: object centre, U[0.3, 0.7]
  rot_phase continuous: rotation modulo the shape's symmetry period, in [0, 1) (meaningless for circles)
  bg_lum    continuous: background luminance U[0.1, 0.9]
Derived: curved (shape in {circle, arc}), warm (red/yellow/magenta), textured (texture != solid).

Class label (24 classes) = shape * 4 + warm * 2 + textured, so a classifier has to read shape,
hue and texture, while size, position, rotation and background are nuisances.

Output: data/vis_shapes/images.npy (uint8 [N, 64, 64, 3]) and factors.npz.
Images [0, 40000) are train, [40000, 50000) test. Image i is drawn from seed 1_000_000 + i,
so the dataset is deterministic and regenerable:  uv run python sim/vis_shapes.py
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "vis_shapes")
N_TRAIN, N_TEST = 40000, 10000
RES, SS = 64, 2                      # final resolution, supersampling factor
SHAPES = ["circle", "square", "triangle", "arc", "cross", "bar"]
SYM = {"circle": 1, "square": 4, "triangle": 3, "arc": 1, "cross": 4, "bar": 2}   # rotational symmetry order
COLORS = ["red", "yellow", "green", "cyan", "blue", "magenta"]
WARM = {0, 1, 5}
TEXTURES = ["solid", "stripes0", "stripes45", "stripes90", "stripes135", "dots"]
BGS = ["plain", "gradient", "noise"]
CATEGORICAL = ["shape", "color", "texture", "bg_type", "curved", "warm", "textured", "label"]
CONTINUOUS = ["size", "x", "y", "rot_phase", "bg_lum"]
N_CLASSES = 24


def hsv_to_rgb(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return np.array([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i], np.float32)


def _rot(pts, a):
    c, s = np.cos(a), np.sin(a)
    return pts @ np.array([[c, s], [-s, c]])


def shape_polygon(shape, r, a):
    """Vertices (unit-centred, radius r) of the shape rotated by a, or None for the circle."""
    if shape == "square":
        h = 0.82 * r
        p = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    elif shape == "triangle":
        t = np.pi / 2 + np.arange(3) * 2 * np.pi / 3
        p = np.stack([np.cos(t), np.sin(t)], 1) * r * 1.1
    elif shape == "arc":                       # 240-degree annulus sector, opening toward +x before rotation
        t = np.linspace(np.pi / 3, 2 * np.pi - np.pi / 3, 40)
        outer = np.stack([np.cos(t), np.sin(t)], 1) * r
        inner = np.stack([np.cos(t[::-1]), np.sin(t[::-1])], 1) * r * 0.58
        p = np.concatenate([outer, inner])
    elif shape == "cross":
        w, l = 0.3 * r, r
        p = np.array([[-w, -l], [w, -l], [w, -w], [l, -w], [l, w], [w, w], [w, l], [-w, l],
                      [-w, w], [-l, w], [-l, -w], [-w, -w]])
    elif shape == "bar":
        w, l = 0.28 * r, 1.15 * r
        p = np.array([[-l, -w], [l, -w], [l, w], [-l, w]])
    else:
        return None
    return _rot(p, a)


def render(i: int):
    rng = np.random.default_rng(1_000_000 + i)
    shape = int(rng.integers(6))
    color = int(rng.integers(6))
    texture = 0 if rng.random() < 0.5 else int(rng.integers(1, 6))
    bg_type = int(rng.integers(3))
    size = float(rng.uniform(0.16, 0.34))
    x, y = (float(v) for v in rng.uniform(0.3, 0.7, 2))
    rot = float(rng.uniform(0, 2 * np.pi))
    bg_lum = float(rng.uniform(0.1, 0.9))
    period = 2 * np.pi / SYM[SHAPES[shape]]
    rot_phase = (rot % period) / period

    W = RES * SS
    m = Image.new("L", (W, W), 0)
    d = ImageDraw.Draw(m)
    cx, cy, r = x * W, y * W, size * W
    poly = shape_polygon(SHAPES[shape], r, rot)
    if poly is None:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    else:
        d.polygon([(cx + px, cy + py) for px, py in poly], fill=255)
    mask = np.asarray(m, np.float32)[..., None] / 255

    yy, xx = np.mgrid[0:W, 0:W].astype(np.float32) / SS          # in 64-px units
    fg = hsv_to_rgb(color / 6 + rng.uniform(-0.03, 0.03), rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0))
    dark = fg * 0.3
    if texture == 0:
        tex = np.broadcast_to(fg, (W, W, 3))
    elif texture <= 4:
        th = np.deg2rad(45 * (texture - 1))
        ph = rng.uniform(0, 2 * np.pi)
        on = np.sin(2 * np.pi * (xx * np.cos(th) + yy * np.sin(th)) / 5.0 + ph) > 0
        tex = np.where(on[..., None], fg, dark)
    else:
        ox, oy = rng.uniform(0, 6, 2)
        on = (((xx + ox) % 6 - 3) ** 2 + ((yy + oy) % 6 - 3) ** 2) < 1.6 ** 2
        tex = np.where(on[..., None], dark, fg)

    if bg_type == 0:
        bg = np.full((W, W), bg_lum, np.float32)
    elif bg_type == 1:
        a = rng.uniform(0, 2 * np.pi)
        bg = bg_lum + 0.22 * ((xx - 32) * np.cos(a) + (yy - 32) * np.sin(a)) / 32
    else:
        lo = rng.normal(0, 1, (6, 6)).astype(np.float32)
        up = np.asarray(Image.fromarray(lo).resize((W, W), Image.BICUBIC))
        bg = bg_lum + 0.12 * up
    bg = np.repeat(np.clip(bg, 0, 1)[..., None], 3, -1)
    img = bg * (1 - mask) + tex * mask
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = np.asarray(Image.fromarray(img).resize((RES, RES), Image.BOX))
    f = dict(shape=shape, color=color, texture=texture, bg_type=bg_type, size=size, x=x, y=y,
             rot_phase=rot_phase, bg_lum=bg_lum, rot=rot)
    return img, f


def derived(fac: dict) -> dict:
    fac["curved"] = np.isin(fac["shape"], [0, 3]).astype(np.int64)
    fac["warm"] = np.isin(fac["color"], list(WARM)).astype(np.int64)
    fac["textured"] = (fac["texture"] > 0).astype(np.int64)
    fac["label"] = fac["shape"] * 4 + fac["warm"] * 2 + fac["textured"]
    return fac


def main():
    os.makedirs(OUT, exist_ok=True)
    n = N_TRAIN + N_TEST
    imgs = np.lib.format.open_memmap(os.path.join(OUT, "images.npy"), "w+", np.uint8, (n, RES, RES, 3))
    facs = []
    with Pool(10) as pool:
        for i, (im, f) in enumerate(pool.imap(render, range(n), chunksize=256)):
            imgs[i] = im
            facs.append(f)
            if i % 10000 == 0:
                print("vis_shapes", i, flush=True)
    imgs.flush()
    fac = {k: np.array([f[k] for f in facs]) for k in facs[0]}
    fac = derived(fac)
    np.savez(os.path.join(OUT, "factors.npz"), **fac)
    print("done", {k: v.shape for k, v in fac.items()})


if __name__ == "__main__":
    sys.exit(main())
