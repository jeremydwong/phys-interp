"""Alternative renderings of the same arm trajectories, for generalisation tests.

  original   : sim.arm.ArmRenderer (blue upper arm, orange forearm, grey floor+wall, white backdrop)
  swap       : upper arm orange, forearm blue
  grey       : both links dark grey
  dark       : near-black backdrop
  noobst     : floor and wall not drawn (physics unchanged; contacts become invisible)
  cartoon    : flat 2-D line drawing (PIL) of the same kinematics, same colours, no shading
"""
from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.arm import xml, L1, L2, FLOOR_Z, WALL_X  # noqa: E402
from analysis.vlm_common import RES, renderer  # noqa: E402

BLUE, ORANGE = "0.16 0.33 0.78 1", "0.85 0.45 0.12 1"
VARIANTS = ["original", "swap", "grey", "dark", "noobst", "cartoon"]


class XmlRenderer:
    def __init__(self, s, size=RES):
        self.m = mujoco.MjModel.from_xml_string(s)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, size, size)

    def __call__(self, qs):
        out = np.empty((len(qs), self.r.height, self.r.width, 3), np.uint8)
        for i, q in enumerate(qs):
            self.d.qpos[:] = q
            mujoco.mj_forward(self.m, self.d)
            self.r.update_scene(self.d, camera="side")
            out[i] = self.r.render()
        return out


class CartoonRenderer:
    """Pinhole projection matching the MuJoCo 'side' camera (pos y=-2.6, z=-0.05, fovy 45)."""
    def __init__(self, size=RES):
        self.size = size
        self.f = (size / 2) / np.tan(np.deg2rad(22.5))
        self.depth = 2.6

    def px(self, x, z):
        return (self.size / 2 + self.f * x / self.depth, self.size / 2 - self.f * (z + 0.05) / self.depth)

    def __call__(self, qs):
        from PIL import Image, ImageDraw
        out = []
        for q in qs:
            im = Image.new("RGB", (self.size, self.size), (242, 242, 237))
            d = ImageDraw.Draw(im)
            fy = self.px(0, FLOOR_Z)[1]
            d.rectangle([0, fy, self.size, self.size], fill=(140, 145, 153))
            wx = self.px(WALL_X, 0)[0]
            d.rectangle([wx, 0, self.size, fy], fill=(140, 145, 153))
            e = (L1 * np.cos(q[0]), L1 * np.sin(q[0]))
            h = (e[0] + L2 * np.cos(q[0] + q[1]), e[1] + L2 * np.sin(q[0] + q[1]))
            # joint axis is -y, so positive q rotates x toward +z
            w = max(2, int(0.085 * self.f / self.depth))
            d.line([self.px(0, 0), self.px(*e)], fill=(41, 84, 199), width=w)
            d.line([self.px(*e), self.px(*h)], fill=(217, 115, 31), width=w)
            out.append(np.asarray(im))
        return np.stack(out)


_R = {}


def variant_renderer(kind):
    if kind in _R:
        return _R[kind]
    if kind == "original":
        r = renderer()
    elif kind == "cartoon":
        r = CartoonRenderer()
    else:
        s = xml()
        if kind == "swap":
            s = s.replace(BLUE, "TMP").replace(ORANGE, BLUE).replace("TMP", ORANGE)
        elif kind == "grey":
            s = s.replace(BLUE, "0.25 0.25 0.27 1").replace(ORANGE, "0.25 0.25 0.27 1")
        elif kind == "dark":
            s = s.replace('rgba="0.95 0.95 0.93 1"', 'rgba="0.08 0.08 0.1 1"')
        elif kind == "noobst":
            s = xml(with_obstacles=False)
        r = XmlRenderer(s)
    _R[kind] = r
    return r
