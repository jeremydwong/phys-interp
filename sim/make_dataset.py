"""Generate the two-link-arm datasets.

main : fixed gravity, floor + wall present. 2400 episodes x 150 frames (3 s at 50 Hz).
grav : gravity direction randomized per episode, no obstacles, so the only cue to
       "down" is how the arm moves. 1200 episodes.

Each dataset is saved as data/<name>/states.npz (all logged fields, shape [E, T, ...])
and data/<name>/frames64.npy (uint8 [E, T, 64, 64, 3]).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.arm import FIELDS, ArmRenderer, simulate  # noqa: E402

MODES = ["passive", "sines", "reach"]
N_FRAMES = 150


def _job(args):
    seed, mode, gravity_angle, with_obstacles = args
    ep = simulate(seed, mode, N_FRAMES, gravity_angle, with_obstacles)
    r = _renderer(gravity_angle, with_obstacles)
    frames = r(ep["q"])
    return ep, frames


_RENDERERS = {}


def _renderer(gravity_angle, with_obstacles):
    key = with_obstacles  # camera/scene don't depend on gravity
    if key not in _RENDERERS:
        _RENDERERS[key] = ArmRenderer(64, 0.0, with_obstacles)
    return _RENDERERS[key]


def make(name: str, n_eps: int, randomize_gravity: bool, with_obstacles: bool, seed0: int):
    out = os.path.join("data", name)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed0)
    jobs = []
    for i in range(n_eps):
        ga = float(rng.uniform(-np.pi, np.pi)) if randomize_gravity else 0.0
        jobs.append((seed0 + i, MODES[i % 3], ga, with_obstacles))
    frames = np.lib.format.open_memmap(os.path.join(out, "frames64.npy"), "w+", np.uint8,
                                       (n_eps, N_FRAMES, 64, 64, 3))
    states = {k: [] for k in FIELDS + ["gravity_angle", "mode"]}
    with Pool(14) as pool:
        for i, (ep, fr) in enumerate(pool.imap(_job, jobs, chunksize=8)):
            frames[i] = fr
            for k in states:
                states[k].append(ep[k])
            if i % 200 == 0:
                print(name, i, flush=True)
    frames.flush()
    np.savez_compressed(os.path.join(out, "states.npz"),
                        **{k: np.asarray(v) for k, v in states.items()},
                        seed=np.array([j[0] for j in jobs]))
    print("done", name)


if __name__ == "__main__":
    make("main", 2400, randomize_gravity=False, with_obstacles=True, seed0=0)
    make("grav", 1200, randomize_gravity=True, with_obstacles=False, seed0=100_000)
