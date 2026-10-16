"""Simulation helpers for the gravity-vector study (states only, no rendering).

Wraps sim/arm.py (unchanged) so that we can additionally control:
  * gravity magnitude g,
  * the exact initial state (q0, qd0),
  * a *hidden* constant joint torque applied through qfrc_applied, which is NOT
    reported in tau_act (used for the cue-conflict experiment),
  * an offset added to reach-controller targets (so a scenario can be rotated rigidly).

Also generates the extra datasets under data/grav_extra/:
  train_extra.npz : 9000 more training episodes (same distribution as data/grav)
  analysis.npz    : 1000 held-out episodes used for all neuron-level analyses
  vary_g.npz      : 600 held-out episodes with |g| ~ U(4, 16) m/s^2 (OOD magnitude test)
Run:  uv run python analysis/grav_sim.py
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim.arm import (RECORD_EVERY, DT, build, log_state, make_controller,  # noqa: E402
                     random_init)

EXTRA = os.path.join(ROOT, "data", "grav_extra")
KEEP = ["q", "qd", "qdd", "tau_act", "g_bias", "c_bias", "ee_xz", "elbow_xz",
        "M11", "M12", "M22"]
MODES = ["passive", "sines", "reach"]


def simulate_custom(seed: int, mode: str, n_frames: int = 150, gravity_angle: float = 0.0,
                    g: float = 9.81, q0=None, qd0=None, hidden_tau=None, q1_offset: float = 0.0,
                    ctrl=None) -> dict:
    rng = np.random.default_rng(seed)
    m = build(gravity_angle=gravity_angle, g=g, with_obstacles=False)
    d, scratch = mujoco.MjData(m), mujoco.MjData(m)
    random_init(m, d, rng)                     # consumes rng exactly like sim.arm.simulate
    if q0 is not None:
        d.qpos[:] = q0
    if qd0 is not None:
        d.qvel[:] = qd0
    d.qpos[0] += q1_offset
    mujoco.mj_forward(m, d)
    T = n_frames * RECORD_EVERY * DT
    if ctrl is None:
        base = make_controller(rng, mode, T)
        if q1_offset != 0.0 and mode == "reach":
            off = np.array([q1_offset, 0.0])
            ctrl = lambda t, q, qd: base(t, q - off, qd)   # noqa: E731  (targets rotate with scene)
        else:
            ctrl = base
    out = {k: [] for k in KEEP}
    for f in range(n_frames):
        for _ in range(RECORD_EVERY):
            d.ctrl[:] = ctrl(d.time, d.qpos.copy(), d.qvel.copy())
            if hidden_tau is not None:
                d.qfrc_applied[:] = hidden_tau
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        rec = log_state(m, d, scratch)
        for k in KEEP:
            out[k].append(rec[k])
    out = {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
    out["gravity_angle"] = np.float32(gravity_angle)
    out["g"] = np.float32(g)
    out["mode"] = mode
    return out


def _job(a):
    return simulate_custom(*a)


def stack(eps: list[dict]) -> dict:
    return {k: np.stack([e[k] for e in eps]) for k in eps[0]}


def make(name: str, n: int, seed0: int, g_range=None):
    rng = np.random.default_rng(seed0)
    jobs = []
    for i in range(n):
        ga = float(rng.uniform(-np.pi, np.pi))
        g = float(rng.uniform(*g_range)) if g_range else 9.81
        jobs.append((seed0 + i, MODES[i % 3], 150, ga, g))
    with Pool(12) as pool:
        eps = pool.map(_job, jobs, chunksize=16)
    os.makedirs(EXTRA, exist_ok=True)
    np.savez(os.path.join(EXTRA, f"{name}.npz"), **stack(eps))
    print("saved", name, n, flush=True)


def load_extra(name: str) -> dict:
    z = np.load(os.path.join(EXTRA, f"{name}.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    make("train_extra", 9000, 500_000)
    make("analysis", 1000, 900_000)
    make("vary_g", 600, 950_000, g_range=(4.0, 16.0))   # out-of-distribution |g| test
