"""Planar two-link arm in a vertical plane, with a floor and a wall it can hit.

Every quantity in the equations of motion is logged per frame so that hidden units
of downstream models can be compared against ground truth:

    M(q) qdd + c(q, qd) + g(q) = tau_act + tau_contact - b qd

Sign convention: g and c are MuJoCo's bias terms (qfrc_bias split into gravity and
velocity parts), so the torque gravity *applies* to joint i is -g_i.
"""
from __future__ import annotations

import numpy as np
import mujoco

L1, L2 = 0.50, 0.45          # link lengths (m)
FLOOR_Z = -0.86              # floor height relative to shoulder (m)
WALL_X = 0.68                # wall position (m)
DT = 0.002                   # physics step (s)
RECORD_EVERY = 10            # -> 50 Hz frames
FPS = 1.0 / (DT * RECORD_EVERY)
TAU_MAX = np.array([25.0, 12.0])
DAMPING = 0.05


def xml(gravity_angle: float = 0.0, g: float = 9.81, with_obstacles: bool = True) -> str:
    """gravity_angle rotates gravity in the x-z plane (0 = straight down)."""
    gx, gz = g * np.sin(gravity_angle), -g * np.cos(gravity_angle)
    obstacles = f"""
    <geom name="floor" type="box" pos="0 0 {FLOOR_Z - 0.05}" size="1.2 0.3 0.05" rgba="0.55 0.57 0.6 1"/>
    <geom name="wall" type="box" pos="{WALL_X + 0.05} 0 0" size="0.05 0.3 1.2" rgba="0.55 0.57 0.6 1"/>
    """ if with_obstacles else ""
    return f"""
<mujoco model="two_link_arm">
  <option timestep="{DT}" gravity="{gx} 0 {gz}" integrator="RK4">
    <flag energy="enable"/>
  </option>
  <visual><global offwidth="448" offheight="448"/><quality shadowsize="0"/></visual>
  <default><geom contype="1" conaffinity="1" friction="0.6 0.005 0.0001" solref="0.01 1"/></default>
  <worldbody>
    <light directional="true" pos="0 -2 2" dir="0 1 -0.5" diffuse="0.8 0.8 0.8" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <camera name="side" pos="0 -2.6 -0.05" xyaxes="1 0 0 0 0 1" fovy="45"/>
    <geom name="backdrop" type="plane" pos="0 0.4 0" zaxis="0 -1 0" size="3 3 0.1" rgba="0.95 0.95 0.93 1" contype="0" conaffinity="0"/>
    {obstacles}
    <geom name="base" type="sphere" pos="0 0 0" size="0.045" rgba="0.15 0.15 0.18 1" contype="0" conaffinity="0"/>
    <body name="upper" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 -1 0" damping="{DAMPING}"/>
      <geom name="upper_g" type="capsule" fromto="0 0 0 {L1} 0 0" size="0.045" mass="2.0" rgba="0.16 0.33 0.78 1"/>
      <body name="fore" pos="{L1} 0 0">
        <joint name="elbow" type="hinge" axis="0 -1 0" damping="{DAMPING}"/>
        <geom name="fore_g" type="capsule" fromto="0 0 0 {L2} 0 0" size="0.04" mass="1.2" rgba="0.85 0.45 0.12 1"/>
        <site name="ee" pos="{L2} 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder" gear="1" ctrlrange="-{TAU_MAX[0]} {TAU_MAX[0]}"/>
    <motor joint="elbow" gear="1" ctrlrange="-{TAU_MAX[1]} {TAU_MAX[1]}"/>
  </actuator>
</mujoco>
"""


def build(**kw) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml(**kw))


# ---------------------------------------------------------------- controllers
def make_controller(rng: np.random.Generator, mode: str, T: float):
    if mode == "passive":
        return lambda t, q, qd: np.zeros(2)
    if mode == "sines":
        freqs = rng.uniform(0.2, 2.0, size=(2, 3))
        phases = rng.uniform(0, 2 * np.pi, size=(2, 3))
        amps = rng.uniform(0.2, 0.7, size=(2, 3)) * TAU_MAX[:, None] / 1.5
        return lambda t, q, qd: np.clip((amps * np.sin(2 * np.pi * freqs * t + phases)).sum(1), -TAU_MAX, TAU_MAX)
    if mode == "reach":   # PD to random targets switching every ~0.6-1.2 s; no gravity compensation
        switch = np.cumsum(rng.uniform(0.6, 1.2, size=int(T / 0.6) + 2))
        targets = rng.uniform([-np.pi, -2.6], [np.pi * 0.5, 2.6], size=(len(switch) + 1, 2))
        kp, kd = np.array([40.0, 18.0]), np.array([4.0, 1.5])
        def ctrl(t, q, qd):
            tgt = targets[np.searchsorted(switch, t)]
            err = np.angle(np.exp(1j * (tgt - q)))
            return np.clip(kp * err - kd * qd, -TAU_MAX, TAU_MAX)
        return ctrl
    raise ValueError(mode)


# ---------------------------------------------------------------- logging
FIELDS = [
    "q", "qd", "qdd", "tau_act", "g_bias", "c_bias", "tau_contact",
    "M11", "M12", "M22", "inertial_self", "interaction",
    "contact_flag", "contact_force", "contact_floor", "contact_wall", "contact_on_fore",
    "elbow_xz", "ee_xz", "ee_vel", "ke", "pe",
]


def _contact_info(m, d):
    total, floor, wall, fore = 0.0, 0.0, 0.0, 0.0
    f6 = np.zeros(6)
    gid_floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    gid_wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    gid_fore = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fore_g")
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(m, d, i, f6)
        fn = abs(f6[0])
        total += fn
        if gid_floor in (c.geom1, c.geom2): floor += fn
        if gid_wall in (c.geom1, c.geom2): wall += fn
        if gid_fore in (c.geom1, c.geom2): fore += fn
    return total, floor, wall, fore


def log_state(m, d, scratch) -> dict:
    """Decompose the equation of motion at the current (post-step) state."""
    # gravity bias: rne with zero velocity
    scratch.qpos[:] = d.qpos
    scratch.qvel[:] = 0
    mujoco.mj_forward(m, scratch)
    g_bias = scratch.qfrc_bias.copy()
    c_bias = d.qfrc_bias - g_bias
    M = np.zeros((2, 2))
    mujoco.mj_fullM(m, d, M)
    qdd = d.qacc.copy()
    total, floor, wall, fore = _contact_info(m, d)
    ee = d.site("ee")
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, ee.id, vel6, 0)
    return dict(
        q=d.qpos.copy(), qd=d.qvel.copy(), qdd=qdd,
        tau_act=d.actuator_force.copy(), g_bias=g_bias, c_bias=c_bias,
        tau_contact=d.qfrc_constraint.copy(),
        M11=M[0, 0], M12=M[0, 1], M22=M[1, 1],
        inertial_self=np.array([M[0, 0] * qdd[0], M[1, 1] * qdd[1]]),
        # torque terms each joint receives from the *other* joint's motion
        interaction=np.array([M[0, 1] * qdd[1] + c_bias[0], M[1, 0] * qdd[0] + c_bias[1]]),
        contact_flag=float(d.ncon > 0), contact_force=total,
        contact_floor=floor, contact_wall=wall, contact_on_fore=fore,
        elbow_xz=d.body("fore").xpos[[0, 2]].copy(), ee_xz=ee.xpos[[0, 2]].copy(),
        ee_vel=vel6[[3, 5]].copy(), ke=d.energy[1], pe=d.energy[0],
    )


def random_init(m, d, rng, max_tries=50):
    for _ in range(max_tries):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = rng.uniform(-np.pi, np.pi, 2)
        d.qvel[:] = rng.normal(0, 2.0, 2)
        mujoco.mj_forward(m, d)
        if d.ncon == 0:
            return
    raise RuntimeError("could not find a collision-free initial state")


def simulate(seed: int, mode: str, n_frames: int = 150, gravity_angle: float = 0.0,
             with_obstacles: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    m = build(gravity_angle=gravity_angle, with_obstacles=with_obstacles)
    d, scratch = mujoco.MjData(m), mujoco.MjData(m)
    random_init(m, d, rng)
    T = n_frames * RECORD_EVERY * DT
    ctrl = make_controller(rng, mode, T)
    out = {k: [] for k in FIELDS}
    for f in range(n_frames):
        for _ in range(RECORD_EVERY):
            d.ctrl[:] = ctrl(d.time, d.qpos.copy(), d.qvel.copy())
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)   # refresh qacc/contacts at the recorded state
        rec = log_state(m, d, scratch)
        for k in FIELDS:
            out[k].append(rec[k])
    out = {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
    out["gravity_angle"] = np.float32(gravity_angle)
    out["mode"] = mode
    return out


class ArmRenderer:
    """Renders qpos sequences from the fixed side camera."""
    def __init__(self, size: int = 64, gravity_angle: float = 0.0, with_obstacles: bool = True):
        self.m = build(gravity_angle=gravity_angle, with_obstacles=with_obstacles)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, size, size)

    def __call__(self, qpos_seq: np.ndarray) -> np.ndarray:
        frames = np.empty((len(qpos_seq), self.r.height, self.r.width, 3), np.uint8)
        for i, q in enumerate(qpos_seq):
            self.d.qpos[:] = q
            mujoco.mj_forward(self.m, self.d)
            self.r.update_scene(self.d, camera="side")
            frames[i] = self.r.render()
        return frames
