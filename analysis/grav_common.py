"""Shared helpers for the gravity-vector analyses (grav_*.py).

* analysis windows from data/grav_extra/analysis.npz (1000 held-out episodes)
* ridge probes for the gravity unit vector
* ideal-observer baseline: least-squares inverse dynamics for the gravity vector
* angle utilities, results json I/O
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.grav_sim import load_extra  # noqa: E402
from models.grav_model import CTX, features  # noqa: E402
from sim.arm import DAMPING, FPS, L1, L2, build  # noqa: E402

RESULTS = os.path.join(ROOT, "reports", "results")
DT_F = 1.0 / FPS
G = 9.81
N_PROBE_TRAIN = 700     # analysis episodes [0,700) fit probes, [700,1000) evaluate


def wrap(a):
    return np.angle(np.exp(1j * np.asarray(a)))


def angerr_deg(pred, true):
    return np.degrees(np.abs(wrap(pred - true)))


def update_json(section: str, data: dict):
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, "grav.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[section] = data
    json.dump(d, open(p, "w"), indent=1, default=float)


# ------------------------------------------------------------------ windows
def analysis_windows(starts=(0, 86), name="analysis"):
    """Returns dict with X [N,CTX,8], q, qd, tau, g_bias, ee [N,CTX,..], theta [N], ep [N]."""
    st = load_extra(name)
    out = {}
    for k in ("q", "qd", "tau_act", "g_bias", "ee_xz", "qdd"):
        out[k] = np.concatenate([st[k][:, s:s + CTX] for s in starts])
    E = len(st["q"])
    out["theta"] = np.tile(st["gravity_angle"], len(starts)).astype(np.float64)
    out["ep"] = np.tile(np.arange(E), len(starts))
    out["mode"] = np.tile(st["mode"], len(starts))
    out["start"] = np.repeat(np.array(starts), E)
    out["X"] = features(out["q"], out["qd"], out["tau_act"])
    return out


# ------------------------------------------------------------------ probes
def ridge_fit(H, Y, lam=1e-2):
    """H [N,D], Y [N,K] -> (W [D+1,K]) with intercept; lam relative to mean eigenvalue."""
    mu, sd = H.mean(0), H.std(0) + 1e-6
    Z = (H - mu) / sd
    A = Z.T @ Z / len(Z)
    W = np.linalg.solve(A + lam * np.eye(len(A)), Z.T @ (Y - Y.mean(0)) / len(Z))
    return dict(mu=mu, sd=sd, W=W, b=Y.mean(0))


def ridge_apply(p, H):
    return ((H - p["mu"]) / p["sd"]) @ p["W"] + p["b"]


def unitvec(theta):
    return np.stack([np.cos(theta), np.sin(theta)], -1)


def vec_angle(Y):
    return np.arctan2(Y[..., 1], Y[..., 0])


# ------------------------------------------------------------------ ideal observer
class Dyn:
    """Rigid-body terms of the arm from MuJoCo: M(q), c(q,qd), gravity basis."""

    def __init__(self):
        self.m0 = build(gravity_angle=0.0, g=0.0, with_obstacles=False)
        self.md = build(gravity_angle=0.0, g=1.0, with_obstacles=False)        # g = (0,-1)
        self.mx = build(gravity_angle=np.pi / 2, g=1.0, with_obstacles=False)  # g = (+1, 0)
        self.d0, self.dd, self.dx = (mujoco.MjData(m) for m in (self.m0, self.md, self.mx))

    def terms(self, q, qd):
        """q, qd [N,2] -> M [N,2,2], c [N,2], Gd [N,2], Gx [N,2] (bias forces, MuJoCo sign)."""
        N = len(q)
        M, c, Gd, Gx = np.zeros((N, 2, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2))
        for i in range(N):
            self.d0.qpos[:] = q[i]; self.d0.qvel[:] = qd[i]
            mujoco.mj_forward(self.m0, self.d0)
            mujoco.mj_fullM(self.m0, self.d0, M[i])
            c[i] = self.d0.qfrc_bias
            for m, d, out in ((self.md, self.dd, Gd), (self.mx, self.dx, Gx)):
                d.qpos[:] = q[i]; d.qvel[:] = 0
                mujoco.mj_forward(m, d)
                out[i] = d.qfrc_bias
        return M, c, Gd, Gx


def g_bias_from(q, theta, g=G, dyn=None):
    """Gravity bias (MuJoCo sign) for postures q [N,2] under gravity angles theta [N]."""
    dyn = dyn or Dyn()
    _, _, Gd, Gx = dyn.terms(q, np.zeros_like(q))
    return g * (np.cos(theta)[:, None] * Gd + np.sin(theta)[:, None] * Gx)


def ideal_observer(q, qd, tau, qdd_true=None, dyn=None):
    """Causal least-squares estimate of the gravity vector at every context position.

    Equation of motion  M qdd + c + g [cosθ Gd + sinθ Gx] = tau - b qd  is linear in
    u = g (cosθ, sinθ). With qdd estimated from the *observed* trajectory (central difference
    of qd around frame t-1, torque averaged over the two intervals), position t can use
    residuals for frames 1..t-1. Returns u_hat [N,T,2] (nan where unavailable).
    If qdd_true is given, uses the simulator's qdd at each frame instead (oracle)."""
    dyn = dyn or Dyn()
    N, T, _ = q.shape
    if qdd_true is None:
        qdd = np.full_like(qd, np.nan)
        qdd[:, 1:-1] = (qd[:, 2:] - qd[:, :-2]) / (2 * DT_F)
        tau_m = np.full_like(tau, np.nan)
        tau_m[:, 1:-1] = 0.25 * tau[:, :-2] + 0.5 * tau[:, 1:-1] + 0.25 * tau[:, 2:]
        lag = 1   # residual at frame k needs frame k+1
    else:
        qdd, tau_m, lag = qdd_true, tau, 0
    M, c, Gd, Gx = dyn.terms(q.reshape(-1, 2).astype(np.float64), qd.reshape(-1, 2).astype(np.float64))
    M, c, Gd, Gx = (a.reshape(N, T, *a.shape[1:]) for a in (M, c, Gd, Gx))
    r = tau_m - DAMPING * qd - np.einsum("ntij,ntj->nti", M, np.nan_to_num(qdd)) - c   # [N,T,2]
    A = np.stack([Gd, Gx], -1)                                                       # [N,T,2(eq),2(u)]
    ok = np.isfinite(qdd).all(-1) & np.isfinite(tau_m).all(-1)
    A = np.where(ok[..., None, None], A, 0)
    r = np.where(ok[..., None], r, 0)
    AtA = np.cumsum(np.einsum("ntei,ntej->ntij", A, A), 1)
    Atr = np.cumsum(np.einsum("ntei,nte->nti", A, r), 1)
    u = np.full((N, T, 2), np.nan)
    for t in range(T):
        tt = t - lag        # residual frames available at context position t
        if tt < 0:
            continue
        S, b = AtA[:, tt], Atr[:, tt]
        good = np.linalg.det(S) > 1e-8
        u[good, t] = np.linalg.solve(S[good], b[good][..., None])[..., 0]
    return u


def hand_xz(q):
    """Forward kinematics, world (x, z) of the hand; q [...,2]."""
    x = L1 * np.cos(q[..., 0]) + L2 * np.cos(q[..., 0] + q[..., 1])
    z = L1 * np.sin(q[..., 0]) + L2 * np.sin(q[..., 0] + q[..., 1])
    return np.stack([x, z], -1)


def elbow_xz(q):
    return np.stack([L1 * np.cos(q[..., 0]), L1 * np.sin(q[..., 0])], -1)


# ------------------------------------------------------------------ pooled probe + encoding subspace
def frame_angle(theta, q, frame):
    """Gravity angle in a given frame. theta [N] or [N,T]; q [N,T,2] -> [N,T]."""
    th = theta[:, None] if np.ndim(theta) == 1 else theta
    if frame == "world":
        return th + 0 * q[..., 0]
    if frame == "upper":
        return th - q[..., 0]
    if frame == "fore":
        return th - q[..., 0] - q[..., 1]
    raise ValueError(frame)


def pooled_probe(model, layer: int, t_min: int = 16, w=None, lam=1e-2, frame="world"):
    """Fit, on analysis episodes [0,700), a ridge probe resid{layer} -> (cos φ, sin φ) pooled over
    context positions >= t_min, where φ is gravity in the given frame ('world': θg, 'upper': θg - q1).
    Also returns the *encoding* map B (least-squares h ≈ μ + B u, u the unit vector).
    Returns dict(probe, B [D,2], mu [D], Binv [2,D], var_frac)."""
    from models.grav_model import run
    w = w or analysis_windows()
    tr = w["ep"] < N_PROBE_TRAIN
    _, acts = run(model, w["X"][tr], keys=[f"resid{layer}"])
    H = acts[f"resid{layer}"][:, t_min:].reshape(-1, acts[f"resid{layer}"].shape[-1])
    U = unitvec(frame_angle(w["theta"][tr], w["q"][tr], frame)[:, t_min:]).reshape(-1, 2)
    probe = ridge_fit(H, U, lam)
    Uc = U - U.mean(0)
    mu = H.mean(0) - np.linalg.lstsq(Uc, H - H.mean(0), rcond=None)[0].T @ U.mean(0)
    B = np.linalg.lstsq(Uc, H - H.mean(0), rcond=None)[0].T          # [D,2]
    Binv = np.linalg.pinv(B)                                         # [2,D]
    frac = ((Uc @ B.T) ** 2).sum() / ((H - H.mean(0)) ** 2).sum()
    return dict(probe=probe, B=B, mu=mu, Binv=Binv, var_frac=float(frac), layer=layer, frame=frame)


def decode(pp, H):
    return vec_angle(ridge_apply(pp["probe"], H))


def probe_layer() -> int:
    """Residual layer with the lowest late-context probe error (set by grav_probe.py)."""
    d = json.load(open(os.path.join(RESULTS, "grav.json")))
    return int(d["probe"]["probe_best_layer"])


def probe_frame() -> str:
    """Frame ('world' or 'upper') in which gravity is best linearly decodable (set by grav_probe.py)."""
    d = json.load(open(os.path.join(RESULTS, "grav.json")))
    return d["probe"]["best_frame"]


def decode_world(pp, H, q):
    """Decode gravity with a pooled probe of either frame and convert it to a world-frame angle.
    H [...,D], q [...,2] matching leading dims."""
    a = decode(pp, H)
    if pp["frame"] == "upper":
        a = a + q[..., 0]
    elif pp["frame"] == "fore":
        a = a + q[..., 0] + q[..., 1]
    return wrap(a)
