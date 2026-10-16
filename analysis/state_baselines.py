"""Model accuracy vs baselines on held-out episodes.

Baselines for the one-step target [dq1, dq2, dqd1, dqd2]:
  linear      : least-squares on the same 8 inputs (sin q, cos q, qd, tau)
  physics -C  : MuJoCo integration of the true rigid-body dynamics for 20 ms with tau held at tau_act(t),
                obstacles removed (gravity + Coriolis + inertia, no contact)
  physics +C  : the same with floor and wall present (the true simulator; residual error comes only from
                the controller changing tau within the 20 ms step) -> a ceiling on achievable R²
"""
from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.state_common import CACHE, RECORD_EVERY, data, list_checkpoints, load_model, run, update_results  # noqa
from models.state_mlp import OUT_NAMES, r2  # noqa: E402
from sim.arm import build  # noqa: E402


def simulate_step(q, qd, tau, with_obstacles):
    m = build(with_obstacles=with_obstacles)
    d = mujoco.MjData(m)
    out = np.zeros((len(q), 4))
    for i in range(len(q)):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q[i]; d.qvel[:] = qd[i]; d.ctrl[:] = tau[i]
        for _ in range(RECORD_EVERY):
            mujoco.mj_step(m, d)
        out[i, :2] = d.qpos - q[i]
        out[i, 2:] = d.qvel - qd[i]
    return out


def main():
    d = data()
    S, D = d["S"], d["D"]
    res = {}
    # trained model
    model, ck = load_model("final")
    y = run(model, d["X"])["y"]
    P = y * d["sd"] + d["mu"]
    res["model"] = r2(P, D).tolist()
    m0, _ = load_model("step000000")
    res["random_init"] = r2(run(m0, d["X"])["y"] * d["sd"] + d["mu"], D).tolist()
    # linear
    A = np.c_[d["Xtr"], np.ones(len(d["Xtr"]))]
    W = np.linalg.lstsq(A, d["Dtr"], rcond=None)[0]
    res["linear"] = r2(np.c_[d["X"], np.ones(len(d["X"]))] @ W, D).tolist()
    # physics
    from multiprocessing import Pool
    q, qd, tau = S["q"].astype(np.float64), S["qd"].astype(np.float64), S["tau_act"].astype(np.float64)
    chunks = np.array_split(np.arange(len(q)), 8)
    for name, obs in (("physics_no_contact", False), ("physics_with_contact", True)):
        with Pool(8) as pool:
            parts = pool.starmap(simulate_step, [(q[c], qd[c], tau[c], obs) for c in chunks])
        Pp = np.concatenate(parts)
        res[name] = r2(Pp, D).tolist()
        os.makedirs(CACHE, exist_ok=True)
        np.save(os.path.join(CACHE, f"{name}.npy"), Pp.astype(np.float32))
    cf = S["contact_flag"] > 0
    nxt = d["st"]["contact_flag"][2000:, 1:].reshape(-1) > 0     # contact at frame t+1
    near = cf | nxt
    res["model_noncontact_frames"] = r2(P[~near], D[~near]).tolist()
    res["model_contact_frames"] = r2(P[near], D[near]).tolist()
    res["frac_contact_frames"] = float(near.mean())
    res["out_names"] = OUT_NAMES
    res["n_test_frames"] = int(len(D))
    res["final_step"] = int(ck["step"])
    idx = list_checkpoints()
    res["test_loss_curve"] = [(c["step"], c["test_loss"]) for c in idx["log"]]
    for k, v in res.items():
        print(k, np.round(v, 4) if isinstance(v, list) and k != "test_loss_curve" and k != "out_names" else "")
    update_results("accuracy", res)


if __name__ == "__main__":
    main()
