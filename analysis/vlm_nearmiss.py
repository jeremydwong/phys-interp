"""D3: explanations that can lose. Simulator-written examples and near misses for the 14 gallery units.

The essay's test ("Explanations That Can Lose"): from an explanation alone, write new examples that
should make the neuron fire and near misses that should not, then measure. Here the "writer" is
the simulator. Each gallery unit's explanation is its census label plus the direction read off its
census tuning curve (e.g. V8#2591 "hand height" fires when the hand is at floor level). For each
unit we write 8 positive clips (the labelled variable at its firing extreme, everything the label
does not constrain randomised) and 8 near misses (the same clips with only the labelled variable
moved to a non-firing value, or the mirrored pose for signed torques).

Clips are produced by sim/arm.py's MuJoCo model: a desired joint trajectory q(t) = q0 + v t + a t²/2
is tracked with computed torque (applied as qfrc_applied, not clipped), or the arm is dropped
passively onto the floor for the contact-torque units. All 32 targets are logged by
sim.arm.log_state, so we can check that the writer actually hit the labelled variable. Clips use
the corpus protocol (8 frames, stride 2, 320 px) and the corpus prompt.

Response of a unit to a clip = the census pole of the unit (pooled max, or -min for - units), max
over the 4 temporal groups, expressed as f = (a - corpus median) / (corpus max - corpus median)
(ours: half of the units have a large offset, so the raw a / max used by the essay cannot go below
0.2 for them; the raw ratio is also stored).
Scoring (ours, after the essay's appendix thresholds): a positive "hits" if f >= 0.5, a near miss is
"rejected" if f < 0.2. A unit's explanation WON if >= 50% of positives hit and >= 50% of near
misses are rejected; TIED if positives hit but near misses also fire; LOST if < 50% of positives hit.

Usage: uv run python analysis/vlm_nearmiss.py [make|run|analyze]
"""
from __future__ import annotations

import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import FAMILIES, savefig, setup_style, targets, INK, INK2, MUTED  # noqa: E402
from analysis.vlm_common import VDIR, n_used, renderer  # noqa: E402
from sim.arm import DT, FLOOR_Z, L1, L2, RECORD_EVERY, WALL_X, build, log_state, FIELDS  # noqa: E402

N_EX = 8
NAMES = None
LEAD = 4           # recorded frames before the clip starts
CLIP = 16          # recorded frames covered by the 8 shown frames (stride 2)


# ------------------------------------------------------------------ simulation

class Sim:
    def __init__(self):
        self.m = build()
        self.d = mujoco.MjData(self.m)
        self.s = mujoco.MjData(self.m)
        self.M = np.zeros((2, 2))

    def track(self, q0, v, a, passive=False, n_rec=LEAD + CLIP, settle=0):
        """Track q(t) = q0 + v t + a t^2/2 with computed torque (or fall passively). Returns the
        recorded dict (50 Hz frames) of the last LEAD+CLIP frames."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q0
        d.qvel[:] = 0 if passive else v
        mujoco.mj_forward(m, d)
        out = {k: [] for k in FIELDS}
        total = settle + n_rec
        for f in range(total):
            for _ in range(RECORD_EVERY):
                if passive:
                    d.qfrc_applied[:] = 0
                else:
                    t = d.time
                    qd_des = v + a * t
                    q_des = q0 + v * t + 0.5 * a * t * t
                    mujoco.mj_fullM(m, d, self.M)
                    e, ed = q_des - d.qpos, qd_des - d.qvel
                    d.qfrc_applied[:] = self.M @ (a + 900.0 * e + 60.0 * ed) + d.qfrc_bias + 0.05 * d.qvel
                mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            if f >= settle:
                rec = log_state(m, d, self.s)
                for k in FIELDS:
                    out[k].append(rec[k])
        return {k: np.asarray(v_, np.float32)[None] for k, v_ in out.items()}


def fk(q):
    e = np.array([L1 * np.cos(q[0]), L1 * np.sin(q[0])])
    h = e + L2 * np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1])])
    return e, h


def ik(x, z, elbow):
    c2 = (x * x + z * z - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1:
        return None
    q2 = elbow * np.arccos(c2)
    q1 = np.arctan2(z, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


def clear(q, margin=0.06):
    """No link point within `margin` of the floor or the wall."""
    e, h = fk(q)
    pts = np.stack([np.linspace(0, 1, 12)[:, None] * e, e + np.linspace(0, 1, 12)[:, None] * (h - e)]).reshape(-1, 2)
    return (pts[:, 1] > FLOOR_Z + margin).all() and (pts[:, 0] < WALL_X - margin).all()


def traj_clear(q0, v, a, margin=0.06):
    T = (LEAD + CLIP) * RECORD_EVERY * DT
    return all(clear(q0 + v * t + 0.5 * a * t * t, margin) for t in np.linspace(0, T, 8))


def grav1(sim, q):
    sim.s.qpos[:] = q
    sim.s.qvel[:] = 0
    mujoco.mj_forward(sim.m, sim.s)
    return -sim.s.qfrc_bias[0]


def jac(q):
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q.sum()), np.cos(q.sum())
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]])


def rand_pose(rng, sim=None, cond=None, margin=0.06):
    for _ in range(10000):
        q = rng.uniform(-np.pi, np.pi, 2)
        if clear(q, margin) and (cond is None or cond(q)):
            return q
    raise RuntimeError("no pose")


Z = np.zeros(2)


def mirror(q):
    return np.array([np.pi - q[0], -q[1]])


def sets_for(name, rng, sim):
    """Returns (explanation, positives, near misses); each item = dict(q0, v, a, passive)."""
    P, N = [], []

    def item(q0, v=Z, a=Z, passive=False):
        return dict(q0=np.asarray(q0, float), v=np.asarray(v, float), a=np.asarray(a, float), passive=passive)

    while len(P) < N_EX:
        if name == "hand_at_floor":
            x, el = rng.uniform(-0.65, 0.5), rng.choice([-1, 1])
            q, qn = ik(x, FLOOR_Z + 0.045, el), ik(x, FLOOR_Z + 0.26, el)
            if q is None or qn is None or not clear(q, -0.01) or not clear(qn):
                continue
            # the hand rests on the floor: check only the elbow/upper arm clear the floor
            e, _ = fk(q)
            if e[1] < FLOOR_Z + 0.08 or e[0] > WALL_X - 0.08:
                continue
            P.append(item(q)); N.append(item(qn))
        elif name == "shoulder_still":
            q = rand_pose(rng, margin=0.25)
            v2 = rng.uniform(-4, 4)
            vn = np.array([rng.choice([-1, 1]) * rng.uniform(3.5, 4.5), v2])
            if not (traj_clear(q, np.array([0, v2]), Z) and traj_clear(q, vn, Z)):
                continue
            P.append(item(q, [0, v2])); N.append(item(q, vn))
        elif name in ("no_shoulder_acc", "no_elbow_acc"):
            j = 0 if name == "no_shoulder_acc" else 1
            q = rand_pose(rng, margin=0.3)
            v = rng.uniform(-2, 2, 2)
            an = np.zeros(2); an[j] = rng.choice([-1, 1]) * 40.0
            if not (traj_clear(q, v, Z) and traj_clear(q, v, an)):
                continue
            P.append(item(q, v)); N.append(item(q, v, an))
        elif name in ("grav1_neg", "grav1_pos"):
            sgn = -1 if name == "grav1_neg" else 1
            q = rand_pose(rng, cond=lambda q: sgn * grav1(sim, q) >= 9.0 and clear(mirror(q)), margin=0.1)
            v = rng.uniform(-0.4, 0.4, 2)
            if not (traj_clear(q, v, Z) and traj_clear(mirror(q), v * np.array([-1, -1]), Z)):
                continue
            P.append(item(q, v)); N.append(item(mirror(q), -v))
        elif name == "cor2_large":
            q2 = rng.choice([-1, 1]) * rng.uniform(1.2, 1.9)
            q = rand_pose(rng, cond=lambda q: True, margin=0.3)
            q = np.array([q[0], q2])
            v = np.array([rng.choice([-1, 1]) * rng.uniform(5, 7), 0.0])
            if not (traj_clear(q, v, Z) and clear(q)):
                continue
            P.append(item(q, v)); N.append(item(q))
        elif name == "cor2_zero":
            q2 = rng.choice([-1, 1]) * rng.uniform(1.2, 1.9)
            q = rand_pose(rng, margin=0.3)
            q = np.array([q[0], q2])
            v2 = rng.uniform(-3, 3)
            vn = np.array([rng.choice([-1, 1]) * rng.uniform(5, 7), v2])
            if not (clear(q) and traj_clear(q, np.array([0, v2]), Z) and traj_clear(q, vn, Z)):
                continue
            P.append(item(q, [0, v2])); N.append(item(q, vn))
        elif name == "contact_shoulder_pos":
            # drop the arm onto the floor right of the shoulder; near miss: same final pose held 5 cm higher
            x, el = rng.uniform(0.1, 0.55), rng.choice([-1, 1])
            q = ik(x, FLOOR_Z + 0.10, el)
            if q is None or not clear(q, 0.02):
                continue
            rec = sim.track(q, Z, Z, passive=True, n_rec=LEAD + CLIP, settle=40)
            qf = rec["q"][0, -1]
            if rec["tau_contact"][0, LEAD:, 0].mean() < 0.5:
                continue
            e, h = fk(qf)
            qn = ik(h[0], h[1] + 0.06, el)
            if qn is None or not clear(qn, 0.02):
                continue
            P.append(item(q, passive=True)); N.append(item(qn))
        elif name == "upper_arm_down":
            q2 = rng.uniform(-2.6, 2.6)
            q = np.array([-np.pi / 2 + rng.uniform(-0.1, 0.1), q2])
            qn = np.array([-np.pi / 2 + rng.choice([-1, 1]) * rng.uniform(0.65, 0.8), q2])
            v = rng.uniform(-0.4, 0.4, 2)
            if not (traj_clear(q, v, Z) and traj_clear(qn, v, Z)):
                continue
            P.append(item(q, v)); N.append(item(qn, v))
        elif name == "hand_vz_zero":
            q = rand_pose(rng, margin=0.3)
            J = jac(q)
            if abs(np.linalg.det(J)) < 0.05:
                continue
            vx = rng.choice([0.0, rng.uniform(-1.2, 1.2)])
            v = np.linalg.solve(J, [vx, 0.0])
            vn = np.linalg.solve(J, [vx, rng.choice([-1, 1]) * 1.5])
            if np.abs(np.r_[v, vn]).max() > 8 or not (traj_clear(q, v, Z) and traj_clear(q, vn, Z)):
                continue
            P.append(item(q, v)); N.append(item(q, vn))
        elif name == "arm_raised":
            q = np.array([np.pi / 2 + rng.uniform(-0.3, 0.3), rng.uniform(-0.4, 0.4)])
            qn = np.array([rng.choice([0.0, np.pi]) + rng.uniform(-0.15, 0.15), rng.uniform(-0.4, 0.4)])
            v = rng.uniform(-0.3, 0.3, 2)
            if not (traj_clear(q, v, Z) and traj_clear(qn, v, Z)):
                continue
            P.append(item(q, v)); N.append(item(qn, v))
        else:
            raise ValueError(name)
    return P, N


UNITS = {
    # gallery label: (set generator, explanation written from the census label + tuning direction)
    "V8#2591": ("hand_at_floor", "hand height (ee_z): fires when the hand is at floor level; near miss: hand 21 cm higher"),
    "V3#1493": ("shoulder_still", "shoulder velocity q̇₁ (tuned at 0): fires when the shoulder does not rotate; near miss: same clip with q̇₁ = ±4 rad/s"),
    "V3#2792": ("no_shoulder_acc", "shoulder acceleration q̈₁ (tuned near 0): fires with no shoulder acceleration; near miss: q̈₁ = ±40 rad/s²"),
    "V8#160": ("grav1_neg", "shoulder gravity torque: fires for strong negative gravity torque (arm out to one side); near miss: mirrored pose"),
    "V8#2386": ("cor2_large", "elbow Coriolis/centripetal torque (dips at 0): fires when |cor₂| is large (fast shoulder, bent elbow); near miss: same pose still"),
    "V18#4014": ("contact_shoulder_pos", "shoulder contact torque ≈ +3 N·m: arm resting on the floor right of the shoulder; near miss: same pose held 6 cm above the floor"),
    "V8#1726": ("arm_raised", "potential energy: fires when the arm is raised; near miss: arm horizontal"),
    "L2−#1848": ("upper_arm_down", "sin q₁ ≈ −1: upper arm pointing straight down; near miss: tilted 37–46°"),
    "L8−#4747": ("hand_vz_zero", "hand vertical velocity ≈ 0; near miss: same pose with hand vz = ±1.5 m/s"),
    "L16−#6072": ("no_elbow_acc", "elbow acceleration q̈₂ ≈ 0; near miss: q̈₂ = ±40 rad/s²"),
    "L2−#789": ("grav1_pos", "shoulder gravity torque: fires for strong positive gravity torque; near miss: mirrored pose"),
    "L8−#7671": ("cor2_zero", "elbow Coriolis torque ≈ 0 (no shoulder rotation); near miss: same pose with q̇₁ = ±5–7 rad/s"),
    "L2−#8960": ("contact_shoulder_pos", "shoulder contact torque large and positive: arm resting on the floor right of the shoulder; near miss: same pose 6 cm above"),
    "L2#9169": ("arm_raised", "potential energy high: arm raised; near miss: arm horizontal"),
}


def spec_label(sp):
    part, L, pole = sp["key"].split("_")[1:4]
    return f"{'V' if part == 'vision' else 'L'}{L[1:]}{'−' if pole == 'neg' else ''}#{sp['unit']}"


def make():
    sim = Sim()
    out = {}
    for lab, (gen, expl) in UNITS.items():
        rng = np.random.default_rng(sum(map(ord, lab)))
        P, N = sets_for(gen, rng, sim)
        out[lab] = dict(gen=gen, explanation=expl, pos=P, near=N)
        print(lab, gen, flush=True)
    recs, meta = [], []
    for lab, d in out.items():
        for kind in ("pos", "near"):
            for i, it in enumerate(d[kind]):
                rec = sim.track(it["q0"], it["v"], it["a"], passive=it["passive"], settle=40 if it["passive"] else 0)
                recs.append(rec)
                meta.append((lab, kind, i))
    st = {k: np.concatenate([r[k] for r in recs]) for k in FIELDS}
    Y, vs = targets(st)
    q = st["q"]
    shown = LEAD + 2 * np.arange(8)
    mid = LEAD + 4 * np.arange(4) + 1
    np.savez(os.path.join(VDIR, "nearmiss_clips.npz"), q=q[:, shown], Y=Y[:, mid], meta=np.array(meta, dtype=object),
             explanations=json.dumps({k: v["explanation"] for k, v in out.items()}))
    print("clips", len(recs))


def run():
    import torch
    from analysis.vlm_common import load_model
    from analysis.vlm_capture import capture
    z = np.load(os.path.join(VDIR, "nearmiss_clips.npz"), allow_pickle=True)
    with open(os.path.join(VDIR, "gallery_spec.json")) as f:
        spec = json.load(f)
    units = [(sp["part"], sp["layer"], sp["unit"]) for sp in spec]
    clips = [renderer()(q) for q in z["q"]]
    model, proc = load_model()
    maps, _ = capture(model, proc, clips, units)
    out = {}
    for sp, m in zip(spec, maps):
        mm_ = m.reshape(m.shape[0], m.shape[1], -1)
        out[f"max_{spec_label(sp)}"] = mm_.max(-1)
        out[f"min_{spec_label(sp)}"] = mm_.min(-1)
    np.savez(os.path.join(VDIR, "nearmiss_acts.npz"), **out)
    del model
    torch.cuda.empty_cache()


def analyze():
    from analysis.vlm_gallery import unit_values
    setup_style()
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(VDIR, "nearmiss_clips.npz"), allow_pickle=True)
    acts = np.load(os.path.join(VDIR, "nearmiss_acts.npz"))
    meta = z["meta"]
    Y = z["Y"]
    from analysis.common import load_states
    names = [v.name for v in targets(load_states())[1]]
    with open(os.path.join(VDIR, "gallery_spec.json")) as f:
        spec = json.load(f)
    expl = json.loads(str(z["explanations"]))
    NG, _ = n_used("video")
    res = {"scoring": __doc__.split("Response of a unit")[1].split("Usage")[0].strip(), "units": {}}
    for sp in spec:
        lab = spec_label(sp)
        (part, l, pole), pos, neg = unit_values(sp["key"], sp["unit"], NG)
        corpus = pos if pole == "+" else -neg
        cmax, cmed = float(corpus.max()), float(np.median(corpus))
        a = acts[f"max_{lab}"] if pole == "+" else -acts[f"min_{lab}"]          # [n_clips, 4]
        a = a.max(1)
        rows = [i for i, mt in enumerate(meta) if mt[0] == lab]
        kinds = np.array([meta[i][1] for i in rows])
        f = (a[rows] - cmed) / (cmax - cmed)
        raw = a[rows] / cmax
        j = names.index(sp["var"])
        var_pos = Y[rows][kinds == "pos", :, j].mean(1)
        var_near = Y[rows][kinds == "near", :, j].mean(1)
        hit = float((f[kinds == "pos"] >= 0.5).mean())
        rej = float((f[kinds == "near"] < 0.2).mean())
        verdict = "lost" if hit < 0.5 else "won" if rej >= 0.5 else "tied"
        # corpus percentile of the median positive / near-miss response
        pct = lambda x: float((corpus < x).mean() * 100)
        res["units"][lab] = dict(var=sp["var"], census_r2=sp["r2"], explanation=expl[lab],
                                 corpus_max=cmax, corpus_median=cmed,
                                 f_pos=f[kinds == "pos"].tolist(), f_near=f[kinds == "near"].tolist(),
                                 raw_ratio_pos=raw[kinds == "pos"].tolist(), raw_ratio_near=raw[kinds == "near"].tolist(),
                                 pos_hit_rate=hit, near_reject_rate=rej, verdict=verdict,
                                 median_pos_corpus_pct=pct(np.median(a[rows][kinds == "pos"])),
                                 median_near_corpus_pct=pct(np.median(a[rows][kinds == "near"])),
                                 var_value_pos=var_pos.tolist(), var_value_near=var_near.tolist())
        print(lab, verdict, "hit", hit, "rej", rej, "f_pos med %.2f near med %.2f" % (np.median(f[kinds == 'pos']), np.median(f[kinds == 'near'])),
              "var pos %.2f near %.2f" % (var_pos.mean(), var_near.mean()), flush=True)
    with open(os.path.join(VDIR, "nearmiss.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    U = res["units"]
    labs = list(U)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    rng = np.random.default_rng(0)
    for i, lab in enumerate(labs):
        u = U[lab]
        col = FAMILIES[next(sp["fam"] for sp in spec if spec_label(sp) == lab)][0]
        fp, fn = np.clip(u["f_pos"], -0.6, 1.6), np.clip(u["f_near"], -0.6, 1.6)
        ax.scatter(i - 0.15 + rng.uniform(-0.07, 0.07, len(fp)), fp, s=12, color=col, lw=0, label="written example" if i == 0 else None)
        ax.scatter(i + 0.15 + rng.uniform(-0.07, 0.07, len(fn)), fn, s=12, facecolor="none", edgecolor=MUTED, lw=0.8,
                   label="near miss" if i == 0 else None)
        ax.text(i, 1.68, u["verdict"].upper(), ha="center", fontsize=6.5, fontweight="bold",
                color={"won": "#1baf7a", "tied": "#eda100", "lost": "#e34948"}[u["verdict"]])
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.axhline(0.2, color=MUTED, lw=0.8, ls=":")
    ax.axhline(0.0, color=MUTED, lw=0.5)
    ax.set_xticks(range(len(labs)), [f"{l}\n{U[l]['var']}" for l in labs], fontsize=5.8)
    ax.set_ylim(-0.65, 1.8)
    ax.set_ylabel("response  (a − corpus median) / (max − median)", fontsize=7)
    ax.set_title("D3  Simulator-written examples and near misses (dashed: hit ≥ 0.5; dotted: reject < 0.2)", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower left", ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, "vlm_nearmiss.png")


if __name__ == "__main__":
    {"make": make, "run": run, "analyze": analyze}[sys.argv[1]]()
