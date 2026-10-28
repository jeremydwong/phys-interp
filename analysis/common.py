"""Shared definitions for every analysis: physical target variables, data loading, figure style.

All three model studies (state MLP, video model, VLM) use these, so a "gravity torque"
neuron means the same thing everywhere and every figure uses the same colours.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "reports", "figs")

# ------------------------------------------------------------------ data

N_TRAIN_EPS = 2000          # main: episodes [0, 2000) train, [2000, 2400) test
N_TRAIN_EPS_GRAV = 1000     # grav: [0, 1000) train, [1000, 1200) test


def load_states(name: str = "main") -> dict:
    z = np.load(os.path.join(DATA, name, "states.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_frames(name: str = "main") -> np.ndarray:
    """uint8 memmap [E, T, 64, 64, 3]."""
    return np.load(os.path.join(DATA, name, "frames64.npy"), mmap_mode="r")


# ------------------------------------------------------------------ targets

@dataclass(frozen=True)
class Var:
    name: str      # column key
    label: str     # pretty label for figures
    family: str    # colour family


FAMILIES = {
    # family: (light-surface colour, description)
    "posture":     ("#2a78d6", "joint angles / positions"),
    "velocity":    ("#6250d6", "velocities"),
    "accel":       ("#e87ba4", "accelerations"),
    "gravity":     ("#1baf7a", "gravity torque"),
    "interaction": ("#eb6834", "interaction torque"),
    "contact":     ("#e34948", "contact"),
    "applied":     ("#eda100", "applied (motor) torque"),
    "energy":      ("#008300", "energy"),
    "none":        ("#b5b3ab", "unexplained"),
}


def targets(st: dict) -> tuple[np.ndarray, list[Var]]:
    """Build the per-frame target matrix [E, T, V] from a states dict.

    Torques are written as the torque *acting on* the joint (gravity torque = -g_bias),
    so a positive gravity torque at the shoulder pulls the arm in the +q1 direction.
    """
    q, qd, qdd = st["q"], st["qd"], st["qdd"]
    cols, vs = [], []

    def add(arr, name, label, fam):
        cols.append(arr.astype(np.float32))
        vs.append(Var(name, label, fam))

    add(np.sin(q[..., 0]), "sin_q1", "sin q₁ (shoulder)", "posture")
    add(np.cos(q[..., 0]), "cos_q1", "cos q₁ (shoulder)", "posture")
    add(np.sin(q[..., 1]), "sin_q2", "sin q₂ (elbow)", "posture")
    add(np.cos(q[..., 1]), "cos_q2", "cos q₂ (elbow)", "posture")
    add(np.sin(q[..., 0] + q[..., 1]), "sin_q12", "sin(q₁+q₂) forearm", "posture")
    add(np.cos(q[..., 0] + q[..., 1]), "cos_q12", "cos(q₁+q₂) forearm", "posture")
    add(st["ee_xz"][..., 0], "ee_x", "hand x", "posture")
    add(st["ee_xz"][..., 1], "ee_z", "hand height", "posture")
    add(st["elbow_xz"][..., 1], "elbow_z", "elbow height", "posture")
    add(qd[..., 0], "qd1", "q̇₁ shoulder vel", "velocity")
    add(qd[..., 1], "qd2", "q̇₂ elbow vel", "velocity")
    add(st["ee_vel"][..., 0], "ee_vx", "hand vx", "velocity")
    add(st["ee_vel"][..., 1], "ee_vz", "hand vz", "velocity")
    add(qd[..., 1] ** 2, "qd2_sq", "q̇₂² (centripetal)", "velocity")
    add(np.clip(qdd[..., 0], -150, 150), "qdd1", "q̈₁ shoulder acc", "accel")
    add(np.clip(qdd[..., 1], -300, 300), "qdd2", "q̈₂ elbow acc", "accel")
    add(-st["g_bias"][..., 0], "grav1", "gravity torque, shoulder", "gravity")
    add(-st["g_bias"][..., 1], "grav2", "gravity torque, elbow", "gravity")
    add(np.clip(-st["interaction"][..., 0], -60, 60), "int1", "interaction torque, shoulder", "interaction")
    add(np.clip(-st["interaction"][..., 1], -30, 30), "int2", "interaction torque, elbow", "interaction")
    add(-st["c_bias"][..., 0], "cor1", "Coriolis/centripetal, shoulder", "interaction")
    add(-st["c_bias"][..., 1], "cor2", "Coriolis/centripetal, elbow", "interaction")
    add(st["tau_act"][..., 0], "tau1", "motor torque, shoulder", "applied")
    add(st["tau_act"][..., 1], "tau2", "motor torque, elbow", "applied")
    add(st["contact_flag"], "contact", "any contact", "contact")
    add(np.log1p(st["contact_floor"]), "c_floor", "log floor force", "contact")
    add(np.log1p(st["contact_wall"]), "c_wall", "log wall force", "contact")
    add(np.log1p(st["contact_on_fore"]), "c_fore", "log force on forearm", "contact")
    add(np.clip(st["tau_contact"][..., 0], -60, 60), "ctq1", "contact torque, shoulder", "contact")
    add(np.clip(st["tau_contact"][..., 1], -60, 60), "ctq2", "contact torque, elbow", "contact")
    add(st["ke"], "ke", "kinetic energy", "energy")
    add(st["pe"], "pe", "potential energy", "energy")
    return np.stack(cols, -1), vs


# ------------------------------------------------------------------ analysis helpers

def r2_per_unit(acts: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Squared Pearson correlation between every unit and every target. acts [N,U], Y [N,V] -> [U,V]."""
    a = (acts - acts.mean(0)) / (acts.std(0) + 1e-8)
    y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    return ((a.T @ y) / len(a)) ** 2


def binned_r2(x: np.ndarray, y: np.ndarray, bins: int = 30) -> float:
    """Fraction of variance in activation y explained by a nonlinear 1-D function of x
    (piecewise-constant over quantile bins). Catches rectified / tuned responses
    that Pearson r misses."""
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    means = np.bincount(idx, y, bins) / np.maximum(np.bincount(idx, minlength=bins), 1)
    resid = y - means[idx]
    return 1.0 - resid.var() / (y.var() + 1e-12)


# ------------------------------------------------------------------ figure style

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SHOULDER, ELBOW = "#2a78d6", "#eb6834"   # matches the rendered arm (blue upper arm, orange forearm)
DIVERGING = ["#1c5cab", "#5598e7", "#9ec5f4", "#f0efec", "#f2a7a6", "#e66767", "#b52f2f"]
SEQUENTIAL = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _installed_fonts(names):
    """Keep only installed families (avoids a findfont warning per text element)."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [n for n in names if n in have] or ["DejaVu Sans"]


def setup_style():
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        # listed in font.family (not behind the sans-serif alias) so matplotlib falls back per glyph:
        # Arial (Windows) has no subscript digits, DejaVu Sans supplies q₁, q₂.
        "font.family": _installed_fonts(["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]),
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2.0, "axes.prop_cycle": mpl.cycler(color=SERIES),
        "image.cmap": "viridis",
    })
    div = LinearSegmentedColormap.from_list("physdiv", DIVERGING)
    seq = LinearSegmentedColormap.from_list("physseq", SEQUENTIAL)
    for cm in (div, seq):
        try:
            mpl.colormaps.register(cm)
        except ValueError:
            pass
    return div, seq


def savefig(fig, name: str):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path
