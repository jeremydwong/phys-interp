"""Figure for report 1: what each model sees."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib.pyplot as plt
from analysis.common import load_states, load_frames, setup_style, savefig, INK2
from sim.arm import ArmRenderer

setup_style()
st = load_states("main"); fr = load_frames("main")
# find a test episode with a contact onset in the middle
best = (0, None, None)
for e in range(2000, 2400):
    c = st["contact_flag"][e]
    on = np.where((c[1:] > 0) & (c[:-1] == 0))[0] + 1
    on = on[(on > 20) & (on < 120)]
    for t in on:
        if c[t - 8:t].sum() == 0:
            speed = np.linalg.norm(st["ee_vel"][e, t - 1])
            if speed > best[0]: best = (speed, e, t)
_, e, t0 = best
idx = np.arange(t0 - 5, t0 + 4)
hi = ArmRenderer(224)(st["q"][e][idx])
g = load_states("grav"); gf = load_frames("grav")
fig, axes = plt.subplots(3, len(idx), figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.04, hspace=0.18))
for j, t in enumerate(idx):
    axes[0, j].imshow(hi[j]); axes[1, j].imshow(fr[e, t]); axes[2, j].imshow(gf[1000, 40 + j * 2])
    for i in range(3):
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([]); axes[i, j].grid(False)
        for s in axes[i, j].spines.values(): s.set_visible(False)
    lab = f"t{t - t0:+d}" if t != t0 else "contact"
    axes[0, j].set_title(lab, fontsize=7, color=INK2, loc="center", fontweight="normal")
for i, name in enumerate(["VLM · 224 px", "video model · 64 px", "gravity set · 64 px"]):
    axes[i, 0].set_ylabel(name, fontsize=7, color=INK2, rotation=0, ha="right", va="center")
savefig(fig, "setup_frames.png")
print(e, t0)
