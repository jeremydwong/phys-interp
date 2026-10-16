"""Run every video-model analysis end to end (after models/video_train.py has produced checkpoints).

    uv run python analysis/video_run_all.py [--train]

Writes figures to reports/figs/video_*.png and numbers to reports/results/video_*.json,
then merges them into reports/results/video.json.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STEPS = ["video_predict", "video_census", "video_probes", "video_gallery", "video_featvis",
         "video_contact", "video_accel", "video_causal", "video_dynamics"]


def main():
    if "--train" in sys.argv:
        subprocess.run([sys.executable, os.path.join(ROOT, "models", "video_train.py")], check=True)
    for s in STEPS:
        print(f"=== {s}", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "analysis", f"{s}.py")], check=True, cwd=ROOT)
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "reports", "results", "video_*.json"))):
        out[os.path.basename(p)[len("video_"):-5]] = json.load(open(p))
    json.dump(out, open(os.path.join(ROOT, "reports", "results", "video.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
