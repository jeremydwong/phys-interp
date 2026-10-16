"""Run the full state-MLP (candidate C) study end to end.

    uv run python analysis/state_run_all.py            # analyses only (expects trained checkpoints)
    uv run python analysis/state_run_all.py --train    # retrain first (60k steps)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = ["state_baselines", "state_census", "state_probes", "state_segments", "state_gallery", "state_featvis",
         "state_contact", "state_causal", "state_dynamics"]

if __name__ == "__main__":
    if "--train" in sys.argv:
        subprocess.run([sys.executable, os.path.join(ROOT, "models", "state_mlp.py"), "60000"], check=True, cwd=ROOT)
    for s in STEPS:
        print(f"==== {s}", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "analysis", f"{s}.py")], check=True, cwd=ROOT)
