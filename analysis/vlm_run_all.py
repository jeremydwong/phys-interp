"""Run the VLM study stage by stage, one process per step (each step loads and frees its own model).

    uv run python analysis/vlm_run_all.py [stage ...]     # default: all stages in order

Stages: extract, census, probe, gallery, contact, behaviour, general.
After each stage the JSON results in data/vlm/ are copied to reports/results/vlm_<name>.json
and merged into reports/results/vlm.json. Set PYTHONUTF8=1 on Windows (scripts print unicode).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "reports", "results")
VDIR = os.path.join(ROOT, "data", "vlm")

STAGES = {
    "extract": [["vlm_extract", "video"], ["vlm_extract", "image"], ["vlm_extract", "randvis"]],
    "census": [["vlm_census"], ["vlm_census", "fig"]],
    "probe": [["vlm_probe"], ["vlm_probe", "fig"]],
    "probe_mlp": [["vlm_probe", "mlp"]],          # max-pooled unit probes + figures only (resume after "probe" main)
    "gallery": [["vlm_gallery"], ["vlm_gallery", "maps"]],
    "contact": [["vlm_contact"]],
    "behaviour": [["vlm_behaviour", "collect"], ["vlm_behaviour", "analyze"]],
    "general": [["vlm_general", "bands"], ["vlm_general", "collect"], ["vlm_general", "analyze"]],
}


def collect_results():
    os.makedirs(RESULTS, exist_ok=True)
    for p in glob.glob(os.path.join(VDIR, "*.json")):
        name = os.path.basename(p)[:-5]
        shutil.copy(p, os.path.join(RESULTS, f"vlm_{name}.json"))
    out = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "vlm_*.json"))):
        out[os.path.basename(p)[len("vlm_"):-5]] = json.load(open(p))
    json.dump(out, open(os.path.join(RESULTS, "vlm.json"), "w"), indent=1)


def main():
    stages = sys.argv[1:] or list(STAGES)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    for st in stages:
        for script, *args in STAGES[st]:
            print(f"=== {st}: {script} {' '.join(args)}", flush=True)
            subprocess.run([sys.executable, os.path.join(ROOT, "analysis", f"{script}.py"), *args],
                           check=True, cwd=ROOT, env=env)
        collect_results()
        print(f"=== stage {st} done", flush=True)


if __name__ == "__main__":
    main()
