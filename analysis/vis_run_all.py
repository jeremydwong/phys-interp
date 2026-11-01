"""Rerun the whole vis_ ladder study (after data + training; see reports/results/vis.md "How to rerun").

  uv run python sim/vis_shapes.py
  uv run python models/vis_train.py --data {shapes,cifar} --step {old,recipe,arch} --seed {0,1,2} --epochs {40,60}
  uv run python models/vis_train.py --data {shapes,cifar} --step readout --epochs {30,40}
  uv run python analysis/vis_run_all.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import vis_ablate, vis_census, vis_dict, vis_featvis, vis_figures, vis_rotation  # noqa: E402

if __name__ == "__main__":
    vis_census.main()
    vis_ablate.main()
    vis_dict.main()
    vis_rotation.main()
    vis_featvis.main()
    vis_figures.main()
