"""Run the whole imnet_ study (pretrained ImageNet classifiers on ImageNet val; see reports/results/imnet.md).

  uv run python analysis/imnet_run_all.py --data imagenet_val       # main run
  uv run python analysis/imnet_run_all.py --data imagenetv2         # secondary check (10k images)

Needs data/imagenet_val/data/validation-0000N-of-00014.parquet (HF ILSVRC/imagenet-1k, validation split),
data/imagenet_val/meta/classes.py (same repo) and, for the secondary set and the held-out fits,
data/imagenetv2/images/ (ImageNetV2 matched-frequency); WordNet in data/nltk_data (nltk.download("wordnet")).
Steps are cached / resumable: data/imnet_cache/<dataset>/..., reports/results/imnet_*.json.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import imnet_common as C  # noqa: E402
from analysis import imnet_extract, imnet_figures, imnet_purity, imnet_rotation, imnet_sae, imnet_weak  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="imagenet_val", choices=["imagenet_val", "imagenetv2"])
    ap.add_argument("--steps", default="cache,extract,purity,sae,rotation,weak,figures")
    a = ap.parse_args()
    steps = a.steps.split(",")
    other = imnet_sae.FIT[a.data]
    if "cache" in steps:
        for ds in (a.data, other):          # the other set is where SAEs and cutoffs are fit
            C.build_image_cache(ds)
    if "extract" in steps:
        for ds in (a.data, other):
            imnet_extract.main(ds)
    if "purity" in steps:
        imnet_purity.main(a.data)
    if "sae" in steps:
        imnet_sae.main(a.data)
        imnet_sae.zero_layer_control(a.data)
    if "rotation" in steps:
        imnet_rotation.main(a.data)
    if "weak" in steps:
        imnet_weak.main(a.data)
    if "figures" in steps:
        imnet_figures.main(a.data)
