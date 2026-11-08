"""Run the pretrained imnet_ pipeline on the essay's own specimen, Lucid's 2014 InceptionV1 (arch
`inceptionv1_lucid`, see imnet_common.LucidInceptionV1), so every pretrained number has a like-for-like value.

  uv run python analysis/imnet_lucid_pipeline.py --steps extract,purity,weak,sae2,rotation,figures [--weak-minutes 40]

Stages reuse the existing modules unchanged (imnet_extract, imnet_purity, imnet_weak, imnet_train + imnet_sae2,
imnet_rotation, imnet_figures / imnet_sae2_fig); results land in the existing json files under the new arch key
(imnet_purity.json, imnet_weak.json, imnet_sae2.json runs l5b_*, imnet_rotation.json) and the per-stage wall time
and peak GPU memory in reports/results/imnet_lucid_pipeline.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import imnet_common as C  # noqa: E402

ARCH = "inceptionv1_lucid"
OUT = os.path.join(C.RESULTS, "imnet_lucid_pipeline.json")


def _log(stage, t0, extra=None):
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out.setdefault("stages", {})[stage] = dict(minutes=(time.time() - t0) / 60,
                                               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, **(extra or {}))
    out["arch"] = ARCH
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[pipeline] {stage}: {out['stages'][stage]['minutes']:.1f} min, peak {out['stages'][stage]['peak_gpu_gb']:.2f} GB",
          flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main(steps, weak_minutes=None, dataset="imagenet_val"):
    torch.cuda.reset_peak_memory_stats()
    if "extract" in steps:
        from analysis import imnet_extract
        t0 = time.time()
        m1008, gather = C.lucid_class_map()
        z = dict(__import__("numpy").load(C.LUCID_MAP))
        extra = dict(modal_map_classes_covered=int(z["modal_cov"]), matched_map_differs_on_outputs=int(z["n_diff"]))
        for trained in (True, False):
            imnet_extract.extract(dataset, ARCH, trained, force=trained)  # the old file is imnet_lucid.py's schema
        _log("extract", t0, extra)
    if "purity" in steps:
        from analysis import imnet_purity
        t0 = time.time()
        imnet_purity.main(dataset, archs=(ARCH,))
        _log("purity", t0)
    if "weak" in steps:
        from analysis import imnet_weak
        t0 = time.time()
        imnet_weak.main(dataset, (ARCH,), weak_minutes)
        _log("weak", t0, dict(max_minutes=weak_minutes))
    if "sae2" in steps:
        from analysis import imnet_sae2, imnet_train
        t0 = time.time()
        imnet_train.extract(ARCH, True, "imagenet_val")
        imnet_train.extract(ARCH, True)
        imnet_train.extract(ARCH, False, "imagenet_val")
        imnet_train.extract(ARCH, False, n_img=20_000)
        _log("sae2_dumps", t0)
        t0 = time.time()
        imnet_sae2.main(only=["l5b"])
        _log("sae2", t0)
    if "rotation" in steps:
        from analysis import imnet_rotation
        t0 = time.time()
        imnet_rotation.main(dataset, archs=(ARCH,))
        _log("rotation", t0)
    if "figures" in steps:
        from analysis import imnet_figures, imnet_sae2_fig
        t0 = time.time()
        imnet_figures.main(dataset)
        imnet_sae2_fig.main()
        _log("figures", t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="extract,purity,weak,sae2,rotation,figures")
    ap.add_argument("--weak-minutes", type=float, default=None, help="time cap for the weak grid (headline cells first)")
    a = ap.parse_args()
    main(a.steps.split(","), a.weak_minutes)
