"""Data access for the vis_ study: the procedural shapes set (sim/vis_shapes.py) and CIFAR-10.

load(name) -> dict with
  xtr, xte    uint8 [N, H, W, 3]       ytr, yte  int64 class labels
  ftr, fte    {factor: array}           ground-truth factors (CIFAR: only "label")
  kinds       {factor: "cat" | "cont"}
  n_classes, stem_stride, mean, std, res
"""
from __future__ import annotations

import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load(name: str) -> dict:
    if name == "shapes":
        from sim.vis_shapes import CATEGORICAL, CONTINUOUS, N_CLASSES, N_TRAIN
        x = np.load(os.path.join(DATA, "vis_shapes", "images.npy"), mmap_mode="r")
        z = np.load(os.path.join(DATA, "vis_shapes", "factors.npz"))
        f = {k: z[k] for k in CATEGORICAL + CONTINUOUS}
        kinds = {k: "cat" for k in CATEGORICAL} | {k: "cont" for k in CONTINUOUS}
        xtr, xte = np.ascontiguousarray(x[:N_TRAIN]), np.ascontiguousarray(x[N_TRAIN:])
        ftr = {k: v[:N_TRAIN] for k, v in f.items()}
        fte = {k: v[N_TRAIN:] for k, v in f.items()}
        mean, std = _stats(xtr)
        return dict(xtr=xtr, xte=xte, ytr=ftr["label"].astype(np.int64), yte=fte["label"].astype(np.int64),
                    ftr=ftr, fte=fte, kinds=kinds, n_classes=N_CLASSES, stem_stride=2, mean=mean, std=std, res=64)
    if name == "cifar":
        d = os.path.join(DATA, "cifar10", "cifar-10-batches-py")

        def rd(files):
            xs, ys = [], []
            for fn in files:
                with open(os.path.join(d, fn), "rb") as fh:
                    b = pickle.load(fh, encoding="latin1")
                xs.append(b["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
                ys.append(np.array(b["labels"]))
            return np.ascontiguousarray(np.concatenate(xs)), np.concatenate(ys).astype(np.int64)
        xtr, ytr = rd([f"data_batch_{i}" for i in range(1, 6)])
        xte, yte = rd(["test_batch"])
        mean, std = _stats(xtr)
        return dict(xtr=xtr, xte=xte, ytr=ytr, yte=yte, ftr={"label": ytr}, fte={"label": yte},
                    kinds={"label": "cat"}, n_classes=10, stem_stride=1, mean=mean, std=std, res=32)
    raise ValueError(name)


def _stats(x):
    s = x[:: max(1, len(x) // 5000)].astype(np.float32) / 255
    return tuple(float(v) for v in s.mean((0, 1, 2))), tuple(float(v) for v in s.std((0, 1, 2)))
