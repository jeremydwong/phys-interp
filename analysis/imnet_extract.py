"""imnet_ extraction: one pass over a dataset per model (trained and random-init).

Per model, saves data/imnet_cache/<dataset>/<tag>_extract.npz:
  pred, margin [N]           top-1 prediction and top-1 minus top-2 softmax probability
  <layer>_pos, <layer>_neg   [N, C] float16 pole scores of PRE-activations: spatial max and -(spatial min)
and <tag>_samples.npy: last-layer PRE-activation vectors at `n_pos` random positions per image [n_pos * N, C]
float16 (post = ReLU / GELU of these), with the image index of each row in <tag>_samples_idx.npy.

Layers: the last layer (imnet_common.last_layer), a secondary view (googlenet: none beyond the branch
pre-activations; convnext: the stage-3 output = final residual stream, 768 ch) and one mid layer
(googlenet inception4e; convnext the last block of stage 2).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import imnet_common as C  # noqa: E402


def layers(arch):
    if arch == "googlenet":
        return [C.last_layer(arch), "inception4e"]
    mid = {"convnext_tiny": "features.5.8.mlp", "convnext_small": "features.5.26.mlp"}[arch]
    return [C.last_layer(arch), "features.7", mid]


def extract(dataset, arch, trained, n_pos=2, bs=200, force=False):
    d = C.cache_dir(dataset)
    t = C.tag(arch, trained)
    fn = os.path.join(d, f"{t}_extract.npz")
    if os.path.exists(fn) and not force:
        return fn
    torch.backends.cudnn.benchmark = True
    m = C.load_model(arch, trained)
    L = layers(arch)
    rec = C.Recorder(m, arch).attach(L)
    _, y = C.images(dataset)
    N = len(y)
    pos = {n: None for n in L}
    neg = {n: None for n in L}
    samples, sidx = [], []
    g = torch.Generator(device="cpu").manual_seed(0)

    def on_batch(j, acts):
        for n in L:
            a = acts[n].float()
            if pos[n] is None:
                pos[n] = np.zeros((N, a.shape[1]), np.float16)
                neg[n] = np.zeros((N, a.shape[1]), np.float16)
            fl = a.flatten(2)
            pos[n][j] = fl.amax(-1).cpu().numpy().astype(np.float16)
            neg[n][j] = (-fl.amin(-1)).cpu().numpy().astype(np.float16)
        a = acts[L[0]].float().flatten(2)                       # [B, C, P]
        P = a.shape[2]
        pi = torch.randint(0, P, (len(j), n_pos), generator=g).to(a.device)
        v = a.gather(2, pi[:, None, :].expand(-1, a.shape[1], -1))  # [B, C, n_pos]
        samples.append(v.permute(0, 2, 1).reshape(-1, a.shape[1]).cpu().numpy().astype(np.float16))
        sidx.append(np.repeat(j, n_pos))

    t0 = time.time()
    pred, margin = C.predict(m, dataset, bs=bs, rec=rec, record=L, on_batch=on_batch)
    rec.detach()
    acc = float((pred == y).mean())
    print(f"[extract] {dataset} {t}: top-1 {100 * acc:.3f}%  ({time.time() - t0:.0f} s)", flush=True)
    np.savez(fn, pred=pred, margin=margin, layers=np.array(L),
             **{f"{n}_pos": pos[n] for n in L}, **{f"{n}_neg": neg[n] for n in L})
    np.save(os.path.join(d, f"{t}_samples.npy"), np.concatenate(samples))
    np.save(os.path.join(d, f"{t}_samples_idx.npy"), np.concatenate(sidx))
    del m
    torch.cuda.empty_cache()
    return fn


def load(dataset, arch, trained):
    d = C.cache_dir(dataset)
    t = C.tag(arch, trained)
    z = np.load(os.path.join(d, f"{t}_extract.npz"))
    return {k: z[k] for k in z.files}


def load_samples(dataset, arch, trained, mmap=True):
    d = C.cache_dir(dataset)
    t = C.tag(arch, trained)
    return np.load(os.path.join(d, f"{t}_samples.npy"), mmap_mode="r" if mmap else None)


def main(dataset="imagenet_val", archs=C.MODELS):
    for arch in archs:
        for trained in (True, False):
            extract(dataset, arch, trained)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["imagenet_val"]))
