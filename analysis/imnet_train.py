"""imnet_ ImageNet TRAINING images and all-position activation dumps for the essay-scale SAEs (imnet_sae2.py).

Essay ("What Dictionaries Learn", its squeezed-model SAE): "trained on the squeezed layer at 25.6 million positions
from 200,000 ImageNet training images." Here (OURS):

data      46 of the 294 train parquet shards of the HF dataset ILSVRC/imagenet-1k, evenly spread over the shard list
          (round(linspace(0, 293, 46))), 4,358 images each = 200,468 images in data/imagenet_train/data. The HF shards
          are shuffled (a single shard covers ~987 classes), so evenly spread shards cover all classes roughly
          uniformly (checked in `coverage()`). Same eval preprocessing as imnet_common (shorter side 256, bilinear +
          antialias, centre crop 224) into data/imnet_cache/imagenet_train/images_u8.npy (30 GB) + labels.npy.
acts      For each (model, layer): the layer's PRE-activation at ALL 7x7 positions of every image, float16, row
          order = image-major then position (row = 49 * image + h * 7 + w), written as a .npy with plain file writes
          and read back with seek + fromfile (no memmap, so the process working set stays small under the memory
          guard). 200,468 x 49 = 9.82M rows per layer (60 GB at 3072-d).
          Layers: googlenet inception5b (1024, post = ReLU), convnext_tiny features.7.2.mlp (3072, post = GELU),
          convnext_tiny features.7 (stage-3 output = final residual stream, 768, no nonlinearity), optionally
          convnext_small features.7.2.mlp. Random-init controls on the first `n_init` images only.
          The extraction pass also stores per-image pole scores (spatial max / -min of the pre-activation) for the
          top-20 dominance rerun on training images.
held out  ImageNet val (data/imnet_cache/imagenet_val): the first 10,000 val images in a fixed random order, all 49
          positions (490k rows), same format.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import queue
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import imnet_common as C  # noqa: E402

TRAIN_DIR = os.path.join(C.DATA, "imagenet_train")
SHARDS = [int(s) for s in np.round(np.linspace(0, 293, 46)).astype(int)]
ACTS = os.path.join(C.CACHE, "sae2_acts")
LAYERS = {"googlenet": ["inception5b"], "convnext_tiny": ["features.7.2.mlp", "features.7"],
          "convnext_small": ["features.7.2.mlp"]}


def shard_files():
    return [os.path.join(TRAIN_DIR, "data", f"train-{i:05d}-of-00294.parquet") for i in SHARDS]


def download(workers=4):
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download

    def one(i):
        for attempt in range(5):
            try:
                return hf_hub_download("ILSVRC/imagenet-1k", f"data/train-{i:05d}-of-00294.parquet",
                                       repo_type="dataset", local_dir=TRAIN_DIR)
            except Exception as e:  # network hiccups
                print(f"  shard {i} attempt {attempt}: {e}", flush=True)
                time.sleep(10)
        raise RuntimeError(f"shard {i} failed")
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for n, f in enumerate(ex.map(one, SHARDS)):
            print(f"  [{n + 1}/{len(SHARDS)}] {os.path.basename(f)} ({time.time() - t0:.0f} s)", flush=True)


def coverage():
    import pyarrow.parquet as pq
    lab = np.concatenate([np.asarray(pq.read_table(f, columns=["label"]).column("label").to_pylist())
                          for f in shard_files()])
    cnt = np.bincount(lab, minlength=1000)
    return dict(n_images=int(len(lab)), n_classes=int((cnt > 0).sum()), per_class_min=int(cnt.min()),
                per_class_median=float(np.median(cnt)), per_class_max=int(cnt.max()),
                per_class_cv=float(cnt.std() / cnt.mean()))


# ------------------------------------------------------------------ raw .npy writer / reader (no memmap)

class NpyWriter:
    """Append rows to a .npy file with plain writes (header fixed up front)."""

    def __init__(self, fn, shape, dtype):
        self.fn, self.shape, self.dtype = fn, tuple(shape), np.dtype(dtype)
        self.f = open(fn + ".tmp", "wb")
        np.lib.format.write_array_header_1_0(self.f, dict(descr=np.lib.format.dtype_to_descr(self.dtype),
                                                           fortran_order=False, shape=self.shape))
        self.n = 0

    def write(self, a):
        a = np.ascontiguousarray(a, dtype=self.dtype)
        self.f.write(a.tobytes())
        self.n += len(a)

    def close(self):
        assert self.n == self.shape[0], (self.n, self.shape)
        self.f.close()
        os.replace(self.fn + ".tmp", self.fn)


class NpyReader:
    """Random row-block reads from a .npy without memmap: rows(i, n) -> array [n, ...]."""

    def __init__(self, fn):
        self.fn = fn
        with open(fn, "rb") as f:
            v = np.lib.format.read_magic(f)
            rd = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
            self.shape, _, self.dtype = rd(f)
            self.off = f.tell()
        self.row = int(np.prod(self.shape[1:])) * np.dtype(self.dtype).itemsize
        self.f = open(fn, "rb", buffering=0)
        self.lock = threading.Lock()

    def __len__(self):
        return self.shape[0]

    def rows(self, i, n):
        n = min(n, self.shape[0] - i)
        with self.lock:
            self.f.seek(self.off + i * self.row)
            b = self.f.read(n * self.row)
        return np.frombuffer(b, self.dtype).reshape(n, *self.shape[1:])

    def take(self, idx):
        """Sorted-ish arbitrary rows (one read per row; use for small sets)."""
        return np.stack([self.rows(int(i), 1)[0] for i in idx])


# ------------------------------------------------------------------ image cache

def build_image_cache(workers=10):
    from multiprocessing import Pool
    import pyarrow.parquet as pq
    d = C.cache_dir("imagenet_train")
    fimg, flab = os.path.join(d, "images_u8.npy"), os.path.join(d, "labels.npy")
    if os.path.exists(flab):
        return
    files = shard_files()
    tasks = [(f, g) for f in files for g in range(pq.ParquetFile(f).num_row_groups)]
    N = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    w = NpyWriter(fimg, (N, 224, 224, 3), np.uint8)
    labels = []
    t0 = time.time()
    with Pool(workers, maxtasksperchild=20) as pool:
        for x, lab in pool.imap(C._parquet_task, tasks):
            w.write(x)
            labels.append(lab)
            print(f"  cached {w.n}/{N} ({time.time() - t0:.0f} s)", flush=True)
    w.close()
    np.save(flab, np.concatenate(labels))


def train_batches(idx_range=None, bs=200):
    """Yield (global indices, input tensor) over the training-image cache, sequential reads, prefetched."""
    R = NpyReader(os.path.join(C.cache_dir("imagenet_train"), "images_u8.npy"))
    lo, hi = idx_range or (0, len(R))
    q = queue.Queue(maxsize=3)

    def reader():
        for i in range(lo, hi, bs):
            x = R.rows(i, min(bs, hi - i))
            q.put((np.arange(i, i + len(x)), torch.from_numpy(x.copy()).pin_memory()))
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()
    while (item := q.get()) is not None:
        j, xb = item
        yield j, C.to_input(xb)


def labels(dataset="imagenet_train"):
    return np.load(os.path.join(C.cache_dir(dataset), "labels.npy"))


# ------------------------------------------------------------------ activation dumps

def act_path(dataset, arch, trained, layer):
    os.makedirs(ACTS, exist_ok=True)
    return os.path.join(ACTS, f"{dataset}_{C.tag(arch, trained)}_{layer}.npy")


VAL_N = 10_000


def val_index():
    """Fixed random 10,000 of the 50,000 val images (sorted), the held-out positions."""
    return np.sort(np.random.default_rng(0).choice(50_000, VAL_N, replace=False))


@torch.no_grad()
def extract(arch, trained=True, dataset="imagenet_train", n_img=None, bs=200, poles=True):
    """All-position pre-activation dump of LAYERS[arch] for one model; train: first n_img images (default all)."""
    L = LAYERS[arch]
    fns = {l: act_path(dataset, arch, trained, l) for l in L}
    if all(os.path.exists(f) for f in fns.values()):
        return fns
    torch.backends.cudnn.benchmark = True
    m = C.load_model(arch, trained)
    rec = C.Recorder(m, arch).attach(L)
    rec.record = set(L)
    if dataset == "imagenet_train":
        N = n_img or len(labels())
        it = train_batches((0, N), bs)
    else:
        idx = val_index()
        N = len(idx)
        it = C.batches(dataset, idx, bs)
    writers, pos, neg = {}, {}, {}
    preds = []
    t0 = time.time()
    for j, x in it:
        with torch.autocast("cuda", torch.float16):
            lo = m(x)
        preds.append(lo.float().argmax(1).cpu().numpy())
        for l in L:
            a = rec.acts[l]                                    # [B, C, 7, 7] pre-activation (or residual)
            B, Ch = a.shape[:2]
            if l not in writers:
                writers[l] = NpyWriter(fns[l], (N * a.shape[2] * a.shape[3], Ch), np.float16)
                pos[l], neg[l] = np.zeros((N, Ch), np.float16), np.zeros((N, Ch), np.float16)
            fl = a.float().flatten(2)
            sl = slice(writers[l].n // fl.shape[2], writers[l].n // fl.shape[2] + B)
            pos[l][sl] = fl.amax(-1).cpu().numpy()
            neg[l][sl] = (-fl.amin(-1)).cpu().numpy()
            writers[l].write(fl.permute(0, 2, 1).reshape(-1, Ch).half().cpu().numpy())
        rec.acts = {}
        done = writers[L[0]].n // 49
        if done % 10000 < bs:
            print(f"  [extract] {dataset} {C.tag(arch, trained)} {done}/{N} ({time.time() - t0:.0f} s)", flush=True)
    rec.detach()
    for l in L:
        writers[l].close()
        if poles:
            np.savez(fns[l].replace(".npy", "_poles.npz"), pos=pos[l], neg=neg[l], pred=np.concatenate(preds))
    del m
    torch.cuda.empty_cache()
    return fns


def main(step="all"):
    if step in ("download", "all"):
        download()
        cov = coverage()
        print("[train] coverage", cov, flush=True)
        json.dump(cov, open(os.path.join(TRAIN_DIR, "coverage.json"), "w"), indent=1)
    if step in ("cache", "all"):
        build_image_cache()
    if step in ("extract", "all"):
        for arch in ("googlenet", "convnext_tiny"):
            extract(arch, True, "imagenet_val")
            extract(arch, True)
            extract(arch, False, "imagenet_val")
            extract(arch, False, n_img=20_000)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["all"]))
