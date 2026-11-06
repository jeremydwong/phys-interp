"""Shared pieces of the imnet_ study: the essay's own setting (pretrained ImageNet classifiers, ImageNet val).

Data. `imagenet_val`: ILSVRC2012 validation, 50,000 images, from the HF dataset ILSVRC/imagenet-1k, 14 parquet
files in data/imagenet_val/data (global index = file order, then row order). `imagenetv2`: ImageNetV2
matched-frequency, 10,000 images in data/imagenetv2/images/<split>/<class index>/*.jpeg. Both are preprocessed
once (resize shorter side to 256, bilinear + antialias, centre crop 224; the standard eval transform) into a
uint8 memmap data/imnet_cache/<dataset>/images_u8.npy [N, 224, 224, 3] plus labels.npy, so every later pass
(extraction, the 150 or so ablation passes) reads crops from disk instead of decoding JPEGs.

Classes and superclasses. Class index = the standard ILSVRC2012 index (sorted WordNet ids; the order of
classes.py in the HF repo). Superclass (OURS): each class goes to the shallowest WordNet ancestor, on its longest
hypernym path, that covers at most K of the 1000 classes. K = 60 gives 125 groups ("bird", "structure",
"working dog", "terrier", ...); K = 130 gives 47 coarser groups ("canine", "device", "conveyance", ...).

Models. torchvision googlenet (IMAGENET1K_V1) and convnext_tiny / convnext_small (IMAGENET1K_V1), and a random-
init control of each (torchvision's default init, same seed every time, eval mode).
"""
from __future__ import annotations

import collections
import glob
import io
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "imnet_cache")
RESULTS = os.path.join(ROOT, "reports", "results")
FIGS = os.path.join(ROOT, "reports", "figs")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
MODELS = ("googlenet", "convnext_tiny", "convnext_small")
ESSAY = "Cammarata (2026), Neural Networks Are Already Clean, Goodfire"


def cache_dir(dataset):
    d = os.path.join(CACHE, dataset)
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------ images

def _prep(img):
    from PIL import Image
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img))
    img = img.convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    nw, nh = max(256, round(w * s)), max(256, round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR, reducing_gap=None)
    l, t = (nw - 224) // 2, (nh - 224) // 2
    return np.asarray(img.crop((l, t, l + 224, t + 224)), dtype=np.uint8)


def _parquet_task(args):
    fn, rg = args
    import pyarrow.parquet as pq
    t = pq.ParquetFile(fn).read_row_group(rg, columns=["image", "label"])
    imgs = t.column("image").to_pylist()
    lab = np.asarray(t.column("label").to_pylist(), np.int64)
    return np.stack([_prep(d["bytes"]) for d in imgs]), lab


def _files_task(fns):
    return np.stack([_prep(open(f, "rb").read()) for f in fns])


def parquet_files():
    return sorted(glob.glob(os.path.join(DATA, "imagenet_val", "data", "validation-*-of-00014.parquet")))


def imagenetv2_files():
    base = glob.glob(os.path.join(DATA, "imagenetv2", "images", "*"))[0]
    out = []
    for c in range(1000):
        for f in sorted(glob.glob(os.path.join(base, str(c), "*"))):
            out.append((f, c))
    return out


def build_image_cache(dataset, workers=10):
    """Decode + preprocess every image once into a uint8 memmap. Idempotent."""
    from multiprocessing import Pool
    d = cache_dir(dataset)
    fimg, flab = os.path.join(d, "images_u8.npy"), os.path.join(d, "labels.npy")
    if os.path.exists(flab):
        return
    if dataset == "imagenet_val":
        import pyarrow.parquet as pq
        files = parquet_files()
        assert len(files) == 14, f"expected 14 parquet files, found {len(files)}"
        tasks = [(f, g) for f in files for g in range(pq.ParquetFile(f).num_row_groups)]
        N = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        fn_task = _parquet_task
    else:
        items = imagenetv2_files()
        N = len(items)
        tasks = [[f for f, _ in items[i:i + 500]] for i in range(0, N, 500)]
        fn_task = _files_task
    X = np.lib.format.open_memmap(fimg + ".tmp", mode="w+", dtype=np.uint8, shape=(N, 224, 224, 3))
    labels = []
    i = 0
    with Pool(workers) as pool:
        for r in pool.imap(fn_task, tasks):
            x, lab = r if isinstance(r, tuple) else (r, None)
            X[i:i + len(x)] = x
            if lab is not None:
                labels.append(lab)
            i += len(x)
            print(f"  cached {i}/{N}", flush=True)
    assert i == N
    X.flush()
    del X
    os.replace(fimg + ".tmp", fimg)
    y = np.concatenate(labels) if labels else np.asarray([c for _, c in imagenetv2_files()], np.int64)
    np.save(flab, y)


def images(dataset):
    d = cache_dir(dataset)
    return np.load(os.path.join(d, "images_u8.npy"), mmap_mode="r"), np.load(os.path.join(d, "labels.npy"))


def to_input(xu8):
    """uint8 [B, 224, 224, 3] (numpy) -> normalised float [B, 3, 224, 224] on DEV, channels_last."""
    if not torch.is_tensor(xu8):
        xu8 = torch.from_numpy(np.ascontiguousarray(xu8))
    x = xu8.to(DEV, non_blocking=True).permute(0, 3, 1, 2).float()
    x = (x - MEAN.to(DEV)) / STD.to(DEV)
    return x.contiguous(memory_format=torch.channels_last)


def batches(dataset, idx=None, bs=200):
    """Yield (global indices, input tensor) over the dataset (or the given sorted indices). A reader thread
    prefetches the next uint8 batches from the memmap while the GPU works."""
    import queue
    import threading
    X, _ = images(dataset)
    idx = np.arange(len(X)) if idx is None else np.sort(np.asarray(idx))
    q = queue.Queue(maxsize=3)

    def reader():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            xb = X[j[0]:j[-1] + 1] if np.all(np.diff(j) == 1) else X[j]
            q.put((j, torch.from_numpy(np.ascontiguousarray(xb)).pin_memory()))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    while (item := q.get()) is not None:
        j, xb = item
        yield j, to_input(xb)


# ------------------------------------------------------------------ classes

def class_meta():
    """-> (wnids [1000], short names [1000]) in the standard index order."""
    s = open(os.path.join(DATA, "imagenet_val", "meta", "classes.py"), encoding="utf-8").read()
    w = re.findall(r'"(n\d{8})"\s*:\s*"([^"]*)"', s)
    assert len(w) == 1000
    return [a for a, _ in w], [b.split(",")[0].strip() for _, b in w]


def superclasses(K=60):
    """OURS. -> (group id per class [1000], group names). Shallowest ancestor (longest hypernym path) covering
    <= K of the 1000 classes."""
    import nltk
    nltk.data.path.insert(0, os.path.join(DATA, "nltk_data"))
    from nltk.corpus import wordnet as wn
    wnids, _ = class_meta()
    syn = [wn.synset_from_pos_and_offset("n", int(x[1:])) for x in wnids]
    cnt = collections.Counter()
    for sy in syn:
        for a in {a for p in sy.hypernym_paths() for a in p}:
            cnt[a] += 1
    sup = []
    for sy in syn:
        p = max(sy.hypernym_paths(), key=len)
        sup.append(next(a for a in p if cnt[a] <= K))
    names = sorted({s.name() for s in sup})
    gid = np.array([names.index(s.name()) for s in sup])
    return gid, [n.split(".")[0] for n in names]


# ------------------------------------------------------------------ models and layers

def load_model(arch, trained=True, seed=0):
    import torchvision.models as tvm
    torch.manual_seed(seed)
    if arch == "googlenet":
        if trained:     # torchvision drops the aux heads itself and sets transform_input=True
            m = tvm.googlenet(weights="IMAGENET1K_V1")
        else:
            m = tvm.googlenet(weights=None, aux_logits=False, init_weights=True, transform_input=True)
            m = _bn_calibrated(m.to(DEV), seed)
    else:
        m = getattr(tvm, arch)(weights="IMAGENET1K_V1" if trained else None)
    return m.eval().to(DEV).to(memory_format=torch.channels_last)


@torch.no_grad()
def _bn_calibrated(m, seed, n=2000):
    """Random-init GoogLeNet control (OURS). torchvision's init (truncated normal, std 0.01) with fresh BatchNorm
    running stats (mean 0, var 1) makes eval-mode activations shrink ~100x per layer, so the last layer is exactly 0
    in fp16 and the control is degenerate. We re-estimate every BN's running mean/var on n ImageNetV2 images
    (cumulative average, fp32, no gradient, weights untouched), so each BN standardises its input as it would in
    training. Cached in data/imnet_cache/googlenet_init_s<seed>_bncal.pt."""
    fn = os.path.join(CACHE, f"googlenet_init_s{seed}_bncal.pt")
    if os.path.exists(fn):
        m.load_state_dict(torch.load(fn))
        return m
    bns = [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm2d)]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None
        b.train()
    for _, x in batches("imagenetv2", np.arange(n), 100):
        m(x)
    m.eval()
    os.makedirs(CACHE, exist_ok=True)
    torch.save(m.state_dict(), fn)
    return m


def last_layer(arch):
    """The tensors called "the last layer" here (see imnet.md):
    googlenet: inception5b, its 1024 output channels = concat of the 4 branch outputs; we hook each branch's final
      BatchNorm (pre-ReLU), so the poles see the pre-activation; post = ReLU(pre) is the block output exactly.
    convnext: the last block's MLP hidden, pwconv1 output (pre-GELU), 3072 (tiny) / 3072 (small) units;
      post = GELU(pre). Secondary: the stage-3 output (768 channels, residual stream)."""
    if arch == "googlenet":
        return "inception5b"
    return "features.7.2.mlp"


def block_names(arch):
    """Layers of the weak-activation grid, in essay order. googlenet: conv1..3 + 9 inception blocks (the essay's
    conv2d0..2, mixed3a..5b); convnext: every block's MLP hidden."""
    if arch == "googlenet":
        return ["conv1", "conv2", "conv3", "inception3a", "inception3b", "inception4a", "inception4b", "inception4c",
                "inception4d", "inception4e", "inception5a", "inception5b"]
    depth = {"convnext_tiny": (3, 3, 9, 3), "convnext_small": (3, 3, 27, 3)}[arch]
    return [f"features.{2 * s + 1}.{b}.mlp" for s in range(4) for b in range(depth[s])]


def _resolve(m, name):
    for p in name.split("."):
        m = getattr(m, p) if not p.isdigit() else m[int(p)]
    return m


class Recorder:
    """Forward hooks that capture / edit the named layers.

    For googlenet inception blocks the captured tensor is the pre-ReLU concat of branch BN outputs [B, C, H, W];
    for conv1..3 the pre-ReLU BN output. For convnext ".mlp" names it is the pwconv1 output (pre-GELU) [B, H, W, 4C],
    returned as [B, 4C, H, W]. `edit[name] = f(post) -> post` modifies the post-activation in the forward pass
    (googlenet: the block/conv output; convnext: the GELU output, residual bypass intact)."""

    def __init__(self, model, arch):
        self.m, self.arch, self.handles = model, arch, []
        self.record, self.acts, self.edit = set(), {}, {}

    def attach(self, names):
        for n in names:
            if self.arch == "googlenet":
                mod = _resolve(self.m, n)
                if n.startswith("inception"):
                    bns = [mod.branch1.bn, mod.branch2[1].bn, mod.branch3[1].bn, mod.branch4[1].bn]
                    store = {}
                    for k, bn in enumerate(bns):
                        self.handles.append(bn.register_forward_hook(self._store(store, k)))
                    self.handles.append(mod.register_forward_hook(self._inc_out(n, store)))
                else:
                    self.handles.append(mod.bn.register_forward_hook(self._pre(n)))
                    self.handles.append(mod.register_forward_hook(self._post_edit(n)))
            elif not n.endswith(".mlp"):        # plain module output (e.g. convnext stage output, residual stream)
                self.handles.append(_resolve(self.m, n).register_forward_hook(self._pre(n)))
                self.handles.append(_resolve(self.m, n).register_forward_hook(self._post_edit(n)))
            else:
                blk = _resolve(self.m, n[:-4])
                self.handles.append(blk.block[3].register_forward_hook(self._pre_nhwc(n)))
                self.handles.append(blk.block[4].register_forward_hook(self._post_edit_nhwc(n)))
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _store(self, store, k):
        def f(mod, inp, out):
            store[k] = out.clone()      # BasicConv2d applies ReLU in place right after the BN
        return f

    def _inc_out(self, n, store):
        def f(mod, inp, out):
            if n in self.record:
                self.acts[n] = torch.cat([store[k] for k in range(4)], 1)
            if n in self.edit:
                return self.edit[n](out)
        return f

    def _pre(self, n):
        def f(mod, inp, out):
            if n in self.record:
                self.acts[n] = out.clone()  # clone: googlenet's BasicConv2d applies ReLU in place after the BN
        return f

    def _post_edit(self, n):
        def f(mod, inp, out):
            if n in self.edit:
                return self.edit[n](out)
        return f

    def _pre_nhwc(self, n):
        def f(mod, inp, out):
            if n in self.record:
                self.acts[n] = out.permute(0, 3, 1, 2)
        return f

    def _post_edit_nhwc(self, n):
        def f(mod, inp, out):
            if n in self.edit:
                return self.edit[n](out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return f


def post_act(arch, pre):
    return F.relu(pre) if arch == "googlenet" else F.gelu(pre)


@torch.no_grad()
def predict(model, dataset, idx=None, bs=200, rec=None, record=(), on_batch=None):
    """Top-1 predictions (and top-2 margin) over the dataset. `on_batch(j, rec.acts)` sees recorded activations."""
    preds, margins = [], []
    if rec is not None:
        rec.record = set(record)
    for j, x in batches(dataset, idx, bs):
        with torch.autocast("cuda", torch.float16):
            lo = model(x)
        lo = lo.float()
        t2 = lo.softmax(1).topk(2, 1).values
        preds.append(lo.argmax(1).cpu().numpy())
        margins.append((t2[:, 0] - t2[:, 1]).cpu().numpy())
        if on_batch is not None:
            on_batch(j, rec.acts)
        if rec is not None:
            rec.acts = {}
    if rec is not None:
        rec.record = set()
    return np.concatenate(preds), np.concatenate(margins)


def tag(arch, trained):
    return arch + ("" if trained else "_init")
