"""imnet_ SAEs retrained the essay's way (Gao et al. 2024 recipe, ImageNet TRAINING images, essay scale).

Question: does the near-copy share of SAE activity rise toward the essay's 88% ("On the last layer, 88% of its
activity is near-copies of single neurons") once the SAE is trained like the essay's ("a TopK SAE with 640
features, 2 active at each position (the essay's SAE recipe plus the usual auxiliary loss that keeps features from
dying), trained on the squeezed layer at 25.6 million positions from 200,000 ImageNet training images. It explains
82% of the variance of held-out positions.")?

Recipe (Gao et al. 2024; our scale choices marked OURS):
  data      imnet_train: all 49 positions of 200,468 ImageNet train images (9.82M vectors per layer, fp16 on disk),
            streamed in chunks of random 8,192-row blocks (OURS) and shuffled within the chunk.
  training  25.6M training vectors (the essay's number of positions) = 6,250 steps x 4,096 (OURS: we read the
            essay's "25.6 million positions" as training tokens), normalised MSE, AuxK (k_aux = d/2, alpha = 1/32,
            dead = not fired in the last 2M training vectors (OURS; Gao: 10M)), Adam eps 6.25e-10,
            lr = 2e-4 / sqrt(n_feat / 2^14) (Gao's width scaling, OURS to apply it here), grad clip 1, unit-norm
            decoder rows with parallel-gradient removal, encoder = decoder^T at init, b_pre at the geometric
            median, inputs scaled to unit mean squared norm.
  held out  10,000 ImageNet val images x 49 positions (490k vectors): FVE and mean activation per feature.
  near copy decoder row with max_i |cos(d_f, e_i)| >= 0.9 (clean_tools.near_copy_share, OURS); share = mean
            activation of near-copy features / total.
  controls  rotated basis (activations x Q^T before training, near copies against the rotated axes; also against
            the true neurons Q e_i), random-init network (trained on the first 20,000 images' positions),
            planted neurons (synthetic sparse codes on the neuron axes, and the same rotated).
  removal   imnet_sae.removal_test (x -> x - sum_{unmatched} z_f d_f, error kept; splices) on all 50k val images,
            plus the whole-block-zeroing control.
Writes reports/results/imnet_sae2.json (one entry per run, resumable).
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import clean_tools as ct  # noqa: E402
from analysis import imnet_common as C  # noqa: E402
from analysis import imnet_train as T  # noqa: E402

OUT = os.path.join(C.RESULTS, "imnet_sae2.json")
SAVE = os.path.join(C.CACHE, "sae2_saes")
LAYER = {"g5b": ("googlenet", "inception5b"), "cnt_mlp": ("convnext_tiny", "features.7.2.mlp"),
         "cnt_res": ("convnext_tiny", "features.7"), "cns_mlp": ("convnext_small", "features.7.2.mlp")}
TOKENS = 25_600_000
BS = 4096
DEAD = 2_000_000
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class BlockSampler:
    """Stream batches from an on-disk [N, d] fp16 .npy: each chunk = `nblk` random 8,192-row blocks read with
    seek+read (no memmap), shuffled, moved to the GPU in fp16; a reader thread prepares the next chunk."""

    def __init__(self, fn, bs=BS, chunk_bytes=0.6e9, block=8192, seed=0, n_rows=None):
        self.R = T.NpyReader(fn)
        self.N = n_rows or len(self.R)
        self.d = self.R.shape[1]
        self.bs, self.block = bs, block
        self.nblk = max(1, int(chunk_bytes / (2 * self.d * block)))
        self.rng = np.random.default_rng(seed)
        self.q = queue.Queue(maxsize=1)
        self.cur, self.pos = None, 0
        self.stop = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        nb = (self.N + self.block - 1) // self.block
        while not self.stop:
            starts = self.rng.choice(nb, min(self.nblk, nb), replace=False) * self.block
            x = np.concatenate([self.R.rows(int(s), min(self.block, self.N - int(s))) for s in starts])
            x = x[self.rng.permutation(len(x))]
            self.q.put(torch.from_numpy(x).pin_memory())

    def __call__(self, step):
        if self.cur is None or self.pos + self.bs > len(self.cur):
            self.cur = self.q.get().to("cuda", non_blocking=True)
            self.pos = 0
        b = self.cur[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return b

    def close(self):
        self.stop = True
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass


def init_sample(fn, n=50_000, seed=1):
    R = T.NpyReader(fn)
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(len(R) // 49, n // 49, replace=False)) * 49     # whole images
    return torch.from_numpy(np.concatenate([R.rows(int(s), 49) for s in starts]))


def eval_set(arch, trained, layer):
    """Held-out val positions as a CPU fp16 tensor (490k x d; 3 GB at 3072-d)."""
    R = T.NpyReader(T.act_path("imagenet_val", arch, trained, layer))
    return torch.from_numpy(np.ascontiguousarray(R.rows(0, len(R))))


def transform_for(arch, layer, Q=None):
    def f(x):
        x = x.float()
        if layer.endswith(".mlp") or arch == "googlenet" and layer.startswith("inception"):
            x = C.post_act(arch, x)
        return x @ Q.T if Q is not None else x
    return f


def lr_for(n_feat):
    return 2e-4 / (n_feat / 2 ** 14) ** 0.5


def summarise(sae, Q=None):
    share, frac, c = ct.near_copy_share(sae["W_dec"], sae["mean_z"])
    mz = sae["mean_z"]
    live = mz > 0
    r = dict(fve=sae["fve"], dead_frac_train=sae["dead_frac"], live_frac_val=float(live.mean()),
             near_copy_share=share, near_copy_frac_live=frac,
             share_by_kappa={str(kap): float(mz[c >= kap].sum() / mz.sum()) for kap in (0.5, 0.7, 0.8, 0.9, 0.95)},
             cos_hist=np.histogram(c[live], bins=20, range=(0, 1))[0].tolist(),
             cos_hist_weighted=np.histogram(c[live], bins=20, range=(0, 1), weights=mz[live] / mz.sum())[0].tolist(),
             cos_median_live=float(np.median(c[live])) if live.any() else 0.0,
             n_neurons_matched=int(len(np.unique(np.abs(np.asarray(sae["W_dec"].cpu())).argmax(1)[c >= 0.9]))))
    if Q is not None:   # near copies of the TRUE neurons, seen through the rotation (rows of Q^T = Q e_i)
        s2, f2, _ = ct.near_copy_share(sae["W_dec"], sae["mean_z"], basis=Q.T)
        r.update(near_copy_share_true_neurons=s2, near_copy_frac_live_true_neurons=f2)
    return r, c


def run_one(name, layer_key, width, k, rotated=False, trained=True, tokens=TOKENS, removal=True, save=False):
    arch, layer = LAYER[layer_key]
    fn = T.act_path("imagenet_train", arch, trained, layer)
    d = T.NpyReader(fn).shape[1]
    n_feat = width * d
    Q = torch.as_tensor(ct.random_rotation(d, 1234), dtype=torch.float32, device="cuda") if rotated else None
    tf = transform_for(arch, layer, Q)
    steps = tokens // BS
    samp = BlockSampler(fn, seed=zlib.crc32(name.encode()))
    t0 = time.time()
    sae = ct.train_topk_sae(init_sample(fn), n_feat, k, steps=steps, bs=BS, lr=lr_for(n_feat), seed=0,
                            sampler=samp, X_eval=eval_set(arch, trained, layer), transform=tf, aux_k=min(512, d // 2),
                            aux_coef=1 / 32, dead_tokens=DEAD, norm_mse=True, adam_eps=6.25e-10, grad_clip=1.0,
                            b_pre_init="median", scale_inputs=True, grad_proj=True, sparse=True,
                            log_every=500)
    samp.close()
    r, c = summarise(sae, Q)
    r.update(layer=layer_key, arch=arch, tensor=layer, trained=trained, rotated=rotated, width=width, n_feat=n_feat,
             k=k, d=d, steps=steps, tokens=steps * BS, lr=lr_for(n_feat), aux_k=min(512, d // 2), dead_tokens=DEAD,
             train_rows=samp.N, minutes=(time.time() - t0) / 60,
             history=[dict(h) for h in sae["history"]])
    print(f"[sae2] {name}: fve {r['fve']:.3f} dead {r['dead_frac_train']:.3f} near-copy share {r['near_copy_share']:.3f}"
          f" (live frac {r['near_copy_frac_live']:.3f}) {r['minutes']:.1f} min", flush=True)
    if removal and trained and not rotated:
        from analysis import imnet_sae as S
        near = torch.as_tensor(c >= 0.9, device="cuda")
        r["removal"] = S.removal_test("imagenet_val", arch, sae, near, bs=50,
                                      layer=layer if layer_key == "cnt_res" else None)
        print(f"[sae2] {name}: removal {r['removal']['remove_unmatched']['delta_pp']:+.2f} pp, "
              f"splice_all {r['removal']['splice_all']['delta_pp']:+.2f} pp", flush=True)
    if save:
        os.makedirs(SAVE, exist_ok=True)
        torch.save({k2: v.half().cpu() for k2, v in sae.items() if torch.is_tensor(v)} | dict(mean_z=sae["mean_z"]),
                   os.path.join(SAVE, f"{name}.pt"))
    del sae
    torch.cuda.empty_cache()
    return r


def planted_runs(d=1024, width=4, steps=6250):
    """Planted-neuron check with the new recipe: sparse non-negative codes on the neuron axes (n_act active of d,
    exponential magnitudes, Zipf-ish frequencies), streamed; and the same data under a random rotation."""
    out = {}
    g = torch.Generator(device="cuda").manual_seed(0)
    p = (1.0 / torch.arange(1, d + 1, device="cuda").float() ** 0.5)
    Q = torch.as_tensor(ct.random_rotation(d, 99), dtype=torch.float32, device="cuda")

    def gen(n, n_act):
        idx = torch.multinomial(p.expand(n, -1), n_act, replacement=False, generator=g)
        val = torch.empty(n, n_act, device="cuda").exponential_(generator=g)
        return torch.zeros(n, d, device="cuda").scatter_(1, idx, val)

    for n_act in (2, 8):
        Xe = gen(100_000, n_act)
        for rot in (False, True):
            tf = (lambda x: x.float() @ Q.T) if rot else (lambda x: x.float())
            sae = ct.train_topk_sae(gen(50_000, n_act), width * d, n_act, steps=steps, bs=BS, lr=lr_for(width * d),
                                    sampler=lambda s: gen(BS, n_act), X_eval=Xe, transform=tf, aux_k=min(512, d // 2),
                                    dead_tokens=DEAD, norm_mse=True, adam_eps=6.25e-10, grad_clip=1.0,
                                    b_pre_init="median", scale_inputs=True, grad_proj=True, sparse=True)
            share, frac, _ = ct.near_copy_share(sae["W_dec"], sae["mean_z"])
            _, _, cr = ct.near_copy_share(sae["W_dec"], sae["mean_z"], basis=Q.T if rot else None)
            out[f"k{n_act}_{'rotated' if rot else 'axes'}"] = dict(
                fve=sae["fve"], near_copy_share=share, near_copy_frac_live=frac, dead_frac_train=sae["dead_frac"],
                share_matching_planted=float(sae["mean_z"][cr >= 0.9].sum() / sae["mean_z"].sum()))
            print(f"[sae2] planted k={n_act} rot={rot}: {out[f'k{n_act}_' + ('rotated' if rot else 'axes')]}", flush=True)
    return out


@torch.no_grad()
def zero_resid_control():
    """Whole-tensor zeroing control for the residual-stream removal test (stage-3 output -> head sees zeros)."""
    from analysis import imnet_extract as E
    m = C.load_model("convnext_tiny", True)
    rec = C.Recorder(m, "convnext_tiny").attach(["features.7"])
    rec.edit = {"features.7": lambda a: a * 0}
    _, y = C.images("imagenet_val")
    p, _ = C.predict(m, "imagenet_val", rec=rec)
    rec.detach()
    base = E.load("imagenet_val", "convnext_tiny", True)["pred"]
    return dict(top1=float((p == y).mean()), delta_pp=float(100 * ((p == y).mean() - (base == y).mean())),
                changed=float((p != base).mean()))


SWEEP = (
    # name, layer, width, k, rotated, trained
    *[(f"{L}_w{w}_k{k}", L, w, k, False, True) for L in ("g5b", "cnt_res", "cnt_mlp") for w in (4, 16) for k in (2, 8, 32)],
    *[(f"{L}_w64_k{k}", L, 64, k, False, True) for L in ("g5b", "cnt_res") for k in (2, 32)],
    *[(f"{L}_w16_k{k}_rot", L, 16, k, True, True) for L in ("g5b", "cnt_res", "cnt_mlp") for k in (2, 32)],
    *[(f"{L}_w16_k{k}_init", L, 16, k, False, False) for L in ("g5b", "cnt_res", "cnt_mlp") for k in (2, 32)],
    # second pass: higher k (toward FVE >= 0.8) and ConvNeXt-S's last MLP
    *[(f"{L}_w{w}_k{k}", L, w, k, False, True) for L in ("g5b", "cnt_res", "cnt_mlp") for w, k in ((4, 128), (16, 64))],
    *[(f"cns_mlp_w{w}_k{k}", "cns_mlp", w, k, False, True) for w in (4, 16) for k in (2, 8, 32)],
    ("cns_mlp_w16_k32_rot", "cns_mlp", 16, 32, True, True),
)


def load_out():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def main(only=None):
    out = load_out()
    if "planted" not in out:
        out["planted"] = planted_runs()
        json.dump(out, open(OUT, "w"), indent=1)
    if "zero_resid_control" not in out:
        out["zero_resid_control"] = zero_resid_control()
        json.dump(out, open(OUT, "w"), indent=1)
    out.setdefault("runs", {})
    for name, L, w, k, rot, tr in SWEEP:
        if name in out["runs"] or (only and not any(o in name for o in only)):
            continue
        if not os.path.exists(T.act_path("imagenet_train", LAYER[L][0], tr, LAYER[L][1])):
            print(f"[sae2] skip {name}: no activations", flush=True)
            continue
        r = run_one(name, L, w, k, rot, tr)
        out = load_out()
        out.setdefault("runs", {})[name] = r
        json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
