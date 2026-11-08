"""imnet_ seed replication of the two settings that mattered in imnet_lucid_sweep.py: `default` vs `path`.

Why. The sweep rendered each of its 12 mixed5b units once per setting, and read "path anchoring moves the renders
toward the essay's 2026 column" off single renders: per-unit rank swings of tens (670 junco 397 -> 1 on the own
net) could be the anchor or could be the seed, and one seed per unit cannot tell these apart. Here the same 12
units (units_order of imnet_lucid_sweep.json: the 8 purity-picked + 41, 273, 642, 934) are rendered with 5 seeds
(0-4) under both settings and every render is scored by the sweep's three judges (the net itself through the
recovered class map, ConvNeXt-T, torchvision GoogLeNet; rank of the unit's top-20 majority class, lower is better).

Paired design. Both settings go through imnet_lucid_sweep.render_own (lucent's chain copied step by step: 128 px
Fourier image + colour decorrelation, pad 12 / jitter 8 / scale / rotate / jitter 4, x*255-117, bilinear upsample
to 224, Adam 0.05, 512 steps), which seeds torch and numpy, so for a given seed and batch the default and the path
render start from the same image and see the same transform draws; the only difference is the anchor term in the
first 64 steps (rho * cos(A(x), r_bar) on mixed5a, as in the sweep). Units are batched 8 per optimisation in the
sweep's order, so the `path` seed-0 renders of the sweep (reports/figs/imnet_fv2/inceptionv1_lucid/<unit>_path.png)
are the same code path and are reused (copied to the _s0 name). The sweep's `default` seed-0 renders are not
reusable: they came from imnet_lucid.py through lucent's render_vis, which seeds torch only, and lucent's jitter /
scale / rotate draw from the unseeded numpy RNG, so they are not reproducible from any seed; default is re-rendered
for all five seeds.

Writes reports/figs/imnet_fv2/inceptionv1_lucid/<unit>_<variant>_s<seed>.png (224 px), reports/results/
imnet_lucid_seeds.json (per unit x variant x seed: objective on the saved render, objective / dataset max, the three
ranks; per variant and judge: geometric-mean rank over picked / random units per seed, mean and sd across seeds;
per judge: units where path beats default in >= 3 of 5 seeds, and the paired count of path < default over the 60
(unit, seed) pairs with a two-sided sign-test p), reports/figs/imnet_lucid_seeds_grid.png (12 rows x [5 default |
5 path], ranks under each cell) and a .jpg copy for the web report.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from math import comb

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import imnet_common as C  # noqa: E402
from analysis.imnet_lucid import LAYER, LR, TAG, _font, branch_of, judge_ranks, load_lucid, own_ranks  # noqa: E402
from analysis.imnet_lucid_sweep import (CFG, JSON as SWEEP_JSON, NB, N_PICKED, OUT_DIR, PATH_STEPS, TOPK, UNITS, Hooks,  # noqa: E402
                                        gmean, objective_on, path_targets, render_own)

warnings.filterwarnings("ignore")
DEV = C.DEV
VARIANTS = ["default", "path"]
SEEDS = [0, 1, 2, 3, 4]
JSON = os.path.join(C.RESULTS, "imnet_lucid_seeds.json")
GRID = os.path.join(C.FIGS, "imnet_lucid_seeds_grid.png")
JUDGES = ["own", "convnext", "googlenet"]


def fn_of(u, v, s):
    return os.path.join(OUT_DIR, f"{u}_{v}_s{s}.png")


def sign_test_p(wins, losses):
    """Two-sided exact sign test on wins vs losses (ties dropped)."""
    n, k = wins + losses, min(wins, losses)
    if n == 0:
        return None
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


# ------------------------------------------------------------------ grid

def grid(res, short, out, T=160, gap=4, block_gap=28, label_w=250):
    units, seeds = res["units_order"], res["seeds"]
    f, fb, fs = _font(16), _font(18, True), _font(13)
    cell_h = T + 18
    row_h = cell_h + 8
    W = label_w + 2 * len(seeds) * (T + gap) + block_gap + 10
    H = 54 + len(units) * row_h
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    x_def = label_w
    x_path = label_w + len(seeds) * (T + gap) + block_gap
    title = "InceptionV1 (Lucid 2014) mixed5b: default vs path anchor, seeds 0-4"
    d.text((0, 6), title, font=fb, fill=(30, 30, 30))
    d.text((d.textlength(title, font=fb) + 30, 9), "under each cell: rank of the majority class, own net / ConvNeXt-T / GoogLeNet",
           font=fs, fill=(90, 90, 90))
    d.text((x_def, 30), "default (lucent recipe, 512 steps)      seeds 0 .. 4", font=f, fill=(60, 60, 60))
    d.text((x_path, 30), "path (+ anchor, first 64 steps)      seeds 0 .. 4", font=f, fill=(60, 60, 60))
    for i, u in enumerate(units):
        r = res["units"][str(u)]
        y0 = 54 + i * row_h
        d.text((0, y0 + 4), f"unit {u} ({r['role']})", font=f, fill=(40, 40, 40))
        d.text((0, y0 + 26), f"{short[r['majority']]} {r['count']}/20", font=f, fill=(60, 60, 60))
        for v, x0 in (("default", x_def), ("path", x_path)):
            for j, s in enumerate(seeds):
                rv = r["variants"][v][str(s)]
                x = x0 + j * (T + gap)
                im.paste(Image.open(fn_of(u, v, s)).convert("RGB").resize((T, T), Image.BICUBIC), (x, y0))
                d.text((x + 2, y0 + T + 2), f"{rv['rank_own']} / {rv['rank_convnext']} / {rv['rank_googlenet']}", font=fs, fill=(50, 50, 50))
    im.save(out)
    jpg = out[:-4] + ".jpg"
    for q in (88, 80, 72, 65):
        im.save(jpg, quality=q, optimize=True)
        if os.path.getsize(jpg) < 2.5 * 2 ** 20:
            break
    print("wrote", out, im.size, "jpg", f"{os.path.getsize(jpg) / 2 ** 20:.2f} MB", flush=True)


# ------------------------------------------------------------------ main

def main(force=False):
    t_all = time.time()
    torch.backends.cudnn.benchmark = True
    os.makedirs(OUT_DIR, exist_ok=True)
    sweep = json.load(open(SWEEP_JSON))
    units = [int(u) for u in sweep["units_order"]]
    assert units == list(UNITS)
    z = np.load(os.path.join(C.cache_dir("imagenet_val"), f"{TAG}_extract.npz"))
    S, m1008 = z[f"{LAYER}_pos"].astype(np.float32), z["map1008"]
    X, _ = C.images("imagenet_val")
    _, names = C.class_meta()
    short = [str(n).split(",")[0] for n in names]
    model = load_lucid()
    for q in model.parameters():
        q.requires_grad_(False)
    hk = Hooks(model)
    res = dict(tag=TAG, layer=LAYER, units_order=units, picked=units[:N_PICKED], random=units[N_PICKED:], seeds=SEEDS,
               variants=VARIANTS, cfg={v: CFG[v] for v in VARIANTS}, lr=LR, path_steps=PATH_STEPS, topk_path=TOPK, batch=NB,
               objective=sweep["objective"],
               design="both variants through imnet_lucid_sweep.render_own (torch + numpy seeded): same seed => same initial "
                      "image and transform draws, the anchor is the only difference. path seed 0 reused from the sweep; "
                      "default seed 0 re-rendered (the sweep's came from lucent render_vis with an unseeded numpy RNG).",
               units={str(u): dict(role="picked" if u in units[:N_PICKED] else "random", majority=sweep["units"][str(u)]["majority"],
                                   count=sweep["units"][str(u)]["count"], branch=list(branch_of(u)),
                                   dataset_max=sweep["units"][str(u)]["dataset_max"], rho=sweep["units"][str(u)]["rho"],
                                   variants={v: {} for v in VARIANTS}) for u in units},
               stages={})
    classes = [res["units"][str(u)]["majority"] for u in units]
    smax = np.array([res["units"][str(u)]["dataset_max"] for u in units])
    rho = torch.tensor([res["units"][str(u)]["rho"] for u in units], device=DEV)
    rbar = None

    def targets():
        """r_bar, computed on first need (only when some render is missing)."""
        nonlocal rbar
        if rbar is None:
            t0 = time.time()
            rbar = path_targets(model, hk, X, S, units)
            res["stages"]["path_targets"] = dict(time_s=time.time() - t0)
            torch.cuda.empty_cache()
        return rbar

    # renders
    for v in VARIANTS:
        for s in SEEDS:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            t0 = time.time()
            reused = False
            if v == "path" and s == 0 and not force and all(os.path.exists(os.path.join(OUT_DIR, f"{u}_path.png")) for u in units):
                for u in units:
                    shutil.copyfile(os.path.join(OUT_DIR, f"{u}_path.png"), fn_of(u, v, s))
                reused = True
            elif not force and all(os.path.exists(fn_of(u, v, s)) for u in units):
                reused = True
            else:
                for b in range(0, len(units), NB):
                    chunk = units[b:b + NB]
                    sl = slice(b, b + len(chunk))
                    out, _ = render_own(model, hk, chunk, CFG[v], rho=rho[sl], rbar=targets()[sl], seed=s)
                    for u, im in zip(chunk, out):
                        im.resize((224, 224), Image.BICUBIC).save(fn_of(u, v, s))
                torch.cuda.synchronize()
            res["stages"][f"{v}_s{s}"] = dict(time_s=time.time() - t0, peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30, reused=reused)
            print(f"[{v} seed {s}] {'reused' if reused else f'{time.time() - t0:.0f} s'}, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB", flush=True)
            torch.cuda.empty_cache()
    # judges, all 120 renders at once per judge
    convnext = C.load_model("convnext_tiny").to(memory_format=torch.contiguous_format)
    googlenet = C.load_model("googlenet").to(memory_format=torch.contiguous_format)
    for v in VARIANTS:
        for s in SEEDS:
            ims = [Image.open(fn_of(u, v, s)).convert("RGB") for u in units]
            obj = objective_on(model, hk, ims, units)
            ranks = dict(own=own_ranks(model, ims, classes, m1008), convnext=judge_ranks(convnext, ims, classes),
                         googlenet=judge_ranks(googlenet, ims, classes))
            for i, u in enumerate(units):
                res["units"][str(u)]["variants"][v][str(s)] = dict(objective=float(obj[i]), ratio=float(obj[i] / smax[i]),
                                                                   **{f"rank_{k}": ranks[k][i] for k in JUDGES})
    torch.cuda.empty_cache()
    # summaries
    R = lambda u, v, s, k: res["units"][str(u)]["variants"][v][str(s)][f"rank_{k}"]  # noqa: E731
    summ = {}
    for v in VARIANTS:
        summ[v] = {}
        for k in JUDGES:
            for role, grp in (("picked", units[:N_PICKED]), ("random", units[N_PICKED:])):
                per_seed = [gmean([R(u, v, s, k) for u in grp]) for s in SEEDS]
                summ[v][f"gmean_{k}_{role}"] = dict(per_seed=per_seed, mean=float(np.mean(per_seed)), sd=float(np.std(per_seed, ddof=1)))
        summ[v]["ratio_median"] = float(np.median([res["units"][str(u)]["variants"][v][str(s)]["ratio"] for u in units for s in SEEDS]))
    comp = {}
    for k in JUDGES:
        wins = sum(R(u, "path", s, k) < R(u, "default", s, k) for u in units for s in SEEDS)
        losses = sum(R(u, "path", s, k) > R(u, "default", s, k) for u in units for s in SEEDS)
        ties = len(units) * len(SEEDS) - wins - losses
        per_unit = {str(u): int(sum(R(u, "path", s, k) < R(u, "default", s, k) for s in SEEDS)) for u in units}
        comp[k] = dict(paired_path_wins=int(wins), paired_default_wins=int(losses), ties=int(ties), n_pairs=len(units) * len(SEEDS),
                       sign_test_p=sign_test_p(wins, losses), units_path_majority=int(sum(w >= 3 for w in per_unit.values())),
                       units_path_majority_picked=int(sum(per_unit[str(u)] >= 3 for u in units[:N_PICKED])),
                       units_path_majority_random=int(sum(per_unit[str(u)] >= 3 for u in units[N_PICKED:])),
                       path_wins_per_unit=per_unit)
    res["summary"], res["comparison"] = summ, comp
    res["time_s"] = time.time() - t_all
    json.dump(res, open(JSON, "w"), indent=1)
    for v in VARIANTS:
        print(f"[{v}] " + "; ".join(f"{k} picked {summ[v][f'gmean_{k}_picked']['mean']:.1f}+-{summ[v][f'gmean_{k}_picked']['sd']:.1f} "
                                    f"random {summ[v][f'gmean_{k}_random']['mean']:.1f}+-{summ[v][f'gmean_{k}_random']['sd']:.1f}" for k in JUDGES), flush=True)
    for k in JUDGES:
        c = comp[k]
        print(f"[{k}] paired path<default {c['paired_path_wins']}/{c['n_pairs']} (default better {c['paired_default_wins']}, ties {c['ties']}, "
              f"p={c['sign_test_p']:.3g}); units with path better in >=3/5 seeds: {c['units_path_majority']}/12", flush=True)
    grid(res, short, GRID)
    hk.detach()
    print(f"done in {time.time() - t_all:.0f} s", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-render every seed, including the reusable ones")
    main(ap.parse_args().force)
