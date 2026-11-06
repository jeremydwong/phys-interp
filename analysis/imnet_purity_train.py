"""imnet_ top-20 class / superclass dominance rerun on 50,000 ImageNet TRAINING images (secondary check).

The essay's neuron scans may use "50,000 training images" rather than val. Same proxy as imnet_purity.py (OURS):
the 20 strongest images per pole (pole = spatial max of the pre-activation, or minus the spatial min); a pole is
class-dominated if >= 10 of the 20 share one class, superclass-dominated if >= 10 share one WordNet group (K = 60 /
K = 130). Images: the first 50,000 of our 200,455 training images (the HF shards are shuffled, so this is a random
subset), and the next 50,000 as a second, disjoint sample. Pole scores come from the imnet_train extraction pass.
Writes the "top20_train" entry of reports/results/imnet_sae2.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import imnet_common as C  # noqa: E402
from analysis import imnet_purity as P  # noqa: E402
from analysis import imnet_train as T  # noqa: E402

LAYERS = (("googlenet", "inception5b"), ("convnext_tiny", "features.7.2.mlp"), ("convnext_tiny", "features.7"),
          ("convnext_small", "features.7.2.mlp"))


def main():
    fn = os.path.join(C.RESULTS, "imnet_sae2.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    val = json.load(open(os.path.join(C.RESULTS, "imnet_purity.json")))["imagenet_val"]
    y_all = T.labels()
    sup60, _ = C.superclasses(60)
    sup130, _ = C.superclasses(130)
    res = {}
    for arch, layer in LAYERS:
        pf = T.act_path("imagenet_train", arch, True, layer).replace(".npy", "_poles.npz")
        if not os.path.exists(pf):
            continue
        z = np.load(pf)
        r = dict(top1_train=float((z["pred"] == y_all[:len(z["pred"])]).mean()))
        for nm, sl in (("train_0_50k", slice(0, 50_000)), ("train_50k_100k", slice(50_000, 100_000))):
            y = y_all[sl]
            r[nm] = {p: {g: dict(dominated=v["dominated"], mean_modal_share=v["mean_modal_share"])
                         for g, v in P.pole_purity(z[p][sl].astype(np.float32), y, sup60, sup130).items() if g != "top"}
                     for p in ("pos", "neg")}
            r[nm]["chance_class"] = P.chance_dominated(y)
        v = val.get(C.tag(arch, True), {}).get(layer)
        if v:
            r["val_50k"] = {p: {g: dict(dominated=v[p][g]["dominated"]) for g in ("class", "super60", "super130")}
                            for p in ("pos", "neg")}
        res[f"{C.tag(arch, True)}:{layer}"] = r
        a = r["train_0_50k"]["pos"]
        print(f"[purity-train] {arch} {layer}: + class {a['class']['dominated']:.3f} super60 "
              f"{a['super60']['dominated']:.3f} (val: {r.get('val_50k', {}).get('pos', {}).get('class')})", flush=True)
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    out["top20_train"] = res
    json.dump(out, open(fn, "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
