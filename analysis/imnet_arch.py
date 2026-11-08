"""imnet_ network anatomy: each studied network as a left-to-right row of stages, for a reader new to CNNs.

Why. Report 2 makes claims like "InceptionV1 is a plain feed-forward stack of Inception modules; ConvNeXt is a
residual stack of depthwise-conv blocks with LayerNorm", and compares model sizes. This script grounds those
claims by instantiating every PyTorch module we use, walking it with forward hooks at a 1x3x224x224 input, and
recording per stage: output shape, parameter count, block style and whether it is the layer we study.

Networks (CPU only):
  (a) inceptionv1_lucid  Lucid's 2014 InceptionV1 (inception5h weights, torch-lucent port).  LRN, no BatchNorm,
                         no aux heads (verified by walking the modules and the forward source).
  (b) googlenet          torchvision GoogLeNet IMAGENET1K_V1: the BatchNorm re-implementation, aux heads present.
  (c) convnext_tiny      torchvision ConvNeXt-T IMAGENET1K_V1: depths 3/3/9/3, widths 96/192/384/768.
  (d) resnet50           torchvision ResNet-50, architecture only; the V1 / V2 weights share it and differ by recipe.
  (e) the project's own small nets of the vis_ study (models/vis_models.py): PlainCNN (M_old / M_recipe) and
      ConvNeXtS in sizes s / m / l, instantiated without weights at the shapes-set input (64 px).
  (f) the essay's three ConvNeXts (92k, 7.06M, 28.6M params) as text-only reference rows.

Outputs: reports/results/imnet_arch.json, reports/results/imnet_arch.md, reports/figs/imnet_arch.png.
Run: uv run python analysis/imnet_arch.py
"""
from __future__ import annotations

import inspect
import json
import math
import os
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import common  # noqa: E402

RESULTS = os.path.join(ROOT, "reports", "results")
FIGS = os.path.join(ROOT, "reports", "figs")
DEV = torch.device("cpu")
torch.set_num_threads(4)

# style key -> (legend label, colour). Coloured = carries weights; greys = parameter-free plumbing.
STYLES = {
    "conv": ("conv + ReLU (plain)", "#2a78d6"),
    "bnconv": ("conv + BatchNorm + ReLU", "#6da7ec"),
    "inception": ("Inception module (1x1 | 3x3 | 5x5 | pool branches, concatenated)", "#eb6834"),
    "bottleneck": ("bottleneck residual block (1x1-3x3-1x1 + skip), BatchNorm", "#1baf7a"),
    "convnext": ("ConvNeXt block (7x7 depthwise -> LayerNorm -> 1x1 MLP x4, GELU -> layer scale -> skip)", "#6250d6"),
    "head": ("linear head (classes)", "#eda100"),
    "aux": ("auxiliary head (training only)", "#e87ba4"),
    "stem": ("patchify stem (4x4/4 conv + LayerNorm)", "#9ec5f4"),
    "downsample": ("downsample (LayerNorm + 2x2/2 conv)", "#a9a79e"),
    "lrn": ("local response norm (LRN)", "#c4c2b9"),
    "pool": ("max pool", "#d7d5cc"),
    "gap": ("global average pool", "#e6e4dc"),
}
DEPTH = {"conv": 1, "bnconv": 1, "inception": 2, "bottleneck": 3, "convnext": 3, "head": 1, "stem": 1,
         "downsample": 1, "aux": 0, "lrn": 0, "pool": 0, "gap": 0}
HF_URL = "https://huggingface.co/facebook/convnext-tiny-224/raw/main/config.json"
# values read from HF_URL on 2026-11-08 (used if the fetch inside the script fails)
HF_FALLBACK = {"depths": [3, 3, 9, 3], "hidden_sizes": [96, 192, 384, 768], "patch_size": 4, "num_stages": 4,
               "layer_scale_init_value": 1e-6, "hidden_act": "gelu", "image_size": 224, "model_type": "convnext"}


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def n_weight_layers(m):
    return sum(isinstance(x, (nn.Conv2d, nn.Linear)) for x in m.modules())


def norm_kinds(m):
    kinds = set()
    for x in m.modules():
        if isinstance(x, nn.modules.batchnorm._BatchNorm):
            kinds.add("BatchNorm")
        if isinstance(x, nn.LayerNorm):
            kinds.add("LayerNorm")
        if isinstance(x, nn.LocalResponseNorm):
            kinds.add("LRN")
    return kinds


def stage(name, style, params, out, blocks=None, studied=False, note=""):
    out = [int(v) for v in out]
    return dict(name=name, style=style, params=int(params), out=out, spatial=(out[1] if len(out) == 3 else 1),
                blocks=blocks, depth=DEPTH[style] * (blocks or 1), studied=studied, note=note)


def hook_shapes(model, mods: dict, x):
    """mods: tag -> module. Returns tag -> output shape (without batch) after one forward pass."""
    shapes, handles = {}, []
    for tag, mod in mods.items():
        handles.append(mod.register_forward_hook(lambda m, i, o, tag=tag: shapes.__setitem__(tag, tuple(o.shape[1:]))))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return shapes


def meta_of(W):
    m = W.meta
    acc = m["_metrics"]["ImageNet-1K"]
    return dict(enum=str(W), url=W.url, num_params=m["num_params"], acc1=acc["acc@1"], acc5=acc["acc@5"],
                recipe=m["recipe"], min_size=list(m.get("min_size", ())), docs=" ".join(m.get("_docs", "").split()))


def finish(net, stages):
    net["stages"] = stages
    net["n_stages"] = len(stages)
    net["depth"] = sum(s["depth"] for s in stages if s["style"] != "aux")
    net["params_in_stages"] = sum(s["params"] for s in stages)
    assert net["params_in_stages"] == net["n_params"], (net["key"], net["params_in_stages"], net["n_params"])
    return net


# ------------------------------------------------------------------ (a) Lucid InceptionV1

def anatomy_lucid():
    import importlib
    from lucent.modelzoo import inceptionv1
    mod = importlib.import_module("lucent.modelzoo.inceptionv1.InceptionV1")   # the submodule holding model_urls
    m = inceptionv1(pretrained=True).to(DEV).eval()
    x = torch.zeros(1, 3, 224, 224)
    order = ["conv2d0", "maxpool0", "conv2d1", "conv2d2", "maxpool1", "mixed3a", "mixed3b", "maxpool4",
             "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "maxpool10", "mixed5a", "mixed5b",
             "softmax2_pre_activation_matmul"]
    shapes = hook_shapes(m, {k: getattr(m, k) for k in order}, x)
    src = inspect.getsource(type(m).forward)
    n_lrn = src.count("F.local_response_norm(")
    pnames = [n for n, _ in m.named_parameters()]
    aux = sorted({n.split("_")[0] for n in pnames if n.startswith(("softmax0", "softmax1", "head"))})
    kinds = norm_kinds(m)
    assert n_lrn == 2 and not aux and "BatchNorm" not in kinds, (n_lrn, aux, kinds)

    def P(prefix):
        return sum(p.numel() for n, p in m.named_parameters() if n.startswith((prefix + "_", prefix + ".")))

    S = []
    S.append(stage("conv2d0", "conv", P("conv2d0"), shapes["conv2d0"], note="7x7/2 conv, 64 ch"))
    S.append(stage("maxpool0", "pool", 0, shapes["maxpool0"], note="3x3/2"))
    S.append(stage("lrn0", "lrn", 0, shapes["maxpool0"], note="F.local_response_norm(size=9, alpha=1e-4, beta=0.5)"))
    S.append(stage("conv2d1", "conv", P("conv2d1"), shapes["conv2d1"], note="1x1 conv, 64 ch"))
    S.append(stage("conv2d2", "conv", P("conv2d2"), shapes["conv2d2"], note="3x3 conv, 192 ch"))
    S.append(stage("lrn1", "lrn", 0, shapes["conv2d2"], note="F.local_response_norm(size=9, alpha=1e-4, beta=0.5)"))
    S.append(stage("maxpool1", "pool", 0, shapes["maxpool1"], note="3x3/2"))
    for k in ["mixed3a", "mixed3b", "maxpool4", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "maxpool10",
              "mixed5a", "mixed5b"]:
        if k.startswith("maxpool"):
            S.append(stage(k, "pool", 0, shapes[k], note="3x3/2"))
        else:
            branches = {b: getattr(m, f"{k}_{b}_pre_relu_conv").out_channels
                        for b in ("1x1", "3x3", "5x5", "pool_reduce")}
            S.append(stage(k, "inception", P(k), shapes[k], studied=(k == "mixed5b"),
                           note="branch widths 1x1 %d | 3x3 %d | 5x5 %d | pool_reduce %d, concatenated" % tuple(branches.values())))
    S.append(stage("avgpool0", "gap", 0, (shapes["mixed5b"][0], 1, 1), note="7x7 average pool (functional)"))
    S.append(stage("softmax2", "head", P("softmax2_pre_activation_matmul"), shapes["softmax2_pre_activation_matmul"],
                   note="Linear 1024 -> 1008 (2012 devkit class order; 8 unused outputs)"))
    net = dict(key="inceptionv1_lucid", name="InceptionV1 (Lucid, 2014 weights)", year=2014, family="Inception",
               source="torch-lucent 0.1.8 lucent.modelzoo.inceptionv1(pretrained=True); port of the TF inception5h graph",
               input="224x224x3 RGB, RGB*255 - 117 (mean-subtracted, no std scaling)", n_outputs=1008,
               n_params=n_params(m), n_weight_layers=n_weight_layers(m), norm="LRN (2 layers, functional); no BatchNorm",
               residual=False,
               weights=dict(url=mod.model_urls["inceptionv1"], recipe="2014 Google training (Szegedy et al. 2015), TF inception5h release 2015",
                            acc1=69.13, acc1_note="OURS: top-1 on the 50k ILSVRC2012 val, 224 centre crops, 1008->1000 class map recovered "
                            "as the modal true label per output (reports/results/imnet_lucid.json class_map/acc_rgb)"),
               studied="mixed5b (1024 ch, 7x7), the four branch convs pre-ReLU",
               verified=dict(lrn_calls_in_forward=n_lrn, aux_heads=aux, norm_modules=sorted(kinds),
                             relu="RedirectedReLU helper (gradient trick for feature vis); no BN"),
               notes=["Feed-forward: every stage consumes only the previous stage's output; no skip connections.",
                      "The two auxiliary softmax heads of the 2014 paper are absent from the inception5h graph and the lucent port.",
                      "Conv biases present (no BN); LRN after the first pool and after conv2d2 as in the 2014 paper."])
    return finish(net, S)


# ------------------------------------------------------------------ (b) torchvision GoogLeNet

def anatomy_googlenet():
    from torchvision.models import GoogLeNet_Weights, googlenet
    W = GoogLeNet_Weights.IMAGENET1K_V1
    # torchvision drops aux1/aux2 (sets them to None) after loading unless aux_logits=True is passed explicitly
    m = googlenet(weights=W, aux_logits=True).to(DEV).eval()
    x = torch.zeros(1, 3, 224, 224)
    order = ["conv1", "maxpool1", "conv2", "conv3", "maxpool2", "inception3a", "inception3b", "maxpool3",
             "inception4a", "inception4b", "inception4c", "inception4d", "inception4e", "maxpool4",
             "inception5a", "inception5b", "avgpool", "fc"]
    shapes = hook_shapes(m, {k: getattr(m, k) for k in order}, x)
    S = []
    for k in order:
        mod = getattr(m, k)
        if k.startswith("conv"):
            c = mod.conv
            S.append(stage(k, "bnconv", n_params(mod), shapes[k], note=f"{c.kernel_size[0]}x{c.kernel_size[1]}/{c.stride[0]} conv (no bias) + BN + ReLU"))
        elif k.startswith("maxpool"):
            S.append(stage(k, "pool", 0, shapes[k], note="3x3/2, ceil_mode"))
        elif k.startswith("inception"):
            br = (mod.branch1.conv.out_channels, mod.branch2[1].conv.out_channels, mod.branch3[1].conv.out_channels,
                  mod.branch4[1].conv.out_channels)
            S.append(stage(k, "inception", n_params(mod), shapes[k], studied=(k == "inception5b"),
                           note="branch widths 1x1 %d | 3x3 %d | 3x3(5x5 slot) %d | pool 1x1 %d, each conv + BN + ReLU" % br))
            if k == "inception4a":
                with torch.no_grad():
                    cap = {}
                    h = m.inception4a.register_forward_hook(lambda mm, i, o: cap.__setitem__("o", o))
                    m(x)
                    h.remove()
                    a = m.aux1(cap["o"])
                S.append(stage("aux1", "aux", n_params(m.aux1), tuple(a.shape[1:]), note="avgpool 5x5/3 -> 1x1 conv 128 -> FC 1024 -> FC 1000; off in eval()"))
            if k == "inception4d":
                with torch.no_grad():
                    cap = {}
                    h = m.inception4d.register_forward_hook(lambda mm, i, o: cap.__setitem__("o", o))
                    m(x)
                    h.remove()
                    a = m.aux2(cap["o"])
                S.append(stage("aux2", "aux", n_params(m.aux2), tuple(a.shape[1:]), note="same as aux1; off in eval()"))
        elif k == "avgpool":
            S.append(stage(k, "gap", 0, shapes[k], note="AdaptiveAvgPool2d(1) + dropout 0.2"))
        elif k == "fc":
            S.append(stage(k, "head", n_params(mod), shapes[k], note="Linear 1024 -> 1000"))
    kinds = norm_kinds(m)
    assert "BatchNorm" in kinds and "LRN" not in kinds
    net = dict(key="googlenet", name="GoogLeNet (torchvision, BatchNorm)", year="2014 arch; BN re-implementation", family="Inception",
               source="torchvision 0.29 googlenet(weights=IMAGENET1K_V1), aux_logits=True, transform_input=True",
               input="224x224x3 RGB, ImageNet mean/std normalised (then re-scaled inside by transform_input)", n_outputs=1000,
               n_params=n_params(m), n_params_excl_aux=n_params(m) - n_params(m.aux1) - n_params(m.aux2),
               n_weight_layers=n_weight_layers(m), norm="BatchNorm after every conv; no LRN", residual=False,
               weights=meta_of(W), studied="inception5b (1024 ch, 7x7), the branch convs pre-ReLU",
               verified=dict(norm_modules=sorted(kinds), aux_heads=["aux1", "aux2"]),
               notes=["Same topology as (a) (3 stem convs, 9 Inception modules, 4 max-pools, GAP, 1 FC), but every conv is "
                      "bias-free conv + BatchNorm + ReLU, the LRN layers are gone, the 5x5 branch is a 3x3 conv (TF-slim "
                      "convention) and the two auxiliary heads are present as modules (used only in training).",
                      "Weights: torchvision's meta says 'ported from the original paper'; they are a BatchNorm-era re-training, "
                      "not the 2014 release. Top-1 69.78 vs our 69.13 for the Lucid weights.",
                      "torchvision's num_params (6,624,904) excludes the aux heads; with aux_logits=True the module has 13,004,888, "
                      "and torchvision warns that the aux heads are NOT pretrained (random init)."])
    return finish(net, S)


# ------------------------------------------------------------------ (c) torchvision ConvNeXt-T

def anatomy_convnext():
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    W = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    m = convnext_tiny(weights=W).to(DEV).eval()
    x = torch.zeros(1, 3, 224, 224)
    f = m.features
    mods = {f"f{i}": f[i] for i in range(len(f))}
    mods["avgpool"], mods["classifier"] = m.avgpool, m.classifier
    shapes = hook_shapes(m, mods, x)
    S = [stage("stem", "stem", n_params(f[0]), shapes["f0"], note="4x4/4 conv 3 -> 96 + LayerNorm (patchify)")]
    depths, widths = [], []
    for si, i in enumerate([1, 3, 5, 7]):
        if i > 1:
            S.append(stage(f"down{si}", "downsample", n_params(f[i - 1]), shapes[f"f{i - 1}"], note="LayerNorm + 2x2/2 conv"))
        blocks = list(f[i])
        dim = blocks[0].block[0].out_channels
        hidden = blocks[0].block[3].out_features
        depths.append(len(blocks))
        widths.append(dim)
        S.append(stage(f"stage{si + 1}", "convnext", n_params(f[i]), shapes[f"f{i}"], blocks=len(blocks),
                       studied=(si == 2),
                       note=f"{len(blocks)} blocks, {dim} ch, MLP hidden {hidden} (x4), layer scale init 1e-6, stochastic depth in training"))
    S.append(stage("avgpool", "gap", 0, shapes["avgpool"], note="AdaptiveAvgPool2d(1)"))
    S.append(stage("classifier", "head", n_params(m.classifier), shapes["classifier"], note="LayerNorm + Linear 768 -> 1000"))
    kinds = norm_kinds(m)
    assert kinds == {"LayerNorm"}
    net = dict(key="convnext_tiny", name="ConvNeXt-T (torchvision)", year=2022, family="ConvNeXt",
               source="torchvision 0.29 convnext_tiny(weights=IMAGENET1K_V1)",
               input="224x224x3 RGB, ImageNet mean/std normalised", n_outputs=1000,
               n_params=n_params(m), n_weight_layers=n_weight_layers(m), norm="LayerNorm (one per block, in stem, downsamples and head); no BatchNorm",
               residual=True, weights=meta_of(W), depths=depths, widths=widths,
               studied="stage 3 (features[5]), last block (9 of 9), the 1536 MLP hidden units after GELU",
               verified=dict(norm_modules=sorted(kinds), depths=depths, widths=widths),
               notes=["Residual stream: every block adds its output to its input (x + gamma * MLP(LN(dwconv(x)))), so the "
                      "same 96/192/384/768-channel tensor runs through each stage and is only rewritten at the downsamples.",
                      "Blocks use GELU, not ReLU, and one LayerNorm per block instead of BatchNorm after every conv."])
    return finish(net, S)


# ------------------------------------------------------------------ (d) torchvision ResNet-50 (architecture only)

def anatomy_resnet():
    from torchvision.models import ResNet50_Weights, resnet50
    m = resnet50(weights=None).to(DEV).eval()
    x = torch.zeros(1, 3, 224, 224)
    order = ["conv1", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    shapes = hook_shapes(m, {k: getattr(m, k) for k in order}, x)
    S = [stage("stem", "bnconv", n_params(m.conv1) + n_params(m.bn1), shapes["conv1"], note="7x7/2 conv 3 -> 64 + BN + ReLU"),
         stage("maxpool", "pool", 0, shapes["maxpool"], note="3x3/2")]
    for k in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(m, k)
        S.append(stage(k, "bottleneck", n_params(layer), shapes[k], blocks=len(layer), studied=(k == "layer4"),
                       note=f"{len(layer)} bottleneck blocks, {layer[0].conv1.out_channels} -> {layer[0].conv3.out_channels} ch; first block has a 1x1 projection shortcut"))
    S.append(stage("avgpool", "gap", 0, shapes["avgpool"], note="AdaptiveAvgPool2d(1)"))
    S.append(stage("fc", "head", n_params(m.fc), shapes["fc"], note="Linear 2048 -> 1000"))
    kinds = norm_kinds(m)
    assert kinds == {"BatchNorm"}
    net = dict(key="resnet50", name="ResNet-50 (torchvision, architecture only)", year=2015, family="ResNet",
               source="torchvision 0.29 resnet50(weights=None); weights V1 / V2 share this module",
               input="224x224x3 RGB, ImageNet mean/std normalised", n_outputs=1000,
               n_params=n_params(m), n_weight_layers=n_weight_layers(m), norm="BatchNorm after every conv", residual=True,
               weights={"IMAGENET1K_V1": meta_of(ResNet50_Weights.IMAGENET1K_V1), "IMAGENET1K_V2": meta_of(ResNet50_Weights.IMAGENET1K_V2)},
               studied="layer4 (2048 ch, 7x7) if used",
               verified=dict(norm_modules=sorted(kinds), blocks=[len(getattr(m, k)) for k in ("layer1", "layer2", "layer3", "layer4")]),
               notes=["V1 and V2 weights have the same num_params (the same module); V2 adds ~4.7 top-1 points purely by "
                      "training recipe (long schedule, heavy augmentation, EMA, label smoothing; see the recipe URLs).",
                      "'50' = 49 convs on the main path (1 stem + 16 blocks x 3) + 1 FC; the 4 projection shortcuts are extra."])
    return finish(net, S)


# ------------------------------------------------------------------ (e) the project's small nets

def anatomy_project():
    from models.vis_models import ARCH_SIZES, ConvNeXtS, PlainCNN
    nets = []
    res, n_cls, stem_stride = 64, 24, 2   # the shapes set (CIFAR: 32 px, 10 classes, stem stride 1)
    x = torch.rand(1, 3, res, res)

    m = PlainCNN(n_cls, stem_stride=stem_stride).eval()
    with torch.no_grad():
        m(x, record=m.neuron_layers)
    S = []
    widths = [c.out_channels for c in m.convs[1::2]]
    for s, w in enumerate(widths):
        names = [f"s{s + 1}c{k + 1}" for k in range(2)]
        p = sum(n_params(m.convs[2 * s + k]) + n_params(m.bns[2 * s + k]) for k in range(2))
        S.append(stage(f"stage{s + 1}", "bnconv", p, m.acts[names[-1]].shape[1:], blocks=2, studied=(s == 3),
                       note=f"2 x (3x3 conv + BN + ReLU), {w} ch" + (" (first conv stride 2)" if s == 0 else "")))
        if s < 3:
            S.append(stage(f"pool{s + 1}", "pool", 0, (w, m.acts[names[-1]].shape[2] // 2, m.acts[names[-1]].shape[3] // 2), note="2x2/2"))
    S.append(stage("gap", "gap", 0, (widths[-1], 1, 1), note="spatial mean"))
    S.append(stage("head", "head", n_params(m.head), (n_cls,), note=f"Linear {widths[-1]} -> {n_cls}"))
    nets.append(finish(dict(key="vis_plaincnn", name="PlainCNN (ours, M_old / M_recipe)", year=2026, family="plain CNN",
                            source="models/vis_models.py PlainCNN(24, stem_stride=2); no weights loaded",
                            input=f"{res}x{res}x3 shapes images in [0,1], normalised inside", n_outputs=n_cls,
                            n_params=n_params(m), n_weight_layers=n_weight_layers(m), norm="BatchNorm after every conv",
                            residual=False, weights=dict(url=None, recipe="models/vis_train.py --step old|recipe",
                                                         acc1=None, acc1_note="shapes test acc 99.99% (old) / 99.96% (recipe); CIFAR-10 91.2 / 89.5 (reports/results/vis.md)"),
                            studied="s4c2 (256 units at 4x4), the last conv's BN output pre-ReLU",
                            notes=["VGG-style: 8 convs in 4 stages of 2, max-pool between stages, no skip connections.",
                                   "The CIFAR variant differs only in stem stride (1), input (32 px) and head (10 classes)."]), S))

    for size in ("s", "m", "l"):
        cfg = ARCH_SIZES[size]
        m = ConvNeXtS(n_cls, stem_stride=stem_stride, **cfg).eval()
        with torch.no_grad():
            m(x, record=m.neuron_layers + ["resid"])
        S = []
        H0 = m.acts["s1b1"].shape[2]
        S.append(stage("stem", "stem", n_params(m.stem), (cfg["dims"][0], H0, H0), note="4x4/2 conv + LayerNorm"))
        for s, (d, n) in enumerate(zip(cfg["dims"], cfg["depths"])):
            if s > 0:
                S.append(stage(f"down{s}", "downsample", n_params(m.down[s - 1]), (d,) + tuple(m.acts[f"s{s + 1}b1"].shape[2:]), note="LayerNorm + 2x2/2 conv"))
            S.append(stage(f"stage{s + 1}", "convnext", n_params(m.stages[s]), (d,) + tuple(m.acts[f"s{s + 1}b1"].shape[2:]), blocks=n,
                           studied=(s == 3), note=f"{n} block(s), {d} ch, MLP hidden {4 * d}"))
        S.append(stage("gap", "gap", 0, (cfg["dims"][-1], 1, 1), note="spatial mean + LayerNorm"))
        S.append(stage("head", "head", n_params(m.head) + n_params(m.norm), (n_cls,), note=f"Linear {cfg['dims'][-1]} -> {n_cls}"))
        label = {"s": "0.2M", "m": "1.2M, M_arch", "l": "4.6M"}[size]
        nets.append(finish(dict(key=f"vis_convnexts_{size}", name=f"ConvNeXtS-{size} (ours, {label})", year=2026, family="ConvNeXt",
                                source=f"models/vis_models.py ConvNeXtS(24, stem_stride=2, **ARCH_SIZES['{size}']); no weights loaded",
                                input=f"{res}x{res}x3 shapes images in [0,1], normalised inside", n_outputs=n_cls,
                                n_params=n_params(m), n_weight_layers=n_weight_layers(m), norm="LayerNorm (per block, stem, downsamples, head)",
                                residual=True, depths=list(cfg["depths"]), widths=list(cfg["dims"]),
                                weights=dict(url=None, recipe="models/vis_train.py --step arch --size " + size, acc1=None,
                                             acc1_note="M_arch: shapes test acc 100%, CIFAR-10 86.9% (reports/results/vis.md)" if size == "m" else "size sweep, see vis.md"),
                                studied="s4b1 MLP hidden units (4 x last width) at 4x4",
                                notes=["Miniature of ConvNeXt-T: same block, 4 stages, patchify stem, LN downsamples; depths "
                                       f"{list(cfg['depths'])}, widths {list(cfg['dims'])}."]), S))
    return nets


TEXT_ROWS = [
    dict(key="essay_92k", name="ConvNeXt 92k (essay)", n_params=92_000, blocks=None,
         note="smallest of the essay's three ConvNeXts; parameters from the essay text, no code or weights"),
    dict(key="essay_7m", name="ConvNeXt 7.06M (essay)", n_params=7_060_000, blocks=None,
         note="middle size of the essay's sweep; text only"),
    dict(key="essay_28m", name="ConvNeXt 28.6M (essay)", n_params=28_600_000, blocks=18,
         note="largest; 18 blocks (3/3/9/3 like ConvNeXt-T, whose torchvision count is 28,589,128); text only"),
]


def hf_check(convnext_net):
    out = dict(url=HF_URL, fetched=False, config=None, matched=None)
    try:
        with urllib.request.urlopen(HF_URL, timeout=15) as r:
            cfg = json.loads(r.read().decode())
        out["fetched"] = True
    except Exception as e:  # noqa: BLE001
        cfg = dict(HF_FALLBACK)
        out["fetch_error"] = repr(e)
        out["fallback"] = "values recorded from a WebFetch of the same URL on 2026-11-08"
    keep = ["depths", "hidden_sizes", "patch_size", "num_stages", "layer_scale_init_value", "hidden_act", "image_size", "model_type"]
    out["config"] = {k: cfg.get(k) for k in keep}
    out["matched"] = (list(cfg.get("depths", [])) == convnext_net["depths"] and list(cfg.get("hidden_sizes", [])) == convnext_net["widths"])
    out["torchvision"] = dict(depths=convnext_net["depths"], widths=convnext_net["widths"])
    return out


# ------------------------------------------------------------------ figure

def fmt_params(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def box_w(p):
    return max(0.2, 0.104 * math.log10(p + 1))


def box_h(spatial):
    return max(0.13, 0.085 + 0.062 * math.log2(max(spatial, 1)))


def _fits(lines, w, fs=7.0):
    return max(len(l) for l in lines) * 0.0515 * fs / 7 <= w - 0.03


def draw(nets, text_rows, fn):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    common.setup_style()
    plt.rcParams["axes.grid"] = False
    W, H = 14, 10.5
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    short = {"inceptionv1_lucid": "InceptionV1\n(Lucid 2014)", "googlenet": "GoogLeNet\n(torchvision, BN)",
             "convnext_tiny": "ConvNeXt-T\n(torchvision)", "resnet50": "ResNet-50\n(torchvision)",
             "vis_plaincnn": "PlainCNN, ours\n(M_old / M_recipe)", "vis_convnexts_s": "ConvNeXtS-s, ours\n(0.2M)",
             "vis_convnexts_m": "ConvNeXtS-m, ours\n(M_arch)", "vis_convnexts_l": "ConvNeXtS-l, ours\n(4.6M)"}
    X0, XR = 1.7, 11.9
    GAP = 0.045
    ROW, TROW = 0.93, 0.3
    y = H - 0.9
    ax.text(0.25, H - 0.3, "Network anatomy: each model as a left-to-right row of stages", fontsize=12, fontweight="bold", va="center")
    ax.text(0.25, H - 0.52, "box height ~ log(spatial size of the output; grey numbers below), box width ~ log10(parameters); grey = no weights; "
            "thick outline = the layer we study;\na coloured line running through the blocks = residual stream (each block adds to its input); "
            "aux heads hang below the trunk they branch from",
            fontsize=8, color=common.INK2, va="top", linespacing=1.3)
    for net in nets:
        y -= ROW
        yb = y + 0.26
        ax.text(0.25, yb + 0.24, short.get(net["key"], net["name"]), fontsize=8.5, fontweight="bold", va="center")
        ax.text(0.25, yb + 0.03, f"{net['year'] if isinstance(net['year'], int) else '2014 / BN'}  ·  "
                + ("residual" if net["residual"] else "feed-forward"), fontsize=7.5, color=common.INK2, va="center")
        x = X0
        last_sp, res_x, parent = None, [], None
        for st in net["stages"]:
            w, h = box_w(st["params"]), box_h(st["spatial"])
            col = STYLES[st["style"]][1]
            studied = st["studied"]
            if st["style"] == "aux":   # side branch: hang below the module it reads from
                px, pw = parent
                w, h = 0.42, 0.13
                xa = px + pw / 2 - w / 2
                ax.plot([px + pw / 2, px + pw / 2], [yb, y + 0.03 + h], color=col, lw=1.0, zorder=1)
                ax.add_patch(Rectangle((xa, y + 0.03), w, h, facecolor=col, edgecolor="white", linewidth=0.5, zorder=2))
                ax.text(xa + w / 2, y + 0.03 + h / 2, "aux", fontsize=7, ha="center", va="center", zorder=3, color=common.INK)
                continue
            ax.add_patch(Rectangle((x, yb), w, h, facecolor=col, edgecolor=common.INK if studied else "white",
                                   linewidth=1.8 if studied else 0.5, zorder=2))
            lab = st["name"].replace("mixed", "").replace("inception", "").replace("stage", "stage ").replace("layer", "layer ")
            lab = lab.replace("softmax2", "fc").replace("classifier", "fc").replace("conv2d", "conv")
            lab_short = lab.replace("stage ", "S").replace("layer ", "L")
            blocks = f" ×{st['blocks']}" if st["blocks"] and st["style"] in ("convnext", "bottleneck", "bnconv") else ""
            ch = [f"{st['out'][0]} ch"] if len(st["out"]) == 3 and st["style"] not in ("pool", "lrn", "gap", "downsample") else []
            candidates = [[lab + blocks] + ch, [lab_short + blocks] + ch, [lab_short + blocks], [lab_short]]
            if st["style"] == "head":
                candidates = [[f"fc→{st['out'][0]}"], ["fc"]]
            n_fit = max(1, int(h / 0.125))
            lines = next((c[:n_fit] for c in candidates if _fits(c[:n_fit], w)), None)
            dark = st["style"] in ("conv", "inception", "bottleneck", "convnext", "head")
            if lines is not None and w >= 0.3:
                ax.text(x + w / 2, yb + h / 2, "\n".join(lines), fontsize=7, ha="center", va="center", zorder=3,
                        color="white" if dark else common.INK, linespacing=1.1)
            elif h >= 0.33:
                ax.text(x + w / 2, yb + h / 2, {"pool": "pool", "lrn": "LRN", "gap": "avg", "downsample": "down"}.get(st["style"], lab_short),
                        fontsize=7, ha="center", va="center", rotation=90, zorder=3, color="white" if dark else common.INK)
            if studied:
                ax.text(x + w / 2, yb + h + 0.05, "studied", fontsize=7, ha="center", va="bottom", color=common.INK, fontweight="bold")
            if st["style"] in ("convnext", "bottleneck"):
                res_x.append((x, x + w))
            sp = st["spatial"]
            if sp != last_sp and len(st["out"]) == 3:
                ax.text(x, yb - 0.04, f"{sp}", fontsize=7, ha="left", va="top", color=common.MUTED)
                last_sp = sp
            parent = (x, w)
            x += w + GAP
        if res_x:
            xa, xb = res_x[0][0] - 0.03, res_x[-1][1] + 0.03
            rc = STYLES["convnext"][1] if net["family"] == "ConvNeXt" else STYLES["bottleneck"][1]
            yl = yb + 0.065
            ax.plot([xa, xb], [yl, yl], color=rc, lw=2.6, zorder=1, solid_capstyle="butt")
            ax.text(x + 0.08, yl, "← residual stream", fontsize=7, ha="left", va="center", color=rc)
        acc = net["weights"].get("acc1") if "acc1" in net["weights"] else None
        if net["key"] == "resnet50":
            acc_s = f"top-1 {net['weights']['IMAGENET1K_V1']['acc1']:.1f} (V1) / {net['weights']['IMAGENET1K_V2']['acc1']:.1f} (V2)"
        elif acc is None:
            acc_s = "no ImageNet accuracy (shapes / CIFAR nets)"
        else:
            acc_s = f"top-1 {acc:.1f}%" + (" (ours)" if net["key"] == "inceptionv1_lucid" else "")
        if net["key"] == "googlenet":
            ptxt = f"{net['n_params_excl_aux'] / 1e6:.2f}M params (+{(net['n_params'] - net['n_params_excl_aux']) / 1e6:.2f}M in 2 aux heads)"
        else:
            ptxt = f"{net['n_params'] / 1e6:.2f}M params"
        ax.text(XR, yb + 0.32, ptxt, fontsize=8.5, fontweight="bold", va="center")
        ax.text(XR, yb + 0.15, f"depth {net['depth']} weight layers on the main path", fontsize=7.5, color=common.INK2, va="center")
        ax.text(XR, yb - 0.01, acc_s, fontsize=7.5, color=common.INK2, va="center")
    y -= 0.1
    for tr in text_rows:
        y -= TROW
        yb = y + 0.06
        ax.text(0.25, yb + 0.08, tr["name"], fontsize=8.5, fontweight="bold", va="center")
        w = box_w(tr["n_params"])
        ax.add_patch(Rectangle((X0, yb), w, 0.17, facecolor="none", edgecolor=STYLES["convnext"][1], hatch="////", linewidth=0.8))
        ax.text(X0 + w + 0.1, yb + 0.085, ("18 blocks; " if tr["blocks"] else "") + "reference row from the essay's text only (no code or weights); box width = log10(params)",
                fontsize=7.5, color=common.INK2, va="center")
        ax.text(XR, yb + 0.085, f"{tr['n_params'] / 1e6:.2f}M params", fontsize=8.5, fontweight="bold", va="center")
    handles = [Patch(facecolor=c, edgecolor=common.AXIS, linewidth=0.5, label=l) for k, (l, c) in STYLES.items()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, 0.005), ncol=2, fontsize=7, frameon=False,
              handlelength=1.4, columnspacing=1.5, title="stage styles", title_fontsize=7.5, alignment="left")
    fig.savefig(fn, dpi=200, bbox_inches=None)
    plt.close(fig)


# ------------------------------------------------------------------ markdown

def write_md(nets, text_rows, hf, fn):
    L = ["# Network anatomy (imnet_arch)", "",
         "What each studied network is made of, stage by stage, read off the instantiated PyTorch modules "
         "(`analysis/imnet_arch.py`; data in `reports/results/imnet_arch.json`, figure `reports/figs/imnet_arch.png`). "
         "Depth = weight layers (convs / linears) on the main path, counting an Inception module as 2 (its deepest branch), "
         "a bottleneck block as 3 and a ConvNeXt block as 3; auxiliary heads and projection shortcuts are not counted.", "",
         "| Network | Year | Parameters | Depth | Top-1 (ImageNet val) | Normalization | Residual stream | Block style | Input | Weights downloaded from |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    styles = {
        "inceptionv1_lucid": "A plain feed-forward stack: 3 ordinary convs (7x7, 1x1, 3x3) with ReLU and two LRN layers, then 9 Inception modules, each running a 1x1, a 1x1->3x3, a 1x1->5x5 and a pool->1x1 branch side by side on the same input and gluing the four outputs together along the channel axis; 4 max-pools shrink the image 224 -> 7; global average pool and one linear layer to 1008 outputs.",
        "googlenet": "Same skeleton as the Lucid net (3 convs, 9 Inception modules, 4 pools, GAP, 1 FC) but each conv is followed by BatchNorm then ReLU, there is no LRN, the '5x5' branch is really a 3x3 conv, and two auxiliary classifier heads hang off modules 4a and 4d (used only during training).",
        "convnext_tiny": "A residual stack: a 4x4 patchify stem, then 4 stages of 3/3/9/3 ConvNeXt blocks (7x7 depthwise conv -> LayerNorm -> 1x1 expand x4 -> GELU -> 1x1 shrink -> per-channel scale, added back to the block's input), with LayerNorm + 2x2 conv downsamples between stages; GAP, LayerNorm, linear to 1000.",
        "resnet50": "A residual stack: 7x7 stem conv + BN + ReLU and a max-pool, then 4 stages of 3/4/6/3 bottleneck blocks (1x1 shrink -> 3x3 -> 1x1 expand, each with BN + ReLU, added to the block's input, the first block of a stage using a 1x1 projection on the skip); GAP and one linear to 1000.",
        "vis_plaincnn": "VGG-style: 4 stages of 2 x (3x3 conv + BN + ReLU) at 32/64/128/256 channels, max-pool between stages, GAP, linear to 24 shape classes; no skips.",
        "vis_convnexts_s": "Miniature ConvNeXt (dims 16/32/64/96, depths 1/1/2/1), same block as ConvNeXt-T.",
        "vis_convnexts_m": "Miniature ConvNeXt (dims 32/64/128/256, depths 1/1/3/1), parameter-matched to PlainCNN.",
        "vis_convnexts_l": "Miniature ConvNeXt (dims 64/128/256/512, depths 1/1/3/1).",
    }
    for n in nets:
        w = n["weights"]
        if n["key"] == "resnet50":
            acc = f"{w['IMAGENET1K_V1']['acc1']:.2f} (V1) / {w['IMAGENET1K_V2']['acc1']:.2f} (V2)"
            url = f"V1 {w['IMAGENET1K_V1']['url']}; V2 {w['IMAGENET1K_V2']['url']}"
        elif w.get("acc1") is None:
            acc = "n/a; " + w.get("acc1_note", "")
            url = "none (trained by us, `models/vis_train.py`)"
        else:
            acc = f"{w['acc1']:.2f}" + (" (OURS, see json)" if n["key"] == "inceptionv1_lucid" else "")
            url = w["url"]
        ptxt = (f"{n['n_params_excl_aux']:,} (+{n['n_params'] - n['n_params_excl_aux']:,} in 2 untrained aux heads)"
                if "n_params_excl_aux" in n else f"{n['n_params']:,}")
        L.append(f"| {n['name']} | {n['year']} | {ptxt} | {n['depth']} ({n['n_weight_layers']} conv/linear modules in all) | {acc} | "
                 f"{n['norm']} | {'yes' if n['residual'] else 'no'} | {styles[n['key']]} | {n['input']} | {url} |")
    for t in text_rows:
        L.append(f"| {t['name']} | 2026 | {t['n_params']:,} | {'18 blocks' if t['blocks'] else '?'} | n/a | LayerNorm (by construction) | yes | "
                 f"{t['note']} | ? | none (essay text only) |")
    L += ["", "GoogLeNet's parameter count includes its two auxiliary heads (%s without them). "
          % f"{[n for n in nets if n['key'] == 'googlenet'][0]['n_params_excl_aux']:,}"
          + "ResNet-50 V1 and V2 weights have the same num_params; V2 differs only by training recipe (V1 recipe: %s; V2 recipe: %s)."
          % ([n for n in nets if n["key"] == "resnet50"][0]["weights"]["IMAGENET1K_V1"]["recipe"],
             [n for n in nets if n["key"] == "resnet50"][0]["weights"]["IMAGENET1K_V2"]["recipe"]), ""]
    L += ["## Stage tables", ""]
    for n in nets:
        L += [f"### {n['name']}", "", f"Source: {n['source']}. Studied layer: {n['studied']}.", "",
              "| stage | style | params | output (C, H, W) | blocks | note |", "|---|---|---|---|---|---|"]
        for s in n["stages"]:
            L.append(f"| {s['name']}{' **(studied)**' if s['studied'] else ''} | {STYLES[s['style']][0].split(' (')[0]} | {s['params']:,} | "
                     f"{'x'.join(map(str, s['out']))} | {s['blocks'] or ''} | {s['note']} |")
        L += [""] + [f"- {t}" for t in n["notes"]] + [""]
    L += ["## Glossary (plain words)", "",
          "- **Inception module**: instead of choosing one filter size, the module runs several small convolutions "
          "(1x1, 3x3, 5x5) and a pooling branch side by side on the same input and stacks their outputs along the channel "
          "axis, so the next layer sees all of them at once.",
          "- **Residual connection**: the block computes a correction and adds it to its own input (output = input + f(input)), "
          "so information can flow straight through a deep network and each block only has to learn a small change. The "
          "tensor that is passed along and added to is called the residual stream.",
          "- **BatchNorm vs LayerNorm**: both rescale activations to a steady range before the next layer. BatchNorm "
          "normalises each channel using statistics over the whole mini-batch (and keeps running averages for test time); "
          "LayerNorm normalises each position using the statistics across its own channels, so it needs no batch and behaves "
          "the same in training and at test time. LRN (local response normalisation, 2012-2014) divides each activation by "
          "the energy of its neighbouring channels; it has no learned parameters.",
          "- **Depthwise convolution**: a convolution in which each channel is filtered by its own small kernel (here 7x7) "
          "without mixing channels; channel mixing is done separately by 1x1 layers. It is much cheaper than a full conv.",
          "- **Inverted bottleneck**: the block widens the representation first (1x1 to 4x the channels, then a nonlinearity) "
          "and narrows it back, the opposite of ResNet's bottleneck, which narrows first. In ConvNeXt the wide 4x hidden "
          "layer is where the units we study live.", "",
          "## Where this information comes from", "",
          "We instantiate each PyTorch module on the CPU and walk it: forward hooks on the top-level stages record the output "
          "shape at a 1x3x224x224 input (64x64 for our own nets), parameter counts are summed over each stage's modules, and "
          "normalisation / auxiliary heads / LRN are verified by listing module types and, for the lucent port, reading the "
          "forward() source (LRN is a functional call there, not a module). torchvision defines the architectures in Python "
          "code and stores training-recipe metadata (num_params, acc@1, recipe URL, download URL) in each weights enum "
          "(`ResNet50_Weights.IMAGENET1K_V1.meta`, `.url`); lucent stores its download URL in `lucent/modelzoo/inceptionv1/"
          "InceptionV1.py` (`model_urls`). HuggingFace-hosted models instead describe the architecture in a `config.json`: "
          f"for facebook/convnext-tiny-224 we read {hf['url']} ({'fetched live' if hf['fetched'] else 'fetch failed in the script; values from a WebFetch on 2026-11-08'}): "
          f"depths {hf['config']['depths']}, hidden_sizes {hf['config']['hidden_sizes']}, patch_size {hf['config']['patch_size']}, "
          f"layer_scale_init_value {hf['config']['layer_scale_init_value']}, hidden_act {hf['config']['hidden_act']}. "
          f"Compared with torchvision's instantiated module (depths {hf['torchvision']['depths']}, widths {hf['torchvision']['widths']}): "
          f"**{'matched' if hf['matched'] else 'DID NOT MATCH'}**. The essay's three ConvNeXt sizes (92k, 7.06M, 28.6M; 18 blocks for the largest) "
          "are copied from its text; we have no code or weights for them. The 28.6M figure agrees with ConvNeXt-T's 28,589,128 "
          "to three digits, so it is very likely the same 3/3/9/3 architecture.", ""]
    with open(fn, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    nets = [anatomy_lucid(), anatomy_googlenet(), anatomy_convnext(), anatomy_resnet()] + anatomy_project()
    hf = hf_check(nets[2])
    for n in nets:
        print(f"{n['name']:48s} {n['n_params']:>12,}  depth {n['depth']:3d}  stages {n['n_stages']:2d}  {n['norm']}")
    print("HF config matched torchvision:", hf["matched"], "(fetched live)" if hf["fetched"] else "(fallback values)")
    out = dict(networks=nets, text_rows=TEXT_ROWS, hf_check=hf, styles={k: v[0] for k, v in STYLES.items()},
               depth_rule=DEPTH, torch=torch.__version__, torchvision=__import__("torchvision").__version__,
               essay="Cammarata (2026), Neural Networks Are Already Clean, Goodfire")
    with open(os.path.join(RESULTS, "imnet_arch.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    write_md(nets, TEXT_ROWS, hf, os.path.join(RESULTS, "imnet_arch.md"))
    draw(nets, TEXT_ROWS, os.path.join(FIGS, "imnet_arch.png"))
    print("wrote", os.path.join(RESULTS, "imnet_arch.json"), os.path.join(RESULTS, "imnet_arch.md"), os.path.join(FIGS, "imnet_arch.png"))


if __name__ == "__main__":
    main()
