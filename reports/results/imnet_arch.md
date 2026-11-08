# Network anatomy (imnet_arch)

What each studied network is made of, stage by stage, read off the instantiated PyTorch modules (`analysis/imnet_arch.py`; data in `reports/results/imnet_arch.json`, figure `reports/figs/imnet_arch.png`). Depth = weight layers (convs / linears) on the main path, counting an Inception module as 2 (its deepest branch), a bottleneck block as 3 and a ConvNeXt block as 3; auxiliary heads and projection shortcuts are not counted.

| Network | Year | Parameters | Depth | Top-1 (ImageNet val) | Normalization | Residual stream | Block style | Input | Weights downloaded from |
|---|---|---|---|---|---|---|---|---|---|
| InceptionV1 (Lucid, 2014 weights) | 2014 | 7,066,380 | 22 (58 conv/linear modules in all) | 69.13 (OURS, see json) | LRN (2 layers, functional); no BatchNorm | no | A plain feed-forward stack: 3 ordinary convs (7x7, 1x1, 3x3) with ReLU and two LRN layers, then 9 Inception modules, each running a 1x1, a 1x1->3x3, a 1x1->5x5 and a pool->1x1 branch side by side on the same input and gluing the four outputs together along the channel axis; 4 max-pools shrink the image 224 -> 7; global average pool and one linear layer to 1008 outputs. | 224x224x3 RGB, RGB*255 - 117 (mean-subtracted, no std scaling) | https://github.com/ProGamerGov/pytorch-old-tensorflow-models/raw/master/inception5h.pth |
| GoogLeNet (torchvision, BatchNorm) | 2014 arch; BN re-implementation | 6,624,904 (+6,379,984 in 2 untrained aux heads) | 22 (64 conv/linear modules in all) | 69.78 | BatchNorm after every conv; no LRN | no | Same skeleton as the Lucid net (3 convs, 9 Inception modules, 4 pools, GAP, 1 FC) but each conv is followed by BatchNorm then ReLU, there is no LRN, the '5x5' branch is really a 3x3 conv, and two auxiliary classifier heads hang off modules 4a and 4d (used only during training). | 224x224x3 RGB, ImageNet mean/std normalised (then re-scaled inside by transform_input) | https://download.pytorch.org/models/googlenet-1378be20.pth |
| ConvNeXt-T (torchvision) | 2022 | 28,589,128 | 59 (59 conv/linear modules in all) | 82.52 | LayerNorm (one per block, in stem, downsamples and head); no BatchNorm | yes | A residual stack: a 4x4 patchify stem, then 4 stages of 3/3/9/3 ConvNeXt blocks (7x7 depthwise conv -> LayerNorm -> 1x1 expand x4 -> GELU -> 1x1 shrink -> per-channel scale, added back to the block's input), with LayerNorm + 2x2 conv downsamples between stages; GAP, LayerNorm, linear to 1000. | 224x224x3 RGB, ImageNet mean/std normalised | https://download.pytorch.org/models/convnext_tiny-983f1562.pth |
| ResNet-50 (torchvision, architecture only) | 2015 | 25,557,032 | 50 (54 conv/linear modules in all) | 76.13 (V1) / 80.86 (V2) | BatchNorm after every conv | yes | A residual stack: 7x7 stem conv + BN + ReLU and a max-pool, then 4 stages of 3/4/6/3 bottleneck blocks (1x1 shrink -> 3x3 -> 1x1 expand, each with BN + ReLU, added to the block's input, the first block of a stage using a 1x1 projection on the skip); GAP and one linear to 1000. | 224x224x3 RGB, ImageNet mean/std normalised | V1 https://download.pytorch.org/models/resnet50-0676ba61.pth; V2 https://download.pytorch.org/models/resnet50-11ad3fa6.pth |
| PlainCNN (ours, M_old / M_recipe) | 2026 | 1,179,384 | 9 (9 conv/linear modules in all) | n/a; shapes test acc 99.99% (old) / 99.96% (recipe); CIFAR-10 91.2 / 89.5 (reports/results/vis.md) | BatchNorm after every conv | no | VGG-style: 4 stages of 2 x (3x3 conv + BN + ReLU) at 32/64/128/256 channels, max-pool between stages, GAP, linear to 24 shape classes; no skips. | 64x64x3 shapes images in [0,1], normalised inside | none (trained by us, `models/vis_train.py`) |
| ConvNeXtS-s (ours, 0.2M) | 2026 | 203,848 | 20 (20 conv/linear modules in all) | n/a; size sweep, see vis.md | LayerNorm (per block, stem, downsamples, head) | yes | Miniature ConvNeXt (dims 16/32/64/96, depths 1/1/2/1), same block as ConvNeXt-T. | 64x64x3 shapes images in [0,1], normalised inside | none (trained by us, `models/vis_train.py`) |
| ConvNeXtS-m (ours, 1.2M, M_arch) | 2026 | 1,182,392 | 23 (23 conv/linear modules in all) | n/a; M_arch: shapes test acc 100%, CIFAR-10 86.9% (reports/results/vis.md) | LayerNorm (per block, stem, downsamples, head) | yes | Miniature ConvNeXt (dims 32/64/128/256, depths 1/1/3/1), parameter-matched to PlainCNN. | 64x64x3 shapes images in [0,1], normalised inside | none (trained by us, `models/vis_train.py`) |
| ConvNeXtS-l (ours, 4.6M) | 2026 | 4,625,752 | 23 (23 conv/linear modules in all) | n/a; size sweep, see vis.md | LayerNorm (per block, stem, downsamples, head) | yes | Miniature ConvNeXt (dims 64/128/256/512, depths 1/1/3/1). | 64x64x3 shapes images in [0,1], normalised inside | none (trained by us, `models/vis_train.py`) |
| ConvNeXt 92k (essay) | 2026 | 92,000 | ? | n/a | LayerNorm (by construction) | yes | smallest of the essay's three ConvNeXts; parameters from the essay text, no code or weights | ? | none (essay text only) |
| ConvNeXt 7.06M (essay) | 2026 | 7,060,000 | ? | n/a | LayerNorm (by construction) | yes | middle size of the essay's sweep; text only | ? | none (essay text only) |
| ConvNeXt 28.6M (essay) | 2026 | 28,600,000 | 18 blocks | n/a | LayerNorm (by construction) | yes | largest; 18 blocks (3/3/9/3 like ConvNeXt-T, whose torchvision count is 28,589,128); text only | ? | none (essay text only) |

GoogLeNet's parameter count includes its two auxiliary heads (6,624,904 without them). ResNet-50 V1 and V2 weights have the same num_params; V2 differs only by training recipe (V1 recipe: https://github.com/pytorch/vision/tree/main/references/classification#resnet; V2 recipe: https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621).

## Stage tables

### InceptionV1 (Lucid, 2014 weights)

Source: torch-lucent 0.1.8 lucent.modelzoo.inceptionv1(pretrained=True); port of the TF inception5h graph. Studied layer: mixed5b (1024 ch, 7x7), the four branch convs pre-ReLU.

| stage | style | params | output (C, H, W) | blocks | note |
|---|---|---|---|---|---|
| conv2d0 | conv + ReLU | 9,472 | 64x112x112 |  | 7x7/2 conv, 64 ch |
| maxpool0 | max pool | 0 | 64x56x56 |  | 3x3/2 |
| lrn0 | local response norm | 0 | 64x56x56 |  | F.local_response_norm(size=9, alpha=1e-4, beta=0.5) |
| conv2d1 | conv + ReLU | 4,160 | 64x56x56 |  | 1x1 conv, 64 ch |
| conv2d2 | conv + ReLU | 110,784 | 192x56x56 |  | 3x3 conv, 192 ch |
| lrn1 | local response norm | 0 | 192x56x56 |  | F.local_response_norm(size=9, alpha=1e-4, beta=0.5) |
| maxpool1 | max pool | 0 | 192x28x28 |  | 3x3/2 |
| mixed3a | Inception module | 163,696 | 256x28x28 |  | branch widths 1x1 64 | 3x3 128 | 5x5 32 | pool_reduce 32, concatenated |
| mixed3b | Inception module | 388,736 | 480x28x28 |  | branch widths 1x1 128 | 3x3 192 | 5x5 96 | pool_reduce 64, concatenated |
| maxpool4 | max pool | 0 | 480x14x14 |  | 3x3/2 |
| mixed4a | Inception module | 372,716 | 508x14x14 |  | branch widths 1x1 192 | 3x3 204 | 5x5 48 | pool_reduce 64, concatenated |
| mixed4b | Inception module | 447,720 | 512x14x14 |  | branch widths 1x1 160 | 3x3 224 | 5x5 64 | pool_reduce 64, concatenated |
| mixed4c | Inception module | 510,104 | 512x14x14 |  | branch widths 1x1 128 | 3x3 256 | 5x5 64 | pool_reduce 64, concatenated |
| mixed4d | Inception module | 605,376 | 528x14x14 |  | branch widths 1x1 112 | 3x3 288 | 5x5 64 | pool_reduce 64, concatenated |
| mixed4e | Inception module | 868,352 | 832x14x14 |  | branch widths 1x1 256 | 3x3 320 | 5x5 128 | pool_reduce 128, concatenated |
| maxpool10 | max pool | 0 | 832x7x7 |  | 3x3/2 |
| mixed5a | Inception module | 1,107,984 | 832x7x7 |  | branch widths 1x1 256 | 3x3 320 | 5x5 128 | pool_reduce 128, concatenated |
| mixed5b **(studied)** | Inception module | 1,444,080 | 1024x7x7 |  | branch widths 1x1 384 | 3x3 384 | 5x5 128 | pool_reduce 128, concatenated |
| avgpool0 | global average pool | 0 | 1024x1x1 |  | 7x7 average pool (functional) |
| softmax2 | linear head | 1,033,200 | 1008 |  | Linear 1024 -> 1008 (2012 devkit class order; 8 unused outputs) |

- Feed-forward: every stage consumes only the previous stage's output; no skip connections.
- The two auxiliary softmax heads of the 2014 paper are absent from the inception5h graph and the lucent port.
- Conv biases present (no BN); LRN after the first pool and after conv2d2 as in the 2014 paper.

### GoogLeNet (torchvision, BatchNorm)

Source: torchvision 0.29 googlenet(weights=IMAGENET1K_V1), aux_logits=True, transform_input=True. Studied layer: inception5b (1024 ch, 7x7), the branch convs pre-ReLU.

| stage | style | params | output (C, H, W) | blocks | note |
|---|---|---|---|---|---|
| conv1 | conv + BatchNorm + ReLU | 9,536 | 64x112x112 |  | 7x7/2 conv (no bias) + BN + ReLU |
| maxpool1 | max pool | 0 | 64x56x56 |  | 3x3/2, ceil_mode |
| conv2 | conv + BatchNorm + ReLU | 4,224 | 64x56x56 |  | 1x1/1 conv (no bias) + BN + ReLU |
| conv3 | conv + BatchNorm + ReLU | 110,976 | 192x56x56 |  | 3x3/1 conv (no bias) + BN + ReLU |
| maxpool2 | max pool | 0 | 192x28x28 |  | 3x3/2, ceil_mode |
| inception3a | Inception module | 155,872 | 256x28x28 |  | branch widths 1x1 64 | 3x3 128 | 3x3(5x5 slot) 32 | pool 1x1 32, each conv + BN + ReLU |
| inception3b | Inception module | 340,224 | 480x28x28 |  | branch widths 1x1 128 | 3x3 192 | 3x3(5x5 slot) 96 | pool 1x1 64, each conv + BN + ReLU |
| maxpool3 | max pool | 0 | 480x14x14 |  | 3x3/2, ceil_mode |
| inception4a | Inception module | 364,512 | 512x14x14 |  | branch widths 1x1 192 | 3x3 208 | 3x3(5x5 slot) 48 | pool 1x1 64, each conv + BN + ReLU |
| aux1 | auxiliary head | 3,188,968 | 1000 |  | avgpool 5x5/3 -> 1x1 conv 128 -> FC 1024 -> FC 1000; off in eval() |
| inception4b | Inception module | 425,232 | 512x14x14 |  | branch widths 1x1 160 | 3x3 224 | 3x3(5x5 slot) 64 | pool 1x1 64, each conv + BN + ReLU |
| inception4c | Inception module | 486,192 | 512x14x14 |  | branch widths 1x1 128 | 3x3 256 | 3x3(5x5 slot) 64 | pool 1x1 64, each conv + BN + ReLU |
| inception4d | Inception module | 573,312 | 528x14x14 |  | branch widths 1x1 112 | 3x3 288 | 3x3(5x5 slot) 64 | pool 1x1 64, each conv + BN + ReLU |
| aux2 | auxiliary head | 3,191,016 | 1000 |  | same as aux1; off in eval() |
| inception4e | Inception module | 803,840 | 832x14x14 |  | branch widths 1x1 256 | 3x3 320 | 3x3(5x5 slot) 128 | pool 1x1 128, each conv + BN + ReLU |
| maxpool4 | max pool | 0 | 832x7x7 |  | 3x3/2, ceil_mode |
| inception5a | Inception module | 978,944 | 832x7x7 |  | branch widths 1x1 256 | 3x3 320 | 3x3(5x5 slot) 128 | pool 1x1 128, each conv + BN + ReLU |
| inception5b **(studied)** | Inception module | 1,347,040 | 1024x7x7 |  | branch widths 1x1 384 | 3x3 384 | 3x3(5x5 slot) 128 | pool 1x1 128, each conv + BN + ReLU |
| avgpool | global average pool | 0 | 1024x1x1 |  | AdaptiveAvgPool2d(1) + dropout 0.2 |
| fc | linear head | 1,025,000 | 1000 |  | Linear 1024 -> 1000 |

- Same topology as (a) (3 stem convs, 9 Inception modules, 4 max-pools, GAP, 1 FC), but every conv is bias-free conv + BatchNorm + ReLU, the LRN layers are gone, the 5x5 branch is a 3x3 conv (TF-slim convention) and the two auxiliary heads are present as modules (used only in training).
- Weights: torchvision's meta says 'ported from the original paper'; they are a BatchNorm-era re-training, not the 2014 release. Top-1 69.78 vs our 69.13 for the Lucid weights.
- torchvision's num_params (6,624,904) excludes the aux heads; with aux_logits=True the module has 13,004,888, and torchvision warns that the aux heads are NOT pretrained (random init).

### ConvNeXt-T (torchvision)

Source: torchvision 0.29 convnext_tiny(weights=IMAGENET1K_V1). Studied layer: stage 3 (features[5]), last block (9 of 9), the 1536 MLP hidden units after GELU.

| stage | style | params | output (C, H, W) | blocks | note |
|---|---|---|---|---|---|
| stem | patchify stem | 4,896 | 96x56x56 |  | 4x4/4 conv 3 -> 96 + LayerNorm (patchify) |
| stage1 | ConvNeXt block | 237,888 | 96x56x56 | 3 | 3 blocks, 96 ch, MLP hidden 384 (x4), layer scale init 1e-6, stochastic depth in training |
| down1 | downsample | 74,112 | 192x28x28 |  | LayerNorm + 2x2/2 conv |
| stage2 | ConvNeXt block | 918,144 | 192x28x28 | 3 | 3 blocks, 192 ch, MLP hidden 768 (x4), layer scale init 1e-6, stochastic depth in training |
| down2 | downsample | 295,680 | 384x14x14 |  | LayerNorm + 2x2/2 conv |
| stage3 **(studied)** | ConvNeXt block | 10,817,280 | 384x14x14 | 9 | 9 blocks, 384 ch, MLP hidden 1536 (x4), layer scale init 1e-6, stochastic depth in training |
| down3 | downsample | 1,181,184 | 768x7x7 |  | LayerNorm + 2x2/2 conv |
| stage4 | ConvNeXt block | 14,289,408 | 768x7x7 | 3 | 3 blocks, 768 ch, MLP hidden 3072 (x4), layer scale init 1e-6, stochastic depth in training |
| avgpool | global average pool | 0 | 768x1x1 |  | AdaptiveAvgPool2d(1) |
| classifier | linear head | 770,536 | 1000 |  | LayerNorm + Linear 768 -> 1000 |

- Residual stream: every block adds its output to its input (x + gamma * MLP(LN(dwconv(x)))), so the same 96/192/384/768-channel tensor runs through each stage and is only rewritten at the downsamples.
- Blocks use GELU, not ReLU, and one LayerNorm per block instead of BatchNorm after every conv.

### ResNet-50 (torchvision, architecture only)

Source: torchvision 0.29 resnet50(weights=None); weights V1 / V2 share this module. Studied layer: layer4 (2048 ch, 7x7) if used.

| stage | style | params | output (C, H, W) | blocks | note |
|---|---|---|---|---|---|
| stem | conv + BatchNorm + ReLU | 9,536 | 64x112x112 |  | 7x7/2 conv 3 -> 64 + BN + ReLU |
| maxpool | max pool | 0 | 64x56x56 |  | 3x3/2 |
| layer1 | bottleneck residual block | 215,808 | 256x56x56 | 3 | 3 bottleneck blocks, 64 -> 256 ch; first block has a 1x1 projection shortcut |
| layer2 | bottleneck residual block | 1,219,584 | 512x28x28 | 4 | 4 bottleneck blocks, 128 -> 512 ch; first block has a 1x1 projection shortcut |
| layer3 | bottleneck residual block | 7,098,368 | 1024x14x14 | 6 | 6 bottleneck blocks, 256 -> 1024 ch; first block has a 1x1 projection shortcut |
| layer4 **(studied)** | bottleneck residual block | 14,964,736 | 2048x7x7 | 3 | 3 bottleneck blocks, 512 -> 2048 ch; first block has a 1x1 projection shortcut |
| avgpool | global average pool | 0 | 2048x1x1 |  | AdaptiveAvgPool2d(1) |
| fc | linear head | 2,049,000 | 1000 |  | Linear 2048 -> 1000 |

- V1 and V2 weights have the same num_params (the same module); V2 adds ~4.7 top-1 points purely by training recipe (long schedule, heavy augmentation, EMA, label smoothing; see the recipe URLs).
- '50' = 49 convs on the main path (1 stem + 16 blocks x 3) + 1 FC; the 4 projection shortcuts are extra.

### PlainCNN (ours, M_old / M_recipe)

Source: models/vis_models.py PlainCNN(24, stem_stride=2); no weights loaded. Studied layer: s4c2 (256 units at 4x4), the last conv's BN output pre-ReLU.

| stage | style | params | output (C, H, W) | blocks | note |
|---|---|---|---|---|---|
| stage1 | conv + BatchNorm + ReLU | 10,208 | 32x32x32 | 2 | 2 x (3x3 conv + BN + ReLU), 32 ch (first conv stride 2) |
| pool1 | max pool | 0 | 32x16x16 |  | 2x2/2 |
| stage2 | conv + BatchNorm + ReLU | 55,552 | 64x16x16 | 2 | 2 x (3x3 conv + BN + ReLU), 64 ch |
| pool2 | max pool | 0 | 64x8x8 |  | 2x2/2 |
| stage3 | conv + BatchNorm + ReLU | 221,696 | 128x8x8 | 2 | 2 x (3x3 conv + BN + ReLU), 128 ch |
| pool3 | max pool | 0 | 128x4x4 |  | 2x2/2 |
| stage4 **(studied)** | conv + BatchNorm + ReLU | 885,760 | 256x4x4 | 2 | 2 x (3x3 conv + BN + ReLU), 256 ch |
| gap | global average pool | 0 | 256x1x1 |  | spatial mean |
| head | linear head | 6,168 | 24 |  | Linear 256 -> 24 |

- VGG-style: 8 convs in 4 stages of 2, max-pool between stages, no skip connections.
- The CIFAR variant differs only in stem stride (1), input (32 px) and head (10 classes).

### ConvNeXtS-s (ours, 0.2M)

Source: models/vis_models.py ConvNeXtS(24, stem_stride=2, **ARCH_SIZES['s']); no weights loaded. Studied layer: s4b1 MLP hidden units (4 x last width) at 4x4.

| stage | style | params | output (C, H, W) | blocks | note |
|---|---|---|---|---|---|
| stem | patchify stem | 816 | 16x32x32 |  | 4x4/2 conv + LayerNorm |
| stage1 | ConvNeXt block | 2,976 | 16x32x32 | 1 | 1 block(s), 16 ch, MLP hidden 64 |
| down1 | downsample | 2,112 | 32x16x16 |  | LayerNorm + 2x2/2 conv |
| stage2 | ConvNeXt block | 10,048 | 32x16x16 | 1 | 1 block(s), 32 ch, MLP hidden 128 |
| down2 | downsample | 8,320 | 64x8x8 |  | LayerNorm + 2x2/2 conv |
| stage3 | ConvNeXt block | 72,960 | 64x8x8 | 2 | 2 block(s), 64 ch, MLP hidden 256 |
| down3 | downsample | 24,800 | 96x4x4 |  | LayerNorm + 2x2/2 conv |
| stage4 **(studied)** | ConvNeXt block | 79,296 | 96x4x4 | 1 | 1 block(s), 96 ch, MLP hidden 384 |
| gap | global average pool | 0 | 96x1x1 |  | spatial mean + LayerNorm |
| head | linear head | 2,520 | 24 |  | Linear 96 -> 24 |

- Miniature of ConvNeXt-T: same block, 4 stages, patchify stem, LN downsamples; depths [1, 1, 2, 1], widths [16, 32, 64, 96].

### ConvNeXtS-m (ours, 1.2M, M_arch)

Source: models/vis_models.py ConvNeXtS(24, stem_stride=2, **ARCH_SIZES['m']); no weights loaded. Studied layer: s4b1 MLP hidden units (4 x last width) at 4x4.

| stage | style | params | output (C, H, W) | blocks | note |
|---|---|---|---|---|---|
| stem | patchify stem | 1,632 | 32x32x32 |  | 4x4/2 conv + LayerNorm |
| stage1 | ConvNeXt block | 10,048 | 32x32x32 | 1 | 1 block(s), 32 ch, MLP hidden 128 |
| down1 | downsample | 8,320 | 64x16x16 |  | LayerNorm + 2x2/2 conv |
| stage2 | ConvNeXt block | 36,480 | 64x16x16 | 1 | 1 block(s), 64 ch, MLP hidden 256 |
| down2 | downsample | 33,024 | 128x8x8 |  | LayerNorm + 2x2/2 conv |
| stage3 | ConvNeXt block | 415,488 | 128x8x8 | 3 | 3 block(s), 128 ch, MLP hidden 512 |
| down3 | downsample | 131,584 | 256x4x4 |  | LayerNorm + 2x2/2 conv |
| stage4 **(studied)** | ConvNeXt block | 539,136 | 256x4x4 | 1 | 1 block(s), 256 ch, MLP hidden 1024 |
| gap | global average pool | 0 | 256x1x1 |  | spatial mean + LayerNorm |
| head | linear head | 6,680 | 24 |  | Linear 256 -> 24 |

- Miniature of ConvNeXt-T: same block, 4 stages, patchify stem, LN downsamples; depths [1, 1, 3, 1], widths [32, 64, 128, 256].

### ConvNeXtS-l (ours, 4.6M)

Source: models/vis_models.py ConvNeXtS(24, stem_stride=2, **ARCH_SIZES['l']); no weights loaded. Studied layer: s4b1 MLP hidden units (4 x last width) at 4x4.

| stage | style | params | output (C, H, W) | blocks | note |
|---|---|---|---|---|---|
| stem | patchify stem | 3,264 | 64x32x32 |  | 4x4/2 conv + LayerNorm |
| stage1 | ConvNeXt block | 36,480 | 64x32x32 | 1 | 1 block(s), 64 ch, MLP hidden 256 |
| down1 | downsample | 33,024 | 128x16x16 |  | LayerNorm + 2x2/2 conv |
| stage2 | ConvNeXt block | 138,496 | 128x16x16 | 1 | 1 block(s), 128 ch, MLP hidden 512 |
| down2 | downsample | 131,584 | 256x8x8 |  | LayerNorm + 2x2/2 conv |
| stage3 | ConvNeXt block | 1,617,408 | 256x8x8 | 3 | 3 block(s), 256 ch, MLP hidden 1024 |
| down3 | downsample | 525,312 | 512x4x4 |  | LayerNorm + 2x2/2 conv |
| stage4 **(studied)** | ConvNeXt block | 2,126,848 | 512x4x4 | 1 | 1 block(s), 512 ch, MLP hidden 2048 |
| gap | global average pool | 0 | 512x1x1 |  | spatial mean + LayerNorm |
| head | linear head | 13,336 | 24 |  | Linear 512 -> 24 |

- Miniature of ConvNeXt-T: same block, 4 stages, patchify stem, LN downsamples; depths [1, 1, 3, 1], widths [64, 128, 256, 512].

## Glossary (plain words)

- **Inception module**: instead of choosing one filter size, the module runs several small convolutions (1x1, 3x3, 5x5) and a pooling branch side by side on the same input and stacks their outputs along the channel axis, so the next layer sees all of them at once.
- **Residual connection**: the block computes a correction and adds it to its own input (output = input + f(input)), so information can flow straight through a deep network and each block only has to learn a small change. The tensor that is passed along and added to is called the residual stream.
- **BatchNorm vs LayerNorm**: both rescale activations to a steady range before the next layer. BatchNorm normalises each channel using statistics over the whole mini-batch (and keeps running averages for test time); LayerNorm normalises each position using the statistics across its own channels, so it needs no batch and behaves the same in training and at test time. LRN (local response normalisation, 2012-2014) divides each activation by the energy of its neighbouring channels; it has no learned parameters.
- **Depthwise convolution**: a convolution in which each channel is filtered by its own small kernel (here 7x7) without mixing channels; channel mixing is done separately by 1x1 layers. It is much cheaper than a full conv.
- **Inverted bottleneck**: the block widens the representation first (1x1 to 4x the channels, then a nonlinearity) and narrows it back, the opposite of ResNet's bottleneck, which narrows first. In ConvNeXt the wide 4x hidden layer is where the units we study live.

## Where this information comes from

We instantiate each PyTorch module on the CPU and walk it: forward hooks on the top-level stages record the output shape at a 1x3x224x224 input (64x64 for our own nets), parameter counts are summed over each stage's modules, and normalisation / auxiliary heads / LRN are verified by listing module types and, for the lucent port, reading the forward() source (LRN is a functional call there, not a module). torchvision defines the architectures in Python code and stores training-recipe metadata (num_params, acc@1, recipe URL, download URL) in each weights enum (`ResNet50_Weights.IMAGENET1K_V1.meta`, `.url`); lucent stores its download URL in `lucent/modelzoo/inceptionv1/InceptionV1.py` (`model_urls`). HuggingFace-hosted models instead describe the architecture in a `config.json`: for facebook/convnext-tiny-224 we read https://huggingface.co/facebook/convnext-tiny-224/raw/main/config.json (fetched live): depths [3, 3, 9, 3], hidden_sizes [96, 192, 384, 768], patch_size 4, layer_scale_init_value 1e-06, hidden_act gelu. Compared with torchvision's instantiated module (depths [3, 3, 9, 3], widths [96, 192, 384, 768]): **matched**. The essay's three ConvNeXt sizes (92k, 7.06M, 28.6M; 18 blocks for the largest) are copied from its text; we have no code or weights for them. The 28.6M figure agrees with ConvNeXt-T's 28,589,128 to three digits, so it is very likely the same 3/3/9/3 architecture.
