# Results: our neuron-cleanliness tools on the essay's own setting (ImageNet, prefix `imnet_`)

Step 1 of the replication of Cammarata (2026), *Neural Networks Are Already Clean*, Goodfire
(https://already-clean.goodfire.com/). Our small-model study (`vis.md`) reproduced the tools step but not the
SAE, rotation or architecture claims. Here the same tools (`analysis/clean_tools.py`, after the 2026-11 fixes)
run on pretrained ImageNet classifiers over the 50,000 ImageNet validation images, at the layers where the
essay reports numbers. The question: **do our tools reproduce the essay's numbers in its own setting?** If they
do, the earlier small-model misses come from scale; if not, our measurements differ from the essay's.

**Decision.** **Mostly yes for the activation-level claims, no for the dictionary claim, so the answer is
split.** In the essay's setting, three of the four claims reproduce with our tools: the weak-activation grid
(GoogLeNet's last layer matches to within 0.2 pp at p25–p95), the ConvNeXt-over-InceptionV1 ordering of
last-layer interpretability, and the rotation search landing on ConvNeXt neurons (on 256-unit sets, with a
random-init control at 0%). Those were misses or weak results on our 1.2M-parameter models, so **scale (or
the real ImageNet models) accounts for them.** The SAE near-copy claim does not reproduce: 0% of SAE activity in
GoogLeNet and 8–9% in ConvNeXt-T, against the essay's 88%. Our planted-neuron check recovers 98%, so
the SAE tool works. Either the essay's SAE or "near-copy" definition differs from ours, or the claim
depends on the essay's own models. For that claim **our measurement and the essay's differ**, and scale alone
does not explain the vis_ miss. The absolute interpretability numbers (essay ~70% / ~99%, human-judged) are
not reproduced by any automatic proxy we have. Class dominance gives 5% / 38%, and one non-blind reader gives
42% / 71%. Only the ordering is a match.

Every threshold the essay does not state is ours and is marked **(ours)**. Essay quotes are from the live page
(fetched 2026-11-05); the weak-activation grid values are the figure's cell labels, saved verbatim in
`imnet_essay_weak_grid.json`.

## Claim-by-claim table

| claim (essay) | essay number | ours (ImageNet val, 50k) | random-init control | verdict |
|---|---|---|---|---|
| Last layer interpretable, InceptionV1 retrained vs ConvNeXt ("A History of Microscope Errors") | "more like 70%" vs "about 99%" (human reading) | top-20 ≥ 50% one class, + pole: **GoogLeNet 4.7%, ConvNeXt-T 38.4%**, ConvNeXt-S 27.8%. Superclass (K = 60): 17% / 52% / 40%. One reader, n = 24: 42% / 71% / 54% | 0% (all three) | **ordering reproduces** (ConvNeXt ≫ InceptionV1, ×8 by class dominance, ×1.7 by reading); absolute levels do not; proxy is stricter than legibility |
| SAE near-copies ("What Dictionaries Learn") | "On the last layer, 88% of its activity is near-copies of single neurons" | GoogLeNet inception5b: **0.0%** (k = 8 and 32). ConvNeXt-T last MLP: **8.8% / 7.7%** (k = 8 / 32). ConvNeXt-T residual stream: 0.0% | 0% (rotated basis 0% everywhere; planted neurons 98%) | **does not reproduce** |
| Removing unmatched SAE directions | "leaves accuracy unchanged, and the few answers it flips were already near-ties" | GoogLeNet −12.3 / −21.8 pp. ConvNeXt-T −1.5 / −1.0 pp, flipped answers are near-ties (median top-1/top-2 margin 0.08–0.10 vs 0.69 overall), but zeroing the whole block costs only −1.1 pp | – | **not reproduced / uninformative** (GoogLeNet breaks; on ConvNeXt the block barely matters) |
| Rotation search ("What Dictionaries Learn") | "starting from a random one, you land back on the neurons, in every block" | ConvNeXt-T last MLP, 256-unit sets: **76% of found axes within \|cos\| ≥ 0.9 of a neuron** (raw), 94% (whitened). 1,024-unit sets: 20% / 44%. Full 3,072: 0% (search under-converged). GoogLeNet: 0% everywhere | ConvNeXt-T init 0% / 4%; GoogLeNet init 0% | **reproduces for ConvNeXt** (on sets where our optimiser converges), **not for GoogLeNet** |
| Weak activations, Inception final layer ("Weak Activations: Further Measurements") | keep strongest quarter (p75): 69.47% vs 70.08% (−0.60 pp); p95 −7.72; p99 −33.63 | GoogLeNet inception5b: p75 **−0.49 pp** (69.29 vs 69.78%); p95 −7.52; p99 −47.81 | – (random init is at chance) | **reproduces** (p25–p95 within 0.2 pp; p99 harsher) |
| Weak activations, ConvNeXt MLP branches, one block at a time | last block ≈ 0 at every cutoff (medium: +0.02 to +0.06 pp); other blocks −0.0 to −0.7 pp up to p95, except the first block (−4.1 at p95) | last block **−0.01 to +0.01 pp**; all 18 blocks within 0.4 pp of the essay's row at p25–p95, first block −3.5 at p95 | – | **reproduces** (block by block) |
| Cutoff at all 18 blocks | p75 −1.02, p95 −19.12, **p99 → 3.80%** (medium); large p99 → 27.78% | ConvNeXt-T all 18 blocks: p75 −4.59, p95 −55.26, **p99 → 11.96%** | – | **reproduces qualitatively** (tolerant to p50, collapse by p99); ours is more fragile from p75 |
| Weakest 75% vs strongest at equal mass (medium ConvNeXt, all 18 blocks) | "accuracy 72% → 71% · 1 in 10 answers change" vs "72% → 4% · 96 in 100 answers change" | ConvNeXt-T: weakest 75% 82.1 → 77.5% (13% of answers change); mass-matched strongest 82.1 → 0.9% (99% change) | – | **reproduces qualitatively** (the asymmetry is the same; our weak cut costs more) |

## 1. Setup

**Data.** ILSVRC2012 validation, 50,000 images, 1000 classes, from the HF dataset `ILSVRC/imagenet-1k` (14
parquet files, `data/imagenet_val/`). Preprocessing is the standard eval transform: shorter side to 256
(bilinear, antialiased), centre crop 224, ImageNet mean/std. The crops are cached once as a uint8 memmap
(`data/imnet_cache/imagenet_val/images_u8.npy`, 7.5 GB). Labels use the standard index. Checks: torchvision
GoogLeNet top-1 **69.78%** (reference 69.78%) and ConvNeXt-T **82.14%** (reference 82.52%, which uses resize
236 instead of 256). ConvNeXt-S scores 83.28%. ImageNetV2 (matched-frequency, 10,000 images,
`data/imagenetv2/`) is the held-out "fit" set for the SAEs and weak-activation cutoffs, and a secondary check.

**Models** (torchvision, `IMAGENET1K_V1`):
- `googlenet` (InceptionV1, 6.6M parameters). This is a modern torchvision retrain with BatchNorm, closest to
  the essay's "retrained with more reasonable modern regularization" rung (~70%). It is **not** the 2014
  checkpoint, and **not** the essay's own retrain: the essay's baseline is 70.08% and ours is 69.78%.
- `convnext_tiny` (28.6M; 18 blocks in stages 3/3/9/3, like the essay's grid) and `convnext_small` (50M,
  27 blocks in stage 3). The essay's ConvNeXts are its own models with a **SwiGLU** MLP ("Magnitude of
  SwiGLU product"). torchvision's use GELU, so our "neuron" is the GELU MLP hidden unit, the nearest analogue.
- Random-init control of each, measured the same way. For GoogLeNet the torchvision init (std 0.01) with fresh
  BatchNorm statistics drives the last layer to exactly 0 in fp16, which makes that control degenerate. We
  re-estimated the BN running statistics on 2,000 ImageNetV2 images, leaving the weights untouched
  **(ours)**. ConvNeXt uses LayerNorm and needs no fix.

**"The last layer"** (the tensor each claim is measured on):
- GoogLeNet: **`inception5b`**, 1024 channels at 7×7. Its output is the concat of four post-ReLU branches.
  Poles use the **pre-ReLU** branch BatchNorm outputs, captured with a clone because torchvision's ReLU is
  in-place. The weak grid, SAE and rotation use the post-ReLU block output. This matches the essay's
  `mixed5b`.
- ConvNeXt: the **last block's MLP hidden** (`features.7.2`, pwconv1 output, 3072 units at 7×7). Poles use
  the pre-GELU values. The weak grid, SAE and rotation use the post-GELU values with the residual bypass
  intact. As secondary views we also measured the stage-3 output (the final residual stream, 768 channels)
  and one mid layer (the last block of stage 2).
- Pole score per image: the spatial max of the pre-activation (+ pole) and minus the spatial min (− pole).

Runtime on the shared RTX 5070 Ti: about 5 hours wall-clock, most of it the weak grid (155 passes over
50k images) and the 12k-feature SAEs. Peak GPU memory stayed under 6 GB.

## 2. Last-layer interpretability (`imnet_purity.png`, `imnet_top20_*.png`)

Essay: "I put it around 5% with tools of that era, others at ~0%. With better interpretability tools ... it's
more like 50% on the same network. ... Retraining the same architecture with more reasonable modern
regularization it's more like 70%. In the ConvNeXt of the same size, it's about 99%." Neurons are shown as
"Each neuron's 20 strongest responses out of 50,000 images, in order."

Our proxy **(ours)**: for each pole, take the 20 strongest of the 50,000 val images. Class-dominated means ≥ 10
of the 20 share one ground-truth class. Superclass-dominated means ≥ 10 share one WordNet group, where each
class goes to its shallowest ancestor covering ≤ 60 classes (125 groups such as "bird", "working dog",
"structure"). The coarser K = 130 version has 47 groups. A random 20-image set is dominated with probability
< 0.01%.

| model, last layer | + pole: class | + pole: superclass (K=60) | + pole: superclass (K=130) | − pole: class | − pole: superclass | one reader, + pole (n = 24) |
|---|---|---|---|---|---|---|
| GoogLeNet inception5b (1024) | **4.7%** | 17.2% | 26.1% | 2.6% | 18.8% | **42%** (10/24) |
| ConvNeXt-T last MLP (3072) | **38.4%** | 51.9% | 58.7% | 0.1% | 1.4% | **71%** (17/24) |
| ConvNeXt-S last MLP (3072) | 27.8% | 39.6% | 46.9% | 4.2% | 7.9% | 54% (13/24) |
| random init (all three) | 0.0% | 0.0% | ≤ 0.1% | 0.0% | 0.0% | – |

Secondary layers, + pole, class / superclass: GoogLeNet inception4e 2.3% / 19.2%. ConvNeXt-T stage-2 last
MLP 3.0% / 13.8%, and its final residual stream (stage-3 output) 1.2% / 2.0%. ConvNeXt-S stage-2 last MLP
**56.1% / 65.1%**, residual stream 0.3% / 0.5%.

- **The ordering reproduces.** By class dominance ConvNeXt-T's last layer is 8× cleaner than GoogLeNet's
  (38% vs 4.7%), and by superclass dominance 3× (52% vs 17%). Random-init controls are 0%.
- **The absolute numbers do not, and this proxy cannot test them.** Class dominance is much stricter than the
  essay's yardstick. Many legible GoogLeNet units have a theme that crosses classes: digital displays across
  clock and watch classes, totem poles, small lights in the dark, face masks, keyboards and typewriters (see
  `imnet_top20_googlenet.png`). A reader names these, but they are not one class. The closest we came to the
  essay's yardstick was one reader (the agent that ran this study, not blind to the model) reading the
  contact sheets: 42% (95% CI 22–63%) vs 71% (49–87%). That is below the essay's 70% / 99%, with the same
  ordering. It is a rough check, not a replication of the essay's agent and human audit. Our InceptionV1 is
  also a different retrain from the essay's.
- **Negative poles are not where the meaning is** in these networks: ConvNeXt's pre-GELU minima are 0.1%
  class-dominated. The essay counts neurons by their strongest responses, so the headline uses + poles.
- ConvNeXt-S's last block is *less* class-dominated than ConvNeXt-T's (28% vs 38%), but its stage-2 output is
  much more so (56%). The essay notes "inexpensive final-block deletion". Here the late class detectors may sit
  one stage earlier.
- **The census tools give no signal at 1000 classes.** `clean_tools.census` against class and superclass
  indicators labels 0% of last-layer poles "clean" or even "partial", in every model, raw or percentile-rank
  (mean best R² 0.02–0.04). One class is 0.1% of images, so a perfect single-class detector explains little
  variance of a continuous pole score. `clean_tools.purity` on the top 1% (500 images, chance-corrected)
  gives 0.7% / 8.9% / 6.2% "one job" (GoogLeNet / ConvNeXt-T / -S). Both measures were built for few-factor
  ground truth (vis_, Pluto). For ImageNet the top-20 dominance above is the usable automatic proxy.
- ImageNetV2 (10 images per class) cannot reach ≥ 10 of 20 in one class. Superclass dominance there keeps the
  ordering: GoogLeNet 3.2%, ConvNeXt-T 31.4%, ConvNeXt-S 23.7%, random init 0%.

## 3. Dictionaries: SAE near-copies (`imnet_sae_rotation.png`, left)

Essay: "On the last layer, 88% of its activity is near-copies of single neurons, and the rest is faint:
removing every dictionary direction that matches no neuron leaves accuracy unchanged, and the few answers it
flips were already near-ties."

Ours **(ours)**: a TopK SAE (`clean_tools.train_topk_sae`) with 4× as many features as neurons, k ∈ {8, 32},
4,000 Adam steps × 2,048 vectors. It is fit on last-layer post-activation vectors from ImageNetV2 (100k vectors,
10 random positions per image) and evaluated on val. A feature is a near-copy if its decoder direction has
max |cos| ≥ 0.9 with one neuron axis. The share is the near-copy features' mean activation over the total.

| last layer | k | SAE FVE | near-copy share of activity | near-copy share of live features | rotated-basis control | random init |
|---|---|---|---|---|---|---|
| GoogLeNet inception5b | 8 | 0.43 | **0.0%** | 0.0% | 0.0% | 0.0% |
| GoogLeNet inception5b | 32 | 0.52 | **0.0%** | 0.0% | 0.0% | 0.0% |
| ConvNeXt-T last MLP | 8 | 0.36 | **8.8%** | 3.7% | 0.0% | 0.0% |
| ConvNeXt-T last MLP | 32 | 0.46 | **7.7%** | 1.6% | 0.0% | 0.0% |
| ConvNeXt-T residual stream (768) | 8 / 32 | 0.57 / 0.65 | 0.0% / 0.0% | 0.0% | 0.0% | 0.0% |
| planted: sparse codes on the neuron axes | 8 | 0.99 | **98.4%** | – | rotated plant: 0.0% (100% match the planted directions) | – |

- **Does not reproduce.** In GoogLeNet no SAE feature is within |cos| 0.9 of a neuron. The decoder-to-neuron
  cosines peak at 0.1–0.2, about what random directions in 1024-d give. In ConvNeXt-T a tail of features is
  neuron-aligned: the cosine histogram has mass up to 0.9–1.0, and random init has none above 0.3. That tail
  carries 8–9% of SAE activity, not 88%. The planted check shows the tool can find near-copies when they are
  there, and the rotated-basis control shows the 8% is a real basis preference.
- **Removal test.** We remove the non-near-copy features' contribution, x → x − Σ_{f unmatched} z_f d_f, which
  keeps the SAE error term. GoogLeNet loses 12.3 / 21.8 pp (k = 8 / 32), because with no near-copies every
  feature is "unmatched". ConvNeXt-T loses 1.5 / 1.0 pp, and the flipped answers are near-ties: median
  baseline top-1/top-2 margin 0.10 / 0.08, against 0.69 over all images. That matches the essay's wording,
  but **the control makes it uninformative**: zeroing the entire last-block MLP costs only 1.1 pp in ConvNeXt-T,
  so removing anything from that block is cheap. In GoogLeNet zeroing inception5b drops accuracy to chance.
- Our SAEs are modest (FVE 0.36–0.65; no auxiliary loss; 4k steps). A stronger SAE might find more
  near-copies. But in GoogLeNet fewer than 2% of SAE features have |cos| > 0.5 to any neuron and none reach 0.9, so nothing in our runs
  points to the essay's 88% being one better optimiser away. The essay's SAE figure is anchored
  "convnext-sae-residual", so it may have used the residual stream. There we find 0%.

## 4. Rotation search (`imnet_sae_rotation.png`, right)

Essay: "if you search the rotations of a layer for the one where a handful of units carry the most activity,
starting from a random one, you land back on the neurons, in every block."

Ours **(ours)**: `clean_tools.rotation_search` (fixed version) maximises E Σᵢ (Rx)ᵢ⁴ over orthogonal R on 40,000
last-layer post-activation vectors. "raw" uses centred, globally scaled activations; "ICA" uses ZCA-whitened
ones. Each search starts from a random rotation, and also from the identity as a sanity check. We report the
share of found axes within |cos| ≥ 0.9 of a neuron, with mean max |cos| in brackets. Random directions give
0.12 (d = 1024), 0.07 (3072) and 0.19 (256).

| layer, unit set | steps | raw, random start | raw, identity start | ICA, random start | random init: raw / identity / ICA |
|---|---|---|---|---|---|
| ConvNeXt-T last MLP, 3 × 256 units | 1,000 | **76% (0.88)** | 94% (0.97) | **94% (0.97)** | 0% / 67% / 4% |
| ConvNeXt-T last MLP, 1,024 units | 3,000 | 20% (0.46) | 47% (0.82) | 44% (0.78) | 0% / 47% / 0% |
| ConvNeXt-T last MLP, all 3,072 | 1,000 | 0% (0.07) | 21% (0.75) | 0.4% (0.26) | 0% / 87% / 0% |
| GoogLeNet inception5b, 3 × 256 | 1,000 | 0% (0.26) | 7% (0.66) | 0.1% (0.35) | 0% / 87% / 0% |
| GoogLeNet inception5b, all 1,024 | 3,000 | 0% (0.12) | 2% (0.62) | 0% (0.17) | 0% / 97% / 0% |

- **ConvNeXt: reproduces where the search converges.** On 256-unit sets, three quarters of the axes found from
  a random rotation are neurons, and 94% after whitening. The random-init ConvNeXt gives 0%, so this is
  learned. With 1,024 units the share falls, but the random start still ends *below* the identity start's
  objective (1.25 vs 1.40× the neuron basis), so the optimiser has not converged. At 3,072 units, 1,000
  normalised steps barely move a random rotation (end objective 1.1× vs 3.2× from the identity). The
  full-layer 0% is an optimiser limit, not evidence against the claim.
- **GoogLeNet: does not reproduce.** Even started at the identity, the search leaves the neurons (2–7% of axes
  stay) and finds rotations with a 2.6–11× higher objective. From a random start nothing lands near a neuron,
  raw or whitened. Under this objective the neuron basis of GoogLeNet's last layer is not a local optimum.
- Random-init identity starts mostly stay put (47–99% of axes) but random starts never find the neurons. In untrained nets
  the axes are a local optimum without a basin around them.

## 5. Weak activations (`imnet_weak_grid.png`)

Essay ("Weak Activations: Further Measurements"): "Keeping roughly the strongest quarter of positive
activations in Inception's final layer preserves 69.47% accuracy, versus 70.08% originally. ConvNeXt's late MLP
branches are similarly tolerant of weak-response removal, with their residual bypass intact." "The bottom row
of each ConvNeXt panel applies the cutoff at all 18 blocks at once. At p99, medium falls to 3.80% accuracy
and large to 27.78%".

Cutoff **(ours, chosen to match that wording)**: per neuron, τ_p is the p-th percentile of its activation
magnitude, estimated on ImageNetV2 (5,000 images × 4 random positions). GoogLeNet uses each layer's post-ReLU
output, takes the percentile over **positive** activations, and zeroes values < τ_p. ConvNeXt uses each
block's post-GELU MLP hidden, takes the percentile over |GELU(h)| at all positions, and zeroes |a| < τ_p; the
residual bypass is untouched. One layer at a time, plus "all" (every ConvNeXt block at once). Top-1 is measured
on all 50,000 val images. Repeated identical runs differ by about ±0.01 pp (fp16, cuDNN).

**GoogLeNet**, change in top-1 (pp). Ours, with the essay's InceptionV1 in brackets (base 69.78% vs 70.08%):

| layer (essay name) | p25 | p50 | p75 | p95 | p99 |
|---|---|---|---|---|---|
| conv1 (conv2d0) | −28.9 (−0.2) | −63.3 (−3.5) | −69.3 (−22.4) | −69.6 (−59.2) | −69.6 (−66.6) |
| conv2 (conv2d1) | −6.2 (−0.3) | −24.4 (−4.4) | −47.6 (−20.9) | −66.8 (−58.9) | −69.2 (−68.6) |
| conv3 (conv2d2) | −2.4 (−0.1) | −12.8 (−0.9) | −42.3 (−11.2) | −67.4 (−47.0) | −69.5 (−64.7) |
| inception3a (mixed3a) | −0.9 (+0.0) | −5.6 (−0.7) | −31.2 (−5.4) | −68.6 (−40.6) | −69.6 (−64.6) |
| inception3b (mixed3b) | −0.2 (+0.0) | −1.5 (−0.1) | −9.2 (−2.1) | −57.2 (−24.3) | −68.9 (−54.4) |
| inception4a (mixed4a) | −0.1 (−0.0) | −1.0 (−0.1) | −11.0 (−1.9) | −68.4 (−19.7) | −69.6 (−49.4) |
| inception4b (mixed4b) | −0.6 (−0.0) | −3.0 (−0.2) | −11.8 (−2.2) | −55.5 (−18.3) | −68.9 (−43.9) |
| inception4c (mixed4c) | −0.2 (−0.1) | −1.8 (−0.3) | −10.8 (−2.0) | −58.4 (−18.9) | −69.1 (−42.6) |
| inception4d (mixed4d) | −0.2 (−0.0) | −1.4 (−0.1) | −6.5 (−1.5) | −38.5 (−12.5) | −65.5 (−33.9) |
| inception4e (mixed4e) | −0.0 (−0.0) | −0.2 (−0.0) | −1.9 (−0.8) | −17.8 (−8.2) | −50.0 (−25.4) |
| inception5a (mixed5a) | +0.0 (+0.0) | −0.2 (−0.1) | −1.7 (−1.7) | −14.4 (−13.9) | −43.3 (−34.5) |
| **inception5b (mixed5b)** | −0.0 (−0.0) | −0.1 (−0.0) | **−0.5 (−0.6)** | **−7.5 (−7.7)** | −47.8 (−33.6) |

Mass-matched pair at inception5b (OURS, `clean_tools.weak_strong_thresholds`): removing the weakest 75% of
nonzero activations costs −0.48 pp (6.5% of answers change). Removing the strongest 12.7%, which carry the same
total activation, costs −6.12 pp (20.9% change).

**ConvNeXt-T** (base 82.14%), change in top-1 (pp), with the essay's ConvNeXt medium (base 72.08%) for the
matching rows:

| block (essay row) | p25 | p50 | p75 | p95 | p99 |
|---|---|---|---|---|---|
| f1.0 (stage0.block0) | +0.03 (+0.00) | -0.05 (-0.05) | -0.15 (-0.27) | -3.53 (-4.07) | -5.63 (-13.44) |
| f1.1 (stage0.block1) | +0.02 (-0.00) | -0.01 (-0.01) | -0.11 (-0.05) | -0.39 (-0.48) | -0.44 (-1.09) |
| f1.2 (stage0.block2) | +0.01 (+0.00) | -0.10 (-0.04) | -0.51 (-0.13) | -0.48 (-0.66) | -0.53 (-1.60) |
| f3.0 (stage1.block0) | +0.00 (-0.03) | -0.06 (-0.00) | -0.27 (-0.06) | -0.31 (-0.61) | -0.32 (-1.08) |
| f3.1 (stage1.block1) | -0.01 (+0.00) | -0.05 (-0.02) | -0.38 (-0.08) | -0.56 (-0.45) | -0.37 (-0.81) |
| f3.2 (stage1.block2) | +0.02 (-0.02) | +0.00 (-0.02) | -0.10 (-0.07) | -0.15 (-0.44) | -0.16 (-0.96) |
| f5.0 (stage2.block0) | +0.01 (-0.02) | +0.00 (-0.02) | +0.02 (-0.05) | -0.09 (-0.39) | -0.10 (-0.76) |
| f5.1 (stage2.block1) | +0.01 (-0.01) | +0.02 (+0.00) | -0.06 (-0.05) | -0.18 (-0.22) | -0.28 (-0.42) |
| f5.2 (stage2.block2) | +0.00 (+0.01) | -0.05 (+0.03) | -0.12 (-0.01) | -0.19 (-0.12) | -0.11 (-0.35) |
| f5.3 (stage2.block3) | +0.00 (+0.02) | +0.01 (-0.02) | -0.05 (-0.03) | -0.12 (-0.16) | -0.19 (-0.33) |
| f5.4 (stage2.block4) | +0.02 (-0.02) | -0.02 (-0.04) | -0.07 (-0.05) | -0.15 (-0.17) | -0.22 (-0.43) |
| f5.5 (stage2.block5) | +0.00 (-0.02) | +0.01 (-0.01) | -0.01 (-0.02) | -0.10 (-0.12) | -0.15 (-0.31) |
| f5.6 (stage2.block6) | +0.00 (-0.01) | +0.02 (-0.01) | -0.01 (-0.02) | -0.20 (-0.11) | -0.26 (-0.44) |
| f5.7 (stage2.block7) | +0.02 (-0.02) | +0.00 (-0.03) | -0.06 (-0.05) | -0.12 (-0.20) | -0.32 (-0.52) |
| f5.8 (stage2.block8) | +0.00 (+0.01) | +0.03 (+0.01) | +0.00 (+0.00) | -0.10 (-0.06) | -0.10 (-0.23) |
| f7.0 (stage3.block0) | -0.01 (+0.01) | +0.01 (-0.00) | +0.01 (-0.02) | -0.04 (-0.37) | -0.05 (-0.65) |
| f7.1 (stage3.block1) | +0.01 (+0.02) | +0.01 (+0.01) | +0.03 (+0.03) | -0.03 (+0.00) | -0.09 (-0.36) |
| **f7.2** (stage3.block2) | +0.01 (+0.02) | +0.00 (+0.02) | +0.01 (+0.03) | -0.01 (+0.03) | -0.00 (+0.06) |
| **all** (all) | -0.04 (-0.01) | -0.61 (-0.13) | -4.59 (-1.02) | -55.26 (-19.12) | -70.18 (-68.28) |

Mass-matched pairs: in the last block, the weakest 75% cost +0.02 pp and the mass-matched strongest −0.71 pp.
In all 18 blocks, the weakest 75% take 82.1 → 77.5% (13.4% of answers change) and the mass-matched strongest
take it to 0.9% (99.0% change). Essay (medium, all 18 blocks): "Removing the weakest 75%: accuracy 72% → to
71% · 1 in 10 answers change. Removing the strongest 7%: accuracy 72% → to 4% · 96 in 100 answers change."

- **Last layers reproduce closely.** GoogLeNet's final layer matches the essay cell for cell through p95,
  and p75 is −0.49 vs −0.60 pp. The ConvNeXt last block is flat at every cutoff, as in the essay.
- **ConvNeXt single blocks reproduce closely, block by block.** Every one of the 18 blocks of our ConvNeXt-T
  is within 0.4 pp of the essay's ConvNeXt medium at p25–p95. The exception is the first block, where both are
  the most sensitive (p95 −3.5 vs −4.1 pp; p99 −5.6 vs −13.4). One block at a time, cutting everything below
  p95 costs ≤ 0.6 pp anywhere past the first block. Our p99 cells are milder than the essay's.
- **The all-blocks row and the weak-vs-strong asymmetry reproduce in shape**, but our ConvNeXt-T is more
  fragile from p75 on (−4.6 vs −1.0 pp; p99 12.0% vs 3.8%).
- **Early GoogLeNet layers are far more fragile in ours** (conv1 p25 −28.9 vs −0.2 pp). The essay's
  InceptionV1 is a different network: its own retrain, with layer names from the Lucid model. Our
  "percentile of positive activations" may also differ from its cutoff in layers where almost every
  activation is positive. We did not try to resolve this.

## 6. What changed after the clean_tools fixes (vision study)

See `vis.md`, "Changes after the clean_tools fixes". `poles()` (2-D input) changed no vis number. `purity()`
restricted to categorical factors left every trained-model number unchanged and lowered the shapes
random-init controls (late + poles with one main job 89% → 50%; "read correctly" controls 39/38/15% →
30/29/10%). `rotation_search` with a decaying step left the vis rotation results unchanged (0% aligned; ICA
mean |cos| within 0.003). Its identity-start check shows the small CNNs' neuron basis is not a local optimum
of the raw objective, the same as GoogLeNet here. No vis headline claim changed.

## 7. Caveats

1. **Models are not the essay's.** Our InceptionV1 is torchvision's BN retrain, and our ConvNeXts use a GELU
   MLP where the essay's use SwiGLU, with different training and sizes. ConvNeXt-T is 28.6M parameters,
   not the essay's 7.06M parameter-matched ConvNeXt; the essay's "ConvNeXt of the same size" is that 7.06M one.
2. **Interpretability proxy.** Class or superclass dominance of the top 20 is an automatic lower bound on
   legibility. The single-reader check is small (n = 24), not blind, and done by an AI agent.
3. **SAE recipe.** Ours is a plain TopK SAE, 4× wide, k ∈ {8, 32}, 4k steps, fit on ImageNetV2 vectors. The
   essay does not give its recipe or its exact near-copy criterion. The 0.9 cosine threshold and the
   activity-share definition are ours.
4. **Rotation objective and optimiser** are ours. At d ≥ 1024 the search from a random start is
   under-converged, so the subset results carry the ConvNeXt verdict.
5. **Weak-grid cutoffs** are estimated on ImageNetV2 activations (5k images, 4 positions each) and applied to
   val. The ConvNeXt cutoff uses |GELU| as our stand-in for "magnitude of SwiGLU product".
6. Top-1 numbers use fp16 autocast. Paired differences below ~0.02 pp are noise.

## How to rerun

```
uv run python analysis/imnet_run_all.py --data imagenet_val     # cache, extract, purity, sae, rotation, weak, figures
uv run python analysis/imnet_run_all.py --data imagenetv2 --steps purity    # secondary purity check
```
Needs `data/imagenet_val/data/validation-*-of-00014.parquet` and `data/imagenet_val/meta/classes.py` (HF
`ILSVRC/imagenet-1k`), `data/imagenetv2/images/` (ImageNetV2 matched-frequency) and WordNet in
`data/nltk_data` (`nltk.download("wordnet")`). Steps resume from `data/imnet_cache/` and the JSONs; set
`IMNET_RECOMPUTE=1` to redo SAE or rotation entries. Code: `analysis/imnet_common.py` (data, models, hooks),
`imnet_extract.py`, `imnet_purity.py`, `imnet_sae.py`, `imnet_rotation.py`, `imnet_weak.py`, `imnet_figures.py`.
Numbers: `reports/results/imnet.json` (summary), `imnet_purity.json`, `imnet_sae.json`, `imnet_rotation.json`,
`imnet_weak.json`, `imnet_essay_weak_grid.json` (essay values), `imnet_legibility_ours.json` (reader judgments).
