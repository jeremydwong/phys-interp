# Results: our neuron-cleanliness tools on the essay's own setting (ImageNet, prefix `imnet_`)

Step 1 of the replication of Cammarata (2026), *Neural Networks Are Already Clean*, Goodfire
(https://already-clean.goodfire.com/). Our small-model study (`vis.md`) reproduced the tools step but not the
SAE, rotation or architecture claims. Here the same tools (`analysis/clean_tools.py`, after the 2026-11 fixes)
run on pretrained ImageNet classifiers over the 50,000 ImageNet validation images, at the layers where the
essay reports numbers. The question: **do our tools reproduce the essay's numbers in its own setting?** If they
do, the earlier small-model misses come from scale; if not, our measurements differ from the essay's.

**Decision.** **Yes for the activation-level claims and, once the SAE is trained the essay's way, for the
dictionary claim on ConvNeXt; not for GoogLeNet.** In the essay's setting, three of the four claims reproduce with our tools: the weak-activation grid
(GoogLeNet's last layer matches to within 0.2 pp at p25–p95), the ConvNeXt-over-InceptionV1 ordering of
last-layer interpretability, and the rotation search landing on ConvNeXt neurons (on 256-unit sets, with a
random-init control at 0%). Those were misses or weak results on our 1.2M-parameter models, so **scale (or
the real ImageNet models) accounts for them.** With our first, weak SAEs the near-copy claim did not reproduce (0% of SAE activity in GoogLeNet, 8–9% in
ConvNeXt-T, against the essay's 88%). **Retrained the essay's way (AuxK loss, 25.6M training positions; section
3b), ConvNeXt-T's last MLP reaches 84% at FVE 0.77 (essay: 88% at 0.82) and ConvNeXt-S 76%**, with the
rotated-basis and random-init controls at 0%. So that miss came from our SAE, and the dictionary claim
reproduces for ConvNeXt's MLP neurons. It does not for GoogLeNet's last layer (0–12% except very dense SAEs)
or for ConvNeXt's residual stream (0% everywhere). The absolute interpretability numbers (essay ~70% / ~99%, human-judged) are
not reproduced by any automatic proxy we have. Class dominance gives 5% / 38%, and one non-blind reader gives
42% / 71%. Only the ordering is a match.
**On the essay's own 2014 InceptionV1** (section after 5): the weak grid reproduces cell for cell in all 12 layers
and the SAE near-copy claim reproduces (87.5% at FVE 0.84 vs 88% at 0.82), while rotation search and absolute
purity (2.4% class-dominated) still do not.

Every threshold the essay does not state is ours and is marked **(ours)**. Essay quotes are from the live page
(fetched 2026-11-05); the weak-activation grid values are the figure's cell labels, saved verbatim in
`imnet_essay_weak_grid.json`.

## Claim-by-claim table

| claim (essay) | essay number | ours (ImageNet val, 50k) | random-init control | verdict |
|---|---|---|---|---|
| Last layer interpretable, InceptionV1 retrained vs ConvNeXt ("A History of Microscope Errors") | "more like 70%" vs "about 99%" (human reading) | top-20 ≥ 50% one class, + pole: **GoogLeNet 4.7%, ConvNeXt-T 38.4%**, ConvNeXt-S 27.8%. Superclass (K = 60): 17% / 52% / 40%. One reader, n = 24: 42% / 71% / 54% | 0% (all three) | **ordering reproduces** (ConvNeXt ≫ InceptionV1, ×8 by class dominance, ×1.7 by reading); absolute levels do not; proxy is stricter than legibility |
| SAE near-copies ("What Dictionaries Learn") | "On the last layer, 88% of its activity is near-copies of single neurons" | Step-1 SAE: GoogLeNet 0.0%, ConvNeXt-T last MLP 8.8% / 7.7%. **Essay-recipe retrain (3b): ConvNeXt-T last MLP 84.1% (4×, k = 8, FVE 0.77), ConvNeXt-S 75.7%**; GoogLeNet inception5b 0–12% (66% only at k = 128, no control); ConvNeXt-T residual 0.0% | 0% (rotated basis and random init, every layer; the rotated SAE recovers the true neurons at 79–94%; planted neurons 99%) | **reproduces for ConvNeXt's last MLP** (after retraining the SAE); not for GoogLeNet or the residual stream |
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

## 3b. SAE retrained the essay's way (`imnet_sae2.png`, prefix `imnet_sae2`)

The section 3 SAEs were weak (FVE 0.36–0.65; up to 41% of ConvNeXt-T features never fired). We retrained
them with the recipe of Gao et al. (2024, arXiv 2406.04093), which the essay's SAE description points to:
TopK with an **AuxK loss** (features dead for 2M tokens reconstruct the residual x − x̂ with their own top-k_aux,
weight 1/32), normalised MSE, unit-norm decoder with the parallel gradient removed, encoder initialised as the
decoder transpose, b_pre at the geometric median, Adam ε 6.25e-10, gradient clip 1, batch 4,096. Data is
essay scale: **25.6M positions sampled from 9.8M last-layer vectors of 200,455 ImageNet training images**,
6,250 steps; FVE and near-copy share are measured on ImageNet val positions. Widths 4×, 16×, 64×;
k ∈ {2, 8, 32, 64, 128}. 47 runs in all, with a rotated-basis and a random-init control per layer.
Code: `clean_tools.train_topk_sae` (the recipe is keyword options; the defaults are unchanged),
`analysis/imnet_train.py`, `imnet_sae2.py`, `imnet_sae2_fig.py`. Numbers: `imnet_sae2.json`.

Near-copy share of SAE activity (|cos| ≥ 0.9 to a neuron axis, weighted by mean activation, as in section 3):

| layer | 4× k=2 | 4× k=8 | 4× k=32 | 4× k=128 | 16× k=2 | 16× k=8 | 16× k=32 | 16× k=64 | rotated basis (16×, k=2 / 32) | random init (16×, k=2 / 32) |
|---|---|---|---|---|---|---|---|---|---|---|
| ConvNeXt-T last MLP (3072) | 68.9% (FVE 0.52) | **84.1% (FVE 0.77)** | 75.3% (0.94) | 31.3% (0.97) | 31.3% (0.44) | 54.3% (0.70) | 46.2% (0.88) | 24.2% (0.93) | **0.0% / 0.0%** | **0.0% / 0.0%** |
| ConvNeXt-S last MLP (3072) | 10.9% (0.36) | 41.4% (0.51) | **75.7% (0.75)** | – | 4.7% (0.33) | 22.1% (0.47) | 47.4% (0.68) | – | – / 0.0% | – |
| GoogLeNet inception5b (1024) | 0.0% (0.41) | 0.0% (0.60) | 11.6% (0.76) | 66.1% (0.91) | 0.0% (0.41) | 0.0% (0.60) | 9.4% (0.76) | 30.7% (0.83) | 0.0% / 0.0% | 0.0% / 0.1% |
| ConvNeXt-T residual (768) | 0.0% (0.49) | 0.0% (0.63) | 0.0% (0.74) | 0.0% (0.86) | 0.0% (0.49) | 0.0% (0.65) | 0.0% (0.78) | 0.0% (0.82) | 0.0% / 0.0% | 0.0% / 0.0% |
| planted sparse codes on the axes | 99.8% (k=2) | 98.9% (k=8) | | | | | | | rotated plant: 0.0%, 99.98% match the planted directions | |

64× runs (GoogLeNet, ConvNeXt-T residual, k = 2 / 32) give 0.0–0.2%.

- **ConvNeXt's last MLP reproduces the essay's number.** The best ConvNeXt-T SAE puts **84% of its activity on
  near-copies of single neurons at FVE 0.77**; the essay reports 88% at FVE 0.82. ConvNeXt-S reaches 76% at
  FVE 0.75. The share falls when the SAE is much wider (16×) or much denser (k = 128): with spare features the
  SAE splits a neuron into several directions, none of them within 0.9.
- **The controls say the alignment is a property of the network, not of the SAE.** Training the same SAE on the
  same activations written in a random orthonormal basis gives **0% near-copies of the rotated axes**, but
  measured against the true neurons (rotated back) the same SAE has **79% (k = 2) and 94% (k = 32)** of its
  activity on near-copies for ConvNeXt-T, and 83% for ConvNeXt-S. So the SAE finds the neuron directions
  whatever basis it is given. Random-init networks give 0%. In GoogLeNet the rotated SAE finds 14% (k = 32).
- **The section 3 miss came from the SAE, not the network.** Same layer, same near-copy definition: 8.8% with
  the weak SAE, 84% with the essay-scale one.
- **The residual stream stays at 0%** in every setting. The essay's figure anchor is "convnext-sae-residual",
  but its 88% sentence says "the last layer"; the residual stream has no per-coordinate nonlinearity, so a
  preferred basis is not expected there. Our reading: the 88% is about the MLP's neurons.
- **GoogLeNet is mixed.** Near-copies appear only for dense SAEs (k ≥ 32): 66% at 4× k = 128 (FVE 0.91), with a
  flat profile from |cos| 0.8 to 0.95, i.e. features lying almost exactly on single neurons. There is no
  rotated control at k = 128, so we do not count this as a reproduction.
- **Removal test.** Removing every feature that matches no neuron, ConvNeXt-T best run: −0.19 pp (flipped answers
  have median top-1/top-2 margin 0.04 vs 0.69 overall, so they were near-ties); splicing in the whole SAE:
  −0.02 pp. Across the ConvNeXt-T runs: −0.19 to −1.30 pp; ConvNeXt-S: −0.01 to −0.12 pp. This matches the
  essay's wording, but zeroing the whole ConvNeXt-T last MLP costs only −1.1 pp, so the test cannot tell much
  on this block (no ConvNeXt-S block control was run). On GoogLeNet, removal costs −19 to −66 pp against
  −69.7 pp for zeroing the layer; on the ConvNeXt-T residual, −74 to −81 pp against −82.0 pp.
- **Top-20 on training images** (the essay's scans may use 50k training images): GoogLeNet 4.4% / 4.2%
  class-dominated (two disjoint 50k samples; val 4.7%), ConvNeXt-T last MLP 42.0% / 43.2% (val 38.4%),
  ConvNeXt-S last MLP 32.0% / 32.5% (val 27.8%), ConvNeXt-T residual 1.3% (val 1.2%). The choice of images does not
  explain the gap to the essay's ~99%.

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
  activation is positive. Resolved in the next section: on the essay's own 2014 weights the same cutoff
  reproduces every early-layer cell, so the gap was the model.

## The same pipeline on the essay's own InceptionV1 (2014 weights)

Everything above used torchvision's GoogLeNet, a BatchNorm retrain. `imnet_vis.md` found that the essay's
specimen is the **original 2014 InceptionV1 (inception5h) that Lucid ships**, loaded here through lucent
(`from lucent.modelzoo import inceptionv1`). To give every pretrained number a like-for-like value on the essay's
own network, the pipeline was rerun on it unchanged (arch `inceptionv1_lucid`; driver
`analysis/imnet_lucid_pipeline.py`). Numbers sit under that key in `imnet_purity.json`, `imnet_weak.json` and
`imnet_rotation.json`, as runs `l5b_*` in `imnet_sae2.json`, and the per-stage wall time and peak GPU memory in
`imnet_lucid_pipeline.json` (extract 1.8 min, purity 2.0, weak grid 84, SAE dumps 9.3, 12 SAE runs 101, rotation
4.9; peak 5.1 GB; the SAE runs include three 50k-image removal passes each in fp32).

**Wrapper (ours).** `imnet_common.LucidInceptionV1` takes the pipeline's normalised input, converts it to lucent's
scale (RGB in [0, 255] minus 117) and runs the net in fp32 (under fp16 autocast its local response norms overflow
and top-1 falls to 3%), in sub-batches of 50 because lucent's flat forward keeps every intermediate alive. Its
1008 outputs are in the 2012 devkit order; the 1000 standard logits are gathered through a one-to-one map
recovered from val (maximum-weight matching between outputs and ground-truth labels on the vote matrix). The
modal-label map of `imnet_lucid.py` covers only 995 classes; the matched map covers 1000 and differs from it on 5
outputs. Top-1 **69.12%** (modal map 69.13%; essay 70.08%). "The last layer" is **`mixed5b`**: the pre-ReLU concat
of the four branch convs (1x1 | 3x3 | 5x5 | pool_reduce, 1024 channels at 7x7) for the poles, the post-ReLU block
output for the weak grid, SAE and rotation, exactly as for `inception5b`. The random-init control is lucent's
`InceptionV1(pretrained=False)`. The BN recalibration does not apply (no BatchNorm); instead each conv's weight
and bias are scaled by one scalar so its output has unit std on 200 ImageNetV2 images, weight directions
untouched (**ours**; without it the default init's activations vanish by mixed5b).

### Last-layer top-20 purity (`imnet_top20_inceptionv1_lucid.png`, `imnet_top20_inceptionv1_lucid_init.png`)

| last layer | top-1 | + pole: class | + pole: superclass (K=60) | + pole: superclass (K=130) | − pole: class | − pole: superclass (K=60) | random init, + pole (class / super60) | essay (human legibility) |
|---|---|---|---|---|---|---|---|---|
| **InceptionV1 2014 (Lucid), mixed5b** | 69.12% | **2.4%** | 20.9% | 31.8% | 3.6% | 29.6% | 0.0% / 0.0% | ~5% (2017 tools), ~50% (new tools) |
| GoogLeNet inception5b (torchvision) | 69.78% | 4.7% | 17.2% | 26.1% | 2.6% | 18.8% | 0.0% / 0.0% | ~70% (essay's own retrain) |
| ConvNeXt-T last MLP | 82.14% | 38.4% | 51.9% | 58.7% | 0.1% | 1.4% | 0.0% / 0.0% | ~99% |

Mid layer, + pole class / superclass: mixed4e 1.6% / 26.1% (GoogLeNet inception4e 2.3% / 19.2%). The census and
top-1% purity tools give no signal here either (mean best R² 0.03; 3.1% "one job"). The contact sheet shows the
essay's picture of this network: most units mix two or three unrelated things (lifeboat + giant panda, academic
gown + colobus monkey, balloon + white dogs), and one of 24 random units is class-dominated.

### Weak activations (`imnet_weak_grid_inceptionv1_lucid.png`)

Change in top-1 (pp) when activations below each neuron's p-th percentile of positive activations are zeroed, one
layer at a time (cutoff definition of section 5, unchanged). Cells read **ours on the 2014 net (essay's grid for
the same network; ours on torchvision GoogLeNet)**. Base 69.12% (essay 70.08%; GoogLeNet 69.78%).

| layer | p25 | p50 | p75 | p95 | p99 |
|---|---|---|---|---|---|
| conv2d0 | −0.16 (−0.17; −28.88) | −3.73 (−3.50; −63.29) | −23.48 (−22.36; −69.26) | −58.03 (−59.21; −69.64) | −65.49 (−66.60; −69.64) |
| conv2d1 | −0.79 (−0.33; −6.18) | −6.52 (−4.39; −24.40) | −25.27 (−20.94; −47.63) | −58.03 (−58.94; −66.83) | −67.09 (−68.62; −69.24) |
| conv2d2 | −0.07 (−0.10; −2.42) | −0.74 (−0.90; −12.78) | −9.11 (−11.24; −42.33) | −44.30 (−47.02; −67.38) | −61.25 (−64.65; −69.51) |
| mixed3a | −0.11 (+0.02; −0.88) | −0.96 (−0.72; −5.61) | −5.60 (−5.35; −31.22) | −36.92 (−40.62; −68.64) | −60.51 (−64.61; −69.56) |
| mixed3b | −0.02 (+0.01; −0.21) | −0.28 (−0.12; −1.51) | −1.76 (−2.14; −9.19) | −21.35 (−24.35; −57.23) | −50.33 (−54.40; −68.89) |
| mixed4a | −0.03 (−0.02; −0.14) | −0.25 (−0.14; −0.98) | −2.26 (−1.92; −11.00) | −18.36 (−19.65; −68.44) | −44.56 (−49.44; −69.62) |
| mixed4b | −0.00 (−0.05; −0.63) | −0.34 (−0.21; −2.98) | −2.15 (−2.20; −11.84) | −16.80 (−18.27; −55.47) | −40.70 (−43.92; −68.93) |
| mixed4c | −0.04 (−0.09; −0.21) | −0.47 (−0.29; −1.83) | −2.51 (−2.03; −10.77) | −18.78 (−18.87; −58.44) | −41.86 (−42.63; −69.06) |
| mixed4d | +0.01 (−0.04; −0.18) | −0.29 (−0.14; −1.37) | −1.82 (−1.45; −6.50) | −12.92 (−12.51; −38.54) | −32.71 (−33.95; −65.52) |
| mixed4e | +0.01 (−0.03; −0.00) | −0.08 (−0.04; −0.22) | −0.61 (−0.81; −1.86) | −7.72 (−8.23; −17.77) | −24.11 (−25.40; −49.99) |
| mixed5a | −0.05 (+0.01; +0.01) | −0.22 (−0.12; −0.18) | −1.77 (−1.70; −1.72) | −13.42 (−13.87; −14.40) | −32.82 (−34.53; −43.34) |
| **mixed5b** | −0.02 (−0.04; −0.01) | −0.11 (−0.04; −0.06) | **−0.76 (−0.60; −0.49)** | **−7.98 (−7.72; −7.52)** | −29.57 (−33.63; −47.81) |

The essay's own number: "Keeping roughly the strongest quarter of positive activations in Inception's final layer
preserves 69.47% accuracy, versus 70.08% originally" (−0.61 pp). Ours on the same network at p75: **68.36% versus
69.12% (−0.76 pp)**, 9.2% of answers change (GoogLeNet −0.49 pp). Mass-matched pair at mixed5b (ours): the weakest
75% of nonzero activations cost −0.76 pp (9.2% of answers change); the strongest 13.9%, carrying the same total
activation, cost −1.70 pp (12.1%). GoogLeNet: −0.48 vs −6.12 pp.

### SAE near-copies, essay recipe (`imnet_sae2.png`, runs `l5b_*`)

Same recipe and data as section 3b (TopK + AuxK, 25.6M training positions from the 200k ImageNet training images,
FVE and near-copy share on held-out val positions), the GoogLeNet grid minus the 64x runs. Near-copy share of SAE
activity (|cos| ≥ 0.9 to a neuron axis), FVE in brackets.

| last layer | 4× k=2 | 4× k=8 | 4× k=32 | 4× k=128 | 16× k=2 | 16× k=8 | 16× k=32 | 16× k=64 | rotated basis (16×, k=2 / 32) | random init (16×, k=2 / 32) |
|---|---|---|---|---|---|---|---|---|---|---|
| **InceptionV1 2014 (Lucid), mixed5b** | 0.5% (0.32) | 27.4% (0.49) | 79.3% (0.72) | 99.7% (0.96) | 0.2% (0.32) | 21.9% (0.50) | 76.6% (0.72) | **87.5% (0.84)** | **0.0% / 0.0%** (true neurons through the rotation: 0.8% / 79.2%) | **0.0% / 0.0%** |
| GoogLeNet inception5b (torchvision) | 0.0% (0.41) | 0.0% (0.60) | 11.6% (0.76) | 66.1% (0.91) | 0.0% (0.41) | 0.0% (0.60) | 9.4% (0.76) | 30.7% (0.83) | 0.0% / 0.0% (true neurons: 0.0% / 14.3%) | 0.0% / 0.1% |
| ConvNeXt-T last MLP | 68.9% (0.52) | 84.1% (0.77) | 75.3% (0.94) | 31.3% (0.97) | 31.3% (0.44) | 54.3% (0.70) | 46.2% (0.88) | 24.2% (0.93) | 0.0% / 0.0% (true neurons: 79% / 94%) | 0.0% / 0.0% |
| essay, "the last layer" | 88% of activity at FVE 0.82 | | | | | | | | | |

Removal test on the 2014 net (remove every feature with no neuron match, error kept): 16× k=64 **−3.71 pp**
(flipped answers' median top-1/top-2 margin 0.61 against 0.999 over all images; this net's softmax is very
peaked), 4× k=128 **+0.00 pp** (splicing in the whole SAE −0.15 pp; flipped margin 0.06), 4× / 16× k=32 −14.5 /
−16.0 pp, k=8 −52 / −54 pp, k=2 −45 / −47 pp. Zeroing mixed5b sends top-1 to 0% (500-image check). GoogLeNet:
−19 to −61 pp at every k.

### Rotation search

Share of found axes within |cos| ≥ 0.9 of a neuron (mean max |cos| in brackets), 1,000 steps, as in section 4. The
1,024-unit / 3,000-step run was skipped for time.

| layer, unit set | raw, random start | raw, identity start | ICA, random start | random init: raw / identity / ICA | identity-start objective ÷ neuron basis |
|---|---|---|---|---|---|
| **InceptionV1 2014 (Lucid), mixed5b, 3 × 256 units** | **1.3% (0.36)** | 10.2% (0.60) | 0.1% (0.41) | 0% / 69% / 0% | 1.59 |
| InceptionV1 2014 (Lucid), mixed5b, all 1,024 | 0% (0.13) | 0% (0.49) | 0% (0.18) | 0% / 82% / 0% | 7.66 |
| GoogLeNet inception5b, 3 × 256 | 0% (0.26) | 7% (0.66) | 0.1% (0.35) | 0% / 87% / 0% | 2.64 |
| GoogLeNet inception5b, all 1,024 | 0% (0.11) | 37% (–) | 0% (0.16) | 0% / 99% / 0% | – |
| ConvNeXt-T last MLP, 3 × 256 | 76% (0.88) | 94% (0.97) | 94% (0.97) | 0% / 67% / 4% | 1.40 |
| essay | "starting from a random one, you land back on the neurons, in every block" | | | | |

### Reading

On the essay's own network, two of the four pretrained claims that had failed or wobbled on torchvision's
GoogLeNet now reproduce, and two still do not. **The weak-activation grid reproduces cell for cell in all twelve
layers**, including the early layers where torchvision GoogLeNet lost 30–60 pp more than the essay (conv2d0 at p25:
−0.16 vs the essay's −0.17, against −28.9 for GoogLeNet), so that discrepancy was the BatchNorm retrain, not our
cutoff; the essay's headline number (70.08% → 69.47%, −0.61 pp) comes out as 69.12% → 68.36% (−0.76 pp). **The
dictionary claim reproduces too**: a 16× SAE with k = 64 puts 87.5% of its activity on near-copies of single
neurons at FVE 0.84 (essay: 88% at 0.82), the rotated-basis control is at 0% while recovering 79% of the true
neurons through the rotation, and the random-init network is at 0%; GoogLeNet's inception5b never passed 31% at a
comparable FVE. The share depends strongly on k (0.5% at k = 2, 99.7% at k = 128), and the essay's second sentence
("removing every dictionary direction that matches no neuron leaves accuracy unchanged") holds only for the k = 128
SAE, which is nearly a copy of the neuron basis; at k = 64 the removal costs 3.7 pp. **Rotation search still does
not land on the neurons**: as for GoogLeNet, the identity start drifts away (the objective ends 1.6–7.7× above the
neuron basis), so under our fourth-moment objective the 2014 neuron basis is not even a local optimum, and the
essay's objective is not stated. **Top-20 purity is 2.4% class-dominated**, the lowest of the three networks and
below the 4.7% of the retrain, which matches the essay's ordering (2014 net < modern retrain < ConvNeXt) and sits
near its "around 5% with tools of that era". But our yardstick is not the essay's: we count units whose 20
strongest images are at least half one class (or one WordNet superclass, 21% here), while the essay counts units a
human finds legible from feature visualisations plus examples, which is how it gets "more like 50%" on this same
layer with new tools; polysemantic units with two clear themes count for the essay and not for us, and nothing
here measures the 5% → 50% microscope effect. Random-init controls are 0% throughout. Caveats specific to this
run: the class map was recovered from val labels (a near-permutation, 5 ambiguous outputs), the random-init
scaling is ours, and the SAE sweep for this net lacks a 64× width and a rotated control at k = 64 or 128.

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
3. **SAE recipe.** Section 3 used a plain TopK SAE (4×, k ∈ {8, 32}, 4k steps, ImageNetV2 vectors); section 3b
   follows Gao et al. (2024). The essay's stated SAE recipe (640 features, k = 2, 25.6M positions, FVE 0.82) is
   for its squeezed 10-channel mixed3b layer; its 88% sentence names no layer width, k or threshold, so our
   sweep, the near-copy criterion and the share definition are ours. The 0.9 cosine threshold and the
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
The essay's own 2014 InceptionV1: `analysis/imnet_lucid_pipeline.py` (driver; arch `inceptionv1_lucid` in `imnet_common.py`), `imnet_lucid_pipeline.json` (stage times, peak GPU), figures `imnet_top20_inceptionv1_lucid*.png`, `imnet_weak_grid_inceptionv1_lucid.png`.
