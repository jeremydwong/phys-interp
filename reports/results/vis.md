# Results: the "already clean" ladder in miniature (vision, prefix `vis_`)

Do the "small number of fixed improvements" in Cammarata's essay *Neural Networks Are Already Clean*
(https://already-clean.goodfire.com/) make neurons measurably more interpretable? The essay's headline
example is InceptionV1's last layer: "I put it around 5% with tools of that era ... With better
interpretability tools (fixed feature visualization, methods for visualizing attribution etc) it's more like
50% on the same network ... Retraining the same architecture with more reasonable modern regularization it's
more like 70%. In the ConvNeXt of the same size, it's about 99%." ("A History of Microscope Errors").
We rebuilt the three steps (tools, recipe, architecture) on small vision networks, measured them against
ground truth and against random-init controls, and also checked four of the essay's side claims (weak
activations, raw shape vs percentiles, dictionaries vs neurons, rotation search).

**Short answer.** Only the *tools* step reproduces, and it reproduces clearly on both datasets: on CIFAR-10
the share of last-layer poles our pipeline reads correctly goes from **8% ± 6** (2017-era tools) to
**44% ± 5** (fixed tools) on the same weights; on the shapes set from 25% ± 4 to 67% ± 17. Almost all of the
gain comes from transformation-robust, decorrelated feature visualization. The *recipe* step does not help
(CIFAR 41% ± 13, shapes 54% ± 13), and the census against ground truth gets *worse* (CIFAR class-clean
poles 9% → 1%). The *architecture* step does not reproduce: the parameter-matched ConvNeXt is no better on
shapes (68% ± 5) and much worse on CIFAR (8% ± 5). The weak-activation claim reproduces best in the
ConvNeXt; the SAE near-copy and rotation-search claims do not reproduce at all in our small nets. All
numbers are means over 3 training seeds with 95% t-intervals. These are small models (1.2M parameters) on
small data, trained for minutes; see "Where this does not reproduce the essay" before generalising.

Every threshold below that the essay does not state is ours and is marked **(ours)**.

## 1. Setup

**Datasets.**
- *Shapes* (`sim/vis_shapes.py`, regenerate with `uv run python sim/vis_shapes.py`, 18 s): 50,000 procedurally
  rendered 64×64 images (40k train, 10k test; image *i* uses seed 1,000,000 + *i*) of one object on a grey
  background (`reports/figs/vis_shapes_examples.png`). Ground-truth factors: shape (circle, square, triangle,
  arc, cross, bar; circle/arc are the curved ones), hue (6), texture (solid, stripes at 0/45/90/135°, dots),
  background type (plain/gradient/noise), size, x, y, rotation phase, background luminance, plus the derived
  groupings curved, warm, textured. Class (24) = shape × warm/cool × solid/textured, so a classifier must
  read shape, hue and texture; size, position, rotation and background are nuisances.
- *CIFAR-10* (torchvision), 50k/10k. The only ground truth is the class.

**Models** (`models/vis_models.py`, `models/vis_train.py`), all ≈1.18M parameters:

| step | architecture | recipe | shapes test acc | CIFAR test acc (train acc) |
|---|---|---|---|---|
| M_old | plain VGG-style CNN: 8 × (conv3×3-BN-ReLU), max-pool between 4 stages (32/64/128/256 ch), GAP, linear | **old**: SGD+momentum 0.9, lr 0.05, coupled L2 5e-4, step decay ×0.1 at 50/75%, crop+flip only | 99.99% | 91.2% ± 0.6 (98.7%) |
| M_recipe | same | **modern**: AdamW 2e-3, wd 0.05, warmup+cosine, label smoothing 0.1, mixup 0.8 / cutmix 1.0, RandAugment-lite (affine + brightness/contrast, no hue), random erasing 0.25, EMA 0.999 | 99.96% | 89.5% ± 0.2 (93.0%) |
| M_arch | ConvNeXt-style: dw7×7 → LN → Linear(4C) → GELU → Linear(C) → layer scale → residual; stages 32/64/128/256 ch with 1/1/3/1 blocks | modern | 100% | 86.9% ± 1.0 (91.9%) |

Shapes: 40 epochs; CIFAR: 60 epochs; 3 seeds each (0, 1, 2). Every model has a random-init control: the
saved initial weights of the same seed, analysed identically. The essay "never specifies the regularization
recipe" (cammarata_math.md §2); ours is the ConvNeXt-paper recipe scaled down **(ours)**.

**What a "neuron" is.** Plain CNN: each conv channel's BN output (pre-ReLU). ConvNeXt: each block's MLP hidden
unit before GELU (the elementwise, privileged basis; 4× wider than the residual stream). The essay does not
say which ConvNeXt tensor it calls a neuron (cammarata_math.md §1.4). We also report the ConvNeXt's final
residual-stream channels as a secondary view. "Late" = the last neuron layer (M_old `s4c2`, 256 units at
4×4; M_arch `s4b1`, 1024 units at 4×4); "mid" = last layer of stage 3; "early" = last layer of stage 1.

## 2. Measurements (`analysis/clean_tools.py`, reusable)

- **Poles.** Positive pole score = spatial max of the pre-activation per image; negative pole score = −(spatial
  min). Essay: "Each neuron's 20 strongest responses out of 50,000 images"; the negative pole is the argmin set.
- **Census (ground truth).** Per pole, R² of the score against each factor. Continuous factor: `binned_r2`
  (30 quantile bins; `clean_tools.eta2` reproduces `analysis.common.binned_r2` exactly). Categorical factor:
  the best one-vs-rest indicator R² ("the unit detects value *v*") **(ours)**: a plain 24-way lookup would
  call any mixture of classes "clean"; in a first pass it labelled 100% of M_old's late units clean.
  Clean ≥ 0.5, partial 0.25–0.5, else unexplained (repo convention). Conjunction: best pair (2×2 indicator
  table, indicator × 6 bins, or 6×6 bins) beats the best single by > 0.15 (repo convention).
  *Clean on a task factor* = clean and the best factor is shape/hue/texture or their groupings/class
  (shapes), or the class (CIFAR).
- **One main job (strong activations)** **(ours)**: the pole's strong set is either the top 1% by rank
  (percentile view) or every image above half the pole's raw range, (s − median) ≥ ½ (max − median),
  clipped to [20, 1000] images (raw-shape view). For every factor (continuous ones in quintiles) the modal
  value's share *p*, chance-corrected against the factor's largest marginal share *p₀*: (p − p₀)/(1 − p₀).
  One main job if the best corrected purity ≥ 0.5. On CIFAR this is the essay's last-layer measure, class
  purity of the strongest images.
- **Feature visualization** (`analysis/vis_featvis.py`), on 48 random late units per model, both poles, 3 FV
  seeds, 256 Adam steps at lr 0.05; objective = the pole score:
  *pixel* ("2009"): sigmoid pixels, no transforms. *robust* ("2017"): Lucid's 1/f Fourier parameterization +
  colour decorrelation (as in `analysis/video_featvis.py`) + 4 random translate/scale/rotate copies per step.
  *path*: robust + for the first 64 steps a term ρ·cos(A(x), r̄), where A is gradient × activation on the
  previous neuron layer, r̄ its mean over the pole's 60 strongest test images and ρ the pole's activation range
  (essay appendix "Feature Visualization in Later Layers": "traced the paths they take, and gave the
  optimization an incentive to follow those paths, often only for the first 64 steps"; "Path-guided
  visualizations use the strongest 60 examples"; the exact term is our formalization, cammarata_math.md §3.5).
- **FV legibility** **(ours)**: an independently trained readout (`--step readout`, a small ResNet; shapes:
  multi-head factor readout, 100% on shape/hue/texture/background, R² ≥ 0.996 on size/x/y/luminance; CIFAR:
  classifier, 91.9%) reads the final FV image. A pole's FV is legible if the readout's value of the pole's
  job factor equals the job's modal value (quintile for continuous factors; rotation phase has no head and is
  never legible). Also reported: FV activation / dataset max, seed agreement.
- **Tool verdict** **(ours)**, per late-layer pole, FV seed 0:
  *2017-era tools* = positive pole only, percentile (top-1%) purity, pixel FV; a pole counts if its strong set
  has one main job **and** its FV is legible for that job; negative poles are never inspected, so they count
  as not read. *Fixed tools* = both poles, raw-shape purity, path-anchored FV.
- **Raw shape** **(ours)**: a clean pole is a *dial* if its best factor still explains ≥ half as much variance
  among the weak images (0 < r ≤ ½, r = (s − median)/(max − median)) as overall, else a *detector*. We also
  rerun the late census on each unit's percentile ranks instead of raw values.
- **Weak vs strong removal** (`analysis/vis_ablate.py`) **(ours)**: per neuron and every spatial position,
  post-activation |a| below the 75th percentile of its nonzero activations (on 1,000 training images) is
  zeroed ("weakest 75%"); separately, everything above a threshold chosen so the removed activation mass
  equals the weak mask's mass ("strongest, same total activation"). Masks in every neuron layer at once.
- **Dictionaries** (`analysis/vis_dict.py`) **(ours)**: TopK SAE (4× features, k = d/32, 4,000 steps) on
  last-layer post-activation vectors at every position of 6,250 training images (100k vectors). Near copy:
  max |cos(decoder, neuron axis)| ≥ 0.9; near-copy share = share of mean SAE activity. Controls: the same
  activations randomly rotated; random init. Splice: replace the layer by the reconstruction from all /
  near-copy-only / other features.
- **Rotation search** (`analysis/vis_rotation.py`) **(ours)**: on 3 random 256-unit subsets, maximise
  E Σᵢ (Rx)ᵢ⁴ over orthogonal R from a random rotation, (a) on centred raw activations, (b) after ZCA whitening
  (kurtosis ICA; patterns compared to neuron axes). On synthetic sparse data both recover the neuron basis
  (100% of axes at |cos| ≥ 0.9) and fail on a rotated basis (0%), so the tools work when the answer is yes.

## 3. The ladder (headline, `reports/figs/vis_ladder.png`)

Last neuron layer, share of poles (both signs; for 2017-era tools the negative poles count as unread). Mean
± 95% CI over 3 seeds; random-init control in brackets.

| step | shapes: read correctly | shapes: clean on task factor | CIFAR: read correctly | CIFAR: class-clean |
|---|---|---|---|---|
| M_old, 2017-era tools | 25 ± 4 [8 ± 24] | 23 ± 10 [0] | **8 ± 6** [0] | 10 ± 4 [0] |
| M_old, fixed tools | **67 ± 17** [39 ± 31] | 28 ± 19 [0] | **44 ± 5** [0] | 9 ± 2 [0] |
| M_recipe, fixed tools | 54 ± 13 [38 ± 35] | 8 ± 2 [0] | 41 ± 13 [0] | 1 ± 1 [0] |
| M_arch, fixed tools | 68 ± 5 [15 ± 16] | 27 ± 7 [0] | 8 ± 5 [0] | 5 ± 2 [0] |

Positive poles only (the essay counts neurons; most of what we read is on the positive side):

| step | shapes | CIFAR |
|---|---|---|
| M_old, 2017-era tools | 49 ± 8 | 16 ± 13 |
| M_old, 2017 FV but otherwise old tools | 58 ± 5 | 73 ± 14 |
| M_old, fixed tools | 85 ± 15 | 69 ± 14 |
| M_recipe, fixed tools | 50 ± 18 | 39 ± 20 |
| M_arch, fixed tools | 66 ± 8 | 13 ± 5 |

Reading the tables:
- **Tools: +36 to +42 points, reproduces.** On CIFAR the jump is entirely in feature visualization: pixel FVs
  of M_old's late units are legible for 16% ± 8 of poles, robust FVs for 57% ± 5, path-anchored for 62% ± 6
  (`vis_fv_legibility.png`; random-init: 1/15/16%). Swapping only the FV (positive poles, percentile purity)
  already moves 16% → 73%. Negative poles add little on CIFAR because they rarely have one job (28% ± 2 of
  M_old's negative poles vs 87% ± 2 of positive). Path anchoring adds +5 ± 7 on M_old and +18 on M_recipe
  over robust FV, in the direction the essay describes but not separable from seed noise on M_old.
- **On shapes the tools step is real but the control is high.** Random-init plain CNNs have many
  luminance/background detectors that a robust FV renders correctly (39% ± 31 read, all of them on nuisance
  factors: `vis_jobs.png`). Trained late units are clean only on task factors and random-init late units only
  on nuisances, so the census column (task factors) is the cleaner comparison there.
- **Recipe: no gain, the census gets worse.** M_recipe reads about the same (CIFAR 41 vs 44) and its
  class-clean share collapses (CIFAR 9% → 1%, shapes 28% → 8%). Its positive poles are less often single-class
  (CIFAR one-main-job 71% ± 8 vs 87% ± 2), but its negative poles are much more often single-class (55% ± 7 vs
  28% ± 2). So the modern recipe moves meaning onto the negative side rather than removing it; the
  percentile/pixel tools never see that side.
- **Architecture: does not reproduce.** The ConvNeXt's late MLP neurons are about as readable as M_old's on
  shapes and much less on CIFAR: one-main-job 66% ± 1 (positive) and 20% ± 16 (negative), and even robust FVs
  are rarely legible (14% ± 4). Its final residual-stream channels are worse still (class-clean 1%,
  one-main-job 45%).

## 4. Depth, jobs and raw shape (`vis_census_depth.png`, `vis_jobs.png`, `vis_dial.png`)

- On shapes, early layers are the cleanest (M_old 55% ± 25 of poles clean, mostly hue; random-init 17%, mostly
  luminance). Mid layers are the least clean in every model. On CIFAR nothing below the last layer is clean
  against the class (expected: the class is the only factor we have), so CIFAR depth trends are uninformative.
- Trained clean poles encode task factors; random-init clean poles encode nuisances (luminance, background).
  The raw "clean" share alone would have said random-init ConvNeXts are cleaner than trained ones on shapes
  (41% vs 27% late), which is why the control matters.
- **Percentiles vs raw values.** Rerunning the late census on percentile ranks instead of raw activations
  removes almost all clean poles: shapes M_old 43% → 0.5%, CIFAR M_old 17% → 0%, CIFAR M_arch 9% → 0%. A late
  unit's job lives in the top of its raw range; percentiles flatten it. This supports "Percentiles flatten
  that" (essay, "Percentiles or raw activations?").
- **Dials vs detectors.** Our raw-shape split puts most clean late positive poles on CIFAR M_old among dials
  (99%), because a class indicator still explains the weak range (a class unit is graded, not a step), while
  shapes M_old late positive poles are mostly detectors (23% dials). Early shapes poles split by sign
  (positive 17% dials, negative 76%). We do not see the essay's simple "early = dials, late = detectors"
  pattern; our dial/detector rule is a crude formalization.

## 5. Weak activations (`vis_weak.png`)

Every neuron layer at once, test accuracy:

| | intact | weakest 75% removed | strongest, same mass, removed (share of activations) |
|---|---|---|---|
| shapes M_old | 100 | 54 ± 24 | 5 ± 5 (22%) |
| shapes M_recipe | 100 | 88 ± 10 | 5 ± 1 (18%) |
| shapes M_arch | 100 | 85 ± 28 | 22 ± 9 (5%) |
| CIFAR M_old | 91.2 | 80.4 ± 1.5 | 10.2 ± 1.6 (18%) |
| CIFAR M_recipe | 89.5 | 69.9 ± 5.6 | 14.5 ± 6.3 (18%) |
| CIFAR M_arch | 86.9 | **85.0 ± 2.7** (4.9% of answers change) | **63.6 ± 9.8 (2.2%)** |

Essay (medium ConvNeXt, all 18 blocks): weakest 75% 72 → 71; strongest 7% 72 → 4. The asymmetry reproduces
in every model (strong activations matter far more per unit of mass), and the ConvNeXt is closest to the
essay's "almost no cost" for weak activations (−1.9 points). But in the plain CNNs weak activations do
matter (−11 to −20 points on CIFAR, −46 on shapes M_old), and the ConvNeXt's activations are so heavy-tailed
that matching mass removes only 2.2% of them, so the strong mask is smaller and hurts less than in the essay.
Masking only the last layer changes nothing in any model (≤ 1.5 points).

## 6. Dictionaries and rotations (`vis_dict.png`)

Essay: "On the last layer, 88% of its activity is near-copies of single neurons, and the rest is faint:
removing every dictionary direction that matches no neuron leaves accuracy unchanged"; a rotation search
from a random start lands "back on the neurons, in every block" ("What Dictionaries Learn").

| last layer | SAE FVE | near-copy share of SAE activity | splice: all features | near-copy features only | other features only | ICA search: mean max \|cos\| to a neuron (random-init) |
|---|---|---|---|---|---|---|
| shapes M_old | 0.99 | **0.0%** (rotated basis 0.0%) | 100% | 4.0% | 100% | 0.25 (0.25) |
| shapes M_recipe | 0.96 | 0.0% | 99.9% | 4.3% | 99.9% | 0.27 (0.25) |
| shapes M_arch | 0.99 | 0.9% ± 1.9 | 100% | 76% ± 38 | 100% | **0.43 ± 0.04** (0.31) |
| CIFAR M_old | CIFAR_OLD_FVE | CIFAR_OLD_NC | CIFAR_OLD_ALL | CIFAR_OLD_NCO | CIFAR_OLD_OTH | CIFAR_OLD_ICA |
| CIFAR M_recipe | CIFAR_REC_FVE | CIFAR_REC_NC | CIFAR_REC_ALL | CIFAR_REC_NCO | CIFAR_REC_OTH | CIFAR_REC_ICA |
| CIFAR M_arch | CIFAR_ARCH_FVE | CIFAR_ARCH_NC | CIFAR_ARCH_ALL | CIFAR_ARCH_NCO | CIFAR_ARCH_OTH | CIFAR_ARCH_ICA |

(Shapes: 3 seeds. CIFAR: CIFAR_DICT_N. A random direction in 256-d has mean max |cos| 0.19 to the axes.)

- **Near copies: does not reproduce.** In every model almost no SAE feature is within |cos| 0.9 of a neuron;
  the decoder-to-neuron cosines peak at 0.2–0.3 (`vis_dict.png`, top). Spliced back in, the SAE reconstructs
  the layer well enough to keep accuracy, and essentially all of that comes from features that are *not*
  neuron copies. The tool itself works: on synthetic data whose sources are the neurons, the same SAE gives a
  99.99% near-copy share. "Near-copy features only" leaves just the SAE's bias (the mean activation); for the
  ConvNeXt that still gives 76–82% accuracy because its last MLP block only writes a delta onto a residual
  stream that already carries the class.
- **Rotation search.** On raw activations the L4 search never returns to the neurons (mean max |cos| 0.20,
  = random): the objective is dominated by the few highest-variance directions. After whitening (ICA) the
  plain CNNs still land on random-looking directions (0.23–0.27, the same as their random init), but the
  **ConvNeXt's** found axes are clearly pulled toward its neurons (shapes 0.43 ± 0.04 vs 0.31 at random init;
  CIFAR CIFAR_ARCH_ICA_SHORT), though only CIFAR_ARCH_ICA_AL of axes reach |cos| ≥ 0.9. This is the one
  dictionary-side measurement where the architecture step points the essay's way. Part of the pull comes from
  the GELU itself (random-init ConvNeXts are already above the 0.19 random baseline).

## 7. Size sweep (`vis_sweep.png`)

RESULTS_SWEEP

## 8. Galleries (`vis_gallery_shapes.png`, `vis_gallery_cifar.png`)

Two late units per model (seed 0), the highest-purity positive pole of distinct jobs, each with both poles: top-5
test images by raw score, then pixel / robust / path-anchored FVs (seed 0). On CIFAR, M_old and M_recipe
units show recognisable airplane and bird FVs once robust FV is on; pixel FVs are noise. ConvNeXt FVs stay
mushy on CIFAR. On shapes, robust FVs show stripes and colour patches matching the unit's job, but few look like
the object.

## Where this does not reproduce the essay

1. **Recipe and architecture steps.** Neither helps in our setting; on CIFAR the ConvNeXt is much worse. Likely
   contributors, none tested: (a) scale: 1.2M-parameter models, 32×32 / 64×64 images, 10/24 classes, versus
   ImageNet; the shapes task saturates at ~100% for every model, so there is little pressure to organise;
   (b) the modern recipe is under-trained at 60 CIFAR epochs (train accuracy 93% vs 99% for the old recipe; a
   180-epoch run reached 90.9% test at epoch 100 before it was killed by a GPU out-of-memory error from a
   shared-GPU job, so no long-recipe checkpoint was analysed); (c) our ConvNeXt "neuron" choice (MLP hidden
   units, 4× wider, each writing a delta to the residual stream) may not be the essay's; (d) mixup/cutmix
   train on blended images, which plausibly spreads class evidence across units.
2. **SAE near-copies and rotation search** do not reproduce (section 6).
3. **One-main-job purity saturates on shapes** for trained and random-init plain CNNs alike (88–99%), so on
   shapes it does not discriminate; on CIFAR random init is 0% and the measure works.
4. Our verdict needs both a pure strong set and a legible FV judged by a readout. That is stricter than a
   human reading a gallery and different from the essay's agent pipeline; absolute percentages are not
   comparable to the essay's 5/50/70/99, only the differences between steps are.

## How to rerun

```
uv run python sim/vis_shapes.py                                    # 18 s
# 3 seeds x {old, recipe, arch} x {shapes 40 ep, cifar 60 ep}; ~1-17 min each on the 5070 Ti (shared GPU)
uv run python models/vis_train.py --data shapes --step old --seed 0 --epochs 40
uv run python models/vis_train.py --data cifar  --step arch --seed 2 --epochs 60
uv run python models/vis_train.py --data shapes --step readout --epochs 30   # FV judge
uv run python models/vis_train.py --data cifar  --step readout --epochs 40
uv run python models/vis_train.py --data cifar --step arch --size l --seed 0 --epochs 60   # size sweep (s|l)
uv run python analysis/vis_run_all.py      # census, ablation, SAE, rotation, FV, figures, vis.json
```
Analyses skip models already in their JSON unless `VIS_RECOMPUTE=1`; `VIS_DATASETS=shapes` restricts the
datasets. Checkpoints are in `models/vis_ckpt/` (weights not in git), caches in `data/vis_cache/`.
Numbers: `reports/results/vis.json` (ladder table, sweep) and the per-analysis `vis_census.json`,
`vis_featvis.json`, `vis_ablate.json`, `vis_dict.json`, `vis_rotation.json`.

## Porting the measurements (e.g. to Pluto)

`analysis/clean_tools.py` has no image or dataset code: `poles`, `census` / `census_summary`, `purity` /
`strong_set`, `dial_detector`, `rank_transform`, `weak_strong_thresholds` + `mask_fn` (an edit hook on a
post-activation tensor), `train_topk_sae` + `near_copy_share`, `random_rotation` + `rotation_search`. Inputs
are pole scores [N, U] and factor dicts {name: values} with kinds "cat" / "cont". For a token or unit model,
pass per-example (or per-timestep) maxima and minima of each neuron's pre-activation as the poles.
