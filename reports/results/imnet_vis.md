# imnet_vis: pretrained-ImageNet-network visuals, side by side with the essay's figures

Companion to `imnet.md`. These are the pictures that let a reader compare our pretrained-net results with the
essay's figures by eye (Cammarata 2026, "A History of Microscope Errors", "How I Think About Neurons", appendix
"Feature Visualization in Later Layers"). Code: `analysis/imnet_featvis.py` (feature vis) and
`analysis/imnet_gallery.py` (figures). Numbers: `reports/results/imnet_featvis.json`.

Models and layers are those of `imnet.md`: torchvision GoogLeNet `inception5b` (1024 channels, pre-ReLU concat of
the branch BatchNorm outputs) and ConvNeXt-T `features.7.2.mlp` (3072 hidden units, pwconv1 output pre-GELU).
Note on the GoogLeNet: torchvision's checkpoint is a BatchNorm retrain (69.8% top-1 on our 50k val crops), so it is
closer to the essay's "retrained InceptionV1 with modern regularization (~70% interpretable)" than to the lost
2014 checkpoint the essay puts at ~5% (old tools) / ~50% (fixed tools). Pole scores come from the cached
extraction (`data/imnet_cache/imagenet_val/<arch>_extract.npz`, 50,000 val images): positive pole = spatial max of
the pre-activation, negative pole = minus the spatial min.

## What was done

**Neuron sample** (per model, 20 units): the 8 units with the highest top-20 class purity (modal class count
among the 20 strongest val images, ties broken by the mean top-20 score), plus 12 drawn uniformly at random
with numpy seed 0 from the remaining units. The figures mark the two groups (blue titles = picked, black = random).

| model | picked by purity (unit: class k/20) | random (unit: class k/20) |
|---|---|---|
| GoogLeNet | 748 hourglass 20 · 502 shower cap 17 · 660 hourglass 17 · 717 hourglass 17 · 476 banjo 15 · 600 banjo 14 · 506 hourglass 14 · 520 banjo 14 | 16 black grouse 3 · 41 totem pole 9 · 76 geyser 5 · 177 bow 2 · 271 ringlet 5 · 310 Brittany spaniel 3 · 517 Airedale 5 · 645 mousetrap 4 · 665 hourglass 5 · 832 Norwich terrier 3 · 862 ruddy turnstone 5 · 935 toaster 3 |
| ConvNeXt-T | 312 indigo bunting 20 · 847 bee eater 20 · 3032 rock beauty 20 · 552 parachute 20 · 1510 coucal 20 · 702 plunger 20 · 1634 American coot 20 · 1568 bulbul 20 | 50 Christmas stocking 11 · 125 hummingbird 3 · 230 monitor 3 · 537 gorilla 3 · 827 eel 3 · 945 limpkin 19 · 1566 bath towel 13 · 1952 crossword puzzle 13 · 1996 fireboat 11 · 2497 vine snake 6 · 2603 kimono 2 · 2803 snail 19 |

The random draws agree with the purity numbers of `imnet.md`: 0 of 12 random GoogLeNet units are class-dominated
(>= 10 of 20) against 6 of 12 for ConvNeXt-T (val-set dominance 4.7% vs 38.4%). Picked GoogLeNet units repeat two
classes (hourglass x4, banjo x3); picked ConvNeXt-T units are eight different classes, all 20/20.

**Feature visualization** (`imnet_featvis.py`, 224 x 224, fp32, positive pole, objective = spatial max of the
unit's pre-activation, Adam lr 0.05, 256 steps, FV seed 0, all 20 units of a model optimised as one batch):
- `pixel` ("2009-style"): raw-pixel sigmoid parameterization, plain gradient ascent, no transforms.
- `robust` ("2017"): Fourier parameterization with 1/f scaling and colour decorrelation, random translation
  (+-14 px), scale 0.9-1.1, rotation +-10 deg, 4 samples per step (the `vis_featvis.py` implementation).
- `path` ("path-anchored", OUR formalization of the essay's appendix): `robust` plus, for the first 64 steps,
  `loss = -s(x) - rho * cos(A(x), r_bar)`, with `A(x)` the gradient x activation profile on the previous layer
  (GoogLeNet: the inception5a output; ConvNeXt-T: the residual stream entering block features.7.2), `r_bar` its
  mean over the pole's 60 strongest val images, `rho` the pole's activation range (max - median).
Two units per model (the first picked and the first random) were also run with FV seed 1.
GPU time: GoogLeNet 17 / 42 / 41 s and ConvNeXt-T 25 / 85 / 91 s for pixel / robust / path (plus 6 s of
path targets each), peak memory 5.7 / 6.6 GB. The whole run took about 5 minutes on the RTX 5070 Ti.

**Figures** (`imnet_gallery.py`, 200 dpi):
- `reports/figs/imnet_gallery_<arch>.png` (2404 x 2716 px): the essay's ladybug figure with the microscope
  timeline on the left. One row per unit: the three FVs, then the 20 strongest val images in order; a
  white-on-dark label names the class of every image outside the row's majority class; the row title gives
  "unit N · majority class · k of 20" and the sample role; the right margin gives FV activation / dataset max.
- `reports/figs/imnet_gallery/<arch>/<unit>.jpg` (2404 x 124 px, JPEG quality 92): each row on its own.
- `reports/figs/imnet_fv_seeds_<arch>.png` (656 x 274 px): seed stability, 2 units x 3 settings x 2 seeds.
- `reports/figs/imnet_negpoles_<arch>.png` (1528 x 2020 px): the essay's "+Taxis / -Ambulances" figure: the 6
  units where both poles are most class-dominated (max over units of min(purity+, purity-) on the top 20 of each
  pole), 8 strongest images per pole.
- `reports/figs/imnet_shapes_<arch>.png` (1399 x 1070 px): claim D, the raw shape of the positive pole over the
  50,000 images (score / max vs image rank) for the 12 random units, with the percentile-rank version dashed.

## FV objective vs dataset max

FV activation divided by the unit's largest pole score over the 50,000 val images (median and IQR over the 20
units; per unit in the json and in each gallery row's right margin):

| model | 2009-style (pixel) | 2017 (robust) | path-anchored (ours) |
|---|---|---|---|
| GoogLeNet inception5b | 7.97 (6.7-9.4) | 3.68 (3.3-4.6) | 3.69 (3.1-4.6) |
| ConvNeXt-T features.7.2.mlp | 5.98 (5.2-8.2) | 3.83 (3.3-5.4) | 3.66 (3.1-5.1) |

Every setting drives the unit well past anything a natural image reaches. The pixel setting does it with an image
that is visually pure noise (the adversarial route); the Fourier + transformation-robust settings reach about half
as far with images that have structure. Purity-picked units sit lower (GoogLeNet picked 1.7-4.5x, ConvNeXt-T
picked 2.4-3.9x under `robust`) than random units (3.2-5.8x and 3.2-5.7x): the clean units are harder to over-drive.
Seed stability: the objective moves by at most 0.4x between seeds 0 and 1 (e.g. GoogLeNet 748: robust 1.68 / 2.09,
path 2.07 / 2.24; ConvNeXt-T 312: robust 2.43 / 2.55, path 2.57 / 2.26), and the two seeds look alike in style
but not in layout (`imnet_fv_seeds_*.png`).

## Observations (from looking at the figures)

1. **GoogLeNet inception5b neurons look like mixtures, as the essay's "inception-last-layer" figure shows.** Among
   the 12 random units the best majority is 9 of 20 (unit 41, totem pole, whose other 11 images are radiators,
   mailboxes, a sea lion, a banana and a pickup) and the median is 4; rows like unit 177 ("bow", 2 of 20: wallaby,
   plow, hog, horned viper, moped, hatchet, water bottle, assault rifle) have no visible theme. Even the purity-
   picked units mix: the banjo units take in drums, shower caps and a keyboard, and three hourglass units share the
   same handful of hourglass photos. Only 6 of the 1024 units (and 2 of those 6 are the same hourglass photos) reach
   14/20 or more.
2. **ConvNeXt-T's purity-picked units look like single concepts, and about half the random ones do.** The 8
   picked units are 20/20 (indigo bunting, bee eater, rock beauty, parachute, coucal, plunger, coot, bulbul) with
   varied poses and backgrounds, like the essay's ladybug row. Of the 12 random units, 6 are class-dominated
   (snail 19, limpkin 19, bath towel 13, crossword puzzle 13, fireboat 11, Christmas stocking 11) and several of
   the rest have a readable non-class theme: unit 1566 is "cloth wrapped around a face" (bath towel, cowboy hat,
   bib, nipple), unit 2603 is "white robes and uniforms" (kimono, lab coat, abaya, cloak, pajama), unit 2497 is
   "green snake on a branch" (vine snake, green mamba, green snake), unit 827 is "slug-like shapes". Units 125, 230
   and 537 (3/20 each) look like mixtures. So the by-eye picture is well above the 38% class-dominance statistic and
   well below the essay's 99% for its ConvNeXt; our ConvNeXt-T is a different training run from the essay's.
3. **Our 2017-style and path-anchored FVs show the GoogLeNet units as textured fragments, not objects, and the
   ConvNeXt-T FVs are dense ornamental texture.** The 2009-style images are noise in every case. The `robust`
   GoogLeNet images have object-like pieces (unit 502, shower cap: a rounded cap-like form and skin tones; unit 41,
   totem pole: a carved vertical form) but none reads as cleanly as the essay's 2017 column. For ConvNeXt-T the
   `robust` images are busy, high-frequency, multi-coloured texture with no single object in any of the 20 units,
   even for the 20/20 bird units. This is a microscope failure on our side, not evidence against the neurons: the
   dataset examples of the same units are unambiguous. The essay's 2026 settings include regularization tweaks we
   have not reproduced (its exact recipe is not given), and Lucid-era settings tuned on InceptionV1 are known to
   transfer poorly to LayerNorm / GELU / 7x7-depthwise networks.
4. **Path anchoring changes the layout of the FV but not its legibility.** Side by side, `path` and `robust` images
   of the same unit have the same palette and texture vocabulary and a different arrangement, with the same
   objective ratio (3.69 vs 3.68 median for GoogLeNet, 3.66 vs 3.83 for ConvNeXt-T); for several units (GoogLeNet
   41, 748; ConvNeXt-T 1634, 702) the path image has a larger central form. Nothing becomes recognisable that was
   not already so. The 64-step anchor is cheap (the same wall time as `robust`) but, in this implementation, is not
   the missing ingredient.
5. **Negative poles: "related but distinct" holds for ConvNeXt-T, not for GoogLeNet.** ConvNeXt-T's best
   two-sided units read like the essay's examples: +racer / -minibus (18 and 10 of 20), +Madagascar cat (ring-
   tailed lemur) / -malinois and other dogs, +Persian cat / -Rhodesian ridgeback and hounds, +book jacket /
   -Sealyham terrier, +stupa / -dowitcher (a shorebird; the + side also admits oystercatchers), and +bow tie / -bow
   tie, where the negative pole is bow ties on clothing in street scenes and the positive is formal portraits. On
   GoogLeNet the same selection gives +banjo / -harp (14, 10), +hourglass / -sax, +cowboy hat / -hourglass,
   +tennis ball / -hourglass: the negative poles are class-dominated at 9-11 of 20 only because the same three
   hourglass and harp photo sets keep recurring, and the pairs are not semantically related. Over the whole layer
   only 2.6% of GoogLeNet negative poles are class-dominated (imnet.md), so this figure shows its 6 best cases.
6. **Raw activation shapes (claim D).** Every unit of both models is a sharp spike on the raw scale: the top 1% of
   images carry the top half of the range and the curve is flat over the remaining 49,000, while the percentile
   version is a diagonal by construction. The ConvNeXt-T pre-GELU scores go negative for most images (the median is
   -1.2 to -6.9 against maxima of 9-26), so "score / max" shows the pole as a detector with a long inactive tail;
   the GoogLeNet BN outputs have a shorter negative tail. Among the 12 random units the shapes do not separate the
   class-dominated units from the mixtures (snail 19/20 and kimono 2/20 look alike), so the shape alone is not a
   purity signal here; the point of the figure is that percentile ranks would erase the spike in all of them.

## Files

- `analysis/imnet_featvis.py`, `analysis/imnet_gallery.py`
- `reports/results/imnet_featvis.json` (per unit: role, majority class and count, dataset max and median, rho,
  FV objective and objective / dataset max per setting and seed; per arch: the negative-pole selection)
- `reports/figs/imnet_fv/<arch>/<unit>_<setting>[_s1].png` (66 per arch, 224 px)
- `reports/figs/imnet_gallery_{googlenet,convnext_tiny}.png`, `reports/figs/imnet_gallery/<arch>/<unit>.jpg`,
  `reports/figs/imnet_fv_seeds_*.png`, `reports/figs/imnet_negpoles_*.png`, `reports/figs/imnet_shapes_*.png`
