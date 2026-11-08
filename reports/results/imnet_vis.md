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
  timeline on the left. One row per unit: the four FVs (2009-style, 2017, path-anchored, tuned), then the 20 strongest val images in order; a
  white-on-dark label names the class of every image outside the row's majority class; the row title gives
  "unit N · majority class · k of 20" and the sample role; the right margin gives FV activation / dataset max.
- `reports/figs/imnet_gallery/<arch>/<unit>.jpg` (2404 x 124 px, JPEG quality 92): each row on its own.
- `reports/figs/imnet_fv_seeds_<arch>.png` (880 x 274 px): seed stability, 2 units x 4 settings x 2 seeds.
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
   transfer poorly to LayerNorm / GELU / 7x7-depthwise networks. **Update: see "Tuning the microscope" below;
   the main fault was our objective (spatial max), and on ConvNeXt-T the missing ingredient is a magnitude
   constraint.**
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

## Tuning the microscope (`analysis/imnet_featvis2.py`, figures `imnet_fv2_<arch>.jpg`)

Observation 3 above said our 2017-style renders were textured fragments where the essay's are objects. The
tuning sweep asks which of our settings is to blame, on 6 units per model (GoogLeNet 748 hourglass, 502 shower
cap, 476 banjo, 41 totem pole, 177 bow, 16 black grouse; ConvNeXt-T 312 indigo bunting, 702 plunger, 552
parachute, 945 limpkin, 50 Christmas stocking, 2803 snail), with one objective yardstick so the choice is not
made by eye: **the other pretrained net reads each render** (ConvNeXt-T judges GoogLeNet's renders and vice
versa) and we record the rank of the unit's top-20 majority class in its prediction. A render "reads as the unit"
if that independent judge ranks the class in its top 5. The unit's own net is also recorded but is not
independent (a render optimised against a net is adversarial to it: the own-net rank is 1 for almost every
ConvNeXt-T render, whatever it looks like). The selection rule, fixed before the renders were viewed: the
variant with the lowest geometric-mean independent rank over the 6 units.

| variant | what changes | GoogLeNet: gm rank, top-5 | ConvNeXt-T: gm rank, top-5 |
|---|---|---|---|
| robust (our "2017") | objective = spatial **max** of the pre-activation, 4 samples, 256 steps | 202, 0/6 | 79, 1/6 |
| mean | objective = spatial **mean** (Lucid's channel objective), Lucid transforms (jitter 16, scale 0.9-1.1, rotate 10), 8 samples, 512 steps | 34, 2/6 | 61, 0/6 |
| center | objective = centre cell of the 7x7 map (Lucid's neuron objective) | 83, 0/6 | 73, 2/6 |
| center_tv | center + total-variation penalty (0.3 x dataset max per unit of mean abs gradient) | 104, 1/6 | 96, 0/6 |
| mean_lowf | mean with Fourier scale 1/f^1.5 and larger transforms (jitter 24, scale 0.75-1.25) | **34, 2/6** | 34, 2/6 |
| maco | magnitude-constrained (Fel et al. 2023): Fourier magnitude fixed to the mean spectrum of 2,000 val images, only the phase is optimised; 6 random crops (40-60% of a 448 px canvas) + noise 0.05 per step; NAdam lr 1.0, 512 steps | 41, 2/6 | 10, 2/6 |
| maco_path | maco + our path anchor for the first 64 steps | 53, 2/6 | 8, 3/6 |
| maco_long | maco, 1,000 steps | 43, 2/6 | 13, 1/6 |
| maco_lowf | maco_long with an extra 1/sqrt(f) on the magnitude and a grey start (one phase shared by the colour channels + 0.1 rad noise) | 90, 0/6 | **4.3, 3/6** |
| maco_dc | maco_lowf with the phase optimised in Lucid's decorrelated colour space | 36, 2/6 | 8, 3/6 |
| maco_dc1 | maco_dc with 1/f instead of 1/sqrt(f) | 177, 1/6 | 9, 2/6 |

(gm rank = geometric mean over the 6 units of the independent judge's rank of the majority class; top-5 = units
whose render it ranks in the top 5. Per-unit ranks, objective values and times are in `imnet_featvis2.json`;
every render is in `reports/figs/imnet_fv2/<arch>/` and on the contact sheets `imnet_fv2_<arch>.jpg`, where each
cell is labelled "independent rank / own rank".)

**What was wrong.** Mostly the objective. We optimised the spatial *max* of the unit's map, which is the pole
score the purity statistics use, but as a feature-vis objective it rewards one spiky position and leaves the
rest of the image free to be noise. Lucid's channel objective (the spatial mean) alone takes GoogLeNet's
shower-cap unit from rank 42 to rank 1 and the banjo unit from 490 to 5, and the renders show caps on heads and a
figure holding a banjo. A stronger low-frequency prior helps both nets. On ConvNeXt-T the Lucid family stays
busy and high-frequency however it is set; the magnitude constraint (MACO), which forbids the optimiser from
putting energy where natural images have none, is what brings its renders into the judge's top ranks (plunger and
snail units at rank 1, bunting at 1-2) and, notably, keeps them inside the natural activation range (objective /
dataset max 0.3-1.2 against 3-6 for every Lucid-family setting: the renders stop over-driving the unit).

**Decision** (by the rule above): "tuned" = `mean_lowf` for GoogLeNet and `maco_lowf` for ConvNeXt-T. On
GoogLeNet three variants tie within the noise of 6 units (mean 34.1, mean_lowf 33.7, maco_dc 36.0); the rule
picks mean_lowf, whose renders are also the most legible by eye (hourglasses with sand, shower caps on heads). On
ConvNeXt-T maco_lowf wins clearly; maco_dc is the runner-up and has more natural colours (brown snail shells, a
red plunger cup), so a reader who prefers it can look at that column on the contact sheet. The tuned setting was
then rendered for all 20 gallery units of each model (plus FV seed 1 for the two seed units) and added as a fourth
column to `imnet_gallery_<arch>.png`, `imnet_fv_seeds_<arch>.png` and the json (`fv["tuned"]`, `ratio["tuned"]`,
and `judge[setting]` = independent / own rank for every setting and unit).

**Result on the full sample** (`judge_summary` in the json; "class-dominated" = majority >= 10 of 20, so that the
unit has a class for the judge to find):

| model | setting | class-dominated units whose render the independent judge ranks top-5 | median rank, all 20 units |
|---|---|---|---|
| GoogLeNet inception5b | 2009-style | 0 of 8 | 457 |
| | 2017 (our first settings) | 0 of 8 | 406 |
| | path-anchored | 0 of 8 | 505 |
| | **tuned (mean_lowf)** | **5 of 8** | 182 |
| ConvNeXt-T last MLP | 2009-style | 0 of 14 | 413 |
| | 2017 (our first settings) | 0 of 14 | 349 |
| | path-anchored | 0 of 14 | 273 |
| | **tuned (maco_lowf)** | **4 of 14** | 33 |

**Reading (corrected 7 Nov).** The tuned GoogLeNet column is a clear step up from our first settings: the
hourglass units render as small hourglasses with sand, the shower-cap unit as caps on heads, the banjo units as a
player with an instrument, and the mixture units (random draws, 2-9 of 20) as texture mixtures. But put beside the
essay's 2017 InceptionV1 column it is not close: theirs is one large, smoothly shaded object filling the frame in
natural colours; ours tiles the frame with small copies, washed-out contrast and high-frequency sparkle. For
ConvNeXt-T the tuned renders are the right colour and rough shape (a blue bird-shaped blob for the indigo-bunting
unit, a red cup on a stick for the plunger unit, a spiral shell for the snail unit) and an independent classifier
reads 4 of the 14 class-dominated units from them, against 0 before, but they are not objects. The judge is a
coarse yardstick (it rewards "the class is findable somewhere in the frame", not "one clean object"), so the
numbers overstate the visual gain. Two untested differences are the most likely cause: the model (the essay uses
the original 2014 InceptionV1 weights; ours is torchvision's BatchNorm retrain) and our re-implementation of the
2017 recipe rather than the Lucid library. Done: see "The essay's own InceptionV1" below; the model was the difference.
Path anchoring, with or without the other fixes, does not change the judge's verdict on either model
(it was never the missing ingredient here; the essay introduces it for a different failure, the dog basin).
Time: the sweep is 11 variants x 6 units, 15-136 s each; the full-sample renders took 167 s (GoogLeNet) and
495 s (ConvNeXt-T).

## The essay's own InceptionV1: Lucid's 2014 weights (`analysis/imnet_lucid.py`)

**Why.** After the tuning sweep our GoogLeNet renders were still far below the essay's 2017 InceptionV1 column by
eye (small tiled shapes with high-frequency sparkle, against one large smoothly shaded object). Two differences
were untested: our re-implementation of the 2017 recipe, and the model (torchvision's GoogLeNet is a BatchNorm
retrain; the essay's figures use the original 2014 inception5h weights that Lucid ships). The smell test that
settled it: the same library with the same defaults (lucent 0.1.8, Lucid's PyTorch port: 128 px Fourier image with
colour decorrelation, pad 12 / jitter 8 / scale 0.9-1.1 / rotate +-10 / jitter 4, bilinear upsample to 224, Adam
0.05, 512 steps) gives the essay's look on the 2014 weights at once and mush on torchvision's GoogLeNet and on
ConvNeXt-T (`reports/figs/imnet_lucid_models.png`). **The specimen was the difference, not the recipe.**

**What was done.** The 2014 net (lucent `inceptionv1(pretrained=True)`, input RGB in [0,1] x 255 - 117) was run over
the 50,000 val crops of the shared cache; mixed5b pre-ReLU pole scores (concat of the four branch convs, 1024 ch)
saved to `data/imnet_cache/imagenet_val/inceptionv1_lucid_extract.npz` like the other nets. Its 1,008 outputs use
the 2012 devkit class order, so the map to the standard order was recovered as the modal true label per output
index and checked by the accuracy it yields: **69.1% top-1 in RGB, 53.4% with the channels flipped** (so RGB is
the right order; 995 distinct classes are covered). The unit sample is the same rule as the other models (8 by
top-20 purity, 12 random, seed 0). Renders: lucent defaults, objective = spatial mean of the unit's pre-ReLU branch
channel, 8 units per batched optimisation (31-33 s per batch of 8 on the 5070 Ti; a first single-unit run after the
extraction pass took 312 s per unit because the GPU cache was full: free it first). Judges as before: ConvNeXt-T
(independent) and the net itself through the recovered map. Figures: `imnet_gallery_inceptionv1_lucid.png`
(20 units: render + 20 strongest photos, 1400 x 5200), `imnet_lucid_models.png` (one microscope, three nets),
`imnet_lucid_seeds.png` (seeds 0 and 1 for units 852 and 860: same bird, different layout). Numbers:
`reports/results/imnet_lucid.json`.

| unit | role | top-20 majority | ConvNeXt-T rank | own rank | by eye (render) |
|---|---|---|---|---|---|
| 852 | picked | goldfinch 14 | 290 | 83 | one large bird: eye, beak, feather texture |
| 860 | picked | jacamar 14 | 13 | 65 | green/blue bird heads |
| 670 | picked | junco 13 | 247 | 397 | grey plumage, no whole bird |
| 105 | picked | junco 13 | 23 | 11 | a small grey bird with an eye |
| 59 | picked | corkscrew 13 | 161 | 1 | twisted metal shape |
| 894 | picked | goldfinch 12 | 168 | 27 | yellow-green bird parts |
| 813 | picked | gasmask 11 | 9 | 1 | a mask-like face with a round filter |
| 1013 | picked | oystercatcher 11 | 594 | 387 | a bird head with an eye, clutter |
| 16 | random | basketball 6 | 306 | 224 | eyes and curved shapes (mixture unit) |
| 41 | random | Pekinese 8 | 4 | 511 | dog faces |
| 273 | random | radio telescope 9 | 5 | 2 | dish shapes against sky |
| 934 | random | toy terrier 5 | 81 | 79 | a dog face |
| others | random | 2-8 of 20 | 33-870 | 23-443 | animal-part collages |

Judge summary: class-dominated units (majority >= 10/20) = 8; the independent judge ranks **0 of 8** in its top 5
(median rank over all 20 units: 268); the net itself ranks 2 of 8 (median 81). For comparison, under the same
lucent defaults the ConvNeXt-T judge ranks torchvision GoogLeNet's hourglass and shower-cap renders 1st although
they are mush to a human, and ranks the Lucid goldfinch render 290th although it is plainly a bird.

**Reading.** (1) The renders have the essay's 2017 look on every unit: large smooth forms, natural palette, legible
subjects on the purer units (goldfinch, jacamar, junco, gasmask, Pekinese, radio dishes, terrier) and animal-part
collages on the mixture units, which is how the essay describes the 2014 net's late layer. This is the first time
our feature visualizations match the essay's figures by eye; the microscope question for claim A is answered on
the essay's own specimen. (2) The 2014 net's purest last-layer units are only 11-14 of 20 class-dominated, against
20 of 20 for torchvision's GoogLeNet and ConvNeXt-T: the old net is messier in the dataset examples too, as the
essay's ladder (~5% / ~50% for this net) says. (3) Our independent-classifier judge does not track human
legibility for this rendering style: ConvNeXt-T reads none of the Lucid renders and reads the mushy torchvision
ones. The judge was a fair way to pick among settings on one model; it is not a measure of essay-quality, and the
earlier "5 of 8 read correctly" numbers should be read with that in mind. (4) The essay's 2026 column (cleaner single objects on the same net) is partly reproduced by the path anchor
and no further: see the next subsection. The other pretrained analyses (purity, negative poles, weak grid, SAE)
have not been rerun on the 2014 net.

### Toward the essay's 2026 column (`analysis/imnet_lucid_sweep.py`, figure `imnet_lucid_sweep.jpg`)

The essay's timeline shows a further step from 2017 to 2026 on the same InceptionV1 neurons: one clean object,
less clutter of faces and animal parts. Six variants on 12 of the units above (the 8 picked + 41, 273, 642, 934),
one FV seed each, all with lucent's transforms and colour decorrelation; path / path_long / tv use our own loop
that copies lucent's chain exactly (128 px image, standard transforms, x*255-117, bilinear upsample to 224, Adam
0.05). Path anchor as in `imnet_featvis.py`: previous layer = `mixed5a` (post-ReLU concat, 832 ch), A = sum over
positions of a_prev * ds/da_prev, r_bar = mean A over the unit's 60 strongest val images, rho = dataset max -
median of the pole score, first 64 steps only. TV weight per unit = 10% of the default render's objective per
unit of TV (307-659; realised penalty 4.7% of the objective at step 0). Three judges (rank of the majority class):
the net itself (through the recovered map), ConvNeXt-T, torchvision GoogLeNet. Geometric means over the 8 picked
/ 4 random units:

| variant | what | own net | ConvNeXt-T | GoogLeNet | median obj / dataset max | time (12 units) |
|---|---|---|---|---|---|---|
| default | lucent defaults, 128 px, 512 steps | 26.5 / 77.3 | 86.4 / 34.5 | 108.9 / 52.5 | 1.72 | (reused) |
| long | 128 px, 2560 steps | 19.7 / 56.5 | 69.6 / 31.1 | 95.4 / 17.8 | 1.88 | 259 s |
| hires | 224 px, 1024 steps, no upsample | 27.9 / 45.3 | 124.4 / 48.1 | 107.3 / 81.2 | 3.51 | 122 s |
| path | default + path anchor, first 64 steps | **5.0** / 35.3 | 56.2 / 11.5 | **54.8** / 18.6 | 1.83 | 56 s |
| path_long | path, 2560 steps | 7.0 / **27.4** | **38.2** / **10.8** | 57.4 / **15.4** | 1.90 | 263 s |
| tv | default + total-variation penalty | 22.4 / 30.6 | 65.9 / 24.5 | 81.8 / 23.7 | 1.91 | 50 s |

(Per unit and variant: objective, ratio, three ranks, path cosine and TV at step 0, in `imnet_lucid_sweep.json`;
every render in `reports/figs/imnet_fv2/inceptionv1_lucid/`; contact sheet cells labelled "own / ConvNeXt-T /
GoogLeNet". Peak GPU 2.4 GB; 13 min in all.)

**Reading (by eye, then the numbers).** The path anchor is the only tweak that moves the renders toward the 2026
look, and it moves them part of the way: greyer, flatter backgrounds, fewer texture tiles, and more often one
whole thing (860 one perched bird with head and beak; 670 one bird head with an eye instead of armour and faces;
894 a single bird silhouette; 273 one dish on legs; 934 one animal head; 59 one corkscrew shape). It still has
extra eyes, duplicated parts and feather spill (852, 105), so it is not the essay's 2026 render (a single
photographic-looking object). All three judges agree on the direction (picked-unit geometric mean 26.5 -> 5.0
own, 86 -> 56 ConvNeXt-T, 109 -> 55 GoogLeNet; 670 junco goes from ranks 397 / 247 / 507 to 1 / 15 / 67).
path_long scores best on two independent judges but by eye regresses toward clutter once the anchor is released
(face-like blobs regrow on 852 and 860), with one exception (642 grows a goose head). long alone is more saturated
and more cluttered than default; hires is the worst by eye and by judge (fractal high-frequency clutter, ratio 3.5:
the usual exploit signature); tv is default, slightly smoother. So the essay's "path-guided optimisation for the
first 64 steps" is a real ingredient and we have it; its "few subtle tweaks to reasonable settings" we do not,
and the remaining gap to the 2026 column is theirs to publish or ours to find. Single seed; per-unit rank
swings of tens are partly seed noise, the direction across three judges is the robust part.

## Files

- `analysis/imnet_featvis.py`, `analysis/imnet_gallery.py`, `analysis/imnet_featvis2.py` (microscope tuning), `analysis/imnet_lucid.py` (the essay's InceptionV1 via lucent), `analysis/imnet_lucid_sweep.py` (2026-column sweep on it)
- `reports/results/imnet_lucid_sweep.json`, `reports/figs/imnet_lucid_sweep.jpg`, `reports/figs/imnet_fv2/inceptionv1_lucid/<unit>_<variant>.png`
- `reports/results/imnet_lucid.json`, `reports/figs/imnet_gallery_inceptionv1_lucid.png`, `reports/figs/imnet_lucid_models.png`, `reports/figs/imnet_lucid_seeds.png`, `reports/figs/imnet_fv/inceptionv1_lucid/<unit>_lucid[_s1].png`
- `reports/results/imnet_featvis2.json`, `reports/figs/imnet_fv2_<arch>.jpg`, `reports/figs/imnet_fv2/<arch>/<unit>_<variant>.png`
- `reports/results/imnet_featvis.json` (per unit: role, majority class and count, dataset max and median, rho,
  FV objective and objective / dataset max per setting and seed; per arch: the negative-pole selection)
- `reports/figs/imnet_fv/<arch>/<unit>_<setting>[_s1].png` (88 per arch, 224 px; `_tuned` from imnet_featvis2.py)
- `reports/figs/imnet_gallery_{googlenet,convnext_tiny}.png`, `reports/figs/imnet_gallery/<arch>/<unit>.jpg`,
  `reports/figs/imnet_fv_seeds_*.png`, `reports/figs/imnet_negpoles_*.png`, `reports/figs/imnet_shapes_*.png`
