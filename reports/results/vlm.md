# Results: Qwen3-VL-4B-Instruct on arm videos (VLM study)

All numbers are on test episodes of `data/main` (300 episodes drawn with seed 0 from [2000, 2400)).
Key numbers are in `reports/results/vlm.json`, which merges the per-analysis `vlm_<name>.json`
files. Activations are cached in `data/vlm/` (not in git, ~18 GB). Everything ran on the msi GPU
(RTX 5070 Ti Laptop, 12 GB) in bf16. Logs are in `models/vlm_logs/`.

The first pass (section 0) was census and probe first. Sections 1–4 follow
`docs/vlm_essay_next.md`, which re-orders the work the way N. Cammarata's essay
"Neural Networks Are Already Clean" (https://already-clean.goodfire.com/) reads neurons: fix the
tools, read units openly, test explanations so they can lose, and only then re-score the census.
Item codes (E1, B3, D3 ...) refer to `reports/research/essay_todos_2026-10-30.html`.
Thresholds, controls and scoring rules that the essay does not specify are **ours** and are
marked as such.

## 0. First pass

### 0.1 What the model sees

| | |
|---|---|
| model | `Qwen/Qwen3-VL-4B-Instruct`, bf16, SDPA attention, prompt "Describe the motion of the arm." |
| clip | 8 sim frames, stride 2 (s, s+2, …, s+14), so 25 fps and 0.30 s of motion; rendered at 320×320 px by `sim.arm.ArmRenderer` |
| corpus | 300 test episodes × 9 non-overlapping clips (start frames 0, 16, …, 128) = 2,700 clips |
| temporal patching | the patch embedding is a Conv3d with temporal patch 2, so 8 frames become **4 temporal groups**; group g holds frames s+4g and s+4g+2 and is aligned to the ground truth of the sim frame between them (s+4g+1) |
| spatial patching | 16 px patches, 20×20 = 400 vision patches per group; the 2×2 merger gives 10×10 = 100 LM tokens per group (400 visual tokens per clip) |
| unit read-out | per temporal group (10,800 rows), the max (and for SwiGLU also the min) over the group's 400 patches (vision) or 100 visual tokens (LM) |
| hooked layers | vision MLP hidden units (GELU, 4,096 wide) of blocks 3, 8, 13, 18, 23 (of 24); LM MLP hidden units (SwiGLU product, input to `down_proj`, 9,728 wide) of decoder layers 2, 8, 16, 24, 32 (of 36); all residual streams (mean-pooled per group) |
| controls | same frames given as single images (one group's midpoint frame); a randomly re-initialized vision encoder on the same clips; raw pixels (PCA-512) |

### 0.2 Census vs random-init control

Method (as in the other studies): binned R² (30 quantile bins) of each live unit against all 32
targets of `analysis/common.targets()`; clean ≥ 0.5, partial 0.25–0.5, otherwise unexplained (ours).
Conjunction flag in this study: the best variables of two different families both reach 0.25 and a
6×6 joint bin model beats the best single variable by ≥ 0.10 (ours; looser than the 0.15 in
CLAUDE.md). LM units are read at both poles (+ = max-pool, − = −min-pool).

| stream | layer | pole | live | clean | partial | unexplained | conj. | top families (clean+partial) |
|---|---|---|---|---|---|---|---|---|
| video vision | 3 | + | 2219 | 152 (6.8%) | 728 (32.8%) | 60.3% | 12 | posture 669, energy 140 |
| video vision | 8 | + | 3369 | 361 (10.7%) | 1436 (42.6%) | 46.7% | 67 | posture 1220, energy 353 |
| video vision | 13 | + | 3868 | 321 (8.3%) | 1669 (43.1%) | 48.6% | 69 | posture 1337, energy 412 |
| video vision | 18 | + | 3235 | 33 (1.0%) | 324 (10.0%) | 89.0% | 12 | posture 217, energy 78 |
| video vision | 23 | + | 4060 | 43 (1.1%) | 780 (19.2%) | 79.7% | 13 | posture 500, energy 178 |
| **random-init vision** | 3 | + | 4096 | 133 (3.2%) | 938 (22.9%) | 73.9% | 0 | posture 927, energy 58 |
| **random-init vision** | 8 | + | 4089 | 97 (2.4%) | 894 (21.9%) | 75.8% | 6 | posture 853, energy 55 |
| **random-init vision** | 13 | + | 4093 | 100 (2.4%) | 882 (21.5%) | 76.0% | 2 | posture 838, energy 68 |
| **random-init vision** | 18 | + | 4092 | 116 (2.8%) | 853 (20.8%) | 76.3% | 2 | posture 811, energy 69 |
| **random-init vision** | 23 | + | 4092 | 114 (2.8%) | 932 (22.8%) | 74.4% | 2 | posture 882, velocity 67 |
| image vision | 3 | + | 2177 | 189 (8.7%) | 996 (45.8%) | 45.6% | 9 | posture 1094, energy 58 |
| image vision | 8 | + | 3214 | 532 (16.6%) | 1628 (50.7%) | 32.8% | 68 | posture 1773, energy 236 |
| image vision | 13 | + | 3764 | 509 (13.5%) | 2005 (53.3%) | 33.2% | 97 | posture 2111, energy 253 |
| image vision | 18 | + | 2948 | 47 (1.6%) | 467 (15.8%) | 82.6% | 19 | posture 420, energy 59 |
| image vision | 23 | + | 4025 | 82 (2.0%) | 1404 (34.9%) | 63.1% | 22 | posture 1219, energy 166 |
| video LM | 2 | + | 7001 | 115 (1.6%) | 2905 (41.5%) | 56.9% | 34 | posture 2569, energy 258 |
| video LM | 2 | − | 6976 | 111 (1.6%) | 2954 (42.3%) | 56.1% | 38 | posture 2599, energy 257 |
| video LM | 8 | + | 9728 | 91 (0.9%) | 2303 (23.7%) | 75.4% | 61 | posture 1594, energy 485 |
| video LM | 8 | − | 9728 | 128 (1.3%) | 2373 (24.4%) | 74.3% | 54 | posture 1714, energy 467 |
| video LM | 16 | ± | 9728 | 2 / 2 | 221 / 208 | 97.7% / 97.8% | 2 / 1 | posture, energy |
| video LM | 24 | ± | 9728 | 1 / 0 | 54 / 58 | 99.4% | 1 / 0 | posture, energy |
| video LM | 32 | ± | 9728 | 0 / 1 | 36 / 34 | 99.6% | 0 | posture, energy |

**Figure `vlm_census.png`** (five panels: video, random-init, single image, LM + pole, LM − pole).

Findings.
- The trained vision encoder is cleaner than its random-init control only in the middle blocks
  (V8: 10.7% clean vs 2.4%; V13: 8.3% vs 2.4%). At V18 and V23 the trained encoder is **less**
  clean than random init (1.0–1.1% vs 2.8%). A random encoder is a smooth function of the image,
  so 20–23% of its units are already partial on posture: on this testbed, posture labels are cheap.
- Almost everything that is clean is posture or potential energy (which is a function of posture).
  Velocity, gravity, interaction and contact units are a few per layer; no acceleration or applied
  torque unit reaches 0.5 anywhere.
- Single frames give a *cleaner* census than video (V8: 16.6% vs 10.7%). With video, some units
  respond to motion, which the per-frame target set describes worse.
- LM units at the visual tokens lose their alignment to the arm's state quickly with depth:
  ~1.6% clean at L2, ~1% at L8, then essentially none from L16 on (>97% unexplained). Both
  SwiGLU poles behave alike.
- Partial R² beyond posture+velocity (each dynamics target residualised on a gradient-boosted
  model of the true posture and velocity) never exceeds 0.29 for any unit (V18 max); above a
  permutation null (99.9%) but tiny in absolute size.

### 0.3 Probes

Ridge, 5-fold CV grouped by episode, all 32 targets, every layer (`vlm_probe.py`).
**Figures `vlm_probe_heatmap.png`, `vlm_probe_families.png`, `vlm_probe_partial.png`.**

| best-layer R² | sin q₁ | q̇₁ | q̇₂ | hand vz | q̈₁ | grav. torque sh. | interaction sh. | any contact | wall force | KE |
|---|---|---|---|---|---|---|---|---|---|---|
| raw pixels (PCA-512, both frames of a group) | 1.00 | 0.11 | 0.02 | 0.30 | 0.03 | 1.00 | 0.02 | 0.62 | 0.49 | 0.14 |
| random-init vision residual | 0.97 | 0.20 | 0.02 | 0.30 | 0.03 | 0.98 | 0.01 | 0.74 | 0.60 | 0.65 |
| vision residual, video | 1.00 | 0.20 | 0.03 | 0.47 | 0.11 | 1.00 | 0.11 | 0.82 | 0.67 | 0.84 |
| LM residual at visual tokens, video | 1.00 | 0.23 | 0.04 | 0.46 | 0.12 | 1.00 | 0.13 | 0.81 | 0.63 | 0.83 |
| LM residual, single image | 1.00 | 0.02 | 0.01 | 0.01 | 0.11 | 1.00 | 0.11 | 0.68 | 0.61 | 0.17 |
| max-pooled LM MLP units (L8, both poles) | 1.00 | 0.31 | 0.07 | – | 0.15 | 1.00 | 0.19 | 0.82 | 0.64 | 0.79 |

- Posture and gravity torque (a function of posture) are linear in every layer, including raw pixels.
- Velocity is weakly decodable: q̇₁ R² ≈ 0.2–0.3, elbow velocity ≈ 0. Single images carry no
  velocity (as they must), so the video gain is real but small. Speed (KE, q̇₂²) is far more
  decodable than signed velocity (0.84 vs 0.2): the model encodes "how much motion", not direction.
- Acceleration, interaction and motor torque stay at R² ≤ 0.15.
- **Beyond posture+velocity, nothing is decodable.** Probes on y − E[y | true posture, velocity]
  reach R² ≤ 0.02 for every dynamics target (`vlm_probe_partial.png`). Contact R² 0.82 is
  explained by posture (GBM on posture+velocity reaches 0.92).

### 0.4 Verbal answers vs probes (behaviour)

Five two-choice questions, 300 balanced clips each, drawn from all 400 test episodes
(`vlm_behaviour.py`). Verbal = greedy answer; logit AUC = first-token logit difference; probe =
5-fold logistic regression on the last-prompt-token residual (best of 37 LM layers) or the mean
final vision block. **Figure `vlm_behaviour.png`.**

| question | verbal bal. acc. | answer-logit AUC | vision probe | best LM probe (layer) | shuffled-label probe |
|---|---|---|---|---|---|
| touching floor or wall? | 0.50 (says "no" 99%) | 0.64 | 0.94 | 0.96 (L8) | 0.50 |
| floor or wall? (given contact) | 0.88 | 0.95 | 1.00 | 1.00 (L1) | 0.50 |
| speeding up or slowing down? | 0.59 | 0.67 | 0.74 | 0.88 (L25) | 0.50 |
| blue or orange part rotating faster? | 0.53 (says "blue" 97%) | 0.81 | 0.71 | 0.91 (L22) | 0.49 |
| will it hit floor/wall right after? | 0.51 (says "no" 99%) | 0.60 | 0.65 | 0.66 (L17) | 0.47 |

The model encodes contact (probe 0.96) but never says so (it answers "no" to 99% of contact
clips). It encodes which link moves faster (probe 0.91, logit AUC 0.81) but its greedy answer is
a fixed "blue". Anticipation of an impact 0.2 s ahead is weak in both the answer and the probes
(0.66). Only the floor/wall distinction, which is a static spatial judgement, reaches language.

### 0.5 Contact events

Event-aligned activity around 402 floor and 244 wall contact onsets (`vlm_contact.py`,
**figure `vlm_contact_events.png`**). The contact probes (LM L6 and vision V23) track the true
contact fraction with no lead: their mean in the 12 frames before onset is *below* a far
baseline (−0.10 to −0.14 SD), so there is no anticipation. The four most contact-selective units
(V3#623, V8#1004, V13#917, V18#3771; contact binned R² 0.50–0.53) all have the *opposite* sign:
they fire for "free and close to a surface" and drop at contact. Before onset they rise by
0.14–0.38 SD over baseline (0.17–0.45 SD after regressing out clearance), which is consistent with
an approach signal, but it is small and could still be a clearance nonlinearity our 20-bin
clearance model misses.

### 0.6 Generalisation renders and activation bands

The same trajectories rendered with swapped link colours, grey links, a dark backdrop, no visible
floor/wall, and a flat 2-D cartoon (30 episodes × 3 clips; `vlm_general.py`,
**figure `vlm_generalisation.png`**).
- Posture units mostly survive recolouring (V8#2591 hand height: tuning R² 0.81 → 0.83 swap,
  0.83 grey, 0.69 cartoon), but **V8#2591 dies when the floor is not drawn (0.02)**: its spatial
  map lies on the floor strip, so it reads "hand near the floor" from the image, not hand height.
  The same holds for L2#9169 "potential energy" (0.68 → 0.07). Labels from the census can hide
  this.
- V8#1726 (potential energy; map on the region above the shoulder) is invariant to everything
  (0.78–0.82).
- Linear probes trained on the original renders (LM L6 residual) transfer for shoulder posture
  under recolouring (sin q₁ 0.98 swap, 0.93 grey, 0.92 dark), but the forearm angle sin(q₁+q₂)
  fails on grey links (0.97 → −0.06): the probe finds the forearm by its colour. Contact transfers
  to grey links (0.77) but not to the no-obstacle render (−0.78, as it should: contact is
  invisible there). Velocity probes do not transfer to any variant.
- V18#4014 (labelled "contact torque, shoulder") is silent (constant 0) on the no-obstacle and
  cartoon renders; its tuning is reported as n/a (the first-pass figure printed R² = 1.00 there,
  which was an artefact of binned R² on a constant; fixed).

Activation-quantile bands (**figure `vlm_bands.png`**): for the four band units, the census
variable explains the 50–90th percentile band best (R² 0.30–0.73) and loses the top 1% to other
variables (e.g. V8#2591's top 1% is better explained by hand vz; V3#1493 "q̇₁" is best explained by
cos q₂ in its top 1%). The labelled variable describes the mid-range, not the strongest
activations. Section 4 quantifies this.

### 0.7 Caveat on the main metric (C2)

Binned R² is a simulation-style score: it asks how well a single variable predicts a unit's
activation over all 10,800 groups. The essay warns that such scores "reward fine details of weak
activations that matter little for understanding what a neuron does", and that binning is only a
partial fix because "neurons differ a lot in their natural thresholds and sparsity". Most groups
sit in a unit's weak range, so a "clean" label here can be a description of weak activity, and an
"unexplained" unit can still have a crisp strong pole. Sections 3 (explanations that can lose) and
4 (strong-activation census) are the checks.

### 0.8 Figure fixes in this pass

All ten first-pass figures were reopened. Fixed: clipped unit labels and hidden tuning-axis labels
in the galleries, overlapping x labels and legend in `vlm_behaviour.png`, legends over data in
`vlm_contact_events.png`, `vlm_generalisation.png` and `vlm_probe_partial.png`, the label on the
vision/LM divider in `vlm_probe_heatmap.png`, an empty band in `vlm_generalisation.png`, and the
R² = 1 artefact for constant units (now n/a) in `vlm_bands.png` and `vlm_generalisation.png`.
The census figure gained the random-init panel.

## 1. Tool fixes

### 1.1 E1: SwiGLU sign regimes ("echo")

**Method.** One extra GPU pass over all 2,700 clips (`vlm_recapture.py`, 13 min) hooked
`gate_proj` and `up_proj` of LM layers 2, 8, 16, 24, 32, recomputed a = silu(g)·v at the 100 visual
tokens of each group, and stored the sign regime (g, v) of the token that gives each group's pooled
max and pooled min. The recomputed argmax token equals the cached one for 99.2–99.5% of
(group, unit) pairs (the rest are bf16 ties). Then (`vlm_regimes.py`) each unit pole's top-100 groups
and the essay's p75–p99 band were tagged. Definitions (ours): for the + pole the regular regime is
+×+ and the echo is −×−; for the − pole the regular regime is +×− and the echo is −×+ (the gate is
negative in both echoes, so |a| ≤ 0.28·|v|). A pole is re-labelled "echo" if > 50% of its top 100
is in the echo regime. **Figure `vlm_regimes.png`**, numbers in `vlm_regimes.json`.

| LM layer | + pole: units with echo top-100 | − pole: units with echo top-100 | top 10 entirely echo (+ / −) | gate closed at both poles | regular at both poles |
|---|---|---|---|---|---|
| L2 | **98.8%** | **98.6%** | 97.3% / 97.1% | **97.2%** | 0.1% |
| L8 | 52.5% | 53.5% | 34.4% / 34.5% | 30.2% | 24.1% |
| L16 | 46.6% | 46.6% | 30.1% / 31.1% | 29.3% | 36.1% |
| L24 | 39.9% | 39.7% | 19.9% / 19.8% | 20.3% | 40.6% |
| L32 | 11.0% | 11.6% | 2.9% / 3.2% | 1.8% | 79.3% |

The essay found the echo to be "the whole top of the list" for about 4% of neurons. At Qwen's visual
tokens it is the rule in the early LM: at L2 almost every unit keeps its gate closed (g < 0) at both
of its pooled extremes, so its "+ pole" is −×− and its "− pole" is −×+: the two census poles are the
two signs of the up-projection v read through a closed gate, not two features. This is why the L2
+ and − censuses in §0.2 are nearly identical. The distribution is bimodal at every layer (a unit is
almost always entirely regular or entirely echo), and the share falls with depth (11% at L32).

Gallery re-labelling. Five of the seven gallery LM units are gate-closed (echo) on their census pole,
at every rank band and in p75–p99: L2−#1848 (sin q₁), L2−#789 (shoulder gravity torque),
L2−#8960 (shoulder contact torque), L2+#9169 (potential energy) and L8−#7671 (elbow Coriolis).
Their labels describe what v carries while the unit is off, not an open-gate response; they are
re-labelled "gate-closed (echo)". L8−#4747 (hand vz) and L16−#6072 (q̈₂) are regular (+×−). Whether
a closed-gate signal of this size is read downstream is tested by steering in §3.2.

### 1.2 E2: residual "sink" coordinates

**Method.** In the same recapture pass we summed x² per residual coordinate over every vision token
(24 block outputs) and over every LM visual token and every LM token (37 hidden states), and counted
per token which coordinate is largest and whether it holds > 50% / > 10% of that token's x²
(`sink_stats.npz`). Coordinates holding > 10% of a layer's energy are the "sinks" removed below.
Then (`vlm_sink.py`) we re-ran (a) the standardised ridge probes of §0.3 and (b) a PCA probe (ours:
PCA-64 of the raw, unstandardised mean-pooled residual, then ridge on the 64 PCs) with and without
the sink coordinates. **Figure `vlm_sink.png`**, numbers in `vlm_sink.json`.

| stream | layers | sink coordinate | share of the layer's Σx² | tokens where it holds > 50% |
|---|---|---|---|---|
| vision | V0–V9 | none (largest coordinate ≤ 4%) | – | – |
| vision | V10–V22 | 62 | 45–58% at V10–V16 (> 50% at V11–V13), 25–40% at V17–V22 | 1–2% |
| vision | V23 (last block) | 201 | **92%** | **96%** |
| LM, visual tokens | L4–L16 | 0 | 14–37% | 0–18% (coordinate 0 is the largest coordinate in ~100% of tokens) |
| LM, visual tokens | L17–L34 | 4 | **78–96%** | 0.5–1.6% |
| LM, all tokens | L7–L34 | 4 | 85–94% | 0.6–19% |
| LM | L36 (after final norm) | none | 4% | 0% |

So Qwen3-VL has the essay's ConvNeXt pathology in both towers: the last vision block puts 92% of
its energy into coordinate 201 on almost every token, and from layer 17 on the LM puts 95% of the
visual tokens' energy into coordinate 4. In the LM the coordinate-4 energy sits in a few tokens with
massive values (it is the top coordinate of only ~1% of tokens), the familiar "massive activation"
pattern; coordinate 0 is a mild sink present on every token. In the mean-pooled cache coordinate 4
still holds 51–68% of the energy, and PC1 of the raw residual is the sink (PC1 explains 88–98% of
raw variance at L20–L32 with 96–97% of its loading on coordinate 4; without it PC1 drops to
31–54%). Vision V23: PC1 97% → 70% without coordinate 201.

**Effect on our results: none.** Removing the sink coordinates changes the mean probe R² (32
targets) by < 0.005 at every tested layer, for the standardised ridge (as expected, since
standardisation already puts each coordinate on equal footing) and also for the PCA-64 probe (the
sink takes one of 64 components). Any analysis that uses raw residual geometry (PC1, distances,
cosine maps) would be dominated by the sinks; we have none in the VLM study, but anyone adding one
should drop coordinates 201 (vision) and 4 and 0 (LM) first.

### 1.3 B2: the negative pole of the vision units (pre-GELU)

**Method.** The recapture pass also stored the vision blocks' `linear_fc1` output (pre-GELU) pooled
per group: max, min and mean over the 400 patches. `vlm_negpole.py` runs the census on the − pole
(−min of the pre-activation) and on the + pole (max of the pre-activation) of every unit, and for the
7 gallery vision units compares the poles. **Figure `vlm_negpole.png`**, numbers in
`vlm_negpole.json`.

| block | live units (pre-act) | + pole clean / partial | − pole clean / partial | both poles ≥ partial | … with different best family | − pole partial, + pole not |
|---|---|---|---|---|---|---|
| V3 | 3,886 | 267 / 1,521 | 217 / 1,426 | 777 | 325 | 856 |
| V8 | 4,096 | 546 / 2,210 | 613 / 2,378 | 1,996 | 1,029 | 995 |
| V13 | 4,096 | 437 / 2,258 | 229 / 2,029 | 1,502 | 789 | 756 |
| V18 | 4,096 | 99 / 1,518 | 90 / 1,754 | 765 | 436 | 1,079 |
| V23 | 4,096 | 49 / 1,043 | 11 / 835 | 228 | 138 | 618 |

- Reading before the GELU revives units: every unit is live on its pre-activation (vs 2,219–4,060 live
  after GELU), and the + pole census is cleaner (V8: 546 clean vs 361).
- The − pole is about as often labelled as the + pole in V3–V18, and for 600–1,100 units per block it
  is the only labelled pole. "Different best family" is mostly posture vs potential energy or gravity
  torque, which are themselves functions of posture, so it rarely means a new distinction.
- For the gallery units the − pole is weaker and mostly not a physical distinction: V3#1493
  ("q̇₁", really a stillness unit) has a − pole at R² 0.19; V8#160 (shoulder gravity torque) has the
  same variable at both poles (0.79 / 0.40); V8#1726 (potential energy) too (0.89 / 0.75). The
  strongest-negative patch is usually on the background or the floor (outlined in the figure), so the
  pooled min mostly reports what the unit does on empty image regions, which shifts with the global
  arm configuration. Max/min pooling over 400 patches is the limiting tool here (essay item E3).
- Contact vs free: no gallery unit's − pole separates contact from free motion better than its + pole.
  The best − pole contact AUC is 0.19 for V18#4014 (i.e. its − pole is larger in free motion; AUC of
  "free" 0.81), against + pole AUCs up to 0.91 (V3#1493 fires when the arm is still, which in this
  data mostly means resting on the floor). So the answer to the spec's question is no.


## 2. Reading units openly

### 2.1 B3: blind, free-form descriptions of 40 random units

**Method.** `vlm_blind.py strips` drew 40 live units at random (seed 2026; 4 per hooked layer: 20
vision, 20 LM; not chosen by the census) and wrote one image per unit,
`reports/figs/vlm_blind/U01.png … U40.png`: the 12 strongest positive and 12 strongest negative
groups (distinct episodes; vision units on the pre-GELU fc1 output, LM units on silu(g)·v), each tile
showing the frame with its motion trail, the strongest patch/token outlined and the raw activation,
plus a histogram of the raw pooled activations. The images carry no unit id, layer, label or ground
truth; the key is in `data/vlm/blind_key.json`. A separate agent, told to open only those 40
images, wrote a verb-style description of each pole and a strength rating (clear / weak / none) to
`reports/results/vlm_blind_descriptions.json`. Only after that file was saved did `vlm_blind.py
join` attach the census labels (binned R², both poles), and I compared each description with the
better-scoring pole's label (`vlm_blind.json`, `vlm_blind_compare.json`; the comparison categories
and judgements are ours).

The describer found a clear pattern on at least one pole for 27 of 40 units (positive poles: 14
clear, 16 weak, 10 none; negative poles: 21 clear, 15 weak, 4 none). Recurring themes in its own
words: many poles are read out at a fixed background patch (a corner, the left edge, behind the wall,
the floor under the shoulder) whatever the arm does, some with a near-constant value (e.g. U03 is
+8.25 at the bottom-left corner in every tile, likely a position/register artefact), but the arm pose
still differs between the poles; "hand down on the floor vs hand held up" is the most common
distinction; others are hand at the wall or in the floor–wall corner, body-part detectors (hand,
forearm vs shoulder), swinging vs still, and specific shapes (elbow folded shut, links in a straight
line).

| judgement (ours) | units | n |
|---|---|---|
| same as the census label | U03, U09, U11, U20, U21, U24, U29, U34, U37 | 9 |
| more general than the label | U08 ("arm hanging at rest, hand near the floor" vs cor₂/pe) | 1 |
| different, description better: an extreme or a conjunction of our variables that the all-data label missed | U01, U06, U07, U14, U15, U16, U26, U27, U30, U32, U33, U39, U40 | 13 |
| different, description better: not expressible in our 32 variables | U13 (hand detector whose sign flips when the hand is against the wall), U18 (forearm pressed along the wall face; its label is cos q₁), U25 (forearm-vs-shoulder part detector) | 3 |
| different, undecided | U12 ("links in a straight line" vs ke) | 1 |
| no pattern in the description (weak/none on both poles) | U02, U04, U05, U10, U17, U19, U22, U23, U28, U31, U35, U36, U38 | 13 |

**The free description beats the census label for 16 of 40 units (40%)**; the label is better for 3
(U28, U31, U36: partial labels where the describer saw no pattern); the two agree for 10. Census
classes of the same 40 units: 3 clean, 15 partial, 22 unexplained. Of the 22 "unexplained" units,
12 got a clear description. Most of those descriptions are still about our variables, but about one
extreme of them ("hand resting on the floor", "arm hanging straight down", "hand in the floor–wall
corner"): detectors of a rare state that an all-groups binned R² scores poorly (§0.7, §4). Only 3
descriptions need a concept outside the 32 targets, and all three involve where on the image the unit
fires (part identity, contact with the wall face). Caveats: one describer, 40 units, judgements by one
reader who had seen the census for other units; the fixed-background-patch readout means "where the
unit fires" is often not where the arm is, which is a property of max-pooling (E3), not of the unit.

### 2.2 C4: dial vs detector (raw tuning shape)

**Method.** For the 14 gallery units, the mean raw activation in 30 equal-width bins of the unit's
best variable (1st–99th percentile range, all 10,800 groups), classified by a rule that is ours
(the essay gives none): *tuned* if the curve peaks (or dips) inside the range; otherwise, from the
higher end, the width w over which the curve falls from 75% to 25% decides *detector* (w < 0.25 of
the range, with ≥ 40% of the range flat beyond it), *dial* (w ≥ 0.3) or *intermediate*.
`vlm_tuning.py`, **figure `vlm_tuning.png`**, `vlm_tuning.json`.

| class | units |
|---|---|
| dial (2) | V8#1726 potential energy; L2−#8960 shoulder contact torque (noisy) |
| detector (4) | V8#2591 hand height (fires only with the hand at floor level, ee_z ≈ −0.82); V8#160 and L2−#789 shoulder gravity torque (opposite signs; step at ≈ ±7 N·m); L2−#1848 sin q₁ (fires only with the upper arm pointing straight down) |
| tuned (7) | V3#1493 q̇₁, L8−#4747 hand vz, V3#2792 q̈₁, L16−#6072 q̈₂, V8#2386 and L8−#7671 elbow Coriolis, V18#4014 shoulder contact torque |
| intermediate (1) | L2#9169 potential energy (rises, then saturates over most of the range) |

The essay expected posture and gravity units to be dials and contact units detectors. Here it is
mostly the other way round: the posture and gravity units are detectors of one extreme pose (hand on
the floor, upper arm straight down, arm held out sideways), and only potential energy is a dial. All
seven "tuned" units peak (or, for V8#2386, dip) at **zero** of a velocity, acceleration or Coriolis
variable: they respond to the arm being still, not to a signed rate. Their census labels (q̇₁, hand
vz, q̈₁, q̈₂, cor₂) name a variable they are not tuned along; the shared description is "stillness".
V18#4014 is silent (exactly 0) in 90% of groups and fires on a few contact-torque values near
+3 N·m, which is a very sparse detector rather than a tuning.

## 3. Explanations that can lose

### 3.1 D3: simulator-written examples and near misses

**Method** (`vlm_nearmiss.py`; **figure `vlm_nearmiss.png`**; `vlm_nearmiss.json`). The essay's test
writes new examples from an explanation alone and checks that they fire, and near misses that
should not. Here the writer is the simulator. Each gallery unit's explanation is its census label
plus the direction read off its tuning curve (e.g. "V8#2591 hand height: fires when the hand is at
floor level"). For each unit, 8 clips put the labelled variable at its firing extreme with
everything the label does not constrain randomised (pose, other joint's velocity, side), and 8 near
misses change only the labelled variable (hand 21 cm higher; q̇₁ = ±4 rad/s; q̈ = ±40 rad/s²; the
mirrored pose for signed gravity torque; the same pose held 6 cm above the floor for contact
torque; arm horizontal instead of raised for potential energy). Clips are MuJoCo trajectories of
`sim/arm.py` (computed-torque tracking of q₀ + v t + a t²/2, or a passive drop onto the floor for the
contact clips), rendered and fed to Qwen exactly as the corpus; all 32 targets are logged, so we
checked that the writer hit the labelled variable (e.g. positives ee_z = −0.81, near misses −0.60;
grav₁ = ∓9.8; contact torque +1 to +7.6 N·m vs 0). Response = the unit's census pole, max over the
4 groups, as f = (a − corpus median)/(corpus max − corpus median) (ours; many units have a large
offset). Scoring (ours, with the spec's thresholds): a positive hits if f ≥ 0.5, a near miss is
rejected if f < 0.2; WON = ≥ 50% hits and ≥ 50% rejections, TIED = hits but near misses also fire,
LOST = < 50% hits.

| unit | label (explanation) | hits | near misses rejected | median positive (corpus pct.) | median near miss (corpus pct.) | verdict |
|---|---|---|---|---|---|---|
| V8#2591 | hand height: hand at floor level | 5/8 | 8/8 | 88 | 45 | **won** |
| V8#160 | shoulder gravity torque ≤ −9 N·m | 7/8 | 8/8 | 73 | 46 | **won** |
| V8#2386 | elbow Coriolis large (fast shoulder, bent elbow) | 4/8 | 6/8 | 95 | 56 | **won** (borderline) |
| L2−#1848 | upper arm straight down | 7/8 | 8/8 | 99 | 37 | **won** |
| L2−#789 | shoulder gravity torque ≥ +9 N·m | 7/8 | 2/8 | 99 | 70 | tied |
| V3#1493 | shoulder not rotating (q̇₁ ≈ 0) | 0/8 | 8/8 | 71 | 49 | lost |
| V3#2792 | no shoulder acceleration | 1/8 | 8/8 | 73 | 73 | lost |
| V18#4014 | shoulder contact torque ≈ +3 N·m (arm resting on the floor) | 0/8 | 8/8 | 0 | 0 | lost |
| V8#1726 | potential energy: arm raised | 1/8 | 8/8 | 91 | 28 | lost (near: median f 0.46) |
| L8−#4747 | hand vertical velocity ≈ 0 | 0/8 | 7/8 | 56 | 48 | lost |
| L16−#6072 | no elbow acceleration | 0/8 | 7/8 | 64 | 74 | lost |
| L8−#7671 | no elbow Coriolis torque | 1/8 | 6/8 | 74 | 51 | lost |
| L2−#8960 | shoulder contact torque large (arm resting on the floor) | 2/8 | 4/8 | 90 | 82 | lost |
| L2#9169 | potential energy: arm raised | 0/8 | 0/8 | 72 | 96 | lost (near misses fire more) |

**4 won, 1 tied, 9 lost.** The explanations that survive are the detector-like posture and gravity
labels (hand at the floor, upper arm down, arm held out to one side), and they also reject their
near misses (including the mirrored pose: V8#160 is specific to the sign of the torque). The
"stillness" units all lose: still clips written by the simulator, even with zero kinetic energy,
only reach f ≈ 0–0.45, so whatever makes their corpus top examples fire is more than "q̇ = 0".
Both contact-torque labels lose: an arm resting on the floor with +1 to +7.6 N·m of contact torque
does not drive V18#4014 at all (f = 0 for every clip) and drives L2−#8960 about as much as the same
pose held off the floor. L2#9169 "potential energy" fires more for a horizontal arm than for a
raised one, so the label is wrong in the direction it claims. L2−#789 fires for its positive-torque
poses, but its mirrored near misses still sit at the 70th corpus percentile (median f 0.22, just above
the rejection line), so it is only partly sign-specific, unlike V8#160.
Caveats: 8 + 8 clips per unit; a single corpus maximum can be an outlier (which makes "f ≥ 0.5"
strict); the written clips are short and mostly slow, which is not the corpus distribution.

## 4. Re-score: strong-activation census (C1)

**Method** (`vlm_strong.py`; **figure `vlm_strong.png`**; `vlm_strong.json`). For a seeded sample of
up to 1,000 live units per census key (video vision, LM both poles, and the random-init vision
control), binned R² (10 quantile bins) against the 32 targets on (a) all groups, (b) the "strong
half" (a ≥ median + 0.5·(max − median); ours, because many units have a large offset, so "above half
the max" would include almost every group), (c) the top 1% (108 groups). Units with < 30 strong
groups are "too sparse". Each unit also gets a shuffled-target null on the same subset.

| key | all: clean / partial | strong half: clean / partial / too sparse | top 1%: clean / partial | null (top 1%): clean+partial |
|---|---|---|---|---|
| vision V3 | 8% / 34% | 6% / 22% / 21% | 10% / 45% | 2% |
| vision V8 | 12% / 43% | 3% / 22% / 26% | 5% / 42% | 3% |
| vision V13 | 9% / 45% | 4% / 22% / 23% | 6% / 39% | 2% |
| vision V18 | 1% / 9% | 1% / 11% / 66% | 4% / 46% | 2% |
| vision V23 | 2% / 19% | 1% / 17% / 27% | 3% / 39% | 2% |
| random-init vision (5 blocks) | 2–3% / 21–22% | 4–5% / 28–30% / 14–17% | 6–7% / 44–47% | 1–3% |
| LM L2 (+ / −) | 1–2% / 41% | 2% / 19–24% / 17–20% | 3–4% / 41–44% | 1–2% |
| LM L8 (+ / −) | 1% / 25% | 1–2% / 20–21% / 25–28% | 2–3% / 41–42% | 2% |
| LM L16 (+ / −) | 0% / 1–3% | 0% / 12–13% / 20–21% | 1% / 23–24% | 2% |
| LM L24 (+ / −) | 0% / 1% | 0% / 10–11% / 37% | 0–1% / 20% | 2–3% |
| LM L32 (+ / −) | 0% / 0–1% | 0% / 9–11% / 46% | 0–1% / 18–19% | 2% |

Results.
- **Restricting to strong activations does not make the units clean.** Clean rates stay at 0–10% in
  every layer; nothing like the essay's "closer to 97% non-arbitrary". What rises is the *partial*
  rate in the top 1% of the deep LM layers (L16–L32: 18–24% vs ≤ 3% on all groups; null 2%): the
  strongest activations of late-layer units are more physical than their weak range.
- The strong-subset question is different from the all-data one: it asks what orders the strong
  activations among themselves, so the best variable there is rarely the census variable (2–8% of
  units) or even its family (14–40%).
- The random-init encoder scores the same as the trained one in the top 1% (6–7% clean, 44–47%
  partial vs 3–10% / 39–46%). On this testbed a strong-activation R² against posture-like variables is
  not evidence of learned structure without that control.
- The "described, not one of our 32 variables" bucket (from B3, 40 random units, §2.1):

| B3 sample (40 units) | n |
|---|---|
| labelled by the all-data census (clean or partial) | 18 (3 clean, 15 partial) |
| census unexplained, but clearly described by the blind reader | 12 |
| … of which the description is an extreme or a conjunction of our variables (e.g. "hand resting on the floor", "arm hanging straight down", "hand in the floor–wall corner") | 12 |
| census label present, but a description outside the 32 variables is better (U13, U18, U25) | 3 |
| neither a census label nor a clear description | 10 |

So with blind reading added, 30 of 40 random units (75%) have either a label or a clear
description, against 18 of 40 (45%) by census alone; the remaining 25% have no pattern either way.
Most of the gain comes from detectors of extreme states that an all-groups simulation score misses,
not from concepts outside our variable set.

