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
