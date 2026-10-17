# Results: a video-prediction model (candidate B)

All numbers are on held-out test episodes [2000, 2400) of `data/main` unless stated otherwise. Unit-level analyses use 19,600 test windows (400 episodes, frame *t* every third frame in [3, 147]). Probes are fit on training episodes [0, 400) and evaluated on the test episodes. Key numbers are in `reports/results/video.json`, which merges the per-analysis `video_<name>.json` files. To reproduce everything, run `uv run python analysis/video_run_all.py`; add `--train` to retrain first.

This is the second pass. The Mac run stopped at 4k steps (val arm-pixel MSE 0.0057); its checkpoints are kept in `models/video_ckpt_mac4k/`. This model was retrained on the msi GPU for 30k steps and every analysis was rerun on it.

## Model and accuracy

**Setup.** The model sees four consecutive 64×64 RGB frames (t−3 … t, 50 Hz, so 60 ms of motion) and predicts frames t+1 and t+2. The architecture (`models/video_model.py`, 1.75 M parameters) has three parts:

- **Per-frame ConvNeXt encoder** (`enc0`–`enc3`): 4×4 patchify to 16×16 tokens of width 128, then 4 blocks (depthwise 7×7, LayerNorm, 128→512 GELU→128). The encoder sees **one frame at a time**, so its units can only know what a still image shows. This is a built-in control.
- **Factorized space-time transformer** (`tmp0`–`tmp3`): 4 blocks, each with spatial attention within a frame, causal temporal attention per location across the 4 frames, and a 128→512 GELU MLP.
- **Light decoder**: from the frame-t tokens to 64×64 pixel deltas for t+1 and t+2.

"Neurons" are the 512 post-GELU MLP channels of each of the 8 blocks (4,096 units). Each unit is read at frame t as the max over the 16×16 positions.

**Training.** Loss is pixel MSE on t+1 (weight 1) and t+2 (weight 0.5). Pixels where the arm moves (t−1 … t+2, dilated 5×5) get 26× weight; otherwise "copy the last frame" dominates. The only augmentation is a ±2.5% brightness jitter. Optimizer: AdamW (lr 8·10⁻⁴, weight decay 0.05 on matrices, betas 0.9/0.98), batch 64, 1,000 warm-up steps, cosine decay to 0 over 30k steps (1.9 M sampled windows, ≈6.6 passes over the ~290k training windows; 91 min on the RTX 5070 Ti Laptop). The forward pass ran under bf16 autocast with TF32 matmuls; validation ran in fp32. Checkpoints were saved at 11 log-spaced steps (0, 100, 188, 355, 669, 1262, 2378, 4481, 8446, 15918, 30000). The step-0 checkpoint is the random-init control.

**Convergence.** On 800 validation windows (episodes 2300–2399), arm-pixel MSE (t+1) fell from 0.0230 at init (equal to copy-last) to 0.00412 at 5k, 0.00204 at 22.5k and 0.00189 at 28.5k–30k. It stayed flat at 0.00189 for the last 1.5k steps, so this is a plateau. The cosine schedule reaches lr ≈ 0 there, so a longer or larger run could still go lower.

**Figure `video_training.png`.** Left: validation MSE on moving-arm pixels for frame t+1 (log–log), with the copy-last-frame (dashed) and linear-extrapolation (dotted) baselines. Right: weighted training loss (EMA) over training.

**Accuracy (test, MSE on moving-arm pixels):**

| windows | n | model t+1 | copy-last t+1 | linear extrap. t+1 | model t+2 | copy-last t+2 | skill vs copy (t+1) |
|---|---|---|---|---|---|---|---|
| all | 19,600 | **0.00127** | 0.0141 | 0.0211 | **0.00256** | 0.0233 | **0.91** |
| free motion | 11,092 | 0.00192 | 0.0231 | 0.0343 | 0.00386 | 0.0383 | 0.92 |
| in contact | 8,508 | 0.00043 | 0.0024 | 0.0039 | 0.00086 | 0.0037 | 0.82 |
| contact onset (between t and t+1) | 617 | 0.00193 | 0.0190 | 0.0297 | 0.00364 | 0.0251 | 0.90 |
| *Mac 4k-step checkpoint, all* | 19,600 | 0.00369 | 0.0140 | 0.0211 | 0.00710 | 0.0232 | 0.74 |

Skill = 1 − MSE_model / MSE_copy. Retraining cut the error by about 3× at both horizons. The model handles contact onsets as well as free motion (skill 0.90). Arm pixels during sustained contact are mostly stationary, so absolute errors there are small and relative skill is lower.

**Figure `video_prediction.png`.** Three test windows (free motion, floor impact, wall impact). Columns: the four input frames, the true and predicted frame t+1, the true and predicted change (t+1 minus t; red = pixel gets brighter, blue = darker), and the predicted t+2. At impact the predicted arm stops at the surface instead of passing through it, but it is visibly blurrier than in free motion.

## 1. Census: how many neurons have one job?

**Method.** This is the same census as `state.md`. For each live unit (active on >1% of windows), we take the binned R² of its activation against each of the 32 targets at frame t (30 quantile bins). A unit is *clean* if its best R² ≥ 0.5, *partial* if 0.25–0.5, and otherwise *unexplained*. It is a *conjunction* if its best 2-D fit (8×8 bins) beats its best 1-D fit by >0.15.

**State control (our formalization).** A 2-layer MLP predicts each unit's activation from the instantaneous ground-truth state (sin/cos q, q̇). It is fit on probe-train episodes and evaluated on test. We then redo the binned fit on the residual ("partial R²"). A unit whose label survives on the residual carries information beyond the visible state.

| | live units | clean | partial | unexplained | conjunction |
|---|---|---|---|---|---|
| trained (all 8 blocks) | 4,084 / 4,096 | **24.6%** | 56.8% | 18.6% | 34.6% |
| random init | 3,374 / 4,096 | 6.5% | 42.1% | 51.4% | 16.0% |

Unlike the state MLP, where random features are already 25% clean, training here **does** make units cleaner, about 4× over the control. The reason is that a random convnet on pixels is not a smooth function of a few angles.

Clean counts by family (trained): posture 476, energy 294, velocity 141, gravity 63, interaction/Coriolis 27, contact 5, **acceleration 0**, applied torque 0. Per block, the clean fraction is 14% / 17% / 26% / 29% in the encoder (enc0–3) and 16% / 32% / 24% / 38% in the space-time stage (tmp0–3). Clean units on derived quantities (acceleration, gravity, interaction, contact, energy) make up 2–8% of the encoder and up to 23% of `tmp3`.

**What survives the state control.** It is very little. The median held-out R² of a unit predicted from the instantaneous state is 0.78. Of the 1,075 units with a non-posture/velocity label, only **13** keep a partial R² ≥ 0.25 once the state is regressed out. The largest residual fits are on posture (cos q₁ 0.46, elbow height 0.42) and potential energy (0.34). For every dynamics variable the residual fit is ≤ 0.21: contact force on the forearm 0.21, Coriolis 0.20, q̈ 0.15, contact torque 0.17.

**Alignment.** Averaged over units, the best R² peaks for targets at frame t (0.445), against t−1 (0.421) and t+1 (0.423). The units encode the present state, not the predicted next frame.

**Figure `video_census.png`.** Best single physical variable for each live unit, per block, for the trained network (left) and the random-init control (right). Solid segments are clean units coloured by the family of the best variable. Light tints are partial units of the same family, and grey is unexplained. Numbers above the bars are live units out of 512. The dotted line separates the per-frame encoder from the space-time stage.

**Figure `video_census_control.png`.** Left: for every partial or clean unit, the best binned R² of the raw activation (x) against the best binned R² after regressing out the instantaneous state (y). Points are coloured by the family of the raw label, and the diagonal is no change. Right: histogram of the held-out R² with which the state (q, q̇) predicts each unit's activation.

**Interpretation.** Most clean units are clean for posture (encoder) or for speed and stillness (space-time stage), and almost everything is explained by the current state. "Energy" units are the second-largest clean family. Their tuning curves (gallery below) show that most of them are *stillness detectors*: they peak at kinetic energy ≈ 0 or at q̇ ≈ 0 and fall off on both sides. No unit is clean for acceleration or motor torque.

## 2. Linear probes of the residual stream

**Method.** Ridge probes are fit on the residual stream after every block (avg-pooled to 4×4, 2,048-d), on probe-train episodes 0–399, with test R² on 2000–2399. Controls are the random-init network and ridge on raw pixels (last frame, or all 4 frames). A 2-layer MLP probe on `tmp3` and `enc3` tests nonlinear decodability. A state-only MLP (ground-truth sin/cos q, q̇) and a state-history MLP (q, q̇ at t−3 … t) give ceilings.

| target | raw pixels (4 fr.) | enc3 | tmp3 | tmp3, random init | tmp3 MLP probe | state MLP | state-history MLP |
|---|---|---|---|---|---|---|---|
| joint angles (sin q₂) | 0.47 | 1.00 | 0.99 | 0.75 | 0.99 | 1.00 | 1.00 |
| q̇₁ / q̇₂ | 0.10 / 0.01 | 0.03 / 0.01 | **0.90 / 0.63** | 0.01 / 0.00 | 0.96 / 0.89 | 1.00 / 1.00 | 1.00 / 1.00 |
| q̈₁ / q̈₂ | 0.04 / 0.04 | 0.13 / 0.10 | 0.33 / 0.29 | 0.04 / 0.02 | **0.68 / 0.54** | 0.55 / 0.41 | 0.85 / 0.75 |
| gravity torque, shoulder / elbow | 1.00 / 0.98 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.98 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| interaction torque, shoulder / elbow | 0.04 / 0.04 | 0.14 / 0.19 | 0.39 / 0.38 | 0.01 / 0.03 | 0.64 / 0.69 | 0.56 / 0.63 | 0.78 / 0.83 |
| Coriolis/centripetal, shoulder / elbow | −0.03 / 0.03 | 0.10 / 0.21 | 0.36 / 0.52 | 0.00 / 0.03 | 0.69 / 0.84 | 0.99 / 0.99 | 0.98 / 0.98 |
| motor torque, shoulder / elbow | 0.07 / 0.08 | 0.08 / 0.13 | 0.29 / 0.25 | 0.06 / 0.06 | 0.49 / 0.41 | 0.21 / 0.30 | 0.64 / 0.58 |
| any contact / log floor force / log wall force | 0.65 / 0.70 / 0.63 | 0.72 / 0.73 / 0.64 | **0.90 / 0.84 / 0.80** | 0.70 / 0.71 / 0.65 | 0.91 / 0.85 / 0.83 | 0.82 / 0.79 / – | 0.86 / 0.82 / – |
| contact torque, shoulder / elbow | 0.20 / 0.19 | 0.30 / 0.16 | 0.41 / 0.34 | 0.25 / 0.12 | 0.47 / 0.45 | – | – |
| kinetic energy | 0.21 | 0.18 | 0.99 | 0.37 | 0.98 | 1.00 | 0.99 |

**Figure `video_probes.png`.** Test R² (×100) of ridge probes for each of the 32 targets (columns, coloured by family) from raw pixels (last frame; 4 frames), from the random-init network (enc3, tmp3) and from each block of the trained network (rows). White lines separate the groups.

**Figure `video_probe_layers.png`.** The same ridge R² across depth for selected targets. Solid lines are the trained network, dotted lines random init, and dashed horizontal lines ridge on the 4 raw frames. The dotted vertical line separates encoder and space-time blocks. Panels: state (joint angles, joint velocities), acceleration and gravity torque, interaction torque, floor force and contact torque.

**Interpretation.**

- **Velocity is created by the space-time stage, exactly where the architecture says it must be.** The per-frame encoder has R² ≈ 0.02 for q̇. It jumps to 0.58 at `tmp0`, the first temporal attention, and reaches 0.90 (shoulder) and 0.63 (elbow) at `tmp3`. The random-init space-time stage gets 0.01. This is the cleanest result of the study.
- **Gravity torque: yes, but it is free.** Gravity torque is a function of posture, and posture is linearly decodable from raw pixels (R² 0.98–1.00, also at random init). Its decodability is no evidence of computation.
- **Acceleration and torques are encoded nonlinearly, and partly beyond the instantaneous state.** Linear R² for q̈ is only 0.33 and 0.29. An MLP probe on `tmp3` reaches 0.68 and 0.54, which beats an MLP given the true instantaneous state (0.55 and 0.41), because the frames carry history. It does not reach the state-history ceiling (0.85 and 0.75). Motor torque follows the same pattern (MLP 0.49 and 0.41, versus 0.21 and 0.30 from the state). The model therefore carries a nonlinear, history-based estimate of what the motor is doing. It is not in single units (census: 0 acceleration units).
- **Coriolis torque is decodable but not computed linearly.** The state MLP gets 0.99, but `tmp3` gets 0.36–0.52 linearly and 0.69–0.84 with an MLP. A product of velocities is only decodable after a nonlinearity.
- **Contact** rises steadily through the space-time stage (any contact 0.72 → 0.90). The single-frame encoder already gets 0.72, because contact is partly visible as geometry: arm touching the floor or wall.

## 3. Neuron gallery

**Method.** Exemplary units are picked automatically from the census: the best unit per family and stage, plus one high-variance unexplained unit. For each, the gallery shows the top-5 test clips from distinct episodes and the 3 weakest clips. Each clip is shown as the last input frame plus a faint motion trail of t−3 … t−1, with the unit's 16×16 activation map overlaid in magenta ('+' = argmax). A tuning curve (mean and 10–90% band) against the best variable is also shown.

| unit | label | best R² | top-3 variables | R² from state |
|---|---|---|---|---|
| enc2.342 | hand height | 0.90 | hand height, potential energy, elbow height | 0.97 |
| tmp0.500 | cos q₁ | 0.88 | cos q₁, gravity torque (shoulder), hand x | 0.95 |
| tmp3.406 | q̇₁ (shoulder velocity) | 0.90 | q̇₁, Coriolis (elbow), kinetic energy | 0.95 |
| tmp2.40 | hand v_z | 0.62 | hand v_z, hand height, kinetic energy | 0.81 |
| enc0.222 | contact torque, elbow (partial, conj.) | 0.31 | contact torque, potential energy, cos q₁ | 0.53 |
| tmp3.48 | contact torque, shoulder | 0.60 | contact torque, any contact, kinetic energy | 0.85 |
| enc3.401 | gravity torque, shoulder | 0.86 | gravity torque, cos q₁, hand x | 0.96 |
| tmp3.286 | Coriolis/centripetal, elbow | 0.75 | Coriolis, kinetic energy, q̇₁ | 0.86 |
| tmp3.20 | kinetic energy | 0.93 | kinetic energy, q̇₁, Coriolis | 0.95 |
| tmp3.465 | unexplained | 0.05 | – | 0.71 |

**Figure `video_gallery_a.png`.** Neuron gallery I, for posture, velocity and contact units. Each row is one unit: name, best variable, binned R² and status. Then come the 5 strongest test clips (distinct episodes; last input frame with the earlier frames as a faint trail, the unit's activation map in magenta, '+' at the argmax, activation in the corner), the tuning curve against the best variable (mean, 10–90% band) and the 3 weakest clips (blue '+' = argmax).

**Figure `video_gallery_b.png`.** Neuron gallery II, for contact, gravity, Coriolis, kinetic-energy and one unexplained unit, in the same layout.

**Interpretation.**

- Posture units are crisp and spatially localized. `enc2.342` fires on the hand when it is high. The top clips all show the arm raised, with the activation peak on the hand.
- Several of the cleanest velocity/energy units are *stillness* units. `tmp3.406` (q̇₁, R² 0.90) and `tmp3.20` (kinetic energy, R² 0.93) have tuning curves that peak sharply at zero velocity or zero energy. Their top clips show an arm at a turning point or resting. `tmp2.40` ("hand v_z") also peaks at v_z = 0. The high R² comes from a narrow peak, not from a signed velocity code.
- `tmp3.286` (Coriolis, elbow) has a V-shaped tuning curve. It fires for fast motion of either sign and reads as a speed unit.
- Encoder "contact" units (`enc0.222`) are only partial (R² 0.31). They respond to the forearm lying near the floor, a geometric cue.

## 4. Feature visualization

**Method.** The 4-frame input clip is optimized to maximize the unit's frame-t activation at its modal argmax position. There are three settings, after the essay's microscope comparison:

- **naive ("2009")**: pixel parameterization, plain gradient ascent.
- **Fourier ("2017")**: decorrelated Fourier parameterization with ±1 px jitter.
- **robust ("2026")**: strong transformation robustness (±3 px jitter, 0.92–1.08 scale, the same transform for all 4 frames), plus a scene-palette parameterization. The palette is a static background plus two Fourier-parameterized alpha masks painted in the upper-arm and forearm colours. It is a domain prior we added; it is not in the essay.

We report the activation reached and the activation after a random 2-px shift.

| unit | dataset max at position | naive | naive, shifted | Fourier | Fourier, shifted | robust | robust, shifted |
|---|---|---|---|---|---|---|---|
| tmp0.500 (cos q₁) | 3.0 | 7.6 | 2.6 | 6.5 | 5.5 | 3.6 | 3.6 |
| tmp3.406 (q̇₁ / stillness) | 4.8 | 11.9 | 3.0 | 0.0 | 0.0 | 4.8 | 6.9 |
| tmp3.48 (contact torque) | 4.5 | 10.5 | 2.2 | 8.7 | 7.4 | 5.8 | 6.1 |
| tmp3.286 (Coriolis) | 3.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.6 | 5.1 |
| enc3.401 (gravity torque) | 4.6 | 0.0 | 0.0 | 0.0 | 0.0 | 6.9 | 6.6 |

**Figure `video_featvis.png`.** Feature visualizations of five units (rows). Columns: naive pixel optimization ("2009"), Fourier parameterization ("2017"), the four frames t−3 … t of the robust scene-palette optimization ("2026"), and the unit's top dataset clip. The magenta '+' marks the optimized position. Numbers give the activation reached.

**Interpretation.** Naive optimization reaches ≈2.5× the dataset maximum, but the result is adversarial: a 2-px shift loses 65–80% of it. In 3 of 5 units, both naive and Fourier optimization stay at zero, because GELU units that start inactive get no gradient. The robust setting is the only one that activates all five units, at or above the dataset maximum, and it survives shifts. Its images are arm-coloured blobs around the target position, not clean arm poses, so they are readable only alongside the dataset examples.

## 5. Contact

**Method.** We take the top-3 encoder units and top-3 space-time units by census R² with any contact variable and compute them densely on test episodes. Events are clean contact onsets (no contact in the 6 preceding frames) at frame k, with lag = t − k, where t is the last input frame. Lag −1 means the contact happens between the last input frame and the first predicted frame. Near misses are local minima of arm–surface clearance below 2 cm with no contact within ±6 frames.

*Anticipation*: among windows with no contact at t and clearance(t) < 6 cm (5,687 windows, 890 with contact at t+1), we compute the AUC of each unit's activation for "contact at t+1". The baseline is a logistic regression on ground-truth clearance and approach speed.

| unit | best variable (R²) | AUC, contact at t+1 |
|---|---|---|
| enc0.222 | contact torque, elbow (0.31) | 0.53 |
| enc0.483 | contact torque, elbow (0.28) | 0.53 |
| enc1.121 | contact torque, shoulder (0.28) | 0.50 |
| **tmp3.48** | contact torque, shoulder (0.60) | **0.76** |
| tmp0.245 | contact torque, elbow (0.60) | 0.51 |
| tmp2.327 | any contact (0.59) | 0.67 |
| state baseline (clearance + approach speed) | – | 0.73 |
| logistic on all 6 units | – | 0.78 |

Events: 341 floor onsets, 205 wall onsets, 71 floor near misses and 36 wall near misses.

**Figure `video_contact.png`.** Mean activation (z-scored, ±1 s.e.) of the six contact units around floor onsets (solid red), wall onsets (solid amber), floor near misses (dotted red) and wall near misses (dotted amber). The x axis is frames from contact onset or closest approach (t − k). The solid line marks the onset. The dashed line marks lag −1, when the last input frame is just before contact. Titles give each unit's best variable with its binned R² and its AUC for contact at t+1.

**Interpretation.**

- **The encoder contact units are proximity detectors, not contact detectors.** They respond about as strongly to near misses as to real contacts. `enc1.121` responds more strongly to floor near misses than to onsets. Their anticipation AUC is at chance (0.50–0.53).
- **`tmp3.48` behaves like a floor-impact unit.** It is already at +1 z at lag −1, before any contact is visible in its input. It peaks at the onset, returns to baseline within 2 frames, and does not respond to near misses. Its anticipation AUC (0.76) beats the ground-truth clearance-and-speed baseline (0.73). Its response to wall onsets is about half as large.
- `tmp2.327` responds to both floor and wall onsets, more strongly to the wall. It is a transient "impact" unit, without anticipation at lag −1.
- `tmp0.245` codes "resting on the floor" (sustained after onset) rather than impact.

## 6. Acceleration, gravity and segment specificity

**Method.** We compare probe R² for q̈, gravity, interaction and motor torque across raw pixels, `enc3` and `tmp3` (ridge and MLP), and against what the instantaneous state can explain (section 2). For every unit whose best variable is a joint-indexed torque, we compute a selectivity index SI = (R²_shoulder − R²_elbow) / (R²_shoulder + R²_elbow).

**Figure `video_accel.png`.** Left: held-out R² for joint accelerations and torques from raw pixels (ridge), the encoder `enc3` (ridge), the space-time `tmp3` (ridge and MLP probe) and the true instantaneous state (MLP). Right: histogram of the shoulder-vs-elbow selectivity index for the 480 units whose best variable is a joint-specific torque (360 gravity, 101 interaction, 19 contact). −1 means elbow only and +1 shoulder only.

**Results.** The encoder-to-space-time gain is large for every dynamics target: q̈₁ 0.13 → 0.33, interaction torque (shoulder) 0.14 → 0.39, Coriolis (elbow) 0.21 → 0.52, motor torque (shoulder) 0.08 → 0.29. Among the 480 torque units, 27.5% are shoulder-selective (SI > 0.5), 4.2% are elbow-selective (SI < −0.5) and 68% are mixed. The histogram is bimodal, with one mode near SI ≈ −0.15 and one near +0.55.

**Interpretation.** Segment-specific torque units exist mainly for the shoulder, and they are almost all gravity-torque units: shoulder gravity torque is a near-linear function of cos q₁, which posture units encode. True elbow-specific units are rare. This matches the probes: elbow velocity and torques are consistently harder to decode than shoulder ones. The forearm is thinner and shorter in the image.

## 7. Causal tests

**Method.** We mean-ablate 32 contact units and, separately, 32 acceleration units. Each ablation replaces the unit's hidden activation at all positions and all 4 frames with its dataset mean. Contact units are the top 32 by census R² with any contact variable, from all 8 blocks. Acceleration units are the top 32 by R² with q̈₁ or q̈₂, from `tmp1`–`tmp3`; most of them have kinetic energy as their best variable (see caveats).

The control ablates the same number of random live units from the same layers, with 5 draws. There are 300 contact-onset windows (contact begins between t and t+1) and 300 free-motion windows. Metrics:

- arm-pixel MSE at t+1 and t+2;
- a counterfactual score on pixels where the true frame differs from a "no obstacles" re-simulation, s = MSE(pred, true) / (MSE(pred, true) + MSE(pred, counterfactual)). 0 means the prediction looks like reality and 1 means it looks like the arm passing through the floor.

| condition (onset windows) | arm MSE t+1 | arm MSE t+2 | counterfactual score t+2 |
|---|---|---|---|
| intact | 0.00208 | 0.00431 | 0.129 |
| ablate 32 contact units | 0.00219 | 0.00453 | 0.131 |
| ablate 32 random units (contact layers) | 0.00256 ± 0.00078 | 0.00509 ± 0.00093 | 0.146 ± 0.021 |
| ablate 32 acceleration units | 0.00252 | 0.00470 | 0.132 |
| ablate 32 random units (acceleration layers) | 0.00228 ± 0.00014 | 0.00501 ± 0.00046 | 0.146 ± 0.018 |

On free-motion windows, the t+2 arm MSE is 0.00401 intact, 0.00418 with contact ablation and 0.00491 with acceleration ablation.

**Figure `video_causal.png`.** Arm-pixel MSE at t+2 on contact-onset windows (left) and free-motion windows (middle), and the counterfactual score at t+2 on onset windows (right; dotted line at 0.5 = halfway to "no obstacles"). Bars: intact, contact units ablated, random units from the contact layers (mean ± s.d. over 5 draws), acceleration units ablated, random units from the acceleration layers.

**Figure `video_causal_examples.png`.** Three contact-onset windows (rows). Columns: input frame t, true t+2, predicted t+2, predicted t+2 with the contact units ablated, and the counterfactual t+2 with obstacles removed. Then come the differences (ablated − intact) and (counterfactual − true); red = brighter, blue = darker.

**Interpretation (negative).** Ablating the contact units barely changes the predictions: +5% arm MSE, with the counterfactual score unchanged at 0.13. It hurts *less* than ablating random units from the same layers. The model does not route "the floor stops the arm" through these units in a way that single-set mean ablation reveals. The information is either redundant (contact is linearly decodable at R² 0.7 even at random init) or distributed. Ablating the acceleration units raises t+1 error by 21% on onsets and 27% on free motion, more than random units at t+1 (+10%). At t+2 the effect is within the random-ablation spread.

## 8. Training dynamics and activation bands

**Method.** (a) For every checkpoint we ran the census on 4.8k test windows (stride 12) and recorded the fraction of live units labelled clean per family. (b) We tracked 3 units chosen at the final checkpoint, recording the R² with their top-3 final variables and their top clips at each checkpoint. (c) At the final checkpoint we split each tracked unit's samples by activation percentile (0–50, 50–90, 90–99, 99–100). For each band we report the variables with the largest standardized mean shift and the within-band binned R².

**Figure `video_dynamics.png`.** Left: fraction of live units that are clean (best binned R² ≥ 0.5), stacked by family, over training (log x; step 0 plotted at 1). Right three panels: binned R² of three tracked units with their top-3 final variables at each checkpoint.

**Figure `video_dynamics_clips.png`.** Top-3 test clips of each tracked unit (rows) at five checkpoints (column groups), with the activation map overlaid. The best variable and its R² at that checkpoint are under each group.

**Figure `video_bands.png`.** For each tracked unit, the mean (z-scored) of its top-3 variables within each activation-percentile band.

**Results.**

- The clean fraction is 6% at init and stays flat to step ~355. It rises between ~700 and 4,500 steps (8% → 22%) and plateaus at 24% from 8k steps. Validation error was still falling after the census plateaued (arm MSE 0.0032 at 8.4k → 0.0019 at 30k).
- Tracked units acquire their final tuning between 1k and 5k steps. `enc2.342` changes from a weak cos q₁ unit (R² 0.34) to a hand-height unit (0.90). `tmp3.406` goes from sin q₁ to kinetic energy (0.50 at 1.3k) to q̇₁ (0.88 at 8k). `tmp3.48` switches from hand x to kinetic energy (R² 0.83 at 4.5k) and then to contact torque (0.62 at 30k).
- **Bands.** `enc2.342` is monotonic: hand height rises steadily from the 0–50 band to the 99–100 band (z −0.69 → +2.5). `tmp3.406`'s top bands are not about velocity at all. The 90–100% bands are dominated by *floor contact* (z +1.2 to +1.6), and its signed q̇₁ mean is 0 in every band. It fires most when the arm rests on the floor. Its high R² for q̇₁ comes from the stillness peak, and this unit is better described as "arm at rest, often on the floor". For `tmp3.48`, contact dominates every band above the median.

## Caveats

- **One seed, one size.** The planned 2×-width variant (does the census get cleaner with capacity?) was not run in this pass.
- **bf16 autocast** was used for the training forward pass on CUDA, while the Mac 4k-step run was fp32. All validation and analyses ran in fp32.
- **Convergence** is a plateau under a cosine schedule that reaches lr ≈ 0. A longer schedule or larger model would likely lower the error further.
- **Binned R² rewards narrow peaks.** Several "clean" velocity and energy units are stillness detectors whose R² comes from a sharp peak at zero (sections 3 and 8). Family counts for velocity and energy overstate how much signed velocity is coded.
- **Aliasing.** Shoulder gravity torque is a function of cos q₁, and potential energy of the heights. Units labelled "gravity" are posture units by another name, and gravity torque is decodable from raw pixels (R² 0.998).
- **The census uses the max over 16×16 positions at frame t.** Units that code something only at a particular location are summarized by their peak.
- **The "acceleration" ablation set** is the top 32 by q̈ R², but most of those units' best variable is kinetic energy. No unit is clean for acceleration.
- **Feature visualization is not bit-reproducible across runs** (robust activations differ by ~5% between reruns), and the scene-palette prior is ours.
- **Contact anticipation:** only one unit (`tmp3.48`) beats the ground-truth clearance-and-speed baseline, and only by 0.03 AUC.
- The accuracy numbers in "Model and accuracy" use the analysis test set (19,600 windows). The convergence numbers use the 800-window validation set from training, whose arm mask differs slightly.
