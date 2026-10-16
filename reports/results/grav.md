# Gravity-vector extraction in a dynamics transformer

**Question.** We train a model only to predict the next frame of a two-link arm swinging in a vertical plane. Gravity points in a random direction θg in each episode, and nothing in the input says which way. Does the model work out which way is down? If it does, how do its neurons encode the direction, and does the network actually use that encoding?

**Short answer.** Yes. The model infers gravity from about two frames of motion (40 ms), and after that it predicts as well as a control model that is given θg as an input. The direction is carried in the **upper-arm (body) frame, θg − q₁**, not in the world frame. A linear readout of world-frame θg is barely better than chance, while the body-frame readout reaches about 11°. The code is a distributed, low-dimensional rotating plane. It is **not** a population of sharply tuned "head-direction"-like cells: only 3 of 2048 MLP units have R² ≥ 0.3 for any gravity angle. Overwriting that body-frame plane at block 4 makes the open-loop rollouts fall toward the steered direction. The same edit in a random plane, or in the world-frame plane, does not. A hidden constant torque that imitates a rotated gravity fools the model by exactly the amount it fools an ideal inverse-dynamics observer, with slope 1.00.

Reproduce with:
`analysis/grav_sim.py` → `models/grav_train.py {incontext,given,h1}` (GRAV_STEPS=30000) → `analysis/grav_probe.py` → `grav_tuning.py` → `grav_equivariance.py` → `grav_steer.py` → `grav_conflict.py` → `grav_magnitude.py`.
All numbers below are also in `reports/results/grav.json`.

---

## 0. Data and models

**Data.**
- Main data: `data/grav`, 1200 episodes × 150 frames at 50 Hz, no obstacles, with θg ~ U(−π, π) per episode and three controllers (passive, sines, reach).
- Extra data (`data/grav_extra`, generated with the unchanged `sim/arm.py` via `analysis/grav_sim.py`):
  - 9000 additional training episodes from the same distribution.
  - 1000 held-out "analysis" episodes, used for every probe and neuron analysis.
  - 600 episodes with |g| ~ U(4, 16) m/s², used for the out-of-distribution magnitude test.
- Probes are fit on analysis episodes [0, 700) and evaluated on [700, 1000).
- We checked that rotating the scene (θg + α, q₁ + α) reproduces the simulated trajectories to within 5·10⁻⁷ rad. This rotation is therefore an exact symmetry of the data.

**Models** (`models/grav_model.py`, `grav_train.py`).
- Architecture: pre-LN causal transformer, residual width 128, 4 blocks, 4 heads, GELU MLPs of width 512 (4×), learned positional embeddings, 0.80 M parameters. It is trained on random 64-frame windows.
- Input per frame: sin q, cos q, q̇/[6, 14], τ/[11, 4.5].
- Target: z-scored (Δq, Δq̇) to the next frame.
- Training: AdamW (lr 1e-3, warm-up 1k steps, cosine decay, weight decay 0.05), batch 256, 30k steps for all three variants. Test loss had flattened by about 20k steps.
- Variants:
  - **in-context**: θg is never given.
  - **given**: (cos θg, sin θg) is appended to every frame.
  - **history-1**: attention is masked to the current frame only, so θg cannot be inferred from motion.

**Prediction error vs context length.** Test set: `data/grav` episodes 1000–1199, windows starting at frames 0, 43 and 85 (600 windows).

| model | median SE, frame 1 | frame 2 | frame 5 | frames 33–64 | mean SE, frames 33–64 |
|---|---|---|---|---|---|
| in-context | 8.4e-3 | 6.5e-5 | 3.2e-5 | 1.9e-5 | 2.24e-3 |
| θg given | 6.4e-5 | 2.0e-5 | 1.4e-5 | 1.2e-5 | 2.24e-3 |
| history-1 | 7.9e-3 | 8.0e-3 | 6.2e-3 | 4.8e-3 | 1.09e-2 |

- At frame 1 the in-context model is exactly as bad as the history-1 model. One more frame cuts its median error by 130×.
- After that a gap of about 1.6× to the oracle remains in the median. The mean errors over frames 33–64 are identical to the oracle's. The mean is dominated by rare, fast, chaotic elbow spins that neither model predicts.
- History is worth about 250× in median error.

> **Figure `grav_context_loss.png`.** Next-frame squared error (averaged over the 4 z-scored targets) vs number of frames observed, for the in-context model (blue), the model given θg (green) and the history-1 model (orange). Left: mean over 600 test windows. It is dominated by a few windows with extreme elbow velocities, which makes it noisy, and the blue and green curves overlap after about 5 frames. Right: median over windows. Without being told gravity, the in-context model starts at the history-1 level and drops 130× after a single extra frame.

---

## 1. How fast is gravity inferred, and in which frame?

**Method.**
- At every layer (embedding and after each block) and every context position, we fit a separate ridge probe (λ = 0.01 on standardized features) from the residual stream to a unit vector. The angle of that vector is either θg (world/camera frame) or θg − q₁ (gravity relative to the upper arm at that frame).
- Error is the absolute angular error on held-out episodes. Chance is 90°.
- **Ideal observer.** Least-squares inverse dynamics. M(q)q̈ + c(q, q̇) + g[cos θ·G↓(q) + sin θ·G→(q)] = τ − bq̇ is linear in g(cos θ, sin θ).
  - The observer uses MuJoCo's M, c and G with q̈ from a central difference of the *observed* q̇ and the torque averaged across the two intervals.
  - It uses only frames available at that context position (from frame 3 on).
- **Nonlinear check.** MLP probes (two layers of 256 units) on late positions.

**Results.**
- **Body frame, linear.** The θg − q₁ probe improves from 64° at frame 1 (a posture prior) to 29° at frame 2, 20° at frame 10, and **11.0° median (21° mean) at frame 64** after blocks 3–4.
- **World frame, linear.** World-frame θg is close to undecodable: 46–53° median at every layer and position. The history-1 model is no better for world θg (50°), and the plain embedding decodes body-frame θg at 48°.
- **Nonlinear probes.** Late positions decode world θg to 5.0° and body-frame θg to 3.2° (block 3). The world-frame information is present, but only nonlinearly, as a product of the body-frame code with the posture code (cos θg = cos φ cos q₁ − sin φ sin q₁).
- **Ideal observer.** 0.57° median at frame 3 and 0.26° at frame 64. Passive episodes: 0.02°. The physics is nearly unambiguous within 3 frames.
- **The model's readable code is much coarser than the physics allows.** Late-context errors by controller: passive 5°, reach 17°, sines 20°. This is not costly for prediction, because the model only needs the gravity *torque*, and its prediction error is close to the oracle's.

> **Figure `grav_probe_time.png`.**
> - Left: median angular error of a linear probe for world-frame θg vs frames observed, one line per residual layer (light to dark = embedding to block 4). Dashed orange: history-1 control (best layer). Dotted: ideal inverse-dynamics observer using finite-difference accelerations. Grey line: chance (90°).
> - Middle: the same for gravity in the upper-arm frame, θg − q₁. Here the residual stream is linearly informative after 2 frames and improves through about 30 frames.
> - Right: body-frame probe at block 4, split by controller. The dotted observer curves for passive episodes fall below 0.1° and leave the plotted range.
> - Both axes are logarithmic.

**Interpretation.** The quantity that matters for dynamics, gravity torque, depends only on θg − q₁ and q₂. The network keeps exactly that relative variable as a linear 2-D feature, much like an otolith mounted on the upper arm, and never needs to form an allocentric "down".

---

## 2. Population code in MLP neurons: no head-direction cells

**Method.**
- Data: all 2048 MLP hidden units (post-GELU, 4 layers × 512) at positions ≥ 24 (every 3rd frame, 28k samples).
- For each unit we compute binned R² (36 circular bins) against θg (world), θg − q₁ (upper arm), θg − q₁ − q₂ (forearm), q₁ and q₂.
- The best frame's mean tuning curve is fit with a von Mises curve, a + b·exp(κ(cos(x − μ) − 1)). A unit counts as "weakly tuned" if best-frame R² ≥ 0.1 and the fit R² ≥ 0.8, and "tuned" if best-frame R² ≥ 0.3.
- **Gain-field control.** Cross-validated R² of a regression on kinematics (posture harmonics, q̇, q̇², τ, posture × q̇) with and without gravity harmonics × posture harmonics. Their difference is the "gravity R² beyond kinematics". The interaction terms make this test independent of frame.

**Results.**
- **Almost no tuned units.**
  - Only **3 / 2048** units are tuned at R² ≥ 0.3, all in MLP layer 1 and all in the upper-arm frame.
  - **84** are weakly tuned: 59 in the upper-arm frame, 25 in the forearm frame, **0** in the world frame.
  - Among tuned units, the median R² is 0.33 in the upper-arm frame and 0.02 in the world frame. No tuned unit prefers the world frame.
- **Broad tuning.** The tuning is very broad: the median FWHM of weakly tuned units is 360°. The curves are rectified-cosine-like rather than bump-like (example units: FWHM 255–360°).
- **Clustered preferred directions.** Preferred directions are not uniform. They cluster along an axis at θg − q₁ ≈ 180° and ≈ 0° (gravity perpendicular to the upper arm, i.e. maximal shoulder gravity torque of either sign). The Rayleigh test for unimodal clustering gives p = 0.10, which is weak because the distribution is bimodal.
- **Gravity explains little single-unit variance.** Gravity adds a median of only 0.7–1.4 % of variance beyond kinematics per unit, and only 60 units gain more than 0.1. Single units mainly track posture, especially the elbow angle q₂ (median R² 0.19 in layer 0).
- **The code is distributed.** A few broadly tuned units sit inside a 2-D linear subspace of the residual stream (section 4). This is the opposite of the sparse, sharply tuned, uniformly tiling head-direction-cell population.

> **Figure `grav_tuning_examples.png`.** The 8 MLP units most strongly tuned to gravity direction, in their best frame (orange = upper-arm frame θg − q₁, green = forearm frame θg − q₁ − q₂). Polar plots show mean activation (minus its minimum) against angle, with 0 = gravity along the reference link's "down" (θ = 0 is straight down in world coordinates). Dashed lines show the von Mises fit. Grey shows the same unit plotted against world-frame θg, which is almost flat in every case. Titles give layer, unit index, binned R² and FWHM.

> **Figure `grav_population.png`.**
> - Top: preferred-direction histograms of the weakly tuned units (R² ≥ 0.1, von Mises fit R² ≥ 0.8) for each frame. The world frame has none.
> - Bottom left: tuning widths (FWHM). Almost all are ≥ 255°.
> - Bottom middle: distribution over all units of best-frame marginal R² (orange) and gravity R² beyond kinematics (green), on a log count axis. Dashed lines mark the 0.1 and 0.3 thresholds.
> - Bottom right: per-unit R² against world θg vs against θg − q₁, coloured by layer. Every unit lies on or above the diagonal.

---

## 3. Gravity-torque units vs gravity-direction units

**Method.** For each unit we compute binned R² (30 quantile bins, `common.binned_r2`) against the gravity torque acting at the shoulder and at the elbow (−g_bias), which depend jointly on θg − q₁ and q₂. We compare it with R² against θg alone.

**Results.**
- No unit tracks world θg (max R² < 0.08). Many units carry some gravity-torque information: up to R² = 0.43, with 3 units ≥ 0.3, all in layer 1.
- Shoulder torque is tracked about 2× better than elbow torque in every layer: mean R² 0.04 vs 0.02 in layer 1. This matches the preferred-direction cluster at "gravity perpendicular to the upper arm".
- The mean R² against the full (θg − q₁, q₂) state (0.14–0.25) is driven mainly by q₂: posture alone gives 0.10–0.24.
- Gravity-related selectivity peaks in MLP layers 1–2, after the first attention layer has aggregated history.

> **Figure `grav_torque_units.png`.**
> - Left: per-unit R² against gravity torque (better of shoulder and elbow) vs R² against world θg, coloured by layer. Units that carry gravity information do so as torque or body-frame direction, never as world direction.
> - Right: mean single-unit R² per MLP layer. Bars from left to right: θg, θg − q₁, shoulder gravity torque, elbow gravity torque, gravity beyond kinematics (cross-validated, clipped at 0), q₂, and joint (θg − q₁, q₂). Note: the θg and q₂ bars share the same blue. Read them by position.

---

## 4. Equivariance under rotating the whole scene

**Method.**
- Take 200 held-out windows. Rotate each by α ∈ {0°, 15°, …, 345°}: q₁ + α at every frame and θg + α. q₂, velocities and torques are unchanged.
- Decode with pooled probes (block 4, positions ≥ 16) in each frame. A body-frame code should be *invariant* under this rotation and a world-frame code should be *equivariant*.
- Per MLP unit, the rotation-variance index is Var_α(a) / Var_total(a).

**Results.**
- **Body-frame readout is invariant.** The raw body-frame readout (θg − q₁) changes by a median of only **1.6°** across rotations (90th percentile 6.4°). Adding q₁ back gives a decoded world gravity that rotates with α to within 1.6° (median).
- **World-frame readout is not.** The world-frame probe tracks α only on average: median 13° deviation, 90th percentile 51°, absolute error 57°.
- **The residual stream is 95–96 % rotation-invariant.** This holds after blocks 1–4 (86 % at the embedding, where only sin/cos q₁ rotate). The α-dependent remainder is 99.8 % first harmonic, i.e. a single rotating 2-D plane.
- **MLP units are almost invariant.** Median rotation-variance index is 0.004–0.021, and 0.010 for upper-arm-tuned units. The handful of units with a high index (> 0.3, mostly layers 0–1) are those where R²(world) is slightly above R²(upper-arm). Both R² values are small for these units, so they most likely carry world-frame posture (q₁) rather than gravity; we did not check this further.

> **Figure `grav_equivariance.png`.**
> - Left: decoded rotation of gravity vs imposed scene rotation α (median and 5–95 % band over 200 windows × 48 positions). Blue: world-frame probe. Green: upper-arm-frame probe plus q₁. Orange: raw upper-arm readout, which stays at 0 as predicted for a body-frame code. Dashed: identity.
> - Middle: fraction of residual-stream variance that is invariant to α (orange), and share of the α-dependent variance in the first harmonic (blue), per layer.
> - Right: per-unit rotation-variance index vs R²(world θg) − R²(θg − q₁), coloured by layer.

---

## 5. Causal steering of the gravity representation

**Method.**
- **Scenario.** 80 "hold then release" episodes. A PD controller moves the arm to a nearby posture and holds it under the true θg for 32 frames (|q̇| < 0.3 rad/s at release). The motors are then switched off and the model rolls out open-loop for 25 frames (0.5 s), feeding back its own predictions with τ = 0.
- **Intervention.**
  - At residual layer L and every position, replace the component of h in the gravity-encoding plane B (a least-squares encoding map h ≈ μ + B·u, fit on analysis windows, positions ≥ 16) with the encoding of a steered direction θ′ = θg + δ, δ ∈ {0°, 30°, …, 330°}.
  - In the body frame the target at each position is θ′ − q₁(t), using the model's own rolled-out q₁.
  - The plane holds only 1.9 % of residual variance at block 4.
- **Controls.** A random 2-D plane with matched column norms, and the world-frame encoding plane.
- **Readouts.**
  1. *Behavioural gravity*: the ideal observer run on the model's rolled-out trajectory.
  2. The direction of hand displacement over the first 80 ms, compared with a MuJoCo rollout from the same state under θ′.

**Results.**
- **Unsteered baseline.** Without steering, the behavioural gravity of the model's rollouts is within 24° of the truth (median). Open-loop rollouts accumulate error: hand RMSE 0.20 m over 0.5 s. MuJoCo itself gives 0.08° through the same observer.
- **Block 4, body frame.** The behavioural gravity follows the steered direction for every δ. The median error relative to θ′ is **19.5°**, flat across δ (17–24°), i.e. at the unsteered noise floor. The rollout's gravity can be rotated all the way round, including δ = 180°.
- **Earlier layers.** Steering at blocks 2 and 3 also works, with median errors of 28° and 27°. It is weaker for δ near 180° (39–56°). Block 1 gives 48°.
- **Hand fall direction.** The first-80-ms hand direction matches MuJoCo under θ′ to within 28° (median over δ) at block 4, against 61° for the random plane.
- **Controls fail.**
  - The random plane leaves gravity unchanged: median 59° away from θ′, mostly near the unsteered behaviour.
  - Writing the *world-frame* encoding plane does not steer at all (85°). The direction in which world θg is linearly correlated with the residual stream is not what the network reads.

> **Figure `grav_steer.png`.**
> - Left: behavioural gravity of the rolled-out trajectory (ideal-observer estimate, relative to true θg) vs steered offset δ. Blue lines: body-frame steering at blocks 1–4 (light to dark). Orange squares: random-plane control at block 4. Pink triangles: world-frame plane at block 4. Dashed: identity. Values are medians over 80 scenarios. Controls are plotted wrapped to (−180°, 180°], so a flat line near 0 means no effect.
> - Right: median absolute difference between the model's hand-displacement direction over the first 80 ms after release and that of a MuJoCo rollout under θ′, vs δ.

> **Figure `grav_steer_strip.png`.** Three release scenarios (rows) steered by δ = 0°, 90°, 180° and 270° (columns) at block 4 (body frame). The arm is drawn every 100 ms after release (blue upper arm, orange forearm, faint to opaque). Black: hand path predicted by the steered model. Dashed: MuJoCo under the steered gravity. Grey arrow: true gravity. Green arrow: steered gravity. The model's arm falls toward the green arrow, not the grey one. Its paths differ in detail from MuJoCo because of compounding open-loop error.

---

## 6. Cue conflict: is an unseen constant torque mistaken for gravity?

**Method.**
- **Construction.** For true gravity θg and a decoy direction θ′ = θg + δ (δ ∈ [−90°, 90°]), let q* be the posture hanging straight along θ′. Apply a constant joint torque τh = g_bias(q*; θg). This makes q* a static equilibrium under the true gravity, so near q* the arm looks exactly as it would under gravity θ′.
- The passive arm starts at rest from q* plus a small (0.15 rad) or large (1.2 rad) perturbation, for 40 base angles θg.
- **Hidden condition:** τh is applied in MuJoCo, but the model is told τ = 0. **Visible condition:** the model is given τ = τh.
- We read gravity with the block-4 body-frame probe, converted to the world frame, as a circular mean over positions ≥ 16. We compare it with the ideal observer under the same assumption about τ.

**Results.**
- **Torque hidden.**
  - The model reports the decoy. The slope of decoded offset vs δ over |δ| ≤ 45° is **1.00** for small swings (observer 1.01) and **1.03** for large swings (observer 1.09). For small swings and |δ| ≤ 60° the decoded offset is within 0.2° of δ, and the spread across episodes is < 1.2°.
  - The prediction error stays at the unperturbed level (≈ 5·10⁻⁶) for small swings. The illusion is complete: the model's whole world model behaves as if gravity were θ′.
  - With large swings the predictions get up to 100× worse as |δ| grows. The posture-dependence mismatch is then visible in the motion, but the gravity estimate is not revised away from the decoy. The ideal observer behaves the same way: the least-squares best-fit gravity is still close to θ′.
- **Torque visible.** The model correctly reports the true gravity: slope −0.06 (small) and 0.06 (large). Offsets are < 5° for |δ| ≤ 45° and grow to about 20° at |δ| = 90° (large holding torques at unusual postures).
- **Where the ambiguity sits.** The model does not confuse applied torque with gravity when the torque is observed. When it is not observed, the model is fooled exactly as far as the physics is ambiguous. In our construction that is almost entirely, because over a ±1.2 rad swing a constant torque is hard to tell apart from a rotated gravity.

> **Figure `grav_conflict.png`.**
> - Left: decoded gravity offset (decoded − true θg) vs decoy offset δ when the constant torque is hidden. Blue: small swing (0.15 rad). Orange: large swing (1.2 rad). Shading: interquartile range over 40 episodes. Dotted: ideal observer. Dashed: "fully fooled" identity.
> - Middle: the same with the torque given to the model as input.
> - Right: next-frame prediction error (z-scored MSE, positions ≥ 16) for each condition.

---

## 7. Supplement: gravity magnitude (out of distribution)

**Method.** The model was trained only at |g| = 9.81. We tested it on 600 episodes with |g| ~ U(4, 16) m/s², all at block 4, positions ≥ 32:
- Linear (ridge) and MLP probes for |g|, trained on 400 episodes and tested on 200.
- The direction probe fit at 9.81.
- The norm of that direction probe's output.

**Results.**
- **Direction.** The direction probe still works: median 13–16° error for |g| ∈ [6, 16] and 21° for [4, 6], vs 11° in distribution.
- **Prediction.** Next-frame error does not rise with |g − 9.81| (per-episode medians 3–13·10⁻⁵ in every bin). The in-context computation therefore generalizes to unseen gravity strengths.
- **Magnitude.** |g| is **not linearly decodable** (R² = 0.01), but an MLP probe recovers it (R² = 0.80), and the *length* of the direction-probe output correlates with |g| (r = 0.80). This fits a code in which gravity is a 2-D vector in a plane, with direction as angle and strength as radius. (Ideal observer: R² = 0.997.)

> **Figure `grav_magnitude.png`.**
> - Left: per-episode median direction error of the 9.81-trained body-frame probe vs |g|. Dashed: training value.
> - Middle: decoded vs true |g| for linear (green) and MLP (purple) probes on 200 held-out episodes.
> - Right: per-episode next-frame error vs |g|.

---

## Caveats

1. **Data.** The model was trained on 10 000 episodes: data/grav train plus 9 000 extra from the same generator. All neuron analyses use a separate generated held-out set. One seed and one architecture were trained, and no hyperparameter sweep was done.
2. **Probe readability vs physics.** Linear probes reach 11° in the body frame, far worse than the ideal observer (0.3°), while prediction matches the oracle model. The readable 2-D plane is therefore a coarse summary; the precise information is nonlinear (MLP probes: 3°). "Gravity is encoded in the body frame" refers to the *linear* geometry and to what steering uses. World θg is recoverable nonlinearly.
3. **Steering.**
   - The steering evaluation relies on open-loop rollouts, which drift (unsteered behavioural error 24°, hand RMSE 0.2 m over 0.5 s). Steering accuracy is therefore bounded by this floor rather than measured against perfect physics.
   - The behavioural readout uses finite-difference accelerations of model trajectories.
   - The intervention overwrites a 2-D plane at all positions using a per-position target computed from q₁, so it is an informed intervention and not a fixed vector addition.
4. **Single-unit statistics.** These depend on thresholds (R² ≥ 0.1/0.3, binning, positions ≥ 24). The near-absence of world-frame or sharply tuned units holds at every threshold we looked at, but the counts are not robust numbers. The gain-field regression basis is hand-chosen, and in later layers it explains only about 20 % of unit variance, so "gravity beyond kinematics" may be underestimated for units with nonlinear kinematic selectivity.
5. **Cue-conflict scope.** The cue-conflict construction is near-degenerate by design (equilibrium at the decoy posture). Other torque patterns, such as torque at a single joint, were not tested.
6. **Mean vs median error.** Mean test errors are dominated by rare high-speed elbow spins (heavy tails). We report medians alongside means.
7. **Magnitude.** The magnitude results are out-of-distribution probes of a model never trained with varying |g|. No model was trained on varied |g|.
