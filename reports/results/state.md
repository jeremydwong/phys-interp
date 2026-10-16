# Results: a forward-dynamics MLP (candidate C)

All numbers are on the 400 held-out test episodes (59,600 transitions) unless stated otherwise. Key numbers are in `reports/results/state.json`. To reproduce everything, run `uv run python analysis/state_run_all.py` (add `--train` to retrain).

## Model and accuracy

**Setup.** The model takes the arm state at frame *t*, as sin q, cos q, q̇/10 and τ_act/τ_max (8 inputs). It is not given acceleration, gravity or contact information. It predicts the standardized one-step change [Δq₁, Δq₂, Δq̇₁, Δq̇₂] over 20 ms. The architecture is a linear embedding into a 256-wide residual stream, then 4 pre-LayerNorm blocks (256→1024 ReLU→256), a final LayerNorm and a linear head (2.1 M parameters). Training used AdamW (lr 3·10⁻⁴, weight decay 0.05 on matrices, betas 0.9/0.98), batch 1024, 500 warm-up steps and a cosine schedule for 60k steps (≈206 epochs). Checkpoints were saved at 12 log-spaced steps (0, 30, 64, …, 60000). The step-0 checkpoint is the random-initialization control.

A first run with lr 10⁻³ killed 85–90% of the ReLU units in blocks 2–4 during steps 300–3000. That run was discarded, and the lower learning rate was used for every result below.

**Accuracy (test R² for Δq₁, Δq₂, Δq̇₁, Δq̇₂):**

| predictor | Δq₁ | Δq₂ | Δq̇₁ | Δq̇₂ |
|---|---|---|---|---|
| **trained MLP** | **0.999** | **0.998** | **0.965** | **0.971** |
| linear regression on the same inputs | 0.978 | 0.957 | 0.096 | 0.080 |
| rigid-body physics, no contact (MuJoCo, τ held for 20 ms) | 0.981 | 0.943 | 0.172 | −0.317 |
| full simulator, τ held for 20 ms (predictability ceiling) | 1.000 | 1.000 | 0.992 | 0.991 |
| random-init network | <0 | <0 | <0 | <0 |

Contact occurs at *t* or *t*+1 on 42.8% of test transitions. On contact-free steps the MLP reaches 0.987 / 0.974 for Δq̇. On steps involving contact it reaches 0.933 / 0.969. The final test MSE is 0.0164 (standardized). It was still falling slowly at the end of training (0.0184 at step 36k), so this is near a plateau but not fully converged.

**Dead units.** Even at lr 3·10⁻⁴, only 780, 461, 382 and 321 of 1024 units are live (active on >1% of frames) in blocks 1–4. That is 1,944 of 4,096 units. Units die mainly between steps ≈600 and 6,000 (Fig. `state_dynamics_census.png`). All unit-level analyses below use live units only.

## 1. Census: how many neurons have one job?

**Method.** For each live unit we computed the binned R² of its activation against each of the 32 targets (30 quantile bins, piecewise-constant fit). A unit is *clean* if its best R² is ≥ 0.5, *partial* if it is 0.25–0.5, and otherwise *unexplained*. A unit is a *conjunction* if the best 2-D binned fit (10×10 quantile cells, over pairs from its top-6 variables) beats the best 1-D fit by more than 0.15.

The elbow gravity torque is mathematically identical to cos(q₁+q₂). Likewise, elbow height is identical to sin q₁ (|r| > 0.99). Units whose best fit is one of these aliased variables are flagged; this applies to 1.3% of live units.

| | live units | clean | partial | unexplained | conjunction | clean **on a derived quantity**¹ |
|---|---|---|---|---|---|---|
| trained (all blocks) | 1,944 | **21.3%** | 37.6% | 41.0% | 56.1% | **7.5%** |
| random init | 4,079 | 25.0% | 63.3% | 11.8% | 72.4% | 4.6% |

¹ A derived quantity is one of acceleration, gravity, interaction/Coriolis, contact or energy, i.e. a variable that is not trivially present in the input.

Per block, the trained clean fractions are 23.8%, 29.9%, 11.0% and 15.3%. The derived-quantity clean fractions are 4.5%, 12.6%, 5.5% and 10.0%. The most common clean labels are as follows:

- **Block 1:** joint-angle sin/cos, plus 20 shoulder-gravity units.
- **Block 2:** hand x (64 units) and shoulder gravity torque (39 units).
- **Block 3:** hand x, contact and gravity.
- **Block 4:** kinetic energy, shoulder velocity, Coriolis and contact torque.

**Figure `state_census.png`.** Best single physical variable for each live unit, per block, for the trained network and the random-init control. Solid segments are clean units (best binned R² ≥ 0.5), coloured by the family of the best variable. Light tints are partial units (0.25–0.5) of the same family. Hatched segments are unexplained (< 0.25). The percentage above each bar is the clean fraction. The x labels give live units (trained / random) out of 1024.

**Interpretation (largely negative).** The trained network is *not* cleaner than chance by this metric. Because the input is only 8-dimensional, every unit of a random network is a smooth function of a few angles and velocities. A quarter of random units are therefore already "clean", mostly on posture variables.

Training changes *what* units are clean for, not *how many*. The share of units cleanly tuned to a computed physical quantity rises from 4.6% to 7.5% overall, and to 12.6% in block 2, mainly through shoulder-gravity units. At the same time the late blocks become much less interpretable: 51% (block 3) and 57% (block 4) of their live units are unexplained by any single target.

## 2. Linear probes of the residual stream

**Method.** We fit ridge probes (λ = 10⁻² on standardized features) on 80k training frames and report test R² for the raw input, the embedding and the output of each block.

**Figure `state_probes_heatmap.png`.** Test R² (×100) of linear probes for each of the 32 targets (columns, coloured by family) at each depth (rows). Top: trained network. Bottom: random-init control.

| target (block-4 residual) | trained | random init | raw input |
|---|---|---|---|
| q̈₁ / q̈₂ (instantaneous, clipped) | 0.74 / 0.61 | 0.59 / 0.47 | 0.17 / 0.15 |
| gravity torque, shoulder / elbow | 0.99 / 0.99 | 1.00 / 0.96 | 0.96 / 0.24 |
| interaction torque, shoulder / elbow | 0.60 / 0.67 | 0.50 / 0.52 | 0.10 / 0.13 |
| Coriolis/centripetal, shoulder / elbow | 0.82 / 0.79 | 0.59 / 0.58 | 0.10 / 0.17 |
| any contact / log floor force / log wall force | 0.84 / 0.83 / 0.75 | 0.70 / 0.71 / 0.59 | 0.30 / 0.25 / 0.17 |
| contact torque, shoulder / elbow | 0.53 / 0.45 | 0.44 / 0.33 | 0.23 / 0.23 |

**Interpretation.**

- **State:** yes. Joint angles, velocities and motor torque are decodable at R² ≥ 0.98 at every depth. This is trivial, because they are inputs.
- **Gravity torque:** yes, but this is weak evidence of computation. Shoulder gravity torque is already linear in the input (R² 0.96), and both versions are decodable after one random block.
- **Coriolis, interaction torque and contact:** this is where training adds the most over random features (+0.1 to +0.2 R²). The gains grow gradually with depth.
- **Contact torques:** remain the least decodable quantities (≤ 0.53).

The low numbers for q̈ and interaction torque come mostly from contact spikes; see the next section.

## 7. Is acceleration represented as a sum of torque components?

**Method.** We decomposed the true acceleration exactly as q̈ = M⁻¹(τ_act − g − c + τ_contact − bq̇) = a_act + a_grav + a_Cor + a_contact + a_damp. We then probed each contribution, the net torque Mq̈, q̈ itself and the step-mean acceleration Δq̇/Δt that the model is trained on. To avoid contact spikes dominating, probes for the non-contact terms were fit and evaluated on contact-free transitions (57% of test frames). a_contact was probed on all frames.

**Figure `state_accel_probes.png`.** Linear-probe test R² for the acceleration and its physical components at each depth. Left: shoulder. Right: elbow. Solid lines are the trained network and dashed lines the random-init control. Non-contact terms are evaluated on contact-free transitions only.

**Results (trained vs random, block 4, shoulder / elbow):**

| quantity | trained | random |
|---|---|---|
| step-mean acceleration | 0.96 / 0.95 | 0.75 / 0.67 |
| instantaneous q̈ | 0.90 / 0.85 | 0.77 / 0.77 |
| net torque | 0.91 / 0.91 | 0.76 / 0.80 |
| Coriolis/centripetal part | 0.86 / 0.84 | 0.73 / 0.73 |
| motor part | 0.95 / 0.93 | 0.95 / 0.96 |
| gravity part | 0.99 / 0.98 | 0.99 / 0.97 |
| contact part (all frames) | 0.72 / 0.65 | 0.57 / 0.53 |

On contact-free frames the Coriolis/centripetal part carries most of the acceleration variance (79% / 71%). The motor part carries 25% / 42% and gravity only 8% / 5%.

**Interpretation.**

- Every component is linearly available, and the sum improves steadily with depth. Training mainly buys the velocity-dependent (Coriolis) term and its combination through M⁻¹. The gravity and motor terms are equally decodable from random features.
- The step-mean acceleration is the only quantity that keeps rising sharply in the last block (0.82 → 0.96 for the shoulder from block 1 to block 4). This is consistent with components being computed in blocks 1–3 and combined late.
- Linear probes cannot test additivity directly, because the probe of a sum equals the sum of the probes. The evidence for decomposition is therefore only that the components are separately decodable before their combination is. The direction edits in §8 give a causal test for the gravity component.

## 3. Neuron gallery

**Figure `state_gallery.png`** (headline figure). Twelve exemplar units, selected automatically as the best unit, by binned R², for each target variable among units whose best variable lies in the same family (falling back to any unit when none does).

- **Left:** unit id (block / index), physical role, binned R² and, where it differs, the unit's actual best variable and best 2-D conjunction.
- **Tuning curve:** mean activation (line) and 10–90% range (band) across 30 quantile bins of the variable. Units on the x axis: gravity, interaction, Coriolis and contact torques in N·m; accelerations in rad/s²; velocity in rad/s; hand height in m; contact force as log(1+N).
- **Five centre images:** the five held-out frames with the highest activation, at most one per episode.
- **Three right images:** the three frames with the lowest pre-activation (the unit's negative pole).

Each frame is the arm pose rendered from the side camera. The upper arm is blue and the forearm orange; the floor is at the bottom and the wall at the right. A trail shows the past 0.12 s of the hand (dark) and elbow (grey), and a red arrow shows 0.1 s of hand velocity. The black number in each image is the activation and the coloured number is the variable's value.

**Interpretation.** The gallery has three kinds of unit.

- **Clean, physically meaningful units.**
  - The shoulder gravity-torque units (L2/826, L2/975; R² 0.86 and 0.85) fire for an outstretched, near-horizontal arm, where gravity's moment about the shoulder is largest (τ_g ≈ −13 N·m). Their negative pole is the arm hanging or folded near vertical, where the torque is near zero or positive.
  - The shoulder-velocity unit L1/321 (R² 0.93) is a rectified linear function of q̇₁.
  - The Coriolis-elbow unit L2/760 (R² 0.63) responds to a fast-swinging, folded arm.
- **Clean only after relabelling.**
  - The "elbow gravity" unit L2/458 is equally well described as a forearm-angle unit, because the two variables are identical. It is also a conjunction (2-D fit with potential energy, 0.86).
  - The best "floor contact" and "wall contact" units are best explained by elbow contact torque. Their examples show the forearm pressed on the floor.
- **Not clean.** The acceleration units (best R² 0.32 and 0.18) have top examples that are contact-induced acceleration spikes, clipped at ±150 / ±300 rad/s².

## 4. Receptive fields and feature visualization

**Method.** We evaluated units on a 90×90 grid of (q₁, q₂) with q̇ = 0 and τ = 0, next to the analytic gravity torque −g(q) from MuJoCo. Poses that intersect the floor or wall are hatched.

**Figure `state_featvis_gravity_rf.png`.** Posture receptive fields at rest (q̇ = 0, τ = 0) for the two best shoulder-gravity units and the two best elbow-gravity units. Each pair is shown beside the analytic gravity torque at the corresponding joint (diverging colour map: red is positive torque, i.e. it pulls toward +q; blue is negative). Activations use a sequential colour map. Hatched regions are poses that penetrate the wall (centred near q = 0) or the floor (near q₁ = −π/2), which never occur in data. The inset value is the binned R² between the unit's map and the analytic map over feasible poses.

**Results.** The shoulder-gravity units' resting receptive fields are rectified copies of the negative lobe of the analytic shoulder-gravity map (grid R² 0.67 and 0.68, vs 0.86 and 0.85 on data). They are therefore genuinely "arm extended toward the wall, gravity pulling down" detectors, not velocity- or contact-dependent artifacts.

The elbow-gravity units have *no* gravity-like receptive field at rest (grid R² 0.03 and 0.10). Their fields are small blobs next to the floor-penetration region. Their data R² comes from a correlation with the forearm angle in the data distribution.

**Feature visualization.** Gradient ascent ran on each unit's pre-activation over valid inputs: angles as free parameters fed through sin/cos, |q̇| ≤ 15 rad/s via tanh, and |τ| ≤ τ_max via tanh. We used 64 restarts × 400 Adam steps (lr 0.05) and kept up to four distinct optima per unit.

**Figure `state_featvis_optimized.png`.** Input-optimized ("superstimulus") arm states for six units. Each row shows up to four distinct local optima, ordered by pre-activation. The text in each image gives the optimized joint velocities (rad/s) and motor torques (N·m), and the red arrow the implied hand velocity. Rows with fewer than four images converged to fewer distinct optima.

**Results.** The optima are interpretable but out of distribution.

- **Shoulder-gravity unit L2/826:** a horizontal arm pointing at the wall (pre-activation 3.81 vs 3.70 maximum in data). Its forearm intersects the wall (clearance −0.22 m).
- **Floor-contact unit L2/834:** a pose pushing the forearm 0.11 m *into* the floor (pre-activation 4.63 vs 3.59 data maximum).
- **"Shoulder acceleration" unit L3/11:** all four optima are at rest (q̇ ≈ 0) with the forearm penetrating the floor. Its strongest drive is therefore "arm inside the floor", where the model expects a large contact reaction.

Contact-related units thus extrapolate their tuning linearly past the obstacle surface, like a penetration-depth detector.

## 5. Segment discrimination

**Method.** For units whose best variable is a gravity (including the forearm-angle alias), Coriolis, interaction or contact-torque quantity with R² ≥ 0.25, we computed SI = (R²_shoulder − R²_elbow)/(R²_shoulder + R²_elbow). Here R² is the binned R² against the shoulder and elbow versions of the same quantity. The two target versions are only weakly related (binned R² 0.18 / 0.09 for gravity; 0.06 / 0.10 for Coriolis).

**Figure `state_segment_selectivity.png`.** Selectivity index between the shoulder (+1) and elbow (−1) versions of the same torque quantity, for torque-tuned units (best binned R² ≥ 0.25). Interaction torque (4 units) is omitted from the plot.

**Results.**

| quantity | units | shoulder-specific (SI > 0.5) | elbow-specific (SI < −0.5) | mixed |
|---|---|---|---|---|
| gravity | 177 | 80.2% | 0.6% | 19.2% |
| Coriolis/centripetal | 119 | 3.4% | 16.0% | 80.7% |
| contact torque | 40 | 0% | 2.5% | 97.5% |
| interaction | 4 | – | – | – |

**Interpretation.** Gravity is represented almost exclusively at the shoulder. Shoulder gravity torque is a nontrivial combination of cos q₁ and cos(q₁+q₂) that the model must build. Elbow gravity, by contrast, is a single input-like feature (cos(q₁+q₂)) that the model never needs to isolate.

Coriolis units lean toward the elbow but are mostly mixed. Contact-torque units are mixed, which matches the physics: a single forearm contact produces torque at both joints.

## 6. Contact

**Method and constraints of the data.** The upper arm cannot reach either obstacle (reach 0.545 m vs wall at 0.68 m and floor 0.86 m below the shoulder). All 1,082 test contact onsets are therefore forearm contacts, and an upper-arm/forearm split does not exist. We split onsets instead by surface and impact speed:

- **Floor:** 691 onsets, median hand speed normal to the surface 2.7 m/s.
- **Wall:** 391 onsets, median 3.9 m/s.

As controls we used near misses: local minima of clearance in (0, 2 cm] with no contact within ±6 frames (112 floor, 71 wall).

Because the model predicts *t* → *t*+1, a unit that anticipates a bounce must fire at frame −1, the last frame before contact. We therefore also selected "impact units" functionally, by the d′ between frame −1 activations and near-miss activations. 57 units have d′ > 0.5, 50 of them in block 4. No unit reaches d′ > 1, because slow impacts are weak.

**Figure `state_contact_psth.png`.** Event-aligned activations around contact onsets. Columns are surface × impact speed (split at the per-surface median). Top row: true log(1+normal force). Rows below:

- three units selected by census R² with floor force, wall force or elbow contact torque;
- three "impact units" selected by d′ between the pre-impact frame and near-miss frames.

Solid red lines are the mean ± s.e.m. aligned to the first contact frame (red vertical line). The dashed grey line is the mean aligned to the clearance minimum of near misses (< 2 cm, no contact). The dotted line is the unit's mean on contact-free frames.

**Results.**

- **Impact unit L4/49.** It is silent on contact-free frames (mean 0.01) and silent at near misses (≤ 0.06). It fires sharply on the frame *before* fast impacts (0.60 floor, 0.70 wall), more weakly before slow impacts (0.12 / 0.16), and is back at zero once contact has begun. It is a clean "collision in the next 20 ms, scaled by approach speed" detector, although no census target captures it (best R² 0.06).
- **Impact unit L4/967** behaves similarly but stays partly active during sustained contact.
- **Census-selected "floor contact" unit L2/237** (best variable: elbow contact torque). It responds to floor proximity as much as to contact: near-miss activation 0.78 vs 0.98 at onset for slow impacts. It also stays elevated while the arm rests on the floor. It never responds at the wall.

Floor and wall are handled by largely separate units.

## 8. Causal interventions

**Gravity units.** We defined gravity units as the 66 live units with clean (R² ≥ 0.5) best fits to shoulder gravity, elbow gravity or forearm angle (20 / 40 / 6 / 0 in blocks 1–4). On 34k contact-free test frames we compared the change in predicted acceleration (Δq̇/Δt) with the change expected if gravity were scaled by *s*: Δq̈* = (s − 1)M(q)⁻¹τ_g(q).

- **Unit ablation.** Zero-ablating all 66 units removes only **25%** of gravity (pooled regression slope 0.25). The R² of the change explained by the gravity direction is just 0.01, because the ablation causes large changes that are mostly not gravity-shaped. Mean ablation removes ≈0%. Ablating 66 random live units from the same blocks gives −0.04, and ablating the top 25 velocity units gives 0.06.
- **Unit gain.** Scaling the units by *s* gives effective gravity changes of −0.25 at *s* = 0 and ≈0 for *s* ≥ 1. A model with a linear gravity channel would give *s* − 1.
- **Direction edits.** We added (s − 1)·τ_g along a residual-stream direction, either the *encoding* direction (regression of the residual on τ_g) or the *decoding* (probe) direction, and continued the forward pass.
  - Decoding-direction edits have no effect (|slope| ≤ 0.03 after blocks 1–3), despite R² 0.99 probes.
  - Encoding-direction edits after block 2 change predicted acceleration in the gravity direction with slope −0.82 at *s* = 0 and +0.60 at *s* = 2. After block 3 the slopes are −0.61 and +0.46. These are monotonic, roughly 50–80% of ideal, and of the correct sign, though the fit explains only R² ≈ 0.1 of the per-frame change.
  - Edits at the embedding or after block 1 are unstable or non-monotonic.

**Figure `state_causal_gravity.png`.** Causal tests of the gravity representation on contact-free test transitions.

- **(a)** Fraction of gravity removed: the slope of the change in predicted acceleration on the analytic −M⁻¹τ_g, against the number of top gravity-tuned units zero- or mean-ablated, with random-unit and velocity-unit controls. The dotted line marks full removal.
- **(b)** Effective gravity change when all 66 gravity units are multiplied by a gain *s*. The dotted line is the ideal *s* − 1.
- **(c)** The same when (s − 1)·τ_g is written into the residual stream along the gravity encoding direction after each depth (solid) or along the probe (decoding) direction (red dashed; every layer gives the same flat line).
- **(d)** Per-frame change in predicted shoulder (blue) and elbow (orange) acceleration for *s* = 0 written after block 3, against the analytic expectation.

**Contact units.** We tested two unit sets:

- the 23 clean contact-family census units;
- the 57 impact units (d′ > 0.5).

We compared the model's contact component (prediction minus rigid-body, no-contact physics) with the true contact-induced Δq̇ on steps where contact begins.

| ablation | slope on contact-onset steps | contact-free R² (Δq̇₁ / Δq̇₂) |
|---|---|---|
| intact | 0.98 | 0.987 / 0.974 |
| 23 census contact units | 0.94 | 0.884 / 0.660 |
| 57 impact units | **0.77** | 0.977 / 0.934 |
| 57 random units (control for impact) | 1.08 | 0.971 / 0.920 |

Ablating the census contact units barely touches the bounce but damages free motion: they are not a contact module. Ablating impact units removes ≈21% of the predicted bounce while hurting free motion no more than a random ablation of the same size. They are causally and specifically involved in collisions, but they carry only part of the bounce.

**Figure `state_causal_contact.png`.** Autoregressive rollout of a passive test episode through a floor impact (episode 2139). Left: the lowest point of the arm (minimum of elbow and hand height) for the true trajectory, the intact model, the model with 73 contact and impact units zero-ablated, and the model with 73 random units ablated. The horizontal line is the height at which the forearm touches the floor. Right: rendered poses at five times for the truth, the intact model and the ablated model.

In the rollout, the intact model reproduces the impact and bounce closely (minimum height −0.834 m vs −0.836 m true). With the union of contact and impact units ablated (73 units), the arm does *not* pass through the floor. Instead the dynamics after the impact diverge: the arm rebounds to −0.22 m. The predicted "bounce disappears" result is therefore **not** observed. Contact handling is distributed enough that ablation corrupts rather than removes it.

## 9. Training dynamics and activation bands

**Method.** At every checkpoint we recomputed binned R² for all live units (30k fixed test frames). We tracked four units by index: the best units for shoulder gravity, floor contact, shoulder interaction and shoulder velocity in the final model.

**Figure `state_dynamics_census.png`.** Left: fraction of live units that are clean in each block across training (solid). Dashed lines show the fraction clean on a derived quantity (acceleration, gravity, interaction, contact or energy). Step 0 is plotted at 10. Right: test MSE across training.

**Figure `state_dynamics_units.png`.** Tuning of four final-model units, tracked by index across training. Lines show binned R² against the four variables each unit was most tuned to at any checkpoint; the thick line is the unit's target variable. Images show the unit's top-2 held-out examples at four checkpoints (vertical grey lines).

**Results.**

- **Census over training.** The clean fraction first *rises* in every block, from 22–30% at initialization to 29–34% around step 300. It then falls in blocks 3–4, to 10–15%, as units die and the loss keeps dropping. The fraction clean on a derived quantity grows over the same period (block 2: 4% → ~18% at step ~10k, ending at 13%).
- **L1/321 (velocity).** It undergoes a sharp transition: q̇₁ R² is 0.3 before step 1k and 0.93 after step 3k. It narrows and does not broaden.
- **L2/826 (shoulder gravity).** It starts weakly gravity-like (R² 0.35 at initialization, inherited from its random projection of cos q₁). It sharpens to 0.91 by step 6k, then slightly *broadens*: gravity R² falls to 0.85 while hand-x and cos q₁ rise to ≈0.7.
- **L2/834 (floor contact).** It is *re-purposed*. It starts as an elbow-angle unit (sin q₂ 0.72), loses that tuning around step 600, and becomes a potential-energy / hand-height / floor-contact unit.
- **L3/11.** It is untuned for the first ≈3k steps. It then acquires a joint tuning to kinetic energy, contact torque and Coriolis torque (each ≈0.65) and never becomes single-variable.

**Activation bands.** Within each band of a unit's active frames we fit binned R² (10 bins).

**Figure `state_bands.png`.** What explains a unit's activation in different bands. Left: activation against the unit's target variable (grey: all frames; red: top 1% of active frames; blue: 50–90th percentile). Right: the three variables with the highest within-band binned R² (10 bins), coloured by family.

**Results.**

| unit | top 1% of active frames | 50–90th percentile band |
|---|---|---|
| gravity L2/826 | elbow velocity / Coriolis (q̇₂ 0.46, Coriolis 0.47) | gravity (0.47) |
| L3/11 | posture (PE 0.88) | contact torque (0.32) |
| velocity L1/321 | q̇₁ (0.70) | q̇₁ (0.85) |

For L2/826, the strongest responses are modulated by a *different* variable than the bulk of its tuning. The velocity unit keeps the same meaning across bands. The contact-like unit L2/834 is explained by elbow velocity in its top band but by contact torque / forearm angle (0.69) in its mid band.

## Caveats

1. **Low-dimensional input inflates "cleanness."** With 8 inputs, a random network already has 25% clean units, and gravity torque is linearly decodable from the raw input (R² 0.96). Unit- or probe-level claims about posture, velocity or gravity are therefore weak unless compared with the random-init control. This is why we report the "derived quantity" fraction.
2. **Aliased targets.** Elbow gravity torque ≡ cos(q₁+q₂) and elbow height ≡ sin q₁. A unit "tuned to elbow gravity" cannot be distinguished from a forearm-angle unit using data alone. The resting receptive fields (§4) suggest the forearm-angle reading is the right one.
3. **Dead units.** More than half of all units are dead in the trained model, even at the reduced learning rate. Census fractions are of live units, so the population sizes differ between the trained and random networks.
4. **Choice of thresholds and estimator.** The census uses 1-D binned R² with fixed thresholds (0.5 / 0.25 / +0.15 for conjunctions). With 10×10 cells, conjunction fits are easy in an 8-D input space: 72% of random units qualify. Different bins or thresholds shift the percentages, though not the trained-vs-random comparison. Census values are also not corrected for the multiple comparisons implicit in taking the maximum over 32 targets, although the bias is ≈0.0005 per fit at N = 59,600.
5. **Instantaneous q̈ is a poor target near contact.** It is dominated by clipped contact spikes. The step-mean acceleration (Δq̇/Δt) is what the model predicts, so acceleration claims rest on contact-free frames.
6. **Exemplar selection is automated and biased toward high R².** The gallery shows the best unit for each variable, not typical units. Several "contact" and "acceleration" exemplars are only partial, or are best explained by another variable; this is stated in the figure.
7. **Causal effects are partial.** No unit-level intervention behaves like an isolated gravity or contact channel. The direction-level gravity edit is monotonic and correctly signed but explains only ≈10% of the per-frame variance of the change. The contact rollout did not show the hypothesized fall-through.
8. **Training is not fully converged.** Test loss was still decreasing slowly at 60k steps. Only one seed was trained, so unit identities and census numbers will vary across seeds.
9. **Rendering.** The arm is rendered at 144 px and cropped to the workspace. Velocity arrows and trails are drawn from logged kinematics, not inferred by the model.
