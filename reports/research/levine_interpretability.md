# Levine on Robot Interpretability

*Research note, compiled 2026-10-16 for the phys-interp project. It covers what Sergey Levine (UC Berkeley RAIL; co-founder of Physical Intelligence) has said about interpreting robot controllers and robot foundation models, what his lab's work does that bears on interpretability, what outside groups have found inside his lab's models, and what this means for neuron-level interpretability of dynamics models.*

**Evidence labels used throughout.** <span class="tag said">SAID</span> is Levine's own words, quoted and dated. <span class="tag lab">LAB</span> is what a paper or blog post from RAIL or Physical Intelligence (PI) does or claims; Levine is a co-author but the claim is the paper's. <span class="tag ext">EXT</span> is work by other groups. <span class="tag syn">SYNTH</span> is my interpretation. Quotes are short verbatim excerpts. I pulled them with a page-reading tool, so check the linked sources before citing them in print.

## 1. Summary

- **I found no essay, talk or interview where Levine treats mechanistic or neuron-level interpretability as a research direction.** The word "interpretability" appears in his lab's papers (mainly ECoT) but not as a topic in his Substack essays or in the podcast transcripts I checked (Dwarkesh 2025; developing.dev 2026). He doesn't argue against it either. As far as I can tell, he has not taken a position. <span class="tag syn">SYNTH</span>
- **For him, interpretability is a byproduct of making the model emit things people can read and act on**: language subtasks, reasoning chains, value estimates. These are mainly levers for *correction, steering and data*, and only secondarily windows into the model. In 2022 he wrote that one can "examine the resulting value functions to see what the agent thinks each token in an utterance will do." <span class="tag said">SAID</span>
- **His lab's own evidence says these readable outputs are not faithful explanations.** In Chen et al. (2025, with Levine), most of the benefit of embodied chain-of-thought comes from *better representations learned by training to produce reasoning*. Their figures show reasoning traces with "incorrect or hallucinated features (though they still lead to task successes)." This is the strongest reason to look at activations rather than trust the text. <span class="tag lab">LAB</span>
- **The one place PI looked inside a model was at the representation level, not the neuron level.** In the human-to-robot transfer work (Dec 2025), t-SNE of mean-pooled final-layer VLM embeddings shows human and robot data merging as pre-training grows more diverse. Levine described the result as "they just line up perfectly" (Aug 2026). <span class="tag said">SAID</span> <span class="tag lab">LAB</span>
- **On world models, he is sceptical of predicting "everything."** Text "already" carries abstracted representations, while video prediction must choose what to model (Plato's Cave essay, Jun 2025; Dwarkesh, Sep 2025). He prefers representations that capture "what you really need to achieve your goal," and treats value functions and simulators as interchangeable ways to "consider counterfactuals." <span class="tag said">SAID</span> For a dynamics-interpretability researcher, this predicts that VLAs will encode task-relevant physical variables, not a complete physics state. <span class="tag syn">SYNTH</span>
- **His "bitter lesson" view ("any component that is not learned … will eventually become the bottleneck," Jul 2025) cuts against hand-built interpretable structure.** His recent view of language as a "scaffold" for prior knowledge (Aug 2026) partly offsets this. <span class="tag said">SAID</span>
- **Outside groups have started opening up his lab's open models** (OpenVLA, π0, π0-FAST, π0.5). They find FFN directions for speed and height that can be steered (Häon et al., CoRL 2025), SAE features in π0.5 that are mostly memorization but partly general and causally steerable (Swann et al., 2026), linear world-state probes in OpenVLA (Molinari et al., 2025; Lu et al., 2025), and failure-predictive probes in π0.5 (Yang and Mokalla, ICML 2026). <span class="tag ext">EXT</span>
- **I found no published work that probes these models for force, torque or contact.** Every probe so far targets symbolic state, end-effector kinematics, gripper state, or the dynamics of the embedding. LIBERO and SimplerEnv run on physics engines that log contact forces, so this is an open niche that fits this project directly. <span class="tag syn">SYNTH</span>

## 2. What Levine has said

### Essays (Substack, "Learning and Control", [sergeylevine.substack.com](https://sergeylevine.substack.com/archive))

| Date | Piece | What bears on interpretability |
|---|---|---|
| 2021-10-26 | [Understanding the World Through Action](https://sergeylevine.substack.com/p/understanding-the-world-through-action-rl-as-a-foundation-for-scalable-self-supervised-learning-636e4e243001) | He argues that agents trained to *control* must form "an understanding of their environment that is causal and generalizable." He writes that "solving such goal-conditioned RL problems corresponds to learning a kind of dynamics model," and contrasts this with model-based RL, "where the model objective is largely disconnected from actually bringing about desired outcomes." So he expects dynamics knowledge to show up *implicitly* inside policies and value functions. |
| 2022-12-04 | [Offline RL and Large Language Models](https://sergeylevine.substack.com/p/offline-rl-and-large-language-models) | His most explicit interpretability-style statement. Value functions are "really just another kind of predictive model," and "the advantage function tells us how much better (or worse) the agent thinks a given token is … we can examine the resulting value functions to see what the agent thinks each token in an utterance will do." In his toxicity example, words like "censor" get low advantage values. |
| 2024-10-06 | [The Promise of Generalist Robotic Policies](https://sergeylevine.substack.com/p/the-promise-of-generalist-robotic) | He describes VLMs "trained to generate intermediate inferences (e.g., positions of objects, substeps in a plan)," and robots that "take human speech, turn it into corrections." He says internet-scale pre-training doesn't cover "the nitty-gritty details of physical grippers, contact forces, and joint angles," and that "Internet-scale training is not enough by itself." |
| 2025-06-08 | [Language Models in Plato's Cave](https://sergeylevine.substack.com/p/language-models-in-platos-cave) | LLMs "skip the neurons all together and reconstruct the mind from the shadow it casts on the Internet". They "copy some aspects of human mental representations." Video models "have so far not yielded the same results." The field still needs "a way to autonomously acquire representations from physical experience." He doesn't discuss interpretability methods, but this is a claim about *what the representations are*. |
| 2025-07-21 | [Sporks of AGI](https://sergeylevine.substack.com/p/sporks-of-agi) | His bitter-lesson corollary: "any component that is not learned but instead designed by hand will eventually become the bottleneck to its performance." Hand-designed inductive biases "require us to build in how we think we think." |

### Talks and interviews

- **Dwarkesh Patel podcast, 2025-09-12** ([transcript](https://www.dwarkesh.com/p/sergey-levine)). <span class="tag said">SAID</span>
    - On video world models: "If you want to predict everything that'll happen in the future, you can do so in many different ways." On text: "The representations are already there." On goal-relevant state: "Representing your context in the right form, that captures what you really need to achieve your goal."
    - On value functions versus simulators: "Optimal decision making at its core, regardless of how you do it, requires considering counterfactuals," whether "by using a value function or something" else.
    - On reasoning: the model "could involve outputting intermediate steps," and "learning for these systems is not just learning from raw actions, it's also learning from words."
    - On emergence he can't explain: "You just have this kind of compositionality that emerges when you do learning at scale," and, about the unexpected T-shirt behaviour, "We didn't know it would do that."
    - I found no explicit discussion of interpretability or inspecting internals in the transcript.
- **developing.dev interview, 2026-08-24** ([page](https://www.developing.dev/p/sergey-levine-current-state-of-humanoid)). <span class="tag said">SAID</span>
    - On representation analysis by PI colleagues (Suraj Nair and Simar Kareer): "they looked at the representations inside the model." With enough robot pre-training, "it's all task identity and minimal sensitivity to embodiment," and "they just line up perfectly."
    - On multimodal reasoning: "you can have a model that does some intermediate thinking, and that thinking can be done in different modalities." He gives the example of imagined subgoal images that let a UR5 fold shirts.
    - On language: "what's important is not language, it's prior knowledge that you can put in as a scaffold on your learning process."
    - On failure modes: π0.5's "mistakes are sort of semantically sensible."
- **X/Twitter, July 2024** ([ECoT thread](https://x.com/svlevine/status/1811623922405630242)). He announced ECoT, saying it distils Gemini and Grounding DINO into reasoning data and can take in human corrections through the chain of thought. I only saw this through search snippets.
- **ICLR 2026 invited talk**, "Robotic Learning with Autonomous Data" ([listing](https://iclr.cc/virtual/2026/10021470)), and a **CoLM 2026 talk** titled roughly "Someday, ALL AI will be Built on Physical Interaction" (known only from a third-party link roundup). I found no transcripts for either.

**Gaps I could not fill.** I found no Levine statement on mechanistic interpretability, sparse autoencoders, probing, circuits, or the "Mechanistic interpretability for steering VLAs" line of work. I didn't check TWIML, Robot Brains or Gradient Dissent episodes in full. Search found only The Gradient (2021), Eye on AI and Automate podcasts, and their descriptions didn't mention interpretability.

## 3. What his lab's work does that bears on interpretability

| Work | Mechanism | What is inspectable | Caveat |
|---|---|---|---|
| **ECoT**, Zawalski, Chen, Pertsch, Mees, Finn, Levine, CoRL 2024 ([arXiv 2407.08693](https://arxiv.org/abs/2407.08693), [site](https://embodied-cot.github.io/)) | OpenVLA is fine-tuned to generate text before each action: task, plan, subtask, move primitive, gripper pixel position, and object bounding boxes. | The authors say it helps humans "interpret a policy's failures," for example by "misidentifying relevant scene objects," and lets operators correct the policy by editing the reasoning. It gave +28% absolute success over OpenVLA on generalization tasks. | The text is generated by the policy, so nothing guarantees it is causal. |
| **Training Strategies for Efficient Embodied Reasoning**, Chen, Belkhale, Mirchandani, Mees, Driess, Pertsch, Levine, CoRL 2025 ([arXiv 2505.08243](https://arxiv.org/abs/2505.08243)) | Tests *why* reasoning helps. Is it representation learning, curriculum, or expressivity? | Finds "learning to generate reasonings does lead to better VLA representations" and "the main benefit … is not increased policy expressivity." Fig. 6 shows "incorrect or hallucinated features (though they still lead to task successes)." | This is direct evidence from Levine's own group that reasoning traces are unfaithful. It argues for looking at internals. |
| **Hi Robot**, PI, Feb 2025 ([post](https://www.pi.website/research/hirobot)) | A high-level VLM "whispers" language commands to π0. | Because the commands are language, "we can inspect them." Users can correct them in real time ("that's not trash"). | The commands are a designed bottleneck, so they explain the high-level choice, not the motor control. |
| **π0.5**, PI, Apr 2025 ([post](https://www.pi.website/blog/pi05)) | One model predicts a text subtask, "analogously to chain-of-thought inference," and then actions. | The subtask string, for example "pick up the pillow." | Same caveat as Hi Robot. Below the subtask, the flow-matching action expert is opaque. |
| **Knowledge Insulation**, Driess et al. with Levine, NeurIPS 2025 ([arXiv 2505.23705](https://arxiv.org/abs/2505.23705), [post](https://www.pi.website/research/knowledge_insulation)) | Stops action-expert gradients from reaching the VLM backbone, which is trained on discrete FAST tokens instead. | Shows that action-expert gradients make the backbone "lose some of the knowledge acquired during web-scale pre-training." | This gives a hypothesis about where things live: semantic knowledge sits in the backbone, continuous motor detail in the action expert. It hasn't been tested mechanistically. |
| **π\*0.6 / Recap**, PI, Nov 2025 ([post](https://www.pi.website/blog/pistar06), [arXiv 2511.14759](https://arxiv.org/abs/2511.14759)) | A learned value function (the negative number of steps to completion) gives advantages, and the policy is conditioned on those advantages. | The post plots value over an episode. It "increases when the robot makes progress, and stays flat when there is little progress." | The value plot is a scalar progress readout used for credit assignment, not an explanation. The post doesn't frame it as interpretability. |
| **V-GPS**, Nakamoto, Mees, Kumar, Levine, CoRL 2024 ([arXiv 2410.13816](https://arxiv.org/abs/2410.13816)) | An offline-RL value function re-ranks actions sampled from Octo, OpenVLA and similar policies, without touching their weights. | A separate value model makes a score per action that people can inspect. | It steers from outside the policy and doesn't look inside it. |
| **Human-to-robot transfer**, Kareer et al., PI, Dec 2025 ([arXiv 2512.22414](https://arxiv.org/abs/2512.22414), [post](https://www.pi.website/research/human_to_robot)) | t-SNE of mean-pooled final-layer VLM tokens for human versus robot data. | Representations go from "disjoint" to overlapping as pre-training diversity grows, and the overlap tracks transfer. | This is representation geometry, with no neurons, no causal tests, and t-SNE only. |
| **RL Token**, PI, Mar 2026 ([post](https://www.pi.website/research/rlt)) | An encoder-decoder compresses VLA embeddings into one "RL token" for a small actor-critic. | It gives a bottleneck summary that could in principle be probed. | The post doesn't analyse what the token encodes. |
| **π0.7**, PI, Apr 2026 ([post](https://www.pi.website/blog/pi07)) | "Steerable" through multimodal prompts: speed and quality metadata, visual subgoal images (possibly from a world model), and language coaching. | The prompt fields are human-set control dimensions. | The post concedes it is "hard to track down the precise episodes that informed this behavior." Weights are not public as far as I found. |
| **Older RAIL work on value-function representations** (for example Kumar, Agarwal, Ghosh, Levine, *Implicit Under-Parameterization*, ICLR 2021; Eysenbach et al., *Contrastive RL*, NeurIPS 2022) | Feature-rank collapse in deep Q-networks. In contrastive RL, representation inner products *are* goal-conditioned values. | Representation-level diagnostics, plus a geometry where dot products have a defined meaning. | This is the closest thing to "representation analysis" in his RL work. It is for training stability, not for explanation. |

**Pattern.** <span class="tag syn">SYNTH</span> Every lab mechanism is *interpretability by interface*. The model is built to emit a readable intermediate (text, subgoal image, value), and that intermediate is then used for control, correction or data. None of the papers checks whether the readable channel is faithful to the computation, except Chen et al. 2025, and their answer is that it often is not.

## 4. External mechanistic and probing work on his lab's models

| Paper | Model(s) | Method | Finding |
|---|---|---|---|
| Häon, Stocking, Chuang, Tomlin (Berkeley, Tomlin lab), *Mechanistic Interpretability for Steering VLAs*, CoRL 2025 ([arXiv 2509.00328](https://arxiv.org/abs/2509.00328)) | OpenVLA (Llama-2-7B), π0 / π0-FAST (PaliGemma) | They project FFN value vectors through the LM head to tokens, cluster them (fast/slow/up/low/high/safe/risk), and overwrite their activations at inference. | Zero-shot control of speed and height in LIBERO and on a real UR5. The steering beat prompt changes. They studied no force or contact. |
| Swann, McGranahan, Buurmeijer, Kennedy, Schwager (Stanford), *SAEs Reveal Interpretable and Steerable Features in VLA Models*, 2026 ([arXiv 2603.19183](https://arxiv.org/abs/2603.19183)) | π0.5 (PaliGemma layers), on LIBERO and DROID hardware | SAEs on hidden activations, plus a metric that separates general features from memorization features. | Most features are episode-specific memorization. A smaller set of general motion and semantic features can be steered causally, including in directions prompting can't reach. |
| Molinari, Nevali, Navani, Younis, *Emergent World Representations in OpenVLA*, 2025 ([arXiv 2509.24559](https://arxiv.org/abs/2509.24559)) | OpenVLA (and the v01 checkpoint) | Lasso and MLP probes on layers 7/15/22/30 predict the embedding change Δe over K steps, with permutation tests. | Activations beat an embedding baseline, peaking in middle layers. Evidence is weaker for the early checkpoint and for LIBERO-only fine-tuning. Linear probes beat MLPs. |
| Lu, Li, Shahani, Herbers, Scheutz (Tufts), *Probing a VLA for Symbolic States*, 2025 ([arXiv 2502.04558](https://arxiv.org/abs/2502.04558)) | OpenVLA, 33 layers | Linear probes for object and action predicates. | Accuracy above 0.9 at most layers, with no clear layer ordering. |
| Yang and Mokalla, *Interpretable Signals Reveal Failure-Relevant Representations in VLAs*, ICML 2026 ([page](https://icml.cc/virtual/2026/79411)) | π0.5 action expert plus attention | Probes for action direction and gripper state, plus attention summaries. | Confidence in the action-direction probe predicts failure best. Gripper-state probes are weaker. |
| Zhang et al. (Leicester/Birmingham), *Embodied Interpretability*, 2026 ([arXiv 2605.00321](https://arxiv.org/abs/2605.00321)) | π0.5 fine-tuned | An interventional saliency score from masks, and a "nuisance mass ratio." | Reliance on nuisance regions correlates with failure (r ≈ −0.77). |
| *Robotic Steering*, 2025 ([arXiv 2511.22697](https://arxiv.org/abs/2511.22697)) | VLAs on a Franka arm | Picks task-relevant attention heads for fine-tuning. | Reported to beat LoRA. |
| Foutter et al. (Stanford/NVIDIA), *Do VLAs Mean What They Say?*, 2026 ([arXiv 2607.04681](https://arxiv.org/abs/2607.04681)) | Alpamayo (driving), not Levine models | A faithfulness critic used as an RL reward. | Functional reasoning and faithful reasoning come apart. It builds on the ECoT paradigm. |
| Wulff et al., *Latent Cluster Analysis for VLAs*, 2026 ([arXiv 2609.02634](https://arxiv.org/abs/2609.02634)) | GR00T N1.5 (NVIDIA), included for contrast | Clusters diffusion-decoder embeddings. | Middle layers separate spatiotemporal and kinematic features. No force. |

None of these papers involves Levine as an author, and none quotes a response from him. They cite his models only as objects of study.

## 5. Relevance to neuron-level interpretability of dynamics models

### How his thinking maps onto this project

- **Prediction to test.** <span class="tag syn">SYNTH</span> Levine's position is that action-trained representations capture what you need for the goal, not everything. Applied to the two-link arm, that predicts VLA and policy units will track **variables you need for control** (contact onset, progress, end-effector velocity) more cleanly than **variables you don't** (gravity torque or Coriolis terms that a position-controlled robot never has to compensate). The repo's 32 shared targets let you test this directly by comparing binned R² for each family between action-trained models and the next-frame predictors (`state_`, `video_`).
- **Faithfulness.** <span class="tag lab">LAB</span> Chen et al. 2025 show that reasoning text can be wrong while the action is right. The `vlm_` study already plans "verbal vs probe accuracy." The same comparison on ECoT, where the stated gripper position or move primitive is compared with a probe on hidden state and with ground truth, would be the first faithfulness test of embodied reasoning on Levine's own models.
- **Where to look.** <span class="tag lab">LAB</span> Knowledge Insulation implies a split: semantics in the PaliGemma backbone, continuous motor detail in the action expert. Contact and force, if they are represented at all, should sit in the action expert and the proprioceptive pathway. Häon et al. and Swann et al. looked mainly at the backbone.

### Experiments that fit a 12 GB GPU (RTX 5070 Ti laptop)

Memory notes: Octo-small and Octo-base are 27M and 93M parameters, so trivial. OpenVLA-7B needs 16.8 GB in bf16 but **7.0 GB in int4, with the same Bridge success rate** (OpenVLA paper, Table 2). openpi π0/π0.5 inference needs more than 8 GB and LoRA fine-tuning more than 22.5 GB (openpi README), so plan for inference and activation caching only, using the PyTorch port for hooks. The ECoT checkpoints are OpenVLA-based, so the same int4 path should apply (I haven't verified this).

1. **Contact-force census in LIBERO (π0.5-LIBERO or OpenVLA-LIBERO).** Roll out the released LIBERO checkpoints in robosuite/MuJoCo. Log `cfrc_ext`, contact normal forces from `mj_contactForce`, `qfrc_actuator` and gripper finger forces on every step, using the same logging approach as `sim/arm.py`. Cache action-expert and backbone activations with forward hooks. Run the project's per-unit census (binned R², top-k examples) against force magnitude, contact onset and slip. Then ablate or clamp the top units and measure the change in commanded gripper closure and in contact force. This is the most direct transfer of the arm methodology, and no one has done it yet.
2. **Force-direction steering à la Häon.** In OpenVLA int4, project FFN value vectors to tokens and look for clusters around *press, push, squeeze, gentle, hard, tight*. Inject them in LIBERO and measure **contact force** (not just displacement) during grasp and insertion. If no force-semantic cluster steers force, that supports the prediction above that position-trained VLAs encode kinematics, not dynamics.
3. **Implicit dynamics in Octo.** Octo is small enough for exhaustive neuron work. Use OXE subsets that log force/torque (for example the Stanford Kuka multimodal dataset; RH20T outside OXE). Check that a given subset has F/T before relying on it. Probe Octo's readout and transformer units for measured wrist F/T, which Octo never receives as input. A positive result means vision-based inference of force. Compare against a shuffled-label control.
4. **ECoT faithfulness test.** Run the ECoT OpenVLA-Bridge checkpoint in SimplerEnv (WidowX, SAPIEN physics, which exposes contacts) or open-loop on Bridge frames. Compare three things for gripper position and move direction: (a) what the text says, (b) a linear probe on the token before the text, and (c) ground truth. Then edit the text, for example "move down" to "move up," and measure the change in action. This is the intervention test Foutter et al. call impractical, and here it is cheap.
5. **Value-function units (Recap-style).** On frozen π0.5 or Octo features, train a small steps-to-completion value head as in π\*0.6. Find units whose activity drops at failed grasps or slips, using contact logs from experiment 1, and check whether those units are the same ones that encode contact. This tests Levine's 2022 claim that value functions can be read to see "what the agent thinks."
6. **Embodiment-invariance on the toy arm.** Feed the repo's rendered two-link-arm clips to the SigLIP vision tower of π0/π0.5 and to OpenVLA's DINOv2+SigLIP. Rerun the `vlm_` census and compare it with Qwen3-VL-4B and V-JEPA 2. PI's human-to-robot result predicts task-level, embodiment-agnostic features in late layers. The arm data tests whether low-level dynamics features survive in robot-pretrained vision towers.

## 6. Open questions

1. Does any open VLA represent contact force or torque at all, given that its inputs are images and joint positions and its outputs are positions or deltas? Or is "force" only gripper state plus timing?
2. Is the subtask text in π0.5 or ECoT causally upstream of the actions, or is it a parallel readout? (Chen et al. suggest it is partly decorative at test time.)
3. Do monosemantic "dynamics neurons" appear in flow-matching action experts, or does the denoising-time variable mix everything? Activations probably need to be indexed by flow time τ as well as by environment step.
4. Does knowledge insulation produce a cleaner split between semantic units (backbone) and motor units (action expert) than standard co-training does?
5. Do value functions trained on robot data (Recap, V-GPS) have units that line up with physical failure causes like slip, collision or missed contact? Or do they only encode progress?
6. Would Levine see neuron-level interpretability as a "spork," a hand-designed structure, or as a tool for finding where learned components bottleneck? He hasn't said, and it would be worth asking him directly.
7. Is the "memorization versus general feature" split that Swann et al. found in π0.5 also present in small dynamics models trained on simulated physics, like the ones in this repo? If so, does it shrink with data diversity, as PI's human-to-robot result would suggest?

## 7. References

**Levine, primary**

- Levine, S. "Understanding the World Through Action." Substack, 2021-10-26. https://sergeylevine.substack.com/p/understanding-the-world-through-action-rl-as-a-foundation-for-scalable-self-supervised-learning-636e4e243001
- Levine, S. "Offline RL and Large Language Models." Substack, 2022-12-04. https://sergeylevine.substack.com/p/offline-rl-and-large-language-models
- Levine, S. "The Promise of Generalist Robotic Policies." Substack, 2024-10-06. https://sergeylevine.substack.com/p/the-promise-of-generalist-robotic
- Levine, S. "Language Models in Plato's Cave." Substack, 2025-06-08. https://sergeylevine.substack.com/p/language-models-in-platos-cave
- Levine, S. "Sporks of AGI." Substack, 2025-07-21. https://sergeylevine.substack.com/p/sporks-of-agi
- Dwarkesh Patel podcast with Levine, "Fully autonomous robots are much closer than you think," 2025-09-12. https://www.dwarkesh.com/p/sergey-levine
- developing.dev interview, "Sergey Levine: Current State of Humanoid Robotics, China & Future Predictions," 2026-08-24. https://www.developing.dev/p/sergey-levine-current-state-of-humanoid
- Levine on X, ECoT announcement, July 2024. https://x.com/svlevine/status/1811623922405630242
- ICLR 2026 invited talk listing. https://iclr.cc/virtual/2026/10021470

**RAIL / Physical Intelligence**

- Zawalski et al. "Robotic Control via Embodied Chain-of-Thought Reasoning." CoRL 2024. https://arxiv.org/abs/2407.08693 · https://embodied-cot.github.io/
- Chen et al. "Training Strategies for Efficient Embodied Reasoning." CoRL 2025. https://arxiv.org/abs/2505.08243
- Nakamoto, Mees, Kumar, Levine. "Steering Your Generalists (V-GPS)." CoRL 2024. https://arxiv.org/abs/2410.13816
- Kim et al. "OpenVLA." 2024. https://arxiv.org/abs/2406.09246 · https://github.com/openvla/openvla
- Octo Model Team et al. "Octo: An Open-Source Generalist Robot Policy." 2024. https://octo-models.github.io/
- PI. "Hi Robot." 2025-02-26. https://www.pi.website/research/hirobot
- PI. "π0.5." 2025-04-22. https://www.pi.website/blog/pi05
- Driess et al. "Knowledge Insulating VLA Models." NeurIPS 2025. https://arxiv.org/abs/2505.23705 · https://www.pi.website/research/knowledge_insulation
- PI. "π\*0.6: a VLA That Learns From Experience." 2025-11-17. https://www.pi.website/blog/pistar06 · https://arxiv.org/abs/2511.14759
- Kareer et al. "Emergence of Human to Robot Transfer in VLAs." 2025-12. https://arxiv.org/abs/2512.22414 · https://www.pi.website/research/human_to_robot
- PI. "Precise Manipulation with Efficient Online RL (RL Token)." 2026-03-19. https://www.pi.website/research/rlt
- PI. "π0.7: a Steerable Model with Emergent Capabilities." 2026-04-16. https://www.pi.website/blog/pi07
- openpi repository (π0, π0-FAST, π0.5 weights). https://github.com/Physical-Intelligence/openpi

**External interpretability work**

- Häon, Stocking, Chuang, Tomlin. "Mechanistic Interpretability for Steering VLA Models." CoRL 2025. https://arxiv.org/abs/2509.00328
- Swann et al. "Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models." 2026. https://arxiv.org/abs/2603.19183
- Molinari et al. "Emergent World Representations in OpenVLA." 2025. https://arxiv.org/abs/2509.24559
- Lu et al. "Probing a VLA Model for Symbolic States and Integration into a Cognitive Architecture." 2025. https://arxiv.org/abs/2502.04558
- Yang and Mokalla. "Interpretable Signals Reveal Failure-Relevant Representations in VLAs." ICML 2026. https://icml.cc/virtual/2026/79411
- Zhang et al. "Embodied Interpretability." 2026. https://arxiv.org/abs/2605.00321
- "Mechanistic Finetuning of VLAs via Few-Shot Demonstrations (Robotic Steering)." 2025. https://arxiv.org/abs/2511.22697
- Foutter et al. "Do VLAs Mean What They Say?" 2026. https://arxiv.org/abs/2607.04681
- Wulff et al. "Latent Cluster Analysis for VLA Models." 2026. https://arxiv.org/abs/2609.02634
- "TA-VLA: Elucidating the Design Space of Torque-aware VLA Models." 2025. https://arxiv.org/abs/2509.07962
