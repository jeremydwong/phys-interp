# phys-interp

Neuron-level interpretability of models that predict physical dynamics, in the style of
Cammarata, "Neural Networks Are Already Clean" (https://already-clean.goodfire.com/).
Testbed: a MuJoCo planar two-link arm in a vertical plane, with a floor and a wall, where
every term of the equation of motion is logged, so hidden units can be checked against
ground-truth gravity torque, interaction torque, contact force, acceleration and state.

## Machines
- **Mac (M4 Max, 48 GB, MPS)**: authoring, light analysis, reports. Heavy jobs make the
  laptop lag; don't run long training or VLM inference here.
- **msi (Windows, RTX 5070 Ti Laptop 12 GB, 31 GB RAM, CUDA)**: all training and model
  inference. Repo at `C:\Git\phys-interp`. Reached through herdr (`ssh msi herdr ... --session desktop`,
  workspace `phys-interp`, agent `physgpu`). Its first task is `docs/msi_task.md`.
- Use uv for everything (`uv run python ...`). On the Mac, don't use the system python3
  (it's Julia's conda).
- Data is not in git. Regenerate with `uv run python sim/make_dataset.py` (deterministic seeds).

## Layout
- `sim/arm.py`: the arm model (`xml`, `simulate`, `ArmRenderer`). Sign convention:
  `M qdd + c + g = tau_act + tau_contact - b qd`, with g and c from MuJoCo's `qfrc_bias`. The
  EOM residual is 1e-5.
- `sim/make_dataset.py`: `data/main` (2400 eps x 150 frames at 50 Hz, floor and wall, modes
  passive/sines/reach; train eps [0,2000), test [2000,2400)) and `data/grav` (1200 eps, random
  gravity direction, no obstacles; train [0,1000)).
- `analysis/common.py`: the 32 shared target variables (`targets()`), the families and colours,
  `binned_r2`, and `setup_style()`. Every study must use these.
- Per-study prefixes: `state_` (residual MLP forward model, "candidate C"), `video_` (ConvNeXt +
  factorized space-time video predictor, "candidate B"), `vlm_` (Qwen3-VL-4B-Instruct on rendered
  clips), `grav_` (causal transformer that must infer the gravity direction in context).
  Code is in `models/` and `analysis/`, figures in `reports/figs/<prefix>_*.png`, write-ups in
  `reports/results/<prefix>.md` and `.json`.
- `reports/research/cammarata_math*`: the sourced math of the interpretability improvements (KaTeX).
- Report 1 (overview, published artifact): https://claude.ai/artifact/QR2538YtzPNDj91di8ETvy
  (source in the Mac session's scratchpad: `physics-neurons.html`).

## State as of 2026-10-16
- Simulator and datasets are done and verified.
- First-pass analyses ran on the Mac and were interrupted by a network outage, then resumed:
  - state: model trained, ~13 figures. Results write-up pending.
  - grav: models trained (in-context, gravity-given, history-1 controls), 9 figures. Remaining:
    equivariance, steering, cue conflict, magnitude.
  - video: trained to ~5.6k steps (killed to stop the lag). Analyses run on the best
    checkpoint. **It needs a longer retrain on msi.**
  - vlm: Qwen3-VL-4B activations cached (`data/vlm`, ~1 GB, Mac only). Census, probes and
    galleries come from the cache. **Behaviour questions and generalisation renders are deferred to msi.**
- Report 1 still needs: the math section (from `reports/research/cammarata_math_fragment.html`,
  loaded with KaTeX) and a "which frames each model sees" section (`reports/figs/setup_frames.png`).
- Report 2 (findings) is not written yet. It gets built from `reports/results/*.md` once all four exist.

## Plan for agentic next steps
Run these as parallel agents where independent. Each agent owns one prefix, commits to its
own branch `msi/<topic>` or `mac/<topic>`, and finishes with `reports/results/<prefix>.md`.

1. **msi setup** (`docs/msi_task.md`): CUDA torch for sm_120 through a platform-marked uv index,
   regenerate data, pull Qwen3-VL-4B / 8B (nf4) / V-JEPA 2 ViT-L, `analysis/device.py`, and
   benchmarks in `reports/results/msi_bench.md`.
2. **msi: video model v2.** Retrain `models/video_train.py` on CUDA to convergence (≥20k steps, or
   until val plateaus), with log-spaced checkpoints. Rerun `analysis/video_run_all.py`. Add a
   larger variant (2x width) to test whether neurons get cleaner with capacity (the essay's
   92k → 7M → 28M size sweep).
3. **msi: VLM.** Rerun the extraction on CUDA for more clips (all 400 test eps, several end
   frames each). Run the deferred behaviour questions (verbal vs probe accuracy) and the
   generalisation renders (recoloured arm, new scene). Repeat the census for Qwen3-VL-8B (nf4)
   and V-JEPA 2 ViT-L (a video world model, closest to "a visual system that predicts dynamics").
4. **msi: feature visualization at scale.** Use transformation-robust, Fourier-parameterized
   feature vis for the video model and the Qwen vision tower. Compare 2017-style settings with
   the fixed settings, as the essay's microscope timeline does.
5. **Mac or msi: recursive-generalization test.** Train state and video models with more
   checkpoints, track each unit's top-example set across training, and compute the essay's
   locality statistic (new members are near existing members, not near the seed centroid).
6. **Mac: reports.** Finish report 1 (math and frames sections), then write report 2 from the
   four results files. Report 2 must answer: gravity-torque units? segment-specific torque
   units (shoulder vs elbow)? contact-force units? state units? acceleration units? It needs
   Distill-style galleries, censuses, probe heatmaps, PSTHs, steering and training replays.
   Publish it as an HTML artifact.
7. **Stretch:** a slot/interaction-network model ("candidate A"), tested for whether edge
   messages carry pairwise contact forces (Cranmer et al. 2020). Also humanoid/AddBiomechanics
   for pose → load.

## Conventions
- Census: `binned_r2` per unit vs the 32 targets. ≥0.5 counts as clean, 0.25–0.5 as partial,
  otherwise unexplained. A unit is a conjunction if its best pair beats its best single
  variable by >0.15. Always compare against a random-init control.
- Figures: `common.setup_style()`, 200 dpi PNG, ~7 in full width, family colours from `FAMILIES`.
- Report numbers honestly, including negative results. Mark our own formalizations as such.
