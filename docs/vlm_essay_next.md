# VLM study: next steps that follow the essay

Spec written 2026-10-30 from `reports/research/essay_todos_2026-10-30.html` (the essay to-do
report, which links each item to https://already-clean.goodfire.com/). Item codes (B1, C3, E1 ...)
refer to that report. This doc picks the items that apply to the Qwen3-VL-4B study and turns
them into concrete jobs. Branch: `msi/first-pass`. Prefix: `vlm_`.

## Principle

The essay reads a neuron from what it does first (top examples at both poles, sign regimes,
inputs and readers, steering), and only then names it. Probes and censuses come last. Our VLM
work so far is census and probe first. So the order below is: fix the tools (E), read units
openly (B), then test explanations so they can lose (D), then re-score the census (C).

## 0. Close out the first pass (no new GPU work) — DONE (vlm.md §0)

- Open every `reports/figs/vlm_*.png`; fix overlapping labels, clipped titles, unreadable ticks.
- Write `reports/results/vlm.md` and `vlm.json` from `data/vlm/*_results.json`, `census.json`,
  `reports/results/vlm_*.json`. Must cover: which frames the model sees (8 frames, stride,
  fps, 320 px, temporal patch 2 → 4 groups), census vs random-init control, probes,
  verbal vs probe accuracy (behaviour), generalisation renders, contact events.
- Add the C2 caveat: binned R² is a simulation-style score; the essay says such scores reward
  weak-activation detail.

## 1. Tool fixes (E1, E2, B2) — DONE (vlm.md §1)

- **E1 SwiGLU sign regimes.** In `vlm_capture.py` also hook `gate_proj` and `up_proj` outputs
  for the LM layers. For each LM gallery unit, recapture top/bottom 100 clips and tag each by
  regime (+×+, +×−, −×+, −×−). Re-label any unit whose top list is mostly −×− ("echo").
  Output: `vlm_regimes.json`, figure `vlm_regimes.png` (stacked bars per unit, per rank band).
- **E2 sink dimensions.** Per layer (vision residual, LM residual), the coordinates holding
  >50% (and >10%) of squared-norm energy. Re-run the residual probe PCA with them removed;
  report the change. Output: `vlm_sink.json`.
- **B2 vision negative pole.** Hook vision `fc1` pre-activations (before GELU) for the gallery
  vision units; show both poles. Does the − pole carry a distinction (e.g. contact vs free)?

## 2. Read units openly (B3, C4) — DONE (vlm.md §2)

- **B3 blind descriptions.** For ~40 random LM + vision units (not chosen by census), show
  top-12 and bottom-12 clips (frames strip) plus raw activation values, and write a free-form
  verb description *before* looking at the census label. Then compare: same / more general /
  different / no pattern. Report the fraction where the free description beats the label.
  The describer must not see census labels (separate agent or a blinded file).
- **C4 dial vs detector.** For the 14 gallery units, plot raw activation vs best variable;
  classify dial (monotone across range) vs detector (flat then steep). Figure `vlm_tuning.png`.

## 3. Explanations that can lose (D3, D1) — DONE (vlm.md §3)

- **D3 simulator-written maxima and near misses.** For each labelled gallery unit (e.g. "LM
  contact −pole"), write arm states with `sim/arm.py` that should maximise the labelled
  variable, plus near misses (same pose without contact; mirrored pose; same speed other
  joint). Render with the same clip pipeline, run Qwen, measure firing as fraction of the unit's
  corpus max. Score won/tied/lost as in the essay's appendix (new ≥ 0.5 max, near miss
  < 0.2 max). Output `vlm_nearmiss.json`, figure `vlm_nearmiss.png`.
- **D1 steering.** Add a hook that sets/scales a gallery LM unit (the down_proj input) and rerun
  the `vlm_behaviour` questions on 64 clips. Does turning up a "contact" unit change the verbal
  contact answer? Control: as many random units from the same layer.

## 4. Re-score (C1) — DONE (vlm.md §4)

- Strong-activation census: R² restricted to clips above half the unit's max, and to the top
  1%. Report clean/partial/unexplained for "all" vs "strong", plus a "described, not one of our
  32 variables" bucket from B3.

## Deferred

- C5 longer clips (16/32 frames), F2 size sweep with Qwen3-VL-8B nf4 (benchmarked:
  6.78 GiB peak, 5.3 clips/s), D4 projection-out of probe directions, G1 (no checkpoints for Qwen).

## Done criteria

Each job commits its code, json, figure, a log in `models/vlm_logs/`, and a section in
`reports/results/vlm.md`, then pushes `msi/first-pass`. Negative results are reported.
