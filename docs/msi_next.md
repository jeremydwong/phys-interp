# msi: take over the first pass (after docs/msi_task.md)

The Mac's network is going down, and the Mac is too slow anyway. Everything from the first pass
is on branch `mac/first-pass-wip`: code, figures, and `reports/results/{state,grav}.md`.
Checkpoints are copied by scp into `models/*_ckpt*/` and `data/grav_extra/` (gitignored); if
they are missing, retrain from the commands in the .md files.

1. `git fetch && git checkout -b msi/first-pass origin/mac/first-pass-wip`. Merge your
   `msi/setup` work (CUDA index, `analysis/device.py`) into it.
2. Make the training and analysis scripts device-agnostic. Replace hard-coded `"mps"` with
   `analysis.device.get_device()`.
3. **Video model (prefix video_).** The Mac run was stopped at about step 5.6k (best checkpoints are
   in `models/video_ckpt*`). Retrain `models/video_train.py` on CUDA until validation plateaus
   (at least 20k steps), keeping log-spaced checkpoints. Then run every video analysis
   (`analysis/video_*.py`, or `video_run_all.py` if present) and write `reports/results/video.md`
   and `video.json` in the same format as `state.md`: per-analysis method, numbers, figure
   filenames with full captions, interpretation, and a caveats list. Follow the
   video-model spec in CLAUDE.md and the conventions in `analysis/common.py`.
4. **VLM (prefix vlm_).** Qwen3-VL-4B-Instruct. Re-extract activations on CUDA
   (`analysis/vlm_extract.py`; the Mac cache in `data/vlm` was not copied). Finish the census,
   probes, galleries, contact PSTH, quantile bands, behaviour questions (verbal accuracy vs probe
   accuracy) and generalisation renders. Document exactly which frames, resolution, fps and
   temporal-patch grouping the model saw. Write `reports/results/vlm.md` and `vlm.json`.
5. Check every figure (open the PNG) and fix overlaps. Commit to `msi/first-pass` and push
   often (at least after each study), so the Mac can pull results whenever it is online.
6. When both are done, push and summarise. The Mac will write report 2 from
   `reports/results/*.md`.
