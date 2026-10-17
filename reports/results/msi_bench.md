# msi GPU worker: setup and benchmarks

Run on 2026-10-16. Reproduce with `uv run python analysis/gpu_bench.py [--models qwen4b vjepa qwen8b] [--clips 32] [--batch 4]`.

## Machine and software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, sm_120 (Blackwell), 12 GB (12,226 MiB), 46 SMs, 36 MB L2, ~70 W under load |
| driver | 610.78 (CUDA UMD 13.3), WDDM |
| CPU / RAM | 20 logical cores / 31 GB, Windows 11 Home 10.0.26200 |
| Python | 3.12.12 (uv-managed) |
| torch / CUDA / cuDNN | 2.14.1+cu132 / 13.2 / 9.24 |
| transformers / bitsandbytes / mujoco | 5.20.1 / 0.50.2 / 3.16.0 |

## 1. Environment

**CUDA torch.** `uv.lock` pins torch 2.14.1, which is **not published on the cu128 index**. It is on cu130 and cu132, and the driver supports CUDA 13.3, so `pyproject.toml` uses the newest index that carries it:

```toml
[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu132"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cuda", marker = "sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cuda", marker = "sys_platform == 'win32'" }]
```

macOS still resolves torch 2.14.1 / torchvision 0.29.1 from PyPI; the lock has both variants.

The verification command gives `True NVIDIA GeForce RTX 5070 Ti Laptop GPU (12, 0)` and a finite 4096² matmul sum. `bitsandbytes>=0.50.2` was added for the 8B nf4 test; it has macOS arm64 wheels, so the Mac lock still resolves.

**MuJoCo offscreen rendering** works on Windows with no extra setup. `ArmRenderer(64)(simulate(0,'reach')['q'][:3]).shape` returns `(3, 64, 64, 3)`.

**Problems hit and fixes.**

- The network is slow (~0.5 MB/s per connection) and drops often (DNS failures, resets).
  - `uv sync` failed once mid-download. Retrying with `UV_HTTP_TIMEOUT=900 UV_HTTP_RETRIES=10` worked.
  - `huggingface_hub` Xet downloads stalled at 0 bytes. Plain HTTP (`HF_HUB_DISABLE_XET=1`) resumed only within one call. A process-level retry restarted from zero and lost ~3 GB.
  - Fix: large LFS files were fetched with `curl -C - --retry` straight into the default cache path (`~/.cache/huggingface/hub/models--*/blobs/<sha256>`), sha256-verified, then `snapshot_download` was run to create the snapshot. Downloads are throttled per connection, so one process per model raised total throughput.
- The repo directory is owned by BUILTIN\Administrators, so git reports "dubious ownership". The global git config was not changed; commands use `git -c safe.directory=C:/Git/phys-interp`. To make it permanent, run `git config --global --add safe.directory C:/Git/phys-interp`.
- The Windows console is cp1252, so scripts that print `−` or `²` crash. Run the analysis scripts with `PYTHONUTF8=1`.
- Arial (the matplotlib fallback on Windows) has no subscript digits, so q₁ rendered as boxes. On `msi/first-pass`, `common.setup_style()` now lists the fonts in `font.family`, which makes matplotlib fall back per glyph to DejaVu Sans.

## 2. Datasets

`uv run python sim/make_dataset.py` produced `data/main` (2400 × 150 frames, 4.2 GB) and `data/grav` (1200 eps, 2.1 GB) in a few minutes with 14 worker processes.

Sanity values for comparison with the Mac (`states.npz`, `q[0, :3]`):

| dataset | q[0, 0] | q[0, 1] | q[0, 2] |
|---|---|---|---|
| main | (2.023843, 2.631776) | (2.086303, 2.670235) | (2.156063, 2.707455) |
| grav | (−2.427466, −0.154075) | (−2.514618, −0.095948) | (−2.611332, −0.024780) |

The Mac's values were not recorded in the repo, so the comparison has to be done on the Mac.

## 3. Models (default HF cache)

| model | hub id | size | status |
|---|---|---|---|
| Qwen3-VL-4B-Instruct | `Qwen/Qwen3-VL-4B-Instruct` | 8.9 GB (2 shards, bf16) | downloaded, sha256-verified |
| V-JEPA 2 ViT-L | `facebook/vjepa2-vitl-fpc64-256` (hub id confirmed) | 1.3 GB | downloaded (`original/model.pth`, the 4.8 GB raw checkpoint, skipped; not needed by transformers) |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | 17.5 GB (4 shards) | **download in progress** at ~0.5 MB/s; nf4 test pending (see below) |

## 4. Inference benchmark

**Protocol.** 32 test episodes (2000–2031) of `data/main`. Each clip is 8 sim frames (20, 24, …, 48; stride 4, 0.64 s), rendered at 224 px with `sim.arm.ArmRenderer(224)`. Batch size is 4. The first batch is warm-up and is excluded. Throughput includes preprocessing (processor and host→device) but not rendering, which takes 2–3 s for 32 clips on the CPU. Attention is SDPA. Forward hooks are on 3 vision-tower MLPs and 3 LM MLPs for Qwen (first, middle, last), and on 4 encoder MLPs for V-JEPA. Module names are printed by the script; examples are `model.visual.blocks.{i}.mlp`, `model.language_model.layers.{i}.mlp` and `encoder.layer.{i}.mlp`.

| model | dtype | load | resident VRAM | peak VRAM | throughput | input seen by the model |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | bf16 | 23 s | 8.32 GiB | **8.83 GiB** | **9.2 clips/s** | `video_grid_thw` [4, 14, 14]: 8 frames → 4 temporal groups (temporal patch 2) of 14×14 patches (16 px) → 49 LM tokens per group after 2×2 merge; 245 tokens per prompt |
| V-JEPA 2 ViT-L | bf16 | 5 s | 0.61 GiB | **0.78 GiB** | **18.3 clips/s** | resized to 256 px: 8 frames → 4 tubelets × 16×16 = 1,024 tokens |
| Qwen3-VL-8B-Instruct (nf4) | – | – | – | – | – | pending download |

**Activations (last batch).**

Qwen3-VL-4B, vision MLP outputs over 3,136 patches (4 clips × 4 groups × 196):

| layer | shape | std | absmax |
|---|---|---|---|
| block 0 | 3136 × 1024 | 0.58 | 10.9 |
| block 12 | 3136 × 1024 | 0.28 | 31.8 |
| block 23 | 3136 × 1024 | 29.0 | 2,224 |

Qwen3-VL-4B, LM MLP outputs:

| layer | shape | std | absmax |
|---|---|---|---|
| layer 0 | 4 × 245 × 2560 | 0.22 | 5.9 |
| layer 18 | 4 × 245 × 2560 | 0.33 | 29.8 |
| layer 35 | 4 × 245 × 2560 | 11.2 | 5,920 |

All values are finite. The large maxima in the last vision block and the last LM layer come from a few outlier ("massive activation") channels, which are typical of these models. They are not bf16 overflow.

V-JEPA 2, encoder MLP outputs, 4 × 1024 tokens × 1024:

| layer | std | absmax |
|---|---|---|
| 0 | 2.0 | 93 |
| 8 | 1.26 | 27 |
| 15 | 1.27 | 30 |
| 23 | 2.9 | 133 |

All values are finite.

**Training throughput (video model, for reference).** `models/video_train.py` (1.75 M params, batch 64, 64×64 frames, train split held on the GPU, 5.6 GB peak):

| setting | time per step |
|---|---|
| fp32 | 384 ms |
| TF32 | 344 ms |
| bf16 autocast + TF32 | 158 ms |

The 30k-step run took 91 min, including validation.

## 5. Notes for the VLM study

- The VLM study (`analysis/vlm_extract.py`) uses 320 px clips (20×20 patches → 100 tokens per group), which costs more per clip than the 224 px benchmark. The 4B model fits comfortably at batch 4 in bf16 (≈9 GB). Batch 8 would be close to the 12 GB limit.
- flash-attn was not installed. SDPA works on sm_120, and the transformers warning about flash-attn is harmless.
