# Task for the msi machine (RTX 5070 Ti Laptop, 12 GB VRAM, 31 GB RAM, Windows)

You are the GPU worker for this repo. The Mac (M4 Max) built the simulator and is finishing a
first round of neuron analyses; it is too slow and makes the laptop lag, so inference and
training move here. Work only inside `C:\Git\phys-interp`. Commit to a branch `msi/<topic>`
and push; don't push to `main`.

## 1. Environment (CUDA PyTorch for Blackwell)
- The RTX 50-series (sm_120) needs CUDA 12.8+ PyTorch wheels. PyPI's Windows torch is CPU-only.
- Add a uv index for the PyTorch CUDA wheels, scoped with a platform marker so macOS keeps
  using PyPI torch, e.g. in pyproject.toml:
  ```toml
  [[tool.uv.index]]
  name = "pytorch-cuda"
  url = "https://download.pytorch.org/whl/cu128"   # or the newest cuXXX that has the torch version in uv.lock
  explicit = true

  [tool.uv.sources]
  torch = [{ index = "pytorch-cuda", marker = "sys_platform == 'win32'" }]
  torchvision = [{ index = "pytorch-cuda", marker = "sys_platform == 'win32'" }]
  ```
  Pick a torch version that exists on that index; adjust the pin if needed. Run `uv lock` and
  `uv sync`. Verify: `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.randn(4096,4096,device='cuda').matmul(torch.randn(4096,4096,device='cuda')).sum().item())"`.
- Verify MuJoCo offscreen rendering works on Windows:
  `uv run python -c "import sys; sys.path.insert(0,'.'); from sim.arm import *; print(ArmRenderer(64)(simulate(0,'reach')['q'][:3]).shape)"`.

## 2. Regenerate the datasets (not in git)
`uv run python sim/make_dataset.py` → `data/main` (2400 eps, ~4.2 GB) and `data/grav` (1200 eps).
Seeds are fixed. Sanity check against the Mac: `states.npz` `q[0,:3]` should be close to
the Mac's values (report them; small float differences across platforms are acceptable).

## 3. Pull models
- `Qwen/Qwen3-VL-4B-Instruct` (the model the Mac VLM study uses; bf16 ≈ 9 GB, fits in 12 GB).
- `Qwen/Qwen3-VL-8B-Instruct` — only fits in 4-bit (bitsandbytes nf4) or with CPU offload; try nf4
  and report whether activations look sane vs the 4B.
- `facebook/vjepa2-vitl-fpc64-256` (V-JEPA 2 ViT-L, video world model; check the exact hub id).
Use `huggingface_hub.snapshot_download`; keep the default HF cache.

## 4. Device-agnostic inference harness
- Add `analysis/device.py` with `get_device()` (cuda → mps → cpu) and `default_dtype()`
  (bf16 on cuda/mps). Don't edit other workers' files yet; the Mac will merge its
  analysis code later and switch it to `get_device()`.
- Add `analysis/gpu_bench.py`: for Qwen3-VL-4B, render 8-frame 224 px clips of the arm from
  `data/main/states.npz` test episodes with `sim.arm.ArmRenderer(224)`, run them as video input
  with a forward hook on a few vision-tower and LLM MLP layers (print module names first),
  and report clips/second, peak VRAM, and activation shapes. Same for V-JEPA 2 ViT-L.
  Use SDPA attention (`attn_implementation="sdpa"`); flash-attn on Windows is optional.
- Write results to `reports/results/msi_bench.md` (GPU, driver, torch/CUDA versions, throughput,
  VRAM, any errors and fixes).

## 5. Report back
Commit and push branch `msi/setup`. Then stop and summarise: what works, throughput numbers,
anything that failed. The next instructions (moving the VLM and video analyses here) will
reference `CLAUDE.md` on `main`.
