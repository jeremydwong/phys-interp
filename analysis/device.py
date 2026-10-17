"""Device selection shared by every analysis, so the same code runs on the Mac (mps) and the msi GPU (cuda)."""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    """cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device: torch.device | None = None) -> torch.dtype:
    """bf16 on cuda/mps, fp32 on cpu."""
    device = device or get_device()
    return torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
