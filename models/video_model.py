"""Candidate B video-prediction model: 4 frames in -> frames t+1, t+2 out.

Built to be readable:
  * per-frame ConvNeXt encoder (patchify 4x4/4 -> 16x16 tokens; depthwise 7x7, LN, 4x GELU MLP,
    layer scale, residual). The encoder sees ONE frame at a time, so its units can only know
    what is visible in a still image (posture, contact geometry) -- a built-in control.
  * factorized space-time transformer: spatial attention within each frame, causal temporal
    attention per spatial location over the 4 frames, then a wide GELU MLP; pre-LN residual.
  * light decoder from the frame-t tokens (16x16) -> 64x64 pixel deltas for t+1 and t+2.

"Neurons" = post-GELU channels of every block's MLP hidden layer. `record=True` stores them
(for frame t only) in `self.acts[name]` as [B, H, 16, 16]; `edits` lets a caller modify a
layer's hidden activations (ablation / steering) through a callable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_IN, N_OUT, RES, GRID = 4, 2, 64, 16


class LN2d(nn.Module):
    """LayerNorm over channels of a [B, C, H, W] map."""
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, c, mult=4, ls=1e-1, name=""):
        super().__init__()
        self.name = name
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.ln = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, mult * c)
        self.fc2 = nn.Linear(mult * c, c)
        self.gamma = nn.Parameter(ls * torch.ones(c))

    def forward(self, x, hook):        # x [N, C, H, W]
        h = self.dw(x).permute(0, 2, 3, 1)
        h = F.gelu(self.fc1(self.ln(h)))           # [N, H, W, 4C]  <- neurons
        h = hook(self.name, h)
        h = self.fc2(h) * self.gamma
        return x + h.permute(0, 3, 1, 2)


class STBlock(nn.Module):
    """Factorized space-time block on tokens [B, T, S, C]."""
    def __init__(self, c, heads=4, mult=4, ls=1e-1, name=""):
        super().__init__()
        self.name = name
        self.ln_s, self.ln_t, self.ln_m = nn.LayerNorm(c), nn.LayerNorm(c), nn.LayerNorm(c)
        self.heads = heads
        self.qkv_s, self.out_s = nn.Linear(c, 3 * c), nn.Linear(c, c)
        self.qkv_t, self.out_t = nn.Linear(c, 3 * c), nn.Linear(c, c)
        self.fc1 = nn.Linear(c, mult * c)
        self.fc2 = nn.Linear(mult * c, c)
        self.g_s = nn.Parameter(ls * torch.ones(c))
        self.g_t = nn.Parameter(ls * torch.ones(c))
        self.g_m = nn.Parameter(ls * torch.ones(c))

    def _attn(self, h, qkv, out, causal):
        N, L, C = h.shape
        q, k, v = qkv(h).reshape(N, L, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return out(a.transpose(1, 2).reshape(N, L, C))

    def forward(self, x, hook):
        B, T, S, C = x.shape
        h = self.ln_s(x).reshape(B * T, S, C)
        x = x + self.g_s * self._attn(h, self.qkv_s, self.out_s, False).reshape(B, T, S, C)
        h = self.ln_t(x).transpose(1, 2).reshape(B * S, T, C)
        a = self._attn(h, self.qkv_t, self.out_t, True)
        x = x + self.g_t * a.reshape(B, S, T, C).transpose(1, 2)
        h = F.gelu(self.fc1(self.ln_m(x)))          # [B, T, S, 4C]  <- neurons
        h = hook(self.name, h)
        return x + self.g_m * self.fc2(h)


class VideoPredictor(nn.Module):
    def __init__(self, c=128, n_enc=4, n_tmp=4, heads=4, mult=4):
        super().__init__()
        self.c = c
        self.stem = nn.Conv2d(3, c, 4, stride=4)
        self.stem_ln = LN2d(c)
        self.pos = nn.Parameter(torch.zeros(1, c, GRID, GRID))
        self.enc = nn.ModuleList([ConvNeXtBlock(c, mult, name=f"enc{i}") for i in range(n_enc)])
        self.tpos = nn.Parameter(torch.zeros(1, N_IN, 1, c))
        self.tmp = nn.ModuleList([STBlock(c, heads, mult, name=f"tmp{i}") for i in range(n_tmp)])
        self.dec_ln = nn.LayerNorm(c)
        self.dec = nn.Sequential(
            nn.Conv2d(c, 256, 1), nn.GELU(),
            nn.PixelShuffle(2),                     # 64 ch @ 32
            nn.Conv2d(64, 96, 3, padding=1), nn.GELU(),
            nn.PixelShuffle(2),                     # 24 ch @ 64
            nn.Conv2d(24, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 3 * N_OUT, 3, padding=1),
        )
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.tpos, std=0.02)
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)
        self.record = False
        self.acts: dict[str, torch.Tensor] = {}
        self.residual: dict[str, torch.Tensor] = {}
        self.edits: dict[str, callable] = {}

    @property
    def layer_names(self):
        return [b.name for b in self.enc] + [b.name for b in self.tmp]

    def _hook(self, name, h):
        """h: enc -> [B*T, H, W, 4C]; tmp -> [B, T, S, 4C]."""
        if name in self.edits:
            h = self.edits[name](name, h)
        if self.record:
            if name.startswith("enc"):
                Bt = h.shape[0]
                hh = h.reshape(Bt // N_IN, N_IN, GRID, GRID, -1)[:, -1]
            else:
                hh = h[:, -1].reshape(h.shape[0], GRID, GRID, -1)
            self.acts[name] = hh.permute(0, 3, 1, 2)
        return h

    def forward(self, x):
        """x: float [B, 4, 3, 64, 64] in [0,1]. Returns predicted frames [B, 2, 3, 64, 64]."""
        B = x.shape[0]
        last = x[:, -1]
        z = (x - 0.5).reshape(B * N_IN, 3, RES, RES)
        z = self.stem_ln(self.stem(z)) + self.pos
        for blk in self.enc:
            z = blk(z, self._hook)
            if self.record:
                self.residual[blk.name] = z.reshape(B, N_IN, self.c, GRID, GRID)[:, -1]
        z = z.reshape(B, N_IN, self.c, GRID * GRID).transpose(2, 3) + self.tpos   # [B,T,S,C]
        for blk in self.tmp:
            z = blk(z, self._hook)
            if self.record:
                self.residual[blk.name] = z[:, -1].transpose(1, 2).reshape(B, self.c, GRID, GRID)
        h = self.dec_ln(z[:, -1]).transpose(1, 2).reshape(B, self.c, GRID, GRID)
        d = self.dec(h).reshape(B, N_OUT, 3, RES, RES)
        return last[:, None] + d


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = VideoPredictor(**ck.get("cfg", {}))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


if __name__ == "__main__":
    m = VideoPredictor()
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")
    y = m(torch.rand(2, 4, 3, 64, 64))
    print(y.shape)
