"""Small vision models for the vis_ ladder study (Cammarata's InceptionV1 -> ConvNeXt ladder in miniature).

PlainCNN  ("M_old" / "M_recipe"): VGG-style feed-forward stack, conv3x3-BN-ReLU, 2 convs per stage,
          max-pool between stages, no residual path. Same architecture for both recipes.
ConvNeXtS ("M_arch"): ConvNeXt-style, depthwise 7x7 -> LayerNorm -> Linear(4C) -> GELU -> Linear(C)
          -> layer scale -> residual add; LN + 2x2 stride-2 conv downsampling between stages.

"Neurons" (what every measurement reads):
  PlainCNN:  each conv's BN output (pre-ReLU) z, post-activation a = ReLU(z). Layers "s{stage}c{k}".
  ConvNeXtS: each block's MLP hidden pre-activation z = Linear1(LN(dw(x))), a = GELU(z). Layers "s{stage}b{k}".
             This is the privileged (elementwise-nonlinear) basis; the residual stream is not used.

Both models expose
  model.neuron_layers           ordered list of layer names
  model(x, record=set_of_names) -> logits; model.acts[name] = z as [B, C, H, W] (pre-activation, detached
                                  unless model.keep_graph is True)
  model.edit = {name: fn}       fn(a, name) -> a', applied to the post-activation (ablation hooks)
Inputs are images in [0, 1], [B, 3, H, W]; normalisation happens inside (model.mean / model.std buffers).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Base(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.acts: dict[str, torch.Tensor] = {}
        self.post: dict[str, torch.Tensor] = {}   # post-activations (kept only when keep_graph, for attribution)
        self.record: set[str] = set()
        self.edit: dict = {}
        self.keep_graph = False

    def _neuron(self, name, z, act):
        if name in self.record:
            self.acts[name] = z if self.keep_graph else z.detach()
        a = act(z)
        if self.keep_graph and name in self.record:
            self.post[name] = a
        if name in self.edit:
            a = self.edit[name](a, name)
        return a

    def forward(self, x, record=None):
        if record is not None:
            self.record = set(record)
        self.acts, self.post = {}, {}
        x = (x - self.mean) / self.std
        return self._forward(x)


class PlainCNN(_Base):
    def __init__(self, n_classes, widths=(32, 64, 128, 256), convs=2, stem_stride=1,
                 mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)):
        super().__init__(mean, std)
        self.convs, self.bns, self.names, self.pool_after = nn.ModuleList(), nn.ModuleList(), [], []
        cin = 3
        for s, w in enumerate(widths):
            for k in range(convs):
                stride = stem_stride if (s == 0 and k == 0) else 1
                self.convs.append(nn.Conv2d(cin, w, 3, stride, 1, bias=False))
                self.bns.append(nn.BatchNorm2d(w))
                self.names.append(f"s{s + 1}c{k + 1}")
                self.pool_after.append(k == convs - 1 and s < len(widths) - 1)
                cin = w
        self.head = nn.Linear(cin, n_classes)
        self.neuron_layers = list(self.names)

    def _forward(self, x):
        for conv, bn, name, pool in zip(self.convs, self.bns, self.names, self.pool_after):
            x = self._neuron(name, bn(conv(x)), F.relu)
            if pool:
                x = F.max_pool2d(x, 2)
        return self.head(x.mean((2, 3)))


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, dim, ls_init=1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), ls_init))


class ConvNeXtS(_Base):
    def __init__(self, n_classes, dims=(32, 64, 128, 256), depths=(1, 1, 3, 1), stem_stride=1,
                 mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), ls_init=1e-6):
        super().__init__(mean, std)
        if stem_stride == 1:
            self.stem = nn.Sequential(nn.Conv2d(3, dims[0], 3, 1, 1), LayerNorm2d(dims[0]))
        else:
            self.stem = nn.Sequential(nn.Conv2d(3, dims[0], 4, stem_stride, 1), LayerNorm2d(dims[0]))
        self.down = nn.ModuleList([nn.Sequential(LayerNorm2d(dims[i]), nn.Conv2d(dims[i], dims[i + 1], 2, 2))
                                   for i in range(len(dims) - 1)])
        self.stages = nn.ModuleList([nn.ModuleList([Block(d, ls_init) for _ in range(n)]) for d, n in zip(dims, depths)])
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], n_classes)
        self.neuron_layers = [f"s{s + 1}b{b + 1}" for s, n in enumerate(depths) for b in range(n)]

    def _forward(self, x):
        x = self.stem(x)
        for s, blocks in enumerate(self.stages):
            if s > 0:
                x = self.down[s - 1](x)
            for b, blk in enumerate(blocks):
                name = f"s{s + 1}b{b + 1}"
                h = blk.dw(x).permute(0, 2, 3, 1)                    # B H W C
                z = blk.fc1(blk.ln(h))                              # B H W 4C
                if name in self.record or name in self.edit:
                    zc = z.permute(0, 3, 1, 2)
                    a = self._neuron(name, zc, F.gelu).permute(0, 2, 3, 1)
                else:
                    a = F.gelu(z)
                x = x + (blk.gamma * blk.fc2(a)).permute(0, 3, 1, 2)
        if "resid" in self.record:          # final residual stream (not a privileged basis; secondary view)
            self.acts["resid"] = x.detach()
        return self.head(self.norm(x.mean((2, 3))))


class Readout(nn.Module):
    """Independent judge for feature-visualization legibility (a small ResNet with a different design
    from the studied models). heads: {name: n_out}; categorical heads output logits, continuous heads 1 value."""

    def __init__(self, heads: dict, width=48, stem_stride=1, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        w = width
        layers = [nn.Conv2d(3, w, 3, stem_stride, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU()]
        for i, (cin, cout) in enumerate([(w, w), (w, 2 * w), (2 * w, 4 * w), (4 * w, 4 * w)]):
            layers += [_Res(cin, cout, 1 if i == 0 else 2)]
        self.body = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({k: nn.Linear(4 * w, n) for k, n in heads.items()})

    def forward(self, x):
        h = self.body((x - self.mean) / self.std).mean((2, 3))
        return {k: m(h) for k, m in self.heads.items()}


class _Res(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                   if (stride != 1 or cin != cout) else nn.Identity())

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sc(x))


ARCH_SIZES = {   # ConvNeXt size sweep (essay: 92k / 7.06M / 28.6M); "m" is parameter-matched to PlainCNN
    "s": dict(dims=(16, 32, 64, 96), depths=(1, 1, 2, 1)),
    "m": dict(dims=(32, 64, 128, 256), depths=(1, 1, 3, 1)),
    "l": dict(dims=(64, 128, 256, 512), depths=(1, 1, 3, 1)),
}


def build(kind: str, n_classes: int, stem_stride: int, mean, std, size: str = "m"):
    if kind == "plain":
        return PlainCNN(n_classes, stem_stride=stem_stride, mean=mean, std=std)
    if kind == "convnext":
        return ConvNeXtS(n_classes, stem_stride=stem_stride, mean=mean, std=std, **ARCH_SIZES[size])
    raise ValueError(kind)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for k, kw in [("plain", {}), ("convnext", {"size": "s"}), ("convnext", {"size": "m"}), ("convnext", {"size": "l"})]:
        m = build(k, 24, 2, (0.5,) * 3, (0.25,) * 3, **kw)
        y = m(torch.rand(2, 3, 64, 64), record=m.neuron_layers)
        print(k, kw, f"{n_params(m) / 1e6:.3f}M", {n: tuple(a.shape[1:]) for n, a in m.acts.items()})
