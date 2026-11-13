"""
Swipe encoder with exactly the inputs and outputs of FUTO's `honorable_sturgeon`, so the
keyboard's swipe engine can use it unchanged:

    inputs   features     [1, 2, 64]   path (x, y) resampled to 64 points, model frame 0..1
             layout_keys  [1, 64, 2]   key centres, padded to 64 keys
             layout_mask  [1, 64]      which key slots are real
    outputs  log_emissions [1, 32, 65] log-probabilities over 64 key slots + blank (last)
             coefficients  [1, 32, 64] (only used by FUTO's layout refiner; we output features)
             lambda        [1, 32, 1]  (ditto)

It works for any keyboard layout because keys are not fixed outputs: each key slot is scored
from its position (a learned key embedding of its coordinates, plus how close the finger was).

Only operations that the phone's swipe engine supports are used: everything heavy runs on
XNNPACK (conv, linear, matmul, elementwise), the rest is limited to FUTO's op list
(log_softmax, where, layer_norm, ...). `export.py` checks this.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

T_IN = 64
T_OUT = 32
MAX_KEYS = 64
COEFF_DIM = 64


@dataclass
class ModelConfig:
    dim: int = 192           # channels
    blocks: int = 8          # conv blocks
    kernel: int = 7
    expansion: int = 3
    attn_layers: int = 2     # self-attention layers after the conv stack (0 = none)
    heads: int = 4
    key_dim: int = 128       # size of the key / timestep embeddings that are matched
    fourier: int = 16        # Fourier features for coordinates
    dropout: float = 0.1

    @staticmethod
    def preset(name: str) -> "ModelConfig":
        return {
            # ~1.5M params: about 2x FUTO's size
            "small": ModelConfig(dim=160, blocks=6, attn_layers=1, key_dim=96),
            # ~4M params
            "medium": ModelConfig(dim=224, blocks=8, attn_layers=2, key_dim=128),
            # ~9M params: still well under 100 ms on a flagship phone
            "large": ModelConfig(dim=320, blocks=10, attn_layers=3, heads=8, key_dim=160),
            # tiny, for smoke tests
            "tiny": ModelConfig(dim=48, blocks=2, attn_layers=1, heads=2, key_dim=32, fourier=4),
        }[name]

    def to_dict(self):
        return asdict(self)


class Fourier(nn.Module):
    """Fixed sin/cos features of 2-D coordinates (frequencies 1..2^(n-1) * pi)."""
    def __init__(self, n: int):
        super().__init__()
        freqs = torch.tensor([math.pi * (2.0 ** i) for i in range(n)], dtype=torch.float32)
        self.register_buffer("freqs", freqs)
        self.out_dim = 2 + 4 * n

    def forward(self, xy):  # [..., 2]
        x = xy[..., 0:1] * self.freqs
        y = xy[..., 1:2] * self.freqs
        return torch.cat([xy, torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)


class ConvBlock(nn.Module):
    """ConvNeXt-style 1-D block: depthwise conv -> norm -> pointwise MLP, residual."""
    def __init__(self, dim, kernel, dilation, expansion, dropout):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.dw = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * expansion)
        self.pw2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # [B, C, T]
        y = self.dw(x).transpose(1, 2)            # [B, T, C]
        y = self.pw2(F.gelu(self.pw1(self.norm(y))))
        return x + self.drop(y).transpose(1, 2)


class AttnBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, dim * 3)
        self.ff2 = nn.Linear(dim * 3, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # [B, T, C]
        B, T, C = x.shape
        h = self.heads
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.reshape(B, T, h, C // h).transpose(1, 2)
        k = k.reshape(B, T, h, C // h).transpose(1, 2)
        v = v.reshape(B, T, h, C // h).transpose(1, 2)
        att = torch.softmax((q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(C // h)), dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff2(F.gelu(self.ff1(self.norm2(x)))))


class SwipeEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.fourier = Fourier(cfg.fourier)
        # Per point: Fourier(x, y) + velocity (dx, dy)
        self.stem = nn.Conv1d(self.fourier.out_dim + 2, d, 5, padding=2)
        dilations = [1, 2, 4, 1, 2, 4, 8, 1, 2, 4, 8, 1]
        self.blocks1 = nn.ModuleList([ConvBlock(d, cfg.kernel, dilations[i], cfg.expansion, cfg.dropout)
                                      for i in range(cfg.blocks // 2)])
        self.down = nn.Conv1d(d, d, 2, stride=2)  # 64 -> 32 steps
        self.blocks2 = nn.ModuleList([ConvBlock(d, cfg.kernel, dilations[i], cfg.expansion, cfg.dropout)
                                      for i in range(cfg.blocks - cfg.blocks // 2)])
        self.attn = nn.ModuleList([AttnBlock(d, cfg.heads, cfg.dropout) for _ in range(cfg.attn_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.to_query = nn.Linear(d, cfg.key_dim)
        self.to_blank = nn.Linear(d, 1)
        self.to_coeff = nn.Linear(d, COEFF_DIM)
        self.to_lambda = nn.Linear(d, 1)
        self.key_mlp = nn.Sequential(
            nn.Linear(self.fourier.out_dim, cfg.key_dim * 2), nn.GELU(), nn.Linear(cfg.key_dim * 2, cfg.key_dim))
        # Learned weight on "how close was the finger to this key" (per output step)
        self.to_dist_w = nn.Linear(d, 1)
        # Fixed averaging conv to get path positions at the 32 output steps
        self.register_buffer("avg_w", torch.full((2, 1, 2), 0.5))
        # Central difference as a fixed depthwise conv (runs on XNNPACK)
        self.register_buffer("diff_w", torch.tensor([[[-0.5, 0.0, 0.5]], [[-0.5, 0.0, 0.5]]]))
        self.scale = 1.0 / math.sqrt(cfg.key_dim)

    def forward(self, features, layout_keys, layout_mask):
        # features [B,2,64], layout_keys [B,64,2], layout_mask [B,64] (bool)
        B = features.shape[0]
        pts = features.transpose(1, 2)                          # [B,64,2]
        vel = F.conv1d(features, self.diff_w, padding=1, groups=2)  # [B,2,64] velocity
        x = torch.cat([self.fourier(pts).transpose(1, 2), vel], dim=1)
        x = self.stem(x)
        for b in self.blocks1:
            x = b(x)
        x = self.down(x)
        for b in self.blocks2:
            x = b(x)
        h = x.transpose(1, 2)                                   # [B,32,C]
        for a in self.attn:
            h = a(h)
        h = self.out_norm(h)

        q = self.to_query(h)                                    # [B,32,D]
        e = self.key_mlp(self.fourier(layout_keys))             # [B,64,D]
        logits = (q @ e.transpose(1, 2)) * self.scale           # [B,32,64]

        # Distance prior: path position at each output step vs. each key
        p = F.conv1d(features, self.avg_w, stride=2, groups=2).transpose(1, 2)  # [B,32,2]
        dx = p[..., 0:1] - layout_keys[..., 0].unsqueeze(1)    # [B,32,64]
        dy = p[..., 1:2] - layout_keys[..., 1].unsqueeze(1)
        dist2 = dx * dx + dy * dy
        w = torch.sigmoid(self.to_dist_w(h)) * 40.0             # [B,32,1] (sigmoid: phone-safe op)
        logits = logits - w * dist2

        mask = layout_mask.unsqueeze(1).expand(B, T_OUT, MAX_KEYS)
        logits = torch.where(mask, logits, torch.full_like(logits, -1e4))
        blank = self.to_blank(h)                                # [B,32,1]
        log_emissions = torch.log_softmax(torch.cat([logits, blank], dim=-1), dim=-1)
        return log_emissions, self.to_coeff(h), self.to_lambda(h)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
