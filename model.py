"""A U-Net denoising model for DDPM, from scratch.

Residual blocks with time conditioning (group norm -> SiLU -> conv, time
embedding added after the first conv), self-attention at low resolutions, and
an encoder-decoder structure with skip connections. No ``diffusers`` — just
``torch.nn``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding (NeRF/Vaswani-style)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_embedding(t, self.net[0].in_features))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = ch // n_heads
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).reshape(B, 3, self.n_heads, self.head_dim, H * W).unbind(1)
        scale = self.head_dim**-0.5
        attn = torch.einsum("bhdl,bhdm->bhlm", q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhlm,bhdm->bhdl", attn, v).reshape(B, C, H, W)
        return self.proj(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, use_attn, dropout):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, time_dim, dropout)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return self.attn(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, use_attn, dropout):
        super().__init__()
        self.res1 = ResBlock(in_ch * 2, out_ch, time_dim, dropout)  # concat skip
        self.res2 = ResBlock(out_ch, out_ch, time_dim, dropout)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return self.attn(x)


class UNet(nn.Module):
    """DDPM U-Net for ``image_size x image_size`` inputs."""

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 64,
        ch_mults: tuple[int, ...] = (1, 2, 2, 2),
        time_dim: int = 256,
        attn_res: tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        image_size: int = 32,
    ):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        chs = [base_ch * m for m in ch_mults]

        self.in_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i in range(len(chs) - 1):
            use_attn = (image_size >> i) in attn_res
            self.down_blocks.append(DownBlock(chs[i], chs[i + 1], time_dim, use_attn, dropout))
            self.downsamplers.append(nn.Conv2d(chs[i + 1], chs[i + 1], 3, stride=2, padding=1))

        self.mid1 = ResBlock(chs[-1], chs[-1], time_dim, dropout)
        self.mid_attn = SelfAttention(chs[-1])
        self.mid2 = ResBlock(chs[-1], chs[-1], time_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.upsamplers.append(nn.ConvTranspose2d(chs[i + 1], chs[i + 1], 4, stride=2, padding=1))
            use_attn = (image_size >> i) in attn_res
            self.up_blocks.append(UpBlock(chs[i + 1], chs[i], time_dim, use_attn, dropout))

        self.out_norm = nn.GroupNorm(32, chs[0])
        self.out_conv = nn.Conv2d(chs[0], in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        x = self.in_conv(x)

        skips: list[torch.Tensor] = []
        for block, downsample in zip(self.down_blocks, self.downsamplers):
            x = block(x, t_emb)
            skips.append(x)
            x = downsample(x)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        for block, upsample in zip(self.up_blocks, self.upsamplers):
            x = upsample(x)
            x = block(x, skips.pop(), t_emb)

        x = F.silu(self.out_norm(x))
        return self.out_conv(x)
