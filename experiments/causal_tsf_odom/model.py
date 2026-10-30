import math
import torch
from torch import nn


class Haar(nn.Module):
    """Right-aligned Haar features; the input window is already in the past."""
    def forward(self, x):
        even, odd = x[..., 0::2], x[..., 1::2]
        return (even + odd) * 0.70710678, (even - odd) * 0.70710678


class GraphMix(nn.Module):
    """Two physical nodes (accelerometer and gyroscope), no fake body graph."""
    def __init__(self, width):
        super().__init__()
        self.node = nn.Conv1d(3, width, 1)
        self.out = nn.Conv1d(width * 2, width, 1)
        self.edge = nn.Sequential(nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1))

    def forward(self, x):
        nodes = torch.stack((self.node(x[:, :3]), self.node(x[:, 3:])), dim=1)
        summary = nodes.mean(-1)
        sim = torch.sigmoid(self.edge(summary).squeeze(-1))
        low = nodes.mean(1)
        high = nodes[:, 0] - nodes[:, 1]
        return self.out(torch.cat((low * sim[:, :1, None], high * (1 - sim[:, 1:, None])), dim=1))


class PatchMix(nn.Module):
    """Small cross-patch mixer inspired by lightweight ICDE forecasting models."""
    def __init__(self, width, patches):
        super().__init__()
        self.mix = nn.Linear(patches, patches, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, h):
        mixed = self.mix(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(h + mixed)


class CausalTSF(nn.Module):
    def __init__(self, variant="time", width=64, patches=10):
        super().__init__()
        if variant not in {"time", "tsf", "tsf_patch", "tsf_gru"}:
            raise ValueError(f"unknown variant {variant}")
        self.variant = variant
        self.width = width
        self.time = nn.Sequential(nn.Conv1d(6, width, 9, padding=4), nn.GELU(),
                                  nn.Conv1d(width, width, 5, padding=2), nn.GELU())
        # A one-way sequence mixer over the already-complete current window.
        # It is causal with respect to output blocks and gives the TSF branch
        # an order-aware summary without reading a future block.
        self.temporal_gru = nn.GRU(width, width, batch_first=True)
        if variant != "tsf_gru":
            for parameter in self.temporal_gru.parameters():
                parameter.requires_grad_(False)
        self.freq = nn.Sequential(nn.Conv1d(12, width, 1), nn.GELU(), nn.Conv1d(width, width, 1))
        self.graph = GraphMix(width)
        self.haar = Haar()
        self.wave = nn.Conv1d(6, width, 1)
        branches = 1 if variant == "time" else 4
        self.patch = PatchMix(width, patches) if variant == "tsf_patch" else nn.Identity()
        self.fuse = nn.Sequential(nn.Linear(width * branches, width * 2), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(width * 2, width))
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))

    def forward(self, x):
        time = self.time(x)
        if self.variant == "time":
            z = time.mean(-1)
        else:
            # Real and imaginary parts retain signed information, unlike magnitude-only spectra.
            spec = torch.fft.rfft(x, dim=-1)
            freq = self.freq(torch.cat((spec.real, spec.imag), dim=1))
            graph = self.graph(x)
            low, high = self.haar(x)
            wave = self.wave(torch.cat((low, high), dim=-1))
            wave = torch.nn.functional.adaptive_avg_pool1d(wave, x.shape[-1])
            if self.variant == "tsf_gru":
                _, hidden = self.temporal_gru(time.transpose(1, 2))
                time_summary = hidden[-1]
            else:
                time_summary = time.mean(-1)
            tokens = torch.nn.functional.adaptive_avg_pool1d(time, 10).transpose(1, 2)
            time = self.patch(tokens).transpose(1, 2)
            z = torch.cat((time_summary, freq.mean(-1), graph.mean(-1), wave.mean(-1)), dim=1)
        return self.head(self.fuse(z))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
