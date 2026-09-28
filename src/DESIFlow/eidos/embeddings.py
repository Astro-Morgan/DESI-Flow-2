"""
Learned embeddings for the Eidos token sequence.

ScaleToken: embeds the per-spectrum magnitude scale (log10 s, from SmoothStarletPath) as ONE dedicated token,
so the spectral tokens stay magnitude-invariant and the scale sits in a single known position (Plato can be
given every token except this one).

    x      = log10(s) - center
    feats  = [x, sin(2 pi x / P_k), cos(2 pi x / P_k)]   for periods P_k geometric in [min_period, max_period] dex
    token  = MLP(feats) + learned type embedding          -> (B, 1, d_token)

The sinusoidal features give the MLP both fine resolution (min_period ~0.05 dex) and an unambiguous code over a
wide range (max_period ~20 dex >> DESI's spread of log10 s); x itself keeps an explicitly monotonic channel.
"""
import math
import torch
import torch.nn as nn


class ScaleToken(nn.Module):
    def __init__(self, d_token: int, n_freqs: int = 16, min_period: float = 0.05, max_period: float = 20.,
                 center: float = 0.5, hidden: int = None):
        super().__init__()
        self.center = center
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), n_freqs)
        self.register_buffer("omega", 2 * math.pi / periods)
        hidden = hidden or d_token
        self.mlp = nn.Sequential(nn.Linear(2 * n_freqs + 1, hidden), nn.SiLU(), nn.Linear(hidden, d_token))
        self.type_embedding = nn.Parameter(torch.zeros(d_token))                     # marks the token as "scale"

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        """scale: (B,) linear scale s > 0  ->  (B, 1, d_token)"""
        x = (torch.log10(scale.clamp_min(1e-30)) - self.center).unsqueeze(-1)       # (B, 1)
        feats = torch.cat([x, torch.sin(x * self.omega), torch.cos(x * self.omega)], dim=-1)
        return (self.mlp(feats) + self.type_embedding).unsqueeze(1)
