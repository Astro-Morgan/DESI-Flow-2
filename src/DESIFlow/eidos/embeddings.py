"""
Embeddings for the Eidos token sequence.

ScaleToken: deterministic, parameter-free encoding of the per-spectrum magnitude scale s (from SmoothStarletPath,
in the input flux units, i.e. DESI's 1e-17 erg/s/cm^2/A) as ONE dedicated token, so the spectral tokens stay
magnitude-invariant and the scale sits in a single known position (Plato can be given every token except this one).

    x     = log10(s)
    token = [x, sin(2 pi x / P_1..K), cos(2 pi x / P_1..K), 1, 0, ..., 0]      (B, 1, d_token),  d_token >= 2K + 2

Nothing here is learned: no current objective depends on the scale, so a learned embedding would receive no
information-carrying gradient and could discard it. The encoding is exact (x itself is slot 0, so the scale is
always recoverable), the sinusoids (periods P_k geometric in [min_period, max_period] dex) give the transformer's
learned Q/K/V projections fine-resolution and wide-range features to read if reconstruction finds the scale useful,
and the constant-1 slot marks the token as the scale token.
"""
import math
import torch
import torch.nn as nn


class ScaleToken(nn.Module):
    def __init__(self, d_token: int, n_freqs: int = 16, min_period: float = 0.05, max_period: float = 20.):
        super().__init__()
        if d_token < 2 * n_freqs + 2:
            raise ValueError(f"d_token={d_token} < 2*n_freqs+2={2 * n_freqs + 2}")
        self.d_token = d_token
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), n_freqs)
        self.register_buffer("omega", 2 * math.pi / periods)

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        """scale: (B,) linear scale s > 0  ->  (B, 1, d_token)"""
        x = torch.log10(scale.clamp_min(1e-30)).unsqueeze(-1)                       # (B, 1)
        feats = torch.cat([x, torch.sin(x * self.omega), torch.cos(x * self.omega), torch.ones_like(x)], dim=-1)
        pad = feats.new_zeros(feats.shape[0], self.d_token - feats.shape[1])
        return torch.cat([feats, pad], dim=-1).unsqueeze(1)
