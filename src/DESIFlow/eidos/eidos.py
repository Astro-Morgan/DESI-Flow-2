"""
Eidos - The Autoencoder that hosts Plato (the hellinger metric encoder)
Inception-like CNN + Transformer Masked Denoising Autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DESIFlow.eidos.starlet import SmoothStarletPath, RawStarletPath, normalize_raw

_ACT = {"silu": nn.SiLU, "gelu": nn.GELU}
_IVAR_ENC = {"log1p": torch.log1p, "sqrt": torch.sqrt, "raw": lambda v: v}


class CNNBranch(nn.Module):
    """
    Shallow CNN for one starlet scale: `n_layers` x [Conv1d(k, dilation) -> GroupNorm(1) -> activation].
    The dilation follows the a-trous scheme of the scale it serves (2^(j-1) for detail scale j), so with k = 5 and two
    layers the receptive field is 1 + n_layers (k-1) 2^(j-1) = 4 * 2^j + 1 px -- the footprint of starlet scale j.
    Input (B, in_channels, N) -> (B, channel_size, N), same length (reflect padding).
    """
    def __init__(self, in_channels: int, channel_size: int, dilation: int, kernel_size: int = 5, n_layers: int = 2,
                 activation: str = "silu"):
        super().__init__()
        self.channel_size = channel_size
        self.dilation = dilation
        pad = dilation * (kernel_size - 1) // 2
        layers, c_in = [], in_channels
        for _ in range(n_layers):
            layers += [nn.Conv1d(c_in, channel_size, kernel_size, dilation=dilation, padding=pad, padding_mode="reflect",
                                 bias=False),
                       nn.GroupNorm(1, channel_size), _ACT[activation]()]
            c_in = channel_size
        self.net = nn.Sequential(*layers)
        self.receptive_field = 1 + n_layers * (kernel_size - 1) * dilation

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """
    Eidos CNN front end.

    1. Hardcoded paths: SmoothStarletPath(x_smooth) -> sqrtP (2(J+1) ch), support (J+1 ch), scale s;
       raw arrays magnitude-normalized by s (normalize_raw), then RawStarletPath -> raw channels (J+1 ch).
    2. Per-scale grouping, for each of the J detail scales and the coarse channel (J+1 groups):
           x_i = [raw_i, smooth_i+, smooth_i-, support_i, flux, enc(ivar), good]            (7 channels, full res)
       smooth_i+/- are multiplied by the constant sqrt(N) (input conditioning only: brings sqrtP ~1e-3 to the
       O(0.1-1) range of the other channels; a global constant, so the Hellinger geometry is only rescaled).
       enc(ivar) = log1p(ivar) by default (normalized ivar spans ~0..1e4).
    3. One independent CNNBranch per group, dilation 2^(i-1) (coarse group uses the last detail scale's dilation).
    4. Mixing: concat all branch outputs -> 1x1 conv -> `len(down_channels)` stride-2 conv stages -> 1x1 projection
       to d_token. Every conv is followed by GroupNorm(1) and the activation, except the final projection.

    forward(x_raw, x_smooth) -> tokens (B, N_tok, d_token), scale (B,)
        N_tok = ceil(N / 2^len(down_channels)); token t is centred on log-lambda pixel 2^len(down_channels) * t
        (see token_wave). The scale feeds the (separate) ScaleToken; no starlet fast track yet.
    """
    def __init__(self, n_scales: int = 9, new_len: int = 13000, branch_channels: int = 16, kernel_size: int = 5,
                 branch_layers: int = 2, mix_channels: int = 64, down_channels: tuple = (128, 256, 256, 256),
                 d_token: int = 256, activation: str = "silu", ivar_encoding: str = "log1p"):
        super().__init__()
        self.n_scales = n_scales
        self.new_len = new_len
        self.ivar_encoding = ivar_encoding
        self.sqrtp_gain = float(np.sqrt(new_len))
        self.smooth_path = SmoothStarletPath(n_scales)
        self.raw_path = RawStarletPath(n_scales)

        dilations = [2 ** j for j in range(n_scales)] + [2 ** (n_scales - 1)]
        self.branches = nn.ModuleList([CNNBranch(7, branch_channels, d, kernel_size, branch_layers, activation)
                                       for d in dilations])

        act = _ACT[activation]
        layers = [nn.Conv1d((n_scales + 1) * branch_channels, mix_channels, 1, bias=False),
                  nn.GroupNorm(1, mix_channels), act()]
        c_in = mix_channels
        for c_out in down_channels:
            layers += [nn.Conv1d(c_in, c_out, 5, stride=2, padding=2, padding_mode="reflect", bias=False),
                       nn.GroupNorm(1, c_out), act()]
            c_in = c_out
        layers += [nn.Conv1d(c_in, d_token, 1)]
        self.mixer = nn.Sequential(*layers)
        self.downsample = 2 ** len(down_channels)

    def scale_groups(self, x_raw, x_smooth):
        """Returns the list of J+1 per-scale inputs (B, 7, N) and the magnitude scale s (B,)."""
        sqrt_p, support, scale = self.smooth_path(x_smooth)
        xn = normalize_raw(x_raw, scale)
        raw = self.raw_path(xn)
        shared = [xn[:, 0:1], _IVAR_ENC[self.ivar_encoding](xn[:, 1:2]), xn[:, 2:3]]
        groups = [torch.cat([raw[:, i:i + 1], self.sqrtp_gain * sqrt_p[:, 2 * i:2 * i + 2], support[:, i:i + 1], *shared], 1)
                  for i in range(self.n_scales + 1)]
        return groups, scale

    def forward(self, x_raw: torch.Tensor, x_smooth: torch.Tensor):
        groups, scale = self.scale_groups(x_raw, x_smooth)
        feats = torch.cat([branch(g) for branch, g in zip(self.branches, groups)], 1)   # (B, (J+1)*Cb, N)
        tokens = self.mixer(feats)                                                     # (B, d_token, N_tok)
        return tokens.transpose(1, 2), scale

    def token_wave(self, new_wave: torch.Tensor) -> torch.Tensor:
        """Log-lambda grid (new_len,) -> wavelength at each token centre (N_tok,)."""
        return new_wave[:: self.downsample]
