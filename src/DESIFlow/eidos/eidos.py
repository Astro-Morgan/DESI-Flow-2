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


def to_odd(x):
    x = int(round(x))
    return x if x % 2 == 1 else x + 1


def starlet_footprint(j: int) -> int:
    """Support (px) of the a-trous B3 starlet detail scale j (1-indexed): 4 (2^j - 1) + 1."""
    return 4 * (2 ** j - 1) + 1


class MaskedGroupNorm(nn.Module):
    """
    GroupNorm(num_groups=1) whose per-sample statistics are WEIGHTED averages over positions:
        mu  = sum_{c,t} w_t x_ct / (C sum_t w_t),   var = sum_{c,t} w_t (x_ct - mu)^2 / (C sum_t w_t)
        y   = (x - mu) / sqrt(var + eps) * gamma + beta
    w (B, 1, L) in [0, 1] (0 = masked / no data). No batch statistics; with w == 1 this is exactly GroupNorm(1).
    Masked positions are still normalized (with the valid-data statistics) but never contribute to the statistics,
    so gaps and low-quality regions do not set the normalization.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels)) if affine else None
        self.bias = nn.Parameter(torch.zeros(num_channels)) if affine else None

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        wsum = w.sum(dim=(1, 2), keepdim=True)                                          # (B, 1, 1)
        w = torch.where(wsum > 0, w, torch.ones_like(w))                                # no data at all -> unweighted
        wsum = w.sum(dim=(1, 2), keepdim=True) * C
        mu = (x * w).sum(dim=(1, 2), keepdim=True) / wsum
        var = ((x - mu) ** 2 * w).sum(dim=(1, 2), keepdim=True) / wsum
        y = (x - mu) * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return y


class ConvNormAct(nn.Module):
    """Conv1d (no bias) -> MaskedGroupNorm -> activation; forward(x, w) with w the norm weights at the OUTPUT length."""
    def __init__(self, c_in, c_out, kernel_size, dilation=1, stride=1, activation="silu"):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, stride=stride, dilation=dilation,
                              padding=dilation * (kernel_size - 1) // 2, padding_mode="reflect", bias=False)
        self.norm = MaskedGroupNorm(c_out)
        self.act = _ACT[activation]()

    def forward(self, x, w):
        return self.act(self.norm(self.conv(x), w))


class CNNBranch(nn.Module):
    """
    Shallow CNN for one starlet scale, sized by that scale's FEATURE WIDTH (px): `n_layers` x ConvNormAct whose
    combined receptive field ~ feature_width. Each layer spans (feature_width - 1) / n_layers px: dense kernels while
    that fits in `base_k` taps, otherwise `base_k` taps dilated to cover it.
    Input (B, in_channels, N) -> (B, out_channels, N), same length.
    """
    def __init__(self, in_channels: int, out_channels: int, feature_width: int, n_layers: int = 2, base_k: int = 31,
                 activation: str = "silu"):
        super().__init__()
        self.feature_width = feature_width
        span = (feature_width - 1) / n_layers                                          # px per layer
        k = min(to_odd(span + 1), to_odd(base_k))
        d = max(1, int(round(span / (k - 1))))
        self.kernel_size, self.dilation = k, d
        self.layers = nn.ModuleList([ConvNormAct(in_channels if i == 0 else out_channels, out_channels, k, d,
                                                 activation=activation) for i in range(n_layers)])
        self.receptive_field = 1 + n_layers * (k - 1) * d

    def forward(self, x, w):
        for layer in self.layers:
            x = layer(x, w)
        return x


class CNN(nn.Module):
    """
    Eidos CNN front end.

    1. Magnitude normalization by the per-spectrum scale s = M / N_good (from SmoothStarletPath), applied the same
       way to every path, so all inputs are at the same O(1) scale and s is tracked separately (ScaleToken):
         raw:    [flux / s, ivar s^2, good] (normalize_raw) -> RawStarletPath -> raw_i = detail_i / s
         smooth: sign-split sqrt(|detail_i| / s) = sqrtP * sqrt(N_good)   (Hellinger coordinates up to the known
                 per-spectrum factor sqrt(N_good); the metric uses the exact sqrtP)
    2. Per-scale grouping, for each of the J detail scales and the coarse channel (J+1 groups):
           x_i = [raw_i, smooth_i+, smooth_i-, support_i, flux/s, log1p(ivar s^2), good]     (7 channels, full res)
    3. One independent CNNBranch per group, feature width = footprint of starlet scale i (coarse: scale J).
    4. Mixing: concat branch outputs -> 1x1 ConvNormAct -> stride-2 ConvNormAct stages -> 1x1 projection to d_token.
    Every norm is a MaskedGroupNorm with weights w = good * q, q = ivar/(ivar+1) on normalized ivar (quality="snr",
    down-weights pixels with per-pixel S/N below ~1) or q = 1 (quality="good"); after each stride-2 stage the
    weights are downsampled by the same k=5/stride-2 window (average over valid positions).

    forward(x_raw, x_smooth) -> tokens (B, N_tok, d_token), scale (B,)
        N_tok = ceil(N / 2^len(down_channels)); token t is centred on log-lambda pixel 2^len(down_channels) * t.
    """
    def __init__(self, n_scales: int = 9, new_len: int = 13000, branch_channels: int = 16, branch_layers: int = 2,
                 base_k: int = 31, mix_channels: int = 64, down_channels: tuple = (128, 256, 256, 256),
                 d_token: int = 256, activation: str = "silu", ivar_encoding: str = "log1p", quality: str = "snr"):
        super().__init__()
        self.n_scales = n_scales
        self.new_len = new_len
        self.ivar_encoding = ivar_encoding
        self.quality = quality
        self.smooth_path = SmoothStarletPath(n_scales)
        self.raw_path = RawStarletPath(n_scales)

        widths = [starlet_footprint(j) for j in range(1, n_scales + 1)] + [starlet_footprint(n_scales)]
        self.branches = nn.ModuleList([CNNBranch(7, branch_channels, fw, branch_layers, base_k, activation)
                                       for fw in widths])
        self.mix_in = ConvNormAct((n_scales + 1) * branch_channels, mix_channels, 1, activation=activation)
        chans = [mix_channels] + list(down_channels)
        self.down = nn.ModuleList([ConvNormAct(chans[i], chans[i + 1], 5, stride=2, activation=activation)
                                   for i in range(len(down_channels))])
        self.proj = nn.Conv1d(chans[-1], d_token, 1)
        self.downsample = 2 ** len(down_channels)

    def scale_groups(self, x_raw, x_smooth):
        """Returns the J+1 per-scale inputs (B, 7, N), the norm weights (B, 1, N) and the scale s (B,)."""
        sqrt_p, support, scale = self.smooth_path(x_smooth)
        n_good = (x_smooth[:, 1] > 0).sum(-1).clamp_min(1).to(sqrt_p.dtype).view(-1, 1, 1)
        smooth = sqrt_p * n_good.sqrt()                                                # = sqrt(|detail| / s)
        xn = normalize_raw(x_raw, scale)
        raw = self.raw_path(xn)
        good = xn[:, 2:3]
        q = xn[:, 1:2] / (xn[:, 1:2] + 1) if self.quality == "snr" else torch.ones_like(good)
        shared = [xn[:, 0:1], _IVAR_ENC[self.ivar_encoding](xn[:, 1:2]), good]
        groups = [torch.cat([raw[:, i:i + 1], smooth[:, 2 * i:2 * i + 2], support[:, i:i + 1], *shared], 1)
                  for i in range(self.n_scales + 1)]
        return groups, good * q, scale

    def forward(self, x_raw: torch.Tensor, x_smooth: torch.Tensor):
        groups, w, scale = self.scale_groups(x_raw, x_smooth)
        x = torch.cat([branch(g, w) for branch, g in zip(self.branches, groups)], 1)    # (B, (J+1)*Cb, N)
        x = self.mix_in(x, w)
        for stage in self.down:
            w = self._downsample_weights(w)
            x = stage(x, w)
        return self.proj(x).transpose(1, 2), scale                                     # (B, N_tok, d_token)

    @staticmethod
    def _downsample_weights(w):
        """Average of the weights over each k=5 / stride-2 window (valid positions only, matching the conv grid)."""
        return F.avg_pool1d(w, 5, stride=2, padding=2, count_include_pad=False)

    def token_wave(self, new_wave: torch.Tensor) -> torch.Tensor:
        """Log-lambda grid (new_len,) -> wavelength at each token centre (N_tok,)."""
        return new_wave[:: self.downsample]
