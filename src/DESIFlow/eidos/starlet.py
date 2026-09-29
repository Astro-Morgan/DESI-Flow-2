"""
Hardcoded (parameter-free) starlet processing paths for the Eidos CNN.

Both paths use the weighted (normalized-convolution) isotropic undecimated B3-spline starlet:
    c_0 = f,  w_0 = weights
    c_{j+1} = conv_j(c_j * w_j) / conv_j(w_j),   w_{j+1} = conv_j(w_j)       (conv_j: B3 kernel dilated by 2^j)
    detail_j = c_j - c_{j+1}          f = c_J + sum_j detail_j  exactly (telescoping)
Weights are ivar * good, so bad or noisy pixels are down-weighted and gaps are filled from coarser levels
without inpainting the input. Channels are ordered [detail_1 .. detail_J, coarse] (J+1 channels) and are
zeroed on not-good pixels (reconstruction is exact on good pixels).

RawStarletPath     x_raw (B, 3, N) = [flux, ivar, good]  ->  channels (B, J+1, N)   (signed; units of the input,
                   so feed it normalize_raw(x_raw, scale) for magnitude-invariant channels)
SmoothStarletPath  x_smooth (B, 2, N) = [flux_s, ivar_s]  ->  sqrtP (B, 2(J+1), N), support (B, J+1, N), scale (B,)
    sqrtP: Hellinger coordinates of the L1 composition the metric is defined on,
           P(i, lam, +/-) = max(+/-detail_i(lam), 0) / M,   M = sum_i sum_{good lam} |detail_i(lam)|,
           ordered [s1+, s1-, s2+, s2-, ..., coarse+, coarse-]; sum over good pixels of sqrtP^2 = 1, and
           for two spectra on the same pixels  0.5 * ||sqrtP_a - sqrtP_b||^2 = the compositional Hellinger^2.
    support: good-pixel mask passed through the same kernel cascade; channel i = fraction of the footprint of
             detail_i (the footprint of c_{i+1}) that lies on good pixels, in [0, 1] (coarse uses c_J's footprint).
    scale:   s = M / N_good, the composition mass per good pixel (flux units). The magnitude normalization for the
             whole network: raw inputs are divided by it (normalize_raw) and log10(s) is passed separately
             (eidos.embeddings.ScaleToken), so the spectral content is magnitude-invariant and the scale explicit.

normalize_raw(x_raw, scale) -> [flux / s, ivar * s^2, good]   (S/N per pixel unchanged)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_B3 = [1./16., 1./4., 3./8., 1./4., 1./16.]


class WeightedStarlet(nn.Module):
    """Parameter-free weighted starlet. forward(flux, weights, good=None) -> channels (B, J+1, N), support or None."""

    def __init__(self, n_scales: int = 9):
        super().__init__()
        self.n_scales = n_scales
        self.register_buffer("h", torch.tensor(_B3, dtype=torch.float32).view(1, 1, 5))

    def _conv(self, x, j):
        d = 2 ** j
        return F.conv1d(F.pad(x, (2 * d, 2 * d), mode="reflect"), self.h.to(x.dtype), dilation=d)

    def forward(self, flux: torch.Tensor, weights: torch.Tensor, good: torch.Tensor = None):
        # per-spectrum weight rescaling: the transform is invariant to it, and it keeps float32 well-conditioned
        w = weights / weights.amax(-1, keepdim=True).clamp_min(1e-30)
        c, w = flux.unsqueeze(1), w.unsqueeze(1)                                     # (B, 1, N)
        g = good.to(flux.dtype).unsqueeze(1) if good is not None else None
        details, supports = [], []
        for j in range(self.n_scales):
            num, den = self._conv(c * w, j), self._conv(w, j)
            c_next = torch.where(den > 1e-8, num / den.clamp_min(1e-8), torch.zeros_like(den))
            details.append(c - c_next)
            c, w = c_next, den
            if g is not None:
                g = self._conv(g, j)
                supports.append(g)
        channels = torch.cat(details + [c], dim=1)                                   # (B, J+1, N)
        support = torch.cat(supports + [g], dim=1).clamp(0, 1) if g is not None else None
        return channels, support


class RawStarletPath(nn.Module):
    """Weighted starlet of the raw (resampled, unsmoothed) spectrum. x_raw: (B, 3, N) = [flux, ivar, good]."""

    def __init__(self, n_scales: int = 9):
        super().__init__()
        self.starlet = WeightedStarlet(n_scales)

    def forward(self, x_raw: torch.Tensor):
        flux, ivar, good = x_raw[:, 0], x_raw[:, 1], x_raw[:, 2] > 0
        channels, _ = self.starlet(flux * good, ivar * good)
        return channels * good.unsqueeze(1)


def normalize_raw(x_raw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Magnitude-normalize the raw arrays by the per-spectrum scale s (B,): [flux / s, ivar * s^2, good]."""
    s = scale.view(-1, 1)
    return torch.stack([x_raw[:, 0] / s, x_raw[:, 1] * s ** 2, x_raw[:, 2]], dim=1)


class SmoothStarletPath(nn.Module):
    """Weighted starlet of the smoothed spectrum -> Hellinger coordinates of its L1 composition + support + scale.
    x_smooth: (B, 2, N) = [flux_s, ivar_s]; good where ivar_s > 0."""

    def __init__(self, n_scales: int = 9):
        super().__init__()
        self.starlet = WeightedStarlet(n_scales)

    def forward(self, x_smooth: torch.Tensor):
        flux, ivar = x_smooth[:, 0], x_smooth[:, 1]
        good = ivar > 0
        channels, support = self.starlet(flux * good, ivar * good, good)
        channels = channels * good.unsqueeze(1)
        mass = channels.abs().sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)         # M
        pos, neg = channels.clamp_min(0) / mass, (-channels).clamp_min(0) / mass
        sqrt_p = torch.stack([pos, neg], dim=2).flatten(1, 2).sqrt()                 # [s1+, s1-, s2+, s2-, ...]
        scale = mass.view(-1) / good.sum(-1).clamp_min(1)                            # M per good pixel
        return sqrt_p, support, scale
