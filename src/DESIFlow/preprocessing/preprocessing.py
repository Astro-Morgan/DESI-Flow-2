"""
This file has all the helper functions for preprocessing
Taking in DESI spectral arrays, cleaning them, formatting them, preparing them for model input
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

C_KMS = 299792.458


def _weighted_regression(flux, w, idx, K):
    """Inverse-variance-weighted kernel regression with a banded kernel table.
    flux, w: (B, L); idx, K: (N, W). Returns value, propagated ivar, and the good-kernel fraction, each (B, N)."""
    w_win = w[:, idx]                                                                  # (B, N, W)
    num = ((w * flux)[:, idx] * K).sum(-1)
    den = (w_win * K).sum(-1)                                                          # sum K w
    den2 = (w_win * K ** 2).sum(-1)                                                    # sum K^2 w
    ok = den > 0
    value = torch.where(ok, num / den.clamp_min(1e-30), torch.zeros_like(num))
    ivar = torch.where(ok, den ** 2 / den2.clamp_min(1e-30), torch.zeros_like(num))
    good_frac = ((w_win > 0).to(K.dtype) * K).sum(-1) / K.sum(-1).clamp_min(1e-30)
    return value, ivar, good_frac


class Preprocessor(nn.Module):
    """
    Resamples native (linear) DESI pixels onto a log-lambda grid in two ways, both as inverse-variance-weighted
    kernel regressions evaluated directly at the log-grid points (banded (index, weight) kernel tables):

      raw    : linear interpolation between the two neighbouring native pixels (hat kernel one native pixel
               wide) -- no smoothing beyond what resampling requires. Input for Eidos.
      smooth : Gaussian kernel fixed in velocity (sigma_v), v_ik = c ln(lam_i / lam_k), truncated at `truncate`
               sigma -- the denoised spectrum the metric is computed on. Identical smoothing at every
               wavelength (redshift-equivariant); one linear operator smooths and resamples, no aliasing.

        value(lam_k) = sum_i K_ik w_i f_i / sum_i K_ik w_i,   w_i = ivar_i * good_i

    Noise is propagated analytically (independent native pixels with variance 1/ivar):
        ivar_out(lam_k) = (sum_i K_ik w_i)^2 / sum_i K_ik^2 w_i
    Exact per output pixel; neighbouring output pixels are correlated (they share native pixels), which the
    per-pixel ivar does not describe -- most strongly for `smooth`, and for `raw` in the blue where one native
    pixel spans several log pixels.

    forward(x) with x: (B, 3, spec_len) = [flux, ivar, mask], mask a bitmask with 0 == good
      -> x_raw:    (B, 3, new_len) = [flux, ivar, good]   good = 1 only if EVERY native pixel interpolated from
                                                           is good (no extension into masked pixels)
         x_smooth: (B, 2, new_len) = [flux_s, ivar_s]     good where ivar_s > 0 (>= 1 good native pixel in reach)
    Not-good output pixels have flux = ivar = 0. No inpainting: gaps stay not-good (the metric renormalizes over
    mutually good rest-frame pixels).
    """
    def __init__(self, wave_min: float = 3600., wave_max: float = 9824., spec_len: int = 7781, new_len: int = 13000,
                 sigma_v: float = 77., truncate: float = 4.):
        super().__init__()
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.spec_len = spec_len
        self.new_len = new_len
        self.sigma_v = sigma_v
        self.truncate = truncate

        native_wave = np.linspace(wave_min, wave_max, spec_len)
        new_wave = np.logspace(np.log10(wave_min), np.log10(wave_max), new_len)
        d_lambda = (wave_max - wave_min) / (spec_len - 1)
        pos = (new_wave - wave_min) / d_lambda                                         # fractional native pixel

        # smooth: Gaussian in velocity; widest reach in native pixels is at the red end
        reach = int(np.ceil(truncate * sigma_v / C_KMS * wave_max / d_lambda)) + 1
        idx = np.floor(pos).astype(np.int64)[:, None] - reach + np.arange(2 * reach + 2)[None, :]
        in_bounds = (idx >= 0) & (idx < spec_len)
        idx = np.clip(idx, 0, spec_len - 1)
        dv = C_KMS * np.log(native_wave[idx] / new_wave[:, None])                      # exact velocity offsets
        kernel = np.exp(-0.5 * (dv / sigma_v) ** 2) * in_bounds * (np.abs(dv) <= truncate * sigma_v)

        # raw: hat kernel one native pixel wide = linear interpolation between the two neighbours
        idx_raw = np.floor(pos).astype(np.int64)[:, None] + np.arange(2)[None, :]
        in_bounds_raw = (idx_raw >= 0) & (idx_raw < spec_len)
        idx_raw = np.clip(idx_raw, 0, spec_len - 1)
        kernel_raw = np.clip(1.0 - np.abs(idx_raw - pos[:, None]), 0.0, None) * in_bounds_raw

        self.register_buffer('idx', torch.as_tensor(idx, dtype=torch.long))
        self.register_buffer('kernel', torch.as_tensor(kernel, dtype=torch.float32))
        self.register_buffer('idx_raw', torch.as_tensor(idx_raw, dtype=torch.long))
        self.register_buffer('kernel_raw', torch.as_tensor(kernel_raw, dtype=torch.float32))
        self.register_buffer('new_wave', torch.as_tensor(new_wave, dtype=torch.float64))

    def forward(self, x: torch.Tensor):
        # x = [flux, ivar, mask], mask is bitmask 0==good
        flux, ivar, mask = x[:, 0], x[:, 1], x[:, 2]
        good = (mask == 0) & torch.isfinite(flux) & torch.isfinite(ivar) & (ivar > 0)

        flux = torch.nan_to_num(flux, 0.0)
        ivar = torch.nan_to_num(ivar, 0.0)
        w = ivar * good                                                                # (B, L)

        flux_r, ivar_r, frac_r = _weighted_regression(flux, w, self.idx_raw, self.kernel_raw)
        good_r = frac_r > 1 - 1e-6                                                     # all interpolated-from px good
        flux_r = torch.where(good_r, flux_r, torch.zeros_like(flux_r))
        ivar_r = torch.where(good_r, ivar_r, torch.zeros_like(ivar_r))

        flux_s, ivar_s, _ = _weighted_regression(flux, w, self.idx, self.kernel)

        x_raw = torch.stack([flux_r, ivar_r, good_r.to(flux_r.dtype)], dim=1)
        x_smooth = torch.stack([flux_s, ivar_s], dim=1)
        return x_raw, x_smooth
