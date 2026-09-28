"""
This file has all the helper functions for preprocessing
Taking in DESI spectral arrays, cleaning them, formatting them, preparing them for model input
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_odd(x):
    x = int(x)
    return x if x % 2 == 1 else x + 1

def CosineInterpolate(x):
    flux, ivar, mask = x[:, 0], x[:, 1], x[:, 2]
    B, L = flux.shape
    device = flux.device

    # Define valid pixels exactly as your Kernel Regressor does
    is_valid = (mask == 0) & (ivar > 0)

    # Base index array: [0, 1, 2, ..., L-1] broadcast to [B, L]
    idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    # Find the nearest valid pixel to the LEFT
    # Set invalid spots to -1, then take cumulative max.
    left_valid = torch.where(is_valid, idx, torch.tensor(-1, device=device))
    left_idx, _ = torch.cummax(left_valid, dim=1)

    #  Find the nearest valid pixel to the RIGHT
    # Set invalid spots to L, then take cumulative min.
    right_valid = torch.where(is_valid, idx, torch.tensor(L, device=device))
    right_idx, _ = torch.cummin(right_valid, dim=1)

    # Handle Edge Cases (Extrapolation)
    # If a gap touches the far left (no valid left neighbor), use the right neighbor.
    left_idx = torch.where(left_idx == -1, right_idx, left_idx)
    # If a gap touches the far right (no valid right neighbor), use the left neighbor.
    right_idx = torch.where(right_idx == L, left_idx, right_idx)

    # Failsafe clamp in case an entire spectrum is masked (prevents CUDA gather errors)
    left_idx = left_idx.clamp(0, L - 1)
    right_idx = right_idx.clamp(0, L - 1)

    # Extract the flux values at the boundary indices
    left_val = torch.gather(flux, 1, left_idx)
    right_val = torch.gather(flux, 1, right_idx)

    # Calculate the fractional distance (\mu) between 0.0 and 1.0
    dist = right_idx - left_idx
    # Prevent division by zero at valid pixels (where left_idx == right_idx)
    dist = torch.where(dist == 0, torch.tensor(1, device=device), dist)
    mu = (idx - left_idx).float() / dist.float()

    # Apply Cosine Smoothing Weight: (1 - cos(mu * pi)) / 2
    mu_2 = (1.0 - torch.cos(mu * np.pi)) / 2.0
    interp_flux = left_val * (1.0 - mu_2) + right_val * mu_2

    # Merge the inpainted gaps back into the original valid flux
    inpainted_flux = torch.where(is_valid, flux, interp_flux)
    return inpainted_flux

class KernelRegressor(nn.Module):
    """
    Applies a constant 77 km/s Nadaraya-Watson kernel regression to the spectrum to interpolate masked pixels
    Combined with the CosineInterpolate function to inpaint masks with the KR estimate in regions with many valid pixels,
    inpaints with the cosine interpolation in regions with few valid pixels
    """
    def __init__(self, wave_min=3600.0, wave_max=9824.0, spec_len=7781, sigma_v=77.0):
        super().__init__()
        self.register_buffer('wave_grid', torch.linspace(wave_min, wave_max, spec_len))
        self.d_lambda = (wave_max - wave_min) / (spec_len - 1)
        self.c_kms = 299792.458
        self.sigma_v = sigma_v

    def forward(self, x):
        flux, ivar, mask = x[:, 0], x[:, 1], x[:, 2]
        B, L = flux.shape
        flux = torch.nan_to_num(flux, 0.0);
        ivar = torch.nan_to_num(ivar, 0.0)
        max_sigma_pixels = (self.sigma_v / self.c_kms) * self.wave_grid[-1] / self.d_lambda
        kernel_radius = max(int(torch.ceil(4.0 * max_sigma_pixels).item()), 3);
        kernel_size = 2 * kernel_radius + 1

        w_valid = (mask == 0).float() * (ivar > 0).float()
        weights = w_valid * (ivar + 1e-8)

        f_pad = F.pad((flux * weights).view(B, 1, L), (kernel_radius, kernel_radius), mode='reflect')
        w_pad = F.pad(weights.view(B, 1, L), (kernel_radius, kernel_radius), mode='reflect')

        m_pad = F.pad(w_valid.view(B, 1, L), (kernel_radius, kernel_radius), mode='reflect')

        f_win = F.unfold(f_pad.unsqueeze(3), kernel_size=(kernel_size, 1), padding=0).view(B, kernel_size, L)
        w_win = F.unfold(w_pad.unsqueeze(3), kernel_size=(kernel_size, 1), padding=0).view(B, kernel_size, L)
        m_win = F.unfold(m_pad.unsqueeze(3), kernel_size=(kernel_size, 1), padding=0).view(B, kernel_size, L)

        t = torch.arange(kernel_size, device=flux.device).float() - kernel_radius;
        t = t.view(1, kernel_size, 1)
        lambda_c = self.wave_grid.view(1, 1, L);
        delta_v = self.c_kms * torch.log(torch.clamp(1.0 + t * (self.d_lambda / lambda_c), min=1e-4))
        kernel_weights = torch.exp(-0.5 * (delta_v / self.sigma_v) ** 2)

        f_smooth = (f_win * kernel_weights).sum(dim=1) / ((w_win * kernel_weights).sum(dim=1) + 1e-8)

        # Visibility Score (alpha): Unweighted fraction of the Gaussian kernel that covers valid data
        visibility = (m_win * kernel_weights).sum(dim=1) / kernel_weights.sum(dim=1).clamp(min=1e-8)

        return f_smooth, visibility

class LogLambdaResampler(nn.Module):
    # Settings here need checked to keep spectrum properly/oversampled
    def __init__(self, wave_min=3600.0, wave_max=9824.0, spec_len=7781):
        super().__init__()
        log_wave = np.logspace(np.log10(wave_min), np.log10(wave_max), spec_len)
        # Create grid normalized to [-1, 1] for grid_sample
        grid_x = 2.0 * (torch.tensor(log_wave).float() - wave_min) / (wave_max - wave_min) - 1.0
        self.register_buffer('grid', torch.stack([grid_x, torch.zeros_like(grid_x)], dim=-1).view(1, 1, spec_len, 2))

    def forward(self, x, mode='bilinear'):
        """
        x: shape [B, L] or [B, C, L]
        mode: 'bilinear' (for flux/weights) or 'nearest' (for strict masks)
        """
        has_channel = (x.dim() == 3)
        if not has_channel:
            x = x.unsqueeze(1)  # [B, 1, L]

        # grid_sample expects 4D input: [N, C, H, W]
        # We dummy out the H dimension to 1
        x_4d = x.unsqueeze(2)

        resampled = F.grid_sample(
            x_4d,
            self.grid.expand(x.shape[0], -1, -1, -1),
            mode=mode,
            align_corners=True,
            padding_mode='zeros'  # Drops out-of-bounds queries to 0
        ).squeeze(2)  # Remove dummy H dimension

        return resampled if has_channel else resampled.squeeze(1)

class Preprocessor(nn.Module):
    def __init__(self, wave_min=3600.0, wave_max=9824.0, spec_len=7781, sigma_v=77.0):
        super().__init__()
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.spec_len = spec_len
        self.sigma_v = sigma_v

        self.kernel_regressor = KernelRegressor(wave_min, wave_max, spec_len, sigma_v)
        self.log_lambda_resampler = LogLambdaResampler(wave_min, wave_max, spec_len)

    def forward(self, x, augment: bool = False):
        # x: [B, 3, L] -> Channel 0: flux, Channel 1: ivar, Channel 2: mask (1=dead, 0=valid) (<-Check convention here, if i rebuild the data cache the masks will be true bitmasks)
        if augment:
            B, L = x.shape[0], x.shape[2]
            flux, ivar, mask = x[:, 0], x[:, 1], x[:, 2]

            idx_shuffle = torch.randperm(B, device=x.device)
            ivar_rand = ivar[idx_shuffle]
            mask_rand = mask[idx_shuffle]

            # noise injection
            sigma = torch.zeros_like(ivar_rand)
            valid_native = (mask == 0) & (ivar > 0) & (mask_rand == 0) & (ivar_rand > 0)
            sigma[valid_native] = torch.rsqrt(ivar_rand[valid_native])

            noise_level = torch.empty(flux.shape[0], 1, device=x.device).uniform_(0.0, 3.0)
            # Multiply by valid_native.float() so we inject noise where data is real
            noise = torch.randn_like(flux) * sigma * noise_level * valid_native.float()
            flux_noisy = flux + noise

            # Stochastic Masking (Drop up to 75%)
            B, L = flux.shape
            # keep_p between 0.25 (75% drop) and 1.0 (0% drop)
            keep_p = 0.25 + torch.rand(B, 1, device=x.device) * 0.75

            # bernoulli draws independent samples for every single pixel. 1 = keep, 0 = drop
            rmask = torch.bernoulli(keep_p.expand(B, L))
            is_dropped = (rmask == 0)

            # Apply the stochastic mask
            flux_aug = torch.where(is_dropped, torch.tensor(0.0, device=x.device), flux_noisy)
            ivar_aug = torch.where(is_dropped, torch.tensor(0.0, device=x.device), ivar_rand)
            mask_aug = torch.where(is_dropped, torch.tensor(1.0, device=x.device), mask_rand)  # 1 = invalid

            # Overwrite x
            x = torch.stack([flux_aug, ivar_aug, mask_aug], dim=1)

        # Inpainting & Resampling to log space
        flux_kr, alpha = self.kernel_regressor(x)
        flux_cos = CosineInterpolate(x)

        # Blend the hallucinations in the gaps
        flux_blend = (flux_kr * alpha) + (flux_cos * (1.0 - alpha))

        # Explicitly restore flux where valid
        is_valid = (x[:, 2] == 0) & (x[:, 1] > 0)
        flux_inpainted = torch.where(is_valid, x[:, 0], flux_blend)

        # Create the spatial weight array for the metric calculation
        weight_array = (1.0 - x[:, 2].float()) * torch.sqrt(x[:, 1] + 1e-12)

        # Resample to Log Lambda Space
        f_log = self.log_lambda_resampler(flux_inpainted, mode='bilinear')
        w_log = self.log_lambda_resampler(weight_array, mode='bilinear')

        # Stack into [Batch, Channels (3), Length]
        return torch.stack([f_log, w_log], dim=1)