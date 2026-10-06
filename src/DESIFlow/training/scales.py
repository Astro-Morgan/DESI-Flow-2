"""
Per-starlet-scale power of a residual relative to pure noise (the E_j statistic).

For a native-grid residual [d - s*model, ivar, mask] the spectrum goes through the model's own preprocessor (so it sits on the same 23 km/s
log-lambda lattice with propagated ivar as the encoder input) and the weighted starlet (scale j ~ 23.2 * 2^j km/s). The same is done for simulated
pure-noise spectra drawn from the quoted ivar. The excess

    E_j = (mean squared starlet coefficient of the residual at scale j) / (the same for pure noise)

is 1 where the residual is noise at that scale and above 1 where the model misses structure or the quoted noise is too small. Scale 1 is almost
pure noise in real spectra, so E_1 checks the ivar calibration (DR2 ivar is understated at high S/N); E_j - E_1 at coarser scales is structured misfit.
"""
import torch


@torch.no_grad()
def scale_power(pre, path, resid, ivar, mask, noise=None):
    """Mean squared starlet coefficient per scale (B, J+1) of a native-grid spectrum [resid, ivar, mask] on the model's lattice, over good lattice pixels."""
    xr, _ = pre(torch.stack([resid if noise is None else noise, ivar, mask], 1))
    ch = path(xr)
    good = (xr[:, 2] > 0).unsqueeze(1)
    return (ch ** 2 * good).sum(-1) / good.sum(-1).clamp_min(1)


@torch.no_grad()
def scale_excess(pre, path, resid, ivar, mask, n_noise=2, gen=None):
    """Per-object excess E (B, J+1) = residual power / pure-noise power at each starlet scale."""
    num = scale_power(pre, path, resid, ivar, mask)
    den = torch.zeros_like(num)
    sig = torch.where(ivar > 0, ivar.clamp_min(1e-30).rsqrt(), torch.zeros_like(ivar))
    for _ in range(n_noise):
        den += scale_power(pre, path, resid, ivar, mask, noise=torch.randn(ivar.shape, device=ivar.device, generator=gen) * sig) / n_noise
    return num / den.clamp_min(1e-30)
