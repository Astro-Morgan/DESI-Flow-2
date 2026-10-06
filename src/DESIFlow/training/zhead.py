"""
Linear redshift head on the Eidos content latents, target log1p(z) (a redshift is a translation in log-lambda; this is
axis 0 of the metric latent).

    mode "flat":   y = w . flatten(latents) + b            w: n_latents * d_latent numbers
    mode "pooled": y = mean_i (w . latent_i) + b           w: d_latent numbers shared by all latents

Reads latents[:, 1:] only (the scale token is not content). Trains only on spectra whose z is trusted: z_mask (B,) bool
selects them; reconstruction uses every spectrum.
"""
import torch
import torch.nn as nn


class ZHead(nn.Module):
    def __init__(self, n_latents, d_latent, mode="flat"):
        super().__init__()
        if mode not in ("flat", "pooled"):
            raise ValueError(mode)
        self.mode = mode
        self.lin = nn.Linear(n_latents * d_latent if mode == "flat" else d_latent, 1)

    def forward(self, content):
        """content (B, n_latents, d_latent) -> predicted log1p(z) (B,)"""
        if self.mode == "flat":
            return self.lin(content.flatten(1)).squeeze(-1)
        return self.lin(content).squeeze(-1).mean(-1)

    @staticmethod
    def log_loss(pred, z, z_mask=None, eps=1000.0 / 299792.458):
        """Mean of 1/2 log(Delta^2 + eps^2) over the trusted spectra, Delta = pred - log1p(z) (= (z_pred - z)/(1+z) to first order; a velocity dv is
        Delta = dv/c). Equal reward for every halving of the error until the floor eps (default 1000 km/s, DESI's standard precision cut); the gradient
        Delta/(Delta^2 + eps^2) GROWS as the error shrinks toward eps, unlike MSE. None if there is no trusted redshift."""
        err = 0.5 * torch.log((pred - torch.log1p(z)) ** 2 + eps ** 2)
        if z_mask is None:
            return err.mean()
        return (err * z_mask).sum() / z_mask.sum() if z_mask.any() else None

    @staticmethod
    def loss(pred, z, z_mask=None):
        """MSE in log1p(z) over the trusted spectra; None if there are none in the batch."""
        err = (pred - torch.log1p(z)) ** 2
        if z_mask is None:
            return err.mean()
        return (err * z_mask).sum() / z_mask.sum() if z_mask.any() else None
