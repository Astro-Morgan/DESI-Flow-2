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
import torch.nn.functional as F


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


class ZClassHead(nn.Module):
    """Linear softmax CLASSIFIER over fine ln(1+z) bins on the flattened content latents (the benchmark's precise redshift head: sigma_NMAD ~ 3e-4 where the linear
    regression head reaches 5e-3 to 2e-2). Soft Gaussian labels (sigma 0.003 in ln(1+z)), prediction = probability-weighted mean of the bins within +-4 of the argmax
    (differentiable through the weights). forward -> logits (B, n_bins); predict(logits) -> ln(1+z) (B,); loss(logits, z) -> cross entropy against the soft labels."""

    def __init__(self, n_latents, d_latent, y_max=1.6, width=0.002, sigma=0.003):
        super().__init__()
        self.n_bins = int(round(y_max / width)) + 1
        self.register_buffer("centers", torch.linspace(0, y_max, self.n_bins))
        self.sigma = sigma
        self.trunk = nn.Linear(n_latents * d_latent, self.n_bins)

    def forward(self, content):
        return self.trunk(content.flatten(1))

    def loss(self, logits, z, z_mask=None):
        y = torch.log1p(z).clamp(0, float(self.centers[-1])).unsqueeze(-1)
        target = torch.softmax(-0.5 * ((self.centers - y) / self.sigma) ** 2, dim=-1)
        ce = -(target * F.log_softmax(logits, dim=-1)).sum(-1)
        if z_mask is None:
            return ce.mean()
        return (ce * z_mask).sum() / z_mask.sum() if z_mask.any() else None

    def predict(self, logits):
        p = torch.softmax(logits, dim=-1)
        k = p.argmax(-1, keepdim=True)
        idx = torch.arange(self.n_bins, device=p.device).unsqueeze(0)
        w = p * ((idx - k).abs() <= 4)
        return (w * self.centers).sum(-1) / w.sum(-1).clamp_min(1e-9)
