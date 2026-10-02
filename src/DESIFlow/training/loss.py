"""
Reconstruction loss for masked denoising, scored on the native pixels the decoder is queried at.

    r_i = (flux_i - s * pred_i) * sqrt(ivar_i)         normalized residual in sigma units; pred = decoder output = flux / s
                                                        (identical to the residual in normalized units flux/s, ivar*s^2)
    rho(r) = r^2                    (huber_c None)   or   Huber: r^2 for |r| <= c, c (2|r| - c) beyond (undetected cosmics,
                                                        strong lines early in training)
    loss = sum_{hidden good px} rho(r) / (number of hidden good px)       pooled over the batch

Only hidden pixels are scored: the latent has more numbers than the spectrum has pixels, so scoring visible pixels would
reward copying their noise. The plain chi2 (mean r^2) over hidden / visible pixels is returned for monitoring: pure
noise gives 1; below 1 on the TRAINING set means the model has memorized noise.
"""
import torch


def normalized_residual(pred, scale, flux, ivar):
    return (flux - scale.unsqueeze(-1) * pred) * ivar.clamp_min(0).sqrt()


def huber(r, c):
    a = r.abs()
    return torch.where(a <= c, r * r, c * (2 * a - c))


def hidden_loss(pred, scale, flux, ivar, hidden, huber_c=None):
    """pred (B, Q) = flux/s at all Q native pixels; scale (B,); flux, ivar (B, Q) the ORIGINAL data; hidden (B, Q) bool,
    subset of the natively good pixels. Returns the scalar training loss."""
    r = normalized_residual(pred, scale, flux, ivar)
    rho = r * r if huber_c is None else huber(r, huber_c)
    w = hidden.to(rho.dtype)
    return (rho * w).sum() / w.sum().clamp_min(1.0)


@torch.no_grad()
def chi2_stats(pred, scale, flux, ivar, hidden, good):
    """Mean chi2 (r^2) over the hidden pixels and over the visible good pixels, and the pixel counts."""
    r2 = normalized_residual(pred, scale, flux, ivar) ** 2
    vis = good & ~hidden
    nh, nv = hidden.sum().clamp_min(1), vis.sum().clamp_min(1)
    return {"chi2_hidden": (r2 * hidden).sum() / nh, "chi2_visible": (r2 * vis).sum() / nv,
            "n_hidden": hidden.sum(), "n_visible": vis.sum()}
