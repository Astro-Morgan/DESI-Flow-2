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


def weighted_mse_loss(pred, scale, flux, ivar, valid, hidden=None, hidden_weight=1.0, w_cap=100.0):
    """ivar-weighted MSE, per object, in normalized flux units (the default pretraining reconstruction loss).

        r_i = flux_i / s - pred_i                                    residual in the encoder's normalized units (pred = flux / s)
        w_i = ivar_i * (hidden_weight if pixel i is hidden else 1)   on the natively good, scored pixels (valid); 0 elsewhere
        L_b = sum_i w_i r_i^2 / sum_i w_i                            weighted mean squared residual of object b
        loss = mean over the objects that have any weight of L_b

    Differs from chi2 (sum ivar r^2 / N), which grows as the object's S/N^2: here each object's loss is divided by its own total weight, so bright and faint
    objects contribute comparably (the pooled chi2 was dominated by a few bright objects and had heavy-tailed gradients). Pixel weights are capped at w_cap times
    the object's median weight so that a pathological ivar (DESI occasionally has 1e12) cannot decide its object's loss.

    pred, flux, ivar (B, Q) are the queried pixels (ORIGINAL data); valid (B, Q) bool = natively good pixels that are scored (hidden AND visible ones when the
    loss is on all pixels); hidden (B, Q) bool marks the artificially hidden ones (None = no pixel is up-weighted). Natively bad pixels have ivar 0 and valid False.
    """
    w = ivar.clamp_min(0) * valid.to(ivar.dtype)
    if w_cap:
        med = torch.where(valid, w, torch.full_like(w, float("nan"))).nanmedian(dim=1, keepdim=True).values      # NaN for an object without valid pixels (its w is 0 anyway)
        w = torch.minimum(w, (w_cap * med).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0))
    if hidden is not None and hidden_weight != 1.0:
        w = w * torch.where(hidden, torch.full_like(w, hidden_weight), torch.ones_like(w))
    r = flux / scale.unsqueeze(-1) - pred
    tot = w.sum(1)
    per_obj = (w * r * r).sum(1) / tot.clamp_min(1e-30)
    has = tot > 0
    return (per_obj * has).sum() / has.sum().clamp_min(1)
