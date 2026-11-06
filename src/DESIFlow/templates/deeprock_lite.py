"""
DeepRock-style template learning in the observed frame (no redshift): K learnable templates, the coefficients of every spectrum
solved in closed form by weighted least squares with gradient flowing through the solve, and a chi2 reconstruction loss.

Same design as deeprock_dr2.py (WLS projection c = (T^T W T + ridge I)^-1 T^T W f, AdamW on the templates); here the loss is a
masked-denoising chi2: spans of pixels are hidden, the coefficients are fitted on the visible pixels and the chi2 is scored on
the hidden ones, so the templates cannot learn the noise of a spectrum. Data are kept in whitened units: g = f * s and
omega = w / s^2, with s_l = sqrt(mean ivar at pixel l), so chi2 = sum omega (g - c U)^2 is identical to sum w (f - c T)^2 with
T = U / s, and the matrices are well conditioned.
"""
import numpy as np
import torch


def whiten(f, w, s=None):
    """flux f (N, L), ivar w (N, L) with w = 0 on bad pixels -> g, omega, s (L,)."""
    if s is None:
        good = (w > 0).astype(np.float64)
        s = np.sqrt((w * good).sum(0) / np.maximum(good.sum(0), 1.0))
        s = np.where(s > 0, s, np.median(s[s > 0]))
    return (f * s).astype(np.float32), (w / s ** 2).astype(np.float32), s.astype(np.float32)


def wls_project(U, g, om, ridge=1e-3):
    """c (B, K) = argmin sum om (g - c U)^2 + ridge |c|^2; U (K, L), g and om (B, L). Differentiable."""
    Uw = om[:, None, :] * U[None, :, :]                                                     # (B, K, L)
    A = torch.einsum("bkl,jl->bkj", Uw, U) + ridge * torch.eye(U.shape[0], device=U.device)[None]
    b = torch.einsum("bkl,bl->bk", Uw, g)
    return torch.linalg.solve(A, b[..., None])[..., 0]


def span_mask(B, L, n_spans=36, len_range=(5, 250), gen=None, device="cpu"):
    """(B, L) bool, True = hidden: n_spans spans per row with log-uniform lengths."""
    lo, hi = np.log(len_range[0]), np.log(len_range[1])
    ln = torch.exp(torch.rand(B, n_spans, generator=gen, device=device) * (hi - lo) + lo).long()
    st = (torch.rand(B, n_spans, generator=gen, device=device) * L).long()
    d = torch.zeros(B, L + 1, device=device)
    d.scatter_add_(1, st, torch.ones_like(st, dtype=torch.float32))
    d.scatter_add_(1, (st + ln).clamp(max=L), -torch.ones_like(st, dtype=torch.float32))
    return d.cumsum(1)[:, :L] > 0


def masked_chi2(U, g, om, hidden, ridge=1e-3):
    """per-row hidden-pixel chi2 per pixel (fit on the visible pixels), the baseline of a zero model, and the visible-pixel chi2."""
    om_vis = om * (~hidden)
    c = wls_project(U, g, om_vis, ridge)
    r = g - c @ U
    om_hid = om * hidden
    n_hid = (om_hid > 0).sum(1).clamp(min=1)
    n_vis = (om_vis > 0).sum(1).clamp(min=1)
    return (om_hid * r ** 2).sum(1) / n_hid, (om_hid * g ** 2).sum(1) / n_hid, (om_vis * r ** 2).sum(1) / n_vis


def svd_init(g, om, K, n=20000, seed=0):
    X = (g * om.sqrt())[torch.randperm(len(g), generator=torch.Generator().manual_seed(seed))[:n].to(g.device)]
    _, _, V = torch.svd_lowrank(X, q=min(K + 10, X.shape[1]), niter=3)
    return V[:, :K].T.contiguous()


@torch.no_grad()
def order_orthonormal(U, g, om, n=20000, ridge=1e-3):
    """Orthonormal basis of the span of U ordered by the energy it carries on the data.

    The reconstructions are R = C U (C the WLS coefficients). With U^T = Q P (QR), R = (C P^T) Q^T, so the SVD of the small matrix
    C P^T (n x K) gives the right singular vectors in the K-dimensional space and the basis is V^T Q^T.
    """
    idx = torch.randperm(len(g), generator=torch.Generator().manual_seed(1))[:n].to(g.device)
    C = torch.cat([wls_project(U, g[idx[a:a + 1000]], om[idx[a:a + 1000]], ridge) for a in range(0, len(idx), 1000)])
    Q, P = torch.linalg.qr(U.T)                                                             # Q (L, K), P (K, K)
    _, S, Vh = torch.linalg.svd(C @ P.T, full_matrices=False)
    return (Vh @ Q.T).contiguous(), S


def train_templates(g, om, K, steps=1500, batch=512, lr=3e-3, ridge=1e-3, seed=0, log=None, init=None, mask_kw=None):
    """g, om: whitened training tensors on the device. Returns the learnable templates U (K, L) and the loss history."""
    dev = g.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    U = torch.nn.Parameter(svd_init(g, om, K, seed=seed) if init is None else init.clone())
    opt = torch.optim.AdamW([U], lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []
    for step in range(steps):
        idx = torch.randint(0, len(g), (batch,), generator=gen, device=dev)
        hid = span_mask(batch, g.shape[1], gen=gen, device=dev, **(mask_kw or {}))
        loss = masked_chi2(U, g[idx], om[idx], hid, ridge)[0].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        hist.append(float(loss.detach()))
        if log is not None and (step % 250 == 0 or step == steps - 1):
            log(f"  K={K} step {step} hidden chi2 {np.mean(hist[-50:]):.4f}")
    return U.detach(), np.array(hist)


@torch.no_grad()
def evaluate(U, g, om, seed=123, chunk=1024, ridge=1e-3, mask_kw=None):
    """mean hidden-pixel chi2 per row, zero-model baseline per row, visible chi2 per row, with fixed masks."""
    gen = torch.Generator(device=g.device).manual_seed(seed)
    out = [[], [], []]
    for a in range(0, len(g), chunk):
        hid = span_mask(len(g[a:a + chunk]), g.shape[1], gen=gen, device=g.device, **(mask_kw or {}))
        for o, v in zip(out, masked_chi2(U, g[a:a + chunk], om[a:a + chunk], hid, ridge)):
            o.append(v.cpu().numpy())
    return tuple(np.concatenate(o) for o in out)
