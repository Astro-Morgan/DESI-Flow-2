"""
The target metric on the DESIFlow stack: compositional Hellinger distance over (scale, pixel, sign) of the weighted starlet of the NW-smoothed, log-resampled spectrum, in the rest frame.

    smooth_spectra   native (flux, ivar, mask) -> Preprocessor 'smooth' path (77 km/s Gaussian kernel, ivar weights) on the 13000-px log grid
    rest_cells       integer log-pixel shift to the rest frame (round(ln(1+z) / dlnlam)), weighted starlet (J = 8), drop s1, signed split into + / - masses
    hellinger_cross  D[i, j] = sqrt(sum (sqrt P_i - sqrt P_j)^2) / sqrt 2 with the compositions P renormalised over the pair's MUTUALLY good rest-frame pixels
                     (evaluated as D^2 = 1 - sum sqrt(P_i P_j) with two matrix products; `hellinger_cross_direct` is the plain definition)

Same computation as scripts/metric_analysis/real_degrade_v11.py (pipeline "A"), with one difference: the rest frame is the log grid widened by the redshift spread of the set (every object keeps
all of its pixels; pixels an object does not reach are simply not good) instead of the common window all objects cover. Cells are comparable only inside one call of `rest_cells`, because the
frame is set by the shifts of the objects handed to it: build the cells of everything that will be compared in one go.
"""
import numpy as np
import torch
from DESIFlow.preprocessing.preprocessing import Preprocessor
from DESIFlow.eidos.starlet import WeightedStarlet

SQ2 = float(np.sqrt(2.0))
J_SCALES = 8


def rest_shifts(z, dlnl):
    """Integer log-pixel shift that brings redshift z to the rest frame."""
    return np.round(np.log1p(np.asarray(z, np.float64)) / dlnl).astype(np.int64)


@torch.no_grad()
def smooth_spectra(pre, flux, ivar, mask, device, chunk=64):
    """Preprocessor smooth path on native (N, 7781) arrays -> (flux_s, ivar_s), float64 tensors (N, 13000) on `device`; ivar_s > 0 marks the good output pixels."""
    out_f, out_w = [], []

    def piece(v, a):                                                                      # numpy / memmap rows or torch rows
        x = v[a:a + chunk]
        return x.to(torch.float32) if isinstance(x, torch.Tensor) else torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)

    for a in range(0, len(flux), chunk):
        x = torch.stack([piece(v, a).to(device) for v in (flux, ivar, mask)], 1)
        _, xs = pre(x)
        out_f.append(xs[:, 0].double()); out_w.append(xs[:, 1].double())
    return torch.cat(out_f), torch.cat(out_w)


@torch.no_grad()
def rest_cells(flux_s, ivar_s, shift, ws, dtype=torch.float64):
    """Rest-frame starlet cells. flux_s, ivar_s: (N, L) smoothed log-grid spectra; shift: (N,) integer shifts; ws: WeightedStarlet (J = 8, double).
    Returns cells (N, 8, 2, LR) = [scales 2..8 + coarse] x [positive, negative mass] and good (N, LR) bool, LR = L + max(shift) - min(shift)."""
    N, L = flux_s.shape
    shift = np.asarray(shift, np.int64)
    smax = int(shift.max())
    LR = L + smax - int(shift.min())
    f, w = flux_s.new_zeros(N, LR), flux_s.new_zeros(N, LR)
    for i in range(N):
        o = smax - int(shift[i])                                                          # observed pixel p sits at rest pixel p - shift_i + smax
        f[i, o:o + L], w[i, o:o + L] = flux_s[i], ivar_s[i]
    good = w > 0
    ch, _ = ws(f * good, w * good, good)                                                  # (N, J + 1, LR)
    ch = (ch * good.unsqueeze(1))[:, 1:]                                                  # drop s1
    return torch.stack([ch.clamp_min(0), (-ch).clamp_min(0)], 2).to(dtype), good


@torch.no_grad()
def hellinger_cross(cA, gA, cB, gB, min_overlap=1000):
    """D[i, j] = Hellinger distance between A_i and B_j over their mutually good rest-frame pixels (composition renormalised per pair); +inf where fewer than `min_overlap` pixels are shared.
    Exact algebra: with P the composition over the mutual pixels, D^2 = 1 - sum sqrt(P_i P_j) = 1 - sum sqrt(c_i c_j) / sqrt(S_ij S_ji), where S_ij = sum of A_i's cell mass over the mutual pixels.
    The cells vanish outside the good pixels, so the numerator is one matrix product of the square-rooted cells and the two S matrices are products of the pixel masses with the good masks."""
    dt = cA.dtype
    sA = cA.sqrt().reshape(len(cA), -1)
    sB = sA if cB is cA else cB.sqrt().reshape(len(cB), -1)
    mA, mB = cA.sum(dim=(1, 2)), cB.sum(dim=(1, 2))                                         # pixel masses (N, LR)
    fA, fB = gA.to(dt), gB.to(dt)
    S_AB = mA @ fB.T                                                                      # mass of A_i over the pixels A_i and B_j share
    S_BA = (mB @ fA.T).T                                                                  # mass of B_j over the same pixels
    bc = (sA @ sB.T) / (S_AB * S_BA).clamp_min(1e-300).sqrt()
    D = (1.0 - bc).clamp_min(0).sqrt()
    return torch.where((fA @ fB.T) >= min_overlap, D, torch.full_like(D, float("inf")))


@torch.no_grad()
def hellinger_paired(cA, gA, cB, gB, chunk=64, min_overlap=1000):
    """D[i] = Hellinger distance between A_i and B_i (matched rows), same conventions as `hellinger_cross`."""
    out = []
    for a in range(0, len(cA), chunk):
        ca, cb = cA[a:a + chunk], cB[a:a + chunk]
        fa, fb = gA[a:a + chunk].to(cA.dtype), gB[a:a + chunk].to(cA.dtype)
        num = (ca.sqrt() * cb.sqrt()).sum(dim=(1, 2, 3))
        S_A, S_B = (ca.sum(dim=(1, 2)) * fb).sum(-1), (cb.sum(dim=(1, 2)) * fa).sum(-1)
        d = (1.0 - num / (S_A * S_B).clamp_min(1e-300).sqrt()).clamp_min(0).sqrt()
        out.append(torch.where((fa * fb).sum(-1) >= min_overlap, d, torch.full_like(d, float("inf"))))
    return torch.cat(out)


@torch.no_grad()
def hellinger_cross_direct(cA, gA, cB, gB, b_chunk=128, min_overlap=1000):
    """Reference implementation of `hellinger_cross` straight from the definition (explicit per-pair renormalisation; slow, used by the tests)."""
    nA, nB = len(cA), len(cB)
    D = torch.empty(nA, nB, dtype=cA.dtype, device=cA.device)
    for i in range(nA):
        for b in range(0, nB, b_chunk):
            g = (gA[i].unsqueeze(0) & gB[b:b + b_chunk]).to(cA.dtype)
            ci, cj = cA[i].unsqueeze(0) * g[:, None, None, :], cB[b:b + b_chunk] * g[:, None, None, :]
            Pi = ci / ci.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-300)
            Pj = cj / cj.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-300)
            d = ((Pi.sqrt() - Pj.sqrt()) ** 2).sum(dim=(1, 2, 3)).sqrt() / SQ2
            D[i, b:b + b_chunk] = torch.where(g.sum(-1) >= min_overlap, d, torch.full_like(d, float("inf")))
    return D


class RestMetric:
    """The metric pipeline with its modules on one device: cells(...) from native spectra, cross(...) between two sets of cells."""

    def __init__(self, device="cpu", J=J_SCALES, sigma_v=77.0):
        self.device = torch.device(device)
        self.pre = Preprocessor(sigma_v=sigma_v).to(self.device)
        self.dlnl = float(torch.log(self.pre.new_wave[1] / self.pre.new_wave[0]))
        self.ws = WeightedStarlet(J).double().to(self.device)

    def cells(self, flux, ivar, mask, z):
        fs, ws = smooth_spectra(self.pre, flux, ivar, mask, self.device)
        return rest_cells(fs, ws, rest_shifts(z, self.dlnl), self.ws)

    @staticmethod
    def cross(A, B, **kw):
        return hellinger_cross(A[0], A[1], B[0], B[1], **kw)

    @staticmethod
    def paired(A, B, **kw):
        return hellinger_paired(A[0], A[1], B[0], B[1], **kw)
