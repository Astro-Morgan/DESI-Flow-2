"""
Gradient surgery for the redshift head (PCGrad, Yu et al. 2020), with reconstruction as the PROTECTED task.

Two losses, three kinds of parameters:
    shared    (encoder: CNN + Perceiver)  get gradient from both losses   g_rec and g_z
    rec_only  (decoder)                   get gradient from reconstruction only
    z_only    (the z head)                get gradient from the z loss only

Only the z gradient on the shared parameters can disturb reconstruction. If it points against the reconstruction
gradient (g_z . g_rec < 0) its anti-parallel component is removed:

    g_z'  =  g_z - min(0, g_z . g_rec) / |g_rec|^2 * g_rec            (g_rec itself is never modified)

and the update uses  g_rec + g_z'  on the shared parameters,  g_rec  on the decoder,  g_z  on the head. Then
g_rec . (g_rec + g_z') >= |g_rec|^2: to first order a gradient step lowers the reconstruction loss at least as much as
a reconstruction-only step. This is exact at the parameter level (two backward passes through the encoder); the
guarantee is for the gradient direction, an adaptive optimizer (Adam) rescales it per parameter.

surgery=False gives the plain sum g_rec + g_z (same two passes, so the conflict statistics are logged either way).
"""
import torch


def _dot(a, b):
    """Dot product of two lists of tensors in float64 (None counts as zero)."""
    total = None
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        d = torch.dot(x.detach().flatten().double(), y.detach().flatten().double())
        total = d if total is None else total + d
    return total if total is not None else torch.zeros((), dtype=torch.float64)


def project_out_conflict(g_z, g_rec):
    """g_z, g_rec: lists of tensors (shared parameters). Returns (g_z', dot, |g_rec|^2, |g_z|^2) with g_z' = g_z minus its
    component anti-parallel to g_rec (unchanged when g_z . g_rec >= 0)."""
    dot, nr2, nz2 = _dot(g_z, g_rec), _dot(g_rec, g_rec), _dot(g_z, g_z)
    if dot < 0 and nr2 > 0:
        coef = (dot / nr2).item()
        g_z = [None if gz is None else (gz - coef * gr if gr is not None else gz) for gz, gr in zip(g_z, g_rec)]
    return g_z, dot, nr2, nz2


def backward_with_surgery(loss_rec, loss_z, shared, rec_only, z_only, surgery=True):
    """Sets .grad on every parameter in shared + rec_only + z_only (replacing any existing .grad) and returns statistics:
        cos (g_rec, g_z on the shared parameters, before surgery), conflict (cos < 0), norm_rec, norm_z (shared,
        before surgery), norm_z_after (shared, after surgery; = norm_z when not projected), surgery (applied or not).
    loss_z None (no trusted redshift in the batch) -> plain reconstruction backward, stats = {}."""
    n = len(shared)
    g_rec = torch.autograd.grad(loss_rec, shared + rec_only, retain_graph=loss_z is not None, allow_unused=True)
    zeros = lambda p: torch.zeros_like(p)
    if loss_z is None:
        for p, g in zip(shared + rec_only, g_rec):
            p.grad = zeros(p) if g is None else g
        return {}
    g_z = torch.autograd.grad(loss_z, shared + z_only, allow_unused=True)
    gr_sh, gz_sh = list(g_rec[:n]), list(g_z[:n])
    gz_proj, dot, nr2, nz2 = project_out_conflict(gz_sh, gr_sh) if surgery else (gz_sh, _dot(gz_sh, gr_sh), _dot(gr_sh, gr_sh), _dot(gz_sh, gz_sh))
    for p, gr, gz in zip(shared, gr_sh, gz_proj):
        gr = zeros(p) if gr is None else gr
        p.grad = gr if gz is None else gr + gz
    for p, g in zip(rec_only, g_rec[n:]):
        p.grad = zeros(p) if g is None else g
    for p, g in zip(z_only, g_z[n:]):
        p.grad = zeros(p) if g is None else g
    nr, nz = nr2.sqrt().item(), nz2.sqrt().item()
    cos = (dot / (nr2.sqrt() * nz2.sqrt())).item() if nr > 0 and nz > 0 else 0.0
    nz_after = _dot(gz_proj, gz_proj).sqrt().item()
    return {"cos": cos, "conflict": float(cos < 0), "norm_rec": nr, "norm_z": nz, "norm_z_after": nz_after,
            "surgery": float(surgery and cos < 0)}
