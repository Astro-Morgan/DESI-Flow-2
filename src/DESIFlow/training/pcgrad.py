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


def backward_with_surgery(loss_rec, loss_z, shared, rec_only, z_only, surgery=True, z_cap=False):
    """Sets .grad on every parameter in shared + rec_only + z_only (replacing any existing .grad) and returns statistics:
        cos (g_rec, g_z on the shared parameters, before surgery), conflict (cos < 0), norm_rec, norm_z (shared,
        before surgery), norm_z_after (shared, after surgery; = norm_z when not projected), surgery (applied or not),
        along = g_rec . update / |g_rec|^2 on the shared parameters (>= 1 guaranteed with surgery).
    z_cap: after the surgery, scale the z gradient on the shared parameters down so that its norm never exceeds the reconstruction gradient's (the z
        head is a guest: never louder than reconstruction; keeps along >= 1); stat z_cap = the scale applied (1 = not binding).
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
    cap = 1.0
    if z_cap:
        nza, nra = _dot(gz_proj, gz_proj).sqrt().item(), nr2.sqrt().item()
        if nza > nra > 0:
            cap = nra / nza
            gz_proj = [None if g is None else g * cap for g in gz_proj]
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
    # progress kept: g_rec . (g_rec + g_z') / |g_rec|^2  -- 1 means the encoder update moves the reconstruction loss exactly
    # as a reconstruction-only step would; >1 helps it; <1 means the z gradient leaks an adversarial component into the encoder
    along = 1.0 + (_dot(gz_proj, gr_sh) / nr2).item() if nr > 0 else 1.0
    return {"cos": cos, "conflict": float(cos < 0), "norm_rec": nr, "norm_z": nz, "norm_z_after": nz_after,
            "surgery": float(surgery and cos < 0), "along": along, "z_cap": cap}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Accumulated / scheduled variant (used by the grid Eidos pretraining): the two raw gradients are computed per MICRO-batch, summed over the micro-batches of a step, and the
# surgery is applied once to the WHOLE-batch gradients (exactly equal to a single big batch). `both` = the z heads: they receive g_rec (through the decoder's z feed) + g_z.

def route_grads(loss_rec, loss_z, shared, rec_only, both):
    """The two raw gradients PCGrad needs: g_rec over shared + rec_only + both, g_z over shared + both (None if loss_z is None). Frees the graph."""
    g_rec = list(torch.autograd.grad(loss_rec, shared + rec_only + both, retain_graph=loss_z is not None, allow_unused=True))
    g_z = list(torch.autograd.grad(loss_z, shared + both, allow_unused=True)) if loss_z is not None else None
    return g_rec, g_z


def acc_grads(a, b):
    """elementwise sum of two gradient lists (None = zero); a may be None"""
    if b is None:
        return a
    if a is None:
        return b
    return [y if x is None else x if y is None else x + y for x, y in zip(a, b)]


def route_apply(g_rec, g_z, shared, rec_only, both, surgery=True, z_cap=True, z_scale=1.0):
    """PCGrad on (possibly accumulated) batch gradients, as backward_with_surgery, plus a group `both` (the z heads, not shared with anything else) whose gradient is
    g_rec + g_z without surgery. z_cap: the z gradient on the shared parameters is capped at the reconstruction gradient's norm; z_scale: schedule multiplier on it (the heads'
    own gradient is not scaled; 0 = the z loss does not reach the encoder). Sets .grad; returns stats (cos, z_cap, along)."""
    n, m = len(shared), len(rec_only)
    zeros = lambda p: torch.zeros_like(p)
    if g_z is None:
        for p, g in zip(shared + rec_only + both, g_rec):
            p.grad = zeros(p) if g is None else g
        return {}
    gr_sh, gz_sh = list(g_rec[:n]), list(g_z[:n])
    gz_proj, dot, nr2, nz2 = project_out_conflict(gz_sh, gr_sh) if surgery else (gz_sh, _dot(gz_sh, gr_sh), _dot(gr_sh, gr_sh), _dot(gz_sh, gz_sh))
    cap = 1.0
    if z_cap:
        nza, nra = _dot(gz_proj, gz_proj).sqrt().item(), nr2.sqrt().item()
        if nza > nra > 0:
            cap = nra / nza
            gz_proj = [None if g is None else g * cap for g in gz_proj]
    if z_scale != 1.0:
        gz_proj = [None if g is None else g * z_scale for g in gz_proj]
    for p, gr, gz in zip(shared, gr_sh, gz_proj):
        gr = zeros(p) if gr is None else gr
        p.grad = gr if gz is None else gr + gz
    for p, g in zip(rec_only, g_rec[n:n + m]):
        p.grad = zeros(p) if g is None else g
    for p, gr, gz in zip(both, g_rec[n + m:], g_z[n:]):
        p.grad = (zeros(p) if gr is None else gr) + (zeros(p) if gz is None else gz)
    nr, nz = nr2.sqrt().item(), nz2.sqrt().item()
    return {"cos": (dot / (nr2.sqrt() * nz2.sqrt())).item() if nr > 0 and nz > 0 else 0.0, "z_cap": cap,
            "along": 1.0 + (_dot(gz_proj, gr_sh) / nr2).item() if nr > 0 else 1.0}
