"""
Reader: queries read the encoder's content latents by cross-attention. The Eidos decoder (wavelength queries -> flux)
and Plato (learned block queries -> metric coordinates) are both readers; they differ in the queries and the read-out.

Cross-attention mirrors the Perceiver's with the roles swapped:
    Perceiver: latents (steered positions) ask spectral tokens (fixed positions)
    Reader:    queries (given positions)   ask latents (each announces its own position)
    latent i, head h, read l announces  p = b[i, h] + Delta_h(z_i)   (fixed base buffer, layer-specific, an even tiling
        across the velocity range like the encoder's; Delta = zero-init linear head on the latent's content)
    the rotary half of each head's q is rotated by the query's position v, of k by p -> the score depends on (v - p) and on
    content; the other half is content-only. Values are never rotated. Periods geometric in [min_period, max_period] km/s.
Queries never attend to each other: a query's answer depends on the latents and its own position only, so any subset of
wavelengths can be queried and gives the same values as querying all of them.
positional=False (queries with no wavelength, e.g. Plato's block queries): no rotation, no base/steering; content-only.

Size settings: n_latent_sa (self-attention blocks over the latents first, no positions, cheap), n_reads, mlp_ratio (MLP
width after each read as a multiple of d_query; 0 = no MLP).

forward(queries (B, Q, d_query), latents (B, N, d_latent), query_v (Q,) or (B, Q) km/s) -> (B, Q, d_query)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from DESIFlow.eidos.perceiver import C_KMS, SelfAttention, _ffn, _rotate


def velocity(wave: torch.Tensor, wave0) -> torch.Tensor:
    """Wavelengths -> positions v = c ln(lambda / lambda_0) in km/s (float32); wave0 must be the encoder's origin
    (Preprocessor.new_wave[0], the first token's wavelength)."""
    return (C_KMS * torch.log(wave.double() / float(wave0))).float()


class ReadLayer(nn.Module):
    def __init__(self, n_latents, d_query, d_latent, n_heads, mlp_ratio=2, positional=True, rot_frac=0.5,
                 min_period=150., max_period=1e6, v_range=(0., 3e5), delta_scale=1e4):
        super().__init__()
        self.h, self.dh = n_heads, d_query // n_heads
        self.positional = positional
        self.n_rot = 2 * int(self.dh * rot_frac / 2)                                  # even number of rotated dims
        self.delta_scale = delta_scale
        self.norm_q, self.norm_kv = nn.LayerNorm(d_query), nn.LayerNorm(d_latent)
        self.q = nn.Linear(d_query, d_query, bias=False)
        self.k, self.v = nn.Linear(d_latent, d_query, bias=False), nn.Linear(d_latent, d_query, bias=False)
        self.qn, self.kn = nn.LayerNorm(self.dh), nn.LayerNorm(self.dh)
        self.out = nn.Linear(d_query, d_query, bias=False)
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.ffn = _ffn(d_query, mlp_ratio) if mlp_ratio else None
        if positional and self.n_rot:
            periods = torch.logspace(math.log10(min_period), math.log10(max_period), self.n_rot // 2)
            self.register_buffer("omega", 2 * math.pi / periods)
            base = torch.linspace(*v_range, n_latents).view(n_latents, 1).repeat(1, n_heads)
            self.register_buffer("base", base)                                        # (N, H) km/s, fixed (not learned)
            self.delta = nn.Linear(d_latent, n_heads)                                 # zero-init steering head
            nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)

    def positions(self, zn):
        """Announced position of every latent, per head (B, N, H) km/s; zn = normalized latents."""
        return self.base.unsqueeze(0) + self.delta_scale * self.delta(zn)

    def forward(self, x, z, q_v=None):
        B, Q, _ = x.shape
        N = z.shape[1]
        zn = self.norm_kv(z)
        q = self.qn(self.q(self.norm_q(x)).view(B, Q, self.h, self.dh))
        k = self.kn(self.k(zn).view(B, N, self.h, self.dh))
        v = self.v(zn).view(B, N, self.h, self.dh)
        if self.positional and self.n_rot:
            r = self.n_rot
            q = torch.cat([_rotate(q[..., :r], q_v.view(B, Q, 1).expand(B, Q, self.h), self.omega), q[..., r:]], -1)
            k = torch.cat([_rotate(k[..., :r], self.positions(zn), self.omega), k[..., r:]], -1)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + torch.tanh(self.gate) * self.out(a.transpose(1, 2).reshape(B, Q, -1))
        return x if self.ffn is None else x + self.ffn(x)


class Reader(nn.Module):
    def __init__(self, n_latents=64, d_latent=256, d_query=128, n_heads=4, n_reads=2, n_latent_sa=0, mlp_ratio=2,
                 positional=True, latent_heads=8, **rope):
        super().__init__()
        self.n_latents, self.positional = n_latents, positional
        self.latent_sa = nn.ModuleList(SelfAttention(d_latent, latent_heads) for _ in range(n_latent_sa))
        self.reads = nn.ModuleList(ReadLayer(n_latents, d_query, d_latent, n_heads, mlp_ratio, positional, **rope)
                                   for _ in range(n_reads))

    def forward(self, queries, latents, query_v=None):
        B, Q, _ = queries.shape
        if latents.shape[1] != self.n_latents:
            raise ValueError(f"reader expects {self.n_latents} latents, got {latents.shape[1]} "
                             f"(the encoder output has the scale token in slot 0: pass out[:, 1:])")
        if self.positional:
            if query_v is None:
                raise ValueError("positional reader needs query_v (km/s)")
            query_v = query_v.float()                                                 # positions stay float32
            query_v =query_v.unsqueeze(0).expand(B, -1) if query_v.dim() == 1 else query_v
            if query_v.shape != (B, Q):
                raise ValueError(f"query_v must be (Q,) or (B, Q) = {(B, Q)}, got {tuple(query_v.shape)}")
        for sa in self.latent_sa:
            latents = sa(latents)
        for read in self.reads:
            queries = read(queries, latents, query_v)
        return queries
