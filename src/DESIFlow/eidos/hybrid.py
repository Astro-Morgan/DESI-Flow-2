"""
Hybrid decoder for the grid Eidos: a reader-style front half and a convolutional back half.

    token latents (B, N, bott)                                  N = 813 tokens, one per 16 log-lambda pixels
      -> lift Linear(bott -> d)                                  so the LayerNorms below do not eat the latent's mean and scale (the plain reader LayerNorms the raw
                                                                 latent: identically 0 at width 1, mean and scale lost at width 8)
      -> output GRID of G = factor x N query points, d channels each (a learned per-phase embedding [+ `extra`, e.g. the redshift feed])
      -> n_cross x CROSS-attention: grid points read the lifted latents, rotary position = wavelength (km/s)
      -> n_self  x SELF-attention among the grid points, rotary position = wavelength
      -> conv stack: log2(16 / factor) x UpShuffleGN (x2 up-sampling conv, GroupNorm(1) optional, SiLU) -> Conv1d -> new_len log-lambda pixels
      -> linear interpolation at the query velocities (any observed-frame wavelengths)

Pre-norm everywhere, QK-LayerNorm per head (as in the Eidos reader), plain residuals. Positions: token t sits at log-lambda pixel 16 t (Perceiver.token_velocity), grid
point g at pixel (16 / factor) g: the up-sampled sequence is cropped by half a block so that the output block of g is centred there. Key positions are fixed at the token
centres (no latent steering). conv_norm=False (default): the GroupNorm(1) over (channels x the whole spectrum) was measured to hurt the brightest-line objects.

forward(latents (B, N, bott), query_v (Q,) or (B, Q) km/s [, extra (B, G, d)]) -> (B, Q) normalized flux (flux / s)
render(latents [, extra]) -> (B, new_len) on the log-lambda grid.   Grid / token velocities and kms_per_px are buffers / attributes.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from DESIFlow.eidos.perceiver import _rotate, _ffn


class UpShuffleGN(nn.Module):
    """x2 up-sampling by a conv emitting 2 interleaved phases, then (optionally) GroupNorm(groups=1) and SiLU."""

    def __init__(self, ci, co, k=5, norm=False):
        super().__init__()
        self.conv, self.co = nn.Conv1d(ci, 2 * co, k, padding=k // 2), co
        self.gn = nn.GroupNorm(1, co) if norm else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        y = self.conv(x).view(B, self.co, 2, L).permute(0, 1, 3, 2).reshape(B, self.co, 2 * L)
        return F.silu(self.gn(y))


class RotAttn(nn.Module):
    """pre-norm attention block with wavelength rotary positions (every head dim rotated) + pre-norm FFN. cross=True: queries x attend to ctx; else x attends to itself."""

    def __init__(self, d, n_heads=4, min_period=400., max_period=1e6, cross=False, norm_kv=True, ffn_ratio=2):
        super().__init__()
        self.h, self.dh, self.cross = n_heads, d // n_heads, cross
        if d % n_heads or self.dh % 2:
            raise ValueError(f"d={d} must split into {n_heads} heads of even dimension")
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d) if (cross and norm_kv) else None
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.qn, self.kn = nn.LayerNorm(self.dh), nn.LayerNorm(self.dh)
        self.out = nn.Linear(d, d, bias=False)
        self.ffn = _ffn(d, ffn_ratio)
        self.register_buffer("omega", 2 * math.pi / torch.logspace(math.log10(min_period), math.log10(max_period), self.dh // 2))

    def forward(self, x, ctx, xv, cv):
        """x (B, L, d); ctx (B, N, d) or None (self-attention); xv (L,), cv (N,) positions in km/s"""
        B, L, d = x.shape
        xn = self.norm_q(x)
        if ctx is None:
            kv, cv = xn, xv
        else:
            kv = self.norm_kv(ctx) if self.norm_kv is not None else ctx
        N = kv.shape[1]
        q = self.qn(self.q(xn).view(B, L, self.h, self.dh))
        k = self.kn(self.k(kv).view(B, N, self.h, self.dh))
        v = self.v(kv).view(B, N, self.h, self.dh)
        q = _rotate(q, xv.view(1, L, 1).expand(B, L, self.h), self.omega)
        k = _rotate(k, cv.view(1, N, 1).expand(B, N, self.h), self.omega)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + self.out(a.transpose(1, 2).reshape(B, L, d))
        return x + self.ffn(x)


class HybridDecoder(nn.Module):
    def __init__(self, bott, n_tokens, kms_per_px, new_len=13000, factor=2, d=96, n_heads=4, n_cross=1, n_self=1, norm_kv=True, conv_norm=False, min_period=400.):
        """kms_per_px = c ln(new_wave[1] / new_wave[0]) (velocity of one log-lambda pixel, 23.1 km/s); tokens sit at pixel 16 t."""
        super().__init__()
        if factor not in (1, 2, 4):
            raise ValueError("factor must be 1, 2 or 4 (grid points per token)")
        self.n_tokens, self.factor, self.new_len, self.kms_per_px, self.d = n_tokens, factor, new_len, float(kms_per_px), d
        self.G, self.block = n_tokens * factor, 16 // factor                              # grid points, log pixels per grid point
        self.shift = self.block // 2
        if self.G * self.block < new_len + self.shift:
            raise ValueError("grid too short for new_len")
        self.lift = nn.Linear(bott, d)
        self.grid_embed = nn.Parameter(torch.randn(factor, d) * 0.02)                       # one learned vector per phase within a token
        self.cross = nn.ModuleList(RotAttn(d, n_heads, min_period, cross=True, norm_kv=norm_kv) for _ in range(n_cross))
        self.selfs = nn.ModuleList(RotAttn(d, n_heads, min_period, cross=False) for _ in range(n_self))
        widths = {2: [48, 32], 3: [64, 48, 32], 4: [80, 64, 48, 32]}[int(math.log2(self.block))]
        ups, c = [], d
        for w in widths:
            ups.append(UpShuffleGN(c, w, norm=conv_norm)); c = w
        self.ups, self.out = nn.Sequential(*ups), nn.Conv1d(c, 1, 5, padding=2)
        self.register_buffer("token_v", self.kms_per_px * 16.0 * torch.arange(n_tokens, dtype=torch.float32), persistent=False)
        self.register_buffer("grid_v", self.kms_per_px * float(self.block) * torch.arange(self.G, dtype=torch.float32), persistent=False)

    def render(self, latents, extra=None):
        """latents (B, N, bott) -> (B, new_len) normalized flux on the log-lambda grid. extra (B, G, d): additive feature on the grid queries (the z feed)."""
        B = latents.shape[0]
        h = self.lift(latents)
        x = self.grid_embed.repeat(self.n_tokens, 1).unsqueeze(0).expand(B, -1, -1)
        if extra is not None:
            x = x + extra
        for blk in self.cross:
            x = blk(x, h, self.grid_v, self.token_v)
        for blk in self.selfs:
            x = blk(x, None, self.grid_v, self.grid_v)
        y = self.out(self.ups(x.transpose(1, 2)))[:, 0]
        return y[:, self.shift: self.shift + self.new_len]

    def sample(self, y, query_v):
        """linear interpolation of the log-grid output y (B, new_len) at velocities query_v (Q,) or (B, Q) km/s measured from the first log pixel."""
        pos = (query_v.float() / self.kms_per_px).clamp(0, self.new_len - 1.001)
        i0 = pos.floor().long()
        w = pos - i0
        if i0.dim() == 1:
            i0, w = i0.unsqueeze(0).expand(y.shape[0], -1), w.unsqueeze(0).expand(y.shape[0], -1)
        return y.gather(1, i0) * (1 - w) + y.gather(1, i0 + 1) * w

    def forward(self, latents, query_v, extra=None):
        return self.sample(self.render(latents, extra).float(), query_v)
