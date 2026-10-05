"""
Perceiver encoder for Eidos: learned latents repeatedly cross-attend to the CNN's spectral tokens.

Cross-attention with query-chosen positions (per layer, per head):
    token k has position v_k = c ln(lambda_k / lambda_0)  (km/s)
    latent i, head h, layer l picks  p = b[i, h] + Delta_h(z_i)
        b: base position (layer-specific), a fixed buffer: an even tiling of the token range (the steering below does the moving)
        Delta: linear head on the latent's current state (zero-init), so positions can be steered by content
    the rotary half of each head's q is rotated by p, of k by v_k  ->  scores depend on (v_k - p) + content;
    the other half is unrotated (content-only matching). Values are never rotated.
    Rotary periods geometric in [min_period, max_period] km/s (default 500 .. 1e6: above the ~370 km/s token
    spacing up to beyond the ~300,000 km/s spectral span).
Latent self-attention has no positions (latents are a set). Pre-norm, QK-norm, gated residuals, SiLU FFNs.
Schedule (legacy): CA, SA, CA, SA, SA, CA, SA x3, ...

forward(tokens (B, L, d_ctx), token_v (L,) km/s, scale (B,)) -> (B, 1 + n_latents, d_latent)
    position 0 = ScaleToken(scale) (deterministic, appended after the stack: the content latents never see it)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from DESIFlow.eidos.embeddings import ScaleToken

C_KMS = 299792.458


def _rotate(x, pos, omega):
    """x (..., 2m) rotated pairwise (first half, second half) by angles pos[..., None] * omega (m,)."""
    ang = pos.unsqueeze(-1) * omega
    c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def _ffn(d, ratio=4):
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ratio * d), nn.SiLU(), nn.Linear(ratio * d, d))


class CrossAttention(nn.Module):
    def __init__(self, n_latents, d_lat, d_ctx, n_heads, rot_frac=0.5, min_period=500., max_period=1e6,
                 v_range=(0., 3e5), delta_scale=1e4):
        super().__init__()
        self.h, self.dh = n_heads, d_lat // n_heads
        self.n_rot = 2 * int(self.dh * rot_frac / 2)                                  # even number of rotated dims
        self.delta_scale = delta_scale
        self.norm_q, self.norm_kv = nn.LayerNorm(d_lat), nn.LayerNorm(d_ctx)
        self.q, self.k, self.v = nn.Linear(d_lat, d_lat, bias=False), nn.Linear(d_ctx, d_lat, bias=False), nn.Linear(d_ctx, d_lat, bias=False)
        self.qn, self.kn = nn.LayerNorm(self.dh), nn.LayerNorm(self.dh)
        self.out = nn.Linear(d_lat, d_lat, bias=False)
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.ffn = _ffn(d_lat)
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), max(self.n_rot // 2, 1))
        self.register_buffer("omega", 2 * math.pi / periods)
        base = torch.linspace(*v_range, n_latents).view(n_latents, 1).repeat(1, n_heads)
        self.register_buffer("base", base)                                            # (N, H) km/s, fixed (not learned)
        self.delta = nn.Linear(d_lat, n_heads)                                        # zero-init steering head
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)

    def positions(self, z):
        return self.base.unsqueeze(0) + self.delta_scale * self.delta(self.norm_q(z))  # (B, N, H)

    def forward(self, z, ctx, ctx_v):
        B, N, _ = z.shape
        L = ctx.shape[1]
        zn, cn = self.norm_q(z), self.norm_kv(ctx)
        q = self.qn(self.q(zn).view(B, N, self.h, self.dh))
        k = self.kn(self.k(cn).view(B, L, self.h, self.dh))
        v = self.v(cn).view(B, L, self.h, self.dh)
        if self.n_rot:
            r = self.n_rot
            q = torch.cat([_rotate(q[..., :r], self.positions(z), self.omega), q[..., r:]], -1)
            k = torch.cat([_rotate(k[..., :r], ctx_v.view(1, L, 1).expand(B, L, self.h), self.omega), k[..., r:]], -1)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        z = z + torch.tanh(self.gate) * self.out(a.transpose(1, 2).reshape(B, N, -1))
        return z + self.ffn(z)


class SelfAttention(nn.Module):
    def __init__(self, d_lat, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d_lat // n_heads
        self.norm = nn.LayerNorm(d_lat)
        self.qkv = nn.Linear(d_lat, 3 * d_lat, bias=False)
        self.qn, self.kn = nn.LayerNorm(self.dh), nn.LayerNorm(self.dh)
        self.out = nn.Linear(d_lat, d_lat, bias=False)
        self.ffn = _ffn(d_lat)

    def forward(self, z):
        B, N, D = z.shape
        q, k, v = self.qkv(self.norm(z)).view(B, N, 3, self.h, self.dh).unbind(2)
        a = F.scaled_dot_product_attention(self.qn(q).transpose(1, 2), self.kn(k).transpose(1, 2), v.transpose(1, 2))
        z = z + self.out(a.transpose(1, 2).reshape(B, N, D))
        return z + self.ffn(z)


class Perceiver(nn.Module):
    def __init__(self, n_latents=64, d_latent=256, d_ctx=256, n_heads=8, n_cross=4, rot_frac=0.5,
                 min_period=500., max_period=1e6, v_range=(0., 3e5)):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d_latent) * 0.02)
        self.layers = nn.ModuleList()
        for i in range(n_cross):
            self.layers.append(CrossAttention(n_latents, d_latent, d_ctx, n_heads, rot_frac, min_period, max_period, v_range))
            self.layers.extend(SelfAttention(d_latent, n_heads) for _ in range(i + 1))
        self.scale_token = ScaleToken(d_latent)

    @staticmethod
    def token_velocity(token_wave: torch.Tensor) -> torch.Tensor:
        """Token wavelengths (L,) -> positions v = c ln(lambda / lambda_0) in km/s (float32)."""
        return (C_KMS * torch.log(token_wave / token_wave[0])).float()

    def forward(self, tokens, token_v, scale):
        z = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for layer in self.layers:
            z = layer(z, tokens, token_v) if isinstance(layer, CrossAttention) else layer(z)
        return torch.cat([self.scale_token(scale).to(z.dtype), z], dim=1)
