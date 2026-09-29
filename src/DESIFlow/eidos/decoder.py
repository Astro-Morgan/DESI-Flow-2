"""
Eidos decoder: the function (content latents, wavelength) -> flux, built from a Reader.

    query at wavelength lambda = one learned constant vector (same for every query) + its position
        v = c ln(lambda / lambda_0) km/s (reader.velocity, lambda_0 = Preprocessor.new_wave[0]) -- no data enters
        the decoder except through the latents, so nothing can be copied around the bottleneck
    the queries read the latents (reader.Reader), then a LayerNorm + linear head gives d_out numbers per query (flux)

Output units are the CNN's normalized flux, flux / s, at the queried wavelengths. Absolute flux = s * output (exact, no
parameters), so the decoder never sees the scale token: pass the encoder output WITHOUT slot 0.

rot_frac defaults to 1.0 here (every head dim rotated: 16 frequencies at d_query/n_heads = 32): the queries have no content
of their own, so a content-only half has nothing to match. Fitting free latents to real spectra it renders ~2x better
than 0.5 at equal steps (tests/test_eidos_decoder.py).

Queries can sit anywhere (a set, not a grid): the native pixel wavelengths, only the hidden pixels being scored, a
different subset per sample ((B, Q) positions), or finer than native at inference.

forward(latents (B, N, d_latent), query_v (Q,) or (B, Q) km/s) -> (B, Q)   (or (B, Q, d_out) if d_out > 1)
"""
import torch
import torch.nn as nn
from DESIFlow.eidos.reader import Reader


class Decoder(nn.Module):
    def __init__(self, n_latents=64, d_latent=256, d_query=128, n_heads=4, n_reads=2, n_latent_sa=0, mlp_ratio=2,
                 d_out=1, rot_frac=1.0, **rope):
        super().__init__()
        self.reader = Reader(n_latents, d_latent, d_query, n_heads, n_reads, n_latent_sa, mlp_ratio, positional=True,
                             rot_frac=rot_frac, **rope)
        self.query_token = nn.Parameter(torch.randn(d_query) * 0.02)
        self.head = nn.Sequential(nn.LayerNorm(d_query), nn.Linear(d_query, d_out))
        self.d_out = d_out

    def forward(self, latents, query_v):
        B = latents.shape[0]
        Q = query_v.shape[-1]
        queries = self.query_token.expand(B, Q, -1)
        y = self.head(self.reader(queries, latents, query_v))
        return y.squeeze(-1) if self.d_out == 1 else y
