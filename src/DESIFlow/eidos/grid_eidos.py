"""
EidosGrid: the grid Eidos (benchmark winner, see the physics-linked masked-modeling campaign): per-token latents instead of a Perceiver set.

    raw_input (B, 3, 7781) = [flux, ivar, mask]                       native DESI pixels, mask bitmask 0 = good
      Preprocessor -> x_raw (B, 3, 13000), x_smooth (B, 2, 13000)     observed-frame log-lambda grid, 23 km/s per pixel
      CNN          -> tokens (B, 813, d_token), scale s (B,)          per-scale weighted-starlet branches, inputs / s; one token per 16 pixels = 370 km/s
      proj + n_layers x TokenBlock + LayerNorm + to_lat               token self-attention (default: a learned relative-position table + a content-conditioned offset
                                                                      preference, see tokens.py; or the plain table, or rotary positions),
                                                                      then ONE bottleneck of `bott` numbers per token (813 x 8 = 6504 < 7781 flux values)
      latents (B, 1 + 813, bott)                                      slot 0 = scale token (feature 0 = log10 s, exact); slots 1.. = the per-token latents
      HybridDecoder -> flux / s at query_wave (B, Q)                  grid cross-attention -> self-attention -> conv, optional redshift feed on the grid queries

encode(raw_input)                          -> latents
decode(latents, query_wave, z=None, gate=1.0) -> normalized flux at query_wave (Angstrom, (Q,) shared or (B, Q) per sample; any observed-frame wavelengths)
forward(raw_input, query_wave, z=None, gate=1.0)

Redshift feed (zfeed=True): the DECODER (never the encoder) may receive z = (z_reg, z_cls), the two heads' ln(1+z) predictions (B,) each. It enters ADDITIVELY on the grid
queries, zero-initialised and multiplied by `gate`:  gate * (MLP(Fourier(rest-frame ln lambda of every grid point under z_cls)) + MLP([z_reg, z_cls, 20 (z_cls - z_reg)])).
Opening the gate therefore does not disturb what the decoder has learned. The z heads themselves live in DESIFlow.training.zhead (ZHead regression, ZClassHead softmax).
Parameter groups for gradient surgery: partition_params() -> (shared = encoder, rec_only = decoder + feed). Plato reads latents[:, 1:].
"""
import math
import numpy as np
import torch
import torch.nn as nn
from DESIFlow.preprocessing.preprocessing import Preprocessor
from DESIFlow.eidos.cnn import CNN
from DESIFlow.eidos.embeddings import ScaleToken
from DESIFlow.eidos.hybrid import HybridDecoder
from DESIFlow.eidos.perceiver import Perceiver, C_KMS
from DESIFlow.eidos.reader import velocity
from DESIFlow.eidos.tokens import TokenBlock


class EidosGrid(nn.Module):
    def __init__(self, cnn_kwargs=None, d=128, n_layers=6, n_heads=4, bott=8, pos="cbias", decoder_kwargs=None, zfeed=True, tok_min_period=800.):
        """cnn_kwargs overrides the CNN constructor arguments (a small model for smoke tests); d, n_layers, n_heads: token attention width, depth, heads; bott: numbers per token;
        pos: 'cbias' (default: table + content-conditioned offset preference), 'bias' (the plain learned table) or 'rope' (rotary on the token velocities); decoder_kwargs overrides HybridDecoder (factor, d, n_heads, n_cross, n_self,
        norm_kv, conv_norm)."""
        super().__init__()
        cnn_kwargs = dict(cnn_kwargs or {})
        self.preprocessor = Preprocessor()
        self.cnn = CNN(**cnn_kwargs)
        new_wave = self.preprocessor.new_wave
        self.wave0 = float(new_wave[0])
        self.register_buffer("token_v", Perceiver.token_velocity(self.cnn.token_wave(new_wave)), persistent=False)
        n_tok = self.token_v.numel()
        self.n_latents, self.d_latent, self.pos = n_tok, bott, pos
        self.proj = nn.Linear(cnn_kwargs.get("d_token", 256), d)
        self.blocks = nn.ModuleList([TokenBlock(d, n_heads, n_tok, pos=pos, token_v=self.token_v, min_period=tok_min_period) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.to_lat = nn.Linear(d, bott)
        self.scale_token = ScaleToken(bott)
        dlog = math.log(float(new_wave[1]) / float(new_wave[0]))
        dk = dict(factor=2, d=96, n_heads=4, n_cross=1, n_self=1, norm_kv=True, conv_norm=False)
        dk.update(decoder_kwargs or {})
        self.decoder = HybridDecoder(bott, n_tok, C_KMS * dlog, self.cnn.new_len, **dk)
        self.zfeed = zfeed
        if zfeed:
            dd = self.decoder.d
            self.register_buffer("per", torch.logspace(math.log10(0.002), math.log10(1.5), 32), persistent=False)
            self.register_buffer("grid_ln", (math.log(self.wave0) + dlog * self.decoder.block * torch.arange(self.decoder.G, dtype=torch.float64)).float(), persistent=False)
            self.zpos = nn.Sequential(nn.Linear(65, dd), nn.SiLU(), nn.Linear(dd, dd))
            self.cond = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, dd))
            for m in (self.zpos[2], self.cond[2]):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        self._native_m = None

    # ---------------------------------------------------------------- encoder
    def preprocess(self, raw_input):
        return self.preprocessor(raw_input)

    def encode(self, raw_input):
        """raw_input (B, 3, 7781) = [flux, ivar, mask] -> latents (B, 1 + 813, bott) with the scale token in slot 0"""
        x_raw, x_smooth = self.preprocessor(raw_input)
        tokens, scale = self.cnn(x_raw, x_smooth)
        h = self.proj(tokens)
        for blk in self.blocks:
            h = blk(h)
        lat = self.to_lat(self.norm(h))
        return torch.cat([self.scale_token(scale).to(lat.dtype), lat], dim=1)

    # ---------------------------------------------------------------- decoder
    def _fourier(self, rest):
        ang = rest.unsqueeze(-1) * (2 * math.pi / self.per)
        return torch.cat([torch.sin(ang), torch.cos(ang), torch.ones_like(rest).unsqueeze(-1)], dim=-1)

    def z_feed(self, z, gate=1.0):
        """z = (z_reg, z_cls) each (B,) ln(1+z) -> additive feature on the grid queries (B, G, d), zero at initialisation; None if there is no z or no feed."""
        if z is None or not self.zfeed:
            return None
        z_reg, z_cls = z
        rest = self.grid_ln.unsqueeze(0) - z_cls.unsqueeze(1)                              # rest-frame ln lambda of every grid point under the classifier's z
        cond = self.cond(torch.stack([z_reg, z_cls, 20.0 * (z_cls - z_reg)], dim=-1))
        return gate * (self.zpos(self._fourier(rest)) + cond.unsqueeze(1))

    def decode(self, latents, query_wave, z=None, gate=1.0):
        """normalized flux (flux / s) at query_wave (Angstrom, (Q,) or (B, Q)) from the content latents (slot 0 is not read)"""
        return self.decoder(latents[:, 1:], velocity(query_wave, self.wave0), self.z_feed(z, gate))

    def decode_native(self, latents, qi=None, z=None, gate=1.0, native_wave=None):
        """Training fast path: render the log-lambda grid once and map it to the native pixels with a dense interpolation matrix (a GEMM; the gather-based
        decode() has a scatter-add backward that is slow on GPUs). qi (B, Q) selects native pixels. native_wave: Angstrom (default linspace(3600, 9824, 7781))."""
        y = self.decoder.render(latents[:, 1:], self.z_feed(z, gate)).float() @ self._native_matrix(native_wave, latents.device)
        return y if qi is None else y.gather(1, qi)

    def _native_matrix(self, native_wave, device):
        if self._native_m is None or self._native_m.device != device:
            wave = torch.linspace(3600., 9824., 7781, dtype=torch.float64) if native_wave is None else torch.as_tensor(native_wave, dtype=torch.float64)
            x = (C_KMS * torch.log(wave / self.wave0) / self.decoder.kms_per_px).clamp(0, self.decoder.new_len - 1.001)        # float64, as the position of each native pixel on the log grid
            i0 = x.floor().long()
            M = torch.zeros(self.decoder.new_len, wave.numel())
            cols = torch.arange(wave.numel())
            M[i0, cols] = 1 - (x - i0).float()
            M[i0 + 1, cols] += (x - i0).float()
            self._native_m = M.to(device)
        return self._native_m

    def forward(self, raw_input, query_wave, z=None, gate=1.0):
        return self.decode(self.encode(raw_input), query_wave, z, gate)

    # ---------------------------------------------------------------- parameter groups
    def partition_params(self):
        """(shared, rec_only) parameter lists for gradient surgery: shared = CNN + token attention + bottleneck (get gradient from reconstruction AND the z losses),
        rec_only = decoder + redshift feed (reconstruction only)."""
        shared = [p for n, p in self.named_parameters() if n.startswith(("cnn.", "proj.", "blocks.", "norm.", "to_lat."))]
        rec_only = [p for n, p in self.named_parameters() if n.startswith(("decoder.", "zpos.", "cond."))]
        return shared, rec_only
