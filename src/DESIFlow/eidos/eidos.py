"""
Eidos: masked-denoising autoencoder for DESI spectra. Raw spectrum -> latents -> flux at any wavelengths.
Plato (the metric head) reads the same latents.

    raw_input (B, 3, 7781) = [flux, ivar, mask]                       native DESI pixels, mask bitmask 0 = good
      Preprocessor -> x_raw (B, 3, 13000), x_smooth (B, 2, 13000)     observed-frame log-lambda grid, 23 km/s per pixel
      CNN          -> tokens (B, 813, 256), scale s (B,)              per-scale weighted-starlet branches, inputs / s
      Perceiver    -> latents (B, 65, 256)                            slot 0 = scale token (feature 0 = log10 s, exact),
                                                                      slots 1..64 = content latents (magnitude-invariant)
      Decoder      -> flux / s at query_wave (B, Q)                   reads slots 1..64 only

encode(raw_input)             -> latents
decode(latents, query_wave)   -> normalized flux at query_wave: Angstrom, (Q,) shared or (B, Q) per sample, any
                                 observed-frame wavelengths (native pixels, only the hidden pixels, finer than native).
                                 Queries are independent, so any subset gives the same values as querying all of them.
forward(raw_input, query_wave) = decode(encode(raw_input), query_wave)

Units: the output is flux / s in the input flux units. Absolute flux = s * output with s = 10 ** latents[:, 0, 0].

Masking and the loss are not part of this class: hiding pixels means setting mask bits in raw_input before the call,
scoring them means choosing query_wave. Plato reads latents[:, 1:], the same slice the decoder reads.
"""

import torch
import torch.nn as nn
from DESIFlow.preprocessing.preprocessing import Preprocessor
from DESIFlow.eidos.cnn import CNN
from DESIFlow.eidos.perceiver import Perceiver
from DESIFlow.eidos.decoder import Decoder
from DESIFlow.eidos.reader import velocity


class Eidos(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocessor = Preprocessor()
        self.cnn = CNN()
        self.perceiver = Perceiver()
        self.decoder = Decoder()
        # positions in km/s are measured from the first log-lambda pixel, for the encoder tokens and the decoder queries
        self.wave0 = float(self.preprocessor.new_wave[0])
        token_v = Perceiver.token_velocity(self.cnn.token_wave(self.preprocessor.new_wave))
        self.register_buffer("token_v", token_v, persistent=False)

    def preprocess(self, raw_input):
        # Utility function to generate model input without embedding
        # Expects raw_input to be [flux, ivar, mask]
        return self.preprocessor(raw_input)

    def encode(self, raw_input):
        # Expects raw_input to be [flux, ivar, mask]; returns (B, 65, 256) with the scale token in slot 0
        x_raw, x_smooth = self.preprocessor(raw_input)
        tokens, scale = self.cnn(x_raw, x_smooth)
        return self.perceiver(tokens, self.token_v, scale)

    def decode(self, latents, query_wave):
        # Reconstruct flux / s at query_wave (Angstrom, (Q,) or (B, Q)) from the content latents (slot 0 is not read)
        return self.decoder(latents[:, 1:], velocity(query_wave, self.wave0))

    def forward(self, raw_input, query_wave):
        # Full pass from raw spectrum to reconstruction, used for denoising
        return self.decode(self.encode(raw_input), query_wave)
