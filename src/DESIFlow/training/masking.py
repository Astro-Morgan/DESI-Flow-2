"""
Hiding for masked denoising of DESI spectra, applied to the NATIVE pixels before the preprocessor.

A hidden pixel looks exactly like a pixel DESI itself masked: mask != 0, flux = ivar = 0, so the preprocessor gives it
zero weight, the starlet paths see a gap (support channel), and no scale (s) or token carries its value. Hiding after
the CNN would not hide anything (the starlet / CNN footprints reach up to 2045 log px, far beyond one token).

SpanMasker draws, per spectrum:
    a hidden fraction f ~ U(frac) of the natively good pixels, reached by placing spans whose lengths are log-uniform
    in len_range native px (the last span is truncated to the remaining budget, so small f means short spans);
    with probability camera_prob a whole camera is hidden first (b: before the r camera starts, r: between the end of
    b and the start of z, z: after the end of r -- only the part covered by that camera alone, as for a missing camera).
Only natively good pixels can be hidden (and scored); natively bad pixels stay as they are.
"""
import numpy as np
import torch

WAVE_MIN, WAVE_MAX, N_PIX = 3600., 9824., 7781
_PX = (WAVE_MAX - WAVE_MIN) / (N_PIX - 1)                                          # 0.8 A, shared by all cameras
_idx = lambda wave: int(round((wave - WAVE_MIN) / _PX))
# native DESI camera coverage: b 3600-5800, r 5760-7620, z 7520-9824 A
CAMERA_RANGES = {"b": (0, _idx(5760)), "r": (_idx(5800), _idx(7520)), "z": (_idx(7620), N_PIX)}


def run_lengths(mask):
    """mask (B, N) bool -> (B, N) int: for every True pixel the length of the contiguous True run it sits in (0 elsewhere)."""
    out = np.zeros(mask.shape, np.int64)
    for b in range(mask.shape[0]):
        m = np.concatenate([[False], mask[b], [False]])
        edges = np.flatnonzero(m[1:] != m[:-1])
        for s, e in zip(edges[0::2], edges[1::2]):
            out[b, s:e] = e - s
    return out


class SpanMasker:
    def __init__(self, frac=(0.05, 0.40), len_range=(3, 1500), camera_prob=0.05):
        self.frac, self.len_range, self.camera_prob = frac, len_range, camera_prob

    def sample(self, good, rng):
        """good (B, N) bool numpy -> (hidden (B, N) bool subset of good, span_union (B, N) bool before intersecting with good)."""
        B, N = good.shape
        lo, hi = self.len_range
        span = np.zeros((B, N), bool)
        for b in range(B):
            target = rng.uniform(*self.frac) * good[b].sum()
            if rng.random() < self.camera_prob:
                a, e = CAMERA_RANGES[("b", "r", "z")[rng.integers(3)]]
                span[b, a:e] = True
            for _ in range(256):
                have = (span[b] & good[b]).sum()
                if have >= target:
                    break
                length = int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
                length = max(lo, min(length, int(target - have) + 1))
                start = int(rng.integers(0, N - length + 1))
                span[b, start:start + length] = True
        return span & good, span


def apply_hidden(x, hidden):
    """x (B, 3, N) = [flux, ivar, mask] (tensor), hidden (B, N) bool tensor -> new input with the hidden pixels removed."""
    flux, ivar, mask = x[:, 0], x[:, 1], x[:, 2]
    zero = torch.zeros_like(flux)
    return torch.stack([torch.where(hidden, zero, flux), torch.where(hidden, zero, ivar),
                        torch.where(hidden, torch.ones_like(mask), mask)], 1)
