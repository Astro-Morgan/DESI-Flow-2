import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from DESIFlow.data.resolution_study import ND, NODES, lsf_moments
from DESIFlow.data.dr2_cache import N_PIX


def gaussian_bands(sig_px, shift=0.0, n=3):
    """(n, ND, N_PIX) bands whose column j is a normalised Gaussian sampled at offsets -5..+5 (width sig_px, centre shift)."""
    off = np.arange(ND) - ND // 2
    col = np.exp(-0.5 * ((off - shift) / sig_px) ** 2)
    col /= col.sum()
    return np.broadcast_to(col[None, :, None], (n, ND, N_PIX)).copy()


def test_moments_recover_width_centroid_and_norm_of_a_gaussian_lsf():
    R = gaussian_bands(1.3, shift=0.2)
    s, mu, sig = lsf_moments(R)
    assert s.shape == (3, len(NODES)) and np.allclose(s, 1.0)
    assert np.allclose(mu, 0.2, atol=2e-3) and np.allclose(sig, 1.3, atol=2e-2)
    return f"lsf_moments: sigma {sig.mean():.3f} px (true 1.3), centroid {mu.mean():.3f} (true 0.2), column sum {s.mean():.3f}"


def test_moments_separate_a_wider_lsf_and_ignore_zero_columns():
    R = np.concatenate([gaussian_bands(1.0, n=1), gaussian_bands(1.6, n=1), np.zeros((1, ND, N_PIX))])
    s, mu, sig = lsf_moments(R)
    assert sig[0].mean() < sig[1].mean() and np.all(s[2] == 0) and np.all(np.isfinite(sig))
    return f"lsf_moments: widths {sig[0].mean():.2f} < {sig[1].mean():.2f} px; an all-zero matrix gives column sum 0 and no NaN"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
