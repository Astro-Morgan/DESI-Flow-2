import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from DESIFlow.data.extraction_toy import toy


def test_whitening_normalisation_and_noise_identities():
    t = toy()
    n = len(t["x"])
    assert np.abs(t["Q"] @ t["C"] @ t["Q"].T - np.eye(n)).max() < 1e-10                    # Q whitens the noise of F_hat
    assert np.abs(t["R"].sum(1) - 1).max() < 1e-12                                          # rows of R sum to 1
    cov = t["R"] @ t["C"] @ t["R"].T
    assert np.abs(cov - np.diag(1 / t["S"] ** 2)).max() < 1e-10                             # F_tilde: independent noise, ivar = S^2
    return "Q C Q^T = I, rows of R sum to 1, cov(F_tilde) = diag(1/S^2)"


def test_f_hat_is_unbiased_and_r_width_matches_the_psf():
    t = toy()
    rng = np.random.default_rng(1)
    est = np.mean([np.linalg.solve(t["info"], t["A"].T @ ((t["A"] @ t["F_true"] + rng.normal(0, 1, len(t["p"]))) / t["N"])) for _ in range(2000)], 0)
    assert np.abs(est - t["F_true"]).max() < 0.5                                            # E[F_hat] = F_true (noise 6 per estimate, 2000 draws)
    j = len(t["x"]) // 2
    r = t["R"][j]
    w = np.sqrt((r * (t["x"] - j) ** 2).sum())
    assert abs(w - 0.75) < 0.05                                                              # the PSF sigma of the toy, 1.5 pixels = 0.75 output pixels
    return f"F_hat unbiased (mean of 2000 draws), width of a row of R {w:.2f} = PSF 0.75 output pixels"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
