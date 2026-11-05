"""
A one-dimensional toy of spectro-perfectionism extraction (Bolton & Schlegel 2010), to see F_hat, C, Q, S and R.

n output wavelengths F_j (the unknown spectrum), m = 2n detector pixels p_k. Every wavelength spreads over the pixels as a Gaussian
of rms width `psf_sigma` pixels, A[k, j] = the fraction of unit flux at wavelength j that lands in pixel k, so p = A F + noise.

    F_hat = (A^T N^-1 A)^-1 A^T N^-1 p      least squares, deconvolved, noise correlated with covariance C = (A^T N^-1 A)^-1
    Q       any matrix with Q^T Q = C^-1     (here the symmetric square root C^-1/2): Q F_hat has uncorrelated, unit-variance noise
    S_i   = sum_j Q_ij                       the response of independent measurement i to a flat spectrum of height 1
    R     = diag(1/S) Q                      rows sum to 1: output pixel i measures a weighted average of the true spectrum
    F_til = R F_hat                          has E[F_til] = R F_true and ivar_i = S_i^2

Run:  python -m DESIFlow.data.extraction_toy --fig FIG_DIR --figcache CACHE_DIR
"""
import argparse
import os

import numpy as np


def toy(n=60, psf_sigma=1.5, noise=1.0, seed=0, line_at=30.0):
    m = 2 * n
    pix = (np.arange(m) + 0.5) / 2.0                                                       # pixel centres in output-grid units
    A = np.exp(-0.5 * ((pix[:, None] - np.arange(n)[None, :] - 0.5) / (psf_sigma / 2.0)) ** 2)
    A /= A.sum(0, keepdims=True)                                                           # each wavelength's light is conserved
    N = np.full(m, noise ** 2)
    x = np.arange(n)
    F_true = 5.0 + 20.0 * np.exp(-0.5 * ((x - line_at) / 0.7) ** 2) + 8.0 * np.exp(-0.5 * ((x - 12.0) / 0.7) ** 2)
    rng = np.random.default_rng(seed)
    p = A @ F_true + rng.normal(0.0, noise, m)
    info = A.T @ (A / N[:, None])                                                          # A^T N^-1 A, the information matrix
    C = np.linalg.inv(info)
    F_hat = C @ (A.T @ (p / N))
    w, V = np.linalg.eigh(info)
    Q = (V * np.sqrt(w)) @ V.T                                                             # symmetric square root, Q^T Q = C^-1
    S = Q.sum(1)
    R = Q / S[:, None]
    F_til = R @ F_hat
    return dict(A=A, N=N, F_true=F_true, p=p, pix=pix, info=info, C=C, F_hat=F_hat, Q=Q, S=S, R=R, F_til=F_til, x=x)


def main(figdir, cachedir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = toy()
    n = len(t["x"])
    Cr = t["R"] @ t["C"] @ t["R"].T                                                        # covariance of F_til
    print("F_hat noise (rms over pixels): %.2f   F_til noise: %.2f   (data pixel noise 1.00)" % (np.sqrt(np.diag(t["C"])).mean(), 1 / t["S"].mean()))
    print("max |Q C Q^T - I| = %.1e ; max |cov(F_til) - diag(1/S^2)| = %.1e ; max |row sum of R - 1| = %.1e"
          % (np.abs(t["Q"] @ t["C"] @ t["Q"].T - np.eye(n)).max(), np.abs(Cr - np.diag(1 / t["S"] ** 2)).max(), np.abs(t["R"].sum(1) - 1).max()))
    j = n // 2
    r = t["R"][j]
    sig_R = np.sqrt((r * (t["x"] - j) ** 2).sum() - ((r * (t["x"] - j)).sum()) ** 2)
    print("rms width of a row of R: %.2f output pixels (the PSF: 0.75)" % sig_R)
    print("neighbour correlation of F_hat noise: %.2f (lag 1), %.2f (lag 2)" % (t["C"][j, j + 1] / t["C"][j, j], t["C"][j, j + 2] / t["C"][j, j]))
    fig, ax = plt.subplots(3, 2, figsize=(12, 11))
    ax[0, 0].plot(t["pix"], t["p"], ".", ms=4, color="gray", label="detector pixels p (noisy, blurred)")
    ax[0, 0].plot(t["x"], t["F_true"], color="k", lw=1, label="true spectrum F")
    ax[0, 0].set_xlabel("position along the dispersion axis [output-grid pixels]"); ax[0, 0].set_ylabel("counts"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("1. what the detector records")
    ax[0, 1].plot(t["x"], t["F_true"], color="k", lw=1, label="true F")
    ax[0, 1].errorbar(t["x"], t["F_hat"], np.sqrt(np.diag(t["C"])), fmt=".", color="tab:red", ms=4, label="F_hat +- sqrt(C_jj)")
    ax[0, 1].set_xlabel("output-grid pixel j"); ax[0, 1].set_ylabel("counts"); ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("2. least squares F_hat: unbiased but noisy and ringing")
    ax[1, 0].plot(t["x"], t["R"] @ t["F_true"], color="k", lw=1, label="R F_true (what the extraction can see)")
    ax[1, 0].errorbar(t["x"], t["F_til"], 1 / t["S"], fmt=".", color="tab:blue", ms=4, label="F_tilde +- 1/S")
    ax[1, 0].set_xlabel("output-grid pixel i"); ax[1, 0].set_ylabel("counts"); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("3. F_tilde = R F_hat: independent errors, smooth R F_true")
    for jj in (10, 20, 30, 45):
        ax[1, 1].plot(t["x"], t["R"][jj], label=f"row i = {jj}")
    ax[1, 1].set_xlabel("true wavelength pixel j"); ax[1, 1].set_ylabel("R_ij (weights sum to 1)"); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("4. rows of R: what each output pixel measures")
    im = ax[2, 0].imshow(t["C"] / np.sqrt(np.outer(np.diag(t["C"]), np.diag(t["C"]))), vmin=-1, vmax=1, cmap="RdBu_r")
    ax[2, 0].set_title("5. noise correlation of F_hat (neighbours anti-correlated)"); ax[2, 0].set_xlabel("pixel j"); ax[2, 0].set_ylabel("pixel j")
    plt.colorbar(im, ax=ax[2, 0], label="correlation coefficient")
    im = ax[2, 1].imshow(Cr / np.sqrt(np.outer(np.diag(Cr), np.diag(Cr))), vmin=-1, vmax=1, cmap="RdBu_r")
    ax[2, 1].set_title("6. noise correlation of F_tilde (none)"); ax[2, 1].set_xlabel("pixel i"); ax[2, 1].set_ylabel("pixel i")
    plt.colorbar(im, ax=ax[2, 1], label="correlation coefficient")
    fig.tight_layout()
    os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fig.savefig(os.path.join(figdir, "extraction_toy.png"), dpi=150)
    np.savez(os.path.join(cachedir, "extraction_toy.npz"), **{k: v for k, v in t.items()}, cov_F_tilde=Cr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    a = ap.parse_args()
    main(a.fig, a.figcache)
