"""
Reconnaissance for the SKY-template step: what do the cached OBJTYPE == SKY spectra (sky-fibre residuals after sky subtraction,
flux calibrated) look like? Per pixel, over a random sample of sky rows: mean and rms of the normalised residual
r = flux * sqrt(ivar) on good pixels (pure noise would give mean 0 and rms 1), and the mean calibrated residual flux.

Run:  python -m DESIFlow.data.sky_residual_recon --cache DR2_CACHE --fig FIG_DIR --figcache CACHE_DIR [--n 30000]
"""
import argparse
import os

import numpy as np

from DESIFlow.data.dr2_cache import N_PIX

LAM = 3600.0 + 0.8 * np.arange(N_PIX)


def main(cache, figdir, cachedir, n=30000, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F, I, M = (np.load(os.path.join(cache, "sky", k + ".npy"), mmap_mode="r") for k in ("flux", "ivar", "mask"))
    ng = np.load(os.path.join(cache, "sky", "ngood.npy"), mmap_mode="r")
    ok = np.flatnonzero(np.asarray(ng) > 0)
    rows = np.sort(np.random.default_rng(seed).choice(ok, n, replace=False))
    s1 = np.zeros(N_PIX); s2 = np.zeros(N_PIX); cnt = np.zeros(N_PIX); f1 = np.zeros(N_PIX); w1 = np.zeros(N_PIX)
    for a in range(0, n, 2000):
        r = rows[a:a + 2000]
        f, iv, m = np.asarray(F[r], np.float64), np.asarray(I[r], np.float64), np.asarray(M[r])
        g = (m == 0) & (iv > 0)
        x = np.where(g, f * np.sqrt(iv), 0.0)
        s1 += x.sum(0); s2 += (x ** 2).sum(0); cnt += g.sum(0)
        f1 += np.where(g, f, 0.0).sum(0); w1 += g.sum(0)
    mean, rms = s1 / np.maximum(cnt, 1), np.sqrt(s2 / np.maximum(cnt, 1))
    fmean = f1 / np.maximum(w1, 1)
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(LAM, rms, lw=0.6); ax[0].axhline(1, color="gray", lw=0.8)
    ax[0].set_ylabel("rms of flux x sqrt(ivar)\n(1 = pure noise)"); ax[0].set_yscale("log")
    ax[0].set_title(f"Cached SKY-fibre spectra (sky-subtracted residuals): {n} random rows")
    ax[1].plot(LAM, mean, lw=0.6); ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set_ylabel("mean of flux x sqrt(ivar)\n(0 = unbiased)")
    ax[2].plot(LAM, fmean, lw=0.6); ax[2].axhline(0, color="gray", lw=0.8)
    ax[2].set_ylabel("mean calibrated residual flux\n[1e-17 erg/s/cm2/A]"); ax[2].set_xlabel("observed wavelength [Angstrom]")
    os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fig.savefig(os.path.join(figdir, "sky_residual_recon.png"), dpi=150, bbox_inches="tight")
    np.savez(os.path.join(cachedir, "sky_residual_recon.npz"), wavelength_A=LAM, rms_normalised_residual=rms, mean_normalised_residual=mean,
             mean_residual_flux=fmean, n_good=cnt, rows=rows)
    q = np.percentile(rms, [5, 50, 95, 99.9])
    print("rms of normalised residual: p5 %.3f median %.3f p95 %.3f p99.9 %.2f | max %.1f at %.0f A" % (*q, rms.max(), LAM[np.argmax(rms)]))
    print("mean normalised residual: median %.4f, min %.3f, max %.3f" % (np.median(mean), mean.min(), mean.max()))
    for lo, hi in ((3600, 3800), (5560, 5900), (7400, 7700), (9600, 9824)):
        m = (LAM >= lo) & (LAM < hi)
        print("  %5d-%5d A: mean residual flux %.4f, mean normalised %.3f, rms %.3f" % (lo, hi, fmean[m].mean(), mean[m].mean(), rms[m].mean()))
    for name, lam in (("[OI] 5577", 5577), ("[OI] 6300", 6300), ("OH 7993", 7993), ("OH 8344", 8344), ("OH 8827", 8827), ("OH 9439", 9439)):
        j = int(round((lam - 3600) / 0.8)); w = slice(max(j - 3, 0), j + 4)
        print("  %-10s rms max in +-3 px: %.2f" % (name, rms[w].max()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    ap.add_argument("--n", type=int, default=30000)
    a = ap.parse_args()
    main(a.cache, a.fig, a.figcache, a.n)
