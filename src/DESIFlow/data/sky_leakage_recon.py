"""
Does the mean sky-fibre residual scale with the sky brightness, or does it track the calibration vector?

For a sample of Loa exposures, per petal and camera, over the sky fibres (cframe OBJTYPE SKY, FIBERSTATUS 0):
    A(lambda)     inverse-variance weighted mean calibrated residual (cframe flux)  [1e-17 erg/s/cm2/A]
    s(lambda)     median over the same fibres of the sky model in the same units, sky / fluxcalib
    C(lambda)     median fluxcalib over the same fibres
Across (exposure, petal) pairs, per wavelength: the weighted least-squares fit A = a + kappa * s (an additive offset a and a
fractional leakage kappa of the sky), and the same fit for the residual in electrons, A * C = a_e + kappa_e * (s * C).

Run:  python -m DESIFlow.data.sky_leakage_recon --out OUT --fig FIG_DIR --figcache CACHE_DIR [--n-exp 24 --workers 8]
"""
import argparse
import multiprocessing as mp
import os

import numpy as np

ROOT = "/global/cfs/cdirs/desi/spectro/redux/loa"
CAM = {"b": (3600.0, 2751), "r": (5760.0, 2326), "z": (7520.0, 2881)}


def one(args):
    import fitsio
    night, expid, petal = args
    d = f"{ROOT}/exposures/{night}/{expid:08d}/"
    out = {}
    for c, (l0, n) in CAM.items():
        try:
            tag = f"{c}{petal}-{expid:08d}.fits.gz"
            sky = fitsio.read(d + "sky-" + tag, ext="SKY")
            fc = fitsio.read(d + "fluxcalib-" + tag, ext="FLUXCALIB")
            fm = fitsio.read(d + "cframe-" + tag, ext="FIBERMAP", columns=["OBJTYPE", "FIBERSTATUS"])
            F = fitsio.FITS(d + "cframe-" + tag)
            flux, ivar, mask = F["FLUX"].read(), F["IVAR"].read(), F["MASK"].read()
        except Exception:
            return None
        sel = (np.char.strip(fm["OBJTYPE"].astype(str)) == "SKY") & (fm["FIBERSTATUS"] == 0)
        if sel.sum() < 10:
            return None
        w = np.where((mask[sel] == 0) & (ivar[sel] > 0), ivar[sel], 0.0)
        A = (w * flux[sel]).sum(0) / np.where(w.sum(0) > 0, w.sum(0), np.nan)
        sig = 1.0 / np.sqrt(np.where(w.sum(0) > 0, w.sum(0), np.nan))
        with np.errstate(all="ignore"):
            s = np.nanmedian(np.where(fc[sel] > 0, sky[sel] / fc[sel], np.nan), 0)
        C = np.median(fc[sel], 0)
        out[c] = np.stack([A, sig, s, C])
    return (night, expid, petal), out


def main(out, figdir, cachedir, n_exp=24, workers=8):
    import fitsio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = fitsio.FITS(ROOT + "/exposures-loa.fits")["EXPOSURES"].read(columns=["NIGHT", "EXPID", "SURVEY", "PROGRAM"])
    m = np.flatnonzero((np.char.strip(t["SURVEY"].astype(str)) == "main") & (t["NIGHT"] > 20210601))
    pick = m[np.linspace(0, len(m) - 1, n_exp).astype(int)]
    jobs = [(int(t["NIGHT"][i]), int(t["EXPID"][i]), p) for i in pick for p in range(10)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = [r for r in pool.imap_unordered(one, jobs) if r is not None]
    print(len(res), "of", len(jobs), "exposure-petal pairs read", flush=True)
    os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    cache = {}
    for c, (l0, n) in CAM.items():
        ok = [r for r in res if c in r[1]]
        X = np.stack([r[1][c] for r in ok])                                               # (pairs, 4, n)
        A, sig, s, C = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        lam = l0 + 0.8 * np.arange(n)
        a_fit, k_fit, k_err, k_e, a_e = (np.full(n, np.nan) for _ in range(5))
        for j in range(n):
            g = np.isfinite(A[:, j]) & np.isfinite(s[:, j]) & np.isfinite(sig[:, j]) & (sig[:, j] > 0)
            if g.sum() < 30:
                continue
            for y, x, w, ka, aa, err in ((A[g, j], s[g, j], 1 / sig[g, j] ** 2, k_fit, a_fit, k_err),):
                M = np.stack([np.ones(g.sum()), x], 1)
                cov = np.linalg.inv(M.T @ (M * w[:, None]))
                beta = cov @ (M.T @ (w * y))
                aa[j], ka[j], err[j] = beta[0], beta[1], np.sqrt(cov[1, 1])
            ye, xe = A[g, j] * C[g, j], s[g, j] * C[g, j]
            we = 1 / (sig[g, j] * C[g, j]) ** 2
            M = np.stack([np.ones(g.sum()), xe], 1)
            beta = np.linalg.inv(M.T @ (M * we[:, None])) @ (M.T @ (we * ye))
            a_e[j], k_e[j] = beta[0], beta[1]
        wmean = np.nansum(A / sig ** 2, 0) / np.nansum(1 / sig ** 2, 0)
        for a_, y, lab in ((ax[0], np.nanmedian(s, 0), None), (ax[1], a_fit, None), (ax[2], k_fit, None), (ax[3], np.nanmedian(C, 0), None)):
            a_.plot(lam, y, lw=0.7, label=f"camera {c}")
        cache.update({f"{c}_wavelength_A": lam, f"{c}_median_sky_model": np.nanmedian(s, 0), f"{c}_offset_a": a_fit, f"{c}_leakage_kappa": k_fit,
                      f"{c}_leakage_kappa_err": k_err, f"{c}_median_fluxcalib": np.nanmedian(C, 0), f"{c}_weighted_mean_residual": wmean,
                      f"{c}_offset_electrons": a_e, f"{c}_leakage_electrons": k_e, f"{c}_n_pairs": np.array(len(ok))})
        print(c, "pairs", len(ok), "| median kappa %.4f (err %.4f) | median offset a %.4f | median kappa_e %.4f | median offset_e %.2f e/A" %
              (np.nanmedian(k_fit), np.nanmedian(k_err), np.nanmedian(a_fit), np.nanmedian(k_e), np.nanmedian(a_e)), flush=True)
    ax[0].set_ylabel("median sky model\n[1e-17 erg/s/cm2/A]"); ax[0].set_yscale("log"); ax[0].legend()
    ax[1].set_ylabel("offset a\n[1e-17 erg/s/cm2/A]"); ax[1].axhline(0, color="gray", lw=0.8)
    ax[2].set_ylabel("sky leakage kappa\n(residual per unit sky)"); ax[2].axhline(0, color="gray", lw=0.8); ax[2].set_ylim(-0.05, 0.05)
    ax[3].set_ylabel("median fluxcalib C\n[1e17 cm2 count s/erg]"); ax[3].set_xlabel("observed wavelength [Angstrom]"); ax[3].set_yscale("log")
    ax[0].set_title(f"Sky-fibre residual vs sky brightness: fit A = a + kappa s over {len(res)} exposure-petal pairs")
    fig.savefig(os.path.join(figdir, "sky_leakage_recon.png"), dpi=150, bbox_inches="tight")
    np.savez(os.path.join(cachedir, "sky_leakage_recon.npz"), **cache)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True); ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    ap.add_argument("--n-exp", type=int, default=24); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.out, a.fig, a.figcache, a.n_exp, a.workers)
