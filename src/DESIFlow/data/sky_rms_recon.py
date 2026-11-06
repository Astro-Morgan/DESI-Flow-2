"""
What sets the wavelength dependence of the rms of flux * sqrt(ivar) on sky fibres (panel 1 of sky_residual_recon)?

flux * sqrt(ivar) is the same in electrons and in calibrated flux units, so it cannot contain the calibration. It is
(counts - sky model) / sigma_counts. Candidate causes tested here, on sky fibres of a sample of Loa exposures:
  * the variance model depends on the electron counts (read noise vs Poisson), and the counts follow the sensitivity and the sky
    spectrum, so an error in it would look like a sensitivity curve: rms against the sky model m_e [electron/A] per camera, and
    rms against wavelength at fixed counts;
  * the sky model is fitted to these very fibres: expected rms^2 = (1 - h) / (1 + h) with h ~ 1 / n_sky: rms^2 against 1/n_sky.

Run:  python -m DESIFlow.data.sky_rms_recon --fig FIG_DIR --figcache CACHE_DIR [--n-exp 24 --workers 8]
"""
import argparse
import multiprocessing as mp
import os

import numpy as np

ROOT = "/global/cfs/cdirs/desi/spectro/redux/loa"
CAM = {"b": (3600.0, 2751), "r": (5760.0, 2326), "z": (7520.0, 2881)}
EDGES = np.logspace(0.0, 3.6, 37)                                                         # sky model, electron/A per fibre


def one(args):
    import fitsio
    night, expid, petal = args
    d = f"{ROOT}/exposures/{night}/{expid:08d}/"
    out = {}
    for c, (l0, n) in CAM.items():
        try:
            tag = f"{c}{petal}-{expid:08d}.fits.gz"
            sky = fitsio.read(d + "sky-" + tag, ext="SKY")
            fm = fitsio.read(d + "cframe-" + tag, ext="FIBERMAP", columns=["OBJTYPE", "FIBERSTATUS"])
            F = fitsio.FITS(d + "cframe-" + tag)
            flux, ivar, mask = F["FLUX"].read(), F["IVAR"].read(), F["MASK"].read()
        except Exception:
            return None
        sel = (np.char.strip(fm["OBJTYPE"].astype(str)) == "SKY") & (fm["FIBERSTATUS"] == 0)
        if sel.sum() < 10:
            return None
        good = (mask[sel] == 0) & (ivar[sel] > 0)
        r = np.where(good, flux[sel] * np.sqrt(ivar[sel]), 0.0)
        m = sky[sel]
        k = np.digitize(m, EDGES) - 1
        ok = good & (k >= 0) & (k < len(EDGES) - 1)
        s2 = np.bincount(k[ok], (r[ok]) ** 2, len(EDGES) - 1); cnt = np.bincount(k[ok], minlength=len(EDGES) - 1)
        lo = good & (m < 15); hi = good & (m > 150)                                          # fixed counts, rms against wavelength
        out[c] = dict(n_sky=int(sel.sum()), s2=s2, cnt=cnt, rms_all=np.sqrt((r ** 2).sum() / max(good.sum(), 1)),
                      lo_s2=(r ** 2 * lo).sum(0), lo_n=lo.sum(0), hi_s2=(r ** 2 * hi).sum(0), hi_n=hi.sum(0))
    return (night, expid, petal), out


def main(figdir, cachedir, n_exp=24, workers=8):
    import fitsio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = fitsio.FITS(ROOT + "/exposures-loa.fits")["EXPOSURES"].read(columns=["NIGHT", "EXPID", "SURVEY"])
    m = np.flatnonzero((np.char.strip(t["SURVEY"].astype(str)) == "main") & (t["NIGHT"] > 20210601))
    pick = m[np.linspace(0, len(m) - 1, n_exp).astype(int)]
    jobs = [(int(t["NIGHT"][i]), int(t["EXPID"][i]), p) for i in pick for p in range(10)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = [r for r in pool.imap_unordered(one, jobs) if r is not None]
    print(len(res), "exposure-petal pairs", flush=True)
    os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    mid = np.sqrt(EDGES[1:] * EDGES[:-1])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    cache = {"sky_model_bin_centres_e_per_A": mid}
    for c, (l0, n) in CAM.items():
        s2 = sum(r[1][c]["s2"] for r in res if c in r[1]); cnt = sum(r[1][c]["cnt"] for r in res if c in r[1])
        rms = np.sqrt(s2 / np.maximum(cnt, 1)); rms[cnt < 2000] = np.nan
        ax[0].plot(mid, rms, label=f"camera {c}"); cache[f"{c}_rms_vs_counts"] = rms; cache[f"{c}_count_vs_counts"] = cnt
        lam = l0 + 0.8 * np.arange(n)
        for key, ls in (("lo", "-"), ("hi", "--")):
            s = sum(r[1][c][key + "_s2"] for r in res if c in r[1]); nn = sum(r[1][c][key + "_n"] for r in res if c in r[1])
            y = np.sqrt(s / np.maximum(nn, 1)); y[nn < 500] = np.nan
            ax[1].plot(lam, y, ls=ls, lw=0.6, color={"b": "tab:blue", "r": "tab:orange", "z": "tab:green"}[c], label=f"{c}, {'< 15' if key == 'lo' else '> 150'} e/A")
            cache[f"{c}_rms_vs_wavelength_{key}"] = y
        cache[f"{c}_wavelength_A"] = lam
        ns = np.array([r[1][c]["n_sky"] for r in res if c in r[1]]); rr = np.array([r[1][c]["rms_all"] for r in res if c in r[1]])
        ax[2].plot(1 / ns, rr ** 2, ".", ms=3, alpha=0.5, label=f"camera {c}")
        cache[f"{c}_inv_nsky"] = 1 / ns; cache[f"{c}_rms2_pair"] = rr ** 2
        sl = np.polyfit(1 / ns, rr ** 2, 1)
        print(f"{c}: rms^2 = {sl[1]:.4f} + {sl[0]:.2f}/n_sky  (leverage expectation 1 - 2/n_sky);  median n_sky {np.median(ns):.0f}; rms at counts 1-10 / 10-100 / 100-1000 e/A: "
              + " / ".join("%.3f" % np.sqrt(np.nansum((s2 / np.maximum(cnt, 1))[(mid >= lo) & (mid < hi)] * cnt[(mid >= lo) & (mid < hi)]) / max(cnt[(mid >= lo) & (mid < hi)].sum(), 1)) for lo, hi in ((1, 10), (10, 100), (100, 1000))))
    ax[0].set_xscale("log"); ax[0].axhline(1, color="gray", lw=0.8); ax[0].set_xlabel("sky model counts [electron/Angstrom per fibre]"); ax[0].set_ylabel("rms of flux x sqrt(ivar)"); ax[0].legend()
    ax[0].set_title("rms against counts")
    ax[1].axhline(1, color="gray", lw=0.8); ax[1].set_xlabel("observed wavelength [Angstrom]"); ax[1].set_ylabel("rms of flux x sqrt(ivar)"); ax[1].legend(fontsize=7, ncol=3)
    ax[1].set_title("rms against wavelength at fixed counts"); ax[1].set_ylim(0.6, 2.0)
    ax[2].set_xlabel("1 / number of sky fibres used"); ax[2].set_ylabel("rms squared (pair average)"); ax[2].legend(); ax[2].set_title("in-sample leverage?")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "sky_rms_recon.png"), dpi=150)
    np.savez(os.path.join(cachedir, "sky_rms_recon.npz"), **cache)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    ap.add_argument("--n-exp", type=int, default=24); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.fig, a.figcache, a.n_exp, a.workers)
