"""
Would sky-shaped additive templates help? Regress sky-fibre residuals on the exposure's own sky model and its derivatives.

Per fibre (sky fibres of Loa cframes), per camera: the calibrated residual r(lambda) is fitted, with inverse-variance weights, by
    r ~ c0 * s + c1 * ds/dlambda + c2 * d2s/dlambda2          (s = sky / fluxcalib of that fibre: the exposure's own sky model)
and the chi2 before and after the nested fits (s; s + s'; s + s' + s'') is accumulated over fibres, separately for sky-line
pixels (the 8% of pixels with the largest s - running median of s) and for the rest. Under pure noise the fit removes about
3 chi2 per fibre and camera (a control with the basis shuffled over wavelength gives the chance level). The chi2 removed beyond the shuffled
null is the usefulness of such a template (chi2 per pixel is about 0.9, so a fraction of the 'excess' is not meaningful).

Run:  python -m DESIFlow.data.sky_line_regression --fig FIG_DIR --figcache CACHE_DIR [--n-exp 24 --workers 8]
"""
import argparse
import multiprocessing as mp
import os

import numpy as np

ROOT = "/global/cfs/cdirs/desi/spectro/redux/loa"
CAM = {"b": (3600.0, 2751), "r": (5760.0, 2326), "z": (7520.0, 2881)}


def fit_nested(B, r, w):
    """chi2 before and after the nested weighted fits with the first 1, 2, 3 columns of B (n, 3); returns (before, [after1, after2, after3])."""
    out = []
    for k in (1, 2, 3):
        X = B[:, :k]
        A = X.T @ (X * w[:, None]) + 1e-12 * np.eye(k)
        c = np.linalg.solve(A, X.T @ (w * r))
        out.append(float((w * (r - X @ c) ** 2).sum()))
    return float((w * r ** 2).sum()), out


def one(args):
    import fitsio
    from scipy.ndimage import median_filter
    night, expid, petal = args
    d = f"{ROOT}/exposures/{night}/{expid:08d}/"
    rng = np.random.default_rng(expid * 10 + petal)
    res, demo = {}, {}
    for c, (l0, n) in CAM.items():
        try:
            tag = f"{c}{petal}-{expid:08d}.fits.gz"
            sky = fitsio.read(d + "sky-" + tag, ext="SKY"); fc = fitsio.read(d + "fluxcalib-" + tag, ext="FLUXCALIB")
            fm = fitsio.read(d + "cframe-" + tag, ext="FIBERMAP", columns=["OBJTYPE", "FIBERSTATUS"])
            F = fitsio.FITS(d + "cframe-" + tag)
            flux, ivar, mask = F["FLUX"].read(), F["IVAR"].read(), F["MASK"].read()
        except Exception:
            return None
        sel = np.flatnonzero((np.char.strip(fm["OBJTYPE"].astype(str)) == "SKY") & (fm["FIBERSTATUS"] == 0))
        if len(sel) < 10:
            return None
        acc = dict(nfib=0, nL=0, nC=0, L_before=0.0, C_before=0.0, L_after=np.zeros(3), C_after=np.zeros(3), L_null=0.0, C_null=0.0)
        for i in sel:
            with np.errstate(all="ignore"):
                s = np.where(fc[i] > 0, sky[i] / fc[i], 0.0)
            good = (mask[i] == 0) & (ivar[i] > 0) & np.isfinite(s)
            if good.sum() < 500:
                continue
            ds = np.gradient(s, 0.8); d2 = np.gradient(ds, 0.8)
            B = np.stack([s / (np.abs(s).max() + 1e-9), ds / (np.abs(ds).max() + 1e-9), d2 / (np.abs(d2).max() + 1e-9)], 1)
            line = (s - median_filter(s, 101, mode="nearest")) > np.percentile(s - median_filter(s, 101, mode="nearest"), 92)
            L, Cn = good & line, good & ~line
            r, w = flux[i].astype(np.float64), ivar[i].astype(np.float64)
            for tag_, m in (("L", L), ("C", Cn)):
                b4, af = fit_nested(B[m], r[m], w[m])
                acc[tag_ + "_before"] += b4; acc[tag_ + "_after"] += np.array(af); acc["n" + tag_] += int(m.sum())
                perm = rng.permutation(len(s))                                              # null: the same basis at random wavelengths
                _, nu = fit_nested(B[perm][m], r[m], w[m]); acc[tag_ + "_null"] += nu[2]
            acc["nfib"] += 1
        res[c] = acc
        if petal == 0:
            Bm = np.stack([np.where(fc[sel] > 0, sky[sel] / fc[sel], 0.0).mean(0)], 1)
            demo[c] = (np.average(flux[sel], 0, weights=np.where(mask[sel] == 0, ivar[sel], 0.0) + 1e-12), Bm[:, 0])
    return (night, expid, petal), res, demo


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
        out = [r for r in pool.imap_unordered(one, jobs) if r is not None]
    print(len(out), "exposure-petal pairs", flush=True)
    os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    cache = {}
    labels = ["sky s", "+ ds/dlambda", "+ d2s/dlambda2"]
    for ci, c in enumerate("brz"):
        T = {k: (sum(r[1][c][k] for r in out if c in r[1])) for k in ("nfib", "nL", "nC", "L_before", "C_before", "L_after", "C_after", "L_null", "C_null")}
        n = T["nfib"]
        rem = [(T["L_before"] - T["L_after"][0]) / n, (T["L_after"][0] - T["L_after"][1]) / n, (T["L_after"][1] - T["L_after"][2]) / n]
        null = (T["L_before"] - T["L_null"]) / n
        total = sum(rem)
        print(f"{c}: {int(n)} fibre-camera fits; line-pixel chi2 per fibre {T['L_before'] / n:.0f} ({T['nL'] / n:.0f} pixels), other pixels {T['C_before'] / T['nC']:.3f} per pixel; "
              f"chi2 removed per fibre by s / +ds / +d2s: {rem[0]:.1f} / {rem[1]:.1f} / {rem[2]:.1f} (3-term {total:.1f}, shuffled-basis null {null:.1f}, real {total - null:.1f} = {100 * (total - null) / (T['L_before'] / n):.1f}% of the line-pixel chi2)", flush=True)
        for k in range(3):
            ax[0].bar(ci + 0.25 * (k - 1), rem[k], 0.22, color=("tab:blue", "tab:orange", "tab:green")[k], label=labels[k] if ci == 0 else None)
        ax[0].plot([ci - 0.4, ci + 0.4], [null] * 2, "k--", lw=1, label="chance (shuffled basis, all 3 terms)" if ci == 0 else None)
        cache.update({f"{c}_{k}": np.asarray(v) for k, v in T.items()}); cache[f"{c}_chi2_removed_per_fibre_s_ds_d2s"] = np.array(rem); cache[f"{c}_chi2_removed_null_per_fibre"] = null
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["b", "r", "z"]); ax[0].set_xlabel("camera"); ax[0].set_ylabel("chi2 removed per fibre in sky-line pixels (added by each term)")
    ax[0].legend(); ax[0].set_title("what sky-shaped terms remove from the sky-line residual")
    demo = [r[2] for r in out if "r" in r[2]]
    if demo:
        res_, s_ = demo[0]["r"]
        lam = CAM["r"][0] + 0.8 * np.arange(len(res_)); w_ = (lam > 7240) & (lam < 7340)
        ax[1].plot(lam[w_], res_[w_], color="k", lw=0.8, label="mean sky-fibre residual of one petal [1e-17 erg/s/cm2/A]")
        ax[1].plot(lam[w_], 0.01 * s_[w_], color="tab:red", lw=0.8, label="1% of the sky model")
        ax[1].set_xlabel("observed wavelength [Angstrom]"); ax[1].set_ylabel("flux [1e-17 erg/s/cm2/A]"); ax[1].legend(fontsize=8); ax[1].set_title("one OH-line region, camera r")
        cache.update(demo_wavelength_A=lam[w_], demo_residual=res_[w_], demo_sky_model=s_[w_])
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "sky_line_regression.png"), dpi=150)
    np.savez(os.path.join(cachedir, "sky_line_regression.npz"), **cache)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    ap.add_argument("--n-exp", type=int, default=24); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.fig, a.figcache, a.n_exp, a.workers)
