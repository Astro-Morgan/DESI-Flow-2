"""
Is the pipeline ivar right on TARGET spectra? Repeat observations as an independent check.

For targets with exactly two cached coadds (independent exposures, different survey/program), the normalised difference
    d = (f1 - f2) * sqrt( ivar1 * ivar2 / (ivar1 + ivar2) )
has rms 1 on every pixel if both ivar are right and the object and the calibration are the same in both. Per wavelength pixel, split
by the median S/N of the target (faint objects are noise and sky dominated, so this probes the variance model, not the objects).
Reported: rms of d and the robust width 1.4826 * median(|d|).

Run:  python -m DESIFlow.data.ivar_repeat_check --cache DR2_CACHE --fig FIG_DIR --figcache CACHE_DIR [--n 20000]
"""
import argparse
import os

import numpy as np

from DESIFlow.data.dr2_cache import N_PIX

LAM = 3600.0 + 0.8 * np.arange(N_PIX)
CATS = {"faint (S/N < 1.5)": (-np.inf, 1.5), "middle (1.5-5)": (1.5, 5.0), "bright (> 5)": (5.0, np.inf)}


def main(cache, figdir, cachedir, n=20000, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = os.path.join(cache, "tgt")
    mem, start, cnt = np.load(d + "/sc_members.npy", mmap_mode="r"), np.load(d + "/sc_member_start.npy"), np.load(d + "/sc_n_total.npy")
    snr, st = np.load(d + "/snr_med.npy", mmap_mode="r"), np.load(d + "/status.npy", mmap_mode="r")
    F, I, M = (np.load(d + f"/{k}.npy", mmap_mode="r") for k in ("flux", "ivar", "mask"))
    two = np.flatnonzero(cnt == 2)
    sel = np.sort(np.random.default_rng(seed).choice(two, n, replace=False))
    r1, r2 = np.asarray(mem[start[sel]]), np.asarray(mem[start[sel] + 1])
    s1, s2 = np.asarray(snr[r1]), np.asarray(snr[r2])
    keep = np.isfinite(s1) & np.isfinite(s2)
    r1, r2, s1, s2 = r1[keep], r2[keep], s1[keep], s2[keep]
    ss = np.minimum(s1, s2)
    print("pairs", len(r1), "| snr categories:", {k: int(((ss >= lo) & (ss < hi)).sum()) for k, (lo, hi) in CATS.items()}, flush=True)
    out = {}
    for k, (lo, hi) in CATS.items():
        sub = np.flatnonzero((ss >= lo) & (ss < hi))
        s2sum, cn = np.zeros(N_PIX), np.zeros(N_PIX)
        keepd = []
        for a in range(0, len(sub), 2000):
            ii = sub[a:a + 2000]
            o1, o2 = np.argsort(r1[ii]), np.argsort(r2[ii])
            f1 = np.empty((len(ii), N_PIX), np.float32); i1 = np.empty_like(f1); m1 = np.empty((len(ii), N_PIX), np.uint8)
            f2, i2, m2 = np.empty_like(f1), np.empty_like(f1), np.empty_like(m1)
            f1[o1], i1[o1], m1[o1] = F[r1[ii][o1]], I[r1[ii][o1]], M[r1[ii][o1]]
            f2[o2], i2[o2], m2[o2] = F[r2[ii][o2]], I[r2[ii][o2]], M[r2[ii][o2]]
            good = (m1 == 0) & (m2 == 0) & (i1 > 0) & (i2 > 0)
            dd = np.where(good, (f1 - f2) * np.sqrt(np.where(good, i1 * i2 / (i1 + i2 + 1e-30), 0.0)), np.nan)
            s2sum += np.nansum(dd ** 2, 0); cn += good.sum(0)
            keepd.append(np.abs(dd[:500]))
        absd = np.concatenate(keepd)
        with np.errstate(all="ignore"):
            mad = 1.4826 * np.nanmedian(absd, 0)
        rms = np.sqrt(s2sum / np.maximum(cn, 1)); rms[cn < 500] = np.nan
        out[k] = (rms, mad, cn)
        print(f"{k}: {len(sub)} pairs, median rms {np.nanmedian(rms):.3f}, median robust width {np.nanmedian(mad):.3f}", flush=True)
        for lo_, hi_ in ((3600, 3900), (4000, 5500), (5800, 7400), (7700, 9300), (9500, 9824)):
            m = (LAM >= lo_) & (LAM < hi_)
            print(f"   {lo_}-{hi_} A: rms {np.nanmedian(rms[m]):.3f}  robust {np.nanmedian(mad[m]):.3f}", flush=True)
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for k, col in zip(CATS, ("tab:blue", "tab:orange", "tab:red")):
        ax[0].plot(LAM, out[k][0], lw=0.5, color=col, label=k)
        ax[1].plot(LAM, out[k][1], lw=0.5, color=col, label=k)
    for a, lab in zip(ax, ("rms of the normalised difference", "robust width 1.4826 median|d|")):
        a.axhline(1, color="gray", lw=0.8); a.set_ylabel(lab); a.set_ylim(0.6, 1.6)
    ax[0].legend(); ax[1].set_xlabel("observed wavelength [Angstrom]")
    ax[0].set_title("Repeat observations of the same target: do the stated ivar match the scatter?")
    os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fig.savefig(os.path.join(figdir, "ivar_repeat_check.png"), dpi=150, bbox_inches="tight")
    np.savez(os.path.join(cachedir, "ivar_repeat_check.npz"), wavelength_A=LAM, **{f"{k}_{n_}": v for k, vv in out.items() for n_, v in zip(("rms", "robust", "n_pix"), vv)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    main(a.cache, a.fig, a.figcache, a.n)
