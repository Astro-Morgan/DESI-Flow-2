"""
Do stacked faint TARGET spectra show the junction steps of the SKY-fibre residual stack?

Per-pixel inverse-variance weighted means of N random faint target rows (cache tgt/, snr_med < 1.5) and N random SKY rows (cache sky/,
the sky-fibre residuals). At each camera junction (5780 and 7570 A) the step is the jump of a straight line fitted on the two sides
(60-200 A away, the 30 A around the junction left out) evaluated at the junction. A step in the sky stack that is absent in the
target stack means the sky-fibre offsets do not transfer; a step in both means they are shared.

Run:  python -m DESIFlow.data.stack_junction_check --cache DR2_CACHE --fig FIG --figcache CACHE [--n 30000]
"""
import argparse
import os

import numpy as np

LAM = 3600.0 + 0.8 * np.arange(7781)
JUNCTIONS = (5780.0, 7570.0)


def stack(cache, kind, n, seed, snr_max=None):
    d = os.path.join(cache, kind)
    F, I, M = (np.load(os.path.join(d, k + ".npy"), mmap_mode="r") for k in ("flux", "ivar", "mask"))
    st, ng, sn = (np.asarray(np.load(os.path.join(d, k + ".npy"), mmap_mode="r")) for k in ("status", "ngood", "snr_med"))
    ok = (st == 1) & (ng >= 4000)
    if snr_max is not None:
        ok &= np.isfinite(sn) & (sn < snr_max)
    rows = np.sort(np.random.default_rng(seed).choice(np.flatnonzero(ok), n, replace=False))
    num, den, w2 = np.zeros(7781), np.zeros(7781), np.zeros(7781)
    for a in range(0, n, 2000):
        r = rows[a:a + 2000]
        f, iv, m = np.asarray(F[r], np.float64), np.asarray(I[r], np.float64), np.asarray(M[r])
        w = np.where(m == 0, iv, 0.0)
        num += (w * f).sum(0); den += w.sum(0)
    return num / np.maximum(den, 1e-30), 1 / np.sqrt(np.maximum(den, 1e-30))


def step_at(lam, y, e, j):
    left = (lam >= j - 200) & (lam < j - 30) & (lam >= j - 200); right = (lam > j + 30) & (lam <= j + 200)
    out = []
    for sel in (left, right):
        w = 1 / e[sel] ** 2
        A = np.stack([np.ones(sel.sum()), lam[sel] - j], 1)
        cov = np.linalg.inv(A.T @ (A * w[:, None])); beta = cov @ (A.T @ (w * y[sel]))
        out.append((beta[0], cov[0, 0]))
    return out[1][0] - out[0][0], float(np.sqrt(out[0][1] + out[1][1]))


def main(cache, figdir, cachedir, n=30000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {"faint targets (S/N < 1.5)": stack(cache, "tgt", n, 0, 1.5), "SKY rows": stack(cache, "sky", n, 1)}
    os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    cache_out = {"wavelength_A": LAM}
    for i, (name, (y, e)) in enumerate(res.items()):
        ax[i, 0].plot(LAM, y, lw=0.5); ax[i, 0].set_title(name + f": stack of {n} rows"); ax[i, 0].set_xlabel("observed wavelength [Angstrom]"); ax[i, 0].set_ylabel("weighted mean flux [1e-17 erg/s/cm2/A]")
        ax[i, 0].set_ylim(np.nanpercentile(y, 1) - 0.1, np.nanpercentile(y, 99) + 0.1)
        cache_out[name.split()[0] + "_mean"] = y; cache_out[name.split()[0] + "_err"] = e
        for jj, j in enumerate(JUNCTIONS):
            m = (LAM > j - 200) & (LAM < j + 200)
            ax[i, 1 + jj].errorbar(LAM[m][::4], y[m][::4], e[m][::4], fmt=".", ms=2, lw=0.5)
            s, se = step_at(LAM, y, e, j)
            ax[i, 1 + jj].set_title(f"{name}, junction {j:.0f} A: step {s:+.4f} +- {se:.4f}"); ax[i, 1 + jj].set_xlabel("observed wavelength [Angstrom]"); ax[i, 1 + jj].set_ylabel("weighted mean flux [1e-17 erg/s/cm2/A]")
            ax[i, 1 + jj].axvline(j, color="gray", lw=0.6, ls=":")
            print(f"{name:28s} junction {j:.0f}: step {s:+.4f} +- {se:.4f} [1e-17 erg/s/cm2/A]", flush=True)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "stack_junction_check.png"), dpi=150); np.savez(os.path.join(cachedir, "stack_junction_check.npz"), **cache_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True); ap.add_argument("--n", type=int, default=30000)
    a = ap.parse_args()
    main(a.cache, a.fig, a.figcache, a.n)
