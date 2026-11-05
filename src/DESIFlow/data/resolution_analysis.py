"""
Figures and numbers of the resolution-matrix study from the sample written by resolution_study.py.

Every figure NAME.png is written together with NAME.npz holding all the plotted arrays (fig dir and cache dir are separate).
Summary numbers go to summary.json in the study directory.

Run:  python -m DESIFlow.data.resolution_analysis --study STUDY_DIR --cache DR2_CACHE --fig FIG_DIR --figcache CACHE_DIR
"""
import argparse
import json
import os

import numpy as np

from DESIFlow.data.dr2_cache import N_PIX, load_index
from DESIFlow.data.resolution_study import NODES, ND, PX_A

LAM = 3600.0 + PX_A * NODES
BANDS = {"b (3800-5500 A)": (3800, 5500), "r (5900-7400 A)": (5900, 7400), "z (7700-9500 A)": (7700, 9500)}


def take(ds, rows):
    """h5py fancy read of arbitrary unique rows (h5py needs increasing indices)."""
    rows = np.asarray(rows)
    o = np.argsort(rows)
    out = np.empty(len(rows), ds.dtype)
    out[o] = ds[rows[o]]
    return out


def load(study, cache):
    import h5py
    t = dict(np.load(os.path.join(study, "table.npz")))
    idx = load_index(cache)
    g = t["group"]
    t["prog"] = np.array([f"{idx['group_survey'][i]}|{idx['group_program'][i]}" for i in g])
    t["z"] = np.full(len(g), np.nan)
    t["spectype"] = np.full(len(g), "", "U8")
    for c, name in ((0, "tgt"), (1, "sky")):
        m = t["cls"] == c
        with h5py.File(os.path.join(cache, name, "metadata.h5"), "r") as f:
            z = f["zall"]
            t["z"][m] = take(z["Z"], t["cache_row"][m])
            t["spectype"][m] = np.char.decode(take(z["SPECTYPE"], t["cache_row"][m]), "ascii")
    t["sigma_A"] = t["sigma_px"] * PX_A
    return t


def omega2(y, labels, min_count=4):
    """Bias-corrected share of the variance of y explained by a categorical factor (omega squared, clipped at 0)."""
    u, inv, cnt0 = np.unique(labels, return_inverse=True, return_counts=True)
    keep = cnt0[inv] >= min_count                                                      # levels with a single sample carry no within-level information
    y, labels = y[keep], labels[keep]
    u, inv = np.unique(labels, return_inverse=True)
    n, k = len(y), len(u)
    cnt = np.bincount(inv)
    means = np.bincount(inv, y) / cnt
    ssb = np.sum(cnt * (means - y.mean()) ** 2)
    ssw = np.sum((y - means[inv]) ** 2)
    msw = ssw / max(n - k, 1)
    return max((ssb - (k - 1) * msw) / (ssb + ssw + msw), 0.0)


def band_mean(sig, lo, hi):
    m = (LAM >= lo) & (LAM <= hi)
    return sig[:, m].mean(1)


def savefig(fig, name, figdir, cachedir, **arrays):
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(cachedir, exist_ok=True)
    fig.savefig(os.path.join(figdir, name + ".png"), dpi=150, bbox_inches="tight")
    np.savez(os.path.join(cachedir, name + ".npz"), **arrays)


def substitution_errors(study, t, chunk=400):
    """
    Relative error eps of a model LSF when the true matrix R_i is replaced by a stand-in R_hat, per spectrum:
    eps = || R_hat - R_i || / || R_i ||  per column (pixel) of the matrix, median over the columns that are normalised to 1 +- 0.3. A line of matched-filter S/N
    s then gets a chi2 error of about (s * eps)^2, so eps = 1/s is the line S/N where the stand-in starts to matter.
    Stand-ins: the mean of the spectrum's own class, the mean of the SKY sample, the SKY mean of the same petal, and a Gaussian
    with the median width of the class at the node pixels (nodes only).
    """
    lib = np.load(os.path.join(study, "library_R_float16.npy"), mmap_mode="r")
    n = len(lib)
    sums = {c: np.zeros((ND, N_PIX)) for c in (0, 1)}
    psum = {}
    cnt = {c: 0 for c in (0, 1)}
    pcnt = {}
    for a in range(0, n, chunk):
        X = np.asarray(lib[a:a + chunk], np.float64)
        cl, pe = t["cls"][a:a + chunk], t["petal"][a:a + chunk]
        for c in (0, 1):
            sums[c] += X[cl == c].sum(0)
            cnt[c] += int((cl == c).sum())
        for p in np.unique(pe[cl == 1]):
            sel = (cl == 1) & (pe == p)
            psum[p] = psum.get(p, 0) + X[sel].sum(0)
            pcnt[p] = pcnt.get(p, 0) + int(sel.sum())
    mean = {c: sums[c] / cnt[c] for c in (0, 1)}
    pmean = {p: psum[p] / pcnt[p] for p in psum if pcnt[p] >= 20}
    eps = {k: np.full(n, np.nan) for k in ("own_class_mean", "sky_mean", "sky_mean_same_petal")}
    eps_col = {k: np.zeros((2, len(NODES))) for k in eps}
    colcount = {k: np.zeros((2, len(NODES))) for k in eps}
    for a in range(0, n, chunk):
        X = np.asarray(lib[a:a + chunk], np.float64)
        cl, pe = t["cls"][a:a + chunk], t["petal"][a:a + chunk]
        nrm = np.sqrt((X ** 2).sum(1))
        valid = np.abs(X.sum(1) - 1.0) < 0.3                                           # drop empty / badly normalised columns
        for i in range(len(X)):
            c = cl[i]
            for k, ref in (("own_class_mean", mean[c]), ("sky_mean", mean[1]), ("sky_mean_same_petal", pmean.get(pe[i], None))):
                if ref is None:
                    continue
                e = np.sqrt(((X[i] - ref) ** 2).sum(0)) / np.where(nrm[i] > 0, nrm[i], np.nan)
                e = np.where(valid[i], e, np.nan)
                eps[k][a + i] = np.nanmedian(e)
                ok = np.isfinite(e[NODES])
                eps_col[k][c] += np.where(ok, e[NODES], 0.0)
                colcount[k][c] += ok
    for k in eps_col:
        eps_col[k] = eps_col[k] / np.maximum(colcount[k], 1)
    # Gaussian stand-in at the node columns (width = class median at the node)
    off = np.arange(ND) - ND // 2
    gauss = np.full(n, np.nan)
    med = {c: np.median(t["sigma_px"][t["cls"] == c], 0) for c in (0, 1)}
    for a in range(0, n, chunk):
        X = np.asarray(lib[a:a + chunk], np.float64)[:, :, NODES]
        for i in range(len(X)):
            s = med[t["cls"][a + i]]
            G = np.exp(-0.5 * (off[:, None] / s[None, :]) ** 2)
            G /= G.sum(0)
            nr = np.sqrt((X[i] ** 2).sum(0))
            e = np.sqrt(((X[i] - G) ** 2).sum(0)) / np.where(nr > 0, nr, np.nan)
            gauss[a + i] = np.nanmedian(np.where(np.abs(X[i].sum(0) - 1.0) < 0.3, e, np.nan))
    eps["gaussian_median_width"] = gauss
    return eps, eps_col, mean


def main(study, cache, figdir, cachedir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    t = load(study, cache)
    sky, tgt = t["cls"] == 1, t["cls"] == 0
    tgt1, tgt3 = tgt & (t["n_exp"] == 1), tgt & (t["n_exp"] >= 3)
    summ = {"n_spectra": int(len(sky)), "n_sky": int(sky.sum()), "n_tgt": int(tgt.sum()), "n_tgt_single_exposure": int(tgt1.sum()),
            "n_tgt_3plus_exposures": int(tgt3.sum()), "sky_n_exp_median": float(np.median(t["n_exp"][sky]))}

    # ---------------------------------------------------------------- figure 1: LSF width vs wavelength
    fig, ax = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    arrays = {"wavelength_A": LAM}
    for name, m, col in (("SKY", sky, "tab:blue"), ("TGT, all", tgt, "tab:red"), ("TGT, 1 exposure", tgt1, "tab:orange")):
        q = np.percentile(t["sigma_A"][m], [16, 50, 84], axis=0)
        ax[0].plot(LAM, q[1], color=col, label=f"{name} (N={int(m.sum())})")
        ax[0].fill_between(LAM, q[0], q[2], color=col, alpha=0.2)
        arrays[f"sigma_A_p16_p50_p84_{name.replace(', ', '_').replace(' ', '_')}"] = q
    ax[0].set_ylabel("LSF rms width sigma [Angstrom]")
    ax[0].legend()
    ax[0].set_title("Width of the resolution-matrix columns of sampled DR2 spectra (band: 16-84 percent)")
    ratio = np.median(t["sigma_A"][sky], 0) / np.median(t["sigma_A"][tgt1], 0)
    ks = np.array([stats.ks_2samp(t["sigma_A"][sky][:, j], t["sigma_A"][tgt1][:, j]).statistic for j in range(len(NODES))])
    ax[1].plot(LAM, ratio, color="k", label="median SKY / median TGT (1 exposure)")
    ax[1].axhline(1, color="gray", lw=0.8)
    ax[1].set_ylabel("ratio of median widths")
    ax[1].set_xlabel("observed wavelength [Angstrom]")
    ax2 = ax[1].twinx()
    ax2.plot(LAM, ks, color="tab:green", ls="--", label="KS distance SKY vs TGT (1 exposure)")
    ax2.set_ylabel("Kolmogorov-Smirnov distance")
    ax[1].legend(loc="upper left"); ax2.legend(loc="upper right")
    savefig(fig, "resolution_lsf_width_vs_wavelength", figdir, cachedir, **arrays, ratio_sky_over_tgt1=ratio, ks_sky_vs_tgt1=ks)
    plt.close(fig)
    summ["sigma_A_median_by_band"] = {n: {"sky": float(np.median(band_mean(t["sigma_A"][sky], *r))), "tgt1": float(np.median(band_mean(t["sigma_A"][tgt1], *r))),
                                         "tgt3plus": float(np.median(band_mean(t["sigma_A"][tgt3], *r)))} for n, r in BANDS.items()}
    summ["ratio_sky_over_tgt1_range"] = [float(ratio.min()), float(ratio.max())]
    summ["ks_sky_vs_tgt1_median_max"] = [float(np.median(ks)), float(ks.max())]

    # ---------------------------------------------------------------- figure 2: distributions per camera
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    arrays = {}
    for a, (name, r) in zip(ax, BANDS.items()):
        lo = min(np.percentile(band_mean(t["sigma_A"], *r), 0.5), 1e9)
        hi = np.percentile(band_mean(t["sigma_A"], *r), 99.5)
        bins = np.linspace(lo, hi, 50)
        for lab, m, col in (("SKY", sky, "tab:blue"), ("TGT, 1 exposure", tgt1, "tab:orange"), ("TGT, 3+ exposures", tgt3, "tab:red")):
            v = band_mean(t["sigma_A"][m], *r)
            h, _ = np.histogram(v, bins, density=True)
            a.step(0.5 * (bins[1:] + bins[:-1]), h, where="mid", color=col, label=lab)
            arrays[f"{name.split()[0]}_{lab.replace(', ', '_').replace(' ', '_').replace('+', 'plus')}_hist"] = h
        arrays[f"{name.split()[0]}_bin_centres_A"] = 0.5 * (bins[1:] + bins[:-1])
        a.set_xlabel("band-mean LSF width sigma [Angstrom]")
        a.set_title(name)
    ax[0].set_ylabel("probability density [1/Angstrom]")
    ax[0].legend()
    savefig(fig, "resolution_lsf_width_distributions", figdir, cachedir, **arrays)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 3: what explains the variation
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    arrays = {"wavelength_A": LAM}
    summ["omega2_band_mean"] = {}
    for a, (title, m) in zip(ax, (("SKY", sky), ("TGT, single exposure", tgt1))):
        fac = {"petal (spectrograph)": t["petal"][m], "fibre": t["fiber"][m], "month": t["night"][m] // 100, "survey|program": t["prog"][m]}
        sig = t["sigma_A"][m]
        for lab, g in fac.items():
            w = np.array([omega2(sig[:, j], g) for j in range(len(NODES))])
            a.plot(LAM, w, label=lab)
            arrays[f"omega2_{title.split(',')[0]}_{lab.split()[0].replace('|', '_')}"] = w
            summ["omega2_band_mean"].setdefault(title, {})[lab] = float(w.mean())
        a.set_title(f"{title} (N={int(m.sum())})")
        a.set_xlabel("observed wavelength [Angstrom]")
    ax[0].set_ylabel("share of the variance of sigma explained (omega squared)")
    ax[0].legend()
    savefig(fig, "resolution_variance_sources", figdir, cachedir, **arrays)
    plt.close(fig)

    # ---------------------------------------------------------------- redshift / type dependence (target, single exposure)
    sb = band_mean(t["sigma_A"], 3800, 9500)
    m = tgt1 & np.isfinite(t["z"])
    X = np.column_stack([np.eye(10)[np.clip(t["petal"][m], 0, 9)], (t["night"][m] // 100 - (t["night"][m] // 100).mean())])
    res = sb[m] - X @ np.linalg.lstsq(X, sb[m], rcond=None)[0]
    summ["sigma_vs_z_spearman_raw"] = float(stats.spearmanr(sb[m], t["z"][m])[0])
    summ["sigma_vs_z_spearman_after_petal_and_month"] = float(stats.spearmanr(res, t["z"][m])[0])
    summ["sigma_by_spectype_median_A"] = {s: float(np.median(sb[m & (t["spectype"] == s)])) for s in np.unique(t["spectype"][m]) if (m & (t["spectype"] == s)).sum() >= 30}

    # ---------------------------------------------------------------- figure 4: substitution error
    eps, eps_col, mean = substitution_errors(study, t)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    arrays = {"wavelength_A": LAM}
    bins = np.logspace(-2.5, 0, 60)
    summ["substitution_eps"] = {}
    for k, col in (("own_class_mean", "tab:green"), ("sky_mean", "tab:blue"), ("sky_mean_same_petal", "tab:purple"), ("gaussian_median_width", "tab:gray")):
        for lab, m in (("tgt", tgt), ("sky", sky)):
            v = eps[k][m]
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            summ["substitution_eps"][f"{k}|{lab}"] = {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "critical_line_snr_median": float(1 / np.median(v)), "critical_line_snr_p95": float(1 / np.percentile(v, 95))}
        v = eps[k][tgt]
        v = v[np.isfinite(v)]
        h, _ = np.histogram(v, bins, density=True)
        ax[0].step(np.sqrt(bins[1:] * bins[:-1]), h, where="mid", color=col, label=f"{k.replace('_', ' ')} (target spectra)")
        arrays[f"hist_tgt_{k}"] = h
    arrays["eps_bin_centres"] = np.sqrt(bins[1:] * bins[:-1])
    ax[0].set_xscale("log"); ax[0].set_xlabel("relative LSF error eps = |R_standin - R_true| / |R_true|"); ax[0].set_ylabel("probability density [per unit eps]")
    ax[0].legend(fontsize=8)
    ax[0].set_title("error of a stand-in matrix (all matrix columns)")
    for k, col in (("own_class_mean", "tab:green"), ("sky_mean", "tab:blue"), ("sky_mean_same_petal", "tab:purple")):
        ax[1].plot(LAM, eps_col[k][0], color=col, label=f"{k.replace('_', ' ')}")
        arrays[f"eps_vs_wavelength_tgt_{k}"] = eps_col[k][0]
    ax[1].set_xlabel("observed wavelength [Angstrom]"); ax[1].set_ylabel("mean eps at the node columns (target spectra)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("where the stand-in is worst")
    savefig(fig, "resolution_substitution_error", figdir, cachedir, **arrays)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 5: column normalisation
    fig, ax = plt.subplots(figsize=(10, 4.5))
    q = np.percentile(t["colsum"], [1, 16, 50, 84, 99], axis=0)
    ax.plot(LAM, q[2], color="k", label="median")
    ax.fill_between(LAM, q[1], q[3], color="tab:blue", alpha=0.3, label="16-84 percent")
    ax.fill_between(LAM, q[0], q[4], color="tab:blue", alpha=0.15, label="1-99 percent")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("observed wavelength [Angstrom]"); ax.set_ylabel("column sum of the resolution band (ideal: 1)")
    ax.set_title("Normalisation of the sampled resolution matrices, all spectra"); ax.legend()
    savefig(fig, "resolution_column_sum_vs_wavelength", figdir, cachedir, wavelength_A=LAM, percentiles_1_16_50_84_99=q)
    plt.close(fig)
    summ["column_sum_outside_5pct_fraction_by_band"] = {n: float((np.abs(t["colsum"][:, (LAM >= r[0]) & (LAM <= r[1])] - 1) > 0.05).mean()) for n, r in BANDS.items()}
    summ["column_sum_outside_5pct_fraction_overall"] = float((np.abs(t["colsum"] - 1) > 0.05).mean())

    # ---------------------------------------------------------------- matrix health
    lib = np.load(os.path.join(study, "library_R_float16.npy"), mmap_mode="r")
    sub = np.sort(np.random.default_rng(0).choice(len(lib), 1500, replace=False))
    X = np.asarray(lib[sub], np.float32)
    cs = X.sum(1)
    summ["health"] = {"column_sum_p1_p50_p99": [float(v) for v in np.percentile(cs, [1, 50, 99])],
                      "edge_diagonal_weight_max": float(np.abs(X[:, [0, ND - 1]]).sum(1).max() / 1.0),
                      "edge_diagonal_weight_median": float(np.median(np.abs(X[:, [0, ND - 1]]).sum(1))),
                      "fraction_negative_entries": float((X < 0).mean()),
                      "centroid_px_p1_p50_p99": [float(v) for v in np.percentile(t["centroid"], [1, 50, 99])]}
    with open(os.path.join(study, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--fig", required=True)
    ap.add_argument("--figcache", required=True)
    a = ap.parse_args()
    main(a.study, a.cache, a.fig, a.figcache)
