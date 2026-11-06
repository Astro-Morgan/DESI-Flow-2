"""
Visual comparison of the instrument-template candidates and of what they do to real target residuals.

Figures (each with a same-named .npz data cache):
  template_comparison_shapes   the first four global K=20 templates and the per-camera templates (b: 3, r: 4, z: 12 available, first four shown),
                               in flux units per unit coefficient, on observed wavelength, camera by camera
  template_comparison_fits     four example targets (2 galaxies, 2 stars, fixed random draw, not selected by outcome): per camera the smoothed
                               data - Redrock model residual and the fits of four candidates, each fitted on the visible pixels only
                               (the hidden spans of the nuisance test are grey); hidden-pixel chi2 per pixel of every candidate in the titles
  template_comparison_population   median over all 4000 targets of the fitted correction (fraction of the model) and of the remainder after the fit

Run:  python -m DESIFlow.templates.template_fit_comparison --res target_residuals.npz --templates K020.npz --percam-out sky_templates --fig FIG --figcache CACHE
"""
import argparse
import json
import os

import numpy as np

from DESIFlow.data.target_nuisance_test import BRZ0, EDGE_ZONES, NPIX, OFF, fit_predict, legendre_basis, pixel_grid, span_mask
from DESIFlow.templates.camera_templates import choose_k

COL = {"mult3": "tab:green", "global": "tab:blue", "percam": "tab:red", "mult3+percam": "k"}
LAB = {"mult3": "multiplicative Legendre x model (9)", "global": "global instrument K=20 (20)", "percam": "per-camera templates 3/4/12 (19)", "mult3+percam": "multiplicative + per-camera (28)"}


def nw(f, w, sig):
    h = int(4 * sig); x = np.arange(-h, h + 1); k = np.exp(-0.5 * (x / sig) ** 2)
    num = np.convolve(w * f, k, "same"); den = np.convolve(w, k, "same")
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def load_sets(tpl_path, percam_out):
    idx, cam, lam, edge = pixel_grid()
    T = np.load(tpl_path)["templates_flux"][:, idx].T
    blocks, ksel, pc = [], {}, {}
    for c in "brz":
        S = json.load(open(os.path.join(percam_out, "instrument_cam", c, "summary.json"))); K = choose_k(S); ksel[c] = K
        t = np.load(os.path.join(percam_out, "instrument_cam", c, f"K{K:03d}.npz"))["templates_flux"]; pc[c] = t
        blk = np.zeros((sum(NPIX.values()), K)); blk[OFF[c]:OFF[c] + NPIX[c]] = t.T; blocks.append(blk)
    return idx, lam, edge, T, np.hstack(blocks), ksel, pc


def cam_lambda(c):
    return 3600.0 + 0.8 * (BRZ0[c] + np.arange(NPIX[c]))


def shade_edges(ax, lo, hi):
    for a, b in EDGE_ZONES:
        if b > lo and a < hi:
            ax.axvspan(max(a, lo), min(b, hi), color="tab:orange", alpha=0.12, lw=0)


def fig_shapes(tpl_path, pc, figdir, cachedir, nshow=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = np.load(tpl_path)["templates_flux"]
    fig, ax = plt.subplots(2 * nshow, 3, figsize=(18, 2.0 * 2 * nshow), sharex="col")
    cache = {}
    for j, c in enumerate("brz"):
        lam = cam_lambda(c); sl = slice(BRZ0[c], BRZ0[c] + NPIX[c]); lo, hi = lam[0], lam[-1]
        for k in range(nshow):
            a = ax[k, j]; y = g[k, sl]; a.plot(lam, y, lw=0.6, color="tab:blue"); shade_edges(a, lo, hi); a.axhline(0, color="gray", lw=0.5)
            cache[f"global_{c}_{k + 1}"] = y
            a.set_ylabel(f"global #{k + 1}\n[1e-17 erg/s/cm2/A per unit coeff.]" if j == 0 else "", fontsize=7)
            a2 = ax[nshow + k, j]
            if k < len(pc[c]):
                y2 = pc[c][k]; a2.plot(lam, y2, lw=0.6, color="tab:red"); shade_edges(a2, lo, hi); a2.axhline(0, color="gray", lw=0.5); cache[f"percam_{c}_{k + 1}"] = y2
            else:
                a2.text(0.5, 0.5, f"camera {c} uses only {len(pc[c])} templates" if k == len(pc[c]) else "", transform=a2.transAxes, ha="center", fontsize=8, color="gray")
            a2.set_ylabel(f"per-camera #{k + 1}\n[1e-17 erg/s/cm2/A per unit coeff.]" if j == 0 else "", fontsize=7)
        ax[0, j].set_title(f"camera {c} (observed {lo:.0f}-{hi:.0f} A); orange = camera-edge zones", fontsize=9)
        ax[-1, j].set_xlabel("observed wavelength [Angstrom]")
        cache[f"wavelength_{c}_A"] = lam
    fig.suptitle(f"Instrument templates: global K=20 (blue, rows 1-{nshow}) vs per-camera (red; selected K per camera: " + ", ".join(f"{c}={len(pc[c])}" for c in "brz") + f"; first {nshow} shown). Template order = energy; signs are arbitrary.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(figdir, "template_comparison_shapes.png"), dpi=130); plt.close(fig)
    np.savez(os.path.join(cachedir, "template_comparison_shapes.npz"), **cache)


def fig_fits(R, W, M, st, T, Bpc, lam, figdir, cachedir, seed=11):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    gal = rng.choice(np.where(st == "GALAXY")[0], 2, replace=False); star = rng.choice(np.where(st == "STAR")[0], 2, replace=False)
    sel = list(gal) + list(star); P = legendre_basis(2)
    fig, ax = plt.subplots(len(sel), 3, figsize=(19, 3.6 * len(sel)))
    cache = {"targets": np.array(sel), "spectype": st[sel]}
    for row, i in enumerate(sel):
        mc = M[i][:, None] * P
        bases = {"mult3": mc, "global": T, "percam": Bpc, "mult3+percam": np.hstack([mc, Bpc])}
        hid = span_mask(len(R[i]), np.random.default_rng(1000 * i)); vis = (~hid).astype(float); hs = hid & (W[i] > 0)
        preds = {n: fit_predict(B, R[i], W[i], vis)[0] for n, B in bases.items()}
        chi = {n: float((W[i] * (R[i] - p) ** 2)[hs].sum() / hs.sum()) for n, p in preds.items()}
        chi0 = float((W[i] * R[i] ** 2)[hs].sum() / hs.sum())
        for j, c in enumerate("brz"):
            sl = slice(OFF[c], OFF[c] + NPIX[c]); x = cam_lambda(c); a = ax[row, j]
            sm = nw(R[i][sl], W[i][sl], 8.0); a.plot(x, sm, color="0.5", lw=1.0, label="data - Redrock model (NW smooth, 8 px)" if j == 0 else None)
            for n in bases:
                a.plot(x, nw(preds[n][sl], np.ones(NPIX[c]), 1.0), color=COL[n], lw=1.0, label=LAB[n] if j == 0 else None)
            h = hid[sl]; start = None
            for k in range(len(h) + 1):
                if k < len(h) and h[k] and start is None: start = k
                if (k == len(h) or not h[k]) and start is not None: a.axvspan(x[start], x[k - 1], color="0.85", lw=0, zorder=0); start = None
            lim = np.nanpercentile(np.abs(sm), 99); a.set_ylim(-1.6 * lim, 1.6 * lim)
            if j == 0: a.set_ylabel(f"target {i} ({st[i]}, z={0:.2f})\n[1e-17 erg/s/cm2/A]".replace("0.00", "%.2f" % 0) if False else f"target {i}: {st[i]}\n[1e-17 erg/s/cm2/A]", fontsize=8)
            if row == len(sel) - 1: a.set_xlabel("observed wavelength [Angstrom]")
            cache[f"t{i}_{c}_wavelength_A"] = x; cache[f"t{i}_{c}_data_nw"] = sm
            for n in bases: cache[f"t{i}_{c}_{n}"] = preds[n][sl]
        ax[row, 1].set_title(f"target {i} ({st[i]}) hidden-pixel chi2/px: none {chi0:.2f} | mult3 {chi['mult3']:.2f} | global {chi['global']:.2f} | per-camera {chi['percam']:.2f} | mult3+per-cam {chi['mult3+percam']:.2f}", fontsize=9)
        cache[f"t{i}_chi2"] = np.array([chi0, chi["mult3"], chi["global"], chi["percam"], chi["mult3+percam"]])
    ax[0, 0].legend(fontsize=7, loc="upper left"); fig.suptitle("Fits to Redrock-model residuals, fitted on visible pixels only (grey = hidden spans, scored in the titles); columns = cameras b, r, z", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(os.path.join(figdir, "template_comparison_fits.png"), dpi=120); plt.close(fig)
    np.savez(os.path.join(cachedir, "template_comparison_fits.npz"), **cache)


def fig_population(R, W, M, st, T, Bpc, figdir, cachedir, binw=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = len(R); P = legendre_basis(2); keys = ["mult3", "global", "percam", "mult3+percam"]
    corr = {n: np.full(R.shape, np.nan) for n in keys}
    for i in range(N):
        mc = M[i][:, None] * P; bases = {"mult3": mc, "global": T, "percam": Bpc, "mult3+percam": np.hstack([mc, Bpc])}
        for n in keys: corr[n][i] = fit_predict(bases[n], R[i], W[i], np.ones(len(R[i])))[0]
    good = (M > 1.0) & (W > 0); Ms = np.where(M != 0, M, 1)
    fr = np.where(good, R / Ms, np.nan); cf = {n: np.where(good, corr[n] / Ms, np.nan) for n in keys}; rem = {n: np.where(good, (R - corr[n]) / Ms, np.nan) for n in keys}
    groups = {"galaxies": st == "GALAXY", "stars": st == "STAR"}
    fig, ax = plt.subplots(2, 2, figsize=(18, 9), sharex=True, sharey="row"); cache = {}
    for g, (gname, gm) in enumerate(groups.items()):
        for c in "brz":
            starts = np.arange(OFF[c], OFF[c] + NPIX[c] - binw, binw); x = 3600.0 + 0.8 * (BRZ0[c] + (starts - OFF[c]) + binw / 2)
            med = lambda A: np.array([np.nanmedian(A[gm][:, s:s + binw]) for s in starts])
            d = 100 * med(fr); ax[0, g].plot(x, d, color="0.55", lw=1.2, label="data - model" if c == "b" else None); cache[f"{gname}_{c}_x"] = x; cache[f"{gname}_{c}_data"] = d
            for n in keys:
                y = 100 * med(cf[n]); z = 100 * med(rem[n]); cache[f"{gname}_{c}_{n}_fit"] = y; cache[f"{gname}_{c}_{n}_remainder"] = z
                ax[0, g].plot(x, y, color=COL[n], lw=1.0, label=LAB[n] if c == "b" else None); ax[1, g].plot(x, z, color=COL[n], lw=1.0)
            ax[1, g].plot(x, d, color="0.55", lw=0.8, alpha=0.7)
        ax[0, g].set_title(f"{gname}: {gm.sum()} targets"); ax[1, g].set_xlabel("observed wavelength [Angstrom] (cameras overlap)")
        ax[0, g].set_ylim(-4, 4); ax[1, g].set_ylim(-4, 4)
        for a in ax[:, g]: a.axhline(0, color="gray", lw=0.6)
    ax[0, 0].set_ylabel("median fitted correction / model [percent]"); ax[1, 0].set_ylabel("median remainder (data - model - fit) / model [percent]\n(grey: before the fit)"); ax[0, 0].legend(fontsize=8, ncol=1)
    fig.suptitle("What each candidate absorbs of the common residual: median over targets in 8 A bins, model flux > 1e-17 (fits use all pixels)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(figdir, "template_comparison_population.png"), dpi=130); plt.close(fig)
    np.savez(os.path.join(cachedir, "template_comparison_population.npz"), **cache)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--res", required=True); ap.add_argument("--templates", required=True); ap.add_argument("--percam-out", required=True)
    ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True); ap.add_argument("--which", default="shapes,fits,population")
    a = ap.parse_args(); os.makedirs(a.fig, exist_ok=True); os.makedirs(a.figcache, exist_ok=True)
    d = np.load(a.res); R, W, M, st = d["resid"].astype(np.float64), d["ivar"].astype(np.float64), d["model"].astype(np.float64), d["spectype"]
    idx, lam, edge, T, Bpc, ksel, pc = load_sets(a.templates, a.percam_out); print("per-camera K:", ksel)
    w = a.which.split(",")
    if "shapes" in w: fig_shapes(a.templates, pc, a.fig, a.figcache)
    if "fits" in w: fig_fits(R, W, M, st, T, Bpc, lam, a.fig, a.figcache)
    if "population" in w: fig_population(R, W, M, st, T, Bpc, a.fig, a.figcache)
