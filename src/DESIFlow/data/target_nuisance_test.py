"""
Which nuisance terms reduce the residual of bright targets against their Redrock models?

Input: target_residuals.npz (per-camera residual = data - Redrock model, ivar, model; b 2751 + r 2326 + z 2881 pixels). Per target, with the
same hidden pixels for every method (36 random spans, about 25% of the pixels, two draws), the coefficients are fitted by weighted least
squares on the visible pixels and the chi2 per pixel is scored on the hidden ones:
  none        the Redrock model as it is
  add3        additive per-camera Legendre polynomials of degree <= 2 (9 terms; what Redrock's archetype mode does)
  mult3       multiplicative per-camera Legendre polynomials of degree <= 2 times the model (9 terms)
  mult1       one multiplicative scale per camera (3 terms)
  inst20      the K = 20 instrument templates (additive, brz grid), 20 terms
  inst4       the first four instrument templates, 4 terms
  mult3+inst20, mult1+inst20
Reported: mean hidden chi2 per pixel, the improvement over none with its standard error over targets, overall and split into the 150 A
camera-edge zones and the interior; also the median fractional residual (data - model) / model against wavelength.

Run:  python -m DESIFlow.data.target_nuisance_test --res RES.npz --templates K020.npz --fig FIG --figcache CACHE [--sets NAME=path ...]
"""
import argparse
import json
import os

import numpy as np
from numpy.polynomial import legendre

NPIX = {"b": 2751, "r": 2326, "z": 2881}
OFF = {"b": 0, "r": 2751, "z": 2751 + 2326}
BRZ0 = {"b": 0, "r": 2700, "z": 4900}                                                           # brz grid index of the first pixel of each camera
EDGE_ZONES = [(3600, 3750), (5650, 5800), (5760, 5910), (7470, 7620), (7520, 7670), (9674, 9824)]


def pixel_grid():
    idx = np.concatenate([BRZ0[c] + np.arange(NPIX[c]) for c in "brz"])
    cam = np.concatenate([np.full(NPIX[c], i) for i, c in enumerate("brz")])
    lam = 3600.0 + 0.8 * idx
    edge = np.zeros(len(lam), bool)
    for lo, hi in EDGE_ZONES:
        edge |= (lam >= lo) & (lam < hi)
    return idx, cam, lam, edge


def legendre_basis(deg=2):
    cols = []
    for c in "brz":
        x = np.linspace(-1, 1, NPIX[c])
        for n in range(deg + 1):
            v = np.zeros(sum(NPIX.values()))
            v[OFF[c]:OFF[c] + NPIX[c]] = legendre.legval(x, [0] * n + [1])
            cols.append(v)
    return np.stack(cols, 1)


def span_mask(L, rng, n_spans=36, len_range=(5, 250)):
    hid = np.zeros(L, bool)
    ln = np.exp(rng.uniform(np.log(len_range[0]), np.log(len_range[1]), n_spans)).astype(int)
    st = rng.integers(0, L, n_spans)
    for a, b in zip(st, st + ln):
        hid[a:b] = True
    return hid


def fit_predict(X, r, w, vis):
    ww = w * vis
    A = X.T @ (X * ww[:, None]); A += 1e-9 * np.trace(A) / len(A) * np.eye(len(A))
    c = np.linalg.solve(A, X.T @ (ww * r))
    return X @ c, c


def main(res_path, tpl_path, figdir, cachedir, extra_sets=None, percam_out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(res_path)
    R, W, M, st = d["resid"].astype(np.float64), d["ivar"].astype(np.float64), d["model"].astype(np.float64), d["spectype"]
    idx, cam, lam, edge = pixel_grid()
    z = np.load(tpl_path)
    T = z["templates_flux"][:, idx].T                                                            # (Npix, 20), flux units per unit coefficient
    P = legendre_basis(2)
    names = ["add3", "mult3", "mult1", "inst20", "inst4", "mult3+inst20", "mult1+inst20"]
    for k, v in (extra_sets or {}).items():
        names.append(k)
    Bpc = None
    if percam_out:                                                                                  # per-camera templates, each camera with its own K
        from DESIFlow.templates.camera_templates import choose_k
        blocks, ksel = [], {}
        for c in "brz":
            S = json.load(open(os.path.join(percam_out, "instrument_cam", c, "summary.json"))); K = choose_k(S); ksel[c] = K
            blk = np.zeros((sum(NPIX.values()), K)); blk[OFF[c]:OFF[c] + NPIX[c]] = np.load(os.path.join(percam_out, "instrument_cam", c, f"K{K:03d}.npz"))["templates_flux"].T
            blocks.append(blk)
        Bpc = np.hstack(blocks); names += ["percam", "mult3+percam"]
        print(f"per-camera templates: K = {ksel} (the K with the best hold-out chi2 of each camera), {Bpc.shape[1]} terms")
    rng = np.random.default_rng(0)
    N = len(R)
    chi = {n: np.zeros((N, 3)) for n in ["none"] + names}                                          # all, edge, interior hidden chi2 per pixel
    mult1 = np.zeros((N, 3))
    for i in range(N):
        mc = M[i][:, None] * P                                                                   # multiplicative columns
        bases = {"add3": P, "mult3": mc, "mult1": mc[:, [0, 3, 6]], "inst20": T, "inst4": T[:, :4], "mult3+inst20": np.hstack([mc, T]), "mult1+inst20": np.hstack([mc[:, [0, 3, 6]], T])}
        if Bpc is not None:
            bases["percam"] = Bpc; bases["mult3+percam"] = np.hstack([mc, Bpc])
        for k, B in (extra_sets or {}).items():
            bases[k] = B(i, M[i], idx) if callable(B) else B
        sums = {n: np.zeros(6) for n in chi}
        for rep in range(2):
            hid = span_mask(len(R[i]), np.random.default_rng(1000 * i + rep)); vis = (~hid).astype(float)
            sel = hid & (W[i] > 0)
            for n in chi:
                pred = 0.0 if n == "none" else fit_predict(bases[n], R[i], W[i], vis)[0]
                e = W[i] * (R[i] - pred) ** 2
                sums[n] += [e[sel].sum(), sel.sum(), e[sel & edge].sum(), (sel & edge).sum(), e[sel & ~edge].sum(), (sel & ~edge).sum()]
        for n in chi:
            s = sums[n]; chi[n][i] = [s[0] / max(s[1], 1), s[2] / max(s[3], 1), s[4] / max(s[5], 1)]
        mult1[i] = fit_predict(bases["mult1"], R[i], W[i], np.ones(len(R[i])))[1]
    out = {}
    print("hidden-pixel chi2 per pixel (mean over targets) and the improvement over the model as it is: delta = chi2(method) - chi2(none), mean +- standard error")
    groups = {"all targets": np.ones(N, bool), "galaxies": st == "GALAXY", "stars": st == "STAR"}
    for gname, gm in groups.items():
        print(f"--- {gname}: {gm.sum()} targets; none: all {chi['none'][gm, 0].mean():.4f}, edge zones {chi['none'][gm, 1].mean():.4f}, interior {chi['none'][gm, 2].mean():.4f}")
        for n in names:
            dd = chi[n][gm] - chi["none"][gm]
            out[(gname, n)] = (dd.mean(0), dd.std(0) / np.sqrt(gm.sum()))
            print(f"  {n:13s} all {dd[:, 0].mean():+.4f} +- {dd[:, 0].std() / np.sqrt(gm.sum()):.4f} | edge {dd[:, 1].mean():+.4f} +- {dd[:, 1].std() / np.sqrt(gm.sum()):.4f} | interior {dd[:, 2].mean():+.4f} +- {dd[:, 2].std() / np.sqrt(gm.sum()):.4f} | targets improved {100 * (dd[:, 0] < 0).mean():.0f}%")
    print("multiplicative scale per camera (fitted on all pixels): mean and std over targets [percent]: b %.2f +- %.2f, r %.2f +- %.2f, z %.2f +- %.2f" % tuple(
        v for c in range(3) for v in (100 * mult1[:, c].mean(), 100 * mult1[:, c].std())))
    os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for a, (gname, gm) in zip(ax[:2], list(groups.items())[1:]):
        x = np.arange(len(names)); wd = 0.27
        for j, (zn, col) in enumerate((("all", "tab:blue"), ("edge zones", "tab:orange"), ("interior", "tab:green"))):
            a.bar(x + (j - 1) * wd, [out[(gname, n)][0][j] for n in names], wd, yerr=[out[(gname, n)][1][j] for n in names], color=col, label=zn)
        a.set_xticks(x); a.set_xticklabels(names, rotation=35, ha="right", fontsize=8); a.axhline(0, color="gray", lw=0.8)
        a.set_ylabel("change of the hidden-pixel chi2 per pixel (negative = better)"); a.set_title(f"{gname}: {gm.sum()} targets"); a.legend(fontsize=8)
    good = (M > 1.0) & (W > 0)
    frac = np.where(good, R / np.where(M != 0, M, 1), np.nan)
    bins = np.arange(0, frac.shape[1], 10)
    med = np.array([np.nanmedian(frac[:, a:a + 10]) for a in bins]); lo = np.array([np.nanpercentile(frac[:, a:a + 10], 16) for a in bins]); hi = np.array([np.nanpercentile(frac[:, a:a + 10], 84) for a in bins])
    xl = lam[bins]
    ax[2].plot(xl, 100 * med, ".", ms=2, color="k"); ax[2].fill_between(xl, 100 * lo, 100 * hi, color="tab:blue", alpha=0.15, lw=0)
    ax[2].set_ylim(-8, 8); ax[2].axhline(0, color="gray", lw=0.8); ax[2].set_xlabel("observed wavelength [Angstrom] (cameras overlap)"); ax[2].set_ylabel("(data - model) / model [percent]")
    ax[2].set_title("median fractional residual, model flux > 1e-17 (band 16-84%)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "target_nuisance_test.png"), dpi=150); plt.close(fig)
    np.savez(os.path.join(cachedir, "target_nuisance_test.npz"), methods=np.array(["none"] + names), chi2_all_edge_interior=np.stack([chi[n] for n in ["none"] + names]),
             spectype=st, mult1_coeff=mult1, wavelength_A=xl, frac_resid_median=med, frac_resid_p16=lo, frac_resid_p84=hi)
    return out, chi, names


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--res", required=True); ap.add_argument("--templates", required=True); ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True); ap.add_argument("--percam-out", default=None)
    a = ap.parse_args()
    main(a.res, a.templates, a.fig, a.figcache, percam_out=a.percam_out)
