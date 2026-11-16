"""
Calibrated prior for the sky-template coefficients of the joint fit (the fiducial K = 64 sky templates).

Two data sets: the DR1 sky-spectrum VAC (9176 sky spectra, with the tile, so the observing program, and the sky brightness) gives the coefficient distribution of the sky ITSELF; the SKY-fibre residual rows
(tile-split; the same rows as the instrument prior) give the coefficients of the unmodelled sky that is left after the sky subtraction (jointly projected on the 23 instrument + 64 sky columns, sky block kept).
Candidate priors for the SKY-row sky coefficients, calibrated on training rows and scored by the held-out mean log predictive density (log N or t of c_i with scatter S + N_i, N_i the estimation covariance):
  A  the prior used so far: zero-mean diagonal Gaussian, sigma = 1% of the std of the VAC coefficients (all rows)
  B  zero-mean diagonal Gaussian, sigma = eps_program x std of the VAC coefficients of the program, eps_program calibrated
  C  Student-t, scatter eps_program^2 x the covariance of the VAC coefficients of the program (shape from the VAC, amplitude from the SKY rows)
  D  Student-t fitted directly to the sky coefficients of the SKY rows of the program (EM, full 64 x 64 scatter)

Run:  python -m DESIFlow.templates.sky_prior --root ROOT_V2 --cache DR2_CACHE --fig FIG --figcache CACHE
"""
import argparse
import os

import numpy as np

from DESIFlow.templates import instrument_prior as ip
from DESIFlow.templates.fiducial import load_fiducial

VAC = "/global/cfs/cdirs/desi/public/dr1/vac/dr1/skyspec/v1.0/sky_spectra_vac_v1.fits"
PROGS = ["dark", "bright", "backup"]


def tile_program_map(cache, root):
    """tile -> observing program from the SKY rows (tile from the negative TARGETID, program from the row's group)."""
    d = os.path.join(root, "..", "sky_templates", "instrument"); out = {}
    for nm in ("train", "eval"):
        z = np.load(os.path.join(d, f"{nm}_rows.npz")); prog = ip.programs_of(cache, z["cache_rows"])
        for t, p in zip(z["tile"], prog):
            out.setdefault(int(t), p)
    return out


def vac_coefficients(fid_sky, tmap):
    import fitsio
    import torch
    F = fitsio.FITS(VAC); flux = F["FLUX"].read().astype(np.float64); meta = F["METADATA"].read(); fin = np.isfinite(flux); keep = fin.mean(1) > 0.9
    f = np.where(fin, flux, 0.0); w = np.where(fin, 1.0 / (0.1 ** 2 + (0.03 * np.abs(f)) ** 2), 0.0)
    c, N, k = ip.coefficients(fid_sky, f[keep], w[keep], np.arange(keep.sum()))
    idx = np.flatnonzero(keep)[k]; tile = meta["TILEID"][idx]; prog = np.array([tmap.get(int(t), "unknown") for t in tile])
    return c, prog, dict(night=meta["NIGHT"][idx], skymag_r=meta["SKY_MAG_R_SPEC"][idx], moonfrac=meta["MOONFRAC"][idx], tile=tile)


def run(root, cache, figdir, cachedir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = load_fiducial(); X = np.vstack([F.inst, F.sky]); ni = len(F.inst); d = os.path.join(root, "seed0", "instrument"); dr = os.path.join(root, "..", "sky_templates", "instrument")
    ftr = np.load(os.path.join(d, "train_flux.npy"), mmap_mode="r"); wtr = np.load(os.path.join(d, "train_ivar.npy"), mmap_mode="r"); fev = np.load(os.path.join(d, "eval_flux.npy"), mmap_mode="r"); wev = np.load(os.path.join(d, "eval_ivar.npy"), mmap_mode="r")
    ctr, ntr, ktr = ip.coefficients(X, ftr, wtr, np.arange(0, len(wtr), 5)); cte, nte, kte = ip.coefficients(X, fev, wev, np.arange(0, len(wev), 2))
    ptr = ip.programs_of(cache, np.load(os.path.join(dr, "train_rows.npz"))["cache_rows"][ktr]); pte = ip.programs_of(cache, np.load(os.path.join(dr, "eval_rows.npz"))["cache_rows"][kte])
    cs_tr, ns_tr, cs_te, ns_te = ctr[:, ni:], ntr[:, ni:, ni:], cte[:, ni:], nte[:, ni:, ni:]; print("SKY rows", len(cs_tr), len(cs_te), {p: int((ptr == p).sum()) for p in set(ptr)}, flush=True)
    tmap = tile_program_map(cache, root); cv, pv, mv = vac_coefficients(F.sky, tmap); print("VAC spectra", len(cv), {p: int((pv == p).sum()) for p in set(pv)}, flush=True)
    sig_all = cv.std(0); res = {}
    from scipy.special import gammaln
    def ll_gauss(c, N, mu, Sd):
        out = []
        for ci, Ni in zip(c, N):
            C = np.diag(Sd) + Ni; r = ci - mu; out.append(-0.5 * (r @ np.linalg.solve(C, r) + np.linalg.slogdet(C)[1] + len(r) * np.log(2 * np.pi)))
        return np.array(out)
    summary = {}
    for q in PROGS:
        mtr, mte = ptr == q, pte == q; mv_ = pv == q
        if mtr.sum() < 300 or mte.sum() < 100 or mv_.sum() < 50:
            print(q, "skipped: SKY rows", int(mtr.sum()), int(mte.sum()), "VAC", int(mv_.sum())); continue
        sig_q = cv[mv_].std(0); Svac = np.cov(cv[mv_].T); ev, V = np.linalg.eigh(Svac); Svac = (V * np.maximum(ev, 1e-6 * ev.max())) @ V.T
        lA = ll_gauss(cs_te[mte], ns_te[mte], np.zeros(64), (0.01 * sig_all) ** 2).mean()
        eps = np.logspace(-4, -0.5, 36); lt = [ll_gauss(cs_tr[mtr][::6], ns_tr[mtr][::6], np.zeros(64), (e * sig_q) ** 2).mean() for e in eps]; eB = eps[int(np.argmax(lt))]
        lB = ll_gauss(cs_te[mte], ns_te[mte], np.zeros(64), (eB * sig_q) ** 2).mean()
        best = None
        for e in np.logspace(-3.5, -0.5, 13):
            for nu in (1.5, 3, 10, 1e3):
                v = ip.loglik_t(cs_tr[mtr][::6], ns_tr[mtr][::6], np.zeros(64), e ** 2 * Svac, nu).mean()
                if best is None or v > best[0]:
                    best = (v, e, nu)
        _, eC, nuC = best; lC = ip.loglik_t(cs_te[mte], ns_te[mte], np.zeros(64), eC ** 2 * Svac, nuC).mean()
        muD, SD, nuD = ip.fit_t(cs_tr[mtr], ns_tr[mtr]); lD = ip.loglik_t(cs_te[mte], ns_te[mte], muD, SD, nuD).mean()
        summary[q] = dict(n_sky_train=int(mtr.sum()), n_sky_test=int(mte.sum()), n_vac=int(mv_.sum()), A=lA, B=lB, C=lC, D=lD, eps_B=float(eB), eps_C=float(eC), nu_C=float(nuC), nu_D=float(nuD),
                          vac_median_c1=float(np.median(cv[mv_][:, 0])), vac_std_c1=float(sig_q[0]), sky_std_c1=float(1.4826 * np.median(np.abs(cs_tr[mtr][:, 0] - np.median(cs_tr[mtr][:, 0])))), skymag_r_median=float(np.median(mv["skymag_r"][mv_])))
        res[q] = dict(muD=muD, SD=SD, nuD=nuD, Svac=Svac, eC=eC, nuC=nuC, eB=eB, sig_q=sig_q)
        print(f"{q:7s} held-out mean log density: A (1% diag, all VAC) {lA:9.2f} | B (eps {eB:.4f} x VAC std) {lB:9.2f} | C (t, eps {eC:.4f} x VAC covariance, nu {nuC:g}) {lC:9.2f} | D (t fitted to SKY rows, nu {nuD:g}) {lD:9.2f}", flush=True)
        print(f"        sky-r magnitude of the VAC spectra of this program: median {summary[q]['skymag_r_median']:.2f}; coefficient 1: VAC median {summary[q]['vac_median_c1']:.3g}, VAC std {summary[q]['vac_std_c1']:.3g}, SKY-row robust sigma {summary[q]['sky_std_c1']:.3g}")
    import json
    json.dump(summary, open(os.path.join(root, "sky_prior.json"), "w"), indent=1)
    np.savez(os.path.join(root, "sky_prior.npz"), sig_all=sig_all, **{f"{k}_{q}": v for q, r in res.items() for k, v in r.items() if not np.isscalar(v)}, **{f"{k}_{q}": np.array(v) for q, r in res.items() for k, v in r.items() if np.isscalar(v)})
    figure(root, figdir, cachedir)


def figure(root, figdir, cachedir):
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = json.load(open(os.path.join(root, "sky_prior.json"))); z = np.load(os.path.join(root, "sky_prior.npz")); sig_all = z["sig_all"]; qs = list(summary)
    res = {q: dict(SD=z[f"SD_{q}"], nuD=float(z[f"nuD_{q}"]), Svac=z[f"Svac_{q}"], eC=float(z[f"eC_{q}"]), sig_q=z[f"sig_q_{q}"]) for q in qs}
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    for k, nm, col in (("A", "A: 1% diagonal (so far)", "gray"), ("B", "B: calibrated diagonal", "tab:blue"), ("C", "C: t, VAC covariance x eps^2", "tab:green"), ("D", "D: t fitted to the SKY rows", "tab:red")):
        ax[0].plot(range(len(qs)), [summary[q][k] for q in qs], "o-", color=col, label=nm)
    ax[0].set_xticks(range(len(qs))); ax[0].set_xticklabels(qs); ax[0].set_ylabel("held-out mean log predictive density of the SKY-row sky coefficients"); ax[0].legend(fontsize=8); ax[0].set_title("which prior describes the unmodelled sky of held-out SKY rows", fontsize=9)
    for q, col in zip(qs, ("tab:blue", "tab:orange", "tab:red")):
        ax[1].plot(np.sqrt(np.diag(res[q]["SD"])), "o-", ms=3, color=col, label=f"{q}: t fitted to SKY rows (nu {res[q]['nuD']:g})"); ax[1].plot(res[q]["sig_q"] * res[q]["eC"], ":", color=col)
    ax[1].plot(0.01 * sig_all, "k--", label="1% of the VAC std (so far)"); ax[1].set_yscale("log"); ax[1].set_xlabel("sky template number"); ax[1].set_ylabel("prior scale of the coefficient"); ax[1].legend(fontsize=7); ax[1].set_title("scale by template (dotted: eps x VAC std)", fontsize=9)
    for q, col in zip(qs, ("tab:blue", "tab:orange", "tab:red")):
        ax[2].scatter(np.sqrt(np.diag(res[q]["Svac"])), np.sqrt(np.diag(res[q]["SD"])), s=8, color=col, label=f"{q}: eps = {res[q]['eC']:.4f}")
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("VAC coefficient std of the program (the sky itself)"); ax[2].set_ylabel("SKY-row residual scale of the same template"); ax[2].plot([1e-1, 1e4], [1e-3, 1e2], color="gray", ls=":"); ax[2].legend(fontsize=8)
    ax[2].set_title("residual scale against the scale of the sky (dotted: 1%)", fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(figdir, "sky_prior_calibration.png"), dpi=120)
    np.savez(os.path.join(cachedir, "sky_prior_calibration.npz"), programs=np.array(qs), **{f"{k}_{q}": summary[q][k] for q in qs for k in ("A", "B", "C", "D")}, **{f"sigD_{q}": np.sqrt(np.diag(res[q]["SD"])) for q in qs}, **{f"vacstd_{q}": res[q]["sig_q"] for q in qs}, sig_all=sig_all)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    ap.add_argument("--figure-only", action="store_true"); a = ap.parse_args()
    figure(a.root, a.fig, a.figcache) if a.figure_only else run(a.root, a.cache, a.fig, a.figcache)
