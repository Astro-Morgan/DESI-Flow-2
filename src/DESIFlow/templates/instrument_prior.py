"""
Empirical prior for the coefficients of the instrument templates, from their distribution on the actual SKY-fibre rows.

Every SKY row is fitted with the fiducial instrument columns (b K=5, r K=6, z K=12, times the camera ramps; 23 coefficients, jointly over the merged grid, exactly the columns of
the joint fit) by weighted least squares, giving a coefficient vector c_i and its estimation covariance N_i = (X W_i X^T)^-1. Three priors are compared:
  P0  zero-mean Gaussian, diagonal, sigma = robust (MAD) scatter                       (the prior used so far)
  P1  Gaussian with the empirical mean and the FULL covariance                         Sigma_true = Sigma_emp - mean(N_i), made positive definite
  P2  P1 conditioned on the observing program of the fibre's tile (dark / bright / backup; other -> all)
Held-out comparison: the mean log predictive density of the coefficient vectors of held-out SKY rows (tiles not used for the prior), log N(c_i; mu, Sigma + N_i).
Row selection: rows with at least half of the pixels good; the 0.5% of the rows with the largest Mahalanobis distance are dropped once when the covariance is estimated.

Run:  python -m DESIFlow.templates.instrument_prior --root ROOT_V2 --cache DR2_CACHE --fig FIG --figcache CACHE
Output: ROOT/instrument_prior.npz (coefficients, programs, priors) and figure instrument_prior.png
"""
import argparse
import os

import numpy as np

from DESIFlow.templates import joint_fit_demo as jd

PROGS = ["dark", "bright", "backup"]


def coefficients(inst, flux, ivar, idx, chunk=400):
    import torch
    X = torch.tensor(inst); out, noise, keep = [], [], []
    for a in range(0, len(idx), chunk):
        ii = idx[a:a + chunk]; f = torch.tensor(np.asarray(flux[ii], np.float64)); w = torch.tensor(np.asarray(ivar[ii], np.float64))
        A = torch.einsum("ml,rl,nl->rmn", X, w, X); A = A + 1e-9 * torch.diagonal(A, dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(len(X), dtype=torch.float64)[None]
        b = torch.einsum("ml,rl->rm", X, w * f); good = (w > 0).sum(1) > 0.5 * w.shape[1]
        N = torch.linalg.inv(A); c = torch.einsum("rmn,rn->rm", N, b)
        out.append(c[good].numpy()); noise.append(N[good].numpy()); keep.append(ii[good.numpy()])
    return np.concatenate(out), np.concatenate(noise), np.concatenate(keep)


def programs_of(cache, rows):
    sky_rows = np.load(os.path.join(cache, "index", "sky_rows.npy")); gs = np.load(os.path.join(cache, "index", "group_start.npy")); prog = np.load(os.path.join(cache, "index", "group_program.npy"))
    return np.array([str(prog[g]).strip() for g in np.searchsorted(gs, sky_rows[rows], side="right") - 1])


def gaussian(c, noise, trim=0.005):
    mu = c.mean(0); S = np.cov(c.T)
    for _ in range(2):
        d = c - mu; m2 = np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d); keep = m2 < np.quantile(m2, 1 - trim); mu = c[keep].mean(0); S = np.cov(c[keep].T)
    S = S - noise.mean(0); ev, V = np.linalg.eigh(S); ev = np.maximum(ev, 1e-3 * ev.max()); S = (V * ev) @ V.T
    return mu, S


def loglik(c, noise, mu, S):
    ll = []
    for ci, Ni in zip(c, noise):
        C = S + Ni; d = ci - mu; sign, ld = np.linalg.slogdet(C); ll.append(-0.5 * (d @ np.linalg.solve(C, d) + ld + len(d) * np.log(2 * np.pi)))
    return np.array(ll)


def build(root, cache, n_fit=30000, n_test=10000):
    fid = jd.Fiducial(root, os.path.join(root, "..", "physical_templates"), "/pscratch/sd/a/andrewmo/V2_Real/RepoClone/Legacy/k32_final.pt") if False else None
    return fid


def run(root, cache, phys_dir, legacy, figdir, cachedir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fid = jd.Fiducial(root, phys_dir, legacy); d = os.path.join(root, "seed0", "instrument")
    ftr = np.load(os.path.join(d, "train_flux.npy"), mmap_mode="r"); wtr = np.load(os.path.join(d, "train_ivar.npy"), mmap_mode="r"); fev = np.load(os.path.join(d, "eval_flux.npy"), mmap_mode="r"); wev = np.load(os.path.join(d, "eval_ivar.npy"), mmap_mode="r")
    dr = os.path.join(root, "..", "sky_templates", "instrument"); rtr = np.load(os.path.join(dr, "train_rows.npz"))["cache_rows"]; rev = np.load(os.path.join(dr, "eval_rows.npz"))["cache_rows"]
    c_fit, n_fit, k_fit = coefficients(fid.inst, ftr, wtr, np.arange(0, len(wtr), 5)); c_te, n_te, k_te = coefficients(fid.inst, fev, wev, np.arange(0, len(wev), 2))
    p_fit = programs_of(cache, rtr[k_fit]); p_te = programs_of(cache, rev[k_te]); print("rows", len(c_fit), len(c_te), {p: int((p_fit == p).sum()) for p in set(p_fit)}, flush=True)
    sig0 = fid.prior_inst; mu1, S1 = gaussian(c_fit, n_fit); pri = {p: gaussian(c_fit[p_fit == p], n_fit[p_fit == p]) for p in PROGS if (p_fit == p).sum() > 300}
    ll0 = loglik(c_te, n_te, np.zeros(c_te.shape[1]), np.diag(sig0 ** 2)); ll1 = loglik(c_te, n_te, mu1, S1)
    ll2 = np.array([loglik(c_te[i:i + 1], n_te[i:i + 1], *pri.get(p_te[i], (mu1, S1)))[0] for i in range(len(c_te))])
    print("held-out mean log predictive density per row: P0 diag zero-mean %.2f | P1 full covariance + mean %.2f | P2 program-conditioned %.2f" % (ll0.mean(), ll1.mean(), ll2.mean()))
    for p in PROGS:
        m = p_te == p
        if m.sum(): print("   ", p, int(m.sum()), "P0 %.2f P1 %.2f P2 %.2f" % (ll0[m].mean(), ll1[m].mean(), ll2[m].mean()))
    # tails: Mahalanobis^2 of the held-out coefficients under P2 against chi2 with 23 dof
    from scipy.stats import chi2 as chi2d
    m2 = np.array([(c_te[i] - pri.get(p_te[i], (mu1, S1))[0]) @ np.linalg.solve(pri.get(p_te[i], (mu1, S1))[1] + n_te[i], c_te[i] - pri.get(p_te[i], (mu1, S1))[0]) for i in range(len(c_te))])
    print("held-out Mahalanobis^2 / dof: median %.2f (1 for a Gaussian prior that fits); fraction above the chi2 99th percentile: %.3f" % (np.median(m2) / chi2d.median(23) * 1.0, (m2 > chi2d.ppf(0.99, 23)).mean()))
    np.savez(os.path.join(root, "instrument_prior.npz"), c_fit=c_fit, p_fit=p_fit, c_test=c_te, p_test=p_te, mu_all=mu1, S_all=S1, sig0=sig0, **{f"mu_{p}": v[0] for p, v in pri.items()}, **{f"S_{p}": v[1] for p, v in pri.items()})
    fig, ax = plt.subplots(2, 3, figsize=(18, 10)); lab = [f"{c}{j + 1}" for c in "brz" for j in range(jd.K_CAM[c])]
    for p, col in zip(PROGS, ("tab:blue", "tab:orange", "tab:red")):
        if p in pri: ax[0, 0].plot(np.sqrt(np.diag(pri[p][1])), "o-", color=col, label=f"{p} (n = {(p_fit == p).sum()})")
    ax[0, 0].plot(sig0, "k--", label="current prior (MAD sigma, all rows)"); ax[0, 0].set_yscale("log"); ax[0, 0].set_xticks(range(23)); ax[0, 0].set_xticklabels(lab, fontsize=6, rotation=90); ax[0, 0].set_ylabel("prior sigma of the coefficient"); ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_title("coefficient scatter by observing program", fontsize=9)
    for p, col in zip(PROGS, ("tab:blue", "tab:orange", "tab:red")):
        if p in pri: ax[0, 1].plot(pri[p][0], "o-", color=col, label=p)
    ax[0, 1].axhline(0, color="gray", lw=0.6); ax[0, 1].set_xticks(range(23)); ax[0, 1].set_xticklabels(lab, fontsize=6, rotation=90); ax[0, 1].set_ylabel("prior mean of the coefficient"); ax[0, 1].legend(fontsize=7); ax[0, 1].set_title("coefficient means (the zero-mean prior ignores these)", fontsize=9)
    Cn = S1 / np.sqrt(np.outer(np.diag(S1), np.diag(S1))); im = ax[0, 2].imshow(Cn, vmin=-1, vmax=1, cmap="RdBu_r"); ax[0, 2].set_title("correlation of the coefficients (all rows)", fontsize=9); fig.colorbar(im, ax=ax[0, 2], shrink=0.8)
    ax[1, 0].bar(range(3), [ll0.mean(), ll1.mean(), ll2.mean()], color=["gray", "tab:blue", "tab:green"]); ax[1, 0].set_xticks(range(3)); ax[1, 0].set_xticklabels(["P0 diagonal\nzero mean", "P1 full cov.\n+ mean", "P2 + program"]); ax[1, 0].set_ylim(min(ll0.mean(), ll1.mean(), ll2.mean()) - 2, max(ll0.mean(), ll1.mean(), ll2.mean()) + 1)
    ax[1, 0].set_ylabel("mean log predictive density of held-out SKY rows"); ax[1, 0].set_title("how well each prior describes held-out SKY rows (higher = better)", fontsize=9)
    ax[1, 1].hist(m2, bins=np.logspace(-0.5, 3, 60), density=True, alpha=0.6, label="held-out Mahalanobis^2 (P2)"); xx = np.logspace(-0.5, 3, 200); ax[1, 1].plot(xx, chi2d.pdf(xx, 23), "k", label="chi2, 23 dof (Gaussian prior that fits)"); ax[1, 1].set_xscale("log"); ax[1, 1].set_xlabel("Mahalanobis distance squared"); ax[1, 1].set_ylabel("density"); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("tails of the coefficient distribution", fontsize=9); ax[1, 2].axis("off")
    fig.tight_layout(); os.makedirs(figdir, exist_ok=True); fig.savefig(os.path.join(figdir, "instrument_prior.png"), dpi=120); np.savez(os.path.join(cachedir, "instrument_prior.npz"), labels=np.array(lab), ll0=ll0, ll1=ll1, ll2=ll2, m2=m2, sig0=sig0, corr=Cn,
                                                                                                                             **{f"sigma_{p}": np.sqrt(np.diag(v[1])) for p, v in pri.items()}, **{f"mean_{p}": v[0] for p, v in pri.items()})


# ------------------------------------------------------------------------------------------------ heavy-tailed (multivariate Student-t) prior
def fit_t(c, noise, nus=(1.5, 2, 3, 4, 6, 10, 30), iters=60):
    """EM fit of a multivariate Student-t (location mu, scatter S, degrees of freedom nu by profile over a grid; the scatter has the mean estimation noise removed)."""
    from scipy.special import gammaln
    n, d = c.shape; best = None
    for nu in nus:
        mu = np.median(c, 0); S = np.diag((1.4826 * np.median(np.abs(c - mu), 0)) ** 2)
        for _ in range(iters):
            m2 = np.einsum("ni,ij,nj->n", c - mu, np.linalg.inv(S), c - mu); u = (nu + d) / (nu + m2); mu = (u[:, None] * c).sum(0) / u.sum(); dd = c - mu; S = (u[:, None, None] * np.einsum("ni,nj->nij", dd, dd)).sum(0) / n
        m2 = np.einsum("ni,ij,nj->n", c - mu, np.linalg.inv(S), c - mu); ll = (gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi) - 0.5 * np.linalg.slogdet(S)[1] - (nu + d) / 2 * np.log1p(m2 / nu)).sum()
        if best is None or ll > best[0]:
            best = (ll, nu, mu, S)
    _, nu, mu, S = best; floor = 1e-3 * np.linalg.eigvalsh(S).max(); S = S - np.median(noise, axis=0)                  # the median estimation noise (the mean is dominated by near-singular rows)
    ev, V = np.linalg.eigh(S); ev = np.maximum(ev, floor); return mu, (V * ev) @ V.T, nu


def loglik_t(c, noise, mu, S, nu):
    from scipy.special import gammaln
    d = c.shape[1]; out = []
    for ci, Ni in zip(c, noise):
        C = S + Ni; r = ci - mu; m2 = r @ np.linalg.solve(C, r); out.append(gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi) - 0.5 * np.linalg.slogdet(C)[1] - (nu + d) / 2 * np.log1p(m2 / nu))
    return np.array(out)


def run_t(root, cache, phys_dir, legacy, figdir, cachedir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(root, "instrument_prior.npz")); c, p, ct, pt = z["c_fit"], z["p_fit"], z["c_test"], z["p_test"]
    fid = jd.Fiducial(root, phys_dir, legacy); d = os.path.join(root, "seed0", "instrument")
    fev = np.load(os.path.join(d, "eval_flux.npy"), mmap_mode="r"); wev = np.load(os.path.join(d, "eval_ivar.npy"), mmap_mode="r"); ftr = np.load(os.path.join(d, "train_flux.npy"), mmap_mode="r"); wtr = np.load(os.path.join(d, "train_ivar.npy"), mmap_mode="r")
    _, n_fit, k_fit = coefficients(fid.inst, ftr, wtr, np.arange(0, len(wtr), 5)); _, n_te, k_te = coefficients(fid.inst, fev, wev, np.arange(0, len(wev), 2))
    sig_all = 1.4826 * np.median(np.abs(c - np.median(c, 0)), 0); priors = {}
    for q in PROGS + ["all"]:
        m = np.ones(len(c), bool) if q == "all" else (p == q)
        if m.sum() < 300: continue
        priors[q] = fit_t(c[m], n_fit[m]); print(q, "n", int(m.sum()), "nu", priors[q][2], "scale b1 %.2f r1 %.2f z1 %.2f" % tuple(np.sqrt(np.diag(priors[q][1]))[[0, 5, 11]]), "median b1 %.2f" % priors[q][0][0], flush=True)
    ll_p0 = loglik(ct, n_te, np.zeros(ct.shape[1]), np.diag(sig_all ** 2)); ll_t = np.array([loglik_t(ct[i:i + 1], n_te[i:i + 1], *priors.get(pt[i], priors["all"]))[0] for i in range(len(ct))]); ll_ta = loglik_t(ct, n_te, *priors["all"])
    print("held-out mean log predictive density: diag Gaussian zero-mean (MAD, all rows) %.2f | Student-t, all rows %.2f | Student-t per program %.2f" % (ll_p0.mean(), ll_ta.mean(), ll_t.mean()))
    for q in PROGS:
        m = pt == q
        if m.sum(): print("   ", q, int(m.sum()), "P0 %.2f t-all %.2f t-program %.2f" % (ll_p0[m].mean(), ll_ta[m].mean(), ll_t[m].mean()))
    np.savez(os.path.join(root, "instrument_prior_t.npz"), sig_all=sig_all, **{f"mu_{q}": v[0] for q, v in priors.items()}, **{f"S_{q}": v[1] for q, v in priors.items()}, **{f"nu_{q}": v[2] for q, v in priors.items()})
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5)); lab = [f"{k}{j + 1}" for k in "brz" for j in range(jd.K_CAM[k])]
    for q, col in zip(PROGS, ("tab:blue", "tab:orange", "tab:red")):
        if q in priors: ax[0].plot(np.sqrt(np.diag(priors[q][1])), "o-", color=col, label=f"{q}: Student-t scale, nu = {priors[q][2]:g}")
    ax[0].plot(sig_all, "k--", label="MAD sigma of all rows"); ax[0].plot(fid.prior_inst, "x:", color="gray", label="prior used so far (first 3000 hold-out rows: backup-dominated)"); ax[0].set_yscale("log"); ax[0].set_xticks(range(23)); ax[0].set_xticklabels(lab, fontsize=6, rotation=90)
    ax[0].set_ylabel("prior scale of the coefficient"); ax[0].legend(fontsize=7); ax[0].set_title("heavy-tailed prior scale by observing program", fontsize=9)
    ax[1].bar(range(3), [ll_p0.mean(), ll_ta.mean(), ll_t.mean()], color=["gray", "tab:blue", "tab:green"]); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["diagonal Gaussian\nzero mean", "Student-t\nall rows", "Student-t\nper program"]); ax[1].set_ylim(min(ll_p0.mean(), ll_t.mean()) - 2, max(ll_p0.mean(), ll_t.mean()) + 2)
    ax[1].set_ylabel("mean log predictive density of held-out SKY rows"); ax[1].set_title("which prior describes held-out SKY rows best (higher = better)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "instrument_prior_t.png"), dpi=120); np.savez(os.path.join(cachedir, "instrument_prior_t.npz"), labels=np.array(lab), ll_p0=ll_p0, ll_t_all=ll_ta, ll_t_prog=ll_t, sig_all=sig_all, old_prior=fid.prior_inst, **{f"scale_{q}": np.sqrt(np.diag(v[1])) for q, v in priors.items()}, **{f"nu_{q}": v[2] for q, v in priors.items()})



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--phys", required=True); ap.add_argument("--legacy", required=True); ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    ap.add_argument("--t", action="store_true"); a = ap.parse_args()
    (run_t if a.t else run)(a.root, a.cache, a.phys, a.legacy, a.fig, a.figcache)
