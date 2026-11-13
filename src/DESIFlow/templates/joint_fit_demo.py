"""
Joint fit of the 32 legacy DeepRock templates with the fiducial instrument (per camera) and sky templates on DR2 spectra, then subtraction of the fitted
instrument and sky parts and a refit of the physical templates.

Model of a spectrum (7781-pixel brz grid, redshift z known from zall):
    f ~ sum_k c_k phi_k(lambda / (1 + z))                      legacy rest-frame templates (32 x 13903, log10 lambda 400..9824 A, 1e-4 dex), linear interpolation
      + sum_cam sum_j d_cj B_cj(lambda) W_cam(lambda)          per-camera instrument templates (b K=5, r K=6, z K=12; long-trained) times the camera weights (ramps in the overlaps)
      + sum_j s_j S_j(lambda)                                  sky templates (K=64, long-trained)
All coefficients by weighted least squares with Gaussian priors on the instrument and sky coefficients (instrument: robust scatter of the coefficients on held-out SKY
rows; sky: SKY_FRAC = 1% of the scatter of the coefficients on VAC sky spectra) and a negligible ridge on the physical ones.

Reported per object: energy of every component E_c = sum omega m_c^2 (omega = ivar; chi2 units, per pixel it is the mean chi2 the component carries), shares,
chi2 of physical-only / +instrument / +sky / joint fits, the cleaned flux f - m_inst - m_sky, the change of the physical model between the physical-only fit to the raw flux
and the physical refit to the cleaned flux. (The refit to the cleaned flux reproduces the joint physical coefficients exactly: the joint normal equations for c are the
normal equations of the refit; the comparison that carries information is raw-physical-only against cleaned.)

Run:  python -m DESIFlow.templates.joint_fit_demo --root ROOT_V2 --phys DATA_DIR --cache DR2_CACHE --legacy K32.pt --fig FIG --figcache CACHE
"""
import argparse
import json
import os

import numpy as np

L = 7781
LAM = 3600.0 + 0.8 * np.arange(L)
RX0, RD = np.log10(400.0), 1e-4
CAMS = {"b": (0, 2751), "r": (2700, 5026), "z": (4900, 7781)}
K_CAM = {"b": 5, "r": 6, "z": 12}
K_SKY = 64
SKY_FRAC = 0.01


def nw(f, w, sig):
    h = int(4 * sig); k = np.exp(-0.5 * (np.arange(-h, h + 1) / sig) ** 2)
    num = np.convolve(w * f, k, "same"); den = np.convolve(w, k, "same")
    return np.where(den > 1e-12 * max(den.max(), 1e-30), num / np.where(den > 0, den, 1), np.nan)


def camera_weights():
    l = np.arange(L, dtype=np.float64); wb = np.clip((2750.0 - l) / 50.0, 0, 1); wz = np.clip((l - 4900.0) / 125.0, 0, 1)
    return {"b": wb, "r": 1.0 - wb - wz, "z": wz}


class Fiducial:
    def __init__(self, root, phys_dir, legacy_pt, use_R=False):
        import torch
        self.T = torch.load(legacy_pt, map_location="cpu", weights_only=False)["templates"].numpy().astype(np.float64); self.N = self.T.shape[1]
        self.Rm = np.load(os.path.join(phys_dir, "R_mean.npy")).astype(np.float64) if use_R else None
        W = camera_weights(); cols, kinds = [], []
        for c in "brz":
            K = K_CAM[c]; t = np.load(os.path.join(root, "long", "seed0", c, f"K{K:03d}.npz"))["templates_flux"].astype(np.float64); a0, a1 = CAMS[c]
            for k in range(K):
                v = np.zeros(L); v[a0:a1] = t[k]; cols.append(v * W[c]); kinds.append(f"inst_{c}")
        self.inst = np.array(cols); self.inst_kind = np.array(kinds)
        self.sky = np.load(os.path.join(root, "long", "seed0", "sky", f"K{K_SKY:03d}.npz"))["templates_flux"].astype(np.float64)
        self.prior_inst, self.prior_sky = self._priors(root)
        self.t_priors, self.t_center, self.t_scale = {}, {}, {}                    # empirical Student-t prior of the instrument coefficients per observing program (instrument_prior.py --t)
        tp = os.path.join(root, "instrument_prior_t.npz")
        if os.path.exists(tp):
            zt = np.load(tp)
            for q in ("dark", "bright", "backup", "all"):
                if f"mu_{q}" in zt.files:
                    self.t_priors[q] = (zt[f"mu_{q}"], np.linalg.inv(zt[f"S_{q}"]), float(zt[f"nu_{q}"])); self.t_center[q] = zt[f"mu_{q}"]; self.t_scale[q] = np.sqrt(np.diag(zt[f"S_{q}"]))

    def center_scale(self, prog):
        """location and scale of the instrument-coefficient prior of a program (the Student-t fit; falls back to zero and the old sigma)."""
        if USE_T and self.t_priors:
            q = prog if prog in self.t_priors else "all"; return self.t_center[q], self.t_scale[q]
        return np.zeros(len(self.prior_inst)), self.prior_inst

    def zscore(self, theta_inst, prog):
        mu, sc = self.center_scale(prog); return np.abs(theta_inst - mu) / sc

    def _priors(self, root):
        import torch
        d = os.path.join(root, "seed0", "instrument"); f = np.load(os.path.join(d, "eval_flux.npy"), mmap_mode="r"); w = np.load(os.path.join(d, "eval_ivar.npy"), mmap_mode="r")
        sig = np.zeros(len(self.inst))
        for c in "brz":
            a0, a1 = CAMS[c]; idx = np.flatnonzero(self.inst_kind == f"inst_{c}"); X = torch.tensor(self.inst[idx][:, a0:a1])
            fc = torch.tensor(np.asarray(f[:3000, a0:a1], np.float64)); wc = torch.tensor(np.asarray(w[:3000, a0:a1], np.float64)); keep = (wc > 0).sum(1) > 0.5 * (a1 - a0)
            A = torch.einsum("ml,rl,nl->rmn", X, wc, X) + 1e-9 * torch.eye(len(idx), dtype=torch.float64); b = torch.einsum("ml,rl->rm", X, wc * fc)
            co = torch.linalg.solve(A[keep], b[keep][..., None])[..., 0].numpy(); sig[idx] = 1.4826 * np.median(np.abs(co - np.median(co, 0)), 0)
        ds = os.path.join(root, "seed0", "sky"); fs = np.load(os.path.join(ds, "train_flux.npy"), mmap_mode="r")[:1500]; ws = np.load(os.path.join(ds, "train_ivar.npy"), mmap_mode="r")[:1500]
        X = torch.tensor(self.sky); cs = []
        for a in range(0, 1500, 250):
            fc = torch.tensor(np.asarray(fs[a:a + 250], np.float64)); wc = torch.tensor(np.asarray(ws[a:a + 250], np.float64))
            A = torch.einsum("ml,rl,nl->rmn", X, wc, X) + 1e-9 * torch.eye(len(X), dtype=torch.float64); cs.append(torch.linalg.solve(A, torch.einsum("ml,rl->rm", X, wc * fc)[..., None])[..., 0].numpy())
        return sig, SKY_FRAC * np.concatenate(cs).std(0)

    def phys(self, z):
        rl = np.log10(LAM / (1 + z)); p = (rl - RX0) / RD; ok = (p >= 0) & (p <= self.N - 1); i0 = np.clip(np.floor(p).astype(int), 0, self.N - 2); fr = np.clip(p - i0, 0, 1)
        X = (self.T[:, i0] * (1 - fr) + self.T[:, i0 + 1] * fr) * ok
        if self.Rm is not None:
            Xp = np.pad(X, ((0, 0), (5, 5))); X = sum(self.Rm[d][None, :] * Xp[:, d:d + L] for d in range(11))
        return X

    def columns(self, z, parts=("phys", "inst", "sky"), sky_scale=1.0):
        X, prec, tag = [], [], []
        P = self.phys(z) if "phys" in parts else None
        if P is not None:
            X.append(P); prec.append(np.full(len(P), 0.0)); tag += ["phys"] * len(P)
        if "inst" in parts:
            X.append(self.inst); prec.append(1.0 / np.maximum(self.prior_inst, 1e-12) ** 2); tag += ["inst"] * len(self.inst)
        if "sky" in parts:
            X.append(self.sky); prec.append(1.0 / (np.maximum(self.prior_sky, 1e-12) * sky_scale) ** 2); tag += ["sky"] * len(self.sky)
        return np.vstack(X), np.concatenate(prec), np.array(tag)


USE_T = True           # instrument-coefficient prior: the empirical Student-t per observing program (False: the old zero-mean diagonal Gaussian)
NONNEG = True          # standing rule (user, 2026-11-12): the cumulative physical model must be >= 0 at every pixel


def _solve(A, b, X, tag, nonneg):
    if not nonneg or not (tag == "phys").any():
        th = np.linalg.solve(A, b); return th, th @ X
    ev, V = np.linalg.eigh(A); Ah = (V / np.sqrt(np.maximum(ev, 1e-300))) @ V.T                 # A^-1/2
    th0 = np.linalg.solve(A, b); mod0 = th0[tag == "phys"] @ X[tag == "phys"]
    if mod0.min() >= 0:
        return th0, th0 @ X
    from scipy.optimize import nnls
    Gt = np.zeros((len(tag), X.shape[1])); Gt[tag == "phys"] = X[tag == "phys"]
    lam, _ = nnls(Ah @ Gt, -(Ah @ b), maxiter=20000)
    th = np.linalg.solve(A, b + Gt @ lam); return th, th @ X


def wls(X, prec, tag, f, w, ridge_phys=1e-6, nonneg=None, inst_prior=None, irls=10):
    """regularised weighted least squares; with nonneg the cumulative physical model sum_k c_k phi_k is constrained to be >= 0 at every pixel.
    The inequality-constrained QP  min 1/2 th'A th - b'th  s.t.  G th >= 0  is solved through its dual, a non-negative least squares problem:
    th = A^-1 (b + G' lam), lam = argmin_{lam >= 0} |A^-1/2 (b + G' lam)|^2  (scipy nnls).
    inst_prior = (mu, Sinv, nu): a multivariate Student-t prior on the instrument coefficients (location mu, inverse scatter Sinv, degrees of freedom nu; nu = inf is a Gaussian),
    fitted by iteratively reweighted least squares (scale mixture: the prior precision is u Sinv with u = (nu + d) / (nu + Mahalanobis^2) updated from the current coefficients)."""
    nonneg = NONNEG if nonneg is None else nonneg
    Xw = X * w[None]; A = Xw @ X.T; d = np.diag(A).copy(); pp = prec.copy(); pp[tag == "phys"] = ridge_phys * d[tag == "phys"]
    if inst_prior is not None:
        pp[tag == "inst"] = 0.0
    A[np.diag_indices_from(A)] += pp; b = Xw @ f
    if inst_prior is None:
        return _solve(A, b, X, tag, nonneg)
    mu, Sinv, nu = inst_prior; ii = np.flatnonzero(tag == "inst"); u = 1.0
    for _ in range(irls if np.isfinite(nu) else 1):
        A2 = A.copy(); b2 = b.copy(); A2[np.ix_(ii, ii)] += u * Sinv; b2[ii] += u * (Sinv @ mu); th, m = _solve(A2, b2, X, tag, nonneg)
        if np.isfinite(nu):
            r = th[ii] - mu; u = (nu + len(ii)) / (nu + r @ Sinv @ r)
    return th, m


def fit_object(fid, z, f, w, nonneg=None, inst_prior=None, prog=None, parts_list=(("phys",), ("phys", "inst"), ("phys", "sky"), ("phys", "inst", "sky")), hidden=None, sky_scale=1.0):
    """fits for each part set; returns dict parts -> (theta, tag, model, per-tag models). hidden: bool mask excluded from the fit."""
    if inst_prior is None and prog is not None and USE_T and fid.t_priors:
        inst_prior = fid.t_priors.get(prog, fid.t_priors["all"])
    wf = w * (~hidden if hidden is not None else 1.0); out = {}
    for parts in parts_list:
        X, prec, tag = fid.columns(z, parts, sky_scale); th, m = wls(X, prec, tag, f, wf, nonneg=nonneg, inst_prior=inst_prior if "inst" in parts else None)
        out[parts] = dict(theta=th, tag=tag, model=m, comp={t: th[tag == t] @ X[tag == t] for t in set(tag)}, X=X)
    return out


def chi2(f, m, w, mask=None):
    sel = (w > 0) if mask is None else ((w > 0) & mask)
    return float((w * (f - m) ** 2)[sel].sum()), int(sel.sum())


def programs_of_cache_rows(cache, rows):
    """observing program (dark / bright / backup / other) of the tiles of target cache rows."""
    tr = np.load(os.path.join(cache, "index", "tgt_rows.npy")); gs = np.load(os.path.join(cache, "index", "group_start.npy")); pg = np.load(os.path.join(cache, "index", "group_program.npy"))
    return np.array([str(pg[g]).strip() for g in np.searchsorted(gs, tr[np.asarray(rows)], side="right") - 1])


def load_objects(phys_dir, cache, n_pop=300, seed=1, chi_max=1.3):
    """held-out galaxies of the physical-template experiment with a Redrock chi2/px < chi_max (the 5% with a bad Redrock fit are mostly mislabelled stars: the whole S/N > 30 end
    of the sample fails it): showcase (top-S/N of five z bins + one typical), a random population, and one deliberately kept contaminant (the highest-S/N spectrum with chi2/px > 5)."""
    f = np.load(os.path.join(phys_dir, "eval_flux.npy"), mmap_mode="r"); w = np.load(os.path.join(phys_dir, "eval_ivar.npy"), mmap_mode="r"); m = np.load(os.path.join(phys_dir, "eval_meta.npz"))
    zz = np.load(os.path.join(phys_dir, "eval_zall.npz")); rr = zz["chi2_per_pix"]; z, snr = m["z"], m["snr_med"]; rng = np.random.default_rng(seed); clean = rr < chi_max; show = []
    for lo, hi in [(0.02, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
        s = np.flatnonzero((z >= lo) & (z < hi) & clean); show.append(int(s[np.argsort(snr[s])[-1]]))
    mid = np.flatnonzero((z > 0.2) & (z < 0.8) & clean & (snr > np.percentile(snr[clean], 40)) & (snr < np.percentile(snr[clean], 60))); show.append(int(rng.choice(mid)))
    pop = rng.choice(np.flatnonzero(clean), n_pop, replace=False); bad = np.flatnonzero(rr > 5); con = int(bad[np.argmax(snr[bad])])
    return f, w, z, snr, m["cache_row"], show, pop, con, rr


def load_qsos(cache, n=2, seed=0):
    import fitsio
    from DESIFlow.data.dr2_cache import load_index, ZALL
    idx = load_index(cache); snr = np.asarray(np.load(os.path.join(cache, "tgt", "snr_med.npy"), mmap_mode="r")); st = np.asarray(np.load(os.path.join(cache, "tgt", "status.npy"), mmap_mode="r"))
    fst = np.asarray(idx["fiberstatus"])[np.asarray(idx["tgt_rows"])]; ok = np.flatnonzero((st == 1) & np.isfinite(snr) & (snr > 3) & (fst == 0)); rng = np.random.default_rng(seed)
    cand = np.sort(rng.choice(ok, 300000, replace=False)); zr = np.asarray(idx["tgt_rows"])[cand]
    d = fitsio.FITS(ZALL)[1].read(columns=["Z", "SPECTYPE", "ZWARN", "DELTACHI2", "CHI2"], rows=zr); good = (np.char.strip(d["SPECTYPE"].astype(str)) == "QSO") & (d["ZWARN"] == 0) & (d["DELTACHI2"] > 200) & (d["Z"] > 1.0) & (d["Z"] < 2.5)
    rrq = d["CHI2"] / (7958.0 * 0.97); good &= rrq < 2.5
    s = np.flatnonzero(good); pick = s[np.argsort(snr[cand[s]])[-n:]]
    flux, ivar, mask = (np.load(os.path.join(cache, "tgt", f"{k}.npy"), mmap_mode="r") for k in ("flux", "ivar", "mask")); rows = cand[pick]
    return [dict(f=np.asarray(flux[r], np.float64), w=np.asarray(np.where(mask[r] == 0, ivar[r], 0.0), np.float64), z=float(d["Z"][i]), snr=float(snr[r]), kind="QSO", rr=float(rrq[i]), prog=programs_of_cache_rows(cache, [r])[0]) for r, i in zip(rows, pick)]


def run(root, phys_dir, cache, legacy, figdir, cachedir, use_R=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from DESIFlow.data.target_nuisance_test import span_mask
    fid = Fiducial(root, phys_dir, legacy, use_R=use_R); F, W, Z, SNR, CR, show, pop, con, RR = load_objects(phys_dir, cache); PR = programs_of_cache_rows(cache, CR)
    objs = [dict(f=np.asarray(F[i], np.float64), w=np.asarray(W[i], np.float64), z=float(Z[i]), snr=float(SNR[i]), kind="galaxy", rr=float(RR[i]), prog=PR[i]) for i in show] + load_qsos(cache) \
        + [dict(f=np.asarray(F[con], np.float64), w=np.asarray(W[con], np.float64), z=float(Z[con]), snr=float(SNR[con]), kind="contaminant", rr=float(RR[con]), prog=PR[con])]
    res, cache_out = [], {}
    for n, o in enumerate(objs):
        fits = fit_object(fid, o["z"], o["f"], o["w"], prog=o["prog"]); j = fits[("phys", "inst", "sky")]; p = fits[("phys",)]; good = o["w"] > 0
        E = {c: float((o["w"] * j["comp"][c] ** 2).sum()) for c in ("phys", "inst", "sky")}; Ed = float((o["w"] * o["f"] ** 2).sum()); npx = int(good.sum())
        chis = {"phys": chi2(o["f"], p["model"], o["w"])[0], "phys+inst": chi2(o["f"], fits[("phys", "inst")]["model"], o["w"])[0], "phys+sky": chi2(o["f"], fits[("phys", "sky")]["model"], o["w"])[0], "joint": chi2(o["f"], j["model"], o["w"])[0]}
        clean = o["f"] - j["comp"]["inst"] - j["comp"]["sky"]
        refit = fit_object(fid, o["z"], clean, o["w"], prog=o["prog"], parts_list=(("phys",),))[("phys",)]
        dm = float(np.sqrt((o["w"] * (refit["comp"]["phys"] - p["comp"]["phys"]) ** 2).sum() / (o["w"] * p["comp"]["phys"] ** 2).sum()))
        ident = float(np.abs(refit["theta"] - j["theta"][j["tag"] == "phys"]).max() / np.abs(j["theta"][j["tag"] == "phys"]).max())
        res.append(dict(kind=o["kind"], z=o["z"], snr=o["snr"], npix=npx, energy_per_pix={c: E[c] / npx for c in E}, data_energy_per_pix=Ed / npx, share_of_model_energy={c: E[c] / sum(E.values()) for c in E},
                        chi2_per_pix={k: v / npx for k, v in chis.items()}, physical_model_change=dm, refit_equals_joint_maxrel=ident))
        cache_out[f"o{n}"] = dict(z=o["z"], snr=o["snr"], f=o["f"], w=o["w"], joint=j["model"], phys_only=p["model"], phys_joint=j["comp"]["phys"], inst=j["comp"]["inst"], sky=j["comp"]["sky"], clean=clean, refit=refit["comp"]["phys"])
        print(n, o["kind"], f"z={o['z']:.3f} S/N={o['snr']:.1f}", {c: round(res[-1]["share_of_model_energy"][c], 3) for c in E}, "chi2/px", {k: round(v, 3) for k, v in res[-1]["chi2_per_pix"].items()}, "dmodel", round(dm, 3), "ident", f"{ident:.1e}", flush=True)
    # population (hidden-pixel test, energy shares)
    P = []
    for i in pop:
        f, w, z = np.asarray(F[i], np.float64), np.asarray(W[i], np.float64), float(Z[i])
        if (w > 0).sum() < 4000: continue
        fits = fit_object(fid, z, f, w, prog=PR[i]); j = fits[("phys", "inst", "sky")]; E = {c: float((w * j["comp"][c] ** 2).sum()) for c in ("phys", "inst", "sky")}
        hid = span_mask(L, np.random.default_rng(int(i))); ft = fit_object(fid, z, f, w, hidden=hid, prog=PR[i]); sel = hid & (w > 0)
        h = {"+".join(k): float((w * (f - v["model"]) ** 2)[sel].sum() / max(sel.sum(), 1)) for k, v in ft.items()}
        p0 = fits[("phys",)]["comp"]["phys"]; dm = float(np.sqrt((w * (j["comp"]["phys"] - p0) ** 2).sum() / (w * p0 ** 2).sum()))
        P.append(dict(snr=float(SNR[i]), z=z, shares=[E[c] / sum(E.values()) for c in ("phys", "inst", "sky")], per_pix=[E[c] / (w > 0).sum() for c in ("phys", "inst", "sky")], hidden=h, dmodel=dm))
    json.dump(dict(objects=res, population_n=len(P)), open(os.path.join(root, "joint_fit_demo.json"), "w"), indent=1)
    return fid, objs, cache_out, res, P, show


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True); ap.add_argument("--phys", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--legacy", required=True)
    ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True); ap.add_argument("--bounded", action="store_true"); ap.add_argument("--no-nonneg", action="store_true"); ap.add_argument("--old-prior", action="store_true")
    a = ap.parse_args()
    from DESIFlow.templates import joint_fit_demo as _jd
    if a.no_nonneg:
        _jd.NONNEG = False
    if a.old_prior:
        _jd.USE_T = False
    from DESIFlow.templates import joint_fit_figures as figs
    if a.bounded:
        figs.make_bounded(a.root, a.phys, a.cache, a.legacy, a.fig, a.figcache); raise SystemExit
    figs.make(a.root, a.phys, a.cache, a.legacy, a.fig, a.figcache)
