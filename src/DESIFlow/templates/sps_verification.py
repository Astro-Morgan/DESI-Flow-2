"""
Independent verification that the instrument templates remove real calibration / sky-subtraction error: a stellar-population-synthesis (SPS) fit.

Physical model: the fastspecfit SPS templates (FSPS, C3K + MIST, Chabrier IMF; 5 age bins at solar metallicity, version 3.0.1;
/global/cfs/cdirs/desi/science/gqp/templates/fastspecfit/3.0.1) with a Calzetti dust screen and a velocity dispersion, plus Gaussian emission lines; all amplitudes >= 0. This model
knows nothing about the legacy DeepRock templates, Redrock's templates or the SKY rows. Variants fitted to each clean galaxy (catalogue z, Redrock chi2/px < 1.3):
  V0  SPS only
  V1  SPS + the fiducial instrument templates (Student-t prior of the galaxy's observing program, fitted jointly) + sky templates
  V2  control: V1 with every instrument and sky template reversed in wavelength within its camera (same smoothness, amplitude statistics and prior, no calibration meaning)
Scores: in-sample chi2/px; hidden-pixel chi2/px (36 hidden spans, fitted on the visible pixels; two draws); the dust and dispersion grid is chosen on V0.
Cross-model check: the instrument component of V1 against the instrument component of the legacy-template joint fit of the same galaxy (a different physical model with different mismatch):
correlation of the two components across pixels, against the same statistic between different galaxies.

Run:  python -m DESIFlow.templates.sps_verification --root ROOT_V2 --phys PHYS --cache CACHE --legacy K32.pt --fig FIG --figcache CACHE [--n 200 --workers 12]
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
from scipy.optimize import nnls

from DESIFlow.templates import joint_fit_demo as jd

TPL = "/global/cfs/cdirs/desi/science/gqp/templates/fastspecfit/3.0.1/ftemplates-chabrier-3.0.1.fits"
LINES = [("[OII]3727", 3727.09), ("[OII]3729", 3729.88), ("[NeIII]", 3868.76), ("Heps", 3970.07), ("Hdelta", 4101.74), ("Hgamma", 4340.47), ("Hbeta", 4861.33), ("[OIII]4959", 4958.91), ("[OIII]5007", 5006.84),
         ("HeI", 5875.6), ("[OI]", 6300.30), ("[NII]6548", 6548.05), ("Halpha", 6562.80), ("[NII]6584", 6583.45), ("[SII]6716", 6716.44), ("[SII]6731", 6730.82)]
AV = [0.0, 0.15, 0.3, 0.5, 0.8, 1.2]
SIGV = [120.0, 220.0]
_W = {}


def calzetti(lam_A, av):
    x = 1.0e4 / lam_A; k = np.where(lam_A < 6300, 2.659 * (-2.156 + 1.509 * x - 0.198 * x ** 2 + 0.011 * x ** 3) + 4.05, 2.659 * (-1.857 + 1.040 * x) + 4.05)
    return 10 ** (-0.4 * np.maximum(k, 0) * av / 4.05)


class SPS:
    def __init__(self, Rm):
        import fitsio
        F = fitsio.FITS(TPL); w = F["WAVE"].read(); fl = F["FLUX"].read().T                       # (5, nwave)
        lg = np.arange(np.log10(1200.0), np.log10(12000.0), 2.0e-5); self.lg = lg; self.lib = {}
        base = np.array([np.interp(lg, np.log10(w), f) for f in fl]); base /= np.median(base[:, (10 ** lg > 4000) & (10 ** lg < 6000)], axis=1)[:, None]
        for s in SIGV:
            sig = s / (3e5 * np.log(10)) / 2.0e-5; h = int(5 * sig) + 1; k = np.exp(-0.5 * (np.arange(-h, h + 1) / sig) ** 2); k /= k.sum()
            self.lib[s] = np.array([np.convolve(b, k, mode="same") for b in base])
        self.Rm = Rm; sd = np.sqrt(np.maximum((Rm * ((np.arange(11) - 5.0) ** 2)[:, None]).sum(0), 1e-6)) * 0.8; self.sigR = sd            # instrumental sigma [A] per pixel

    def stellar(self, z, sigv):
        lam = jd.LAM; lr = np.log10(lam / (1 + z)); X = np.array([np.interp(lr, self.lg, b) for b in self.lib[sigv]])
        Xp = np.pad(X, ((0, 0), (5, 5))); return sum(self.Rm[d][None, :] * Xp[:, d:d + jd.L] for d in range(11))

    def lines(self, z, sigv=100.0):
        lam = jd.LAM; cols = []
        for nm, l0 in LINES:
            c = l0 * (1 + z); s = np.sqrt((c * sigv / 3e5) ** 2 + np.interp(c, lam, self.sigR) ** 2); cols.append(np.where(np.abs(lam - c) < 6 * s, np.exp(-0.5 * ((lam - c) / s) ** 2), 0.0) / (s * np.sqrt(2 * np.pi)))
        cols = np.array(cols); cols[:, (lam < 3600) | (lam > 9824)] = 0.0; return cols


def solve(Ppos, Qfree, w, f, prior=None, sky_sig=None, irls=4):
    """non-negative amplitudes of Ppos and free (signed) coefficients of Qfree = [instrument | sky] with the priors; NNLS on the variable-split augmented system.
    prior = (mu, Sinv, nu) for the instrument block (first len(mu) columns of Qfree), sky_sig the Gaussian sigma of the sky block."""
    sw = np.sqrt(w); npos = len(Ppos)
    A0 = [Ppos * sw[None]]; 
    if Qfree is None or len(Qfree) == 0:
        D = (np.vstack(A0)).T; sc = np.linalg.norm(D, axis=0); sc[sc == 0] = 1.0; x, _ = nnls(D / sc, sw * f, maxiter=10000); x = x / sc; return x, None
    nq = len(Qfree); ni = len(prior[0]); u = 1.0; Qw = Qfree * sw[None]
    for it in range(irls):
        Lc = np.linalg.cholesky(u * prior[1] + 1e-12 * np.eye(ni)).T if np.isfinite(prior[2]) or it == 0 else Lc                  # Cholesky factor of the current prior precision
        pri_rows = np.zeros((nq, npos + 2 * nq)); pri_rows[:ni, npos:npos + ni] = Lc; pri_rows[:ni, npos + nq:npos + nq + ni] = -Lc; k = np.arange(ni, nq)
        pri_rows[k, npos + k] = 1.0 / sky_sig; pri_rows[k, npos + nq + k] = -1.0 / sky_sig
        top = np.hstack([Ppos.T * sw[:, None], Qw.T, -Qw.T]); D = np.vstack([top, pri_rows]); rhs = np.concatenate([sw * f, Lc @ prior[0], np.zeros(nq - ni)])
        sc = np.linalg.norm(D, axis=0); sc[sc == 0] = 1.0; x, _ = nnls(D / sc, rhs, maxiter=20000); x = x / sc; th = x[npos:npos + nq] - x[npos + nq:]
        r = th[:ni] - prior[0]; m2 = r @ prior[1] @ r; u = (prior[2] + ni) / (prior[2] + m2) if np.isfinite(prior[2]) else 1.0
    return x[:npos], th


def one(args):
    i = args; fid, sps = _W["fid"], _W["sps"]; F, Wt, Z, PR = _W["F"], _W["W"], _W["Z"], _W["PR"]
    f = np.asarray(F[i], np.float64); w = np.asarray(Wt[i], np.float64); z = float(Z[i]); prog = PR[i]; ip = fid.t_priors.get(prog, fid.t_priors["all"])
    from DESIFlow.data.target_nuisance_test import span_mask
    Qreal = np.vstack([fid.inst, fid.sky]); Qflip = np.vstack([fid.inst_flip, fid.sky[:, ::-1]]); sig_sky = np.maximum(fid.prior_sky, 1e-12)
    S = {s: sps.stellar(z, s) for s in SIGV}; Ln = sps.lines(z)

    ni = len(fid.inst); line_px = np.sum(fid.sky ** 2, 0) > np.percentile(np.sum(fid.sky ** 2, 0), 90)           # pixels carrying sky-line structure (top 10% of the sky-template energy)

    def qmat(v):
        Q = (Qreal if v in (1, 3, 5) else Qflip).copy()
        if v in (3, 4):
            Q[ni:] = 0.0                                                                                       # instrument terms only
        if v in (5, 6):
            Q[:ni] = 0.0                                                                                       # sky terms only
        return Q

    def fit(variant, wt, grid=None):
        best = None
        for s in SIGV:
            for av in AV:
                if grid is not None and (s, av) != grid:
                    continue
                P = np.vstack([S[s] * calzetti(jd.LAM / (1 + z), av)[None], Ln]); vis = np.linalg.norm(P * np.sqrt(wt)[None], axis=1) / np.maximum(np.linalg.norm(P * np.sqrt(w)[None], axis=1), 1e-300); P = P * (vis > 0.3)[:, None]       # columns whose peak is hidden are dropped
                if variant == 0:
                    x, _ = solve(P, None, wt, f); m = x @ P; q = None
                else:
                    Qv = qmat(variant); x, th = solve(P, Qv, wt, f, prior=ip, sky_sig=sig_sky); m = x @ P + th @ Qv; q = th
                c = float((wt * (f - m) ** 2).sum())
                if best is None or c < best[0]:
                    best = (c, (s, av), m, q, x @ P)
        return best
    npx = (w > 0).sum(); out = dict(i=int(i), z=z, snr=float(_W["SNR"][i]), prog=prog)
    b0 = fit(0, w); g = b0[1]; b1 = fit(1, w, g); b2 = fit(2, w, g)
    out.update(chi0=b0[0] / npx, chi1=b1[0] / npx, chi2=b2[0] / npx, grid=[float(g[0]), float(g[1])])
    inst_sps = b1[3][:len(fid.inst)] @ fid.inst; out["inst_sps"] = inst_sps.astype(np.float32)
    hs, hl, hc = [], [], []
    for rep in range(2):
        hid = span_mask(jd.L, np.random.default_rng(1000 * int(i) + rep)); sel = hid & (w > 0); wv = w * (~hid); c0 = fit(0, wv); g2 = c0[1]; cs = [c0] + [fit(v, wv, g2) for v in range(1, 7)]
        for acc, m_ in ((hs, sel), (hl, sel & line_px), (hc, sel & ~line_px)):
            acc.append([float((w * (f - c[2]) ** 2)[m_].sum() / max(m_.sum(), 1)) for c in cs])
    out["hid_all7"] = np.mean(hs, 0).tolist(); out["hid_line7"] = np.mean(hl, 0).tolist(); out["hid_cont7"] = np.mean(hc, 0).tolist(); out["hid"] = out["hid_all7"][:3]
    lg = jd.fit_object(fid, z, f, w, prog=prog, parts_list=(("phys", "inst", "sky"),))[("phys", "inst", "sky")]; out["inst_leg"] = lg["comp"]["inst"].astype(np.float32)
    out["model0"] = b0[2].astype(np.float32); out["model1"] = b1[2].astype(np.float32); out["f"] = f.astype(np.float32); out["w"] = w.astype(np.float32)
    return out


def init(root, phys, cache, legacy):
    _W["fid"] = jd.Fiducial(root, phys, legacy); _W["sps"] = SPS(_W["fid"].Rm if _W["fid"].Rm is not None else np.load(os.path.join(phys, "R_mean.npy")).astype(np.float64))
    F, Wt, Z, SNR, CR, show, pop, con, RR = jd.load_objects(phys, cache); _W.update(F=F, W=Wt, Z=Z, SNR=SNR, PR=jd.programs_of_cache_rows(cache, CR), pop=pop)


def main(root, phys, cache, legacy, figdir, cachedir, n=200, workers=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    init(root, phys, cache, legacy)
    idx = [int(i) for i in _W["pop"] if (np.asarray(_W["W"][i]) > 0).sum() > 4000][:n]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(one, idx, chunksize=2)
    chi = np.array([[r["chi0"], r["chi1"], r["chi2"]] for r in res]); hid = np.array([r["hid"] for r in res]); snr = np.array([r["snr"] for r in res])
    def stat(a, b):
        d = a - b; return f"{d.mean():+.4f} +- {d.std() / np.sqrt(len(d)):.4f} ({100 * (d < 0).mean():.0f}% negative)"
    print(f"{len(res)} clean galaxies, S/N {snr.min():.1f}-{snr.max():.1f}; in-sample chi2/px: SPS-only {chi[:,0].mean():.3f}, + instrument/sky {chi[:,1].mean():.3f}, + reversed controls {chi[:,2].mean():.3f}")
    print("in-sample  V1 - V0:", stat(chi[:, 1], chi[:, 0]), "| V2 - V0:", stat(chi[:, 2], chi[:, 0]), "| V1 - V2:", stat(chi[:, 1], chi[:, 2]))
    print(f"hidden chi2/px: SPS-only {hid[:,0].mean():.3f}, + instrument/sky {hid[:,1].mean():.3f}, + reversed {hid[:,2].mean():.3f}")
    print("hidden     V1 - V0:", stat(hid[:, 1], hid[:, 0]), "| V2 - V0:", stat(hid[:, 2], hid[:, 0]), "| V1 - V2:", stat(hid[:, 1], hid[:, 2]))
    # cross-model consistency of the instrument component
    def smooth(x, k=41):
        return np.convolve(x, np.ones(k) / k, mode="same")
    good = (jd.LAM > 3800) & (jd.LAM < 9600); cors, null, amp = [], [], []
    IS = np.array([smooth(r["inst_sps"]) for r in res]); IL = np.array([smooth(r["inst_leg"]) for r in res])
    for a in range(len(res)):
        cors.append(np.corrcoef(IS[a][good], IL[a][good])[0, 1] if IS[a][good].std() > 0 and IL[a][good].std() > 0 else np.nan)
        b = (a + 1 + np.random.default_rng(a).integers(0, len(res) - 1)) % len(res); null.append(np.corrcoef(IS[a][good], IL[b][good])[0, 1] if IS[a][good].std() > 0 and IL[b][good].std() > 0 else np.nan)
        band = (jd.LAM > 4000) & (jd.LAM < 5000); amp.append((IS[a][band].mean(), IL[a][band].mean()))
    cors, null, amp = np.array(cors), np.array(null), np.array(amp)
    H = {k: np.array([r[k] for r in res]) for k in ("hid_all7", "hid_line7", "hid_cont7")}; names = ["V0 SPS only", "V1 real inst+sky", "V2 reversed inst+sky", "V3 real inst only", "V4 reversed inst only", "V5 real sky only", "V6 reversed sky only"]
    for k, nm in (("hid_all7", "all hidden pixels"), ("hid_line7", "hidden sky-line pixels (top 10% sky-template energy)"), ("hid_cont7", "hidden continuum pixels")):
        print(f"decomposition, {nm}: chi2/px V0 {H[k][:,0].mean():.3f}; paired change vs V0: " + ", ".join(f"{names[v].split(' ', 1)[1]} {(H[k][:, v] - H[k][:, 0]).mean():+.4f}+-{(H[k][:, v] - H[k][:, 0]).std() / np.sqrt(len(H[k])):.4f}" for v in range(1, 7)))
    print(f"cross-model instrument component: correlation across pixels, same galaxy median {np.nanmedian(cors):.2f}; different galaxies median {np.nanmedian(null):.2f}; 4000-5000 A level correlation across galaxies {np.corrcoef(amp[:,0], amp[:,1])[0,1]:.2f}, median ratio SPS/legacy {np.nanmedian(amp[:,0] / np.where(np.abs(amp[:,1]) > 0.05, amp[:,1], np.nan)):.2f}")
    fig, ax = plt.subplots(2, 3, figsize=(18, 10)); lab = ["instrument + sky\n(real) - SPS only", "reversed control\n- SPS only", "real - reversed\ncontrol"]
    for j, (d, nm) in enumerate(((chi, "in-sample"), (hid, "hidden pixels"))):
        dd = [d[:, 1] - d[:, 0], d[:, 2] - d[:, 0], d[:, 1] - d[:, 2]]; ax[0, j].bar(range(3), [x.mean() for x in dd], yerr=[x.std() / np.sqrt(len(x)) for x in dd], color=["tab:red", "tab:gray", "tab:green"]); ax[0, j].set_xticks(range(3)); ax[0, j].set_xticklabels(lab, fontsize=8); ax[0, j].axhline(0, color="k", lw=0.8)
        ax[0, j].set_ylabel(f"paired change of the {nm} chi2 per pixel (negative = better)"); ax[0, j].set_title(f"{len(res)} clean galaxies, SPS fit ({nm})", fontsize=9)
    ax[0, 2].hist(cors[np.isfinite(cors)], bins=np.linspace(-1, 1, 41), alpha=0.6, label="same galaxy"); ax[0, 2].hist(null[np.isfinite(null)], bins=np.linspace(-1, 1, 41), alpha=0.6, label="different galaxies"); ax[0, 2].set_xlabel("correlation of the SPS-derived and the legacy-derived instrument component across pixels"); ax[0, 2].set_ylabel("galaxies"); ax[0, 2].legend(fontsize=8)
    ax[1, 0].scatter(amp[:, 1], amp[:, 0], s=8); ax[1, 0].axline((0, 0), slope=1, color="gray", ls=":"); ax[1, 0].set_xlabel("4000-5000 A mean instrument component, legacy-template fit [1e-17 erg/s/cm2/A]"); ax[1, 0].set_ylabel("same, SPS fit [1e-17 erg/s/cm2/A]")
    ex = np.argsort(-np.abs(amp[:, 0]))[:2].tolist() + [int(np.argsort(snr)[len(snr) // 2])]
    for k, e in enumerate(ex[:2]):
        r = res[e]; a = ax[1, 1 + k]; g = r["w"] > 0; a.plot(jd.LAM[g], r["f"][g], color="0.8", lw=0.3); a.plot(jd.LAM, jd.nw(r["f"], r["w"], 6.0), "k", lw=0.9, label="data (NW smooth)"); a.plot(jd.LAM, jd.nw(r["model0"], np.ones(jd.L), 1.0), "tab:blue", lw=0.9, label="SPS only")
        a.plot(jd.LAM, jd.nw(r["model1"], np.ones(jd.L), 1.0), "tab:red", lw=0.9, label="SPS + instrument + sky"); a.plot(jd.LAM, r["inst_sps"], "tab:orange", lw=0.7, label="instrument component (SPS fit)"); a.plot(jd.LAM, r["inst_leg"], "tab:green", lw=0.7, ls="--", label="instrument component (legacy fit)")
        a.set_ylim(-1, np.nanpercentile(jd.nw(r["f"], r["w"], 6.0), 99) * 1.4); a.legend(fontsize=6); a.set_xlabel("observed wavelength [Angstrom]"); a.set_ylabel("flux [1e-17 erg/s/cm2/A]"); a.set_title(f"z = {r['z']:.3f}, S/N {r['snr']:.1f}", fontsize=9)
    fig.tight_layout(); os.makedirs(figdir, exist_ok=True); fig.savefig(os.path.join(figdir, "sps_verification.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5)); cols = ["tab:red", "tab:gray", "tab:orange", "tab:olive", "tab:purple", "tab:pink"]
    for j, (k, nm) in enumerate((("hid_all7", "all hidden pixels"), ("hid_line7", "hidden sky-line pixels"), ("hid_cont7", "hidden continuum pixels"))):
        d = [H[k][:, v] - H[k][:, 0] for v in range(1, 7)]; ax[j].bar(range(6), [x.mean() for x in d], yerr=[x.std() / np.sqrt(len(x)) for x in d], color=cols); ax[j].set_xticks(range(6)); ax[j].set_xticklabels(["inst+sky\nreal", "inst+sky\nreversed", "inst\nreal", "inst\nreversed", "sky\nreal", "sky\nreversed"], fontsize=8); ax[j].axhline(0, color="k", lw=0.8)
        ax[j].set_ylabel("paired change of the hidden-pixel chi2 per pixel vs SPS only (negative = better)"); ax[j].set_title(nm, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "sps_verification_decomposition.png"), dpi=110); plt.close(fig); np.savez(os.path.join(cachedir, "sps_verification_decomposition.npz"), **H)
    np.savez(os.path.join(cachedir, "sps_verification.npz"), chi=chi, hid=hid, snr=snr, z=np.array([r["z"] for r in res]), corr_same=cors, corr_null=null, amp_sps_leg=amp, grid=np.array([r["grid"] for r in res]))
    json.dump([{k: v for k, v in r.items() if k in ("i", "z", "snr", "prog", "chi0", "chi1", "chi2", "hid", "grid")} for r in res], open(os.path.join(root, "sps_verification.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True); ap.add_argument("--phys", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--legacy", required=True); ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        import time
        init(a.root, a.phys, a.cache, a.legacy); idx = [int(i) for i in _W["pop"]][:3]
        for i in idx:
            t0 = time.time(); r = one(i); print(i, "z %.3f S/N %.1f grid" % (r["z"], r["snr"]), r["grid"], "chi2/px V0 %.3f V1 %.3f V2 %.3f | hidden V0 %.3f V1 %.3f V2 %.3f | %.1fs" % (r["chi0"], r["chi1"], r["chi2"], *r["hid"], time.time() - t0), flush=True)
    else:
        main(a.root, a.phys, a.cache, a.legacy, a.fig, a.figcache, a.n, a.workers)
