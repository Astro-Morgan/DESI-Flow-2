"""
Choice of the number of templates K for the per-camera instrument templates (b, r, z) and the sky templates, from dense K sweeps over several training seeds,
a smoothing cubic B-spline of the hold-out hidden-pixel chi2 against ln K, and a bootstrap over seeds; then figures of the final templates and of their fits to
held-out data.

stages
  select   spline fit + K* per group -> k_selection.json, figure k_selection (chi2 against K with seeds, spline, K*)
  fits     final templates (trained at K*, fresh seed, in OUT/final) against held-out rows: figures template_fits_instrument, template_fits_sky
  atlas    every final template: figures templates_instrument_cam, templates_sky

K*: the spline minimum when it lies inside the swept range (instrument templates overfit beyond it); when the spline is still falling at the end of the range
(sky) the knee: the smallest K at which a doubling of K lowers the spline chi2 by less than KNEE (default 5%, i.e. d ln chi2 / d ln K > -0.07).

Run:  python -m DESIFlow.templates.k_selection select|fits|atlas --root ROOT --fig FIG --figcache CACHE [--seeds 0,1,2]
"""
import argparse
import json
import os

import numpy as np
from scipy.interpolate import splev, splrep

KNEE = 0.05


def load_group(root, seeds, group):
    """-> K (n,), chi2 (n_seeds, n), sem (n_seeds, n), baseline."""
    res = {}
    for s in seeds:
        p = os.path.join(root, f"seed{s}", "sky" if group == "sky" else f"instrument_cam/{group}", "summary.json")
        res[s] = {int(k): v for k, v in json.load(open(p)).items()}
    K = sorted(set.intersection(*[set(r) for r in res.values()]))
    chi = np.array([[res[s][k]["hidden_chi2"] for k in K] for s in seeds]); sem = np.array([[res[s][k]["hidden_chi2_sem"] for k in K] for s in seeds])
    return np.array(K), chi, sem, res[seeds[0]][K[0]]["baseline_chi2"]


def fit_spline(K, y, sig, log_y=False):
    """cubic smoothing B-spline in ln K (weights 1/sigma, smoothing s = number of points); returns a callable and the dense grid."""
    x = np.log(K); yy = np.log(y) if log_y else y; w = 1.0 / (sig / y if log_y else sig)
    tck = splrep(x, yy, w=w, k=3, s=len(x))
    xd = np.linspace(x[0], x[-1], 2000)
    v = splev(xd, tck)
    return np.exp(xd), (np.exp(v) if log_y else v), tck


def choose(Kd, yd, log_y=False):
    """spline minimum if interior, otherwise the knee."""
    i = int(np.argmin(yd))
    if i < len(yd) - 5 and i > 0:
        return float(Kd[i]), "minimum"
    ly = np.log(yd) if not log_y else np.log(yd)
    # smallest K where doubling K lowers chi2 by < KNEE
    for k, y in zip(Kd, yd):
        if 2 * k > Kd[-1]:
            break
        y2 = np.interp(2 * k, Kd, yd)
        if (y - y2) / y < KNEE:
            return float(k), "knee"
    return float(Kd[-1]), "knee"


def camera_rows(root, cam, seed=0):
    """per-row hold-out hidden chi2 (fixed masks) of every saved K of one camera, seed `seed`: (n_K, n_rows); cached in ROOT/rows_cam_<cam>.npz."""
    import torch
    from DESIFlow.templates.camera_templates import CAMS
    from DESIFlow.templates.deeprock_lite import evaluate
    path = os.path.join(root, f"rows_cam_{cam}.npz")
    if os.path.exists(path):
        z = np.load(path); return z["K"], z["rows"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); a0, a1 = CAMS[cam]; L = a1 - a0
    d = os.path.join(root, f"seed{seed}", "instrument"); od = os.path.join(root, f"seed{seed}", "instrument_cam", cam)
    ks = sorted(int(f[1:4]) for f in os.listdir(od) if f.startswith("K") and f.endswith(".npz"))
    s = np.load(os.path.join(od, f"K{ks[0]:03d}.npz"))["pixel_scale_s"]
    fe = np.load(os.path.join(d, "eval_flux.npy"), mmap_mode="r")[:, a0:a1]; we = np.load(os.path.join(d, "eval_ivar.npy"), mmap_mode="r")[:, a0:a1]
    GE = torch.tensor(np.asarray(fe) * s[None], dtype=torch.float32, device=dev); OE = torch.tensor(np.asarray(we) / s[None] ** 2, dtype=torch.float32, device=dev)
    mk = dict(n_spans=max(6, int(36 * L / 7781)), len_range=(5, 250)); rows = []
    for k in ks:
        B = torch.tensor(np.load(os.path.join(od, f"K{k:03d}.npz"))["templates_whitened"], device=dev)
        rows.append(evaluate(B, GE, OE, mask_kw=mk)[0])
    np.savez(path, K=np.array(ks), rows=np.array(rows, np.float32)); return np.array(ks), np.array(rows, np.float32)


def sky_rows(root, seed=0):
    od = os.path.join(root, f"seed{seed}", "sky"); ks = sorted(int(f[1:4]) for f in os.listdir(od) if f.startswith("K") and f.endswith(".npz"))
    return np.array(ks), np.array([np.load(os.path.join(od, f"K{k:03d}.npz"))["hidden_chi2_per_row_eval"] for k in ks], np.float32)


def select(root, seeds, figdir, cachedir, nboot=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = ["b", "r", "z", "sky"]; out = {}; cache = {}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9)); rng = np.random.default_rng(0)
    for a, g in zip(ax.ravel(), groups):
        K, chi, sem, base = load_group(root, seeds, g); log_y = g == "sky"
        m = chi.mean(0); sd = np.sqrt((sem ** 2).mean(0) + chi.var(0, ddof=1) / len(seeds)) if len(seeds) > 1 else sem[0]
        Kd, yd, _ = fit_spline(K, m, sd, log_y); kstar, how = choose(Kd, yd, log_y)
        boots = []
        Kr, R = (sky_rows(root, seeds[0]) if g == "sky" else camera_rows(root, g, seeds[0]))
        keep = [list(Kr).index(k) for k in K if k in set(Kr)]; R = R[keep]                       # rows of seed 0 for the swept K
        for _ in range(nboot):                                                                  # paired bootstrap over held-out rows: the same rows for every K
            ii = rng.integers(0, R.shape[1], R.shape[1]); mb = R[:, ii].mean(1) + (m - R.mean(1))
            Kb, yb, _ = fit_spline(K, mb, sd, log_y); boots.append(choose(Kb, yb, log_y)[0])
        lo, med, hi = np.percentile(boots, [16, 50, 84])
        out[g] = dict(K_star=kstar, rule=how, boot_16=float(lo), boot_median=float(med), boot_84=float(hi), spline_min=float(yd.min()), chi2_at_kstar=float(np.interp(kstar, Kd, yd)),
                      seed_scatter_median=float(np.median(chi.std(0, ddof=1))) if len(seeds) > 1 else None, baseline=float(base))
        for s_i, s in enumerate(seeds):
            a.plot(K, chi[s_i], ".", ms=4, alpha=0.5, label=f"seed {s}")
        a.errorbar(K, m, sd, fmt="k.", ms=5, lw=0.8, label="mean over seeds +- combined error"); a.plot(Kd, yd, color="tab:red", lw=1.5, label="cubic B-spline in ln K")
        a.axvline(kstar, color="tab:red", ls=":", label=f"K* = {kstar:.0f} ({how}; bootstrap 16-84%: {lo:.0f}-{hi:.0f})"); a.axhline(base, color="gray", ls="--", lw=0.8, label="zero model")
        a.set_xscale("log"); a.set_xlabel("number of templates K"); a.set_ylabel("hold-out hidden-pixel chi2 per pixel")
        if log_y: a.set_yscale("log")
        else: a.set_ylim(m.min() - 0.004, m.min() + 0.06 * (base - m.min()) + 0.02)
        a.set_title({"b": "camera b instrument", "r": "camera r instrument", "z": "camera z instrument", "sky": "sky templates (DR1 VAC, assumed weights)"}[g], fontsize=10); a.legend(fontsize=7)
        cache.update({f"{g}_K": K, f"{g}_chi2_seeds": chi, f"{g}_sem": sem, f"{g}_spline_K": Kd, f"{g}_spline": yd, f"{g}_boot": np.array(boots)})
    fig.tight_layout(); os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fig.savefig(os.path.join(figdir, "k_selection.png"), dpi=130); np.savez(os.path.join(cachedir, "k_selection.npz"), **cache)
    json.dump(out, open(os.path.join(root, "k_selection.json"), "w"), indent=1)
    for g in groups:
        print(g, out[g])
    return out


def wls_fit(T, f, w, hidden=None):
    """coefficients of templates T (K, L) for flux f (L,) with ivar w (L,) (hidden pixels excluded), ridge 1e-9 relative."""
    ww = w * (~hidden if hidden is not None else 1.0)
    A = (T * ww) @ T.T; A += 1e-9 * np.trace(A) / len(A) * np.eye(len(A))
    return np.linalg.solve(A, (T * ww) @ f)


def _spans(L, rng, n=None):
    from DESIFlow.data.target_nuisance_test import span_mask
    return span_mask(L, rng, n_spans=n or max(6, int(36 * L / 7781)))


def fits(root, figdir, cachedir, ks, n_ex=3, seed=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from DESIFlow.templates.camera_templates import CAMS
    from DESIFlow.templates.sky_instrument_sweep import nw_smooth
    fin = os.path.join(root, "final"); rng = np.random.default_rng(seed)
    # ---------------- instrument
    f = np.load(os.path.join(fin, "instrument", "eval_flux.npy"), mmap_mode="r"); w = np.load(os.path.join(fin, "instrument", "eval_ivar.npy"), mmap_mode="r")
    rows = np.sort(rng.choice(len(f), n_ex, replace=False)); cache = {"rows": rows}
    fig, ax = plt.subplots(n_ex + 1, 3, figsize=(19, 3.6 * (n_ex + 1)))
    for j, c in enumerate("brz"):
        a0, a1 = CAMS[c]; K = ks[c]; T = np.load(os.path.join(fin, "instrument_cam", c, f"K{K:03d}.npz"))["templates_flux"].astype(np.float64)
        lam = 3600.0 + 0.8 * np.arange(a0, a1)
        for i, r in enumerate(rows):
            fr = np.asarray(f[r, a0:a1], np.float64); wr = np.asarray(w[r, a0:a1], np.float64)
            hid = _spans(a1 - a0, np.random.default_rng(100 + i)); coef = wls_fit(T, fr, wr, hid); fit = coef @ T
            good = wr > 0; sm = nw_smooth(fr, wr, 11.0)
            a = ax[i, j]; a.plot(lam[good], fr[good], color="0.75", lw=0.4, label="held-out SKY-fibre residual row")
            a.plot(lam, sm, color="k", lw=1.0, label="data, NW smooth (sigma 11 px)"); a.plot(lam, nw_smooth(fit, np.ones_like(fit), 3.0), color="tab:red", lw=1.0, label=f"K={K} templates, fitted on visible pixels")
            s0 = None
            for k in range(len(hid) + 1):
                if k < len(hid) and hid[k] and s0 is None: s0 = k
                if (k == len(hid) or not hid[k]) and s0 is not None: a.axvspan(lam[s0], lam[k - 1], color="0.9", lw=0, zorder=0); s0 = None
            sel = hid & good
            chi_t = (wr * (fr - fit) ** 2)[sel].sum() / max(sel.sum(), 1); chi_0 = (wr * fr ** 2)[sel].sum() / max(sel.sum(), 1)
            a.set_title(f"camera {c}, row {r}: hidden chi2/px {chi_0:.2f} (zero model) -> {chi_t:.2f}", fontsize=9)
            lim = np.nanpercentile(np.abs(sm), 99.5) * 1.5; a.set_ylim(-lim, lim); a.set_ylabel("flux [1e-17 erg/s/cm2/A]")
            cache[f"{c}_row{r}_wavelength"] = lam; cache[f"{c}_row{r}_data"] = fr * good; cache[f"{c}_row{r}_data_nw"] = sm; cache[f"{c}_row{r}_fit"] = fit
            if i == 0 and j == 0: a.legend(fontsize=7, loc="upper right")
        # population: hold-out hidden chi2 per row, zero model vs templates (first 3000 rows)
        hz, ht = [], []
        for r in range(0, min(len(f), 6000), 2):
            fr = np.asarray(f[r, a0:a1], np.float64); wr = np.asarray(w[r, a0:a1], np.float64)
            if (wr > 0).sum() < 0.5 * (a1 - a0): continue
            hid = _spans(a1 - a0, np.random.default_rng(r)); coef = wls_fit(T, fr, wr, hid); sel = hid & (wr > 0)
            if sel.sum() == 0: continue
            hz.append((wr * fr ** 2)[sel].sum() / sel.sum()); ht.append((wr * (fr - coef @ T) ** 2)[sel].sum() / sel.sum())
        hz, ht = np.array(hz), np.array(ht); bins = np.linspace(0.5, 2.0, 60)
        ax[n_ex, j].hist(hz, bins, alpha=0.5, label=f"zero model, mean {hz.mean():.3f}"); ax[n_ex, j].hist(ht, bins, alpha=0.5, label=f"K={K} templates, mean {ht.mean():.3f}")
        ax[n_ex, j].set_xlabel("hidden-pixel chi2 per pixel, one row"); ax[n_ex, j].set_ylabel("held-out rows"); ax[n_ex, j].legend(fontsize=8); ax[n_ex, j].set_title(f"camera {c}: {len(hz)} held-out rows", fontsize=9)
        cache[f"{c}_hist_zero"] = hz; cache[f"{c}_hist_fit"] = ht
        for i in range(n_ex): ax[i, j].set_xlabel("observed wavelength [Angstrom]")
    fig.suptitle("Final per-camera instrument templates against held-out SKY-fibre rows (held-out tiles); grey bands = hidden spans scored in the titles", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(os.path.join(figdir, "template_fits_instrument.png"), dpi=110); plt.close(fig)
    np.savez(os.path.join(cachedir, "template_fits_instrument.npz"), **cache)
    # ---------------- sky
    Ks = ks["sky"]; T = np.load(os.path.join(fin, "sky", f"K{Ks:03d}.npz"))["templates_flux"].astype(np.float64); lam = 3600.0 + 0.8 * np.arange(7781)
    fs = np.load(os.path.join(fin, "sky", "eval_flux.npy"), mmap_mode="r"); ws = np.load(os.path.join(fin, "sky", "eval_ivar.npy"), mmap_mode="r")
    rows = np.sort(rng.choice(len(fs), n_ex, replace=False)); cache = {"rows": rows, "wavelength": lam}
    fig, ax = plt.subplots(n_ex + 1, 3, figsize=(19, 3.6 * (n_ex + 1)))
    for i, r in enumerate(rows):
        fr = np.asarray(fs[r], np.float64); wr = np.asarray(ws[r], np.float64)
        hid = _spans(7781, np.random.default_rng(200 + i)); coef = wls_fit(T, fr, wr, hid); fit = coef @ T; sel = hid & (wr > 0)
        chi_t = (wr * (fr - fit) ** 2)[sel].sum() / sel.sum(); chi_0 = (wr * fr ** 2)[sel].sum() / sel.sum()
        for k, (lo, hi, nm) in enumerate([(3600, 9824, "full range"), (7300, 8000, "zoom 7300-8000 A"), (5500, 6000, "zoom 5500-6000 A")]):
            m = (lam >= lo) & (lam <= hi); a = ax[i, k]
            a.plot(lam[m], fr[m], color="k", lw=0.7, label="held-out VAC sky spectrum"); a.plot(lam[m], fit[m], color="tab:red", lw=0.7, label=f"K={Ks} templates (visible-pixel fit)")
            a.plot(lam[m], (fr - fit)[m], color="tab:blue", lw=0.5, label="residual"); a.set_yscale("symlog", linthresh=1.0)
            s0 = None
            for q in range(7781 + 1):
                if q < 7781 and hid[q] and s0 is None: s0 = q
                if (q == 7781 or not hid[q]) and s0 is not None:
                    if lam[min(q, 7780)] >= lo and lam[s0] <= hi: a.axvspan(lam[s0], lam[min(q, 7780)], color="0.9", lw=0, zorder=0)
                    s0 = None
            a.set_xlim(lo, hi); a.set_ylabel("flux [1e-17 erg/s/cm2/A] (symlog)")
            if k == 0: a.set_title(f"row {r}: hidden chi2/px {chi_0:.1f} (zero model) -> {chi_t:.3f}", fontsize=9)
            else: a.set_title(nm, fontsize=9)
            if i == 0 and k == 0: a.legend(fontsize=7)
        cache[f"row{r}_data"] = fr; cache[f"row{r}_fit"] = fit
    hz, ht, fr_err = [], [], []
    for r in range(0, min(len(fs), 2000)):
        fr = np.asarray(fs[r], np.float64); wr = np.asarray(ws[r], np.float64); hid = _spans(7781, np.random.default_rng(r)); coef = wls_fit(T, fr, wr, hid); sel = hid & (wr > 0)
        hz.append((wr * fr ** 2)[sel].sum() / sel.sum()); ht.append((wr * (fr - coef @ T) ** 2)[sel].sum() / sel.sum())
        br = fr > np.percentile(fr, 90); fr_err.append(np.median(np.abs(fr - coef @ T)[sel & br] / fr[sel & br]) if (sel & br).any() else np.nan)
    hz, ht, fr_err = np.array(hz), np.array(ht), np.array(fr_err)
    ax[n_ex, 0].hist(np.log10(hz), 50, alpha=0.5, label=f"zero model, median {np.median(hz):.0f}"); ax[n_ex, 0].hist(np.log10(ht), 50, alpha=0.5, label=f"K={Ks} templates, median {np.median(ht):.2f}")
    ax[n_ex, 0].set_xlabel("log10 hidden-pixel chi2 per pixel (assumed weights)"); ax[n_ex, 0].set_ylabel("held-out sky spectra"); ax[n_ex, 0].legend(fontsize=8)
    ax[n_ex, 1].hist(100 * fr_err[np.isfinite(fr_err)], 50); ax[n_ex, 1].set_xlabel("median |data - fit| / data on the brightest 10% of pixels [percent]"); ax[n_ex, 1].set_ylabel("held-out sky spectra")
    ax[n_ex, 2].axis("off")
    fig.suptitle("Final sky templates against held-out DR1 VAC sky spectra (held-out nights)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(os.path.join(figdir, "template_fits_sky.png"), dpi=110); plt.close(fig)
    cache.update({"hist_zero": hz, "hist_fit": ht, "bright_frac_err": fr_err}); np.savez(os.path.join(cachedir, "template_fits_sky.npz"), **cache)


def atlas(root, figdir, cachedir, ks, nlines=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from DESIFlow.templates.camera_templates import CAMS
    fin = os.path.join(root, "final"); cache = {}
    fig, ax = plt.subplots(1, 3, figsize=(20, 11))
    for j, c in enumerate("brz"):
        a0, a1 = CAMS[c]; z = np.load(os.path.join(fin, "instrument_cam", c, f"K{ks[c]:03d}.npz")); T = z["templates_flux"]; lam = 3600.0 + 0.8 * np.arange(a0, a1)
        for k in range(len(T)):
            v = T[k] / np.abs(T[k]).max(); ax[j].plot(lam, v * 0.45 - k, lw=0.5, color="k"); ax[j].text(lam[0], -k + 0.2, f"#{k + 1}  peak {np.abs(T[k]).max():.3g}", fontsize=6)
        ax[j].set_yticks([]); ax[j].set_xlabel("observed wavelength [Angstrom]"); ax[j].set_title(f"camera {c}: all K={ks[c]} templates (each scaled to its own peak; peak in 1e-17 erg/s/cm2/A per unit coefficient)", fontsize=8)
        cache[f"{c}_templates_flux"] = T; cache[f"{c}_wavelength"] = lam; cache[f"{c}_energy"] = z["energy"]
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "templates_instrument_cam.png"), dpi=110); plt.close(fig)
    np.savez(os.path.join(cachedir, "templates_instrument_cam.npz"), **cache)
    z = np.load(os.path.join(fin, "sky", f"K{ks['sky']:03d}.npz")); T = z["templates_flux"]; lam = 3600.0 + 0.8 * np.arange(7781); K = len(T)
    fig = plt.figure(figsize=(20, 12)); gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], width_ratios=[4, 1])
    ax0 = fig.add_subplot(gs[0, 0]); im = ax0.imshow(T / np.abs(T).max(1, keepdims=True), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, extent=[lam[0], lam[-1], K + 0.5, 0.5])
    ax0.set_xlabel("observed wavelength [Angstrom]"); ax0.set_ylabel("template number (ordered by energy)"); ax0.set_title(f"all K={K} sky templates, each scaled to its own peak (colour = fraction of peak)"); fig.colorbar(im, ax=ax0, pad=0.01)
    ax1 = fig.add_subplot(gs[0, 1]); ax1.semilogy(np.arange(1, K + 1), z["energy"], ".-"); ax1.set_xlabel("template number"); ax1.set_ylabel("energy of the template on the training data (whitened units)")
    ax2 = fig.add_subplot(gs[1, 0])
    for k in range(nlines):
        ax2.plot(lam, T[k] / np.abs(T[k]).max() * 0.45 - k, lw=0.5, color="k"); ax2.text(lam[0], -k + 0.15, f"#{k + 1}", fontsize=7)
    ax2.set_yticks([]); ax2.set_xlabel("observed wavelength [Angstrom]"); ax2.set_title(f"first {nlines} sky templates as lines (each scaled to its own peak)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "templates_sky.png"), dpi=110); plt.close(fig)
    np.savez(os.path.join(cachedir, "templates_sky.npz"), templates_flux=T, wavelength=lam, energy=z["energy"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["select", "fits", "atlas"]); ap.add_argument("--root", required=True); ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(",")]
    if a.stage == "select":
        select(a.root, seeds, a.fig, a.figcache)
    else:
        ksel = json.load(open(os.path.join(a.root, "k_selection.json"))); ks = {g: int(round(ksel[g]["K_star"])) for g in ksel}
        (fits if a.stage == "fits" else atlas)(a.root, a.fig, a.figcache, ks)
