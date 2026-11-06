"""
Instrument and sky templates by DeepRock-style chi2 reconstruction, with a sweep of the number of templates K.

instrument   training data: the cached SKY rows (sky-fibre residuals after sky subtraction, calibrated, observed frame, with their
             ivar and mask); rows with a derivable tile (TARGETID < 0: TILEID = -TARGETID // 10000) are split BY TILE (tile % 20 == 0
             held out), so the hold-out has independent exposures;
sky          training data: the 9,176 DR1 sky spectra of the skyspec VAC (1e-17 erg/s/cm2/A/arcsec2, no ivar: the weights are an
             assumption, ivar = 1 / (a^2 + (b f)^2) with a = 0.1 and b = 0.03, i.e. 3% relative accuracy with a small floor);
             held out BY NIGHT (night % 10 == 0).

Each K is trained independently (not nested) with the masked-denoising chi2 of deeprock_lite and saved with its metrics; nothing
is discarded. The reported "mean chi2" is the chi2 per hidden pixel on the hold-out rows with the coefficients fitted on the
visible pixels, against the zero-model baseline.

Run:  python -m DESIFlow.templates.sky_instrument_sweep prep|sweep|figures|all --cache DR2_CACHE --out OUT --fig FIG --figcache CACHE
"""
import argparse
import json
import os
import time

import numpy as np

KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
VAC = "/global/cfs/cdirs/desi/public/dr1/vac/dr1/skyspec/v1.0/sky_spectra_vac_v1.fits"
LAM = 3600.0 + 0.8 * np.arange(7781)


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------------------------------------ data
def prep_instrument(cache, out, n_train=150000, n_eval=20000, seed=0):
    from DESIFlow.data.dr2_cache import load_index
    d = os.path.join(out, "instrument"); os.makedirs(d, exist_ok=True)
    idx = load_index(cache)
    tid = np.asarray(idx["targetid"])[np.asarray(idx["sky_rows"])]
    st = np.load(os.path.join(cache, "sky", "status.npy"), mmap_mode="r"); ng = np.load(os.path.join(cache, "sky", "ngood.npy"), mmap_mode="r")
    ok = (np.asarray(st) == 1) & (np.asarray(ng) >= 4000) & (tid < 0)
    tile = (-tid) // 10000
    hold = tile % 20 == 0
    rng = np.random.default_rng(seed)
    pools = {"train": np.flatnonzero(ok & ~hold), "eval": np.flatnonzero(ok & hold)}
    _log(f"instrument pools: train {len(pools['train'])} rows, eval {len(pools['eval'])} rows (rows with a derivable tile and >= 4000 good pixels); "
         f"tiles held out: {len(np.unique(tile[ok & hold]))} of {len(np.unique(tile[ok]))}")
    F, I, M = (np.load(os.path.join(cache, "sky", k + ".npy"), mmap_mode="r") for k in ("flux", "ivar", "mask"))
    for name, n in (("train", n_train), ("eval", n_eval)):
        rows = np.sort(rng.choice(pools[name], min(n, len(pools[name])), replace=False))
        f = np.empty((len(rows), 7781), np.float32); w = np.empty_like(f)
        for a in range(0, len(rows), 5000):
            r = rows[a:a + 5000]
            ff, ii, mm = np.asarray(F[r]), np.asarray(I[r]), np.asarray(M[r])
            f[a:a + 5000], w[a:a + 5000] = ff, np.where(mm == 0, ii, 0.0)
        np.save(os.path.join(d, f"{name}_flux.npy"), f); np.save(os.path.join(d, f"{name}_ivar.npy"), w)
        np.savez(os.path.join(d, f"{name}_rows.npz"), cache_rows=rows, targetid=tid[rows], tile=tile[rows])
        _log(f"saved {name}: {len(rows)} rows")


def prep_sky(out, a=0.1, b=0.03):
    import fitsio
    d = os.path.join(out, "sky"); os.makedirs(d, exist_ok=True)
    F = fitsio.FITS(VAC)
    flux = F["FLUX"].read().astype(np.float64); meta = F["METADATA"].read()
    night = meta["NIGHT"]
    fin = np.isfinite(flux)
    keep = fin.mean(1) > 0.9
    f = np.where(fin, flux, 0.0)
    w = np.where(fin, 1.0 / (a ** 2 + (b * np.abs(f)) ** 2), 0.0)
    hold = (night % 10 == 0)
    _log(f"sky VAC: {keep.sum()} usable spectra of {len(keep)}; held-out nights: {len(np.unique(night[keep & hold]))} ({(keep & hold).sum()} spectra); "
         f"flux percentiles 1/50/99: {np.percentile(f[fin], [1, 50, 99]).round(2)}; weights ivar = 1/({a}^2 + ({b} f)^2)")
    for name, sel in (("train", keep & ~hold), ("eval", keep & hold)):
        np.save(os.path.join(d, f"{name}_flux.npy"), f[sel].astype(np.float32)); np.save(os.path.join(d, f"{name}_ivar.npy"), w[sel].astype(np.float32))
        np.savez(os.path.join(d, f"{name}_rows.npz"), night=night[sel], expid=meta["EXPID"][sel])
    json.dump({"a": a, "b": b}, open(os.path.join(d, "weights.json"), "w"))


# ------------------------------------------------------------------------------------------------ sweep
def sweep(kind, out, ks=KS, steps=1500, batch=512, lr=3e-3, seed=0):
    import torch
    from DESIFlow.templates.deeprock_lite import evaluate, order_orthonormal, train_templates
    d = os.path.join(out, kind)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_tr, w_tr = np.load(os.path.join(d, "train_flux.npy"), mmap_mode="r"), np.load(os.path.join(d, "train_ivar.npy"), mmap_mode="r")
    # pixel scale s_l = sqrt(mean ivar over good pixels), accumulated in chunks (no big host copies)
    sw, sn = np.zeros(w_tr.shape[1]), np.zeros(w_tr.shape[1])
    for a in range(0, len(w_tr), 10000):
        c = np.asarray(w_tr[a:a + 10000]); sw += c.sum(0, dtype=np.float64); sn += (c > 0).sum(0)
    s = np.sqrt(sw / np.maximum(sn, 1)); s = np.where(s > 0, s, np.median(s[s > 0])).astype(np.float32)

    def to_gpu(fm, wm):
        G = torch.empty(fm.shape, dtype=torch.float32, device=dev); O = torch.empty_like(G)
        for a in range(0, len(fm), 10000):
            fc, wc = np.asarray(fm[a:a + 10000]), np.asarray(wm[a:a + 10000])
            G[a:a + 10000] = torch.from_numpy(fc * s[None]).to(dev); O[a:a + 10000] = torch.from_numpy(wc / s[None] ** 2).to(dev)
        return G, O
    G, O = to_gpu(f_tr, w_tr)
    GE, OE = to_gpu(np.load(os.path.join(d, "eval_flux.npy"), mmap_mode="r"), np.load(os.path.join(d, "eval_ivar.npy"), mmap_mode="r"))
    _log(f"{kind}: train {tuple(G.shape)}, eval {tuple(GE.shape)} on {dev}; pixel scale s from {s.min():.3g} to {s.max():.3g}")
    spath = os.path.join(d, "summary.json")
    summary = {int(k): v for k, v in json.load(open(spath)).items()} if os.path.exists(spath) else {}      # keep earlier K
    for K in ks:
        path = os.path.join(d, f"K{K:03d}.npz")
        t0 = time.time()
        U, hist = train_templates(G, O, K, steps=steps, batch=batch, lr=lr, seed=seed, log=_log)
        B, S = order_orthonormal(U, G, O)
        hid, base, vis = evaluate(B, GE, OE)
        sem = float(hid.std() / np.sqrt(len(hid)))
        summary[K] = dict(hidden_chi2=float(hid.mean()), hidden_chi2_sem=sem, hidden_chi2_median=float(np.median(hid)), baseline_chi2=float(base.mean()),
                          visible_chi2=float(vis.mean()), seconds=time.time() - t0)
        np.savez(path, templates_whitened=B.cpu().numpy(), templates_flux=(B.cpu().numpy() / s[None]), pixel_scale_s=s, energy=S.cpu().numpy(), K=K,
                 wavelength_A=LAM, train_loss_history=hist, hidden_chi2_per_row_eval=hid, baseline_per_row_eval=base, **{k: v for k, v in summary[K].items()})
        _log(f"{kind} K={K}: hold-out hidden chi2 {summary[K]['hidden_chi2']:.4f} +- {sem:.4f} (zero model {summary[K]['baseline_chi2']:.4f}), in-sample visible {summary[K]['visible_chi2']:.4f}, {summary[K]['seconds']:.0f} s")
        json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1)
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ figures
def nw_smooth(f, w, sigma):
    """Gaussian-kernel weighted mean with sqrt(ivar) weights over the good pixels (the smoothing of the user's notebook)."""
    from scipy.ndimage import gaussian_filter
    good = w > 0
    num = gaussian_filter(f * np.sqrt(w) * good, sigma); den = gaussian_filter(np.sqrt(w) * good, sigma)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def figures(out, figdir, cachedir, dr2_cache=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from scipy.ndimage import median_filter
    from DESIFlow.templates.deeprock_lite import wls_project, whiten
    os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    titles = {"instrument": "instrument templates (SKY-fibre residual rows)", "sky": "sky templates (DR1 sky VAC)"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    cache = {}
    for a, kind in zip(ax, ("instrument", "sky")):
        S = json.load(open(os.path.join(out, kind, "summary.json")))
        k = np.array(sorted(int(x) for x in S)); h = np.array([S[str(x)]["hidden_chi2"] for x in k]); e = np.array([S[str(x)]["hidden_chi2_sem"] for x in k])
        v = np.array([S[str(x)]["visible_chi2"] for x in k]); b = S[str(k[0])]["baseline_chi2"]
        a.errorbar(k, h, e, marker="o", ms=4, color="tab:blue", label="hold-out, hidden pixels (the score)")
        a.plot(k, v, marker="s", ms=3, ls=":", color="tab:orange", label="hold-out, visible pixels (fitted)")
        a.axhline(b, color="gray", ls="--", label="zero model (K = 0)")
        a.set_xscale("log"); a.set_xlabel("number of templates K"); a.set_ylabel("mean chi2 per pixel"); a.set_title(titles[kind]); a.legend(fontsize=8)
        if kind == "sky":
            a.set_yscale("log")
        cache.update({f"{kind}_K": k, f"{kind}_hidden_chi2": h, f"{kind}_hidden_chi2_sem": e, f"{kind}_visible_chi2": v, f"{kind}_baseline": b})
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "template_chi2_vs_K.png"), dpi=150); np.savez(os.path.join(cachedir, "template_chi2_vs_K.npz"), **cache)
    plt.close(fig)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for kind, Kref, Ks in (("instrument", 16, (4, 16, 64)), ("sky", 32, (8, 32, 64))):
        d = os.path.join(out, kind)
        fe, we = np.load(os.path.join(d, "eval_flux.npy"), mmap_mode="r"), np.load(os.path.join(d, "eval_ivar.npy"), mmap_mode="r")
        sets = {K: np.load(os.path.join(d, f"K{K:03d}.npz")) for K in Ks}
        s = sets[Ks[0]]["pixel_scale_s"]
        n = min(len(fe), 3000)
        f, w = np.asarray(fe[:n]), np.asarray(we[:n])
        n0, extra_label = len(f), None
        if kind == "instrument" and dr2_cache:                                                    # the user's own example: cache row 1000 of sky/
            from DESIFlow.data.dr2_cache import load_index
            Fc, Ic, Mc = (np.load(os.path.join(dr2_cache, "sky", k + ".npy"), mmap_mode="r") for k in ("flux", "ivar", "mask"))
            ix = load_index(dr2_cache); tid1 = int(np.asarray(ix["targetid"])[int(np.asarray(ix["sky_rows"])[1000])])
            tile1 = (-tid1) // 10000 if tid1 < 0 else None
            f = np.vstack([f, np.asarray(Fc[1000])[None]]); w = np.vstack([w, np.where(np.asarray(Mc[1000]) == 0, np.asarray(Ic[1000]), 0.0)[None]])
            extra_label = f"your notebook row (cache row 1000)\nTARGETID {tid1}, " + (f"tile {tile1}, {'held out' if tile1 % 20 == 0 else 'not held out'}" if tile1 is not None else "no tile derivable: never in the training or hold-out sets")
        g, om, _ = whiten(f, w, s)
        U = torch.tensor(sets[Kref if Kref in sets else Ks[1]]["templates_whitened"], device=dev)
        c = wls_project(U, torch.tensor(g, device=dev), torch.tensor(om, device=dev))
        chi_full = ((torch.tensor(om, device=dev) * (torch.tensor(g, device=dev) - c @ U) ** 2).sum(1) / (torch.tensor(om, device=dev) > 0).sum(1)).cpu().numpy()
        base = ((om * g ** 2).sum(1) / np.maximum((om > 0).sum(1), 1))
        gain = base - chi_full
        order = np.argsort(gain[:n0])
        pick = [order[int(0.5 * n0)], order[int(0.97 * n0)], order[int(0.1 * n0)]]
        labels = ["typical gain", "large gain", "small gain"]
        if extra_label:
            pick.append(n0); labels.append(extra_label)
        ncol = 3 if kind == "instrument" else 2
        fig, ax = plt.subplots(len(pick), ncol, figsize=(5.2 * ncol, 3.2 * len(pick)))
        ex = {"wavelength_A": LAM}
        for r, (i, lab) in enumerate(zip(pick, labels)):
            gi, oi = torch.tensor(g[i:i + 1], device=dev), torch.tensor(om[i:i + 1], device=dev)
            preds = {}
            for K in Ks:
                Uk = torch.tensor(sets[K]["templates_whitened"], device=dev)
                preds[K] = ((wls_project(Uk, gi, oi) @ Uk)[0].cpu().numpy()) / s
                ex[f"row{i}_pred_K{K}"] = preds[K]
            ex[f"row{i}_flux"] = f[i]; ex[f"row{i}_ivar"] = w[i]
            wins = [(3600, 9824), (3600, 4300), (9200, 9824)] if kind == "instrument" else [(3600, 9824), (8200, 8900)]
            for cidx, (lo, hi) in enumerate(wins):
                a = ax[r, cidx]; m = (LAM >= lo) & (LAM <= hi)
                if kind == "instrument":
                    a.plot(LAM[m], np.where(w[i] > 0, f[i], np.nan)[m], lw=0.6, alpha=0.4, color="tab:gray", label="data (masked pixels left out)")
                    a.plot(LAM[m], nw_smooth(f[i], w[i], 11.0 / 6.0)[m], lw=0.7, color="k", label="data, smooth sigma = 11/6 px (your notebook)")
                    a.plot(LAM[m], nw_smooth(f[i], w[i], 20.0)[m], lw=1.3, color="tab:purple", label="data, smooth sigma = 20 px")
                else:
                    a.plot(LAM[m], f[i][m], color="k", lw=0.8, label="data")
                for K, col in zip(Ks, ("tab:blue", "tab:orange", "tab:red")):
                    a.plot(LAM[m], preds[K][m], color=col, lw=1, label=f"K = {K}")
                if kind == "instrument":
                    lo_y, hi_y = np.nanpercentile(preds[Ks[-1]][m], [0.5, 99.5]); pad = max(hi_y - lo_y, 0.3)
                    a.set_ylim(lo_y - pad, hi_y + pad)
                a.set_xlabel("observed wavelength [Angstrom]"); a.set_ylabel("flux [1e-17 erg/s/cm2/A]" if kind == "instrument" else "sky [1e-17 erg/s/cm2/A/arcsec2]", fontsize=8 if kind == "sky" else 10)
                if cidx == 0:
                    a.set_title(f"{lab}\nchi2/pixel {base[i]:.2f} -> {chi_full[i]:.2f} (K = {Kref}, fitted on all pixels)", fontsize=8); a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(figdir, f"template_examples_{kind}.png"), dpi=150); np.savez(os.path.join(cachedir, f"template_examples_{kind}.npz"), **ex)
        plt.close(fig)
        _log(f"{kind} examples saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["prep", "sweep", "figures", "all"])
    ap.add_argument("--cache", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default=None); ap.add_argument("--figcache", default=None)
    ap.add_argument("--kinds", default="instrument,sky"); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--ks", default=None, help="comma list of K for the sweep stage (default: the standard ladder)")
    a = ap.parse_args()
    kinds = a.kinds.split(",")
    if a.stage in ("prep", "all"):
        if "instrument" in kinds:
            prep_instrument(a.cache, a.out)
        if "sky" in kinds:
            prep_sky(a.out)
    if a.stage in ("sweep", "all"):
        for k in kinds:
            sweep(k, a.out, ks=[int(x) for x in a.ks.split(",")] if a.ks else KS, steps=a.steps)
    if a.stage in ("figures", "all"):
        figures(a.out, a.fig, a.figcache, a.cache)
