"""
Camera overlap test on TARGET spectra: is the b-r / r-z mismatch additive or multiplicative, and how good is an additive stand-in?

The healpix coadds keep the cameras separate (the cache merged them). Per bright target and overlap, the ivar-weighted mean flux of each
camera over a window both cameras cover is measured: b and r over 5762-5798 A, r and z over 7522-7578 A (the telluric A band, which starts
near 7594 A, is left out). With Delta = F_second - F_first and Fbar their mean, the mismatch is modelled as Delta = a + m * Fbar:
an additive offset a, a multiplicative error m, or both.

Also stored per target and camera: 8 window means over the camera interior (b 3800-5650, r 5900-7450, z 7750-9250 A), the shape of the
spectrum. If a multiplicative error m acts on a whole camera, the best constant additive stand-in is the camera mean of m F; the
leftover is m (F(lambda) - mean F), so its rms relative to the flux is |m| * rms(F)/mean(F) over the camera: reported from the stored windows.

Run:  python -m DESIFlow.data.camera_overlap_test --cache DR2_CACHE --out OUT --fig FIG --figcache CACHE [--n 6000 --workers 12]
"""
import argparse
import multiprocessing as mp
import os

import numpy as np

from DESIFlow.data.dr2_cache import coadd_path, load_index

OV = {"br": ("b", "r", 5762.0, 5798.0), "rz": ("r", "z", 7522.0, 7578.0)}
INTERIOR = {"b": (3800.0, 5650.0), "r": (5900.0, 7450.0), "z": (7750.0, 9250.0)}
_W = {}


def _init(cache):
    _W["idx"] = load_index(cache)


def wmean(f, iv, m, sel):
    w = np.where((m[sel] == 0) & (iv[sel] > 0), iv[sel], 0.0)
    sw = w.sum()
    return ((w * f[sel]).sum() / sw, 1.0 / np.sqrt(sw), int((w > 0).sum())) if sw > 0 else (np.nan, np.nan, 0)


def one_group(args):
    import fitsio
    g, rows = args
    idx = _W["idx"]
    path = coadd_path(str(idx["group_survey"][g]), str(idx["group_program"][g]), int(idx["group_healpix"][g]))
    out = []
    try:
        F = fitsio.FITS(path)
        ftid = F["FIBERMAP"].read(columns=["TARGETID"])["TARGETID"]
        wave = {c: F[f"{c.upper()}_WAVELENGTH"].read() for c in "brz"}
        for zr in rows:
            tid = int(idx["targetid"][zr])
            hit = np.flatnonzero(ftid == tid)
            if len(hit) == 0:
                continue
            p = int(hit[0])
            cam = {c: [F[f"{c.upper()}_{k}"][p:p + 1, :][0] for k in ("FLUX", "IVAR", "MASK")] for c in "brz"}
            rec = {"zall_row": zr}
            for name, (c1, c2, lo, hi) in OV.items():
                for c in (c1, c2):
                    v, e, n = wmean(*cam[c], (wave[c] >= lo) & (wave[c] <= hi))
                    rec[f"{name}_{c}_mean"], rec[f"{name}_{c}_err"], rec[f"{name}_{c}_n"] = v, e, n
            for c in "brz":
                lo, hi = INTERIOR[c]
                edges = np.linspace(lo, hi, 9)
                rec[f"shape_{c}"] = np.array([wmean(*cam[c], (wave[c] >= a) & (wave[c] < b))[0] for a, b in zip(edges[:-1], edges[1:])])
            out.append(rec)
    except Exception as e:
        return []
    return out


def sample(cache, n, seed=0):
    idx = load_index(cache)
    snr = np.asarray(np.load(os.path.join(cache, "tgt", "snr_med.npy"), mmap_mode="r"))
    st = np.asarray(np.load(os.path.join(cache, "tgt", "status.npy"), mmap_mode="r"))
    fst = np.asarray(idx["fiberstatus"])[np.asarray(idx["tgt_rows"])]
    ok = (st == 1) & np.isfinite(snr) & (fst == 0)
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(ok & (snr > 3) & (snr <= 10)), np.flatnonzero(ok & (snr > 10))]
    sel = np.concatenate([rng.choice(p, n // 2, replace=False) for p in pools])
    zr = np.asarray(idx["tgt_rows"])[sel]
    gs = np.asarray(idx["group_start"])
    grp = np.searchsorted(gs, zr, side="right") - 1
    return zr, grp, snr[sel]


def collect(cache, out, n=6000, workers=12):
    os.makedirs(out, exist_ok=True)
    zr, grp, snr = sample(cache, n)
    jobs = [(int(g), [int(z) for z in zr[grp == g]]) for g in np.unique(grp)]
    recs = []
    with mp.get_context("spawn").Pool(workers, _init, (cache,)) as pool:
        for k, r in enumerate(pool.imap_unordered(one_group, jobs, chunksize=4), 1):
            recs += r
            if k % 200 == 0:
                print(f"{k}/{len(jobs)} groups, {len(recs)} targets", flush=True)
    keys = [k for k in recs[0] if not k.startswith("shape")]
    table = {k: np.array([r[k] for r in recs]) for k in keys}
    for c in "brz":
        table[f"shape_{c}"] = np.stack([r[f"shape_{c}"] for r in recs])
    np.savez(os.path.join(out, "overlap_table.npz"), **table)
    print("saved", len(recs), "targets", flush=True)


def wls_fit(x, y, e, clip=4.0):
    """y = a + m x with weights 1/e^2 and iterative clipping; returns a, m and their standard errors."""
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    for _ in range(5):
        X = np.stack([np.ones(keep.sum()), x[keep]], 1); w = 1 / (e[keep] ** 2 + (0.003 * np.abs(x[keep])) ** 2)         # 0.3% floor on the scatter
        cov = np.linalg.inv(X.T @ (X * w[:, None])); beta = cov @ (X.T @ (w * y[keep]))
        r = (y - (beta[0] + beta[1] * x)) * np.sqrt(1 / (e ** 2 + (0.003 * np.abs(x)) ** 2))
        keep = np.isfinite(r) & (np.abs(r) < clip * np.nanstd(r[keep]))
    return beta[0], beta[1], np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), int(keep.sum())


def analyse(out, figdir, cachedir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = dict(np.load(os.path.join(out, "overlap_table.npz")))
    os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cache, summary = {}, {}
    for j, (name, (c1, c2, lo, hi)) in enumerate(OV.items()):
        f1, f2 = t[f"{name}_{c1}_mean"], t[f"{name}_{c2}_mean"]
        e1, e2 = t[f"{name}_{c1}_err"], t[f"{name}_{c2}_err"]
        n1, n2 = t[f"{name}_{c1}_n"], t[f"{name}_{c2}_n"]
        fb, d, de = 0.5 * (f1 + f2), f2 - f1, np.hypot(e1, e2)
        ok = (n1 >= 20) & (n2 >= 20) & np.isfinite(d) & (fb > 0.3)
        fb, d, de, f1, f2 = fb[ok], d[ok], de[ok], f1[ok], f2[ok]
        a, m, ea, em, nn = wls_fit(fb, d, de)
        edges = np.array([0.3, 1, 2, 4, 8, 16, 32, 64, 160]); mid = np.sqrt(edges[1:] * edges[:-1])
        bm, bd, bde, br = [], [], [], []
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            s = (fb >= lo_) & (fb < hi_)
            if s.sum() < 15:
                bm.append(np.nan); bd.append(np.nan); bde.append(np.nan); br.append(np.nan); continue
            bm.append(np.median(d[s])); bde.append(1.2533 * 1.4826 * np.median(np.abs(d[s] - np.median(d[s]))) / np.sqrt(s.sum())); br.append(np.median(f2[s] / f1[s])); bd.append(s.sum())
        ax[0, j].hexbin(fb, d, gridsize=60, xscale="log", bins="log", mincnt=1, cmap="Greys", extent=(np.log10(0.3), np.log10(160), -4, 4))
        x = np.logspace(np.log10(0.3), np.log10(160), 50)
        ax[0, j].plot(x, a + m * x, color="tab:red", label=f"fit: a = {a:+.3f} +- {ea:.3f}, m = {100 * m:+.2f}% +- {100 * em:.2f}%")
        ax[0, j].errorbar(mid, bm, bde, fmt="o", color="tab:blue", ms=4, label="binned median")
        ax[0, j].axhline(0, color="gray", lw=0.8); ax[0, j].set_xscale("log"); ax[0, j].set_ylim(-4, 4)
        ax[0, j].set_xlabel(f"mean flux of the two cameras in the overlap [1e-17 erg/s/cm2/A]"); ax[0, j].set_ylabel(f"{c2} - {c1} flux in the overlap [1e-17 erg/s/cm2/A]")
        ax[0, j].set_title(f"{c1}-{c2} overlap {lo:.0f}-{hi:.0f} A: {len(fb)} targets"); ax[0, j].legend(fontsize=7)
        ax[1, j].errorbar(mid, 100 * (np.array(br) - 1), 100 * np.array(bde) / mid, fmt="o-", color="tab:green", ms=4)
        ax[1, j].axhline(100 * m, color="tab:red", ls="--", label="multiplicative term m of the fit"); ax[1, j].axhline(0, color="gray", lw=0.8)
        ax[1, j].set_xscale("log"); ax[1, j].set_ylim(-15, 15); ax[1, j].legend(fontsize=7)
        ax[1, j].set_xlabel("mean flux of the two cameras in the overlap [1e-17 erg/s/cm2/A]"); ax[1, j].set_ylabel(f"median ({c2} / {c1} - 1) [percent]")
        # additive stand-in error: |m| * rms(F)/mean(F) over the camera interior of the second and first camera
        shp = {}
        for c in (c1, c2):
            S = t[f"shape_{c}"][ok]
            good = np.isfinite(S).all(1) & (S.mean(1) > 0.3)
            shp[c] = np.nanstd(S[good], 1) / S[good].mean(1)
        summary[name] = dict(a=a, a_err=ea, m=m, m_err=em, n=len(fb), median_ratio_minus1_bins=[float(v) for v in np.array(br) - 1], bin_mid=[float(v) for v in mid],
                             shape_rms_over_mean_median={c: float(np.median(v)) for c, v in shp.items()}, shape_rms_over_mean_p90={c: float(np.percentile(v, 90)) for c, v in shp.items()},
                             median_flux=float(np.median(fb)))
        cache.update({f"{name}_flux_mean": fb, f"{name}_delta": d, f"{name}_delta_err": de, f"{name}_bin_mid": mid, f"{name}_bin_median_delta": np.array(bm), f"{name}_bin_ratio": np.array(br)})
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "camera_overlap_test.png"), dpi=150); np.savez(os.path.join(cachedir, "camera_overlap_test.npz"), **cache)
    import json
    json.dump(summary, open(os.path.join(out, "overlap_summary.json"), "w"), indent=1)
    for name, s in summary.items():
        print(f"{name}: {s['n']} targets (median overlap flux {s['median_flux']:.2f}); fit Delta = a + m Fbar: a = {s['a']:+.4f} +- {s['a_err']:.4f} [1e-17], m = {100 * s['m']:+.3f}% +- {100 * s['m_err']:.3f}%")
        print("   median ratio-1 by flux bin [%]:", " ".join(f"{100 * v:+.1f}" for v in s["median_ratio_minus1_bins"]), "| bins:", " ".join(f"{v:.1f}" for v in s["bin_mid"]))
        print("   spectral shape rms/mean inside each camera: median", {c: round(v, 3) for c, v in s["shape_rms_over_mean_median"].items()}, "p90", {c: round(v, 3) for c, v in s["shape_rms_over_mean_p90"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["collect", "analyse", "all"])
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True); ap.add_argument("--fig", default=None); ap.add_argument("--figcache", default=None)
    ap.add_argument("--n", type=int, default=6000); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.stage in ("collect", "all"):
        collect(a.cache, a.out, a.n, a.workers)
    if a.stage in ("analyse", "all"):
        analyse(a.out, a.fig, a.figcache)
