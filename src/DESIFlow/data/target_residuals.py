"""
Per-camera residuals of bright, secure targets against their Redrock models (for testing calibration nuisance terms).

For targets with a secure Redrock fit (ZWARN == 0, large DELTACHI2; GALAXY with z < 1.5, where the IGM does not enter the observed
range, and STAR), the model is rebuilt from the zall COEFF, Z and SUBTYPE with the installed Redrock templates (Template.eval, then the
coadd's resolution matrix of each camera), and stored per camera with the data ivar and mask: concatenated over b (2751 px), r (2326)
and z (2881). The rebuilt model is validated against the CHI2 of zall.

Run:  python -m DESIFlow.data.target_residuals --cache DR2_CACHE --out OUT [--n-gal 3000 --n-star 1000 --workers 12]
"""
import argparse
import multiprocessing as mp
import os

import numpy as np

from DESIFlow.data.dr2_cache import coadd_path, load_index, ZALL

NPIX = {"b": 2751, "r": 2326, "z": 2881}
START = {"b": 3600.0, "r": 5760.0, "z": 7520.0}
_W = {}


def _init(cache):
    from redrock.templates import load_templates
    _W["idx"] = load_index(cache)
    _W["tpl"] = {(t.template_type, t.sub_type): t for t in load_templates()}


def one_group(args):
    import fitsio
    from desispec.resolution import Resolution
    g, rows, info = args
    idx, tpl = _W["idx"], _W["tpl"]
    path = coadd_path(str(idx["group_survey"][g]), str(idx["group_program"][g]), int(idx["group_healpix"][g]))
    out = []
    try:
        F = fitsio.FITS(path)
        ftid = F["FIBERMAP"].read(columns=["TARGETID"])["TARGETID"]
        wave = {c: F[f"{c.upper()}_WAVELENGTH"].read() for c in "brz"}
        for zr in rows:
            z, st, sub, coeff, chi2 = info[zr]
            key = ("GALAXY", "") if st == "GALAXY" else ("STAR", sub)
            if key not in tpl:
                continue
            hit = np.flatnonzero(ftid == int(idx["targetid"][zr]))
            if len(hit) == 0:
                continue
            p = int(hit[0])
            res, ivs, mds, chi_self = [], [], [], 0.0
            for c in "brz":
                f, iv, mk = (F[f"{c.upper()}_{k}"][p:p + 1, :][0] for k in ("FLUX", "IVAR", "MASK"))
                R = Resolution(F[f"{c.upper()}_RESOLUTION"][p:p + 1, :, :][0])
                m = tpl[key].eval(coeff, wave[c].astype(np.float64), z, R=R)
                w = np.where(mk == 0, iv, 0.0)
                chi_self += float((w * (f - m) ** 2).sum())
                res.append(f - m); ivs.append(w); mds.append(m)
            out.append(dict(zall_row=zr, spectype=st, z=z, chi2_zall=chi2, chi2_rebuilt=chi_self, resid=np.concatenate(res).astype(np.float32),
                            ivar=np.concatenate(ivs).astype(np.float32), model=np.concatenate(mds).astype(np.float32)))
    except Exception as e:
        return []
    return out


def select(cache, n_gal, n_star, seed=0):
    import fitsio
    idx = load_index(cache)
    snr = np.asarray(np.load(os.path.join(cache, "tgt", "snr_med.npy"), mmap_mode="r"))
    st = np.asarray(np.load(os.path.join(cache, "tgt", "status.npy"), mmap_mode="r"))
    fst = np.asarray(idx["fiberstatus"])[np.asarray(idx["tgt_rows"])]
    ok = np.flatnonzero((st == 1) & np.isfinite(snr) & (snr > 8) & (fst == 0))
    rng = np.random.default_rng(seed)
    cand = np.sort(rng.choice(ok, min(len(ok), 25 * (n_gal + n_star)), replace=False))
    zr = np.asarray(idx["tgt_rows"])[cand]
    h = fitsio.FITS(ZALL)[1]
    d = h.read(columns=["Z", "SPECTYPE", "SUBTYPE", "ZWARN", "DELTACHI2", "COEFF", "CHI2"], rows=zr)
    stp, sub = np.char.strip(d["SPECTYPE"].astype(str)), np.char.strip(d["SUBTYPE"].astype(str))
    good = (d["ZWARN"] == 0) & (d["DELTACHI2"] > 500)
    gal = np.flatnonzero(good & (stp == "GALAXY") & (d["Z"] < 1.5) & (d["Z"] > 0.0))[:n_gal]
    star = np.flatnonzero(good & (stp == "STAR") & (sub != "") & (sub != "N/A"))[:n_star]
    pick = np.concatenate([gal, star])
    info = {int(zr[i]): (float(d["Z"][i]), str(stp[i]), str(sub[i]), np.asarray(d["COEFF"][i], np.float64), float(d["CHI2"][i])) for i in pick}
    print(f"selected {len(gal)} galaxies and {len(star)} stars from {len(cand)} bright candidates", flush=True)
    return info


def collect(cache, out, n_gal=3000, n_star=1000, workers=12):
    os.makedirs(out, exist_ok=True)
    idx = load_index(cache)
    info = select(cache, n_gal, n_star)
    zr = np.array(sorted(info))
    grp = np.searchsorted(np.asarray(idx["group_start"]), zr, side="right") - 1
    jobs = [(int(g), [int(z) for z in zr[grp == g]], {int(z): info[int(z)] for z in zr[grp == g]}) for g in np.unique(grp)]
    recs = []
    with mp.get_context("spawn").Pool(workers, _init, (cache,)) as pool:
        for k, r in enumerate(pool.imap_unordered(one_group, jobs, chunksize=2), 1):
            recs += r
            if k % 200 == 0:
                print(f"{k}/{len(jobs)} groups, {len(recs)} targets", flush=True)
    ratio = np.array([r["chi2_rebuilt"] / r["chi2_zall"] for r in recs])
    print(f"collected {len(recs)} targets; rebuilt chi2 / zall CHI2: median {np.median(ratio):.4f}, 16-84% {np.percentile(ratio, 16):.4f}-{np.percentile(ratio, 84):.4f}", flush=True)
    np.savez(os.path.join(out, "target_residuals.npz"), resid=np.stack([r["resid"] for r in recs]), ivar=np.stack([r["ivar"] for r in recs]), model=np.stack([r["model"] for r in recs]),
             zall_row=np.array([r["zall_row"] for r in recs]), spectype=np.array([r["spectype"] for r in recs]), z=np.array([r["z"] for r in recs]),
             chi2_zall=np.array([r["chi2_zall"] for r in recs]), chi2_rebuilt=np.array([r["chi2_rebuilt"] for r in recs]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-gal", type=int, default=3000); ap.add_argument("--n-star", type=int, default=1000); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    collect(a.cache, a.out, a.n_gal, a.n_star, a.workers)
