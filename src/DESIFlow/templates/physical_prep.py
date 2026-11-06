"""
Data for the physical-template experiment: secure galaxies from the DR2 cache with their redshifts, a mean resolution matrix, and the
nuisance-prior numbers.

Selection: cache rows with status == 1, FIBERSTATUS == 0, snr_med > SNR_MIN, then ZALL columns Z, SPECTYPE, ZWARN, DELTACHI2: GALAXY, ZWARN == 0,
DELTACHI2 > 100, ZMIN < z < ZMAX. Held out by TARGETID % 10 == 0. Flux and ivar are the cached healpix coadds on the 7781-pixel brz grid
(overlaps merged); bad pixels have ivar 0.
Mean resolution: the mean of the 12,437 library matrices (R[d, j] = response at offset d-5 pixels from pixel j).

Run:  python -m DESIFlow.templates.physical_prep --cache DR2_CACHE --lib LIB.npy --out OUT [--n-train 80000 --n-eval 10000]
"""
import argparse
import os

import numpy as np

from DESIFlow.data.dr2_cache import load_index, ZALL


def main(cache, lib, out, n_train, n_eval, snr_min=4.0, zmin=0.02, zmax=1.0, seed=0):
    import fitsio
    os.makedirs(out, exist_ok=True)
    idx = load_index(cache)
    snr = np.asarray(np.load(os.path.join(cache, "tgt", "snr_med.npy"), mmap_mode="r"))
    st = np.asarray(np.load(os.path.join(cache, "tgt", "status.npy"), mmap_mode="r"))
    fst = np.asarray(idx["fiberstatus"])[np.asarray(idx["tgt_rows"])]
    tid = np.asarray(idx["targetid"])[np.asarray(idx["tgt_rows"])]
    ok = np.flatnonzero((st == 1) & np.isfinite(snr) & (snr > snr_min) & (fst == 0))
    rng = np.random.default_rng(seed)
    cand = np.sort(rng.choice(ok, min(len(ok), 6 * (n_train + n_eval)), replace=False))
    zr = np.asarray(idx["tgt_rows"])[cand]
    d = fitsio.FITS(ZALL)[1].read(columns=["Z", "SPECTYPE", "ZWARN", "DELTACHI2"], rows=zr)
    stp = np.char.strip(d["SPECTYPE"].astype(str))
    good = (stp == "GALAXY") & (d["ZWARN"] == 0) & (d["DELTACHI2"] > 100) & (d["Z"] > zmin) & (d["Z"] < zmax)
    sel = np.flatnonzero(good); print(f"{len(cand)} candidates, {len(sel)} secure galaxies", flush=True)
    held = (tid[cand[sel]] % 10) == 0
    tr, ev = sel[~held][:n_train], sel[held][:n_eval]
    flux, ivar, mask = (np.load(os.path.join(cache, "tgt", f"{k}.npy"), mmap_mode="r") for k in ("flux", "ivar", "mask"))
    for name, s in (("train", tr), ("eval", ev)):
        rows = cand[s]; f = np.empty((len(rows), 7781), np.float32); w = np.empty_like(f)
        from concurrent.futures import ThreadPoolExecutor

        def read(a):
            r = rows[a:a + 250]
            f[a:a + 250] = flux[r]; w[a:a + 250] = np.where(mask[r] == 0, ivar[r], 0.0)
            return a
        with ThreadPoolExecutor(24) as ex:
            for k, a in enumerate(ex.map(read, range(0, len(rows), 250))):
                if k % 8 == 0:
                    print(name, a, flush=True)
        np.save(os.path.join(out, f"{name}_flux.npy"), f); np.save(os.path.join(out, f"{name}_ivar.npy"), w)
        np.savez(os.path.join(out, f"{name}_meta.npz"), cache_row=rows, z=d["Z"][s].astype(np.float32), deltachi2=d["DELTACHI2"][s].astype(np.float32), snr_med=snr[rows])
    L = np.load(lib, mmap_mode="r"); Rm = np.zeros((11, 7781), np.float64)
    for a in range(0, L.shape[0], 1000):
        Rm += np.asarray(L[a:a + 1000]).astype(np.float64).sum(0)
    np.save(os.path.join(out, "R_mean.npy"), (Rm / L.shape[0]).astype(np.float32))
    print("done", len(tr), len(ev))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--lib", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=80000); ap.add_argument("--n-eval", type=int, default=10000)
    a = ap.parse_args(); main(a.cache, a.lib, a.out, a.n_train, a.n_eval)
