"""
Sampling and summary statistics of DR2 resolution matrices (study of 2026-11, see docs/resolution_matrices.md).

DESI stores a resolution matrix as `ndiag` = 11 diagonals: R[d, j] is the response of observed pixel j + (d - 5)... measured from
the column of true pixel j, so column j of the band is the line-spread function (LSF) of pixel j sampled at offsets -5..+5 pixels
(0.8 A each). `lsf_moments` reduces each column to its normalisation, centroid and width, `sample_groups` draws random
healpix groups and returns, for sky and target rows alike, those statistics together with the exposure information the
coadd file carries (fibre, petal, night, tile), and keeps the full matrices (float16) as a library for later tests.

Run:  python -m DESIFlow.data.resolution_study --cache DR2_CACHE --out OUT_DIR [--groups 700 --per-class 10 --workers 12]
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from DESIFlow.data.dr2_cache import N_PIX, SKIP_HDUS, coadd_path, load_index

ND = 11
NODES = np.arange(60, N_PIX, 120)                                                     # 65 pixels, one every 96 A
PX_A = 0.8                                                                            # A per pixel


def lsf_moments(R, nodes=NODES):
    """R (n, ND, N_PIX) -> column sum, centroid (px) and rms width (px) at the node pixels, each (n, len(nodes))."""
    off = np.arange(ND) - ND // 2
    Rn = np.asarray(R, np.float64)[:, :, nodes]
    s = Rn.sum(1)
    safe = np.where(s != 0, s, 1.0)
    mu = (Rn * off[None, :, None]).sum(1) / safe
    var = (Rn * off[None, :, None] ** 2).sum(1) / safe - mu ** 2
    return s, mu, np.sqrt(np.clip(var, 0, None))


_W = {}


def _init(cache, per_class, seed):
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    _W.update(cache=cache, idx=load_index(cache), per_class=per_class, seed=seed)
    for c in ("tgt", "sky"):
        _W[c + "_rows"] = np.asarray(_W["idx"][c + "_rows"])
        _W[c + "_ngood"] = np.load(os.path.join(cache, c, "ngood.npy"), mmap_mode="r")
        _W[c + "_status"] = np.load(os.path.join(cache, c, "status.npy"), mmap_mode="r")


def sample_group(g):
    import fitsio
    from desispec.io import read_spectra
    from desispec.coaddition import coadd_cameras
    idx, rng = _W["idx"], np.random.default_rng([_W["seed"], g])
    s, e = int(idx["group_start"][g]), int(idx["group_stop"][g])
    ot = np.asarray(idx["objtype"][s:e])
    fst = np.asarray(idx["fiberstatus"][s:e])
    pick = []                                                                            # (class, zall row, cache row)
    for ci, (c, code) in enumerate((("tgt", 1), ("sky", 2))):
        rows = s + np.flatnonzero((ot == code) & (fst == 0))
        crow = np.searchsorted(_W[c + "_rows"], rows)
        ok = (np.asarray(_W[c + "_status"][crow]) == 1) & (np.asarray(_W[c + "_ngood"][crow]) > 0)
        rows, crow = rows[ok], crow[ok]
        if len(rows) == 0:
            continue
        k = rng.choice(len(rows), min(_W["per_class"], len(rows)), replace=False)
        pick += [(ci, int(rows[i]), int(crow[i])) for i in k]
    if not pick:
        return None
    path = coadd_path(str(idx["group_survey"][g]), str(idx["group_program"][g]), int(idx["group_healpix"][g]))
    tid = np.array([idx["targetid"][r] for _, r, _ in pick])
    ftid = fitsio.read(path, ext="FIBERMAP", columns=["TARGETID"])["TARGETID"]
    pos = np.array([np.flatnonzero(ftid == t)[0] for t in tid])
    o = np.argsort(pos)
    pick, tid, pos = [pick[i] for i in o], tid[o], pos[o]
    sp = coadd_cameras(read_spectra(path, rows=pos, skip_hdus=("EXP_FIBERMAP", "SCORES")))
    assert np.array_equal(np.asarray(sp.fibermap["TARGETID"]), tid)
    R = np.asarray(sp.resolution_data["brz"], np.float32)
    s0, mu, sig = lsf_moments(R)
    ex = fitsio.read(path, ext="EXP_FIBERMAP", columns=["TARGETID", "NIGHT", "MJD", "TILEID", "PETAL_LOC", "FIBER", "EXPTIME"])
    rec = {k: [] for k in ("cls", "zall_row", "cache_row", "group", "n_exp", "n_tile", "n_fiber", "petal", "fiber", "night", "mjd", "exptime")}
    for (ci, r, cr), t in zip(pick, tid):
        m = ex[ex["TARGETID"] == t]
        rec["cls"].append(ci); rec["zall_row"].append(r); rec["cache_row"].append(cr); rec["group"].append(g)
        rec["n_exp"].append(len(m)); rec["n_tile"].append(len(np.unique(m["TILEID"]))); rec["n_fiber"].append(len(np.unique(m["FIBER"])))
        rec["petal"].append(int(m["PETAL_LOC"][0]) if len(m) else -1); rec["fiber"].append(int(m["FIBER"][0]) if len(m) else -1)
        rec["night"].append(int(np.median(m["NIGHT"])) if len(m) else -1); rec["mjd"].append(float(np.mean(m["MJD"])) if len(m) else np.nan)
        rec["exptime"].append(float(m["EXPTIME"].sum()) if len(m) else np.nan)
    return rec, s0.astype(np.float32), mu.astype(np.float32), sig.astype(np.float32), R.astype(np.float16)


def sample_groups(cache, out, n_groups=700, per_class=10, workers=12, max_rows=3000, seed=0):
    os.makedirs(out, exist_ok=True)
    idx = load_index(cache)
    size = np.asarray(idx["group_stop"]) - np.asarray(idx["group_start"])
    cand = np.flatnonzero(size <= max_rows)
    groups = np.sort(np.random.default_rng(seed).choice(cand, min(n_groups, len(cand)), replace=False))
    recs, cols, mus, sigs, Rs = [], [], [], [], []
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, _init, (cache, per_class, seed)) as pool:
        for k, res in enumerate(pool.imap_unordered(sample_group, [int(g) for g in groups], chunksize=1), 1):
            if res is not None:
                recs.append(res[0]); cols.append(res[1]); mus.append(res[2]); sigs.append(res[3]); Rs.append(res[4])
            if k % 50 == 0 or k == len(groups):
                print(f"resolution study {k}/{len(groups)} groups, {sum(len(r['cls']) for r in recs)} spectra, {(time.time() - t0) / 60:.1f} min", flush=True)
    table = {key: np.concatenate([np.asarray(r[key]) for r in recs]) for key in recs[0]}
    np.savez(os.path.join(out, "table.npz"), **table, colsum=np.concatenate(cols), centroid=np.concatenate(mus), sigma_px=np.concatenate(sigs), nodes=NODES)
    np.save(os.path.join(out, "library_R_float16.npy"), np.concatenate(Rs))
    print("saved", len(table["cls"]), "spectra to", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", type=int, default=700)
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    sample_groups(a.cache, a.out, a.groups, a.per_class, a.workers)
