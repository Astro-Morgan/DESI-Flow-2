"""
DR2 (Loa) healpix spectral cache built from zall-pix-loa.fits (ZALL below, read-only) and the healpix coadd files.

Two caches, each a directory of row-matched files, so row i of every file in a directory is row i of that directory's
metadata.h5 and no index lookup is needed:
    tgt/   the OBJTYPE == TGT rows of zall (zall order)          sky/   the OBJTYPE == SKY rows (kept apart for sky templates)
        flux.npy ivar.npy mask.npy   float32 / float32 / uint8 (n, 7781); np.load(..., mmap_mode="r")
        status.npy   uint8: 0 not processed, 1 ok, 2 target absent from its coadd file, 3 coadd file missing
        snr_med.npy  float32: median over good pixels of flux * sqrt(ivar);  ngood.npy uint16: good pixel count
        metadata.h5  uncompressed; group "zall" holds every zall column except COEFF, plus zall_row (the row in the
                     original catalog); finalize adds status, snr_med, ngood (and sc_index in tgt/) as columns
    tgt/ also holds the super-coadds across (survey, program), see supercoadd.
Rows of OBJTYPE N/A, BAD and NON are not cached.

Pixels: the official camera coadd (desispec coadd_cameras) on the 0.8 A linear grid 3600-9824 A (7781 px). mask is
binary, 1 = bad (DESI mask != 0, or ivar <= 0 / non-finite after the camera coadd), and bad pixels carry flux = ivar = 0,
exactly how the masker hides pixels. The pixel mask is therefore a pure function of the DESI bitmask.
Resolution matrices are a separate pass (stage resolution): float16, (n, 11, 7781) per spectrum = 171 kB (float16 keeps
rms 3e-4 of R @ f, mean error 1e-7; decimating the matrix along wavelength or a global PCA does not work, see the notes
in the memory node dr2-spectral-cache). sky/resolution.npy covers every sky row; for tgt/ pass --rows with the
compact row numbers of the secure-z sample once it exists (tgt/resolution_<tag>.npy + resolution_<tag>_rows.npy).

Stages (python -m DESIFlow.data.dr2_cache <stage>):  index -> alloc -> build -> supercoadd -> metadata -> finalize,
and resolution (--cls sky|tgt [--rows rows.npy --tag name]) at any time after build.
All of them resume where they stopped. Environment: DESI_DR2_CACHE (output directory), DESI_ZALL_LOA, DESI_HEALPIX_LOA.
"""
import argparse
import glob
import multiprocessing as mp
import os
import subprocess
import sys
import time
import warnings

import numpy as np

ZALL = os.environ.get("DESI_ZALL_LOA", "/global/cfs/cdirs/desi/spectro/redux/loa/zcatalog/v1/zall-pix-loa.fits")
HEALPIX = os.environ.get("DESI_HEALPIX_LOA", "/global/cfs/cdirs/desi/spectro/redux/loa/healpix")
N_PIX = 7781
OBJ = {"N/A": 0, "TGT": 1, "SKY": 2, "BAD": 3, "NON": 4}
# desispec reads these HDUs by name; the resolution matrices are about 4/5 of the reading and coadding time
SKIP_HDUS = ("EXP_FIBERMAP", "SCORES", "RESOLUTION", "B_RESOLUTION", "R_RESOLUTION", "Z_RESOLUTION")
CHUNK = 1000                                                                          # spectra coadded at once per worker
SPEC_ARRAYS = {"flux": "f4", "ivar": "f4", "mask": "u1"}
ROW_ARRAYS = {"status": "u1", "snr_med": "f4", "ngood": "u2"}
CLASSES = {"tgt": OBJ["TGT"], "sky": OBJ["SKY"]}


def cache_dir(path=None):
    path = path or os.environ.get("DESI_DR2_CACHE")
    if not path:
        raise SystemExit("set DESI_DR2_CACHE or pass --cache")
    return path


def cdir(cache, cls):
    return os.path.join(cache, cls)


def _env():
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")                        # Lustre


# ------------------------------------------------------------------------------------------------ index
def build_index(cache, chunk=4_000_000):
    """Read the key columns of zall once and save them as npy files in cache/index (rows of a group must be contiguous)."""
    import fitsio
    out = os.path.join(cache, "index")
    os.makedirs(out, exist_ok=True)
    h = fitsio.FITS(ZALL)[1]
    n = h.get_nrows()
    tid, obj, fst = np.empty(n, np.int64), np.empty(n, np.uint8), np.empty(n, np.int32)
    starts, keys = [], []
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d = h.read(columns=["TARGETID", "SURVEY", "PROGRAM", "HEALPIX", "OBJTYPE", "COADD_FIBERSTATUS"], rows=np.arange(a, b))
        tid[a:b], fst[a:b] = d["TARGETID"], d["COADD_FIBERSTATUS"]
        ot = np.char.strip(d["OBJTYPE"].astype("U8"))
        obj[a:b] = _codes(ot)
        key = np.char.add(np.char.add(np.char.strip(d["SURVEY"].astype("U8")), "|"), np.char.strip(d["PROGRAM"].astype("U8")))
        key = np.char.add(np.char.add(key, "|"), d["HEALPIX"].astype("U9"))
        ch = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        starts += (ch + a).tolist()
        keys += key[ch].tolist()
        print(f"index {b}/{n}", flush=True)
    keep = [i for i in range(len(keys)) if i == 0 or keys[i] != keys[i - 1]]          # a group spanning two chunks
    start = np.array([starts[i] for i in keep], np.int64)
    key = [keys[i] for i in keep]
    if len(set(key)) != len(key):
        raise RuntimeError("zall rows of a (survey, program, healpix) group are not contiguous")
    sv, pr, hp = zip(*(k.split("|") for k in key))
    np.save(os.path.join(out, "targetid.npy"), tid)
    np.save(os.path.join(out, "objtype.npy"), obj)
    np.save(os.path.join(out, "fiberstatus.npy"), fst)
    np.save(os.path.join(out, "group_start.npy"), start)
    np.save(os.path.join(out, "group_stop.npy"), np.r_[start[1:], n])
    np.save(os.path.join(out, "group_survey.npy"), np.array(sv, "U8"))
    np.save(os.path.join(out, "group_program.npy"), np.array(pr, "U8"))
    np.save(os.path.join(out, "group_healpix.npy"), np.array(hp, np.int64))
    print(f"index done: {n} rows, {len(key)} groups; TGT {int((obj == 1).sum())} SKY {int((obj == 2).sum())}", flush=True)


def _codes(ot):
    out = np.full(ot.shape, 255, np.uint8)
    for k, v in OBJ.items():
        out[ot == k] = v
    return out


def load_index(cache):
    d = os.path.join(cache, "index")
    return {os.path.basename(f)[:-4]: np.load(f, mmap_mode="r") for f in glob.glob(os.path.join(d, "*.npy"))}


# ------------------------------------------------------------------------------------------------ files
def alloc(cache, stripe=8):
    """Save the zall rows of each class (index/tgt_rows.npy, sky_rows.npy) and create the row-matched npy files (the
    directories are striped across OSTs first, ignored if lfs is missing). The files are sparse until written."""
    idx = load_index(cache)
    for cls, code in CLASSES.items():
        rows = np.flatnonzero(np.asarray(idx["objtype"]) == code)
        np.save(os.path.join(cache, "index", f"{cls}_rows.npy"), rows)
        d = cdir(cache, cls)
        os.makedirs(d, exist_ok=True)
        try:
            subprocess.run(["lfs", "setstripe", "-c", str(stripe), "-S", "4m", d], check=False, capture_output=True)
        except FileNotFoundError:
            pass
        for name, dt in {**SPEC_ARRAYS, **ROW_ARRAYS}.items():
            path = os.path.join(d, name + ".npy")
            if not os.path.exists(path):
                shape = (len(rows), N_PIX) if name in SPEC_ARRAYS else (len(rows),)
                np.lib.format.open_memmap(path, mode="w+", dtype=dt, shape=shape).flush()
                print("created", path, shape, dt, flush=True)


class RowWriter:
    """pwrite rows of a row-matched npy file (no shared mmap pages between workers, only the rows we own are touched)."""

    def __init__(self, path):
        m = np.load(path, mmap_mode="r")
        self.offset, self.rowbytes, self.dtype = m.offset, int(np.prod(m.shape[1:], dtype=np.int64)) * m.dtype.itemsize, m.dtype
        del m
        self.fd = os.open(path, os.O_WRONLY)

    def write(self, rows, data):
        """rows sorted ascending (n,), data (n, ...) of self.dtype; contiguous rows go out in one pwrite."""
        data = np.ascontiguousarray(data, self.dtype)
        rows = np.asarray(rows)
        br = np.flatnonzero(np.r_[True, np.diff(rows) != 1, True])
        for a, b in zip(br[:-1], br[1:]):
            buf = memoryview(data[a:b].reshape(-1).view(np.uint8))
            pos, off = 0, self.offset + int(rows[a]) * self.rowbytes
            while pos < len(buf):
                pos += os.pwrite(self.fd, buf[pos:], off + pos)

    def close(self):
        os.close(self.fd)


def coadd_path(survey, program, healpix):
    return f"{HEALPIX}/{survey}/{program}/{healpix // 100}/{healpix}/coadd-{survey}-{program}-{healpix}.fits"


# ------------------------------------------------------------------------------------------------ build
def camera_coadd(path, positions):
    """Official desispec camera coadd of the given fibermap rows of a healpix coadd file -> (targetid, flux, ivar, mask)."""
    from desispec.io import read_spectra
    from desispec.coaddition import coadd_cameras
    sp = coadd_cameras(read_spectra(path, rows=positions, skip_hdus=SKIP_HDUS))
    return np.asarray(sp.fibermap["TARGETID"]), sp.flux["brz"], sp.ivar["brz"], sp.mask["brz"]


def to_cache_format(flux, ivar, mask):
    """float64 coadd -> float32 flux / ivar with bad pixels zeroed, uint8 binary mask (1 = bad), median S/N, good count."""
    bad = (mask != 0) | ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar <= 0)
    flux, ivar = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(ivar, nan=0.0, posinf=0.0, neginf=0.0)
    flux, ivar = np.where(bad, 0.0, flux), np.where(bad, 0.0, ivar)
    snr = np.where(bad, np.nan, flux * np.sqrt(ivar))
    ngood = (~bad).sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                               # all-bad rows
        med = np.nanmedian(snr, axis=1)
    return flux.astype(np.float32), ivar.astype(np.float32), bad.astype(np.uint8), med.astype(np.float32), ngood.astype(np.uint16)


_W = {}


def _worker_init(cache):
    _env()
    _W["cache"], _W["idx"] = cache, load_index(cache)
    _W["w"] = {c: {k: RowWriter(os.path.join(cdir(cache, c), k + ".npy")) for k in list(SPEC_ARRAYS) + list(ROW_ARRAYS)} for c in CLASSES}
    os.makedirs(os.path.join(cache, "ledger"), exist_ok=True)
    _W["log"] = open(os.path.join(cache, "ledger", f"w{os.getpid()}.log"), "a", buffering=1)


def process_group(g):
    idx, w, t0 = _W["idx"], _W["w"], time.time()
    s, e = int(idx["group_start"][g]), int(idx["group_stop"][g])
    ot = np.asarray(idx["objtype"][s:e])
    want = s + np.flatnonzero(np.isin(ot, list(CLASSES.values())))
    crow = {c: np.asarray(idx[c + "_rows"]) for c in CLASSES}                         # zall rows of each class, sorted
    counts = [0, 0, 0]                                                                # ok, absent from file, file missing

    def put_status(rows, code):
        for c, v in CLASSES.items():
            sel = np.asarray(idx["objtype"][rows]) == v
            if sel.any():
                w[c]["status"].write(np.searchsorted(crow[c], rows[sel]), np.full(int(sel.sum()), code, np.uint8))

    if len(want):
        path = coadd_path(str(idx["group_survey"][g]), str(idx["group_program"][g]), int(idx["group_healpix"][g]))
        tid = np.asarray(idx["targetid"][want])
        if not os.path.exists(path):
            put_status(want, 3)
            counts[2] = len(want)
        else:
            import fitsio
            ftid = fitsio.read(path, ext="FIBERMAP", columns=["TARGETID"])["TARGETID"]
            order = np.argsort(ftid)
            loc = np.searchsorted(ftid, tid, sorter=order).clip(0, len(ftid) - 1)
            pos = order[loc]
            found = ftid[pos] == tid
            if (~found).any():
                put_status(want[~found], 2)
                counts[1] = int((~found).sum())
            want, pos = want[found], pos[found]
            srt = np.argsort(pos)
            want, pos = want[srt], pos[srt]
            for a in range(0, len(want), CHUNK):
                rows, p = want[a:a + CHUNK], pos[a:a + CHUNK]
                got, fl, iv, mk = camera_coadd(path, p)
                if not np.array_equal(got, np.asarray(idx["targetid"][rows])):
                    raise RuntimeError(f"group {g}: row order from read_spectra differs from the request")
                fl, iv, mk, med, ng = to_cache_format(fl, iv, mk)
                kind = np.asarray(idx["objtype"][rows])
                for c, v in CLASSES.items():
                    sel = kind == v
                    if not sel.any():
                        continue
                    cr, o = np.searchsorted(crow[c], rows[sel]), np.argsort(np.searchsorted(crow[c], rows[sel]))
                    for k, arr in (("flux", fl), ("ivar", iv), ("mask", mk), ("snr_med", med), ("ngood", ng)):
                        w[c][k].write(cr[o], arr[sel][o])
                    w[c]["status"].write(cr[o], np.ones(int(sel.sum()), np.uint8))
                counts[0] += len(rows)
    _W["log"].write(f"{g} ok {counts[0]} absent {counts[1]} nofile {counts[2]} {time.time() - t0:.1f}s\n")
    return g, counts


_RES = {}


def _res_init(cache, cls, out_path, subset):
    _env()
    _RES.update(idx=load_index(cache), cls=cls, crow=np.load(os.path.join(cache, "index", f"{cls}_rows.npy")), sub=subset,
                out=RowWriter(out_path), log=open(os.path.join(cache, "ledger", f"res_{cls}_{os.getpid()}.log"), "a", buffering=1))


def process_group_res(g):
    """Resolution matrices (brz, float16) of the rows of one group that belong to the class (and the subset, if any)."""
    import fitsio
    from desispec.io import read_spectra
    from desispec.coaddition import coadd_cameras
    idx, crow, sub, t0, n = _RES["idx"], _RES["crow"], _RES["sub"], time.time(), 0
    s, e = int(idx["group_start"][g]), int(idx["group_stop"][g])
    rows = s + np.flatnonzero(np.asarray(idx["objtype"][s:e]) == CLASSES[_RES["cls"]])
    comp = np.searchsorted(crow, rows)
    if sub is not None:
        keep = np.isin(comp, sub)
        rows, comp = rows[keep], comp[keep]
    if len(rows):
        path = coadd_path(str(idx["group_survey"][g]), str(idx["group_program"][g]), int(idx["group_healpix"][g]))
        if os.path.exists(path):
            ftid = fitsio.read(path, ext="FIBERMAP", columns=["TARGETID"])["TARGETID"]
            order = np.argsort(ftid)
            pos = order[np.searchsorted(ftid, np.asarray(idx["targetid"][rows]), sorter=order).clip(0, len(ftid) - 1)]
            ok = ftid[pos] == np.asarray(idx["targetid"][rows])
            rows, comp, pos = rows[ok], comp[ok], pos[ok]
            srt = np.argsort(pos)
            rows, comp, pos = rows[srt], comp[srt], pos[srt]
            a = 0
            while a < len(rows):
                # read a contiguous block of the file (slab reads are fast, scattered rows of a big file are not), keep the wanted rows
                b = int(np.searchsorted(pos, pos[a] + 2 * CHUNK, side="left"))
                b = min(max(b, a + 1), len(rows))
                lo, hi = int(pos[a]), int(pos[b - 1])
                sp = read_spectra(path, rows=np.arange(lo, hi + 1), skip_hdus=("EXP_FIBERMAP", "SCORES"))[pos[a:b] - lo]
                sp = coadd_cameras(sp)
                if not np.array_equal(np.asarray(sp.fibermap["TARGETID"]), np.asarray(idx["targetid"][rows[a:b]])):
                    raise RuntimeError(f"group {g}: row order from read_spectra differs from the request")
                c = comp[a:b]
                out = c if sub is None else np.searchsorted(sub, c)
                o = np.argsort(out)
                _RES["out"].write(out[o], np.asarray(sp.resolution_data["brz"], np.float16)[o])
                n += len(c)
                a = b
    _RES["log"].write(f"{g} ok {n} {time.time() - t0:.1f}s\n")
    return g, n


def resolution(cache, cls, workers=20, rows_file=None, tag=None):
    """Write float16 resolution matrices for every row of the class, or for the compact rows listed in rows_file."""
    sub = None if rows_file is None else np.unique(np.load(rows_file))
    name = "resolution" if tag is None else f"resolution_{tag}"
    path = os.path.join(cdir(cache, cls), name + ".npy")
    n = len(np.load(os.path.join(cache, "index", f"{cls}_rows.npy"))) if sub is None else len(sub)
    if sub is not None:
        np.save(os.path.join(cdir(cache, cls), name + "_rows.npy"), sub)
    if not os.path.exists(path):
        np.lib.format.open_memmap(path, mode="w+", dtype="f2", shape=(n, 11, N_PIX)).flush()
    idx = load_index(cache)
    size = np.asarray(idx["group_stop"]) - np.asarray(idx["group_start"])
    done = set()
    for f in glob.glob(os.path.join(cache, "ledger", f"res_{cls}_*.log")):
        with open(f) as fh:
            done.update(int(l.split()[0]) for l in fh if l.strip())
    todo = [int(g) for g in np.argsort(-size, kind="stable") if int(g) not in done]
    print(f"resolution {cls}: {n} rows, {len(todo)} groups to do, {workers} workers -> {path}", flush=True)
    t0, rows = time.time(), 0
    with mp.get_context("spawn").Pool(workers, _res_init, (cache, cls, path, sub)) as pool:
        for k, (g, m) in enumerate(pool.imap_unordered(process_group_res, todo, chunksize=1), 1):
            rows += m
            if k % 100 == 0 or k == len(todo):
                print(f"resolution {k}/{len(todo)} groups, {rows} spectra ({rows / (time.time() - t0):.0f}/s), {(time.time() - t0) / 60:.1f} min", flush=True)


def done_groups(cache):
    done = set()
    for f in glob.glob(os.path.join(cache, "ledger", "w*.log")):
        with open(f) as fh:
            done.update(int(l.split()[0]) for l in fh if l.strip() and l.split()[1] == "ok")
    return done


def build(cache, workers=20, groups=None):
    idx = load_index(cache)
    size = np.asarray(idx["group_stop"]) - np.asarray(idx["group_start"])
    todo = np.arange(len(size)) if groups is None else np.asarray(groups)
    done = done_groups(cache)
    todo = [int(g) for g in todo[np.argsort(-size[todo], kind="stable")] if int(g) not in done]
    print(f"build: {len(todo)} groups to do ({len(done)} already done), {workers} workers", flush=True)
    t0, rows, absent = time.time(), 0, 0
    with mp.get_context("spawn").Pool(workers, _worker_init, (cache,)) as pool:
        for k, (g, c) in enumerate(pool.imap_unordered(process_group, todo, chunksize=1), 1):
            rows, absent = rows + c[0], absent + c[1] + c[2]
            if k % 100 == 0 or k == len(todo):
                dt = time.time() - t0
                print(f"build {k}/{len(todo)} groups, {rows} spectra ({rows / dt:.0f}/s), {absent} not found, {dt / 60:.1f} min", flush=True)


# ------------------------------------------------------------------------------------------------ super-coadd
def combine(flux, ivar, bad, use):
    """
    Inverse-variance combination of m coadds of one target, pixel by pixel (the official coadd's estimator).

    flux, ivar (m, N) with bad pixels already zeroed, bad (m, N) bool, use (m,) bool contributions that may enter.
    Returns flux (N,), ivar (N,), bad (N,) bool, chi2_red (reduced chi2 of the contributions about the combination over
    the pixels where two or more contribute; nan if there are none) and the median S/N of the result.
    """
    w = np.where(bad | ~use[:, None], 0.0, ivar.astype(np.float64))
    W = w.sum(0)
    ok = W > 0
    f = np.where(ok, (w * flux).sum(0) / np.where(ok, W, 1.0), 0.0)
    n = (w > 0).sum(0)
    dof = np.clip(n - 1, 0, None).sum()
    chi2 = (w * (flux - f) ** 2).sum()
    snr = f[ok] * np.sqrt(W[ok])
    return f, W, ~ok, (chi2 / dof if dof > 0 else np.nan), (np.median(snr) if len(snr) else np.nan)


def sc_plan(cache):
    """Targets with two or more tgt rows -> (mem, start, count, tid): the member tgt rows of those targets, concatenated
    in TARGETID order; target j owns mem[start[j]:start[j] + count[j]]."""
    idx = load_index(cache)
    tid = np.asarray(idx["targetid"])[np.asarray(idx["tgt_rows"])]
    rows = np.argsort(tid, kind="stable")
    tid = tid[rows]
    first = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    count = np.diff(np.r_[first, len(tid)])
    multi = count > 1
    first, count = first[multi], count[multi]
    start = np.r_[0, np.cumsum(count)[:-1]]
    mem = rows[np.repeat(first - start, count) + np.arange(int(count.sum()))]
    return mem, start, count, tid[first]


def _sc_init(cache):
    _env()
    d = cdir(cache, "tgt")
    r = {k: np.load(os.path.join(d, k + ".npy"), mmap_mode="r") for k in ("flux", "ivar", "mask", "status")}
    idx = load_index(cache)
    fst = np.asarray(idx["fiberstatus"])[np.asarray(idx["tgt_rows"])]
    _W.update(cache=cache, r=r, fst=fst, w={k: RowWriter(os.path.join(d, f"sc_{k}.npy")) for k in ("flux", "ivar", "mask")})


def sc_chunk(args):
    a, b, members, first, count = args                                                # members / first are local to the job
    r = _W["r"]
    out = {k: [] for k in ("f", "i", "m", "chi2", "snr", "nused", "fallback")}
    for j in range(a, b):
        mem = members[first[j - a]:first[j - a] + count[j - a]]
        flux, ivar, bad = np.asarray(r["flux"][mem]), np.asarray(r["ivar"][mem]), np.asarray(r["mask"][mem]) != 0
        good_status = np.asarray(r["status"][mem]) == 1
        use = good_status & (_W["fst"][mem] == 0)
        fallback = not use.any()
        if fallback:
            use = good_status if good_status.any() else np.ones(len(mem), bool)
        f, W, badpx, chi2, snr = combine(flux, ivar, bad, use)
        for k, v in zip(out, (f.astype(np.float32), W.astype(np.float32), badpx.astype(np.uint8), chi2, snr, int(use.sum()), fallback)):
            out[k].append(v)
    rows = np.arange(a, b)
    _W["w"]["flux"].write(rows, np.stack(out["f"]))
    _W["w"]["ivar"].write(rows, np.stack(out["i"]))
    _W["w"]["mask"].write(rows, np.stack(out["m"]))
    return a, b, np.array(out["chi2"], np.float32), np.array(out["snr"], np.float32), np.array(out["nused"], np.uint8), np.array(out["fallback"], bool)


def supercoadd(cache, workers=20, per_chunk=250):
    """
    Super-coadds across (survey, program) for every TARGETID that has two or more rows in tgt/ (files in tgt/).

    Single-row targets would just repeat their healpix coadd, so they are not stored: tgt/sc_index.npy (int32, one entry per
    tgt row) is the row of the sc_* arrays, or -1 when the super-coadd IS the row's own healpix coadd. Contributions are
    pixelwise inverse-variance weighted; a row enters only if its status is ok and COADD_FIBERSTATUS == 0 (if no row
    qualifies all rows with status ok enter and sc_fallback is set). No S/N gate: a noisy coadd gets a small weight.
    Tables (sc_*.npy, one entry per stored target): targetid, n_total, n_used, fallback, chi2_red (about the combination,
    a flag for variable objects or calibration mismatch), snr_med, member_start (offsets into sc_members.npy, tgt rows).
    """
    members, first, count, tid = sc_plan(cache)
    m, d = len(first), cdir(cache, "tgt")
    print(f"supercoadd: {m} targets with 2+ rows ({int(count.sum())} rows)", flush=True)
    path = lambda k: os.path.join(d, k + ".npy")
    for k, dt in (("flux", "f4"), ("ivar", "f4"), ("mask", "u1")):
        if not os.path.exists(path("sc_" + k)):
            np.lib.format.open_memmap(path("sc_" + k), mode="w+", dtype=dt, shape=(m, N_PIX)).flush()
    sc_index = np.full(len(load_index(cache)["tgt_rows"]), -1, np.int32)
    sc_index[members] = np.repeat(np.arange(m, dtype=np.int32), count)
    np.save(path("sc_index"), sc_index)
    np.save(path("sc_members"), members)
    np.save(path("sc_member_start"), first)
    np.save(path("sc_targetid"), tid)
    np.save(path("sc_n_total"), count.astype(np.uint16))
    stats = {k: np.full(m, np.nan if k in ("chi2_red", "snr_med") else 0, np.float32 if k in ("chi2_red", "snr_med") else np.uint8) for k in ("chi2_red", "snr_med", "n_used", "fallback")}
    jobs = []
    for a in range(0, m, per_chunk):
        b = min(a + per_chunk, m)
        jobs.append((a, b, members[first[a]:first[b - 1] + count[b - 1]], first[a:b] - first[a], count[a:b]))
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, _sc_init, (cache,)) as pool:
        for k, (a, b, chi2, snr, nused, fb) in enumerate(pool.imap_unordered(sc_chunk, jobs), 1):
            stats["chi2_red"][a:b], stats["snr_med"][a:b], stats["n_used"][a:b], stats["fallback"][a:b] = chi2, snr, nused, fb
            if k % 50 == 0 or k == len(jobs):
                print(f"supercoadd {k}/{len(jobs)} chunks, {(time.time() - t0) / 60:.1f} min", flush=True)
    for k, v in stats.items():
        np.save(path("sc_" + k), v)


# ------------------------------------------------------------------------------------------------ metadata
def _h5_dtype(dt):
    """fitsio hands strings back as unicode; hdf5 stores them as fixed-width bytes."""
    return np.dtype(f"S{dt.itemsize // 4}") if dt.kind == "U" else dt.newbyteorder("=")


def metadata(cache, chunk=1_000_000):
    """<class>/metadata.h5 (uncompressed, group "zall"): every zall column except COEFF for the rows of the class, plus zall_row."""
    import fitsio
    import h5py
    _env()
    h = fitsio.FITS(ZALL)[1]
    cols = [c for c in h.get_colnames() if c != "COEFF"]
    first = h.read(columns=cols, rows=[0])
    for cls in CLASSES:
        rows = np.load(os.path.join(cache, "index", f"{cls}_rows.npy"))
        n = len(rows)
        with h5py.File(os.path.join(cdir(cache, cls), "metadata.h5"), "w") as f:
            g = f.create_group("zall")
            for c in cols:
                g.create_dataset(c, shape=(n,) + first[c].shape[1:], dtype=_h5_dtype(first[c].dtype))
            g.create_dataset("zall_row", data=rows)
            for a in range(0, n, chunk):
                d = h.read(columns=cols, rows=rows[a:a + chunk])
                for c in cols:
                    g[c][a:a + len(d)] = np.char.encode(d[c], "ascii") if d[c].dtype.kind == "U" else d[c]
                print(f"metadata {cls} {a + len(d)}/{n}", flush=True)


def finalize(cache):
    """Append the derived row-matched columns (status, snr_med, ngood; sc_index in tgt/) to each metadata.h5 (group "derived")."""
    import h5py
    for cls in CLASSES:
        d = cdir(cache, cls)
        with h5py.File(os.path.join(d, "metadata.h5"), "a") as f:
            g = f.require_group("derived")
            for k in list(ROW_ARRAYS) + ["sc_index"]:
                p = os.path.join(d, k + ".npy")
                if os.path.exists(p):
                    if k in g:
                        del g[k]
                    g[k] = np.load(p, mmap_mode="r")[:]
        print("finalized", cls, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["index", "alloc", "build", "supercoadd", "metadata", "finalize", "resolution"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--cls", choices=list(CLASSES), default="sky", help="resolution stage: which cache")
    ap.add_argument("--rows", default=None, help="resolution stage: npy of compact row numbers (default: all rows of the class)")
    ap.add_argument("--tag", default=None, help="resolution stage: name suffix when --rows is given")
    ap.add_argument("--groups", type=int, nargs="*", default=None, help="build only these group numbers (pilot)")
    a = ap.parse_args(argv)
    cache = cache_dir(a.cache)
    {"index": lambda: build_index(cache), "alloc": lambda: alloc(cache), "build": lambda: build(cache, a.workers, a.groups),
     "supercoadd": lambda: supercoadd(cache, a.workers), "metadata": lambda: metadata(cache), "finalize": lambda: finalize(cache),
     "resolution": lambda: resolution(cache, a.cls, a.workers, a.rows, a.tag)}[a.stage]()


if __name__ == "__main__":
    main()
