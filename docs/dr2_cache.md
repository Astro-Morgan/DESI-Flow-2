# DR2 (Loa) spectral cache

A row-matched, memory-mappable cache of every DESI DR2 (`loa`) healpix coadd of `OBJTYPE == TGT` and `OBJTYPE == SKY`,
built from `zcatalog/v1/zall-pix-loa.fits` and the `healpix/{survey}/{program}/{healpix//100}/{healpix}/coadd-*.fits`
files. Code: `src/DESIFlow/data/dr2_cache.py`, tests `tests/test_dr2_cache.py`, `tests/test_dr2_supercoadd_plan.py`.

## Contents

zall has 64,100,684 rows (137 columns). `OBJTYPE`: TGT 51,457,579, SKY 9,489,661, N/A 3,124,966, BAD 28,324, NON 154;
only TGT and SKY are cached. The TGT rows hold 49,775,647 distinct TARGETIDs (1.68M rows are a further coadd of a target
already present under another survey/program). There are 56,780 (survey, program, healpix) groups, contiguous in zall.

Two directories, `tgt/` and `sky/`, each row-matched to its own `metadata.h5`: row `i` of every file in a directory is
row `i` of that directory's metadata (no index lookup).

| file | content |
|---|---|
| `flux.npy`, `ivar.npy` | float32 (n, 7781) on the 0.8 A linear grid 3600-9824 A; bad pixels have flux = ivar = 0 |
| `mask.npy` | uint8 (n, 7781), 1 = bad: DESI `mask != 0`, non-finite flux or ivar, or ivar <= 0 |
| `status.npy` | uint8: 1 ok, 2 target absent from its coadd file, 3 coadd file missing (all 60,947,240 rows are 1) |
| `snr_med.npy`, `ngood.npy` | median over good pixels of flux * sqrt(ivar); number of good pixels |
| `metadata.h5` | uncompressed; group `zall` = every zall column except `COEFF`, plus `zall_row` (row in the original catalog); group `derived` = status, snr_med, ngood (and `sc_index` in tgt/) |
| `tgt/sc_*.npy` | super-coadds, see below |
| `sky/resolution.npy` | float16 (n, 11, 7781) resolution matrices of every sky row |

Counts: 197,810 TGT rows and 64,234 SKY rows have no good pixel (fully masked; `snr_med` is NaN). Median `snr_med`:
2.28 (TGT), -0.03 (SKY).

## Method

1. **Camera coadd.** For each group, the targets of the class are located in the coadd file by TARGETID and read with
   `desispec.io.read_spectra`; the b, r, z cameras are combined with the official `desispec.coaddition.coadd_cameras`
   (inverse-variance weighted in the overlaps, OR of the masks). The resolution HDUs are skipped in this pass, which makes
   reading plus coadding about 4x faster.
2. **Cache format.** float32; pixels with `mask != 0`, non-finite values or ivar <= 0 are marked bad (mask 1) and carry
   flux = ivar = 0, the same convention the span masker uses to hide pixels, so the model logic is unchanged
   whether it sees the DESI bitmask or this binary mask.
3. **Super-coadds** (tgt/): for each TARGETID with two or more rows (1,614,340 targets, 3,296,272 rows), the healpix
   coadds are combined pixel by pixel with inverse-variance weights, `flux = sum(w f) / sum(w)`, `ivar = sum(w)`,
   `w = ivar` on good pixels. A row enters only if `COADD_FIBERSTATUS == 0` (95.7% of super-coadds use every row; if no row
   qualifies, all rows with status ok enter and `sc_fallback` is set, 0.10% of targets). There is no S/N gate: a noisy
   coadd gets a small weight and gating would bias the S/N distribution. `sc_chi2_red` is the reduced chi2 of the
   contributions about the combination (median 1.08, 99th percentile 13), a flag for variable objects or calibration
   mismatches; it is stored, not used to clip. Single-row targets are not stored again: `tgt/sc_index.npy` (int32 per tgt
   row) is the row of the `sc_*` arrays, or -1 when the super-coadd is the row's own healpix coadd. Tables: `sc_targetid`,
   `sc_n_total`, `sc_n_used`, `sc_fallback`, `sc_chi2_red`, `sc_snr_med`, `sc_members` + `sc_member_start`.
4. **Resolution matrices** (stage `resolution`): the brz-merged matrix of `coadd_cameras`, 11 diagonals x 7781 pixels, stored
   as float16 (171 kB per spectrum). Sky: every row (1.6 TB). Target rows: only for a chosen subset (the VI/ensemble
   secure-z sample, once defined): `--cls tgt --rows rows.npy --tag name`. Float16 was chosen after tests on coadds of
   several programs (R applied to a line spectrum, rms relative error): float16 3e-4 with mean signed error 1e-7;
   thinning the matrix along wavelength and interpolating fails (3e-2 rms for every second column, 1e-1 for every eighth: the
   matrix is not smooth in pixel index, because of the sub-pixel phase of the grid against the CCD pixels); a global PCA over
   the 85,591 numbers per matrix fails (rms 0.2 even on held-out rows of its own training files).

## Validation

* Pilot (4 groups, 30,496 spectra): cached flux, ivar, mask and `snr_med` are bit-identical to an independent full read
  that includes the resolution matrices; rows outside the pilot are still zero.
* Production build: 56,776 groups (+ the 4 pilot ones), 60,916,744 spectra in 115 minutes with 20 workers (8,800/s), 0
  errors; every row has status 1.
* Super-coadds: 25 random multi-row targets recombined independently in numpy from freshly read official camera coadds
  match the cache to a worst relative difference of 7e-8.
* Metadata: `TARGETID` of each metadata row equals the index TARGETID of its `zall_row`, and OBJTYPE is TGT / SKY as
  expected.
* Resolution stage: bit-identical to an independent full read on a test group, for both read strategies below.
  The first 9.41M sky rows were written with scattered-row reads (`read_spectra(rows=...)`); the 19 largest groups
  (74,852 sky rows, up to 32k rows per coadd file) stalled with that strategy, so the committed code reads contiguous
  blocks of the file and then selects the wanted rows. Both strategies run the same `coadd_cameras` on the same rows, so
  the result is the same; only the file access pattern differs.
* Sky resolution (9,489,661 rows): the per-group ledger counts equal the number of sky rows in every one of the 56,780
  groups; in a random sample of 20,000 rows the all-zero matrices (133, 0.67%) are exactly the rows with `ngood == 0`
  (fully masked spectra, whose coadd file stores a zero matrix), and no other row is zero or NaN.
* Not validated: the cross-exposure official `desispec.coaddition.coadd` (healpix coadds have no per-exposure fibermap rows).

## Reproduce

```bash
source <cache>/env.sh                    # DESI environment + DESI_DR2_CACHE
python -m DESIFlow.data.dr2_cache index  # zall key columns -> <cache>/index (checks group contiguity)
python -m DESIFlow.data.dr2_cache alloc  # sparse row-matched npy files
python -m DESIFlow.data.dr2_cache build --workers 20 [--groups g ...]
python -m DESIFlow.data.dr2_cache supercoadd --workers 20
python -m DESIFlow.data.dr2_cache metadata
python -m DESIFlow.data.dr2_cache finalize
python -m DESIFlow.data.dr2_cache resolution --cls sky --workers 20
```

Every stage resumes where it stopped (`build` and `resolution` through the logs in `<cache>/ledger`). On NERSC the stages
ran on a login node in tmux; the cache directory is `RepoClone/Claude/cache/dr2_loa`.
