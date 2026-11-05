# Resolution matrices of DR2 spectra: are they suitable for template training?

Study of 2026-11-04 (roadmap step 1). Code: `src/DESIFlow/data/resolution_study.py` (sampling) and
`resolution_analysis.py` (figures and numbers); tests `tests/test_resolution_study.py`. Figures live in `docs/figures/` with a
same-named `.npz` holding every plotted array.

## What a resolution matrix is here

DESI stores the line-spread function (LSF) of a coadded spectrum as 11 diagonals: `R[d, j]` for `d = 0..10` is the response
at offset `d - 5` pixels (0.8 A each) from pixel `j`. A model spectrum `m` on the 0.8 A grid is compared with data as `R @ m`.
Templates "trained through R" are therefore templates of the unconvolved spectrum, so they can be reused with another
instrument's matrix (Spec-S5) by swapping `R`.

## Sample

700 random healpix groups (groups of at most 3000 rows, 97% of all groups), up to 10 SKY and 10 TGT spectra per group with
`COADD_FIBERSTATUS == 0` and at least one good pixel: 12,437 spectra (5,555 SKY; 6,882 TGT, of which 4,254 are coadds of a
single exposure and 1,039 of three or more). The full matrices are kept as a float16 library (2.0 GB,
`cache/resolution_study/library_R_float16.npy`) with their exposure information (fibre, petal, night, tile). SKY targets are
almost always single exposures (median 1); a TGT coadd is an inverse-variance weighted average of several exposures' matrices.

## Findings

1. **Normalisation and shape are sound.** Rows sum to 0.9992-1.0000 in all three cameras (flux conserving, as DESI
   normalises rows). Column sums wander (z camera 1st-99th percentile 0.72-1.14) because the native pixel size varies across the
   camera; that is not an error. The 11-diagonal band keeps all but 0.2% of the weight (median of the two outermost diagonals;
   maximum 6%). 27% of the entries are negative (small lobes, as expected for a Gauss-Hermite style PSF). Centroids are within
   +-0.14 px (1st-99th percentile).
2. **LSF width** (rms sigma, median over the camera): b 0.68 A, r 0.64 A, z 0.74 A, i.e. resolving power of about 2800 (b,
   at 4500 A), 4400 (r, 6600 A) and 4900 (z, 8600 A). Figure `resolution_lsf_width_vs_wavelength`.
3. **SKY and TGT matrices are the same population.** The median widths agree to within 1.5% - 2.0% at every wavelength
   (ratio SKY / single-exposure TGT between 0.985 and 1.020); the Kolmogorov-Smirnov distance is 0.03 in the blue and up to 0.17
   at a few z-camera wavelengths (near 8060, 8840 and 9500 A; the cause is not investigated). Single-exposure and 3+ exposure
   TGT coadds also agree (figure `resolution_lsf_width_distributions`).
4. **Nothing about the object matters.** The width correlates with redshift at Spearman 0.069 (0.064 after removing petal and
   month), and the medians for GALAXY, QSO and STAR are 0.687, 0.686 and 0.684 A. So the matrix does not know what it is
   applied to.
5. **What does matter is the hardware and the moment**, and only a little: of the variance of sigma across spectra, the petal
   explains about 5% (SKY) to 9% (TGT), the fibre about 15% to 28% (a noisy estimate: most fibres appear only a few times in a
   sample this size), the month 1-2% and the survey/program under 1%. More than 70% remains, which must be exposure-to-exposure
   variation (focus, temperature, weights). Figure `resolution_variance_sources`.
6. **How wrong is a stand-in?** For each spectrum, the relative error of a stand-in `eps = |R_standin - R_true| / |R_true|`
   per column (median over columns). For a line of matched-filter S/N `s`, the chi2 error of using the stand-in is about
   `(s eps)^2`, so a stand-in starts to matter for lines with `s > 1 / eps`:

   | stand-in for a TGT spectrum | median eps | 95th percentile | line S/N where it starts to matter (median / 95th) |
   |---|---|---|---|
   | mean of TGT spectra | 2.0% | 3.6% | 49 / 28 |
   | mean of SKY spectra | 2.2% | 3.9% | 45 / 25 |
   | mean of SKY spectra of the same petal | 1.9% | 3.7% | 52 / 27 |
   | Gaussian with the median width | 2.9% | 4.6% | 35 / 22 |

   The error is worst in the z camera (7-10% near 8000, 8840 and 9500 A); figure `resolution_substitution_error`.
   Using SKY matrices for TGT costs almost nothing: 2.2% against 2.0%.

## Reading of the results (what they imply, not yet tested on fits)

* The cached matrices are suitable for template training: normalised, flux conserving and varying by about 2% per object
  around a single population mean.
* Training a template is a (weighted) linear fit, so replacing each object's matrix by a draw from the same population, or by
  the population mean, gives templates that are unbiased to first order; the covariance of the matrices enters only at about
  `eps^2` (a few 1e-4). That is the "unbiased low-memory approximation" asked for, and it already exists: the 12,437-matrix
  library (2 GB), or one mean matrix per petal.
* A generative model of matrices would have to reproduce a 2% scatter that is already a small effect and depends on hardware
  state, not on the object; it is not needed for template training. If wanted later, condition it on fibre and time, not on
  redshift or type.
* Per-object matrices only matter for bright objects, with lines above S/N of about 25-50. Template training weights objects
  by median S/N, so these objects carry the most weight: give them their true matrix (for TGT, recompute from the coadd file, or
  store `resolution --cls tgt --rows ...` for the secure-z sample) or draw from the library. The denoising prior (roadmap step 8)
  should use the true matrix for them.

## Not done

* The direct test: fit real bright spectra with templates convolved with the true matrix, a population mean, and a library
  draw, and compare the chi2 and the line profiles. The `eps` argument above is a bound, not a measurement on data.
* The cause of the narrow z-camera features (8060, 8840, 9500 A) in the widths and in the SKY-TGT distance.
* Sample selection: groups above 3000 rows (3% of groups, but many rows) and fibres with a nonzero `COADD_FIBERSTATUS` were
  not sampled.
