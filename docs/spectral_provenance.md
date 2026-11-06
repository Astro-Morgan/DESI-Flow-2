# Provenance of the fluxes and spectra around the SKY / instrument-template question (2026-11-05)

How each spectrum is made, from the exposure to our cache. Sources: the installed desispec 0.73.3 docstrings (read), the headers
of Loa files (read), and the literature for the steps whose code was not read (marked "not read"). The Loa production
(`SPECPROD=loa`, nights 20201214-20240409; config says "Kibo configuration") reduced the exposures with desispec 0.66.0-0.66.2
(`SPECPROD kibo` in the headers) and built the healpix coadds and redshifts with desispec 0.67.0 and redrock 0.20.2. The
installed 0.73.3 is newer, so a docstring below describes Loa only if it also holds in 0.66/0.67; this is flagged where it matters.

## The chain, per exposure, petal and camera

| # | spectrum (file) | units | how it is made | what it contains |
|---|---|---|---|---|
| 1 | `frame` (extracted fibre spectra) | electron/A per exposure | 2-D spectro-perfectionism extraction of each fibre from the preprocessed CCD image (Bolton & Schlegel 2010; not read), then division by the fibre flat. The flux is the resolution-convolved estimate `R f`, with the resolution matrix `R` (11 diagonals) stored | signal + sky + instrument response, uncalibrated, noise from the extraction covariance |
| 2 | `fiberflatexp` | dimensionless | relative fibre throughput per wavelength: `compute_fiberflat` divides the flat-lamp data by a deconvolved mean spectrum, smooths along wavelength with a spline (node spacing 5 A) and clips outliers. Whether the exposure flat is the night flat or corrected with sky lines: not read | fibre-to-fibre throughput |
| 3 | `sky` (sky model, all 500 fibres) | electron/A | `compute_sky`: `Sky[fibre, i] = R[fibre, i, j] Flux[j]`, one deconvolved `Flux` per petal and camera fitted to the sky fibres (those listed in `SKYTARGETID`) with iterative 4-sigma clipping, convolved with each fibre's own `R`. Terms stored in the file: per-fibre throughput corrections (`THRPUTCORR`), wavelength shift and LSF-width adjustments fitted on sky lines and interpolated across fibres with a PCA (`DWAVECOEFF`, `DLSFCOEFF`), a sky-gradient PCA (`SKYGRADPCACOEFF`), and a calibration-error term added to its variance. Whether the optional offset fit was on in kibo is not recorded | sky emission (airglow, scattered moon and twilight light, zodiacal light) times the instrument response, in the instrument's electrons |
| 4 | `sframe` | electron/A | `subtract_sky`: frame minus sky model (variances added), optionally a residual achromatic throughput correction | signal + the sky-model error, uncalibrated |
| 5 | `stdstars` | 1e-17 erg/s/cm2/A | stellar models fitted to the standard stars of the petal (about 10) | physical-flux models of the standards |
| 6 | `fluxcalib` | 1e17 cm2 count s / erg | `compute_flux_calibration`: solve `frame_std = R_fibre C F_model` for the mean throughput `C(lambda)` (deconvolved, so sharp telluric lines are handled), a scale factor per fibre, outlier rejection; the returned vector is `C_fibre` for a flat spectrum, divided by a flat-to-PSF flux factor, with `FIBERCORR` per fibre. Standard-star models are unattenuated, so `C` contains extinction and telluric absorption of that exposure | the calibration of the petal, an average over its standards; its error is not stored |
| 7 | `cframe` | 1e-17 erg/s/cm2/A | `apply_flux_calibration`: `flux = sframe_flux / C_fibre`, `ivar = ivar C^2`; resolution `R` unchanged by default | the quantity we call data |
| 8 | healpix `coadd` | 1e-17 erg/s/cm2/A | per target and camera: inverse-variance weighted mean over exposures with 4-sigma cosmic-ray clipping; resolution coadded the same way. Loa's coadd fibermap has no `COADD_NORM`, so the per-exposure renormalisation that newer desispec offers was not applied | the same, averaged over exposures (and over fibres, since the same target can sit on different fibres) |
| 9 | our cache (`tgt/`, `sky/`) | 1e-17 erg/s/cm2/A | `coadd_cameras` (inverse-variance weighted in the overlaps, mask OR), then float32 and a binary mask with bad pixels set to 0 | as 8, on the 7781-pixel brz grid |
| 10 | our super-coadds | same | pixelwise inverse-variance mean of the coadds of a target, `COADD_FIBERSTATUS == 0` rows only | as 8 |
| 11 | DR1 sky VAC | 1e-17 erg/s/cm2/A/arcsec2 | undocumented ("derived from the pipeline"); not reproduced as `sky / fluxcalib` of the same exposure | one spectrum per DR1 exposure with conditions |

## What a cached target spectrum contains

Work in electrons first. Let `T` be the true instrument response (counts per unit physical flux), `f` the true object spectrum,
`s` the true sky radiance in physical units, `m_e` the sky model in electrons (fitted to the counts of the sky fibres) and
`C_fit` the pipeline calibration. The counts are `d = T (R f + s) + noise`, so the cframe is

    y = (d - m_e) / C_fit = (T / C_fit) R f  +  (T s - m_e) / C_fit  +  noise / C_fit
                            g R f               delta_e / C_fit           noise / C_fit

with `g = T / C_fit` the calibration error and `delta_e = T s - m_e` the sky-subtraction error in electrons. Equivalently
`delta_e / C_fit = g (s - m_e / T)`: in physical terms the additive term carries a factor `g`, but it multiplies a quantity
(`s - m_e / T`, the model error in physical units) that is about 1% of the sky, so a 5% calibration error changes it by about
0.05% of the sky. The calibration drops out of the sky subtraction itself (done in electrons, before the division); it enters the
additive term only through the division by `C_fit(lambda)`, which is known exactly per exposure and petal (`fluxcalib`) and falls by
a factor of 5-10 at the blue edge of each camera and in the telluric A band (figure `sky_leakage_recon`, bottom panel).

* `g` multiplies the object: only flux can reveal it.
* `A = delta_e / C_fit` is the additive sky-subtraction error; a structure that is smooth in electrons shows edges in flux because
  of the division by `C_fit`.

## What the cached SKY rows contain

For a sky fibre `f = 0`, so the cached row is `delta_e / C_fit + noise / C_fit` (calibrated and averaged over the exposures of that coadd; a sky
TARGETID is exposure-specific, so these are single exposures). Two properties follow:

* They are in-sample. The sky fibres are the data the model `m` was fitted to, so their `A` is smaller than at a target fibre
  by roughly the leverage of the fit. The honest measure of `A` at a target is a leave-fibres-out residual.
* They are blind to `g` in practice (it multiplies a term that is about 1% of the sky), so they cannot be used to measure it.

## What the exposure sky model is good for

`sky / fluxcalib` is the sky model in the cache's units. It is a fitted model of a spectrum that is itself the sky emission times
the instrument response; its calibration-error term is `g`. Training templates on it would be a model of a model, and the
pipeline has already subtracted it. Its use is as a covariate: how bright the sky was at each wavelength in the exposure, which
predicts the size of `A` (sky-line residuals scale with line flux) and so can be used to condition or weight the residual
templates.

## Open items (not verified)

* How the exposure fibre flat is derived (night flat or sky-line corrected).
* Whether the optional offset fit (`fit_offsets`) was on in kibo; it would change the mean offset seen in the SKY rows.
* Standard-star fitting details (model grid, colours used).
* The recipe of the DR1 sky VAC (contacts on its documentation page: M. Dowicz, D. Kirkby).

## Test of the electron-space picture (24 exposures, `sky_leakage_recon.py`)

119 of 240 exposure-petal pairs could be read (the rest failed to open; not investigated). Across pairs, per wavelength, the
sky-fibre residual was regressed on the sky model, in flux units and in electrons:

* flux units: slope -0.002 (b), -0.006 (r), -0.013 (z), growing to -0.04 at the z red edge, offset about +0.03e-17;
* electrons: slope -0.0002, -0.0002, -0.0005 (consistent with none) and a constant offset of -0.7 (b), -1.1 (r), -1.8 (z) e/A,
  -4.7 e/A at the z red edge. Divided by `C_fit` this gives about -0.03e-17 in all cameras and -0.13e-17 at the z red edge,
  against an observed mean residual of -0.02e-17 and -0.10e-17.

The flux-unit slope is not trustworthy: the residual and the sky model are both divided by `C_fit`, which varies between
exposures and petals, and a shared divisor creates a correlation. Whether the mean residual is a constant electron deficit
divided by `C_fit` or a fraction of the sky is not settled by this sample (correlation of the smoothed mean residual with the
electron-offset prediction: 0.33 b, 0.04 r, 0.94 z; with the sky-fraction prediction 0.43, 0.12, 0.94).

## Test of the variance model (24 exposures, `sky_rms_recon.py`)

The rms of `flux * sqrt(ivar)` on sky fibres is the same in electrons and in calibrated flux (the factor `C` cancels), so the
smooth wavelength dependence of panel 1 of `sky_residual_recon` cannot be a calibration (shape or amplitude) effect. It is
set by the electron counts: against the sky model counts per fibre the three cameras fall on one curve,

| counts (e/A per fibre) | rms b | rms r | rms z |
|---|---|---|---|
| 1-10 | 0.73 | 0.77 | 0.87 |
| 10-100 | 0.92 | 0.97 | 1.01 |
| 100-1000 | 0.98 | 0.98 | 1.02 |

(down to 0.57 at 3 e/A in b). At fixed high counts the rms is about 1 at all wavelengths apart from sky-line spikes in r and z.
The counts follow sensitivity times sky brightness, so a counts-dependent error in the variance looks like a sensitivity curve:
the stated variance is too large where the counts are low (variance overestimated by 1/rms^2, up to a factor 3 in b at 3 e/A).
The rms does not depend on the number of sky fibres used (no sign of in-sample leverage). The cause is not determined; the
candidates are the read-noise or gain used in the variance model, the Poisson term being estimated from the noisy data (the
preprocessing code comments that this is biased), and the extra variance added to the sky model. The residuals of sky fibres
may not carry over to target fibres, whose counts include the object.
