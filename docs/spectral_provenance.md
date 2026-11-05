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

Write `f` for the true spectrum, `C` for the true calibration of the fibre, `C_fit` for the pipeline's, `S` for the true sky
(in electrons) and `m` for the sky model:

    y = (C / C_fit) R f  +  (S - m) / C_fit  +  noise / C_fit
        multiplicative       additive            noise
        g(lambda) R f        A(lambda)

* `g = C / C_fit` is the calibration error. It multiplies the object, so only flux can reveal it.
* `A = (S - m) / C_fit` is the additive sky-subtraction error. It is the same kind of quantity for every fibre of the petal, with
  fibre-dependent parts (throughput, LSF, position gradient) that the model only partly captures.

## What the cached SKY rows contain

For a sky fibre `f = 0`, so the cached row is `A + noise` (calibrated and averaged over the exposures of that coadd; a sky
TARGETID is exposure-specific, so these are single exposures). Two properties follow:

* They are in-sample. The sky fibres are the data the model `m` was fitted to, so their `A` is smaller than at a target fibre
  by roughly the leverage of the fit. The honest measure of `A` at a target is a leave-fibres-out residual.
* They are blind to `g`, because nothing multiplies.

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
