# SKY templates: reconnaissance (2026-11-05)

Roadmap step 2 asks for templates that absorb sky features and instrumental systematics. Before planning them: what are the
"SKY" rows of the cache, what are the public DR1 sky spectra, and what sky products exist for Loa?

## 1. What the cached SKY rows are

OBJTYPE SKY rows are sky-fibre spectra after sky subtraction and flux calibration: the residual of the fibre and the sky model
that the pipeline fitted to the sky fibres of that petal. In a random sample of 30,000 cached SKY rows
(`src/DESIFlow/data/sky_residual_recon.py`, figure `sky_residual_recon`):

* the normalised residual `flux * sqrt(ivar)` has rms 0.99 (median over pixels), so the rows are close to pure noise, with
  localised excess: up to 1.3 at [OI] 5577 A, 1.2 near 9440 A, 1.8 at the red end, and spikes between 0.5 and 2 on sky
  lines in the r and z cameras (the lower values are pixels where the pipeline inflates the variance);
* the mean normalised residual is negative everywhere, from -0.01 at 3600 A through -0.06 at 6000-7500 A to -0.35 at 9800 A; in
  flux this is a mean over-subtraction of about 0.1-0.2e-17 erg/s/cm2/A over most of the range, 0.4-0.5e-17 at the red edge, and
  -2e-17 and -7e-17 on single pixels near 5577 A and 9300-9400 A;
* these are residuals of fibres the model was fitted to, so they are probably smaller than the residuals at target fibres
  (not measured).

## 2. DR1 sky spectra VAC (`dr1/vac/dr1/skyspec/v1.0/sky_spectra_vac_v1.fits`)

9,176 spectra, one per exposure, nights 20201214-20220613, on the 7781-pixel brz grid (3600-9824 A, 0.8 A), flux in
1e-17 erg/s/cm2/A/arcsec2, with exposure conditions (airmass, moon and sun geometry, seeing, transparency, synthetic g/r/z sky
magnitudes). The documentation page says only "derived from the DESI spectroscopic pipeline" and the data model is empty; the
page names M. Dowicz and D. Kirkby as contacts. I could not reproduce it: the ratio of the VAC to the median of
`sky / fluxcalib` over fibres in the same exposure (Loa products, camera b) differs between exposures (0.04, 0.08, 0.31, 0.30)
and does not follow the exposure time, and the shape differs by 7-10% after one scalar. So the recipe is unknown, and it covers
only DR1 exposures (about 45% of the 20,196 Loa exposures).

## 3. Sky products in Loa (`redux/loa/exposures/<night>/<expid>/`)

For every science exposure, per camera and petal (30 files each): `frame` (extracted fibre flux, electrons/A), `sky`
(per-fibre sky model, electrons/A, plus throughput correction, wavelength-shift, LSF-width and sky-gradient coefficients and the
ids of the sky fibres used), `sframe` (frame minus sky), `fluxcalib` (calibration vector, 1e17 cm2 count s / erg) and `cframe`
(calibrated flux, with fibermap and resolution). Dividing `sky` by `fluxcalib` gives the sky model in the same flux units as
the cached spectra. These have defined provenance (desispec) and cover all 20,196 exposures.

## 4. Consequences

* The cached `sky/` rows are the right data for the ADDITIVE sky-subtraction error (same pipeline, calibration and noise
  as the targets), but they are noise-dominated and contain no sky brightness information.
* The sky spectrum itself, and the conditions that drive the residual, come from the exposure files (or the DR1 VAC for
  part of the survey).
* Sky-fibre data cannot show MULTIPLICATIVE calibration drift (sensitivity at camera edges): there is no flux to multiply.
* The float16 resolution matrices of `sky/` (1.6 TB) are not needed for residual templates; the 12,437-matrix library of
  `docs/resolution_matrices.md` already covers realistic LSFs.
