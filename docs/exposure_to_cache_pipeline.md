# From an exposure to a cached TGT or SKY row: forms, measured versus modelled

2026-11-05. Companion to `docs/spectral_provenance.md` (file-level list); this note gives the analytic form of each step and says
what is measured, what is fitted and what is assumed. Sources: desispec 0.73.3 source and docstrings (read where cited), headers
of Loa files, and the literature where marked "literature, not read". Loa's exposures were reduced with desispec 0.66.0-0.66.2
(`SPECPROD kibo`) and its healpix coadds made with 0.67.0, so the code read here is newer than what produced the data;
the HDUs present in the Loa files (`THRPUTCORR`, `SKYGRADPCACOEFF`, `DWAVECOEFF`, `DLSFCOEFF`, `fiberflatexp`) confirm that
the corresponding features were in use.

Legend: **M** measured (a number read from detectors or imaging), **F** fitted (parameters of a model estimated from the data
of this exposure or of calibration data), **A** assumed (an external model, catalogue or prior taken as true).

## 0. Notation

`d` raw counts; `T_i(lambda)` the true response (counts per unit physical flux) of fibre `i`; `f_i` the true spectrum of what
fibre `i` sees; `s` the true sky radiance; `R_i` the resolution matrix of fibre `i` (rows sum to 1); `n` noise.

## 1. Detector to image (`preproc`; read)

    image = [ (raw - bias - overscan) * gain - dark * exptime ] / pixflat      [electrons per pixel]

* Bias, overscan, gain, read noise: **M** (overscan columns and rows of the exposure itself, plus master bias and dark frames from
  calibration exposures).
* Dark and pixel flat: **M** in calibration exposures, applied as fixed maps (**A** for the exposure).
* Variance: `var = read_noise^2 + Poisson(counts)` with the counts estimated from the data themselves (the code comment says this
  is biased), so the ivar of the image is a **model** built from measured values.
* Cosmic rays: pixels flagged when they are `cosmics_nsig` above the background and inconsistent with the PSF (**F**: a detection
  rule, not a repair; flagged pixels are masked). CTE and dark-trail corrections where configured (**A**).

## 2. Image to fibre spectra (specter `ex2d`; spectro-perfectionism, Bolton & Schlegel 2010, literature)

For a bundle of fibres and a wavelength patch the pixels are `p = A F + n`, with `A` the PSF model (the response of pixel `p` to
unit flux in fibre `i` at output wavelength `lambda_j`) and `F` the unknown spectrum on the output grid (0.8 A). The weighted
least-squares solution and covariance are

    F_hat = (A^T N^-1 A)^-1 A^T N^-1 p ,    C = (A^T N^-1 A)^-1

`F_hat` is deconvolved, noisy and correlated. Take a matrix square root `Q` with `Q^T Q = C^-1` (Bolton & Schlegel use `C^-1/2`), let `S = rowsum(Q)` and

    F_tilde = S^-1 Q F_hat ,    R = S^-1 Q ,    ivar(F_tilde) = S^2 ,    E[F_tilde] = R F_true

so the stored `frame.flux = F_tilde` (electrons/A) has uncorrelated noise, and `R` (stored as 11 diagonals) is its resolution
matrix with unit row sums (this is why the cached matrices have row sums of 1). Divide-and-conquer over bundles and
wavelength patches with overlaps stitches the full result (read).

* The PSF model `A`: a parametric (Gauss-Hermite) model of the spot shape with trace and wavelength solution, **F** from arc-lamp
  and flat exposures, then shifted for the exposure (`shifted-input-psf`, `fit-psf` files in `exposures/`): trace and
  wavelength shifts **F** from the exposure's own sky lines and traces (literature / file names; not read).
* The pixel data and `N`: **M** plus the variance model of section 1.

## 3. Fibre flat (`compute_fiberflat`, humidity correction; read)

The night flat from lamp exposures: iterate `mean(lambda)` = deconvolved average of all fibres, then

    ff_i(lambda) = smooth_spline( F_tilde_i(lambda) / [R_i mean](lambda) ) ,   node spacing 5 A, outlier clipping,

and the exposure flat `fiberflatexp` is the night flat multiplied by a humidity-dependent correction, `desi_compute_humidity_corrected_fiberflat`
(a tabulated model of the flat versus humidity, optionally adjusted with the sky fibres). Then `frame <- frame / ff_i`.

* Flat-lamp counts: **M**. Mean spectrum, spline smoothing, humidity dependence: **F** (the humidity model is **A** at the
  level of the exposure). The code states it has not been tested against real transmission variations.
* What this removes: fibre-to-fibre and wavelength-dependent throughput differences relative to the mean. What remains is
  `eps_i = T_i / T_hat_i`, the flat error.

## 4. Sky model (`compute_sky_linear`; read)

For the 500 fibres of a petal and camera, using the sky fibres only (`SKYTARGETID`), the model is

    m_i(lambda) = (1 + t_i) * R_i [ S + sum_k PC_k ( a_k (x_i - <x>) + b_k (y_i - <y>) ) ] + sector offsets

with parameters: `S_j`, the deconvolved sky flux at every output pixel `j` (**F**); `a_k, b_k`, amplitudes of a sky-gradient
PCA along the focal plane (**F** amplitudes; the PCA itself is **A**, built from earlier data); `t_i`, a throughput correction
per fibre from a PCA model of throughput versus fibre position (**F**, PCA **A**); sector offsets (**F**, optional). The normal
equations are

    A_pq = sum_i sum_w ivar_iw (d model_iw / d p)(d model_iw / d q) ,   B_p = sum_i sum_w ivar_iw (d model_iw / d p)(f_iw - model_iw)

solved by Cholesky; iterate with a chi2 > 4 sigma rejection (the worst outlier per wavelength among fibres) for at least 5
iterations; parameter covariance `A^-1`. Afterwards the wavelength shift and LSF width of the model are adjusted on sky lines
by solving a 2x2 system per fibre and line (derivatives of the model with respect to wavelength and LSF width), interpolated to
all fibres with a PCA (stored as `DWAVECOEFF`, `DLSFCOEFF`). The model variance includes a calibration-error term.

* Counts of the sky fibres: **M**. Everything above: **F** to those same sky fibres. The model for a target fibre is an
  **interpolation** of that fit (its own `R_i`, throughput correction and gradient position): not a measurement at that fibre.

`sframe_i = frame_i - m_i` (variances add).

## 5. Calibration (`stdstars`, `compute_flux_calibration`; read)

* Standard stars: about 10 per petal, selected by colour (Gaia BP-RP or Legacy G-R, within `delta-color` of the stellar locus).
  Their model spectra come from a template grid in (Teff, logg, [Fe/H]) matched by chi2 with a fractional template error (0.1),
  normalised to the star's catalogue magnitude and reddened with the extinction map: model flux **A** (grid, magnitude, E(B-V)),
  Teff/logg/[Fe/H] **F**.
* The calibration vector `C(lambda)` is fitted from

      chi2 = sum_stars sum_w ivar_sw [ f_sw - sigma_s ( R_s ( C . M_s ) )_w ]^2 + weak prior to median(C)

  with `M_s` the model of star `s`, `sigma_s` a free scale per star (normalised so that mean(1/sigma) = 1), iterated with
  outlier rejection, holes in the data interpolated, and the calibration linearly interpolated across the interstellar Ca H&K and Na D lines (+-6 A, read). `C` is deconvolved, so
  sharp telluric lines are handled. Star counts: **M**; `C(lambda)` and `sigma_s`: **F**; the stellar models and the assumption
  that all stars share one `C`: **A**.
* Because the models are unattenuated, `C` contains instrument response, atmospheric extinction and telluric absorption of
  this exposure. A point-source correction for the seeing (`FLAT_TO_PSF_FLUX`, a model of the fibre light fraction versus
  positioning offset and seeing, **A**) is folded in, and `PSF_TO_FIBER_FLUX` is stored to convert to fibre flux. The per-fibre
  calibration `C_i` is the deconvolved `C` convolved with that fibre's `R_i`.

## 6. The calibrated frame

    cframe_i = sframe_i / C_i ,    ivar_cframe = ivar_sframe * C_i^2     [1e-17 erg/s/cm2/A]

In terms of the truth, with `m_i` the model of section 4 and `T s_i` the sky in the fibre (electrons):

    cframe_i = ( eps_i T_bar / C_i ) R_i f_i  +  ( eps_i T_bar s - m_i ) / C_i  +  noise / C_i
               g_i R_i f_i                         additive A_i

* The sky subtraction happens in electrons, so the calibration does not enter it; it enters through `1/C_i(lambda)` on the
  additive term and as the factor `g_i` on the object.
* **M**: the counts. **F**: `m_i`, `C_i`, `ff_i`. The error terms `g_i - 1`, `eps_i - 1` and `s - m_i` are never measured;
  they are what remains after the fits.

## 7. Exposures to a healpix coadd (`coadd`, 0.67.0 for Loa; docstring read)

Per target and camera, over the exposures `e` that observed it:

    f = sum_e w_e f_e / sum_e w_e ,   w_e = ivar_e (masked pixels excluded),   ivar = sum_e w_e ,   R = sum_e w_e R_e / sum_e w_e

with a cosmic-ray clip of the pixels that deviate more than 4 sigma from the others (direction not checked). Loa's fibermap
has no `COADD_NORM`, so the per-exposure renormalisation of newer desispec was not applied.
Everything here is **M** (the exposure spectra) combined with a **F** weighting and clipping rule.

## 8. Cameras to brz and our cache (`coadd_cameras`, ours)

Overlap pixels (5760-5800, 7520-7620 A): ivar-weighted mean, masks OR-ed; elsewhere copy. Then float32, bad pixels (mask,
non-finite, ivar <= 0) set to flux = ivar = 0 and a binary mask. Super-coadds: the same ivar-weighted mean over the healpix
coadds of one TARGETID.

## 9. What a cached row is

* **TGT row**: `sum_e w_e cframe_e` over the exposures of the target (different fibres, petals and nights): the object, times a
  fibre- and exposure-averaged `g`, plus the exposure-averaged additive error `A / C`.
* **SKY row**: OBJTYPE SKY fibres are placed on blank sky (positions chosen from imaging, literature) and carry no object, so
  `f = 0`: the row is `A_i + noise / C_i` for one exposure. The model `m_i` was fitted to those very fibres (section 4), so
  `A` in these rows is the in-sample residual: smaller than at a target fibre by about the leverage of the fit, and blind to
  `g` (it multiplies only `R f = 0` and a ~1% term).

## 10. Summary: what is measured, fitted, assumed

| quantity | type | how |
|---|---|---|
| CCD counts, overscan, calibration frames | M | detector readout |
| image variance | F | read noise + Poisson from the data (biased) |
| PSF shape, trace, wavelength solution | F | arcs and flats, shifted by the exposure's sky lines |
| `F_tilde`, `R`, ivar of the frame | F | weighted least squares per bundle, matrix square-root normalisation |
| fibre flat `ff_i` | F | lamp spectra, spline smooth, humidity model |
| sky model `m_i` | F | linear fit to sky fibres, 4-sigma clipped; gradient and throughput PCAs assumed |
| sky-line wavelength and LSF adjustments | F | 2x2 linear fits per line, PCA interpolation |
| standard-star models | A | template grid, catalogue magnitudes, extinction map |
| calibration `C(lambda)`, per-star scales | F | deconvolved weighted fit to the standards |
| point-source and fibre-flux corrections | A | seeing and offset model |
| `g_i`, `eps_i`, `s - m_i` | not measured | residual errors of the fits above |
| coadd weights and clipping | F | ivar weights, 4-sigma clip |
