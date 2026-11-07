# Instrument and sky templates: DeepRock-style chi2 reconstruction, sweep over the number of templates

2026-11-06. Code: `src/DESIFlow/templates/deeprock_lite.py` (trainer), `sky_instrument_sweep.py` (data, sweep, figures); tests
`tests/test_deeprock_lite.py`. Figures: `docs/figures/template_chi2_vs_K`, `template_examples_instrument`,
`template_examples_sky` (each with a same-named `.npz` of the plotted arrays).

## Method

* DeepRock design of `deeprock_dr2.py`, observed frame, no redshift: K learnable templates `T`; per spectrum the coefficients are
  solved by weighted least squares `c = (T^T W T + ridge I)^-1 T^T W f` with gradient through the solve; AdamW (lr 3e-3, cosine,
  1500 steps, batch 512), initialised from a weighted SVD.
* Loss: masked-denoising chi2. 36 random spans (log-uniform 5-250 pixels, about 25% of the pixels) are hidden, the coefficients
  are fitted on the visible pixels and the chi2 per pixel is scored on the hidden ones. The reported "mean chi2" is this score on
  hold-out spectra (fixed masks), with the standard error of the mean over spectra, against the zero model (K = 0).
* Whitened units: `g = f s`, `omega = w / s^2`, `s_l = sqrt(mean ivar)`; identical chi2, better conditioned. Saved templates are
  an orthonormal basis of the learned span (in whitened units) ordered by the energy they carry, plus the same in flux units.
* Each K is trained independently (not nested); one seed.

## Data

| | instrument | sky |
|---|---|---|
| source | cached SKY rows (sky-fibre residuals after sky subtraction, calibrated) | DR1 sky VAC (9,176 per-exposure sky spectra) |
| weights | pipeline ivar and mask | assumed: ivar = 1 / (0.1^2 + (0.03 f)^2) (the VAC has none) |
| rows used | 150,000 train / 20,000 hold-out, TARGETID < 0 (tile derivable), >= 4000 good pixels | 8,372 train / 804 hold-out |
| split | by tile (tile % 20 == 0 held out: 705 of 13,577 tiles) | by night (night % 10 == 0: 32 nights) |

## Result: mean chi2 per pixel on hold-out spectra (hidden pixels)

| K | instrument (zero model 1.0143) | sky (zero model 288.9) |
|---|---|---|
| 1 | 0.9538 | 47.03 |
| 2 | 0.9469 | 20.41 |
| 3 | 0.9421 | 11.33 |
| 4 | 0.9404 | 6.63 |
| 6 | 0.9390 | 2.56 |
| 8 | 0.9380 | 1.58 |
| 12 | 0.9370 | 0.966 |
| 16 | 0.9364 | 0.756 |
| 20 | 0.9362 | (not run) |
| 24 | 0.9363 | 0.505 |
| 32 | 0.9371 | 0.431 |
| 48 | 0.9389 | 0.383 |
| 64 | 0.9412 | 0.367 |
| 80 | (not run) | 0.349 |
| 96 | (not run) | 0.353 |
| 128 | (not run) | 0.344 |

Standard errors: 0.002 (instrument), 0.02-1 (sky, larger at small K); the K ladder was extended on request (instrument K = 20, sky K = 80, 96, 128).

* Instrument: 6% of the 7.7% total gain comes from the first template; the curve is flat within the errors from K = 12 to 32 and
  rises beyond K = 32 while the in-sample (visible-pixel) chi2 keeps falling (0.9196 at K = 64): overfitting.
* Instrument K = 20 (added on request) lies on the plateau, 0.9362 +- 0.0022.
* Sky K = 80, 96 and 128 were added on request: the hold-out chi2 is flat from K = 48 (0.383) to 128 (0.344) within the standard error of
  about 0.03, while the in-sample chi2 keeps falling (0.187 at K = 128), so templates beyond about 48 are overfitting.
* Sky: the chi2 reaches 1 (the assumed 3% accuracy) at K = 12 and keeps improving slowly; the hold-out value separates from the
  in-sample one above K = 24 (0.505 / 0.404 at K = 24, 0.367 / 0.225 at K = 64).

## Files (not in the repository; `RepoClone/Claude/cache/sky_templates/`)

`instrument/K001.npz ... K064.npz` (13 sets, including K020) and `sky/K001.npz ... K128.npz` (15 sets), each with `templates_whitened` (K, 7781), `templates_flux`
(K, 7781), `pixel_scale_s`, `energy`, `wavelength_A`, the hold-out metrics, the per-row hold-out chi2 and the training-loss
history; `summary.json` per kind; the prepared training and hold-out arrays; `run.sh`.

## Caveats

* The instrument rows are the in-sample residuals of the sky fit (the sky model was fitted to these fibres): the structure the
  templates learn is what survives that fit, and may differ at target fibres.
* The sky chi2 values depend on the assumed weights; only the shape of the curve and the comparison with the zero model carry
  over. The recipe of the VAC is not documented.
* Differences between K = 12 and K = 32 (instrument) are within about one standard error; one seed.

## What the K = 20 instrument templates are (`inspect_templates.py`, figures `instrument_templates_K020`, `..._summary`)

The set is an orthonormal basis ordered by the energy it carries on the data. Per-template measures (the energy share, the power
within 150 A of a camera edge, the power on the strongest sky-line pixels, the share explained by a piecewise quadratic per camera, and
the correlation of the coefficient on the hold-out spectra with the measured camera offsets):

| template | energy | what it looks like | evidence |
|---|---|---|---|
| 1 | 72.4% | a smooth positive offset in all three cameras, rising steeply over the last 500 A of z | coefficient correlates 0.72 / 0.79 / 0.75 with the measured mean offset in b / r / z; a flat offset over all cameras lies 94% in this template alone; 20% of a red-edge ramp |
| 2 | 8.3% | the b camera high, r and z near 0: a step at the b/r junction | correlation -0.70 with the r - b offset; step -0.78 rms |
| 3 | 6.0% | r and z slightly positive, then a sharp fall over 9200-9824 A; noisy rise at 3600 A | 47% of its power in the camera-edge zones (14% of the pixels); with #1 and #2 it holds 92% of a z red-edge ramp |
| 4 | 3.0% | the r camera low relative to b and z: steps at both junctions | correlation +0.69 with the z - r offset, -0.38 with r - b |
| 6, 10 | 1.3%, 0.7% | piecewise-smooth shapes inside a camera | camera-quadratic R2 0.62 and 0.60 |
| 12, 13, 14, 20 | 0.5%, 0.5%, 0.5%, 0.2% | swings at the blue edge (3600-4500 A) of different shapes | 24-37% of their power in the edge zones |
| 5, 8, 9, 11, 15, 16 | 0.4-1.8% each | sky-line patterns in r and z (7000-9800 A) | 54-63% of their power on 8% of the pixels |
| 7, 17-19 | 0.2-1.0% | mixed | |

The first four templates together span a pure per-camera constant offset to 95% (b), 93% (r) and 96% (z) (the first two give 90% for
b); a flat offset is 94% in template 1; a z red-edge ramp 92% in the first three. A blue-edge decay of 120 A is only 41% in the first
12 templates and 71% in all 20: the blue-edge swings are spread over several templates.

These are additive structures measured on sky fibres (no signal to multiply), so they are camera offsets and edge behaviour of the
sky-subtracted residual, not by themselves a multiplicative calibration error.

## Camera overlap test on targets (`camera_overlap_test.py`, `stack_junction_check.py`; figures `camera_overlap_test`, `stack_junction_check`)

6,000 targets (half S/N 3-10, half > 10), per-camera arrays of the healpix coadds. Mean flux of each camera over the b-r overlap
(5762-5798 A) and the r-z overlap (7522-7578 A, clear of the telluric A band), Delta = second - first camera, fitted as
Delta = a + m Fbar:

| overlap | a [1e-17] | m | median ratio - 1 above 3e-17 |
|---|---|---|---|
| b - r | +0.009 +- 0.004 | -0.14% +- 0.02% | +0.2 to +0.8% |
| r - z | -0.010 +- 0.002 | +0.20% +- 0.01% | within +-0.3% |

* On average the cameras agree at the overlaps to 0.01e-17 and 0.2%.
* Target by target they do not: the scatter of Delta beyond its measurement error is 1.3-3.4% of the flux for b-r and 0.9-2.0% for
  r-z, and in absolute units it grows with the flux (b-r: 0.03, 0.18, 0.36, 0.88e-17 in the bins 1-3, 3-10, 10-30, 30-160e-17). That is
  a zero-mean, percent-level, multiplicative-type mismatch (some of it may be real differences in line content or LSF inside the windows).
* Additive stand-in: a multiplicative error m on a whole camera leaves, after the best constant additive offset, |m| rms(F)/mean(F)
  of the flux. The spectral shape rms/mean inside a camera is 0.07 (r), 0.06 (z) and 0.25 (b) in the median (0.16, 0.11, 0.49 at the
  90th percentile). For m = 2% the leftover is 0.14% (r), 0.13% (z) and 0.5% (b; 1% at the 90th percentile) of the flux, below the noise
  of any spectrum of S/N below about 100 per pixel. This holds for errors that are constant over a camera, not for shapes that vary
  inside it (the edges).
* The junction steps of the SKY stack do not appear in targets: stacks of 30,000 SKY rows give steps of +0.0048 +- 0.0010 (5780 A)
  and -0.0043 +- 0.0007 (7570 A) [1e-17], stacks of 30,000 faint targets +0.0003 +- 0.0006 and -0.0009 +- 0.0004. The camera-step
  coefficients of the SKY templates are therefore not shown to measure target calibration errors.

Corrections to the section above: the "typical amplitude" (std of the coefficient times the template) was dominated by heavy
tails; the robust median amplitudes are 0.006 (#1), 0.035 (#2), 0.006 (#3), 0.017 (#4) [1e-17 erg/s/cm2/A], 90th percentiles 0.09, 0.32,
0.06, 0.13. The correlations of the coefficients with the measured camera offsets share pixels with the coefficients (window means
and template fits use the same data) and are not clean evidence.

## Target-residual test and per-camera templates (`target_residuals.py`, `target_nuisance_test.py`, `camera_templates.py`; figures `target_nuisance_test`, `camera_template_chi2_vs_K`)

**Per-camera instrument templates.** Each camera trained separately on its own brz-grid segment of the SKY rows (the overlaps are the merged
pixels of the cache), K = 1..48, hold-out hidden-pixel chi2 (zero model in brackets): b best at K = 3, 0.9011 (0.9318); r best at K = 4,
0.9280 (0.9830); z best at K = 12, 0.9683 (1.1046); each camera overfits beyond its best K. 19 terms in all, against 20 for the global
set. Overlap rule for applying per-camera terms to a merged brz spectrum: `coadd_cameras` combines the overlap pixels as the
inverse-variance weighted mean of the two cameras and copies the rest (read in the source); `merge_camera_terms` does the same for the
fitted terms (tested). In the target test below the data are per camera and each camera is fitted separately, so no merging enters.

**Test on targets.** 3,000 galaxies (ZWARN 0, DELTACHI2 > 500, z < 1.5) and 1,000 stars with S/N > 8; the Redrock model rebuilt from
`COEFF` (Template.eval with the camera resolution, validated: rebuilt chi2 / zall CHI2 = 1.014); residual = data - model per camera;
nuisance terms fitted on the visible pixels, chi2 per pixel scored on hidden spans (same spans for every method, two draws). The
model as it is: 5.64 per hidden pixel (galaxies 4.79, stars 8.19), dominated by template mismatch.

| method | terms | gain over the model as it is | edge zones | targets improved |
|---|---|---|---|---|
| one scale per camera | 3 | 0.4% | worse | 44% |
| additive Legendre per camera (deg 2) | 9 | 8.6% | -0.75 | 75% |
| multiplicative Legendre per camera x model | 9 | 10.1% | -0.98 | 73% |
| global instrument templates K = 20 | 20 | 6.5% | -0.35 +- 0.26 | 80% |
| per-camera templates (3, 4, 12) | 19 | 9.2% | -0.77 | 79% |
| multiplicative + global K = 20 | 29 | 13.7% | -1.02 | 75% |
| multiplicative + per-camera templates | 28 | 14.2% | -1.08 | 75% |

(edge zones: change of the hidden chi2 per pixel in the 150 A camera-edge zones.) Paired differences (first minus second, negative favours the
first): per-camera minus global K = 20: -0.153 +- 0.053 (edge -0.43 +- 0.25; stars -0.20 +- 0.14); per-camera minus additive Legendre:
-0.032 +- 0.074 (equal); per-camera minus multiplicative: +0.049 +- 0.041 (equal; stars -0.06 +- 0.07); multiplicative + per-camera
minus multiplicative alone: -0.231 +- 0.038; versus multiplicative + global: -0.026 +- 0.009.

Reading: a per-camera scale does nothing (the cameras agree in amplitude on average, as the overlaps showed); what helps is shape within
a camera. Additive per-camera instrument terms learned from sky fibres reach 90% of the gain of nine free multiplicative terms without
being fitted to targets, and the two kinds of term are complementary. The multiplicative scatter fitted per camera (percent): b
0.6 +- 1.9, r -0.4 +- 0.8, z 0.2 +- 0.3. The median fractional residual (data - model) / model of all targets shows common +-1%
structure (+1-2% at 4000-4400 A, +1% at 5000-5900, -0.7% at 6200-7000, +1% at 8000-9300, falling to -1% at 9500-9800) that is
calibration, Redrock-template mismatch, or both. Polynomials and templates also absorb physical template mismatch.

## Visual comparison of the candidates (template_fit_comparison.py)

Figures `template_comparison_shapes`, `template_comparison_fits`, `template_comparison_population` (each with a same-named .npz).
- Shapes: the per-camera templates carry steps inside a camera (r #2 near 6730 A, z #4 near 8700 A; plausibly CCD amplifier boundaries, not
  verified) and sky-line spikes (r #3-4, z #3); the global K=20 set mixes these into smoother combinations. z templates bend strongly beyond 9300 A.
- Fits (four fixed-seed random targets, not selected by outcome): fitted corrections are small against the smoothed residual, which is dominated by
  template mismatch; hidden-pixel chi2 hardly changes on these four (only star 603 improves slightly). The per-camera fit can overreach at the red end
  of bright stars (star 506, camera z, -130 against a scale of +-50).
- Population (median over targets): galaxies show a +2-2.5% bump at 5000-5800 A in b that falls to ~0 in r at the same wavelengths (a camera
  discontinuity), then -1% at 6200-7000 A. Per-camera and global templates follow the discontinuity and the 6730 A step; the smooth multiplicative
  polynomials cannot. Stars have +-1% structure the fits leave in place (Paschen-region line dips at 8800-9200 A remain).

## Physical templates with and without explicit nuisance terms (physical_templates.py, physical_figures.py)

Question: are explicit instrument/sky terms needed, or do augmented training inputs make linear template fits robust? Setup: K = 24 rest-frame
templates (log-lambda grid, 3e-5 dex) learned through the mean DR2 library resolution matrix at the known redshift of 60,000 secure galaxies
(0.02 < z < 1, snr_med > 4, DELTACHI2 > 100; 8,000 held out by TARGETID % 10), masked-denoising loss, 2500 steps, batch 96. Nuisance group fitted jointly
per spectrum with Gaussian priors: per-camera additive offsets (3), per-camera instrument templates (3/4/12), sky templates (VAC K = 24, prior 1% of the VAC
coefficient scatter), per-camera multiplicative Legendre degree <= 2 times the first-pass model (8 terms, b constant dropped: b is the reference).
Augmentation: multiplicative curves, offsets, global K = 20 instrument-template combinations (real SKY-row coefficients), VAC sky spectra, each scaled to add
about 0.075 chi2/px (natural sizes added 0.23 / 1.9 / 16.5 / 0.011); scored against the clean spectrum. Arms: A plain, B augmented, C nuisance group, D both.

Hidden-pixel chi2 per pixel on held-out real galaxies (paired difference against A in brackets, standard error over rows):

| arm | plain fit | with nuisance fit |
|---|---|---|
| A plain | 1.408 | 1.314 |
| B augmentation | 1.437 (+0.028 +- 0.025) | 1.331 (+0.017 +- 0.017) |
| C nuisance group | 1.683 (+0.275 +- 0.086) | 1.292 (-0.022 +- 0.015) |
| D both | 1.737 (+0.328 +- 0.128) | 1.295 (-0.019 +- 0.012) |

D against C: +0.003 +- 0.005 (real), +0.005 +- 0.003 (augmented). Reading: (1) augmentation alone does nothing for the linear fit (also on augmented
test data: +0.004 +- 0.009); (2) the explicit nuisance group at fit time is what helps (-6.7% for the plain-trained templates); (3) training with the group
adds only ~1.7% (about 1.5 sigma) and makes the templates depend on it (plain fit +0.28); (4) augmentation on top of the group adds nothing.
Caveats: single seed, K = 24 only, galaxies only, mean R stand-in; the nuisance part alone carries a smooth additive level of 1-2e-17 (figure
physical_arms_fits), i.e. the offset priors (sigma 0.7-1.1 from SKY-row camera means) are loose enough that the group absorbs continuum flux; the hyperparameters
were not tuned. Figures: physical_arms_chi2, physical_arms_templates, physical_arms_fits (with .npz caches in docs/figures).

### Simplified group: per-camera instrument (+ sky) templates only (arms E, F; ablation of A-D)

Hidden chi2 per pixel on held-out real galaxies, templates of arm A (plain-trained) evaluated under subsets of the nuisance group: plain 1.408; offsets only
1.392; multiplicative Legendre only 1.396 (no gain); sky only 1.374; per-camera instrument only 1.351 (-0.057 +- 0.021); instrument + sky 1.326; all 1.314.
Instrument + sky therefore carries about 87% of the full group's gain. Training with the simplified group: E (instrument only) 1.338 evaluated with instrument
terms (vs A 1.351: -0.013 +- 0.007); F (instrument + sky) 1.312 evaluated with both (vs A 1.326: -0.014 +- 0.008; vs C with the full group 1.292: +0.021 +- 0.018, not
significant). Templates trained with the simplified group depend less on it (plain fit E 1.49, F 1.55, against 1.68 for C). Instrument-only training is the
simplest option that stays within about 1 sigma of the full group when sky is added at inference.

## Dense K sweeps, spline K selection and final templates (k_selection.py)

Full retrain from scratch (cache/sky_templates_v2/seed0-2; same recipe: 1500 steps, whitened masked-denoising chi2): per-camera instrument templates for
K = 1..32 (every integer), 34, 36, 40, 44, 48; sky templates (VAC, assumed weights) K = 1..32, 34..48 step 2, 56, 64, 72, 80, 96, 112, 128; three training seeds
(seed scatter 0.5e-4 to 1.5e-4 in chi2, far below the hold-out row standard error of 1.2e-3, 6e-3 for r). A cubic smoothing B-spline in ln K (weights 1/error, s = number of
points) is fitted to the seed-mean hold-out chi2; K* = its minimum; uncertainty = paired bootstrap over hold-out rows (the same rows for every K, per-row chi2 cached).

| group | K* (spline minimum) | bootstrap 16-84% | chi2 at K* | zero model |
|---|---|---|---|---|
| camera b | 5 (5.2) | 5.1-5.2 | 0.9014 | 0.9318 |
| camera r | 5 (4.6) | 4.5-4.7 | 0.9282 | 0.9830 |
| camera z | 11 (11.4) | 11.0-11.7 | 0.9680 | 1.1046 |
| sky | 108 | 80-128 | 0.348 | 288.9 |

The instrument curves are very flat at the minimum (b: K = 3-8 differ by < 0.0005, well below one standard error), so K* is nominal within about +-2 for b and r. Sky has a plateau
from K ~ 40: the minimum at 108 gains 5% in chi2 over K = 57; knee alternatives (a doubling of K lowers chi2 by less than 20/10/5%): K = 30 / 45 / 57. Final templates were trained
fresh at K* (seed 100, OUT/final): hold-out hidden chi2 0.9014 (b, K=5), 0.9281 (r, K=5), 0.9682 (z, K=11), sky K=108 0.332 +- 0.027 (in-sample visible 0.185).
Figures: k_selection, template_fits_instrument, template_fits_sky, templates_instrument_cam, templates_sky (+ caches). Held-out rows (held-out tiles): mean hidden chi2 per row zero model ->
templates b 0.960 -> 0.879, r 1.040 -> 0.882, z 1.339 -> 0.947 (the gain comes mostly from the heavy tail of rows with large residuals; typical rows barely move). Held-out
sky spectra (held-out nights): median hidden chi2 161 -> 0.15; median |data - fit| / data on the brightest 10% of pixels about 1.1% (0.3-3.8%); the largest residuals sit on
the strongest lines (5577, 5890/5896 A), i.e. line-width and shift changes. Template anatomy: b #1 broad level plus the 5577 line, b #2-5 are 3600 A edge terms; r #2/#4 sky-line
patterns (6300, A-band 6830, OH 7250), r #3 a step near 6750 A, r #5 noise-like; z #1-2 red-tail curvature, z #4 a step near 8700 A, z #3, #5-11 OH-line clusters beyond 9300 A;
sky #1 smooth continuum, the rest line groups, #15 a single localised line near 4360 A.
