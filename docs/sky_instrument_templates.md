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

## Exploratory follow-ups (2026-11-08): b #1, signal-weighted training, derivative terms (template_diagnostics.py, signal_weighted.py, sky_derivative.py)

**Camera-b template #1 is a solar-type spectrum.** Its high-pass shape has features at Ca K/H (3934/3968 A), the G band and Mg b (sign arbitrary; correlation with Redrock STAR F/G/K templates
-0.2 to -0.3 because of the line forest) and its coefficient on held-out SKY rows rises with sky brightness: median |c1| 1.9 (dark, n=9494), 7.5 (bright, n=7894), 63.9 (backup, n=2612).
Consistent with scattered moonlight / twilight (a solar spectrum) whose fibre-to-fibre variation the sky model, fitted to the sky fibres, cannot remove. Not proven against the moon geometry.

**Capture of the global K=20 shapes** (energy-weighted fraction of the global templates inside the per-camera span, template_capture_K20): b 0.60 (K_b=5), 0.62 (8), 0.68 (20); r 0.65 / 0.70 / 0.71; z 0.89 / 0.91 / 0.91. Yet the global K=20 templates restricted to
a camera and fitted per camera score b 0.906, r 0.938, z 0.971, against 0.905, 0.935, 0.970 for per-camera-trained K=20: the uncaptured structure has no per-camera predictive value.

**Signal-weighted training** (row score e = whole-row zero-model chi2 - 1, noise-only sigma 0.03; 80th percentile 0.00 (b), -0.00 (r), 0.08 (z); 95th 0.10 / 0.17 / 0.61): uniform, score-weighted and top-20%-only training agree within +-0.001 in hold-out chi2 for K = 3..24
in every camera, on all rows and on the high-score rows; the weighted mode is slightly worse (r +0.005). The information in the tail rows is already used; sampling is not the limit.

**Derivative (shift/width) terms for the sky templates**, priors from the scatter of the unregularised coefficients (shift 0.04-0.09 px, width 0.01-0.05 px^2; sigma 0.07 px / 0.03 px^2): at fit time only, they help at small K (K=4: 6.74 -> 5.26, K=8 1.58 -> 1.42, K=12 0.958 -> 0.942) and hurt from K=16
(+0.02-0.03). Training from scratch with the terms in the loop diverged (K >= 32: chi2 17-112), not a valid test. Warm start (plain templates, 600 steps, lr 3e-4) with a plain-loss control: with derivative terms K=4/8/12/24/48 reach 4.84 / 1.37 / 0.831 / 0.441 / 0.320;
the control (plain, 600 more steps) 6.75 / 1.58 / 0.947 / 0.481 / 0.325. The derivative terms are worth 28% / 13% / 12% / 8% / 2% at K = 4 / 8 / 12 / 24 / 48. More important: the extra training alone moves plain K=48 from 0.373 to 0.325, below the plain K=96
value of the 1500-step sweeps (0.334), so the K* = 108 of the sky sweep is partly a training-length artefact; with longer training K ~ 48 reaches that level. The sweep should be rerun with longer training before the sky K is final.

## Long-training sweeps with patience annealing; per-camera + full-grid sets combined (long_sweep.py, 2026-11-08)

Schedule: AdamW lr 3e-3 halved (best templates restored) when a fixed validation set (rows carved out of the training set) has not improved by 0.02% for 4 checks of 100 steps, stop at lr < 2e-5 or 10,000 steps
(median 4,200-7,200 steps, i.e. 3-5x the earlier 1500). One seed. Groups: per-camera instrument b, r, z (K = 1..32, 34, 36, 40, 44, 48), instrument templates on the full 7781-pixel grid (K = 1..40, 44, 48, 56, 64), sky (K = 1..32, 34..48 step 2, 56, 64, 72, 80, 96, 112, 128).
Spline K* (paired bootstrap 16-84%): b 5.2 (5.1-5.3), r 5.6 (5.4-5.8), z 11.8 (11.4-12.1), full grid 18.6 (18.3-19.0), sky 88.9 (87.9-128). Hold-out chi2 at the minimum: b 0.9015, r 0.9281, z 0.9681, full 0.9361 (zero model 1.0143), sky 0.294 (before: 0.348).
The instrument K* are unchanged by the longer training (b 5, r 5 -> 6, z 11 -> 12; full 16-24 before -> 19); the sky chi2 improved 15% and reaches its plateau at K = 64 (0.295; K = 48 0.330, 56 0.313, 80 0.300, 96 0.301, 128 0.303; 32 0.419, 24 0.476), so the sky K is 64-90 rather than the 108 of the 1500-step sweep.

Combining (full-grid set sliced to each camera, stacked with the camera's own set, orthonormalised; hold-out hidden chi2): per-camera alone b 0.9015 / r 0.9281 / z 0.9682; full grid sliced alone 0.9058 / 0.9374 / 0.9710; combined orthonormal (24 / 25 / 31 dims) 0.9076 / 0.9395 / 0.9749.
The combination is worse than the per-camera set alone (+0.0061 +- 0.0001, +0.0114 +- 0.0008, +0.0067 +- 0.0001), and adding the sliced full-grid directions one at a time (orthogonal to the camera's span, ordered by energy) increases the hold-out chi2 monotonically
for m = 1..19 in every camera. Collinearity: principal-angle cosines between the spans > 0.99 for 1 / 1 / 2 directions and > 0.9 for 3 / 4 / 10 (mean cos^2 0.65 / 0.68 / 0.92 for b / r / z), stack condition number 38 / 52 / 71, no singular value below 1e-3 of the largest (nothing is numerically collinear).
The hold-out score on SKY-fibre rows does not reward cross-camera structure; whether it matters for targets (junction steps, overlap behaviour) is not tested here.
Figures: long_k_selection, long_combined_collinearity, long_combined_added_dims (+ .npz). Data: cache/sky_templates_v2/long/{seed0,combined}.

## Joint fit of the 32 legacy templates with the fiducial instrument + sky templates on DR2 spectra (joint_fit_demo.py, joint_fit_figures.py, 2026-11-12; corrected)

Fiducial sets: per-camera instrument b K=5, r K=6, z K=12 (long-trained, times the overlap ramps) and sky K=64; legacy physical templates `Legacy/k32_final.pt` (32 x 13903, rest log10 lambda 400-9824 A, 1e-4 dex; linear interpolation at lambda/(1+z), catalogue z;
the mean resolution matrix changes chi2 by < 2%, so none is applied: the templates were trained on data already at instrument resolution). Priors: instrument = robust SKY-row coefficient scatter, sky = 1% of the VAC coefficient scatter, physical ridge negligible.

**Sample correction.** A first version of this section used the top-S/N galaxy of each z bin of the physical-template hold-out set; those were mislabelled M/L/T dwarfs (Redrock chi2/px 46-1272, the zall SPECTYPE is GALAXY with ZWARN = 0 and DELTACHI2 > 100). In the 8000-spectrum hold-out set 5% have a Redrock chi2/px > 1.5, 0.6% > 3,
and none of the 64 spectra with median S/N > 30 has chi2/px < 1.4. The conclusions drawn from those objects (instrument terms absorbing the misfit of the legacy templates on "bright galaxies") were true only for contaminants and are withdrawn for galaxies. The demonstration now uses only spectra with Redrock chi2/px < 1.3 (galaxies; QSOs < 2.5)
and keeps one contaminant as a control.

Galaxies (S/N 4-22, Redrock chi2/px 1.2-1.3) and QSOs (S/N 37-38), free-prior joint fit: instrument share of the model energy 0.0 / 0.0 / 0.8 / 2.0 / 23.9 (S/N 4, tiny model energy) / 1.1 % for the six galaxies and 0.0 / 0.1 % for the QSOs; chi2 reduction from the instrument templates 1.1, 1.1, 12.5, 3.2, 4.5, 1.5 % (galaxies), 2.7 and 9.9 % (QSOs); sky <= 0.9% (energy share
1e-4 percent or less). The instrument coefficients are 0.1-0.9 of their prior sigma (median |d|/sigma 0.15-0.35): within the range the SKY rows show, and bounding them to +-3 sigma changes nothing (no coefficient at the bound). 300 random clean held-out galaxies (S/N 4-19, median 5.5): instrument share median 0.15% (mean 0.4%, 90th percentile 1.0%, max 8%), sky 3e-4 percent (max 0.006%);
the physical model moves by a median of 3.5% of its norm (90th percentile 9%, max 25%) between the physical-only fit to the raw flux and the physical refit to the cleaned flux (the refit reproduces the joint physical coefficients to 1e-13: the joint normal equations contain the refit); hidden-pixel test (36 spans, fit on visible pixels): + instrument -0.0129 +- 0.0036 chi2/px (1.1%, 70% of galaxies improve), + sky -0.0011 +- 0.0002 (77% improve),
both -0.0128 +- 0.0036. Redshift scan (60 km/s grid): minima move by at most 60-180 km/s for the clean objects (0 for two of them).
Largest galaxy effect: z = 0.33, S/N 12.7: the instrument component is a smooth additive level of 2-3 (1e-17 erg/s/cm2/A) in b falling to ~0.5 in z, i.e. as large as the blue flux itself (3-5), and lowers the physical-only chi2/px from 1.61 to 1.40; whether that is instrument signal or legacy-template mismatch cannot be separated by this fit.
Control, the contaminant (z = 0.47 dwarf, S/N 252, Redrock chi2/px 1272): instrument share 12%, coefficients 15-900 sigma, chi2/px 2113 -> 893, a stepped plateau of 30-700 absorbing the misfit; bounded to +-3 sigma the share is 0 and chi2/px 2073: the signature of a wrong physical model.
Figures: joint_fit_object_0..8 (raw, smoothed, template components, cleaned flux, residuals; object 8 = contaminant), joint_fit_energy, joint_fit_physical_change, joint_fit_bounded (+ .npz).

## Contamination caveat for earlier results (2026-11-12)

The galaxy samples of (a) the physical-template experiment (60,000 / 8,000 spectra, ZWARN = 0, DELTACHI2 > 100, snr_med > 4) and (b) the target-residual test (3,000 galaxies, snr_med > 8, DELTACHI2 > 500) were not filtered for stellar contaminants labelled GALAXY; the bright end is dominated by them (all 64 hold-out spectra with S/N > 30). Their chi2 means (including the baseline 4.79 of the target test) and the paired gains are heavy-tail sensitive
and were not re-run on a Redrock-chi2/px-clean sample. The sky and instrument template sweeps (SKY rows, VAC spectra) are unaffected.

### Degeneracy between the instrument and the physical part (object 5, galaxy z = 0.205, S/N 5.6; 2026-11-12)

4000-5000 A mean flux: data 2.36, physical-only fit 2.24, joint physical component 0.78, instrument component +1.56 +- 0.18 (posterior from the joint normal matrix with the priors), sky 0.01; the posterior correlation between the instrument and the physical band means is -0.93 and their sum is 2.34 +- 0.07:
the data fix the total, not the split. At 3600-4000 A the joint physical component is negative (-0.39) against an instrument component of +2.22. The SKY rows show a median 4000-5000 A level of -0.85 (5-95%: -5.6 to +1.5; 53% have |mean| > 1), so the amplitude is within the sky-fibre range but the sign is opposite to the typical sky-fibre
residual. The cleaned flux in the blue of such a faint object is therefore not a measurement of the instrument; it is a prior-driven partition (the chi2 gain is 1.5%). Ways to break it: non-negativity of the physical part, instrument priors from target fibres or an exposure-level link, a joint fit over the targets of one exposure/petal.

### Standing rule: the cumulative physical model must be non-negative (user, 2026-11-12)

Implemented in `joint_fit_demo.wls` (default on): the inequality-constrained quadratic program (physical model >= 0 at all 7781 observed pixels) is solved through its dual, a non-negative least-squares problem (no QP package in the environment). The refit-equals-joint identity still holds (same KKT conditions).
Effect on the clean sample: the constraint is inactive for most galaxies and binds in 37 of the 300 random galaxies (S/N ~5; median 0.6% of pixels would be negative, max 7.9%) and moves the physical component by a median 0.6% of its norm (max 3.5%); the 300-galaxy means are unchanged (instrument share median 0.15%; hidden-pixel gain
-0.0129 +- 0.0035 chi2/px). Object 5 (z = 0.205, S/N 5.6): the unconstrained joint physical component reached -1.2e-17 (5.9% of pixels negative); constrained, the 4000-5000 A mean splits as physical 1.30 / instrument 1.04 (unconstrained 0.78 / 1.56) at chi2/px 1.106 (1.104), and the physical component is pinned at 0 below 4000 A:
the constraint trims about a third of the instrument offset but the split stays prior-dependent. The dwarf control is also constrained (chi2/px of the physical-only fit rises 2113 -> 2855, joint 893 -> 935). The "bounded" figure (joint_fit_bounded) was made before the rule and uses the unconstrained fit.

## Empirical prior for the instrument coefficients (instrument_prior.py, prior_compare.py; 2026-11-12)

**The prior used so far was backup-dominated.** Its sigma came from the first 3000 hold-out SKY rows, which are ordered by tile and dominated by backup-program (twilight / moonlit) rows; it coincides with the backup-program scale (b#1 62.6 against 2.3 dark, 12.6 bright). The coefficients of 30,000 SKY rows (every 5th training row; 23 coefficients fitted jointly on the merged grid with the joint-fit columns) are extremely heavy-tailed: excess kurtosis 340-7500, standard deviation 20-145 against a MAD sigma of 2-8, and they depend strongly on the observing program
(b#1 median 0.3 dark / 2.0 bright / 49.6 backup; scatter 2.75 / 11 / 60). A moment-matched Gaussian describes held-out rows badly (Mahalanobis^2 median 2.3x its dof, 62% above the chi2 99th percentile).
Heavy-tailed fit: multivariate Student-t (EM, noise-deconvolved scatter, nu on a grid; nu hits the lower grid value 1.5, Cauchy-like tails) per program. Mean log predictive density of 10,000 held-out SKY rows (tiles not used for the fit): diagonal zero-mean Gaussian with the MAD sigma of all rows -131.0; Student-t all rows -67.2; Student-t per program -64.4
(dark -51.2 vs -59.8, bright -70.1 vs -87.1, backup -94.8 vs -522.6). Scale of b#1: 2.35 dark, 12.6 bright, 53.3 backup; location of b#1 0.25 / 1.78 / 46.4.
Effect on the joint fit (300 clean BGS galaxies, all bright program; non-negative physical model; the Student-t prior enters by iteratively reweighted least squares): hidden-pixel gain of the instrument terms -0.0128 +- 0.0034 (old prior), -0.0124 +- 0.0021 (diagonal MAD, all rows), -0.0122 +- 0.0023 (Student-t per program; 69 / 76 / 77% of the galaxies improve);
median instrument energy share 0.147% / 0.029% / 0.020%; faint galaxy (z = 0.205, S/N 5.6), 4000-5000 A split physical / instrument 1.30 / 1.04 (old), 1.79 / 0.44 (MAD), 1.61 / 0.67 (Student-t) with chi2/px 1.106 / 1.110 / 1.110: the empirical priors move the partition toward the physical part at no cost in the fit. The earlier statements that the fitted coefficients are "0.1-0.9 of the prior sigma" refer to the too-wide old prior.
Caveats: the SKY rows are in-sample sky-fit residuals; target fibres may be distributed differently. Figures: instrument_prior, instrument_prior_t, instrument_prior_effect (+ .npz).

### Demo figures regenerated with the empirical instrument prior (2026-11-12)

All joint_fit_* figures and numbers are now produced with the non-negative physical model and the Student-t prior per observing program (nu 1.5, location and scale from the SKY-row coefficients of that program; sky terms keep the 1% VAC prior), the program of each spectrum being that of its tile; no prior is fitted to the target rows (they are the inference set: the SKY residuals of the same program are the model of the unmodelled sky and instrument response being generalised to TGT rows).
This supersedes the numbers of the sections above. Clean galaxies (S/N 4-22) and QSOs: instrument share of the model energy 0.0 / 0.0 / 0.9 / 1.0 / 19.9 (S/N 4, tiny model energy) / 0.1 % (galaxies) and 0.04 / 0.1 % (QSOs); chi2 reduction from the instrument terms 0.9, 0.8, 12.5, 3.1, 4.4, 0.9 % (galaxies), 2.6 and 9.9 % (QSOs); median |coefficient - location| / scale 0.2-3.7 prior scales (log10 -0.76 to +0.57).
Contaminant control (dwarf, Redrock chi2/px 1272): share 22.7%, chi2/px 2855 -> 858, coefficients ~120 prior scales; bounded to +-3 scales (box-constrained, around the prior location) the share is 0 and chi2/px 2077.
300 random clean galaxies (all bright program): instrument energy share median 0.02% (mean 0.17%, 90th percentile 0.43%, max 6.1%), sky <= 0.014%; the physical model moves by a median 1.15% of its norm (90th percentile 6.3%, max 23.8%); hidden-pixel gain of the instrument terms -0.0122 +- 0.0024 chi2/px (76% improve), of the sky terms -0.0013 +- 0.0002 (77%), both -0.0122 +- 0.0023 (77%).
Faint galaxy of the demo (z = 0.205, S/N 5.6): instrument component a smooth blue ramp of ~1 at 4000 A falling to ~0.1 beyond 5760 A (0.11% of the model energy; chi2/px 1.121 -> 1.110), the cleaned blue flux is lowered by ~0.7 instead of ~1.5-2.
Bounded figure: with +-3 prior scales the instrument coefficient sits at the bound in 4-48% of the clean objects (chi2/px 1.44 vs 1.40 free for the z = 0.33 galaxy, 1.64 vs 1.53 for the z = 1.76 QSO): the data want more than three scales for the strongest cases.

## Independent verification with stellar-population synthesis fits (sps_verification.py, 2026-11-16)

Physical model independent of the legacy DeepRock templates, Redrock's templates and the SKY rows: the fastspecfit SPS templates (FSPS with C3K spectra and MIST isochrones, Chabrier IMF, 5 age bins at solar metallicity, version 3.0.1, `/global/cfs/cdirs/desi/science/gqp/templates/fastspecfit/3.0.1`), velocity dispersion (120 or 220 km/s), Calzetti dust (Av 0-1.2), 16 Gaussian emission lines, all amplitudes >= 0 (NNLS), the mean resolution matrix; the dust/dispersion grid is chosen on the SPS-only fit and held fixed for the other variants.
300 clean BGS galaxies (Redrock chi2/px < 1.3, S/N 4-19, bright program, catalogue z). Variants: V0 SPS only; V1 SPS + fiducial instrument templates (Student-t prior of the program) + sky templates, fitted jointly; V2 control with every instrument and sky template reversed in wavelength within its camera (same smoothness, amplitude statistics and prior, no calibration meaning); V3-V6 the instrument-only and sky-only
halves of V1 and V2. Hidden-pixel test: 36 spans, two draws, fitted on the visible pixels.
In-sample chi2/px: SPS-only 1.146, V1 1.119, V2 1.136; V1-V0 -0.0271 +- 0.0012 (100% of galaxies), V2-V0 -0.0100 +- 0.0006, V1-V2 -0.0171 +- 0.0009 (99%).
Hidden pixels: SPS-only 1.273, V1 1.251, V2 1.269; V1-V0 -0.0217 +- 0.0014 (97% negative), V2-V0 -0.0036 +- 0.0007 (59%), V1-V2 -0.0181 +- 0.0011 (98%): the real templates improve the independent model out of sample by 0.0217 chi2/px, a generic equally flexible control by 0.0036.
Decomposition of the hidden-pixel change (real | reversed): all pixels inst+sky -0.0217 | -0.0036, inst only -0.0160 | -0.0035, sky only -0.0073 | -0.0016; sky-line pixels (top 10% of the sky-template energy; SPS-only chi2/px 1.330) inst+sky -0.0936 +- 0.0059 | -0.0008, inst only -0.0444 | +0.0002, sky only -0.0584 | -0.0019; continuum pixels (1.269) inst+sky -0.0143 | -0.0040, inst only -0.0131 | -0.0039, sky only -0.0020 | -0.0016.
So the instrument templates remove a real sky-line residual (7% of the chi2 at sky-line pixels, which the sky templates alone do not fully capture) and a real continuum component (about 0.009 chi2/px beyond the generic control, 0.7% of the continuum chi2).
Cross-model check: the instrument component of V1 (SPS) against that of the legacy-template joint fit of the same galaxy: correlation across pixels (41-pixel smoothed, 3800-9600 A) median 0.66 for the same galaxy against 0.14 between different galaxies; the shape (blue-weighted ramp with steps at the camera junctions) agrees in the examples, but the 4000-5000 A level correlates only 0.12 across galaxies and the SPS amplitude is a median 0.55 of the legacy one:
the shape of the instrument response is supported by two independent physical models, its absolute level (the instrument/physical partition) is not.
Caveats: the SPS model is crude (5 ages, one metallicity), so the improvements also contain SPS mismatch absorbed by the real templates (the reversed control removes generic flexibility but not mismatch aligned with real spectral structure); faint galaxies only; the prior comes from SKY rows; repeat observations of the same targets and a second SPS family (other metallicities) would be the next independent checks.
Figures: sps_verification, sps_verification_decomposition (+ .npz).

## Fiducial set frozen (2026-11-17)

Decision: the instrument templates (per camera, b K=5, r K=6, z K=12) and the sky templates (K=64) under the per-program Student-t prior are the fiducial set; shape is frozen, scale is adjusted after the fact through the coefficient priors (`Fiducial.rescale_prior`). Stored in `artifacts/fiducial_templates_v1.npz` with the loader `DESIFlow.templates.fiducial.load_fiducial()`.
The prior scatter is a full 23 x 23 matrix fitted to the joint coefficients on the merged grid, so cross-camera covariance is included (corr b1-r1 +0.5, r1-z1 +0.64 dark / +0.76 bright / +0.84 backup, b1-z1 +0.35-0.54; mean |corr| 0.10-0.15 within a camera, 0.05-0.09 across cameras). Held-out log predictive density per SKY row, full / block-diagonal / diagonal scatter:
dark -51.24 / -51.59 / -52.18; bright -70.13 / -70.65 / -71.14; backup -94.78 / -95.77 / -96.17, i.e. the cross-camera terms are worth 0.35 / 0.52 / 0.98 nats per row (significant but modest). The sky prior (1% of the VAC coefficient scatter) is an assumption, not a calibrated distribution.
